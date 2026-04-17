# Condition Mapping Pipeline

Maps free-text disease condition labels from ClinicalTrials.gov (AACT) to ICD-10-CM codes and then to CCSR (Clinical Classifications Software Refined) disease categories. The output is a one-row-per-trial feature table used as input to the regression model.

**Source reference:** DXCCSR v2026-1 (AHRQ), 75,725 ICD-10-CM codes.

---

## Pipeline Overview

```
raw_data/conditions_raw.csv          (filtered cohort: 93,701 trials)
raw_data/browse_conditions.csv       (AACT MeSH terms: all AACT trials)
         │
         ▼
Step 00 — Exclusion Taxonomy         stage0_conditions.csv
         │
         ▼
Step 01 — Normalization              stage1_normalized.csv
         │
         ▼
Step 02 — ICD-10 Lookup (TF-IDF)    stage2_icd10.csv  +  manual_review_queue.csv
         │
         ▼
Step 02b — Coverage Review Ranker   review_ranked.csv  +  review_top300.csv
         │
         ▼
Step 03 — CCSR Join                  stage3_with_ccsr.csv  (long form)
                                     stage3_nct_features.csv  (one row per trial)
```

---

## Stage 0 — Exclusion Taxonomy

**Script:** `step00_exclusion_taxonomy.py`  
**Input:** `raw_data/conditions_raw.csv`  
**Output:** `output/stage0_conditions.csv`

Classifies every raw condition string into one of six buckets. Only the `disease` bucket proceeds to Stage 1. Buckets are checked in priority order — the first match wins.

| Bucket | What it catches | Example |
|---|---|---|
| `corrupted` | Excel errors, empty strings, leading `_` or `-` artifact | `#NAME?`, `_unknown` |
| `pk_admin` | Pharmacokinetic or administrative concepts | `bioavailability`, `dose-finding` |
| `demographic` | Age range descriptors, healthy volunteer labels | `18-42 year`, `healthy volunteers` |
| `drug_term` | Drug codes, biologic suffixes, vaccine names, salt forms | `ABBV-154`, `pembrolizumab`, `hydrochloride` |
| `staging_only` | Bare line-of-therapy strings with no surviving disease name | `2L+`, `advanced refractory` |
| `disease` | Everything else — proceeds to Stage 1 | `non-small-cell lung cancer` |

### Output columns — `stage0_conditions.csv`

| Column | Type | Description |
|---|---|---|
| `nct_id` | str | ClinicalTrials.gov trial identifier |
| `condition_raw` | str | Original condition string (lowercased) |
| `exclusion_bucket` | str | One of: `disease`, `demographic`, `pk_admin`, `drug_term`, `staging_only`, `corrupted` |
| `bucket_reason` | str | Sub-reason within the bucket (e.g. `pk_term:bioavailability`, `drug_code`) |

---

## Stage 1 — Normalization and Source Priority

**Script:** `step01_normalize.py`  
**Inputs:** `output/stage0_conditions.csv` (disease rows only), `raw_data/browse_conditions.csv` (MeSH-list terms)  
**Output:** `output/stage1_normalized.csv`

Merges two condition sources, normalizes text, and assigns each trial up to 3 priority-ranked condition slots.

### Two input sources

| Source | What it is | Priority |
|---|---|---|
| `mesh_list` | AACT-curated MeSH terms from `browse_conditions.csv` (type = `mesh-list`) | 4 (highest) or 2 if Tier B sole |
| `raw_condition` | Raw free-text conditions from `conditions_raw.csv`, classified `disease` in Stage 0 | 3 |

### MeSH tier system

MeSH terms are classified into three tiers before priority assignment:

| Tier | Definition | Behavior |
|---|---|---|
| **A** (suppress) | Chapter-level umbrella terms with no diagnostic specificity (e.g. `neoplasms`, `pain`, `infections`) | Always dropped (priority = 0) |
| **B** (demote) | Mid-level terms that are broad but not vacuous (e.g. `leukemia`, `diabetes mellitus`) | Suppressed if a non-A/B sibling exists for the same trial; kept (priority = 2) if it is the only option |
| **keep** | Specific, diagnostically useful MeSH terms | Priority = 4 |

### Normalization steps applied to every string

1. Strip artifacts (quotes, trailing punctuation)
2. Unwrap or trim parentheticals
3. Reverse MeSH comma-inversion (`"leukemia, myeloid, acute"` → `"acute myeloid leukemia"`)
4. British → American spelling (`leukaemia` → `leukemia`, `tumour` → `tumor`)
5. Hyphen normalization (`non small cell` → `non-small-cell`, `b cell` → `b-cell`)
6. Extract clinical modifier flags (see below), then strip staging/therapy words from the query string

### Slot assignment

Within each trial, up to 3 condition slots are assigned. Rank 1 = highest priority. Tie-breaking within the same priority: longer normalized string wins (length is a proxy for specificity).

### Output columns — `stage1_normalized.csv`

| Column | Type | Description |
|---|---|---|
| `nct_id` | str | Trial identifier |
| `slot_rank` | int | Slot position within the trial: 1 (best) → 3 |
| `source` | str | `mesh_list` or `raw_condition` |
| `condition_raw` | str | Original string before normalization |
| `condition_normalized` | str | Cleaned, normalized string used for ICD-10 lookup |
| `mesh_tier` | str | `A`, `B`, `keep`, or `na` (for raw_condition rows) |
| `priority` | int | 4 = best MeSH; 3 = raw condition; 2 = Tier B sole; 0 = suppressed |
| `tier_b_suppressed` | int | 1 if this is a Tier B MeSH row that was suppressed because a better sibling exists |
| `tier_b_only` | int | 1 if this is a Tier B MeSH row kept as the sole available term |
| `metastatic_flag` | int | 1 if the raw string contained "metastatic" or "advanced" (not "advanced age") |
| `relapsed_refractory_flag` | int | 1 if raw string contained "relapsed", "refractory", or "r/r" |
| `line_of_therapy` | int/null | 1, 2, or 3 if a line-of-therapy marker was found; null otherwise |
| `pediatric_flag` | int | 1 if raw string contained "pediatric", "childhood", "juvenile", or "neonatal" |
| `adult_flag` | int | 1 if raw string contained "adult(s)" |
| `biomarker_flag` | int | 1 if raw string contained a specific biomarker (EGFR, HER2, BRAF, ALK, PD-L1, etc.) |
| `normalized_too_short` | int | 1 if the normalized string is fewer than 3 characters (not used for lookup) |

---

## Stage 2 — ICD-10 Lookup

**Script:** `step02_icd10_lookup.py`  
**Input:** `output/stage1_normalized.csv`  
**Outputs:** `output/stage2_icd10.csv`, `output/manual_review_queue.csv`

Maps each normalized condition string to an ICD-10-CM code using a two-step approach.

### Lookup strategy

**Step 1 — Alias dictionary (312 entries):** Exact-match lookup for known lexical mismatches between clinical language and ICD-10 terminology. Covers HIV variants, common abbreviations, oncology terms. Alias matches are always accepted.

**Step 2 — TF-IDF retrieval:** For strings not in the alias dictionary, builds a TF-IDF index over the 75,725 ICD-10-CM code descriptions from DXCCSR and retrieves the top candidate using bigram TF-IDF cosine similarity, then re-scores with a composite:

```
score = 0.40 × token_overlap + 0.30 × Jaccard + 0.30 × containment − 0.15 × length_penalty
```

**Auto-accept threshold:** score ≥ 0.65 → `auto_accepted`. Below threshold → `manual_review`.

### Output columns added in `stage2_icd10.csv`

All Stage 1 columns are carried through, plus:

| Column | Type | Description |
|---|---|---|
| `icd10_code` | str | ICD-10-CM code assigned (dot-free format, e.g. `C349`) |
| `icd10_description` | str | ICD-10-CM description for the assigned code |
| `icd10_confidence` | float | Composite similarity score (0–1) |
| `icd10_status` | str | `auto_accepted`, `manual_review`, or `alias` |
| `match_tier` | str | `alias`, `tfidf_auto`, or `tfidf_flagged` |

### Coverage
- Auto-accepted: ~28% of unique condition strings
- Manual review queue: ~13,800 strings (see `manual_review_queue.csv`)
- Trial-level CCSR coverage after Stage 3: **84.9%** of the 443k universe; **73.9%** of the completed regression cohort

---

## Stage 2b — Coverage-Weighted Review Ranker

**Script:** `step02b_coverage_review.py`  
**Input:** `output/stage2_icd10.csv`, `raw_data/studies.csv`  
**Outputs:** `output/review_ranked.csv`, `output/review_top300.csv`, `output/review_quick_wins.csv`

Ranks the manual review queue by expected impact on trial coverage.

**Composite priority score:**
```
score = 0.35 × nct_id_count_norm
      + 0.25 × enrollment_weight_norm
      + 0.30 × zero_coverage_ncts_norm
      + 0.10 × near_threshold_norm
```

- **nct_id_count:** how many trials use this condition string
- **enrollment_weight:** sum of enrolled participants across those trials
- **zero_coverage_ncts:** how many of those trials currently have no CCSR mapping at all
- **near_threshold:** how close the TF-IDF confidence score is to the 0.65 auto-accept threshold (near misses are cheap to resolve)

Top 300 strings cover ~85,673 trials (51,230 currently zero-coverage).

---

## Stage 3 — CCSR Join

**Script:** `step03_ccsr_join.py`  
**Inputs:** `output/stage2_icd10.csv`, `raw_data/condition_mapping_data/DXCCSR_v2026-1.csv`  
**Outputs:** `output/stage3_with_ccsr.csv` (long form), `output/stage3_nct_features.csv` (one row per trial)

Joins each auto-accepted ICD-10 code to its CCSR category, then pivots to one row per trial with up to three CCSR condition slots.

### Slot compaction

The three CCSR slots are **compacted left**: within each trial, the slots are filled in rank order from Stage 1, but gaps from unmapped conditions are closed. `ccsr_slot1` always contains the first non-null CCSR mapping; `ccsr_slot2` the second; `ccsr_slot3` the third. There are no gaps.

### Output columns — `stage3_nct_features.csv` (model input table)

| Column | Type | Description |
|---|---|---|
| `nct_id` | str | Trial identifier |
| `ccsr_slot1` | str/null | CCSR code for the highest-priority mapped condition (e.g. `NEO030`) |
| `ccsr_slot1_desc` | str/null | Human-readable description of `ccsr_slot1` (e.g. `Non-Hodgkin lymphoma`) |
| `ccsr_slot2` | str/null | CCSR code for the second-priority mapped condition |
| `ccsr_slot2_desc` | str/null | Description of `ccsr_slot2` |
| `ccsr_slot3` | str/null | CCSR code for the third-priority mapped condition |
| `ccsr_slot3_desc` | str/null | Description of `ccsr_slot3` |
| `ccsr_domain` | str/null | First 3 characters of `ccsr_slot1` — coarse disease domain (e.g. `NEO`). Auxiliary feature; `ccsr_slot1` is the primary. |
| `has_ccsr` | int | 1 if any slot has a CCSR mapping, else 0 |
| `metastatic_flag` | int | 1 if any condition slot contained "metastatic" or "advanced" |
| `relapsed_refractory_flag` | int | 1 if any condition slot contained "relapsed", "refractory", or "r/r" |
| `line_of_therapy` | float/null | Maximum line-of-therapy number seen across slots (1, 2, or 3) |
| `pediatric_flag` | int | 1 if any slot contained a pediatric indicator |
| `adult_flag` | int | 1 if any slot contained "adult(s)" |
| `biomarker_flag` | int | 1 if any slot contained a targetable biomarker marker |
| `tier_b_only_flag` | int | 1 if the trial's only MeSH terms were Tier B (broad-but-not-vacuous); indicates weaker condition specificity |

---

## CCSR Domain Reference Table

`ccsr_domain` is the 3-letter prefix of the CCSR category code. Each domain maps to a clinical body system. The table below shows all 20 domains present in the dataset, ordered by frequency in `stage3_nct_features.csv`.

| Domain | Full name | Clinical scope | nct_ids in dataset |
|---|---|---|---|
| `NEO` | Neoplasms | All cancers and tumors — solid tumors, hematologic malignancies, benign neoplasms | 83,090 |
| `CIR` | Circulatory system diseases | Heart disease, hypertension, arrhythmias, coronary disease, stroke, peripheral vascular disease | 38,136 |
| `END` | Endocrine, nutritional, and metabolic diseases | Diabetes, obesity, thyroid disorders, lipid disorders, adrenal and pituitary conditions | 35,227 |
| `NVS` | Nervous system diseases | Neurodegenerative disease (Alzheimer's, Parkinson's, ALS), epilepsy, MS, neuropathy, headache, stroke sequelae | 32,101 |
| `MBD` | Mental and behavioral disorders | Depression, anxiety, schizophrenia, bipolar disorder, ADHD, autism, substance use disorders | 26,979 |
| `MUS` | Musculoskeletal system and connective tissue diseases | Arthritis (RA, OA, psoriatic), osteoporosis, lupus, gout, fibromyalgia, back pain | 22,467 |
| `INF` | Infectious and parasitic diseases | Bacterial, viral, fungal, and parasitic infections — including HIV, COVID-19, hepatitis, TB | 21,569 |
| `DIG` | Digestive system diseases | GI tract conditions — IBD, Crohn's, ulcerative colitis, GERD, liver disease, pancreatitis | 20,818 |
| `RSP` | Respiratory system diseases | Asthma, COPD, pulmonary fibrosis, pulmonary hypertension, pneumonia, cystic fibrosis | 19,726 |
| `SYM` | Symptoms, signs, and ill-defined conditions | Non-specific findings not yet classified to a disease — pain, fatigue, fever, dyspnea | 17,195 |
| `GEN` | Genitourinary system diseases | Kidney disease (CKD, nephritis), bladder conditions, urinary tract infections, sexual disorders | 14,915 |
| `EYE` | Eye and adnexa diseases | Glaucoma, macular degeneration, diabetic retinopathy, uveitis, dry eye, cataracts | 10,724 |
| `SKN` | Skin and subcutaneous tissue diseases | Psoriasis, eczema/atopic dermatitis, acne, alopecia, wound healing, skin infections | 7,414 |
| `PRG` | Pregnancy, childbirth, and puerperium | Obstetric conditions, gestational diabetes, preeclampsia, preterm birth | 5,795 |
| `BLD` | Diseases of blood and blood-forming organs | Anemia (aplastic, hemolytic, sickle cell), coagulation disorders, white blood cell disorders, immune deficiencies | 5,608 |
| `MAL` | Congenital malformations, deformations, and chromosomal abnormalities | Birth defects, structural anomalies, Down syndrome, chromosomal disorders | 4,653 |
| `INJ` | Injury, poisoning, and consequences of external causes | Traumatic injuries, fractures, burns, toxic exposures, sequelae of injury | 4,202 |
| `PNL` | Perinatal conditions | Conditions originating in the perinatal period — prematurity, neonatal jaundice, birth asphyxia | 3,132 |
| `EAR` | Ear and mastoid process diseases | Hearing loss, otitis media, tinnitus, Ménière's disease | 1,775 |
| `FAC` | Factors influencing health status | Carrier status, screening encounters, prophylactic procedures, health-contact reasons without active disease | 616 |

> **Note on `Other_Unclassified`:** Trials where all condition strings were classified as non-disease in Stage 0 (demographic, PK/admin, drug terms) have `has_ccsr = 0` and null slot columns. In the regression model `category` column, these appear as `Other_Unclassified`.

---

## Integration into Regression Pipeline

`stage3_nct_features.csv` is joined into the regression cohort in `4_regression/core/step00_cohort_io.py` (`load_and_join`) via a **left join on `nct_id`**, applied after all study-level cohort filters (COMPLETED status, date range, duration band).

The `ccsr_domain` column is exposed as `category` in the feature matrix (21 levels including `Other_Unclassified`), replacing the former `categorized_output.csv` system.

**Ablation result (test R², dedicated phase models):**

| Feature used as `category` | PHASE1 | PHASE2 | PHASE3 |
|---|---|---|---|
| `ccsr_domain` (21 levels) — current | **0.652** | **0.446** | 0.434 |
| `ccsr_slot1` (~300 codes) | 0.636 | 0.437 | **0.435** |
| No category | 0.636 | 0.418 | 0.428 |

`ccsr_domain` outperforms `ccsr_slot1` on P1/P2 because the 300-code one-hot encoding is too sparse for HGBR — most codes appear in fewer than 50 trials. The coarser domain grouping generalizes better. `ccsr_slot1` and `ccsr_slot2`/`ccsr_slot3` are preserved in the feature table for future use with alternative encodings (target encoding, top-N truncation).

---

## Future Improvements

### 1. Manual review of the ICD-10 queue

**14,033 condition strings** remain in `manual_review_queue.csv` with confidence scores below the 0.65 auto-accept threshold. These strings cover a substantial portion of the trial universe. Resolving them increases CCSR coverage in the regression cohort (currently 73.9% of completed trials).

The queue is pre-ranked by impact in `review_ranked.csv` and `review_top300.csv`. The top entries by priority:

| Rank | Condition string | Trials affected | Zero-coverage trials | TF-IDF best guess |
|---|---|---|---|---|
| 1 | `healthy` | 5,290 | 4,884 | Z76.3 (Healthy person — wrong concept) |
| 2 | `hiv infections` | 4,272 | 3,350 | E88.14 (HIV lipodystrophy — wrong) |
| 3 | `chronic renal insufficiency` | 2,805 | 1,225 | I87.2 (Venous insufficiency — wrong) |
| 4 | `acquired immunodeficiency syndrome` | 2,943 | 1,991 | D84.9 (Immunodeficiency unspec. — wrong) |

**Priority action:** Resolve the top 300 strings in `review_top300.csv`. Each row has a `reviewer_code` column for the correct ICD-10 code and a `candidates_json` column with the top-5 TF-IDF candidates for reference. After review, run `step02c` (not yet written) to ingest overrides and re-run Stage 3.

**2,416 strings** are within 0.10 of the auto-accept threshold (`review_quick_wins.csv`) — these are the cheapest to resolve since the TF-IDF match is likely already correct and needs only a human confirmation.

### 2. Write step02c — Reviewer override ingestion

A script (`step02c_ingest_reviews.py`) is needed to read the `reviewer_code` column from the reviewed `review_top300.csv`, merge those overrides back into `stage2_icd10.csv`, and trigger a partial re-run of Stage 3. This closes the loop between the manual review process and the model.

### 3. Alternative encodings for `ccsr_slot1`

The ablation showed `ccsr_slot1` (~300 codes) does not outperform the coarser `ccsr_domain` (21 levels) with raw one-hot encoding. Two alternatives worth testing:

- **Target encoding:** Replace each CCSR code with its mean target value (duration) computed on the training fold only. Eliminates the sparsity problem and directly captures the duration signal per code. Must be computed inside cross-validation to avoid leakage.
- **Top-N truncation:** One-hot encode only the top 50–100 most frequent CCSR codes; collapse the rest to `"other"`. Reduces sparsity while preserving granularity for common diseases.

### 4. Use `ccsr_slot2` and `ccsr_slot3` as features

`ccsr_slot2` is filled for 25.2% of the regression cohort; `ccsr_slot3` for 5.6%. These secondary conditions are currently unused. Options:

- Add `ccsr_slot2_domain` (3-letter prefix) as a separate categorical feature alongside `ccsr_slot1`.
- Add a binary `has_comorbid_domain` flag: 1 if `ccsr_slot2` exists and its domain differs from `ccsr_slot1`'s domain.
- Test whether secondary conditions add R² signal on top of slot1 alone, particularly for complex multi-indication trials.

### 5. Disease subgroup hierarchy for high-volume domains

The `NEO` domain covers 83,090 condition rows — the most of any domain — but groups radically different trial types (breast cancer, leukemia, GBM, sarcoma). A lookup table mapping specific CCSR codes to subgroups within NEO, MBD, and NVS would create more informative categories. Proposed first-pass (to be validated against observed duration distributions):

- **NEO:** `NEO_SOLID_COMMON` (breast, lung, colorectal, prostate), `NEO_SOLID_OTHER`, `NEO_HEME` (leukemia, MDS), `NEO_LYMPHOMA_MYELOMA`
- **MBD:** `MBD_MOOD_ANXIETY`, `MBD_PSYCHOTIC_BIPOLAR`, `MBD_ADDICTION_NEURODEV`
- **NVS:** `NVS_NEURODEGEN` (Alzheimer's, Parkinson's, ALS), `NVS_ACUTE_CEREBRO` (stroke, TBI), `NVS_OTHER`

Decision rule before adopting any split: each subgroup must have ≥ 100 trials in training, and the duration IQRs of the proposed subgroups must be demonstrably non-overlapping.

### 6. Expand the alias dictionary

The alias dictionary (`step02_icd10_lookup.py`) currently has 312 entries. Common failures visible in the manual review queue (HIV infections → wrong code, AIDS → wrong code, chronic renal insufficiency → wrong code) should be added as direct aliases to bypass TF-IDF entirely for known-bad cases. Each alias should map to the most specific clinically correct ICD-10 code, not the nearest string match.
