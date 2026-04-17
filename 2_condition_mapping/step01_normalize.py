"""
Stage 1 — Normalization and Source Priority

Inputs:
  2_condition_mapping/output/stage0_conditions.csv   (disease bucket from Stage 0)
  raw_data/browse_conditions.csv                     (mesh-list terms)

Output:
  2_condition_mapping/output/stage1_normalized.csv
  One row per (nct_id, slot_rank). Max 3 slots per nct_id.

Columns: nct_id, slot_rank, source, condition_raw, condition_normalized,
         mesh_tier, priority, tier_b_suppressed, tier_b_only,
         metastatic_flag, relapsed_refractory_flag, line_of_therapy,
         pediatric_flag, adult_flag, biomarker_flag, normalized_too_short

Priority model (slot 1 = highest):
  4 — MeSH-list term, not Tier A or Tier B (or Tier B kept as sole option → 2)
  3 — Raw condition (disease bucket from Stage 0)
  2 — Tier B MeSH term kept because no non-AB sibling exists for this nct_id
  0 — Suppressed (Tier A, or Tier B with non-AB sibling present)

Within same priority: longer normalized string gets the earlier slot
(length is a proxy for specificity — more words = more qualified term).
"""
import re
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "raw_data"
STAGE0_OUT = Path(__file__).parent / "output" / "stage0_conditions.csv"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Broadness sets ────────────────────────────────────────────────────────────
# Terms match the exact downcase_mesh_term values in browse_conditions.csv.

TIER_A_EXACT: frozenset[str] = frozenset({
    # Behavior / process / non-disease concepts
    "motor activity", "behavior", "sedentary behavior",
    "psychological well-being", "health behavior", "feeding behavior",
    "patient satisfaction", "patient compliance", "medication adherence",
    "communication", "body weight", "smoking cessation", "smoking",
    "alcohol drinking", "breast feeding", "weight loss",
    # Disease properties masquerading as conditions
    "recurrence", "neoplasm metastasis", "postoperative complications",
    "inflammation",
    # Chapter-level umbrella terms (too broad to distinguish disease)
    "neoplasms", "pain", "disease", "cardiovascular diseases", "heart diseases",
    "lung diseases", "mental disorders", "infections", "wounds and injuries",
    "nervous system diseases", "musculoskeletal diseases", "metabolic diseases",
    "vascular diseases", "virus diseases", "hematologic diseases",
    "autoimmune diseases", "bacterial infections", "communicable diseases",
    "gastrointestinal diseases", "rare diseases", "genetic diseases, inborn",
    "kidney diseases", "liver diseases", "chronic disease", "emergencies",
    "carcinoma",
    # Confirmed AACT artifact — not the neurological condition
    "agnosia",
})

# Tier B: demote if a non-Tier-A/B sibling exists for the same nct_id;
# keep (priority 2) if it's the only disease-like term available.
TIER_B_EXACT: frozenset[str] = frozenset({
    "leukemia", "lymphoma", "hematologic neoplasms", "lung neoplasms",
    "head and neck neoplasms", "gastrointestinal neoplasms",
    "central nervous system neoplasms", "skin neoplasms",
    "coronary disease", "renal insufficiency", "sleep apnea syndromes",
    "depressive disorder", "arthritis", "diabetes mellitus",
    "respiratory tract infections", "bone diseases, metabolic",
    "mood disorders", "myeloproliferative disorders",
})


def get_mesh_tier(term: str) -> str:
    """Return 'A', 'B', or 'keep' for a mesh-list term."""
    if term in TIER_A_EXACT:
        return "A"
    if term in TIER_B_EXACT:
        return "B"
    return "keep"


# ── Normalization patterns ────────────────────────────────────────────────────

_BRITISH = [
    (re.compile(r"\bleukaemia\b", re.I),  "leukemia"),
    (re.compile(r"\bleukaemias\b", re.I), "leukemias"),
    (re.compile(r"\bhaematolog", re.I),   "hematolog"),
    (re.compile(r"\bhaemorrhage\b", re.I),"hemorrhage"),
    (re.compile(r"\bhaemophilia\b", re.I),"hemophilia"),
    (re.compile(r"\btumour\b", re.I),     "tumor"),
    (re.compile(r"\btumours\b", re.I),    "tumors"),
    (re.compile(r"\boedema\b", re.I),     "edema"),
    (re.compile(r"\bpaediatric", re.I),   "pediatric"),
    (re.compile(r"\banaemia\b", re.I),    "anemia"),
    (re.compile(r"\bdiarrhoea\b", re.I),  "diarrhea"),
    (re.compile(r"\bbehaviour\b", re.I),  "behavior"),
    (re.compile(r"\bcolour\b", re.I),     "color"),
]

_HYPHEN_NORMS = [
    # Canonical hyphenated oncology terms that appear with/without hyphens
    (re.compile(r"\bnon[\s]small[\s]cell\b", re.I),  "non-small-cell"),
    (re.compile(r"\bnon-small\s+cell\b", re.I),       "non-small-cell"),
    (re.compile(r"\bb[\s]cell\b", re.I),              "b-cell"),
    (re.compile(r"\bt[\s]cell\b", re.I),              "t-cell"),
    (re.compile(r"\bnk[\s]cell\b", re.I),             "nk-cell"),
    (re.compile(r"\bnatural\s+killer\s+cell\b", re.I),"nk-cell"),
]

# Flags: extract before removal
_FLAG_PATTERNS: dict[str, re.Pattern] = {
    "metastatic_flag":          re.compile(r"\bmetastatic\b|\badvanced(?!\s+age)\b", re.I),
    "relapsed_refractory_flag": re.compile(
        r"\brelapsed[\s/]+refractory\b|\brelapsed\b|\brefractory\b|\br/r\b", re.I
    ),
    "pediatric_flag": re.compile(
        r"\b(pediatric|paediatric|childhood|juvenile|neonatal)\b", re.I
    ),
    "adult_flag":    re.compile(r"\badults?\b", re.I),
    "biomarker_flag": re.compile(
        r"\b(egfr|her[\s-]?2|her2|bcr[\s-]?abl|kras|braf|alk|ros1"
        r"|pd[\s-]?l1|msi[\s-]?h|tmb[\s-]?h|ntrk)\b",
        re.I,
    ),
}

_LINE_OF_THERAPY: list[tuple[int, re.Pattern]] = [
    (1, re.compile(r"\b(1st[\s-]line|first[\s-]line|1l\b)", re.I)),
    (2, re.compile(r"\b(2nd[\s-]line|second[\s-]line|2l\+?\b)", re.I)),
    (3, re.compile(r"\b(3rd[\s-]line|third[\s-]line|3l\+?\b|3\+[\s-]line)", re.I)),
]

# Staging descriptors removed from the query string (after flagging)
_STAGING_REMOVAL = re.compile(
    r"\b("
    r"metastatic"
    r"|advanced(?!\s+age)"          # "advanced" but not "advanced age"
    r"|relapsed[\s/]+refractory"
    r"|relapsed"
    r"|refractory"
    r"|r/r"
    r"|1st[\s-]line|first[\s-]line"
    r"|2nd[\s-]line|second[\s-]line"
    r"|3rd[\s-]line|third[\s-]line|3\+[\s-]line"
    r"|1l\b|2l\+?\b|3l\+?\b"
    r")\b",
    re.I,
)


# ── Normalization helpers ─────────────────────────────────────────────────────

def _clean_artifacts(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    s = s.rstrip(",;").strip()
    return re.sub(r"\s{2,}", " ", s)


def _handle_parentheticals(s: str) -> str:
    # Case 1: entire content is the parenthetical → unwrap it
    if s.startswith("(") and s.endswith(")") and "(" not in s[1:-1]:
        return s[1:-1].strip()
    # Case 2: trailing acronym in parens: "(crc)", "(aaa)", "(cll)" → drop
    result = re.sub(r"\s*\([a-z]{2,6}\)\s*$", "", s)
    if result != s:
        return result.strip()
    # Case 3: leading short qualifier in parens followed by the real name
    m = re.match(r"^\(([^)]{1,40})\)\s+(.+)$", s)
    if m and len(m.group(1).split()) <= 3:
        return m.group(2).strip()
    return s


def _reverse_mesh_inversion(s: str) -> str:
    """Reverse MeSH comma-inverted notation: 'leukemia, myeloid, acute' → 'acute myeloid leukemia'."""
    if "," not in s or len(s) > 70 or ":" in s or ";" in s:
        return s
    parts = [p.strip() for p in s.split(",")]
    first = parts[0]
    first_words = first.split()
    if not (1 <= len(first_words) <= 4):
        return s
    if first_words[0][0].isdigit():
        return s
    _staging = {"advanced", "metastatic", "relapsed", "refractory", "line", "stage"}
    if any(w.lower() in _staging for w in first_words):
        return s
    return " ".join(reversed(parts))


def _fix_british(s: str) -> str:
    for pattern, replacement in _BRITISH:
        s = pattern.sub(replacement, s)
    return s


def _normalize_hyphens(s: str) -> str:
    for pattern, replacement in _HYPHEN_NORMS:
        s = pattern.sub(replacement, s)
    return s


def _extract_flags_and_clean(s: str) -> tuple[str, dict]:
    flags: dict[str, int | None] = {}
    for name, pattern in _FLAG_PATTERNS.items():
        flags[name] = 1 if pattern.search(s) else 0

    # Line of therapy (first match wins)
    flags["line_of_therapy"] = None
    for line_num, pattern in _LINE_OF_THERAPY:
        if pattern.search(s):
            flags["line_of_therapy"] = line_num
            break

    # Remove staging descriptors and line-of-therapy patterns from query string
    cleaned = _STAGING_REMOVAL.sub("", s)
    # Also remove bare line codes that _STAGING_REMOVAL might miss
    cleaned = re.sub(r"\b\d+l\+?\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().rstrip(",;-").strip()
    return cleaned, flags


def normalize(text: str) -> dict:
    """Full normalization pipeline. Returns dict with 'normalized' + feature flags."""
    s = str(text).lower()
    s = _clean_artifacts(s)
    s = _handle_parentheticals(s)
    s = _reverse_mesh_inversion(s)
    s = _fix_british(s)
    s = _normalize_hyphens(s)
    s, flags = _extract_flags_and_clean(s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    too_short = 1 if len(s) < 3 else 0
    return {"normalized": s, "normalized_too_short": too_short, **flags}



# ── Main runner ───────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    logger.info("Loading Stage 0 output (disease bucket)")
    stage0 = pd.read_csv(STAGE0_OUT, low_memory=False)
    raw_disease = stage0[stage0["exclusion_bucket"] == "disease"][
        ["nct_id", "condition_raw"]
    ].copy()
    raw_disease["source"] = "raw_condition"
    raw_disease["mesh_tier"] = "na"
    logger.info("Raw disease rows: %s across %s nct_ids",
                f"{len(raw_disease):,}", f"{raw_disease['nct_id'].nunique():,}")

    logger.info("Loading browse_conditions.csv (mesh-list only)")
    browse = pd.read_csv(RAW_DATA / "browse_conditions.csv", low_memory=False)
    mesh = browse[browse["mesh_type"] == "mesh-list"][["nct_id", "downcase_mesh_term"]].copy()
    mesh = mesh.rename(columns={"downcase_mesh_term": "condition_raw"})
    mesh["source"] = "mesh_list"
    mesh["mesh_tier"] = mesh["condition_raw"].map(get_mesh_tier)
    logger.info("MeSH-list rows: %s across %s nct_ids",
                f"{len(mesh):,}", f"{mesh['nct_id'].nunique():,}")

    logger.info("Tier A mesh terms (will be suppressed): %s rows",
                f"{(mesh['mesh_tier'] == 'A').sum():,}")
    logger.info("Tier B mesh terms (conditional demotion): %s rows",
                f"{(mesh['mesh_tier'] == 'B').sum():,}")

    # ── Normalize unique strings ──────────────────────────────────────────────
    all_strings = pd.concat([
        raw_disease["condition_raw"],
        mesh["condition_raw"],
    ]).unique()
    logger.info("Normalizing %s unique condition strings ...", f"{len(all_strings):,}")

    norm_cache: dict[str, dict] = {s: normalize(s) for s in all_strings}
    norm_df = pd.DataFrame.from_dict(norm_cache, orient="index")
    norm_df.index.name = "condition_raw"
    norm_df = norm_df.reset_index()

    # ── Build unified candidate table ─────────────────────────────────────────
    candidates = pd.concat([raw_disease, mesh], ignore_index=True)
    candidates = candidates.merge(norm_df, on="condition_raw", how="left")

    # Drop rows whose normalized string is empty or too short to query
    before = len(candidates)
    candidates = candidates[candidates["normalized"].str.len() >= 3]
    logger.info("Dropped %s rows with normalized string < 3 chars",
                f"{before - len(candidates):,}")

    # ── Assign priorities (vectorized — no groupby) ───────────────────────────
    # Tier B demotion: suppress Tier B MeSH terms for any nct_id that has at
    # least one non-A/B MeSH term.  Computed once as a set lookup.
    logger.info("Assigning priorities and applying Tier B demotion ...")
    non_ab_nct_ids: set[str] = set(
        candidates.loc[
            (candidates["source"] == "mesh_list") & (candidates["mesh_tier"] == "keep"),
            "nct_id",
        ]
    )

    is_mesh   = candidates["source"] == "mesh_list"
    is_tier_a = candidates["mesh_tier"] == "A"
    is_tier_b = candidates["mesh_tier"] == "B"
    is_keep   = candidates["mesh_tier"] == "keep"
    has_non_ab = candidates["nct_id"].isin(non_ab_nct_ids)

    candidates["priority"] = 3                                          # raw default
    candidates.loc[is_mesh & is_keep,            "priority"] = 4       # best mesh
    candidates.loc[is_mesh & is_tier_b & ~has_non_ab, "priority"] = 2  # sole-option B
    candidates.loc[is_mesh & is_tier_b &  has_non_ab, "priority"] = 0  # suppressed B
    candidates.loc[is_mesh & is_tier_a,          "priority"] = 0       # always dropped

    candidates["tier_b_suppressed"] = (is_mesh & is_tier_b &  has_non_ab).astype(int)
    candidates["tier_b_only"]       = (is_mesh & is_tier_b & ~has_non_ab).astype(int)

    # Drop suppressed terms
    candidates = candidates[candidates["priority"] > 0].copy()

    # ── Deduplicate: same normalized string within nct_id → keep highest priority
    candidates = (
        candidates
        .sort_values(["nct_id", "priority", "normalized"], ascending=[True, False, False])
        .drop_duplicates(subset=["nct_id", "normalized"], keep="first")
    )

    # ── Assign slot ranks (top 3 per nct_id) ─────────────────────────────────
    # Sort within nct_id: priority DESC, then len(normalized) DESC (specificity proxy)
    candidates["_norm_len"] = candidates["normalized"].str.len()
    candidates = candidates.sort_values(
        ["nct_id", "priority", "_norm_len"], ascending=[True, False, False]
    )
    candidates["slot_rank"] = (
        candidates.groupby("nct_id").cumcount() + 1
    )
    candidates = candidates[candidates["slot_rank"] <= 3].copy()
    candidates = candidates.drop(columns=["_norm_len"])

    # ── Output columns ────────────────────────────────────────────────────────
    out_cols = [
        "nct_id", "slot_rank", "source", "condition_raw", "condition_normalized",
        "mesh_tier", "priority", "tier_b_suppressed", "tier_b_only",
        "metastatic_flag", "relapsed_refractory_flag", "line_of_therapy",
        "pediatric_flag", "adult_flag", "biomarker_flag", "normalized_too_short",
    ]
    candidates = candidates.rename(columns={"normalized": "condition_normalized"})

    # Fill missing flag columns defensively
    for col in out_cols:
        if col not in candidates.columns:
            candidates[col] = None

    result = candidates[out_cols].reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_nct = result["nct_id"].nunique()
    slot_dist = result["slot_rank"].value_counts().sort_index()
    src_dist = result["source"].value_counts()
    logger.info("nct_ids with ≥1 slot: %s", f"{n_nct:,}")
    logger.info("Slot distribution:\n%s", slot_dist.to_string())
    logger.info("Source distribution:\n%s", src_dist.to_string())
    logger.info("Tier B only (kept as sole term): %s rows",
                f"{result['tier_b_only'].sum():,}")
    logger.info("Tier B suppressed: see stage0 dropped rows (priority=0, not in output)")
    logger.info("Rows flagged too_short: %s", f"{result['normalized_too_short'].sum():,}")

    out = OUTPUT_DIR / "stage1_normalized.csv"
    result.to_csv(out, index=False)
    logger.info("Saved → %s (%s rows)", out, f"{len(result):,}")
    return result


if __name__ == "__main__":
    run()
