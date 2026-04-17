"""
Step 2b — Coverage-Weighted Manual Review Ranker

Ranks the 14,033 manual_review strings by:
  1. nct_id_count        — how many trials reference this string
  2. enrollment_weight   — total enrollment of affected trials (proxy for importance)
  3. zero_coverage_ncts  — how many affected trials have NO other CCSR code (this string is the only shot)
  4. therapeutic_area    — flag high-value TAs: oncology, cardiovascular, cns, infectious, metabolic
  5. near_threshold      — current best confidence is close to 0.65 (easy wins)

Composite priority score:
  score = 0.35 * nct_norm + 0.25 * enroll_norm + 0.30 * zero_cov_norm + 0.10 * near_thresh_norm

Outputs:
  output/review_ranked.csv         — full ranked list (14,033 rows)
  output/review_top300.csv         — top 300 for immediate review
  output/review_quick_wins.csv     — strings within 0.10 of threshold (likely just need alias)

Usage for reviewer:
  Open review_top300.csv.
  For each row: check candidates_json column (top 5 ICD-10 options scored).
  Fill in 'reviewer_code' and 'reviewer_name' columns.
  After review, run step02c_ingest_reviews.py to bake reviewed codes back in.
"""
import json
import logging
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT   = Path(__file__).parent.parent
REVIEW_FILE    = Path(__file__).parent / "output" / "manual_review_queue.csv"
STAGE2_FILE    = Path(__file__).parent / "output" / "stage2_icd10.csv"
STAGE3_FEAT    = Path(__file__).parent / "output" / "stage3_nct_features.csv"
STUDIES_FILE   = PROJECT_ROOT / "raw_data" / "studies.csv"
OUT_RANKED     = Path(__file__).parent / "output" / "review_ranked.csv"
OUT_TOP300     = Path(__file__).parent / "output" / "review_top300.csv"
OUT_QUICK_WINS = Path(__file__).parent / "output" / "review_quick_wins.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Therapeutic area keyword classifier ───────────────────────────────────────

_TA_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("oncology", re.compile(
        r"\b(cancer|carcinoma|tumor|neoplasm|leukemia|lymphoma|melanoma|sarcoma|"
        r"myeloma|glioma|glioblastoma|blastoma|adenocarcinoma|malignant|metastas|"
        r"oncol|solid tumor|hematolog)\b", re.I)),
    ("cardiovascular", re.compile(
        r"\b(heart|cardiac|coronary|myocardial|atrial|ventricular|arrhythmia|"
        r"hypertension|atherosclerosis|stroke|peripheral artery|aortic|"
        r"endocarditis|pericarditis|cardiomyopathy|angina|ischemic|"
        r"vascular|thrombosis|embolism|coagulation)\b", re.I)),
    ("infectious", re.compile(
        r"\b(hiv|aids|hepatitis|tuberculosis|malaria|influenza|covid|sars|"
        r"infection|bacterial|viral|fungal|parasit|sepsis|pneumonia|"
        r"antimicrobial|antibiotic|antiviral|immunodeficiency)\b", re.I)),
    ("cns", re.compile(
        r"\b(alzheimer|parkinson|dementia|epilepsy|seizure|migraine|"
        r"multiple sclerosis|amyotrophic|depression|schizophrenia|bipolar|"
        r"anxiety|psychiatric|neurolog|neuropath|cognitive|brain|spinal|"
        r"stroke|cerebral|mental disorder)\b", re.I)),
    ("metabolic", re.compile(
        r"\b(diabetes|insulin|obesity|metabolic|dyslipidemia|hyperlipidemia|"
        r"thyroid|adrenal|fatty liver|nash|nafld|glucose|cholesterol|"
        r"hyperglycemia|hypoglycemia|endocrine|renal|kidney)\b", re.I)),
    ("musculoskeletal", re.compile(
        r"\b(arthritis|rheumatoid|osteoporosis|osteoarthritis|lupus|spondylitis|"
        r"fibromyalgia|gout|bone|joint|muscle|tendon|ligament|spine|scoliosis)\b", re.I)),
    ("respiratory", re.compile(
        r"\b(asthma|copd|pulmonary|lung|respiratory|bronchitis|fibrosis|"
        r"emphysema|pleural|trachea|bronchial|dyspnea|cough|wheez)\b", re.I)),
    ("gastrointestinal", re.compile(
        r"\b(crohn|colitis|ibs|irritable bowel|gastric|esophageal|hepatic|"
        r"liver|pancreatic|colorectal|intestinal|bowel|colon|rectal|"
        r"gastroesophageal|reflux|celiac|cholangitis)\b", re.I)),
    ("rare_disease", re.compile(
        r"\b(rare|orphan|congenital|genetic|hereditary|lysosomal|gaucher|"
        r"fabry|pompe|niemann|wilson|phenylketonuria|haemophilia|hemophilia)\b", re.I)),
]


def classify_ta(text: str) -> str:
    for ta, pat in _TA_PATTERNS:
        if pat.search(text):
            return ta
    return "other"


def _norm(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def run() -> None:
    # ── Load inputs ──────────────────────────────────────────────────────────
    logger.info("Loading manual review queue (%s)", REVIEW_FILE.name)
    review = pd.read_csv(REVIEW_FILE, dtype=str)
    logger.info("  %s strings to rank", f"{len(review):,}")

    logger.info("Loading stage2 (nct_id → condition_normalized mapping)")
    stage2 = pd.read_csv(STAGE2_FILE,
                         usecols=["nct_id", "condition_normalized", "icd10_status"],
                         low_memory=False)

    logger.info("Loading stage3 nct features (has_ccsr)")
    feat = pd.read_csv(STAGE3_FEAT, usecols=["nct_id", "has_ccsr"], low_memory=False)
    zero_cov_ncts = set(feat.loc[feat["has_ccsr"] == 0, "nct_id"])
    logger.info("  nct_ids with zero CCSR coverage: %s", f"{len(zero_cov_ncts):,}")

    logger.info("Loading enrollment from studies.csv")
    studies = pd.read_csv(STUDIES_FILE, usecols=["nct_id", "enrollment"],
                          low_memory=False)
    studies["enrollment"] = pd.to_numeric(studies["enrollment"], errors="coerce").fillna(0)
    enroll_map = studies.set_index("nct_id")["enrollment"].to_dict()

    # ── Per-string statistics ─────────────────────────────────────────────────
    # Restrict stage2 to manual_review rows only
    manual_rows = stage2[stage2["icd10_status"] == "manual_review"].copy()
    manual_rows["enrollment"] = manual_rows["nct_id"].map(enroll_map).fillna(0)
    manual_rows["is_zero_cov"] = manual_rows["nct_id"].isin(zero_cov_ncts).astype(int)

    agg = manual_rows.groupby("condition_normalized").agg(
        nct_id_count=("nct_id", "nunique"),
        enrollment_total=("enrollment", "sum"),
        zero_coverage_ncts=("is_zero_cov", "sum"),
    ).reset_index()

    logger.info("Aggregated %s unique strings", f"{len(agg):,}")

    # ── Merge with review queue (has confidence + candidates_json) ────────────
    review["confidence"] = pd.to_numeric(review["confidence"], errors="coerce").fillna(0)
    ranked = review.merge(agg, on="condition_normalized", how="left")
    ranked["nct_id_count"] = ranked["nct_id_count"].fillna(0).astype(int)
    ranked["enrollment_total"] = ranked["enrollment_total"].fillna(0)
    ranked["zero_coverage_ncts"] = ranked["zero_coverage_ncts"].fillna(0).astype(int)

    # ── Therapeutic area ─────────────────────────────────────────────────────
    ranked["therapeutic_area"] = ranked["condition_normalized"].map(classify_ta)

    # ── Near-threshold flag ───────────────────────────────────────────────────
    # Gap to 0.65 threshold: smaller gap = closer to acceptance = easier win
    AUTO_ACCEPT = 0.65
    ranked["gap_to_threshold"] = (AUTO_ACCEPT - ranked["confidence"]).clip(lower=0)
    ranked["near_threshold"] = (ranked["gap_to_threshold"] <= 0.10).astype(int)

    # ── Composite priority score ──────────────────────────────────────────────
    ranked["nct_norm"]       = _norm(ranked["nct_id_count"].astype(float))
    ranked["enroll_norm"]    = _norm(ranked["enrollment_total"])
    ranked["zerocov_norm"]   = _norm(ranked["zero_coverage_ncts"].astype(float))
    ranked["nearthresh_norm"]= _norm((0.10 - ranked["gap_to_threshold"]).clip(lower=0))

    ranked["priority_score"] = (
        0.35 * ranked["nct_norm"]
        + 0.25 * ranked["enroll_norm"]
        + 0.30 * ranked["zerocov_norm"]
        + 0.10 * ranked["nearthresh_norm"]
    ).round(4)

    ranked = ranked.sort_values("priority_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    # ── Add reviewer columns for filling in ──────────────────────────────────
    ranked["reviewer_code"] = ""
    ranked["reviewer_name"] = ""
    ranked["reviewer_notes"] = ""

    # Clean column order
    out_cols = [
        "rank", "condition_normalized", "priority_score",
        "nct_id_count", "enrollment_total", "zero_coverage_ncts",
        "therapeutic_area", "near_threshold", "confidence",
        "selected_code", "selected_name",
        "reviewer_code", "reviewer_name", "reviewer_notes",
        "candidates_json",
    ]
    ranked = ranked[out_cols]

    # ── Outputs ───────────────────────────────────────────────────────────────
    ranked.to_csv(OUT_RANKED, index=False)
    logger.info("Saved review_ranked.csv (%s rows)", f"{len(ranked):,}")

    top300 = ranked.head(300)
    top300.to_csv(OUT_TOP300, index=False)
    logger.info("Saved review_top300.csv")

    quick_wins = ranked[ranked["near_threshold"] == 1].sort_values(
        ["nct_id_count", "confidence"], ascending=[False, False]
    )
    quick_wins.to_csv(OUT_QUICK_WINS, index=False)
    logger.info("Saved review_quick_wins.csv (%s strings within 0.10 of threshold)", f"{len(quick_wins):,}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n=== TOP 30 PRIORITY STRINGS ===")
    cols_print = ["rank", "condition_normalized", "nct_id_count",
                  "enrollment_total", "zero_coverage_ncts", "therapeutic_area",
                  "confidence", "priority_score"]
    print(top300[cols_print].head(30).to_string(index=False))

    logger.info("\n=== QUICK WINS (confidence 0.55–0.64, close to threshold) ===")
    print(quick_wins[cols_print].head(30).to_string(index=False))

    logger.info("\n=== THERAPEUTIC AREA BREAKDOWN (top 300) ===")
    print(top300["therapeutic_area"].value_counts().to_string())

    logger.info("\n=== COVERAGE IMPACT ESTIMATE ===")
    top100_ncts = manual_rows[
        manual_rows["condition_normalized"].isin(ranked.head(100)["condition_normalized"])
    ]
    unique_ncts_top100 = top100_ncts["nct_id"].nunique()
    zero_cov_top100 = top100_ncts[top100_ncts["is_zero_cov"] == 1]["nct_id"].nunique()
    logger.info("Top 100 strings affect %s nct_ids (%s currently zero-coverage)",
                f"{unique_ncts_top100:,}", f"{zero_cov_top100:,}")

    top300_ncts = manual_rows[
        manual_rows["condition_normalized"].isin(ranked.head(300)["condition_normalized"])
    ]
    unique_ncts_top300 = top300_ncts["nct_id"].nunique()
    zero_cov_top300 = top300_ncts[top300_ncts["is_zero_cov"] == 1]["nct_id"].nunique()
    logger.info("Top 300 strings affect %s nct_ids (%s currently zero-coverage)",
                f"{unique_ncts_top300:,}", f"{zero_cov_top300:,}")


if __name__ == "__main__":
    run()
