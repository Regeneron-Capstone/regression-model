"""
Stage 3 — CCSR Join

Input:  2_condition_mapping/output/stage2_icd10.csv
        raw_data/condition_mapping_data/DXCCSR_v2026-1.csv
Output: 2_condition_mapping/output/stage3_with_ccsr.csv  — long form (one row per slot)
        2_condition_mapping/output/stage3_nct_features.csv — one row per nct_id (model input)

For each auto-accepted ICD-10 code, looks up:
  - Default CCSR category (IP) — used as the primary CCSR code per slot
  - CCSR category 1–6 descriptions (carried along for interpretability)

Final feature table (stage3_nct_features.csv) columns:
  nct_id
  ccsr_slot1, ccsr_slot1_desc        — CCSR for the highest-priority condition
  ccsr_slot2, ccsr_slot2_desc        — CCSR for second slot (if any)
  ccsr_slot3, ccsr_slot3_desc        — CCSR for third slot (if any)
  has_ccsr                           — 1 if at least one slot has a CCSR code
  metastatic_flag                    — any slot has metastatic/advanced flag
  relapsed_refractory_flag
  line_of_therapy                    — max line seen across slots
  pediatric_flag
  adult_flag
  biomarker_flag
  tier_b_only_flag                   — nct_id has only Tier B conditions (no specific MeSH)
"""
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
STAGE2_OUT   = Path(__file__).parent / "output" / "stage2_icd10.csv"
DXCCSR_PATH  = PROJECT_ROOT / "raw_data" / "condition_mapping_data" / "DXCCSR_v2026-1.csv"
OUTPUT_LONG  = Path(__file__).parent / "output" / "stage3_with_ccsr.csv"
OUTPUT_FEAT  = Path(__file__).parent / "output" / "stage3_nct_features.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CCSR_NUM_COLS = 6  # DXCCSR has up to 6 CCSR categories per code
XXX000 = "XXX000"  # CCSR unclassifiable code — treat as missing


def load_dxccsr() -> pd.DataFrame:
    """Load DXCCSR, strip embedded quotes, return clean DataFrame."""
    df = pd.read_csv(DXCCSR_PATH, dtype=str, low_memory=False)
    df.columns = [c.strip("'\" ") for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip("'\" ").str.strip()
    df = df[df["ICD-10-CM CODE"].notna() & (df["ICD-10-CM CODE"].str.strip() != "")]
    logger.info("DXCCSR: %s codes loaded", f"{len(df):,}")
    return df


def normalize_icd10_code(code: str) -> str:
    """Remove dots so alias-dict codes (E25.0) match DXCCSR format (E250)."""
    if pd.isna(code) or not isinstance(code, str):
        return ""
    return code.replace(".", "").strip().upper()


def build_ccsr_lookup(dxccsr: pd.DataFrame) -> pd.DataFrame:
    """Return a lookup table keyed on normalized ICD-10 code."""
    lookup = dxccsr.copy()
    lookup["icd10_key"] = lookup["ICD-10-CM CODE"].str.upper()

    # Collect columns we want to carry through
    keep = ["icd10_key", "ICD-10-CM CODE DESCRIPTION",
            "Default CCSR CATEGORY IP", "Default CCSR CATEGORY DESCRIPTION IP"]
    for i in range(1, CCSR_NUM_COLS + 1):
        cat_col  = f"CCSR CATEGORY {i}"
        desc_col = f"CCSR CATEGORY {i} DESCRIPTION"
        if cat_col in lookup.columns:
            keep.extend([cat_col, desc_col])

    lookup = lookup[keep].copy()

    # Blank out XXX000 (unclassifiable) in default IP column
    mask_xxx = lookup["Default CCSR CATEGORY IP"] == XXX000
    lookup.loc[mask_xxx, "Default CCSR CATEGORY IP"] = None
    lookup.loc[mask_xxx, "Default CCSR CATEGORY DESCRIPTION IP"] = None

    return lookup.set_index("icd10_key")


def join_ccsr(stage2: pd.DataFrame, ccsr_lookup: pd.DataFrame) -> pd.DataFrame:
    """Add CCSR columns to stage2 rows that have auto_accepted ICD-10 codes."""
    stage2 = stage2.copy()
    stage2["icd10_key"] = stage2["icd10_code"].map(
        lambda x: normalize_icd10_code(x) if pd.notna(x) else ""
    )

    joined = stage2.join(ccsr_lookup, on="icd10_key", how="left")

    # For rows with no auto-accepted code, clear CCSR fields
    mask_no_code = stage2["icd10_status"] != "auto_accepted"
    ccsr_cols = [c for c in ccsr_lookup.columns if c != "icd10_key"]
    joined.loc[mask_no_code, ccsr_cols] = None

    logger.info("Rows with CCSR default IP assigned: %s",
                f"{joined['Default CCSR CATEGORY IP'].notna().sum():,}")
    return joined


def build_nct_features(stage3: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from long format to one-row-per-nct_id feature table.

    CCSR slots: for each nct_id, take the first non-null Default CCSR CATEGORY IP
    per slot_rank (1, 2, 3).  Flag columns: max across all slots.
    ccsr_domain: first 3 characters of ccsr_slot1 (coarse auxiliary feature;
    ccsr_slot1/2/3 remain the primary condition features for modeling).
    """
    # ── CCSR slots (vectorized, compacted left) ───────────────────────────────
    # Take all rows with a mapped CCSR code, sort by slot_rank (ascending =
    # highest priority first), then assign compacted sequential slot numbers
    # 1→2→3 within each nct_id.  This closes gaps caused by unmapped conditions
    # at slot_rank=1 or slot_rank=2 so the exported columns are always contiguous.
    slots_df = (
        stage3.loc[
            stage3["Default CCSR CATEGORY IP"].notna(),
            ["nct_id", "slot_rank",
             "Default CCSR CATEGORY IP",
             "Default CCSR CATEGORY DESCRIPTION IP"],
        ]
        .sort_values(["nct_id", "slot_rank"])
        .drop_duplicates(subset=["nct_id", "slot_rank"], keep="first")
        .copy()
    )
    # Compact: number each mapped row 1, 2, 3, … within its nct_id
    slots_df["compacted_slot"] = slots_df.groupby("nct_id").cumcount() + 1
    slots_df = slots_df[slots_df["compacted_slot"] <= 3]

    ccsr_wide = slots_df.pivot(
        index="nct_id",
        columns="compacted_slot",
        values=["Default CCSR CATEGORY IP", "Default CCSR CATEGORY DESCRIPTION IP"],
    )
    ccsr_wide.columns = [
        f"ccsr_slot{int(slot)}" if field == "Default CCSR CATEGORY IP"
        else f"ccsr_slot{int(slot)}_desc"
        for field, slot in ccsr_wide.columns
    ]
    for slot in (1, 2, 3):
        for suffix in ("", "_desc"):
            col = f"ccsr_slot{slot}{suffix}"
            if col not in ccsr_wide.columns:
                ccsr_wide[col] = None

    # ── Binary flags (vectorized) ─────────────────────────────────────────────
    flag_cols = [
        "metastatic_flag", "relapsed_refractory_flag",
        "pediatric_flag", "adult_flag", "biomarker_flag",
    ]
    for col in flag_cols:
        stage3[col] = pd.to_numeric(stage3[col], errors="coerce").fillna(0)

    flags = stage3.groupby("nct_id")[flag_cols].max()

    tb_only = (
        stage3.groupby("nct_id")[["tier_b_only", "tier_b_suppressed"]]
        .apply(lambda g: int(
            g["tier_b_only"].fillna(0).astype(int).max() == 1
            and g["tier_b_suppressed"].fillna(0).astype(int).max() == 0
        ), include_groups=False)
        .rename("tier_b_only_flag")
    )

    lot = (
        pd.to_numeric(stage3["line_of_therapy"], errors="coerce")
        .groupby(stage3["nct_id"]).max()
        .rename("line_of_therapy")
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    all_ncts = pd.Index(stage3["nct_id"].unique(), name="nct_id")
    out = (
        pd.DataFrame(index=all_ncts)
        .join(ccsr_wide, how="left")
        .join(flags,    how="left")
        .join(tb_only,  how="left")
        .join(lot,      how="left")
        .reset_index()
    )

    # After compaction slot1 is always the first filled, but derive from any slot
    # to be defensive.
    out["has_ccsr"] = (
        out[["ccsr_slot1", "ccsr_slot2", "ccsr_slot3"]].notna().any(axis=1)
    ).astype(int)

    # Coarse 3-letter domain prefix — auxiliary feature; experiment whether it
    # adds signal beyond ccsr_slot1 directly.
    out["ccsr_domain"] = out["ccsr_slot1"].str[:3]

    for col in flag_cols + ["tier_b_only_flag"]:
        out[col] = out[col].fillna(0).astype(int)

    col_order = [
        "nct_id",
        "ccsr_slot1", "ccsr_slot1_desc",
        "ccsr_slot2", "ccsr_slot2_desc",
        "ccsr_slot3", "ccsr_slot3_desc",
        "ccsr_domain",
        "has_ccsr",
        "metastatic_flag", "relapsed_refractory_flag", "line_of_therapy",
        "pediatric_flag", "adult_flag", "biomarker_flag", "tier_b_only_flag",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    logger.info("nct_id feature table: %s rows, %s with CCSR (%.1f%%)",
                f"{len(out):,}",
                f"{out['has_ccsr'].sum():,}",
                100 * out["has_ccsr"].mean())
    return out


def run() -> None:
    logger.info("Loading Stage 2 output")
    stage2 = pd.read_csv(STAGE2_OUT, low_memory=False)
    logger.info("Stage 2: %s rows, %s nct_ids",
                f"{len(stage2):,}", f"{stage2['nct_id'].nunique():,}")

    dxccsr = load_dxccsr()
    ccsr_lookup = build_ccsr_lookup(dxccsr)

    # Long-form join
    stage3 = join_ccsr(stage2, ccsr_lookup)
    stage3.to_csv(OUTPUT_LONG, index=False)
    logger.info("Saved stage3_with_ccsr.csv → %s rows", f"{len(stage3):,}")

    # CCSR category distribution
    ccsr_dist = stage3["Default CCSR CATEGORY IP"].value_counts().head(20)
    logger.info("Top 20 CCSR categories (default IP):\n%s", ccsr_dist.to_string())

    # Per-nct_id feature table
    logger.info("Building nct_id feature table ...")
    nct_features = build_nct_features(stage3)
    nct_features.to_csv(OUTPUT_FEAT, index=False)
    logger.info("Saved stage3_nct_features.csv → %s rows", f"{len(nct_features):,}")

    # Summary
    logger.info("CCSR slot fill rates:")
    for slot in ("ccsr_slot1", "ccsr_slot2", "ccsr_slot3"):
        n = nct_features[slot].notna().sum()
        logger.info("  %s: %s (%.1f%%)", slot, f"{n:,}", 100 * n / len(nct_features))

    logger.info("Top 20 ccsr_domain values (slot1 prefix):\n%s",
                nct_features["ccsr_domain"].value_counts().head(20).to_string())

    logger.info("Flag distributions:")
    for flag in ("metastatic_flag", "relapsed_refractory_flag",
                 "pediatric_flag", "adult_flag", "biomarker_flag", "tier_b_only_flag"):
        n = nct_features[flag].sum()
        logger.info("  %s: %s (%.1f%%)", flag, f"{n:,}", 100 * n / len(nct_features))


if __name__ == "__main__":
    run()
