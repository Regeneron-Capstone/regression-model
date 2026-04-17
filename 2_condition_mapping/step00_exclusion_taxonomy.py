"""
Stage 0 — Exclusion Taxonomy

Input:  raw_data/conditions_raw.csv
Output: 2_condition_mapping/output/stage0_conditions.csv

Buckets (checked in priority order):
  corrupted    — data artifacts: #NAME?, leading _, leading dash
  pk_admin     — pharmacokinetic / administrative concepts
  demographic  — age range descriptors, healthy volunteer labels
  drug_term    — drug codes, -mab/-nib/-zumab suffixes, vaccine names, salt forms
  staging_only — bare line-of-therapy strings with no surviving disease name
  disease      — everything else → proceeds to Stage 1
"""
import re
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "raw_data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Compiled patterns ─────────────────────────────────────────────────────────

_PK_ADMIN_RE = re.compile(
    r"\b("
    r"bioavailability|pharmacokinetic|pharmacodynamic"
    r"|absorption"          # \b prevents matching "malabsorption"
    r"|excretion"
    r"|maximum\s+tolerated\s+dose"
    r"|dose[\s-]finding"
    r"|abuse\s+potential"
    r"|abuse\s+liability"
    r"|drug[\s-]interaction"
    r")\b",
    re.IGNORECASE,
)

_DEMOGRAPHIC_RE = re.compile(
    r"""
      \d+\s*[-–]\s*\d+\s*year            # 18-42 year
    | \d+\s+years?\s+old\b               # 70 years old
    | \d+\s+years?\s+(and\s+)?(over|older|above)\b
    | \bhealthy\s+(volunteers?|subjects?|adults?|individuals?|participants?|persons?|controls?|people)\b
    | ^\s*healthy\s*$                     # bare "healthy" — healthy volunteer label
    | \bnormal\s+(volunteers?|subjects?|controls?|participants?|adults?)\b
    | ^\s*normal\s+(healthy\s+)?(volunteer|subject|control|adult)s?\s*$
    | vaccinia[\s-]na[iï]ve
    | vaccine[\s-]na[iï]ve
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Biologic / small-molecule name suffixes
_DRUG_SUFFIX_RE = re.compile(
    r"-("
    r"mab|nib|zumab|tinib|ciclib|rafenib|prazole|fenib|umab"
    r"|lizumab|tumumab|ximab|arib|parib|sidenib|vir|tide"
    r")\b",
    re.IGNORECASE,
)

# Drug / compound codes: abbv-154, mk-1234, ncx-470a
# Requires 3+ digit number to avoid matching disease codes like covid-19, hiv-1
_DRUG_CODE_RE = re.compile(r"^[a-z]{1,5}-\d{3,}[a-z0-9]?$", re.IGNORECASE)

# Radiopharmaceuticals: 18f-, 68ga-, 99mtc-, 64cu-, 89zr-, 177lu-, 131i-
# Uses an explicit element-symbol alternation to avoid collisions with:
#   chromosome arm notation (11q-, 17p-, 22q-)
#   ordinal suffixes     (1st-, 2nd-, 3rd-)
_RADIOTRACER_RE = re.compile(
    r"^\d{1,3}m?(?:f|ga|tc|in\b|cu|zr|lu|y(?=[-\s])|i(?=[-\s])|c(?=[-\s]))[-\s]",
    re.IGNORECASE,
)

# Vaccine name indicators (singular and plural)
_VACCINE_RE = re.compile(r"(\d+-valent\b|(?<!\w)vaccines?\b)", re.IGNORECASE)

# Drug salt / formulation markers
_DRUG_SALT_RE = re.compile(
    r"\b(hydrochloride|acetate|phosphate|sulfate|mesylate"
    r"|tartrate|fumarate|maleate|chloride|bromide)\b",
    re.IGNORECASE,
)

# Drug class names ending in mechanism descriptor
_DRUG_CLASS_RE = re.compile(
    r"\b(inhibitors?|agonists?|antagonists?|blockers?|modulators?)\s*$",
    re.IGNORECASE,
)

# Staging words used to test whether anything meaningful remains
_STAGING_STRIP_RE = re.compile(
    r"\b("
    r"advanced|metastatic|relapsed|refractory"
    r"|first[\s-]line|second[\s-]line|third[\s-]line"
    r"|\d+l\+?|r/r"
    r")\b",
    re.IGNORECASE,
)


# ── Individual check functions ────────────────────────────────────────────────
# Each returns a non-empty reason string on match, empty string otherwise.

def _is_corrupted(text: str) -> str:
    s = text.strip()
    if not s or s.lower() in {"n/a", "na", "none", "null", "unknown", "not specified"}:
        return "empty_or_null"
    if "#name?" in s.lower():
        return "excel_formula_error"
    if s.startswith("_"):
        return "leading_underscore"
    # Leading dash followed immediately by a letter: "-unhealthy" is an artifact;
    # "non-small" is not (it doesn't start with dash).
    if re.match(r"^-[a-z]", s):
        return "leading_dash_artifact"
    return ""


def _is_pk_admin(text: str) -> str:
    m = _PK_ADMIN_RE.search(text)
    return f"pk_term:{m.group().strip().lower()}" if m else ""


def _is_demographic(text: str) -> str:
    return "demographic_descriptor" if _DEMOGRAPHIC_RE.search(text) else ""


def _is_drug_term(text: str) -> str:
    t = text.strip()
    if _DRUG_CODE_RE.match(t):
        return "drug_code"
    if _RADIOTRACER_RE.match(t):
        return "radiotracer"
    if _DRUG_SUFFIX_RE.search(t):
        return "biologic_drug_suffix"
    if _VACCINE_RE.search(t) and len(t.split()) <= 6:
        return "vaccine_name"
    if _DRUG_SALT_RE.search(t):
        return "drug_salt_form"
    if _DRUG_CLASS_RE.search(t):
        return "drug_class_name"
    return ""


def _is_staging_only(text: str) -> str:
    t = text.strip()
    # Bare line code with no disease: "2l+", "3l"
    if re.match(r"^\d+[lL]\+?\s*$", t):
        return "bare_line_of_therapy"
    # Only flag if staging words were actually found AND nothing else survives.
    # Check presence BEFORE the punctuation-cleanup sub so that punctuation removal
    # alone (e.g. trailing comma in "λz,") does not falsely set the flag.
    after_staging = _STAGING_STRIP_RE.sub("", t)
    if after_staging == t:          # no staging words were present
        return ""
    cleaned = re.sub(r"[\s/,+]+", " ", after_staging).strip()
    if len(cleaned) <= 2:
        return "staging_only_no_disease_name"
    return ""


_CHECKS: list[tuple] = [
    (_is_corrupted,    "corrupted"),
    (_is_pk_admin,     "pk_admin"),
    (_is_demographic,  "demographic"),
    (_is_drug_term,    "drug_term"),
    (_is_staging_only, "staging_only"),
]


def classify(text: str) -> tuple[str, str]:
    """Return (exclusion_bucket, reason). bucket='disease' means no exclusion."""
    for fn, bucket in _CHECKS:
        reason = fn(str(text))
        if reason:
            return bucket, reason
    return "disease", "no_exclusion_pattern_matched"


# ── Runner ────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    logger.info("Loading conditions_raw.csv")
    df = pd.read_csv(RAW_DATA / "conditions_raw.csv", low_memory=False)
    df = df.rename(columns={"condition_downcase": "condition_raw"})
    logger.info("Loaded %s rows, %s unique nct_ids",
                f"{len(df):,}", f"{df['nct_id'].nunique():,}")

    results = df["condition_raw"].map(
        lambda t: classify(str(t)) if pd.notna(t) else ("corrupted", "null_value")
    )
    df["exclusion_bucket"], df["bucket_reason"] = zip(*results)

    counts = df["exclusion_bucket"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    summary = pd.DataFrame({"count": counts, "pct": pct})
    logger.info("Bucket distribution:\n%s", summary.to_string())

    disease_nct = df[df["exclusion_bucket"] == "disease"]["nct_id"].nunique()
    total_nct = df["nct_id"].nunique()
    logger.info(
        "%s of %s unique nct_ids have ≥1 disease-bucket condition (%.1f%%)",
        f"{disease_nct:,}", f"{total_nct:,}", 100 * disease_nct / total_nct,
    )

    out = OUTPUT_DIR / "stage0_conditions.csv"
    df.to_csv(out, index=False)
    logger.info("Saved → %s", out)
    return df


if __name__ == "__main__":
    run()
