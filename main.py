"""
Run full pipeline: 1. Download → 2. Data exploration → 3. Preprocessing → 4. Modeling.

- **Default (step 4):** single baseline primary regression → ``results/regression_report.txt``
- **``--planning-experiment``:** full staged planning-time run (primary + post-primary strict + combined
  forecast + late-risk + deviation) → ``results/experiments/<UTC>/``

Checkpointing: download scripts skip if data is up to date (see raw_data/.checkpoints/).
Use --skip-download to skip downloads when you already have raw_data.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_SCRIPTS = [
    PROJECT_ROOT / "1_scripts" / "download_studies.py",
    PROJECT_ROOT / "1_scripts" / "download_sponsors.py",
    PROJECT_ROOT / "1_scripts" / "download_browse_conditions.py",
    PROJECT_ROOT / "1_scripts" / "download_interventions.py",
    PROJECT_ROOT / "1_scripts" / "download_eligibilities.py",
    PROJECT_ROOT / "1_scripts" / "download_calculated_values.py",
    PROJECT_ROOT / "1_scripts" / "download_facilities.py",
    PROJECT_ROOT / "1_scripts" / "download_countries.py",
    PROJECT_ROOT / "1_scripts" / "download_designs.py",
    PROJECT_ROOT / "1_scripts" / "download_design_groups.py",
    PROJECT_ROOT / "1_scripts" / "download_design_outcomes.py",
    PROJECT_ROOT / "1_scripts" / "download_browse_interventions.py",
]


def run_script(script_path: Path, step_name: str, quiet: bool = False) -> bool:
    """Run a Python script, return True if successful."""
    print(f"\n{step_name}")
    kwargs: dict = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        **kwargs,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: download → explore → preprocess → regression or planning experiment"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download steps (use when raw_data already exists)",
    )
    parser.add_argument(
        "--planning-experiment",
        action="store_true",
        help="After preprocess, run full planning-time experiment (replaces single train_regression step)",
    )
    parser.add_argument(
        "--experiment-dry-run",
        action="store_true",
        help="With --planning-experiment, print subprocess commands only (no training)",
    )
    parser.add_argument(
        "--late-quantile",
        type=float,
        default=0.75,
        help="With --planning-experiment: late-risk label quantile (default: 0.75)",
    )
    args = parser.parse_args()

    if args.planning_experiment and not (0.0 < args.late_quantile < 1.0):
        print("ERROR: --late-quantile must be in (0, 1)")
        sys.exit(1)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = RESULTS_DIR / f"run_{run_timestamp}_results.txt"

    if args.skip_download:
        print("Skipping download steps (--skip-download)")

    if not args.skip_download:
        print("\n1. Download")
        for script_path in DOWNLOAD_SCRIPTS:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                print("ERROR: 1. Download failed")
                sys.exit(1)

    explore_preprocess = [
        (PROJECT_ROOT / "2_data_exploration" / "run_all.py", "2. Data exploration", True),
        (PROJECT_ROOT / "3_preprocessing" / "preprocess.py", "3. Preprocessing", True),
    ]

    for script_path, step_name, quiet in explore_preprocess:
        if not run_script(script_path, step_name, quiet=quiet):
            print(f"ERROR: {step_name} failed")
            sys.exit(1)

    if args.planning_experiment:
        sys.path.insert(0, str(PROJECT_ROOT / "4_regression"))
        from planning_experiment_runner import run_experiment

        print("\n4. Planning-time experiment (staged models)")
        exp_dir = run_experiment(dry_run=args.experiment_dry_run, late_quantile=args.late_quantile)
        if exp_dir is not None:
            results_path.write_text(
                "Planning-time experiment completed.\n\n"
                f"Experiment directory:\n  {exp_dir}\n\n"
                f"See experiment_summary.txt and experiment.log in that folder.\n",
                encoding="utf-8",
            )
        else:
            results_path.write_text(
                "Planning-time experiment — dry run only (no artifacts).\n"
                "Re-run without --experiment-dry-run to execute training.\n",
                encoding="utf-8",
            )
        print(f"\nRun summary pointer: {results_path}")
        return

    print("\n4. Regression (baseline primary_completion)")
    if not run_script(
        PROJECT_ROOT / "4_regression" / "train_regression.py",
        "4. Regression",
        quiet=False,
    ):
        print("ERROR: 4. Regression failed")
        sys.exit(1)

    regression_report = RESULTS_DIR / "regression_report.txt"
    if regression_report.exists():
        results_path.write_text(regression_report.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nFinal results saved to: {results_path}")


if __name__ == "__main__":
    main()
