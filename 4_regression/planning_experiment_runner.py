"""
Planning-time experiment orchestration (used by ``main.py --planning-experiment`` or run directly).

Runs baseline primary → post-primary strict → combined forecast → late-risk → combined deviation;
artifacts under ``results/experiments/<UTC>/``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

RANDOM_STATE = 42


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _tee_run(
    cmd: list[str],
    *,
    log_file: Path,
    env: dict[str, str],
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    banner = f"\n{'=' * 72}\n$ {' '.join(cmd)}\n{'=' * 72}\n"
    print(banner, end="")
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(banner)
        p = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    reg = str(PROJECT_ROOT / "4_regression")
    env["PYTHONPATH"] = reg if not prev else f"{reg}{os.pathsep}{prev}"
    env["PYTHONHASHSEED"] = "0"
    return env


def _write_experiment_summary(exp_dir: Path, log_path: Path) -> None:
    lines: list[str] = []
    lines.append("PLANNING-TIME EXPERIMENT — SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Experiment directory: {exp_dir}")
    lines.append(f"UTC run id (folder name): {exp_dir.name}")
    lines.append(f"Full log: {log_path.name}")
    lines.append(f"Determinism: PYTHONHASHSEED=0; random_state={RANDOM_STATE} for training/classifier/splits.")
    lines.append("")

    artifacts = [
        "regression_report_baseline_primary.txt",
        "regression_report_post_primary_strict_planning.txt",
        "stage_models/",
        "combined_duration_predictions.csv",
        "late_risk_classification_report.txt",
        "late_risk_predictions.csv",
        "deviation_combined_predictions.csv",
        "deviation_combined_summary.txt",
    ]
    lines.append("Artifacts (expected)")
    lines.append("-" * 72)
    for name in artifacts:
        p = exp_dir / name
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            lines.append(f"  {name}  ({n} files)" if p.exists() else f"  {name}  (missing)")
        elif p.exists():
            lines.append(f"  {name}  ({p.stat().st_size:,} bytes)")
        else:
            lines.append(f"  {name}  (missing)")
    lines.append("")

    def _tail(path: Path, n: int = 40) -> list[str]:
        if not path.exists():
            return [f"  (file not found: {path.name})"]
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        chunk = text[-n:] if len(text) > n else text
        return ["  " + ln for ln in chunk]

    lr = exp_dir / "late_risk_classification_report.txt"
    if lr.exists():
        lines.append("Late-risk classifier (excerpt: val/test metrics)")
        lines.append("-" * 72)
        for ln in lr.read_text(encoding="utf-8", errors="replace").splitlines():
            s = ln.strip()
            if any(
                k in s
                for k in ("Split:", "precision", "recall", "F1", "ROC-AUC", "PR-AUC", "positive rate")
            ):
                lines.append("  " + ln)
        lines.append("")

    dev = exp_dir / "deviation_combined_summary.txt"
    if dev.exists():
        lines.append("Combined forecast deviation (first 120 lines)")
        lines.append("-" * 72)
        for ln in dev.read_text(encoding="utf-8", errors="replace").splitlines()[:120]:
            lines.append("  " + ln)
        lines.append("")

    for label, fname in (
        ("Baseline primary regression (tail)", "regression_report_baseline_primary.txt"),
        ("Post-primary strict regression (tail)", "regression_report_post_primary_strict_planning.txt"),
    ):
        fp = exp_dir / fname
        lines.append(label)
        lines.append("-" * 72)
        lines.extend(_tail(fp, 28))
        lines.append("")

    out = exp_dir / "experiment_summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


def run_experiment(*, dry_run: bool, late_quantile: float) -> Path | None:
    run_id = _utc_run_id()
    exp_dir = EXPERIMENTS_DIR / run_id
    log_path = exp_dir / "experiment.log"

    env = _child_env()
    py = sys.executable

    steps: list[tuple[str, list[str]]] = [
        (
            "1. Baseline primary_completion training",
            [
                py,
                str(PROJECT_ROOT / "4_regression" / "train_regression.py"),
                "--target",
                "primary_completion",
                "--feature-policy",
                "baseline",
                "--report",
                str(exp_dir / "regression_report_baseline_primary.txt"),
                "--random-state",
                str(RANDOM_STATE),
            ],
        ),
        (
            "2. post_primary_completion (strict_planning) training",
            [
                py,
                str(PROJECT_ROOT / "4_regression" / "train_regression.py"),
                "--target",
                "post_primary_completion",
                "--feature-policy",
                "strict_planning",
                "--report",
                str(exp_dir / "regression_report_post_primary_strict_planning.txt"),
                "--random-state",
                str(RANDOM_STATE),
            ],
        ),
        (
            "3. Combined forecast (stage models + CSV)",
            [
                py,
                str(PROJECT_ROOT / "4_regression" / "combined_duration_forecast.py"),
                "--models-dir",
                str(exp_dir / "stage_models"),
                "--output",
                str(exp_dir / "combined_duration_predictions.csv"),
                "--refit",
            ],
        ),
        (
            "4. Late-risk classification",
            [
                py,
                str(PROJECT_ROOT / "4_regression" / "late_risk_classifier.py"),
                "--report",
                str(exp_dir / "late_risk_classification_report.txt"),
                "--predictions",
                str(exp_dir / "late_risk_predictions.csv"),
                "--random-state",
                str(RANDOM_STATE),
                "--late-quantile",
                str(late_quantile),
            ],
        ),
        (
            "5. Deviation analysis (combined predictions)",
            [
                py,
                str(PROJECT_ROOT / "5_deviation" / "deviation_analysis.py"),
                "--target",
                "combined",
                "--combined-csv",
                str(exp_dir / "combined_duration_predictions.csv"),
                "--output-csv",
                str(exp_dir / "deviation_combined_predictions.csv"),
                "--output-summary",
                str(exp_dir / "deviation_combined_summary.txt"),
                "--splits",
                "test",
                "--random-state",
                str(RANDOM_STATE),
            ],
        ),
    ]

    print(f"Experiment directory: {exp_dir}\nLog file: {log_path.name}\n")

    if dry_run:
        for title, cmd in steps:
            print(title)
            print(" ", " ".join(cmd))
        print("\nDry run only — no directory created; no commands executed.")
        return None

    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"Planning-time experiment {run_id}\n"
        f"Started: {datetime.now(timezone.utc).isoformat()}\n"
        f"PYTHONHASHSEED=0  random_state={RANDOM_STATE}\n\n",
        encoding="utf-8",
    )

    for title, cmd in steps:
        print(f"\n>>> {title}")
        _tee_run(cmd, log_file=log_path, env=env)

    _write_experiment_summary(exp_dir, log_path)
    print(f"\nDone. All outputs under:\n  {exp_dir}")
    return exp_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Run full planning-time modeling experiment into results/experiments/")
    p.add_argument("--dry-run", action="store_true", help="Print steps only; do not execute.")
    p.add_argument(
        "--late-quantile",
        type=float,
        default=0.75,
        help="Late-risk label quantile (passed to late_risk_classifier, default: 0.75)",
    )
    args = p.parse_args()
    if not (0.0 < args.late_quantile < 1.0):
        sys.exit("--late-quantile must be in (0, 1)")
    run_experiment(dry_run=args.dry_run, late_quantile=args.late_quantile)


if __name__ == "__main__":
    main()
