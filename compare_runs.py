# Copyright (c) 2026 Isabel Moore. All rights reserved.
"""Compare evaluation metrics across all training runs."""

import json
import os
from glob import glob

RUNS_ROOT = "runs/UAVGuidance_RecurrentPPO"


def load_metrics(run_dir):
    """Load eval/metrics.json from a training run directory.

    Args:
        run_dir: Path to a single training run directory.

    Returns:
        Parsed JSON dict, or None if file not found.
    """
    metrics_path = os.path.join(run_dir, "eval", "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return json.load(f)


def main():
    """Print a comparison table of evaluation metrics across all training runs."""
    run_dirs = sorted(glob(os.path.join(RUNS_ROOT, "*")))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]

    if not run_dirs:
        print(f"No runs found in {RUNS_ROOT}/")
        return

    # Load all metrics
    runs = []
    for run_dir in run_dirs:
        metrics = load_metrics(run_dir)
        if metrics is None:
            print(f"  No eval/metrics.json in {os.path.basename(run_dir)} — run evaluate_meta.py first")
            continue
        runs.append(metrics)

    if not runs:
        print("No evaluation results found. Run: python3 evaluate_meta.py")
        return

    # ---- Header ----
    print()
    print("=" * 100)
    print("COMPARISON OF TRAINING RUNS")
    print("=" * 100)

    # Short names for display (extract date portion)
    names = []
    for r in runs:
        name = r["run_name"]
        short = name.split("_")[0] + "_" + name.split("_")[1] if "_" in name else name
        names.append(short)

    # ---- Main comparison table ----
    col_w = max(18, max(len(n) for n in names) + 2)
    label_w = 22

    def row(label, values, fmt="{}", highlight_min=False, highlight_max=False):
        """Print one row of the comparison table, optionally starring the best value.

        Args:
            label: Row label string.
            values: List of metric values (one per run).
            fmt: Format string applied to each value.
            highlight_min: Star the minimum value.
            highlight_max: Star the maximum value.
        """
        formatted = [fmt.format(v) for v in values]
        # Highlight best
        if highlight_min and all(isinstance(v, (int, float)) for v in values):
            best_idx = values.index(min(values))
            formatted[best_idx] = f"*{formatted[best_idx]}*"
        if highlight_max and all(isinstance(v, (int, float)) for v in values):
            best_idx = values.index(max(values))
            formatted[best_idx] = f"*{formatted[best_idx]}*"
        cols = "".join(f"{v:>{col_w}}" for v in formatted)
        print(f"  {label:<{label_w}}{cols}")

    # Header row
    header = "".join(f"{n:>{col_w}}" for n in names)
    print(f"\n  {'Metric':<{label_w}}{header}")
    print(f"  {'-'*label_w}{('-'*col_w)*len(names)}")

    # Model info
    row("Model", [os.path.basename(r["model_path"]).replace(".zip", "") for r in runs])
    row("Episodes", [r["n_episodes"] for r in runs])
    print()

    # Hit rate
    row("Hit rate", [f"{r['hit_rate']:.1%}" for r in runs])
    row("Hit count", [r["hit_count"] for r in runs], highlight_max=True)
    print()

    # Miss distance
    miss = [r["miss_distance"] for r in runs]
    row("Miss mean (m)", [m["mean"] for m in miss], fmt="{:.0f}", highlight_min=True)
    row("Miss median (m)", [m["median"] for m in miss], fmt="{:.0f}", highlight_min=True)
    row("Miss min (m)", [m["min"] for m in miss], fmt="{:.0f}", highlight_min=True)
    row("Miss max (m)", [m["max"] for m in miss], fmt="{:.0f}")
    row("Miss p25 (m)", [m["p25"] for m in miss], fmt="{:.0f}", highlight_min=True)
    row("Miss p75 (m)", [m["p75"] for m in miss], fmt="{:.0f}", highlight_min=True)
    row("Miss std (m)", [m["std"] for m in miss], fmt="{:.0f}")
    print()

    # Reward
    rew = [r["reward"] for r in runs]
    row("Reward mean", [rw["mean"] for rw in rew], fmt="{:.1f}", highlight_max=True)
    row("Reward std", [rw["std"] for rw in rew], fmt="{:.1f}")
    row("Reward min", [rw["min"] for rw in rew], fmt="{:.1f}")
    row("Reward max", [rw["max"] for rw in rew], fmt="{:.1f}", highlight_max=True)
    print()

    # Sim time / steps
    row("Sim time mean (s)", [r["sim_time"]["mean"] for r in runs], fmt="{:.1f}", highlight_max=True)
    row("Sim time max (s)", [r["sim_time"]["max"] for r in runs], fmt="{:.1f}")
    row("Steps mean", [r["steps"]["mean"] for r in runs], fmt="{:.0f}", highlight_max=True)
    row("Steps max", [r["steps"]["max"] for r in runs], fmt="{}")
    print()

    # ---- Termination breakdown ----
    print(f"  {'--- Termination % ---':<{label_w}}")
    all_reasons = set()
    for r in runs:
        all_reasons.update(r["termination_reasons"].keys())
    for reason in sorted(all_reasons):
        vals = []
        for r in runs:
            if reason in r["termination_reasons"]:
                vals.append(f"{r['termination_reasons'][reason]['pct']:.0%}")
            else:
                vals.append("-")
        cols = "".join(f"{v:>{col_w}}" for v in vals)
        print(f"  {reason:<{label_w}}{cols}")

    print()
    print("  * = best in row")
    print("=" * 100)


if __name__ == "__main__":
    main()
