#!/usr/bin/env python
"""Summarize the old full-Craftax human trajectory dataset.

This dataset was collected with an older Craftax package layout. Run this with a
Python environment that has Craftax installed; the loader aliases the old pickle
module paths to the newer package layout.

Example:
    /Users/daphne/github/multitask_preplay/.venv/bin/python \
        scripts/craftax_full_human_runs_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pufferlib.craftax_full_human_runs import CRAFTAX_RUNS_DIR, run_paths, summarize_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=CRAFTAX_RUNS_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "resources" / "craftax" / "people" / "craftax_full_human_runs_analysis.txt",
    )
    args = parser.parse_args()

    summaries = [summarize_run(path) for path in run_paths(args.runs_dir)]
    total_steps = sum(row["steps"] for row in summaries)
    total_episodes = sum(row["episodes"] for row in summaries)
    total_reward = sum(row["reward_sum"] for row in summaries)

    lines = [
        "Full Craftax human runs overview",
        f"runs_dir: {args.runs_dir}",
        f"runs: {len(summaries)}",
        f"total_steps: {total_steps:,}",
        f"done_events: {total_episodes:,}",
        f"total_reward: {total_reward:.3f}",
        "",
        (
            "run   steps   states  done  last_done  reward_sum  reward_range   "
            "nonzero_reward  unique_actions  action_range  max_ach final_ach"
        ),
    ]
    for row in summaries:
        lines.append(
            f"{row['run']:<5} "
            f"{row['steps']:>7,} "
            f"{row['states']:>7,} "
            f"{row['episodes']:>5} "
            f"{str(row['last_done']):>9} "
            f"{row['reward_sum']:>10.3f} "
            f"[{row['reward_min']:.3f}, {row['reward_max']:.3f}] "
            f"{row['reward_nonzero']:>14,} "
            f"{row['unique_actions']:>14} "
            f"[{row['action_min']}, {row['action_max']}] "
            f"{row['max_achievements']:>8} "
            f"{row['final_achievements']:>9}"
        )

    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n")
    print(report)
    print(f"\nsaved report: {args.out}")


if __name__ == "__main__":
    main()
