#!/usr/bin/env python3
"""Extract symbolic observation/action pairs from a full Craftax human run.

Example:
    python scripts/extract_craftax_full_run_dataset.py --run run1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pufferlib.craftax_full_human_runs import (
    CRAFTAX_RUNS_DIR,
    extract_obs_action_pairs,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="run1", help="Run stem without .pbz2, e.g. run1")
    parser.add_argument("--runs-dir", type=Path, default=CRAFTAX_RUNS_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "resources" / "craftax" / "people" / "run1_obs_actions.npz",
    )
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="Also save next observations for each action.",
    )
    parser.add_argument(
        "--include-reward-done",
        action="store_true",
        help="Also save rewards and done flags for each action.",
    )
    args = parser.parse_args()

    run_path = args.runs_dir / f"{args.run}.pbz2"
    dataset = extract_obs_action_pairs(
        run_path,
        include_next_obs=args.include_next_obs,
        include_reward_done=args.include_reward_done,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **dataset)

    print(f"run: {run_path}")
    for key, value in dataset.items():
        print(f"{key}: shape={value.shape} dtype={value.dtype}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
