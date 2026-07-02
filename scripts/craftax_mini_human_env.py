#!/usr/bin/env python
"""Validate and plot Craftax human trajectories on native Craftax Mini maps.

Examples:
    uv run --with polars --with matplotlib python scripts/craftax_mini_human_env.py
    uv run --with polars --with matplotlib python scripts/craftax_mini_human_env.py --n 6 --split test
"""

from __future__ import annotations

import argparse

import polars as pl

from pufferlib.craftax_mini_human_data import (
    HUMAN_DATAFRAME_PATH,
    HUMAN_PLOT_DIR,
    NativeCraftaxMiniWorlds,
    is_native_bounds_compatible,
    plot_human_trajectories,
    validate_human_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataframe", default=str(HUMAN_DATAFRAME_PATH))
    parser.add_argument("--out", default=str(HUMAN_PLOT_DIR / "human_trajectories_on_craftax_mini_maps.png"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    args = parser.parse_args()

    df = pl.read_parquet(args.dataframe)
    if args.split == "train":
        df = df.filter(pl.col("eval") == False)
    elif args.split == "test":
        df = df.filter(pl.col("eval") == True)

    pool = df.filter((pl.col("success") == 1.0) & (pl.col("path_length") > 2))
    if pool.height < args.n:
        pool = df
    candidate_rows = pool.to_dicts()
    compatible_rows = [row for row in candidate_rows if is_native_bounds_compatible(row)]
    print(
        f"native-compatible rows: {len(compatible_rows)} / {len(candidate_rows)} "
        "after split/success filtering"
    )
    if len(compatible_rows) < args.n:
        raise RuntimeError(
            f"Only {len(compatible_rows)} native-compatible rows available; requested {args.n}."
        )
    rows = pl.DataFrame(compatible_rows).sample(n=args.n, seed=args.seed).to_dicts()

    for row in rows:
        positions, web_actions, env_actions = validate_human_row(row)
        print(
            f"world={row['world']} eval={row['eval']} user={row['user_id']} "
            f"positions={len(positions)} web_actions={len(web_actions)} "
            f"env_action_range=[{env_actions.min()}, {env_actions.max()}]"
        )

    out_path = plot_human_trajectories(rows, args.out, worlds=NativeCraftaxMiniWorlds())
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
