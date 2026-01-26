#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pufferlib.ocean.pushworld.pushworld import PushWorld
from pufferlib.ocean.pushworld import binding as pw_binding


def parse_levels(text):
    if text is None or text.strip() == "":
        return [1, 2, 3, 4]
    return [int(x) for x in text.split(",") if x.strip()]


def run_level(level, episodes, num_envs, max_steps, puzzle_dir, vision, seed):
    rng = np.random.default_rng(seed)

    env = PushWorld(
        puzzle_dir=puzzle_dir,
        levels=[level],
        max_episode_length=max_steps,
        vision=vision,
        num_envs=num_envs,
        report_interval=1_000_000_000,
        seed=seed,
        render_mode="rgb_array",
        render_tile_size=1,
        render_full_map=False,
    )

    env.reset(seed=seed)

    steps = np.zeros(num_envs, dtype=np.int32)
    solved_steps = np.zeros(episodes, dtype=np.int32)
    episode_idx = 0

    while episode_idx < episodes:
        actions = rng.integers(0, 4, size=num_envs, dtype=np.int32)
        _, _, terminals, truncations, _ = env.step(actions)
        steps += 1

        done = (terminals.astype(bool) | truncations.astype(bool))
        if not np.any(done):
            continue

        solved_flags = pw_binding.vec_last_solved(env.c_envs).astype(bool)
        done_indices = np.flatnonzero(done)
        for env_i in done_indices:
            if episode_idx >= episodes:
                break
            if solved_flags[env_i]:
                step_count = int(steps[env_i])
                if step_count > max_steps:
                    step_count = max_steps
                solved_steps[episode_idx] = step_count
            steps[env_i] = 0
            episode_idx += 1

    env.close()

    counts = np.zeros(max_steps, dtype=np.int64)
    solved_only = solved_steps[solved_steps > 0]
    if solved_only.size > 0:
        counts += np.bincount(solved_only - 1, minlength=max_steps)
    solved_cum = np.cumsum(counts) / float(episodes)
    return solved_cum


def save_chart(x, y, level, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2)
    ax.set_title(f"Random Solve Rate (Level {level})")
    ax.set_xlabel("Number of random actions")
    ax.set_ylabel("Proportion solved by step")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    # Label 5 evenly spaced points for readability at low rates.
    if x.size >= 5:
        idxs = np.linspace(0, x.size - 1, 5, dtype=int)
        for idx in idxs:
            ax.scatter([x[idx]], [y[idx]], s=18, color="black", zorder=3)
            ax.annotate(
                f"{y[idx]:.3f}",
                (x[idx], y[idx]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
    out_path = Path(out_dir) / f"random_solve_rate_level{level}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_csv(x, y, level, out_dir):
    out_path = Path(out_dir) / f"random_solve_rate_level{level}.csv"
    data = np.column_stack([x, y])
    np.savetxt(out_path, data, delimiter=",", header="steps,solve_rate", comments="")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate random-action solve probability curves for PushWorld."
    )
    parser.add_argument("--puzzle-dir", default="resources/pushworld/puzzles/train")
    parser.add_argument("--levels", default="1,2,3,4", help="Comma-separated levels.")
    parser.add_argument("--episodes", type=int, default=5000, help="Episodes per level.")
    parser.add_argument("--num-envs", type=int, default=1024, help="Parallel envs.")
    parser.add_argument("--max-steps", type=int, default=200, help="Step horizon per episode.")
    parser.add_argument("--vision", type=int, default=15, help="Egocentric vision size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="experiments/random_action_eval")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export.")
    args = parser.parse_args()

    levels = parse_levels(args.levels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(1, args.max_steps + 1, dtype=np.int32)
    for level in levels:
        y = run_level(
            level=level,
            episodes=args.episodes,
            num_envs=args.num_envs,
            max_steps=args.max_steps,
            puzzle_dir=args.puzzle_dir,
            vision=args.vision,
            seed=args.seed + level,
        )
        save_chart(x, y, level, out_dir)
        if not args.no_csv:
            save_csv(x, y, level, out_dir)

    print(f"Saved charts to: {out_dir}")


if __name__ == "__main__":
    main()
