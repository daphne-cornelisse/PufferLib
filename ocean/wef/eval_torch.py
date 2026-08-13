#!/usr/bin/env python3
"""Eval a converted torch WEF policy on the C env (via wef_env pybind).

Examples:
  # Convert best policy then eval
  python ocean/wef/example.py --ini logs/wef_nice/best_policy.ini --continuous
  python ocean/wef/eval_torch.py --ini logs/wef_nice/best_policy.ini --episodes 20

  # Compare vs random baseline
  python ocean/wef/eval_torch.py --ini logs/wef_nice/best_policy.ini --episodes 20 --also-random
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ocean" / "wef"))

from example import convert_from_ini, load_policy  # noqa: E402


def parse_ini(path: Path) -> dict:
    vals = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip()
    return vals


def make_env(seed: int, **kwargs):
    import wef_env

    return wef_env.WefEnv(
        num_agents=kwargs.get("num_agents", 4),
        min_arena_width=kwargs.get("arena", 70),
        max_arena_width=kwargs.get("arena", 70),
        min_arena_height=kwargs.get("arena", 70),
        max_arena_height=kwargs.get("arena", 70),
        food_distribution=kwargs.get("food_distribution", 1),
        num_food=kwargs.get("num_food", 64),
        episode_length=kwargs.get("episode_length", 512),
        seed=seed,
    )


def eval_policy(env, policy, episodes: int, seed: int) -> dict:
    """Run `episodes` completed episodes; return mean metrics from env log."""
    obs = env.reset(seed=seed)
    state = policy.initial_state(env.num_agents).numpy()
    # Drain any residual log
    env.consume_log()

    completed = 0
    steps = 0
    max_steps = episodes * env.episode_length + env.episode_length
    while completed < episodes and steps < max_steps:
        actions, _, state = policy.act_numpy(obs, state, deterministic=True)
        obs, rew, term, info = env.step(actions.astype(np.float32))
        steps += 1
        if term[0] > 0.5:
            completed += 1
            # Reset recurrent state on episode boundary
            state = policy.initial_state(env.num_agents).numpy()

    stats = env.consume_log()
    stats["completed"] = completed
    stats["steps"] = steps
    return stats


def eval_random(env, episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    env.consume_log()
    completed = 0
    steps = 0
    max_steps = episodes * env.episode_length + env.episode_length
    while completed < episodes and steps < max_steps:
        # Unbounded-ish random logits; env applies sigmoid/tanh/threshold
        actions = rng.normal(0.0, 1.0, size=(env.num_agents, env.num_actions)).astype(
            np.float32
        )
        obs, rew, term, info = env.step(actions)
        steps += 1
        if term[0] > 0.5:
            completed += 1
    stats = env.consume_log()
    stats["completed"] = completed
    stats["steps"] = steps
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ini", type=str, default="logs/wef_nice/best_policy.ini")
    ap.add_argument("--pt", type=str, default=None, help="pre-converted .pt (else convert)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arena", type=float, default=70)
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--num-food", type=int, default=64)
    ap.add_argument("--food-distribution", type=int, default=1)
    ap.add_argument("--episode-length", type=int, default=512)
    ap.add_argument("--also-random", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    ini = parse_ini(Path(args.ini))
    hidden = int(ini.get("hidden_size", 128))
    layers = int(ini.get("num_layers", 3))
    bin_path = ini["load_model_path"]

    pt_path = args.pt
    if pt_path is None:
        pt_path = str(Path(bin_path).with_suffix(".pt"))
        if not Path(pt_path).exists():
            print(f"converting {bin_path} → {pt_path}")
            convert_from_ini(args.ini, pt_path, continuous=True)

    print(f"loading policy {pt_path} (H={hidden}, L={layers})")
    policy = load_policy(
        pt_path,
        obs_size=110,
        hidden_size=hidden,
        num_layers=layers,
        continuous=True,
        num_actions=4,
        device=args.device,
    )

    env_kw = dict(
        num_agents=args.num_agents,
        arena=args.arena,
        num_food=args.num_food,
        food_distribution=args.food_distribution,
        episode_length=args.episode_length,
    )
    env = make_env(args.seed, **env_kw)

    stats = eval_policy(env, policy, args.episodes, args.seed)
    print("=== torch policy (C wef env) ===")
    for k in (
        "completed",
        "n",
        "episode_return",
        "score",
        "perf",
        "food_eaten_mean",
        "eod_rate",
        "episode_length",
    ):
        if k in stats:
            print(f"  {k}: {stats[k]}")
    if "episode_return" in ini:
        print(f"  (training best_policy.ini episode_return={ini['episode_return']})")

    if args.also_random:
        env_r = make_env(args.seed + 1000, **env_kw)
        rstats = eval_random(env_r, args.episodes, args.seed + 1000)
        print("=== random baseline (C wef env) ===")
        for k in ("completed", "n", "episode_return", "score", "perf", "food_eaten_mean"):
            if k in rstats:
                print(f"  {k}: {rstats[k]}")


if __name__ == "__main__":
    main()
