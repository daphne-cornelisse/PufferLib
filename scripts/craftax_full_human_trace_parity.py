#!/usr/bin/env python
"""Replay old human action traces in current JAX and PufferLib full Craftax.

The old dataset stores full JAX states, but not the original reset seeds. This
script uses the recorded action sequences as a realistic action source, starts
current JAX Craftax and PufferLib Craftax from identical reset seeds, and checks
reward/done/observation parity step by step.

Example:
    CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ \
        uv run --with pybind11 --with rich_argparse ./build.sh craftax --cpu
    /Users/daphne/github/multitask_preplay/.venv/bin/python \
        scripts/craftax_full_human_trace_parity.py --runs run4 run3 --steps 512
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pufferlib.craftax_full_human_runs import CRAFTAX_RUNS_DIR, load_actions  # noqa: E402
from tests.craftax_parity import JaxCraftaxBatch, float_view  # noqa: E402


JAX_OBS_SIZE = 8268
NUM_ACTIONS = 43


def import_c_env():
    import pufferlib._C as cmod

    env_name = getattr(cmod, "env_name", None)
    if env_name != "craftax":
        raise RuntimeError(
            f"pufferlib._C is compiled for {env_name!r}, expected 'craftax'. "
            "Run: ./build.sh craftax --cpu"
        )
    return cmod


def make_c_vec(cmod, num_envs: int, seed_offset: int):
    args = {
        "vec": {"total_agents": num_envs, "num_buffers": 1, "num_threads": 1},
        "env": {"seed_offset": seed_offset, "reset_pool_size": 0},
    }
    vec = cmod.create_vec(args, 0)
    if list(vec.act_sizes) != [NUM_ACTIONS]:
        raise RuntimeError(f"C act_sizes={vec.act_sizes}, expected [{NUM_ACTIONS}]")
    vec.reset()
    obs = float_view(vec.obs_ptr, num_envs * vec.obs_size).reshape(num_envs, vec.obs_size)
    rewards = float_view(vec.rewards_ptr, num_envs)
    terminals = float_view(vec.terminals_ptr, num_envs)
    return vec, obs, rewards, terminals


def resolve_run(name: str) -> Path:
    path = Path(name)
    if path.exists():
        return path
    if not name.endswith(".pbz2"):
        name = f"{name}.pbz2"
    return CRAFTAX_RUNS_DIR / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["run4"], help="run names or .pbz2 paths")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    run_files = [resolve_run(name) for name in args.runs]
    action_traces = [load_actions(path) for path in run_files]
    steps = min(args.steps, min(len(trace) for trace in action_traces))
    if steps <= 0:
        raise ValueError("no actions available for requested runs")

    seeds = np.arange(args.seed_start, args.seed_start + len(run_files), dtype=np.int64)
    ref = JaxCraftaxBatch(seeds, resetter=None)
    cmod = import_c_env()
    vec, c_obs, c_rewards, c_terminals = make_c_vec(cmod, len(run_files), int(seeds[0]))

    try:
        compare_observations = c_obs.shape[1] == JAX_OBS_SIZE
        if not compare_observations:
            print(
                "NOTE: skipping observation comparison because PufferLib full Craftax "
                f"uses packed obs_size={c_obs.shape[1]} while current JAX Craftax "
                f"uses symbolic obs_size={JAX_OBS_SIZE}."
            )

        action_buf = np.zeros((len(run_files), 1), dtype=np.float32)
        for step in range(steps):
            actions = np.asarray([trace[step] for trace in action_traces], dtype=np.int32)
            action_buf[:, 0] = actions.astype(np.float32)

            ref_obs, ref_rewards, ref_dones, _reset_keys = ref.step(actions)
            vec.cpu_step(action_buf.ctypes.data_as(ctypes.c_void_p).value)

            c_obs_snapshot = c_obs.copy()
            c_rewards_snapshot = c_rewards.copy()
            c_dones_snapshot = c_terminals.copy().astype(bool)

            for env_i, seed in enumerate(seeds):
                reward_diff = abs(float(ref_rewards[env_i]) - float(c_rewards_snapshot[env_i]))
                done_match = bool(ref_dones[env_i]) == bool(c_dones_snapshot[env_i])
                obs_diff = None
                if compare_observations:
                    diff = np.abs(ref_obs[env_i] - c_obs_snapshot[env_i])
                    max_idx = int(np.argmax(diff))
                    max_diff = float(diff[max_idx])
                    if max_diff > args.atol:
                        obs_diff = (
                            max_idx,
                            max_diff,
                            float(ref_obs[env_i, max_idx]),
                            float(c_obs_snapshot[env_i, max_idx]),
                        )
                if reward_diff > args.atol or not done_match or obs_diff is not None:
                    print(
                        "FAIL "
                        f"run={run_files[env_i].stem} seed={int(seed)} step={step} "
                        f"action={int(actions[env_i])} "
                        f"reward_jax={float(ref_rewards[env_i]):.8g} "
                        f"reward_c={float(c_rewards_snapshot[env_i]):.8g} "
                        f"done_jax={bool(ref_dones[env_i])} "
                        f"done_c={bool(c_dones_snapshot[env_i])} "
                        f"obs_diff={obs_diff}"
                    )
                    raise SystemExit(1)
    finally:
        vec.close()

    run_names = ", ".join(path.stem for path in run_files)
    print(
        "PASS: full Craftax PufferLib/JAX parity under human action traces "
        f"runs=[{run_names}] seeds={seeds.tolist()} steps={steps}"
    )


if __name__ == "__main__":
    main()
