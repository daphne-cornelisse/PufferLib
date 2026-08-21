#!/usr/bin/env python3
"""Measure WEF simulation and training throughput vs KempnerInstitute/wef.

Reports mean ± std over N episodes (default 10):

  1. Original Python env FPS — ``without_render`` in MAEfish.py, animation off
  2. Puffer C env raw SPS, 1 thread (apples-to-apples vs original)
  3. Puffer C env raw SPS, sum of worker threads (training vec layout)
  4. True training SPS from ``./puffer train wef``

Usage (from repo root):

  python ocean/wef/bench_sps.py
  python ocean/wef/bench_sps.py --skip-train
  python ocean/wef/bench_sps.py --episodes 10 --threads 16

Original code is invoked with the wef venv:

  WEF_ROOT=/home/daphne/code/wef
  WEF_PYTHON=/home/daphne/code/wef/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEF_ROOT = Path(os.environ.get("WEF_ROOT", Path.home() / "code" / "wef"))
DEFAULT_WEF_PYTHON = Path(
    os.environ.get("WEF_PYTHON", DEFAULT_WEF_ROOT / ".venv" / "bin" / "python")
)
WEF_INI = ROOT / "config" / "wef.ini"

SPS_LINE_RE = re.compile(r"\bSPS\s+([0-9]+(?:\.[0-9]+)?)([KMBT]?)\b")
JSON_LINE_RE = re.compile(r"^JSON\s+(\{.*\})\s*$")
INI_SPS_RE = re.compile(r"^SPS\s*=\s*(.+)$", re.MULTILINE)


def parse_ini_scalars(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    section = ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        vals[f"{section}.{key.strip()}"] = val.strip()
        vals[key.strip()] = val.strip()
    return vals


def ini_int(vals: dict[str, str], key: str, default: int) -> int:
    raw = vals.get(key)
    if raw is None:
        return default
    return int(float(raw.replace("_", "")))


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)


def fmt_num(x: float) -> str:
    if x != x:  # NaN
        return "n/a"
    ax = abs(x)
    if ax >= 1e6:
        return f"{x / 1e6:.2f}M"
    if ax >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{x:.1f}"


def parse_abbrev(num: str, suffix: str) -> float:
    scale = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
    return float(num) * scale


def parse_sps_series(text: str) -> list[float]:
    out: list[float] = []
    for match in SPS_LINE_RE.finditer(text):
        out.append(parse_abbrev(match.group(1), match.group(2)))
    return out


def parse_csv_floats(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


@dataclass
class EpisodeStats:
    name: str
    env_sps: list[float] = field(default_factory=list)
    agent_sps: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        env_m, env_s = mean_std(self.env_sps)
        ag_m, ag_s = mean_std(self.agent_sps)
        return {
            "name": self.name,
            "n": len(self.env_sps),
            "env_sps_mean": env_m,
            "env_sps_std": env_s,
            "agent_sps_mean": ag_m,
            "agent_sps_std": ag_s,
            **self.extra,
        }


def build_c_bench(out: Path) -> None:
    ray_inc = ROOT / "raylib-5.5_linux_amd64" / "include"
    ray_lib = ROOT / "raylib-5.5_linux_amd64" / "lib" / "libraylib.a"
    if not ray_lib.is_file():
        raise SystemExit(f"missing {ray_lib}; run ./build.sh wef once")
    cc = os.environ.get("CC", "clang")
    cmd = [
        cc,
        "-O3",
        "-DNDEBUG",
        "-mavx2",
        "-mfma",
        "-DPLATFORM_DESKTOP",
        f"-I{ray_inc}",
        f"-I{ROOT / 'src'}",
        f"-I{ROOT / 'vendor'}",
        f"-I{ROOT / 'ocean' / 'wef'}",
        str(ROOT / "ocean" / "wef" / "bench_sps.c"),
        "-o",
        str(out),
        str(ray_lib),
        "-lGL",
        "-lm",
        "-lpthread",
        "-ldl",
        "-lrt",
    ]
    print("compile:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError:
        if out.is_file():
            print(
                f"warning: compile failed; reusing existing {out}",
                flush=True,
            )
            return
        raise


def run_c_bench(
    binary: Path,
    *,
    threads: int,
    total_agents: int,
    num_agents: int,
    episodes: int,
    episode_length: int,
    warmup: int,
    min_arena: int,
    max_arena: int,
    food_distribution: int,
    num_food: int,
    name: str,
) -> EpisodeStats:
    cmd = [
        str(binary),
        "--threads",
        str(threads),
        "--total-agents",
        str(total_agents),
        "--num-agents",
        str(num_agents),
        "--episodes",
        str(episodes),
        "--episode-length",
        str(episode_length),
        "--warmup",
        str(warmup),
        "--min-arena",
        str(min_arena),
        "--max-arena",
        str(max_arena),
        "--food-distribution",
        str(food_distribution),
        "--num-food",
        str(num_food),
    ]
    print("\n===", name, "===")
    print("run:", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd, cwd=ROOT, check=True, text=True, capture_output=True
    )
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    payload = None
    for line in proc.stdout.splitlines():
        match = JSON_LINE_RE.match(line)
        if match:
            payload = json.loads(match.group(1))
    if payload is None:
        raise RuntimeError("C bench did not print a JSON summary line")
    wall_env = [float(ep["env_sps"]) for ep in payload["episodes_data"]]
    sum_thread = [float(ep["sum_thread_env_sps"]) for ep in payload["episodes_data"]]
    n_agents = float(payload["num_agents"])
    env_sps = sum_thread
    agent_sps = [e * n_agents for e in sum_thread]
    stats = EpisodeStats(
        name=name,
        env_sps=env_sps,
        agent_sps=agent_sps,
        extra={
            "threads": threads,
            "num_envs": payload["num_envs"],
            "wall_env_sps_mean": payload["mean_env_sps"],
            "wall_env_sps_std": payload["std_env_sps"],
            "sum_thread_env_sps_mean": payload["mean_sum_thread_env_sps"],
            "sum_thread_env_sps_std": payload["std_sum_thread_env_sps"],
            "sum_thread_env_sps": sum_thread,
            "wall_env_sps": wall_env,
            "per_thread_env_sps_mean": (
                payload["mean_sum_thread_env_sps"] / threads if threads else 0.0
            ),
        },
    )
    return stats


def original_all_args(seed: int, episode_length: int, num_agents: int) -> dict:
    """MAEfish.py __main__ defaults, animation off (render_mode=None)."""
    return {
        "arena_size_min": (50, 150),
        "arena_size_max": (50, 150),
        "num_agents": num_agents,
        "energy_food": 20,
        "render_mode": None,
        "food_drift": 0.01,
        "food_drag": 0.2,
        "collective_sensing_mode": 1,
        "timestamp": "bench_sps",
        "max_food_eaten_per_step": 1,
        "feedback_action": True,
        "feedback_displacement": True,
        "pfeeder": 0,
        "prandom": 1,
        "urandom": 0,
        "prob_n_patch": 0,
        "save_vid": False,
        "allow_aggression": 1,
        "knollen_mode": 1,
        "ampullary_mode": 1,
        "mormyromast_mode": 1,
        "morm_selfimage_mode": 1,
        "morm_consimage_mode": 1,
        "sensing_model_type": "dynamic",
        "backwards": True,
        "max_episode_length": episode_length,
        "homing_distance": 4.0,
        "required_homing_steps": 20,
        "homing_mode": False,
        "is_eval": False,
        "rw_eod_rate": 0.2,
        "rw_freeze": 1,
        "task": "foraging",
        "p_init_closeby": 0.0,
        "ampullary_intrinsic_only": True,
        "ampullary_ema": False,
        "max_food_sensing_radius": 15,
        "base_food_multiplier": 1.0,
        "auxs": None,
        "multiplier_linear": 2.0,
        "multiplier_angular": 3.0,
        "seed": seed,
    }


def _original_thread_loop(
    thread_id: int,
    episodes: int,
    steps: int,
    warmup: int,
    num_agents: int,
    seed0: int,
    wef_root: str,
    conn,
) -> None:
    """One original env on one process; report per-episode elapsed seconds."""
    wef_fish = str(Path(wef_root) / "onpolicy" / "custom" / "fish")
    sys.path.insert(0, wef_fish)
    os.chdir(wef_fish)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import numpy as np
    from MAEFish import MultiAgentFishEnv, TestTurnsAgent  # noqa: E402

    seed = seed0 + thread_id * 1009
    all_args = original_all_args(seed, steps, num_agents)
    env = MultiAgentFishEnv(all_args, seed=seed)
    agents = [
        TestTurnsAgent(
            max_angular_velocity=1,
            allow_aggression=all_args["allow_aggression"],
            agent_id=i,
        )
        for i in range(env.num_agents)
    ]
    for _ in range(warmup):
        actions = np.array([a.get_action() for a in agents], dtype=object)
        env.step(actions)
    elapsed_eps = []
    for _ in range(episodes):
        t0 = time.perf_counter()
        for _ in range(steps):
            actions = np.array([a.get_action() for a in agents], dtype=object)
            env.step(actions)
        elapsed_eps.append(time.perf_counter() - t0)
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass
    conn.send(elapsed_eps)
    conn.close()


def run_original_worker(args: argparse.Namespace) -> None:
    """Must run under the Kempner wef venv (imports MAEFish)."""
    wef_fish = Path(args.wef_root) / "onpolicy" / "custom" / "fish"
    sys.path.insert(0, str(wef_fish))
    os.chdir(wef_fish)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    episodes = args.episodes
    steps = args.original_steps
    warmup = args.original_warmup
    threads = max(1, int(args.threads))
    print("=== original KempnerInstitute/wef (without_render, animation off) ===")
    print(
        f"  episodes={episodes}  steps/ep={steps}  warmup={warmup}  "
        f"num_agents={args.original_num_agents}  threads={threads}",
        flush=True,
    )

    if threads == 1:
        import numpy as np
        from MAEFish import MultiAgentFishEnv, TestTurnsAgent  # noqa: E402

        env_sps: list[float] = []
        for ep in range(episodes):
            seed = args.seed + ep
            all_args = original_all_args(seed, steps, args.original_num_agents)
            env = MultiAgentFishEnv(all_args, seed=seed)
            agents = [
                TestTurnsAgent(
                    max_angular_velocity=1,
                    allow_aggression=all_args["allow_aggression"],
                    agent_id=i,
                )
                for i in range(env.num_agents)
            ]
            for _ in range(warmup):
                actions = np.array([a.get_action() for a in agents], dtype=object)
                env.step(actions)
            t0 = time.perf_counter()
            for _ in range(steps):
                actions = np.array([a.get_action() for a in agents], dtype=object)
                env.step(actions)
            elapsed = time.perf_counter() - t0
            fps = steps / elapsed if elapsed > 0 else 0.0
            env_sps.append(fps)
            print(f"    ep {ep}  wall={elapsed:.4f}s  env_FPS={fps:.1f}", flush=True)
            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        pipes = [ctx.Pipe(duplex=False) for _ in range(threads)]
        procs = []
        for t in range(threads):
            recv, send = pipes[t]
            p = ctx.Process(
                target=_original_thread_loop,
                args=(
                    t,
                    episodes,
                    steps,
                    warmup,
                    args.original_num_agents,
                    args.seed,
                    args.wef_root,
                    send,
                ),
            )
            p.start()
            send.close()
            procs.append((p, recv))
        per_thread = [recv.recv() for _, recv in procs]
        for p, recv in procs:
            p.join()
            recv.close()
            if p.exitcode != 0:
                raise SystemExit(f"original worker thread exited {p.exitcode}")
        env_sps = []
        for ep in range(episodes):
            thread_s = [per_thread[t][ep] for t in range(threads)]
            wall = max(thread_s)
            sum_thr = sum(
                (steps / s) if s > 0 else 0.0 for s in thread_s
            )
            env_sps.append(sum_thr)
            print(
                f"    ep {ep}  wall={wall:.4f}s  sum_thread_env_SPS={sum_thr:.1f}  "
                f"per_thread={sum_thr / threads:.1f}",
                flush=True,
            )

    em, es = mean_std(env_sps)
    print(f"  env_SPS   mean={em:.1f}  std={es:.1f}  threads={threads}")
    payload = {
        "num_agents": args.original_num_agents,
        "threads": threads,
        "episode_length": steps,
        "episodes": episodes,
        "mean_env_sps": em,
        "std_env_sps": es,
        "episodes_data": [{"env_sps": e} for e in env_sps],
    }
    print("JSON", json.dumps(payload), flush=True)


def run_original(args: argparse.Namespace) -> EpisodeStats:
    python = Path(args.wef_python)
    if not python.is_file():
        raise SystemExit(
            f"original wef python not found: {python}\n"
            "Set --wef-python / WEF_PYTHON to the Kempner wef venv interpreter."
        )
    cmd = [
        str(python),
        str(Path(__file__).resolve()),
        "--original-worker",
        "--wef-root",
        str(args.wef_root),
        "--episodes",
        str(args.episodes),
        "--original-steps",
        str(args.original_steps),
        "--original-warmup",
        str(args.original_warmup),
        "--original-num-agents",
        str(args.original_num_agents),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
    ]
    print("\n=== original (KempnerInstitute/wef) ===")
    print("run:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    payload = None
    for line in proc.stdout.splitlines():
        match = JSON_LINE_RE.match(line)
        if match:
            payload = json.loads(match.group(1))
    if payload is None:
        raise RuntimeError("original worker did not print a JSON summary")
    env_sps = [float(ep["env_sps"]) for ep in payload["episodes_data"]]
    n_thr = int(payload.get("threads", args.threads))
    return EpisodeStats(
        name=f"original ({n_thr} thread{'s' if n_thr != 1 else ''})",
        env_sps=env_sps,
        agent_sps=env_sps,
        extra={
            "threads": n_thr,
            "num_agents": payload["num_agents"],
            "note": "MAEfish.py without_render, render_mode=None",
        },
    )


ORIG_FPS_RE = re.compile(
    r"timesteps\s+(\d+)/(\d+),\s+FPS\s+(\d+)", re.IGNORECASE
)


def incremental_sps(logs: list[tuple[int, float]]) -> list[float]:
    """Convert cumulative (steps, fps) logs into per-interval env SPS."""
    out: list[float] = []
    prev_steps = 0
    prev_t = 0.0
    for steps, fps in logs:
        if fps <= 0:
            continue
        t = steps / fps
        if prev_t > 0 and t > prev_t and steps > prev_steps:
            out.append((steps - prev_steps) / (t - prev_t))
        prev_steps, prev_t = steps, t
    return out


def run_original_train(args: argparse.Namespace) -> EpisodeStats:
    """MAPPO training throughput from Kempner train_fish.py (paper-ish flags)."""
    python = Path(args.wef_python)
    if not python.is_file():
        raise SystemExit(f"original wef python not found: {python}")
    fish_dir = Path(args.wef_root) / "onpolicy" / "custom" / "fish"
    num_agents = 4
    threads = args.original_train_threads
    episode_length = args.episode_length
    episodes = args.episodes + args.train_warmup_episodes
    num_env_steps = episodes * episode_length * threads
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_parent = ROOT / "logs" / "wef"
    cmd = [
        str(python),
        "train_fish.py",
        "--experiment_name",
        "bench_sps_original_train",
        "--results_parent_dir",
        str(out_parent),
        "--timestamp",
        stamp,
        "--num_agents",
        str(num_agents),
        "--num_env_steps",
        str(num_env_steps),
        "--episode_length",
        str(episode_length),
        "--max_episode_length",
        str(episode_length),
        "--n_rollout_threads",
        str(threads),
        "--render_episodes",
        "0",
        "--log_interval",
        "1",
        "--save_interval",
        "100000",
        "--allow_aggression",
        "1",
        "--agent_size_mode",
        "random",
        "--weight_decay",
        "0.000001",
        "--sensing_model_type",
        "dynamic",
        "--noise_frac_morm",
        "0.05",
        "--noise_frac_amp",
        "0.05",
        "--noise_frac_amp_cons_eod",
        "1.0",
        "--noise_frac_knollen",
        "0.05",
        "--size_speed_exponent",
        "1",
        "--train_sensor_dropout_p",
        "0.0",
        "--p_init_closeby",
        "0.0",
        "--base_food_multiplier",
        "1.0",
        "--hidden_size",
        "512",
        "--max_food_sensing_radius",
        "15",
        "--train_food_scaling_min",
        "0.5",
        "--train_food_scaling_max",
        "2",
        "--use_bite_cooldown",
        "--mormyromast_mode",
        "1",
        "--ampullary_mode",
        "1",
        "--knollen_mode",
        "1",
        "--knollen_metadata_mode",
        "relative",
        "--collective_sensing_mode",
        "1",
        "--data_chunk_length",
        "100",
        "--use_orthogonal",
        "--rnn_type",
        "GRU",
        "--motion_order",
        "1",
        "--multiplier_linear",
        "2.0",
        "--multiplier_angular",
        "4.0",
        "--ampullary_intrinsic_only",
        "--auxs",
        "eods",
        "--asym_eating",
        "--food_orientation_drift",
        "0.1",
        "--penalize_effort_over_frac",
        "0.5",
        "--prandom",
        "5",
        "--urandom",
        "1",
        "--prob_n_patch",
        "0",
        "--pfeeder",
        "0",
        "--feedback_displacement",
        "--gamma",
        "0.995",
        "--seed",
        str(args.seed),
    ]
    shim = Path("/tmp/tbx_shim")
    (shim / "tensorboardX").mkdir(parents=True, exist_ok=True)
    (shim / "tensorboardX" / "__init__.py").write_text(
        "class SummaryWriter:\n"
        "    def __init__(self, *a, **k): pass\n"
        "    def add_scalar(self, *a, **k): pass\n"
        "    def add_scalars(self, *a, **k): pass\n"
        "    def close(self): pass\n"
        "    def flush(self): pass\n"
        "    def __getattr__(self, name):\n"
        "        return lambda *a, **k: None\n"
    )
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": str(shim) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    print("\n=== original train (Kempner MAPPO) ===")
    print("run:", " ".join(cmd), flush=True)
    print(
        f"  episodes={episodes}  episode_length={episode_length}  "
        f"n_rollout_threads={threads}  num_env_steps={num_env_steps}",
        flush=True,
    )
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(fish_dir),
        text=True,
        capture_output=True,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    log_path = out_parent / f"original_train_{stamp}.log"
    log_path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))
    sys.stdout.write(proc.stdout[-8000:] if len(proc.stdout) > 8000 else proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:] if proc.stderr else "")
        raise SystemExit(
            f"original train_fish.py failed ({proc.returncode}); log: {log_path}"
        )

    logs: list[tuple[int, float]] = []
    for line in (proc.stdout + "\n" + (proc.stderr or "")).splitlines():
        match = ORIG_FPS_RE.search(line)
        if match:
            logs.append((int(match.group(1)), float(match.group(3))))
    env_inc = incremental_sps(logs)
    # First logged FPS is the warmup baseline; increments are subsequent episodes.
    if len(env_inc) > args.episodes:
        env_inc = env_inc[: args.episodes]
    overall_env = num_env_steps / elapsed if elapsed > 0 else float("nan")
    print(
        f"  original train SPS intervals={len(env_inc)}  "
        f"mean={mean_std(env_inc)[0]:.1f}  overall={overall_env:.1f}",
        flush=True,
    )
    return EpisodeStats(
        name="original train (MAPPO)",
        env_sps=env_inc,
        agent_sps=env_inc,
        extra={
            "threads": threads,
            "num_agents": num_agents,
            "timesteps": num_env_steps,
            "wall_s": elapsed,
            "overall_env_sps": overall_env,
            "overall_agent_sps": overall_env,
            "log_path": str(log_path),
            "note": "train_fish.py rmappo, paper flags, render_episodes=0",
        },
    )


def train_timesteps(args: argparse.Namespace, cfg: dict[str, str]) -> int:
    if args.train_timesteps is not None:
        return args.train_timesteps
    total_agents = ini_int(cfg, "vec.total_agents", 512)
    episode_length = ini_int(cfg, "env.episode_length", 2048)
    horizon = ini_int(cfg, "train.horizon", 32)
    # One extra warmup episode so the first CUDA-graph sample is dropped.
    episodes = args.episodes + args.train_warmup_episodes
    raw = episodes * episode_length * total_agents
    batch = total_agents * horizon
    return int(math.ceil(raw / batch) * batch)


def run_train(args: argparse.Namespace, cfg: dict[str, str]) -> EpisodeStats:
    binary = Path(args.puffer)
    if not binary.is_file():
        raise SystemExit(
            f"missing {binary}; build with ./build.sh wef --headless"
        )
    total_agents = ini_int(cfg, "vec.total_agents", 512)
    num_agents = ini_int(cfg, "env.num_agents", 4)
    episode_length = ini_int(cfg, "env.episode_length", 2048)
    timesteps = train_timesteps(args, cfg)
    run_id = args.train_run_id or f"wef_sps_{int(time.time())}"
    log_path = ROOT / "logs" / "wef" / f"{run_id}.ini"
    cmd = [
        str(binary),
        "train",
        f"--train.total_timesteps={timesteps}",
        "--base.eval_episodes=0",
        "--base.checkpoint_interval=0",
        "--base.wandb=0",
        f"--base.run_id={run_id}",
        f"--sweep.downsample={max(args.episodes + args.train_warmup_episodes, 2)}",
    ]
    skip_overlay = {
        "train.total_timesteps",
        "train.prio_alpha",
        "train.prio_beta0",
    }
    for key, val in cfg.items():
        if "." not in key or key in skip_overlay:
            continue
        section = key.split(".", 1)[0]
        if section not in ("vec", "policy", "train"):
            continue
        if val.startswith("#") or not val:
            continue
        cmd.append(f"--{key}={val.replace('_', '')}")
    print("\n=== ./puffer train ===")
    print("run:", " ".join(cmd), flush=True)
    print(
        f"  timesteps={timesteps}  (~{args.episodes}+{args.train_warmup_episodes} "
        f"episodes × {episode_length} × {total_agents} agents)",
        flush=True,
    )
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "TERM": "dumb"},
    )
    elapsed = time.perf_counter() - t0
    out_dir = ROOT / "logs" / "wef"
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{run_id}.stdout"
    stdout_path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(
            f"./puffer train failed ({proc.returncode}); "
            f"full log: {stdout_path}"
        )

    dash_sps = parse_sps_series(proc.stdout)
    # First logs include CUDA-graph capture; the last tick is often a leftover
    # epoch with a tiny dt (absurd SPS). Drop both.
    if len(dash_sps) > 2:
        dash_sps = dash_sps[:-1]
    skip = min(args.train_skip_logs, max(0, len(dash_sps) - 1))
    used = dash_sps[skip:]
    log_sps: list[float] = []
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = INI_SPS_RE.search(text)
        if match:
            log_sps = parse_csv_floats(match.group(1))
            if args.train_warmup_episodes > 0 and len(log_sps) > args.train_warmup_episodes:
                log_sps = log_sps[args.train_warmup_episodes :]
            if len(log_sps) > 1:
                log_sps = log_sps[:-1]

    # Prefer the 10 episode-binned log values; fall back to dashboard samples.
    series = log_sps if len(log_sps) >= 2 else used
    overall = timesteps / elapsed if elapsed > 0 else float("nan")
    stats = EpisodeStats(
        name="puffer train wef",
        env_sps=series,
        agent_sps=series,
        extra={
            "threads": ini_int(cfg, "vec.num_threads", 16),
            "total_agents": total_agents,
            "num_agents": num_agents,
            "timesteps": timesteps,
            "wall_s": elapsed,
            "overall_agent_sps": overall,
            "overall_env_sps": overall,
            "dashboard_sps": used,
            "log_sps": log_sps,
            "log_path": str(log_path),
            "stdout_path": str(stdout_path),
            "run_id": run_id,
        },
    )
    em, es = mean_std(series)
    print(
        f"  train SPS  mean={em:.1f}  std={es:.1f}  "
        f"overall={overall:.1f}  samples={len(series)}",
        flush=True,
    )
    return stats


def print_table(rows: list[EpisodeStats], original: EpisodeStats | None) -> None:
    print("\n" + "=" * 78)
    print("WEF throughput comparison")
    print("=" * 78)
    header = (
        f"{'setup':<28} {'env SPS':>16} {'agent SPS':>16} {'vs orig':>10}"
    )
    print(header)
    print("-" * 78)
    orig_env = float("nan")
    if original is not None and original.env_sps:
        orig_env = statistics.fmean(original.env_sps)
    for row in rows:
        s = row.summary()
        vs = (
            s["env_sps_mean"] / orig_env
            if orig_env and orig_env == orig_env and orig_env > 0
            else float("nan")
        )
        vs_s = "n/a" if vs != vs else f"{vs:.1f}x"
        env_s = f"{fmt_num(s['env_sps_mean'])} ± {fmt_num(s['env_sps_std'])}"
        ag_s = f"{fmt_num(s['agent_sps_mean'])} ± {fmt_num(s['agent_sps_std'])}"
        print(f"{s['name']:<28} {env_s:>16} {ag_s:>16} {vs_s:>10}")
        if "per_thread_env_sps_mean" in s:
            print(
                f"{'  per thread':<28} {fmt_num(s['per_thread_env_sps_mean']):>16}"
            )
        if "overall_agent_sps" in s:
            print(
                f"{'  overall (steps/wall)':<28} "
                f"{fmt_num(s['overall_env_sps']):>16} "
                f"{fmt_num(s['overall_agent_sps']):>16}"
            )
    print("-" * 78)
    print(
        "SPS as each program measures it (no ×num_agents).\n"
        "Original FPS and C-bench env SPS count env.step / puf_step.\n"
        "Puffer train SPS is the dashboard value (delta global_step / dt).\n"
        "Raw sim 'sum of threads' is wall-clock throughput of the env pool "
        "(independent workers, no policy)."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = parse_ini_scalars(WEF_INI) if WEF_INI.is_file() else {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--episode-length", type=int,
                   default=ini_int(cfg, "env.episode_length", 2048))
    p.add_argument("--threads", type=int,
                   default=ini_int(cfg, "vec.num_threads", 16))
    p.add_argument("--total-agents", type=int,
                   default=ini_int(cfg, "vec.total_agents", 512))
    p.add_argument("--num-agents", type=int,
                   default=ini_int(cfg, "env.num_agents", 4))
    p.add_argument("--min-arena", type=int,
                   default=ini_int(cfg, "env.min_arena_width", 30))
    p.add_argument("--max-arena", type=int,
                   default=ini_int(cfg, "env.max_arena_width", 400))
    p.add_argument("--food-distribution", type=int,
                   default=ini_int(cfg, "env.food_distribution", 2))
    p.add_argument("--num-food", type=int,
                   default=ini_int(cfg, "env.num_food", 64))
    p.add_argument("--warmup", type=int, default=1,
                   help="untimed C-env episodes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wef-root", type=str, default=str(DEFAULT_WEF_ROOT))
    p.add_argument("--wef-python", type=str, default=str(DEFAULT_WEF_PYTHON))
    p.add_argument("--original-num-agents", type=int, default=3,
                   help="MAEfish.py __main__ default is 3")
    p.add_argument("--original-steps", type=int, default=None,
                   help="steps per original episode (default: --episode-length)")
    p.add_argument("--original-warmup", type=int, default=8)
    p.add_argument("--puffer", type=str, default=str(ROOT / "puffer"))
    p.add_argument("--train-timesteps", type=int, default=None)
    p.add_argument("--train-run-id", type=str, default=None)
    p.add_argument("--train-warmup-episodes", type=int, default=1)
    p.add_argument("--train-skip-logs", type=int, default=3)
    p.add_argument("--skip-original", action="store_true")
    p.add_argument("--skip-original-train", action="store_true")
    p.add_argument("--original-train-threads", type=int, default=16,
                   help="n_rollout_threads for original MAPPO (paper default 16)")
    p.add_argument("--skip-sim", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--original-train-only", action="store_true")
    p.add_argument("--skip-single-thread", action="store_true")
    p.add_argument("--original-worker", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--no-rebuild", action="store_true",
                   help="reuse existing wef_bench_sps if present")
    p.add_argument(
        "--overlay-ini",
        type=str,
        default="",
        help="overlay [vec]/[policy]/[train] from another ini (e.g. config/drive.ini)",
    )
    args = p.parse_args(argv)
    if args.original_steps is None:
        args.original_steps = args.episode_length
    return args


def main() -> None:
    args = parse_args()
    if args.original_worker:
        run_original_worker(args)
        return

    cfg = parse_ini_scalars(WEF_INI) if WEF_INI.is_file() else {}
    if args.overlay_ini:
        overlay_path = Path(args.overlay_ini)
        if not overlay_path.is_file():
            raise SystemExit(f"overlay ini not found: {overlay_path}")
        ov = parse_ini_scalars(overlay_path)
        for key, val in ov.items():
            if "." not in key:
                continue
            if key.split(".", 1)[0] in ("vec", "policy", "train"):
                cfg[key] = val
        args.threads = ini_int(cfg, "vec.num_threads", args.threads)
        args.total_agents = ini_int(cfg, "vec.total_agents", args.total_agents)
        print(
            f"overlay {overlay_path}: vec.total_agents={args.total_agents} "
            f"vec.num_threads={args.threads} "
            f"policy={cfg.get('policy.hidden_size')}x{cfg.get('policy.num_layers')} "
            f"horizon={cfg.get('train.horizon')}",
            flush=True,
        )
    rows: list[EpisodeStats] = []
    original: EpisodeStats | None = None

    if args.original_train_only:
        rows.append(run_original_train(args))
        print_table(rows, None)
        if args.json_out:
            payload = {
                "rows": [
                    {**row.summary(), "env_sps": row.env_sps, "agent_sps": row.agent_sps}
                    for row in rows
                ]
            }
            Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {args.json_out}")
        return

    if not args.skip_original:
        original = run_original(args)
        rows.append(original)

    if not args.skip_sim:
        binary = ROOT / "wef_bench_sps"
        if args.no_rebuild and binary.is_file():
            print(f"reusing {binary}", flush=True)
        else:
            build_c_bench(binary)
        if not args.skip_single_thread:
            rows.append(
                run_c_bench(
                    binary,
                    threads=1,
                    total_agents=args.num_agents,
                    num_agents=args.num_agents,
                    episodes=args.episodes,
                    episode_length=args.episode_length,
                    warmup=args.warmup,
                    min_arena=args.min_arena,
                    max_arena=args.max_arena,
                    food_distribution=args.food_distribution,
                    num_food=args.num_food,
                    name="puffer sim (1 thread)",
                )
            )
        rows.append(
            run_c_bench(
                binary,
                threads=args.threads,
                total_agents=args.total_agents,
                num_agents=args.num_agents,
                episodes=args.episodes,
                episode_length=args.episode_length,
                warmup=args.warmup,
                min_arena=args.min_arena,
                max_arena=args.max_arena,
                food_distribution=args.food_distribution,
                num_food=args.num_food,
                name=f"puffer sim ({args.threads} threads)",
            )
        )

    if not args.skip_original_train:
        rows.append(run_original_train(args))

    if not args.skip_train:
        rows.append(run_train(args, cfg))

    print_table(rows, original)
    if args.json_out:
        payload = {
            "rows": [
                {
                    **row.summary(),
                    "env_sps": row.env_sps,
                    "agent_sps": row.agent_sps,
                }
                for row in rows
            ]
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
