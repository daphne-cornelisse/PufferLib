#!/usr/bin/env python3
"""Craftax eval: death rollouts plus training-log plots.

Death eval compiles scripts/craftax_death_eval.c and writes tail videos plus
a diagnosis under logs/craftax_clean/death_eval/.

Plot mode reads every *.ini in a log directory and writes env/perf, env/score,
and floor achievement-rate curves.

Usage:
    python scripts/craftax_death_eval.py
    python scripts/craftax_death_eval.py --seeds 0,1,2,3,4,5,6,7,8,9
    python scripts/craftax_death_eval.py --plot-only --plot-dir logs/craftax_clean_act_mask
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "craftax_death_eval.c"
BIN = ROOT / "build" / "craftax_death_eval"
OUT = ROOT / "logs" / "craftax_clean" / "death_eval"
WEIGHTS = ROOT / "resources" / "craftax_clean" / "craftax_clean_weights.bin"
PLOT_DIR = ROOT / "logs" / "craftax_clean"

FLOOR_KEYS = [
    ("env/floor_0_overworld", "0 overworld"),
    ("env/floor_1_dungeon", "1 dungeon"),
    ("env/floor_2_gnomish_mines", "2 mines"),
    ("env/floor_3_sewers", "3 sewers"),
    ("env/floor_4_vault", "4 vault"),
    ("env/floor_5_troll_mines", "5 troll mines"),
    ("env/floor_6_fire_realm", "6 fire"),
    ("env/floor_7_ice_realm", "7 ice"),
    ("env/floor_8_graveyard", "8 graveyard"),
]


def compile_eval() -> None:
    BIN.parent.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    if system == "Darwin":
        raylib = ROOT / "raylib-5.5_macos"
        ldflags = [
            str(raylib / "lib" / "libraylib.a"),
            "-framework", "Cocoa",
            "-framework", "IOKit",
            "-framework", "CoreVideo",
            "-framework", "OpenGL",
        ]
    else:
        raylib = ROOT / "raylib-5.5_linux_amd64"
        ldflags = [
            str(raylib / "lib" / "libraylib.a"),
            "-lGL", "-ldl",
        ]
    cmd = [
        os.environ.get("CC", "clang"),
        "-std=c11",
        "-O2",
        "-Wno-unused-function",
        "-Wno-unused-variable",
        "-DPLATFORM_DESKTOP",
        "-DPUF_CRAFTAX_NET",
        "-I", str(ROOT),
        "-I", str(ROOT / "src"),
        "-I", str(ROOT / "vendor"),
        "-I", str(ROOT / "ocean" / "craftax_clean"),
        "-I", str(raylib / "include"),
        str(SRC),
        "-o", str(BIN),
        "-lm", "-lpthread",
        *ldflags,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def parse_ini_metrics(path: Path) -> dict:
    metrics: dict[str, list[float]] = {}
    run_id = path.stem
    in_metrics = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            in_metrics = s[1:-1] == "metrics"
            continue
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key == "run_id":
            run_id = val
        if not in_metrics:
            continue
        try:
            metrics[key] = [float(x) for x in val.split(",") if x.strip()]
        except ValueError:
            pass
    return {"path": path, "run_id": run_id, "m": metrics}


def plot_logs(log_dir: Path) -> int:
    import matplotlib.pyplot as plt

    runs = []
    for path in sorted(log_dir.glob("*.ini")):
        run = parse_ini_metrics(path)
        m = run["m"]
        if "env/perf" not in m or "agent_steps" not in m:
            continue
        n = min(len(m["env/perf"]), len(m["agent_steps"]))
        if n == 0:
            continue
        run["steps"] = [x / 1e6 for x in m["agent_steps"][:n]]
        run["perf"] = m["env/perf"][:n]
        run["final_perf"] = run["perf"][-1]
        runs.append(run)
    if not runs:
        print(f"no ini files with env/perf in {log_dir}")
        return 1

    for run in runs:
        score = run["m"].get("env/score") or []
        n = min(len(score), len(run["steps"]))
        run["score"] = score[:n]
        run["final_score"] = run["score"][-1] if n else 0.0
        ach = run["m"].get("env/achievement_rate") or []
        n = min(len(ach), len(run["steps"]))
        run["achievement_rate"] = ach[:n]
        run["final_ach"] = run["achievement_rate"][-1] if n else 0.0
    runs.sort(key=lambda r: r["final_perf"])
    cmap = plt.cm.plasma
    n_runs = max(len(runs) - 1, 1)

    def plot_metric(values_key, ylabel, title, path, ymin, ymax):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, run in enumerate(runs):
            ys = run[values_key]
            n = min(len(ys), len(run["steps"]))
            if n == 0:
                continue
            color = cmap(i / n_runs)
            best = i == len(runs) - 1
            ax.plot(run["steps"][:n], ys[:n], color=color,
                lw=2.2 if best else 0.9, alpha=1.0 if best else 0.45,
                label=run["run_id"] if best else None)
        ax.set_xlabel("env steps (M)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
        if any(ax.get_legend_handles_labels()[0]):
            ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"wrote {path}")

    plot_metric("perf", "env/perf",
        f"Craftax perf  ({len(runs)} runs in {log_dir})",
        log_dir / "perf.png", 0,
        max(1.0, max(r["final_perf"] for r in runs) * 1.05))
    plot_metric("score", "env/score",
        f"Craftax score  ({len(runs)} runs in {log_dir})",
        log_dir / "score.png",
        min(0.0, min(r["final_score"] for r in runs) * 1.05),
        max(1.0, max(r["final_score"] for r in runs) * 1.05))
    plot_metric("achievement_rate", "env/achievement_rate",
        f"Craftax achievement rate  ({len(runs)} runs in {log_dir})",
        log_dir / "achievement_rate.png", 0,
        max(1.0, max((r["final_ach"] for r in runs), default=0.0) * 1.05))

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.ravel(), FLOOR_KEYS):
        for i, run in enumerate(runs):
            series = run["m"].get(key)
            if not series:
                continue
            n = min(len(series), len(run["steps"]))
            color = cmap(i / n_runs)
            best = i == len(runs) - 1
            ax.plot(run["steps"][:n], series[:n], color=color,
                lw=2.0 if best else 0.8, alpha=1.0 if best else 0.4)
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
    fig.supxlabel("env steps (M)")
    fig.supylabel("episode ratio unlocking floor")
    fig.suptitle(f"Floor achievement rates  ({len(runs)} runs)")
    fig.tight_layout()
    floors_path = log_dir / "floors.png"
    fig.savefig(floors_path, dpi=130)
    plt.close(fig)
    print(f"wrote {floors_path}")

    print("run_id                              steps_M  perf    score   ach     dungeon  mines   sewers")
    for run in reversed(runs):
        m = run["m"]
        def last(key):
            xs = m.get(key) or [0.0]
            return xs[-1]
        print(f"{run['run_id']:<34s} {run['steps'][-1]:7.1f}  "
              f"{run['final_perf']:.3f}  {run['final_score']:6.2f}  "
              f"{run['final_ach']:.3f}  "
              f"{last('env/floor_1_dungeon'):.3f}   "
              f"{last('env/floor_2_gnomish_mines'):.3f}   "
              f"{last('env/floor_3_sewers'):.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9",
                    help="comma-separated world seeds (reset-pool indices)")
    ap.add_argument("--sample", action="store_true",
                    help="softmax-sample actions instead of greedy argmax")
    ap.add_argument("--full", action="store_true",
                    help="record a full spawn-to-death playthrough (30 fps)")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--plot-dir", default=str(PLOT_DIR),
                    help="directory of sweep *.ini files to plot")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip death rollouts; only plot ini metrics")
    ap.add_argument("--skip-plot", action="store_true",
                    help="skip ini metric plots")
    args = ap.parse_args()
    rc = 0
    if not args.plot_only:
        if not args.skip_compile:
            compile_eval()
        Path(args.out).mkdir(parents=True, exist_ok=True)
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            ap.error("need at least one seed")
        cmd = [str(BIN)]
        if args.sample:
            cmd.append("--sample")
        if args.full:
            cmd.append("--full")
        cmd.extend([args.weights, args.out, *seeds])
        print(" ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            return rc
    if not args.skip_plot:
        rc = plot_logs(Path(args.plot_dir))
    return rc


if __name__ == "__main__":
    sys.exit(main())
