#!/usr/bin/env python3
"""Replay the official Craftax human playthrough (`run1`) and write an mp4.

The dataset is the mixed-skill human trajectory zip from
https://github.com/MichaelTMatthews/Craftax (Offline Dataset). `run1` is the
only trajectory that finishes the game. Trajectories were recorded on Craftax
v1.1.0, so this script must run with that package version.

Usage (from repo root, after extracting the zip):
    /tmp/craftax110/bin/python scripts/render_craftax_run1.py
"""
from __future__ import annotations

import argparse
import bz2
import os
import pickle
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJ = ROOT / "logs/craftax_clean/run1_replay/people/run1.pbz2"
DEFAULT_OUT = ROOT / "logs/craftax_clean/run1_replay/run1.mp4"
DEFAULT_LOG = ROOT / "logs/craftax_clean/run1_replay/run1_summary.txt"

LEVEL_NAMES = [
    "overworld",
    "dungeon",
    "gnomish_mines",
    "sewers",
    "vault",
    "troll_mines",
    "fire_realm",
    "ice_realm",
    "graveyard",
]


def load_compressed_pickle(path: Path):
    with bz2.BZ2File(path, "rb") as f:
        return pickle.load(f)


def stack_states(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--scale", type=int, default=4, help="nearest-neighbor upsample")
    args = ap.parse_args()

    if not args.traj.exists():
        print(f"missing trajectory {args.traj}", file=sys.stderr)
        return 1

    from craftax.constants import (
        Achievement,
        BLOCK_PIXEL_SIZE_IMG,
        INVENTORY_OBS_HEIGHT,
        OBS_DIM,
    )
    from craftax.renderer import render_craftax_pixels

    print(f"loading {args.traj}", flush=True)
    data = load_compressed_pickle(args.traj)
    states = data["state"]
    actions = np.asarray(data["action"])
    rewards = np.asarray([float(np.asarray(r)) for r in data["reward"]], dtype=np.float32)
    dones = np.asarray([bool(np.asarray(d)) for d in data["done"]])
    n = len(states)
    print(
        f"run1: {n} states, {len(actions)} actions, "
        f"reward_sum={rewards.sum():.2f}, last_done={bool(dones[-1])}",
        flush=True,
    )

    render_one = jax.jit(
        lambda s: render_craftax_pixels(
            s, BLOCK_PIXEL_SIZE_IMG, do_night_noise=False
        )
    )
    render_batch = jax.jit(
        jax.vmap(
            lambda s: render_craftax_pixels(
                s, BLOCK_PIXEL_SIZE_IMG, do_night_noise=False
            )
        )
    )

    warmup = np.asarray(render_one(states[0]))
    tile = BLOCK_PIXEL_SIZE_IMG
    h0, w0 = warmup.shape[:2]
    scale = args.scale
    w = w0 * scale
    h = h0 * scale
    print(
        f"frame {w0}x{h0} -> {w}x{h}  obs={OBS_DIM} inv={INVENTORY_OBS_HEIGHT} tile={tile}",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20",
        "-loglevel", "error",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    ach_by_value = {a.value: a.name for a in Achievement}
    n_ach_slots = max(ach_by_value) + 1
    prev_ach = np.zeros(n_ach_slots, dtype=np.int32)
    timeline = []
    max_level = 0
    indices = list(range(0, n, args.stride))
    batch = args.batch
    written = 0
    try:
        i = 0
        while i < len(indices):
            chunk_idx = indices[i : i + batch]
            chunk = [states[j] for j in chunk_idx]
            if len(chunk) == 1:
                frames = np.asarray(render_one(chunk[0]))[None]
            else:
                frames = np.asarray(render_batch(stack_states(chunk)))
            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)
            if scale != 1:
                frames = np.repeat(np.repeat(frames, scale, axis=1), scale, axis=2)
            proc.stdin.write(frames.tobytes())
            written += len(chunk)

            for local, si in enumerate(chunk_idx):
                st = chunk[local]
                level = int(np.asarray(st.player_level))
                if level > max_level:
                    max_level = level
                    timeline.append(
                        f"t={si:5d}  ENTER {LEVEL_NAMES[level]} (floor {level})  "
                        f"pos={np.asarray(st.player_position).tolist()}  "
                        f"hp={float(np.asarray(st.player_health)):.1f}"
                    )
                    print(timeline[-1], flush=True)
                ach = np.asarray(st.achievements).astype(np.int32).reshape(-1)
                if ach.size > prev_ach.size:
                    prev_ach = np.pad(prev_ach, (0, ach.size - prev_ach.size))
                newly = np.where((prev_ach[: ach.size] == 0) & (ach == 1))[0]
                for a in newly:
                    name = ach_by_value.get(int(a), f"ACH_{a}")
                    timeline.append(
                        f"t={si:5d}  {name}  "
                        f"floor={LEVEL_NAMES[int(np.asarray(st.player_level))]}  "
                        f"hp={float(np.asarray(st.player_health)):.1f}"
                    )
                    print(timeline[-1], flush=True)
                prev_ach = ach
            if written % 512 == 0 or i + batch >= len(indices):
                last = chunk[-1]
                print(
                    f"  rendered {written}/{len(indices)}  "
                    f"floor={int(np.asarray(last.player_level))}  "
                    f"pos={np.asarray(last.player_position).tolist()}  "
                    f"hp={float(np.asarray(last.player_health)):.1f}  "
                    f"boss={int(np.asarray(last.boss_progress))}",
                    flush=True,
                )
            i += batch
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        print(f"ffmpeg failed with {rc}", file=sys.stderr)
        return rc

    # play_craftax auto-resets on done, so the final stored state is a new episode.
    win_i = n - 1
    for i in range(n - 1, -1, -1):
        if int(np.asarray(states[i].timestep)) > 100:
            win_i = i
            break
    last = states[win_i]
    n_ach = int(np.asarray(last.achievements).sum())
    duration_s = written / float(args.fps)
    summary = [
        "Craftax official human playthrough: run1",
        f"trajectory: {args.traj}",
        f"states: {n}  actions: {len(actions)}  stride: {args.stride}",
        f"reward_sum: {rewards.sum():.3f}  last_done: {bool(dones[-1])}",
        f"win_state_index: {win_i} (last frames may be auto-reset)",
        f"final_floor: {int(np.asarray(last.player_level))} ({LEVEL_NAMES[int(np.asarray(last.player_level))]})",
        f"max_floor: {max_level} ({LEVEL_NAMES[max_level]})",
        f"boss_progress: {int(np.asarray(last.boss_progress))}",
        f"achievements: {n_ach}/{len(ach_by_value)}",
        f"final_hp: {float(np.asarray(last.player_health)):.2f}",
        f"final_pos: {np.asarray(last.player_position).tolist()}",
        f"timestep: {int(np.asarray(last.timestep))}",
        f"video: {args.out}  {w}x{h}  {args.fps} fps  {written} frames  {duration_s:.1f}s",
        "",
        "timeline:",
        *timeline,
        "",
        "final achievements:",
        *[
            f"  {ach_by_value[i_a]}"
            for i_a in sorted(ach_by_value)
            if i_a < len(prev_ach) and prev_ach[i_a]
        ],
    ]
    args.log.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"wrote {args.out}")
    print(f"wrote {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
