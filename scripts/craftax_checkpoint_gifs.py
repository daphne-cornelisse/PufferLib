#!/usr/bin/env python3
"""Roll out craftax_scale_best.bin for 5 seeds and save MP4s.

MP4s are named by episode return and length:
    recordings/craftax_ret{score}_len{steps}.mp4
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "resources" / "craftax" / "craftax_scale_best.bin"
PUFFER = ROOT / "puffer"
OUT_DIR = ROOT / "recordings"
SEEDS = [11, 23, 47, 89, 131]
EVAL_RE = re.compile(
    r"CPU_EVAL\b.*\bscore=(?P<score>[-0-9.]+).*?\bsteps=(?P<steps>-?\d+)"
)


def run_episode(seed: int, mp4: Path) -> tuple[float, int]:
    cmd = [
        str(PUFFER),
        "eval",
        str(CKPT),
        "--policy.hidden_size=1024",
        "--policy.num_layers=4",
        f"--env.seed_offset={seed}",
        "--eval_episodes=1",
        "--headless",
        "--video",
        str(mp4),
        "--video-fps=15",
    ]
    print("Running", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )
    match = None
    for line in proc.stdout.splitlines():
        print(line, flush=True)
        found = EVAL_RE.search(line)
        if found:
            match = found
    if not match:
        raise RuntimeError(f"no CPU_EVAL line for seed={seed}\n{proc.stdout}")
    return float(match.group("score")), int(match.group("steps"))


def transcode_mp4(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "scale=720:-2:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "23",
        "-movflags", "+faststart",
        "-an",
        "-loglevel", "error",
        str(dst),
    ]
    print("Writing", dst.name, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    if not PUFFER.exists():
        raise SystemExit(f"missing {PUFFER}; build with: bash build.sh craftax")
    if not CKPT.exists():
        raise SystemExit(f"missing checkpoint {CKPT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    used: dict[str, int] = {}
    for seed in SEEDS:
        tmp = OUT_DIR / f"craftax_ckpt_seed{seed}.mp4"
        score, steps = run_episode(seed, tmp)
        stem = f"craftax_ret{score:.2f}_len{steps}"
        if stem in used:
            stem = f"{stem}_seed{seed}"
        used[stem] = seed
        out = OUT_DIR / f"{stem}.mp4"
        if tmp.resolve() != out.resolve():
            transcode_mp4(tmp, out)
            tmp.unlink(missing_ok=True)
        print(f"seed={seed} return={score:.4f} length={steps} -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
