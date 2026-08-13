#!/usr/bin/env python3
"""
Record WEF policy videos from the best sweep checkpoint.

Loads logs/wef/best_policy.ini (from scripts/sweep_eval.py) and runs
`./puffer render wef` (or ./build_cpu) for a few standard env settings:

  1) patchy_70   — 70×70 cm, 4 agents, patchy food
  2) uniform_70  — 70×70 cm, 4 agents, uniform food
  3) 1v1_scarce  — 70×70 cm, 2 agents, scarce patchy food

Outputs (default): logs/wef/videos/<setting>.mp4

Requires a headless-capable binary when no display is available:
  ./build.sh wef --headless
  # or for CPU-only render:
  ./build.sh wef --cpu --headless

Usage:
  python scripts/wef_videos.py
  python scripts/wef_videos.py --frames 512 --fps 20 --settings patchy_70,uniform_70
  python scripts/wef_videos.py --format gif --frames 300
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_INI = ROOT / "logs" / "wef" / "best_policy.ini"
DEFAULT_OUT_DIR = ROOT / "logs" / "wef" / "videos"

FOOD_UNIFORM = 0
FOOD_PATCHY = 1


@dataclass
class PolicySpec:
    name: str
    load_model_path: str
    hidden_size: int
    num_layers: int


@dataclass
class VideoSetting:
    name: str
    description: str
    num_agents: int
    arena: float
    food_distribution: int
    num_food: int
    episode_length: int = 512
    # num_patches = ceil(patch_density * arena * arena); default 0.001 → 5 on 70×70
    patch_density: float = 0.001
    patch_radius: float = 6.0
    patch_radius_std: float = 1.5


def patch_density_for_count(arena: float, num_patches: int) -> float:
    """Choose density so ceil(density * arena^2) == num_patches (for num_patches >= 1)."""
    if num_patches < 1:
        raise ValueError("num_patches must be >= 1")
    # ceil(d * A) = n  ⇔  n-1 < d*A <= n  → use d = n / A
    return float(num_patches) / (arena * arena)


SETTINGS: dict[str, VideoSetting] = {
    "patchy_70": VideoSetting(
        name="patchy_70",
        description="70×70 cm, 4 agents, patchy food (80 pellets, 3 patches)",
        num_agents=4,
        arena=70.0,
        food_distribution=FOOD_PATCHY,
        num_food=80,
        patch_density=patch_density_for_count(70.0, 3),
    ),
    "uniform_70": VideoSetting(
        name="uniform_70",
        description="70×70 cm, 4 agents, uniform food",
        num_agents=4,
        arena=70.0,
        food_distribution=FOOD_UNIFORM,
        num_food=64,
    ),
    "1v1_scarce": VideoSetting(
        name="1v1_scarce",
        description="70×70 cm, 1v1, scarce patchy food",
        num_agents=2,
        arena=70.0,
        food_distribution=FOOD_PATCHY,
        num_food=8,
    ),
}

DEFAULT_ORDER = list(SETTINGS.keys())


def parse_policy_ini(path: Path) -> PolicySpec:
    text = path.read_text(encoding="utf-8", errors="replace")
    vals: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip()
    model = vals.get("load_model_path", "")
    if not model:
        raise SystemExit(f"no load_model_path in {path}")
    if not Path(model).is_file():
        cand = ROOT / model
        if cand.is_file():
            model = cand.as_posix()
        else:
            raise SystemExit(f"checkpoint not found: {model}")
    return PolicySpec(
        name=vals.get("policy_name") or vals.get("run_id") or path.stem,
        load_model_path=model,
        hidden_size=int(float(vals.get("hidden_size", "128"))),
        num_layers=int(float(vals.get("num_layers", "3"))),
    )


def resolve_renderer(explicit: Path | None) -> tuple[Path, str]:
    """
    Return (binary, mode) where mode is 'render' (./puffer) or 'cpu' (./build_cpu).
    Paths are absolute so subprocess can exec them without relying on PATH.
    """
    def _mode(path: Path) -> str:
        name = path.name
        if name == "build_cpu" or "cpu" in name:
            return "cpu"
        return "render"

    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_absolute():
            cand = (Path.cwd() / path).resolve()
            if not cand.is_file():
                cand = (ROOT / path).resolve()
            path = cand
        else:
            path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"renderer not found: {explicit}")
        return path, _mode(path)

    puffer = ROOT / "puffer"
    build_cpu = ROOT / "build_cpu"
    if not os.environ.get("DISPLAY") and build_cpu.is_file():
        return build_cpu.resolve(), "cpu"
    if puffer.is_file():
        return puffer.resolve(), "render"
    if build_cpu.is_file():
        return build_cpu.resolve(), "cpu"
    raise SystemExit(
        "No renderer binary found.\n"
        "  Build GPU trainer:  ./build.sh wef --headless\n"
        "  Build CPU eval:     ./build.sh wef --cpu --headless"
    )


def setting_cmd(
    *,
    binary: Path,
    mode: str,
    policy: PolicySpec,
    setting: VideoSetting,
    out_path: Path,
    num_frames: int,
    fps: float,
    seed: int,
) -> list[str]:
    arena_i = int(round(setting.arena))
    common = [
        f"base.load_model_path={policy.load_model_path}",
        f"base.seed={seed}",
        f"base.num_frames={num_frames}",
        f"base.fps={fps}",
        f"base.gif_path={out_path.as_posix()}",
        f"policy.hidden_size={policy.hidden_size}",
        f"policy.num_layers={policy.num_layers}",
        f"vec.total_agents={setting.num_agents}",
        "vec.num_buffers=1",
        "vec.num_threads=1",
        f"env.num_agents={setting.num_agents}",
        f"env.food_distribution={setting.food_distribution}",
        f"env.num_food={setting.num_food}",
        f"env.patch_radius={setting.patch_radius}",
        f"env.patch_radius_std={setting.patch_radius_std}",
        f"env.patch_density={setting.patch_density}",
        f"env.min_arena_width={arena_i}",
        f"env.max_arena_width={arena_i}",
        f"env.min_arena_height={arena_i}",
        f"env.max_arena_height={arena_i}",
        f"env.episode_length={setting.episode_length}",
    ]
    if mode == "render":
        return [str(binary), "render", "wef", *common]
    return [str(binary), "wef", *common]


def record_video(
    *,
    binary: Path,
    mode: str,
    policy: PolicySpec,
    setting: VideoSetting,
    out_path: Path,
    num_frames: int,
    fps: float,
    seed: int,
    dry_run: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    cmd = setting_cmd(
        binary=binary,
        mode=mode,
        policy=policy,
        setting=setting,
        out_path=out_path,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
    )

    print(f"\n=== {setting.name} ===")
    print(f"  {setting.description}")
    print(f"  out: {out_path}")
    print(f"  frames={num_frames}  fps={fps}  seed={seed}")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        return

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))

    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout[-4000:])
        if not proc.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr[-6000:])
        hint = ""
        err = (proc.stderr or "") + (proc.stdout or "")
        if (
            "Failed to initialize" in err
            or "DISPLAY" in err
            or "GLFW" in err
            or "No displays" in err
        ):
            hint = (
                "\nHint: no display — rebuild headless:\n"
                "  ./build.sh wef --headless\n"
                "  # or: ./build.sh wef --cpu --headless\n"
            )
        raise SystemExit(
            f"render failed for {setting.name} (exit {proc.returncode}){hint}"
        )

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise SystemExit(
            f"render finished but video missing/empty: {out_path}\n"
            "Rebuild with raylib PLATFORM_MEMORY if headless: "
            "./build.sh wef --headless"
        )
    print(f"  wrote {out_path}  ({out_path.stat().st_size} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--policy-ini",
        type=Path,
        default=DEFAULT_POLICY_INI,
        help="best policy ini from sweep_eval.py (default logs/wef/best_policy.ini)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory (default logs/wef/videos)",
    )
    ap.add_argument(
        "--settings",
        default=",".join(DEFAULT_ORDER),
        help=(
            "comma-separated setting names, or 'all'. "
            f"Available: {', '.join(SETTINGS)}"
        ),
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=512,
        help="frames to record per video (default 512)",
    )
    ap.add_argument("--fps", type=float, default=20.0, help="output fps (default 20)")
    ap.add_argument(
        "--format",
        choices=("mp4", "gif"),
        default="mp4",
        help="container (default mp4 via libx264; gif uses palette encode)",
    )
    ap.add_argument("--seed", type=int, default=0, help="base.seed for env RNG")
    ap.add_argument(
        "--puffer",
        type=Path,
        default=None,
        help="path to puffer or build_cpu (default: ./puffer then ./build_cpu)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    ap.add_argument("--list", action="store_true", help="list settings and exit")
    args = ap.parse_args()

    if args.list:
        print("Available settings:")
        for key in DEFAULT_ORDER:
            s = SETTINGS[key]
            print(f"  {key:16s}  {s.description}")
        return

    if not args.policy_ini.is_file():
        raise SystemExit(
            f"policy ini not found: {args.policy_ini}\n"
            "  Run: python scripts/sweep_eval.py wef"
        )

    policy = parse_policy_ini(args.policy_ini)
    binary, mode = resolve_renderer(args.puffer)

    names = [n.strip() for n in args.settings.split(",") if n.strip()]
    if names == ["all"]:
        names = list(DEFAULT_ORDER)
    unknown = [n for n in names if n not in SETTINGS]
    if unknown:
        raise SystemExit(
            f"unknown settings: {unknown}\n  choose from: {', '.join(SETTINGS)}"
        )

    print(f"Policy: {policy.name}")
    print(f"  checkpoint: {policy.load_model_path}")
    print(f"  hidden_size={policy.hidden_size}  num_layers={policy.num_layers}")
    print(f"Renderer: {binary}  mode={mode}")
    print(f"Output:   {args.out_dir}/  (format=.{args.format})")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(names):
        setting = SETTINGS[name]
        out_path = args.out_dir / f"{setting.name}.{args.format}"
        record_video(
            binary=binary,
            mode=mode,
            policy=policy,
            setting=setting,
            out_path=out_path,
            num_frames=args.frames,
            fps=args.fps,
            seed=args.seed + i,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print("\nDone. Videos:")
        for name in names:
            p = args.out_dir / f"{SETTINGS[name].name}.{args.format}"
            if p.is_file():
                print(f"  {p}")


if __name__ == "__main__":
    main()
