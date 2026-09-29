#!/usr/bin/env python3
"""Plot PufferLib drive sweep metrics vs trial index.

Reads `sweep run=N score=...` from a ./puffer sweep log and the matching
logs/drive/sweep_*_<idx>.ini [metrics] dumps for offroad, collision, etc.

  uv run python ocean/drive/plot_sweep.py
  uv run python ocean/drive/plot_sweep.py --log ~/.grok/long-running-background-tasks/puffer_drive_PID.log
  uv run python ocean/drive/plot_sweep.py --out-dir logs/drive
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from configparser import ConfigParser
from pathlib import Path

SWEEP_RE = re.compile(
    r"sweep run=(?P<run>\d+)\s+score=(?P<score>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?:\s+cost=(?P<cost>[-+]?\d+(?:\.\d+)?))?"
    r"(?:\s+steps=(?P<steps>[-+]?\d+(?:\.\d+)?))?"
    r"(?:\s+random=(?P<random>\d+))?"
    r"(?:\s+gp_obs=(?P<gp_obs>\d+))?"
    r"(?:\s+pareto=(?P<pareto>\d+))?"
)

INI_LAST = (
    ("score", "env/score"),
    ("offroad", "env/offroad_rate"),
    ("collision", "env/collision_rate"),
    ("completion", "env/completion_rate"),
    ("dnf", "env/dnf_rate"),
    ("return", "env/episode_return"),
    ("clean_collision", "env/clean_collision_rate"),
    ("perf", "env/perf"),
)

DEFAULT_LOG_GLOBS = (
    str(Path.home() / ".grok/long-running-background-tasks/puffer_drive_*.log"),
)


def parse_sweep_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = SWEEP_RE.search(line)
        if not m:
            continue
        row = {k: v for k, v in m.groupdict().items() if v is not None}
        row["run"] = int(row["run"])
        row["score"] = float(row["score"])
        for key in ("cost", "steps"):
            if key in row:
                row[key] = float(row[key])
        if "random" in row:
            row["random"] = int(row["random"])
        rows.append(row)
    rows.sort(key=lambda r: r["run"])
    return rows


def _csv_floats(raw: str) -> list[float]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def parse_trial_ini(path: Path) -> dict:
    text = path.read_text(errors="replace")
    if "[metrics]" not in text:
        return {}
    series: dict[str, list[float]] = {}
    in_metrics = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[metrics]":
            in_metrics = True
            continue
        if in_metrics and stripped.startswith("["):
            break
        if not in_metrics or not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        vals = _csv_floats(raw)
        if vals:
            series[key.strip()] = vals
    out: dict = {"path": str(path), "curves": series}
    for dest, src in INI_LAST:
        if src in series:
            out[dest] = series[src][-1]
    if "agent_steps" in series:
        out["ini_steps"] = series["agent_steps"][-1]
    cfg = ConfigParser(interpolation=None)
    cfg.optionxform = str
    try:
        cfg.read_string(text)
        if cfg.has_option("policy", "hidden_size"):
            out["hidden_size"] = float(cfg.get("policy", "hidden_size"))
        if cfg.has_option("policy", "num_layers"):
            out["num_layers"] = float(cfg.get("policy", "num_layers"))
        if cfg.has_option("train", "total_timesteps"):
            out["total_timesteps"] = float(cfg.get("train", "total_timesteps"))
        if cfg.has_option("vec", "total_agents"):
            out["total_agents"] = float(cfg.get("vec", "total_agents"))
        if cfg.has_option("base", "run_id"):
            out["run_id"] = cfg.get("base", "run_id")
    except Exception:
        pass
    return out


def attach_inis(rows: list[dict], logs_dir: Path) -> None:
    by_idx: dict[int, Path] = {}
    for path in logs_dir.glob("sweep_*_*.ini"):
        m = re.search(r"_(\d+)\.ini$", path.name)
        if m:
            by_idx[int(m.group(1))] = path
    for row in rows:
        path = by_idx.get(row["run"])
        if path is None:
            continue
        extra = parse_trial_ini(path)
        for k, v in extra.items():
            if k == "curves":
                row["curves"] = v
            elif k not in row:
                row[k] = v
        if "score" not in extra and "score" in row:
            pass
        elif "score" in extra:
            row["score"] = extra["score"]


def latest_sweep_log() -> Path | None:
    candidates = []
    for pattern in DEFAULT_LOG_GLOBS:
        for match in glob.glob(pattern):
            p = Path(match)
            if p.is_file() and parse_sweep_log(p):
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _running_best(ys: list[float]) -> list[float]:
    best = []
    running = float("-inf")
    for y in ys:
        running = max(running, y)
        best.append(running)
    return best


def plot_index(rows: list[dict], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    xs = [r["run"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=12)

    def series(key):
        return xs, [r[key] for r in rows if key in r], [r["run"] for r in rows if key in r]

    ax = axes[0, 0]
    scores = [r["score"] for r in rows]
    ax.plot(xs, scores, marker="o", color="tab:blue", label="score")
    ax.plot(xs, _running_best(scores), color="tab:orange", label="best so far")
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Score")

    ax = axes[0, 1]
    for key, label, color, ls in (
        ("offroad", "offroad", "tab:orange", "-"),
        ("collision", "collision", "tab:red", "-"),
        ("clean_collision", "clean collision", "tab:pink", "--"),
    ):
        _, ys, xk = series(key)
        if ys:
            ax.plot(xk, ys, marker="o", color=color, linestyle=ls, label=label)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Safety")

    ax = axes[1, 0]
    for key, label, color in (
        ("completion", "completion", "tab:green"),
        ("dnf", "dnf", "tab:purple"),
    ):
        _, ys, xk = series(key)
        if ys:
            ax.plot(xk, ys, marker="o", color=color, label=label)
    ax.set_xlabel("sweep index")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Completion / DNF")

    ax = axes[1, 1]
    _, ys, xk = series("return")
    if ys:
        ax.plot(xk, ys, marker="o", color="tab:blue", label="episode return")
    ax.set_xlabel("sweep index")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Episode return")

    if xs:
        for a in axes.ravel():
            a.set_xticks(xs)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_curves(rows: list[dict], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    fig.suptitle(title, fontsize=12)
    panels = (
        ("env/score", "score"),
        ("env/offroad_rate", "offroad"),
        ("env/collision_rate", "collision"),
    )
    for ax, (key, label) in zip(axes, panels):
        for row in rows:
            curves = row.get("curves") or {}
            ys = curves.get(key)
            xs = curves.get("agent_steps")
            if not ys or not xs:
                continue
            ax.plot([x / 1e6 for x in xs], ys, alpha=0.75, linewidth=1.2,
                    label=f"run {row['run']}")
        ax.set_title(label)
        ax.set_xlabel("steps (M)")
        if label != "score":
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    if len(rows) <= 16:
        axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def latest_log() -> Path | None:
    return latest_sweep_log()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/drive"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/drive"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Deprecated alias for score-only PNG; ignored if --out-dir is used")
    args = parser.parse_args()

    log = args.log or latest_sweep_log()
    if log is None:
        print("no sweep log found; pass --log", file=sys.stderr)
        return 1
    log = log.resolve()
    rows = parse_sweep_log(log)
    if not rows:
        print(f"no sweep run= lines in {log}", file=sys.stderr)
        return 1
    attach_inis(rows, args.logs_dir)

    best = max(rows, key=lambda r: r["score"])
    title = (
        f"{log.name}  n={len(rows)}  "
        f"best={best['score']:.4f} @ run {best['run']}"
    )
    out_dir = args.out_dir
    metrics_path = out_dir / "sweep_metrics.png"
    curves_path = out_dir / "sweep_curves.png"
    score_path = args.out or (out_dir / "sweep_score.png")

    plot_index(rows, metrics_path, title)
    plot_curves(rows, curves_path, title + "  (learning curves)")
    # Keep a score-only PNG for the earlier one-liner workflow.
    import matplotlib.pyplot as plt
    xs = [r["run"] for r in rows]
    ys = [r["score"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, marker="o", linewidth=1.4, color="tab:blue", label="score")
    ax.plot(xs, _running_best(ys), linewidth=1.6, color="tab:orange", label="best so far")
    ax.set_xlabel("sweep index")
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xticks(xs)
    fig.tight_layout()
    score_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(score_path, dpi=140)
    plt.close(fig)

    print(
        f"wrote {metrics_path} {curves_path} {score_path}  "
        f"trials={len(rows)}  best={best['score']:.4f} @ run {best['run']}"
        + (f"  id={best.get('run_id')}" if best.get("run_id") else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
