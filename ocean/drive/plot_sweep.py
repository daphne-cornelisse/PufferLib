#!/usr/bin/env python3
"""Plot drive sweep score vs trial index.

Parses `sweep run=N score=...` lines from a ./puffer sweep log.

  uv run python ocean/drive/plot_sweep.py
  uv run python ocean/drive/plot_sweep.py --log ~/.grok/long-running-background-tasks/puffer_drive_PID.log
  uv run python ocean/drive/plot_sweep.py --out logs/drive/sweep_score.png
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

SWEEP_RE = re.compile(
    r"sweep run=(?P<run>\d+)\s+score=(?P<score>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?:\s+cost=(?P<cost>[-+]?\d+(?:\.\d+)?))?"
    r"(?:\s+steps=(?P<steps>[-+]?\d+(?:\.\d+)?))?"
    r"(?:\s+random=(?P<random>\d+))?"
    r"(?:\s+gp_obs=(?P<gp_obs>\d+))?"
    r"(?:\s+pareto=(?P<pareto>\d+))?"
)

DEFAULT_LOG_GLOBS = (
    str(Path.home() / ".grok/long-running-background-tasks/puffer_drive_*.log"),
)


def parse_sweep(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = SWEEP_RE.search(line)
        if not m:
            continue
        row = {k: v for k, v in m.groupdict().items() if v is not None}
        row["run"] = int(row["run"])
        row["score"] = float(row["score"])
        if "cost" in row:
            row["cost"] = float(row["cost"])
        if "steps" in row:
            row["steps"] = float(row["steps"])
        if "random" in row:
            row["random"] = int(row["random"])
        rows.append(row)
    rows.sort(key=lambda r: r["run"])
    return rows


def latest_sweep_log() -> Path | None:
    candidates = []
    for pattern in DEFAULT_LOG_GLOBS:
        for match in glob.glob(pattern):
            p = Path(match)
            if p.is_file() and parse_sweep(p):
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_sweep(rows: list[dict], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    xs = [r["run"] for r in rows]
    ys = [r["score"] for r in rows]
    best = []
    running = float("-inf")
    for y in ys:
        running = max(running, y)
        best.append(running)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, marker="o", linewidth=1.4, color="tab:blue", label="score")
    ax.plot(xs, best, linewidth=1.6, color="tab:orange", label="best so far")
    ax.set_xlabel("sweep index")
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if xs:
        ax.set_xticks(xs)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("logs/drive/sweep_score.png"))
    args = parser.parse_args()

    log = args.log or latest_sweep_log()
    if log is None:
        print("no sweep log found; pass --log", file=sys.stderr)
        return 1
    log = log.resolve()
    rows = parse_sweep(log)
    if not rows:
        print(f"no sweep run= lines in {log}", file=sys.stderr)
        return 1

    best = max(rows, key=lambda r: r["score"])
    title = (
        f"{log.name}  n={len(rows)}  "
        f"best={best['score']:.4f} @ run {best['run']}"
    )
    plot_sweep(rows, args.out, title)
    print(
        f"wrote {args.out}  trials={len(rows)}  "
        f"best={best['score']:.4f} @ run {best['run']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
