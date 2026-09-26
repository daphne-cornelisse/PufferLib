#!/usr/bin/env python3
"""Plot drive training score, offroad rate, and collision rate.

Reads a live dashboard log (./puffer train stdout) or a finished
logs/drive/<run>.ini [metrics] dump.

  uv run python ocean/drive/plot_train.py
  uv run python ocean/drive/plot_train.py --log ~/.grok/long-running-background-tasks/puffer_drive_PID.log
  uv run python ocean/drive/plot_train.py --watch 15 --out logs/drive/train_plot.png
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
FRAME_SPLIT_RE = re.compile(r"[╭┌]")
ABBREV_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)([KMBT])?$", re.I)
FLOAT_RE = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

DASH_FIELDS = (
    ("epoch", re.compile(r"Epoch\s+(\d+)")),
    ("steps", re.compile(r"Steps\s+(\S+)")),
    ("sps", re.compile(r"SPS\s+(\S+)")),
    ("loss", re.compile(r"Losses\s+" + FLOAT_RE)),
    ("policy_loss", re.compile(r"policy\s+" + FLOAT_RE)),
    ("value_loss", re.compile(r"value\s+" + FLOAT_RE)),
    ("entropy", re.compile(r"entropy\s+" + FLOAT_RE)),
    ("kl", re.compile(r"\bkl\s+" + FLOAT_RE)),
    ("clipfrac", re.compile(r"clipfrac\s+" + FLOAT_RE)),
    ("score", re.compile(r"score\s+" + FLOAT_RE)),
    ("completion_rate", re.compile(r"completion_rate\s+" + FLOAT_RE)),
    ("offroad_rate", re.compile(r"offroad_rate\s+" + FLOAT_RE)),
    ("collision_rate", re.compile(r"collision_rate\s+" + FLOAT_RE)),
    ("clean_collision_rate", re.compile(r"clean_collision_rate\s+" + FLOAT_RE)),
    ("dnf_rate", re.compile(r"dnf_rate\s+" + FLOAT_RE)),
    ("episode_return", re.compile(r"episode_return\s+" + FLOAT_RE)),
    ("episode_length", re.compile(r"episode_length\s+" + FLOAT_RE)),
    ("perf", re.compile(r"\bperf\s+" + FLOAT_RE)),
)

INI_KEY_MAP = {
    "agent_steps": "steps",
    "SPS": "sps",
    "epoch": "epoch",
    "env/score": "score",
    "env/perf": "perf",
    "env/completion_rate": "completion_rate",
    "env/offroad_rate": "offroad_rate",
    "env/collision_rate": "collision_rate",
    "env/clean_collision_rate": "clean_collision_rate",
    "env/dnf_rate": "dnf_rate",
    "env/episode_return": "episode_return",
    "env/episode_length": "episode_length",
    "loss/total": "loss",
    "loss/policy": "policy_loss",
    "loss/value": "value_loss",
    "loss/entropy": "entropy",
    "loss/kl": "kl",
    "loss/clipfrac": "clipfrac",
}

DEFAULT_LOG_GLOBS = (
    str(Path.home() / ".grok/long-running-background-tasks/puffer_drive_*.log"),
    "logs/drive/*.ini",
)


def parse_abbrev(token: str) -> float | None:
    token = token.strip().rstrip("│|")
    m = ABBREV_RE.match(token)
    if not m:
        try:
            return float(token)
        except ValueError:
            return None
    value = float(m.group(1))
    suf = (m.group(2) or "").upper()
    scale = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suf]
    return value * scale


def _match_float(rx: re.Pattern, text: str) -> float | None:
    m = rx.search(text)
    if not m:
        return None
    if rx.pattern.startswith("Steps") or rx.pattern.startswith("SPS"):
        return parse_abbrev(m.group(1))
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_dashboard(text: str) -> list[dict]:
    text = ANSI_RE.sub("", text)
    rows: list[dict] = []
    for chunk in FRAME_SPLIT_RE.split(text):
        if "score" not in chunk or "Steps" not in chunk:
            continue
        row: dict = {}
        for key, rx in DASH_FIELDS:
            val = _match_float(rx, chunk)
            if val is not None:
                row[key] = val
        if "steps" in row and "score" in row:
            rows.append(row)
    return rows


def parse_ini_metrics(text: str) -> list[dict]:
    if "[metrics]" not in text:
        return []
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
        key = key.strip()
        mapped = INI_KEY_MAP.get(key)
        if mapped is None:
            continue
        values = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            values.append(float(part))
        if values:
            series[mapped] = values
    if "steps" not in series:
        return []
    n = len(series["steps"])
    rows = []
    for i in range(n):
        row = {}
        for key, vals in series.items():
            if i < len(vals):
                row[key] = vals[i]
        rows.append(row)
    return rows


def parse_log(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    if path.suffix == ".ini" or text.lstrip().startswith("# PufferLib log"):
        rows = parse_ini_metrics(text)
        if rows:
            return rows
    return parse_dashboard(text)


def latest_log() -> Path | None:
    candidates: list[Path] = []
    for pattern in DEFAULT_LOG_GLOBS:
        for match in glob.glob(pattern):
            p = Path(match)
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def rolling_mean(xs: list[float], window: int) -> list[float]:
    if window <= 1 or len(xs) < 2:
        return xs
    w = min(window, len(xs))
    out = []
    acc = 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
            out.append(acc / w)
        else:
            out.append(acc / (i + 1))
    return out


def _xy(rows: list[dict], key: str, smooth: int) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in rows:
        if key in row and "steps" in row:
            xs.append(row["steps"] / 1e6)
            ys.append(row[key])
    return xs, rolling_mean(ys, smooth)


def plot_rows(rows: list[dict], out: Path, title: str, smooth: int, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=12)

    series = (
        ("score", "score", "tab:blue"),
        ("offroad_rate", "offroad", "tab:orange"),
        ("collision_rate", "collision", "tab:red"),
    )
    for key, label, color in series:
        xs, ys = _xy(rows, key, smooth)
        if xs:
            ax.plot(xs, ys, label=label, color=color, linewidth=1.8)

    ax.set_xlabel("steps (M)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def render(log: Path, out: Path, smooth: int, show: bool) -> int:
    rows = parse_log(log)
    if not rows:
        print(f"no training frames in {log}", file=sys.stderr)
        return 1
    last = rows[-1]
    title = (
        f"{log.name}  n={len(rows)}  "
        f"steps={last.get('steps', 0)/1e6:.1f}M  "
        f"score={last.get('score', float('nan')):.3f}  "
        f"offroad={last.get('offroad_rate', float('nan')):.3f}  "
        f"collision={last.get('collision_rate', float('nan')):.3f}"
    )
    plot_rows(rows, out, title, smooth, show)
    print(
        f"wrote {out}  frames={len(rows)}  "
        f"steps={last.get('steps', 0):.0f}  "
        f"score={last.get('score', float('nan')):.3f}  "
        f"offroad={last.get('offroad_rate', float('nan')):.3f}  "
        f"collision={last.get('collision_rate', float('nan')):.3f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None, help="Dashboard .log or logs/drive/<run>.ini")
    parser.add_argument("--out", type=Path, default=Path("logs/drive/train_plot.png"))
    parser.add_argument("--smooth", type=int, default=1, help="Rolling-mean window in epochs (1 = off)")
    parser.add_argument("--watch", type=float, nargs="?", const=10.0, default=None,
                        help="Resave every N seconds (default 10 if flag is bare)")
    parser.add_argument("--show", action="store_true", help="Open an interactive window")
    args = parser.parse_args()

    log = args.log or latest_log()
    if log is None:
        print("no training log found; pass --log", file=sys.stderr)
        return 1
    log = log.resolve()
    if not log.is_file():
        print(f"missing log {log}", file=sys.stderr)
        return 1

    if args.watch is None:
        return render(log, args.out, args.smooth, args.show)

    print(f"watching {log} every {args.watch:.0f}s -> {args.out}")
    while True:
        try:
            render(log, args.out, args.smooth, False)
        except Exception as exc:
            print(f"plot failed: {exc}", file=sys.stderr)
        time.sleep(args.watch)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
