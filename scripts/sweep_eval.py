#!/usr/bin/env python3
"""
Sweep evaluation dashboard.

Loads logs/<env>/sweep_*.ini and writes:
  logs/<env>/sweep_eval.pdf          — curves ranked by env/episode_return
  logs/<env>/sweep_eval_metrics.pdf  — 1×5 seaborn panels (perf, food, eod, collisions, bites)
  logs/<env>/best_policy             — run_id of the top-return policy (for next eval)

  python scripts/sweep_eval.py wef
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# Perceptually uniform sequential colormap (wall-clock: fast → slow).
HEATMAP_CMAP = plt.get_cmap("viridis")

SWEEP_RE = re.compile(r"^sweep_(\d+)_(\d+)\.ini$")
# Newer logs: bare run_id.ini (e.g. 1786478287071.ini)
RUN_ID_RE = re.compile(r"^(\d+)\.ini$")
CKPT_RE = re.compile(r"^(\d+)\.bin$")
DEFAULT_METRIC = "env/episode_return"

# Seaborn 1×N panels: (metric key, y-label / title)
# env/perf = fraction of food pellets cleared (food_eaten / num_food), not return/max.
SEABORN_METRICS: list[tuple[str, str]] = [
    ("env/perf", "perf"),
    ("env/food_eaten_mean", "food_eaten_mean"),
    ("env/eod_rate", "eod_rate"),
    ("env/collisions_fish", "collisions_fish"),
    ("env/bites", "bites"),
]
BEST_GREEN = "#0a7a32"
RUN_GRAY = "#999999"


@dataclass
class SweepRun:
    idx: int
    ts: int
    run_id: str
    path: Path
    final_return: float = float("nan")
    max_return: float = float("nan")
    final_score: float = float("nan")
    final_steps: float = float("nan")
    final_uptime: float = float("nan")
    steps: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    uptime: list[float] = field(default_factory=list)
    # Optional env metric series (same length as uptime when present)
    perf: list[float] = field(default_factory=list)
    food_eaten_mean: list[float] = field(default_factory=list)
    eod_rate: list[float] = field(default_factory=list)
    collisions_fish: list[float] = field(default_factory=list)
    bites: list[float] = field(default_factory=list)
    hidden_size: int = 128
    num_layers: int = 3


def parse_float_series(csv: str) -> list[float]:
    out: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            break
    return out


def parse_ini_metrics(text: str) -> dict[str, list[float]]:
    if "[metrics]" not in text:
        return {}
    body = text.split("[metrics]", 1)[1]
    body = re.split(r"\n\[", body, maxsplit=1)[0]
    metrics: dict[str, list[float]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        metrics[key.strip()] = parse_float_series(val)
    return metrics


def parse_ini_int(text: str, key: str, default: int) -> int:
    m = re.search(rf"^{re.escape(key)}\s*=\s*([^\n#]+)", text, re.M)
    if not m:
        return default
    try:
        return int(float(m.group(1).strip()))
    except ValueError:
        return default


def parse_ini_str(text: str, key: str, default: str = "") -> str:
    m = re.search(rf"^{re.escape(key)}\s*=\s*([^\n#]+)", text, re.M)
    if not m:
        return default
    return m.group(1).strip()


def load_runs(log_dir: Path, metric_key: str = DEFAULT_METRIC) -> list[SweepRun]:
    runs: list[SweepRun] = []
    bare_idx = 0
    for path in sorted(log_dir.glob("*.ini")):
        if path.name in ("best_policy.ini",):
            continue
        m = SWEEP_RE.match(path.name)
        if m:
            ts, idx = int(m.group(1)), int(m.group(2))
        else:
            m2 = RUN_ID_RE.match(path.name)
            if not m2:
                continue
            ts = int(m2.group(1))
            idx = bare_idx
            bare_idx += 1
        text = path.read_text(errors="replace")
        metrics = parse_ini_metrics(text)
        returns = metrics.get(metric_key, [])
        if not returns:
            continue
        steps = metrics.get("agent_steps", [])
        uptime = metrics.get("uptime", [])
        scores = metrics.get("env/score", [])
        n = min(len(returns), len(steps)) if steps else 0
        n_u = min(len(uptime), len(returns)) if uptime else 0

        def series(key: str) -> list[float]:
            vals = metrics.get(key, [])
            if not vals or not n_u:
                return []
            return vals[: min(len(vals), n_u)]

        run = SweepRun(
            idx=idx,
            ts=ts,
            run_id=parse_ini_str(text, "run_id", path.stem),
            path=path,
            final_return=returns[-1],
            max_return=max(returns),
            final_score=scores[-1] if scores else float("nan"),
            final_steps=steps[-1] if steps else float("nan"),
            final_uptime=uptime[-1] if uptime else float("nan"),
            steps=steps[:n],
            returns=returns[:n],
            uptime=uptime[:n_u],
            perf=series("env/perf"),
            food_eaten_mean=series("env/food_eaten_mean"),
            eod_rate=series("env/eod_rate"),
            collisions_fish=series("env/collisions_fish"),
            bites=series("env/bites"),
            hidden_size=parse_ini_int(text, "hidden_size", 128),
            num_layers=parse_ini_int(text, "num_layers", 3),
        )
        runs.append(run)
    runs.sort(key=lambda r: (r.ts, r.idx))
    return runs


def format_steps(steps: float) -> str:
    if steps >= 1e9:
        return f"{steps / 1e9:.2f}B"
    if steps >= 1e6:
        return f"{steps / 1e6:.1f}M"
    if steps >= 1e3:
        return f"{steps / 1e3:.1f}K"
    return f"{steps:.0f}"


def format_time(sec: float) -> str:
    if sec >= 3600:
        return f"{sec / 3600:.1f}h"
    if sec >= 60:
        return f"{sec / 60:.1f}m"
    return f"{sec:.0f}s"


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Latest step checkpoint: checkpoints/<env>/<run_id>/<step>.bin"""
    if not ckpt_dir.is_dir():
        return None
    best_step = -1
    best_path: Path | None = None
    for path in ckpt_dir.glob("*.bin"):
        m = CKPT_RE.match(path.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def resolve_policy(
    env: str,
    run: SweepRun,
    checkpoint_dir: Path,
) -> tuple[str, Path | None]:
    """Return (policy_name / run_id, latest checkpoint path or None)."""
    ckpt = find_latest_checkpoint(checkpoint_dir / env / run.run_id)
    return run.run_id, ckpt


def write_best_policy(
    out_path: Path,
    policy_name: str,
    ckpt: Path | None,
    run: SweepRun,
) -> None:
    """Write policy name (run_id) for scripting; also a sidecar with details."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(policy_name + "\n", encoding="utf-8")
    detail = out_path.with_suffix(".ini")
    lines = [
        f"policy_name = {policy_name}",
        f"run_id = {run.run_id}",
        f"idx = {run.idx}",
        f"episode_return = {run.final_return}",
        f"hidden_size = {run.hidden_size}",
        f"num_layers = {run.num_layers}",
    ]
    if ckpt is not None:
        lines.append(f"load_model_path = {ckpt.as_posix()}")
    else:
        lines.append("load_model_path =")
    detail.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_sweep_eval(
    env: str,
    runs: list[SweepRun],
    out_path: Path,
    metric_key: str = DEFAULT_METRIC,
    dpi: int = 288,
) -> None:
    if not runs:
        raise SystemExit(f"no sweep runs with metrics in logs/{env}")

    best = max(runs, key=lambda r: r.final_return)
    total_wall = sum(r.final_uptime for r in runs if r.final_uptime == r.final_uptime)

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.4),
        dpi=dpi / 3,  # base fig; savefig uses dpi=
        constrained_layout=True,
    )
    # Reserve bottom strip for best-run footer (axes sit above it)
    fig.set_constrained_layout_pads(
        h_pad=0.04, w_pad=0.04, hspace=0.04, wspace=0.04, rect=(0, 0.10, 1, 1)
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_l, ax_r):
        ax.set_facecolor("white")
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    # ----- Left: training curves (episode return) -----
    steps_max = 1.0
    ret_min, ret_max = float("inf"), float("-inf")
    for r in runs:
        if r.final_steps == r.final_steps:
            steps_max = max(steps_max, r.final_steps)
        for s, ret in zip(r.steps, r.returns):
            steps_max = max(steps_max, s)
            ret_min = min(ret_min, ret)
            ret_max = max(ret_max, ret)
        if r.final_return == r.final_return:
            ret_min = min(ret_min, r.final_return)
            ret_max = max(ret_max, r.final_return)
    if ret_max <= ret_min:
        ret_max = ret_min + 1.0
    pad = 0.08 * (ret_max - ret_min)

    for r in runs:
        if len(r.steps) >= 2:
            ax_l.plot(
                r.steps,
                r.returns,
                color="#999999",
                alpha=0.35,
                linewidth=0.9,
                zorder=2,
            )

    # Highlight top return in green (no "best" labels — visual only)
    if len(best.steps) >= 2:
        ax_l.plot(
            best.steps,
            best.returns,
            color=BEST_GREEN,
            linewidth=2.6,
            zorder=4,
        )
        ax_l.scatter(
            best.steps,
            best.returns,
            s=28,
            color=BEST_GREEN,
            zorder=5,
            edgecolors="none",
        )

    finals_x, finals_y = [], []
    for r in runs:
        if r.final_steps == r.final_steps and r.final_return == r.final_return:
            if r.idx == best.idx:
                continue
            finals_x.append(r.final_steps)
            finals_y.append(r.final_return)
    ax_l.scatter(
        finals_x,
        finals_y,
        s=18,
        color="#00bbbb",
        zorder=3,
        edgecolors="none",
        label="final",
    )
    ax_l.scatter(
        [best.final_steps],
        [best.final_return],
        s=36,
        color=BEST_GREEN,
        zorder=6,
        edgecolors="none",
    )

    ax_l.set_xlim(0, steps_max * 1.02)
    ax_l.set_ylim(ret_min - pad, ret_max + pad)
    ax_l.set_xlabel("agent steps", fontsize=12, color="#333333")
    ax_l.set_ylabel(metric_key, fontsize=12, color="#333333")
    ax_l.set_title(
        "Training curves  (env/episode_return vs agent steps)",
        fontsize=13,
        pad=10,
    )

    ax_l.text(
        0.02,
        0.98,
        f"sweep wall-clock sum {format_time(total_wall)}",
        transform=ax_l.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#c06010",
    )

    legend_elems = [
        Line2D([0], [0], color="#999999", lw=1.2, alpha=0.6, label="all runs"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00bbbb",
            markersize=7,
            label="final",
        ),
    ]
    ax_l.legend(handles=legend_elems, loc="lower right", fontsize=9, framealpha=0.95)

    # Nice step tick labels
    xticks = np.linspace(0, steps_max * 1.02, 6)
    ax_l.set_xticks(xticks)
    ax_l.set_xticklabels([format_steps(v) for v in xticks], fontsize=10)

    # ----- Right: final episode return vs sweep idx -----
    idxs = np.array([r.idx for r in runs], dtype=float)
    rets = np.array([r.final_return for r in runs], dtype=float)
    ups = np.array([r.final_uptime for r in runs], dtype=float)
    valid = np.isfinite(rets)
    idxs, rets, ups = idxs[valid], rets[valid], ups[valid]

    up_finite = ups[np.isfinite(ups)]
    if len(up_finite) > 0:
        up_lo, up_hi = np.percentile(up_finite, [10, 90])
        if up_hi <= up_lo:
            up_lo, up_hi = float(up_finite.min()), float(up_finite.max())
        if up_hi <= up_lo:
            up_hi = up_lo + 1.0
    else:
        up_lo, up_hi = 0.0, 1.0

    norm = Normalize(vmin=up_lo, vmax=up_hi, clip=True)
    sc = ax_r.scatter(
        idxs,
        rets,
        c=np.nan_to_num(ups, nan=(up_lo + up_hi) * 0.5),
        cmap=HEATMAP_CMAP,
        norm=norm,
        s=36,
        edgecolors="#222222",
        linewidths=0.4,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax_r, fraction=0.046, pad=0.04)
    cbar.set_label("wall-clock time", fontsize=10)
    cbar.set_ticks([up_lo, 0.5 * (up_lo + up_hi), up_hi])
    cbar.set_ticklabels(
        [
            f"fast\n{format_time(up_lo)}",
            format_time(0.5 * (up_lo + up_hi)),
            f"slow\n{format_time(up_hi)}",
        ]
    )
    cbar.ax.tick_params(labelsize=9)

    ax_r.set_xlabel("sweep run index", fontsize=12, color="#333333")
    ax_r.set_ylabel("env/episode_return", fontsize=12, color="#333333")
    ax_r.set_title("Final episode return", fontsize=13, pad=10)

    # Footer: top-return policy (reserved bottom margin)
    score_str = (
        f"score={best.final_score:.3f}   "
        if best.final_score == best.final_score
        else ""
    )
    fig.text(
        0.01,
        0.035,
        f"policy={best.run_id}   #{best.idx:04d}   return={best.final_return:.3f}   "
        f"{score_str}"
        f"steps={format_steps(best.final_steps)}   "
        f"wall={format_time(best.final_uptime)}",
        fontsize=10,
        color="#222222",
        ha="left",
        va="center",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _run_metric_series(run: SweepRun, key: str) -> list[float]:
    return {
        "env/perf": run.perf,
        "env/food_eaten_mean": run.food_eaten_mean,
        "env/eod_rate": run.eod_rate,
        "env/collisions_fish": run.collisions_fish,
        "env/bites": run.bites,
    }.get(key, [])


def plot_sweep_metrics_seaborn(
    env: str,
    runs: list[SweepRun],
    out_path: Path,
    dpi: int = 288,
) -> None:
    """1×N seaborn figure: perf | food_eaten_mean | eod_rate | collisions_fish | bites.

    X-axis is wall-clock time (uptime). Best run (by env/episode_return) in green.
    """
    if not runs:
        raise SystemExit(f"no sweep runs with metrics in logs/{env}")

    best = max(runs, key=lambda r: r.final_return)

    # Long-form dataframe for seaborn line plots
    rows: list[dict] = []
    finals: list[dict] = []
    for r in runs:
        is_best = r.run_id == best.run_id and r.idx == best.idx
        for key, label in SEABORN_METRICS:
            ys = _run_metric_series(r, key)
            xs = r.uptime
            n = min(len(xs), len(ys))
            if n < 1:
                continue
            for t, y in zip(xs[:n], ys[:n]):
                rows.append(
                    {
                        "uptime": t,
                        "value": y,
                        "metric": label,
                        "run_id": r.run_id,
                        "is_best": is_best,
                    }
                )
            finals.append(
                {
                    "uptime": xs[n - 1],
                    "value": ys[n - 1],
                    "metric": label,
                    "is_best": is_best,
                }
            )

    if not rows:
        print(
            f"No seaborn metrics series found (need uptime + "
            f"{', '.join(k for k, _ in SEABORN_METRICS)}); skipping {out_path}",
            file=sys.stderr,
        )
        return

    df = pd.DataFrame(rows)
    df_fin = pd.DataFrame(finals)
    metric_order = [label for _, label in SEABORN_METRICS]

    # Proportional type scale for a wide 1×4 panel (not talk/poster sizes)
    title_fs = 12
    label_fs = 10
    tick_fs = 9
    legend_fs = 9

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "axes.titlesize": title_fs,
            "axes.labelsize": label_fs,
            "xtick.labelsize": tick_fs,
            "ytick.labelsize": tick_fs,
            "legend.fontsize": legend_fs,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
        },
    )
    n_panels = len(SEABORN_METRICS)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.9 * n_panels, 3.8),
        dpi=dpi / 3,
        sharex=False,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]
    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.03, wspace=0.08)
    fig.patch.set_facecolor("white")

    for ax, label in zip(axes, metric_order):
        ax.set_facecolor("white")
        sub = df[df["metric"] == label]
        fin = df_fin[df_fin["metric"] == label]
        if sub.empty:
            ax.set_title(label, fontsize=title_fs, pad=6)
            continue

        # All runs (thin gray) — plain plot is much faster than sns units= for large sweeps
        for rid, g in sub[~sub["is_best"]].groupby("run_id", sort=False):
            ax.plot(
                g["uptime"].to_numpy(),
                g["value"].to_numpy(),
                color=RUN_GRAY,
                alpha=0.30,
                linewidth=0.85,
                zorder=2,
            )
        # Best run (green)
        best_df = sub[sub["is_best"]]
        if not best_df.empty:
            ax.plot(
                best_df["uptime"].to_numpy(),
                best_df["value"].to_numpy(),
                color=BEST_GREEN,
                linewidth=2.2,
                zorder=4,
            )
            ax.scatter(
                best_df["uptime"],
                best_df["value"],
                s=22,
                color=BEST_GREEN,
                zorder=5,
                edgecolors="none",
            )

        # Final markers
        fin_o = fin[~fin["is_best"]]
        fin_b = fin[fin["is_best"]]
        if not fin_o.empty:
            ax.scatter(
                fin_o["uptime"],
                fin_o["value"],
                s=14,
                color="#00bbbb",
                zorder=3,
                edgecolors="none",
                alpha=0.85,
            )
        if not fin_b.empty:
            ax.scatter(
                fin_b["uptime"],
                fin_b["value"],
                s=32,
                color=BEST_GREEN,
                zorder=6,
                edgecolors="none",
            )
            # Annotate last recorded best-run value
            last_x = float(fin_b["uptime"].iloc[-1])
            last_y = float(fin_b["value"].iloc[-1])
            ax.annotate(
                f"{last_y:.3g}",
                xy=(last_x, last_y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=tick_fs + 1,
                color=BEST_GREEN,
                zorder=7,
                clip_on=False,
            )

        ax.set_title(label, fontsize=title_fs, pad=6, color="#222222")
        ax.set_xlabel("wall-clock time", fontsize=label_fs, color="#444444")
        ax.set_ylabel(label, fontsize=label_fs, color="#444444")
        ax.tick_params(axis="both", labelsize=tick_fs, colors="#444444")
        ax.grid(True, color="#cccccc", linewidth=0.7, alpha=0.22, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        # Friendly time ticks
        xmax = float(sub["uptime"].max()) if len(sub) else 1.0
        xticks = np.linspace(0, xmax * 1.02, 4)
        ax.set_xlim(0, xmax * 1.02)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_time(v) for v in xticks], fontsize=tick_fs)

    legend_elems = [
        Line2D([0], [0], color=RUN_GRAY, lw=1.2, alpha=0.6, label="all runs"),
        Line2D([0], [0], color=BEST_GREEN, lw=2.2, label="best return"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00bbbb",
            markersize=6,
            label="final",
        ),
    ]
    axes[-1].legend(
        handles=legend_elems,
        loc="upper right",
        fontsize=legend_fs,
        framealpha=0.9,
        edgecolor="#dddddd",
        borderpad=0.4,
        handlelength=1.6,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    sns.reset_orig()
    print(f"Saved plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep eval plot: training curves + finals for env/episode_return"
    )
    ap.add_argument("env", help="environment name (reads logs/<env>/sweep_*.ini)")
    ap.add_argument(
        "--log-dir",
        default=None,
        help="override log dir (default logs/<env>)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output pdf (default logs/<env>/sweep_eval.pdf)",
    )
    ap.add_argument(
        "--metrics-out",
        default=None,
        help="seaborn metrics pdf (default logs/<env>/sweep_eval_metrics.pdf)",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="root checkpoint dir (default checkpoints)",
    )
    ap.add_argument(
        "--policy-out",
        default=None,
        help="write best policy name here (default logs/<env>/best_policy)",
    )
    ap.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"training-curve metric key (default {DEFAULT_METRIC})",
    )
    ap.add_argument("--dpi", type=int, default=288, help="export DPI (default 288)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else Path("logs") / args.env
    if not log_dir.is_dir():
        print(f"log dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)
    out = Path(args.out) if args.out else Path("logs") / args.env / "sweep_eval.pdf"
    metrics_out = (
        Path(args.metrics_out)
        if args.metrics_out
        else Path("logs") / args.env / "sweep_eval_metrics.pdf"
    )
    policy_out = (
        Path(args.policy_out)
        if args.policy_out
        else Path("logs") / args.env / "best_policy"
    )
    ckpt_root = Path(args.checkpoint_dir)
    metric = args.metric if "/" in args.metric else f"env/{args.metric}"

    runs = load_runs(log_dir, metric_key=metric)
    print(f"Loaded {len(runs)} sweep runs from {log_dir}")
    if not runs:
        print("No runs with episode_return metrics.", file=sys.stderr)
        sys.exit(1)

    best = max(runs, key=lambda r: r.final_return)
    policy_name, ckpt = resolve_policy(args.env, best, ckpt_root)
    write_best_policy(policy_out, policy_name, ckpt, best)

    print(f"policy={policy_name}")
    print(f"  return={best.final_return:.4f}  #{best.idx:04d}")
    print(f"  hidden_size={best.hidden_size}  num_layers={best.num_layers}")
    if ckpt is not None:
        print(f"  checkpoint={ckpt.as_posix()}")
        print(
            f"  load: base.load_model_path={ckpt.as_posix()}  "
            f"policy.hidden_size={best.hidden_size}  "
            f"policy.num_layers={best.num_layers}"
        )
    else:
        print(
            f"  checkpoint=(missing under {ckpt_root / args.env / policy_name})",
            file=sys.stderr,
        )
    print(f"  wrote {policy_out}  (+ {policy_out.with_suffix('.ini')})")

    plot_sweep_eval(args.env, runs, out, metric_key=metric, dpi=args.dpi)
    plot_sweep_metrics_seaborn(args.env, runs, metrics_out, dpi=args.dpi)


if __name__ == "__main__":
    main()

