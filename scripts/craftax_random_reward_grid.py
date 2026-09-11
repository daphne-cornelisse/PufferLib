#!/usr/bin/env python3
"""Per-step Craftax reward calendar: columns = rollouts, rows = time.

Saves a CSV first (reward + achievement ids), then draws the figure from it.

    python scripts/craftax_random_reward_grid.py
    python scripts/craftax_random_reward_grid.py --from-csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from craftax_random_achievement_prob import ACHIEVEMENT_NAMES, ROOT, compile_rollout

DEATH_LABEL = "Death"
ARMOUR_LABEL = "Armour"
ACH_REWARD = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 5, 5,
    5, 8, 8, 8, 3, 3, 3, 3, 5, 5, 5, 5, 8, 8, 8, 8,
    8, 8, 3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 5,
    5, 5, 5,
]
REWARD_BY_NAME = dict(zip(ACHIEVEMENT_NAMES, ACH_REWARD))


def generate_csv(bin_path: Path, csv_path: Path, episodes: int, max_len: int,
                 hidden: int, layers: int, seed: int) -> None:
    cmd = [
        str(bin_path), str(episodes), str(max_len),
        str(hidden), str(layers), str(seed), "rewards",
    ]
    print("Running", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=None, text=True,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(proc.stdout)
    print(f"Wrote {csv_path}", flush=True)


def event_label(row) -> str | None:
    raw = str(row.get("achievement_ids") or "").strip()
    if raw:
        ids = [int(x) for x in raw.split(";") if x != ""]
        if ids:
            best = max(ids, key=lambda i: ACH_REWARD[i])
            return ACHIEVEMENT_NAMES[best]
    if int(row["armour_delta"]):
        return ARMOUR_LABEL
    return None


def load_steps(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, keep_default_na=False)
    df["event"] = df.apply(event_label, axis=1)
    return df


def event_colors(df: pd.DataFrame) -> tuple[dict, list[str]]:
    events = [e for e in df["event"].dropna().unique().tolist()]
    ach_events = [e for e in ACHIEVEMENT_NAMES if e in events]
    extra = [e for e in (ARMOUR_LABEL,) if e in events]
    ordered = ach_events + extra
    palette = sns.color_palette("husl", max(len(ach_events), 1))
    colors = {name: palette[i] for i, name in enumerate(ach_events)}
    colors[ARMOUR_LABEL] = "#7f8c8d"
    return colors, ordered


def plot_grid(df: pd.DataFrame, out_path: Path, colors: dict, ordered: list[str]) -> None:
    n_rollouts = int(df["episode"].max()) + 1
    max_t = int(df.groupby("episode")["step"].max().max()) + 1

    fill = np.ones((max_t, n_rollouts, 3), dtype=np.float32)
    have = np.zeros((max_t, n_rollouts), dtype=bool)
    for rec in df.itertuples(index=False):
        step, ep = int(rec.step), int(rec.episode)
        have[step, ep] = True
        if rec.event:
            fill[step, ep] = colors[rec.event]

    # Rasterize outlined squares with a Wait-But-Why-style gap between them.
    # 1px outline on a larger tile keeps the bounding box thin.
    sq, gap, border = 16, 5, 1
    stride = sq + gap
    h = max_t * stride - gap
    w = n_rollouts * stride - gap
    img = np.ones((h, w, 3), dtype=np.float32)
    outline = np.array([0.72, 0.72, 0.72], dtype=np.float32)
    steps_i, eps_i = np.nonzero(have)
    for step, ep in zip(steps_i.tolist(), eps_i.tolist()):
        y0 = step * stride
        x0 = ep * stride
        img[y0:y0 + sq, x0:x0 + sq] = outline
        img[y0 + border:y0 + sq - border, x0 + border:x0 + sq - border] = fill[step, ep]

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(
        img,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, n_rollouts, max_t, 0),
    )
    ax.set_xlim(0, n_rollouts)
    ax.set_ylim(max_t, 0)
    ax.set_xlabel("Rollout", fontsize=20)
    ax.set_ylabel("Time (step)", fontsize=20)
    ax.set_title(
        "A random RL agents experiences per step, colored by achievement",
        fontsize=20,
        pad=22,
    )
    ax.tick_params(axis="both", labelsize=17, length=4, width=1.2)
    xtick_step = max(1, n_rollouts // 10)
    ax.set_xticks(np.arange(0, n_rollouts, xtick_step) + 0.5)
    ax.set_xticklabels([str(i) for i in range(0, n_rollouts, xtick_step)])
    yticks = list(range(0, max_t, max(1, max_t // 12)))
    ax.set_yticks([t + 0.5 for t in yticks])
    ax.set_yticklabels([str(t) for t in yticks])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if ordered:
        handles = [
            Patch(facecolor=colors[name], edgecolor="black", label=name)
            for name in ordered
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=8,
            title="Achievement",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def unlock_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in df.itertuples(index=False):
        raw = str(getattr(rec, "achievement_ids", "") or "").strip()
        if not raw:
            continue
        for token in raw.split(";"):
            if token == "":
                continue
            aid = int(token)
            rows.append({
                "episode": int(rec.episode),
                "step": int(rec.step),
                "achievement": ACHIEVEMENT_NAMES[aid],
            })
    if not rows:
        return pd.DataFrame(columns=["episode", "step", "achievement"])
    out = pd.DataFrame(rows)
    return out.groupby(["episode", "achievement"], as_index=False)["step"].min()


def plot_unlock_lines(df: pd.DataFrame, out_path: Path, n_rollouts: int | None = None) -> None:
    if n_rollouts is None:
        n_rollouts = int(df["episode"].nunique())
    sub = df[df["episode"] < n_rollouts]
    unlocks = unlock_table(sub)
    mean_steps = unlocks.groupby("achievement")["step"].mean().sort_values()
    present = [a for a in mean_steps.index.tolist()]
    xmap = {name: i for i, name in enumerate(present)}
    n_lines = int(unlocks["episode"].nunique())
    line_alpha = max(0.06, min(0.28, 18.0 / n_lines))

    fig, ax = plt.subplots(figsize=(14, 7.5))
    for _, g in unlocks.groupby("episode"):
        g = g.copy()
        g["x"] = g["achievement"].map(xmap)
        g = g.dropna(subset=["x"]).sort_values("x")
        ax.plot(
            g["x"], g["step"],
            color="#1b6ca8", alpha=line_alpha, linewidth=1.0, zorder=2,
        )
        ax.scatter(
            g["x"], g["step"],
            color="#1b6ca8", s=12, alpha=min(0.4, line_alpha + 0.1),
            zorder=3, linewidths=0,
        )
    mean = (
        unlocks.groupby("achievement", as_index=False)["step"].mean()
        .assign(x=lambda d: d["achievement"].map(xmap))
        .dropna(subset=["x"])
        .sort_values("x")
    )
    ax.plot(
        mean["x"], mean["step"],
        color="#111111", linewidth=2.4, marker="D", markersize=7,
        zorder=4, label="Mean",
    )
    ax.plot([], [], color="#1b6ca8", alpha=0.5, linewidth=1.2, label="Rollout")
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("Achievement", fontsize=16)
    ax.set_ylabel("Steps to complete", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(
        f"Steps to complete each achievement ({n_rollouts} rollouts)",
        fontsize=18,
        pad=16,
    )
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_unlock_by_importance(df: pd.DataFrame, out_path: Path,
                              n_rollouts: int | None = None) -> None:
    if n_rollouts is None:
        n_rollouts = int(df["episode"].nunique())
    unlocks = unlock_table(df[df["episode"] < n_rollouts])
    unlocks["importance"] = unlocks["achievement"].map(REWARD_BY_NAME)
    unlocks = unlocks.dropna(subset=["importance"])
    rng = np.random.default_rng(0)
    classes = [1, 3, 5, 8]
    n_lines = int(unlocks["episode"].nunique())
    line_alpha = max(0.05, min(0.25, 16.0 / n_lines))

    fig, ax = plt.subplots(figsize=(10, 7.5))
    per_class = (
        unlocks.groupby(["episode", "importance"], as_index=False)["step"].min()
        .sort_values("importance")
    )
    for _, g in per_class.groupby("episode"):
        ax.plot(
            g["importance"], g["step"],
            color="#1b6ca8", alpha=line_alpha, linewidth=1.0, zorder=2,
        )
    jitter = rng.uniform(-0.12, 0.12, size=len(unlocks))
    ax.scatter(
        unlocks["importance"] + jitter, unlocks["step"],
        color="#1b6ca8", s=14, alpha=min(0.35, line_alpha + 0.08),
        zorder=3, linewidths=0,
    )
    mean = unlocks.groupby("importance", as_index=False)["step"].mean().sort_values("importance")
    ax.plot(
        mean["importance"], mean["step"],
        color="#111111", linewidth=2.4, marker="D", markersize=8,
        zorder=4, label="Mean",
    )
    ax.plot([], [], color="#1b6ca8", alpha=0.5, linewidth=1.2, label="Rollout")
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes], fontsize=14)
    ax.set_xlim(0.4, 8.6)
    ax.set_xlabel("Importance (achievement reward)", fontsize=16)
    ax.set_ylabel("Steps to complete", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(
        f"Steps to complete vs achievement importance ({n_rollouts} rollouts)",
        fontsize=18,
        pad=16,
    )
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument(
        "--plot-ns",
        type=int,
        nargs="+",
        default=[100, 200, 500],
        help="Rollout counts to plot from the saved CSV",
    )
    p.add_argument("--max-len", type=int, default=10000)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--force-compile", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "recordings" / "craftax_random_reward_grid.png",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "recordings" / "craftax_random_reward_steps.csv",
    )
    p.add_argument(
        "--from-csv",
        action="store_true",
        help="Skip rollouts and draw from an existing --csv",
    )
    args = p.parse_args()

    need = max([args.episodes, *args.plot_ns])
    if not args.from_csv:
        bin_path = compile_rollout(force=args.force_compile)
        generate_csv(
            bin_path, args.csv, need, args.max_len,
            args.hidden, args.layers, args.seed,
        )

    df = load_steps(args.csv)
    n_ep = int(df["episode"].nunique())
    n_event = int(df["event"].notna().sum())
    print(
        f"Loaded {n_ep} rollouts, {len(df)} steps, {n_event} nonzero events "
        f"from {args.csv}",
        flush=True,
    )
    colors, ordered = event_colors(df)
    for n in args.plot_ns:
        if n > n_ep:
            print(f"skip {n}: CSV only has {n_ep} rollouts", flush=True)
            continue
        sub = df[df["episode"] < n].copy()
        out = args.out.with_name(f"{args.out.stem}_{n}{args.out.suffix}")
        print(f"Plotting {n} rollouts -> {out}", flush=True)
        plot_grid(sub, out, colors, ordered)
    lines_out = args.out.with_name("craftax_random_unlock_steps.png")
    plot_unlock_lines(df, lines_out)
    imp_out = args.out.with_name("craftax_random_unlock_importance.png")
    plot_unlock_by_importance(df, imp_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
