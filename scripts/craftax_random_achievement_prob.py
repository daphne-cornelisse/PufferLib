#!/usr/bin/env python3
"""Estimate Craftax achievement probabilities under a randomly initialized policy.

Rolls out one randomly initialized 1024x4 CraftaxNet for N episodes (default 100),
each capped at max_len steps (default 10000). Probability is count / N_episodes.
Saves two 1x2 figures: unlocked-only P(unlock) and counts, then a separate
1x2 figure with all 67 achievements including zeros.

From the repo root:

    python scripts/craftax_random_achievement_prob.py
    python scripts/craftax_random_achievement_prob.py --episodes 100 --max-len 10000
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "craftax_random_achievement_rollout.c"
BIN = ROOT / "build" / "craftax_random_achievement_rollout"

ACHIEVEMENT_NAMES = [
    "Collect Wood", "Place Table", "Eat Cow", "Collect Sapling", "Collect Drink",
    "Make Wood Pickaxe", "Make Wood Sword", "Place Plant", "Defeat Zombie",
    "Collect Stone", "Place Stone", "Eat Plant", "Defeat Skeleton",
    "Make Stone Pickaxe", "Make Stone Sword", "Wake Up", "Place Furnace",
    "Collect Coal", "Collect Iron", "Collect Diamond", "Make Iron Pickaxe",
    "Make Iron Sword", "Make Arrow", "Make Torch", "Place Torch",
    "Make Diamond Sword", "Make Iron Armour", "Make Diamond Armour",
    "Enter Gnomish Mines", "Enter Dungeon", "Enter Sewers", "Enter Vault",
    "Enter Troll Mines", "Enter Fire Realm", "Enter Ice Realm", "Enter Graveyard",
    "Defeat Gnome Warrior", "Defeat Gnome Archer", "Defeat Orc Soldier",
    "Defeat Orc Mage", "Defeat Lizard", "Defeat Kobold", "Defeat Troll",
    "Defeat Deep Thing", "Defeat Pigman", "Defeat Fire Elemental",
    "Defeat Frost Troll", "Defeat Ice Elemental", "Damage Necromancer",
    "Defeat Necromancer", "Eat Bat", "Eat Snail", "Find Bow", "Fire Bow",
    "Collect Sapphire", "Learn Fireball", "Cast Fireball", "Learn Iceball",
    "Cast Iceball", "Collect Ruby", "Make Diamond Pickaxe", "Open Chest",
    "Drink Potion", "Enchant Sword", "Enchant Armour", "Defeat Knight",
    "Defeat Archer",
]


def compile_rollout(force: bool = False) -> Path:
    BIN.parent.mkdir(parents=True, exist_ok=True)
    if BIN.exists() and not force and BIN.stat().st_mtime >= SRC.stat().st_mtime:
        return BIN

    plat = platform.system()
    if plat == "Darwin":
        ray = ROOT / "raylib-5.5_macos"
        extra = [
            "-framework", "Cocoa",
            "-framework", "IOKit",
            "-framework", "CoreVideo",
            "-framework", "OpenGL",
        ]
    else:
        ray = ROOT / "raylib-5.5_linux_amd64"
        extra = ["-lGL", "-ldl"]

    cmd = [
        "cc",
        "-std=c11",
        "-O2",
        "-DPLATFORM_DESKTOP",
        "-Wno-unused-function",
        "-I", str(ROOT),
        "-I", str(ROOT / "src"),
        "-I", str(ROOT / "vendor"),
        "-I", str(ROOT / "ocean" / "craftax"),
        "-I", str(ray / "include"),
        str(SRC),
        str(ray / "lib" / "libraylib.a"),
        "-lm",
        "-lpthread",
        *extra,
        "-o", str(BIN),
    ]
    print("Compiling", BIN.name, "...", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    return BIN


def run_rollouts(bin_path: Path, episodes: int, max_len: int, hidden: int,
                 layers: int, seed: int):
    cmd = [
        str(bin_path),
        str(episodes),
        str(max_len),
        str(hidden),
        str(layers),
        str(seed),
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
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    if len(rows) != episodes:
        raise RuntimeError(f"expected {episodes} episodes, got {len(rows)}")

    n_ach = len(ACHIEVEMENT_NAMES)
    flags = np.zeros((episodes, n_ach), dtype=np.float64)
    lengths = np.zeros(episodes, dtype=np.int32)
    terminals = np.zeros(episodes, dtype=np.int32)
    for i, row in enumerate(rows):
        lengths[i] = int(row["length"])
        terminals[i] = int(row["terminal"])
        for a in range(n_ach):
            flags[i, a] = float(row[f"ach_{a}"])

    print(
        f"episodes={episodes}  mean_len={lengths.mean():.1f}  "
        f"terminals={int(terminals.sum())}/{episodes}  "
        f"mean_unlocked={flags.sum(axis=1).mean():.2f}",
        flush=True,
    )
    return flags, lengths, terminals


def style_achievement_axis(ax, names, *, rotation=45, ha="right", fontsize=10) -> None:
    ax.set_xlabel("Achievement")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=rotation, ha=ha, va="top", fontsize=fontsize)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.6, len(names) - 0.4)


def draw_prob_count(axes, data, order, episodes, title_prefix, *,
                    label_values, rotation, ha, fontsize, log_scale=False):
    plot_data = data.copy()
    if log_scale:
        # Log axes cannot draw height 0; park never-unlocked bars at the floor.
        plot_data["prob"] = plot_data["prob"].clip(lower=0.5 / float(episodes))
        plot_data["count"] = plot_data["count"].clip(lower=0.5)
    sns.barplot(
        data=plot_data, x="achievement", y="prob", order=order,
        errorbar=None, color="#00bbbb", ax=axes[0],
    )
    axes[0].set_ylabel("P(unlock) = count / episodes")
    axes[0].set_title("Probability")
    if log_scale:
        axes[0].set_yscale("log")
        axes[0].set_ylim(0.5 / float(episodes), 1.2)
    else:
        axes[0].set_ylim(0.0, 1.12)
    style_achievement_axis(axes[0], order, rotation=rotation, ha=ha, fontsize=fontsize)
    if label_values:
        for i, name in enumerate(order):
            p = float(data.loc[data["achievement"] == name, "prob"].iloc[0])
            if p > 0:
                y = p * 1.15 if log_scale else p + 0.02
                axes[0].text(i, y, f"{p:.2f}", ha="center", va="bottom", fontsize=9)

    sns.barplot(
        data=plot_data, x="achievement", y="count", order=order,
        errorbar=None, color="#1b6ca8", ax=axes[1],
    )
    axes[1].set_ylabel("Unlock count")
    axes[1].set_title("Count")
    if log_scale:
        axes[1].set_yscale("log")
        axes[1].set_ylim(0.5, float(episodes) * 1.5)
    else:
        axes[1].set_ylim(0.0, float(episodes) * 1.12)
    style_achievement_axis(axes[1], order, rotation=rotation, ha=ha, fontsize=fontsize)
    if label_values:
        for i, name in enumerate(order):
            c = int(data.loc[data["achievement"] == name, "count"].iloc[0])
            if c > 0:
                y = c * 1.15 if log_scale else c + 1.5
                axes[1].text(i, y, str(c), ha="center", va="bottom", fontsize=9)


def save_prob_count_figure(data, order, episodes, out_path, *, figsize,
                           label_values, rotation, ha, fontsize, bottom,
                           log_scale=False, suptitle=None) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    draw_prob_count(
        axes, data, order, episodes, "",
        label_values=label_values, rotation=rotation, ha=ha, fontsize=fontsize,
        log_scale=log_scale,
    )
    top = 0.88 if suptitle else 0.92
    fig.subplots_adjust(bottom=bottom, top=top, wspace=0.22, left=0.07, right=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_probs(flags: np.ndarray, out_path: Path, out_all_path: Path,
               episodes: int, max_len: int, hidden: int, layers: int) -> None:
    counts = flags.sum(axis=0)
    probs = counts / float(episodes)
    summary = pd.DataFrame({
        "achievement": ACHIEVEMENT_NAMES,
        "count": counts.astype(int),
        "prob": probs,
    })
    unlocked = summary[summary["count"] > 0].copy()
    if unlocked.empty:
        unlocked = summary.copy()
    unlocked_order = unlocked.sort_values(
        ["count", "achievement"], ascending=[False, True]
    )["achievement"].tolist()
    all_order = summary.sort_values(
        ["count", "achievement"], ascending=[False, True]
    )["achievement"].tolist()

    save_prob_count_figure(
        unlocked, unlocked_order, episodes, out_path,
        figsize=(14, 6.5), label_values=True,
        rotation=45, ha="right", fontsize=10, bottom=0.32,
        suptitle="Unlocked",
    )
    save_prob_count_figure(
        summary, all_order, episodes, out_all_path,
        figsize=(18, 7.5), label_values=False,
        rotation=90, ha="center", fontsize=6, bottom=0.36,
        log_scale=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-len", type=int, default=10000)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--force-compile", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "recordings" / "craftax_random_achievement_prob.png",
    )
    p.add_argument(
        "--out-all",
        type=Path,
        default=ROOT / "recordings" / "craftax_random_achievement_prob_all.png",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "recordings" / "craftax_random_achievement_prob.csv",
    )
    p.add_argument(
        "--from-csv",
        action="store_true",
        help="Skip rollouts and plot from an existing --csv",
    )
    args = p.parse_args()

    if args.from_csv:
        rows = list(csv.DictReader(args.csv.open()))
        flags = np.zeros((len(rows), len(ACHIEVEMENT_NAMES)), dtype=np.float64)
        for i, row in enumerate(rows):
            for a, name in enumerate(ACHIEVEMENT_NAMES):
                flags[i, a] = float(row[name])
        args.episodes = flags.shape[0]
        print(f"Loaded {args.episodes} episodes from {args.csv}", flush=True)
    else:
        bin_path = compile_rollout(force=args.force_compile)
        flags, lengths, terminals = run_rollouts(
            bin_path, args.episodes, args.max_len, args.hidden, args.layers, args.seed
        )
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        header = ["episode", "length", "terminal"] + ACHIEVEMENT_NAMES
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(flags.shape[0]):
                w.writerow(
                    [i, int(lengths[i]), int(terminals[i])]
                    + [int(v) for v in flags[i]]
                )
        print(f"Wrote {args.csv}", flush=True)

    counts = flags.sum(axis=0)
    probs = counts / float(flags.shape[0])
    print("\nachievement                         count   P=count/N")
    for i in np.argsort(-counts, kind="stable"):
        print(f"{ACHIEVEMENT_NAMES[i]:<34} {int(counts[i]):5d}  {probs[i]:8.3f}")

    plot_probs(
        flags, args.out, args.out_all,
        args.episodes, args.max_len, args.hidden, args.layers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
