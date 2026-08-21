#!/usr/bin/env python3
"""1×3 WEF figure: rollout SPS, training SPS, episode return vs wall-clock."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = ROOT / "logs" / "wef" / "sps_return.pdf"
OUT_PNG = ROOT / "logs" / "wef" / "sps_return.png"

UNCON_INI = ROOT / "logs" / "wef_unconstrained" / "sweep_1787181581088_0063.ini"
CON_INI = ROOT / "logs" / "wef_constrained_5M" / "sweep_1787232678353_0286.ini"

# Native measured SPS. No ×num_agents conversion.
# Original rollout: MAEFish without_render FPS, 16 processes, sum of per-process rates.
ORIGINAL_ROLLOUT = [
    3832.195453441468,
    3920.8436562913503,
    3943.6850960967126,
    3972.83499707202,
    4058.523800265363,
    4140.718391624295,
    4264.211740509225,
    4617.682040216008,
    4879.75377575737,
    5294.712713092618,
]
# Puffer rollout: C bench agent-steps/s (puf_step × 4 agents/env, 8192 agents, 16 threads).
PUFFER_ROLLOUT = [
    1850422.269956,
    1851743.336908,
    1858505.361168,
    1855217.416848,
    1853372.61662,
    1849857.583996,
    1848849.658992,
    1848213.572088,
    1851483.329364,
    1848493.48172,
]
# MAPPO train_fish.py --n_rollout_threads=16, GPU, training loop only.
# FPS as printed: vec-env timesteps / elapsed.
ORIGINAL_TRAIN = [
    720, 608, 670, 714, 751, 777, 733, 762, 776, 764, 745, 760, 770, 782, 764,
]
# Puffer dashboard SPS as printed (delta global_step / dt). Leftover last tick dropped.
# Filled in main() from logs/wef/bench_sps.json
PUFFER_TRAIN: list[float] = []
PUFFER_TRAIN_PEAK = 0.0

REF_IMPL = "Reference Python"
PUF_IMPL = "Puffer"
SPS_PALETTE = {REF_IMPL: "#1f77b4", PUF_IMPL: "#0a7a32"}
IMPL_ORDER = [REF_IMPL, PUF_IMPL]
CURVE_PALETTE = {
    "Unconstrained": "#0a7a32",
    "Constrained (5M)": "#7b2d8e",
}


def fmt_sps(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e5:
        return f"{x / 1e3:.0f}K"
    if x >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{x:.0f}"


def parse_float_series(csv: str) -> list[float]:
    out: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def load_curve(path: Path, label: str) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    body = text.split("[metrics]", 1)[1]
    metrics: dict[str, list[float]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("["):
            break
        key, val = line.split("=", 1)
        metrics[key.strip()] = parse_float_series(val)
    uptime = metrics["uptime"]
    ret = metrics["env/episode_return"]
    ns = metrics.get("env/n", [])
    n = min(len(uptime), len(ret))
    # Final downsample bin is often a 10k-episode eval, not the train curve.
    if ns and n > 1 and ns[n - 1] >= 1000:
        n -= 1
    return pd.DataFrame(
        {
            "uptime": uptime[:n],
            "episode_return": ret[:n],
            "run": label,
        }
    )


def load_puffer_dashboard_sps() -> tuple[list[float], float]:
    import json
    path = ROOT / "logs" / "wef" / "bench_sps.json"
    rows = json.loads(path.read_text())["rows"]
    train = next(r for r in rows if "train" in r["name"])
    dash = list(train["dashboard_sps"])
    return dash, max(dash) if dash else 0.0


def sps_frame() -> pd.DataFrame:
    rows = []
    for v in ORIGINAL_ROLLOUT:
        rows.append({"panel": "Rollout", "impl": REF_IMPL, "sps": v})
    for v in PUFFER_ROLLOUT:
        rows.append({"panel": "Rollout", "impl": PUF_IMPL, "sps": v})
    for v in ORIGINAL_TRAIN:
        rows.append({"panel": "Training", "impl": REF_IMPL, "sps": v})
    for v in PUFFER_TRAIN:
        rows.append({"panel": "Training", "impl": PUF_IMPL, "sps": v})
    return pd.DataFrame(rows)


def annotate_bars(ax, sub: pd.DataFrame, panel: str) -> None:
    means = sub.groupby("impl")["sps"].mean().reindex(IMPL_ORDER)
    for patch, impl in zip(ax.patches, IMPL_ORDER):
        if impl not in means.index or pd.isna(means[impl]):
            continue
        val = float(means[impl])
        label = fmt_sps(val)
        if panel == "Training" and impl == PUF_IMPL:
            label = f"{fmt_sps(val)}\npeak {fmt_sps(PUFFER_TRAIN_PEAK)}"
        # Short bars need more gap above the error bar; tall ones sit near the title.
        y_off = 8 if val > 1e5 else 16
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
            xytext=(0, y_off),
            textcoords="offset points",
        )


def main() -> None:
    global PUFFER_TRAIN, PUFFER_TRAIN_PEAK
    PUFFER_TRAIN, PUFFER_TRAIN_PEAK = load_puffer_dashboard_sps()
    sps = sps_frame()
    curves = pd.concat(
        [
            load_curve(UNCON_INI, "Unconstrained"),
            load_curve(CON_INI, "Constrained (5M)"),
        ],
        ignore_index=True,
    )

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        },
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), dpi=160)
    fig.patch.set_facecolor("white")

    for ax, panel in zip(axes[:2], ("Rollout", "Training")):
        sub = sps[sps["panel"] == panel]
        sns.barplot(
            data=sub,
            x="impl",
            y="sps",
            hue="impl",
            hue_order=IMPL_ORDER,
            order=IMPL_ORDER,
            palette=SPS_PALETTE,
            errorbar="sd",
            capsize=0.12,
            err_kws={"linewidth": 1.2, "color": "0.2"},
            width=0.62,
            ax=ax,
            legend=False,
        )
        ax.set_yscale("log")
        ax.set_title(f"{panel} SPS")
        ax.set_xlabel("")
        ax.set_ylabel("SPS" if ax is axes[0] else "")
        ax.set_ylim(1e2, 5e6)
        annotate_bars(ax, sub, panel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("white")

    ax = axes[2]
    sns.lineplot(
        data=curves,
        x="uptime",
        y="episode_return",
        hue="run",
        palette=CURVE_PALETTE,
        linewidth=2.2,
        marker="o",
        markersize=6,
        ax=ax,
        legend=True,
    )
    ax.set_title("Wall clock time for best policy in sweep")
    ax.set_xlabel("wall clock (s)")
    ax.set_ylabel("episode return")
    ax.legend(title="", frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")

    y0 = ax.get_ylim()[0]
    end_ann = [
        # run, t-label, t-label offset, pi offset, pi ha/va
        ("Constrained (5M)", "7s", (0, -14), "center", (9, -2), "left", "center"),
        ("Unconstrained", "135s", (8, -14), "left", (-8, 8), "right", "bottom"),
    ]
    for run, t_label, t_off, t_ha, pi_off, ha, va in end_ann:
        g = curves[curves["run"] == run]
        t_end = float(g["uptime"].iloc[-1])
        r_end = float(g["episode_return"].iloc[-1])
        color = CURVE_PALETTE[run]
        ax.plot(
            [t_end, t_end],
            [y0, r_end],
            linestyle=":",
            color=color,
            linewidth=1.3,
            zorder=3,
            clip_on=False,
        )
        ax.annotate(
            t_label,
            xy=(t_end, y0),
            xytext=t_off,
            textcoords="offset points",
            fontsize=9,
            color=color,
            ha=t_ha,
            va="top",
            annotation_clip=False,
        )
        ax.annotate(
            r"$\pi^*$",
            xy=(t_end, r_end),
            xytext=pi_off,
            textcoords="offset points",
            fontsize=11,
            color=color,
            ha=ha,
            va=va,
            annotation_clip=False,
        )

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    for panel in ("Rollout", "Training"):
        sub = sps[sps["panel"] == panel]
        for impl, g in sub.groupby("impl"):
            print(
                f"{panel:12} {impl:10}  "
                f"{g['sps'].mean():.1f} ± {g['sps'].std(ddof=1):.1f}"
            )
    for label, g in curves.groupby("run"):
        print(
            f"curve {label}: n={len(g)}  "
            f"t={g['uptime'].iloc[0]:.1f}→{g['uptime'].iloc[-1]:.1f}s  "
            f"R={g['episode_return'].iloc[0]:.1f}→{g['episode_return'].iloc[-1]:.1f}"
        )


if __name__ == "__main__":
    main()
