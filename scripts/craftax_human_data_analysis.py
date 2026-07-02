#!/usr/bin/env python
"""Analyze the vendored Craftax human dataframe and plot key statistics.

Examples:
    uv run --with polars --with pandas --with seaborn --with matplotlib \
        python scripts/craftax_human_data_analysis.py
    uv run --with polars --with pandas --with seaborn --with matplotlib \
        python scripts/craftax_human_data_analysis.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from pufferlib.craftax_mini_human_data import (
    GOAL_BLOCK_IDS,
    HUMAN_DATAFRAME_PATH,
    HUMAN_PLOT_DIR,
    MAP_SIZE,
    load_human_experiment_map_metadata,
    load_human_experiment_maps,
    parse_jax_array_string,
    parse_positions_string,
    render_native_map,
)


WEB_ACTION_NAMES = {
    0: "RIGHT",
    1: "DOWN",
    2: "LEFT",
    3: "UP",
    4: "DO",
}

def split_name(eval_value: bool) -> str:
    return "eval" if eval_value else "train"


def add_trajectory_diagnostics(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add parsed trajectory diagnostics and return per-action counts."""
    compatible = []
    min_row = []
    max_row = []
    min_col = []
    max_col = []
    stationary_steps = []
    action_counts = np.zeros(len(WEB_ACTION_NAMES), dtype=np.int64)
    stationary_by_action = np.zeros(len(WEB_ACTION_NAMES), dtype=np.int64)

    for row in df.select(["positions", "actions"]).iter_rows(named=True):
        positions = parse_positions_string(row["positions"])
        actions = parse_jax_array_string(row["actions"])
        deltas = np.diff(positions, axis=0)
        stationary = (deltas[:, 0] == 0) & (deltas[:, 1] == 0)

        compatible.append(bool(np.all((positions >= 0) & (positions < MAP_SIZE))))
        min_row.append(int(positions[:, 0].min()))
        max_row.append(int(positions[:, 0].max()))
        min_col.append(int(positions[:, 1].min()))
        max_col.append(int(positions[:, 1].max()))
        stationary_steps.append(int(stationary.sum()))

        for action in actions:
            action_counts[int(action)] += 1
        for action, is_stationary in zip(actions, stationary):
            if is_stationary:
                stationary_by_action[int(action)] += 1

    diagnosed = df.with_columns(
        [
            pl.Series("native_48x48_compatible", compatible),
            pl.Series("min_row", min_row),
            pl.Series("max_row", max_row),
            pl.Series("min_col", min_col),
            pl.Series("max_col", max_col),
            pl.Series("stationary_steps", stationary_steps),
        ]
    )

    action_df = pl.DataFrame(
        {
            "action_id": list(WEB_ACTION_NAMES),
            "action": [WEB_ACTION_NAMES[idx] for idx in WEB_ACTION_NAMES],
            "count": action_counts,
            "stationary_count": stationary_by_action,
        }
    ).with_columns(
        [
            (pl.col("count") / pl.col("count").sum()).alias("fraction"),
            (pl.col("stationary_count") / pl.col("count")).alias("stationary_fraction"),
        ]
    )
    return diagnosed, action_df


def dataframe_overview(df: pl.DataFrame, action_df: pl.DataFrame) -> str:
    rows = df.height
    users = df["user_id"].n_unique()
    worlds = sorted(df["world"].unique().to_list(), key=lambda value: int(value))
    steps = int(df["path_length"].sum())
    successes = int(df["success"].sum())
    compatible = int(df["native_48x48_compatible"].sum())

    by_split = df.group_by("eval").agg(
        [
            pl.len().alias("trajectories"),
            pl.col("user_id").n_unique().alias("users"),
            pl.col("world").n_unique().alias("worlds"),
            pl.col("success").sum().cast(pl.Int64).alias("successes"),
            pl.col("path_length").sum().alias("steps"),
            pl.col("path_length").mean().alias("mean_length"),
            pl.col("path_length").median().alias("median_length"),
            pl.col("path_length").max().alias("max_length"),
            pl.col("native_48x48_compatible").sum().alias("native_compatible"),
        ]
    ).sort("eval")

    per_user = df.group_by("user_id").agg(
        [
            pl.len().alias("trajectories"),
            pl.col("path_length").sum().alias("steps"),
            pl.col("success").sum().alias("successes"),
            pl.col("eval").sum().alias("eval_trajectories"),
            pl.col("world").n_unique().alias("worlds"),
        ]
    )

    length_quantiles = {
        f"p{int(q * 100):02d}": df.select(pl.col("path_length").quantile(q)).item()
        for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    }

    lines = [
        "Craftax human data overview",
        f"dataframe: {HUMAN_DATAFRAME_PATH}",
        f"rows / trajectories: {rows:,}",
        f"unique global episodes: {df['global_episode_idx'].n_unique():,}",
        f"unique humans: {users:,}",
        f"worlds: {', '.join(worlds)}",
        f"total actions / steps: {steps:,}",
        f"successes: {successes:,} ({successes / rows:.1%})",
        (
            f"native {MAP_SIZE}x{MAP_SIZE} compatible trajectories: "
            f"{compatible:,} / {rows:,} ({compatible / rows:.1%})"
        ),
        "",
        "By split:",
    ]
    for row in by_split.to_dicts():
        lines.append(
            "  "
            f"{split_name(row['eval'])}: trajectories={row['trajectories']:,}, "
            f"users={row['users']:,}, worlds={row['worlds']:,}, "
            f"successes={row['successes']:,}, steps={row['steps']:,}, "
            f"median_len={row['median_length']:.0f}, mean_len={row['mean_length']:.1f}, "
            f"max_len={row['max_length']:,}, native_compatible={row['native_compatible']:,}"
        )

    user_stats = per_user.select(
        [
            pl.col("trajectories").min().alias("min_trajectories"),
            pl.col("trajectories").median().alias("median_trajectories"),
            pl.col("trajectories").mean().alias("mean_trajectories"),
            pl.col("trajectories").max().alias("max_trajectories"),
            pl.col("steps").median().alias("median_steps"),
            pl.col("steps").max().alias("max_steps"),
        ]
    ).to_dicts()[0]
    lines.extend(
        [
            "",
            "Per-human coverage:",
            (
                "  trajectories per human: "
                f"min={user_stats['min_trajectories']:,}, "
                f"median={user_stats['median_trajectories']:.0f}, "
                f"mean={user_stats['mean_trajectories']:.1f}, "
                f"max={user_stats['max_trajectories']:,}"
            ),
            (
                "  steps per human: "
                f"median={user_stats['median_steps']:.0f}, "
                f"max={user_stats['max_steps']:,}"
            ),
            "",
            "Path length quantiles:",
            "  " + ", ".join(f"{key}={value:.0f}" for key, value in length_quantiles.items()),
            "",
            "Web action distribution:",
        ]
    )
    for row in action_df.to_dicts():
        lines.append(
            "  "
            f"{row['action']}: {row['count']:,} ({row['fraction']:.1%}), "
            f"stationary={row['stationary_count']:,} ({row['stationary_fraction']:.1%})"
        )

    bounds = df.select(
        [
            pl.col("min_row").min().alias("min_row"),
            pl.col("max_row").max().alias("max_row"),
            pl.col("min_col").min().alias("min_col"),
            pl.col("max_col").max().alias("max_col"),
        ]
    ).to_dicts()[0]
    lines.extend(
        [
            "",
            "Observed coordinate range:",
            (
                "  "
                f"row=[{bounds['min_row']}, {bounds['max_row']}], "
                f"col=[{bounds['min_col']}, {bounds['max_col']}]"
            ),
        ]
    )
    return "\n".join(lines)


def save_overview_plots(
    df: pl.DataFrame,
    action_df: pl.DataFrame,
    out_dir: Path,
    *,
    show: bool = False,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="notebook", style="whitegrid")

    plot_df = df.with_columns(
        [
            pl.col("eval").map_elements(split_name, return_dtype=pl.String).alias("split"),
            pl.when(pl.col("success") == 1.0)
            .then(pl.lit("success"))
            .otherwise(pl.lit("failure"))
            .alias("outcome"),
        ]
    )
    pdf = pd.DataFrame(plot_df.to_dicts())
    pdf["world_num"] = pd.to_numeric(pdf["world"])
    action_pdf = pd.DataFrame(action_df.to_dicts())

    saved = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.countplot(data=pdf, x="split", hue="outcome", ax=axes[0, 0])
    axes[0, 0].set_title("Trajectories by split and outcome")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("trajectories")

    success_by_split = (
        pdf.groupby("split", as_index=False)["success"].mean().rename(columns={"success": "success_rate"})
    )
    sns.barplot(data=success_by_split, x="split", y="success_rate", ax=axes[0, 1])
    axes[0, 1].set_title("Success rate by split")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("success rate")
    axes[0, 1].set_ylim(0, 1)

    sns.histplot(data=pdf, x="path_length", hue="split", bins=50, element="step", ax=axes[1, 0])
    axes[1, 0].set_title("Path length distribution")
    axes[1, 0].set_xlabel("actions per trajectory")
    axes[1, 0].set_ylabel("trajectories")

    sns.barplot(data=action_pdf, x="action", y="count", ax=axes[1, 1])
    axes[1, 1].set_title("Web action counts")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("actions")
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    path = out_dir / "craftax_human_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    per_user = (
        pdf.groupby("user_id", as_index=False)
        .agg(
            trajectories=("global_episode_idx", "count"),
            steps=("path_length", "sum"),
            successes=("success", "sum"),
            eval_trajectories=("eval", "sum"),
        )
        .assign(success_rate=lambda data: data["successes"] / data["trajectories"])
    )
    per_world = (
        pdf.groupby("world_num", as_index=False)
        .agg(
            trajectories=("global_episode_idx", "count"),
            users=("user_id", "nunique"),
            success_rate=("success", "mean"),
        )
        .sort_values("world_num")
    )
    compatibility = (
        pdf.groupby(["split", "native_48x48_compatible"], as_index=False)
        .size()
        .rename(columns={"size": "trajectories"})
    )
    compatibility["native_48x48_compatible"] = compatibility["native_48x48_compatible"].map(
        {True: "compatible", False: "out of bounds"}
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.histplot(data=per_user, x="trajectories", bins=25, ax=axes[0, 0])
    axes[0, 0].set_title("Trajectories per human")
    axes[0, 0].set_xlabel("trajectories")
    axes[0, 0].set_ylabel("humans")

    sns.histplot(data=per_user, x="steps", bins=30, ax=axes[0, 1])
    axes[0, 1].set_title("Steps per human")
    axes[0, 1].set_xlabel("actions")
    axes[0, 1].set_ylabel("humans")

    sns.barplot(data=per_world, x="world_num", y="trajectories", ax=axes[1, 0])
    axes[1, 0].set_title("Trajectories by world")
    axes[1, 0].set_xlabel("world seed")
    axes[1, 0].set_ylabel("trajectories")

    sns.barplot(data=compatibility, x="split", y="trajectories", hue="native_48x48_compatible", ax=axes[1, 1])
    axes[1, 1].set_title(f"Native {MAP_SIZE}x{MAP_SIZE} map compatibility")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("trajectories")

    fig.tight_layout()
    path = out_dir / "craftax_human_users_worlds_bounds.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    demographic_cols = {"age", "sex"}.intersection(pdf.columns)
    if demographic_cols:
        user_demo_cols = ["user_id"] + sorted(demographic_cols)
        user_demo = pdf[user_demo_cols].drop_duplicates("user_id")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if "age" in user_demo:
            sns.histplot(data=user_demo, x="age", bins=25, ax=axes[0])
            axes[0].set_title("Age distribution")
            axes[0].set_xlabel("age")
            axes[0].set_ylabel("humans")
        else:
            axes[0].axis("off")
        if "sex" in user_demo:
            sns.countplot(data=user_demo, x="sex", ax=axes[1])
            axes[1].set_title("Sex distribution")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("humans")
            axes[1].tick_params(axis="x", rotation=30)
        else:
            axes[1].axis("off")
        fig.tight_layout()
        path = out_dir / "craftax_human_demographics.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    if show:
        plt.show()

    return saved


def save_episode_length_plot(df: pl.DataFrame, out_dir: Path, *, show: bool = False) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="notebook", style="whitegrid")

    plot_df = df.with_columns(
        [
            pl.col("eval").map_elements(split_name, return_dtype=pl.String).alias("split"),
            pl.when(pl.col("success") == 1.0)
            .then(pl.lit("success"))
            .otherwise(pl.lit("failure"))
            .alias("outcome"),
        ]
    )
    pdf = pd.DataFrame(plot_df.to_dicts())
    quantiles = df.select(
        [
            pl.col("path_length").quantile(0.5).alias("median"),
            pl.col("path_length").quantile(0.9).alias("p90"),
            pl.col("path_length").quantile(0.95).alias("p95"),
            pl.col("path_length").quantile(0.99).alias("p99"),
        ]
    ).to_dicts()[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.histplot(
        data=pdf,
        x="path_length",
        hue="split",
        bins=80,
        element="step",
        stat="count",
        common_norm=False,
        ax=axes[0],
    )
    axes[0].set_title("Episode length distribution")
    axes[0].set_xlabel("steps per trajectory")
    axes[0].set_ylabel("trajectories")
    for label, value in quantiles.items():
        axes[0].axvline(value, linestyle="--", linewidth=1)
        axes[0].text(value, axes[0].get_ylim()[1] * 0.92, label, rotation=90, va="top", ha="right")

    sns.boxplot(data=pdf, x="split", y="path_length", hue="outcome", ax=axes[1])
    axes[1].set_title("Episode length by split and outcome")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("steps per trajectory")

    fig.tight_layout()
    path = out_dir / "craftax_human_episode_lengths.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    if show:
        plt.show()

    return [path]


def trajectory_visit_counts(rows: list[dict]) -> np.ndarray:
    visits = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int64)
    for row in rows:
        positions = parse_positions_string(row["positions"])
        if np.any((positions < 0) | (positions >= MAP_SIZE)):
            continue
        np.add.at(visits, (positions[:, 0], positions[:, 1]), 1)
    return visits


def mark_key_objects(
    ax,
    map_level: np.ndarray,
    starts: np.ndarray,
    *,
    label: bool = False,
) -> None:
    for block_id, name in GOAL_BLOCK_IDS.items():
        cells = np.argwhere(map_level == block_id)
        if not len(cells):
            continue
        ax.scatter(
            cells[:, 1],
            cells[:, 0],
            c="#6ff7ff",
            s=48,
            marker="D",
            edgecolors="black",
            linewidths=0.7,
            label=name if label else None,
        )
    if len(starts):
        ax.scatter(
            starts[:, 1],
            starts[:, 0],
            c="white",
            s=42,
            edgecolors="black",
            linewidths=0.8,
            label="configured starts" if label else None,
        )


def format_map_axes(ax) -> None:
    ax.set_xlim(-0.5, MAP_SIZE - 0.5)
    ax.set_ylim(MAP_SIZE - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def save_all_trajectory_overlays(
    df: pl.DataFrame,
    out_dir: Path,
    *,
    show: bool = False,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_maps = load_human_experiment_maps()
    experiment_metadata = load_human_experiment_map_metadata()
    worlds = sorted(df["world"].unique().to_list(), key=lambda value: int(value))
    saved = []
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(alpha=0.0)

    fig, axes = plt.subplots(len(worlds), 2, figsize=(10, 5 * len(worlds)), squeeze=False)
    for row_idx, world in enumerate(worlds):
        world_df = df.filter(pl.col("world") == world)
        compatible_df = world_df.filter(pl.col("native_48x48_compatible"))
        rows = compatible_df.select(["positions"]).to_dicts()
        skipped = world_df.height - compatible_df.height

        world_int = int(world)
        maps = experiment_maps[world_int]
        metadata = experiment_metadata.get(world_int, {})
        starts = np.concatenate(
            [
                metadata.get("train_starts", np.empty((0, 2), dtype=np.int32)),
                metadata.get("eval_starts", np.empty((0, 2), dtype=np.int32)),
            ],
            axis=0,
        )
        visits = trajectory_visit_counts(rows)
        density = np.ma.masked_where(visits == 0, np.log1p(visits))

        map_ax = axes[row_idx, 0]
        density_ax = axes[row_idx, 1]
        map_ax.imshow(render_native_map(maps[0]), interpolation="nearest")
        mark_key_objects(map_ax, maps[0], starts, label=row_idx == 0)
        map_ax.set_title(f"world={world} human experiment map")
        format_map_axes(map_ax)

        image = density_ax.imshow(density, cmap=cmap, interpolation="nearest")
        mark_key_objects(density_ax, maps[0], starts, label=row_idx == 0)
        density_ax.set_title(
            f"trajectory density  trajectories={compatible_df.height:,}  skipped={skipped:,}\n"
            f"visited cells={(visits > 0).sum():,}  max visits={visits.max():,}"
        )
        format_map_axes(density_ax)
        fig.colorbar(image, ax=density_ax, fraction=0.046, pad=0.02, label="log(1 + visits)")

    fig.suptitle(
        "Human experiment maps beside trajectory density: white dots = configured starts, cyan diamonds = goals",
        y=0.998,
    )
    fig.tight_layout()
    combined_path = out_dir / "craftax_human_all_trajectory_overlays.png"
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)
    saved.append(combined_path)

    for world in worlds:
        world_df = df.filter(pl.col("world") == world)
        compatible_df = world_df.filter(pl.col("native_48x48_compatible"))
        rows = compatible_df.select(["positions"]).to_dicts()
        skipped = world_df.height - compatible_df.height
        visits = trajectory_visit_counts(rows)

        world_int = int(world)
        maps = experiment_maps[world_int]
        metadata = experiment_metadata.get(world_int, {})
        starts = np.concatenate(
            [
                metadata.get("train_starts", np.empty((0, 2), dtype=np.int32)),
                metadata.get("eval_starts", np.empty((0, 2), dtype=np.int32)),
            ],
            axis=0,
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), squeeze=False)
        map_ax = axes[0, 0]
        density_ax = axes[0, 1]

        map_ax.imshow(render_native_map(maps[0]), interpolation="nearest")
        mark_key_objects(map_ax, maps[0], starts, label=True)
        map_ax.set_title(f"world={world} human experiment map")
        format_map_axes(map_ax)

        density = np.ma.masked_where(visits == 0, np.log1p(visits))
        image = density_ax.imshow(density, cmap=cmap, interpolation="nearest")
        mark_key_objects(density_ax, maps[0], starts, label=True)
        density_ax.set_title(
            f"trajectory density  trajectories={compatible_df.height:,}  skipped={skipped:,}\n"
            f"visited cells={(visits > 0).sum():,}  max visits={visits.max():,}"
        )
        format_map_axes(density_ax)
        fig.colorbar(image, ax=density_ax, fraction=0.046, pad=0.02, label="log(1 + visits)")

        handles, labels = density_ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncols=len(handles), frameon=False)
            fig.subplots_adjust(bottom=0.12)
        fig.tight_layout()
        path = out_dir / f"craftax_human_trajectory_overlay_world_{world}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        saved.append(path)

    if show:
        plt.show()

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataframe", default=str(HUMAN_DATAFRAME_PATH))
    parser.add_argument("--out-dir", default=str(HUMAN_PLOT_DIR))
    parser.add_argument("--show", action="store_true", help="Display plots interactively after saving them.")
    args = parser.parse_args()

    df = pl.read_parquet(args.dataframe)
    diagnosed_df, action_df = add_trajectory_diagnostics(df)
    report = dataframe_overview(diagnosed_df, action_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "craftax_human_data_analysis.txt"
    report_path.write_text(report + "\n")
    print(report)
    print(f"\nsaved report: {report_path}")

    saved_paths = []
    saved_paths.extend(save_overview_plots(diagnosed_df, action_df, out_dir, show=args.show))
    saved_paths.extend(save_episode_length_plot(diagnosed_df, out_dir, show=args.show))
    saved_paths.extend(save_all_trajectory_overlays(diagnosed_df, out_dir, show=args.show))
    for path in saved_paths:
        print(f"saved plot: {path}")


if __name__ == "__main__":
    main()
