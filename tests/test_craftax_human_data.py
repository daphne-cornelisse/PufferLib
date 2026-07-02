from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pufferlib.craftax_mini_human_data import (
    GOAL_BLOCK_IDS,
    HUMAN_DATAFRAME_PATH,
    HUMAN_EXPERIMENT_MAPS_PATH,
    MAP_SIZE,
    NativeCraftaxMiniWorlds,
    is_native_bounds_compatible,
    load_human_experiment_map_metadata,
    load_human_experiment_maps,
    plot_human_trajectories,
    validate_human_row,
)


EXPECTED_SHA256 = "ed75445777b7f7246db5bf901a0f8cbf6884f8ab8e30580c8df5099b6f63b428"
EXPECTED_EXPERIMENT_MAPS_SHA256 = (
    "d0caaed13633c7d59c5a0635453d4a86fb2de84691a39bdf13b1231eba277ecb"
)
REQUIRED_COLUMNS = {
    "domain",
    "algo",
    "user_id",
    "world",
    "eval",
    "global_episode_idx",
    "corresponding_train_episode_idx",
    "success",
    "path_length",
    "positions",
    "actions",
}


def _load_df():
    pl = pytest.importorskip("polars")
    assert HUMAN_DATAFRAME_PATH.exists(), f"missing {HUMAN_DATAFRAME_PATH}"
    return pl.read_parquet(HUMAN_DATAFRAME_PATH)


def test_craftax_human_dataframe_is_vendored_exactly():
    digest = hashlib.sha256(HUMAN_DATAFRAME_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256


def test_craftax_human_experiment_maps_are_vendored_exactly():
    digest = hashlib.sha256(HUMAN_EXPERIMENT_MAPS_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_EXPERIMENT_MAPS_SHA256

    maps = load_human_experiment_maps()
    metadata = load_human_experiment_map_metadata()
    assert set(maps) == {3, 15, 20, 95}
    assert set(metadata) == {3, 15, 20, 95}

    for world, world_maps in maps.items():
        assert world_maps.shape == (9, MAP_SIZE, MAP_SIZE)
        assert world_maps.dtype == np.uint8

        goal_locations = metadata[world]["goal_locations"]
        goal_blocks = metadata[world]["goal_blocks"]
        assert goal_locations.shape == (3, 2)
        assert set(goal_blocks.tolist()).issubset(set(GOAL_BLOCK_IDS))
        for (row, col), block in zip(goal_locations, goal_blocks):
            assert int(world_maps[0, row, col]) == int(block)


def test_craftax_human_dataframe_schema_and_splits():
    df = _load_df()
    assert df.height > 0
    assert REQUIRED_COLUMNS.issubset(set(df.columns))
    assert set(df["domain"].unique().to_list()) == {"craftax"}
    assert set(df["algo"].unique().to_list()) == {"human"}
    assert set(df["eval"].unique().to_list()) == {False, True}
    assert df.filter(df["success"] == 1.0).height > 0


def test_craftax_human_paths_and_actions_are_internally_consistent():
    df = _load_df()
    rows = df.select(
        [
            "positions",
            "actions",
            "path_length",
            "world",
            "eval",
            "user_id",
        ]
    ).to_dicts()
    for idx, row in enumerate(rows):
        positions, web_actions, env_actions = validate_human_row(
            row,
            require_native_bounds=False,
        )
        assert len(web_actions) == int(row["path_length"]), idx
        assert len(positions) == int(row["path_length"]) + 1, idx
        assert np.all((env_actions >= 0) & (env_actions < 5)), idx


@pytest.mark.xfail(
    strict=True,
    reason="329 rows in the current dataframe contain positions outside the native 48x48 map.",
)
def test_all_craftax_human_rows_are_native_bounds_compatible():
    df = _load_df()
    incompatible = [row for row in df.to_dicts() if not is_native_bounds_compatible(row)]
    assert incompatible == []


def test_craftax_mini_native_maps_match_vendored_human_experiment_maps():
    expected_maps = load_human_experiment_maps()
    worlds = NativeCraftaxMiniWorlds()

    for world, expected in expected_maps.items():
        maps, item_maps, native_start = worlds.world_from_seed(world)
        assert np.array_equal(maps, expected), world
        assert item_maps.shape == (9, MAP_SIZE, MAP_SIZE)
        assert np.all((native_start >= 0) & (native_start < MAP_SIZE))


def test_craftax_human_worlds_and_start_positions_fit_craftax_mini_maps_for_compatible_rows():
    df = _load_df()
    candidate_rows = (
        df.filter((df["success"] == 1.0) & (df["path_length"] > 2))
        .group_by(["eval", "world"])
        .first()
        .sort(["eval", "world"])
        .to_dicts()
    )
    rows = [row for row in candidate_rows if is_native_bounds_compatible(row)][:12]
    assert rows

    expected_maps = load_human_experiment_maps()
    has_task_specific_start = False
    for row in rows:
        positions, _, _ = validate_human_row(row)
        maps = expected_maps[int(row["world"])]
        assert maps.shape == (9, MAP_SIZE, MAP_SIZE)

        start = positions[0]
        block = int(maps[0, start[0], start[1]])
        assert block != 1, f"human start is out-of-bounds for row {row}"
        has_task_specific_start |= not np.array_equal(start, [24, 24])

    # Human collection used task-specific starts; Craftax Mini can represent them, but
    # replay code must set the initial position instead of relying on reset().
    assert has_task_specific_start


def test_craftax_human_trajectory_plot_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    df = _load_df()
    candidate_rows = df.filter((df["success"] == 1.0) & (df["path_length"] > 2)).to_dicts()
    rows = [row for row in candidate_rows if is_native_bounds_compatible(row)][:3]
    assert len(rows) == 3
    out_path = tmp_path / "test_human_trajectories_on_craftax_mini_maps.png"
    plot_human_trajectories(rows, out_path, worlds=NativeCraftaxMiniWorlds())
    assert out_path.exists()
    assert out_path.stat().st_size > 10_000
