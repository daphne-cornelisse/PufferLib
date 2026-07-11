from __future__ import annotations

import ctypes
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HUMAN_DATA_DIR = ROOT / "resources" / "craftax_mini" / "human_data"
HUMAN_DATAFRAME_PATH = HUMAN_DATA_DIR / "craftax_human.parquet"
HUMAN_EXPERIMENT_MAPS_PATH = HUMAN_DATA_DIR / "craftax_human_experiment_maps.npz"
HUMAN_PLOT_DIR = HUMAN_DATA_DIR / "plots"

MAP_SIZE = 48
NUM_LEVELS = 9
GOAL_BLOCK_IDS = {
    10: "diamond",
    21: "sapphire",
    22: "ruby",
}

WEB_TO_ENV_ACTION = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
WEB_ACTION_DELTAS = {
    0: (0, 1),   # RIGHT
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (-1, 0),  # UP
    4: (0, 0),   # DO
}

# A compact categorical palette for native block ids. It is intentionally plain:
# these plots are validation artifacts, not a replacement for raylib rendering.
BLOCK_COLORS = np.asarray(
    [
        [255, 0, 255],    # invalid
        [8, 8, 8],        # out of bounds
        [58, 153, 65],    # grass
        [45, 105, 190],   # water
        [120, 120, 120],  # stone
        [46, 95, 45],     # tree
        [139, 91, 43],    # wood
        [118, 96, 66],    # path
        [35, 35, 35],     # coal
        [150, 105, 82],   # iron
        [96, 220, 220],   # diamond
        [154, 107, 70],   # crafting table
        [80, 80, 85],     # furnace
        [215, 195, 120],  # sand
        [220, 74, 32],    # lava
        [62, 180, 60],    # plant
        [95, 205, 70],    # ripe plant
        [78, 78, 84],     # wall
        [18, 18, 25],     # darkness
        [72, 95, 76],     # moss wall
        [105, 105, 112],  # stalagmite
        [72, 92, 210],    # sapphire
        [190, 42, 60],    # ruby
        [120, 76, 32],    # chest
        [60, 145, 190],   # fountain
        [180, 70, 38],    # fire grass
        [120, 190, 220],  # ice grass
        [92, 86, 80],     # gravel
        [190, 82, 38],    # fire tree
        [160, 210, 230],  # ice shrub
        [190, 65, 34],    # fire enchant table
        [88, 168, 220],   # ice enchant table
        [78, 32, 86],     # necromancer
        [92, 92, 98],     # grave
        [84, 84, 92],     # grave2
        [75, 75, 86],     # grave3
        [128, 42, 138],   # vulnerable necromancer
    ],
    dtype=np.uint8,
)


def parse_jax_array_string(value: str) -> np.ndarray:
    value = value.strip().strip("[]")
    if not value:
        return np.asarray([], dtype=np.int32)
    return np.asarray([int(float(item)) for item in value.split()], dtype=np.int32)


def parse_positions_string(value: str) -> np.ndarray:
    value = value.strip().strip("[]")
    if not value:
        return np.empty((0, 2), dtype=np.int32)
    rows = re.split(r"\]\s*\[", value)
    return np.asarray(
        [[int(item) for item in row.strip().strip("[]").split()] for row in rows],
        dtype=np.int32,
    )


def validate_human_row(
    row: dict,
    *,
    require_native_bounds: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = parse_positions_string(row["positions"])
    web_actions = parse_jax_array_string(row["actions"])

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise AssertionError(f"positions must have shape (T, 2), got {positions.shape}")
    if len(positions) != len(web_actions) + 1:
        raise AssertionError(
            f"positions/actions length mismatch: {len(positions)} vs {len(web_actions)}"
        )
    if require_native_bounds and (np.any(positions < 0) or np.any(positions >= MAP_SIZE)):
        raise AssertionError("positions contain coordinates outside the Craftax 48x48 map")
    if np.any(web_actions < 0) or np.any(web_actions > 4):
        raise AssertionError("human web actions must be in [0, 4]")

    for idx, (action, delta) in enumerate(zip(web_actions, np.diff(positions, axis=0))):
        observed = tuple(int(x) for x in delta)
        expected = WEB_ACTION_DELTAS[int(action)]
        if observed != (0, 0) and observed != expected:
            raise AssertionError(
                f"action/path mismatch at step {idx}: action={action}, "
                f"expected delta {expected} or blocked delta (0, 0), got {observed}"
            )

    return positions, web_actions, WEB_TO_ENV_ACTION[web_actions]


def is_native_bounds_compatible(row: dict) -> bool:
    positions = parse_positions_string(row["positions"])
    return bool(np.all((positions >= 0) & (positions < MAP_SIZE)))


def load_human_experiment_maps(path: Path = HUMAN_EXPERIMENT_MAPS_PATH) -> dict[int, np.ndarray]:
    with np.load(path) as data:
        return {
            int(key.removeprefix("world_")): data[key].copy()
            for key in data.files
            if key.startswith("world_") and key.removeprefix("world_").isdigit()
        }


def load_human_experiment_map_metadata(path: Path = HUMAN_EXPERIMENT_MAPS_PATH) -> dict[int, dict[str, np.ndarray]]:
    metadata: dict[int, dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for key in data.files:
            if not key.startswith("world_"):
                continue
            parts = key.split("_", 2)
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            world = int(parts[1])
            metadata.setdefault(world, {})[parts[2]] = data[key].copy()
    return metadata


class NativeCraftaxMiniWorlds:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        src = tmp / "craftax_mini_human_worlds.c"
        so = tmp / "craftax_mini_human_worlds.so"
        src.write_text(
            r'''
            #include <stdint.h>
            #include <string.h>
            #include "ocean/craftax_mini/craftax_mini.h"

            void craftax_mini_human_world_from_seed(
                uint32_t seed,
                uint8_t* map_out,
                uint8_t* item_map_out,
                int* player_pos_out
            ) {
                Craftax env;
                memset(&env, 0, sizeof(env));
                env.num_agents = 1;
                env.rng = seed;
                env.seed = seed;
                c_init(&env);
                memcpy(map_out, env.state->map, CRAFTAX_NUM_LEVELS * CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE);
                memcpy(item_map_out, env.state->item_map, CRAFTAX_NUM_LEVELS * CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE);
                player_pos_out[0] = env.state->player_position[0];
                player_pos_out[1] = env.state->player_position[1];
                c_close(&env);
            }

            void craftax_mini_human_world_from_index(
                int index,
                uint8_t* map_out,
                uint8_t* item_map_out,
                int* player_pos_out
            ) {
                craftax_mini_human_world_from_seed(
                    (uint32_t)craftax_mini_human_seed_for_index(index),
                    map_out,
                    item_map_out,
                    player_pos_out
                );
            }
            '''
        )
        subprocess.run(
            [
                os.environ.get("CC", "cc"),
                "-std=c99",
                "-O2",
                "-shared",
                "-fPIC",
                "-I",
                str(ROOT),
                "-I",
                str(ROOT / "raylib-5.5_macos" / "include"),
                str(src),
                "-lm",
                "-o",
                str(so),
            ],
            check=True,
            cwd=ROOT,
        )
        self.lib = ctypes.CDLL(str(so))
        self.lib.craftax_mini_human_world_from_seed.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.craftax_mini_human_world_from_seed.restype = None
        self.lib.craftax_mini_human_world_from_index.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.craftax_mini_human_world_from_index.restype = None

    def world_from_seed(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        maps = np.empty((NUM_LEVELS, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        item_maps = np.empty((NUM_LEVELS, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        player_pos = np.empty(2, dtype=np.int32)
        self.lib.craftax_mini_human_world_from_seed(
            ctypes.c_uint32(int(seed)),
            maps.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            item_maps.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            player_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return maps, item_maps, player_pos

    def world_from_index(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        maps = np.empty((NUM_LEVELS, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        item_maps = np.empty((NUM_LEVELS, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        player_pos = np.empty(2, dtype=np.int32)
        self.lib.craftax_mini_human_world_from_index(
            ctypes.c_int32(int(index)),
            maps.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            item_maps.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            player_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return maps, item_maps, player_pos


def render_native_map(map_level: np.ndarray) -> np.ndarray:
    clipped = np.clip(map_level, 0, len(BLOCK_COLORS) - 1)
    return BLOCK_COLORS[clipped]


def plot_human_trajectories(rows: list[dict], out_path: Path, worlds: NativeCraftaxMiniWorlds | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    worlds = worlds or NativeCraftaxMiniWorlds()

    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 5), squeeze=False)
    for ax, row in zip(axes[0], rows):
        positions, _, _ = validate_human_row(row, require_native_bounds=True)
        maps, _, native_start = worlds.world_from_seed(int(row["world"]))
        image = render_native_map(maps[0])
        ax.imshow(image, interpolation="nearest")
        ax.plot(positions[:, 1], positions[:, 0], color="red", linewidth=1.4)
        ax.scatter(positions[0, 1], positions[0, 0], c="yellow", s=80, edgecolors="black", marker="*")
        ax.scatter(positions[-1, 1], positions[-1, 0], c="yellow", s=45, edgecolors="black")
        ax.scatter(native_start[1], native_start[0], c="white", s=35, edgecolors="black")
        ax.set_title(
            f"world={row['world']} eval={row['eval']}\n"
            f"user={row['user_id']} len={row['path_length']}"
        )
        ax.set_xlim(-0.5, MAP_SIZE - 0.5)
        ax.set_ylim(MAP_SIZE - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
