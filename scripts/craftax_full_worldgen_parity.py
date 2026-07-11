#!/usr/bin/env python
"""Compare full PufferLib Craftax native worldgen against current JAX Craftax.

This checks reset-world parity only. It is intentionally separate from the human
trace replay because a reset/worldgen mismatch is enough to invalidate later
state/reward parity.

Example:
    /Users/daphne/github/multitask_preplay/.venv/bin/python \
        scripts/craftax_full_worldgen_parity.py --seeds 0 1 2 3
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]


class NativeCraftaxWorldgen:
    def __init__(self):
        tmp = tempfile.TemporaryDirectory()
        self._tmp = tmp
        tmp_path = Path(tmp.name)
        src = tmp_path / "craftax_full_worldgen.c"
        so = tmp_path / "craftax_full_worldgen.so"
        src.write_text(
            r'''
            #include <stdint.h>
            #include <string.h>
            #include "ocean/craftax/worldgen.h"

            void world_from_seed(
                uint32_t seed,
                uint8_t* map_out,
                uint8_t* item_map_out,
                int* player_pos_out
            ) {
                CraftaxWorldState state;
                craftax_generate_world_from_seed(seed, &state);
                memcpy(map_out, state.map, CRAFTAX_WG_NUM_LEVELS * CRAFTAX_WG_MAP_SIZE * CRAFTAX_WG_MAP_SIZE);
                memcpy(item_map_out, state.item_map, CRAFTAX_WG_NUM_LEVELS * CRAFTAX_WG_MAP_SIZE * CRAFTAX_WG_MAP_SIZE);
                player_pos_out[0] = state.player_position[0];
                player_pos_out[1] = state.player_position[1];
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
                str(src),
                "-lm",
                "-o",
                str(so),
            ],
            check=True,
            cwd=ROOT,
        )
        self.lib = ctypes.CDLL(str(so))
        self.lib.world_from_seed.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.world_from_seed.restype = None

    def world_from_seed(self, seed: int):
        maps = np.empty((9, 48, 48), dtype=np.uint8)
        item_maps = np.empty((9, 48, 48), dtype=np.uint8)
        player_pos = np.empty(2, dtype=np.int32)
        self.lib.world_from_seed(
            ctypes.c_uint32(int(seed)),
            maps.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            item_maps.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            player_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return maps, item_maps, player_pos


def import_jax_craftax():
    from craftax.craftax_env import make_craftax_env_from_name
    import jax

    return jax, make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    jax, env = import_jax_craftax()
    native = NativeCraftaxWorldgen()

    all_equal = True
    for seed in args.seeds:
        native_map, native_item_map, native_pos = native.world_from_seed(seed)
        rng = jax.random.PRNGKey(seed)
        _rng, reset_key = jax.random.split(rng)
        _obs, state = env.reset(reset_key, env.default_params)
        jax_map = np.asarray(state.map, dtype=np.uint8)
        jax_item_map = np.asarray(state.item_map, dtype=np.uint8)
        jax_pos = np.asarray(state.player_position, dtype=np.int32)

        map_equal = np.array_equal(native_map, jax_map)
        item_equal = np.array_equal(native_item_map, jax_item_map)
        pos_equal = np.array_equal(native_pos, jax_pos)
        all_equal &= map_equal and item_equal and pos_equal
        print(f"seed={seed}")
        print(f"  map_equal={map_equal} diff_cells={int(np.count_nonzero(native_map != jax_map))}")
        print(f"  item_map_equal={item_equal} diff_cells={int(np.count_nonzero(native_item_map != jax_item_map))}")
        print(f"  start_equal={pos_equal} native={native_pos.tolist()} jax={jax_pos.tolist()}")

    if not all_equal:
        raise SystemExit(1)
    print("PASS: native full Craftax worldgen matches current JAX Craftax.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
