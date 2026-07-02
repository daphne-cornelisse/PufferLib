#!/usr/bin/env python
"""Check Craftax Mini native maps against the vendored human experiment maps.

Example:
    uv run --with numpy python scripts/craftax_mini_human_map_parity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pufferlib.craftax_mini_human_data import (  # noqa: E402
    NativeCraftaxMiniWorlds,
    load_human_experiment_maps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 15, 20, 95])
    args = parser.parse_args()

    expected_maps = load_human_experiment_maps()
    worlds = NativeCraftaxMiniWorlds()

    all_equal = True
    for seed in args.seeds:
        native_map, _item_map, native_start = worlds.world_from_seed(seed)
        expected_map = expected_maps[seed]
        map_equal = np.array_equal(native_map, expected_map)
        all_equal &= map_equal

        print(f"world={seed}")
        print(
            "  craftax_mini map equal to vendored human map: "
            f"{map_equal} diff_cells={int(np.count_nonzero(native_map != expected_map))}"
        )
        print(f"  native reset start: {native_start.tolist()}")

    if not all_equal:
        raise SystemExit(1)
    print("PASS: Craftax Mini native maps match all vendored human experiment maps.")


if __name__ == "__main__":
    main()
