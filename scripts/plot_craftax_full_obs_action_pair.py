#!/usr/bin/env python3
"""Plot one symbolic observation-action pair from the full Craftax human data.

Examples:
    python scripts/plot_craftax_full_obs_action_pair.py --run run1 --index 0
    python scripts/plot_craftax_full_obs_action_pair.py --dataset resources/craftax/people/run1_obs_actions.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pufferlib.craftax_full_human_runs import (
    CRAFTAX_RUNS_DIR,
    extract_obs_action_pairs,
    load_obs_action_npz,
)


ROOT = Path(__file__).resolve().parents[1]

OBS_ROWS = 9
OBS_COLS = 11
NUM_BLOCK_TYPES = 37
NUM_ITEM_TYPES = 5
NUM_MOB_CLASSES = 5
NUM_MOB_TYPES = 8
NUM_TILE_CHANNELS = NUM_BLOCK_TYPES + NUM_ITEM_TYPES + NUM_MOB_CLASSES * NUM_MOB_TYPES + 1
MAP_OBS_SIZE = OBS_ROWS * OBS_COLS * NUM_TILE_CHANNELS

BLOCK_NAMES = [
    "invalid", "oob", "grass", "water", "stone", "tree", "wood", "path",
    "coal", "iron", "diamond", "table", "furnace", "sand", "lava", "plant",
    "ripe_plant", "wall", "darkness", "wall_moss", "stalagmite", "sapphire",
    "ruby", "chest", "fountain", "fire_grass", "ice_grass", "gravel",
    "fire_tree", "ice_shrub", "ench_fire", "ench_ice", "necromancer",
    "grave", "grave2", "grave3", "necromancer_vulnerable",
]
ITEM_NAMES = ["none", "torch", "ladder_down", "ladder_up", "ladder_down_blocked"]
MOB_CLASS_NAMES = [
    "melee_mobs",
    "passive_mobs",
    "ranged_mobs",
    "mob_projectiles",
    "player_projectiles",
]
ACTION_NAMES = [
    "noop", "left", "right", "up", "down", "do", "sleep", "place_stone",
    "place_table", "place_furnace", "place_plant", "make_wood_pickaxe",
    "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword",
    "make_stone_sword", "make_iron_sword", "rest", "descend", "ascend",
    "make_diamond_pickaxe", "make_diamond_sword", "make_iron_armour",
    "make_diamond_armour", "shoot_arrow", "make_arrow", "cast_fireball",
    "cast_iceball", "place_torch", "drink_potion_red", "drink_potion_green",
    "drink_potion_blue", "drink_potion_pink", "drink_potion_cyan",
    "drink_potion_yellow", "read_book", "enchant_sword", "enchant_armour",
    "make_torch", "level_up_dexterity", "level_up_strength",
    "level_up_intelligence", "enchant_bow",
]
SCALAR_NAMES = [
    "wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "sapling",
    "torches", "arrows", "books", "pickaxe", "sword", "sword_enchantment",
    "bow_enchantment", "bow", "potion_red", "potion_green", "potion_blue",
    "potion_pink", "potion_cyan", "potion_yellow", "health", "food", "drink",
    "energy", "mana", "xp", "dexterity", "strength", "intelligence",
    "dir_left", "dir_right", "dir_up", "dir_down", "armour_0", "armour_1",
    "armour_2", "armour_3", "armour_enchant_0", "armour_enchant_1",
    "armour_enchant_2", "armour_enchant_3", "light_level", "is_sleeping",
    "is_resting", "learned_fireball", "learned_iceball", "player_level",
    "ladder_down_open", "boss_vulnerable",
]


def decode_obs(obs: np.ndarray) -> dict[str, np.ndarray]:
    tile = obs[:MAP_OBS_SIZE].reshape(OBS_ROWS, OBS_COLS, NUM_TILE_CHANNELS)
    scalar = obs[MAP_OBS_SIZE:]

    block = tile[:, :, :NUM_BLOCK_TYPES].argmax(axis=-1)
    item = tile[:, :, NUM_BLOCK_TYPES:NUM_BLOCK_TYPES + NUM_ITEM_TYPES].argmax(axis=-1)
    mob_raw = tile[:, :, NUM_BLOCK_TYPES + NUM_ITEM_TYPES:NUM_BLOCK_TYPES + NUM_ITEM_TYPES + NUM_MOB_CLASSES * NUM_MOB_TYPES]
    mobs = mob_raw.reshape(OBS_ROWS, OBS_COLS, NUM_MOB_CLASSES, NUM_MOB_TYPES).argmax(axis=-1)
    mob_present = mob_raw.reshape(OBS_ROWS, OBS_COLS, NUM_MOB_CLASSES, NUM_MOB_TYPES).max(axis=-1) > 0
    visible = tile[:, :, -1]

    return {
        "block": block,
        "item": item,
        "mobs": mobs,
        "mob_present": mob_present,
        "visible": visible,
        "scalar": scalar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="run1")
    parser.add_argument("--runs-dir", type=Path, default=CRAFTAX_RUNS_DIR)
    parser.add_argument("--dataset", type=Path, default=None, help="Optional cached .npz dataset")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "resources" / "craftax" / "people" / "run1_obs_action_pair.png",
    )
    args = parser.parse_args()

    if args.dataset is not None:
        data = load_obs_action_npz(args.dataset)
    else:
        run_path = args.runs_dir / f"{args.run}.pbz2"
        data = extract_obs_action_pairs(run_path, include_reward_done=True)

    obs = np.asarray(data["obs"][args.index], dtype=np.float32)
    action = int(data["actions"][args.index])
    reward = float(data["rewards"][args.index]) if "rewards" in data else None
    done = bool(data["dones"][args.index]) if "dones" in data else None
    decoded = decode_obs(obs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 1.2, 1.2, 1.8], height_ratios=[1.0, 1.0])
    ax_block = fig.add_subplot(gs[0, 0])
    ax_item = fig.add_subplot(gs[0, 1])
    ax_visible = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[:, 3])
    mob_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    ax_block.imshow(decoded["block"], interpolation="nearest", cmap="tab20")
    ax_block.set_title("Block IDs")
    ax_item.imshow(decoded["item"], interpolation="nearest", cmap="Accent")
    ax_item.set_title("Item IDs")
    ax_visible.imshow(decoded["visible"], interpolation="nearest", cmap="gray", vmin=0.0, vmax=1.0)
    ax_visible.set_title("Visibility")

    for mob_idx, ax in enumerate(mob_axes):
        mob_map = np.where(decoded["mob_present"][:, :, mob_idx], decoded["mobs"][:, :, mob_idx] + 1, 0)
        ax.imshow(mob_map, interpolation="nearest", cmap="viridis")
        ax.set_title(MOB_CLASS_NAMES[mob_idx])

    ax_text.axis("off")
    scalar = decoded["scalar"]
    scalar_lines = []
    for name, value in zip(SCALAR_NAMES, scalar.tolist()):
        if abs(value) > 1e-6:
            scalar_lines.append(f"{name}: {value:.3f}")
    title = f"action={action} ({ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else 'unknown'})"
    if reward is not None:
        title += f"  reward={reward:.3f}"
    if done is not None:
        title += f"  done={done}"
    fig.suptitle(title, fontsize=14)

    summary = [
        "Block legend:",
        *[f"{i}: {name}" for i, name in enumerate(BLOCK_NAMES[:12])],
        "...",
        "",
        "Items:",
        *[f"{i}: {name}" for i, name in enumerate(ITEM_NAMES)],
        "",
        "Nonzero scalars:",
        *scalar_lines[:30],
    ]
    ax_text.text(0.0, 1.0, "\n".join(summary), va="top", family="monospace", fontsize=9)

    for ax in [ax_block, ax_item, ax_visible, *mob_axes]:
        ax.set_xticks(range(OBS_COLS))
        ax.set_yticks(range(OBS_ROWS))
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
