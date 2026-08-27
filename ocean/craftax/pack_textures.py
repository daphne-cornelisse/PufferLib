"""Pack Craftax upstream 16x16 PNG assets into a single shared textures.bin.

Consumed by ocean/craftax, ocean/craftax_classic, and ocean/craftax_clean.
All files live in craftax's asset dir; the classic PNGs that overlap are
byte-identical to the full ones.

Layout: contiguous 16*16*4 RGBA tiles. Order must match the
CRAFTAX_TEX_* / CC_TEX_* / TEX_* enums in the env headers.

  [0..36]  block textures (37) -- BlockType; first 17 entries also valid for classic
  [37..41] player: down, up, left, right, sleep
  [42..46] items: none(blank), torch, ladder_down, ladder_up, ladder_down_blocked
  [47..49] mobs: zombie, skeleton, cow
  [50..53] arrows: down, up, left, right
  [54..61] armour: iron then diamond, each helmet/chest/pants/boots
  [62..65] pickaxes: wood, stone, iron, diamond
  [66..69] swords: wood, stone, iron, diamond
  [70]     bow
  [71..76] potions: red, green, blue, pink, cyan, yellow
  [77..79] HUD-only: sapling, torch_in_inventory, book
  [80..87] melee types: zombie, gnome_warrior, orc_soldier, lizard, knight, troll, pigman, frost_troll
  [88..90] passive types: cow, bat, snail
  [91..98] ranged types: skeleton, gnome_archer, orc_mage, kobold, knight_archer, deep_thing, fire_elemental, ice_elemental
  [99..102] projectiles: dagger, fireball, iceball, slimeball
  [103..104] sword enchant overlays: fire, ice
  [105..106] arrow enchant overlays: fire, ice
  [107..110] armour fire overlays: helmet, chest, pants, boots
  [111..114] armour ice overlays: helmet, chest, pants, boots
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_BIN = ROOT / "resources" / "craftax" / "textures.bin"


def find_assets() -> Path:
    env = os.environ.get("CRAFTAX_ASSETS")
    if env:
        p = Path(env)
        if (p / "iron_helmet.png").exists():
            return p
    try:
        import craftax
        pkg = Path(craftax.__file__).resolve().parent
        for cand in (pkg / "craftax" / "assets", pkg / "assets"):
            if (cand / "iron_helmet.png").exists():
                return cand
    except ImportError:
        pass
    candidates = [
        ROOT / ".venv/lib/python3.12/site-packages/craftax/craftax/assets",
        ROOT / ".venv/lib/python3.10/site-packages/craftax/craftax/assets",
        Path.home() / ".cache/uv/archive-v0/A1lWMW1niJVdk7dF0fJmT/craftax/craftax/assets",
        Path.home() / "github/multitask_preplay/.venv/lib/python3.10/site-packages/craftax/craftax/assets",
    ]
    for cand in candidates:
        if (cand / "iron_helmet.png").exists():
            return cand
    raise FileNotFoundError(
        "craftax assets not found (need iron_helmet.png). "
        "Set CRAFTAX_ASSETS or install the craftax package."
    )


ASSETS = find_assets()

TILE = 16

BLOCK_FILES = [
    "debug_tile.png",            # 0 INVALID
    "debug_tile.png",            # 1 OUT_OF_BOUNDS (overwritten solid grey below)
    "grass.png",                 # 2
    "water.png",                 # 3
    "stone.png",                 # 4
    "tree.png",                  # 5
    "wood.png",                  # 6
    "path.png",                  # 7
    "coal.png",                  # 8
    "iron.png",                  # 9
    "diamond.png",               # 10
    "table.png",                 # 11 crafting table
    "furnace.png",               # 12
    "sand.png",                  # 13
    "lava.png",                  # 14
    "plant_on_grass.png",        # 15
    "ripe_plant_on_grass.png",   # 16
    "wall2.png",                 # 17
    "debug_tile.png",            # 18 DARKNESS (overwritten solid black below)
    "wall_moss.png",             # 19
    "stalagmite.png",            # 20
    "sapphire.png",              # 21
    "ruby.png",                  # 22
    "chest.png",                 # 23
    "fountain.png",              # 24
    "fire_grass.png",            # 25
    "ice_grass.png",             # 26
    "gravel.png",                # 27
    "fire_tree.png",             # 28
    "ice_shrub.png",             # 29
    "enchantment_table_fire.png",# 30
    "enchantment_table_ice.png", # 31
    "necromancer.png",           # 32
    "grave.png",                 # 33
    "grave2.png",                # 34
    "grave3.png",                # 35
    "necromancer_vulnerable.png",# 36
]

PLAYER_FILES = [
    "player-down.png",
    "player-up.png",
    "player-left.png",
    "player-right.png",
    "player-sleep.png",
]

ITEM_FILES = [
    None,                        # NONE -> fully transparent
    "torch_on_path.png",
    "ladder_down.png",
    "ladder_up.png",
    "ladder_down_blocked.png",
]

MOB_FILES = [
    "zombie.png",
    "skeleton.png",
    "cow.png",
]

ARROW_FILES = [
    "arrow-down.png",
    "arrow-up.png",
    "arrow-left.png",
    "arrow-right.png",
]

# Official Craftax HUD icons (constants.py armour_textures[level, slot]).
# Empty (level 0) is a blank tile drawn by the renderer, not packed.
ARMOUR_FILES = [
    "iron_helmet.png",
    "iron_chestplate.png",
    "iron_pants.png",
    "iron_boots.png",
    "diamond_helmet.png",
    "diamond_chestplate.png",
    "diamond_pants.png",
    "diamond_boots.png",
]

# Official Craftax HUD tool icons (constants.py pickaxe/sword/bow textures).
# Empty (level 0) is a blank slot drawn by the renderer, not packed.
WEAPON_FILES = [
    "wood_pickaxe.png",
    "stone_pickaxe.png",
    "iron_pickaxe.png",
    "diamond_pickaxe.png",
    "wood_sword.png",
    "stone_sword.png",
    "iron_sword.png",
    "diamond_sword.png",
    "bow.png",
]

# Official Craftax HUD potion icons (constants.py potion_textures).
POTION_FILES = [
    "potion_red.png",
    "potion_green.png",
    "potion_blue.png",
    "potion_pink.png",
    "potion_cyan.png",
    "potion_yellow.png",
]

# Official Craftax HUD item icons that differ from the world tiles
# (constants.py sapling_texture / torch_inv_texture / book_texture).
HUD_ITEM_FILES = [
    "sapling.png",
    "torch_in_inventory.png",
    "book.png",
]

# Official Craftax per-type mob sprites (constants.py load_mob_texture_set).
# Indexed by type_id; 47-49 remain the 3 generic sprites for classic/full Craftax.
MELEE_TYPE_FILES = [
    "zombie.png",
    "gnome_warrior.png",
    "orc_soldier.png",
    "lizard.png",
    "knight.png",
    "troll.png",
    "pigman.png",
    "frost_troll.png",
]
PASSIVE_TYPE_FILES = [
    "cow.png",
    "bat.png",
    "snail.png",
]
RANGED_TYPE_FILES = [
    "skeleton.png",
    "gnome_archer.png",
    "orc_mage.png",
    "kobold.png",
    "knight_archer.png",
    "deep_thing.png",
    "fire_elemental.png",
    "ice_elemental.png",
]
PROJECTILE_TYPE_FILES = [
    "dagger.png",
    "fireball.png",
    "iceball.png",
    "slimeball.png",
]
# HUD enchant overlays. Level 0 is empty (no tile); fire then ice.
ENCHANT_FILES = [
    "sword_fire_enchantment.png",
    "sword_ice_enchantment.png",
    "arrow_fire_enchantment.png",
    "arrow_ice_enchantment.png",
    "helmet_fire_enchantment.png",
    "chestplate_fire_enchantment.png",
    "pants_fire_enchantment.png",
    "boots_fire_enchantment.png",
    "helmet_ice_enchantment.png",
    "chestplate_ice_enchantment.png",
    "pants_ice_enchantment.png",
    "boots_ice_enchantment.png",
]


def load_tile(name: str | None) -> np.ndarray:
    if name is None:
        return np.zeros((TILE, TILE, 4), dtype=np.uint8)
    p = ASSETS / name
    img = Image.open(p).convert("RGBA").resize((TILE, TILE), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    print(f"craftax assets: {ASSETS}")
    tiles: list[np.ndarray] = []
    for f in BLOCK_FILES:
        tiles.append(load_tile(f))

    # manual overrides to match upstream renderer
    tiles[1] = np.full((TILE, TILE, 4), 128, dtype=np.uint8)
    tiles[1][..., 3] = 255  # out of bounds
    tiles[18] = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    tiles[18][..., 3] = 255  # darkness

    for f in PLAYER_FILES:
        tiles.append(load_tile(f))

    # torch_in_walls doesn't exist in assets; fall back to torch.png if needed
    for f in ITEM_FILES:
        if f is not None and not (ASSETS / f).exists():
            alt = "torch.png" if "torch" in f else f
            tiles.append(load_tile(alt))
        else:
            tiles.append(load_tile(f))

    for f in (
        MOB_FILES + ARROW_FILES + ARMOUR_FILES + WEAPON_FILES
        + POTION_FILES + HUD_ITEM_FILES
        + MELEE_TYPE_FILES + PASSIVE_TYPE_FILES + RANGED_TYPE_FILES
        + PROJECTILE_TYPE_FILES + ENCHANT_FILES
    ):
        tiles.append(load_tile(f))

    blob = np.stack(tiles, axis=0)  # (N, 16, 16, 4) uint8
    assert blob.dtype == np.uint8
    OUT_BIN.write_bytes(blob.tobytes(order="C"))
    print(f"wrote {OUT_BIN} — {blob.shape[0]} tiles, {OUT_BIN.stat().st_size} bytes")


if __name__ == "__main__":
    main()
