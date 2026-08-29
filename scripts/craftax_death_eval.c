// Roll out craftax_clean_weights.bin, record the last 100 steps of one episode
// per seed, and diagnose where / why the agent died.
//
// Build + run via:  python scripts/craftax_death_eval.py

#define PUF_CRAFTAX_NET 1
#include "src/puffercpu.c"
#include "ocean/craftax_clean/craftax_clean.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define TAIL 100
#define DEFAULT_POOL 32
#define MAX_EVAL_SEEDS 32

static const char* LEVEL_NAMES[NUM_LEVELS] = {
    "overworld", "dungeon", "gnomish_mines", "sewers", "vault",
    "troll_mines", "fire_realm", "ice_realm", "graveyard",
};

static const char* PASSIVE_NAMES[NUM_MOB_TYPES] = {
    "cow", "bat", "snail", "passive3", "passive4", "passive5", "passive6", "passive7",
};
static const char* MELEE_NAMES[NUM_MOB_TYPES] = {
    "zombie", "gnome_warrior", "orc_soldier", "lizard",
    "knight", "troll", "pigman", "frost_troll",
};
static const char* RANGED_NAMES[NUM_MOB_TYPES] = {
    "skeleton", "gnome_archer", "orc_mage", "kobold",
    "archer", "deep_thing", "fire_elemental", "ice_elemental",
};
static const char* PROJ_NAMES[8] = {
    "arrow", "dagger", "fireball", "iceball",
    "arrow2", "slimeball", "fireball2", "iceball2",
};

typedef struct Nearby {
    const char* kind;
    const char* name;
    int type_id;
    int row;
    int col;
    int dist;
    int cooldown;
    float health;
} Nearby;

typedef struct Frame {
    int t;
    int action;
    int row;
    int col;
    int level;
    int dir;
    float health;
    int food;
    int drink;
    int energy;
    int mana;
    int xp;
    int strength;
    int dexterity;
    int intelligence;
    int sleeping;
    int resting;
    float light;
    int stand_block;
    int stand_item;
    int pickaxe;
    int sword;
    int bow;
    int arrows;
    int armour[4];
    int monsters_killed;
    int boss_progress;
    int n_near;
    Nearby near[16];
    unsigned char tiles[OBS_ROWS * OBS_COLS];
    unsigned char items[OBS_ROWS * OBS_COLS];
    unsigned char mobs[OBS_ROWS * OBS_COLS];
    float predicted_value;
} Frame;

typedef struct Episode {
    int seed;
    int length;
    int death_level;
    int death_row;
    int death_col;
    float score;
    float health_at_kill_step;
    int timeout;
    int n_tail;
    Frame tail[TAIL];
    int achievements[NUM_ACHIEVEMENTS];
    int potion_mapping[NUM_POTIONS];
    int max_floor;
} Episode;

static int manhattan(int r0, int c0, int r1, int c1) {
    int dr = r0 - r1;
    int dc = c0 - c1;
    if (dr < 0) {
        dr = -dr;
    }
    if (dc < 0) {
        dc = -dc;
    }
    return dr + dc;
}

static char tile_char(int block, int item, int mob, int is_player) {
    if (is_player) {
        return '@';
    }
    if (mob == 1) {
        return 'Z';
    }
    if (mob == 2) {
        return 'S';
    }
    if (mob == 3) {
        return '*';
    }
    if (mob == 4) {
        return 'o';
    }
    if (item == ITEM_LADDER_DOWN) {
        return 'v';
    }
    if (item == ITEM_LADDER_UP) {
        return '^';
    }
    if (item == ITEM_TORCH) {
        return 'i';
    }
    switch (block) {
    case BLOCK_GRASS:
    case BLOCK_PATH:
    case BLOCK_SAND:
    case BLOCK_FIRE_GRASS:
    case BLOCK_ICE_GRASS:
    case BLOCK_GRAVEL:
        return '.';
    case BLOCK_WATER:
    case BLOCK_FOUNTAIN:
        return '~';
    case BLOCK_LAVA:
        return '!';
    case BLOCK_STONE:
    case BLOCK_WALL:
    case BLOCK_WALL_MOSS:
    case BLOCK_STALAGMITE:
        return '#';
    case BLOCK_TREE:
    case BLOCK_FIRE_TREE:
    case BLOCK_ICE_SHRUB:
        return 'T';
    case BLOCK_COAL:
        return 'c';
    case BLOCK_IRON:
        return 'f';
    case BLOCK_DIAMOND:
        return 'd';
    case BLOCK_RUBY:
        return 'r';
    case BLOCK_SAPPHIRE:
        return 's';
    case BLOCK_CRAFTING_TABLE:
        return '=';
    case BLOCK_FURNACE:
        return 'F';
    case BLOCK_CHEST:
        return 'C';
    case BLOCK_PLANT:
        return 'p';
    case BLOCK_RIPE_PLANT:
        return 'P';
    case BLOCK_WOOD:
        return 'w';
    case BLOCK_NECROMANCER:
    case BLOCK_NECROMANCER_VULNERABLE:
        return 'N';
    case BLOCK_GRAVE:
    case BLOCK_GRAVE2:
    case BLOCK_GRAVE3:
        return 'g';
    case BLOCK_ENCHANTMENT_TABLE_FIRE:
    case BLOCK_ENCHANTMENT_TABLE_ICE:
        return 'E';
    case BLOCK_DARKNESS:
        return ' ';
    default:
        return '?';
    }
}

static const char* block_name(int block) {
    switch (block) {
    case BLOCK_GRASS: return "grass";
    case BLOCK_WATER: return "water";
    case BLOCK_STONE: return "stone";
    case BLOCK_TREE: return "tree";
    case BLOCK_WOOD: return "wood";
    case BLOCK_PATH: return "path";
    case BLOCK_COAL: return "coal";
    case BLOCK_IRON: return "iron";
    case BLOCK_DIAMOND: return "diamond";
    case BLOCK_CRAFTING_TABLE: return "table";
    case BLOCK_FURNACE: return "furnace";
    case BLOCK_SAND: return "sand";
    case BLOCK_LAVA: return "lava";
    case BLOCK_PLANT: return "plant";
    case BLOCK_RIPE_PLANT: return "ripe_plant";
    case BLOCK_WALL: return "wall";
    case BLOCK_WALL_MOSS: return "moss_wall";
    case BLOCK_STALAGMITE: return "stalagmite";
    case BLOCK_SAPPHIRE: return "sapphire";
    case BLOCK_RUBY: return "ruby";
    case BLOCK_CHEST: return "chest";
    case BLOCK_FOUNTAIN: return "fountain";
    case BLOCK_FIRE_GRASS: return "fire_grass";
    case BLOCK_ICE_GRASS: return "ice_grass";
    case BLOCK_GRAVEL: return "gravel";
    case BLOCK_FIRE_TREE: return "fire_tree";
    case BLOCK_ICE_SHRUB: return "ice_shrub";
    case BLOCK_NECROMANCER: return "necromancer";
    case BLOCK_GRAVE: return "grave";
    case BLOCK_NECROMANCER_VULNERABLE: return "necromancer_vulnerable";
    default: return "other";
    }
}

static void add_near(Frame* fr, const char* kind, const char* name, int type_id,
        int row, int col, int dist, int cooldown, float health) {
    if (fr->n_near >= 16) {
        return;
    }
    Nearby* n = &fr->near[fr->n_near++];
    n->kind = kind;
    n->name = name;
    n->type_id = type_id;
    n->row = row;
    n->col = col;
    n->dist = dist;
    n->cooldown = cooldown;
    n->health = health;
}

static void collect_mobs(Frame* fr, const State* s) {
    int level = s->player_level;
    int pr = s->player_position[0];
    int pc = s->player_position[1];
    const Mobs* melee = &s->melee_mobs[level];
    for (int i = 0; i < MAX_MELEE_MOBS; i++) {
        if (!melee->mask[i]) {
            continue;
        }
        int r = melee->position[i][0];
        int c = melee->position[i][1];
        int d = manhattan(pr, pc, r, c);
        int tid = clampi(melee->type_id[i], 0, NUM_MOB_TYPES - 1);
        add_near(fr, "melee", MELEE_NAMES[tid], tid, r, c, d,
            melee->attack_cooldown[i], melee->health[i]);
    }
    const Mobs* ranged = &s->ranged_mobs[level];
    for (int i = 0; i < MAX_RANGED_MOBS; i++) {
        if (!ranged->mask[i]) {
            continue;
        }
        int r = ranged->position[i][0];
        int c = ranged->position[i][1];
        int d = manhattan(pr, pc, r, c);
        int tid = clampi(ranged->type_id[i], 0, NUM_MOB_TYPES - 1);
        add_near(fr, "ranged", RANGED_NAMES[tid], tid, r, c, d,
            ranged->attack_cooldown[i], ranged->health[i]);
    }
    const Mobs* passive = &s->passive_mobs[level];
    for (int i = 0; i < MAX_PASSIVE_MOBS; i++) {
        if (!passive->mask[i]) {
            continue;
        }
        int r = passive->position[i][0];
        int c = passive->position[i][1];
        int d = manhattan(pr, pc, r, c);
        int tid = clampi(passive->type_id[i], 0, NUM_MOB_TYPES - 1);
        add_near(fr, "passive", PASSIVE_NAMES[tid], tid, r, c, d, 0, passive->health[i]);
    }
    const Mobs* proj = &s->mob_projectiles[level];
    for (int i = 0; i < MAX_MOB_PROJECTILES; i++) {
        if (!proj->mask[i]) {
            continue;
        }
        int r = proj->position[i][0];
        int c = proj->position[i][1];
        int nr = r + s->mob_projectile_directions[level][i][0];
        int nc = c + s->mob_projectile_directions[level][i][1];
        int d = manhattan(pr, pc, r, c);
        int tid = clampi(proj->type_id[i], 0, 7);
        add_near(fr, "projectile", PROJ_NAMES[tid], tid, r, c, d, 0, proj->health[i]);
        if (nr == pr && nc == pc) {
            add_near(fr, "projectile_incoming", PROJ_NAMES[tid], tid, nr, nc, 0, 0,
                proj->health[i]);
        }
    }
}

static void snapshot_frame(Frame* fr, const Craftax* env, int action) {
    const State* s = &env->state;
    memset(fr, 0, sizeof(*fr));
    fr->t = s->timestep;
    fr->action = action;
    fr->row = s->player_position[0];
    fr->col = s->player_position[1];
    fr->level = s->player_level;
    fr->dir = s->player_direction;
    fr->health = s->player_health;
    fr->food = s->player_food;
    fr->drink = s->player_drink;
    fr->energy = s->player_energy;
    fr->mana = s->player_mana;
    fr->xp = s->player_xp;
    fr->strength = s->player_strength;
    fr->dexterity = s->player_dexterity;
    fr->intelligence = s->player_intelligence;
    fr->sleeping = s->is_sleeping;
    fr->resting = s->is_resting;
    fr->light = s->light_level;
    fr->stand_block = s->map[fr->level][fr->row][fr->col];
    fr->stand_item = s->item_map[fr->level][fr->row][fr->col];
    fr->pickaxe = s->inventory.pickaxe;
    fr->sword = s->inventory.sword;
    fr->bow = s->inventory.bow;
    fr->arrows = s->inventory.arrows;
    memcpy(fr->armour, s->inventory.armour, sizeof(fr->armour));
    fr->predicted_value = env->predicted_value;
    fr->monsters_killed = s->monsters_killed[fr->level];
    fr->boss_progress = s->boss_progress;
    collect_mobs(fr, s);

    int rr = OBS_ROWS / 2;
    int rc = OBS_COLS / 2;
    for (int r = 0; r < OBS_ROWS; r++) {
        for (int c = 0; c < OBS_COLS; c++) {
            int wr = fr->row + (r - rr);
            int wc = fr->col + (c - rc);
            int idx = r * OBS_COLS + c;
            if (wr < 0 || wr >= MAP_SIZE || wc < 0 || wc >= MAP_SIZE) {
                fr->tiles[idx] = BLOCK_OUT_OF_BOUNDS;
                continue;
            }
            fr->tiles[idx] = s->map[fr->level][wr][wc];
            fr->items[idx] = s->item_map[fr->level][wr][wc];
        }
    }
    for (int i = 0; i < fr->n_near; i++) {
        int r = fr->near[i].row - fr->row + rr;
        int c = fr->near[i].col - fr->col + rc;
        if (r < 0 || r >= OBS_ROWS || c < 0 || c >= OBS_COLS) {
            continue;
        }
        int idx = r * OBS_COLS + c;
        if (strcmp(fr->near[i].kind, "melee") == 0) {
            fr->mobs[idx] = 1;
        } else if (strcmp(fr->near[i].kind, "ranged") == 0) {
            fr->mobs[idx] = 2;
        } else if (strcmp(fr->near[i].kind, "projectile") == 0
                || strcmp(fr->near[i].kind, "projectile_incoming") == 0) {
            fr->mobs[idx] = 3;
        } else if (strcmp(fr->near[i].kind, "passive") == 0) {
            fr->mobs[idx] = 4;
        }
    }
}

static void dump_local_map(FILE* fp, const Frame* fr) {
    int rr = OBS_ROWS / 2;
    int rc = OBS_COLS / 2;
    fprintf(fp, "    ");
    for (int c = 0; c < OBS_COLS; c++) {
        fputc(c == rc ? '|' : ' ', fp);
    }
    fputc('\n', fp);
    for (int r = 0; r < OBS_ROWS; r++) {
        fprintf(fp, "  %c ", r == rr ? '-' : ' ');
        for (int c = 0; c < OBS_COLS; c++) {
            int idx = r * OBS_COLS + c;
            fputc(tile_char(fr->tiles[idx], fr->items[idx], fr->mobs[idx],
                r == rr && c == rc), fp);
        }
        fprintf(fp, " %c\n", r == rr ? '-' : ' ');
    }
}

static int armour_sum(const Frame* fr) {
    return fr->armour[0] + fr->armour[1] + fr->armour[2] + fr->armour[3];
}

static void dump_frame(FILE* fp, const Frame* fr, int idx) {
    const char* aname = (fr->action >= 0 && fr->action < ATN_DIM)
        ? craftax_clean_action_names[fr->action] : "?";
    fprintf(fp,
        "-- step %d  t=%d  %s (%d,%d)  act=%d:%s  HP=%.1f food=%d drink=%d "
        "nrg=%d mana=%d  sleep=%d rest=%d  light=%.2f  stand=%s item=%d  "
        "gear=p%d/s%d/b%d/a%d  kills=%d\n",
        idx, fr->t, LEVEL_NAMES[clampi(fr->level, 0, NUM_LEVELS - 1)],
        fr->row, fr->col, fr->action, aname, fr->health, fr->food, fr->drink,
        fr->energy, fr->mana, fr->sleeping, fr->resting, fr->light,
        block_name(fr->stand_block), fr->stand_item,
        fr->pickaxe, fr->sword, fr->bow, armour_sum(fr), fr->monsters_killed);
    for (int i = 0; i < fr->n_near; i++) {
        const Nearby* n = &fr->near[i];
        fprintf(fp, "     %s %s tid=%d at (%d,%d) dist=%d cd=%d hp=%.1f\n",
            n->kind, n->name, n->type_id, n->row, n->col, n->dist, n->cooldown,
            n->health);
    }
}

static int melee_will_hit(const Frame* fr) {
    for (int i = 0; i < fr->n_near; i++) {
        if (strcmp(fr->near[i].kind, "melee") == 0
                && fr->near[i].dist == 1
                && fr->near[i].cooldown <= 0) {
            return i;
        }
    }
    return -1;
}

static int projectile_will_hit(const Frame* fr) {
    for (int i = 0; i < fr->n_near; i++) {
        if (strcmp(fr->near[i].kind, "projectile_incoming") == 0) {
            return i;
        }
        if (strcmp(fr->near[i].kind, "projectile") == 0 && fr->near[i].dist == 0) {
            return i;
        }
    }
    return -1;
}

static int last_potion_action(const Frame* fr) {
    return fr->action >= ACTION_DRINK_POTION_RED
        && fr->action <= ACTION_DRINK_POTION_YELLOW;
}

static void fill_death_cause(const Episode* ep, char* out, size_t out_n) {
    const Frame* last = &ep->tail[ep->n_tail - 1];
    int drop_i = -1;
    float drop_amt = 0.0f;
    int ladder_frames = 0;
    for (int i = 0; i < ep->n_tail; i++) {
        if (ep->tail[i].action == ACTION_ASCEND
                || ep->tail[i].action == ACTION_DESCEND) {
            ladder_frames++;
        }
        if (i + 1 < ep->n_tail) {
            float d = ep->tail[i].health - ep->tail[i + 1].health;
            if (d > drop_amt) {
                drop_amt = d;
                drop_i = i;
            }
        }
    }
    const Frame* crash = (drop_i >= 0) ? &ep->tail[drop_i] : last;
    int crash_melee = melee_will_hit(crash);
    int last_melee = melee_will_hit(last);
    int last_proj = projectile_will_hit(last);
    int potion = last_potion_action(last);
    int starving = last->food <= 0 || last->drink <= 0;
    int sleep_crash = drop_i >= 0 && crash->sleeping && drop_amt >= 3.0f;
    const char* floor = LEVEL_NAMES[clampi(last->level, 0, NUM_LEVELS - 1)];

    if (ep->timeout) {
        snprintf(out, out_n, "timeout at %d steps on %s", ep->length, floor);
    } else if (sleep_crash) {
        const char* killer = crash_melee >= 0 ? crash->near[crash_melee].name : "melee";
        if (last->level != crash->level) {
            snprintf(out, out_n,
                "slept on %s, %s 3.5x sleep hit (%.0f→%.0f HP), died on %s",
                LEVEL_NAMES[clampi(crash->level, 0, NUM_LEVELS - 1)],
                killer, crash->health, ep->tail[drop_i + 1].health, floor);
        } else {
            snprintf(out, out_n,
                "slept on %s and died to %s (3.5x sleep melee, %.0f→%.0f HP)",
                floor, killer, crash->health, ep->tail[drop_i + 1].health);
        }
    } else if (potion) {
        int slot = last->action - ACTION_DRINK_POTION_RED;
        int effect = ep->potion_mapping[slot];
        snprintf(out, out_n, "%s potion on %s (slot %d effect %d)",
            effect == 1 ? "poison" : "potion", floor, slot, effect);
    } else if (last_melee >= 0 && last->sleeping) {
        snprintf(out, out_n, "melee while sleeping: %s on %s",
            last->near[last_melee].name, floor);
    } else if (last_melee >= 0 && last->resting) {
        snprintf(out, out_n, "melee while resting: %s on %s",
            last->near[last_melee].name, floor);
    } else if (last_melee >= 0 && last_proj >= 0) {
        snprintf(out, out_n, "%s melee + %s projectile on %s",
            last->near[last_melee].name, last->near[last_proj].name, floor);
    } else if (last_melee >= 0) {
        snprintf(out, out_n, "%s melee on %s at (%d,%d)%s",
            last->near[last_melee].name, floor, last->row, last->col,
            starving ? "; drink/food empty" : "");
    } else if (last_proj >= 0) {
        snprintf(out, out_n, "%s projectile on %s", last->near[last_proj].name, floor);
    } else if (starving && last->health <= 1.5f) {
        snprintf(out, out_n, "dehydration/starvation on %s (food=%d drink=%d energy=%d)%s",
            floor, last->food, last->drink, last->energy,
            ladder_frames >= 8 ? "; ladder yo-yo" : "");
    } else if (last->energy <= 0 && last->health <= 1.5f) {
        snprintf(out, out_n, "exhaustion on %s (energy=0)", floor);
    } else {
        snprintf(out, out_n, "unclear death on %s (HP %.1f, no adjacent ready melee)",
            floor, last->health);
    }
}

static void write_diagnosis(FILE* fp, const Episode* ep) {
    const Frame* last = &ep->tail[ep->n_tail - 1];
    const char* floor = LEVEL_NAMES[clampi(last->level, 0, NUM_LEVELS - 1)];
    fprintf(fp, "seed %d  length %d  score %.2f  died on %s at (%d,%d)\n",
        ep->seed, ep->length, ep->score, floor, last->row, last->col);
    fprintf(fp, "HP entering death step: %.2f   food=%d drink=%d energy=%d  "
        "sleeping=%d resting=%d  light=%.2f\n",
        last->health, last->food, last->drink, last->energy,
        last->sleeping, last->resting, last->light);
    fprintf(fp, "gear: pickaxe=%d sword=%d bow=%d arrows=%d armour=%d/%d/%d/%d  "
        "str=%d dex=%d int=%d xp=%d  floor_kills=%d\n",
        last->pickaxe, last->sword, last->bow, last->arrows,
        last->armour[0], last->armour[1], last->armour[2], last->armour[3],
        last->strength, last->dexterity, last->intelligence, last->xp,
        last->monsters_killed);
    fprintf(fp, "standing on %s (item %d)  last action %s\n",
        block_name(last->stand_block), last->stand_item,
        last->action >= 0 && last->action < ATN_DIM
            ? craftax_clean_action_names[last->action] : "?");

    int drop_i = -1;
    float drop_amt = 0.0f;
    int sleep_frames = 0;
    int ladder_frames = 0;
    for (int i = 0; i < ep->n_tail; i++) {
        if (ep->tail[i].sleeping) {
            sleep_frames++;
        }
        if (ep->tail[i].action == ACTION_ASCEND
                || ep->tail[i].action == ACTION_DESCEND) {
            ladder_frames++;
        }
        if (i + 1 < ep->n_tail) {
            float d = ep->tail[i].health - ep->tail[i + 1].health;
            if (d > drop_amt) {
                drop_amt = d;
                drop_i = i;
            }
        }
    }
    const Frame* crash = (drop_i >= 0) ? &ep->tail[drop_i] : last;
    int crash_melee = melee_will_hit(crash);
    int last_melee = melee_will_hit(last);
    int last_proj = projectile_will_hit(last);
    int potion = last_potion_action(last);
    int starving = last->food <= 0 || last->drink <= 0;
    int exhausted = last->energy <= 0;
    int sleep_crash = drop_i >= 0 && crash->sleeping && drop_amt >= 3.0f;

    fprintf(fp, "\nprimary cause: ");
    if (ep->timeout) {
        fprintf(fp, "TIMEOUT at %d steps (not a health death)\n", ep->length);
    } else if (sleep_crash) {
        fprintf(fp, "SLEEP MELEE — HP dropped %.1f->%.1f while sleeping on %s "
            "(%s). Sleep multiplies melee damage 3.5x. ",
            crash->health, ep->tail[drop_i + 1].health,
            LEVEL_NAMES[clampi(crash->level, 0, NUM_LEVELS - 1)],
            crash_melee >= 0 ? crash->near[crash_melee].name : "nearby melee");
        if (last->level != crash->level) {
            fprintf(fp, "Then descended/ascended to %s at %.1f HP and died.\n",
                LEVEL_NAMES[clampi(last->level, 0, NUM_LEVELS - 1)], last->health);
        } else {
            fprintf(fp, "Finishing blow on the same floor.\n");
        }
    } else if (potion) {
        int slot = last->action - ACTION_DRINK_POTION_RED;
        int effect = ep->potion_mapping[slot];
        if (effect == 1) {
            fprintf(fp, "POISON POTION (color slot %d maps to damage)\n", slot);
        } else {
            fprintf(fp, "POTION (slot %d effect %d) on the death step, "
                "but that may not be the killing blow\n", slot, effect);
        }
    } else if (last_melee >= 0 && last->sleeping) {
        fprintf(fp, "MELEE WHILE SLEEPING — %s at dist 1, sleep multiplies "
            "melee damage by 3.5x\n", last->near[last_melee].name);
    } else if (last_melee >= 0 && last->resting) {
        fprintf(fp, "MELEE WHILE RESTING — %s at dist 1 woke and hit the agent\n",
            last->near[last_melee].name);
    } else if (last_melee >= 0 && last_proj >= 0) {
        fprintf(fp, "MELEE + PROJECTILE — %s in melee range and %s incoming\n",
            last->near[last_melee].name, last->near[last_proj].name);
    } else if (last_melee >= 0) {
        fprintf(fp, "MELEE — adjacent %s (cd=%d hp=%.1f) at (%d,%d)\n",
            last->near[last_melee].name, last->near[last_melee].cooldown,
            last->near[last_melee].health, last->near[last_melee].row,
            last->near[last_melee].col);
    } else if (last_proj >= 0) {
        fprintf(fp, "PROJECTILE — %s hitting player tile\n",
            last->near[last_proj].name);
    } else if (starving && last->health <= 1.5f) {
        fprintf(fp, "STARVATION/DEHYDRATION — food=%d drink=%d energy=%d; recover "
            "ticks drain HP when food/drink/energy are empty\n",
            last->food, last->drink, last->energy);
    } else if (exhausted && last->health <= 1.5f) {
        fprintf(fp, "EXHAUSTION — energy=%d with low HP\n", last->energy);
    } else {
        fprintf(fp, "UNCLEAR from last-step snapshot (HP was %.2f; no adjacent "
            "ready melee or incoming projectile)\n", last->health);
    }
    if (drop_i >= 0 && drop_amt >= 2.0f) {
        fprintf(fp, "largest HP drop in tail: %.1f at t=%d on %s (sleep=%d rest=%d "
            "act=%s)\n",
            drop_amt, crash->t,
            LEVEL_NAMES[clampi(crash->level, 0, NUM_LEVELS - 1)],
            crash->sleeping, crash->resting,
            crash->action >= 0 && crash->action < ATN_DIM
                ? craftax_clean_action_names[crash->action] : "?");
    }
    if (ladder_frames >= 8) {
        fprintf(fp, "behavior: ladder oscillation — %d/%d tail actions were "
            "ASCEND/DESCEND (stuck on a floor transition)\n",
            ladder_frames, ep->n_tail);
    }
    if (sleep_frames > 0) {
        fprintf(fp, "behavior: sleeping for %d/%d tail frames\n",
            sleep_frames, ep->n_tail);
    }

    if (!ep->timeout) {
        if (starving) {
            fprintf(fp, "contributing: necessities empty (food=%d drink=%d) so HP "
                "was not regenerating and may have been ticking down\n",
                last->food, last->drink);
        }
        if (last->sleeping && last_melee < 0) {
            fprintf(fp, "contributing: was sleeping (incoming hits deal 3.5x)\n");
        }
        if (last->level == 0 && last->light < 0.4f) {
            fprintf(fp, "contributing: overworld night (melee spawn rate up)\n");
        }
        if (armour_sum(last) == 0 && last->level >= 1) {
            fprintf(fp, "contributing: no armour on floor %d\n", last->level);
        }
    }

    fprintf(fp, "\nnearby at death step:\n");
    if (last->n_near == 0) {
        fprintf(fp, "  (none)\n");
    }
    for (int i = 0; i < last->n_near; i++) {
        const Nearby* n = &last->near[i];
        fprintf(fp, "  %s %s dist=%d cd=%d hp=%.1f at (%d,%d)\n",
            n->kind, n->name, n->dist, n->cooldown, n->health, n->row, n->col);
    }

    fprintf(fp, "\nlocal map (9x11, @=player Z=melee S=ranged *=proj o=passive "
        "v/^ = ladders !=lava):\n");
    dump_local_map(fp, last);

    fprintf(fp, "\nHP over last %d steps: ", ep->n_tail);
    for (int i = 0; i < ep->n_tail; i++) {
        fprintf(fp, "%.0f%s", ep->tail[i].health, i + 1 == ep->n_tail ? "\n" : ",");
    }

    int unlocked = 0;
    fprintf(fp, "achievements:");
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        if (ep->achievements[i]) {
            fprintf(fp, " %s;", craftax_clean_ach_names[i]);
            unlocked++;
        }
    }
    fprintf(fp, "  (%d / %d)  max_floor=%d\n", unlocked, NUM_ACHIEVEMENTS, ep->max_floor);
}

static void write_episode(const char* out_dir, const Episode* ep) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/seed_%d_tail.txt", out_dir, ep->seed);
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "failed to write %s: %s\n", path, strerror(errno));
        exit(1);
    }
    write_diagnosis(fp, ep);
    fprintf(fp, "\n===== last %d steps (oldest to newest; state is PRE-step) =====\n",
        ep->n_tail);
    for (int i = 0; i < ep->n_tail; i++) {
        dump_frame(fp, &ep->tail[i], i);
        if (i >= ep->n_tail - 5) {
            dump_local_map(fp, &ep->tail[i]);
        }
    }
    fclose(fp);
    printf("wrote %s\n", path);
}

#define VIDEO_FPS 10
#define VIDEO_FULL_FPS 30
#define VIDEO_HOLD_LAST 20

typedef struct {
    int pipefd[2];
    pid_t pid;
} VideoRecorder;

static int open_video(VideoRecorder* rec, const char* filename, int w, int h,
        int fps) {
    if (pipe(rec->pipefd) == -1) {
        fprintf(stderr, "pipe failed: %s\n", strerror(errno));
        return 0;
    }
    rec->pid = fork();
    if (rec->pid == -1) {
        fprintf(stderr, "fork failed: %s\n", strerror(errno));
        return 0;
    }
    if (rec->pid == 0) {
        close(rec->pipefd[1]);
        dup2(rec->pipefd[0], STDIN_FILENO);
        close(rec->pipefd[0]);
        for (int fd = 3; fd < 256; fd++) {
            close(fd);
        }
        char sz[32];
        snprintf(sz, sizeof(sz), "%dx%d", w, h);
        char fps_s[8];
        snprintf(fps_s, sizeof(fps_s), "%d", fps);
        execlp("ffmpeg", "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", sz, "-r", fps_s, "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "20",
            "-loglevel", "error",
            filename, NULL);
        fprintf(stderr, "exec ffmpeg failed (is ffmpeg on PATH?)\n");
        _exit(1);
    }
    close(rec->pipefd[0]);
    return 1;
}

static void write_video_frame(VideoRecorder* rec, const Image* img) {
    size_t n = (size_t)img->width * (size_t)img->height * 4;
    const unsigned char* p = (const unsigned char*)img->data;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(rec->pipefd[1], p + off, n - off);
        if (w <= 0) {
            break;
        }
        off += (size_t)w;
    }
}

static void close_video(VideoRecorder* rec) {
    close(rec->pipefd[1]);
    waitpid(rec->pid, NULL, 0);
}

static void write_tail_video(Craftax* env, const Episode* ep, const State* states,
        const char* path) {
    if (ep->n_tail <= 0) {
        return;
    }
    State saved_state = env->state;
    float saved_action = env->agents[0].actions[0];
    int saved_max_floor = env->max_floor_accum;
    int saved_len = env->episode_length_accum;
    float saved_ret = env->episode_return_accum;

    float saved_value = env->predicted_value;
    env->state = states[0];
    env->agents[0].actions[0] = (float)ep->tail[0].action;
    env->predicted_value = ep->tail[0].predicted_value;
    env->max_floor_accum = ep->max_floor;
    env->episode_length_accum = ep->tail[0].t;
    env->episode_return_accum = ep->score;
    puf_render(env);
    SetTargetFPS(0);
    Image first = LoadImageFromScreen();
    int w = first.width;
    int h = first.height;
    if (w <= 0 || h <= 0) {
        fprintf(stderr, "screen capture failed for %s\n", path);
        UnloadImage(first);
        return;
    }
    VideoRecorder rec;
    if (!open_video(&rec, path, w, h, VIDEO_FPS)) {
        UnloadImage(first);
        return;
    }
    write_video_frame(&rec, &first);
    UnloadImage(first);

    for (int i = 1; i < ep->n_tail; i++) {
        env->state = states[i];
        env->agents[0].actions[0] = (float)ep->tail[i].action;
        env->predicted_value = ep->tail[i].predicted_value;
        env->max_floor_accum = ep->max_floor;
        env->episode_length_accum = ep->tail[i].t;
        puf_render(env);
        Image img = LoadImageFromScreen();
        write_video_frame(&rec, &img);
        UnloadImage(img);
    }
    for (int i = 0; i < VIDEO_HOLD_LAST; i++) {
        puf_render(env);
        Image img = LoadImageFromScreen();
        write_video_frame(&rec, &img);
        UnloadImage(img);
    }
    close_video(&rec);
    printf("wrote %s\n", path);

    env->state = saved_state;
    env->agents[0].actions[0] = saved_action;
    env->predicted_value = saved_value;
    env->max_floor_accum = saved_max_floor;
    env->episode_length_accum = saved_len;
    env->episode_return_accum = saved_ret;
}

static State* make_reset_pool(int n) {
    State* pool = (State*)calloc((size_t)n, sizeof(State));
    for (int i = 0; i < n; i++) {
        Rng init_key = rng_seed((uint32_t)i);
        Rng discard;
        Rng reset_key;
        rng_split(init_key, &discard, &reset_key);
        Rng unused;
        Rng world_key;
        rng_split(reset_key, &unused, &world_key);
        generate_world_from_key(&pool[i], world_key);
    }
    return pool;
}

static void zero_gru(CraftaxNet* net) {
    int B = net->mingru->batch_size;
    int H = net->mingru->hidden_size;
    int L = net->mingru->num_layers;
    memset(net->mingru->state, 0, (size_t)L * B * H * sizeof(float));
}

static void reset_episode(Craftax* env, CraftaxNet* net, State* pool, int pool_n,
        int seed) {
    env->rng = (unsigned int)seed;
    env->seed = (uint64_t)seed;
    env->reset_pool = pool;
    env->reset_pool_size = pool_n;
    memset(&env->log, 0, sizeof(env->log));
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    env->max_floor_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    zero_gru(net);
    puf_reset(env);
}

static void run_seed(Craftax* env, CraftaxNet* net, State* pool, int pool_n,
        int seed, int greedy, const char* out_dir, int record_full, Episode* ep) {
    memset(ep, 0, sizeof(*ep));
    ep->seed = seed;
    net->greedy = greedy;
    reset_episode(env, net, pool, pool_n, seed);

    Frame ring[TAIL];
    State* state_ring = (State*)calloc(TAIL, sizeof(State));
    int ring_i = 0;
    int ring_n = 0;
    int steps = 0;
    float terminals[1] = {0};
    memcpy(ep->potion_mapping, env->state.potion_mapping, sizeof(ep->potion_mapping));

    VideoRecorder rec;
    int rec_open = 0;
    Image last_img = {0};
    char full_path[1024];
    snprintf(full_path, sizeof(full_path), "%s/seed_%d_full.mp4", out_dir, seed);

    while (1) {
        if ((steps % 2000) == 0) {
            fprintf(stderr, "  seed %d step %d hp=%.1f floor=%d pos=(%d,%d)\n",
                seed, steps, env->state.player_health, env->state.player_level,
                env->state.player_position[0], env->state.player_position[1]);
        }
        int action_i = 0;
        Frame* slot = &ring[ring_i];
        forward_craftax(net, env->agents[0].observations, terminals,
            env->agents[0].actions, env->agents[0].action_mask);
        env->predicted_value = craftax_value(net, 0);
        action_i = (int)env->agents[0].actions[0];
        snapshot_frame(slot, env, action_i);
        state_ring[ring_i] = env->state;
        memcpy(ep->achievements, env->state.achievements, sizeof(ep->achievements));
        if (env->max_floor_accum > ep->max_floor) {
            ep->max_floor = env->max_floor_accum;
        }
        if (record_full) {
            puf_render(env);
            if (steps == 0) {
                SetTargetFPS(0);
            }
            if (last_img.data) {
                UnloadImage(last_img);
            }
            last_img = LoadImageFromScreen();
            if (!rec_open) {
                rec_open = open_video(&rec, full_path, last_img.width, last_img.height,
                    VIDEO_FULL_FPS);
            }
            if (rec_open) {
                write_video_frame(&rec, &last_img);
            }
        }
        ring_i = (ring_i + 1) % TAIL;
        if (ring_n < TAIL) {
            ring_n++;
        }
        puf_step(env);
        steps++;
        terminals[0] = env->agents[0].terminals[0];
        if (terminals[0] > 0.5f) {
            break;
        }
        if (steps >= DEFAULT_MAX_TIMESTEPS) {
            break;
        }
    }
    if (rec_open) {
        for (int i = 0; i < VIDEO_HOLD_LAST && last_img.data; i++) {
            write_video_frame(&rec, &last_img);
        }
        close_video(&rec);
        printf("wrote %s\n", full_path);
    }
    if (last_img.data) {
        UnloadImage(last_img);
    }

    ep->length = steps;
    ep->score = env->log.episode_return;
    ep->n_tail = ring_n;
    int start = (ring_i - ring_n + TAIL) % TAIL;
    State* ordered = (State*)calloc((size_t)ring_n, sizeof(State));
    for (int i = 0; i < ring_n; i++) {
        int idx = (start + i) % TAIL;
        ep->tail[i] = ring[idx];
        ordered[i] = state_ring[idx];
    }
    const Frame* last = &ep->tail[ep->n_tail - 1];
    ep->death_level = last->level;
    ep->death_row = last->row;
    ep->death_col = last->col;
    ep->health_at_kill_step = last->health;
    ep->timeout = last->t >= DEFAULT_MAX_TIMESTEPS - 1;

    if (!record_full) {
        char vid_path[1024];
        snprintf(vid_path, sizeof(vid_path), "%s/seed_%d_tail.mp4", out_dir, seed);
        write_tail_video(env, ep, ordered, vid_path);
    }
    free(state_ring);
    free(ordered);
}

int main(int argc, char** argv) {
    const char* weights_path = "resources/craftax_clean/craftax_clean_weights.bin";
    const char* out_dir = "logs/craftax_clean/death_eval";
    int seeds[MAX_EVAL_SEEDS];
    int n_seeds = 0;
    int greedy = 1;
    int record_full = 0;
    int argi = 1;
    while (argi < argc) {
        if (strcmp(argv[argi], "--sample") == 0) {
            greedy = 0;
            argi++;
        } else if (strcmp(argv[argi], "--full") == 0) {
            record_full = 1;
            argi++;
        } else {
            break;
        }
    }
    if (argi < argc) {
        weights_path = argv[argi++];
    }
    if (argi < argc) {
        out_dir = argv[argi++];
    }
    while (argi < argc && n_seeds < MAX_EVAL_SEEDS) {
        seeds[n_seeds++] = atoi(argv[argi++]);
    }
    if (n_seeds == 0) {
        n_seeds = 10;
        for (int i = 0; i < n_seeds; i++) {
            seeds[i] = i;
        }
    }

    Weights* weights = load_weights(weights_path);
    if (!weights) {
        fprintf(stderr, "failed to load %s\n", weights_path);
        return 1;
    }
    int hidden = 1024;
    int layers = 3;
    int need = craftax_weight_count(hidden, layers);
    int file_floats = weights->size - 7;
    fprintf(stderr, "weights %s file_floats=%d need=%d greedy=%d\n",
        weights_path, file_floats, need, greedy);
    if (need - file_floats > 7 || file_floats > need) {
        fprintf(stderr, "weight count mismatch (file=%d need=%d)\n",
            file_floats, need);
        return 1;
    }
    CraftaxNet* net = init_craftax_net(weights, 1, hidden, layers);

    Craftax env;
    memset(&env, 0, sizeof(env));
    obs_t* observations = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
    float* actions = (float*)calloc(1, sizeof(float));
    float* rewards = (float*)calloc(1, sizeof(float));
    float* terminals = (float*)calloc(1, sizeof(float));
    unsigned char* masks = (unsigned char*)calloc(ATN_DIM, 1);
    memset(masks, 1, ATN_DIM);
    fprintf(stderr, "generating reset pool (%d worlds)...\n", DEFAULT_POOL);
    State* pool = make_reset_pool(DEFAULT_POOL);

    mkdir("logs", 0755);
    mkdir("logs/craftax_clean", 0755);
    mkdir(out_dir, 0755);

    Dict kwargs = {0};
    dict_set(&kwargs, "action_mask", 1);
    dict_set(&kwargs, "seed_offset", 0);
    puf_init(&env, &kwargs);
    env.agents[0].observations = observations;
    env.agents[0].actions = actions;
    env.agents[0].rewards = rewards;
    env.agents[0].terminals = terminals;
    env.agents[0].action_mask = masks;
    env.use_action_mask = 1;

    Episode* eps = (Episode*)calloc((size_t)n_seeds, sizeof(Episode));
    for (int i = 0; i < n_seeds; i++) {
        fprintf(stderr, "=== seed %d (%d/%d) ===\n", seeds[i], i + 1, n_seeds);
        srand((unsigned)(1000 + seeds[i]));
        run_seed(&env, net, pool, DEFAULT_POOL, seeds[i], greedy, out_dir,
            record_full, &eps[i]);
        write_episode(out_dir, &eps[i]);
    }

    char sum_path[1024];
    snprintf(sum_path, sizeof(sum_path), "%s/summary.txt", out_dir);
    FILE* fp = fopen(sum_path, "w");
    if (!fp) {
        fprintf(stderr, "failed to write %s\n", sum_path);
        return 1;
    }
    fprintf(fp, "craftax_clean death eval  weights=%s  greedy=%d  pool=%d  n=%d\n\n",
        weights_path, greedy, DEFAULT_POOL, n_seeds);
    fprintf(fp, "seed\tscore\tlength\tfloor\tdeath\n");
    printf("\n======== DEATH TABLE ========\n");
    printf("%-6s %-8s %-8s %-16s %s\n", "seed", "score", "length", "floor", "death");
    for (int i = 0; i < n_seeds; i++) {
        char cause[320];
        fill_death_cause(&eps[i], cause, sizeof(cause));
        const char* floor = LEVEL_NAMES[clampi(eps[i].death_level, 0, NUM_LEVELS - 1)];
        fprintf(fp, "%d\t%.2f\t%d\t%s\t%s\n",
            eps[i].seed, eps[i].score, eps[i].length, floor, cause);
        printf("%-6d %-8.2f %-8d %-16s %s\n",
            eps[i].seed, eps[i].score, eps[i].length, floor, cause);
    }
    fprintf(fp, "\n");
    for (int i = 0; i < n_seeds; i++) {
        write_diagnosis(fp, &eps[i]);
        fprintf(fp, "\n----------------------------------------\n\n");
    }
    fclose(fp);
    printf("wrote %s\n", sum_path);

    printf("\n======== DEATH EVAL DETAILS ========\n");
    for (int i = 0; i < n_seeds; i++) {
        write_diagnosis(stdout, &eps[i]);
        printf("\n");
    }
    if (record_full) {
        printf("full playthrough videos:\n");
        for (int i = 0; i < n_seeds; i++) {
            printf("  %s/seed_%d_full.mp4\n", out_dir, seeds[i]);
        }
    } else {
        printf("videos (last 100 steps + hold):\n");
        for (int i = 0; i < n_seeds; i++) {
            printf("  %s/seed_%d_tail.mp4\n", out_dir, seeds[i]);
        }
    }
    return 0;
}
