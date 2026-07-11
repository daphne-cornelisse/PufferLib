// Full native Craftax environment for PufferLib Ocean.

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "constants.h"
#include "worldgen.h"
#include <stdio.h>
#include <stdlib.h>

// ============================================================
// State layout declarations matching craftax_state.py field order
// ============================================================
typedef struct CraftaxInventory {
    int32_t wood;
    int32_t stone;
    int32_t coal;
    int32_t iron;
    int32_t diamond;
    int32_t sapling;
    int32_t pickaxe;
    int32_t sword;
    int32_t bow;
    int32_t arrows;
    int32_t armour[4];
    int32_t torches;
    int32_t ruby;
    int32_t sapphire;
    int32_t potions[6];
    int32_t books;
} CraftaxInventory;

typedef struct CraftaxMobs3 {
    int32_t position[CRAFTAX_NUM_LEVELS][3][2];
    float health[CRAFTAX_NUM_LEVELS][3];
    bool mask[CRAFTAX_NUM_LEVELS][3];
    int32_t attack_cooldown[CRAFTAX_NUM_LEVELS][3];
    int32_t type_id[CRAFTAX_NUM_LEVELS][3];
} CraftaxMobs3;

typedef struct CraftaxMobs2 {
    int32_t position[CRAFTAX_NUM_LEVELS][2][2];
    float health[CRAFTAX_NUM_LEVELS][2];
    bool mask[CRAFTAX_NUM_LEVELS][2];
    int32_t attack_cooldown[CRAFTAX_NUM_LEVELS][2];
    int32_t type_id[CRAFTAX_NUM_LEVELS][2];
} CraftaxMobs2;

typedef struct CraftaxState {
    // === Hot data (accessed every step) ===
    int32_t player_position[2];
    int32_t player_level;
    int32_t player_direction;

    float player_health;
    int32_t player_food;
    int32_t player_drink;
    int32_t player_energy;
    int32_t player_mana;
    bool is_sleeping;
    bool is_resting;

    float player_recover;
    float player_hunger;
    float player_thirst;
    float player_fatigue;
    float player_recover_mana;

    int32_t player_xp;
    int32_t player_dexterity;
    int32_t player_strength;
    int32_t player_intelligence;

    CraftaxInventory inventory;

    CraftaxMobs3 melee_mobs;
    CraftaxMobs3 passive_mobs;
    CraftaxMobs2 ranged_mobs;

    CraftaxMobs3 mob_projectiles;
    int32_t mob_projectile_directions[CRAFTAX_NUM_LEVELS][CRAFTAX_MAX_MOB_PROJECTILES][2];
    CraftaxMobs3 player_projectiles;
    int32_t player_projectile_directions[CRAFTAX_NUM_LEVELS][CRAFTAX_MAX_PLAYER_PROJECTILES][2];

    int32_t growing_plants_positions[CRAFTAX_MAX_GROWING_PLANTS][2];
    int32_t growing_plants_age[CRAFTAX_MAX_GROWING_PLANTS];
    bool growing_plants_mask[CRAFTAX_MAX_GROWING_PLANTS];

    int32_t potion_mapping[6];
    bool learned_spells[2];

    int32_t sword_enchantment;
    int32_t bow_enchantment;
    int32_t armour_enchantments[4];

    int32_t boss_progress;
    int32_t boss_timesteps_to_spawn_this_round;

    float light_level;
    bool achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    uint32_t state_rng[2];
    int32_t timestep;
    int32_t fractal_noise_angles[4];

    // === Medium-hot bitmaps, read during mob updates, spawn scans, encode_obs ===
    uint64_t mob_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_all_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_grave_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_water_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];

    // === Cold data (large maps, scattered access) ===
    uint8_t map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    uint8_t item_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    uint8_t light_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];

    int32_t down_ladders[CRAFTAX_NUM_LEVELS][2];
    int32_t up_ladders[CRAFTAX_NUM_LEVELS][2];
    bool chests_opened[CRAFTAX_NUM_LEVELS];
    int32_t monsters_killed[CRAFTAX_NUM_LEVELS];
} CraftaxState;

typedef char CraftaxStateMatchesWorldState[
    (sizeof(CraftaxState) == sizeof(CraftaxWorldState)) ? 1 : -1
];

static inline uint64_t craftax_spawn_all_bit(uint8_t block) {
    return (uint64_t)(
        block == CRAFTAX_BLOCK_GRASS
        || block == CRAFTAX_BLOCK_PATH
        || block == CRAFTAX_BLOCK_FIRE_GRASS
        || block == CRAFTAX_BLOCK_ICE_GRASS
    );
}

static inline uint64_t craftax_spawn_grave_bit(uint8_t block) {
    return (uint64_t)(
        block == CRAFTAX_BLOCK_GRAVE
        || block == CRAFTAX_BLOCK_GRAVE2
        || block == CRAFTAX_BLOCK_GRAVE3
    );
}

static inline uint64_t craftax_spawn_water_bit(uint8_t block) {
    return (uint64_t)(block == CRAFTAX_BLOCK_WATER);
}

static inline void craftax_refresh_spawn_bits_cell(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    uint64_t bit = 1ULL << col;
    uint8_t block = state->map[level][row][col];

    state->spawn_all_bits[level][row] =
        (state->spawn_all_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_all_bit(block)) & bit);
    state->spawn_grave_bits[level][row] =
        (state->spawn_grave_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_grave_bit(block)) & bit);
    state->spawn_water_bits[level][row] =
        (state->spawn_water_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_water_bit(block)) & bit);
}

static inline void craftax_set_map_block(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    int32_t block
) {
    state->map[level][row][col] = (uint8_t)block;
    craftax_refresh_spawn_bits_cell(state, level, row, col);
}

static inline void craftax_refresh_spawn_bits_all(CraftaxState* state) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        for (int32_t row = 0; row < CRAFTAX_MAP_SIZE; row++) {
            uint64_t all_bits = 0;
            uint64_t grave_bits = 0;
            uint64_t water_bits = 0;
            for (int32_t col = 0; col < CRAFTAX_MAP_SIZE; col++) {
                uint8_t block = state->map[level][row][col];
                uint64_t bit = 1ULL << col;
                all_bits |= (0ULL - craftax_spawn_all_bit(block)) & bit;
                grave_bits |= (0ULL - craftax_spawn_grave_bit(block)) & bit;
                water_bits |= (0ULL - craftax_spawn_water_bit(block)) & bit;
            }
            state->spawn_all_bits[level][row] = all_bits;
            state->spawn_grave_bits[level][row] = grave_bits;
            state->spawn_water_bits[level][row] = water_bits;
        }
    }
}

typedef struct CraftaxArena {
    CraftaxState* states;
} CraftaxArena;

#ifdef CRAFTAX_ENABLE_ENV_IMPL
static inline void craftax_change_floor(CraftaxState* state, int32_t action);
static inline void craftax_do_crafting(CraftaxState* state, int32_t action);
static inline void craftax_do_action(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
);
static inline void craftax_place_block(CraftaxState* state, int32_t action);
static inline void craftax_shoot_projectile(
    CraftaxState* state,
    int32_t action
);
static inline void craftax_cast_spell(CraftaxState* state, int32_t action);
static inline void craftax_drink_potion(CraftaxState* state, int32_t action);
static inline void craftax_read_book(
    CraftaxState* state,
    const uint32_t rng_words[2],
    int32_t action
);
static inline void craftax_enchant(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
);
static inline void craftax_boss_logic(CraftaxState* state);
static inline void craftax_level_up_attributes(
    CraftaxState* state,
    int32_t action,
    int32_t max_attribute
);
static inline void craftax_move_player(
    CraftaxState* state,
    int32_t action
);
static inline void craftax_update_mobs(
    CraftaxState* state,
    CraftaxThreefryKey rng
);
static inline void craftax_spawn_mobs(
    CraftaxState* state,
    CraftaxThreefryKey rng
);
static inline void craftax_update_plants(CraftaxState* state);
static inline void craftax_update_player_intrinsics(
    CraftaxState* state,
    int32_t action
);
static inline void craftax_clip_inventory_and_intrinsics(
    CraftaxState* state
);
static inline void craftax_calculate_inventory_achievements(
    CraftaxState* state
);
#endif

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float n;
} Log;

typedef struct Client Client;

typedef struct Craftax {
    Client* client;
    Log log;

    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    unsigned int rng;
    uint64_t seed;
    CraftaxThreefryKey rng_key;
    CraftaxArena* arena;
    CraftaxState* state;
    bool owns_state_storage;
    int32_t render_view_mode;

    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float episode_return_accum;
    int32_t episode_length_accum;
} Craftax;

#ifdef CRAFTAX_ENABLE_RENDERING
#include "render.h"
#endif

#ifdef CRAFTAX_ENABLE_ENV_IMPL

static inline CraftaxThreefryKey craftax_next_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey subkey;
    craftax_threefry_split(*rng, rng, &subkey);
    return subkey;
}

static inline void craftax_copy_world_state_to_state(
    CraftaxState* dst,
    const CraftaxWorldState* src
) {
    memcpy(dst, src, sizeof(*dst));
}

static inline void craftax_generate_state_from_world_key(
    CraftaxThreefryKey world_key,
    CraftaxState* out
) {
    CraftaxWorldState world_state;
    craftax_generate_world_from_key(world_key, &world_state);
    craftax_copy_world_state_to_state(out, &world_state);
    craftax_refresh_spawn_bits_all(out);
}

static inline void craftax_reset_state_from_reset_key(
    CraftaxState* out,
    CraftaxThreefryKey reset_key
) {
    CraftaxThreefryKey unused;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &unused, &world_key);
    craftax_generate_state_from_world_key(world_key, out);
}

// ============================================================
// Reset pool: pre-generate N worlds once, then memcpy on reset.
// Trades world diversity (at most pool_size unique maps per process) for much
// faster resets. Set pool_size=0 to regenerate from the seed every time.
// ============================================================
static int g_craftax_reset_pool_size = 0;
static CraftaxState* g_craftax_reset_pool = NULL;
static int g_craftax_reset_pool_ready = 0;
static float g_craftax_initial_health = 9.0f;

static inline void craftax_set_initial_health(float initial_health) {
    g_craftax_initial_health = initial_health > 0.0f ? initial_health : 9.0f;
}

static inline void craftax_apply_reset_settings(CraftaxState* state) {
    state->player_health = g_craftax_initial_health;
}

// Called from my_init which runs single-threaded during env creation
// (vecenv.h iterates envs sequentially). First caller populates the
// pool; subsequent callers are no-ops.
static inline void craftax_set_reset_pool_size(int n) {
    if (g_craftax_reset_pool_ready) return;
    g_craftax_reset_pool_size = n;
    if (n > 0) {
        g_craftax_reset_pool = (CraftaxState*)calloc((size_t)n, sizeof(CraftaxState));
        for (int i = 0; i < n; i++) {
            CraftaxThreefryKey init_key = craftax_prng_key((uint32_t)i);
            CraftaxThreefryKey discard, reset_key;
            craftax_threefry_split(init_key, &discard, &reset_key);
            craftax_reset_state_from_reset_key(&g_craftax_reset_pool[i], reset_key);
        }
    }
    g_craftax_reset_pool_ready = 1;
}

static inline void craftax_ensure_state_storage(Craftax* env) {
    if (env->state != NULL) {
        return;
    }

    CraftaxArena* arena = (CraftaxArena*)calloc(1, sizeof(CraftaxArena));
    arena->states = (CraftaxState*)calloc(1, sizeof(CraftaxState));
    env->arena = arena;
    env->state = arena->states;
    env->owns_state_storage = true;
}

static inline void craftax_reset_state_from_seed(Craftax* env) {
    craftax_ensure_state_storage(env);
    CraftaxThreefryKey initial_key = craftax_prng_key((uint32_t)env->seed);
    if (g_craftax_reset_pool_size > 0) {
        CraftaxThreefryKey discard;
        craftax_threefry_split(initial_key, &env->rng_key, &discard);
        int idx = (int)(env->seed % (uint64_t)g_craftax_reset_pool_size);
        memcpy(env->state, &g_craftax_reset_pool[idx], sizeof(CraftaxState));
        craftax_apply_reset_settings(env->state);
        return;
    }
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(initial_key, &env->rng_key, &reset_key);
    craftax_reset_state_from_reset_key(env->state, reset_key);
    craftax_apply_reset_settings(env->state);
}

// Hot-path reset used by c_step on episode-done. Consults the reset pool
// when enabled, falls through to generate_world otherwise. Pool index is
// derived from the reset_key so different done events pick different
// pooled worlds. The direct craftax_reset_state_from_reset_key stays
// pool-free so direct callers always regenerate from the provided reset key.
static inline void craftax_reset_state_on_done(
    CraftaxState* out,
    CraftaxThreefryKey reset_key
) {
    if (g_craftax_reset_pool_size > 0) {
        uint32_t idx = reset_key.word[0] % (uint32_t)g_craftax_reset_pool_size;
        memcpy(out, &g_craftax_reset_pool[idx], sizeof(CraftaxState));
        craftax_apply_reset_settings(out);
        return;
    }
    craftax_reset_state_from_reset_key(out, reset_key);
    craftax_apply_reset_settings(out);
}

static inline void craftax_encode_observation(
    const CraftaxState* state,
    float* obs
) {
    if (obs == NULL) {
        return;
    }
    craftax_encode_reset_observation((const CraftaxWorldState*)(const void*)state, obs);
}

static inline float craftax_calculate_light_level(int32_t timestep) {
    float progress = fmodf(
        (float)timestep / (float)CRAFTAX_DAY_LENGTH,
        1.0f
    ) + 0.3f;
    float c = cosf(CRAFTAX_WG_PI * progress);
    return 1.0f - powf(fabsf(c), 3.0f);
}

static inline bool craftax_is_game_over(const CraftaxState* state) {
    return state->timestep >= CRAFTAX_DEFAULT_MAX_TIMESTEPS
        || state->player_health <= 0.0f;
}

static inline void craftax_copy_achievements_to_env(
    Craftax* env,
    const CraftaxState* state
) {
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        env->achievements[i] = state->achievements[i] ? 1.0f : 0.0f;
    }
}

static void add_log(Craftax* env) {
    int unlocked = 0;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        if (env->achievements[i] > 0.5f) {
            unlocked++;
            env->log.achievements[i] += 1.0f;
        }
    }
    env->log.perf += env->episode_return_accum / CRAFTAX_MAX_EPISODE_RETURN;
    env->log.score += env->episode_return_accum;
    env->log.episode_return += env->episode_return_accum;
    env->log.episode_length += (float)env->episode_length_accum;
    env->log.n += 1.0f;
}

static float craftax_gameplay_step(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    bool init_achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    memcpy(init_achievements, state->achievements, sizeof(init_achievements));

    action = state->is_sleeping ? CRAFTAX_ACTION_NOOP : action;
    action = state->is_resting ? CRAFTAX_ACTION_NOOP : action;

    craftax_change_floor(state, action);
    craftax_do_crafting(state, action);

    CraftaxThreefryKey subkey = craftax_next_key(&rng);
    craftax_do_action(state, action, subkey);

    craftax_place_block(state, action);
    craftax_shoot_projectile(state, action);
    craftax_cast_spell(state, action);
    craftax_drink_potion(state, action);

    subkey = craftax_next_key(&rng);
    craftax_read_book(state, subkey.word, action);

    subkey = craftax_next_key(&rng);
    craftax_enchant(state, action, subkey);

    craftax_boss_logic(state);
    craftax_level_up_attributes(state, action, CRAFTAX_MAX_ATTRIBUTE);
    craftax_move_player(state, action);

    subkey = craftax_next_key(&rng);
    craftax_update_mobs(state, subkey);

    subkey = craftax_next_key(&rng);
    craftax_spawn_mobs(state, subkey);

    craftax_update_plants(state);
    craftax_update_player_intrinsics(state, action);
    craftax_clip_inventory_and_intrinsics(state);
    craftax_calculate_inventory_achievements(state);

    float reward = 0.0f;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        int32_t delta = (int32_t)state->achievements[i] - (int32_t)init_achievements[i];
        reward += (float)delta * CRAFTAX_ACHIEVEMENT_REWARD_MAP[i];
    }

    subkey = craftax_next_key(&rng);
    state->timestep += 1;
    state->light_level = craftax_calculate_light_level(state->timestep);
    state->state_rng[0] = subkey.word[0];
    state->state_rng[1] = subkey.word[1];

    return reward;
}

// ============================================================
// Public API expected by vecenv.h
// ============================================================
static void c_init(Craftax* env) {
    env->client = NULL;
    env->num_agents = 1;
    craftax_ensure_state_storage(env);
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(env->log));
    craftax_reset_state_from_seed(env);
#ifdef CRAFTAX_ENABLE_RENDERING
    craftax_reset_player_trail(env);
#endif
}

static void c_reset(Craftax* env) {
    if (env->rewards != NULL) {
        env->rewards[0] = 0.0f;
    }
    if (env->terminals != NULL) {
        env->terminals[0] = 0.0f;
    }
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));

    craftax_reset_state_from_seed(env);
#ifdef CRAFTAX_ENABLE_RENDERING
    craftax_reset_player_trail(env);
#endif
    craftax_encode_observation(env->state, env->observations);
}

static void c_step_gameplay(Craftax* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) action = CRAFTAX_ACTION_NOOP;
    if (action >= CRAFTAX_NUM_ACTIONS) action = CRAFTAX_NUM_ACTIONS - 1;

    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);
    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);

    float reward = craftax_gameplay_step(env->state, action, step_rng);
    bool done = craftax_is_game_over(env->state);
    craftax_copy_achievements_to_env(env, env->state);

    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;
#ifdef CRAFTAX_ENABLE_RENDERING
    craftax_record_player_position(env);
#endif

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        craftax_reset_state_on_done(env->state, reset_key);
#ifdef CRAFTAX_ENABLE_RENDERING
        craftax_reset_player_trail(env);
#endif
    }
}

static void c_step(Craftax* env) {
    c_step_gameplay(env);
    craftax_encode_observation(env->state, env->observations);
}

static void c_close(Craftax* env) {
#ifdef CRAFTAX_ENABLE_RENDERING
    craftax_close_client(env);
#endif
    if (!env->owns_state_storage || env->arena == NULL) {
        return;
    }
    free(env->arena->states);
    free(env->arena);
    env->arena = NULL;
    env->state = NULL;
    env->owns_state_storage = false;
}

#ifdef CRAFTAX_ENABLE_RENDERING
static void c_render_mode(Craftax* env, int view_mode) {
    craftax_set_view_mode(env, view_mode);
    craftax_render_full(env);
}

static void c_render(Craftax* env) {
    craftax_render_full(env);
}
#else
static void c_render_mode(Craftax* env, int view_mode) {
    (void)env;
    (void)view_mode;
}

static void c_render(Craftax* env) {
    (void)env;
}
#endif

#endif
