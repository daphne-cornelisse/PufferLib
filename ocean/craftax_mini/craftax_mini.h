// Goal-conditioned Craftax variant used for Multitask Preplay human-data parity.
//
// This env intentionally reuses the full Craftax state/worldgen implementation
// and only changes the experiment-facing surface: 5 actions, easier reset
// settings, passable non-precious stones, sparse goal reward, and goal terminal.

#pragma once

#define CRAFTAX_ENABLE_ENV_IMPL

// Include the full implementation under private public-entrypoint names so this
// sibling env can provide its own c_init/c_reset/c_step while still reusing the
// full renderer and low-level helpers.
#define c_init craftax_full_c_init
#define c_reset craftax_full_c_reset
#define c_step_native craftax_full_c_step_native
#define c_step_gameplay craftax_full_c_step_gameplay
#define c_step_encode craftax_full_c_step_encode
#define c_step craftax_full_c_step
#define c_close craftax_full_c_close
#define c_render craftax_full_c_render
#include "../craftax/craftax.h"
#include "../craftax/step_crafting.h"
#include "../craftax/step_update_mobs.h"
#include "../craftax/step_spawn_mobs.h"
#include "human_maps.h"
#undef c_init
#undef c_reset
#undef c_step_native
#undef c_step_gameplay
#undef c_step_encode
#undef c_step
#undef c_close
#undef c_render

#define CRAFTAX_MINI_NUM_ACTIONS 5
#define CRAFTAX_MINI_DEFAULT_MAX_TIMESTEPS 200

static int32_t g_craftax_mini_current_goal_block = CRAFTAX_BLOCK_DIAMOND;
static int32_t g_craftax_mini_max_timesteps = CRAFTAX_MINI_DEFAULT_MAX_TIMESTEPS;
static bool g_craftax_mini_use_human_maps = true;

static inline int32_t craftax_mini_action_to_full(int32_t action) {
    switch (action) {
    case 0: return CRAFTAX_ACTION_RIGHT;
    case 1: return CRAFTAX_ACTION_DOWN;
    case 2: return CRAFTAX_ACTION_LEFT;
    case 3: return CRAFTAX_ACTION_UP;
    case 4: return CRAFTAX_ACTION_DO;
    default: return CRAFTAX_ACTION_DO;
    }
}

static inline bool craftax_mini_is_solid_block(int32_t block) {
    // Matches the web experiment simplification: non-precious stones are
    // passable, while goal stones and interactive objects remain blocking.
    return block != CRAFTAX_BLOCK_STONE && craftax_step_is_solid_block(block);
}

static inline bool craftax_mini_valid_land_position(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    bool pos_in_bounds = row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
    int32_t level = craftax_step_jax_index(state->player_level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    int32_t block = state->map[level][map_row][map_col];
    bool in_solid_block = craftax_mini_is_solid_block(block);
    bool in_mob = craftax_step_is_in_mob(state, row, col);
    bool in_lava = block == CRAFTAX_BLOCK_LAVA;
    bool in_water = block == CRAFTAX_BLOCK_WATER;

    bool valid_move = pos_in_bounds && !in_mob && !in_solid_block;
    valid_move = valid_move && !in_water;
    valid_move = valid_move && !in_lava;
    return valid_move;
}

static inline void craftax_mini_move_player_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t direction[2];
    craftax_step_direction(action, direction);

    int32_t proposed_row = state->player_position[0] + direction[0];
    int32_t proposed_col = state->player_position[1] + direction[1];
    bool valid_move = craftax_mini_valid_land_position(
        state,
        proposed_row,
        proposed_col
    );

    state->player_position[0] += (int32_t)valid_move * direction[0];
    state->player_position[1] += (int32_t)valid_move * direction[1];

    bool is_new_direction = direction[0] != 0 || direction[1] != 0;
    state->player_direction = state->player_direction * (1 - (int32_t)is_new_direction)
        + action * (int32_t)is_new_direction;
}

static inline int32_t craftax_mini_inventory_count_for_goal(
    const CraftaxState* state,
    int32_t goal_block
) {
    switch (goal_block) {
    case CRAFTAX_BLOCK_DIAMOND: return state->inventory.diamond;
    case CRAFTAX_BLOCK_SAPPHIRE: return state->inventory.sapphire;
    case CRAFTAX_BLOCK_RUBY: return state->inventory.ruby;
    default: return 0;
    }
}

static inline float craftax_mini_gameplay_step_native(
    Craftax* env,
    int32_t mini_action,
    CraftaxThreefryKey rng
) {
    CraftaxState* state = env->state;
    int32_t action = craftax_mini_action_to_full(mini_action);
    int32_t initial_goal_count = craftax_mini_inventory_count_for_goal(
        state,
        g_craftax_mini_current_goal_block
    );

    action = state->is_sleeping ? CRAFTAX_ACTION_NOOP : action;
    action = state->is_resting ? CRAFTAX_ACTION_NOOP : action;

    craftax_change_floor_native(state, action);
    craftax_do_crafting_native(state, action);

    CraftaxThreefryKey subkey = craftax_step_native_next_key(&rng);
    craftax_do_action_native(state, action, subkey);

    craftax_place_block_native(state, action);
    craftax_shoot_projectile_native(state, action);
    craftax_cast_spell_native(state, action);
    craftax_drink_potion_native(state, action);

    subkey = craftax_step_native_next_key(&rng);
    craftax_read_book_native(state, subkey.word, action);

    subkey = craftax_step_native_next_key(&rng);
    craftax_enchant_native(state, action, subkey);

    craftax_boss_logic_native(state);
    craftax_level_up_attributes_native(state, action, CRAFTAX_MAX_ATTRIBUTE);
    craftax_mini_move_player_native(state, action);

    subkey = craftax_step_native_next_key(&rng);
    craftax_update_mobs_native(state, subkey);

    subkey = craftax_step_native_next_key(&rng);
    craftax_spawn_mobs_native(state, subkey);

    craftax_update_plants_native(state);
    craftax_update_player_intrinsics_native(state, action);
    craftax_clip_inventory_and_intrinsics_native(state, false);
    craftax_calculate_inventory_achievements_native(state);

    subkey = craftax_step_native_next_key(&rng);
    state->timestep += 1;
    state->light_level = craftax_calculate_light_level_native(state->timestep);
    state->state_rng[0] = subkey.word[0];
    state->state_rng[1] = subkey.word[1];

    int32_t current_goal_count = craftax_mini_inventory_count_for_goal(
        state,
        g_craftax_mini_current_goal_block
    );
    return current_goal_count > initial_goal_count ? 1.0f : 0.0f;
}

static inline bool craftax_mini_is_game_over_native(const Craftax* env) {
    const CraftaxState* state = env->state;
    bool reached_goal = craftax_mini_inventory_count_for_goal(
        state,
        g_craftax_mini_current_goal_block
    ) > 0;
    int32_t max_timesteps = g_craftax_mini_max_timesteps > 0
        ? g_craftax_mini_max_timesteps
        : CRAFTAX_MINI_DEFAULT_MAX_TIMESTEPS;
    return reached_goal
        || state->timestep >= max_timesteps
        || state->player_health <= 0.0f;
}

static inline void craftax_mini_apply_reset_settings(Craftax* env) {
    CraftaxState* state = env->state;
    state->inventory.pickaxe = 5;
    state->player_strength = 20;
    state->player_health = (float)craftax_step_get_max_health(state);
}

static inline bool craftax_mini_apply_human_map_for_seed(
    CraftaxState* state,
    int32_t seed
) {
    const uint8_t* map = craftax_mini_human_map_for_seed(seed);
    if (map == NULL) {
        return false;
    }

    memcpy(
        state->map,
        map,
        CRAFTAX_NUM_LEVELS * CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE * sizeof(uint8_t)
    );
    craftax_refresh_spawn_bits_all(state);
    return true;
}

static inline void craftax_mini_reset_state_from_seed(Craftax* env) {
    craftax_reset_state_from_seed(env);
    if (!g_craftax_mini_use_human_maps) {
        return;
    }

    int32_t seed = (int32_t)env->seed;
    if (!craftax_mini_apply_human_map_for_seed(env->state, seed)) {
        seed = craftax_mini_human_seed_for_index((int32_t)env->rng);
        craftax_mini_apply_human_map_for_seed(env->state, seed);
    }
}

static void c_init(Craftax* env) {
    env->client = NULL;
    env->num_agents = 1;
    craftax_ensure_state_storage(env);
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(env->log));
    craftax_wg_init_cell_templates();
    craftax_mini_reset_state_from_seed(env);
    craftax_mini_apply_reset_settings(env);
}

static void c_reset(Craftax* env) {
    if (env->rewards != NULL) env->rewards[0] = 0.0f;
    if (env->terminals != NULL) env->terminals[0] = 0.0f;
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));

    craftax_mini_reset_state_from_seed(env);
    craftax_mini_apply_reset_settings(env);
    craftax_encode_native_observation(env->state, env->observations);
}

static void c_step_gameplay(Craftax* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) action = 0;
    if (action >= CRAFTAX_MINI_NUM_ACTIONS) action = CRAFTAX_MINI_NUM_ACTIONS - 1;

    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);
    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);

    float reward = craftax_mini_gameplay_step_native(env, action, step_rng);
    bool done = craftax_mini_is_game_over_native(env);
    craftax_copy_achievements_to_env(env, env->state);

    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        craftax_reset_state_on_done(env->state, reset_key);
        if (g_craftax_mini_use_human_maps) {
            int32_t seed = craftax_mini_human_seed_for_index((int32_t)env->rng);
            craftax_mini_apply_human_map_for_seed(env->state, seed);
        }
        craftax_mini_apply_reset_settings(env);
    }
}

static void c_step_encode(Craftax* env) {
    craftax_encode_native_observation(env->state, env->observations);
}

static void c_step(Craftax* env) {
    c_step_gameplay(env);
    c_step_encode(env);
}

static void c_close(Craftax* env) {
    craftax_full_c_close(env);
}

static void c_render(Craftax* env) {
    craftax_full_c_render(env);
}
