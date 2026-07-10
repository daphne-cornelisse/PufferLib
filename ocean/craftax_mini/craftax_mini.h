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
#define c_step_gameplay craftax_full_c_step_gameplay
#define c_step_encode craftax_full_c_step_encode
#define c_step craftax_full_c_step
#define c_close craftax_full_c_close
#define c_render craftax_full_c_render
#include "../craftax/craftax.h"
#include "../craftax/step_crafting.h"
#include "../craftax/step_mobs.h"
#include "../craftax/step_spawning.h"
#include "human_maps.h"
#undef c_init
#undef c_reset
#undef c_step_gameplay
#undef c_step_encode
#undef c_step
#undef c_close
#undef c_render

#define CRAFTAX_MINI_NUM_ACTIONS 5
#define CRAFTAX_MINI_DEFAULT_MAX_TIMESTEPS 200

// Positive values select a fixed goal block. Non-positive values cycle the
// three human-task goals across env slots.
static int32_t g_craftax_mini_config_goal_block = CRAFTAX_BLOCK_DIAMOND;
static int32_t g_craftax_mini_max_timesteps = CRAFTAX_MINI_DEFAULT_MAX_TIMESTEPS;
static bool g_craftax_mini_use_human_maps = true;

static inline int32_t craftax_mini_goal_block_for_slot(uint32_t slot) {
    static const int32_t goal_blocks[3] = {
        CRAFTAX_BLOCK_DIAMOND,
        CRAFTAX_BLOCK_SAPPHIRE,
        CRAFTAX_BLOCK_RUBY,
    };
    return goal_blocks[slot % 3u];
}

static inline int32_t craftax_mini_current_goal_block(const Craftax* env) {
    if (g_craftax_mini_config_goal_block > 0) {
        return g_craftax_mini_config_goal_block;
    }
    return craftax_mini_goal_block_for_slot(env->rng);
}

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
    int32_t level = craftax_clamp_index(state->player_level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_clamp_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_clamp_index(col, CRAFTAX_MAP_SIZE);
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

static inline void craftax_mini_move_player(
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

static inline const char* craftax_mini_goal_name(int32_t goal_block) {
    switch (goal_block) {
    case CRAFTAX_BLOCK_DIAMOND: return "diamond";
    case CRAFTAX_BLOCK_SAPPHIRE: return "sapphire";
    case CRAFTAX_BLOCK_RUBY: return "ruby";
    default: return "unknown";
    }
}

static inline Color craftax_mini_goal_color(int32_t goal_block) {
    switch (goal_block) {
    case CRAFTAX_BLOCK_DIAMOND: return (Color){180, 255, 255, 255};
    case CRAFTAX_BLOCK_SAPPHIRE: return (Color){120, 180, 255, 255};
    case CRAFTAX_BLOCK_RUBY: return (Color){255, 120, 120, 255};
    default: return WHITE;
    }
}

static inline float craftax_mini_gameplay_step(
    Craftax* env,
    int32_t mini_action,
    CraftaxThreefryKey rng
) {
    CraftaxState* state = env->state;
    int32_t action = craftax_mini_action_to_full(mini_action);
    int32_t goal_block = craftax_mini_current_goal_block(env);
    int32_t initial_goal_count = craftax_mini_inventory_count_for_goal(
        state,
        goal_block
    );

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
    craftax_mini_move_player(state, action);

    subkey = craftax_next_key(&rng);
    craftax_update_mobs(state, subkey);

    subkey = craftax_next_key(&rng);
    craftax_spawn_mobs(state, subkey);

    craftax_update_plants(state);
    craftax_update_player_intrinsics(state, action);
    craftax_clip_inventory_and_intrinsics(state);
    craftax_calculate_inventory_achievements(state);

    subkey = craftax_next_key(&rng);
    state->timestep += 1;
    state->light_level = craftax_calculate_light_level(state->timestep);
    state->state_rng[0] = subkey.word[0];
    state->state_rng[1] = subkey.word[1];

    int32_t current_goal_count = craftax_mini_inventory_count_for_goal(
        state,
        goal_block
    );
    return current_goal_count > initial_goal_count ? 1.0f : 0.0f;
}

static inline bool craftax_mini_is_game_over(const Craftax* env) {
    const CraftaxState* state = env->state;
    int32_t goal_block = craftax_mini_current_goal_block(env);
    bool reached_goal = craftax_mini_inventory_count_for_goal(
        state,
        goal_block
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
    craftax_encode_observation(env->state, env->observations);
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

    float reward = craftax_mini_gameplay_step(env, action, step_rng);
    bool done = craftax_mini_is_game_over(env);
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
    craftax_encode_observation(env->state, env->observations);
}

static void c_step(Craftax* env) {
    c_step_gameplay(env);
    c_step_encode(env);
}

static void c_close(Craftax* env) {
    craftax_full_c_close(env);
}

static void c_render(Craftax* env) {
    const int view_w = CRAFTAX_RENDER_COLS * CRAFTAX_TEX_DRAW_PX;
    const int view_h = CRAFTAX_RENDER_ROWS * CRAFTAX_TEX_DRAW_PX;
    const int hud_h = 136;
    const int hud_font = 18;
    const int hud_title_font = 24;
    const int hud_line = 24;
    const int hud_pad = 6;

    if (!IsWindowReady()) {
        if (!craftax_ensure_render_display(env)) {
            fprintf(stderr, "WARNING: failed to initialize display for Craftax rendering\n");
            return;
        }
        Client* client = craftax_get_client(env);
        if (client != NULL && client->headless_display) {
            SetConfigFlags(FLAG_WINDOW_HIDDEN);
            SetTargetFPS(6000);
        } else {
            SetTargetFPS(30);
        }
        InitWindow(view_w, view_h + hud_h, "PufferLib Craftax");
    }
    if (!craftax_textures_loaded) craftax_load_textures();
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    CraftaxState* s = env->state;
    int lvl = s->player_level;
    int pr = s->player_position[0];
    int pc = s->player_position[1];
    int half_r = CRAFTAX_RENDER_ROWS / 2;
    int half_c = CRAFTAX_RENDER_COLS / 2;

    BeginDrawing();
    ClearBackground(BLACK);

    for (int vr = 0; vr < CRAFTAX_RENDER_ROWS; vr++) {
        for (int vc = 0; vc < CRAFTAX_RENDER_COLS; vc++) {
            int wr = pr - half_r + vr;
            int wc = pc - half_c + vc;
            int dst_x = vc * CRAFTAX_TEX_DRAW_PX;
            int dst_y = vr * CRAFTAX_TEX_DRAW_PX;

            int blk = CRAFTAX_BLOCK_OUT_OF_BOUNDS;
            if (wr >= 0 && wr < CRAFTAX_MAP_SIZE && wc >= 0 && wc < CRAFTAX_MAP_SIZE) {
                blk = s->map[lvl][wr][wc];
                if (s->light_map[lvl][wr][wc] <= 12) blk = CRAFTAX_BLOCK_DARKNESS;
            }
            if (blk < 0 || blk >= CRAFTAX_NUM_BLOCK_TYPES) blk = 0;
            craftax_draw_tile(blk, dst_x, dst_y, 1.0f);

            if (wr >= 0 && wr < CRAFTAX_MAP_SIZE && wc >= 0 && wc < CRAFTAX_MAP_SIZE) {
                int it = s->item_map[lvl][wr][wc];
                if (it > 0 && it < 5) {
                    craftax_draw_tile(CRAFTAX_TEX_ITEM_BASE + it, dst_x, dst_y, 1.0f);
                }
            }
        }
    }

    int pid = craftax_player_tex_id(s->player_direction, s->is_sleeping);
    craftax_draw_tile(pid, half_c * CRAFTAX_TEX_DRAW_PX, half_r * CRAFTAX_TEX_DRAW_PX, 1.0f);

    if (s->light_level < 1.0f) {
        unsigned char a = (unsigned char)((1.0f - s->light_level) * 140.0f);
        DrawRectangle(0, 0, view_w, view_h, (Color){0, 0, 40, a});
    }

    int hud_y = view_h;
    int32_t goal_block = craftax_mini_current_goal_block(env);
    int goal_count = craftax_mini_inventory_count_for_goal(s, goal_block);
    bool found_goal_gemstone = goal_count > 0;
    Color goal_color = craftax_mini_goal_color(goal_block);

    DrawRectangle(0, hud_y, view_w, hud_h, (Color){20, 20, 20, 255});
    DrawText(TextFormat("task: collect %s", craftax_mini_goal_name(goal_block)),
             4, hud_y + hud_pad, hud_title_font, goal_color);
    DrawText(TextFormat("progress: %d  status: %s",
             goal_count, found_goal_gemstone ? "complete" : "searching"),
             4, hud_y + hud_pad + hud_line + 2, hud_font,
             found_goal_gemstone ? (Color){120, 255, 120, 255} : (Color){220, 220, 220, 255});
    DrawText(TextFormat("ret: %.2f  len: %d / %d",
             env->episode_return_accum, env->episode_length_accum, g_craftax_mini_max_timesteps),
             4, hud_y + hud_pad + 2 * hud_line + 4, hud_font, (Color){220, 210, 140, 255});
    DrawText(TextFormat("HP: %.0f  food: %d  drink: %d  energy: %d  light: %.2f",
             s->player_health, s->player_food, s->player_drink,
             s->player_energy, s->light_level),
             4, hud_y + hud_pad + 3 * hud_line + 4, hud_font, WHITE);
    DrawText(TextFormat("pos: (%d, %d)  level: %d  step: %d",
             s->player_position[0], s->player_position[1], s->player_level, s->timestep),
             4, hud_y + hud_pad + 4 * hud_line + 4, hud_font, (Color){190, 190, 190, 255});

    if (found_goal_gemstone) {
        DrawText("goal item acquired", view_w - 220, hud_y + hud_pad + 2, 22, (Color){120, 255, 120, 255});
    }

    EndDrawing();
}
