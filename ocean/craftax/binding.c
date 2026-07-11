#define CRAFTAX_ENABLE_ENV_IMPL
#define CRAFTAX_ENABLE_RENDERING
#include "craftax.h"
#include "step_crafting.h"
#include "step_mobs.h"
#include "step_spawning.h"

#define OBS_SIZE CRAFTAX_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {CRAFTAX_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define MY_VEC_STEP craftax_vec_step
#define MY_VEC_STEP_RANGE craftax_vec_step_range
#define Env Craftax
#include "vecenv.h"

// Tiled vector step: process agents in cache-friendly chunks. Each env stores a
// lightweight handle while the backing CraftaxState instances live in a shared
// arena allocation.
void craftax_vec_step(StaticVec* vec) {
    memset(vec->rewards, 0, vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, vec->total_agents * sizeof(float));
    Craftax* envs = (Craftax*)vec->envs;
    int size = vec->size;
    #pragma omp parallel for schedule(static)
    for (int tile = 0; tile < size; tile += CRAFTAX_VEC_TILE_SIZE) {
        int end = tile + CRAFTAX_VEC_TILE_SIZE;
        if (end > size) end = size;
        for (int i = tile; i < end; i++) {
            c_step_gameplay(&envs[i]);
            craftax_encode_observation(envs[i].state, envs[i].observations);
        }
    }
}

void craftax_vec_step_range(StaticVec* vec, int env_start, int env_count, int num_workers) {
    (void)num_workers;
    Craftax* envs = (Craftax*)vec->envs;
    int env_end = env_start + env_count;
    for (int tile = env_start; tile < env_end; tile += CRAFTAX_VEC_TILE_SIZE) {
        int end = tile + CRAFTAX_VEC_TILE_SIZE;
        if (end > env_end) end = env_end;
        for (int i = tile; i < end; i++) {
            c_step_gameplay(&envs[i]);
            craftax_encode_observation(envs[i].state, envs[i].observations);
        }
    }
}

static CraftaxState* craftax_alloc_state_arena(int num_envs) {
    return (CraftaxState*)calloc((size_t)num_envs, sizeof(CraftaxState));
}

Env* my_vec_init(
    int* num_envs_out,
    int* buffer_env_starts,
    int* buffer_env_counts,
    Dict* vec_kwargs,
    Dict* env_kwargs
) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    int num_envs = total_agents;

    Env* envs = (Env*)calloc((size_t)num_envs, sizeof(Env));
    CraftaxArena* arena = (CraftaxArena*)calloc(1, sizeof(CraftaxArena));
    arena->states = craftax_alloc_state_arena(num_envs);

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;

    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->rng = (unsigned int)i;
        env->arena = arena;
        env->state = &arena->states[i];
        env->owns_state_storage = false;
        my_init(env, env_kwargs);

        buf_agents += env->num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_vec_close(Env* envs) {
    if (envs == NULL || envs[0].arena == NULL) {
        return;
    }

    CraftaxArena* arena = envs[0].arena;
    free(arena->states);
    free(arena);
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->render_view_mode = CRAFTAX_VIEW_MODE_AGENT;

    float initial_health = 9.0f;
    DictItem* health_item = dict_get_unsafe(kwargs, "initial_health");
    if (health_item != NULL) {
        initial_health = (float)health_item->value;
    }
    craftax_set_initial_health(initial_health);

    uint64_t seed_offset = 0;
    DictItem* item = dict_get_unsafe(kwargs, "seed_offset");
    if (item != NULL) {
        seed_offset = (uint64_t)item->value;
    }
    env->seed = seed_offset + (uint64_t)env->rng;

    // Process-wide reset pool. A size of 0 disables caching and regenerates the
    // world on every reset.
    int reset_pool_size = 0;
    DictItem* pool_item = dict_get_unsafe(kwargs, "reset_pool_size");
    if (pool_item != NULL) reset_pool_size = (int)pool_item->value;
    craftax_set_reset_pool_size(reset_pool_size);

    DictItem* view_mode_item = dict_get_unsafe(kwargs, "view_mode");
    if (view_mode_item != NULL) {
        env->render_view_mode = (int32_t)view_mode_item->value;
    }

    c_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    // Log tech and floor progression checkpoints. Values are per-window means
    // over completed episodes, so each line is a reach/unlock rate.
    struct { const char* name; int idx; } checkpoints[] = {
        {"collect_wood",         0},
        {"make_wood_pickaxe",    5},
        {"make_stone_pickaxe",  13},
        {"collect_iron",        18},
        {"make_iron_pickaxe",   20},
        {"collect_diamond",     19},
        {"floor_1_dungeon",       29},
        {"floor_2_gnomish_mines", 28},
        {"floor_3_sewers",        30},
        {"floor_4_vault",         31},
        {"floor_5_troll_mines",   32},
        {"floor_6_fire_realm",    33},
        {"floor_7_ice_realm",     34},
        {"floor_8_graveyard",     35},
        {"defeat_necromancer",  48},
    };
    for (int i = 0; i < (int)(sizeof(checkpoints) / sizeof(checkpoints[0])); i++) {
        dict_set(out, checkpoints[i].name, log->achievements[checkpoints[i].idx]);
    }
}
