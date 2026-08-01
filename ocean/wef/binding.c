#include "wef.h"

#define NUM_ATNS ACTION_SIZE
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env FishEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_agents")->value;
    env->min_arena_size_x = dict_get(kwargs, "min_arena_width")->value;
    env->min_arena_size_y = dict_get(kwargs, "min_arena_height")->value;
    env->max_arena_size_x = dict_get(kwargs, "max_arena_width")->value;
    env->max_arena_size_y = dict_get(kwargs, "max_arena_height")->value;
    env->arena_size_x = env->min_arena_size_x;
    env->arena_size_y = env->min_arena_size_y;
    env->food_distribution = (FoodDistribution)(int)dict_get(kwargs, "food_distribution")->value;
    env->configured_num_food = (int)dict_get(kwargs, "num_food")->value;
    env->patch_radius_cm = dict_get(kwargs, "patch_radius")->value;
    env->patch_radius_std_cm = dict_get(kwargs, "patch_radius_std")->value;
    env->patch_density = dict_get(kwargs, "patch_density")->value;
    env->electric_field_radius_cm = dict_get(kwargs, "electric_field_radius")->value;
    env->episode_length = (int)dict_get(kwargs, "episode_length")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "food_eaten_mean", log->food_eaten_mean);
    dict_set(out, "eod_rate", log->eod_rate);
    dict_set(out, "collisions_fish", log->collisions_fish);
}
