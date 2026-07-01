#include "robocode.h"
#define OBS_SIZE 10
#define NUM_ATNS 5
#define ACT_SIZES {}
#define OBS_TENSOR_T FloatTensor

#define Env Robocode
#include "vecvenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env.width = 768;
    env.height = 576;
    allocate_env(&env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}