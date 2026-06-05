#include "proteins.h"
#define OBS_SIZE (BINDER_LEN * FEATURE_DIM + NUM_TARGETS * BINDER_LEN * FEATURE_DIM)
#define NUM_ATNS 1
#define ACT_SIZES {BINDER_LEN * FEATURE_DIM}
#define OBS_TENSOR_T FloatTensor

#define Env Proteins
#include "vecenv.h"

void my_init(Proteins* env, Dict* kwargs) {
    // Initialize environment parameters.
    init(env);   
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);   
    dict_set(out, "score", log->score);   
    dict_set(out, "aff_on", log->aff_on);   
    dict_set(out, "aff_off", log->aff_off);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}