// Standalone viewer for goal-conditioned Craftax mini.

#include "craftax_mini.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x ? x : 0xdeadbeef;
    return x;
}

int main(int argc, char** argv) {
    uint64_t seed = (argc > 1) ? strtoull(argv[1], NULL, 10) : (uint64_t)time(NULL);

    Craftax env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.seed = seed;
    env.rng = (uint32_t)seed;

    env.observations = calloc(CRAFTAX_OBS_SIZE, sizeof(float));
    env.actions = calloc(1, sizeof(float));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(float));

    c_init(&env);
    c_reset(&env);
    c_render(&env);

    uint32_t action_rng = (uint32_t)(seed ^ 0x9E3779B9u);
    while (!WindowShouldClose()) {
        env.actions[0] = (float)(xorshift32(&action_rng) % CRAFTAX_MINI_NUM_ACTIONS);
        c_step(&env);
        c_render(&env);
    }

    c_close(&env);
    if (IsWindowReady()) CloseWindow();
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
