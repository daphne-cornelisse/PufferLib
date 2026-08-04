/*
 * Standalone random-policy demo.
 *
 * Build:
 *   ./build.sh wef --local
 * Run:
 *   ./wef
 */

#include <time.h>
#include "wef.h"

static float random_action(unsigned int* rng) {
    return 2.0f * (float)rand_r(rng) / (float)RAND_MAX - 1.0f;
}

int main(void) {
    FishEnv env = {
        .num_agents = 4,
        .arena_size_x = 70.0f,
        .arena_size_y = 70.0f,
        .electric_field_radius_cm = 15.0f,
        .episode_length = 4096,
        .rng = (unsigned int)time(NULL),
    };
    c_allocate(&env);
    c_reset(&env);

    unsigned int demo_rng = env.rng ^ 0x9e3779b9U;
    float target_actions[MAX_AGENTS][ACTION_SIZE] = {{0}};

    c_render(&env);

    while (!WindowShouldClose()) {
        if (env.tick % 45 == 0) {
            for (int i = 0; i < env.num_agents; i++) {
                target_actions[i][0] = random_action(&demo_rng);
                target_actions[i][1] = random_action(&demo_rng);
                target_actions[i][2] =
                    rand_r(&demo_rng) % 100 < 35 ? 1.0f : -1.0f;
                target_actions[i][3] =
                    rand_r(&demo_rng) % 100 < 8 ? 1.0f : -1.0f;
            }
        }

        // Take random actions
        for (int i = 0; i < env.num_agents; i++) {
            float* action = env.actions + i * ACTION_SIZE;
            action[0] += 0.06f * (target_actions[i][0] - action[0]);
            action[1] += 0.08f * (target_actions[i][1] - action[1]);
            action[2] = target_actions[i][2];
            action[3] = target_actions[i][3];
        }

        c_step(&env);
        c_render(&env);
    }

    free_allocated(&env);
    return 0;
}
