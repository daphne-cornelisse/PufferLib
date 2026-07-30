/*
 * Standalone random-policy demo.
 *
 * Build:
 *   ./build.sh wef --local
 * Run:
 *   ./wef
 */

#include <time.h>
#include <stdio.h>
#include "wef.h"

static float random_action(unsigned int* rng) {
    return 2.0f * (float)rand_r(rng) / (float)RAND_MAX - 1.0f;
}

static void print_physics_fixture(FishEnv* env) {
    env->num_agents = 2;
    env->num_food = 1;
    env->agents[0].movement.position_cm = (FishVec2){20.0, 20.0};
    env->agents[0].movement.orientation = 0.25;
    env->agents[0].emits_eod = true;
    env->agents[1].movement.position_cm = (FishVec2){45.0, 40.0};
    env->agents[1].movement.orientation = -1.0;
    env->agents[1].emits_eod = false;
    env->food[0].motion.position_cm = (FishVec2){30.0, 25.0};
    env->food[0].motion.orientation = 0.7;
    env->food[0].active = true;
    calibrate_electroreceptors(env);
    build_electric_scene(env);

    FishVec2 query = {35.0, 30.0};
    FishVec2 eod = measure_electric_field_with_reflections(
        query, env->eod_sources, 4, NULL, 0,
        env->arena_size_cm, FIELD_EPS_M,
        REFLECTION_SCALE, false
    );
    FishVec2 intrinsic = measure_electric_field_with_reflections(
        query, NULL, 0, env->intrinsic_sources,
        (size_t)env->num_intrinsic_sources,
        env->arena_size_cm, FIELD_EPS_M,
        REFLECTION_SCALE, false
    );
    FishVec2 induced = measure_electric_field_with_reflections(
        query, NULL, 0, env->induced_sources,
        (size_t)env->num_induced_sources,
        env->arena_size_cm, FIELD_EPS_M,
        REFLECTION_SCALE, false
    );
    printf(
        "{\"eod\":[%.17g,%.17g],\"intrinsic\":[%.17g,%.17g],"
        "\"induced\":[%.17g,%.17g],\"induced_fish\":[%.17g,%.17g],"
        "\"induced_food\":[%.17g,%.17g],\"morm_cd0\":%.17g,"
        "\"amp_baseline0\":%.17g}\n",
        eod.x, eod.y, intrinsic.x, intrinsic.y, induced.x, induced.y,
        env->induced_sources[1].moment_c_m.x,
        env->induced_sources[1].moment_c_m.y,
        env->induced_sources[2].moment_c_m.x,
        env->induced_sources[2].moment_c_m.y,
        env->mormyromast_cd[0], env->amp_intrinsic_baseline[0]
    );
}

int main(int argc, char** argv) {
    FishEnv env = {
        .num_agents = 4,
        .arena_size_cm = {70.0, 70.0},
        .electric_field_radius_cm = 15.0,
        .episode_length = 4096,
        .rng = (unsigned int)time(NULL),
    };
    c_allocate(&env);
    c_reset(&env);

    if (argc > 1 && strcmp(argv[1], "--physics-fixture") == 0) {
        print_physics_fixture(&env);
        free_allocated(&env);
        return 0;
    }

    unsigned int demo_rng = env.rng ^ 0x9e3779b9U;
    float target_actions[MAX_AGENTS][ACTION_SIZE] = {{0}};
    bool headless = argc > 1 && strcmp(argv[1], "--headless") == 0;

    if (!headless) c_render(&env);

    while (headless ? env.tick < 1000 : !WindowShouldClose()) {
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

        /* Smooth random walk commands; EOD remains a discrete pulse action. */
        for (int i = 0; i < env.num_agents; i++) {
            float* action = env.actions + i * ACTION_SIZE;
            action[0] += 0.06f * (target_actions[i][0] - action[0]);
            action[1] += 0.08f * (target_actions[i][1] - action[1]);
            action[2] = target_actions[i][2];
            action[3] = target_actions[i][3];
        }

        c_step(&env);
        if (!headless) c_render(&env);
    }

    if (headless) {
        printf("wef headless smoke test: 1000 steps, %d food eaten\n",
            env.food_eaten);
    }
    free_allocated(&env);
    return 0;
}
