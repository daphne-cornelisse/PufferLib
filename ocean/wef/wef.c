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
    env->agents[0].pos_x = 20.0f;
    env->agents[0].pos_y = 20.0f;
    env->agents[0].orientation = 0.25f;
    env->agents[0].emits_eod = true;
    env->agents[1].pos_x = 45.0f;
    env->agents[1].pos_y = 40.0f;
    env->agents[1].orientation = -1.0f;
    env->agents[1].emits_eod = false;
    env->food[0].pos_x = 30.0f;
    env->food[0].pos_y = 25.0f;
    env->food[0].orientation = 0.7f;
    env->food[0].active = true;
    calibrate_electroreceptors(env);
    build_electric_scene(env);

    float mono_x[4], mono_y[4], mono_q[4];
    float dip_x[MAX_AGENTS + MAX_FOOD];
    float dip_y[MAX_AGENTS + MAX_FOOD];
    float dip_mx[MAX_AGENTS + MAX_FOOD];
    float dip_my[MAX_AGENTS + MAX_FOOD];
    int n_mono = 0;
    for (int i = 0; i < env->num_agents; i++) {
        for (int p = 0; p < 2; p++) {
            mono_x[n_mono] = env->agents[i].eod_pos_x[p];
            mono_y[n_mono] = env->agents[i].eod_pos_y[p];
            mono_q[n_mono] = env->agents[i].eod_charge[p];
            n_mono++;
        }
    }
    int n_intr = 0;
    for (int i = 0; i < env->num_agents; i++) {
        dip_x[n_intr] = env->agents[i].pos_x;
        dip_y[n_intr] = env->agents[i].pos_y;
        dip_mx[n_intr] = env->agents[i].intrinsic_moment_x;
        dip_my[n_intr] = env->agents[i].intrinsic_moment_y;
        n_intr++;
    }
    if (env->food[0].active) {
        dip_x[n_intr] = env->food[0].pos_x;
        dip_y[n_intr] = env->food[0].pos_y;
        dip_mx[n_intr] = env->food[0].intrinsic_moment_x;
        dip_my[n_intr] = env->food[0].intrinsic_moment_y;
        n_intr++;
    }
    int n_ind = 0;
    float ind_x[MAX_AGENTS + MAX_FOOD];
    float ind_y[MAX_AGENTS + MAX_FOOD];
    float ind_mx[MAX_AGENTS + MAX_FOOD];
    float ind_my[MAX_AGENTS + MAX_FOOD];
    for (int i = 0; i < env->num_agents; i++) {
        ind_x[n_ind] = env->agents[i].pos_x;
        ind_y[n_ind] = env->agents[i].pos_y;
        ind_mx[n_ind] = env->agents[i].induced_moment_x;
        ind_my[n_ind] = env->agents[i].induced_moment_y;
        n_ind++;
    }
    if (env->food[0].active) {
        ind_x[n_ind] = env->food[0].pos_x;
        ind_y[n_ind] = env->food[0].pos_y;
        ind_mx[n_ind] = env->food[0].induced_moment_x;
        ind_my[n_ind] = env->food[0].induced_moment_y;
        n_ind++;
    }

    float eod_x, eod_y, intrinsic_x, intrinsic_y, induced_x, induced_y;
    measure_electric_field_with_reflections(
        35.0f, 30.0f,
        mono_x, mono_y, mono_q, (size_t)n_mono,
        NULL, NULL, NULL, NULL, 0,
        env->arena_size_x, env->arena_size_y, FIELD_EPS_M,
        REFLECTION_SCALE, false, &eod_x, &eod_y
    );
    measure_electric_field_with_reflections(
        35.0f, 30.0f,
        NULL, NULL, NULL, 0,
        dip_x, dip_y, dip_mx, dip_my, (size_t)n_intr,
        env->arena_size_x, env->arena_size_y, FIELD_EPS_M,
        REFLECTION_SCALE, false, &intrinsic_x, &intrinsic_y
    );
    measure_electric_field_with_reflections(
        35.0f, 30.0f,
        NULL, NULL, NULL, 0,
        ind_x, ind_y, ind_mx, ind_my, (size_t)n_ind,
        env->arena_size_x, env->arena_size_y, FIELD_EPS_M,
        REFLECTION_SCALE, false, &induced_x, &induced_y
    );
    printf(
        "{\"eod\":[%.17g,%.17g],\"intrinsic\":[%.17g,%.17g],"
        "\"induced\":[%.17g,%.17g],\"induced_fish\":[%.17g,%.17g],"
        "\"induced_food\":[%.17g,%.17g],\"morm_cd0\":%.17g,"
        "\"amp_baseline0\":%.17g}\n",
        eod_x, eod_y, intrinsic_x, intrinsic_y, induced_x, induced_y,
        env->agents[1].induced_moment_x,
        env->agents[1].induced_moment_y,
        env->food[0].induced_moment_x,
        env->food[0].induced_moment_y,
        env->mormyromast_cd[0], env->amp_intrinsic_baseline[0]
    );
}

int main(int argc, char** argv) {
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
