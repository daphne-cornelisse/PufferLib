/*
 * WEF per-thread rollout throughput (and optional gprof target).
 *
 * Build / run:
 *   ./ocean/wef/run_profile.sh
 *   ./ocean/wef/run_profile.sh --gprof
 *   ./ocean/wef/run_profile.sh --threads 8 --envs-per-thread 4 --steps 200
 *
 * Each OpenMP thread owns a private set of envs and runs reset+step
 * independently. Reports agent-steps/sec overall and per thread.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef WEF_PROFILE_NO_OMP
#include <omp.h>
#endif

#include "wef.h"

#define DEFAULT_NUM_FISH 4
#define DEFAULT_THREADS 8
#define DEFAULT_ENVS_PER_THREAD 4
#define DEFAULT_STEPS 200
#define DEFAULT_RESETS 1

typedef struct {
    int num_fish;
    int threads;
    int envs_per_thread;
    int steps;
    int resets;
} Options;

typedef struct {
    double elapsed_s;
    long long agent_steps;
    double agent_sps;
} ThreadResult;

static double timespec_seconds(struct timespec t0, struct timespec t1) {
    return (double)(t1.tv_sec - t0.tv_sec) +
        (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static float random_action(unsigned int* rng) {
    return 2.0f * (float)rand_r(rng) / (float)RAND_MAX - 1.0f;
}

static void configure_env(FishEnv* env, unsigned int seed, int num_fish) {
    memset(env, 0, sizeof(*env));
    env->num_agents = num_fish;
    env->arena_size_x = 70.0f;
    env->arena_size_y = 70.0f;
    env->min_arena_size_x = 70.0f;
    env->min_arena_size_y = 70.0f;
    env->max_arena_size_x = 70.0f;
    env->max_arena_size_y = 70.0f;
    env->food_distribution = FOOD_RANDOM;
    env->configured_num_food = 64;
    env->electric_field_radius_cm = 15.0f;
    env->episode_length = 4096;
    env->rng = seed ? seed : 1u;
}

static void fill_random_actions(FishEnv* env, unsigned int* rng) {
    for (int a = 0; a < env->num_agents; a++) {
        float* action = env->actions + a * ACTION_SIZE;
        for (int k = 0; k < ACTION_SIZE; k++) {
            action[k] = random_action(rng);
        }
    }
}

static void print_usage(const char* argv0) {
    printf(
        "Usage: %s [options]\n"
        "  --threads N           Worker threads (default %d)\n"
        "  --envs-per-thread N   Envs owned by each thread (default %d)\n"
        "  --num-fish N          Agents per env (default %d)\n"
        "  --steps N             c_step calls per reset (default %d)\n"
        "  --resets N            c_reset cycles per env (default %d)\n"
        "  -h, --help\n",
        argv0,
        DEFAULT_THREADS,
        DEFAULT_ENVS_PER_THREAD,
        DEFAULT_NUM_FISH,
        DEFAULT_STEPS,
        DEFAULT_RESETS
    );
}

static bool parse_options(int argc, char** argv, Options* opt) {
    *opt = (Options){
        .num_fish = DEFAULT_NUM_FISH,
        .threads = DEFAULT_THREADS,
        .envs_per_thread = DEFAULT_ENVS_PER_THREAD,
        .steps = DEFAULT_STEPS,
        .resets = DEFAULT_RESETS,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return false;
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            opt->threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--envs-per-thread") && i + 1 < argc) {
            opt->envs_per_thread = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--num-fish") && i + 1 < argc) {
            opt->num_fish = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            opt->steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--resets") && i + 1 < argc) {
            opt->resets = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
    }
    if (opt->num_fish <= 0 || opt->threads <= 0 ||
            opt->envs_per_thread <= 0 || opt->steps <= 0 ||
            opt->resets <= 0) {
        fprintf(stderr, "all numeric options must be positive\n");
        return false;
    }
#ifdef WEF_PROFILE_NO_OMP
    if (opt->threads != 1) {
        fprintf(stderr, "OpenMP disabled: forcing --threads 1\n");
        opt->threads = 1;
    }
#endif
    return true;
}

/* One thread's private rollout: returns local agent-steps and wall time. */
static ThreadResult run_thread_rollout(
    int thread_id,
    int num_fish,
    int envs_per_thread,
    int steps,
    int resets
) {
    FishEnv* envs =
        (FishEnv*)calloc((size_t)envs_per_thread, sizeof(FishEnv));
    unsigned int* rngs =
        (unsigned int*)calloc((size_t)envs_per_thread, sizeof(unsigned int));
    ThreadResult result = {0};
    if (!envs || !rngs) {
        fprintf(stderr, "thread %d: allocation failed\n", thread_id);
        free(envs);
        free(rngs);
        return result;
    }

    for (int e = 0; e < envs_per_thread; e++) {
        unsigned int seed =
            (unsigned int)(thread_id * 10007 + e + 1);
        configure_env(&envs[e], seed, num_fish);
        c_allocate(&envs[e]);
        rngs[e] = seed * 0x9e3779b9u + 1u;
    }

    /* Warmup (not timed). */
    for (int e = 0; e < envs_per_thread; e++) {
        c_reset(&envs[e]);
        for (int t = 0; t < 3; t++) {
            fill_random_actions(&envs[e], &rngs[e]);
            c_step(&envs[e]);
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int r = 0; r < resets; r++) {
        for (int e = 0; e < envs_per_thread; e++) {
            c_reset(&envs[e]);
            for (int t = 0; t < steps; t++) {
                fill_random_actions(&envs[e], &rngs[e]);
                c_step(&envs[e]);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    result.elapsed_s = timespec_seconds(t0, t1);
    result.agent_steps =
        (long long)resets * (long long)envs_per_thread *
        (long long)steps * (long long)num_fish;
    result.agent_sps = result.elapsed_s > 0.0
        ? (double)result.agent_steps / result.elapsed_s
        : 0.0;

    for (int e = 0; e < envs_per_thread; e++) {
        free_allocated(&envs[e]);
    }
    free(envs);
    free(rngs);
    return result;
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_options(argc, argv, &opt)) {
        return 1;
    }

    ThreadResult* results =
        (ThreadResult*)calloc((size_t)opt.threads, sizeof(ThreadResult));
    if (!results) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    printf("wef per-thread rollout SPS\n");
    printf("  threads=%d  envs_per_thread=%d  num_fish=%d\n",
        opt.threads, opt.envs_per_thread, opt.num_fish);
    printf("  resets=%d  steps_per_reset=%d\n", opt.resets, opt.steps);
    printf("  total_envs=%d  sizeof(FishEnv)=%zu\n",
        opt.threads * opt.envs_per_thread, sizeof(FishEnv));
    fflush(stdout);

    struct timespec wall0, wall1;
    clock_gettime(CLOCK_MONOTONIC, &wall0);

#ifndef WEF_PROFILE_NO_OMP
    #pragma omp parallel num_threads(opt.threads)
    {
        int tid = omp_get_thread_num();
        results[tid] = run_thread_rollout(
            tid, opt.num_fish, opt.envs_per_thread, opt.steps, opt.resets
        );
    }
#else
    results[0] = run_thread_rollout(
        0, opt.num_fish, opt.envs_per_thread, opt.steps, opt.resets
    );
#endif

    clock_gettime(CLOCK_MONOTONIC, &wall1);
    double wall_s = timespec_seconds(wall0, wall1);

    long long total_agent_steps = 0;
    double sum_thread_sps = 0.0;
    double min_thread_sps = 1e300;
    double max_thread_sps = 0.0;
    double sum_thread_elapsed = 0.0;

    printf("\n  per-thread agent SPS:\n");
    for (int t = 0; t < opt.threads; t++) {
        ThreadResult* r = &results[t];
        total_agent_steps += r->agent_steps;
        sum_thread_sps += r->agent_sps;
        sum_thread_elapsed += r->elapsed_s;
        if (r->agent_sps < min_thread_sps) min_thread_sps = r->agent_sps;
        if (r->agent_sps > max_thread_sps) max_thread_sps = r->agent_sps;
        printf("    thread %2d:  SPS=%.1f  steps=%lld  elapsed=%.4fs\n",
            t, r->agent_sps, r->agent_steps, r->elapsed_s);
    }

    double mean_thread_sps = sum_thread_sps / (double)opt.threads;
    /* Aggregate SPS from total work over wall-clock (includes imbalance). */
    double wall_sps =
        wall_s > 0.0 ? (double)total_agent_steps / wall_s : 0.0;
    /* Ideal scale-out: sum of per-thread rates (no load imbalance). */
    double sum_sps = sum_thread_sps;

    printf("\n  summary (agent-steps/sec):\n");
    printf("    per_thread_mean=%.1f\n", mean_thread_sps);
    printf("    per_thread_min=%.1f\n", min_thread_sps);
    printf("    per_thread_max=%.1f\n", max_thread_sps);
    printf("    sum_of_threads=%.1f  (if perfectly parallel)\n", sum_sps);
    printf("    wall_SPS=%.1f  (total work / wall time)\n", wall_sps);
    printf("    wall_elapsed=%.4fs  total_agent_steps=%lld\n",
        wall_s, total_agent_steps);
    printf("    mean_thread_elapsed=%.4fs\n",
        sum_thread_elapsed / (double)opt.threads);

    free(results);
    return 0;
}
