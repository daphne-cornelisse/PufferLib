/*
 * Multi-thread WEF env throughput bench.
 *
 * Each worker owns a slice of the env pool and steps independently (no
 * per-tick barrier). Wall-clock env_SPS is the sum of threads.
 *
 * Build + run via ocean/wef/bench_sps.py, or:
 *   clang -O3 -DNDEBUG -mavx2 -mfma -DPLATFORM_DESKTOP \
 *     -I./raylib-5.5_linux_amd64/include -I./src -I./vendor -I./ocean/wef \
 *     ocean/wef/bench_sps.c -o wef_bench_sps \
 *     raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt
 */

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "wef.h"

#define DEFAULT_TOTAL_AGENTS 512
#define DEFAULT_NUM_AGENTS 4
#define DEFAULT_THREADS 16
#define DEFAULT_EPISODES 10
#define DEFAULT_EPISODE_LENGTH 2048
#define DEFAULT_WARMUP 1

typedef struct {
    int total_agents;
    int num_agents;
    int threads;
    int episodes;
    int episode_length;
    int warmup;
    int min_arena_width;
    int max_arena_width;
    int min_arena_height;
    int max_arena_height;
    int food_distribution;
    int num_food;
} Options;

typedef struct {
    Wef* envs;
    unsigned int* rngs;
    int env_begin;
    int env_end;
    int episode_length;
    int warmup;
    int episodes;
    pthread_barrier_t* barrier;
    double* episode_thread_s; /* [episodes] wall seconds this thread spent */
} Worker;

static double timespec_seconds(struct timespec t0, struct timespec t1) {
    return (double)(t1.tv_sec - t0.tv_sec) +
        (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static float random_action(unsigned int* rng) {
    return 2.0f * (float)rand_r(rng) / (float)RAND_MAX - 1.0f;
}

static void configure_env(Wef* env, unsigned int seed, const Options* opt) {
    Dict kwargs = {0};
    dict_set(&kwargs, "num_agents", opt->num_agents);
    dict_set(&kwargs, "min_arena_width", opt->min_arena_width);
    dict_set(&kwargs, "min_arena_height", opt->min_arena_height);
    dict_set(&kwargs, "max_arena_width", opt->max_arena_width);
    dict_set(&kwargs, "max_arena_height", opt->max_arena_height);
    dict_set(&kwargs, "food_distribution", opt->food_distribution);
    dict_set(&kwargs, "num_food", opt->num_food);
    dict_set(&kwargs, "patch_radius", 6);
    dict_set(&kwargs, "patch_radius_std", 1.5);
    dict_set(&kwargs, "patch_density", 0.001);
    dict_set(&kwargs, "electric_field_radius", 15);
    dict_set(&kwargs, "reflection_wall_range", 100);
    dict_set(&kwargs, "episode_length", opt->episode_length);
    memset(env, 0, sizeof(*env));
    env->rng = seed ? seed : 1u;
    puf_init(env, &kwargs);
    dict_clear(&kwargs);
}

static void bind_agents(
    Wef* env, obs_t* obs, float* act, float* rew, float* term
) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = obs + i * OBS_SIZE;
        env->agents[i].actions = act + i * NUM_ATNS;
        env->agents[i].rewards = rew + i;
        env->agents[i].terminals = term + i;
        env->agents[i].action_mask = NULL;
        env->agents[i].policy = 0;
    }
}

static void fill_random_actions(Wef* env, unsigned int* rng) {
    for (int a = 0; a < env->num_agents; a++) {
        float* action = env->agents[a].actions;
        for (int k = 0; k < ACTION_SIZE; k++) {
            action[k] = random_action(rng);
        }
    }
}

static void step_slice(Worker* w) {
    for (int e = w->env_begin; e < w->env_end; e++) {
        fill_random_actions(&w->envs[e], &w->rngs[e]);
        puf_step(&w->envs[e]);
    }
}

static void* worker_main(void* arg) {
    Worker* w = (Worker*)arg;
    for (int t = 0; t < w->warmup; t++) {
        for (int s = 0; s < w->episode_length; s++) {
            step_slice(w);
        }
    }
    for (int ep = 0; ep < w->episodes; ep++) {
        pthread_barrier_wait(w->barrier);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int s = 0; s < w->episode_length; s++) {
            step_slice(w);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        w->episode_thread_s[ep] = timespec_seconds(t0, t1);
        pthread_barrier_wait(w->barrier);
    }
    return NULL;
}

static void print_usage(const char* argv0) {
    printf(
        "Usage: %s [options]\n"
        "  --total-agents N     Agents across the pool (default %d)\n"
        "  --num-agents N       Agents per env (default %d)\n"
        "  --threads N          Independent worker threads (default %d)\n"
        "  --episodes N         Timed episodes (default %d)\n"
        "  --episode-length N   Steps per episode (default %d)\n"
        "  --warmup N           Untimed episodes (default %d)\n"
        "  --min-arena N        (default 30)\n"
        "  --max-arena N        (default 400)\n"
        "  --food-distribution N  0=uniform 1=patchy 2=random (default 2)\n"
        "  --num-food N         (default 64)\n"
        "  -h, --help\n",
        argv0,
        DEFAULT_TOTAL_AGENTS,
        DEFAULT_NUM_AGENTS,
        DEFAULT_THREADS,
        DEFAULT_EPISODES,
        DEFAULT_EPISODE_LENGTH,
        DEFAULT_WARMUP
    );
}

static bool parse_options(int argc, char** argv, Options* opt) {
    *opt = (Options){
        .total_agents = DEFAULT_TOTAL_AGENTS,
        .num_agents = DEFAULT_NUM_AGENTS,
        .threads = DEFAULT_THREADS,
        .episodes = DEFAULT_EPISODES,
        .episode_length = DEFAULT_EPISODE_LENGTH,
        .warmup = DEFAULT_WARMUP,
        .min_arena_width = 30,
        .max_arena_width = 400,
        .min_arena_height = 30,
        .max_arena_height = 400,
        .food_distribution = FOOD_RANDOM,
        .num_food = 64,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return false;
        } else if (!strcmp(argv[i], "--total-agents") && i + 1 < argc) {
            opt->total_agents = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--num-agents") && i + 1 < argc) {
            opt->num_agents = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            opt->threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--episodes") && i + 1 < argc) {
            opt->episodes = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--episode-length") && i + 1 < argc) {
            opt->episode_length = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            opt->warmup = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--min-arena") && i + 1 < argc) {
            opt->min_arena_width = opt->min_arena_height = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-arena") && i + 1 < argc) {
            opt->max_arena_width = opt->max_arena_height = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--food-distribution") && i + 1 < argc) {
            opt->food_distribution = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--num-food") && i + 1 < argc) {
            opt->num_food = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
    }
    if (opt->total_agents <= 0 || opt->num_agents <= 0 || opt->threads <= 0
        || opt->episodes <= 0 || opt->episode_length <= 0) {
        fprintf(stderr, "numeric options must be positive\n");
        return false;
    }
    if (opt->total_agents % opt->num_agents != 0) {
        fprintf(stderr, "total-agents (%d) must be divisible by num-agents (%d)\n",
            opt->total_agents, opt->num_agents);
        return false;
    }
    if (opt->num_agents > MAX_AGENTS) {
        fprintf(stderr, "num-agents (%d) > MAX_AGENTS (%d)\n",
            opt->num_agents, MAX_AGENTS);
        return false;
    }
    int num_envs = opt->total_agents / opt->num_agents;
    if (num_envs < opt->threads) {
        fprintf(stderr, "need at least one env per thread (%d envs, %d threads)\n",
            num_envs, opt->threads);
        return false;
    }
    return true;
}

static double mean_of(const double* xs, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += xs[i];
    }
    return n ? s / (double)n : 0.0;
}

static double std_of(const double* xs, int n) {
    if (n < 2) {
        return 0.0;
    }
    double m = mean_of(xs, n);
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        double d = xs[i] - m;
        acc += d * d;
    }
    return sqrt(acc / (double)(n - 1));
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_options(argc, argv, &opt)) {
        return 1;
    }

    int num_envs = opt.total_agents / opt.num_agents;
    int nthread = opt.threads;
    Wef* envs = (Wef*)calloc((size_t)num_envs, sizeof(Wef));
    unsigned int* rngs =
        (unsigned int*)calloc((size_t)num_envs, sizeof(unsigned int));
    if (!envs || !rngs) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    printf("wef_bench_sps\n");
    printf("  total_agents=%d  num_agents=%d  num_envs=%d  threads=%d\n",
        opt.total_agents, opt.num_agents, num_envs, nthread);
    printf("  warmup=%d  episodes=%d  episode_length=%d  sizeof(Wef)=%zu\n",
        opt.warmup, opt.episodes, opt.episode_length, sizeof(Wef));
    printf("  arena=[%d,%d]  food_distribution=%d  num_food=%d\n",
        opt.min_arena_width, opt.max_arena_width,
        opt.food_distribution, opt.num_food);
    fflush(stdout);

    size_t per_env_obs = (size_t)opt.num_agents * OBS_SIZE;
    size_t per_env_act = (size_t)opt.num_agents * NUM_ATNS;
    obs_t* all_obs = (obs_t*)calloc((size_t)num_envs * per_env_obs, sizeof(obs_t));
    float* all_act = (float*)calloc((size_t)num_envs * per_env_act, sizeof(float));
    float* all_rew = (float*)calloc((size_t)num_envs * opt.num_agents, sizeof(float));
    float* all_term = (float*)calloc((size_t)num_envs * opt.num_agents, sizeof(float));
    if (!all_obs || !all_act || !all_rew || !all_term) {
        fprintf(stderr, "agent buffer allocation failed\n");
        return 1;
    }
    for (int e = 0; e < num_envs; e++) {
        unsigned int seed = (unsigned int)(e + 1);
        configure_env(&envs[e], seed, &opt);
        bind_agents(
            &envs[e],
            all_obs + (size_t)e * per_env_obs,
            all_act + (size_t)e * per_env_act,
            all_rew + (size_t)e * opt.num_agents,
            all_term + (size_t)e * opt.num_agents
        );
        rngs[e] = seed * 0x9e3779b9u + 1u;
        puf_reset(&envs[e]);
    }

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, (unsigned)nthread);

    Worker* workers = (Worker*)calloc((size_t)nthread, sizeof(Worker));
    pthread_t* tids = (pthread_t*)calloc((size_t)nthread, sizeof(pthread_t));
    double* thread_ep_s =
        (double*)calloc((size_t)nthread * (size_t)opt.episodes, sizeof(double));
    if (!workers || !tids || !thread_ep_s) {
        fprintf(stderr, "worker allocation failed\n");
        return 1;
    }

    int base = num_envs / nthread;
    int rem = num_envs % nthread;
    int cursor = 0;
    for (int t = 0; t < nthread; t++) {
        int count = base + (t < rem ? 1 : 0);
        workers[t] = (Worker){
            .envs = envs,
            .rngs = rngs,
            .env_begin = cursor,
            .env_end = cursor + count,
            .episode_length = opt.episode_length,
            .warmup = opt.warmup,
            .episodes = opt.episodes,
            .barrier = &barrier,
            .episode_thread_s = thread_ep_s + (size_t)t * (size_t)opt.episodes,
        };
        cursor += count;
        pthread_create(&tids[t], NULL, worker_main, &workers[t]);
    }
    for (int t = 0; t < nthread; t++) {
        pthread_join(tids[t], NULL);
    }

    double* env_sps = (double*)calloc((size_t)opt.episodes, sizeof(double));
    double* agent_sps = (double*)calloc((size_t)opt.episodes, sizeof(double));
    double* sum_thread = (double*)calloc((size_t)opt.episodes, sizeof(double));
    long long env_steps_ep =
        (long long)num_envs * (long long)opt.episode_length;
    long long agent_steps_ep = env_steps_ep * (long long)opt.num_agents;

    printf("\n  per-episode (wall = sum of threads):\n");
    for (int ep = 0; ep < opt.episodes; ep++) {
        double wall = 0.0;
        double sum_thr = 0.0;
        for (int t = 0; t < nthread; t++) {
            double s = workers[t].episode_thread_s[ep];
            if (s > wall) {
                wall = s;
            }
            int n_local = workers[t].env_end - workers[t].env_begin;
            if (s > 0.0) {
                sum_thr += (double)n_local * (double)opt.episode_length / s;
            }
        }
        env_sps[ep] = wall > 0.0 ? (double)env_steps_ep / wall : 0.0;
        agent_sps[ep] = wall > 0.0 ? (double)agent_steps_ep / wall : 0.0;
        sum_thread[ep] = sum_thr;
        printf(
            "    ep %d  wall=%.4fs  env_SPS=%.1f  agent_SPS=%.1f  "
            "sum_thread_env_SPS=%.1f\n",
            ep, wall, env_sps[ep], agent_sps[ep], sum_thread[ep]
        );
    }

    double mean_env = mean_of(env_sps, opt.episodes);
    double std_env = std_of(env_sps, opt.episodes);
    double mean_agent = mean_of(agent_sps, opt.episodes);
    double std_agent = std_of(agent_sps, opt.episodes);
    double mean_sum = mean_of(sum_thread, opt.episodes);
    double std_sum = std_of(sum_thread, opt.episodes);

    printf("\n  summary (%d episodes):\n", opt.episodes);
    printf("    env_SPS          mean=%.1f  std=%.1f\n", mean_env, std_env);
    printf("    agent_SPS        mean=%.1f  std=%.1f\n", mean_agent, std_agent);
    printf("    sum_thread_env   mean=%.1f  std=%.1f\n", mean_sum, std_sum);
    printf("    per_thread_env   mean=%.1f\n",
        nthread ? mean_sum / (double)nthread : 0.0);

    printf("JSON {");
    printf("\"threads\":%d,\"num_envs\":%d,\"num_agents\":%d,",
        nthread, num_envs, opt.num_agents);
    printf("\"episode_length\":%d,\"episodes\":%d,",
        opt.episode_length, opt.episodes);
    printf("\"mean_env_sps\":%.6f,\"std_env_sps\":%.6f,", mean_env, std_env);
    printf("\"mean_agent_sps\":%.6f,\"std_agent_sps\":%.6f,",
        mean_agent, std_agent);
    printf("\"mean_sum_thread_env_sps\":%.6f,\"std_sum_thread_env_sps\":%.6f,",
        mean_sum, std_sum);
    printf("\"episodes_data\":[");
    for (int ep = 0; ep < opt.episodes; ep++) {
        printf("%s{\"env_sps\":%.6f,\"agent_sps\":%.6f,\"sum_thread_env_sps\":%.6f}",
            ep ? "," : "", env_sps[ep], agent_sps[ep], sum_thread[ep]);
    }
    printf("]}\n");

    pthread_barrier_destroy(&barrier);
    free(env_sps);
    free(agent_sps);
    free(sum_thread);
    free(thread_ep_s);
    free(workers);
    free(tids);
    free(all_obs);
    free(all_act);
    free(all_rew);
    free(all_term);
    free(envs);
    free(rngs);
    return 0;
}
