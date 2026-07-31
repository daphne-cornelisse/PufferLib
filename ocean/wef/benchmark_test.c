/*
 * Deterministic numerical-stability harness for wef.
 *
 * Creates a 4-fish arena, steps with seeded random actions, and reports
 * observations plus internal environment state. For each seed in {0,1,2},
 * runs the trajectory twice and asserts bit-identical results, then compares
 * fingerprints against the double-precision baseline at
 * artifacts/wef/benchmark_baseline_fingerprints.txt so biophysics changes
 * (e.g. double→float) are easy to detect.
 *
 * Build (standalone wef also builds this binary):
 *   ./build.sh wef --fast
 *   ./wef_benchmark_test
 *   ./wef_benchmark_test --baseline path/to/fingerprints.txt
 *
 * Manual build:
 *   clang -O2 -I./raylib-5.5_linux_amd64/include -I./src -I./vendor \
 *     -I./ocean/wef ocean/wef/benchmark_test.c \
 *     raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -fopenmp \
 *     -DPLATFORM_DESKTOP -o wef_benchmark_test
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "wef.h"

#define FNV_OFFSET 14695981039346656037ULL
#define FNV_PRIME 1099511628211ULL

/* Stability trajectory (single env, small + deterministic). */
#define NUM_FISH 4
#define NUM_STEPS 100
#define NUM_FOOD 16
#define NUM_TEST_SEEDS 3
#define DEFAULT_BASELINE_PATH "artifacts/wef/benchmark_baseline_fingerprints.txt"
/* Reward / episode_return are multiples of 0.5; allow tiny float noise. */
#define BASELINE_REWARD_ATOL 1e-5
#define BASELINE_OBS_SUM_ATOL 1e-4
#define BASELINE_OBS_SUM_RTOL 1e-6

static const unsigned int TEST_SEEDS[NUM_TEST_SEEDS] = {0u, 1u, 2u};

typedef struct SeedMetrics {
    unsigned int seed;
    double reward_sum;
    double final_obs_sum;
    int food_eaten;
    float episode_return;
    unsigned int rng_end;
    uint64_t obs_fnv1a;
    uint64_t rewards_fnv1a;
    uint64_t actions_fnv1a;
    uint64_t state_fnv1a;
} SeedMetrics;

typedef struct BaselineTable {
    bool loaded;
    char path[512];
    char commit[128];
    SeedMetrics seeds[NUM_TEST_SEEDS];
    bool seed_present[NUM_TEST_SEEDS];
} BaselineTable;

/*
 * SPS benchmark sizing from config/wef.ini:
 *   [vec]
 *     total_agents = 4096
 *     num_buffers  = 8
 *     num_threads  = 8
 *   [env]
 *     num_agents = 4
 *
 * Layout matches StaticVec / my_vec_init:
 *   num_envs = total_agents / num_agents = 1024
 *   agents_per_buffer = total_agents / num_buffers = 512
 *   envs_per_buffer   ≈ 512 / 4 = 128
 *   workers_per_buffer = num_threads / num_buffers = 1
 *
 * SPS = total environment steps / wall time, flattened over agents:
 *   steps_counted = total_agents * vec_steps
 *   SPS = steps_counted / elapsed
 * (Same units as puffer train: global_step += total_agents each env step.)
 */
#define TOTAL_AGENTS 4096
#define NUM_BUFFERS 8
#define NUM_THREADS 8
#define NUM_ENVS (TOTAL_AGENTS / NUM_FISH)
#define AGENTS_PER_BUFFER (TOTAL_AGENTS / NUM_BUFFERS)
#define WORKERS_PER_BUFFER (NUM_THREADS / NUM_BUFFERS)
/* Outer vec ticks. Flattened steps timed = TOTAL_AGENTS * SPS_STEPS.
 * 200 * 4096 ≈ 0.8M steps — enough for a stable reading. */
#define SPS_STEPS 200

typedef struct TrajectorySnapshot {
    float observations[(NUM_STEPS + 1) * NUM_FISH * OBS_SIZE];
    float rewards[NUM_STEPS * NUM_FISH];
    float terminals[NUM_STEPS * NUM_FISH];
    float actions[NUM_STEPS * NUM_FISH * ACTION_SIZE];
    double agent_x[NUM_STEPS + 1][NUM_FISH];
    double agent_y[NUM_STEPS + 1][NUM_FISH];
    double agent_orientation[NUM_STEPS + 1][NUM_FISH];
    double agent_size[NUM_STEPS + 1][NUM_FISH];
    double agent_lin_vel[NUM_STEPS + 1][NUM_FISH];
    double agent_ang_vel[NUM_STEPS + 1][NUM_FISH];
    int food_active[NUM_STEPS + 1][MAX_FOOD];
    double food_x[NUM_STEPS + 1][MAX_FOOD];
    double food_y[NUM_STEPS + 1][MAX_FOOD];
    int tick[NUM_STEPS + 1];
    int food_eaten[NUM_STEPS + 1];
    int collisions_fish[NUM_STEPS + 1];
    int eod_agent_steps[NUM_STEPS + 1];
    float episode_return[NUM_STEPS + 1];
    unsigned int rng[NUM_STEPS + 1];
    int num_food;
} TrajectorySnapshot;

static float random_action(unsigned int* rng) {
    return 2.0f * (float)rand_r(rng) / (float)RAND_MAX - 1.0f;
}

static uint64_t fnv1a_update(uint64_t hash, const void* data, size_t nbytes) {
    const unsigned char* bytes = (const unsigned char*)data;
    for (size_t i = 0; i < nbytes; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static uint64_t hash_bytes(const void* data, size_t nbytes) {
    return fnv1a_update(FNV_OFFSET, data, nbytes);
}

static bool floats_identical(const float* a, const float* b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

static bool doubles_identical(const double* a, const double* b, size_t n) {
    return memcmp(a, b, n * sizeof(double)) == 0;
}

static bool all_finite_float(const float* values, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)values[i])) return false;
    }
    return true;
}

/* init() rewrites rng==0 to 1, so map seed 0 to a distinct non-zero value. */
static unsigned int env_rng_from_seed(unsigned int seed) {
    return seed == 0u ? 0xA341316Cu : seed;
}

static void configure_env(FishEnv* env, unsigned int seed) {
    memset(env, 0, sizeof(*env));
    env->num_agents = NUM_FISH;
    env->arena_size_cm = (FishVec2){70.0, 70.0};
    env->min_arena_size_cm = (FishVec2){70.0, 70.0};
    env->max_arena_size_cm = (FishVec2){70.0, 70.0};
    env->food_distribution = FOOD_UNIFORM;
    env->configured_num_food = NUM_FOOD;
    env->fixed_num_patches = 4;
    env->patch_radius_cm = 6.0;
    env->patch_radius_std_cm = 1.5;
    env->patch_density = 0.001;
    env->placement_radius_frac = 0.75;
    env->electric_field_radius_cm = 15.0;
    env->episode_length = 4096;
    env->rng = env_rng_from_seed(seed);
}

static void capture_state(const FishEnv* env, TrajectorySnapshot* snap, int frame) {
    snap->tick[frame] = env->tick;
    snap->food_eaten[frame] = env->food_eaten;
    snap->collisions_fish[frame] = env->collisions_fish;
    snap->eod_agent_steps[frame] = env->eod_agent_steps;
    snap->episode_return[frame] = env->episode_return;
    snap->rng[frame] = env->rng;
    snap->num_food = env->num_food;

    for (int i = 0; i < NUM_FISH; i++) {
        const FishAgentState* agent = &env->agents[i];
        snap->agent_x[frame][i] = agent->movement.position_cm.x;
        snap->agent_y[frame][i] = agent->movement.position_cm.y;
        snap->agent_orientation[frame][i] = agent->movement.orientation;
        snap->agent_size[frame][i] = agent->size;
        snap->agent_lin_vel[frame][i] = agent->movement.linear_velocity;
        snap->agent_ang_vel[frame][i] = agent->movement.angular_velocity;
    }

    for (int i = 0; i < env->num_food; i++) {
        snap->food_active[frame][i] = env->food[i].active ? 1 : 0;
        snap->food_x[frame][i] = env->food[i].motion.position_cm.x;
        snap->food_y[frame][i] = env->food[i].motion.position_cm.y;
    }

    size_t obs_offset = (size_t)frame * NUM_FISH * OBS_SIZE;
    memcpy(
        snap->observations + obs_offset,
        env->observations,
        (size_t)NUM_FISH * OBS_SIZE * sizeof(float)
    );
}

static double timespec_seconds(struct timespec t0, struct timespec t1) {
    return (double)(t1.tv_sec - t0.tv_sec) +
        (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

typedef struct SpsResult {
    double sps; /* Flattened env steps/sec: total_agents * vec_steps / elapsed. */
    double elapsed_s;
    long long total_steps; /* Flattened: total_agents * vec_steps. */
    int num_envs;
    int total_agents;
    int num_buffers;
    int num_threads;
    int vec_steps;
} SpsResult;

/*
 * Pure step loop matching train's StaticVec layout from wef.ini [vec]:
 *   total_agents / num_agents envs, packed into num_buffers, stepped with
 *   OpenMP the same way as vecenv (parallel buffers, workers per buffer).
 * Snapshotting and policy forward are excluded.
 *
 * SPS counts environment steps flattened over agents (each agent slot
 * advances once per vec tick).
 */
static SpsResult measure_sps(void) {
    SpsResult result = {0};
    result.num_envs = NUM_ENVS;
    result.total_agents = TOTAL_AGENTS;
    result.num_buffers = NUM_BUFFERS;
    result.num_threads = NUM_THREADS;
    result.vec_steps = SPS_STEPS;

    FishEnv* envs = (FishEnv*)calloc((size_t)NUM_ENVS, sizeof(FishEnv));
    unsigned int* action_rngs =
        (unsigned int*)calloc((size_t)NUM_ENVS, sizeof(unsigned int));
    int buffer_env_starts[NUM_BUFFERS];
    int buffer_env_counts[NUM_BUFFERS];
    if (envs == NULL || action_rngs == NULL) {
        fprintf(stderr, "failed to allocate SPS envs\n");
        free(envs);
        free(action_rngs);
        return result;
    }

    for (int e = 0; e < NUM_ENVS; e++) {
        /* Distinct seeds per env, same idea as my_vec_init's env index seed. */
        configure_env(&envs[e], (unsigned int)(e + 1));
        /* Training defaults from wef.ini [env]. */
        envs[e].configured_num_food = 64;
        envs[e].food_distribution = FOOD_RANDOM;
        c_allocate(&envs[e]);
        c_reset(&envs[e]);
        action_rngs[e] = (unsigned int)e * 0x9e3779b9u + 1u;
    }

    /* Pack envs into buffers exactly like my_vec_init in src/vecenv.h. */
    {
        int buf = 0;
        int buf_agents = 0;
        buffer_env_starts[0] = 0;
        buffer_env_counts[0] = 0;
        for (int i = 1; i < NUM_BUFFERS; i++) {
            buffer_env_starts[i] = 0;
            buffer_env_counts[i] = 0;
        }
        for (int i = 0; i < NUM_ENVS; i++) {
            buf_agents += envs[i].num_agents;
            buffer_env_counts[buf]++;
            if (buf_agents >= AGENTS_PER_BUFFER && buf < NUM_BUFFERS - 1) {
                buf++;
                buffer_env_starts[buf] = i + 1;
                buffer_env_counts[buf] = 0;
                buf_agents = 0;
            }
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int t = 0; t < SPS_STEPS; t++) {
        /* One thread group per buffer (matches buffer managers in vecenv). */
        #pragma omp parallel for schedule(static) num_threads(NUM_BUFFERS)
        for (int buf = 0; buf < NUM_BUFFERS; buf++) {
            int env_start = buffer_env_starts[buf];
            int env_count = buffer_env_counts[buf];
            int workers = WORKERS_PER_BUFFER;
            if (workers < 1) workers = 1;
            /* Within a buffer: OMP over envs (num_threads/num_buffers workers). */
            #pragma omp parallel for schedule(static) num_threads(workers) \
                if(workers > 1)
            for (int i = env_start; i < env_start + env_count; i++) {
                FishEnv* env = &envs[i];
                for (int a = 0; a < NUM_FISH; a++) {
                    float* action = env->actions + a * ACTION_SIZE;
                    for (int k = 0; k < ACTION_SIZE; k++) {
                        action[k] = random_action(&action_rngs[i]);
                    }
                }
                c_step(env);
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    for (int e = 0; e < NUM_ENVS; e++) {
        free_allocated(&envs[e]);
    }
    free(envs);
    free(action_rngs);

    result.elapsed_s = timespec_seconds(t0, t1);
    result.total_steps = (long long)TOTAL_AGENTS * (long long)SPS_STEPS;
    if (result.elapsed_s > 0.0) {
        result.sps = (double)result.total_steps / result.elapsed_s;
    }
    return result;
}

static void run_trajectory(TrajectorySnapshot* snap, unsigned int seed) {
    FishEnv env;
    configure_env(&env, seed);
    c_allocate(&env);
    c_reset(&env);

    unsigned int action_rng = seed;
    capture_state(&env, snap, 0);

    for (int t = 0; t < NUM_STEPS; t++) {
        for (int i = 0; i < NUM_FISH; i++) {
            float* action = env.actions + i * ACTION_SIZE;
            for (int a = 0; a < ACTION_SIZE; a++) {
                action[a] = random_action(&action_rng);
            }
            size_t act_offset =
                (size_t)t * NUM_FISH * ACTION_SIZE + (size_t)i * ACTION_SIZE;
            memcpy(snap->actions + act_offset, action, ACTION_SIZE * sizeof(float));
        }

        c_step(&env);

        size_t step_offset = (size_t)t * NUM_FISH;
        memcpy(
            snap->rewards + step_offset,
            env.rewards,
            (size_t)NUM_FISH * sizeof(float)
        );
        memcpy(
            snap->terminals + step_offset,
            env.terminals,
            (size_t)NUM_FISH * sizeof(float)
        );
        capture_state(&env, snap, t + 1);
    }

    free_allocated(&env);
}

static bool snapshots_equal(const TrajectorySnapshot* a, const TrajectorySnapshot* b) {
    size_t obs_n = (size_t)(NUM_STEPS + 1) * NUM_FISH * OBS_SIZE;
    size_t step_n = (size_t)NUM_STEPS * NUM_FISH;
    size_t act_n = (size_t)NUM_STEPS * NUM_FISH * ACTION_SIZE;

    if (!floats_identical(a->observations, b->observations, obs_n)) return false;
    if (!floats_identical(a->rewards, b->rewards, step_n)) return false;
    if (!floats_identical(a->terminals, b->terminals, step_n)) return false;
    if (!floats_identical(a->actions, b->actions, act_n)) return false;

    for (int frame = 0; frame <= NUM_STEPS; frame++) {
        if (a->tick[frame] != b->tick[frame]) return false;
        if (a->food_eaten[frame] != b->food_eaten[frame]) return false;
        if (a->collisions_fish[frame] != b->collisions_fish[frame]) return false;
        if (a->eod_agent_steps[frame] != b->eod_agent_steps[frame]) return false;
        if (a->rng[frame] != b->rng[frame]) return false;
        if (memcmp(&a->episode_return[frame], &b->episode_return[frame], sizeof(float)) != 0) {
            return false;
        }
        if (!doubles_identical(a->agent_x[frame], b->agent_x[frame], NUM_FISH)) {
            return false;
        }
        if (!doubles_identical(a->agent_y[frame], b->agent_y[frame], NUM_FISH)) {
            return false;
        }
        if (!doubles_identical(
                a->agent_orientation[frame], b->agent_orientation[frame], NUM_FISH
            )) {
            return false;
        }
        if (!doubles_identical(a->agent_size[frame], b->agent_size[frame], NUM_FISH)) {
            return false;
        }
        if (!doubles_identical(
                a->agent_lin_vel[frame], b->agent_lin_vel[frame], NUM_FISH
            )) {
            return false;
        }
        if (!doubles_identical(
                a->agent_ang_vel[frame], b->agent_ang_vel[frame], NUM_FISH
            )) {
            return false;
        }
        for (int i = 0; i < a->num_food; i++) {
            if (a->food_active[frame][i] != b->food_active[frame][i]) return false;
            if (memcmp(&a->food_x[frame][i], &b->food_x[frame][i], sizeof(double)) != 0) {
                return false;
            }
            if (memcmp(&a->food_y[frame][i], &b->food_y[frame][i], sizeof(double)) != 0) {
                return false;
            }
        }
    }
    return true;
}

static void report_sps(const SpsResult* sps) {
    printf("SPS benchmark (config/wef.ini [vec], env-only):\n");
    printf("\n");
    printf("  total_agents=%d  num_buffers=%d  num_threads=%d\n",
        sps->total_agents, sps->num_buffers, sps->num_threads);
    printf("  num_agents=%d  num_envs=%d  agents_per_buffer=%d  workers_per_buffer=%d\n",
        NUM_FISH, sps->num_envs, AGENTS_PER_BUFFER, WORKERS_PER_BUFFER);
    printf("  vec_steps=%d  total_steps=%lld  (flattened: total_agents * vec_steps)\n",
        sps->vec_steps, sps->total_steps);
    printf("  elapsed=%.3fs\n", sps->elapsed_s);
    printf("  SPS=%.1f  (total env steps/sec, flattened over agents)\n", sps->sps);
}

static SeedMetrics compute_metrics(
    const TrajectorySnapshot* snap, unsigned int seed
) {
    SeedMetrics m = {0};
    m.seed = seed;
    const size_t obs_n = (size_t)(NUM_STEPS + 1) * NUM_FISH * OBS_SIZE;
    const size_t rew_n = (size_t)NUM_STEPS * NUM_FISH;
    const size_t act_n = (size_t)NUM_STEPS * NUM_FISH * ACTION_SIZE;
    const float* final_obs =
        snap->observations + (size_t)NUM_STEPS * NUM_FISH * OBS_SIZE;

    for (size_t i = 0; i < (size_t)NUM_FISH * OBS_SIZE; i++) {
        m.final_obs_sum += (double)final_obs[i];
    }
    for (int t = 0; t < NUM_STEPS; t++) {
        for (int i = 0; i < NUM_FISH; i++) {
            m.reward_sum += (double)snap->rewards[t * NUM_FISH + i];
        }
    }
    m.food_eaten = snap->food_eaten[NUM_STEPS];
    m.episode_return = snap->episode_return[NUM_STEPS];
    m.rng_end = snap->rng[NUM_STEPS];
    m.obs_fnv1a = hash_bytes(snap->observations, obs_n * sizeof(float));
    m.rewards_fnv1a = hash_bytes(snap->rewards, rew_n * sizeof(float));
    m.actions_fnv1a = hash_bytes(snap->actions, act_n * sizeof(float));
    uint64_t state_hash = FNV_OFFSET;
    state_hash = fnv1a_update(state_hash, snap->agent_x, sizeof(snap->agent_x));
    state_hash = fnv1a_update(state_hash, snap->agent_y, sizeof(snap->agent_y));
    state_hash = fnv1a_update(
        state_hash, snap->agent_orientation, sizeof(snap->agent_orientation)
    );
    state_hash = fnv1a_update(state_hash, snap->agent_size, sizeof(snap->agent_size));
    state_hash = fnv1a_update(state_hash, snap->food_x, sizeof(snap->food_x));
    state_hash = fnv1a_update(state_hash, snap->food_y, sizeof(snap->food_y));
    state_hash = fnv1a_update(state_hash, snap->food_active, sizeof(snap->food_active));
    state_hash = fnv1a_update(state_hash, snap->tick, sizeof(snap->tick));
    state_hash = fnv1a_update(state_hash, snap->rng, sizeof(snap->rng));
    m.state_fnv1a = state_hash;
    return m;
}

static int seed_index(unsigned int seed) {
    for (int i = 0; i < NUM_TEST_SEEDS; i++) {
        if (TEST_SEEDS[i] == seed) return i;
    }
    return -1;
}

static bool parse_u64_hex(const char* s, uint64_t* out) {
    char* end = NULL;
    unsigned long long v = strtoull(s, &end, 0);
    if (end == s) return false;
    *out = (uint64_t)v;
    return true;
}

static bool load_baseline(const char* path, BaselineTable* table) {
    memset(table, 0, sizeof(*table));
    snprintf(table->path, sizeof(table->path), "%s", path);

    FILE* f = fopen(path, "r");
    if (f == NULL) {
        return false;
    }

    char line[512];
    int current = -1;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            if (strncmp(line, "# git_commit=", 13) == 0) {
                snprintf(
                    table->commit, sizeof(table->commit), "%s", line + 13
                );
                /* strip trailing newline */
                size_t n = strlen(table->commit);
                while (n > 0 &&
                        (table->commit[n - 1] == '\n' ||
                         table->commit[n - 1] == '\r')) {
                    table->commit[--n] = '\0';
                }
            }
            continue;
        }

        unsigned int seed = 0;
        if (sscanf(line, "[seed %u]", &seed) == 1) {
            current = seed_index(seed);
            if (current >= 0) {
                table->seed_present[current] = true;
                table->seeds[current].seed = seed;
            }
            continue;
        }
        if (current < 0) continue;

        char key[128];
        char value[128];
        if (sscanf(line, "%127[^=]=%127s", key, value) != 2) continue;

        SeedMetrics* m = &table->seeds[current];
        if (strcmp(key, "reward sum") == 0) {
            m->reward_sum = strtod(value, NULL);
        } else if (strcmp(key, "final observation sum") == 0) {
            m->final_obs_sum = strtod(value, NULL);
        } else if (strcmp(key, "food_eaten") == 0) {
            m->food_eaten = (int)strtol(value, NULL, 10);
        } else if (strcmp(key, "episode_return") == 0) {
            m->episode_return = (float)strtod(value, NULL);
        } else if (strcmp(key, "rng") == 0) {
            m->rng_end = (unsigned int)strtoul(value, NULL, 10);
        } else if (strcmp(key, "obs_fnv1a") == 0) {
            parse_u64_hex(value, &m->obs_fnv1a);
        } else if (strcmp(key, "rewards_fnv1a") == 0) {
            parse_u64_hex(value, &m->rewards_fnv1a);
        } else if (strcmp(key, "actions_fnv1a") == 0) {
            parse_u64_hex(value, &m->actions_fnv1a);
        } else if (strcmp(key, "state_fnv1a") == 0) {
            parse_u64_hex(value, &m->state_fnv1a);
        }
    }
    fclose(f);
    table->loaded = true;
    return true;
}

static bool float_close(double a, double b, double atol, double rtol) {
    double diff = fabs(a - b);
    return diff <= atol + rtol * fmax(fabs(a), fabs(b));
}

/* Returns number of mismatched fields (0 = match). */
static int compare_to_baseline(
    const SeedMetrics* got, const BaselineTable* baseline
) {
    int idx = seed_index(got->seed);
    if (idx < 0 || !baseline->loaded || !baseline->seed_present[idx]) {
        printf("  baseline:  MISSING seed=%u in %s\n", got->seed, baseline->path);
        return 1;
    }
    const SeedMetrics* exp = &baseline->seeds[idx];
    int mismatches = 0;

    printf("  baseline compare (commit %s):\n",
        baseline->commit[0] ? baseline->commit : "?");

    #define CHECK_EXACT_U64(name, field) do { \
        bool ok = (got->field == exp->field); \
        printf("    %-22s got=0x%016llx  exp=0x%016llx  %s\n", \
            name, \
            (unsigned long long)got->field, \
            (unsigned long long)exp->field, \
            ok ? "MATCH" : "DIFF"); \
        if (!ok) mismatches++; \
    } while (0)

    #define CHECK_EXACT_INT(name, field) do { \
        bool ok = (got->field == exp->field); \
        printf("    %-22s got=%d  exp=%d  %s\n", \
            name, (int)got->field, (int)exp->field, ok ? "MATCH" : "DIFF"); \
        if (!ok) mismatches++; \
    } while (0)

    #define CHECK_EXACT_UINT(name, field) do { \
        bool ok = (got->field == exp->field); \
        printf("    %-22s got=%u  exp=%u  %s\n", \
            name, (unsigned)got->field, (unsigned)exp->field, \
            ok ? "MATCH" : "DIFF"); \
        if (!ok) mismatches++; \
    } while (0)

    #define CHECK_CLOSE(name, field, atol, rtol) do { \
        bool ok = float_close( \
            (double)got->field, (double)exp->field, atol, rtol); \
        printf("    %-22s got=%.17g  exp=%.17g  diff=%.3g  %s\n", \
            name, (double)got->field, (double)exp->field, \
            fabs((double)got->field - (double)exp->field), \
            ok ? "MATCH" : "DIFF"); \
        if (!ok) mismatches++; \
    } while (0)

    CHECK_EXACT_U64("actions_fnv1a", actions_fnv1a);
    CHECK_EXACT_U64("obs_fnv1a", obs_fnv1a);
    CHECK_EXACT_U64("rewards_fnv1a", rewards_fnv1a);
    CHECK_EXACT_U64("state_fnv1a", state_fnv1a);
    CHECK_EXACT_INT("food_eaten", food_eaten);
    CHECK_EXACT_UINT("rng", rng_end);
    CHECK_CLOSE("reward sum", reward_sum, BASELINE_REWARD_ATOL, 0.0);
    CHECK_CLOSE(
        "episode_return", episode_return, BASELINE_REWARD_ATOL, 0.0
    );
    CHECK_CLOSE(
        "final observation sum",
        final_obs_sum,
        BASELINE_OBS_SUM_ATOL,
        BASELINE_OBS_SUM_RTOL
    );

    #undef CHECK_EXACT_U64
    #undef CHECK_EXACT_INT
    #undef CHECK_EXACT_UINT
    #undef CHECK_CLOSE

    printf("    baseline fields mismatched: %d\n", mismatches);
    return mismatches;
}

static void report_trajectory(
    const TrajectorySnapshot* snap, unsigned int seed, const SeedMetrics* metrics
) {
    const size_t obs_n = (size_t)(NUM_STEPS + 1) * NUM_FISH * OBS_SIZE;
    const float* final_obs =
        snap->observations + (size_t)NUM_STEPS * NUM_FISH * OBS_SIZE;
    const float* final_rewards =
        snap->rewards + (size_t)(NUM_STEPS - 1) * NUM_FISH;
    const float* final_terminals =
        snap->terminals + (size_t)(NUM_STEPS - 1) * NUM_FISH;

    double obs_min = final_obs[0];
    double obs_max = final_obs[0];
    for (size_t i = 0; i < (size_t)NUM_FISH * OBS_SIZE; i++) {
        double v = (double)final_obs[i];
        if (v < obs_min) obs_min = v;
        if (v > obs_max) obs_max = v;
    }

    double cumulative[NUM_FISH] = {0};
    bool any_terminal = false;
    for (int t = 0; t < NUM_STEPS; t++) {
        for (int i = 0; i < NUM_FISH; i++) {
            cumulative[i] += (double)snap->rewards[t * NUM_FISH + i];
            if (snap->terminals[t * NUM_FISH + i] != 0.0f) any_terminal = true;
        }
    }

    printf("------------------------------------------------------------\n");
    printf("seed=%u\n", seed);
    printf("------------------------------------------------------------\n");
    printf("\n");
    printf("  steps=%d  num_agents=%d  num_food=%d  env_rng=%u  action_seed=%u\n",
        NUM_STEPS, NUM_FISH, snap->num_food, env_rng_from_seed(seed), seed);
    printf("  obs shape=(%d, %d, %d)\n", NUM_STEPS + 1, NUM_FISH, OBS_SIZE);

    printf("\n");
    printf("  observations finite=%s\n",
        all_finite_float(snap->observations, obs_n) ? "true" : "false");
    printf("  final observation sum=%.17g\n", metrics->final_obs_sum);
    printf("  final observation min/max=%.17g / %.17g\n", obs_min, obs_max);
    printf("  reward sum=%.17g\n", metrics->reward_sum);
    printf("  cumulative rewards per fish=[");
    for (int i = 0; i < NUM_FISH; i++) {
        printf("%s%.17g", i ? ", " : "", cumulative[i]);
    }
    printf("]\n");
    printf("  any terminal=%s\n", any_terminal ? "true" : "false");

    printf("\n");
    printf("  final tick=%d\n", snap->tick[NUM_STEPS]);
    printf("  food_eaten=%d\n", metrics->food_eaten);
    printf("  collisions_fish=%d\n", snap->collisions_fish[NUM_STEPS]);
    printf("  eod_agent_steps=%d\n", snap->eod_agent_steps[NUM_STEPS]);
    printf("  episode_return=%.17g\n", (double)metrics->episode_return);
    printf("  rng=%u\n", metrics->rng_end);

    printf("\n");
    printf("  obs_fnv1a=0x%016llx\n", (unsigned long long)metrics->obs_fnv1a);
    printf("  rewards_fnv1a=0x%016llx\n", (unsigned long long)metrics->rewards_fnv1a);
    printf("  actions_fnv1a=0x%016llx\n", (unsigned long long)metrics->actions_fnv1a);
    printf("  state_fnv1a=0x%016llx\n", (unsigned long long)metrics->state_fnv1a);

    printf("\n");
    printf("  final environment state (agents):\n");
    printf("\n");
    for (int i = 0; i < NUM_FISH; i++) {
        printf(
            "    fish %d:  pos=(%.17g, %.17g)\n"
            "             ori=%.17g  size=%.17g\n"
            "             lin_vel=%.17g  ang_vel=%.17g\n",
            i,
            snap->agent_x[NUM_STEPS][i],
            snap->agent_y[NUM_STEPS][i],
            snap->agent_orientation[NUM_STEPS][i],
            snap->agent_size[NUM_STEPS][i],
            snap->agent_lin_vel[NUM_STEPS][i],
            snap->agent_ang_vel[NUM_STEPS][i]
        );
        printf("\n");
    }

    printf("  final environment state (active food):\n");
    printf("\n");
    int active_food = 0;
    for (int i = 0; i < snap->num_food; i++) {
        if (!snap->food_active[NUM_STEPS][i]) continue;
        printf(
            "    food %d:  pos=(%.17g, %.17g)\n",
            i,
            snap->food_x[NUM_STEPS][i],
            snap->food_y[NUM_STEPS][i]
        );
        active_food++;
    }
    printf("\n");
    printf("    active_food_count=%d / %d\n", active_food, snap->num_food);

    printf("\n");
    printf("  final observation (per fish, first 8 channels):\n");
    printf("\n");
    for (int i = 0; i < NUM_FISH; i++) {
        const float* obs = final_obs + i * OBS_SIZE;
        printf("    fish %d:  [", i);
        for (int c = 0; c < 8; c++) {
            printf("%s%.6g", c ? ", " : "", (double)obs[c]);
        }
        printf(", ...]\n");
    }

    printf("\n");
    printf("  final rewards:    [");
    for (int i = 0; i < NUM_FISH; i++) {
        printf("%s%.6g", i ? ", " : "", (double)final_rewards[i]);
    }
    printf("]\n");
    printf("  final terminals:  [");
    for (int i = 0; i < NUM_FISH; i++) {
        printf("%s%.6g", i ? ", " : "", (double)final_terminals[i]);
    }
    printf("]\n");
}

static int run_seed(
    unsigned int seed,
    TrajectorySnapshot* a,
    TrajectorySnapshot* b,
    const BaselineTable* baseline,
    bool require_baseline_match
) {
    memset(a, 0, sizeof(*a));
    memset(b, 0, sizeof(*b));

    run_trajectory(a, seed);
    run_trajectory(b, seed);

    int failed = 0;
    if (!snapshots_equal(a, b)) {
        fprintf(stderr, "FAIL seed=%u: two runs were not bit-identical\n", seed);
        failed = 1;
    }

    size_t obs_n = (size_t)(NUM_STEPS + 1) * NUM_FISH * OBS_SIZE;
    size_t rew_n = (size_t)NUM_STEPS * NUM_FISH;
    if (!all_finite_float(a->observations, obs_n) ||
            !all_finite_float(a->rewards, rew_n) ||
            !all_finite_float(a->terminals, rew_n)) {
        fprintf(stderr, "FAIL seed=%u: non-finite values in trajectory\n", seed);
        failed = 1;
    }

    SeedMetrics metrics = compute_metrics(a, seed);
    report_trajectory(a, seed, &metrics);

    printf("\n");
    int baseline_mismatches = 0;
    if (baseline != NULL && baseline->loaded) {
        baseline_mismatches = compare_to_baseline(&metrics, baseline);
        if (require_baseline_match && baseline_mismatches > 0) {
            failed = 1;
        }
    } else {
        printf("  baseline:  skipped (file not loaded)\n");
    }

    printf("\n");
    if (failed) {
        printf("  seed=%u RESULT: FAIL\n", seed);
    } else if (baseline != NULL && baseline->loaded && baseline_mismatches == 0) {
        printf("  seed=%u RESULT: PASS (deterministic, finite, matches baseline)\n",
            seed);
    } else if (baseline != NULL && baseline->loaded) {
        printf(
            "  seed=%u RESULT: PASS_DETERMINISM "
            "(finite + self-consistent; baseline DIFF — outcomes changed)\n",
            seed
        );
    } else {
        printf("  seed=%u RESULT: PASS (deterministic, finite)\n", seed);
    }
    printf("\n");
    return failed;
}

static void print_usage(const char* argv0) {
    printf(
        "Usage: %s [--baseline PATH] [--require-baseline-match] [--skip-sps]\n"
        "\n"
        "  --baseline PATH              Fingerprint file to compare against\n"
        "                               (default: %s)\n"
        "  --require-baseline-match     Exit non-zero if any baseline field differs\n"
        "                               (default: only determinism/finite fail the run;\n"
        "                               baseline diffs are still printed)\n"
        "  --skip-sps                   Skip the multi-env SPS timing section\n",
        argv0,
        DEFAULT_BASELINE_PATH
    );
}

static int run_test(
    const char* baseline_path, bool require_baseline_match, bool skip_sps
) {
    TrajectorySnapshot* a = (TrajectorySnapshot*)calloc(1, sizeof(TrajectorySnapshot));
    TrajectorySnapshot* b = (TrajectorySnapshot*)calloc(1, sizeof(TrajectorySnapshot));
    if (a == NULL || b == NULL) {
        fprintf(stderr, "failed to allocate trajectory snapshots\n");
        free(a);
        free(b);
        return 1;
    }

    BaselineTable baseline = {0};
    if (baseline_path != NULL) {
        if (!load_baseline(baseline_path, &baseline)) {
            fprintf(
                stderr,
                "warning: could not load baseline fingerprints from %s\n",
                baseline_path
            );
        }
    }

    printf("wef numerical stability (C)\n");
    printf("\n");
    printf("  seeds=");
    for (int i = 0; i < NUM_TEST_SEEDS; i++) {
        printf("%s%u", i ? "," : "", TEST_SEEDS[i]);
    }
    printf("  steps=%d  num_agents=%d\n", NUM_STEPS, NUM_FISH);
    if (baseline.loaded) {
        printf("  baseline=%s\n", baseline.path);
        if (baseline.commit[0]) {
            printf("  baseline_commit=%s\n", baseline.commit);
        }
        printf(
            "  require_baseline_match=%s\n",
            require_baseline_match ? "true" : "false"
        );
    } else {
        printf("  baseline=none\n");
    }
    printf("\n");

    if (!skip_sps) {
        SpsResult sps = measure_sps();
        report_sps(&sps);
        printf("\n");
    }

    int failed = 0;
    int baseline_failed_seeds = 0;
    for (int i = 0; i < NUM_TEST_SEEDS; i++) {
        int before = failed;
        failed |= run_seed(
            TEST_SEEDS[i], a, b, &baseline, require_baseline_match
        );
        if (require_baseline_match && failed != before) {
            baseline_failed_seeds++;
        }
        (void)baseline_failed_seeds;
    }

    printf("============================================================\n");
    if (failed) {
        printf("OVERALL RESULT: FAIL\n");
    } else {
        printf("OVERALL RESULT: PASS\n");
    }

    free(a);
    free(b);
    return failed;
}

int main(int argc, char** argv) {
    const char* baseline_path = DEFAULT_BASELINE_PATH;
    bool require_baseline_match = false;
    bool skip_sps = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--baseline") == 0 && i + 1 < argc) {
            baseline_path = argv[++i];
        } else if (strcmp(argv[i], "--require-baseline-match") == 0) {
            require_baseline_match = true;
        } else if (strcmp(argv[i], "--skip-sps") == 0) {
            skip_sps = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    return run_test(baseline_path, require_baseline_match, skip_sps);
}
