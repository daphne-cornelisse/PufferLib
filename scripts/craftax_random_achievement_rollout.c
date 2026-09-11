// Random-policy Craftax rollouts for achievement-probability analysis.
// Built and invoked by scripts/craftax_random_achievement_prob.py.
//
// Prints CSV: episode,length,terminal,ach_0,...,ach_66
// Achievements are 1 if unlocked during that episode, else 0.

#include "src/puffercpu.c"

#define PUF_CRAFTAX_NET 1
#include "ocean/craftax/craftax.h"

static float randn(unsigned int* rng) {
    float u1 = rng_f32(rng);
    float u2 = rng_f32(rng);
    if (u1 < 1e-12f) {
        u1 = 1e-12f;
    }
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

static Weights* make_random_weights(int hidden, int layers, unsigned int seed,
        float std) {
    int n = craftax_weight_count(hidden, layers);
    Weights* w = calloc(1, sizeof(Weights) + (size_t)(n + 7) * sizeof(float));
    if (!w) {
        return NULL;
    }
    w->data = (float*)(w + 1);
    w->size = n + 7;
    unsigned int rng = seed ? seed : 1u;
    for (int i = 0; i < n; i++) {
        w->data[i] = std * randn(&rng);
    }
    return w;
}

static void zero_mingru(CraftaxNet* net) {
    int n = net->mingru->num_layers * net->mingru->batch_size * net->mingru->hidden_size;
    memset(net->mingru->state, 0, (size_t)n * sizeof(float));
}

static void run_episode(CraftaxNet* net, int env_seed, int max_len,
        float* out_ach, int* out_len, int* out_done, int dump_rewards, int ep) {
    Craftax env = {0};
    env.rng = (unsigned int)env_seed;
    Dict kwargs = {0};
    dict_set(&kwargs, "action_mask", 1.0);
    puf_init(&env, &kwargs);

    float obs[OBS_SIZE];
    float act[1];
    float rew[1];
    float term[1];
    unsigned char mask[ATN_DIM];
    memset(obs, 0, sizeof(obs));
    memset(act, 0, sizeof(act));
    memset(rew, 0, sizeof(rew));
    memset(term, 0, sizeof(term));
    memset(mask, 1, sizeof(mask));
    env.agents[0].observations = obs;
    env.agents[0].actions = act;
    env.agents[0].rewards = rew;
    env.agents[0].terminals = term;
    env.agents[0].action_mask = mask;

    puf_reset(&env);
    zero_mingru(net);
    srand((unsigned)env_seed + 17u);

    int done = 0;
    int t = 0;
    int prev_ach[NUM_ACHIEVEMENTS];
    for (; t < max_len; t++) {
        memcpy(prev_ach, env.state.achievements, sizeof(prev_ach));
        int arm0 = equipped_armour(&env.state);
        term[0] = 0.0f;
        forward_craftax(net, obs, term, act, mask);
        puf_step(&env);
        int died = term[0] > 0.5f;
        int arm_delta = died ? 0 : (equipped_armour(&env.state) - arm0);
        if (dump_rewards) {
            char ids[512];
            ids[0] = 0;
            char* p = ids;
            for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
                int now = died ? (env.log.achievements[i] > 0.0f)
                               : env.state.achievements[i];
                if (now && !prev_ach[i]) {
                    if (p != ids) {
                        *p++ = ';';
                    }
                    p += sprintf(p, "%d", i);
                }
            }
            printf("%d,%d,%.6g,%d,%d,%s\n",
                ep, t, rew[0], died, arm_delta, ids);
        }
        if (died) {
            done = 1;
            t += 1;
            break;
        }
    }
    if (!done) {
        t = max_len;
    }

    if (done) {
        for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
            out_ach[i] = env.log.achievements[i] > 0.0f ? 1.0f : 0.0f;
        }
    } else {
        for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
            out_ach[i] = env.achievements[i] ? 1.0f : 0.0f;
        }
    }
    *out_len = t;
    *out_done = done;

    puf_close(&env);
    dict_clear(&kwargs);
}

int main(int argc, char** argv) {
    int n_episodes = 10;
    int max_len = 5000;
    int hidden = 1024;
    int layers = 4;
    unsigned int seed = 73;
    if (argc > 1) n_episodes = atoi(argv[1]);
    if (argc > 2) max_len = atoi(argv[2]);
    if (argc > 3) hidden = atoi(argv[3]);
    if (argc > 4) layers = atoi(argv[4]);
    if (argc > 5) seed = (unsigned int)strtoul(argv[5], NULL, 10);
    int dump_rewards = argc > 6 && strcmp(argv[6], "rewards") == 0;
    if (n_episodes <= 0 || max_len <= 0 || hidden <= 0 || layers <= 0) {
        fprintf(stderr,
            "usage: %s [n_episodes=10] [max_len=5000] [hidden=1024] [layers=4] [seed=73] [rewards]\n",
            argv[0]);
        return 1;
    }

    Weights* weights = make_random_weights(hidden, layers, seed, 0.02f);
    if (!weights) {
        fprintf(stderr, "alloc weights failed\n");
        return 1;
    }
    CraftaxNet* net = init_craftax_net(weights, 1, hidden, layers);
    net->greedy = 0;

    float ach[NUM_ACHIEVEMENTS];
    if (dump_rewards) {
        printf("episode,step,reward,death,armour_delta,achievement_ids\n");
    } else {
        printf("episode,length,terminal");
        for (int a = 0; a < NUM_ACHIEVEMENTS; a++) {
            printf(",ach_%d", a);
        }
        printf("\n");
    }

    for (int ep = 0; ep < n_episodes; ep++) {
        int length = 0;
        int done = 0;
        int env_seed = (int)(seed + 10007u * (unsigned)(ep + 1));
        run_episode(net, env_seed, max_len, ach, &length, &done, dump_rewards, ep);
        if (!dump_rewards) {
            printf("%d,%d,%d", ep, length, done);
            float unlocked = 0.0f;
            for (int a = 0; a < NUM_ACHIEVEMENTS; a++) {
                printf(",%.0f", ach[a]);
                unlocked += ach[a];
            }
            printf("\n");
            fprintf(stderr, "episode %d/%d length=%d terminal=%d unlocked=%.0f\n",
                ep + 1, n_episodes, length, done, unlocked);
        } else {
            fprintf(stderr, "episode %d/%d length=%d terminal=%d\n",
                ep + 1, n_episodes, length, done);
        }
        fflush(stdout);
    }
    return 0;
}
