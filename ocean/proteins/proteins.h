#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "raylib.h"

#define BINDER_LEN   16
#define FEATURE_DIM  4
#define NUM_TARGETS  2   // 0 = ON (want), 1 = OFF (avoid)

typedef struct Log {
    float perf;            // normalized specificity
    float score;           // final specificity gap
    float aff_on;
    float aff_off;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Client {
    float width;
    float height;
} Client;

typedef struct Target {
    float pocket[BINDER_LEN][FEATURE_DIM];
    float weight[BINDER_LEN];
} Target;

typedef struct Binder {
    float feats[BINDER_LEN][FEATURE_DIM];
} Binder;

typedef struct Proteins {
    Client* client;
    Log log;

    float* observations;
    float* actions;
    float* rewards;
    float* terminals;

    Target targets[NUM_TARGETS];
    Binder binder;

    int   num_agents;
    int   tick;
    int   horizon;
    int   frameskip;
    float prev_gap;
    float off_penalty;
    float edit_scale;       // step size per edit
    float episode_return;   // running sum of rewards this episode
    unsigned int rng;
} Proteins;

float compute_binding_affinity(Binder* b, Target* t);
static float specificity_gap(Proteins* env);
void compute_observations(Proteins* env);

static float frand(unsigned int* rng) {
    *rng = (*rng * 1103515245u + 12345u);
    return ((*rng >> 16) & 0x7fff) / 32767.0f;
}

void init(Proteins* env) {
    env->tick        = 0;
    env->horizon     = (env->horizon     > 0)    ? env->horizon     : 64;
    env->frameskip   = (env->frameskip   > 0)    ? env->frameskip   : 1;
    env->off_penalty = (env->off_penalty > 0.0f) ? env->off_penalty : 1.0f;
    env->edit_scale  = (env->edit_scale  > 0.0f) ? env->edit_scale  : 0.1f;
    env->num_agents  = 1;
}

void allocate(Proteins* env) {
    init(env);
    // obs = binder feats + all target pockets
    int obs_dim = BINDER_LEN * FEATURE_DIM
                + NUM_TARGETS * BINDER_LEN * FEATURE_DIM;
    env->observations = (float*)calloc(obs_dim, sizeof(float));
    env->actions      = (float*)calloc(BINDER_LEN * FEATURE_DIM, sizeof(float));
    env->rewards      = (float*)calloc(1, sizeof(float));
    env->terminals    = (float*)calloc(1, sizeof(float));
}

void c_close(Proteins* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

float compute_binding_affinity(Binder* b, Target* t) {
    float aff = 0.0f;
    for (int i = 0; i < BINDER_LEN; i++) {
        float d2 = 0.0f;
        for (int k = 0; k < FEATURE_DIM; k++) {
            float diff = b->feats[i][k] - t->pocket[i][k];
            d2 += diff * diff;
        }
        aff += t->weight[i] * expf(-d2);
    }
    return aff;
}

static float specificity_gap(Proteins* env) {
    float on  = compute_binding_affinity(&env->binder, &env->targets[0]);
    float off = compute_binding_affinity(&env->binder, &env->targets[1]);
    env->log.aff_on  = on;
    env->log.aff_off = off;
    return on - env->off_penalty * off;
}

void compute_observations(Proteins* env) {
    int idx = 0;
    for (int i = 0; i < BINDER_LEN; i++)
        for (int k = 0; k < FEATURE_DIM; k++)
            env->observations[idx++] = env->binder.feats[i][k];
    for (int t = 0; t < NUM_TARGETS; t++)
        for (int i = 0; i < BINDER_LEN; i++)
            for (int k = 0; k < FEATURE_DIM; k++)
                env->observations[idx++] = env->targets[t].pocket[i][k];
}

void add_log(Proteins* env) {
    float gap = specificity_gap(env);
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.score          += gap;
    // perf: normalize by the max achievable gap (sum of ON weights)
    float max_gap = 0.0f;
    for (int i = 0; i < BINDER_LEN; i++) max_gap += env->targets[0].weight[i];
    env->log.perf += (max_gap > 0.0f) ? gap / max_gap : 0.0f;
    env->log.n    += 1.0f;
}

void c_reset(Proteins* env) {
    env->tick = 0;
    env->episode_return = 0.0f;
    for (int t = 0; t < NUM_TARGETS; t++) {
        for (int i = 0; i < BINDER_LEN; i++) {
            env->targets[t].weight[i] = 1.0f;
            for (int k = 0; k < FEATURE_DIM; k++)
                env->targets[t].pocket[i][k] = frand(&env->rng);
        }
    }
    for (int i = 0; i < BINDER_LEN; i++)
        for (int k = 0; k < FEATURE_DIM; k++)
            env->binder.feats[i][k] = 0.5f;

    env->prev_gap = specificity_gap(env);
    compute_observations(env);
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void step_frame(Proteins* env) {
    for (int i = 0; i < BINDER_LEN; i++) {
        for (int k = 0; k < FEATURE_DIM; k++) {
            float delta = env->actions[i * FEATURE_DIM + k];
            float v = env->binder.feats[i][k] + env->edit_scale * delta;
            env->binder.feats[i][k] = clampf(v, 0.0f, 1.0f);
        }
    }
    float gap = specificity_gap(env);
    env->rewards[0] += gap - env->prev_gap;   // dense delta-specificity shaping
    env->prev_gap = gap;
}

void c_step(Proteins* env) {
    env->terminals[0] = 0.0f;
    env->rewards[0]   = 0.0f;

    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env);
    }
    env->episode_return += env->rewards[0];

    if (env->tick >= env->horizon) {
        env->terminals[0] = 1.0f;
        add_log(env);
        c_reset(env);
    }
    compute_observations(env);
}

Client* make_client(Proteins* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width  = 800;
    client->height = 600;
    InitWindow((int)client->width, (int)client->height, "PufferLib Proteins");
    SetTargetFPS(60 / env->frameskip);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Proteins* env) {
    if (env->client == NULL) env->client = make_client(env);
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (IsKeyPressed(KEY_TAB)) ToggleFullscreen();

    Client* c = env->client;
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Binder: one column of bars per residue, FEATURE_DIM cells stacked.
    float margin = 40.0f;
    float cell_w = (c->width - 2*margin) / BINDER_LEN;
    float cell_h = 26.0f;
    for (int i = 0; i < BINDER_LEN; i++) {
        for (int k = 0; k < FEATURE_DIM; k++) {
            float v = env->binder.feats[i][k];
            int x = (int)(margin + i * cell_w);
            int y = (int)(margin + k * cell_h);
            unsigned char s = (unsigned char)(v * 255.0f);
            DrawRectangle(x, y, (int)cell_w - 2, (int)cell_h - 2, (Color){s, 180, 255 - s, 255});
        }
    }

    float on  = compute_binding_affinity(&env->binder, &env->targets[0]);
    float off = compute_binding_affinity(&env->binder, &env->targets[1]);
    float gap = on - env->off_penalty * off;

    DrawText(TextFormat("ON  affinity: %.3f", on),  10, 220, 20, SKYBLUE);
    DrawText(TextFormat("OFF affinity: %.3f", off), 10, 245, 20, RED);
    DrawText(TextFormat("specificity : %.3f", gap), 10, 270, 20, gap > 0 ? GREEN : ORANGE);
    DrawText(TextFormat("tick: %d / %d", env->tick, env->horizon), 10, 295, 20, WHITE);

    EndDrawing();
}