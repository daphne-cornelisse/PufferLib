/* Click: An environment to train an RL agent to click targets fast! 
The agent has to move the mouse from one target to the next and then click.
Environment difficulty can be tuned by changing the average target size and spawning rate.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <raylib.h>

// Constants
#define MAX_TARGETS 5
#define NUM_START_TARGETS 5
#define OBS_SIZE (3 + 3 * MAX_TARGETS + 2)

#define ACTION_SIZE 3
#define TARGET_RADIUS_MIN 2
#define TARGET_RADIUS_MAX 20

// Action space: 2D movement (delta x, delta y) and click status (0 or 1)
#define CONTINUOUS 1
#define NUM_BINS 5
static float DELTA_X[NUM_BINS] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
static float DELTA_Y[NUM_BINS] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
static const float STATUS[2] = {0, 1};

// Define structs
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float targets_hit;
    float targets_total;
    float n;
} Log;

typedef struct {
    float width;
    float height;
} Client;

typedef struct {
    float x;
    float y;
    float status;
} Agent;

typedef struct {
    float x;
    float y;
    float radius;
    float spawn_time;
} Target;

typedef struct {
    Log log;
    Client* client;
    Agent agent;
    Target targets[MAX_TARGETS];
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int width;
    int height;
    int target_spawn_duration;
    unsigned int rng;
    int tick;
    int episode_length;
    int episode_return;
    int targets_hit;
    int targets_total;
    Vector2 prev_mouse;
    int human_input;
    int action_type;
} ClickEnv;

// Util functions
void add_log(ClickEnv* env) {
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->targets_hit;
    env->log.perf += (float)(env->targets_hit);
    env->log.targets_hit += env->targets_hit;
    env->log.targets_total += env->targets_total;
    env->log.n++;
}

int get_human_input(ClickEnv* env) {
    Vector2 mouse = GetMousePosition();

    // If the mouse did not move, return nothing
    if (env->prev_mouse.x < 0) {
        env->prev_mouse = mouse;
        return 0;
    }
    // Set state directly
    float move_x = mouse.x;
    float move_y = mouse.y;
    env->prev_mouse = mouse;

    int moved   = (fabsf(move_x) > 0.5f || fabsf(move_y) > 0.5f);
    int clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

    if (!moved && !clicked) return 0;

    env->actions[0] = move_x; 
    env->actions[1] = move_y;
    env->actions[2] = clicked ? 1 : 0;
    return 1;

}

void init(ClickEnv* env) {
    env->tick = 0;
}

// Environment functions
void compute_observations(ClickEnv* env) {
    // We have one agent and multiple targets.
    env->observations[0] = env->agent.x/env->width;
    env->observations[1] = env->agent.y/env->height;
    env->observations[2] = env->agent.status;

    int obs_idx = 3;
    // For each target, we store its x, y, radius and spawn time
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            env->observations[obs_idx]     = (env->targets[i].x - env->agent.x)/env->width;
            env->observations[obs_idx + 1] = (env->targets[i].y - env->agent.y)/env->height;
            env->observations[obs_idx + 2] = env->targets[i].radius/TARGET_RADIUS_MAX;
        } else {
            env->observations[obs_idx]     = 0.0f;
            env->observations[obs_idx + 1] = 0.0f;
            env->observations[obs_idx + 2] = 0.0f;
        }
        obs_idx += 3;
    }
}

void c_reset(ClickEnv* env) {
    env->tick = 0;
    env->episode_return = 0;
    env->targets_hit = 0;
    env->prev_mouse = (Vector2){ -1.0f, -1.0f };  
    env->targets_total = NUM_START_TARGETS;

    // Spawn a number of targets at random locations
    for (int i = 0; i < NUM_START_TARGETS; i++) {
        env->targets[i].x = rand_r(&env->rng) % env->width;
        env->targets[i].y = rand_r(&env->rng) % env->height;
        env->targets[i].radius = TARGET_RADIUS_MIN + rand_r(&env->rng) % (TARGET_RADIUS_MAX - TARGET_RADIUS_MIN + 1);
        env->targets[i].spawn_time = 0;
    };

    // Initialize the agent (mouse cursor) near the center of the screen
    env->agent.x = env->width / 2 + rand_r(&env->rng) % 21 - 10; 
    env->agent.y = env->height / 2 + rand_r(&env->rng) % 21 - 10;
    env->agent.status = 0; // Not clicked

    compute_observations(env);
}

static inline float clipf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void c_step(ClickEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    if (env->human_input) {
        env->agent.x = env->actions[0];
        env->agent.y = env->actions[1];
        env->agent.status = env->actions[2];
    } else {
        if (env->action_type == CONTINUOUS) {
            env->agent.x += env->actions[0];
            env->agent.y += env->actions[1];
            env->agent.status = env->actions[2];
        } else {
            env->agent.x += DELTA_X[(int)env->actions[0]];
            env->agent.y += DELTA_Y[(int)env->actions[1]];
            env->agent.status = env->actions[2];
        }
    }
    
    env->agent.x = clipf(env->agent.x, 0, env->width);
    env->agent.y = clipf(env->agent.y, 0, env->height);

    int click_status = env->agent.status;
   
    // Update environment
    // Target spawn times and remove targets that have been on the screen for too long
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            env->targets[i].spawn_time += 1;
            if (env->targets[i].spawn_time > env->target_spawn_duration) {
                env->targets[i].spawn_time = -1; // Mark target for removal 
            }
        }
        // Spawn new targets if there are less than the max number of targets
        if (env->targets[i].spawn_time == -1 && rand_r(&env->rng) % 100 < 20) { // 20% chance to spawn a new target
            env->targets[i].x = rand_r(&env->rng) % env->width;
            env->targets[i].y = rand_r(&env->rng) % env->height;
            env->targets[i].radius = TARGET_RADIUS_MIN + rand_r(&env->rng) % (TARGET_RADIUS_MAX - TARGET_RADIUS_MIN + 1);
            env->targets[i].spawn_time = 0; 
            env->targets_total += 1;
        }
    }
    
    // Check if the agent position (mouse cursor) is within the radius of any target and if the click status is 1 (clicked)
    for (int j = 0; j < MAX_TARGETS; j++) {
        if (env->targets[j].spawn_time >= 0) {
            float dist_x = env->agent.x - env->targets[j].x;
            float dist_y = env->agent.y - env->targets[j].y;
            float distance = sqrt(dist_x * dist_x + dist_y * dist_y);
            if (distance <= env->targets[j].radius && click_status == 1) {
                // Target is hit
                env->targets_hit += 1;
                env->rewards[0] += 1.0f;
                env->targets[j].spawn_time = -1; // Mark target for removal
                //printf("%d", env->targets_hit);
            }
        }
    }

    // Increase target size
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            env->targets[i].radius += 1;
        }
    }

    if (env->tick >= env->episode_length) {
        env->terminals[0] = 1; 
        add_log(env);
        c_reset(env);
        //printf("%s\n", "reset");
    } else {
        env->terminals[0] = 0; 
    }

    // printf("%d\n", env->tick);
    // printf("%d\n", env->targets_hit);
    // printf("%d\n", env->targets_total);

    // Update environment stats
    env->tick += 1;    

    compute_observations(env);
}   

void c_render(ClickEnv* env) {

    if (env->client == NULL) {
        InitWindow(env->width, env->height, "Click Environment");
        SetTargetFPS(60);
        env->client = (Client*)malloc(sizeof(Client));
        env->client->width = env->width;
        env->client->height = env->height;
    }

    Client* client = env->client;
    
    BeginDrawing();
    ClearBackground(RAYWHITE);
    
    // Draw targets
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            DrawCircle(env->targets[i].x, env->targets[i].y, env->targets[i].radius, CYAN);
        }
    }

    // Draw agent (mouse cursor)
    Color c = (env->agent.status == 1) ? BLUE : BLACK;
    float x = env->agent.x;
    float y = env->agent.y;

    // Classic arrow cursor, tip anchored at agent position
    Vector2 tip   = { x,      y      };
    Vector2 left  = { x,      y + 16 };
    Vector2 notch = { x + 4,  y + 12 };
    Vector2 right = { x + 11, y + 11 };

    DrawTriangle(tip, left, right, c);
    DrawTriangle(left, notch, right, c);
    DrawTriangleLines(tip, left, right, BLACK);
    DrawTriangleLines(left, notch, right, BLACK);

    DrawText(TextFormat("Timestep: %d", env->tick), 10, 10, 20, BLACK);
    DrawText(TextFormat("Targets hit: %d", env->targets_hit), 200, 10, 20, GREEN);
    DrawText(TextFormat("Reward: %.2f", env->rewards[0]), 420, 10, 20, BLUE);

    EndDrawing();
}

void c_close(ClickEnv* env) {
    if (env->client != NULL) {
        Client* client = env->client;
        CloseWindow();
        free(client);
    }
}
