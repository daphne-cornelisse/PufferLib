#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"

// Observation space
#define NUM_BULLETS 16
#define EGO_FEATURES 8
#define OTHER_FEATURES 8
#define OBS_SIZE EGO_FEATURES + OTHER_FEATURES

// Action space
#define NUM_ACTIONS 5 

// Action space types
#define ACCEL_IDX 0
#define TURN_RADIUS 1
#define GUN_TURN_RADIUS 2
#define RADAR_HEADING 3
#define FIREPOWER 4

static const float ACCEL_VALUES[4] = {
    -2.0f, -1.0f, 0, 1.0f
};

static const float TURN_VALUES[9] = {
    -10.0f, -6.0f, -3.0f, -1.0f, 0, 1.0f, 3.0f, 6.0f, 10.0f
};

static const float GUN_TURN_VALUES[11] = {
    -20.0f, -10.0f, -5.0f, -3.0f, -1.0f, 0, 1.0f, 3.0f, 5.0f, 15.0f, 20.0f
};

static const float RADAR_TURN_VALUES[11] = {
    -45.0f, -25.0f, -10.0f, -5.0f,  -1.0f, 0, 1.0f, 5.0f, 10.0f, 25.0f, 45.0f
};
static const float FIREPOWER_VALUES[6] = {
    0, 0.1f, 0.5f, 1.0f, 2.0f, 3.0f
};

typedef struct {
    float x;
    float y;
    float heading;
    float firepower;
    bool live;
} Bullet;

typedef struct {
    float x;
    float y;
    float v;
    float heading;
    float gun_heading;
    float radar_heading_prev;
    float radar_heading;
    float gun_heat;
    int speed;
    int health;
    int bullet_idx;
} Robot;

typedef struct {
    Robot* robots;
    Bullet* bullets;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int num_controlled_agents; 
    int width;
    int height;
    unsigned int rng;
} Robocode;

typedef struct {
    float perf;
    float episode_return;
    float episode_length;
    float score;
} Log; 

typedef struct {
    Texture2D atlas;
} Client;

Client* make_client(Robocode* env) {
    InitWindow(768, 576, "Robocode environment");
    SetTargetFPS(60);
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->atlas = LoadTexture("resources/robocode/robocode.png");
    return client;
}

void allocate_env(Robocode* env) {
    env->robots = (Robot*)calloc(env->num_agents, sizeof(Robot));
    env->bullets = (Bullet*)calloc(NUM_BULLETS*env->num_agents, sizeof(Bullet));
    env->actions = (float*)calloc(NUM_ACTIONS*env->num_agents, sizeof(float));
}

void free_env(Robocode* env) {
    free(env->robots);
    free(env->actions);
}

// Util functions
float cos_deg(float deg) {
    return cos(deg * 3.14159265358979323846 / 180.0);
}

float sin_deg(float deg) {
    return sin(deg * 3.14159265358979323846 / 180.0);
}

int scan_area(Robocode* env, Robot* robot){
    // Sweep is the signed angle traversed from radar_heading_prev to
    // radar_heading, normalized to (-180, 180]. A robot is scanned if its
    // bearing from us lies inside this wedge and within 1200 units.
    float start = robot->radar_heading_prev;
    float sweep = robot->radar_heading - start;
    if (sweep > 180.0f) sweep -= 360.0f;
    if (sweep < -180.0f) sweep += 360.0f;

    int total_robots = env->num_agents + env->num_agents;
    for (int j = 0; j < total_robots; j++) {
        Robot* other = &env->robots[j];
        if (other == robot) continue;
        if (other->health < 0) continue;

        float dx = other->x - robot->x;
        float dy = other->y - robot->y;
        if (dx*dx + dy*dy > 1200.0f*1200.0f) continue;

        float bearing = atan2f(dy, dx) * 180.0f / 3.14159265358979323846f;
        if (bearing < 0.0f) bearing += 360.0f;

        float diff = bearing - start;
        if (diff > 180.0f) diff -= 360.0f;
        if (diff < -180.0f) diff += 360.0f;

        // diff is inside the wedge when it shares sign with sweep and is shorter.
        if (diff * sweep >= 0.0f && fabsf(diff) <= fabsf(sweep)) return j;
    }
    return -1;
}

void move(Robocode* env, Robot* robot, float distance) {
    float dx = cos_deg(robot->heading);
    float dy = sin_deg(robot->heading);
    float accel = distance;

    if (accel > 1.0) {
        accel = 1.0;
    } else if (accel < -2.0) {
        accel = -2.0;
    }

    robot->v += accel;
    if (robot->v > 8.0) {
        robot->v = 8.0;
    } else if (robot->v < -8.0) {
        robot->v = -8.0;
    }

    float new_x = robot->x + dx * robot->v;
    float new_y = robot->y + dy * robot->v;

    // Collision check
    for (int j = 0; j < env->num_agents; j++) {
        Robot* target = &env->robots[j];
        if (target == robot) {
            continue;
        }
        float dx = target->x - new_x;
        float dy = target->y - new_y;
        float dist = sqrt(dx*dx + dy*dy);
        if (dist > 32.0f) {
            continue;
        }

        target->health -= 0.6;
        robot->health -= 0.6;
        return;
    }
    
    robot->x = new_x;
    robot->y = new_y;

}

float turn(Robocode* env, Robot* robot, float degrees) {
    float abs_v = fabs(robot->v);
    float d_angle = 10 - 0.75*abs_v;
    if (degrees > d_angle) {
        degrees = d_angle;
    } else if (degrees < -d_angle) {
        degrees = -d_angle;
    }

    robot->heading += degrees;
    if (robot->heading > 360) {
        robot->heading -= 360;
    } else if (robot->heading < 0) {
        robot->heading += 360;
    }
    return degrees;
}

void fire(Robocode* env, Robot* robot, float firepower) {
    if (robot->gun_heat > 0) {
        return;
    }
    if (robot->health < firepower) {
        return;
    }
    robot->health -= firepower;

    Bullet* bullet = &env->bullets[robot->bullet_idx];
    robot->bullet_idx = (robot->bullet_idx + 1) % NUM_BULLETS;
    robot->gun_heat += 1.0f + firepower/5.0f;

    bullet->x = robot->x + 64*cos_deg(robot->gun_heading);
    bullet->y = robot->y + 64*sin_deg(robot->gun_heading);
    bullet->heading = robot->gun_heading;
    bullet->firepower = firepower;
    bullet->live = true;
}

void compute_observations(Robocode* env) {
    int obs_idx = 0;
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Robot* robot = &env->robots[agent_idx];

        // Ego state
        env->observations[obs_idx++] = robot->x / env->width;
        env->observations[obs_idx++] = robot->y / env->height;
        env->observations[obs_idx++] = robot->heading * DEG2RAD;
        env->observations[obs_idx++] = robot->gun_heading * DEG2RAD;
        env->observations[obs_idx++] = robot->radar_heading * DEG2RAD;
        env->observations[obs_idx++] = robot->radar_heading_prev * DEG2RAD;
        env->observations[obs_idx++] = robot->speed;
        env->observations[obs_idx++] = robot->health / 100.0f; 

        // Partner observation
        // Note: Current scan implementation is hardcoded for two-player settings.
        int scanned = scan_area(env, robot);
        if (scanned < 0) {
            // Zero out the partner observations if no robot was in its field of view
            memset(&env->observations[obs_idx++], 0, obs_idx+OTHER_FEATURES * sizeof(float));
            obs_idx += OTHER_FEATURES;
            continue;   
        }

        Robot* other = &env->robots[scanned];

        // Relative position rotated into ego (body) frame. Engine convention:
        // cos_deg -> x, sin_deg -> y, so forward = (cos h, sin h), right = (sin h, -cos h).
        float dx_w = other->x - robot->x;
        float dy_w = other->y - robot->y;
        float c = cos_deg(robot->heading);
        float s = sin_deg(robot->heading);
        float dx_ego =  c*dx_w + s*dy_w;   // forward
        float dy_ego = -s*dx_w + c*dy_w;   // perpendicular (matches drive's R(-h))
        // Relative headings: wrap raw delta (in (-360, 360)) to (-180, 180]
        // then scale to [-1, 1].
        float dh_body  = other->heading - robot->heading;
        float dh_gun   = other->heading - robot->gun_heading;
        float dh_radar = other->heading - robot->radar_heading;
        if (dh_body  >  180.0f) dh_body  -= 360.0f; else if (dh_body  < -180.0f) dh_body  += 360.0f;
        if (dh_gun   >  180.0f) dh_gun   -= 360.0f; else if (dh_gun   < -180.0f) dh_gun   += 360.0f;
        if (dh_radar >  180.0f) dh_radar -= 360.0f; else if (dh_radar < -180.0f) dh_radar += 360.0f;
        // Aim error: bearing to target (world) minus my gun heading, wrapped to (-180, 180].
        // ~0 means gun is pointed at the target.
        float bearing = atan2f(dy_w, dx_w) * 180.0f / 3.14159265358979323846f;
        float aim_err = bearing - robot->gun_heading;
        if (aim_err >  180.0f) aim_err -= 360.0f;
        else if (aim_err < -180.0f) aim_err += 360.0f;
        
        env->observations[obs_idx++] = dx_ego / 1200.0f;
        env->observations[obs_idx++] = dy_ego / 1200.0f;
        env->observations[obs_idx++] = dh_body  * DEG2RAD;
        env->observations[obs_idx++] = dh_gun   * DEG2RAD;
        env->observations[obs_idx++] = dh_radar * DEG2RAD;
        env->observations[obs_idx++] = other->health / 100.0f;
        env->observations[obs_idx++] = aim_err * DEG2RAD;
        env->observations[obs_idx++] = 1.0f;
    }
}


void c_reset(Robocode* env) {
    
    // Initialize robot positions, headings, and health
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        
        float x = 16 + rand() % (env->width-32);
        float y = 16 + rand() % (env->height-32);
        robot->x = x;
        robot->y = y;
        robot->v = 0;
        robot->heading = rand() % 360;
        robot->gun_heading = robot->heading;
        robot->radar_heading_prev = robot->heading;
        robot->radar_heading = robot->heading;
        robot->gun_heat = 3.0f;
        robot->speed = 0;
        robot->health = 100;
        robot->bullet_idx = 0;
    }

    // Reset bullets (TODO)

    compute_observations(env);

}

void c_step(Robocode* env) {

    // Update state for all controlled agents in env
    for (int agent_idx = 0; agent_idx <= env->num_controlled_agents; agent_idx++) {
        Robot* robot = &env->robots[agent_idx];
        if (robot->health <= 0) {
            c_reset(env);
            return;
        }

        for (int blt = 0; blt < NUM_BULLETS; blt++) {
            Bullet* bullet = &env->bullets[agent_idx*NUM_BULLETS + blt];
            if (!bullet->live) {
                continue;
            }

            float v = 20.0f - 3.0f*bullet->firepower;
            bullet->x += v*cos_deg(bullet->heading);
            bullet->y += v*sin_deg(bullet->heading);

            // Bounds check
            if (bullet->x < 0 || bullet->x > env->width
                    || bullet->y < 0 || bullet->y > env->height) {
                bullet->live = false;
                continue;
            }

            // Collision check
            for (int j = 0; j < env->num_agents; j++) {
                Robot* target = &env->robots[j];
                float dx = target->x - bullet->x;
                float dy = target->y - bullet->y;
                float dist = sqrt(dx*dx + dy*dy);
                if (dist > 32.0f) {
                    continue;
                }

                float damage = 4*bullet->firepower;
                if (bullet->firepower > 1.0f) {
                    damage += 2*(bullet->firepower - 1.0f);
                }

                target->health -= damage;
                robot->health += 3*bullet->firepower;
                bullet->live = false;
            }
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        int atn_offset = i*NUM_ACTIONS;

        // Cool down gun
        if (robot->gun_heat > 0) {
            robot->gun_heat -= 0.1f;
        }

        // Move
        int move_atn = env->actions[atn_offset];
        move(env, robot, move_atn);

        // Turn
        int turn_atn = env->actions[atn_offset + 1];
        float turn_degrees = turn(env, robot, turn_atn);

        // Gun 
        float gun_degrees = env->actions[atn_offset + 2] + turn_degrees;
        robot->gun_heading += gun_degrees;
        if (robot->gun_heading > 360) {
            robot->gun_heading -= 360;
        } else if (robot->gun_heading < 0) {
            robot->gun_heading += 360;
        }

        // Radar
        float radar_degrees = env->actions[atn_offset + 3] + gun_degrees;
        robot->radar_heading_prev = robot->radar_heading;
        robot->radar_heading += radar_degrees;
        if (robot->radar_heading > 360) {
            robot->radar_heading -= 360;
        } else if (robot->radar_heading < 0) {
            robot->radar_heading += 360;
        }

        // Fire
        float firepower = env->actions[atn_offset + 4];
        if (firepower > 0) {
            fire(env, robot, firepower);
        }

        // Clip position
        if (robot->x < 16) {
            robot->x = 16;
        } else if (robot->x > env->width - 16) {
            robot->x = env->width - 16;
        }
        if (robot->y < 16) {
            robot->y = 16;
        } else if (robot->y > env->height - 16) {
            robot->y = env->height - 16;
        }
    }
}

void close_client(Client* client) {
    UnloadTexture(client->atlas);
    CloseWindow();
}

void c_render(Client* client, Robocode* env) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int x = 0; x < env->width; x+=64) {
        for (int y = 0; y < env->height; y+=64) {
            int src_x = 64 * ((x*33409+ y*30971) % 5);
            Rectangle src_rect = (Rectangle){src_x, 0, 64, 64};
            Vector2 dest_pos = (Vector2){x, y};
            DrawTextureRec(client->atlas, src_rect, dest_pos, WHITE);
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        int atn_offset = i*NUM_ACTIONS;
        int turn_atn = env->actions[atn_offset + 1];
        int gun_atn = env->actions[atn_offset + 2] + turn_atn;
        int radar_atn = env->actions[atn_offset + 3] + gun_atn;

        Robot robot = env->robots[i];
        Vector2 robot_pos = (Vector2){robot.x, robot.y};

        // Radar
        float radar_left = (radar_atn > 0) ? robot.radar_heading: robot.radar_heading_prev;
        float radar_right = (radar_atn > 0) ? robot.radar_heading_prev : robot.radar_heading;
        Vector2 radar_left_pos = (Vector2){
            robot.x + 1200*cos_deg(radar_left),
            robot.y + 1200*sin_deg(radar_left)
        };
        Vector2 radar_right_pos = (Vector2){
            robot.x + 1200*cos_deg(radar_right),
            robot.y + 1200*sin_deg(radar_right)
        };
        DrawTriangle(robot_pos, radar_left_pos, radar_right_pos, (Color){0, 255, 0, 128});

        // Gun 
        Vector2 gun_pos = (Vector2){
            robot.x + 64*cos_deg(robot.gun_heading),
            robot.y + 64*sin_deg(robot.gun_heading)
        };
        //DrawLineEx(robot_pos, gun_pos, 4, WHITE);

        // Robot
        //DrawCircle(robot.x, robot.y, 32, RED);
        //DrawCircle(robot.x, robot.y, 16, WHITE);
        float theta = robot.heading;
        float dx = cos_deg(theta);
        float dy = sin_deg(theta);
        int src_y = 64 + 64*(i%2);
        Rectangle body_rect = (Rectangle){0, src_y, 64, 64};
        Rectangle radar_rect = (Rectangle){64, src_y, 64, 64};
        Rectangle gun_rect = (Rectangle){128, src_y, 64, 64};
        Rectangle dest_rect = (Rectangle){robot.x, robot.y, 64, 64};
        Vector2 origin = (Vector2){32, 32};
        DrawTexturePro(client->atlas, body_rect, dest_rect, origin, robot.heading+90, WHITE);
        DrawTexturePro(client->atlas, radar_rect, dest_rect, origin, robot.radar_heading+90, WHITE);
        DrawTexturePro(client->atlas, gun_rect, dest_rect, origin, robot.gun_heading+90, WHITE);

        DrawText(TextFormat("%i", robot.health), robot.x-16, robot.y-48, 12, WHITE);
    }

    for (int i = 0; i < env->num_agents*NUM_BULLETS; i++) {
        Bullet bullet = env->bullets[i];
        if (!bullet.live) {
            continue;
        }
        Vector2 bullet_pos = (Vector2){bullet.x, bullet.y};
        DrawCircleV(bullet_pos, 4, WHITE);
    }

    EndDrawing();
}