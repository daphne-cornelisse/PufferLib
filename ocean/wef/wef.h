/*
 * Core mechanics and electrostatic biophysics for the weakly electric fish
 * environment.
 *
 * This is a port of the biophysics simulation implemented in
 * https://github.com/KempnerInstitute/wef
 *   onpolicy/custom/fish/movement.py
 *   onpolicy/custom/fish/electric.py
 *   onpolicy/custom/fish/electric_accelerated.py
 *   onpolicy/custom/fish/electric_scene.py
 *
 * Positions and radii use centimeters, matching the Python environment.
 * Electric-field calculations convert positions to meters internally and
 * return V/m. Dipole moments use C*m and monopole charges use C.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

// Constants
#define PI_F 3.14159265358979323846f
#define CM_TO_M 0.01f
#define K_COULOMB 8.99e9f
#define EPSILON_0 8.854e-12f
#define FIELD_EPS_M 1e-5f
#define REFLECTION_SCALE 1.0f
#define SENSOR_EPS 1e-25f

// Values from upstream cfg.py and Table 1 of the accompanying publication
#define SIMULATION_HZ 83.0f
#define BODY_RADIUS_CM 1.0f
#define FOOD_RADIUS_CM 0.25f
#define CONDUCTOR_CONTRAST -0.5f
#define EOD_CHARGE_C 1.11e-15f
#define EOD_POLE_OFFSET_CM 0.5f
#define INTRINSIC_MOMENT_C_M 1.11e-23f
#define FOOD_INTRINSIC_MOMENT_C_M 1.11e-24f

#define NUM_MORMYROMASTS 36
#define NUM_AMPULLARY 24
#define NUM_KNOLLEN 12
#define MAX_AGENTS 4
#define MAX_FOOD 64
#define MAX_DIPOLE_SOURCES (2 * MAX_AGENTS + 2 * MAX_FOOD)
#define OBS_SIZE 110
#define ACTION_SIZE 4
#define EATING_RADIUS_CM 2.0f
#define BITING_RADIUS_CM 3.0f
#define EATING_ANGLE (PI_F / 4.0f)
#define EAT_REWARD 10.0f
#define COLLISION_REWARD -0.5f
#define BITTEN_REWARD -5.0f
#define MAX_PATCHES 90
#define TRACE_LENGTH 128

#define EAT_COOLDOWN_STEPS 3
#define BITE_COOLDOWN_STEPS 5

#define AMPULLARY_MIN_VM 2e-10f
#define AMPULLARY_MAX_VM 2e-8f
#define MORMYROMAST_MIN_VM 5e-8f
#define MORMYROMAST_MAX_VM 5e-2f
#define KNOLLEN_MIN_VM 2e-7f

// Approximate influence range for first-order 
// wall images (cm) for electric field calculations.
#define REFLECTION_WALL_RANGE_CM 100.0f

#define MOTION_FIRST_ORDER 1
#define MOTION_SECOND_ORDER 2

// Arena wall indices for first-order image charges. 
#define WALL_LEFT 0
#define WALL_RIGHT 1
#define WALL_BOTTOM 2
#define WALL_TOP 3
#define NUM_WALLS 4

/* Temporary propose/commit payload for agent motion. */
typedef struct FishMotionProposal {
    float pos_x;
    float pos_y;
    float orientation;
    float linear_velocity;
    float angular_velocity;
} FishMotionProposal;

/* All per-fish state: pose, dynamics, electric sources, sensors, actions. */
typedef struct FishAgent {
    float pos_x;
    float pos_y;
    float orientation;
    float max_linear_velocity;
    float max_angular_velocity;

    float disp_ground_x;
    float disp_ground_y;
    float disp_ego_x;
    float disp_ego_y;
    float last_orientation;
    float linear_velocity;
    float angular_velocity;

    float body_radius_cm;
    float min_linear_velocity;
    float max_linear_acceleration;
    float min_angular_velocity;
    float max_angular_acceleration;
    float linear_drag_factor;
    float angular_drag_factor;

    int motion_order;
    bool backwards;
    bool collided;

    float size;
    int bite_cooldown;  /* remaining steps; 0 = ready */
    int eat_cooldown;   /* remaining steps; 0 = free to move / eat */
    bool emits_eod;
    bool bite_action;
    bool was_bitten;
    bool ate_food;
    float last_action[ACTION_SIZE];
    float ampullary_ema[NUM_AMPULLARY];

    /* EOD monopole pair in world frame (updated each scene rebuild). */
    float eod_pos_x[2];
    float eod_pos_y[2];
    float eod_charge[2];

    /* Body dipoles in world frame (position is agent pos). */
    float intrinsic_moment_x;
    float intrinsic_moment_y;
    float induced_moment_x;
    float induced_moment_y;

    /* Local body-frame sensor geometry (set once at spawn). */
    float morm_local_x[NUM_MORMYROMASTS];
    float morm_local_y[NUM_MORMYROMASTS];
    float morm_nx[NUM_MORMYROMASTS];
    float morm_ny[NUM_MORMYROMASTS];
    float amp_local_x[NUM_AMPULLARY];
    float amp_local_y[NUM_AMPULLARY];
    float amp_nx[NUM_AMPULLARY];
    float amp_ny[NUM_AMPULLARY];
    float knollen_local_x[NUM_KNOLLEN];
    float knollen_local_y[NUM_KNOLLEN];
    float knollen_nx[NUM_KNOLLEN];
    float knollen_ny[NUM_KNOLLEN];
} FishAgent;

float vec_length_squared(float x, float y) {
    return x * x + y * y;
}

float vec_length(float x, float y) {
    return sqrtf(vec_length_squared(x, y));
}

float clamp(float value, float minimum, float maximum) {
    return fminf(maximum, fmaxf(minimum, value));
}

/* Stable wrap to [-pi, pi], identical to movement._wrap_angle. */
float wrap_angle(float angle) {
    return atan2f(sinf(angle), cosf(angle));
}

/* Rotate a vector; add translation separately when transforming a position. */
void rotate(float x, float y, float angle, float* out_x, float* out_y) {
    float c = cosf(angle);
    float s = sinf(angle);
    *out_x = c * x - s * y;
    *out_y = s * x + c * y;
}

void transform_position(
    float local_x, float local_y,
    float origin_x, float origin_y,
    float angle,
    float* out_x, float* out_y
) {
    float rx, ry;
    rotate(local_x, local_y, angle, &rx, &ry);
    *out_x = rx + origin_x;
    *out_y = ry + origin_y;
}

void world_sensor(
    float local_x, float local_y,
    float local_nx, float local_ny,
    float origin_x, float origin_y,
    float angle,
    float* out_x, float* out_y,
    float* out_nx, float* out_ny
) {
    transform_position(
        local_x, local_y, origin_x, origin_y, angle, out_x, out_y
    );
    rotate(local_nx, local_ny, angle, out_nx, out_ny);
}

FishMotionProposal propose_motion(
    const FishAgent* fish,
    float move_command,
    float turn_command,
    bool eating_frozen
) {
    FishMotionProposal proposal = {
        .pos_x = fish->pos_x,
        .pos_y = fish->pos_y,
        .orientation = fish->orientation,
        .linear_velocity = 0.0f,
        .angular_velocity = 0.0f,
    };

    if (fish->motion_order == MOTION_FIRST_ORDER) {
        float move = eating_frozen ? 0.0f : move_command;
        float turn = eating_frozen ? 0.0f : turn_command;
        proposal.angular_velocity = turn * fish->max_angular_velocity;
        proposal.linear_velocity = move * fish->max_linear_velocity;
    } else { // Second-order motion with acceleration and drag
        proposal.linear_velocity =
            fish->linear_velocity +
            move_command * fish->max_linear_acceleration;
        proposal.angular_velocity =
            fish->angular_velocity +
            turn_command * fish->max_angular_acceleration;

        proposal.linear_velocity = clamp(
            proposal.linear_velocity,
            fish->min_linear_velocity,
            fish->max_linear_velocity
        );
        if (!fish->backwards && proposal.linear_velocity < 0.0f) {
            proposal.linear_velocity = 0.0f;
        }

        proposal.linear_velocity *= fish->linear_drag_factor;
        proposal.angular_velocity *= fish->angular_drag_factor;
        proposal.angular_velocity = clamp(
            proposal.angular_velocity,
            fish->min_angular_velocity,
            fish->max_angular_velocity
        );

        if (eating_frozen) {
            proposal.linear_velocity = 0.0f;
            proposal.angular_velocity = 0.0f;
        }
    }

    proposal.orientation = wrap_angle(fish->orientation + proposal.angular_velocity);
    float hx = cosf(proposal.orientation);
    float hy = sinf(proposal.orientation);
    proposal.pos_x = fish->pos_x + hx * proposal.linear_velocity;
    proposal.pos_y = fish->pos_y + hy * proposal.linear_velocity;
    return proposal;
}

bool motion_collides(
    float proposed_x,
    float proposed_y,
    float body_radius_cm,
    const FishAgent* others,
    size_t num_others,
    size_t self_index
) {
    for (size_t i = 0; i < num_others; i++) {
        if (i == self_index) {
            continue;
        }
        float radius = body_radius_cm + others[i].body_radius_cm;
        float dx = proposed_x - others[i].pos_x;
        float dy = proposed_y - others[i].pos_y;
        if (vec_length_squared(dx, dy) < radius * radius) {
            return true;
        }
    }
    return false;
}

/*
 * Commit a proposal and enforce a rectangular arena. Orientation always
 * changes even when translation is blocked, allowing turning in place.
 */
bool commit_motion(
    FishAgent* fish,
    FishMotionProposal proposal,
    bool collided,
    float arena_size_x,
    float arena_size_y
) {
    float previous_x = fish->pos_x;
    float previous_y = fish->pos_y;
    float previous_orientation = fish->orientation;

    fish->last_orientation = previous_orientation;
    fish->orientation = proposal.orientation;
    if (!collided) {
        fish->pos_x = proposal.pos_x;
        fish->pos_y = proposal.pos_y;
    }

    float unclipped_x = fish->pos_x;
    float unclipped_y = fish->pos_y;
    fish->pos_x = clamp(
        fish->pos_x,
        fish->body_radius_cm,
        arena_size_x - fish->body_radius_cm
    );
    fish->pos_y = clamp(
        fish->pos_y,
        fish->body_radius_cm,
        arena_size_y - fish->body_radius_cm
    );
    bool hit_wall = fish->pos_x != unclipped_x || fish->pos_y != unclipped_y;
    bool stopped = collided || hit_wall;

    if (fish->motion_order == MOTION_SECOND_ORDER) {
        fish->linear_velocity =
            stopped ? 0.0f : proposal.linear_velocity;
        fish->angular_velocity =
            stopped ? 0.0f : proposal.angular_velocity;
    }

    fish->collided = collided;
    fish->disp_ground_x = fish->pos_x - previous_x;
    fish->disp_ground_y = fish->pos_y - previous_y;
    rotate(
        fish->disp_ground_x, fish->disp_ground_y,
        -previous_orientation,
        &fish->disp_ego_x, &fish->disp_ego_y
    );
    return hit_wall;
}

bool point_in_forward_cone(
    float pos_x,
    float pos_y,
    float orientation,
    float target_x,
    float target_y,
    float radius_cm,
    float cone_angle
) {
    float dx = target_x - pos_x;
    float dy = target_y - pos_y;
    if (vec_length_squared(dx, dy) >= radius_cm * radius_cm) {
        return false;
    }
    float bearing = atan2f(dy, dx);
    return fabsf(wrap_angle(bearing - orientation)) <= cone_angle * 0.5f;
}

// Electrostatic field model
/*
 * Field from monopole and dipole sources (SoA). Matches
 * electric.measure_electric_field_original, including adding eps_m to
 * distance rather than squared distance. Writes V/m into out_x/out_y.
 */
void measure_electric_field(
    float measurement_x,
    float measurement_y,
    const float* mono_x,
    const float* mono_y,
    const float* mono_q,
    size_t num_monopoles,
    const float* dip_x,
    const float* dip_y,
    const float* dip_mx,
    const float* dip_my,
    size_t num_dipoles,
    float eps_m,
    float* out_x,
    float* out_y
) {
    float measurement_mx = measurement_x * CM_TO_M;
    float measurement_my = measurement_y * CM_TO_M;
    float field_x = 0.0f;
    float field_y = 0.0f;

    for (size_t i = 0; i < num_monopoles; i++) {
        float source_mx = mono_x[i] * CM_TO_M;
        float source_my = mono_y[i] * CM_TO_M;
        float ox = measurement_mx - source_mx;
        float oy = measurement_my - source_my;
        float distance = vec_length(ox, oy) + eps_m;
        float weight =
            K_COULOMB * mono_q[i] / (distance * distance * distance);
        field_x += ox * weight;
        field_y += oy * weight;
    }

    for (size_t i = 0; i < num_dipoles; i++) {
        float source_mx = dip_x[i] * CM_TO_M;
        float source_my = dip_y[i] * CM_TO_M;
        float ox = measurement_mx - source_mx;
        float oy = measurement_my - source_my;
        float distance = vec_length(ox, oy) + eps_m;
        float r2 = distance * distance;
        float r3 = r2 * distance;
        float r5 = r3 * r2;
        float moment_dot_offset = dip_mx[i] * ox + dip_my[i] * oy;
        float axial = 3.0f * moment_dot_offset / r5;
        field_x += K_COULOMB * (ox * axial - dip_mx[i] / r3);
        field_y += K_COULOMB * (oy * axial - dip_my[i] / r3);
    }
    *out_x = field_x;
    *out_y = field_y;
}

void induce_dipole_moment(
    float external_field_x,
    float external_field_y,
    float conductor_radius_cm,
    float conductor_contrast,
    float* out_x,
    float* out_y
) {
    float radius_m = conductor_radius_cm * CM_TO_M;
    float volume_m3 = (4.0f / 3.0f) * PI_F * radius_m * radius_m * radius_m;
    float scale = 3.0f * EPSILON_0 * volume_m3 * conductor_contrast;
    *out_x = external_field_x * scale;
    *out_y = external_field_y * scale;
}

/* Perpendicular distance from a point to a rectangular arena wall. */
static inline float distance_to_wall_cm(
    float pos_x, float pos_y, float arena_size_x, float arena_size_y, int wall
) {
    if (wall == WALL_LEFT) return pos_x;
    if (wall == WALL_RIGHT) return arena_size_x - pos_x;
    if (wall == WALL_BOTTOM) return pos_y;
    if (wall == WALL_TOP) return arena_size_y - pos_y;
    return 0.0f;
}

static inline void reflect_source_across_wall(
    float* pos_x,
    float* pos_y,
    float* moment_x,
    float* moment_y,
    float* charge_c,
    float arena_size_x,
    float arena_size_y,
    int wall,
    float reflection_scale,
    bool flip_on_reflection
) {
    if (wall == WALL_LEFT) {
        *pos_x = -(*pos_x);
        if (moment_x != NULL && flip_on_reflection) {
            *moment_x = -(*moment_x);
        }
    } else if (wall == WALL_RIGHT) {
        *pos_x = 2.0f * arena_size_x - (*pos_x);
        if (moment_x != NULL && flip_on_reflection) {
            *moment_x = -(*moment_x);
        }
    } else if (wall == WALL_BOTTOM) {
        *pos_y = -(*pos_y);
        if (moment_y != NULL && flip_on_reflection) {
            *moment_y = -(*moment_y);
        }
    } else if (wall == WALL_TOP) {
        *pos_y = 2.0f * arena_size_y - (*pos_y);
        if (moment_y != NULL && flip_on_reflection) {
            *moment_y = -(*moment_y);
        }
    }

    if (charge_c != NULL) {
        *charge_c *= reflection_scale * (flip_on_reflection ? -1.0f : 1.0f);
    }
    if (moment_x != NULL && moment_y != NULL) {
        *moment_x *= reflection_scale;
        *moment_y *= reflection_scale;
    }
}

// Field from the original sources plus first-order wall images (near walls only).
void measure_electric_field_with_reflections(
    float measurement_x,
    float measurement_y,
    const float* mono_x,
    const float* mono_y,
    const float* mono_q,
    size_t num_monopoles,
    const float* dip_x,
    const float* dip_y,
    const float* dip_mx,
    const float* dip_my,
    size_t num_dipoles,
    float arena_size_x,
    float arena_size_y,
    float eps_m,
    float reflection_scale,
    bool flip_on_reflection,
    float* out_x,
    float* out_y
) {
    measure_electric_field(
        measurement_x, measurement_y,
        mono_x, mono_y, mono_q, num_monopoles,
        dip_x, dip_y, dip_mx, dip_my, num_dipoles,
        eps_m, out_x, out_y
    );

    int near_walls[NUM_WALLS];
    int num_near_walls = 0;
    for (int wall = 0; wall < NUM_WALLS; wall++) {
        float dist = distance_to_wall_cm(
            measurement_x, measurement_y, arena_size_x, arena_size_y, wall
        );
        if (dist <= REFLECTION_WALL_RANGE_CM) {
            near_walls[num_near_walls++] = wall;
        }
    }
    if (num_near_walls == 0) {
        return;
    }

    for (int w = 0; w < num_near_walls; w++) {
        int wall = near_walls[w];
        for (size_t i = 0; i < num_monopoles; i++) {
            float ix = mono_x[i];
            float iy = mono_y[i];
            float iq = mono_q[i];
            reflect_source_across_wall(
                &ix, &iy, NULL, NULL, &iq,
                arena_size_x, arena_size_y,
                wall, reflection_scale, flip_on_reflection
            );
            float cx, cy;
            measure_electric_field(
                measurement_x, measurement_y,
                &ix, &iy, &iq, 1,
                NULL, NULL, NULL, NULL, 0,
                eps_m, &cx, &cy
            );
            *out_x += cx;
            *out_y += cy;
        }
        for (size_t i = 0; i < num_dipoles; i++) {
            float ix = dip_x[i];
            float iy = dip_y[i];
            float imx = dip_mx[i];
            float imy = dip_my[i];
            reflect_source_across_wall(
                &ix, &iy, &imx, &imy, NULL,
                arena_size_x, arena_size_y,
                wall, reflection_scale, flip_on_reflection
            );
            float cx, cy;
            measure_electric_field(
                measurement_x, measurement_y,
                NULL, NULL, NULL, 0,
                &ix, &iy, &imx, &imy, 1,
                eps_m, &cx, &cy
            );
            *out_x += cx;
            *out_y += cy;
        }
    }
}

// Receptor geometry and transduction
float project_field(
    float field_x, float field_y, float normal_x, float normal_y
) {
    return field_x * normal_x + field_y * normal_y;
}

float
uniform_sensor_angle(size_t sensor_index, size_t num_sensors) {
    return 2.0f * PI_F * (float)sensor_index / (float)num_sensors;
}

/*
 * Mormyromast layout from sensing.calculate_mormyromast_angles: 30% of the
 * receptors span the forward-facing chin region, with the rest covering the
 * remaining circumference. Pass num_chin=10, num_rest=26 for the default 36.f
 */
float mormyromast_angle(
    size_t sensor_index,
    size_t num_chin,
    size_t num_rest,
    float chin_angle
) {
    if (sensor_index < num_chin) {
        if (num_chin == 1) {
            return 0.0f;
        }
        return -0.5f * chin_angle +
            chin_angle * (float)sensor_index / (float)(num_chin - 1);
    }
    size_t rest_index = sensor_index - num_chin;
    if (num_rest == 1) {
        return PI_F;
    }
    return 0.5f * chin_angle +
        (2.0f * PI_F - chin_angle) *
        (float)rest_index / (float)num_rest;
}

/*
 * Sign-preserving logarithmic normalization used by all receptor channels.
 * Values below threshold map to signed zero and values above maximum saturate.
 */
float normalize_sensor_reading(
    float reading,
    float sensor_min,
    float sensor_max,
    float eps
) {
    if (reading == 0.0f) {
        return 0.0f;
    }
    float r = (float)reading;
    float sign = r < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabsf(r);
    magnitude = clamp(magnitude, (float)sensor_min, (float)sensor_max);
    magnitude = fmaxf(magnitude, (float)eps);
    float denominator =
        log10f((float)sensor_max) - log10f((float)sensor_min);
    if (denominator <= 0.0f) {
        return 0.0f;
    }
    float normalized =
        (log10f(magnitude) - log10f((float)sensor_min)) / denominator;
    return (float)(sign * clamp(normalized, 0.0f, 1.0f));
}

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float food_eaten_mean;
    float eod_rate;
    float collisions_fish;
    float n;
} Log;

typedef struct Trace {
    float pos_x[TRACE_LENGTH];
    float pos_y[TRACE_LENGTH];
    int index;
    int count;
} Trace;

typedef struct Client {
    int window_width;
    int window_height;
    int margin;
    bool show_field;
    bool show_sensors;
    Trace traces[MAX_AGENTS];
} Client;

typedef struct FishFood {
    float pos_x;
    float pos_y;
    float orientation;
    bool active;
    // Dipoles in world frame
    float intrinsic_moment_x;
    float intrinsic_moment_y;
    float induced_moment_x;
    float induced_moment_y;
} FishFood;

typedef enum FoodDistribution {
    FOOD_UNIFORM,
    FOOD_PATCHY,
    FOOD_RANDOM,
} FoodDistribution;

typedef struct FishEnv {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int tick;
    int episode_length;
    unsigned int rng;

    float arena_size_x;
    float arena_size_y;
    float min_arena_size_x;
    float min_arena_size_y;
    float max_arena_size_x;
    float max_arena_size_y;
    float electric_field_radius_cm;
    FoodDistribution food_distribution;
    int configured_num_food;
    float patch_radius_cm;
    float patch_radius_std_cm;
    float patch_density;
    FishAgent agents[MAX_AGENTS];
    FishFood food[MAX_FOOD];
    int num_food;
    int food_eaten;
    int eod_agent_steps;
    int collisions_fish;
    float episode_return;

    float mormyromast_cd[NUM_MORMYROMASTS];
    float amp_intrinsic_baseline[NUM_AMPULLARY];
    Client* client;
} FishEnv;

float random_uniform(FishEnv* env, float low, float high) {
    float unit = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return low + (high - low) * unit;
}

float random_multiplier(FishEnv* env, float fraction) {
    return (float)random_uniform(env, 1.0f - (float)fraction, 1.0f + (float)fraction);
}

/* Pack EOD monopoles from all agents into SoA buffers. */
static int pack_eod_sources(
    const FishEnv* env, float* x, float* y, float* q
) {
    int n = 0;
    for (int i = 0; i < env->num_agents; i++) {
        for (int p = 0; p < 2; p++) {
            x[n] = env->agents[i].eod_pos_x[p];
            y[n] = env->agents[i].eod_pos_y[p];
            q[n] = env->agents[i].eod_charge[p];
            n++;
        }
    }
    return n;
}

/* Pack intrinsic dipoles: all agents, then active food. */
static int pack_intrinsic_sources(
    const FishEnv* env, float* x, float* y, float* mx, float* my
) {
    int n = 0;
    for (int i = 0; i < env->num_agents; i++) {
        x[n] = env->agents[i].pos_x;
        y[n] = env->agents[i].pos_y;
        mx[n] = env->agents[i].intrinsic_moment_x;
        my[n] = env->agents[i].intrinsic_moment_y;
        n++;
    }
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) continue;
        x[n] = env->food[i].pos_x;
        y[n] = env->food[i].pos_y;
        mx[n] = env->food[i].intrinsic_moment_x;
        my[n] = env->food[i].intrinsic_moment_y;
        n++;
    }
    return n;
}

/* Pack induced dipoles: all agents, then active food. */
static int pack_induced_sources(
    const FishEnv* env, float* x, float* y, float* mx, float* my
) {
    int n = 0;
    for (int i = 0; i < env->num_agents; i++) {
        x[n] = env->agents[i].pos_x;
        y[n] = env->agents[i].pos_y;
        mx[n] = env->agents[i].induced_moment_x;
        my[n] = env->agents[i].induced_moment_y;
        n++;
    }
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) continue;
        x[n] = env->food[i].pos_x;
        y[n] = env->food[i].pos_y;
        mx[n] = env->food[i].induced_moment_x;
        my[n] = env->food[i].induced_moment_y;
        n++;
    }
    return n;
}

void init_agent_sensors(FishAgent* agent) {
    float r = agent->body_radius_cm;
    for (int s = 0; s < NUM_MORMYROMASTS; s++) {
        float a = mormyromast_angle((size_t)s, 10, 26, PI_F / 3.0f);
        float nx = cosf(a);
        float ny = sinf(a);
        agent->morm_local_x[s] = nx * r;
        agent->morm_local_y[s] = ny * r;
        agent->morm_nx[s] = nx;
        agent->morm_ny[s] = ny;
    }
    for (int s = 0; s < NUM_AMPULLARY; s++) {
        float a = uniform_sensor_angle((size_t)s, NUM_AMPULLARY);
        float nx = cosf(a);
        float ny = sinf(a);
        agent->amp_local_x[s] = nx * r;
        agent->amp_local_y[s] = ny * r;
        agent->amp_nx[s] = nx;
        agent->amp_ny[s] = ny;
    }
    for (int s = 0; s < NUM_KNOLLEN; s++) {
        float a = uniform_sensor_angle((size_t)s, NUM_KNOLLEN);
        float nx = cosf(a);
        float ny = sinf(a);
        agent->knollen_local_x[s] = nx * r;
        agent->knollen_local_y[s] = ny * r;
        agent->knollen_nx[s] = nx;
        agent->knollen_ny[s] = ny;
    }
}


void init(FishEnv* env) {
    if (env->num_agents <= 0) env->num_agents = 4;
    if (env->num_agents > MAX_AGENTS) env->num_agents = MAX_AGENTS;
    if (env->arena_size_x <= 0.0f) env->arena_size_x = 70.0f;
    if (env->arena_size_y <= 0.0f) env->arena_size_y = 70.0f;
    if (env->min_arena_size_x <= 0.0f) {
        env->min_arena_size_x = env->arena_size_x;
    }
    if (env->min_arena_size_y <= 0.0f) {
        env->min_arena_size_y = env->arena_size_y;
    }
    if (env->max_arena_size_x <= 0.0f) {
        env->max_arena_size_x = env->min_arena_size_x;
    }
    if (env->max_arena_size_y <= 0.0f) {
        env->max_arena_size_y = env->min_arena_size_y;
    }
    if (env->configured_num_food <= 0) env->configured_num_food = MAX_FOOD;
    if (env->configured_num_food > MAX_FOOD) {
        env->configured_num_food = MAX_FOOD;
    }
    if (env->patch_radius_cm <= 0.0f) env->patch_radius_cm = 6.0f;
    if (env->patch_radius_std_cm < 0.0f) env->patch_radius_std_cm = 1.5f;
    if (env->patch_density <= 0.0f) env->patch_density = 0.001f;
    if (env->food_distribution < FOOD_UNIFORM ||
            env->food_distribution > FOOD_RANDOM) {
        env->food_distribution = FOOD_UNIFORM;
    }
    if (env->electric_field_radius_cm <= 0.0f) {
        env->electric_field_radius_cm = 15.0f;
    }
    if (env->episode_length <= 0) env->episode_length = 4096;
    if (env->rng == 0) env->rng = 1;
}

void c_allocate(FishEnv* env) {
    init(env);
    env->observations = (float*)calloc((size_t)env->num_agents * OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc((size_t)env->num_agents * ACTION_SIZE, sizeof(float));
    env->rewards = (float*)calloc((size_t)env->num_agents, sizeof(float));
    env->terminals = (float*)calloc((size_t)env->num_agents, sizeof(float));
}

void compute_observations(FishEnv* env) {
    float dip_x[MAX_AGENTS + MAX_FOOD];
    float dip_y[MAX_AGENTS + MAX_FOOD];
    float dip_mx[MAX_AGENTS + MAX_FOOD];
    float dip_my[MAX_AGENTS + MAX_FOOD];

    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->agents[i];
        float* obs = env->observations + i * OBS_SIZE;
        int cursor = 0;

        int n_induced = pack_induced_sources(env, dip_x, dip_y, dip_mx, dip_my);
        /* Mormyromasts: active/collective image after direct-EOD subtraction. */
        for (int sensor_idx = 0; sensor_idx < NUM_MORMYROMASTS; sensor_idx++) {
            float sx, sy, snx, sny;
            world_sensor(
                agent->morm_local_x[sensor_idx], agent->morm_local_y[sensor_idx],
                agent->morm_nx[sensor_idx], agent->morm_ny[sensor_idx],
                agent->pos_x, agent->pos_y, agent->orientation,
                &sx, &sy, &snx, &sny
            );
            float fx, fy;
            measure_electric_field_with_reflections(
                sx, sy,
                NULL, NULL, NULL, 0,
                dip_x, dip_y, dip_mx, dip_my, (size_t)n_induced,
                env->arena_size_x, env->arena_size_y, FIELD_EPS_M,
                REFLECTION_SCALE, false, &fx, &fy
            );
            float reading = project_field(fx, fy, snx, sny);
            if (!agent->emits_eod) {
                reading *= 100.0f;
            }
            reading *= random_multiplier(env, 0.05f);
            obs[cursor++] = normalize_sensor_reading(
                reading, MORMYROMAST_MIN_VM,
                MORMYROMAST_MAX_VM, SENSOR_EPS
            );
        }

        int n_intrinsic = pack_intrinsic_sources(env, dip_x, dip_y, dip_mx, dip_my);
        // Ampullary receptors: intrinsic sources with static self-field removed
        for (int sensor_idx = 0; sensor_idx < NUM_AMPULLARY; sensor_idx++) {
            float sx, sy, snx, sny;
            world_sensor(
                agent->amp_local_x[sensor_idx], agent->amp_local_y[sensor_idx],
                agent->amp_nx[sensor_idx], agent->amp_ny[sensor_idx],
                agent->pos_x, agent->pos_y, agent->orientation,
                &sx, &sy, &snx, &sny
            );
            float fx, fy;
            measure_electric_field_with_reflections(
                sx, sy,
                NULL, NULL, NULL, 0,
                dip_x, dip_y, dip_mx, dip_my, (size_t)n_intrinsic,
                env->arena_size_x, env->arena_size_y, FIELD_EPS_M,
                REFLECTION_SCALE, false, &fx, &fy
            );
            float reading =
                project_field(fx, fy, snx, sny) -
                env->amp_intrinsic_baseline[sensor_idx];
            bool cons_eod = false;
            for (int other = 0; other < env->num_agents; other++) {
                if (other != i && env->agents[other].emits_eod) {
                    cons_eod = true;
                    break;
                }
            }
            reading *= random_multiplier(env, cons_eod ? 0.5f : 0.05f);
            obs[cursor++] = normalize_sensor_reading(
                reading, AMPULLARY_MIN_VM,
                AMPULLARY_MAX_VM, SENSOR_EPS
            );
        }

        /* Knollenorgans: one directional 12-receptor block per conspecific. */
        int metadata_start =
            NUM_MORMYROMASTS + NUM_AMPULLARY +
            NUM_KNOLLEN * (MAX_AGENTS - 1);
        int cons_slot = 0;
        for (int other = 0; other < MAX_AGENTS; other++) {
            if (other == i) continue;
            bool valid = other < env->num_agents && env->agents[other].emits_eod;
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN; sensor_idx++) {
                float value = 0.0f;
                if (valid) {
                    float sx, sy, snx, sny;
                    world_sensor(
                        agent->knollen_local_x[sensor_idx],
                        agent->knollen_local_y[sensor_idx],
                        agent->knollen_nx[sensor_idx],
                        agent->knollen_ny[sensor_idx],
                        agent->pos_x, agent->pos_y, agent->orientation,
                        &sx, &sy, &snx, &sny
                    );
                    float fx, fy;
                    measure_electric_field(
                        sx, sy,
                        env->agents[other].eod_pos_x,
                        env->agents[other].eod_pos_y,
                        env->agents[other].eod_charge,
                        2,
                        NULL, NULL, NULL, NULL, 0,
                        FIELD_EPS_M, &fx, &fy
                    );
                    float raw_knollen = project_field(fx, fy, snx, sny);
                    raw_knollen *= random_multiplier(env, 0.05f);
                    value = fabsf(raw_knollen) <= KNOLLEN_MIN_VM
                        ? 0.0f
                        : (raw_knollen < 0.0f ? -1.0f : 1.0f);
                }
                obs[cursor++] = value;
            }
            bool detected = false;
            int block_start = cursor - NUM_KNOLLEN;
            for (int k = 0; k < NUM_KNOLLEN; k++) {
                if (obs[block_start + k] != 0.0f) {
                    detected = true;
                    break;
                }
            }
            float metadata = -1.0f;
            if (valid && detected) {
                metadata = agent->size - env->agents[other].size;
                metadata += random_uniform(env, -0.05f, 0.05f);
                metadata = clamp(metadata, -1.0f, 1.0f);
            }
            obs[metadata_start + cons_slot] = metadata;
            cons_slot++;
        }
        while (cons_slot < MAX_AGENTS - 1) {
            for (int k = 0; k < NUM_KNOLLEN; k++) obs[cursor++] = 0.0f;
            obs[metadata_start + cons_slot++] = 0.0f;
        }
        cursor = metadata_start + MAX_AGENTS - 1;

        for (int action_idx = 0; action_idx < ACTION_SIZE; action_idx++) {
            obs[cursor++] = agent->last_action[action_idx];
        }
        obs[cursor++] = 0.0f; /* fatigue, retained for upstream layout */
        obs[cursor++] = agent->was_bitten ? 1.0f : 0.0f;
        obs[cursor++] = agent->size;
        obs[cursor++] =
            (float)agent->bite_cooldown / (float)BITE_COOLDOWN_STEPS;
        obs[cursor++] = clamp(
            agent->disp_ego_x / agent->max_linear_velocity, -1.0f, 1.0f);
        obs[cursor++] = clamp(
            agent->disp_ego_y / agent->max_linear_velocity, -1.0f, 1.0f);
        obs[cursor++] =
            (float)agent->eat_cooldown / (float)EAT_COOLDOWN_STEPS;
    }
}

bool position_overlaps_agents(
    const FishEnv* env, float pos_x, float pos_y, int count, float radius
) {
    for (int i = 0; i < count; i++) {
        float dx = pos_x - env->agents[i].pos_x;
        float dy = pos_y - env->agents[i].pos_y;
        float minimum = radius + env->agents[i].body_radius_cm;
        if (vec_length_squared(dx, dy) < minimum * minimum) return true;
    }
    return false;
}

/*
 * Matches SensingModel._calculate_corollary_discharge: baselines are computed
 * once for a representative fish at the arena center, orientation zero.
 */
void calibrate_electroreceptors(FishEnv* env) {
    float center_x = env->arena_size_x * 0.5f;
    float center_y = env->arena_size_y * 0.5f;
    float eod_x[2] = {
        center_x + EOD_POLE_OFFSET_CM,
        center_x - EOD_POLE_OFFSET_CM,
    };
    float eod_y[2] = {center_y, center_y};
    float eod_q[2] = {EOD_CHARGE_C, -EOD_CHARGE_C};
    float dip_x[1] = {center_x};
    float dip_y[1] = {center_y};
    float dip_mx[1] = {INTRINSIC_MOMENT_C_M};
    float dip_my[1] = {0.0f};

    for (int i = 0; i < NUM_MORMYROMASTS; i++) {
        float a = mormyromast_angle((size_t)i, 10, 26, PI_F / 3.0f);
        float nx = cosf(a);
        float ny = sinf(a);
        float sx = center_x + nx * BODY_RADIUS_CM;
        float sy = center_y + ny * BODY_RADIUS_CM;
        float fx, fy;
        measure_electric_field_with_reflections(
            sx, sy,
            eod_x, eod_y, eod_q, 2,
            NULL, NULL, NULL, NULL, 0,
            env->arena_size_x, env->arena_size_y, FIELD_EPS_M,
            REFLECTION_SCALE, false, &fx, &fy
        );
        env->mormyromast_cd[i] = project_field(fx, fy, nx, ny);
    }

    for (int i = 0; i < NUM_AMPULLARY; i++) {
        float a = uniform_sensor_angle((size_t)i, NUM_AMPULLARY);
        float nx = cosf(a);
        float ny = sinf(a);
        float sx = center_x + nx * BODY_RADIUS_CM;
        float sy = center_y + ny * BODY_RADIUS_CM;
        float fx, fy;
        measure_electric_field_with_reflections(
            sx, sy,
            NULL, NULL, NULL, 0,
            dip_x, dip_y, dip_mx, dip_my, 1,
            env->arena_size_x, env->arena_size_y, FIELD_EPS_M,
            REFLECTION_SCALE, false, &fx, &fy
        );
        env->amp_intrinsic_baseline[i] = project_field(fx, fy, nx, ny);
    }
}

void build_electric_scene(FishEnv* env) {
    float eod_x[2 * MAX_AGENTS];
    float eod_y[2 * MAX_AGENTS];
    float eod_q[2 * MAX_AGENTS];

    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->agents[i];
        float ori = agent->orientation;
        float charge_scale = agent->emits_eod ? 1.0f : 0.0f;
        transform_position(
            EOD_POLE_OFFSET_CM, 0.0f,
            agent->pos_x, agent->pos_y, ori,
            &agent->eod_pos_x[0], &agent->eod_pos_y[0]
        );
        transform_position(
            -EOD_POLE_OFFSET_CM, 0.0f,
            agent->pos_x, agent->pos_y, ori,
            &agent->eod_pos_x[1], &agent->eod_pos_y[1]
        );
        agent->eod_charge[0] = EOD_CHARGE_C * charge_scale;
        agent->eod_charge[1] = -EOD_CHARGE_C * charge_scale;
        rotate(
            INTRINSIC_MOMENT_C_M, 0.0f, ori,
            &agent->intrinsic_moment_x, &agent->intrinsic_moment_y
        );
    }

    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) {
            env->food[i].intrinsic_moment_x = 0.0f;
            env->food[i].intrinsic_moment_y = 0.0f;
            continue;
        }
        rotate(
            0.0f, FOOD_INTRINSIC_MOMENT_C_M, env->food[i].orientation,
            &env->food[i].intrinsic_moment_x, &env->food[i].intrinsic_moment_y
        );
    }

    int n_eod = pack_eod_sources(env, eod_x, eod_y, eod_q);

    // EOD fields induce dipoles on prey and agent bodies (Chen et al., Eq. 6) 
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->agents[i];
        float moment_x = 0.0f;
        float moment_y = 0.0f;
        if (!agent->emits_eod) {
            float fx, fy;
            measure_electric_field(
                agent->pos_x, agent->pos_y,
                eod_x, eod_y, eod_q, (size_t)n_eod,
                NULL, NULL, NULL, NULL, 0,
                FIELD_EPS_M, &fx, &fy
            );
            induce_dipole_moment(
                fx, fy, agent->body_radius_cm,
                CONDUCTOR_CONTRAST, &moment_x, &moment_y
            );
        }
        float moment_magnitude = vec_length(moment_x, moment_y);
        float max_moment = EOD_CHARGE_C * agent->body_radius_cm;
        if (EOD_CHARGE_C >= 0.0f && moment_magnitude > max_moment) {
            float moment_scale = max_moment / moment_magnitude;
            moment_x *= moment_scale;
            moment_y *= moment_scale;
        }
        agent->induced_moment_x = moment_x;
        agent->induced_moment_y = moment_y;
    }
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) {
            env->food[i].induced_moment_x = 0.0f;
            env->food[i].induced_moment_y = 0.0f;
            continue;
        }
        float fx, fy, mx, my;
        measure_electric_field(
            env->food[i].pos_x, env->food[i].pos_y,
            eod_x, eod_y, eod_q, (size_t)n_eod,
            NULL, NULL, NULL, NULL, 0,
            FIELD_EPS_M, &fx, &fy
        );
        induce_dipole_moment(
            fx, fy, FOOD_RADIUS_CM, CONDUCTOR_CONTRAST, &mx, &my
        );
        env->food[i].induced_moment_x = mx;
        env->food[i].induced_moment_y = my;
    }
}

void c_reset(FishEnv* env) {
    init(env);

    // Sample arena size 
    env->arena_size_x = random_uniform(
        env, env->min_arena_size_x, env->max_arena_size_x
    );
    env->arena_size_y = random_uniform(
        env, env->min_arena_size_y, env->max_arena_size_y
    );

    // FOOD_RANDOM picks uniform or patchy for this episode only
    FoodDistribution mode = env->food_distribution;
    if (mode == FOOD_RANDOM) {
        mode = (FoodDistribution)(rand_r(&env->rng) % 2);
    }
    env->tick = 0;
    env->food_eaten = 0;
    env->eod_agent_steps = 0;
    env->collisions_fish = 0;
    env->episode_return = 0.0f;

    // Initialize agents with random positions and orientations, avoiding overlaps
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent agent = {0};
        agent.size = random_uniform(env, 0.0f, 1.0f);
        float pos_x, pos_y;
        int attempts = 0;
        do {
            pos_x = random_uniform(env, 3.0f, env->arena_size_x - 3.0f);
            pos_y = random_uniform(env, 3.0f, env->arena_size_y - 3.0f);
        } while (
            position_overlaps_agents(
                env, pos_x, pos_y, i, BODY_RADIUS_CM
            ) && ++attempts < 1000
        );
        agent.pos_x = pos_x;
        agent.pos_y = pos_y;
        agent.orientation = random_uniform(env, -PI_F, PI_F);
        agent.max_linear_velocity = 35.0f / SIMULATION_HZ;
        agent.max_angular_velocity = 3.6f / SIMULATION_HZ;
        agent.body_radius_cm = BODY_RADIUS_CM;
        agent.min_linear_velocity = -5.0f / SIMULATION_HZ;
        agent.max_linear_acceleration = 650.0f / (SIMULATION_HZ * SIMULATION_HZ);
        agent.min_angular_velocity = -3.5f / SIMULATION_HZ;
        agent.max_angular_acceleration = 318.0f / (SIMULATION_HZ * SIMULATION_HZ);
        agent.linear_drag_factor = 0.95f;
        agent.angular_drag_factor = 0.95f;
        agent.motion_order = MOTION_FIRST_ORDER;
        agent.backwards = false;
        float size_multiplier = 1.0f + agent.size;
        agent.min_linear_velocity *= size_multiplier;
        agent.max_linear_velocity *= size_multiplier;
        agent.max_linear_acceleration *= size_multiplier;
        agent.min_angular_velocity *= size_multiplier;
        agent.max_angular_velocity *= size_multiplier;
        agent.max_angular_acceleration *= size_multiplier;
        agent.emits_eod = true;
        init_agent_sensors(&agent);
        env->agents[i] = agent;
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0.0f;
    }

    env->num_food = env->configured_num_food;
    if (mode == FOOD_UNIFORM) {
        for (int i = 0; i < env->num_food; i++) {
            env->food[i] = (FishFood){
                .pos_x = random_uniform(env, 0.0f, env->arena_size_x),
                .pos_y = random_uniform(env, 0.0f, env->arena_size_y),
                .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
                .active = true,
            };
        }
    } else {
        // Patchy: random circular patches, food sampled uniformly in a patch disk
        float centers_x[MAX_PATCHES];
        float centers_y[MAX_PATCHES];
        float radii[MAX_PATCHES];
        float max_radius =
            fminf(env->arena_size_x, env->arena_size_y) * 0.5f;
        int num_patches = (int)clamp(
            ceilf(env->patch_density *
                env->arena_size_x * env->arena_size_y),
            1, MAX_PATCHES
        );
        for (int p = 0; p < num_patches; p++) {
            centers_x[p] = random_uniform(env, 0.0f, env->arena_size_x);
            centers_y[p] = random_uniform(env, 0.0f, env->arena_size_y);
            radii[p] = clamp(
                env->patch_radius_cm + random_uniform(
                    env, -env->patch_radius_std_cm, env->patch_radius_std_cm
                ),
                1.0f, max_radius
            );
        }
        for (int i = 0; i < env->num_food; i++) {
            int p = (int)(rand_r(&env->rng) % (unsigned)num_patches);
            float angle = random_uniform(env, 0.0f, 2.0f * PI_F);
            float r = radii[p] * sqrtf(random_uniform(env, 0.0f, 1.0f));
            env->food[i] = (FishFood){
                .pos_x = clamp(centers_x[p] + r * cosf(angle),
                    0.0f, env->arena_size_x),
                .pos_y = clamp(centers_y[p] + r * sinf(angle),
                    0.0f, env->arena_size_y),
                .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
                .active = true,
            };
        }
    }
    calibrate_electroreceptors(env);
    build_electric_scene(env);
    compute_observations(env);
}

void add_log(FishEnv* env) {
    env->log.episode_length += (float)env->tick;
    env->log.episode_return += env->episode_return;
    env->log.score += (float)env->food_eaten;
    env->log.perf += (float)env->food_eaten / (float)env->num_food;
    env->log.food_eaten_mean += (float)env->food_eaten / (float)env->num_agents;
    env->log.eod_rate += (float)env->eod_agent_steps / (float)(env->tick * env->num_agents);
    env->log.collisions_fish += (float)env->collisions_fish;
    env->log.n += 1.0f;
}

bool try_eat(FishEnv* env, int agent_idx) {
    FishAgent* agent = &env->agents[agent_idx];
    int nearest = -1;
    float nearest_distance = INFINITY;
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) continue;
        if (!point_in_forward_cone(
                agent->pos_x, agent->pos_y, agent->orientation,
                env->food[i].pos_x, env->food[i].pos_y,
                EATING_RADIUS_CM, EATING_ANGLE)) continue;
        float dx = env->food[i].pos_x - agent->pos_x;
        float dy = env->food[i].pos_y - agent->pos_y;
        float distance = vec_length_squared(dx, dy);
        if (distance < nearest_distance) {
            nearest = i;
            nearest_distance = distance;
        }
    }
    if (nearest < 0) return false;
    env->food[nearest].active = false;
    env->food_eaten++;
    agent->ate_food = true;
    agent->eat_cooldown = EAT_COOLDOWN_STEPS;
    return true;
}

void c_step(FishEnv* env) {
    env->tick++;
    
    // Reset agent state variables
    // NOTE(DC): zeroing out rewards and terminals is already done in vecenv, shall we delete it here?
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].was_bitten = false;
        env->agents[i].ate_food = false;
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0.0f;
    }
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->agents[i];
        float* raw_action = env->actions + i * ACTION_SIZE;
        float move = 1.0f / (1.0f + expf(-(float)raw_action[0]));
        float turn = tanhf((float)raw_action[1]);
        agent->emits_eod = raw_action[2] > 0.0f;
        agent->bite_action = raw_action[3] > 0.0f && agent->bite_cooldown <= 0;
        if (agent->bite_action) agent->bite_cooldown = BITE_COOLDOWN_STEPS;
        agent->last_action[0] = move;
        agent->last_action[1] = turn;
        agent->last_action[2] = agent->emits_eod ? 1.0f : 0.0f;
        agent->last_action[3] = agent->bite_action ? 1.0f : 0.0f;
        env->eod_agent_steps += agent->emits_eod ? 1 : 0;

        if (!agent->bite_action && agent->eat_cooldown <= 0) {
            if (try_eat(env, i)) {
                env->rewards[i] += EAT_REWARD;
            }
        }
        FishMotionProposal proposal = propose_motion(
            agent, move, turn, agent->eat_cooldown > 0
        );
        bool collided = motion_collides(
            proposal.pos_x, proposal.pos_y, agent->body_radius_cm,
            env->agents, (size_t)env->num_agents, (size_t)i
        );
        bool hit_wall = commit_motion(
            agent, proposal, collided,
            env->arena_size_x, env->arena_size_y
        );
        env->collisions_fish += collided ? 1 : 0;
        if (collided || hit_wall) {
            env->rewards[i] += COLLISION_REWARD;
        }
    }

    /* Bite resolution occurs after all fish have moved. */
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* attacker = &env->agents[i];
        if (!attacker->bite_action) continue;
        int victim = -1;
        float nearest = INFINITY;
        for (int j = 0; j < env->num_agents; j++) {
            if (i == j) continue;
            if (!point_in_forward_cone(
                    attacker->pos_x, attacker->pos_y, attacker->orientation,
                    env->agents[j].pos_x, env->agents[j].pos_y,
                    BITING_RADIUS_CM, EATING_ANGLE)) continue;
            float dx = env->agents[j].pos_x - attacker->pos_x;
            float dy = env->agents[j].pos_y - attacker->pos_y;
            float distance = vec_length_squared(dx, dy);
            if (distance < nearest) {
                victim = j;
                nearest = distance;
            }
        }
        if (victim >= 0) {
            env->agents[victim].was_bitten = true;
            float size_difference = attacker->size - env->agents[victim].size;
            env->rewards[victim] += BITTEN_REWARD * (1.0f + fmaxf(0.0f, size_difference));
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->agents[i];
        if (agent->eat_cooldown > 0) agent->eat_cooldown -= 1;
        if (agent->bite_cooldown > 0) agent->bite_cooldown -= 1;
        env->episode_return += env->rewards[i];
    }
    build_electric_scene(env);
    compute_observations(env);

    if (env->tick >= env->episode_length || env->food_eaten == env->num_food) {
        add_log(env);
        c_reset(env);
        for (int i = 0; i < env->num_agents; i++) env->terminals[i] = 1.0f;
    }
}

Vector2 world_to_screen(const FishEnv* env, float point_x, float point_y) {
    Client* client = env->client;
    float usable_width = client->window_width - 2.0f * client->margin;
    float usable_height = client->window_height - 2.0f * client->margin;
    return (Vector2){
        client->margin + point_x / env->arena_size_x * usable_width,
        client->window_height - client->margin -
            point_y / env->arena_size_y * usable_height,
    };
}

float render_scale(const FishEnv* env) {
    Client* client = env->client;
    float sx = (client->window_width - 2.0f * client->margin) /
        env->arena_size_x;
    float sy = (client->window_height - 2.0f * client->margin) /
        env->arena_size_y;
    return fminf(sx, sy);
}

Client* make_client(void) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->window_width = 900;
    client->window_height = 900;
    client->margin = 55;
    client->show_field = true;
    client->show_sensors = true;
    InitWindow(client->window_width, client->window_height, "Electric fish");
    SetTargetFPS(60);
    return client;
}

Color field_color(float log_strength) {
    float normalized = clamp((log_strength + 8.0f) / 7.0f, 0.0f, 1.0f);
    unsigned char shade = (unsigned char)(205.0f - 95.0f * normalized);
    return (Color){shade, shade, shade, 190};
}

void render_field(FishEnv* env) {
    float mono_x[2 * MAX_AGENTS];
    float mono_y[2 * MAX_AGENTS];
    float mono_q[2 * MAX_AGENTS];
    float dip_x[MAX_DIPOLE_SOURCES];
    float dip_y[MAX_DIPOLE_SOURCES];
    float dip_mx[MAX_DIPOLE_SOURCES];
    float dip_my[MAX_DIPOLE_SOURCES];
    int n_mono = pack_eod_sources(env, mono_x, mono_y, mono_q);
    int n_intr = pack_intrinsic_sources(env, dip_x, dip_y, dip_mx, dip_my);
    int n_ind = pack_induced_sources(
        env, dip_x + n_intr, dip_y + n_intr, dip_mx + n_intr, dip_my + n_intr
    );
    int n_dip = n_intr + n_ind;

    const int columns = 25;
    const int rows = 25;
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            float pos_x = env->arena_size_x * (column + 0.5f) / columns;
            float pos_y = env->arena_size_y * (row + 0.5f) / rows;
            bool near_fish = false;
            float radius_squared =
                env->electric_field_radius_cm * env->electric_field_radius_cm;
            for (int i = 0; i < env->num_agents; i++) {
                float dx = pos_x - env->agents[i].pos_x;
                float dy = pos_y - env->agents[i].pos_y;
                if (dx * dx + dy * dy <= radius_squared) {
                    near_fish = true;
                    break;
                }
            }
            if (!near_fish) continue;

            float fx, fy;
            measure_electric_field_with_reflections(
                pos_x, pos_y,
                mono_x, mono_y, mono_q, (size_t)n_mono,
                dip_x, dip_y, dip_mx, dip_my, (size_t)n_dip,
                env->arena_size_x, env->arena_size_y,
                FIELD_EPS_M,
                REFLECTION_SCALE,
                false,
                &fx, &fy
            );
            float strength = vec_length(fx, fy);
            if (strength <= 0.0f) continue;
            float log_strength = log10f(strength);
            float arrow_length = 4.0f + 10.0f *
                clamp((log_strength + 8.0f) / 7.0f, 0.0f, 1.0f);
            float dir_x = fx / strength;
            float dir_y = fy / strength;
            Vector2 start = world_to_screen(env, pos_x, pos_y);
            Vector2 end = {
                start.x + dir_x * arrow_length,
                start.y - dir_y * arrow_length,
            };
            Color color = field_color(log_strength);
            DrawCircleV(start, 2.0f, color);
            DrawLineEx(start, end, 1.2f, color);
        }
    }
}

void c_render(FishEnv* env) {
    if (env->client == NULL) env->client = make_client();
    if (IsKeyPressed(KEY_TAB)) ToggleFullscreen();
    if (IsKeyPressed(KEY_F)) {
        env->client->show_field = !env->client->show_field;
    }
    if (IsKeyPressed(KEY_S)) {
        env->client->show_sensors = !env->client->show_sensors;
    }

    for (int i = 0; i < env->num_agents; i++) {
        Trace* trace = &env->client->traces[i];
        if (env->terminals[i]) {
            trace->index = 0;
            trace->count = 0;
        }
        trace->pos_x[trace->index] = env->agents[i].pos_x;
        trace->pos_y[trace->index] = env->agents[i].pos_y;
        trace->index = (trace->index + 1) % TRACE_LENGTH;
        if (trace->count < TRACE_LENGTH) trace->count++;
    }

    BeginDrawing();
    ClearBackground(WHITE);

    Vector2 arena_min = world_to_screen(env, 0.0f, env->arena_size_y);
    Vector2 arena_max = world_to_screen(env, env->arena_size_x, 0.0f);
    DrawRectangleRec(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        WHITE
    );

    if (env->client->show_field) {
        render_field(env);
    }

    const Color color = {157, 122, 216, 255};
    for (int i = 0; i < env->num_agents; i++) {
        Trace* trace = &env->client->traces[i];
        for (int j = 0; j < trace->count - 1; j++) {
            int current =
                (trace->index - j - 1 + TRACE_LENGTH) % TRACE_LENGTH;
            int previous =
                (trace->index - j - 2 + TRACE_LENGTH) % TRACE_LENGTH;
            float alpha =
                0.55f * (float)(trace->count - j) / (float)trace->count;
            DrawLineEx(
                world_to_screen(env, trace->pos_x[current], trace->pos_y[current]),
                world_to_screen(env, trace->pos_x[previous], trace->pos_y[previous]),
                2.0f, ColorAlpha(color, alpha)
            );
        }
    }

    float scale = render_scale(env);
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) continue;
        Vector2 position = world_to_screen(
            env, env->food[i].pos_x, env->food[i].pos_y
        );
        float radius = fmaxf(2.5f, FOOD_RADIUS_CM * scale);
        DrawCircleV(position, radius, (Color){80, 220, 125, 255});
        DrawCircleLines(
            (int)position.x, (int)position.y, radius,
            (Color){190, 255, 205, 255}
        );
    }
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->agents[i];
        Vector2 center = world_to_screen(env, agent->pos_x, agent->pos_y);
        float radius = agent->body_radius_cm * scale;

        if (agent->emits_eod) {
            float pulse = radius + 5.0f +
                5.0f * sinf((float)env->tick * 0.18f);
            DrawCircleLines((int)center.x, (int)center.y, pulse,
                (Color){color.r, color.g, color.b, 120});
        }

        float heading_x = cosf(agent->orientation);
        float heading_y = -sinf(agent->orientation);
        DrawRing(center, radius - 1.5f, radius + 1.5f,
            0.0f, 360.0f, 32, color);
        Vector2 nose = {
            center.x + heading_x * radius * 1.35f,
            center.y + heading_y * radius * 1.35f,
        };
        DrawLineEx(center, nose, 3.0f, color);

        Vector2 positive = world_to_screen(
            env, agent->eod_pos_x[0], agent->eod_pos_y[0]
        );
        Vector2 negative = world_to_screen(
            env, agent->eod_pos_x[1], agent->eod_pos_y[1]
        );
        DrawCircleV(positive, 3.0f, RED);
        DrawCircleV(negative, 3.0f, BLUE);

        if (env->client->show_sensors) {
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN;
                    sensor_idx++) {
                float sx, sy, snx, sny;
                world_sensor(
                    agent->knollen_local_x[sensor_idx],
                    agent->knollen_local_y[sensor_idx],
                    agent->knollen_nx[sensor_idx],
                    agent->knollen_ny[sensor_idx],
                    agent->pos_x, agent->pos_y, agent->orientation,
                    &sx, &sy, &snx, &sny
                );
                DrawCircleV(
                    world_to_screen(env, sx, sy),
                    1.5f, (Color){184, 164, 224, 180}
                );
            }
        }
        DrawText(TextFormat("%d", i + 1),
            (int)(center.x + radius + 4), (int)(center.y - radius), 16, BLACK);
    }

    DrawRectangleLinesEx(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        3.0f, DARKGRAY
    );
    DrawText("Electric fish", 20, 14, 24, BLACK);
    DrawText("F: electric field   S: sensors   TAB: fullscreen",
        20, env->client->window_height - 32, 18, DARKGRAY);
    int active_eods = 0;
    for (int i = 0; i < env->num_agents; i++) {
        active_eods += env->agents[i].emits_eod ? 1 : 0;
    }
    DrawText(TextFormat("step %d   active EODs %d/%d",
        env->tick, active_eods, env->num_agents),
        env->client->window_width - 285, 18, 18, DARKGRAY);
    DrawText(TextFormat("food %d/%d", env->food_eaten, env->num_food),
        env->client->window_width - 145,
        env->client->window_height - 32, 18, DARKGRAY);
    DrawText(TextFormat("field radius %.0f cm", env->electric_field_radius_cm),
        430, env->client->window_height - 32, 18, DARKGRAY);
    EndDrawing();
}

void c_close(FishEnv* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

void free_allocated(FishEnv* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}
