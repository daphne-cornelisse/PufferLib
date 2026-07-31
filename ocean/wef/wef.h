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

typedef struct FishVec2 {
    float x;
    float y;
} FishVec2;

typedef struct FishMonopole {
    FishVec2 position_cm;
    float charge_c;
} FishMonopole;

typedef struct FishDipole {
    FishVec2 position_cm;
    FishVec2 moment_c_m;
} FishDipole;

typedef struct FishSensor {
    FishVec2 position_cm;
    FishVec2 normal;
} FishSensor;

typedef enum FishMotionOrder {
    MOTION_FIRST_ORDER = 1,
    MOTION_SECOND_ORDER = 2,
} FishMotionOrder;

typedef enum FishWall {
    WALL_LEFT = 0,
    WALL_RIGHT = 1,
    WALL_BOTTOM = 2,
    WALL_TOP = 3,
} FishWall;

typedef struct FishMovement {
    FishVec2 position_cm;
    float orientation;
    float max_linear_velocity;
    float max_angular_velocity;

    FishVec2 displacement_ground_cm;
    FishVec2 displacement_ego_cm;
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
} FishMovement;

typedef struct FishMotionProposal {
    FishVec2 position_cm;
    float orientation;
    float linear_velocity;
    float angular_velocity;
} FishMotionProposal;

typedef struct FishFoodMotion {
    FishVec2 position_cm;
    FishVec2 velocity_cm;
    float orientation;
} FishFoodMotion;

float vec_length_squared(FishVec2 v) {
    return v.x * v.x + v.y * v.y;
}

float vec_length(FishVec2 v) {
    return sqrtf(vec_length_squared(v));
}

float clamp(float value, float minimum, float maximum) {
    return fminf(maximum, fmaxf(minimum, value));
}

/* Stable wrap to [-pi, pi], identical to movement._wrap_angle. */
float wrap_angle(float angle) {
    return atan2f(sinf(angle), cosf(angle));
}

/* Rotate a vector; add translation separately when transforming a position. */
FishVec2 rotate(FishVec2 v, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return (FishVec2){c * v.x - s * v.y, s * v.x + c * v.y};
}

FishVec2 transform_position(FishVec2 local_cm, FishVec2 origin_cm, float angle) {
    FishVec2 rotated = rotate(local_cm, angle);
    return (FishVec2){rotated.x + origin_cm.x, rotated.y + origin_cm.y};
}

FishSensor transform_sensor(FishSensor local, FishVec2 origin_cm, float angle) {
    FishSensor world = {
        .position_cm = transform_position(local.position_cm, origin_cm, angle),
        .normal = rotate(local.normal, angle),
    };
    return world;
}

// Biophysics and dynamics 

FishMotionProposal propose_motion(
    const FishMovement* fish,
    float move_command,
    float turn_command,
    bool eating_frozen
) {
    FishMotionProposal proposal = {
        .position_cm = fish->position_cm,
        .orientation = fish->orientation,
        .linear_velocity = 0.0f,
        .angular_velocity = 0.0f,
    };

    if (fish->motion_order == MOTION_FIRST_ORDER) {
        float move = eating_frozen ? 0.0f : move_command;
        float turn = eating_frozen ? 0.0f : turn_command;
        proposal.angular_velocity = turn * fish->max_angular_velocity;
        proposal.linear_velocity = move * fish->max_linear_velocity;
    } else {
        proposal.linear_velocity =
            (float)fish->linear_velocity +
            move_command * (float)fish->max_linear_acceleration;
        proposal.angular_velocity =
            (float)fish->angular_velocity +
            turn_command * (float)fish->max_angular_acceleration;

        proposal.linear_velocity = clamp(
            proposal.linear_velocity,
            (float)fish->min_linear_velocity,
            fish->max_linear_velocity
        );
        if (!fish->backwards && proposal.linear_velocity < 0.0f) {
            proposal.linear_velocity = 0.0f;
        }

        proposal.linear_velocity *= (float)fish->linear_drag_factor;
        proposal.angular_velocity *= (float)fish->angular_drag_factor;
        proposal.angular_velocity = clamp(
            proposal.angular_velocity,
            (float)fish->min_angular_velocity,
            fish->max_angular_velocity
        );

        if (eating_frozen) {
            proposal.linear_velocity = 0.0f;
            proposal.angular_velocity = 0.0f;
        }
    }

    proposal.orientation =
        wrap_angle(fish->orientation + proposal.angular_velocity);
    float hx = cosf(proposal.orientation);
    float hy = sinf(proposal.orientation);
    proposal.position_cm = (FishVec2){
        (float)((float)fish->position_cm.x + hx * proposal.linear_velocity),
        (float)((float)fish->position_cm.y + hy * proposal.linear_velocity),
    };
    return proposal;
}

bool motion_collides(
    FishVec2 proposed_position_cm,
    float body_radius_cm,
    const FishMovement* others,
    size_t num_others,
    size_t self_index
) {
    for (size_t i = 0; i < num_others; i++) {
        if (i == self_index) {
            continue;
        }
        float radius = body_radius_cm + others[i].body_radius_cm;
        FishVec2 delta = {
            proposed_position_cm.x - others[i].position_cm.x,
            proposed_position_cm.y - others[i].position_cm.y,
        };
        if (vec_length_squared(delta) < radius * radius) {
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
    FishMovement* fish,
    FishMotionProposal proposal,
    bool collided,
    FishVec2 arena_size_cm
) {
    FishVec2 previous_position = fish->position_cm;
    float previous_orientation = fish->orientation;

    fish->last_orientation = (float)previous_orientation;
    fish->orientation = proposal.orientation;
    if (!collided) {
        fish->position_cm = proposal.position_cm;
    }

    FishVec2 unclipped = fish->position_cm;
    fish->position_cm.x = clamp(
        fish->position_cm.x,
        fish->body_radius_cm,
        arena_size_cm.x - fish->body_radius_cm
    );
    fish->position_cm.y = clamp(
        fish->position_cm.y,
        fish->body_radius_cm,
        arena_size_cm.y - fish->body_radius_cm
    );
    bool hit_wall = fish->position_cm.x != unclipped.x || fish->position_cm.y != unclipped.y;
    bool stopped = collided || hit_wall;

    if (fish->motion_order == MOTION_SECOND_ORDER) {
        fish->linear_velocity =
            stopped ? 0.0f : (float)proposal.linear_velocity;
        fish->angular_velocity =
            stopped ? 0.0f : (float)proposal.angular_velocity;
    }

    fish->collided = collided;
    fish->displacement_ground_cm = (FishVec2){
        fish->position_cm.x - previous_position.x,
        fish->position_cm.y - previous_position.y,
    };
    fish->displacement_ego_cm = rotate(
        fish->displacement_ground_cm, (float)(-previous_orientation)
    );
    return hit_wall;
}

bool point_in_forward_cone(
    FishVec2 position_cm,
    float orientation,
    FishVec2 target_position_cm,
    float radius_cm,
    float cone_angle
) {
    FishVec2 offset = {
        target_position_cm.x - position_cm.x,
        target_position_cm.y - position_cm.y,
    };
    if (vec_length_squared(offset) >= radius_cm * radius_cm) {
        return false;
    }
    float bearing = atan2f(offset.y, offset.x);
    return fabsf(wrap_angle(bearing - orientation)) <= cone_angle * 0.5f;
}

// Electrostatic field model
FishMonopole transform_monopole(
    FishMonopole local,
    FishVec2 position_cm,
    float orientation,
    bool emits_eod
) {
    FishMonopole world = {
        .position_cm = transform_position(
            local.position_cm, position_cm, orientation
        ),
        .charge_c = emits_eod ? local.charge_c : 0.0f,
    };
    return world;
}

/*
 * Add fields from arbitrary monopoles and dipoles at one measurement point.
 * This follows electric.measure_electric_field_original exactly, including
 * adding eps_m to the distance rather than to squared distance.
 */
FishVec2 measure_electric_field(
    FishVec2 measurement_position_cm,
    const FishMonopole* monopoles,
    size_t num_monopoles,
    const FishDipole* dipoles,
    size_t num_dipoles,
    float eps_m
) {
    FishVec2 measurement_m = {
        measurement_position_cm.x * CM_TO_M,
        measurement_position_cm.y * CM_TO_M,
    };
    FishVec2 field = {0.0f, 0.0f};

    for (size_t i = 0; i < num_monopoles; i++) {
        FishVec2 source_m = {
            monopoles[i].position_cm.x * CM_TO_M,
            monopoles[i].position_cm.y * CM_TO_M,
        };
        FishVec2 offset = {
            measurement_m.x - source_m.x,
            measurement_m.y - source_m.y,
        };
        float distance = vec_length(offset) + eps_m;
        float weight =
            K_COULOMB * monopoles[i].charge_c /
            (distance * distance * distance);
        field.x += offset.x * weight;
        field.y += offset.y * weight;
    }

    for (size_t i = 0; i < num_dipoles; i++) {
        FishVec2 source_m = {
            dipoles[i].position_cm.x * CM_TO_M,
            dipoles[i].position_cm.y * CM_TO_M,
        };
        FishVec2 offset = {
            measurement_m.x - source_m.x,
            measurement_m.y - source_m.y,
        };
        float distance = vec_length(offset) + eps_m;
        float r2 = distance * distance;
        float r3 = r2 * distance;
        float r5 = r3 * r2;
        float moment_dot_offset =
            dipoles[i].moment_c_m.x * offset.x +
            dipoles[i].moment_c_m.y * offset.y;
        float axial = 3.0f * moment_dot_offset / r5;
        field.x += K_COULOMB * (
            offset.x * axial - dipoles[i].moment_c_m.x / r3
        );
        field.y += K_COULOMB * (
            offset.y * axial - dipoles[i].moment_c_m.y / r3
        );
    }
    return field;
}

FishVec2 induce_dipole_moment(
    FishVec2 external_field_vm,
    float conductor_radius_cm,
    float conductor_contrast
) {
    float radius_m = conductor_radius_cm * CM_TO_M;
    float volume_m3 = (4.0f / 3.0f) * PI_F * radius_m * radius_m * radius_m;
    float scale =
        3.0f * EPSILON_0 * volume_m3 * conductor_contrast;
    return (FishVec2){
        external_field_vm.x * scale,
        external_field_vm.y * scale,
    };
}

// Field from the original sources plus their four first-order wall images.
FishVec2 measure_electric_field_with_reflections(
    FishVec2 measurement_position_cm,
    const FishMonopole* monopoles,
    size_t num_monopoles,
    const FishDipole* dipoles,
    size_t num_dipoles,
    FishVec2 arena_size_cm,
    float eps_m,
    float reflection_scale,
    bool flip_on_reflection
) {
    FishVec2 field = measure_electric_field(
        measurement_position_cm,
        monopoles, num_monopoles,
        dipoles, num_dipoles,
        eps_m
    );
    for (int wall = WALL_LEFT; wall <= WALL_TOP; wall++) {
        for (size_t i = 0; i < num_monopoles; i++) {
            FishMonopole image = monopoles[i];
            if (wall == WALL_LEFT) {
                image.position_cm.x = -image.position_cm.x;
            } else if (wall == WALL_RIGHT) {
                image.position_cm.x =
                    2.0f * arena_size_cm.x - image.position_cm.x;
            } else if (wall == WALL_BOTTOM) {
                image.position_cm.y = -image.position_cm.y;
            } else {
                image.position_cm.y =
                    2.0f * arena_size_cm.y - image.position_cm.y;
            }
            image.charge_c *=
                reflection_scale * (flip_on_reflection ? -1.0f : 1.0f);
            FishVec2 contribution = measure_electric_field(
                measurement_position_cm, &image, 1, NULL, 0, eps_m
            );
            field.x += contribution.x;
            field.y += contribution.y;
        }
        for (size_t i = 0; i < num_dipoles; i++) {
            FishDipole image = dipoles[i];
            if (wall == WALL_LEFT) {
                image.position_cm.x = -image.position_cm.x;
                if (flip_on_reflection) {
                    image.moment_c_m.x = -image.moment_c_m.x;
                }
            } else if (wall == WALL_RIGHT) {
                image.position_cm.x =
                    2.0f * arena_size_cm.x - image.position_cm.x;
                if (flip_on_reflection) {
                    image.moment_c_m.x = -image.moment_c_m.x;
                }
            } else if (wall == WALL_BOTTOM) {
                image.position_cm.y = -image.position_cm.y;
                if (flip_on_reflection) {
                    image.moment_c_m.y = -image.moment_c_m.y;
                }
            } else {
                image.position_cm.y =
                    2.0f * arena_size_cm.y - image.position_cm.y;
                if (flip_on_reflection) {
                    image.moment_c_m.y = -image.moment_c_m.y;
                }
            }
            image.moment_c_m.x *= reflection_scale;
            image.moment_c_m.y *= reflection_scale;
            FishVec2 contribution = measure_electric_field(
                measurement_position_cm, NULL, 0, &image, 1, eps_m
            );
            field.x += contribution.x;
            field.y += contribution.y;
        }
    }
    return field;
}

// Receptor geometry and transduction
float
project_field(FishVec2 field, FishVec2 sensor_normal) {
    return (float)(
        (float)field.x * (float)sensor_normal.x +
        (float)field.y * (float)sensor_normal.y
    );
}

FishSensor
radial_sensor(float angle, float body_radius_cm) {
    FishVec2 normal = {cosf(angle), sinf(angle)};
    FishSensor sensor = {
        .position_cm = {
            normal.x * body_radius_cm,
            normal.y * body_radius_cm,
        },
        .normal = normal,
    };
    return sensor;
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
    FishVec2 positions[TRACE_LENGTH];
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

typedef struct FishAgentState {
    FishMovement movement;
    float size;
    int bite_cooldown;  /* remaining steps; 0 = ready */
    int eat_cooldown;   /* remaining steps; 0 = free to move / eat */
    bool emits_eod;
    bool bite_action;
    bool was_bitten;
    bool ate_food;
    float last_action[ACTION_SIZE];
    float ampullary_ema[NUM_AMPULLARY];
} FishAgentState;

typedef struct FishFood {
    FishFoodMotion motion;
    bool active;
} FishFood;

typedef enum FoodDistribution {
    FOOD_UNIFORM,
    FOOD_PATCHY,
    FOOD_N_PATCH,
    FOOD_ONE_PATCH,
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

    FishVec2 arena_size_cm;
    FishVec2 min_arena_size_cm;
    FishVec2 max_arena_size_cm;
    float electric_field_radius_cm;
    FoodDistribution food_distribution;
    FoodDistribution active_food_distribution;
    int configured_num_food;
    int fixed_num_patches;
    float patch_radius_cm;
    float patch_radius_std_cm;
    float patch_density;
    float placement_radius_frac;
    FishAgentState agents[MAX_AGENTS];
    FishFood food[MAX_FOOD];
    int num_food;
    int food_eaten;
    int eod_agent_steps;
    int collisions_fish;
    float episode_return;

    FishMonopole eod_sources[2 * MAX_AGENTS];
    FishDipole intrinsic_sources[MAX_AGENTS + MAX_FOOD];
    FishDipole induced_sources[MAX_AGENTS + MAX_FOOD];
    int num_intrinsic_sources;
    int num_induced_sources;
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

void init(FishEnv* env) {
    if (env->num_agents <= 0) env->num_agents = 4;
    if (env->num_agents > MAX_AGENTS) env->num_agents = MAX_AGENTS;
    if (env->arena_size_cm.x <= 0.0f) env->arena_size_cm.x = 70.0f;
    if (env->arena_size_cm.y <= 0.0f) env->arena_size_cm.y = 70.0f;
    if (env->min_arena_size_cm.x <= 0.0f) {
        env->min_arena_size_cm.x = env->arena_size_cm.x;
    }
    if (env->min_arena_size_cm.y <= 0.0f) {
        env->min_arena_size_cm.y = env->arena_size_cm.y;
    }
    if (env->max_arena_size_cm.x <= 0.0f) {
        env->max_arena_size_cm.x = env->min_arena_size_cm.x;
    }
    if (env->max_arena_size_cm.y <= 0.0f) {
        env->max_arena_size_cm.y = env->min_arena_size_cm.y;
    }
    if (env->configured_num_food <= 0) env->configured_num_food = MAX_FOOD;
    if (env->configured_num_food > MAX_FOOD) {
        env->configured_num_food = MAX_FOOD;
    }
    if (env->fixed_num_patches <= 0) env->fixed_num_patches = 4;
    if (env->patch_radius_cm <= 0.0f) env->patch_radius_cm = 6.0f;
    if (env->patch_radius_std_cm < 0.0f) env->patch_radius_std_cm = 1.5f;
    if (env->patch_density <= 0.0f) env->patch_density = 0.001f;
    if (env->placement_radius_frac <= 0.0f) {
        env->placement_radius_frac = 0.75f;
    }
    env->placement_radius_frac =
        clamp(env->placement_radius_frac, 0.0f, 1.0f);
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
    for (int i = 0; i < env->num_agents; i++) {
        FishAgentState* agent = &env->agents[i];
        float* obs = env->observations + i * OBS_SIZE;
        int cursor = 0;

        /* Mormyromasts: active/collective image after direct-EOD subtraction. */
        for (int sensor_idx = 0; sensor_idx < NUM_MORMYROMASTS;
                sensor_idx++) {
            float angle = mormyromast_angle(
                (size_t)sensor_idx, 10, 26, PI_F / 3.0f
            );
            FishSensor sensor = transform_sensor(
                radial_sensor(angle, agent->movement.body_radius_cm),
                agent->movement.position_cm,
                (float)agent->movement.orientation
            );
            FishVec2 induced = measure_electric_field_with_reflections(
                sensor.position_cm, NULL, 0,
                env->induced_sources, (size_t)env->num_induced_sources,
                env->arena_size_cm, FIELD_EPS_M,
                REFLECTION_SCALE, false
            );
            float reading = project_field(induced, sensor.normal);
            if (!agent->emits_eod) {
                reading *= 100.0f;
            }
            reading *= random_multiplier(env, 0.05f);
            obs[cursor++] = (float)normalize_sensor_reading(
                reading, MORMYROMAST_MIN_VM,
                MORMYROMAST_MAX_VM, SENSOR_EPS
            );
        }

        // Ampullary receptors: intrinsic sources with static self-field removed
        for (int sensor_idx = 0; sensor_idx < NUM_AMPULLARY;
                sensor_idx++) {
            float angle = uniform_sensor_angle(
                (size_t)sensor_idx, NUM_AMPULLARY
            );
            FishSensor sensor = transform_sensor(
                radial_sensor(angle, agent->movement.body_radius_cm),
                agent->movement.position_cm,
                (float)agent->movement.orientation
            );
            FishVec2 field = measure_electric_field_with_reflections(
                sensor.position_cm, NULL, 0,
                env->intrinsic_sources, (size_t)env->num_intrinsic_sources,
                env->arena_size_cm, FIELD_EPS_M,
                REFLECTION_SCALE, false
            );
            float reading =
                project_field(field, sensor.normal) -
                env->amp_intrinsic_baseline[sensor_idx];
            bool cons_eod = false;
            for (int other = 0; other < env->num_agents; other++) {
                if (other != i && env->agents[other].emits_eod) {
                    cons_eod = true;
                    break;
                }
            }
            reading *= random_multiplier(env, cons_eod ? 0.5f : 0.05f);
            float processed = normalize_sensor_reading(
                reading, AMPULLARY_MIN_VM,
                AMPULLARY_MAX_VM, SENSOR_EPS
            );
            obs[cursor++] = (float)processed;
        }

        /* Knollenorgans: one directional 12-receptor block per conspecific. */
        int metadata_start =
            NUM_MORMYROMASTS + NUM_AMPULLARY +
            NUM_KNOLLEN * (MAX_AGENTS - 1);
        int cons_slot = 0;
        for (int other = 0; other < MAX_AGENTS; other++) {
            if (other == i) continue;
            bool valid = other < env->num_agents && env->agents[other].emits_eod;
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN;
                    sensor_idx++) {
                float value = 0.0f;
                if (valid) {
                    float angle = uniform_sensor_angle(
                        (size_t)sensor_idx, NUM_KNOLLEN
                    );
                    FishSensor sensor = transform_sensor(
                        radial_sensor(
                            angle, agent->movement.body_radius_cm
                        ),
                        agent->movement.position_cm,
                        (float)agent->movement.orientation
                    );
                    FishVec2 field = measure_electric_field(
                        sensor.position_cm,
                        &env->eod_sources[2 * other], 2,
                        NULL, 0, FIELD_EPS_M
                    );
                    float raw_knollen =
                        project_field(field, sensor.normal);
                    raw_knollen *= random_multiplier(env, 0.05f);
                    value = fabsf(raw_knollen) <= KNOLLEN_MIN_VM
                        ? 0.0f
                        : (raw_knollen < 0.0f ? -1.0f : 1.0f);
                }
                obs[cursor++] = (float)value;
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
            obs[metadata_start + cons_slot] = (float)metadata;
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
        obs[cursor++] = (float)agent->size;
        obs[cursor++] =
            (float)agent->bite_cooldown / (float)BITE_COOLDOWN_STEPS;
        obs[cursor++] = (float)clamp(
            agent->movement.displacement_ego_cm.x /
                (float)agent->movement.max_linear_velocity, -1.0f, 1.0f);
        obs[cursor++] = (float)clamp(
            agent->movement.displacement_ego_cm.y /
                (float)agent->movement.max_linear_velocity, -1.0f, 1.0f);
        obs[cursor++] =
            (float)agent->eat_cooldown / (float)EAT_COOLDOWN_STEPS;
    }
}

bool
position_overlaps_agents(
    const FishEnv* env, FishVec2 position, int count, float radius
) {
    for (int i = 0; i < count; i++) {
        FishVec2 delta = {
            position.x - env->agents[i].movement.position_cm.x,
            position.y - env->agents[i].movement.position_cm.y,
        };
        float minimum = radius + env->agents[i].movement.body_radius_cm;
        if (vec_length_squared(delta) < minimum * minimum) return true;
    }
    return false;
}

/*
 * Matches SensingModel._calculate_corollary_discharge: baselines are computed
 * once for a representative fish at the arena center, orientation zero.
 */
void calibrate_electroreceptors(FishEnv* env) {
    FishVec2 center = {
        env->arena_size_cm.x * 0.5f,
        env->arena_size_cm.y * 0.5f,
    };
    FishMonopole eod[2] = {
        {
            .position_cm = {
                center.x + EOD_POLE_OFFSET_CM, center.y
            },
            .charge_c = EOD_CHARGE_C,
        },
        {
            .position_cm = {
                center.x - EOD_POLE_OFFSET_CM, center.y
            },
            .charge_c = -EOD_CHARGE_C,
        },
    };
    FishDipole intrinsic = {
        .position_cm = center,
        .moment_c_m = {INTRINSIC_MOMENT_C_M, 0.0f},
    };

    for (int i = 0; i < NUM_MORMYROMASTS; i++) {
        float angle = mormyromast_angle(
            (size_t)i, 10, 26, PI_F / 3.0f
        );
        FishSensor sensor = radial_sensor(angle, BODY_RADIUS_CM);
        sensor.position_cm.x += center.x;
        sensor.position_cm.y += center.y;
        FishVec2 field = measure_electric_field_with_reflections(
            sensor.position_cm, eod, 2, NULL, 0,
            env->arena_size_cm, FIELD_EPS_M,
            REFLECTION_SCALE, false
        );
        env->mormyromast_cd[i] = project_field(field, sensor.normal);
    }

    for (int i = 0; i < NUM_AMPULLARY; i++) {
        float angle = uniform_sensor_angle(
            (size_t)i, NUM_AMPULLARY
        );
        FishSensor sensor = radial_sensor(angle, BODY_RADIUS_CM);
        sensor.position_cm.x += center.x;
        sensor.position_cm.y += center.y;
        FishVec2 field = measure_electric_field_with_reflections(
            sensor.position_cm, NULL, 0, &intrinsic, 1,
            env->arena_size_cm, FIELD_EPS_M,
            REFLECTION_SCALE, false
        );
        env->amp_intrinsic_baseline[i] =
            project_field(field, sensor.normal);
    }
}

void build_electric_scene(FishEnv* env) {
    env->num_intrinsic_sources = 0;
    env->num_induced_sources = 0;
    for (int i = 0; i < env->num_agents; i++) {
        FishAgentState* agent = &env->agents[i];
        FishMonopole positive = {
            .position_cm = {EOD_POLE_OFFSET_CM, 0.0f},
            .charge_c = EOD_CHARGE_C,
        };
        FishMonopole negative = {
            .position_cm = {-EOD_POLE_OFFSET_CM, 0.0f},
            .charge_c = -EOD_CHARGE_C,
        };
        float agent_ori = (float)agent->movement.orientation;
        env->eod_sources[2 * i] = transform_monopole(
            positive, agent->movement.position_cm, agent_ori,
            agent->emits_eod
        );
        env->eod_sources[2 * i + 1] = transform_monopole(
            negative, agent->movement.position_cm, agent_ori,
            agent->emits_eod
        );
        env->intrinsic_sources[env->num_intrinsic_sources++] = (FishDipole){
            .position_cm = agent->movement.position_cm,
            .moment_c_m = rotate(
                (FishVec2){INTRINSIC_MOMENT_C_M, 0.0f},
                agent_ori
            ),
        };
    }

    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) continue;
        FishDipole intrinsic = {
            .position_cm = env->food[i].motion.position_cm,
            .moment_c_m = rotate(
                (FishVec2){0.0f, FOOD_INTRINSIC_MOMENT_C_M},
                env->food[i].motion.orientation
            ),
        };
        env->intrinsic_sources[env->num_intrinsic_sources++] = intrinsic;
    }

    // EOD fields induce dipoles on prey and agent bodies (Chen et al., Eq. 6) 
    for (int i = 0; i < env->num_agents; i++) {
        FishVec2 position = env->agents[i].movement.position_cm;
        FishVec2 moment = {0.0f, 0.0f};
        if (!env->agents[i].emits_eod) {
            FishVec2 field = measure_electric_field(
                position, env->eod_sources, (size_t)(2 * env->num_agents),
                NULL, 0, FIELD_EPS_M
            );
            moment = induce_dipole_moment(
                field, env->agents[i].movement.body_radius_cm,
                CONDUCTOR_CONTRAST
            );
        }
        float moment_magnitude = vec_length(moment);
        float max_moment =
            EOD_CHARGE_C * env->agents[i].movement.body_radius_cm;
        if (EOD_CHARGE_C >= 0.0f && moment_magnitude > max_moment) {
            float moment_scale = max_moment / moment_magnitude;
            moment = (FishVec2){
                moment.x * moment_scale,
                moment.y * moment_scale,
            };
        }
        env->induced_sources[env->num_induced_sources++] = (FishDipole){
            .position_cm = position, .moment_c_m = moment
        };
    }
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) continue;
        FishVec2 position = env->food[i].motion.position_cm;
        FishVec2 field = measure_electric_field(
            position, env->eod_sources, (size_t)(2 * env->num_agents),
            NULL, 0, FIELD_EPS_M
        );
        env->induced_sources[env->num_induced_sources++] = (FishDipole){
            .position_cm = position,
            .moment_c_m = induce_dipole_moment(
                field, FOOD_RADIUS_CM, CONDUCTOR_CONTRAST
            ),
        };
    }
}

void c_reset(FishEnv* env) {
    init(env);
    env->arena_size_cm = (FishVec2){
        random_uniform(
            env, env->min_arena_size_cm.x, env->max_arena_size_cm.x
        ),
        random_uniform(
            env, env->min_arena_size_cm.y, env->max_arena_size_cm.y
        ),
    };
    env->active_food_distribution = env->food_distribution;
    if (env->food_distribution == FOOD_RANDOM) {
        env->active_food_distribution =
            (FoodDistribution)(rand_r(&env->rng) % 4);
    }
    env->tick = 0;
    env->food_eaten = 0;
    env->eod_agent_steps = 0;
    env->collisions_fish = 0;
    env->episode_return = 0.0f;

    for (int i = 0; i < env->num_agents; i++) {
        FishAgentState agent = {0};
        agent.size = random_uniform(env, 0.0f, 1.0f);
        FishVec2 position;
        int attempts = 0;
        do {
            position = (FishVec2){
                (float)random_uniform(env, 3.0f, (float)env->arena_size_cm.x - 3.0f),
                (float)random_uniform(env, 3.0f, (float)env->arena_size_cm.y - 3.0f)
            };
        } while (
            position_overlaps_agents(
                env, position, i, BODY_RADIUS_CM
            ) && ++attempts < 1000
        );
        agent.movement = (FishMovement){
            .position_cm = position,
            .orientation = random_uniform(env, -PI_F, PI_F),
            .max_linear_velocity = 35.0f / (float)SIMULATION_HZ,
            .max_angular_velocity = 3.6f / (float)SIMULATION_HZ,
            .body_radius_cm = BODY_RADIUS_CM,
            .min_linear_velocity = -5.0f / SIMULATION_HZ,
            .max_linear_acceleration = 650.0f / (SIMULATION_HZ * SIMULATION_HZ),
            .min_angular_velocity = -3.5f / SIMULATION_HZ,
            .max_angular_acceleration = 318.0f / (SIMULATION_HZ * SIMULATION_HZ),
            .linear_drag_factor = 0.95f,
            .angular_drag_factor = 0.95f,
            .motion_order = MOTION_FIRST_ORDER,
            .backwards = false,
        };
        float size_multiplier = 1.0f + agent.size;
        agent.movement.min_linear_velocity *= (float)size_multiplier;
        agent.movement.max_linear_velocity *= size_multiplier;
        agent.movement.max_linear_acceleration *= (float)size_multiplier;
        agent.movement.min_angular_velocity *= (float)size_multiplier;
        agent.movement.max_angular_velocity *= size_multiplier;
        agent.movement.max_angular_acceleration *= (float)size_multiplier;
        agent.emits_eod = true;
        env->agents[i] = agent;
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0.0f;
    }

    FishVec2 patch_centers[MAX_PATCHES];
    float patch_radii[MAX_PATCHES];
    float patch_weights[MAX_PATCHES];
    int num_patches = 0;
    float total_patch_weight = 0.0f;
    float minimum_edge = fminf(env->arena_size_cm.x, env->arena_size_cm.y);

    if (env->active_food_distribution == FOOD_PATCHY) {
        num_patches = (int)ceilf(
            env->patch_density *
            env->arena_size_cm.x * env->arena_size_cm.y
        );
        num_patches = (int)clamp(num_patches, 1, MAX_PATCHES);
        for (int i = 0; i < num_patches; i++) {
            patch_centers[i] = (FishVec2){
                random_uniform(env, 0.0f, env->arena_size_cm.x),
                random_uniform(env, 0.0f, env->arena_size_cm.y),
            };
            float u1 = fmaxf(
                random_uniform(env, 0.0f, 1.0f), 1.0f / (float)RAND_MAX
            );
            float u2 = random_uniform(env, 0.0f, 1.0f);
            float normal_sample =
                sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_F * u2);
            patch_radii[i] = clamp(
                env->patch_radius_cm +
                    normal_sample * env->patch_radius_std_cm,
                1.0f, minimum_edge / 2.0f
            );
            patch_weights[i] = patch_radii[i] * patch_radii[i];
            total_patch_weight += patch_weights[i];
        }
    } else if (env->active_food_distribution == FOOD_N_PATCH) {
        num_patches = (int)clamp(
            env->fixed_num_patches, 1, MAX_PATCHES
        );
        float initial_angle = random_uniform(env, 0.0f, 2.0f * PI_F);
        for (int i = 0; i < num_patches; i++) {
            float angle =
                initial_angle + 2.0f * PI_F * i / num_patches;
            patch_centers[i] = (FishVec2){
                env->arena_size_cm.x * 0.5f *
                    (1.0f + env->placement_radius_frac * cosf(angle)),
                env->arena_size_cm.y * 0.5f *
                    (1.0f + env->placement_radius_frac * sinf(angle)),
            };
            patch_radii[i] = fminf(env->patch_radius_cm, minimum_edge / 2.0f);
            patch_weights[i] = patch_radii[i] * patch_radii[i];
            total_patch_weight += patch_weights[i];
        }
    } else if (env->active_food_distribution == FOOD_ONE_PATCH) {
        num_patches = 1;
        patch_centers[0] = (FishVec2){
            env->arena_size_cm.x / 2.0f,
            env->arena_size_cm.y / 2.0f,
        };
        float u1 = fmaxf(
            random_uniform(env, 0.0f, 1.0f), 1.0f / (float)RAND_MAX
        );
        float u2 = random_uniform(env, 0.0f, 1.0f);
        float normal_sample =
            sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_F * u2);
        patch_radii[0] = clamp(
            env->patch_radius_cm +
                normal_sample * env->patch_radius_std_cm,
            0.1f, minimum_edge / 2.0f
        );
        patch_weights[0] = patch_radii[0] * patch_radii[0];
        total_patch_weight = patch_weights[0];
    }

    env->num_food = env->configured_num_food;
    for (int i = 0; i < env->num_food; i++) {
        FishVec2 food_position;
        if (env->active_food_distribution == FOOD_UNIFORM) {
            food_position = (FishVec2){
                random_uniform(env, 0.0f, env->arena_size_cm.x),
                random_uniform(env, 0.0f, env->arena_size_cm.y),
            };
        } else {
            float target =
                random_uniform(env, 0.0f, total_patch_weight);
            int patch = 0;
            while (patch < num_patches - 1 &&
                    target > patch_weights[patch]) {
                target -= patch_weights[patch++];
            }
            int attempts = 0;
            do {
                float angle = random_uniform(env, 0.0f, 2.0f * PI_F);
                float radius = patch_radii[patch] * sqrtf(
                    random_uniform(env, 0.0f, 1.0f)
                );
                food_position = (FishVec2){
                    patch_centers[patch].x + radius * cosf(angle),
                    patch_centers[patch].y + radius * sinf(angle),
                };
            } while (
                (food_position.x < 0.0f ||
                 food_position.x >= env->arena_size_cm.x ||
                 food_position.y < 0.0f ||
                 food_position.y >= env->arena_size_cm.y) &&
                ++attempts < 1000
            );
            food_position.x = clamp(
                food_position.x, 0.0f, env->arena_size_cm.x
            );
            food_position.y = clamp(
                food_position.y, 0.0f, env->arena_size_cm.y
            );
        }
        env->food[i] = (FishFood){
            .motion = {
                .position_cm = food_position,
                .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
            },
            .active = true,
        };
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
    FishAgentState* agent = &env->agents[agent_idx];
    int nearest = -1;
    float nearest_distance = INFINITY;
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) continue;
        if (!point_in_forward_cone(
                agent->movement.position_cm,
                (float)agent->movement.orientation,
                env->food[i].motion.position_cm,
                EATING_RADIUS_CM, EATING_ANGLE)) continue;
        FishVec2 offset = {
            env->food[i].motion.position_cm.x - agent->movement.position_cm.x,
            env->food[i].motion.position_cm.y - agent->movement.position_cm.y,
        };
        float distance = vec_length_squared(offset);
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
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].was_bitten = false;
        env->agents[i].ate_food = false;
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0.0f;
    }

    for (int food_idx = 0; food_idx < env->num_food; food_idx++) {
        if (!env->food[food_idx].active) continue;
        FishFoodMotion* food = &env->food[food_idx].motion;
        food->position_cm.x = clamp(
            food->position_cm.x + food->velocity_cm.x,
            0.0f, env->arena_size_cm.x
        );
        food->position_cm.y = clamp(
            food->position_cm.y + food->velocity_cm.y,
            0.0f, env->arena_size_cm.y
        );
    }

    for (int i = 0; i < env->num_agents; i++) {
        FishAgentState* agent = &env->agents[i];
        float* raw_action = env->actions + i * ACTION_SIZE;
        float move = 1.0f / (1.0f + expf(-(float)raw_action[0]));
        float turn = tanhf((float)raw_action[1]);
        agent->emits_eod = raw_action[2] > 0.0f;
        agent->bite_action =
            raw_action[3] > 0.0f && agent->bite_cooldown <= 0;
        if (agent->bite_action) agent->bite_cooldown = BITE_COOLDOWN_STEPS;
        agent->last_action[0] = (float)move;
        agent->last_action[1] = (float)turn;
        agent->last_action[2] = agent->emits_eod ? 1.0f : 0.0f;
        agent->last_action[3] = agent->bite_action ? 1.0f : 0.0f;
        env->eod_agent_steps += agent->emits_eod ? 1 : 0;

        if (!agent->bite_action && agent->eat_cooldown <= 0) {
            if (try_eat(env, i)) {
                env->rewards[i] += EAT_REWARD;
            }
        }
        FishMotionProposal proposal = propose_motion(
            &agent->movement, move, turn, agent->eat_cooldown > 0
        );
        FishMovement movement_views[MAX_AGENTS];
        for (int j = 0; j < env->num_agents; j++) {
            movement_views[j] = env->agents[j].movement;
        }
        bool collided = motion_collides(
            proposal.position_cm, agent->movement.body_radius_cm,
            movement_views, (size_t)env->num_agents, (size_t)i
        );
        bool hit_wall = commit_motion(
            &agent->movement, proposal, collided, env->arena_size_cm
        );
        env->collisions_fish += collided ? 1 : 0;
        if (collided || hit_wall) {
            env->rewards[i] += COLLISION_REWARD;
        }
    }

    /* Bite resolution occurs after all fish have moved. */
    for (int i = 0; i < env->num_agents; i++) {
        FishAgentState* attacker = &env->agents[i];
        if (!attacker->bite_action) continue;
        int victim = -1;
        float nearest = INFINITY;
        for (int j = 0; j < env->num_agents; j++) {
            if (i == j) continue;
            if (!point_in_forward_cone(
                    attacker->movement.position_cm,
                    (float)attacker->movement.orientation,
                    env->agents[j].movement.position_cm,
                    BITING_RADIUS_CM, EATING_ANGLE)) continue;
            FishVec2 offset = {
                env->agents[j].movement.position_cm.x -
                    attacker->movement.position_cm.x,
                env->agents[j].movement.position_cm.y -
                    attacker->movement.position_cm.y,
            };
            float distance = vec_length_squared(offset);
            if (distance < nearest) {
                victim = j;
                nearest = distance;
            }
        }
        if (victim >= 0) {
            env->agents[victim].was_bitten = true;
            float size_difference = attacker->size - env->agents[victim].size;
            env->rewards[victim] += BITTEN_REWARD * (float)(1.0f + fmaxf(0.0f, size_difference));
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        FishAgentState* agent = &env->agents[i];
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

Vector2 world_to_screen(const FishEnv* env, FishVec2 point_cm) {
    Client* client = env->client;
    float usable_width = client->window_width - 2.0f * client->margin;
    float usable_height = client->window_height - 2.0f * client->margin;
    Vector2 result = {
        (float)(client->margin +
            point_cm.x / env->arena_size_cm.x * usable_width),
        (float)(client->window_height - client->margin -
            point_cm.y / env->arena_size_cm.y * usable_height),
    };
    return result;
}

float render_scale(const FishEnv* env) {
    Client* client = env->client;
    float sx = (client->window_width - 2.0f * client->margin) /
        env->arena_size_cm.x;
    float sy = (client->window_height - 2.0f * client->margin) /
        env->arena_size_cm.y;
    return (float)fminf(sx, sy);
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

void render_field(
    FishEnv* env,
    const FishMonopole* monopoles,
    const FishDipole* dipoles
) {
    const int columns = 25;
    const int rows = 25;
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            FishVec2 position = {
                env->arena_size_cm.x * (column + 0.5f) / columns,
                env->arena_size_cm.y * (row + 0.5f) / rows
            };
            bool near_fish = false;
            float radius_squared =
                env->electric_field_radius_cm * env->electric_field_radius_cm;
            for (int i = 0; i < env->num_agents; i++) {
                float dx = position.x - env->agents[i].movement.position_cm.x;
                float dy = position.y - env->agents[i].movement.position_cm.y;
                if (dx * dx + dy * dy <= radius_squared) {
                    near_fish = true;
                    break;
                }
            }
            if (!near_fish) continue;

            FishVec2 field = measure_electric_field_with_reflections(
                position,
                monopoles, (size_t)(2 * env->num_agents),
                dipoles, (size_t)(
                    env->num_intrinsic_sources + env->num_induced_sources
                ),
                env->arena_size_cm,
                FIELD_EPS_M,
                REFLECTION_SCALE,
                false
            );
            float strength = vec_length(field);
            if (strength <= 0.0f) continue;
            float log_strength = log10f(strength);
            float arrow_length = 4.0f + 10.0f *
                clamp((log_strength + 8.0f) / 7.0f, 0.0f, 1.0f);
            FishVec2 direction = {
                field.x / strength,
                field.y / strength,
            };
            Vector2 start = world_to_screen(env, position);
            Vector2 end = {
                start.x + (float)(direction.x * arrow_length),
                start.y - (float)(direction.y * arrow_length),
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

    FishMonopole monopoles[2 * MAX_AGENTS];
    FishDipole dipoles[MAX_DIPOLE_SOURCES];
    for (int i = 0; i < 2 * env->num_agents; i++) {
        monopoles[i] = env->eod_sources[i];
    }
    for (int i = 0; i < env->num_intrinsic_sources; i++) {
        dipoles[i] = env->intrinsic_sources[i];
    }
    for (int i = 0; i < env->num_induced_sources; i++) {
        dipoles[env->num_intrinsic_sources + i] = env->induced_sources[i];
    }
    for (int i = 0; i < env->num_agents; i++) {
        Trace* trace = &env->client->traces[i];
        if (env->terminals[i]) {
            trace->index = 0;
            trace->count = 0;
        }
        trace->positions[trace->index] =
            env->agents[i].movement.position_cm;
        trace->index = (trace->index + 1) % TRACE_LENGTH;
        if (trace->count < TRACE_LENGTH) trace->count++;
    }

    BeginDrawing();
    ClearBackground(WHITE);

    Vector2 arena_min = world_to_screen(
        env, (FishVec2){0.0f, env->arena_size_cm.y}
    );
    Vector2 arena_max = world_to_screen(
        env, (FishVec2){env->arena_size_cm.x, 0.0f}
    );
    DrawRectangleRec(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        WHITE
    );

    if (env->client->show_field) {
        render_field(env, monopoles, dipoles);
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
                world_to_screen(env, trace->positions[current]),
                world_to_screen(env, trace->positions[previous]),
                2.0f, ColorAlpha(color, alpha)
            );
        }
    }

    float scale = render_scale(env);
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) continue;
        Vector2 position = world_to_screen(
            env, env->food[i].motion.position_cm
        );
        float radius = fmaxf(2.5f, (float)(FOOD_RADIUS_CM * scale));
        DrawCircleV(position, radius, (Color){80, 220, 125, 255});
        DrawCircleLines(
            (int)position.x, (int)position.y, radius,
            (Color){190, 255, 205, 255}
        );
    }
    for (int i = 0; i < env->num_agents; i++) {
        FishAgentState* agent = &env->agents[i];
        FishMovement* fish = &agent->movement;
        Vector2 center = world_to_screen(env, fish->position_cm);
        float radius = (float)(fish->body_radius_cm * scale);

        if (agent->emits_eod) {
            float pulse = radius + 5.0f +
                5.0f * sinf((float)env->tick * 0.18f);
            DrawCircleLines((int)center.x, (int)center.y, pulse,
                (Color){color.r, color.g, color.b, 120});
        }

        float heading_x = cosf((float)fish->orientation);
        float heading_y = -sinf((float)fish->orientation);
        DrawRing(center, radius - 1.5f, radius + 1.5f,
            0.0f, 360.0f, 32, color);
        Vector2 nose = {
            center.x + heading_x * radius * 1.35f,
            center.y + heading_y * radius * 1.35f,
        };
        DrawLineEx(center, nose, 3.0f, color);

        Vector2 positive = world_to_screen(
            env, monopoles[2 * i].position_cm
        );
        Vector2 negative = world_to_screen(
            env, monopoles[2 * i + 1].position_cm
        );
        DrawCircleV(positive, 3.0f, RED);
        DrawCircleV(negative, 3.0f, BLUE);

        if (env->client->show_sensors) {
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN;
                    sensor_idx++) {
                float angle = uniform_sensor_angle(
                    (size_t)sensor_idx, NUM_KNOLLEN
                );
                FishSensor sensor = radial_sensor(
                    angle, fish->body_radius_cm
                );
                sensor = transform_sensor(
                    sensor,
                    fish->position_cm,
                    (float)fish->orientation
                );
                DrawCircleV(
                    world_to_screen(env, sensor.position_cm),
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
