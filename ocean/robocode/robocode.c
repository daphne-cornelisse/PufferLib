#include "robocode.h"

#define NUM_AGENTS 2

void demo() {

    // TODO: Load policy

    // Allocate memory on the stack
    float observations[NUM_AGENTS*OBS_SIZE] = {0};
    float actions[NUM_AGENTS*NUM_ACTIONS] = {0};
    float rewards[NUM_AGENTS] = {0};
    float terminals[NUM_AGENTS] = {0};
    Robot robots[NUM_AGENTS] = {0};
    Bullet bullets[NUM_BULLETS] = {0};
    
    // Make environment
    Robocode env = {
        .width=768, 
        .height=576,
        .num_agents=2,
        .num_controlled_agents=1,
        .observations = observations,
        .actions=actions,
        .terminals=terminals,
        .rewards=rewards,
        .robots=robots,
        .bullets=bullets,
    };
    allocate_env(&env);
    c_reset(&env);

    Client* client = make_client(&env);

    while (!WindowShouldClose()) {

        // Override policy actions with human actions if provided
        if (IsKeyDown(KEY_W)) env.actions[ACCEL_IDX] = 5.0f;
        if (IsKeyDown(KEY_S)) env.actions[ACCEL_IDX] = -5.0f;
        if (IsKeyDown(KEY_A)) env.actions[TURN_RADIUS] = -1.0f;
        if (IsKeyDown(KEY_D)) env.actions[TURN_RADIUS] = 1.0f;
        if (IsKeyDown(KEY_Q)) env.actions[GUN_TURN_RADIUS] = -0.5f;
        if (IsKeyDown(KEY_E)) env.actions[GUN_TURN_RADIUS] = 0.5f;
    
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            env.actions[FIREPOWER] = 1.0f;
        } else {
            env.actions[FIREPOWER] = 0.0f;
        }

        // TODO: Integrate network

        c_step(&env);
        c_render(client, &env);
    }
    CloseWindow();
}

int main() {
    demo();
}