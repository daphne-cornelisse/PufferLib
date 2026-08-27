#ifdef CRAFTAX_CLEAN_IN_HEADER

static int key_to_action(void) {
    static const int map[][2] = {
        {KEY_Q, ACTION_NOOP},
        {KEY_W, ACTION_UP},
        {KEY_UP, ACTION_UP},
        {KEY_D, ACTION_RIGHT},
        {KEY_RIGHT, ACTION_RIGHT},
        {KEY_S, ACTION_DOWN},
        {KEY_DOWN, ACTION_DOWN},
        {KEY_A, ACTION_LEFT},
        {KEY_LEFT, ACTION_LEFT},
        {KEY_SPACE, ACTION_DO},
        {KEY_ONE, ACTION_MAKE_WOOD_PICKAXE},
        {KEY_TWO, ACTION_MAKE_STONE_PICKAXE},
        {KEY_THREE, ACTION_MAKE_IRON_PICKAXE},
        {KEY_FOUR, ACTION_MAKE_DIAMOND_PICKAXE},
        {KEY_FIVE, ACTION_MAKE_WOOD_SWORD},
        {KEY_SIX, ACTION_MAKE_STONE_SWORD},
        {KEY_SEVEN, ACTION_MAKE_IRON_SWORD},
        {KEY_EIGHT, ACTION_MAKE_DIAMOND_SWORD},
        {KEY_T, ACTION_PLACE_TABLE},
        {KEY_TAB, ACTION_SLEEP},
        {KEY_R, ACTION_PLACE_STONE},
        {KEY_F, ACTION_PLACE_FURNACE},
        {KEY_P, ACTION_PLACE_PLANT},
        {KEY_E, ACTION_REST},
        {KEY_COMMA, ACTION_ASCEND},
        {KEY_PERIOD, ACTION_DESCEND},
        {KEY_Y, ACTION_MAKE_IRON_ARMOUR},
        {KEY_U, ACTION_MAKE_DIAMOND_ARMOUR},
        {KEY_I, ACTION_SHOOT_ARROW},
        {KEY_O, ACTION_MAKE_ARROW},
        {KEY_G, ACTION_CAST_FIREBALL},
        {KEY_H, ACTION_CAST_ICEBALL},
        {KEY_J, ACTION_PLACE_TORCH},
        {KEY_Z, ACTION_DRINK_POTION_RED},
        {KEY_X, ACTION_DRINK_POTION_GREEN},
        {KEY_C, ACTION_DRINK_POTION_BLUE},
        {KEY_V, ACTION_DRINK_POTION_PINK},
        {KEY_B, ACTION_DRINK_POTION_CYAN},
        {KEY_N, ACTION_DRINK_POTION_YELLOW},
        {KEY_M, ACTION_READ_BOOK},
        {KEY_K, ACTION_ENCHANT_SWORD},
        {KEY_L, ACTION_ENCHANT_ARMOUR},
        {KEY_LEFT_BRACKET, ACTION_MAKE_TORCH},
        {KEY_RIGHT_BRACKET, ACTION_LEVEL_UP_DEXTERITY},
        {KEY_MINUS, ACTION_LEVEL_UP_STRENGTH},
        {KEY_EQUAL, ACTION_LEVEL_UP_INTELLIGENCE},
        {KEY_SEMICOLON, ACTION_ENCHANT_BOW},
    };
    for (int i = 0; i < (int)(sizeof(map) / sizeof(map[0])); i++) {
        if (IsKeyPressed(map[i][0])) {
            return map[i][1];
        }
    }
    return -1;
}

// Shift + action-panel key. 1 = applied, 0 = policy, -1 = skip tick.
int craftax_clean_human_controls(Craftax* env) {
    int shift = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
    if (!IsWindowReady() || !shift
        || env->state.is_sleeping || env->state.is_resting) {
        return 0;
    }
    int action = key_to_action();
    if (action < 0) {
        return -1;
    }
    env->agents[0].actions[0] = (float)action;
    return 1;
}

#else

#include "craftax_clean.h"

int main(void) {
    Craftax env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.rng = 1;
    env.seed = 1;
    env.use_action_mask = 1;

    env.agents[0].observations = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    env.agents[0].action_mask = (unsigned char*)calloc(ATN_DIM, 1);
    puf_reset(&env);
    env.agents[0].actions[0] = -1.0f;

    puf_render(&env);
    while (!WindowShouldClose()) {
        int action = key_to_action();
        if (action < 0) {
            env.agents[0].actions[0] = -1.0f;
            puf_render(&env);
            continue;
        }
        env.agents[0].actions[0] = (float)action;
        puf_step(&env);
        puf_render(&env);
    }

    puf_close(&env);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    free(env.agents[0].action_mask);
    return 0;
}

#endif
