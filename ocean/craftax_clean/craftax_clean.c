#include "craftax_clean.h"

static int key_to_action(void) {
    // Makes it human playable
    if (IsKeyPressed(KEY_Q)) return ACTION_NOOP;
    if (IsKeyPressed(KEY_W) || IsKeyPressed(KEY_UP)) return ACTION_UP;
    if (IsKeyPressed(KEY_D) || IsKeyPressed(KEY_RIGHT)) return ACTION_RIGHT;
    if (IsKeyPressed(KEY_S) || IsKeyPressed(KEY_DOWN)) return ACTION_DOWN;
    if (IsKeyPressed(KEY_A) || IsKeyPressed(KEY_LEFT)) return ACTION_LEFT;
    if (IsKeyPressed(KEY_SPACE)) return ACTION_DO;
    if (IsKeyPressed(KEY_ONE)) return ACTION_MAKE_WOOD_PICKAXE;
    if (IsKeyPressed(KEY_TWO)) return ACTION_MAKE_STONE_PICKAXE;
    if (IsKeyPressed(KEY_THREE)) return ACTION_MAKE_IRON_PICKAXE;
    if (IsKeyPressed(KEY_FOUR)) return ACTION_MAKE_DIAMOND_PICKAXE;
    if (IsKeyPressed(KEY_FIVE)) return ACTION_MAKE_WOOD_SWORD;
    if (IsKeyPressed(KEY_SIX)) return ACTION_MAKE_STONE_SWORD;
    if (IsKeyPressed(KEY_SEVEN)) return ACTION_MAKE_IRON_SWORD;
    if (IsKeyPressed(KEY_EIGHT)) return ACTION_MAKE_DIAMOND_SWORD;
    if (IsKeyPressed(KEY_T)) return ACTION_PLACE_TABLE;
    if (IsKeyPressed(KEY_TAB)) return ACTION_SLEEP;
    if (IsKeyPressed(KEY_R)) return ACTION_PLACE_STONE;
    if (IsKeyPressed(KEY_F)) return ACTION_PLACE_FURNACE;
    if (IsKeyPressed(KEY_P)) return ACTION_PLACE_PLANT;
    if (IsKeyPressed(KEY_E)) return ACTION_REST;
    if (IsKeyPressed(KEY_COMMA)) return ACTION_ASCEND;
    if (IsKeyPressed(KEY_PERIOD)) return ACTION_DESCEND;
    if (IsKeyPressed(KEY_Y)) return ACTION_MAKE_IRON_ARMOUR;
    if (IsKeyPressed(KEY_U)) return ACTION_MAKE_DIAMOND_ARMOUR;
    if (IsKeyPressed(KEY_I)) return ACTION_SHOOT_ARROW;
    if (IsKeyPressed(KEY_O)) return ACTION_MAKE_ARROW;
    if (IsKeyPressed(KEY_G)) return ACTION_CAST_FIREBALL;
    if (IsKeyPressed(KEY_H)) return ACTION_CAST_ICEBALL;
    if (IsKeyPressed(KEY_J)) return ACTION_PLACE_TORCH;
    if (IsKeyPressed(KEY_Z)) return ACTION_DRINK_POTION_RED;
    if (IsKeyPressed(KEY_X)) return ACTION_DRINK_POTION_GREEN;
    if (IsKeyPressed(KEY_C)) return ACTION_DRINK_POTION_BLUE;
    if (IsKeyPressed(KEY_V)) return ACTION_DRINK_POTION_PINK;
    if (IsKeyPressed(KEY_B)) return ACTION_DRINK_POTION_CYAN;
    if (IsKeyPressed(KEY_N)) return ACTION_DRINK_POTION_YELLOW;
    if (IsKeyPressed(KEY_M)) return ACTION_READ_BOOK;
    if (IsKeyPressed(KEY_K)) return ACTION_ENCHANT_SWORD;
    if (IsKeyPressed(KEY_L)) return ACTION_ENCHANT_ARMOUR;
    if (IsKeyPressed(KEY_LEFT_BRACKET)) return ACTION_MAKE_TORCH;
    if (IsKeyPressed(KEY_RIGHT_BRACKET)) return ACTION_LEVEL_UP_DEXTERITY;
    if (IsKeyPressed(KEY_MINUS)) return ACTION_LEVEL_UP_STRENGTH;
    if (IsKeyPressed(KEY_EQUAL)) return ACTION_LEVEL_UP_INTELLIGENCE;
    if (IsKeyPressed(KEY_SEMICOLON)) return ACTION_ENCHANT_BOW;
    return -1;
}

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
    env.agents[0].action_mask = (unsigned char*)calloc(ATN_DIM, sizeof(unsigned char));
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
