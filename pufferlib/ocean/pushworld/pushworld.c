#include "pushworld.h"
#include <time.h>

int main(int argc, char** argv) {
    const char* puzzle_dir = argc > 1 ? argv[1] : "resources/pushworld/puzzles/train";
    int max_episode_length = argc > 2 ? atoi(argv[2]) : 500;
    int vision = argc > 3 ? atoi(argv[3]) : 15;
    int seed = argc > 4 ? atoi(argv[4]) : 0;

    srand(seed);
    PuzzleSet* puzzles = pw_load_puzzles(puzzle_dir, NULL, 0, 0);
    PushWorld* env = allocate_pushworld(puzzles, max_episode_length, vision);
    c_reset(env);

    int action = -1;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            c_reset(env);
        }

        action = -1;
        if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
            action = PW_ACTION_LEFT;
        } else if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
            action = PW_ACTION_RIGHT;
        } else if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
            action = PW_ACTION_UP;
        } else if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
            action = PW_ACTION_DOWN;
        }

        if (action >= 0) {
            env->actions[0] = action;
            c_step(env);
        }

        c_render(env);
    }

    free_allocated_pushworld(env);
    pw_free_puzzles(puzzles);
    return 0;
}
