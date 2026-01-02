#include "grid.h"
#include "rlgl.h"
#include <unistd.h>
#include <sys/wait.h>
#include <stdint.h>

// Defaults if args not provided via command line
const int MAX_GRID_SIZE = 20;
const int HORIZON = 128;
const float SPEED = 1.0;
const int VISION = 5;
const int DISCRETIZE = 1;
const float DIFFICULTY = 0.85;
const int FPS = 15;
const int SEED = 0;

typedef struct {
    int pipefd[2];
    pid_t pid;
} VideoRecorder;

bool OpenVideo(VideoRecorder *recorder, const char *output_filename, int width, int height, int fps) {
    if (pipe(recorder->pipefd) == -1) {
        fprintf(stderr, "Failed to create pipe\n");
        return false;
    }

    recorder->pid = fork();
    if (recorder->pid == -1) {
        fprintf(stderr, "Failed to fork\n");
        return false;
    }

    if (recorder->pid == 0) {
        close(recorder->pipefd[1]);
        dup2(recorder->pipefd[0], STDIN_FILENO);
        close(recorder->pipefd[0]);
        
        for (int fd = 3; fd < 256; fd++) close(fd);
        
        char size_str[64], fps_str[16];
        snprintf(size_str, sizeof(size_str), "%dx%d", width, height);
        snprintf(fps_str, sizeof(fps_str), "%d", fps);
        
        execlp("ffmpeg", "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", 
               "-s", size_str, "-r", fps_str, "-i", "-", "-c:v", "libx264", 
               "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23", 
               "-loglevel", "error", output_filename, NULL);
        
        fprintf(stderr, "Failed to launch ffmpeg\n");
        exit(1);
    }

    close(recorder->pipefd[0]);
    return true;
}

void WriteFrame(VideoRecorder *recorder, int width, int height) {
    unsigned char *screen_data = rlReadScreenPixels(width, height);
    write(recorder->pipefd[1], screen_data, width * height * 4 * sizeof(*screen_data));
    RL_FREE(screen_data);
}

void CloseVideo(VideoRecorder *recorder) {
    close(recorder->pipefd[1]);
    waitpid(recorder->pid, NULL, 0);
}

int main(int argc, char *argv[]) {
    int horizon = argc >= 2 ? atoi(argv[1]) : HORIZON;
    char* output_path = argc >= 3 ? argv[2] : NULL;
    int fps = argc >= 4 ? atoi(argv[3]) : FPS;
    char* actions_path = argc >= 5 ? argv[4] : NULL;
    
    // Environment parameters
    int seed = argc >= 6 ? atoi(argv[5]) : SEED;
    int max_size = argc >= 7 ? atoi(argv[6]) : MAX_GRID_SIZE-1;
    int size = argc >= 8 ? atoi(argv[7]) : MAX_GRID_SIZE-1;
    float speed = argc >= 9 ? atof(argv[8]) : SPEED;
    int vision = argc >= 10 ? atoi(argv[9]) : VISION;
    int discretize = argc >= 11 ? atoi(argv[10]) : DISCRETIZE;
    float difficulty = argc >= 12 ? atof(argv[11]) : DIFFICULTY; 
    
    int num_agents = 1;

    // Initialize environment
    Grid* env = allocate_grid(max_size, num_agents, horizon, vision, speed, discretize);
    
    State* levels = calloc(1, sizeof(State));
    
    if (size % 2 == 0) {
        size -= 1;
    }
    
    // Generate maze with same seed as Python (important so that the actions and observations align)
    create_maze_level(env, size, size, difficulty, seed);
    init_state(levels, max_size, num_agents);
    get_state(env, levels);
    env->num_maps = 1;
    env->levels = levels;

    // Load actions
    int* loaded_actions = NULL;
    if (actions_path != NULL) {
        FILE* f = fopen(actions_path, "rb");
        if (f != NULL) {
            // Get file size to determine the effective horizon
            fseek(f, 0, SEEK_END);
            long file_size = ftell(f);
            fseek(f, 0, SEEK_SET);
            
            horizon = file_size / sizeof(int);
            loaded_actions = (int*)malloc(horizon * sizeof(int));
            fread(loaded_actions, sizeof(int), horizon, f);
            fclose(f);
            printf("Loaded %d actions\n", horizon);
        }
    }
    
    int render_width = 512, render_height = 512;
    
    c_render(env);
    VideoRecorder recorder = {0};
    bool recording = false;
    if (output_path != NULL) {
        recording = OpenVideo(&recorder, output_path, render_width, render_height, fps);
        if (recording) {
            printf("Recording video to: %s\n", output_path);
        }
    }
    
    // Main loop
    int frame_count = 0;
    while (!WindowShouldClose() && frame_count < horizon) {
        Agent* agent = &env->agents[0];
        env->actions[0] = ATN_FORWARD;

        // TODO: Why are up and down flipped?
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                //env->actions[0] = ATN_FORWARD;
                agent->direction = 3.0*PI/2.0;
            } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                //env->actions[0] = ATN_BACK;
                agent->direction = PI/2.0;
            } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                //env->actions[0] = ATN_LEFT;
                agent->direction = PI;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                //env->actions[0] = ATN_RIGHT;
                agent->direction = 0;
            } else {
                env->actions[0] = ATN_PASS;
            }
        } else {
            if (loaded_actions != NULL) {
                env->actions[0] = loaded_actions[frame_count];
            } else {
                env->actions[0] = rand() % 5;
            }
        }

        c_step(env);
        c_render(env);
        if (recording) WriteFrame(&recorder, render_width, render_height);
        frame_count++;

        // Break if goal reached or effective horizon exceeded
        if (env->terminals[0] || frame_count >= env->horizon) {
            break;
        }
    }
    
    printf("Completed %d frames\n", frame_count);
    
    if (recording) {
        CloseVideo(&recorder);
        printf("Video saved to: %s\n", output_path);
    }
    
    if (loaded_actions) free(loaded_actions);
    if (levels) {
        if (levels->grid) free(levels->grid);
        if (levels->agents) free(levels->agents);
        free(levels);
    }
    
    free_allocated_grid(env);
    CloseWindow();
    return 0;
}