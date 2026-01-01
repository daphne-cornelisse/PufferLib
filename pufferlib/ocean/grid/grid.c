#include "grid.h"
#include "rlgl.h"
#include <unistd.h>
#include <sys/wait.h>

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

    if (recorder->pid == 0) { // Child process: run ffmpeg
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
    // Environment parameters
    int max_size = 32;
    int num_agents = 1;
    int horizon = 128;
    float speed = 1;
    int vision = 5;
    bool discretize = true;
    int seed = 0;

    // Video recording parameters
    int max_frames = argc >= 2 ? atoi(argv[1]) : 100;
    char* output_path = argc >= 3 ? argv[2] : NULL;
    int fps = argc >= 4 ? atoi(argv[3]) : 15;

    // Initialize environment
    Grid* env = allocate_grid(max_size, num_agents, horizon, vision, speed, discretize);
    
    //env->width = 32;
    //env->height = 32; 
    //env->agents[0].spawn_x = 16;
    //env->agents[0].spawn_y = 16;
    //env->agents[0].color = 6;
    //reset(env, seed);
    //load_locked_room_preset(env);
     
    State* levels = calloc(1, sizeof(State));
    create_maze_level(env, 31, 31, 0.85, seed);
    init_state(levels, max_size, num_agents);
    get_state(env, levels);
    env->num_maps = 1;
    env->levels = levels;
    
    //generate_locked_room(env);
    //State state;
    //init_state(&state, env->max_size, env->num_agents);
    //get_state(env, &state);

    /*
    width = height = 31;
    env->width=31;
    env->height=31;
    env->agents[0].spawn_x = 1;
    env->agents[0].spawn_y = 1;
    reset(env, seed);
    generate_growing_tree_maze(env->grid, env->width, env->height, max_size, 0.85, 0);
    env->grid[(env->height-2)*env->max_size + (env->width - 2)] = GOAL;
    */

    // Initialize rendering first
    c_render(env);
    
    // Setup video recorder
    int render_width = 512, render_height = 512;
    VideoRecorder recorder = {0};
    bool recording = false;
    if (output_path != NULL) {
        recording = OpenVideo(&recorder, output_path, render_width, render_height, fps);
        if (recording) {
            printf("Recording video to: %s\n", output_path);
        } else {
            fprintf(stderr, "Failed to start video recording\n");
        }
    }
    
    // Main loop
    int tick = 0, frame_count = 0;
    while (!WindowShouldClose() && frame_count < max_frames) {
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
            for (int i = 0; i < num_agents; i++) {
                env->actions[i] = rand() % 5;
            }
        }

        //env->actions[0] = actions[t];
        tick = (tick + 1)%12;
        bool done = false;
        if (tick % 1 == 0) {
            c_step(env);
            //printf("direction: %f\n", env->agents[0].direction);
        }
        
        c_render(env);
        if (recording) WriteFrame(&recorder, render_width, render_height);
        frame_count++;
    }
    
    printf("Completed %d frames\n", frame_count);
    
    // Cleanup
    if (recording) {
        CloseVideo(&recorder);
        printf("Video saved to: %s\n", output_path);
    }
    
    if (levels) {
        if (levels->grid) free(levels->grid);
        if (levels->agents) free(levels->agents);
        free(levels);
    }
    
    free_allocated_grid(env);
    //TakeScreenshot("output.png"); 
    
    CloseWindow();
    return 0;
}