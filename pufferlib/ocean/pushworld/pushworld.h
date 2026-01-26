#ifndef PUFFERLIB_PUSHWORLD_H
#define PUFFERLIB_PUSHWORLD_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include <stdint.h>
#include <limits.h>
#include <assert.h>

#include "raylib.h"

#if defined(__GNUC__) || defined(__clang__)
#define PW_UNUSED __attribute__((unused))
#else
#define PW_UNUSED
#endif

#define PW_POSITION_LIMIT 128
#define PW_MAX_ID_LEN 16

#define PW_ACTION_LEFT 0
#define PW_ACTION_RIGHT 1
#define PW_ACTION_UP 2
#define PW_ACTION_DOWN 3
#define PW_NUM_ACTIONS 4

#define PW_TILE_EMPTY 0
#define PW_TILE_WALL 1
#define PW_TILE_AGENT_WALL 2
#define PW_TILE_GOAL 3
#define PW_TILE_AGENT 4
#define PW_TILE_MOVABLE 5
#define PW_TILE_GOAL_OBJECT 6
#define PW_NUM_TILES 7

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float no_op_rate;
    float solved;
    float n;
};

typedef struct {
    char id[PW_MAX_ID_LEN];
    int count;
    int cap;
    int* xs;
    int* ys;
} PixelList;

typedef struct {
    int num_cells;
    int* dx;
    int* dy;
    int width;
    int height;
} Shape;

typedef struct {
    int width;
    int height;
    int num_objects;
    int num_goals;
    Shape* objects;
    int* initial_positions;
    unsigned char* is_goal_object;
    int* goal_object_indices;
    int* goal_positions;
    unsigned char* wall_grid;
    unsigned char* agent_wall_grid;
    unsigned char* goal_grid;
    char* name;
    int level_id;
} Puzzle;

typedef struct {
    int num_puzzles;
    Puzzle* puzzles;
    int max_width;
    int max_height;
    int max_objects;
} PuzzleSet;

typedef struct Renderer Renderer;
struct Renderer {
    int cell_size;
    int width;
    int height;
};

typedef struct {
    Renderer* renderer;
    PuzzleSet* puzzles;
    Puzzle* puzzle;
    int puzzle_idx;
    int width;
    int height;
    int num_objects;
    int num_goals;
    int max_episode_length;
    int vision_radius;
    int obs_size;
    int tick;
    int no_op_moves;
    float count_based_reward_coef;
    int count_based_global;
    float episode_return;
    Log log;
    int forced_puzzle_idx;
    unsigned char last_solved;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int* obj_grid;
    uint64_t* count_hashes;
    unsigned char* count_hash_used;
    int* count_hash_counts;
    int count_hash_capacity;
    int count_hash_mask;
    int count_hash_size;
    int* positions;
    unsigned char* moved;
    int* queue;
    int max_width;
    int max_height;
    int max_objects;
    unsigned char coverage_enabled;
    uint32_t* coverage_goal;
    uint32_t* coverage_agent;
    int coverage_stride;
} PushWorld;

static inline int pw_pos(int x, int y) {
    return x * PW_POSITION_LIMIT + y;
}

static inline int pw_pos_x(int pos) {
    return pos / PW_POSITION_LIMIT;
}

static inline int pw_pos_y(int pos) {
    return pos % PW_POSITION_LIMIT;
}

static bool pw_has_suffix(const char* str, const char* suffix) {
    size_t len = strlen(str);
    size_t suf_len = strlen(suffix);
    if (len < suf_len) {
        return false;
    }
    return strcmp(str + len - suf_len, suffix) == 0;
}

static bool pw_starts_with(const char* str, const char* prefix) {
    return strncmp(str, prefix, strlen(prefix)) == 0;
}

static int pw_parse_level_id(const char* name) {
    if (!pw_starts_with(name, "level")) {
        return -1;
    }
    return atoi(name + 5);
}

static bool pw_level_allowed(const int* levels, int num_levels, int level_id) {
    if (num_levels <= 0) {
        return true;
    }
    for (int i = 0; i < num_levels; i++) {
        if (levels[i] == level_id) {
            return true;
        }
    }
    return false;
}

static void pw_lowercase(char* s) {
    for (int i = 0; s[i] != '\0'; i++) {
        s[i] = (char)tolower((unsigned char)s[i]);
    }
}

static PixelList* pw_find_pixel_list(PixelList* list, int count, const char* id) {
    for (int i = 0; i < count; i++) {
        if (strcmp(list[i].id, id) == 0) {
            return &list[i];
        }
    }
    return NULL;
}

static PixelList* pw_get_pixel_list(PixelList** list, int* count, int* cap, const char* id) {
    PixelList* existing = pw_find_pixel_list(*list, *count, id);
    if (existing != NULL) {
        return existing;
    }
    if (*count >= *cap) {
        int new_cap = (*cap == 0) ? 8 : (*cap * 2);
        PixelList* tmp = (PixelList*)realloc(*list, new_cap * sizeof(PixelList));
        if (!tmp) {
            fprintf(stderr, "Failed to allocate pixel list\n");
            exit(1);
        }
        *list = tmp;
        *cap = new_cap;
    }
    PixelList* pl = &(*list)[*count];
    memset(pl, 0, sizeof(PixelList));
    strncpy(pl->id, id, PW_MAX_ID_LEN - 1);
    pl->id[PW_MAX_ID_LEN - 1] = '\0';
    (*count)++;
    return pl;
}

static void pw_add_pixel(PixelList* pl, int x, int y) {
    if (pl->count >= pl->cap) {
        int new_cap = (pl->cap == 0) ? 32 : (pl->cap * 2);
        int* xs = (int*)realloc(pl->xs, new_cap * sizeof(int));
        int* ys = (int*)realloc(pl->ys, new_cap * sizeof(int));
        if (!xs || !ys) {
            fprintf(stderr, "Failed to allocate pixel storage\n");
            exit(1);
        }
        pl->xs = xs;
        pl->ys = ys;
        pl->cap = new_cap;
    }
    pl->xs[pl->count] = x;
    pl->ys[pl->count] = y;
    pl->count++;
}

static void pw_free_pixel_lists(PixelList* list, int count) {
    if (!list) {
        return;
    }
    for (int i = 0; i < count; i++) {
        free(list[i].xs);
        free(list[i].ys);
    }
    free(list);
}

static Shape pw_build_shape(PixelList* pl, int* out_x, int* out_y) {
    int min_x = 1 << 30;
    int min_y = 1 << 30;
    int max_x = -1;
    int max_y = -1;

    for (int i = 0; i < pl->count; i++) {
        int x = pl->xs[i];
        int y = pl->ys[i];
        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
    }

    Shape shape = {0};
    shape.num_cells = pl->count;
    shape.dx = (int*)calloc(shape.num_cells, sizeof(int));
    shape.dy = (int*)calloc(shape.num_cells, sizeof(int));
    shape.width = (max_x - min_x + 1);
    shape.height = (max_y - min_y + 1);

    for (int i = 0; i < pl->count; i++) {
        shape.dx[i] = pl->xs[i] - min_x;
        shape.dy[i] = pl->ys[i] - min_y;
    }

    *out_x = min_x;
    *out_y = min_y;
    return shape;
}

static bool pw_id_in_list(char** list, int count, const char* id) {
    for (int i = 0; i < count; i++) {
        if (strcmp(list[i], id) == 0) {
            return true;
        }
    }
    return false;
}

static void pw_add_id(char*** list, int* count, int* cap, const char* id) {
    if (pw_id_in_list(*list, *count, id)) {
        return;
    }
    if (*count >= *cap) {
        int new_cap = (*cap == 0) ? 8 : (*cap * 2);
        char** tmp = (char**)realloc(*list, new_cap * sizeof(char*));
        if (!tmp) {
            fprintf(stderr, "Failed to allocate id list\n");
            exit(1);
        }
        *list = tmp;
        *cap = new_cap;
    }
    (*list)[*count] = strdup(id);
    if (!(*list)[*count]) {
        fprintf(stderr, "Failed to allocate id string\n");
        exit(1);
    }
    (*count)++;
}

static int pw_object_index(char** list, int count, const char* id) {
    for (int i = 0; i < count; i++) {
        if (strcmp(list[i], id) == 0) {
            return i;
        }
    }
    return -1;
}

static int pw_compare_strings(const void* a, const void* b) {
    const char* const* sa = (const char* const*)a;
    const char* const* sb = (const char* const*)b;
    return strcmp(*sa, *sb);
}

typedef struct {
    char* path;
    int level_id;
} PuzzleFile;

static void pw_add_puzzle_file(PuzzleFile** list, int* count, int* cap, const char* path, int level_id) {
    if (*count >= *cap) {
        int new_cap = (*cap == 0) ? 16 : (*cap * 2);
        PuzzleFile* tmp = (PuzzleFile*)realloc(*list, new_cap * sizeof(PuzzleFile));
        if (!tmp) {
            fprintf(stderr, "Failed to allocate puzzle file list\n");
            exit(1);
        }
        *list = tmp;
        *cap = new_cap;
    }
    (*list)[*count].path = strdup(path);
    (*list)[*count].level_id = level_id;
    if (!(*list)[*count].path) {
        fprintf(stderr, "Failed to allocate puzzle file path\n");
        exit(1);
    }
    (*count)++;
}

static int pw_compare_puzzle_files(const void* a, const void* b) {
    const PuzzleFile* pa = (const PuzzleFile*)a;
    const PuzzleFile* pb = (const PuzzleFile*)b;
    int level_cmp = pa->level_id - pb->level_id;
    if (level_cmp != 0) {
        return level_cmp;
    }
    return strcmp(pa->path, pb->path);
}

static bool pw_is_dir(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

static Puzzle pw_load_puzzle(const char* path, int level_id) {
    PixelList* lists = NULL;
    int list_count = 0;
    int list_cap = 0;

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open puzzle file: %s\n", path);
        exit(1);
    }

    char line[4096];
    int rows = 0;
    int cols = -1;

    while (fgets(line, sizeof(line), f)) {
        char* saveptr = NULL;
        char* token = strtok_r(line, " \t\r\n", &saveptr);
        if (!token) {
            continue;
        }
        rows++;
        int col_count = 0;
        while (token) {
            col_count++;
            char cell[128];
            strncpy(cell, token, sizeof(cell) - 1);
            cell[sizeof(cell) - 1] = '\0';
            pw_lowercase(cell);

            char* cell_save = NULL;
            char* part = strtok_r(cell, "+", &cell_save);
            while (part) {
                if (strcmp(part, ".") != 0) {
                    PixelList* pl = pw_get_pixel_list(&lists, &list_count, &list_cap, part);
                    pw_add_pixel(pl, col_count, rows);
                }
                part = strtok_r(NULL, "+", &cell_save);
            }
            token = strtok_r(NULL, " \t\r\n", &saveptr);
        }
        if (cols == -1) {
            cols = col_count;
        } else if (cols != col_count) {
            fprintf(stderr, "Puzzle rows do not match in %s\n", path);
            fclose(f);
            pw_free_pixel_lists(lists, list_count);
            exit(1);
        }
    }
    fclose(f);

    if (cols <= 0 || rows <= 0) {
        fprintf(stderr, "Puzzle file empty: %s\n", path);
        pw_free_pixel_lists(lists, list_count);
        exit(1);
    }

    int width = cols + 2;
    int height = rows + 2;
    if (width >= PW_POSITION_LIMIT || height >= PW_POSITION_LIMIT) {
        fprintf(stderr, "Puzzle %s exceeds position limit %d\n", path, PW_POSITION_LIMIT);
        pw_free_pixel_lists(lists, list_count);
        exit(1);
    }

    PixelList* agent_pl = pw_find_pixel_list(lists, list_count, "a");
    if (!agent_pl) {
        fprintf(stderr, "Puzzle missing agent: %s\n", path);
        pw_free_pixel_lists(lists, list_count);
        exit(1);
    }

    char** goal_ids = NULL;
    int goal_count = 0;
    int goal_cap = 0;

    for (int i = 0; i < list_count; i++) {
        const char* id = lists[i].id;
        if (id[0] == 'g') {
            pw_add_id(&goal_ids, &goal_count, &goal_cap, id);
        }
    }

    if (goal_count > 1) {
        qsort(goal_ids, goal_count, sizeof(char*), pw_compare_strings);
    }

    char** object_ids = NULL;
    int object_count = 0;
    int object_cap = 0;
    pw_add_id(&object_ids, &object_count, &object_cap, "a");

    for (int i = 0; i < goal_count; i++) {
        const char* goal_id = goal_ids[i];
        char move_id[PW_MAX_ID_LEN];
        snprintf(move_id, sizeof(move_id), "m%s", goal_id + 1);
        if (!pw_find_pixel_list(lists, list_count, move_id)) {
            fprintf(stderr, "Goal %s missing movable %s in %s\n", goal_id, move_id, path);
            pw_free_pixel_lists(lists, list_count);
            exit(1);
        }
        pw_add_id(&object_ids, &object_count, &object_cap, move_id);
    }

    for (int i = 0; i < list_count; i++) {
        const char* id = lists[i].id;
        if (id[0] == 'm') {
            pw_add_id(&object_ids, &object_count, &object_cap, id);
        }
    }

    Puzzle puzzle = {0};
    puzzle.width = width;
    puzzle.height = height;
    puzzle.num_objects = object_count;
    puzzle.num_goals = goal_count;
    puzzle.level_id = level_id;
    puzzle.name = strdup(path);

    puzzle.objects = (Shape*)calloc(object_count, sizeof(Shape));
    puzzle.initial_positions = (int*)calloc(object_count, sizeof(int));
    puzzle.is_goal_object = (unsigned char*)calloc(object_count, sizeof(unsigned char));
    puzzle.goal_positions = (int*)calloc(goal_count, sizeof(int));
    puzzle.goal_object_indices = (int*)calloc(goal_count, sizeof(int));

    puzzle.wall_grid = (unsigned char*)calloc(width * height, sizeof(unsigned char));
    puzzle.agent_wall_grid = (unsigned char*)calloc(width * height, sizeof(unsigned char));
    puzzle.goal_grid = (unsigned char*)calloc(width * height, sizeof(unsigned char));

    if (!puzzle.objects || !puzzle.initial_positions || !puzzle.is_goal_object ||
        !puzzle.goal_positions || !puzzle.goal_object_indices || !puzzle.wall_grid ||
        !puzzle.agent_wall_grid || !puzzle.goal_grid) {
        fprintf(stderr, "Failed to allocate puzzle data for %s\n", path);
        exit(1);
    }

    for (int x = 0; x < width; x++) {
        puzzle.wall_grid[x] = 1;
        puzzle.wall_grid[(height - 1) * width + x] = 1;
    }
    for (int y = 0; y < height; y++) {
        puzzle.wall_grid[y * width] = 1;
        puzzle.wall_grid[y * width + (width - 1)] = 1;
    }

    PixelList* wall_pl = pw_find_pixel_list(lists, list_count, "w");
    if (wall_pl) {
        for (int i = 0; i < wall_pl->count; i++) {
            int x = wall_pl->xs[i];
            int y = wall_pl->ys[i];
            puzzle.wall_grid[y * width + x] = 1;
        }
    }

    PixelList* agent_wall_pl = pw_find_pixel_list(lists, list_count, "aw");
    if (agent_wall_pl) {
        for (int i = 0; i < agent_wall_pl->count; i++) {
            int x = agent_wall_pl->xs[i];
            int y = agent_wall_pl->ys[i];
            puzzle.agent_wall_grid[y * width + x] = 1;
        }
    }

    for (int i = 0; i < object_count; i++) {
        PixelList* pl = pw_find_pixel_list(lists, list_count, object_ids[i]);
        if (!pl) {
            fprintf(stderr, "Missing object %s in %s\n", object_ids[i], path);
            exit(1);
        }
        int anchor_x = 0;
        int anchor_y = 0;
        Shape shape = pw_build_shape(pl, &anchor_x, &anchor_y);
        puzzle.objects[i] = shape;
        puzzle.initial_positions[i] = pw_pos(anchor_x, anchor_y);
    }

    for (int g = 0; g < goal_count; g++) {
        const char* goal_id = goal_ids[g];
        char move_id[PW_MAX_ID_LEN];
        snprintf(move_id, sizeof(move_id), "m%s", goal_id + 1);
        int obj_idx = pw_object_index(object_ids, object_count, move_id);
        if (obj_idx < 0) {
            fprintf(stderr, "Failed to map goal %s to object %s in %s\n", goal_id, move_id, path);
            exit(1);
        }
        PixelList* goal_pl = pw_find_pixel_list(lists, list_count, goal_id);
        if (!goal_pl) {
            fprintf(stderr, "Missing goal %s in %s\n", goal_id, path);
            exit(1);
        }

        int min_x = 1 << 30;
        int min_y = 1 << 30;
        for (int i = 0; i < goal_pl->count; i++) {
            int x = goal_pl->xs[i];
            int y = goal_pl->ys[i];
            if (x < min_x) min_x = x;
            if (y < min_y) min_y = y;
            puzzle.goal_grid[y * width + x] = 1;
        }

        puzzle.goal_positions[g] = pw_pos(min_x, min_y);
        puzzle.goal_object_indices[g] = obj_idx;
        puzzle.is_goal_object[obj_idx] = 1;
    }

    for (int i = 0; i < goal_count; i++) {
        free(goal_ids[i]);
    }
    free(goal_ids);

    for (int i = 0; i < object_count; i++) {
        free(object_ids[i]);
    }
    free(object_ids);

    pw_free_pixel_lists(lists, list_count);

    return puzzle;
}

static void pw_free_puzzle(Puzzle* puzzle) {
    if (!puzzle) {
        return;
    }
    for (int i = 0; i < puzzle->num_objects; i++) {
        free(puzzle->objects[i].dx);
        free(puzzle->objects[i].dy);
    }
    free(puzzle->objects);
    free(puzzle->initial_positions);
    free(puzzle->is_goal_object);
    free(puzzle->goal_positions);
    free(puzzle->goal_object_indices);
    free(puzzle->wall_grid);
    free(puzzle->agent_wall_grid);
    free(puzzle->goal_grid);
    free(puzzle->name);
}

static PuzzleSet* pw_load_puzzles(const char* root_dir, const int* levels, int num_levels, int max_puzzles) {
    PuzzleFile* files = NULL;
    int file_count = 0;
    int file_cap = 0;

    if (!pw_is_dir(root_dir)) {
        fprintf(stderr, "Puzzle directory not found: %s\n", root_dir);
        exit(1);
    }

    DIR* dir = opendir(root_dir);
    if (!dir) {
        fprintf(stderr, "Failed to open puzzle directory: %s\n", root_dir);
        exit(1);
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        char level_path[4096];
        snprintf(level_path, sizeof(level_path), "%s/%s", root_dir, entry->d_name);
        if (pw_is_dir(level_path) && pw_starts_with(entry->d_name, "level")) {
            int level_id = pw_parse_level_id(entry->d_name);
            if (!pw_level_allowed(levels, num_levels, level_id)) {
                continue;
            }
            DIR* level_dir = opendir(level_path);
            if (!level_dir) {
                continue;
            }
            struct dirent* level_entry;
            while ((level_entry = readdir(level_dir)) != NULL) {
                if (pw_has_suffix(level_entry->d_name, ".pwp")) {
                    char file_path[4096];
                    snprintf(file_path, sizeof(file_path), "%s/%s", level_path, level_entry->d_name);
                    pw_add_puzzle_file(&files, &file_count, &file_cap, file_path, level_id);
                }
            }
            closedir(level_dir);
        } else if (pw_has_suffix(entry->d_name, ".pwp")) {
            int level_id = pw_parse_level_id(root_dir);
            if (!pw_level_allowed(levels, num_levels, level_id)) {
                continue;
            }
            char file_path[4096];
            snprintf(file_path, sizeof(file_path), "%s/%s", root_dir, entry->d_name);
            pw_add_puzzle_file(&files, &file_count, &file_cap, file_path, level_id);
        }
    }
    closedir(dir);

    if (file_count == 0) {
        fprintf(stderr, "No puzzles found in %s\n", root_dir);
        exit(1);
    }

    qsort(files, file_count, sizeof(PuzzleFile), pw_compare_puzzle_files);

    int total_files = file_count;
    if (max_puzzles > 0 && file_count > max_puzzles) {
        file_count = max_puzzles;
    }

    PuzzleSet* set = (PuzzleSet*)calloc(1, sizeof(PuzzleSet));
    set->num_puzzles = file_count;
    set->puzzles = (Puzzle*)calloc(file_count, sizeof(Puzzle));

    if (!set->puzzles) {
        fprintf(stderr, "Failed to allocate puzzles\n");
        exit(1);
    }

    for (int i = 0; i < file_count; i++) {
        Puzzle puzzle = pw_load_puzzle(files[i].path, files[i].level_id);
        set->puzzles[i] = puzzle;
        if (puzzle.width > set->max_width) set->max_width = puzzle.width;
        if (puzzle.height > set->max_height) set->max_height = puzzle.height;
        if (puzzle.num_objects > set->max_objects) set->max_objects = puzzle.num_objects;
    }

    for (int i = 0; i < total_files; i++) {
        free(files[i].path);
    }
    free(files);

    return set;
}

static void pw_free_puzzles(PuzzleSet* set) PW_UNUSED;
static void pw_free_puzzles(PuzzleSet* set) {
    if (!set) {
        return;
    }
    for (int i = 0; i < set->num_puzzles; i++) {
        pw_free_puzzle(&set->puzzles[i]);
    }
    free(set->puzzles);
    free(set);
}

static void pw_clear_obj_grid(PushWorld* env) {
    int total = env->width * env->height;
    for (int i = 0; i < total; i++) {
        env->obj_grid[i] = -1;
    }
}

static void pw_place_object(PushWorld* env, int obj_idx) {
    Shape* shape = &env->puzzle->objects[obj_idx];
    int pos = env->positions[obj_idx];
    int base_x = pw_pos_x(pos);
    int base_y = pw_pos_y(pos);
    for (int i = 0; i < shape->num_cells; i++) {
        int x = base_x + shape->dx[i];
        int y = base_y + shape->dy[i];
        env->obj_grid[y * env->width + x] = obj_idx;
    }
}

static void pw_clear_object(PushWorld* env, int obj_idx) {
    Shape* shape = &env->puzzle->objects[obj_idx];
    int pos = env->positions[obj_idx];
    int base_x = pw_pos_x(pos);
    int base_y = pw_pos_y(pos);
    for (int i = 0; i < shape->num_cells; i++) {
        int x = base_x + shape->dx[i];
        int y = base_y + shape->dy[i];
        int idx = y * env->width + x;
        if (env->obj_grid[idx] == obj_idx) {
            env->obj_grid[idx] = -1;
        }
    }
}

static int pw_count_achieved_goals(PushWorld* env) {
    int count = 0;
    for (int g = 0; g < env->num_goals; g++) {
        int obj_idx = env->puzzle->goal_object_indices[g];
        if (env->positions[obj_idx] == env->puzzle->goal_positions[g]) {
            count++;
        }
    }
    return count;
}

static uint64_t pw_hash_state(PushWorld* env) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < env->num_objects; i++) {
        uint64_t v = (uint64_t)env->positions[i];
        hash ^= v + ((uint64_t)i << 32);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int pw_count_state(PushWorld* env, uint64_t hash) {
    if (env->count_hash_capacity <= 0) {
        return 1;
    }
    int mask = env->count_hash_mask;
    int start = (int)(hash & (uint64_t)mask);
    for (int probe = 0; probe < env->count_hash_capacity; probe++) {
        int idx = (start + probe) & mask;
        if (!env->count_hash_used[idx]) {
            if (env->count_hash_size >= env->count_hash_capacity) {
                return INT_MAX;
            }
            env->count_hash_used[idx] = 1;
            env->count_hashes[idx] = hash;
            env->count_hash_counts[idx] = 1;
            env->count_hash_size += 1;
            return 1;
        }
        if (env->count_hashes[idx] == hash) {
            int count = env->count_hash_counts[idx] + 1;
            env->count_hash_counts[idx] = count;
            return count;
        }
    }
    return INT_MAX;
}

static void pw_update_coverage(PushWorld* env) {
    if (!env->coverage_enabled || !env->coverage_goal || !env->coverage_agent) {
        return;
    }

    int stride = env->coverage_stride;

    // Agent coverage (object 0, potentially multi-cell shape).
    int agent_pos = env->positions[0];
    int agent_x = pw_pos_x(agent_pos);
    int agent_y = pw_pos_y(agent_pos);
    Shape* agent_shape = &env->puzzle->objects[0];
    for (int i = 0; i < agent_shape->num_cells; i++) {
        int x = agent_x + agent_shape->dx[i];
        int y = agent_y + agent_shape->dy[i];
        if (x < 0 || y < 0 || x >= env->width || y >= env->height) {
            continue;
        }
        env->coverage_agent[y * stride + x] += 1;
    }

    // Goal object coverage.
    for (int g = 0; g < env->num_goals; g++) {
        int obj_idx = env->puzzle->goal_object_indices[g];
        int pos = env->positions[obj_idx];
        int base_x = pw_pos_x(pos);
        int base_y = pw_pos_y(pos);
        Shape* shape = &env->puzzle->objects[obj_idx];
        for (int i = 0; i < shape->num_cells; i++) {
            int x = base_x + shape->dx[i];
            int y = base_y + shape->dy[i];
            if (x < 0 || y < 0 || x >= env->width || y >= env->height) {
                continue;
            }
            env->coverage_goal[y * stride + x] += 1;
        }
    }
}

static bool pw_hits_wall(PushWorld* env, int obj_idx, int dx, int dy, bool is_agent) {
    Shape* shape = &env->puzzle->objects[obj_idx];
    int pos = env->positions[obj_idx];
    int base_x = pw_pos_x(pos);
    int base_y = pw_pos_y(pos);
    for (int i = 0; i < shape->num_cells; i++) {
        int x = base_x + shape->dx[i] + dx;
        int y = base_y + shape->dy[i] + dy;
        if (x < 0 || y < 0 || x >= env->width || y >= env->height) {
            return true;
        }
        int idx = y * env->width + x;
        if (env->puzzle->wall_grid[idx]) {
            return true;
        }
        if (is_agent && env->puzzle->agent_wall_grid[idx]) {
            return true;
        }
    }
    return false;
}

static bool pw_try_move(PushWorld* env, int action) {
    if (action < 0 || action >= PW_NUM_ACTIONS) {
        return false;
    }
    static const int dxs[PW_NUM_ACTIONS] = {-1, 1, 0, 0};
    static const int dys[PW_NUM_ACTIONS] = {0, 0, -1, 1};
    static const int disps[PW_NUM_ACTIONS] = {-PW_POSITION_LIMIT, PW_POSITION_LIMIT, -1, 1};

    int dx = dxs[action];
    int dy = dys[action];
    int disp = disps[action];

    memset(env->moved, 0, env->num_objects * sizeof(unsigned char));
    int front = 0;
    int back = 0;

    env->queue[back++] = 0;
    env->moved[0] = 1;

    while (front < back) {
        int obj_idx = env->queue[front++];
        bool is_agent = (obj_idx == 0);

        if (pw_hits_wall(env, obj_idx, dx, dy, is_agent)) {
            return false;
        }

        Shape* shape = &env->puzzle->objects[obj_idx];
        int pos = env->positions[obj_idx];
        int base_x = pw_pos_x(pos);
        int base_y = pw_pos_y(pos);

        for (int i = 0; i < shape->num_cells; i++) {
            int x = base_x + shape->dx[i] + dx;
            int y = base_y + shape->dy[i] + dy;
            int other = env->obj_grid[y * env->width + x];
            if (other >= 0 && other != obj_idx && !env->moved[other]) {
                env->moved[other] = 1;
                env->queue[back++] = other;
            }
        }
    }

    for (int i = 0; i < back; i++) {
        pw_clear_object(env, env->queue[i]);
    }
    for (int i = 0; i < back; i++) {
        int obj_idx = env->queue[i];
        env->positions[obj_idx] += disp;
    }
    for (int i = 0; i < back; i++) {
        pw_place_object(env, env->queue[i]);
    }

    return true;
}

static int pw_tile_at(PushWorld* env, int x, int y) {
    int idx = y * env->width + x;
    int obj = env->obj_grid[idx];
    if (obj >= 0) {
        if (obj == 0) {
            return PW_TILE_AGENT;
        }
        if (env->puzzle->is_goal_object[obj]) {
            return PW_TILE_GOAL_OBJECT;
        }
        return PW_TILE_MOVABLE;
    }
    if (env->puzzle->goal_grid[idx]) {
        return PW_TILE_GOAL;
    }
    if (env->puzzle->wall_grid[idx]) {
        return PW_TILE_WALL;
    }
    if (env->puzzle->agent_wall_grid[idx]) {
        return PW_TILE_AGENT_WALL;
    }
    return PW_TILE_EMPTY;
}

static void pw_compute_observations(PushWorld* env) {
    int obs_size = env->obs_size;
    int total = obs_size * obs_size;
    for (int i = 0; i < total; i++) {
        env->observations[i] = (float)PW_TILE_WALL;
    }

    int agent_pos = env->positions[0];
    int agent_x = pw_pos_x(agent_pos);
    int agent_y = pw_pos_y(agent_pos);
    int start_x = agent_x - env->vision_radius;
    int start_y = agent_y - env->vision_radius;

    for (int oy = 0; oy < obs_size; oy++) {
        for (int ox = 0; ox < obs_size; ox++) {
            int gx = start_x + ox;
            int gy = start_y + oy;
            if (gx < 0 || gy < 0 || gx >= env->width || gy >= env->height) {
                continue;
            }
            int tile = pw_tile_at(env, gx, gy);
            env->observations[oy * obs_size + ox] = (float)tile;
        }
    }
}

static void pw_add_log(PushWorld* env) {
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.no_op_rate += env->tick > 0 ? (float)env->no_op_moves / (float)env->tick : 0.0f;
    env->log.solved += env->last_solved ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

static void init_pushworld(PushWorld* env, PuzzleSet* puzzles, int max_episode_length, int vision) {
    env->puzzles = puzzles;
    env->max_episode_length = max_episode_length > 0 ? max_episode_length : 500;
    int vision_size = vision > 0 ? vision : 15;
    if (vision_size % 2 == 0) {
        vision_size += 1; // Force odd window so the agent stays centered.
    }
    env->obs_size = vision_size;
    env->vision_radius = vision_size / 2;
    env->max_width = puzzles->max_width;
    env->max_height = puzzles->max_height;
    env->max_objects = puzzles->max_objects;
    int target = env->max_episode_length * 4;
    if (target < 64) {
        target = 64;
    }
    int cap = 1;
    while (cap < target) {
        cap <<= 1;
    }
    env->count_hash_capacity = cap;
    env->count_hash_mask = cap - 1;
    env->count_hash_size = 0;
    env->forced_puzzle_idx = -1;
    env->last_solved = 0;
    env->coverage_enabled = 0;
    env->coverage_stride = env->max_width;
    env->obj_grid = (int*)calloc(env->max_width * env->max_height, sizeof(int));
    env->count_hashes = (uint64_t*)calloc((size_t)cap, sizeof(uint64_t));
    env->count_hash_used = (unsigned char*)calloc((size_t)cap, sizeof(unsigned char));
    env->count_hash_counts = (int*)calloc((size_t)cap, sizeof(int));
    env->positions = (int*)calloc(env->max_objects, sizeof(int));
    env->moved = (unsigned char*)calloc(env->max_objects, sizeof(unsigned char));
    env->queue = (int*)calloc(env->max_objects, sizeof(int));
    size_t cover_size = (size_t)env->max_width * (size_t)env->max_height;
    env->coverage_goal = (uint32_t*)calloc(cover_size, sizeof(uint32_t));
    env->coverage_agent = (uint32_t*)calloc(cover_size, sizeof(uint32_t));
    if (!env->obj_grid || !env->count_hashes || !env->count_hash_used || !env->count_hash_counts ||
            !env->positions || !env->moved ||
            !env->queue || !env->coverage_goal || !env->coverage_agent) {
        fprintf(stderr, "Failed to allocate pushworld env buffers\n");
        exit(1);
    }
    env->renderer = NULL;
}

static void c_reset(PushWorld* env) {
    int idx = env->forced_puzzle_idx;
    if (idx < 0 || idx >= env->puzzles->num_puzzles) {
        idx = rand() % env->puzzles->num_puzzles;
    }
    env->puzzle_idx = idx;
    env->puzzle = &env->puzzles->puzzles[idx];
    env->width = env->puzzle->width;
    env->height = env->puzzle->height;
    env->num_objects = env->puzzle->num_objects;
    env->num_goals = env->puzzle->num_goals;
    env->tick = 0;
    env->no_op_moves = 0;
    env->episode_return = 0.0f;
    if (!env->count_based_global) {
        memset(env->count_hash_used, 0, env->count_hash_capacity * sizeof(unsigned char));
        memset(env->count_hash_counts, 0, env->count_hash_capacity * sizeof(int));
        env->count_hash_size = 0;
    }
    memcpy(env->positions, env->puzzle->initial_positions, env->num_objects * sizeof(int));
    pw_clear_obj_grid(env);
    for (int i = 0; i < env->num_objects; i++) {
        pw_place_object(env, i);
    }
    pw_compute_observations(env);
    pw_update_coverage(env);
}

static void c_step(PushWorld* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    env->last_solved = 0;
    env->tick++;

    int prev_goals = pw_count_achieved_goals(env);
    bool moved = pw_try_move(env, env->actions[0]);
    if (!moved) {
        env->no_op_moves++;
    }
    pw_update_coverage(env);
    int cur_goals = pw_count_achieved_goals(env);

    bool solved = (cur_goals == env->num_goals);
    float reward = 0.0f;
    if (solved) {
        reward = 10.0f;
        env->terminals[0] = 1;
        env->last_solved = 1;
    } else {
        reward = (float)(cur_goals - prev_goals) - 0.01f;
    }

    if (env->count_based_reward_coef > 0.0f) {
        uint64_t state_hash = pw_hash_state(env);
        int visit_count = pw_count_state(env, state_hash);
        if (visit_count != INT_MAX) {
            reward += env->count_based_reward_coef * 0.1f / (float)visit_count;
        }
    }

    env->rewards[0] = reward;
    env->episode_return += reward;

    bool done = solved;
    if (env->tick >= env->max_episode_length) {
        done = true;
        env->terminals[0] = 1;
    }

    if (done) {
        pw_add_log(env);
        c_reset(env);
        return;
    }

    pw_compute_observations(env);
}

static Renderer* pw_init_renderer(int cell_size, int width, int height) {
    Renderer* renderer = (Renderer*)calloc(1, sizeof(Renderer));
    renderer->cell_size = cell_size;
    renderer->width = width;
    renderer->height = height;
    InitWindow(width * cell_size, height * cell_size, "PufferLib PushWorld");
    SetTargetFPS(60);
    return renderer;
}

static void pw_close_renderer(Renderer* renderer) {
    if (!renderer) {
        return;
    }
    CloseWindow();
    free(renderer);
}

static void c_render(PushWorld* env) {
    if (env->renderer == NULL || env->renderer->width != env->width || env->renderer->height != env->height) {
        if (env->renderer) {
            pw_close_renderer(env->renderer);
        }
        int max_dim = env->width > env->height ? env->width : env->height;
        int cell_size = max_dim <= 64 ? 16 : 512 / max_dim;
        if (cell_size < 4) cell_size = 4;
        env->renderer = pw_init_renderer(cell_size, env->width, env->height);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    Renderer* renderer = env->renderer;
    int ts = renderer->cell_size;

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            Color base = (Color){20, 20, 20, 255};
            if (env->puzzle->wall_grid[idx]) {
                base = (Color){40, 40, 40, 255};
            } else if (env->puzzle->agent_wall_grid[idx]) {
                base = (Color){180, 140, 40, 255};
            } else if (env->puzzle->goal_grid[idx]) {
                base = (Color){90, 20, 20, 255};
            }
            DrawRectangle(x * ts, y * ts, ts, ts, base);

            int obj = env->obj_grid[idx];
            if (obj >= 0) {
                Color color = (Color){70, 110, 255, 255};
                if (obj == 0) {
                    color = (Color){0, 220, 0, 255};
                } else if (env->puzzle->is_goal_object[obj]) {
                    color = (Color){220, 0, 0, 255};
                }
                DrawRectangle(x * ts, y * ts, ts, ts, color);
            }
        }
    }

    EndDrawing();
}

static void c_close(PushWorld* env) {
    if (env->renderer) {
        pw_close_renderer(env->renderer);
        env->renderer = NULL;
    }
    free(env->obj_grid);
    free(env->count_hashes);
    free(env->count_hash_used);
    free(env->count_hash_counts);
    free(env->positions);
    free(env->moved);
    free(env->queue);
    free(env->coverage_goal);
    free(env->coverage_agent);
}

static PushWorld* allocate_pushworld(PuzzleSet* puzzles, int max_episode_length, int vision) PW_UNUSED;
static PushWorld* allocate_pushworld(PuzzleSet* puzzles, int max_episode_length, int vision) {
    PushWorld* env = (PushWorld*)calloc(1, sizeof(PushWorld));
    init_pushworld(env, puzzles, max_episode_length, vision);
    env->observations = (float*)calloc(env->obs_size * env->obs_size, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (!env->observations || !env->actions || !env->rewards || !env->terminals) {
        fprintf(stderr, "Failed to allocate pushworld buffers\n");
        exit(1);
    }
    return env;
}

static void free_allocated_pushworld(PushWorld* env) PW_UNUSED;
static void free_allocated_pushworld(PushWorld* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
    free(env);
}

#endif
