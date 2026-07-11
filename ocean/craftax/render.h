#pragma once

#include "raylib.h"
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

struct Client {
    pid_t xvfb_pid;
    int xvfb_display_num;
    bool headless_display;
    int view_mode;
    bool trail_initialized;
    int32_t trail_level;
    int32_t last_row;
    int32_t last_col;
    int32_t trail_rows[CRAFTAX_TRAIL_LEN];
    int32_t trail_cols[CRAFTAX_TRAIL_LEN];
    int trail_count;
    int trail_head;
};

static Texture2D craftax_textures[CRAFTAX_TEX_NUM];
static bool craftax_textures_loaded = false;

static inline int craftax_normalize_view_mode(int view_mode) {
    return view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? CRAFTAX_VIEW_MODE_FLOOR
        : CRAFTAX_VIEW_MODE_AGENT;
}

static inline Client* craftax_get_client(Craftax* env) {
    if (env->client == NULL) {
        env->client = (Client*)calloc(1, sizeof(Client));
        if (env->client != NULL) {
            env->client->view_mode = craftax_normalize_view_mode(env->render_view_mode);
        }
    }
    return env->client;
}

static inline int craftax_ensure_render_display(Craftax* env) {
#if defined(PLATFORM_MEMORY)
    (void)env;
    return 1;
#else
    Client* client = craftax_get_client(env);
    if (client == NULL) {
        return 0;
    }

    if (getenv("DISPLAY") != NULL) {
        return 1;
    }

    client->headless_display = true;
    if (client->xvfb_display_num == 0) {
        client->xvfb_display_num = 99;
    }

    char display_name[16];
    char lock_path[64];
    char socket_path[64];
    snprintf(display_name, sizeof(display_name), ":%d", client->xvfb_display_num);
    snprintf(lock_path, sizeof(lock_path), "/tmp/.X%d-lock", client->xvfb_display_num);
    snprintf(socket_path, sizeof(socket_path), "/tmp/.X11-unix/X%d", client->xvfb_display_num);

    FILE* f = fopen(lock_path, "r");
    if (f != NULL) {
        pid_t pid = -1;
        fscanf(f, "%d", &pid);
        fclose(f);
        if (pid > 0 && kill(pid, 0) == 0) {
            setenv("DISPLAY", display_name, 1);
            return 1;
        }
        unlink(lock_path);
    }

    if (access(socket_path, F_OK) == 0) {
        setenv("DISPLAY", display_name, 1);
        return 1;
    }

    pid_t child = fork();
    if (child == 0) {
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
        execlp("Xvfb", "Xvfb", display_name, "-screen", "0",
            "1280x720x24", "+extension", "GLX", "-ac", "-noreset", NULL);
        _exit(1);
    }
    if (child < 0) {
        return 0;
    }

    client->xvfb_pid = child;
    setenv("DISPLAY", display_name, 1);
    for (int i = 0; i < 20 && access(lock_path, F_OK) != 0; i++) {
        usleep(100000);
    }
    usleep(200000);
    if (access(lock_path, F_OK) != 0 && access(socket_path, F_OK) != 0) {
        return 0;
    }
    return 1;
#endif
}

static inline void craftax_close_client(Craftax* env) {
    if (env->client == NULL) {
        return;
    }
    if (env->client->xvfb_pid > 0) {
        kill(env->client->xvfb_pid, SIGTERM);
        waitpid(env->client->xvfb_pid, NULL, 0);
    }
    free(env->client);
    env->client = NULL;
}

static inline void craftax_reset_player_trail(Craftax* env) {
    Client* client = craftax_get_client(env);
    if (client == NULL || env->state == NULL) {
        return;
    }
    client->trail_initialized = true;
    client->trail_level = env->state->player_level;
    client->last_row = env->state->player_position[0];
    client->last_col = env->state->player_position[1];
    client->trail_count = 0;
    client->trail_head = 0;
}

static inline void craftax_record_player_position(Craftax* env) {
    Client* client = craftax_get_client(env);
    if (client == NULL || env->state == NULL) {
        return;
    }

    int32_t level = env->state->player_level;
    int32_t row = env->state->player_position[0];
    int32_t col = env->state->player_position[1];

    if (!client->trail_initialized) {
        craftax_reset_player_trail(env);
        return;
    }

    if (client->trail_level != level) {
        craftax_reset_player_trail(env);
        return;
    }

    if (client->last_row == row && client->last_col == col) {
        return;
    }

    client->trail_rows[client->trail_head] = client->last_row;
    client->trail_cols[client->trail_head] = client->last_col;
    client->trail_head = (client->trail_head + 1) % CRAFTAX_TRAIL_LEN;
    if (client->trail_count < CRAFTAX_TRAIL_LEN) {
        client->trail_count += 1;
    }

    client->last_row = row;
    client->last_col = col;
}

static inline void craftax_set_view_mode(Craftax* env, int view_mode) {
    env->render_view_mode = craftax_normalize_view_mode(view_mode);
    Client* client = craftax_get_client(env);
    if (client == NULL) {
        return;
    }
    client->view_mode = env->render_view_mode;
}

static inline int craftax_view_tile_px(const Client* client) {
    return client->view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? CRAFTAX_FLOOR_TILE_PX
        : CRAFTAX_TEX_DRAW_PX;
}

static inline int craftax_view_rows(const Client* client) {
    return client->view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? CRAFTAX_MAP_SIZE
        : CRAFTAX_RENDER_ROWS;
}

static inline int craftax_view_cols(const Client* client) {
    return client->view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? CRAFTAX_MAP_SIZE
        : CRAFTAX_RENDER_COLS;
}

static inline int craftax_view_width(const Client* client) {
    return craftax_view_cols(client) * craftax_view_tile_px(client);
}

static inline int craftax_view_height(const Client* client) {
    return craftax_view_rows(client) * craftax_view_tile_px(client);
}

static inline void craftax_load_textures(void) {
    if (craftax_textures_loaded) return;
    const char* candidates[] = {
        "resources/craftax/textures.bin",
        "../resources/craftax/textures.bin",
        "../../resources/craftax/textures.bin",
    };
    FILE* f = NULL;
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        f = fopen(candidates[i], "rb");
        if (f) break;
    }
    if (!f) {
        fprintf(stderr, "craftax: textures.bin not found in resources/craftax\n");
        exit(1);
    }
    const size_t tile_bytes = CRAFTAX_TEX_TILE_PX * CRAFTAX_TEX_TILE_PX * 4;
    uint8_t* buf = (uint8_t*)malloc(tile_bytes);
    for (int i = 0; i < CRAFTAX_TEX_NUM; i++) {
        if (fread(buf, 1, tile_bytes, f) != tile_bytes) {
            fprintf(stderr, "craftax: short read on textures.bin at tile %d\n", i);
            exit(1);
        }
        Image img = {
            .data = buf,
            .width = CRAFTAX_TEX_TILE_PX,
            .height = CRAFTAX_TEX_TILE_PX,
            .mipmaps = 1,
            .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
        };
        craftax_textures[i] = LoadTextureFromImage(img);
        SetTextureFilter(craftax_textures[i], TEXTURE_FILTER_POINT);
    }
    free(buf);
    fclose(f);
    craftax_textures_loaded = true;
}

static inline int craftax_player_tex_id(int32_t direction, bool sleeping) {
    if (sleeping) return CRAFTAX_TEX_PLAYER_SLEEP;
    switch (direction) {
        case 1: return CRAFTAX_TEX_PLAYER_LEFT;
        case 2: return CRAFTAX_TEX_PLAYER_RIGHT;
        case 3: return CRAFTAX_TEX_PLAYER_UP;
        case 4: return CRAFTAX_TEX_PLAYER_DOWN;
        default: return CRAFTAX_TEX_PLAYER_DOWN;
    }
}

static inline void craftax_draw_tile_tint_scaled(
    int tex_id,
    int dst_x,
    int dst_y,
    int dst_size,
    Color tint
) {
    if (tex_id < 0 || tex_id >= CRAFTAX_TEX_NUM) return;
    Rectangle src = {0, 0, CRAFTAX_TEX_TILE_PX, CRAFTAX_TEX_TILE_PX};
    Rectangle dst = {(float)dst_x, (float)dst_y, (float)dst_size, (float)dst_size};
    DrawTexturePro(craftax_textures[tex_id], src, dst, (Vector2){0, 0}, 0.0f, tint);
}

static inline void craftax_draw_tile_scaled(
    int tex_id,
    int dst_x,
    int dst_y,
    int dst_size,
    float tint_alpha
) {
    Color tint = {255, 255, 255, (unsigned char)(tint_alpha * 255.0f)};
    craftax_draw_tile_tint_scaled(tex_id, dst_x, dst_y, dst_size, tint);
}

static inline void craftax_draw_texture_box(
    int tex_id,
    int dst_x,
    int dst_y,
    int dst_size,
    Color tint
) {
    craftax_draw_tile_tint_scaled(tex_id, dst_x, dst_y, dst_size, tint);
}

static inline int craftax_mob_tex_id(bool passive, bool ranged) {
    if (passive) return 49;
    if (ranged) return 48;
    return 47;
}

static inline int craftax_projectile_tex_id(const int32_t direction[2]) {
    if (direction[0] < 0) return 51;
    if (direction[0] > 0) return 50;
    if (direction[1] < 0) return 52;
    if (direction[1] > 0) return 53;
    return 50;
}

static inline Color craftax_projectile_tint(int32_t type_id, bool hostile) {
    switch (type_id) {
        case CRAFTAX_PROJECTILE_FIREBALL:
        case CRAFTAX_PROJECTILE_FIREBALL2:
            return hostile ? (Color){255, 170, 80, 235} : (Color){255, 220, 120, 235};
        case CRAFTAX_PROJECTILE_ICEBALL:
        case CRAFTAX_PROJECTILE_ICEBALL2:
            return hostile ? (Color){120, 220, 255, 235} : (Color){180, 240, 255, 235};
        case CRAFTAX_PROJECTILE_SLIMEBALL:
            return (Color){120, 255, 140, 235};
        case CRAFTAX_PROJECTILE_DAGGER:
            return (Color){220, 220, 220, 235};
        default:
            return hostile ? (Color){255, 255, 255, 235} : (Color){255, 245, 160, 235};
    }
}

static inline void craftax_draw_world_entity_scaled(
    int tex_id,
    int32_t entity_row,
    int32_t entity_col,
    int32_t top_row,
    int32_t left_col,
    int rows,
    int cols,
    int tile_px,
    Color tint
) {
    int vr = entity_row - top_row;
    int vc = entity_col - left_col;
    if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) {
        return;
    }
    craftax_draw_tile_tint_scaled(
        tex_id,
        vc * tile_px,
        vr * tile_px,
        tile_px,
        tint
    );
}

static inline void craftax_draw_mob_overlays_scaled(
    const CraftaxState* s,
    int lvl,
    int top_row,
    int left_col,
    int rows,
    int cols,
    int tile_px
) {
    const Color white = {255, 255, 255, 255};
    for (int i = 0; i < CRAFTAX_MAX_PASSIVE_MOBS; i++) {
        if (!s->passive_mobs.mask[lvl][i]) continue;
        craftax_draw_world_entity_scaled(
            craftax_mob_tex_id(true, false),
            s->passive_mobs.position[lvl][i][0],
            s->passive_mobs.position[lvl][i][1],
            top_row, left_col, rows, cols, tile_px, white
        );
    }
    for (int i = 0; i < CRAFTAX_MAX_MELEE_MOBS; i++) {
        if (!s->melee_mobs.mask[lvl][i]) continue;
        craftax_draw_world_entity_scaled(
            craftax_mob_tex_id(false, false),
            s->melee_mobs.position[lvl][i][0],
            s->melee_mobs.position[lvl][i][1],
            top_row, left_col, rows, cols, tile_px, white
        );
    }
    for (int i = 0; i < CRAFTAX_MAX_RANGED_MOBS; i++) {
        if (!s->ranged_mobs.mask[lvl][i]) continue;
        craftax_draw_world_entity_scaled(
            craftax_mob_tex_id(false, true),
            s->ranged_mobs.position[lvl][i][0],
            s->ranged_mobs.position[lvl][i][1],
            top_row, left_col, rows, cols, tile_px, white
        );
    }
}

static inline void craftax_draw_projectile_overlays_scaled(
    const CraftaxState* s,
    int lvl,
    int top_row,
    int left_col,
    int rows,
    int cols,
    int tile_px
) {
    for (int i = 0; i < CRAFTAX_MAX_MOB_PROJECTILES; i++) {
        if (!s->mob_projectiles.mask[lvl][i]) continue;
        craftax_draw_world_entity_scaled(
            craftax_projectile_tex_id(s->mob_projectile_directions[lvl][i]),
            s->mob_projectiles.position[lvl][i][0],
            s->mob_projectiles.position[lvl][i][1],
            top_row,
            left_col,
            rows,
            cols,
            tile_px,
            craftax_projectile_tint(s->mob_projectiles.type_id[lvl][i], true)
        );
    }
    for (int i = 0; i < CRAFTAX_MAX_PLAYER_PROJECTILES; i++) {
        if (!s->player_projectiles.mask[lvl][i]) continue;
        craftax_draw_world_entity_scaled(
            craftax_projectile_tex_id(s->player_projectile_directions[lvl][i]),
            s->player_projectiles.position[lvl][i][0],
            s->player_projectiles.position[lvl][i][1],
            top_row,
            left_col,
            rows,
            cols,
            tile_px,
            craftax_projectile_tint(s->player_projectiles.type_id[lvl][i], false)
        );
    }
}

static inline void craftax_draw_trail_overlay(
    const Client* client,
    int lvl,
    int top_row,
    int left_col,
    int rows,
    int cols,
    int tile_px
) {
    if (client->trail_level != lvl) {
        return;
    }
    for (int i = 0; i < client->trail_count; i++) {
        int idx = (client->trail_head - 1 - i + CRAFTAX_TRAIL_LEN) % CRAFTAX_TRAIL_LEN;
        int vr = client->trail_rows[idx] - top_row;
        int vc = client->trail_cols[idx] - left_col;
        if (vr < 0 || vr >= rows || vc < 0 || vc >= cols) {
            continue;
        }
        unsigned char alpha = (unsigned char)(180 - i * 12);
        DrawRectangle(
            vc * tile_px,
            vr * tile_px,
            tile_px,
            tile_px,
            (Color){220, 40, 40, alpha}
        );
    }
}

static inline void craftax_draw_ladder_overlays(
    const Client* client,
    const CraftaxState* s,
    int lvl,
    int top_row,
    int left_col,
    int rows,
    int cols,
    int tile_px
) {
    if (client->view_mode != CRAFTAX_VIEW_MODE_FLOOR) {
        return;
    }

    int inset = tile_px / 10;
    int line_width = tile_px >= 16 ? 3 : 2;
    if (inset < 1) {
        inset = 1;
    }
    int marker_size = tile_px - 2 * inset;

    for (int vr = 0; vr < rows; vr++) {
        for (int vc = 0; vc < cols; vc++) {
            int wr = top_row + vr;
            int wc = left_col + vc;
            if (wr < 0 || wr >= CRAFTAX_MAP_SIZE || wc < 0 || wc >= CRAFTAX_MAP_SIZE) {
                continue;
            }

            int item = s->item_map[lvl][wr][wc];
            bool is_ladder = true;
            switch (item) {
                case CRAFTAX_ITEM_LADDER_DOWN:
                case CRAFTAX_ITEM_LADDER_DOWN_BLOCKED:
                case CRAFTAX_ITEM_LADDER_UP:
                    break;
                default:
                    is_ladder = false;
                    break;
            }
            if (!is_ladder) {
                continue;
            }

            int x = vc * tile_px + inset;
            int y = vr * tile_px + inset;
            DrawRectangleLinesEx(
                (Rectangle){(float)x, (float)y, (float)marker_size, (float)marker_size},
                (float)line_width,
                (Color){255, 230, 80, 255}
            );
        }
    }
}

static inline int craftax_render_block(
    const CraftaxState* s,
    int lvl,
    int wr,
    int wc
) {
    int blk = s->map[lvl][wr][wc];
    if (blk == CRAFTAX_BLOCK_NECROMANCER
            && craftax_wg_is_boss_vulnerable((const CraftaxWorldState*)s)) {
        blk = CRAFTAX_BLOCK_NECROMANCER_VULNERABLE;
    }
    if (s->light_map[lvl][wr][wc] <= 12) {
        blk = CRAFTAX_BLOCK_DARKNESS;
    }
    if (blk < 0 || blk >= CRAFTAX_NUM_BLOCK_TYPES) {
        blk = 0;
    }
    return blk;
}

static inline void craftax_draw_world_view(Craftax* env) {
    Client* client = craftax_get_client(env);
    CraftaxState* s = env->state;
    int lvl = s->player_level;
    int tile_px = craftax_view_tile_px(client);
    int rows = craftax_view_rows(client);
    int cols = craftax_view_cols(client);
    int top_row = client->view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? 0
        : s->player_position[0] - rows / 2;
    int left_col = client->view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? 0
        : s->player_position[1] - cols / 2;

    for (int vr = 0; vr < rows; vr++) {
        for (int vc = 0; vc < cols; vc++) {
            int wr = top_row + vr;
            int wc = left_col + vc;
            int dst_x = vc * tile_px;
            int dst_y = vr * tile_px;
            int blk = CRAFTAX_BLOCK_OUT_OF_BOUNDS;
            if (wr >= 0 && wr < CRAFTAX_MAP_SIZE && wc >= 0 && wc < CRAFTAX_MAP_SIZE) {
                blk = craftax_render_block(s, lvl, wr, wc);
            }
            craftax_draw_tile_scaled(blk, dst_x, dst_y, tile_px, 1.0f);

            if (wr >= 0 && wr < CRAFTAX_MAP_SIZE && wc >= 0 && wc < CRAFTAX_MAP_SIZE) {
                int it = s->item_map[lvl][wr][wc];
                if (it > 0 && it < 5) {
                    if (it == CRAFTAX_ITEM_LADDER_DOWN
                            && s->monsters_killed[lvl] < CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL) {
                        it = CRAFTAX_ITEM_LADDER_DOWN_BLOCKED;
                    }
                    craftax_draw_tile_scaled(CRAFTAX_TEX_ITEM_BASE + it, dst_x, dst_y, tile_px, 1.0f);
                }
            }
        }
    }

    craftax_draw_trail_overlay(client, lvl, top_row, left_col, rows, cols, tile_px);
    craftax_draw_ladder_overlays(client, s, lvl, top_row, left_col, rows, cols, tile_px);
    craftax_draw_mob_overlays_scaled(s, lvl, top_row, left_col, rows, cols, tile_px);
    craftax_draw_projectile_overlays_scaled(s, lvl, top_row, left_col, rows, cols, tile_px);

    int pid = craftax_player_tex_id(s->player_direction, s->is_sleeping);
    int player_row = client->view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? s->player_position[0]
        : rows / 2;
    int player_col = client->view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? s->player_position[1]
        : cols / 2;
    craftax_draw_tile_scaled(
        pid,
        player_col * tile_px,
        player_row * tile_px,
        tile_px,
        1.0f
    );

    if (s->light_level < 1.0f) {
        unsigned char a = (unsigned char)((1.0f - s->light_level) * 140.0f);
        DrawRectangle(0, 0, cols * tile_px, rows * tile_px, (Color){0, 0, 40, a});
    }
}

static inline int craftax_prepare_render(Craftax* env, const char* title, int hud_h) {
    Client* client = craftax_get_client(env);
    if (client == NULL) {
        return 0;
    }
    if (!client->trail_initialized) {
        craftax_reset_player_trail(env);
    }

    if (!IsWindowReady()) {
        if (!craftax_ensure_render_display(env)) {
            fprintf(stderr, "WARNING: failed to initialize display for Craftax rendering\n");
            return 0;
        }
        if (client->headless_display) {
            SetConfigFlags(FLAG_WINDOW_HIDDEN);
            SetTargetFPS(6000);
        } else {
            SetTargetFPS(30);
        }
        InitWindow(craftax_view_width(client), craftax_view_height(client) + hud_h, title);
    }

    if (!craftax_textures_loaded) {
        craftax_load_textures();
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (IsKeyPressed(KEY_V) || IsKeyPressed(KEY_TAB)) {
        client->view_mode = client->view_mode == CRAFTAX_VIEW_MODE_AGENT
            ? CRAFTAX_VIEW_MODE_FLOOR
            : CRAFTAX_VIEW_MODE_AGENT;
    }

    int target_w = craftax_view_width(client);
    int target_h = craftax_view_height(client) + hud_h;
    if (GetScreenWidth() != target_w || GetScreenHeight() != target_h) {
        SetWindowSize(target_w, target_h);
    }
    return 1;
}

static inline void craftax_draw_hud_panel(int x, int y, int w, int h, const char* title) {
    DrawRectangleRounded((Rectangle){(float)x, (float)y, (float)w, (float)h}, 0.18f, 8,
        (Color){28, 31, 35, 235});
    DrawRectangleLinesEx((Rectangle){(float)x, (float)y, (float)w, (float)h}, 2.0f,
        (Color){72, 78, 88, 255});
    if (title != NULL && title[0] != '\0') {
        DrawText(title, x + 10, y + 8, 16, (Color){230, 232, 236, 255});
    }
}

static inline void craftax_draw_hud_bar(
    int x,
    int y,
    int w,
    const char* label,
    int value,
    int max_value,
    Color fill
) {
    if (max_value < 1) max_value = 1;
    if (value < 0) value = 0;
    if (value > max_value) value = max_value;

    int bar_y = y + 18;
    int bar_h = 12;
    int fill_w = (w - 2) * value / max_value;

    DrawText(label, x, y, 15, (Color){215, 217, 221, 255});
    DrawRectangle(x, bar_y, w, bar_h, (Color){55, 60, 68, 255});
    DrawRectangle(x + 1, bar_y + 1, fill_w, bar_h - 2, fill);
    DrawRectangleLines(x, bar_y, w, bar_h, (Color){105, 112, 122, 255});
    DrawText(TextFormat("%d/%d", value, max_value), x + w + 8, y + 8, 14,
        (Color){235, 238, 242, 255});
}

static inline void craftax_draw_hud_slot(
    int x,
    int y,
    int tex_id,
    Color tint,
    const char* label,
    int value
) {
    const int slot = 28;
    DrawRectangleRounded((Rectangle){(float)x, (float)y, (float)slot, (float)slot}, 0.2f, 6,
        (Color){47, 51, 58, 255});
    DrawRectangleLinesEx((Rectangle){(float)x, (float)y, (float)slot, (float)slot}, 1.0f,
        (Color){88, 94, 104, 255});
    craftax_draw_texture_box(tex_id, x + 4, y + 4, 20, tint);
    DrawText(label, x + slot + 8, y + 2, 14, (Color){210, 214, 220, 255});
    DrawText(TextFormat("%d", value), x + slot + 8, y + 15, 15, WHITE);
}

static inline void craftax_draw_hud_badge(
    int x,
    int y,
    int w,
    const char* label,
    Color bg,
    Color fg
) {
    DrawRectangleRounded((Rectangle){(float)x, (float)y, (float)w, 20.0f}, 0.3f, 6, bg);
    DrawText(label, x + 8, y + 3, 14, fg);
}

static inline void craftax_draw_view_mode_badge(Craftax* env, int x, int y) {
    Client* client = craftax_get_client(env);
    const char* label = client->view_mode == CRAFTAX_VIEW_MODE_FLOOR
        ? "View: Full Floor [V]"
        : "View: Agent [V]";
    craftax_draw_hud_badge(x, y, 132, label, (Color){54, 58, 62, 255}, WHITE);
}

static inline void craftax_render_full(Craftax* env) {
    const int hud_h = 248;
    const int top_y = 12;
    const int top_h = 160;
    const int status_y = 180;
    const int status_h = 56;
    if (!craftax_prepare_render(env, "PufferLib Craftax", hud_h)) {
        return;
    }

    Client* client = craftax_get_client(env);
    CraftaxState* s = env->state;
    int lvl = s->player_level;
    int view_w = craftax_view_width(client);
    int view_h = craftax_view_height(client);

    BeginDrawing();
    ClearBackground(BLACK);

    craftax_draw_world_view(env);

    int hud_y = view_h;
    bool compact_hud = view_w < 900;
    int inventory_x = compact_hud ? 268 : 268;
    int inventory_w = compact_hud ? 176 : 236;
    int inventory_col2_x = compact_hud ? 352 : 388;
    int progress_x = compact_hud ? 456 : 516;
    int progress_w = view_w - progress_x - 12;
    int achievements_font = compact_hud ? 22 : 30;
    int progress_font = compact_hud ? 16 : 18;
    DrawRectangle(0, hud_y, view_w, hud_h, (Color){14, 16, 19, 255});
    craftax_draw_hud_panel(12, hud_y + top_y, 244, top_h, "Vitals");
    craftax_draw_hud_bar(24, hud_y + 40, 140, "Health", (int)(s->player_health + 0.5f), 10,
        (Color){198, 58, 62, 255});
    craftax_draw_hud_bar(24, hud_y + 64, 140, "Food", s->player_food, 10,
        (Color){196, 142, 56, 255});
    craftax_draw_hud_bar(24, hud_y + 88, 140, "Drink", s->player_drink, 10,
        (Color){70, 150, 224, 255});
    craftax_draw_hud_bar(24, hud_y + 112, 140, "Energy", s->player_energy, 10,
        (Color){120, 196, 94, 255});
    craftax_draw_hud_bar(24, hud_y + 136, 140, "Mana", s->player_mana, 10,
        (Color){168, 112, 222, 255});

    int ach_count = 0;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) ach_count += s->achievements[i] ? 1 : 0;

    craftax_draw_hud_panel(inventory_x, hud_y + top_y, inventory_w, top_h, "Inventory");
    craftax_draw_hud_slot(inventory_x + 12, hud_y + 40, CRAFTAX_BLOCK_WOOD, WHITE, "Wood", s->inventory.wood);
    craftax_draw_hud_slot(inventory_x + 12, hud_y + 72, CRAFTAX_BLOCK_STONE, WHITE, "Stone", s->inventory.stone);
    craftax_draw_hud_slot(inventory_x + 12, hud_y + 104, CRAFTAX_BLOCK_IRON, WHITE, "Iron", s->inventory.iron);
    craftax_draw_hud_slot(inventory_col2_x, hud_y + 40, CRAFTAX_BLOCK_COAL, WHITE, "Coal", s->inventory.coal);
    craftax_draw_hud_slot(inventory_col2_x, hud_y + 72, CRAFTAX_BLOCK_DIAMOND, WHITE, "Diamond", s->inventory.diamond);
    craftax_draw_hud_slot(inventory_col2_x, hud_y + 104, CRAFTAX_TEX_ITEM_BASE + CRAFTAX_ITEM_TORCH, WHITE,
        "Torch", s->inventory.torches);

    craftax_draw_hud_panel(progress_x, hud_y + top_y, progress_w, top_h, "Progress");
    DrawText(TextFormat("%d / %d achievements", ach_count, CRAFTAX_NUM_ACHIEVEMENTS),
        progress_x + 14, hud_y + 38, achievements_font, (Color){255, 236, 150, 255});
    DrawText(TextFormat("%.1f / 226 reward", env->episode_return_accum),
        progress_x + 14, hud_y + 64, progress_font, (Color){170, 255, 170, 255});
    DrawText(TextFormat("Floor %d   Kills %d / %d",
        s->player_level, s->monsters_killed[lvl], CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL),
        progress_x + 14, hud_y + 92, progress_font, WHITE);
    DrawText(TextFormat("Timestep %d   XP %d",
        s->timestep, s->player_xp),
        progress_x + 14, hud_y + 116, progress_font, (Color){220, 224, 228, 255});

    craftax_draw_hud_panel(12, hud_y + status_y, view_w - 24, status_h, "Status");
    DrawText(TextFormat("Pos (%d, %d)   DEX %d   STR %d   INT %d   Pickaxe %d   Sword %d   Bow %d",
        s->player_position[0], s->player_position[1], s->player_dexterity, s->player_strength,
        s->player_intelligence, s->inventory.pickaxe, s->inventory.sword, s->inventory.bow),
        24, hud_y + 202, 16, WHITE);
    DrawText(TextFormat("Ruby %d   Sapphire %d   Sapling %d   Arrows %d   Books %d   Armour %d %d %d %d",
        s->inventory.ruby, s->inventory.sapphire, s->inventory.sapling, s->inventory.arrows,
        s->inventory.books, s->inventory.armour[0], s->inventory.armour[1],
        s->inventory.armour[2], s->inventory.armour[3]),
        24, hud_y + 222, 15, (Color){215, 218, 222, 255});

    int badge_x = compact_hud ? 266 : 360;
    int badge_y = hud_y + status_y + 10;
    craftax_draw_hud_badge(badge_x, badge_y, 78,
        s->is_sleeping ? "Sleeping" : "Awake",
        s->is_sleeping ? (Color){74, 88, 134, 255} : (Color){58, 96, 68, 255}, WHITE);
    craftax_draw_hud_badge(badge_x + 86, badge_y, 78,
        s->is_resting ? "Resting" : "Active",
        s->is_resting ? (Color){126, 96, 52, 255} : (Color){64, 72, 80, 255}, WHITE);
    craftax_draw_hud_badge(badge_x + 172, badge_y, 78,
        s->learned_spells[0] ? "Fireball" : "No Fire",
        s->learned_spells[0] ? (Color){150, 74, 40, 255} : (Color){54, 58, 62, 255}, WHITE);
    craftax_draw_hud_badge(badge_x + 258, badge_y, 78,
        s->learned_spells[1] ? "Iceball" : "No Ice",
        s->learned_spells[1] ? (Color){54, 112, 160, 255} : (Color){54, 58, 62, 255}, WHITE);
    craftax_draw_view_mode_badge(env, view_w - 150, hud_y + 12);

    EndDrawing();
}
