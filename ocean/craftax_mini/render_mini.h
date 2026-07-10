#pragma once

#ifdef CRAFTAX_ENABLE_RENDERING

static inline void craftax_render_mini(Craftax* env) {
    const int hud_h = 136;
    const int hud_font = 18;
    const int hud_title_font = 24;
    const int hud_line = 24;
    const int hud_pad = 6;

    if (!craftax_prepare_render(env, "PufferLib Craftax Mini", hud_h)) {
        return;
    }

    CraftaxState* s = env->state;
    Client* client = craftax_get_client(env);
    int view_w = craftax_view_width(client);
    int view_h = craftax_view_height(client);
    int hud_y = view_h;
    int32_t goal_block = craftax_mini_current_goal_block(env);
    int goal_count = craftax_mini_inventory_count_for_goal(s, goal_block);
    bool found_goal_gemstone = goal_count > 0;
    Color goal_color = craftax_mini_goal_color(goal_block);

    BeginDrawing();
    ClearBackground(BLACK);

    craftax_draw_world_view(env);

    DrawRectangle(0, hud_y, view_w, hud_h, (Color){20, 20, 20, 255});
    DrawText(TextFormat("task: collect %s", craftax_mini_goal_name(goal_block)),
             4, hud_y + hud_pad, hud_title_font, goal_color);
    DrawText(TextFormat("progress: %d  status: %s",
             goal_count, found_goal_gemstone ? "complete" : "searching"),
             4, hud_y + hud_pad + hud_line + 2, hud_font,
             found_goal_gemstone ? (Color){120, 255, 120, 255} : (Color){220, 220, 220, 255});
    DrawText(TextFormat("ret: %.2f  len: %d / %d",
             env->episode_return_accum, env->episode_length_accum, g_craftax_mini_max_timesteps),
             4, hud_y + hud_pad + 2 * hud_line + 4, hud_font, (Color){220, 210, 140, 255});
    DrawText(TextFormat("HP: %.0f  food: %d  drink: %d  energy: %d  light: %.2f",
             s->player_health, s->player_food, s->player_drink,
             s->player_energy, s->light_level),
             4, hud_y + hud_pad + 3 * hud_line + 4, hud_font, WHITE);
    DrawText(TextFormat("pos: (%d, %d)  level: %d  step: %d",
             s->player_position[0], s->player_position[1], s->player_level, s->timestep),
             4, hud_y + hud_pad + 4 * hud_line + 4, hud_font, (Color){190, 190, 190, 255});

    if (found_goal_gemstone) {
        DrawText("goal item acquired", view_w - 220, hud_y + hud_pad + 2, 22,
                 (Color){120, 255, 120, 255});
    }
    craftax_draw_view_mode_badge(env, view_w - 150, hud_y + 12);

    EndDrawing();
}

#endif
