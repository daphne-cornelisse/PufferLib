/*
 * Sweep dashboard for PufferLib 5c.
 *
 * Loads logs/<env>/sweep_*.ini, plots final env/score with raylib, writes
 * logs/<env>/sweep_plot.png, and with --render records logs/<env>/best.gif
 * from the best checkpoint via ./puffer render.
 *
 * Build:  ./build.sh eval --fast [--headless]
 * Usage:  ./eval ENV [--render]
 */

#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "raylib.h"

#define PUFF_CYAN ((Color){0, 187, 187, 255})
#define PUFF_RED ((Color){220, 60, 80, 255})
#define PUFF_GREEN ((Color){80, 200, 120, 255})
#define PUFF_ORANGE ((Color){240, 160, 50, 255})
#define PUFF_PURPLE ((Color){160, 120, 220, 255})
#define PUFF_WHITE ((Color){241, 241, 241, 255})
#define PUFF_DIM ((Color){140, 160, 160, 255})
#define PUFF_BG ((Color){6, 24, 24, 255})
#define PUFF_PANEL ((Color){12, 36, 36, 255})
#define PUFF_GRID ((Color){20, 50, 50, 255})

#define MAX_RUNS 4096
#define MAX_PATH 1024
#define MAX_METRIC_KEY 128
#define MAX_CURVE 64

typedef struct {
    int idx;
    long ts;
    char run_id[128];
    char log_path[MAX_PATH];
    float final_score;
    float max_score;
    float final_return;
    float final_steps;
    float final_uptime; /* wall-clock seconds for the run */
    int n_curve;
    float steps[MAX_CURVE];
    float scores[MAX_CURVE];
    int hidden_size;
    int num_layers;
    int num_agents;
    int total_agents;
    int num_buffers;
    char ckpt_path[MAX_PATH];
    int has_ckpt;
} SweepRun;

typedef struct {
    char env[64];
    char log_dir[MAX_PATH];
    char checkpoint_dir[MAX_PATH];
    char metric_key[MAX_METRIC_KEY]; /* env/score */
    char plot_path[MAX_PATH];        /* logs/<env>/sweep_plot.png */
    char gif_path[MAX_PATH];         /* logs/<env>/best.gif */
    char puffer_bin[MAX_PATH];
    int do_render;
    int num_frames;
    float fps;
} Options;

static void die(const char* msg) {
    fprintf(stderr, "eval: %s\n", msg);
    exit(1);
}

static void dief(const char* fmt, const char* a) {
    fprintf(stderr, "eval: ");
    fprintf(stderr, fmt, a);
    fprintf(stderr, "\n");
    exit(1);
}

static int is_dir(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int is_file(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static void mkdir_p(const char* path) {
    char tmp[MAX_PATH];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t n = strlen(tmp);
    if (n == 0) {
        return;
    }
    if (tmp[n - 1] == '/') {
        tmp[n - 1] = '\0';
    }
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void mkdir_parent(const char* path) {
    char tmp[MAX_PATH];
    snprintf(tmp, sizeof(tmp), "%s", path);
    char* slash = strrchr(tmp, '/');
    if (!slash || slash == tmp) {
        return;
    }
    *slash = '\0';
    mkdir_p(tmp);
}

static char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    if (out_len) {
        *out_len = n;
    }
    return buf;
}

static const char* find_section(const char* text, const char* section) {
    char needle[96];
    snprintf(needle, sizeof(needle), "[%s]", section);
    const char* p = strstr(text, needle);
    if (!p) {
        return NULL;
    }
    p += strlen(needle);
    while (*p == '\r' || *p == '\n') {
        p++;
    }
    return p;
}

static int parse_kv_line(const char* line, char* key, size_t key_n,
        char* val, size_t val_n) {
    while (*line == ' ' || *line == '\t') {
        line++;
    }
    if (*line == '\0' || *line == '#' || *line == '[') {
        return 0;
    }
    const char* eq = strchr(line, '=');
    if (!eq) {
        return 0;
    }
    size_t kn = (size_t)(eq - line);
    while (kn > 0 && (line[kn - 1] == ' ' || line[kn - 1] == '\t')) {
        kn--;
    }
    if (kn == 0 || kn >= key_n) {
        return 0;
    }
    memcpy(key, line, kn);
    key[kn] = '\0';
    eq++;
    while (*eq == ' ' || *eq == '\t') {
        eq++;
    }
    size_t vn = strlen(eq);
    while (vn > 0 && (eq[vn - 1] == '\n' || eq[vn - 1] == '\r'
            || eq[vn - 1] == ' ' || eq[vn - 1] == '\t')) {
        vn--;
    }
    if (vn >= val_n) {
        vn = val_n - 1;
    }
    memcpy(val, eq, vn);
    val[vn] = '\0';
    return 1;
}

/* Parse comma-separated floats into out[0..maxn); returns count. */
static int parse_float_series(const char* csv, float* out, int maxn) {
    const char* p = csv;
    int n = 0;
    while (*p && n < maxn) {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            p++;
        }
        if (!*p) {
            break;
        }
        char* end = NULL;
        float v = strtof(p, &end);
        if (end == p) {
            break;
        }
        out[n++] = v;
        p = end;
    }
    return n;
}

static int parse_last_float(const char* csv, float* out) {
    float buf[MAX_CURVE];
    int n = parse_float_series(csv, buf, MAX_CURVE);
    if (n <= 0) {
        return 0;
    }
    *out = buf[n - 1];
    return 1;
}

static int parse_max_float(const char* csv, float* out) {
    float buf[MAX_CURVE];
    int n = parse_float_series(csv, buf, MAX_CURVE);
    if (n <= 0) {
        return 0;
    }
    float mx = buf[0];
    for (int i = 1; i < n; i++) {
        if (buf[i] > mx) {
            mx = buf[i];
        }
    }
    *out = mx;
    return 1;
}

static int parse_ini_int(const char* text, const char* key, int* out) {
    char line_key[128], val[512];
    const char* p = text;
    while (*p) {
        const char* eol = strchr(p, '\n');
        size_t n = eol ? (size_t)(eol - p) : strlen(p);
        char line[1024];
        if (n >= sizeof(line)) {
            n = sizeof(line) - 1;
        }
        memcpy(line, p, n);
        line[n] = '\0';
        if (parse_kv_line(line, line_key, sizeof(line_key), val, sizeof(val))) {
            if (strcmp(line_key, key) == 0) {
                *out = (int)strtol(val, NULL, 10);
                return 1;
            }
        }
        if (!eol) {
            break;
        }
        p = eol + 1;
    }
    return 0;
}

static int parse_ini_str(const char* text, const char* key, char* out, size_t out_n) {
    char line_key[128], val[512];
    const char* p = text;
    while (*p) {
        const char* eol = strchr(p, '\n');
        size_t n = eol ? (size_t)(eol - p) : strlen(p);
        char line[1024];
        if (n >= sizeof(line)) {
            n = sizeof(line) - 1;
        }
        memcpy(line, p, n);
        line[n] = '\0';
        if (parse_kv_line(line, line_key, sizeof(line_key), val, sizeof(val))) {
            if (strcmp(line_key, key) == 0) {
                snprintf(out, out_n, "%s", val);
                return 1;
            }
        }
        if (!eol) {
            break;
        }
        p = eol + 1;
    }
    return 0;
}

static int parse_sweep_filename(const char* name, long* ts, int* idx) {
    /* sweep_<timestamp>_<index>.ini */
    if (strncmp(name, "sweep_", 6) != 0) {
        return 0;
    }
    const char* p = name + 6;
    char* end = NULL;
    long t = strtol(p, &end, 10);
    if (end == p || *end != '_') {
        return 0;
    }
    p = end + 1;
    long i = strtol(p, &end, 10);
    if (end == p || strcmp(end, ".ini") != 0) {
        return 0;
    }
    *ts = t;
    *idx = (int)i;
    return 1;
}

static int find_latest_checkpoint(const char* dir, char* out, size_t out_n) {
    DIR* d = opendir(dir);
    if (!d) {
        return 0;
    }
    long best_step = -1;
    char best_name[256] = {0};
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        const char* n = ent->d_name;
        size_t len = strlen(n);
        if (len < 5 || strcmp(n + len - 4, ".bin") != 0) {
            continue;
        }
        char* end = NULL;
        long step = strtol(n, &end, 10);
        if (end == n || strcmp(end, ".bin") != 0) {
            continue;
        }
        if (step > best_step) {
            best_step = step;
            snprintf(best_name, sizeof(best_name), "%s", n);
        }
    }
    closedir(d);
    if (best_step < 0) {
        return 0;
    }
    snprintf(out, out_n, "%s/%s", dir, best_name);
    return 1;
}

static int cmp_run_idx(const void* a, const void* b) {
    const SweepRun* ra = (const SweepRun*)a;
    const SweepRun* rb = (const SweepRun*)b;
    return ra->idx - rb->idx;
}

static int load_sweep_runs(const Options* opt, SweepRun* runs, int max_runs) {
    DIR* d = opendir(opt->log_dir);
    if (!d) {
        dief("cannot open log dir %s", opt->log_dir);
    }

    int n = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL && n < max_runs) {
        long ts = 0;
        int idx = 0;
        if (!parse_sweep_filename(ent->d_name, &ts, &idx)) {
            continue;
        }
        char path[MAX_PATH];
        snprintf(path, sizeof(path), "%s/%s", opt->log_dir, ent->d_name);
        char* text = read_file(path, NULL);
        if (!text) {
            continue;
        }

        const char* metrics = find_section(text, "metrics");
        if (!metrics) {
            free(text);
            continue;
        }

        SweepRun* r = &runs[n];
        memset(r, 0, sizeof(*r));
        r->idx = idx;
        r->ts = ts;
        snprintf(r->log_path, sizeof(r->log_path), "%s", path);
        r->final_score = NAN;
        r->max_score = NAN;
        r->final_return = NAN;
        r->final_steps = NAN;
        r->final_uptime = NAN;
        r->n_curve = 0;
        r->hidden_size = 128;
        r->num_layers = 3;
        r->num_agents = 1;
        r->total_agents = 4096;
        r->num_buffers = 1;

        parse_ini_str(text, "run_id", r->run_id, sizeof(r->run_id));
        if (!r->run_id[0]) {
            snprintf(r->run_id, sizeof(r->run_id), "sweep_%ld_%04d", ts, idx);
        }
        parse_ini_int(text, "hidden_size", &r->hidden_size);
        parse_ini_int(text, "num_layers", &r->num_layers);
        parse_ini_int(text, "num_agents", &r->num_agents);
        parse_ini_int(text, "total_agents", &r->total_agents);
        parse_ini_int(text, "num_buffers", &r->num_buffers);

        float score_series[MAX_CURVE];
        float step_series[MAX_CURVE];
        float uptime_series[MAX_CURVE];
        int n_score = 0, n_step = 0, n_up = 0;

        /* Walk metrics section lines only. */
        const char* p = metrics;
        char key[128], val[4096];
        while (*p && *p != '[') {
            const char* eol = strchr(p, '\n');
            size_t ln = eol ? (size_t)(eol - p) : strlen(p);
            char line[4200];
            if (ln >= sizeof(line)) {
                ln = sizeof(line) - 1;
            }
            memcpy(line, p, ln);
            line[ln] = '\0';
            if (parse_kv_line(line, key, sizeof(key), val, sizeof(val))) {
                if (strcmp(key, opt->metric_key) == 0) {
                    n_score = parse_float_series(val, score_series, MAX_CURVE);
                    if (n_score > 0) {
                        r->final_score = score_series[n_score - 1];
                        parse_max_float(val, &r->max_score);
                    }
                } else if (strcmp(key, "env/episode_return") == 0) {
                    parse_last_float(val, &r->final_return);
                } else if (strcmp(key, "agent_steps") == 0) {
                    n_step = parse_float_series(val, step_series, MAX_CURVE);
                    if (n_step > 0) {
                        r->final_steps = step_series[n_step - 1];
                    }
                } else if (strcmp(key, "uptime") == 0) {
                    n_up = parse_float_series(val, uptime_series, MAX_CURVE);
                    if (n_up > 0) {
                        r->final_uptime = uptime_series[n_up - 1];
                    }
                }
            }
            if (!eol) {
                break;
            }
            p = eol + 1;
        }

        /* Align training curve points (score vs agent_steps). */
        int nc = n_score;
        if (n_step < nc) {
            nc = n_step;
        }
        r->n_curve = nc;
        for (int i = 0; i < nc; i++) {
            r->scores[i] = score_series[i];
            r->steps[i] = step_series[i];
        }

        if (!isnan(r->final_score)) {
            char ckpt_dir[MAX_PATH];
            snprintf(ckpt_dir, sizeof(ckpt_dir), "%s/%s/%s",
                opt->checkpoint_dir, opt->env, r->run_id);
            if (find_latest_checkpoint(ckpt_dir, r->ckpt_path, sizeof(r->ckpt_path))) {
                r->has_ckpt = 1;
            }
            n++;
        }
        free(text);
    }
    closedir(d);

    qsort(runs, (size_t)n, sizeof(SweepRun), cmp_run_idx);
    return n;
}

static SweepRun* find_best_run(SweepRun* runs, int n) {
    SweepRun* best = NULL;
    for (int i = 0; i < n; i++) {
        if (!best || runs[i].final_score > best->final_score) {
            best = &runs[i];
        }
    }
    return best;
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

/* Logical design size; window is PLOT_DPI_SCALE times larger for sharp PNG export. */
#define PLOT_LOGIC_W 1680
#define PLOT_LOGIC_H 820
#define PLOT_DPI_SCALE 3

static float g_ui = 1.0f;

static float ui(float v) { return v * g_ui; }
static int uii(int v) { return (int)lroundf((float)v * g_ui); }

static void draw_text(const char* text, float x, float y, int size, Color c) {
    DrawText(text, (int)lroundf(x), (int)lroundf(y), uii(size), c);
}

static int measure_text(const char* text, int size) {
    return MeasureText(text, uii(size));
}

/*
 * Dark-mode sequential heatmap for the teal panel (PUFF_BG / PUFF_PANEL).
 *
 * Candidates considered for this background:
 *   - full plasma/inferno: cool end is near-black → markers vanish
 *   - viridis: cool end is blue-green → blends with teal panel
 *   - pure heat (red→yellow): distinctive but muddy mid-range
 *   - lifted plasma / "ember": cool end is bright magenta-violet (readable),
 *     then rose → orange → gold (slow). High contrast vs teal, ordered,
 *     easy to read fast vs slow.
 *
 * Fast (t=0) → slow (t=1).
 */
static Color color_heatmap(float t) {
    t = clampf(t, 0.0f, 1.0f);
    /* Lifted plasma / ember stops (sRGB). Min channel ~90 so nothing sinks
     * into the dark teal panel. */
    static const float stops[][3] = {
        { 140.0f,  70.0f, 210.0f }, /* bright violet   (fast) */
        { 200.0f,  70.0f, 180.0f }, /* magenta */
        { 240.0f,  90.0f, 120.0f }, /* coral / rose */
        { 250.0f, 140.0f,  60.0f }, /* orange */
        { 255.0f, 200.0f,  50.0f }, /* amber */
        { 255.0f, 245.0f, 120.0f }, /* soft gold      (slow) */
    };
    const int nseg = (int)(sizeof(stops) / sizeof(stops[0])) - 1;
    float x = t * (float)nseg;
    int i = (int)x;
    if (i >= nseg) {
        i = nseg - 1;
    }
    float f = x - (float)i;
    float r = stops[i][0] + f * (stops[i + 1][0] - stops[i][0]);
    float g = stops[i][1] + f * (stops[i + 1][1] - stops[i][1]);
    float b = stops[i][2] + f * (stops[i + 1][2] - stops[i][2]);
    return (Color){
        (unsigned char)clampf(r, 0.0f, 255.0f),
        (unsigned char)clampf(g, 0.0f, 255.0f),
        (unsigned char)clampf(b, 0.0f, 255.0f),
        255
    };
}

/* Percentile of a float array (sorted copy). q in [0,1]. */
static float percentile_sorted(float* sorted, int n, float q) {
    if (n <= 0) {
        return 0.0f;
    }
    if (n == 1) {
        return sorted[0];
    }
    float pos = q * (float)(n - 1);
    int i = (int)pos;
    float f = pos - (float)i;
    if (i >= n - 1) {
        return sorted[n - 1];
    }
    return sorted[i] * (1.0f - f) + sorted[i + 1] * f;
}

static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static void format_steps(char* buf, size_t n, float steps) {
    if (steps >= 1e9f) {
        snprintf(buf, n, "%.2fB", steps / 1e9f);
    } else if (steps >= 1e6f) {
        snprintf(buf, n, "%.1fM", steps / 1e6f);
    } else if (steps >= 1e3f) {
        snprintf(buf, n, "%.1fK", steps / 1e3f);
    } else {
        snprintf(buf, n, "%.0f", steps);
    }
}

static void format_time(char* buf, size_t n, float sec) {
    if (sec >= 3600.0f) {
        snprintf(buf, n, "%.1fh", sec / 3600.0f);
    } else if (sec >= 60.0f) {
        snprintf(buf, n, "%.1fm", sec / 60.0f);
    } else {
        snprintf(buf, n, "%.0fs", sec);
    }
}

typedef struct {
    float x, y, w, h;
    float x0, x1, y0, y1;
} Panel;

static float panel_x(const Panel* p, float v) {
    return p->x + ((v - p->x0) / (p->x1 - p->x0)) * p->w;
}

static float panel_y(const Panel* p, float v) {
    return p->y + (1.0f - (v - p->y0) / (p->y1 - p->y0)) * p->h;
}

/* Draw axes/grid. y_labels live in the left gutter (axis_gutter wide). */
static void draw_panel_frame(const Panel* p, float axis_gutter) {
    DrawRectangleRec(
        (Rectangle){p->x - axis_gutter, p->y - ui(4), p->w + axis_gutter + ui(8),
            p->h + ui(8)},
        PUFF_PANEL);
    const int y_ticks = 5;
    const int x_ticks = 5;
    float grid_w = ui(1.0f);
    float axis_w = ui(2.0f);
    for (int i = 0; i <= y_ticks; i++) {
        float t = (float)i / (float)y_ticks;
        float y = p->y + p->h * (1.0f - t);
        DrawLineEx((Vector2){p->x, y}, (Vector2){p->x + p->w, y}, grid_w, PUFF_GRID);
        float val = p->y0 + t * (p->y1 - p->y0);
        char lab[32];
        snprintf(lab, sizeof(lab), "%.1f", val);
        int tw = measure_text(lab, 13);
        draw_text(lab, p->x - ui(10) - tw, y - ui(7), 13, PUFF_DIM);
    }
    for (int i = 0; i <= x_ticks; i++) {
        float t = (float)i / (float)x_ticks;
        float x = p->x + p->w * t;
        DrawLineEx((Vector2){x, p->y}, (Vector2){x, p->y + p->h}, grid_w, PUFF_GRID);
    }
    DrawLineEx((Vector2){p->x, p->y}, (Vector2){p->x, p->y + p->h}, axis_w, PUFF_WHITE);
    DrawLineEx((Vector2){p->x, p->y + p->h}, (Vector2){p->x + p->w, p->y + p->h},
        axis_w, PUFF_WHITE);
}

static void plot_sweep(const Options* opt, const SweepRun* runs, int n,
        const SweepRun* best) {
    g_ui = (float)GetScreenWidth() / (float)PLOT_LOGIC_W;
    if (g_ui < 1.0f) {
        g_ui = 1.0f;
    }
    const float W = (float)GetScreenWidth();
    const float H = (float)GetScreenHeight();
    /* Logical layout units; ui() scales them to the high-DPI framebuffer. */
    const float margin = ui(20.0f);
    const float title_h = ui(36.0f);
    const float panel_title_h = ui(28.0f);
    const float annot_h = ui(40.0f);
    const float x_label_h = ui(44.0f);
    const float legend_h = ui(28.0f);
    const float footer_h = ui(40.0f);
    const float gap = ui(40.0f);
    const float y_gutter = ui(72.0f);
    const float cbar_w = ui(70.0f);

    ClearBackground(PUFF_BG);

    char title[256];
    snprintf(title, sizeof(title),
        "%s hyperparameter sweep  ·  %d runs",
        opt->env, n);
    draw_text(title, margin, ui(10), 22, PUFF_WHITE);

    if (n == 0) {
        draw_text("No sweep runs with metrics found.", margin, ui(80), 20, PUFF_RED);
        return;
    }

    float content_top = title_h;
    float content_bot = H - footer_h;
    float col_top = content_top + panel_title_h + annot_h;
    float col_bot = content_bot - x_label_h - legend_h;
    float panel_h = col_bot - col_top;
    if (panel_h < ui(120.0f)) {
        panel_h = ui(120.0f);
    }

    float usable_w = W - 2.0f * margin - gap - 2.0f * y_gutter - cbar_w;
    float panel_w = usable_w * 0.5f;

    Panel left = {
        .x = margin + y_gutter,
        .y = col_top,
        .w = panel_w,
        .h = panel_h,
    };
    Panel right = {
        .x = margin + y_gutter + panel_w + gap + y_gutter,
        .y = col_top,
        .w = panel_w,
        .h = panel_h,
    };

    /* ---------- Left: training curves (score vs agent steps) ---------- */
    float steps_max = 1.0f;
    float score_min = FLT_MAX, score_max = -FLT_MAX;
    float total_wall = 0.0f;
    for (int i = 0; i < n; i++) {
        if (!isnan(runs[i].final_steps) && runs[i].final_steps > steps_max) {
            steps_max = runs[i].final_steps;
        }
        if (!isnan(runs[i].final_uptime)) {
            total_wall += runs[i].final_uptime;
        }
        for (int k = 0; k < runs[i].n_curve; k++) {
            if (runs[i].scores[k] < score_min) {
                score_min = runs[i].scores[k];
            }
            if (runs[i].scores[k] > score_max) {
                score_max = runs[i].scores[k];
            }
            if (runs[i].steps[k] > steps_max) {
                steps_max = runs[i].steps[k];
            }
        }
        if (!isnan(runs[i].final_score)) {
            if (runs[i].final_score < score_min) {
                score_min = runs[i].final_score;
            }
            if (runs[i].final_score > score_max) {
                score_max = runs[i].final_score;
            }
        }
    }
    if (score_max <= score_min) {
        score_max = score_min + 1.0f;
    }
    float spad = 0.08f * (score_max - score_min);
    left.x0 = 0.0f;
    left.x1 = steps_max * 1.02f;
    left.y0 = score_min - spad;
    left.y1 = score_max + spad;

    /* Panel title + annotations ABOVE the axes (no overlap with curves). */
    draw_text("Training curves  (env/score vs agent steps)",
        left.x, content_top + ui(4), 16, PUFF_WHITE);
    {
        char steps_s[32], time_s[32], ann[192], tot[96];
        float best_steps = best && !isnan(best->final_steps) ? best->final_steps : steps_max;
        float best_up = best && !isnan(best->final_uptime) ? best->final_uptime : 0.0f;
        format_steps(steps_s, sizeof(steps_s), best_steps);
        format_time(time_s, sizeof(time_s), best_up);
        snprintf(ann, sizeof(ann),
            "best end: %s steps · wall %s", steps_s, time_s);
        draw_text(ann, left.x, content_top + panel_title_h + ui(2), 13, PUFF_ORANGE);
        format_time(time_s, sizeof(time_s), total_wall);
        snprintf(tot, sizeof(tot), "sweep wall-clock sum %s", time_s);
        draw_text(tot, left.x, content_top + panel_title_h + ui(18), 12, PUFF_DIM);
    }

    draw_panel_frame(&left, y_gutter);

    /* X tick labels: steps */
    for (int i = 0; i <= 5; i++) {
        float t = (float)i / 5.0f;
        float v = left.x0 + t * (left.x1 - left.x0);
        float x = panel_x(&left, v);
        char lab[32];
        format_steps(lab, sizeof(lab), v);
        int tw = measure_text(lab, 12);
        draw_text(lab, x - tw / 2.0f, left.y + left.h + ui(8), 12, PUFF_DIM);
    }
    draw_text("agent steps", left.x + left.w / 2.0f - ui(40.0f),
        left.y + left.h + ui(26), 13, PUFF_DIM);

    /* All training curves (faint) */
    for (int i = 0; i < n; i++) {
        if (runs[i].n_curve < 2) {
            continue;
        }
        for (int k = 1; k < runs[i].n_curve; k++) {
            Vector2 a = {
                panel_x(&left, runs[i].steps[k - 1]),
                panel_y(&left, runs[i].scores[k - 1])
            };
            Vector2 b = {
                panel_x(&left, runs[i].steps[k]),
                panel_y(&left, runs[i].scores[k])
            };
            DrawLineEx(a, b, ui(1.0f), Fade(PUFF_DIM, 0.35f));
        }
    }

    if (best && best->n_curve >= 2) {
        for (int k = 1; k < best->n_curve; k++) {
            Vector2 a = {
                panel_x(&left, best->steps[k - 1]),
                panel_y(&left, best->scores[k - 1])
            };
            Vector2 b = {
                panel_x(&left, best->steps[k]),
                panel_y(&left, best->scores[k])
            };
            DrawLineEx(a, b, ui(3.0f), PUFF_GREEN);
        }
        for (int k = 0; k < best->n_curve; k++) {
            DrawCircleV(
                (Vector2){
                    panel_x(&left, best->steps[k]),
                    panel_y(&left, best->scores[k])
                },
                ui(4.0f), PUFF_GREEN);
        }
    }

    /* Final score markers */
    for (int i = 0; i < n; i++) {
        if (isnan(runs[i].final_steps) || isnan(runs[i].final_score)) {
            continue;
        }
        float x = panel_x(&left, runs[i].final_steps);
        float y = panel_y(&left, runs[i].final_score);
        Color c = (best && runs[i].idx == best->idx) ? PUFF_GREEN : PUFF_CYAN;
        DrawCircleV((Vector2){x, y}, ui(3.0f), c);
    }

    /* Best label: place in top-right of left panel, outside dense curve region. */
    if (best) {
        char ann[128];
        snprintf(ann, sizeof(ann), "best %.2f  (#%04d)", best->final_score, best->idx);
        float bx = panel_x(&left, isnan(best->final_steps) ? left.x1 : best->final_steps);
        float by = panel_y(&left, best->final_score);
        float ax = left.x + left.w - ui(8) - measure_text(ann, 14);
        float ay = left.y + ui(10);
        DrawLineEx((Vector2){bx, by},
            (Vector2){ax + measure_text(ann, 14) * 0.5f, ay + ui(14)},
            ui(1.2f), Fade(PUFF_GREEN, 0.7f));
        draw_text(ann, ax, ay, 14, PUFF_GREEN);
    }

    /* Legend under left x-axis */
    {
        float lx = left.x;
        float ly = left.y + left.h + x_label_h - ui(2);
        DrawLineEx((Vector2){lx, ly + ui(6)}, (Vector2){lx + ui(18), ly + ui(6)},
            ui(1.0f), Fade(PUFF_DIM, 0.6f));
        draw_text("all runs", lx + ui(24), ly, 12, PUFF_DIM);
        DrawLineEx((Vector2){lx + ui(100), ly + ui(6)},
            (Vector2){lx + ui(118), ly + ui(6)}, ui(3.0f), PUFF_GREEN);
        draw_text("best run", lx + ui(124), ly, 12, PUFF_DIM);
        DrawCircleV((Vector2){lx + ui(210), ly + ui(6)}, ui(3.0f), PUFF_CYAN);
        draw_text("final", lx + ui(218), ly, 12, PUFF_DIM);
    }

    /* ---------- Right: sweep idx vs episode return, color = wall clock ---------- */
    float ret_min = FLT_MAX, ret_max = -FLT_MAX;
    int idx_min = runs[0].idx, idx_max = runs[n - 1].idx;
    float* uptimes = (float*)malloc((size_t)n * sizeof(float));
    int n_up = 0;
    for (int i = 0; i < n; i++) {
        if (!isnan(runs[i].final_return)) {
            if (runs[i].final_return < ret_min) {
                ret_min = runs[i].final_return;
            }
            if (runs[i].final_return > ret_max) {
                ret_max = runs[i].final_return;
            }
        }
        if (!isnan(runs[i].final_uptime)) {
            uptimes[n_up++] = runs[i].final_uptime;
        }
    }
    if (ret_max <= ret_min) {
        ret_max = ret_min + 1.0f;
    }
    if (idx_max <= idx_min) {
        idx_max = idx_min + 1;
    }

    /* Percentile color scale: robust min/max so outliers don't wash out contrast. */
    float up_lo = 0.0f, up_hi = 1.0f;
    if (n_up > 0) {
        qsort(uptimes, (size_t)n_up, sizeof(float), cmp_float);
        up_lo = percentile_sorted(uptimes, n_up, 0.10f);
        up_hi = percentile_sorted(uptimes, n_up, 0.90f);
        if (up_hi <= up_lo) {
            up_lo = uptimes[0];
            up_hi = uptimes[n_up - 1];
        }
        if (up_hi <= up_lo) {
            up_hi = up_lo + 1.0f;
        }
    }

    float rpad = 0.08f * (ret_max - ret_min);
    right.x0 = (float)idx_min;
    right.x1 = (float)idx_max;
    right.y0 = ret_min - rpad;
    right.y1 = ret_max + rpad;

    draw_text("Final episode return",
        right.x, content_top + ui(4), 16, PUFF_WHITE);
    {
        char lo[32], hi[32], note[128];
        format_time(lo, sizeof(lo), up_lo);
        format_time(hi, sizeof(hi), up_hi);
        snprintf(note, sizeof(note),
            "color = wall-clock  (scale p10=%s … p90=%s)", lo, hi);
        draw_text(note, right.x, content_top + panel_title_h + ui(2), 13, PUFF_DIM);
    }

    draw_panel_frame(&right, y_gutter);

    for (int i = 0; i <= 5; i++) {
        float t = (float)i / 5.0f;
        float v = right.x0 + t * (right.x1 - right.x0);
        float x = panel_x(&right, v);
        char lab[32];
        snprintf(lab, sizeof(lab), "%d", (int)lroundf(v));
        int tw = measure_text(lab, 12);
        draw_text(lab, x - tw / 2.0f, right.y + right.h + ui(8), 12, PUFF_DIM);
    }
    draw_text("sweep run index", right.x + right.w / 2.0f - ui(52.0f),
        right.y + right.h + ui(26), 13, PUFF_DIM);

    /* Points: wall-clock color (p10–p90). No special best-run callout. */
    for (int i = 0; i < n; i++) {
        if (isnan(runs[i].final_return)) {
            continue;
        }
        float t = 0.5f;
        if (!isnan(runs[i].final_uptime)) {
            t = (runs[i].final_uptime - up_lo) / (up_hi - up_lo);
            t = clampf(t, 0.0f, 1.0f);
        }
        Color c = color_heatmap(t);
        float x = panel_x(&right, (float)runs[i].idx);
        float y = panel_y(&right, runs[i].final_return);
        DrawCircleV((Vector2){x, y}, ui(5.0f), (Color){10, 30, 30, 200});
        DrawCircleV((Vector2){x, y}, ui(4.0f), c);
    }

    /* Colorbar to the RIGHT of the panel (own column — no overlap). */
    {
        float cb_x = right.x + right.w + ui(14);
        float cb_y = right.y + ui(8);
        float cb_h = right.h - ui(16);
        float cb_bw = ui(14.0f);
        const int segs = 64;
        for (int i = 0; i < segs; i++) {
            float t0 = (float)i / (float)segs;
            float t1 = (float)(i + 1) / (float)segs;
            float y0 = cb_y + cb_h * (1.0f - t1);
            float y1 = cb_y + cb_h * (1.0f - t0);
            Color c = color_heatmap(0.5f * (t0 + t1));
            DrawRectangleRec((Rectangle){cb_x, y0, cb_bw, y1 - y0 + 1}, c);
        }
        DrawRectangleLinesEx((Rectangle){cb_x, cb_y, cb_bw, cb_h}, ui(1.0f), PUFF_WHITE);
        char lo[32], hi[32], mid[32];
        format_time(lo, sizeof(lo), up_lo);
        format_time(hi, sizeof(hi), up_hi);
        format_time(mid, sizeof(mid), 0.5f * (up_lo + up_hi));
        /* Labels to the right of the bar only — avoid stacking under the bar. */
        draw_text("slow", cb_x + cb_bw + ui(6), cb_y - ui(2), 11, PUFF_DIM);
        draw_text(hi, cb_x + cb_bw + ui(6), cb_y + ui(12), 12, PUFF_WHITE);
        draw_text(mid, cb_x + cb_bw + ui(6), cb_y + cb_h * 0.5f - ui(6), 12, PUFF_DIM);
        draw_text(lo, cb_x + cb_bw + ui(6), cb_y + cb_h - ui(26), 12, PUFF_WHITE);
        draw_text("fast", cb_x + cb_bw + ui(6), cb_y + cb_h - ui(12), 11, PUFF_DIM);
    }

    free(uptimes);

    /* Footer */
    {
        float ly = H - ui(28);
        if (best) {
            char stats[512];
            char steps_s[32], time_s[32];
            format_steps(steps_s, sizeof(steps_s),
                isnan(best->final_steps) ? 0.0f : best->final_steps);
            format_time(time_s, sizeof(time_s),
                isnan(best->final_uptime) ? 0.0f : best->final_uptime);
            snprintf(stats, sizeof(stats),
                "best #%04d   score=%.3f   return=%.1f   steps=%s   wall=%s   %s",
                best->idx, best->final_score,
                isnan(best->final_return) ? 0.0f : best->final_return,
                steps_s, time_s, best->run_id);
            draw_text(stats, margin, ly, 13, PUFF_WHITE);
        }
    }
}

static void save_plot_png(const char* path) {
    mkdir_parent(path);
    Image img = LoadImageFromScreen();
    /* Fix headless BGRA flip like puf_pipe_frame_fd, for software renderer. */
#if defined(PLATFORM_MEMORY) || defined(GRAPHICS_API_OPENGL_SOFTWARE)
    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    int w = img.width;
    int h = img.height;
    unsigned char* px = (unsigned char*)img.data;
    size_t row = (size_t)w * 4;
    unsigned char* tmp = (unsigned char*)malloc(row);
    if (tmp) {
        for (int y = 0; y < h / 2; y++) {
            unsigned char* a = px + (size_t)y * row;
            unsigned char* b = px + (size_t)(h - 1 - y) * row;
            memcpy(tmp, a, row);
            memcpy(a, b, row);
            memcpy(b, tmp, row);
        }
        free(tmp);
        for (int i = 0; i < w * h; i++) {
            unsigned char r = px[i * 4 + 0];
            unsigned char b = px[i * 4 + 2];
            px[i * 4 + 0] = b;
            px[i * 4 + 2] = r;
        }
    }
#endif
    if (!ExportImage(img, path)) {
        fprintf(stderr, "eval: failed to save plot to %s\n", path);
    } else {
        printf("Saved plot: %s  (%dx%d @ %dx scale ≈ %d DPI)\n",
            path, img.width, img.height, PLOT_DPI_SCALE, 96 * PLOT_DPI_SCALE);
    }
    UnloadImage(img);
}

static int render_best_checkpoint(const Options* opt, const SweepRun* best) {
    if (!best) {
        fprintf(stderr, "eval: no best run to render\n");
        return 1;
    }
    if (!best->has_ckpt) {
        fprintf(stderr, "eval: best run %s has no checkpoint under %s/%s/%s\n",
            best->run_id, opt->checkpoint_dir, opt->env, best->run_id);
        return 1;
    }
    if (!is_file(opt->puffer_bin)) {
        fprintf(stderr,
            "eval: puffer binary not found at %s\n"
            "      build with: ./build.sh %s --headless   (or without --headless)\n",
            opt->puffer_bin, opt->env);
        return 1;
    }

    mkdir_parent(opt->gif_path);

    /* total_agents must be a multiple of num_agents; use one env for render. */
    int agents = best->num_agents > 0 ? best->num_agents : 1;
    char arg_model[MAX_PATH + 64];
    char arg_h[64], arg_L[64], arg_agents[64], arg_frames[64], arg_gif[MAX_PATH + 64];
    char arg_fps[64], arg_buf[64];
    snprintf(arg_model, sizeof(arg_model), "base.load_model_path=%s", best->ckpt_path);
    snprintf(arg_h, sizeof(arg_h), "policy.hidden_size=%d", best->hidden_size);
    snprintf(arg_L, sizeof(arg_L), "policy.num_layers=%d", best->num_layers);
    snprintf(arg_agents, sizeof(arg_agents), "vec.total_agents=%d", agents);
    snprintf(arg_buf, sizeof(arg_buf), "vec.num_buffers=1");
    snprintf(arg_frames, sizeof(arg_frames), "base.num_frames=%d", opt->num_frames);
    snprintf(arg_gif, sizeof(arg_gif), "base.gif_path=%s", opt->gif_path);
    snprintf(arg_fps, sizeof(arg_fps), "base.fps=%g", (double)opt->fps);

    printf("Rendering best policy:\n");
    printf("  run     %s  score=%.3f\n", best->run_id, best->final_score);
    printf("  ckpt    %s\n", best->ckpt_path);
    printf("  gif     %s  (%d frames @ %g fps)\n",
        opt->gif_path, opt->num_frames, (double)opt->fps);
    printf("  cmd     %s render %s ...\n", opt->puffer_bin, opt->env);

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }
    if (pid == 0) {
        execl(opt->puffer_bin, opt->puffer_bin, "render", opt->env,
            arg_model, arg_h, arg_L, arg_agents, arg_buf,
            arg_frames, arg_gif, arg_fps,
            (char*)NULL);
        perror("execl puffer");
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return 1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "eval: puffer render failed (status=%d)\n", status);
        return 1;
    }
    if (is_file(opt->gif_path)) {
        printf("Wrote gif: %s\n", opt->gif_path);
        return 0;
    }
    fprintf(stderr, "eval: puffer finished but gif missing: %s\n", opt->gif_path);
    return 1;
}

static void usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s ENV [--render]\n"
        "\n"
        "Plot logs/ENV/sweep_*.ini to logs/ENV/sweep_plot.png.\n"
        "With --render, also write logs/ENV/best.gif via ./puffer render.\n"
        "\n"
        "Build:  ./build.sh eval --fast [--headless]\n"
        "Render: ./build.sh ENV [--headless]   # needs ./puffer\n",
        argv0);
}

static void parse_args(int argc, char** argv, Options* opt) {
    memset(opt, 0, sizeof(*opt));
    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }
    snprintf(opt->env, sizeof(opt->env), "%s", argv[1]);
    snprintf(opt->log_dir, sizeof(opt->log_dir), "logs/%s", opt->env);
    snprintf(opt->checkpoint_dir, sizeof(opt->checkpoint_dir), "checkpoints");
    snprintf(opt->metric_key, sizeof(opt->metric_key), "env/score");
    snprintf(opt->plot_path, sizeof(opt->plot_path), "logs/%s/sweep_plot.png", opt->env);
    snprintf(opt->gif_path, sizeof(opt->gif_path), "logs/%s/best.gif", opt->env);
    snprintf(opt->puffer_bin, sizeof(opt->puffer_bin), "./puffer");
    opt->do_render = 0;
    opt->num_frames = 300;
    opt->fps = 15.0f;

    for (int i = 2; i < argc; i++) {
        const char* a = argv[i];
        if (strcmp(a, "--render") == 0) {
            opt->do_render = 1;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "eval: unknown argument %s\n", a);
            usage(argv[0]);
            exit(1);
        }
    }
}

int main(int argc, char** argv) {
    Options opt;
    parse_args(argc, argv, &opt);

    if (!is_dir(opt.log_dir)) {
        dief("log dir not found: %s", opt.log_dir);
    }

    SweepRun* runs = (SweepRun*)calloc(MAX_RUNS, sizeof(SweepRun));
    if (!runs) {
        die("oom");
    }
    int n = load_sweep_runs(&opt, runs, MAX_RUNS);
    setbuf(stdout, NULL);
    printf("Loaded %d sweep runs from %s\n", n, opt.log_dir);

    SweepRun* best = find_best_run(runs, n);
    if (best) {
        printf("Best: #%04d  score=%.4f  run_id=%s\n",
            best->idx, best->final_score, best->run_id);
        if (best->has_ckpt) {
            printf("  checkpoint: %s\n", best->ckpt_path);
        } else {
            printf("  checkpoint: (missing)\n");
        }
    }

    int render_status = 0;
    if (opt.do_render) {
        render_status = render_best_checkpoint(&opt, best);
    }

#if defined(PLATFORM_MEMORY)
    SetTraceLogLevel(LOG_WARNING);
#endif
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(PLOT_LOGIC_W * PLOT_DPI_SCALE, PLOT_LOGIC_H * PLOT_DPI_SCALE,
        TextFormat("Puffer sweep — %s", opt.env));
#if defined(PLATFORM_MEMORY)
    SetTargetFPS(0);
#else
    SetTargetFPS(60);
#endif

    int saved = 0;
    while (!WindowShouldClose()) {
        BeginDrawing();
        plot_sweep(&opt, runs, n, best);
        EndDrawing();

        if (!saved) {
            save_plot_png(opt.plot_path);
            saved = 1;
        }

#if defined(PLATFORM_MEMORY)
        break; /* one frame is enough to export the plot */
#endif
    }

    CloseWindow();
    free(runs);

    if (opt.do_render) {
        printf("Done. plot=%s  gif=%s\n", opt.plot_path, opt.gif_path);
        return render_status;
    }
    printf("Done. plot=%s\n", opt.plot_path);
    return 0;
}
