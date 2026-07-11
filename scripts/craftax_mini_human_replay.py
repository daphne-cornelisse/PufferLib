"""Step Craftax Mini through a human episode or random actions and save a GIF.

Examples:
    uv run --with polars --with matplotlib --with pillow python scripts/craftax_mini_human_replay.py --policy human
    uv run --with polars --with matplotlib --with pillow python scripts/craftax_mini_human_replay.py --policy random
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pufferlib.craftax_mini_human_data import (
    BLOCK_COLORS,
    GOAL_BLOCK_IDS,
    HUMAN_DATAFRAME_PATH,
    HUMAN_PLOT_DIR,
    MAP_SIZE,
    WEB_TO_ENV_ACTION,
    is_native_bounds_compatible,
    parse_jax_array_string,
    parse_positions_string,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HUMAN_PLOT_DIR / "replays"
MOVEMENT_ACTIONS = np.asarray([0, 1, 2, 3], dtype=np.int32)


class MiniReplayFrame(ctypes.Structure):
    _fields_ = [
        ("map", ctypes.c_uint8 * (MAP_SIZE * MAP_SIZE)),
        ("player_position", ctypes.c_int32 * 2),
        ("player_direction", ctypes.c_int32),
        ("timestep", ctypes.c_int32),
        ("goal_block", ctypes.c_int32),
        ("reward", ctypes.c_float),
        ("done", ctypes.c_float),
    ]


class NativeCraftaxMiniReplay:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        src = tmp / "craftax_mini_replay.c"
        so = tmp / "craftax_mini_replay.so"
        src.write_text(
            r'''
            #include <stdint.h>
            #include <string.h>
            #include "ocean/craftax_mini/craftax_mini.h"

            typedef struct MiniReplayFrame {
                uint8_t map[CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE];
                int player_position[2];
                int player_direction;
                int timestep;
                int goal_block;
                float reward;
                float done;
            } MiniReplayFrame;

            static void capture_frame(
                Craftax* env,
                int goal_block,
                float reward,
                float done,
                MiniReplayFrame* frame
            ) {
                memcpy(
                    frame->map,
                    env->state->map[0],
                    CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE * sizeof(uint8_t)
                );
                frame->player_position[0] = env->state->player_position[0];
                frame->player_position[1] = env->state->player_position[1];
                frame->player_direction = env->state->player_direction;
                frame->timestep = env->state->timestep;
                frame->goal_block = goal_block;
                frame->reward = reward;
                frame->done = done;
            }

            int craftax_mini_replay_rollout(
                uint32_t world_seed,
                uint32_t slot_id,
                int goal_block,
                int max_timesteps,
                int start_row,
                int start_col,
                const int* actions,
                int num_actions,
                MiniReplayFrame* frames,
                int max_frames
            ) {
                if (max_frames <= 0) {
                    return 0;
                }

                g_craftax_mini_config_goal_block = goal_block;
                g_craftax_mini_max_timesteps = max_timesteps;
                g_craftax_mini_use_human_maps = true;
                craftax_set_reset_pool_size(0);

                Craftax env;
                memset(&env, 0, sizeof(env));
                env.num_agents = 1;
                env.rng = slot_id;
                env.seed = world_seed;
                c_init(&env);

                if (start_row >= 0 && start_col >= 0) {
                    env.state->player_position[0] = start_row;
                    env.state->player_position[1] = start_col;
                }

                int active_goal = craftax_mini_current_goal_block(&env);
                int count = 0;
                capture_frame(&env, active_goal, 0.0f, 0.0f, &frames[count++]);

                for (int i = 0; i < num_actions && count < max_frames; i++) {
                    int action = actions[i];
                    if (action < 0) action = 0;
                    if (action >= CRAFTAX_MINI_NUM_ACTIONS) {
                        action = CRAFTAX_MINI_NUM_ACTIONS - 1;
                    }

                    CraftaxThreefryKey step_key;
                    craftax_threefry_split(env.rng_key, &env.rng_key, &step_key);

                    CraftaxThreefryKey step_rng;
                    CraftaxThreefryKey reset_key;
                    craftax_threefry_split(step_key, &step_rng, &reset_key);
                    (void)reset_key;

                    float reward = craftax_mini_gameplay_step_native(&env, action, step_rng);
                    float done = craftax_mini_is_game_over_native(&env) ? 1.0f : 0.0f;
                    capture_frame(&env, active_goal, reward, done, &frames[count++]);
                    if (done > 0.0f) {
                        break;
                    }
                }

                c_close(&env);
                return count;
            }
            '''
        )
        subprocess.run(
            [
                os.environ.get("CC", "cc"),
                "-std=c99",
                "-O2",
                "-shared",
                "-fPIC",
                "-I",
                str(ROOT),
                "-I",
                str(ROOT / "raylib-5.5_macos" / "include"),
                str(src),
                "-lm",
                "-o",
                str(so),
            ],
            check=True,
            cwd=ROOT,
        )
        self.lib = ctypes.CDLL(str(so))
        self.lib.craftax_mini_replay_rollout.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(MiniReplayFrame),
            ctypes.c_int32,
        ]
        self.lib.craftax_mini_replay_rollout.restype = ctypes.c_int32

    def rollout(
        self,
        *,
        world_seed: int,
        slot_id: int,
        goal_block: int,
        max_timesteps: int,
        start_position: np.ndarray,
        actions: np.ndarray,
    ) -> list[MiniReplayFrame]:
        actions = np.asarray(actions, dtype=np.int32)
        max_frames = len(actions) + 1
        frame_array = (MiniReplayFrame * max_frames)()
        count = self.lib.craftax_mini_replay_rollout(
            ctypes.c_uint32(int(world_seed)),
            ctypes.c_uint32(int(slot_id)),
            ctypes.c_int32(int(goal_block)),
            ctypes.c_int32(int(max_timesteps)),
            ctypes.c_int32(int(start_position[0])),
            ctypes.c_int32(int(start_position[1])),
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(len(actions)),
            frame_array,
            ctypes.c_int32(max_frames),
        )
        return [frame_array[i] for i in range(count)]


def frame_image(frame: MiniReplayFrame, scale: int = 10) -> np.ndarray:
    map_level = np.frombuffer(frame.map, dtype=np.uint8).reshape(MAP_SIZE, MAP_SIZE)
    image = BLOCK_COLORS[np.clip(map_level, 0, len(BLOCK_COLORS) - 1)].copy()

    row, col = int(frame.player_position[0]), int(frame.player_position[1])
    if 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE:
        image[row, col] = np.asarray([255, 245, 40], dtype=np.uint8)
        for rr in range(max(0, row - 1), min(MAP_SIZE, row + 2)):
            for cc in range(max(0, col - 1), min(MAP_SIZE, col + 2)):
                if rr == row or cc == col:
                    image[rr, cc] = np.maximum(image[rr, cc], np.asarray([210, 210, 20], dtype=np.uint8))

    goal_block = int(frame.goal_block)
    goal_cells = np.argwhere(map_level == goal_block)
    for rr, cc in goal_cells:
        image[max(0, rr - 1):min(MAP_SIZE, rr + 2), cc] = np.asarray([255, 255, 255], dtype=np.uint8)
        image[rr, max(0, cc - 1):min(MAP_SIZE, cc + 2)] = np.asarray([255, 255, 255], dtype=np.uint8)
        image[rr, cc] = BLOCK_COLORS[goal_block]

    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def save_gif(frames: list[MiniReplayFrame], out_path: Path, fps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    image_artist = ax.imshow(frame_image(frames[0]), interpolation="nearest")
    title = ax.set_title("")

    goal_names = {block_id: name for block_id, name in GOAL_BLOCK_IDS.items()}

    def update(index: int):
        frame = frames[index]
        image_artist.set_data(frame_image(frame))
        pos = tuple(int(x) for x in frame.player_position)
        goal = goal_names.get(int(frame.goal_block), str(int(frame.goal_block)))
        title.set_text(
            f"frame={index} t={int(frame.timestep)} pos={pos} "
            f"goal={goal} reward={float(frame.reward):.1f} done={int(frame.done)}"
        )
        return image_artist, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_first_last_png(frames: list[MiniReplayFrame], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(min(5, len(frames))))
    tail_start = max(0, len(frames) - 5)
    indices += list(range(tail_start, len(frames)))

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for plot_idx, frame_idx in enumerate(indices[:10]):
        ax = axes.ravel()[plot_idx]
        frame = frames[frame_idx]
        ax.imshow(frame_image(frame, scale=6), interpolation="nearest")
        ax.set_title(
            f"frame {frame_idx}\nt={int(frame.timestep)} "
            f"r={float(frame.reward):.1f} d={int(frame.done)}",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def select_human_row(df, rng: np.random.Generator, split: str, row_index: int | None):
    sub = df.filter((df["success"] == 1.0))
    if split != "any":
        sub = sub.filter(df["eval"] == (split == "test"))
    rows = [row for row in sub.to_dicts() if is_native_bounds_compatible(row)]
    if not rows:
        raise RuntimeError(f"no successful native-bounds-compatible rows for split={split}")
    if row_index is None:
        return rows[int(rng.integers(len(rows)))]
    return rows[row_index % len(rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["human", "random"], default="human")
    parser.add_argument("--split", choices=["train", "test", "any"], default="any")
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--world", type=int, default=3, choices=[3, 15, 20, 95])
    parser.add_argument("--goal-block", type=int, default=10, choices=sorted(GOAL_BLOCK_IDS))
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    start_position = np.asarray([24, 24], dtype=np.int32)
    slot_id = int(args.seed)

    if args.policy == "human":
        pl = __import__("polars")
        df = pl.read_parquet(HUMAN_DATAFRAME_PATH)
        row = select_human_row(df, rng, args.split, args.row_index)
        positions = parse_positions_string(row["positions"])
        web_actions = parse_jax_array_string(row["actions"]).astype(np.int32)
        actions = WEB_TO_ENV_ACTION[web_actions]
        world = int(row["world"])
        goal_block = int(row["task_object_id"])
        start_position = positions[0]
        label = (
            f"human_{'test' if row['eval'] else 'train'}"
            f"_world{world}_goal{GOAL_BLOCK_IDS.get(goal_block, goal_block)}"
            f"_user{row['user_id']}_episode{row['episode_idx']}"
        )
    else:
        world = int(args.world)
        goal_block = int(args.goal_block)
        actions = rng.choice(MOVEMENT_ACTIONS, size=args.max_steps).astype(np.int32)
        label = f"random_world{world}_goal{GOAL_BLOCK_IDS[goal_block]}_seed{args.seed}"

    replay = NativeCraftaxMiniReplay()
    frames = replay.rollout(
        world_seed=world,
        slot_id=slot_id,
        goal_block=goal_block,
        max_timesteps=args.max_timesteps,
        start_position=start_position,
        actions=actions,
    )

    out_dir = Path(args.out_dir)
    gif_path = out_dir / f"craftax_mini_replay_{label}.gif"
    png_path = out_dir / f"craftax_mini_replay_{label}.png"
    save_gif(frames, gif_path, fps=args.fps)
    save_first_last_png(frames, png_path)

    print(
        f"policy={args.policy} world={world} goal={GOAL_BLOCK_IDS.get(goal_block, goal_block)} "
        f"frames={len(frames)} final_done={int(frames[-1].done)} "
        f"final_reward={float(frames[-1].reward):.1f}"
    )
    print(f"saved {gif_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
