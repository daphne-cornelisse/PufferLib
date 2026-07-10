"""Monte Carlo random-policy sanity checks for Craftax Mini.

Runs random rollouts on the 4 vendored human maps for each goal object, estimates
collection probability, and plots trajectory-density overlays.

Example:
    uv run --with matplotlib python ocean/craftax_mini/monte_carlo_simulations.py --rollouts 10
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
    HUMAN_PLOT_DIR,
    MAP_SIZE,
    load_human_experiment_maps,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = HUMAN_PLOT_DIR / "monte_carlo"
HUMAN_WORLDS = (3, 15, 20, 95)
GOAL_BLOCKS = tuple(sorted(GOAL_BLOCK_IDS))
ACTION_NAMES = ("right", "down", "left", "up", "do")


class MiniMonteCarloResult(ctypes.Structure):
    _fields_ = [
        ("success", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("final_row", ctypes.c_int32),
        ("final_col", ctypes.c_int32),
        ("reward_sum", ctypes.c_float),
    ]


class NativeCraftaxMiniMonteCarlo:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        src = tmp / "craftax_mini_monte_carlo.c"
        so = tmp / "craftax_mini_monte_carlo.so"
        src.write_text(
            r'''
            #include <stdint.h>
            #include <string.h>
            #include "ocean/craftax_mini/craftax_mini.h"

            typedef struct MiniMonteCarloResult {
                int32_t success;
                int32_t steps;
                int32_t final_row;
                int32_t final_col;
                float reward_sum;
            } MiniMonteCarloResult;

            void craftax_mini_random_rollout(
                uint32_t world_seed,
                uint32_t slot_id,
                int32_t goal_block,
                int32_t max_timesteps,
                const int32_t* actions,
                int32_t num_actions,
                int32_t* positions_out,
                MiniMonteCarloResult* result
            ) {
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

                int32_t steps = 0;
                float reward_sum = 0.0f;
                int32_t success = 0;
                positions_out[0] = env.state->player_position[0];
                positions_out[1] = env.state->player_position[1];

                for (int32_t i = 0; i < num_actions; i++) {
                    int32_t action = actions[i];
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

                    float reward = craftax_mini_gameplay_step(&env, action, step_rng);
                    reward_sum += reward;
                    steps += 1;

                    positions_out[2 * steps] = env.state->player_position[0];
                    positions_out[2 * steps + 1] = env.state->player_position[1];

                    if (reward > 0.0f) {
                        success = 1;
                    }
                    if (craftax_mini_is_game_over(&env)) {
                        break;
                    }
                }

                result->success = success;
                result->steps = steps;
                result->final_row = env.state->player_position[0];
                result->final_col = env.state->player_position[1];
                result->reward_sum = reward_sum;
                c_close(&env);
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
        self.lib.craftax_mini_random_rollout.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(MiniMonteCarloResult),
        ]
        self.lib.craftax_mini_random_rollout.restype = None

    def rollout(
        self,
        *,
        world_seed: int,
        slot_id: int,
        goal_block: int,
        max_timesteps: int,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, MiniMonteCarloResult]:
        actions = np.asarray(actions, dtype=np.int32)
        positions = np.empty((len(actions) + 1, 2), dtype=np.int32)
        result = MiniMonteCarloResult()
        self.lib.craftax_mini_random_rollout(
            ctypes.c_uint32(int(world_seed)),
            ctypes.c_uint32(int(slot_id)),
            ctypes.c_int32(int(goal_block)),
            ctypes.c_int32(int(max_timesteps)),
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(len(actions)),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(result),
        )
        return positions[: result.steps + 1].copy(), result


def make_actions(rng: np.random.Generator, n: int, include_do: bool) -> np.ndarray:
    high = 5 if include_do else 4
    return rng.integers(0, high, size=n, dtype=np.int32)


def add_density(density: np.ndarray, positions: np.ndarray) -> None:
    in_bounds = positions[
        (positions[:, 0] >= 0)
        & (positions[:, 0] < MAP_SIZE)
        & (positions[:, 1] >= 0)
        & (positions[:, 1] < MAP_SIZE)
    ]
    np.add.at(density, (in_bounds[:, 0], in_bounds[:, 1]), 1)


def render_map(map_level: np.ndarray) -> np.ndarray:
    return BLOCK_COLORS[np.clip(map_level, 0, len(BLOCK_COLORS) - 1)]


def write_summary(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "world",
        "goal",
        "rollouts",
        "successes",
        "success_probability",
        "mean_steps",
        "median_steps",
    ]
    widths = {key: max(len(key), 10) for key in headers}
    for row in rows:
        for key in headers:
            widths[key] = max(widths[key], len(str(row[key])))

    with out_path.open("w") as f:
        f.write("Craftax Mini random-policy Monte Carlo\n")
        f.write("\n")
        f.write(" ".join(key.ljust(widths[key]) for key in headers))
        f.write("\n")
        for row in rows:
            f.write(" ".join(str(row[key]).ljust(widths[key]) for key in headers))
            f.write("\n")


def draw_landmarks(ax, map_level: np.ndarray, goal_block: int) -> None:
    start = np.asarray([24, 24])
    ax.scatter(start[1], start[0], c="white", s=52, edgecolors="black", marker="*", label="start")
    for block_id, name in GOAL_BLOCK_IDS.items():
        cells = np.argwhere(map_level == block_id)
        if len(cells) == 0:
            continue
        is_goal = block_id == goal_block
        ax.scatter(
            cells[:, 1],
            cells[:, 0],
            s=70 if is_goal else 38,
            facecolors="none",
            edgecolors="yellow" if is_goal else BLOCK_COLORS[block_id] / 255.0,
            linewidths=2.4 if is_goal else 1.3,
            label=f"{name}{' goal' if is_goal else ''}",
        )


def style_map_axis(ax) -> None:
    ax.set_xlim(-0.5, MAP_SIZE - 0.5)
    ax.set_ylim(MAP_SIZE - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def positive_density_values(densities: list[np.ndarray]) -> np.ndarray:
    values = [density[density > 0].ravel() for density in densities if np.any(density > 0)]
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(values)


def density_log_norm(densities: list[np.ndarray]):
    from matplotlib.colors import LogNorm

    positive = positive_density_values(densities)
    if len(positive) == 0:
        return LogNorm(vmin=1.0, vmax=1.0)
    return LogNorm(vmin=1.0, vmax=max(float(np.percentile(positive, 99.5)), 1.0))


def plot_goal_density_panels(
    out_path: Path,
    maps: dict[int, np.ndarray],
    densities: dict[int, dict[int, np.ndarray]],
    goal_block: int,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(HUMAN_WORLDS),
        2,
        figsize=(10, 4.2 * len(HUMAN_WORLDS)),
        squeeze=False,
        width_ratios=(1, 1.08),
        constrained_layout=True,
    )
    norm = density_log_norm([densities[goal_block][world] for world in HUMAN_WORLDS])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=(0.03, 0.03, 0.04, 1.0))
    last_heatmap = None

    for row_idx, world in enumerate(HUMAN_WORLDS):
        map_level = maps[world][0]
        density = densities[goal_block][world]

        map_ax = axes[row_idx, 0]
        map_ax.imshow(render_map(map_level), interpolation="nearest")
        draw_landmarks(map_ax, map_level, goal_block)
        style_map_axis(map_ax)
        map_ax.set_title(f"world {world}: map")

        density_ax = axes[row_idx, 1]
        masked = np.ma.masked_where(density <= 0, density)
        last_heatmap = density_ax.imshow(
            masked,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        draw_landmarks(density_ax, map_level, goal_block)
        style_map_axis(density_ax)
        density_ax.set_title(f"world {world}: random-walk visits")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    if last_heatmap is not None:
        cbar = fig.colorbar(last_heatmap, ax=axes[:, 1], fraction=0.025, pad=0.02)
        cbar.set_label("visits per cell (log scale; dark = 0 visits)")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=int, default=10, help="Rollouts per world-goal pair.")
    parser.add_argument("--max-steps", type=int, default=300, help="Actions per rollout.")
    parser.add_argument("--max-timesteps", type=int, default=300, help="Craftax Mini episode timeout.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--movement-only",
        action="store_true",
        help="Sample only movement actions. By default random actions include DO.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    native = NativeCraftaxMiniMonteCarlo()
    maps = load_human_experiment_maps()
    densities = {
        goal_block: {
            world: np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float64)
            for world in HUMAN_WORLDS
        }
        for goal_block in GOAL_BLOCKS
    }
    rows: list[dict] = []
    aggregate = {goal_block: {"successes": 0, "rollouts": 0, "steps": []} for goal_block in GOAL_BLOCKS}

    for world in HUMAN_WORLDS:
        for goal_block in GOAL_BLOCKS:
            successes = 0
            steps: list[int] = []
            for rollout_idx in range(args.rollouts):
                actions = make_actions(rng, args.max_steps, include_do=not args.movement_only)
                slot_id = (
                    world * 1_000_003
                    + goal_block * 10_007
                    + rollout_idx
                    + args.seed * 97
                ) & 0xFFFFFFFF
                positions, result = native.rollout(
                    world_seed=world,
                    slot_id=slot_id,
                    goal_block=goal_block,
                    max_timesteps=args.max_timesteps,
                    actions=actions,
                )
                add_density(densities[goal_block][world], positions)
                successes += int(result.success)
                steps.append(int(result.steps))
                aggregate[goal_block]["successes"] += int(result.success)
                aggregate[goal_block]["rollouts"] += 1
                aggregate[goal_block]["steps"].append(int(result.steps))

            probability = successes / max(args.rollouts, 1)
            rows.append(
                {
                    "world": world,
                    "goal": GOAL_BLOCK_IDS[goal_block],
                    "rollouts": args.rollouts,
                    "successes": successes,
                    "success_probability": f"{probability:.4f}",
                    "mean_steps": f"{float(np.mean(steps)):.1f}",
                    "median_steps": f"{float(np.median(steps)):.1f}",
                }
            )

    for goal_block in GOAL_BLOCKS:
        item = aggregate[goal_block]
        probability = item["successes"] / max(item["rollouts"], 1)
        rows.append(
            {
                "world": "all",
                "goal": GOAL_BLOCK_IDS[goal_block],
                "rollouts": item["rollouts"],
                "successes": item["successes"],
                "success_probability": f"{probability:.4f}",
                "mean_steps": f"{float(np.mean(item['steps'])):.1f}",
                "median_steps": f"{float(np.median(item['steps'])):.1f}",
            }
        )

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "craftax_mini_random_policy_summary.txt"
    write_summary(summary_path, rows)
    plot_paths = []
    for goal_block in GOAL_BLOCKS:
        goal_name = GOAL_BLOCK_IDS[goal_block]
        plot_path = out_dir / f"craftax_mini_random_policy_density_{goal_name}.png"
        plot_goal_density_panels(
            plot_path,
            maps,
            densities,
            goal_block,
            title=(
                f"Random-policy density for {goal_name}, "
                f"{args.rollouts} rollouts per world, "
                f"actions={'movement only' if args.movement_only else 'movement+do'}"
            ),
        )
        plot_paths.append(plot_path)

    for row in rows:
        print(
            f"world={row['world']} goal={row['goal']} "
            f"successes={row['successes']}/{row['rollouts']} "
            f"p={row['success_probability']} mean_steps={row['mean_steps']}"
        )
    print(f"saved {summary_path}")
    for plot_path in plot_paths:
        print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
