import os
import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.pushworld import binding


TILE_COLORS = np.array(
    [
        [20, 20, 20],   # empty
        [40, 40, 40],   # wall
        [180, 140, 40], # agent wall
        [90, 20, 20],   # goal
        [0, 220, 0],    # agent
        [70, 110, 255], # movable
        [220, 0, 0],    # goal object
    ],
    dtype=np.uint8,
)


class PushWorld(pufferlib.PufferEnv):
    def __init__(
        self,
        puzzle_dir="resources/pushworld/puzzles/train",
        vision=15,
        max_episode_length=500,
        horizon=None,
        levels=None,
        train_map=None,
        train_map_idx=-1,
        max_puzzles=0,
        count_based_reward_coef=0.0,
        count_based_global=True,
        num_envs=1024,
        report_interval=128,
        seed=0,
        render_mode="raylib",
        render_tile_size=16,
        render_full_map=False,
        buf=None,
    ):
        if vision <= 0:
            vision = 15
        if vision % 2 == 0:
            vision += 1  # Force odd window so the agent stays centered.
        if horizon is not None:
            max_episode_length = horizon
        self.vision = vision
        self.max_episode_length = max_episode_length
        self.obs_size = vision
        self.render_mode = render_mode
        self.render_tile_size = render_tile_size
        self.render_full_map = render_full_map
        self.num_agents = num_envs
        self.report_interval = report_interval
        if train_map is not None and isinstance(train_map, str) and train_map.strip() == "":
            train_map = None
        train_map_idx = int(train_map_idx) if train_map_idx is not None else -1

        # If a specific map is requested, ignore level filters to ensure it loads.
        if train_map is not None or train_map_idx >= 0:
            levels = None
        self.levels = levels
        self.max_puzzles = max_puzzles
        self.count_based_reward_coef = count_based_reward_coef
        self.count_based_global = count_based_global
        self.train_map = train_map
        self.train_map_idx = train_map_idx

        self.single_observation_space = gymnasium.spaces.Box(
            low=0,
            high=6,
            shape=(self.obs_size * self.obs_size,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Discrete(4)

        super().__init__(buf=buf)

        self.c_state = binding.shared(
            puzzle_dir=puzzle_dir,
            seed=seed,
            levels=levels,
            max_puzzles=max_puzzles,
        )

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            state=self.c_state,
            max_episode_length=max_episode_length,
            vision=vision,
            count_based_reward_coef=count_based_reward_coef,
            count_based_global=count_based_global,
        )
        if num_envs > 0:
            binding.vec_set_coverage_enabled(self.c_envs, 0, 1)

        if self.train_map is not None or self.train_map_idx >= 0:
            idx = self.train_map_idx
            if idx < 0:
                idx = _resolve_puzzle_index(
                    puzzle_dir,
                    self.train_map,
                    levels,
                    max_puzzles,
                )
            num_puzzles = binding.vec_num_puzzles(self.c_envs)
            if idx >= num_puzzles:
                raise ValueError(
                    f"train_map_idx {idx} out of range for {num_puzzles} puzzles"
                )
            binding.vec_set_puzzle_indices(self.c_envs, [idx] * num_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        self.tick += 1
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self, overlay=0):
        if self.render_mode == "rgb_array":
            if self.render_full_map:
                obs = binding.vec_full_map(self.c_envs)
            else:
                obs = self.observations[0].reshape(self.obs_size, self.obs_size)
            obs = np.clip(obs, 0, len(TILE_COLORS) - 1).astype(np.int32)
            tile_size = max(1, int(self.render_tile_size))
            img = TILE_COLORS[obs]
            if tile_size > 1:
                img = np.repeat(img, tile_size, axis=0)
                img = np.repeat(img, tile_size, axis=1)
            return img
        if self.render_mode == "raylib":
            binding.vec_render(self.c_envs)
        return None

    def close(self):
        pass


def _parse_level_id(name):
    return int(name[5:]) if name.startswith("level") else -1


def _collect_puzzle_paths(root_dir, levels, max_puzzles):
    if not os.path.isdir(root_dir):
        raise ValueError(f"Puzzle directory not found: {root_dir}")

    level_set = set(levels) if levels else None
    files = []
    for entry in os.listdir(root_dir):
        if entry in (".", ".."):
            continue
        level_path = os.path.join(root_dir, entry)
        if os.path.isdir(level_path) and entry.startswith("level"):
            level_id = _parse_level_id(entry)
            if level_set is not None and level_id not in level_set:
                continue
            for level_entry in os.listdir(level_path):
                if level_entry.endswith(".pwp"):
                    files.append((level_id, os.path.join(level_path, level_entry)))
        elif entry.endswith(".pwp"):
            level_id = _parse_level_id(os.path.basename(root_dir))
            if level_set is not None and level_id not in level_set:
                continue
            files.append((level_id, os.path.join(root_dir, entry)))

    if not files:
        raise ValueError(f"No puzzles found in {root_dir}")

    files.sort(key=lambda item: (item[0], item[1]))
    if max_puzzles and max_puzzles > 0:
        files = files[:max_puzzles]
    return [path for _, path in files]


def _resolve_puzzle_index(root_dir, train_map, levels, max_puzzles):
    paths = _collect_puzzle_paths(root_dir, levels, max_puzzles)
    train_map = str(train_map)
    candidates = {
        os.path.normpath(train_map),
        os.path.normpath(os.path.join(root_dir, train_map)),
    }
    for idx, path in enumerate(paths):
        norm = os.path.normpath(path)
        if norm in candidates or os.path.basename(norm) in candidates:
            return idx
    raise ValueError(f"train_map {train_map} not found in {root_dir}")
