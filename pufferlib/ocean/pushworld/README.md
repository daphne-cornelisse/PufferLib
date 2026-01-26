# PushWorld

Environment from: https://deepmind-pushworld.github.io/play/

Action Space
`Discrete(4)`

Observation Space
`Box(0, 6, (vision * vision,), float32)`

Import
```python
import pufferlib
from pufferlib.ocean import env_creator

env = env_creator("puffer_pushworld")()
```

Description
PushWorld is a grid-based pushing puzzle. Each level specifies:
- **A**: the agent (may be multi-cell shaped).
- **Gk**: goal region for object k (one or more cells).
- **Mk**: movable object k (one or more cells). Each goal **Gk** requires a corresponding **Mk**.
- **W**: wall (impassable).
- **AW**: agent-only wall (agent cannot enter; other objects can).
- **.**: empty cell.

The agent can move in four cardinal directions. If the agent attempts to move into
a space occupied by movable objects, it pushes them (recursively) if the chain of
objects can move without colliding with walls or bounds. The puzzle is solved when
every goal object is aligned with its corresponding goal region.

Observation Space
The observation is an egocentric, fixed-size window centered on the agent:
- Size is `vision x vision` (odd enforced).
- Out-of-bounds cells are treated as walls.
- Observation is flattened to shape `(vision * vision,)` and uses integer tile codes:

| Tile | Meaning        |
|------|----------------|
| 0    | Empty          |
| 1    | Wall           |
| 2    | Agent wall     |
| 3    | Goal region    |
| 4    | Agent          |
| 5    | Movable object |
| 6    | Goal object    |

Action Space
Discrete actions control movement of the agent:

| Num | Action |
|-----|--------|
| 0   | Left   |
| 1   | Right  |
| 2   | Up     |
| 3   | Down   |

Rewards
Let `cur_goals` be the number of goal objects aligned with their goal regions and
`prev_goals` the count before the action.
- **Solved**: reward = **+10.0**, episode terminates.
- **Otherwise**: reward = **(cur_goals - prev_goals) - 0.01**.

Starting State
The start state is fully determined by the `.pwp` puzzle file. The agent and
movable objects spawn at their specified initial positions.

Episode End
An episode ends if:
- **Termination**: all goals are satisfied (solved).
- **Truncation**: `tick >= max_episode_length`.

Arguments (PushWorld)
These are the env kwargs accepted by `PushWorld`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `puzzle_dir` | str | `resources/pushworld/puzzles/train` | Root directory containing puzzle files and/or `level*` subfolders. |
| `vision` | int | `25` | Egocentric observation size (odd enforced). |
| `max_episode_length` | int | `500` | Max steps per episode. |
| `horizon` | int or None | `None` | Alias for `max_episode_length` if provided. |
| `levels` | list[int] or None | `None` | If set, only load puzzles from these level ids. |
| `train_map` | str or None | `None` | Specific puzzle filename or relative path; overrides `levels`. |
| `train_map_idx` | int | `-1` | Index into sorted puzzle list; overrides `train_map` if >= 0. |
| `max_puzzles` | int | `0` | Cap number of puzzles loaded (0 = no cap). |
| `count_based_reward_coef` | float | `0.0` | Count-based intrinsic reward coefficient. |
| `count_based_global` | bool | `True` | If `True`, counts persist across episodes; else reset each episode. |
| `num_envs` | int | `1024` | Number of parallel agents in this native env instance. |
| `report_interval` | int | `128` | Steps between aggregated log reports. |
| `seed` | int | `0` | RNG seed for puzzle selection / reset. |
| `render_mode` | str | `"raylib"` | `"raylib"` or `"rgb_array"` or `"None"`. |
| `render_tile_size` | int | `16` | Pixel size of tiles for `rgb_array` renders. |
| `render_full_map` | bool | `False` | If `True`, renders the full map instead of egocentric view. |

PWP Puzzle Format
Puzzle files are whitespace-separated grids. Example:
```
.  .  .  .  .
.  A  . M0  .
.  .  .  .  .
.  . G0  .  .
.  .  .  .  .
```

Rules:
- Each cell is a token; tokens can be combined using `+` (e.g., `G0+M0`).
- `A` is required (agent).
- Each `Gk` must have a matching `Mk`.
- Multiple cells with the same token form a shape (agent and objects can be multi-cell).
- The loader automatically adds a 1-cell wall border around the map.

Render
`render_mode="rgb_array"` returns an `H x W x 3` uint8 image, optionally full-map.

W&B Metrics
These are the env-specific charts you’ll see in W&B that aren’t obvious at first glance:

- `environment/no_op_rate`  
  Fraction of steps where the agent’s action resulted in no movement.

- `environment/map_eval_solved_rate`  
  Periodic evaluation over all training puzzles; fraction solved.

- `environment/map_eval_test_solved_rate`  
  Same as above, but for the held-out test puzzle directory if provided.

- `environment/cumulative_solved`  
  Cumulative count of solved episodes (from env logs), aggregated over training.

- `environment/coverage_heatmap`  
  Heatmap of **goal-object** occupancy for a fixed map (live training coverage, env 0).

- `environment/coverage_agent_heatmap`  
  Heatmap of **agent** occupancy for the same fixed map.

Notes
- The goal check is anchor-based: each goal object is considered solved when its
  anchor aligns with the goal’s anchor (top-left of the goal region).
