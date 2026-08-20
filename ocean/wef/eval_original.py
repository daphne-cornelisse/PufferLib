#!/usr/bin/env python3
"""Eval converted Puffer torch policy on original WEF MultiAgentFishEnv (KempnerInstitute/wef).

Run from anywhere (paths are absolute-ish via env vars / defaults):

  WEF_ROOT=/home/daphne/code/wef \\
  /home/daphne/code/wef/.venv/bin/python ocean/wef/eval_original.py \\
      --ini logs/wef_nice/best_policy.ini --episodes 10

Compares:
  - random continuous actions (baseline)
  - converted PufferNet continuous policy
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PUFFER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEF_ROOT = Path(os.environ.get("WEF_ROOT", Path.home() / "code" / "wef"))

sys.path.insert(0, str(PUFFER_ROOT / "ocean" / "wef"))
from example import convert_from_ini, load_policy  # noqa: E402


def parse_ini(path: Path) -> dict:
    vals = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip()
    return vals


def make_fish_args(
    seed: int,
    episode_length: int,
    num_agents: int,
    *,
    arena_min: float,
    arena_max: float,
    prandom: float,
    urandom: float,
    reward_scale: str = "python",
) -> dict:
    """Build MultiAgentFishEnv kwargs.

    reward_scale:
      - "python": Kempner cfg defaults (eat=10, …) — matches original paper/MAPPO numbers
      - "c": ocean/wef/wef.h coeffs (eat=1, …) — matches Puffer training logs
    """
    if reward_scale == "c":
        rewards = {
            "timestep": 0.0,
            "eat": 1.0,
            "proximity_shaping": 0.1,
            "bitten": -0.5,
            "bite": -0.0001,
            "collision": -0.05,
            "effort_over": -0.01,
        }
    elif reward_scale == "python":
        rewards = {}  # use cfg.REWARDS (eat=10, proximity=1, …)
    else:
        raise ValueError(f"unknown reward_scale={reward_scale}")

    return {
        "num_agents": num_agents,
        "render_mode": None,  # set to "rgb_array" by callers that record video
        "feedback_action": True,
        "feedback_displacement": True,
        "pfeeder": 0,
        "prandom": prandom,
        "urandom": urandom,
        "prob_n_patch": 0,
        "save_vid": False,
        "allow_aggression": 1,
        "enable_bite_action": True,
        "use_bite_cooldown": True,
        "agent_size_mode": "random",
        "knollen_mode": 1,
        "ampullary_mode": 1,
        "mormyromast_mode": 1,
        "morm_selfimage_mode": 1,
        "morm_consimage_mode": 1,
        "sensing_model_type": "dynamic",
        "backwards": False,
        "max_episode_length": episode_length,
        "episode_length": episode_length,
        "homing_mode": False,
        "is_eval": True,
        "task": "foraging",
        "p_init_closeby": 0.0,
        "ampullary_intrinsic_only": True,
        "ampullary_ema": False,
        "max_food_sensing_radius": 15,
        "base_food_multiplier": 1.0,
        # Match C wef speeds: MAX_LINEAR 70cm/s / 83Hz with size exponent 1
        "multiplier_linear": 2.0,
        "multiplier_angular": 4.0,
        "size_speed_exponent": 1.0,
        "motion_order": 1,
        "seed": seed,
        "collective_sensing_mode": 1,
        "food_drift": 0.0,
        "food_drag": 0.0,
        "shared_reward": False,
        "dist_perturbation": 0.0,
        "mute_k": 0,
        "train_sensor_dropout_p": 0.0,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment_name": "sim2sim_puffer",
        "cfg_override": {
            "AGENT_PARAMS": {},
            "ENV_PARAMS": {
                "arena_size_min_cm": (arena_min, arena_min),
                "arena_size_max_cm": (arena_max, arena_max),
            },
            "REWARDS": rewards,
            "OBJECT_TYPES": {},
        },
    }


def patch_action_feedback_to_match_c():
    """C packs post-processed last_action without extra tanh; Python applies tanh.

    Monkeypatch EFishAgent.next_observation to skip tanh so obs layout matches C.
    """
    import MAEFish

    orig = MAEFish.EFishAgent.next_observation

    def next_observation(self, feedback_action=True, enable_bite_action=True,
                         use_bite_cooldown=True, agent_size_mode=None,
                         feedback_displacement=False):
        # Call original logic but replace tanh on actions with identity.
        # Cheapest: temporarily patch np.tanh for this call only if used on actions.
        # Cleaner: reimplement the packing matching C.
        action_observations = (
            self.last_action if feedback_action else [0] * self.num_actions
        )
        # C: already-bounded commands, no tanh
        bounded_action_observations = np.asarray(action_observations, dtype=np.float32)

        vel_obs = []
        if self.motion_order == 2:
            lin_norm = self.linear_velocity / self.max_linear_velocity
            ang_norm = self.angular_velocity / self.max_angular_velocity
            vel_obs = [lin_norm, ang_norm]

        _meta_lists = [
            list(self.knollen_metadata[i].values()) for i in self.knollen_metadata.keys()
        ]
        knollen_metadatas = (
            np.concatenate(_meta_lists) if _meta_lists else np.array([])
        )

        displacement_ego_obs = []
        if feedback_displacement:
            denom = self.max_linear_velocity if self.max_linear_velocity > 0 else 1.0
            displacement_ego_obs = np.clip(
                self.displacement_ego / denom, -1.0, 1.0
            ).flatten()

        observation = np.concatenate(
            [
                np.array(self.mormyromast_observations_virtual).flatten(),
                np.array(self.ampullary_observations).flatten(),
                np.array(self.knollen_observations).flatten(),
                np.array(knollen_metadatas).flatten(),
                np.array(bounded_action_observations).flatten(),
                [0.0],
                ([self.was_bitten] if enable_bite_action else []),
                [self.agent_size] if agent_size_mode is not None else [],
                (
                    [self.bite_cooldown_counter]
                    if (enable_bite_action and use_bite_cooldown)
                    else []
                ),
                displacement_ego_obs,
                ([self.eat_cooldown_counter]),
                vel_obs,
            ]
        )
        return observation

    MAEFish.EFishAgent.next_observation = next_observation
    return orig


def stack_obs(obs_list) -> np.ndarray:
    return np.stack([np.asarray(o, dtype=np.float32) for o in obs_list], axis=0)


def _food_this_step(env, info) -> float:
    """Count food consumed on this step (curr_food_consumed is per-step)."""
    total = 0.0
    # Prefer live agent state
    for a in getattr(env, "agent_objects", []):
        if hasattr(a, "curr_food_consumed"):
            total += float(len(a.curr_food_consumed))
    if total > 0:
        return total
    # Fallback: info dicts with food_eaten_count
    if info is None:
        return 0.0
    infos = info if isinstance(info, (list, tuple)) else [info]
    for agent_info in infos:
        if isinstance(agent_info, dict) and "food_eaten_count" in agent_info:
            total += float(agent_info["food_eaten_count"])
        elif isinstance(agent_info, dict):
            for v in agent_info.values():
                if isinstance(v, dict) and "food_eaten_count" in v:
                    total += float(v["food_eaten_count"])
    return total


def eval_random(env, episodes: int, seed: int, episode_length: int) -> dict:
    rng = np.random.default_rng(seed)
    returns = []
    per_agent = []
    foods = []
    for ep in range(episodes):
        obs = env.reset()
        ep_ret = 0.0
        food = 0.0
        for t in range(episode_length):
            acts = [
                rng.normal(0.0, 1.0, size=env.action_space[i].shape[0]).astype(np.float32)
                for i in range(env.num_agents)
            ]
            obs, rew, done, info = env.step(acts)
            ep_ret += float(np.sum(rew))
            food += _food_this_step(env, info)
            if isinstance(done, (list, tuple, np.ndarray)):
                if bool(np.all(done)):
                    break
            elif done:
                break
        returns.append(ep_ret)
        per_agent.append(ep_ret / max(env.num_agents, 1))
        foods.append(food)
    return {
        "n": len(returns),
        "episode_return_sum": float(np.mean(returns)),
        "episode_return_sum_std": float(np.std(returns)),
        "episode_return_per_agent": float(np.mean(per_agent)),
        "food_eaten_total": float(np.mean(foods)),
        "returns": returns,
    }


def eval_policy(env, policy, episodes: int, seed: int, episode_length: int) -> dict:
    returns = []
    per_agent = []
    foods = []
    for ep in range(episodes):
        obs_list = env.reset()
        obs = stack_obs(obs_list)
        assert obs.shape[1] == policy.obs_size, (
            f"obs dim {obs.shape[1]} != policy {policy.obs_size}"
        )
        state = policy.initial_state(env.num_agents).numpy()
        ep_ret = 0.0
        food = 0.0
        for t in range(episode_length):
            actions, _, state = policy.act_numpy(obs, state, deterministic=True)
            acts = [actions[i].astype(np.float32) for i in range(env.num_agents)]
            obs_list, rew, done, info = env.step(acts)
            ep_ret += float(np.sum(rew))
            food += _food_this_step(env, info)
            obs = stack_obs(obs_list)
            finished = False
            if isinstance(done, (list, tuple, np.ndarray)):
                finished = bool(np.all(done))
            else:
                finished = bool(done)
            if finished:
                state = policy.initial_state(env.num_agents).numpy()
                break
        returns.append(ep_ret)
        per_agent.append(ep_ret / max(env.num_agents, 1))
        foods.append(food)
    return {
        "n": len(returns),
        "episode_return_sum": float(np.mean(returns)),
        "episode_return_sum_std": float(np.std(returns)),
        "episode_return_per_agent": float(np.mean(per_agent)),
        "food_eaten_total": float(np.mean(foods)),
        "returns": returns,
    }


def _print_stats(label: str, stats: dict):
    print(f"=== {label} ===")
    print(
        f"  n={stats['n']}  return_sum={stats['episode_return_sum']:.2f} "
        f"± {stats['episode_return_sum_std']:.2f}  "
        f"return_per_agent={stats['episode_return_per_agent']:.2f}  "
        f"food_total={stats['food_eaten_total']:.2f}"
    )
    if "returns" in stats:
        print(f"  returns: {[round(r, 1) for r in stats['returns']]}")


def record_video(
    env,
    policy,
    out_path: Path,
    *,
    episode_length: int,
    seed: int,
    fps: int = 20,
    frames: int | None = None,
    auxs: list[str] | None = None,
) -> dict:
    """Roll out policy in original env and write mp4 via FishRenderer rgb_array."""
    import imageio

    if frames is None:
        frames = episode_length
    if auxs is None:
        auxs = ["eods"]  # arena + EOD subplot; skip heavy auxs for speed

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    env.render_mode = "rgb_array"
    env.save_vid = False  # we write our own path
    obs_list = env.reset()
    # Some MAEFish versions re-seed via env.seed
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass
    obs = stack_obs(obs_list)
    state = policy.initial_state(env.num_agents).numpy()
    ep_ret = 0.0
    food = 0.0
    n_written = 0

    writer = imageio.get_writer(
        out_path.as_posix(), format="ffmpeg", mode="I", fps=fps
    )
    try:
        # Initial frame
        frame = env.render(mode="rgb_array", auxs=auxs)
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        writer.append_data(np.asarray(frame, dtype=np.uint8))
        n_written += 1

        for t in range(frames - 1):
            actions, _, state = policy.act_numpy(obs, state, deterministic=True)
            acts = [actions[i].astype(np.float32) for i in range(env.num_agents)]
            obs_list, rew, done, info = env.step(acts)
            ep_ret += float(np.sum(rew))
            food += _food_this_step(env, info)
            obs = stack_obs(obs_list)

            frame = env.render(mode="rgb_array", auxs=auxs)
            if isinstance(frame, (list, tuple)):
                frame = frame[0]
            writer.append_data(np.asarray(frame, dtype=np.uint8))
            n_written += 1

            if t % 50 == 0:
                print(f"  video frame {n_written}/{frames}", flush=True)

            finished = False
            if isinstance(done, (list, tuple, np.ndarray)):
                finished = bool(np.all(done))
            else:
                finished = bool(done)
            if finished:
                state = policy.initial_state(env.num_agents).numpy()
                # keep recording if more frames requested; env auto-continues after done
    finally:
        writer.close()
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    size = out_path.stat().st_size if out_path.is_file() else 0
    print(f"  wrote {out_path}  ({size} bytes, {n_written} frames @ {fps} fps)")
    print(f"  episode_return_sum={ep_ret:.2f}  food_total={food:.1f}")
    return {"return_sum": ep_ret, "food_total": food, "frames": n_written, "path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wef-root", type=str, default=str(DEFAULT_WEF_ROOT))
    ap.add_argument("--ini", type=str, default=str(PUFFER_ROOT / "logs" / "wef" / "best_policy.ini"))
    ap.add_argument("--pt", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arena-min", type=float, default=70,
                    help="min arena edge cm (default 70 fixed; train used 30)")
    ap.add_argument("--arena-max", type=float, default=70,
                    help="max arena edge cm (default 70 fixed; train used 400)")
    ap.add_argument("--prandom", type=float, default=2.0 / 3.0,
                    help="patchy weight (original default 2/3; paper full run 5)")
    ap.add_argument("--urandom", type=float, default=1.0 / 3.0,
                    help="uniform weight (original default 1/3; paper full run 1)")
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--episode-length", type=int, default=512,
                    help="Puffer train uses 512; original full scripts use 1600")
    ap.add_argument("--reward-scale", choices=("python", "c"), default="python",
                    help="python=cfg eat=10 (paper); c=wef.h eat=1 (Puffer train logs)")
    ap.add_argument("--no-action-feedback-patch", action="store_true",
                    help="keep Python tanh on last_action (diverges from C)")
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--video", action="store_true",
                    help="record one episode mp4 in the original renderer")
    ap.add_argument(
        "--video-out",
        type=str,
        default=None,
        help="mp4 path (default logs/wef/videos/original_patchy_70.mp4)",
    )
    ap.add_argument("--video-frames", type=int, default=None,
                    help="frames to record (default = episode-length)")
    ap.add_argument("--video-fps", type=int, default=20)
    args = ap.parse_args()

    wef_fish = Path(args.wef_root) / "onpolicy" / "custom" / "fish"
    if not wef_fish.is_dir():
        raise SystemExit(f"WEF fish dir not found: {wef_fish}")
    sys.path.insert(0, str(wef_fish))

    if not args.no_action_feedback_patch:
        patch_action_feedback_to_match_c()
        print("patched action feedback to match C (no tanh on last_action)")

    from MAEFish import MultiAgentFishEnv  # noqa: E402

    ini = parse_ini(Path(args.ini))
    hidden = int(ini.get("hidden_size", 128))
    layers = int(ini.get("num_layers", 3))
    bin_path = ini["load_model_path"]
    if not Path(bin_path).is_absolute():
        bin_path = str(PUFFER_ROOT / bin_path)

    pt_path = args.pt
    if pt_path is None:
        pt_path = str(Path(bin_path).with_suffix(".pt"))
    if not Path(pt_path).exists():
        os.chdir(PUFFER_ROOT)
        convert_from_ini(args.ini, pt_path, continuous=True)

    if not Path(pt_path).is_absolute():
        pt_path = str(PUFFER_ROOT / pt_path)

    print(f"loading {pt_path}")
    print(
        f"protocol: agents={args.num_agents} T={args.episode_length} "
        f"arena=[{args.arena_min},{args.arena_max}] "
        f"prandom={args.prandom} urandom={args.urandom} "
        f"reward_scale={args.reward_scale}"
    )
    policy = load_policy(
        pt_path, obs_size=110, hidden_size=hidden, num_layers=layers,
        continuous=True, num_actions=4, device="cpu",
    )

    def mk(seed: int, render_mode=None):
        fa = make_fish_args(
            seed, args.episode_length, args.num_agents,
            arena_min=args.arena_min, arena_max=args.arena_max,
            prandom=args.prandom, urandom=args.urandom,
            reward_scale=args.reward_scale,
        )
        if render_mode is not None:
            fa["render_mode"] = render_mode
            fa["save_vid"] = False
        return MultiAgentFishEnv(fa, seed=seed, is_eval=True)

    if args.video:
        video_out = Path(
            args.video_out
            if args.video_out
            else PUFFER_ROOT / "logs" / "wef" / "videos" / "original_patchy_70.mp4"
        )
        if not video_out.is_absolute():
            video_out = PUFFER_ROOT / video_out
        print(f"=== recording original-sim video → {video_out} ===")
        env_v = mk(args.seed, render_mode="rgb_array")
        record_video(
            env_v,
            policy,
            video_out,
            episode_length=args.episode_length,
            seed=args.seed,
            fps=args.video_fps,
            frames=args.video_frames or args.episode_length,
        )
        return

    env = mk(args.seed)
    obs0 = env.reset()
    print(f"env obs_dim={obs0[0].shape[0]} act_dim={env.action_space[0].shape[0]}")

    if not args.skip_random:
        _print_stats("random baseline (original)", eval_random(
            env, args.episodes, args.seed, args.episode_length))

    env2 = mk(args.seed + 1)
    pstats = eval_policy(env2, policy, args.episodes, args.seed + 1, args.episode_length)
    _print_stats("puffer torch policy (original)", pstats)

    if "episode_return" in ini:
        print(f"(C training log episode_return={ini['episode_return']} with eat≈1 scale)")
    if args.reward_scale == "python":
        print("note: python reward scale uses eat=10; C train logs use eat=1 (~10× lower)")


if __name__ == "__main__":
    main()
