#!/usr/bin/env python3
"""Roll out Puffer + original MAPPO + random in original 70×70; plot stats.

Panels:
  - episode return (strip + mean ± 95% CI) as % of max obtainable
  - mean ± 95% CI for food eaten, bites, collisions, EOD rate, was-bitten

Original MAPPO checkpoint (default):
  checkpoints/wef/original/actor.pt
  (Dyn_F00_Kb_For_S1/models from the WEF release Drive)

Example:
  /path/to/wef/.venv/bin/python ocean/wef/plot_original_returns.py \\
      --ini logs/wef/best_policy.ini --episodes 20
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

PUFFER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEF_ROOT = Path(os.environ.get("WEF_ROOT", Path.home() / "code" / "wef"))
DEFAULT_ORIGINAL_MODEL_DIR = PUFFER_ROOT / "checkpoints" / "wef" / "original"

sys.path.insert(0, str(PUFFER_ROOT / "ocean" / "wef"))
from eval_original import (  # noqa: E402
    _food_this_step,
    make_fish_args,
    parse_ini,
    patch_action_feedback_to_match_c,
    stack_obs,
)
from example import convert_from_ini, load_policy  # noqa: E402


# Episode-level series keys plotted with mean ± 95% CI
STAT_SPECS = [
    ("returns", "episode return\n(sum over agents)", "{:.0f}"),
    ("food", "food eaten\n(total)", "{:.1f}"),
    ("bites", "bites\n(count)", "{:.1f}"),
    ("collisions", "collisions\n(count)", "{:.1f}"),
    ("eod_rate", "EOD rate\n(fraction steps)", "{:.2f}"),
    ("was_bitten", "was-bitten\n(count)", "{:.1f}"),
]


def _infos(info):
    if info is None:
        return []
    if isinstance(info, (list, tuple)):
        return [i for i in info if isinstance(i, dict)]
    if isinstance(info, dict):
        return [info]
    return []


def _accumulate_step(env, info, tallies: dict):
    """Add this step's events into episode tallies (sums over all agents)."""
    tallies["food"] += _food_this_step(env, info)
    n_agents = 0
    eod_on = 0
    for inf in _infos(info):
        n_agents += 1
        if bool(inf.get("bite_other_fish", False)):
            tallies["bites"] += 1
        if bool(inf.get("collided", False)):
            tallies["collisions"] += 1
        if bool(inf.get("was_bitten", False)):
            tallies["was_bitten"] += 1
        if bool(inf.get("emit_eod", False)):
            eod_on += 1
        # also count from post-activation if present
        acts = inf.get("actions_postactivation")
        if acts is not None and len(acts) >= 3 and not bool(inf.get("emit_eod", False)):
            # prefer emit_eod; only fall through if missing
            pass
    tallies["eod_on"] += eod_on
    tallies["agent_steps"] += max(n_agents, 1)


def _empty_tally():
    return {
        "food": 0.0,
        "bites": 0,
        "collisions": 0,
        "was_bitten": 0,
        "eod_on": 0,
        "agent_steps": 0,
    }


def _eat_reward(env) -> float:
    """Positive food-eat coefficient (python cfg default 10, C-scale 1)."""
    rp = getattr(env, "reward_params", None) or {}
    if isinstance(rp, dict) and "eat" in rp:
        return float(rp["eat"])
    return 10.0


def _n_food_available(env) -> int:
    """Food pellets present at episode start (= max obtainable if all eaten)."""
    arena = getattr(env, "arena", None)
    if arena is not None and hasattr(arena, "num_food_pellets"):
        return int(arena.num_food_pellets)
    if arena is not None and hasattr(arena, "food_pellets"):
        return int(len(arena.food_pellets))
    return 0


def max_return_per_agent(env) -> float:
    """Maximum obtainable episode return / num fish.

    All initial food pellets eaten, no penalties, food reward only:
      (n_food * eat_reward) / num_agents
    """
    n_food = _n_food_available(env)
    eat = _eat_reward(env)
    n_agents = max(int(env.num_agents), 1)
    return (n_food * eat) / n_agents


def _finalize_episode(
    ep_ret: float, n_agents: int, tallies: dict, max_pa: float
) -> dict:
    agent_steps = max(tallies["agent_steps"], 1)
    ret_pa = float(ep_ret) / max(n_agents, 1)
    return {
        "return": float(ep_ret),
        "return_per_agent": ret_pa,
        "max_return_per_agent": float(max_pa),
        "return_pct_max": (100.0 * ret_pa / max_pa) if max_pa > 0 else 0.0,
        "food": float(tallies["food"]),
        "bites": float(tallies["bites"]),
        "collisions": float(tallies["collisions"]),
        "was_bitten": float(tallies["was_bitten"]),
        "eod_rate": float(tallies["eod_on"]) / agent_steps,
    }


def eval_policy_rich(env, policy, episodes: int, seed: int, episode_length: int) -> dict:
    series = {k: [] for k, _, _ in STAT_SPECS}
    series["return_per_agent"] = []
    series["max_return_per_agent"] = []
    series["return_pct_max"] = []
    for ep in range(episodes):
        obs_list = env.reset()
        max_pa = max_return_per_agent(env)
        obs = stack_obs(obs_list)
        state = policy.initial_state(env.num_agents).numpy()
        ep_ret = 0.0
        tallies = _empty_tally()
        for _t in range(episode_length):
            actions, _, state = policy.act_numpy(obs, state, deterministic=True)
            acts = [actions[i].astype(np.float32) for i in range(env.num_agents)]
            obs_list, rew, done, info = env.step(acts)
            ep_ret += float(np.sum(rew))
            _accumulate_step(env, info, tallies)
            obs = stack_obs(obs_list)
            finished = (
                bool(np.all(done))
                if isinstance(done, (list, tuple, np.ndarray))
                else bool(done)
            )
            if finished:
                state = policy.initial_state(env.num_agents).numpy()
                break
        ep = _finalize_episode(ep_ret, env.num_agents, tallies, max_pa)
        series["returns"].append(ep["return"])
        series["return_per_agent"].append(ep["return_per_agent"])
        series["max_return_per_agent"].append(ep["max_return_per_agent"])
        series["return_pct_max"].append(ep["return_pct_max"])
        for k in ("food", "bites", "collisions", "eod_rate", "was_bitten"):
            series[k].append(ep[k])
    return _pack_series(series)


def eval_random_rich(env, episodes: int, seed: int, episode_length: int) -> dict:
    rng = np.random.default_rng(seed)
    series = {k: [] for k, _, _ in STAT_SPECS}
    series["return_per_agent"] = []
    series["max_return_per_agent"] = []
    series["return_pct_max"] = []
    for _ep in range(episodes):
        env.reset()
        max_pa = max_return_per_agent(env)
        ep_ret = 0.0
        tallies = _empty_tally()
        for _t in range(episode_length):
            acts = [
                rng.normal(0.0, 1.0, size=env.action_space[i].shape[0]).astype(np.float32)
                for i in range(env.num_agents)
            ]
            _obs, rew, done, info = env.step(acts)
            ep_ret += float(np.sum(rew))
            _accumulate_step(env, info, tallies)
            if isinstance(done, (list, tuple, np.ndarray)):
                if bool(np.all(done)):
                    break
            elif done:
                break
        ep = _finalize_episode(ep_ret, env.num_agents, tallies, max_pa)
        series["returns"].append(ep["return"])
        series["return_per_agent"].append(ep["return_per_agent"])
        series["max_return_per_agent"].append(ep["max_return_per_agent"])
        series["return_pct_max"].append(ep["return_pct_max"])
        for k in ("food", "bites", "collisions", "eod_rate", "was_bitten"):
            series[k].append(ep[k])
    return _pack_series(series)


def _pack_series(series: dict) -> dict:
    out = {"n": len(series["returns"]), "series": series}
    for k, vals in series.items():
        a = np.asarray(vals, dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(a))
        out[f"{k}_std"] = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        out[k] = list(a)
    # aliases used by print / older callers
    out["returns"] = series["returns"]
    out["episode_return_sum"] = out["returns_mean"]
    out["episode_return_sum_std"] = out["returns_std"]
    out["episode_return_per_agent"] = out["return_per_agent_mean"]
    return out


def mean_ci95(vals) -> tuple[float, float, float]:
    """Return (mean, ci_half_width, n) for a 95% Student-t CI of the mean."""
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0
    mean = float(np.mean(a))
    if n == 1:
        return mean, 0.0, 1
    se = float(np.std(a, ddof=1) / np.sqrt(n))
    tcrit = float(sp_stats.t.ppf(0.975, n - 1))
    return mean, tcrit * se, n


def try_load_original_policy(model_dir: Path, env, device: str = "cpu"):
    """Load shared MAPPO R_Actor from original WEF models/ dir (actor.pt).

    Dyn_F00 4-fish actor: obs=110, hidden=512, GRU, continuous act=4.
    """
    model_dir = Path(model_dir)
    candidates = [model_dir / "actor.pt", model_dir / "actor_agent0.pt"]
    actor_path = next((p for p in candidates if p.is_file()), None)
    if actor_path is None:
        print(f"no actor.pt under {model_dir}", file=sys.stderr)
        return None

    wef_root = Path(os.environ.get("WEF_ROOT", DEFAULT_WEF_ROOT))
    if str(wef_root) not in sys.path:
        sys.path.insert(0, str(wef_root))
    # Package __init__ imports missing onpolicy.scripts — stub it.
    sys.modules.setdefault("onpolicy.scripts", types.ModuleType("onpolicy.scripts"))

    try:
        import torch
        from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
        from onpolicy.config import get_config
    except Exception as e:
        print(f"cannot import original policy stack: {e}", file=sys.stderr)
        return None

    try:
        parser = get_config()
        all_args, _ = parser.parse_known_args([])
    except Exception:
        all_args = argparse.Namespace()

    for k, v in dict(
        hidden_size=512,
        use_orthogonal=True,
        gain=0.01,
        use_policy_active_masks=False,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=True,
        recurrent_N=1,
        data_chunk_length=10,
        use_feature_normalization=True,
        use_ReLU=True,
        stacked_frames=1,
        layer_N=1,
        use_centralized_V=True,
        algorithm_name="rmappo",
        attn_mode=None,
        rnn_type="gru",
    ).items():
        setattr(all_args, k, v)

    try:
        import torch

        actor = R_Actor(
            all_args,
            env.observation_space[0],
            env.action_space[0],
            device=torch.device(device),
        )
        state = torch.load(actor_path, map_location=device, weights_only=False)
        actor.load_state_dict(state, strict=True)
        actor.eval()
    except Exception as e:
        print(f"failed to construct/load original actor: {e}", file=sys.stderr)
        return None

    rnn_states = None
    masks = None

    def act(obs_list, deterministic=True):
        nonlocal rnn_states, masks
        import torch

        obs = np.stack([np.asarray(o, dtype=np.float32) for o in obs_list], axis=0)
        n = obs.shape[0]
        if rnn_states is None:
            h = int(getattr(all_args, "hidden_size", 512))
            rn = int(getattr(all_args, "recurrent_N", 1))
            rnn_states = torch.zeros(n, rn, h)
            masks = torch.ones(n, 1)
        with torch.no_grad():
            actions, _logp, rnn_states = actor(
                torch.as_tensor(obs),
                rnn_states,
                masks,
                deterministic=deterministic,
            )
        a = actions.cpu().numpy()
        return [a[i].astype(np.float32) for i in range(n)]

    def reset_state():
        nonlocal rnn_states, masks
        rnn_states = None
        masks = None

    act.reset_state = reset_state
    print(f"loaded original MAPPO actor from {actor_path}")
    return act


def eval_callable_rich(
    env, act_fn, episodes: int, episode_length: int, seed: int
) -> dict:
    """Roll out act(obs_list)->actions with the same rich episode stats."""
    series = {k: [] for k, _, _ in STAT_SPECS}
    series["return_per_agent"] = []
    series["max_return_per_agent"] = []
    series["return_pct_max"] = []
    for ep in range(episodes):
        if hasattr(act_fn, "reset_state"):
            act_fn.reset_state()
        obs_list = env.reset()
        if hasattr(env, "seed"):
            try:
                env.seed(seed + ep * 17)
            except Exception:
                pass
            obs_list = env.reset()
        max_pa = max_return_per_agent(env)
        ep_ret = 0.0
        tallies = _empty_tally()
        for _t in range(episode_length):
            acts = act_fn(obs_list, deterministic=True)
            obs_list, rew, done, info = env.step(acts)
            ep_ret += float(np.sum(rew))
            _accumulate_step(env, info, tallies)
            finished = (
                bool(np.all(done))
                if isinstance(done, (list, tuple, np.ndarray))
                else bool(done)
            )
            if finished:
                if hasattr(act_fn, "reset_state"):
                    act_fn.reset_state()
                break
        ep_stats = _finalize_episode(ep_ret, env.num_agents, tallies, max_pa)
        series["returns"].append(ep_stats["return"])
        series["return_per_agent"].append(ep_stats["return_per_agent"])
        series["max_return_per_agent"].append(ep_stats["max_return_per_agent"])
        series["return_pct_max"].append(ep_stats["return_pct_max"])
        for k in ("food", "bites", "collisions", "eod_rate", "was_bitten"):
            series[k].append(ep_stats[k])
    return _pack_series(series)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wef-root", type=str, default=str(DEFAULT_WEF_ROOT))
    ap.add_argument(
        "--ini", type=str, default=str(PUFFER_ROOT / "logs" / "wef" / "best_policy.ini")
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--episode-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arena", type=float, default=70.0)
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--reward-scale", choices=("python", "c"), default="python")
    ap.add_argument(
        "--original-model-dir",
        type=str,
        default=os.environ.get(
            "ORIGINAL_MODEL_DIR",
            str(DEFAULT_ORIGINAL_MODEL_DIR),
        ),
        help="dir with original MAPPO actor.pt (default checkpoints/wef/original)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(PUFFER_ROOT / "logs" / "wef" / "original_return_compare.pdf"),
    )
    ap.add_argument(
        "--puffer-label",
        type=str,
        default="Puffer",
        help="legend/x-tick label for the Puffer policy",
    )
    args = ap.parse_args()

    wef_root = Path(args.wef_root)
    wef_fish = wef_root / "onpolicy" / "custom" / "fish"
    sys.path.insert(0, str(wef_root))
    sys.path.insert(0, str(wef_fish))
    sys.modules.setdefault("onpolicy.scripts", types.ModuleType("onpolicy.scripts"))
    patch_action_feedback_to_match_c()
    from MAEFish import MultiAgentFishEnv  # noqa: E402

    ini = parse_ini(Path(args.ini))
    hidden = int(ini.get("hidden_size", 128))
    layers = int(ini.get("num_layers", 3))
    bin_path = ini["load_model_path"]
    if not Path(bin_path).is_absolute():
        bin_path = str(PUFFER_ROOT / bin_path)
    pt_path = str(Path(bin_path).with_suffix(".pt"))
    if not Path(pt_path).exists():
        convert_from_ini(args.ini, pt_path, continuous=True)
    if not Path(pt_path).is_absolute():
        pt_path = str(PUFFER_ROOT / pt_path)

    policy = load_policy(
        pt_path,
        obs_size=110,
        hidden_size=hidden,
        num_layers=layers,
        continuous=True,
        num_actions=4,
        device="cpu",
    )
    puffer_label = args.puffer_label
    print(f"Puffer policy ({puffer_label}): {pt_path}  H={hidden} L={layers}")

    def mk(seed: int):
        return MultiAgentFishEnv(
            make_fish_args(
                seed,
                args.episode_length,
                args.num_agents,
                arena_min=args.arena,
                arena_max=args.arena,
                prandom=1.0,
                urandom=0.0,
                reward_scale=args.reward_scale,
            ),
            seed=seed,
            is_eval=True,
        )

    results = {}

    print(
        f"=== {puffer_label} ({args.episodes} eps, {args.arena}cm, T={args.episode_length}) ==="
    )
    env_p = mk(args.seed)
    pstats = eval_policy_rich(
        env_p, policy, args.episodes, args.seed, args.episode_length
    )
    results[puffer_label] = pstats

    print("=== reference ===")
    env_o = mk(args.seed + 2000)
    oact = try_load_original_policy(Path(args.original_model_dir), env_o)
    if oact is not None:
        ostats = eval_callable_rich(
            env_o, oact, args.episodes, args.episode_length, args.seed + 2000
        )
        results["Reference Python"] = ostats
    else:
        print(
            f"Could not load original actor from {args.original_model_dir}; "
            "plotting without it.",
            file=sys.stderr,
        )

    print("=== random baseline ===")
    env_r = mk(args.seed + 1000)
    rstats = eval_random_rich(
        env_r, args.episodes, args.seed + 1000, args.episode_length
    )
    results["Random"] = rstats

    order = list(results.keys())
    for name in order:
        s = results[name]
        m, half, n = mean_ci95(s["returns"])
        pct_m, pct_h, _ = mean_ci95(s["return_pct_max"])
        print(
            f"  [{name}] return_sum={m:.1f} ± {half:.1f}  "
            f"per_agent={s['return_per_agent_mean']:.1f}  "
            f"%max={pct_m:.1f}±{pct_h:.1f}%  "
            f"max_pa={s['max_return_per_agent_mean']:.1f}  "
            f"food={s['food_mean']:.1f}  bites={s['bites_mean']:.1f}  "
            f"coll={s['collisions_mean']:.1f}  eod_rate={s['eod_rate_mean']:.3f}  "
            f"bitten={s['was_bitten_mean']:.1f}  n={n}"
        )

    # ---- plot ----
    # Keep Puffer / π* distinctly green (same as sweep_eval best-run green).
    palette = {
        puffer_label: "#0a7a32",
        "Puffer": "#0a7a32",
        "Puffer C\nimplementation": "#0a7a32",
        "π*": "#0a7a32",
        "reference": "#1f77b4",
        "Reference Python": "#1f77b4",
        "Original": "#1f77b4",
        "Original MAPPO": "#1f77b4",
        "Reference Python\nimplementation": "#1f77b4",
        "Random": "#888888",
    }
    rng = np.random.default_rng(0)

    # Per-episode % of max: return_per_agent / (n_food * eat / n_agents) * 100
    # Max obtainable = all initial food pellets eaten (food reward only).
    pct_by_name = {
        name: np.asarray(results[name]["return_pct_max"], dtype=np.float64)
        for name in order
    }
    all_pct = np.concatenate([pct_by_name[n] for n in order])
    all_max_pa = np.concatenate(
        [
            np.asarray(results[n]["max_return_per_agent"], dtype=np.float64)
            for n in order
        ]
    )
    mean_max_pa = float(np.mean(all_max_pa))
    print(
        f"LHS: return/agent as % of max obtainable "
        f"(n_food×eat / n_fish); mean max/agent={mean_max_pa:.1f}"
    )

    # layout: left big strip for return; right 2×3 bars for behavior stats
    fig = plt.figure(figsize=(12.5, 5.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1, 1, 1])

    # --- left column: per-agent return as % of obtainable max ---
    ax_ret = fig.add_subplot(gs[:, 0])

    for i, name in enumerate(order):
        vals_pct = pct_by_name[name]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals_pct))
        ax_ret.scatter(
            np.full(len(vals_pct), i) + jitter,
            vals_pct,
            s=36,
            alpha=0.7,
            color=palette.get(name, "#444"),
            edgecolors="none",
            zorder=3,
            label=name,
        )
        mean, half, _n = mean_ci95(vals_pct)
        ax_ret.errorbar(
            i,
            mean,
            yerr=half,
            fmt="D",
            color="#222222",
            markersize=7,
            capsize=5,
            elinewidth=1.4,
            zorder=5,
            label="mean ± 95% CI" if i == 0 else None,
        )
        ax_ret.annotate(
            f"{mean:.0f}%",
            xy=(i, mean),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color="#222",
        )
    ax_ret.set_xticks(np.arange(len(order)))
    ax_ret.set_xticklabels(order, fontsize=10)
    ax_ret.set_ylabel(
        "per agent\n(% of max obtainable)",
        fontsize=11,
    )
    ax_ret.set_title(
        f"max = (n_food × eat) / n_fish  (mean {mean_max_pa:.0f})",
        fontsize=11,
    )
    ax_ret.grid(True, alpha=0.22, zorder=0)
    ax_ret.set_axisbelow(True)
    ax_ret.spines["top"].set_visible(False)
    ax_ret.spines["right"].set_visible(False)
    y_hi = max(float(np.max(all_pct)) * 1.08, 5.0)
    y_lo = min(0.0, float(np.min(all_pct)) * 1.08)
    ax_ret.set_ylim(y_lo, y_hi)
    ax_ret.legend(loc="best", fontsize=8, framealpha=0.9)

    # --- remaining panels: mean ± 95% CI bars for behavior stats ---
    bar_specs = [s for s in STAT_SPECS if s[0] != "returns"]
    # absolute per-agent return as first bar panel (raw units, for reference)
    bar_specs = [
        ("return_per_agent", "per agent\n(absolute)", "{:.0f}"),
        *bar_specs,
    ]
    # 2×3 slots in columns 1..3
    slots = [(r, c) for r in range(2) for c in range(1, 4)]
    for idx, (key, ylabel, fmt) in enumerate(bar_specs):
        if idx >= len(slots):
            break
        r, c = slots[idx]
        ax = fig.add_subplot(gs[r, c])
        means = []
        halves = []
        for name in order:
            vals = results[name].get(key) or results[name]["series"].get(key, [])
            m, h, _ = mean_ci95(vals)
            means.append(m)
            halves.append(h)
        xs = np.arange(len(order))
        colors = [palette.get(n, "#444") for n in order]
        ax.bar(
            xs,
            means,
            yerr=halves,
            color=colors,
            width=0.62,
            capsize=4,
            error_kw=dict(elinewidth=1.3, ecolor="#222222", capthick=1.2),
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
        for i, (m, h) in enumerate(zip(means, halves)):
            ax.annotate(
                fmt.format(m),
                xy=(i, m + h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222",
            )
        tick = {
            "Reference Python": "Reference\nPython",
        }
        ax.set_xticks(xs)
        ax.set_xticklabels([tick.get(n, n) for n in order], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.22, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # zero line when data crosses zero
        if min(means) < 0 < max(means) or min(m - h for m, h in zip(means, halves)) < 0:
            ax.axhline(0, color="#999", lw=0.8, zorder=1)

    star_note = (
        "  ·  π* = best solve (time/return)"
        if "π" in puffer_label or puffer_label in ("pi*", "pi_star")
        else ""
    )
    fig.suptitle(
        f"Sim-to-sim transfer · rollouts in the reference Python simulator · "
        f"{args.episodes} rollouts × T={args.episode_length}{star_note}",
        fontsize=12,
    )

    out = Path(args.out)
    if not out.is_absolute():
        out = PUFFER_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    # also PNG for quick preview
    png = out.with_suffix(".png")
    fig.savefig(png, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
