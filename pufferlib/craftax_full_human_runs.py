from __future__ import annotations

import bz2
import importlib
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CRAFTAX_RUNS_ZIP = ROOT / "resources" / "craftax" / "runs.zip"
CRAFTAX_RUNS_DIR = ROOT / "resources" / "craftax" / "people"

OLD_CRAFTAX_MODULE_ALIASES = {
    "craftax.craftax_state": "craftax.craftax.craftax_state",
    "craftax.constants": "craftax.craftax.constants",
}


def install_old_craftax_pickle_aliases() -> None:
    """Allow old Craftax pickles to load under the newer package layout."""
    for old_name, new_name in OLD_CRAFTAX_MODULE_ALIASES.items():
        if old_name not in sys.modules:
            sys.modules[old_name] = importlib.import_module(new_name)


def load_compressed_pickle(path: str | Path) -> Any:
    install_old_craftax_pickle_aliases()
    with bz2.BZ2File(path, "rb") as f:
        return pickle.load(f)


def run_paths(runs_dir: Path = CRAFTAX_RUNS_DIR) -> list[Path]:
    return sorted(Path(runs_dir).glob("run*.pbz2"), key=lambda path: int(path.stem.removeprefix("run")))


def summarize_run(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = load_compressed_pickle(path)
    actions = np.asarray(data["action"], dtype=np.int32)
    rewards = np.asarray(data["reward"], dtype=np.float32)
    dones = np.asarray(data["done"], dtype=np.bool_)

    max_achievements = 0
    final_achievements = 0
    if data["state"]:
        achievement_counts = [
            int(np.count_nonzero(np.asarray(state.achievements, dtype=np.bool_)))
            for state in data["state"]
        ]
        max_achievements = max(achievement_counts)
        final_achievements = achievement_counts[-1]

    return {
        "run": path.stem,
        "states": len(data["state"]),
        "steps": len(actions),
        "episodes": int(np.count_nonzero(dones)),
        "last_done": bool(dones[-1]) if len(dones) else False,
        "reward_sum": float(rewards.sum()) if len(rewards) else 0.0,
        "reward_min": float(rewards.min()) if len(rewards) else 0.0,
        "reward_max": float(rewards.max()) if len(rewards) else 0.0,
        "reward_nonzero": int(np.count_nonzero(rewards)),
        "unique_actions": int(len(set(actions.tolist()))),
        "action_min": int(actions.min()) if len(actions) else None,
        "action_max": int(actions.max()) if len(actions) else None,
        "max_achievements": max_achievements,
        "final_achievements": final_achievements,
    }


def load_actions(path: str | Path) -> np.ndarray:
    data = load_compressed_pickle(path)
    return np.asarray(data["action"], dtype=np.int32)
