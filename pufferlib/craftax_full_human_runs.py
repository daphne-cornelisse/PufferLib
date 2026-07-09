from __future__ import annotations

import bz2
import ctypes
import importlib
import pickle
import subprocess
import sys
import tempfile
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


_OBS_ENCODER = None


def _raylib_include_dir(root: Path) -> Path:
    candidates = [
        root / "raylib-6.0_memory" / "src",
        root / "raylib-6.0_linux_amd64" / "include",
        root / "raylib-5.5_linux_amd64" / "include",
    ]
    for path in candidates:
        if (path / "raylib.h").exists():
            return path
    raise FileNotFoundError("Could not find raylib headers under the repo root.")


class NativeCraftaxObservationEncoder:
    """Encode Craftax EnvState objects into the repo's symbolic observation."""

    def __init__(self) -> None:
        from tests.craftax_parity import OBS_SIZE
        from tests.craftax_state_fixtures import CraftaxState

        self.obs_size = OBS_SIZE
        self.state_type = CraftaxState

        root = ROOT
        raylib_include = _raylib_include_dir(root)
        source = r"""
        #include <stdint.h>
        #define CRAFTAX_ENABLE_ENV_IMPL
        #include "ocean/craftax/craftax.h"
        #include "ocean/craftax/step_crafting.h"
        #include "ocean/craftax/step_mobs.h"
        #include "ocean/craftax/step_spawning.h"

        void encode_state(const CraftaxState* state, float* obs) {
            craftax_encode_native_observation(state, obs);
        }
        """
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        src = tmp_path / "craftax_obs_encode.c"
        so = tmp_path / "craftax_obs_encode.so"
        src.write_text(source)
        subprocess.run(
            [
                "cc",
                "-std=c99",
                "-O2",
                "-shared",
                "-fPIC",
                "-I",
                str(root),
                "-I",
                str(raylib_include),
                str(src),
                "-lm",
                "-o",
                str(so),
            ],
            check=True,
            cwd=root,
        )
        self.lib = ctypes.CDLL(str(so))
        self.lib.encode_state.argtypes = [
            ctypes.POINTER(CraftaxState),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.encode_state.restype = None

    def encode_state(self, state: Any) -> np.ndarray:
        from tests.craftax_state_fixtures import (
            deserialize_jax_state_to_c,
            serialize_jax_state,
        )

        c_state = deserialize_jax_state_to_c(serialize_jax_state(state))
        obs = np.empty(self.obs_size, dtype=np.float32)
        self.lib.encode_state(
            ctypes.byref(c_state),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return obs


def get_observation_encoder() -> NativeCraftaxObservationEncoder:
    global _OBS_ENCODER
    if _OBS_ENCODER is None:
        _OBS_ENCODER = NativeCraftaxObservationEncoder()
    return _OBS_ENCODER


def load_run(path: str | Path) -> dict[str, Any]:
    return load_compressed_pickle(path)


def extract_obs_action_pairs(
    path: str | Path,
    *,
    include_next_obs: bool = False,
    include_reward_done: bool = False,
) -> dict[str, np.ndarray]:
    """Extract symbolic observation/action pairs from a full Craftax run."""
    data = load_run(path)
    states = data["state"]
    actions = np.asarray(data["action"], dtype=np.int32)
    rewards = np.asarray(data["reward"], dtype=np.float32)
    dones = np.asarray(data["done"], dtype=np.bool_)

    if len(states) != len(actions) + 1:
        raise ValueError(
            f"Expected len(states) == len(actions) + 1, got "
            f"{len(states)} states and {len(actions)} actions"
        )

    encoder = get_observation_encoder()
    obs = np.stack([encoder.encode_state(state) for state in states[:-1]], axis=0)
    out = {
        "obs": obs,
        "actions": actions,
    }

    if include_next_obs:
        next_obs = np.stack([encoder.encode_state(state) for state in states[1:]], axis=0)
        out["next_obs"] = next_obs

    if include_reward_done:
        out["rewards"] = rewards
        out["dones"] = dones

    return out


def load_obs_action_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def _concat_sample_dicts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("No dataset parts to concatenate.")
    keys = parts[0].keys()
    for part in parts[1:]:
        if part.keys() != keys:
            raise ValueError("All dataset parts must have the same keys.")
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in keys
    }


class CraftaxFullHumanDataset:
    """PyTorch-style dataset over full Craftax human transitions.

    Each sample always contains:
      - ``obs``: float32 symbolic observation
      - ``action``: int64 action id

    Optional fields:
      - ``next_obs``
      - ``reward``
      - ``done``
      - ``run_id``: integer index of the source run in the input list
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
    ) -> None:
        self.obs = np.asarray(data["obs"], dtype=np.float32)
        self.actions = np.asarray(data["actions"], dtype=np.int64)
        self.next_obs = (
            np.asarray(data["next_obs"], dtype=np.float32)
            if "next_obs" in data else None
        )
        self.rewards = (
            np.asarray(data["rewards"], dtype=np.float32)
            if "rewards" in data else None
        )
        self.dones = (
            np.asarray(data["dones"], dtype=np.bool_)
            if "dones" in data else None
        )
        self.run_ids = (
            np.asarray(data["run_ids"], dtype=np.int64)
            if "run_ids" in data else None
        )

        n = len(self.actions)
        if self.obs.shape[0] != n:
            raise ValueError("obs/actions length mismatch")
        if self.next_obs is not None and self.next_obs.shape[0] != n:
            raise ValueError("next_obs/actions length mismatch")
        if self.rewards is not None and self.rewards.shape[0] != n:
            raise ValueError("rewards/actions length mismatch")
        if self.dones is not None and self.dones.shape[0] != n:
            raise ValueError("dones/actions length mismatch")
        if self.run_ids is not None and self.run_ids.shape[0] != n:
            raise ValueError("run_ids/actions length mismatch")

    @classmethod
    def from_run(
        cls,
        path: str | Path,
        *,
        include_next_obs: bool = False,
        include_reward_done: bool = False,
    ) -> "CraftaxFullHumanDataset":
        data = extract_obs_action_pairs(
            path,
            include_next_obs=include_next_obs,
            include_reward_done=include_reward_done,
        )
        data["run_ids"] = np.zeros(len(data["actions"]), dtype=np.int64)
        return cls(data)

    @classmethod
    def from_runs(
        cls,
        paths: list[str | Path],
        *,
        include_next_obs: bool = False,
        include_reward_done: bool = False,
    ) -> "CraftaxFullHumanDataset":
        parts = []
        for run_id, path in enumerate(paths):
            part = extract_obs_action_pairs(
                path,
                include_next_obs=include_next_obs,
                include_reward_done=include_reward_done,
            )
            part["run_ids"] = np.full(len(part["actions"]), run_id, dtype=np.int64)
            parts.append(part)
        return cls(_concat_sample_dicts(parts))

    @classmethod
    def from_npz(cls, path: str | Path) -> "CraftaxFullHumanDataset":
        return cls(load_obs_action_npz(path))

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample = {
            "obs": self.obs[index],
            "action": self.actions[index],
        }
        if self.next_obs is not None:
            sample["next_obs"] = self.next_obs[index]
        if self.rewards is not None:
            sample["reward"] = self.rewards[index]
        if self.dones is not None:
            sample["done"] = self.dones[index]
        if self.run_ids is not None:
            sample["run_id"] = self.run_ids[index]
        return sample


def make_dataloader(
    dataset: CraftaxFullHumanDataset,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
