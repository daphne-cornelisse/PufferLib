#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pufferlib.craftax_mini_human_data import HUMAN_DATAFRAME_PATH, ROOT


NUM_ACTIONS = 5
DEFAULT_OUT = ROOT / "resources" / "craftax_mini" / "craftax_mini_anchor_policy.bin"


def parse_actions(value: str) -> np.ndarray:
    value = value.strip().strip("[]")
    if not value:
        return np.empty(0, dtype=np.int32)
    return np.asarray([int(float(item)) for item in value.split()], dtype=np.int32)


def build_replay_lib() -> ctypes.CDLL:
    tmp = tempfile.TemporaryDirectory()
    src = Path(tmp.name) / "craftax_mini_anchor_replay.c"
    so = Path(tmp.name) / "craftax_mini_anchor_replay.so"
    src.write_text(
        r'''
        #include <stdint.h>
        #include <string.h>
        #include "ocean/craftax_mini/craftax_mini.h"

        int32_t craftax_mini_anchor_obs_size(void) {
            return CRAFTAX_OBS_SIZE;
        }

        int32_t craftax_mini_anchor_collect(
            uint32_t seed,
            int32_t goal_block,
            const int32_t* actions,
            int32_t num_actions,
            float* obs_out
        ) {
            g_craftax_mini_config_goal_block = goal_block;
            g_craftax_mini_max_timesteps = 300;
            g_craftax_mini_use_human_maps = true;
            craftax_set_reset_pool_size(0);

            Craftax env;
            memset(&env, 0, sizeof(env));
            env.num_agents = 1;
            env.rng = seed;
            env.seed = seed;
            env.observations = calloc(CRAFTAX_OBS_SIZE, sizeof(float));
            env.actions = calloc(1, sizeof(float));
            env.rewards = calloc(1, sizeof(float));
            env.terminals = calloc(1, sizeof(float));
            c_init(&env);

            int32_t wrote = 0;
            for (int32_t i = 0; i < num_actions; i++) {
                c_step_encode(&env);
                memcpy(obs_out + (int64_t)i * CRAFTAX_OBS_SIZE,
                       env.observations,
                       CRAFTAX_OBS_SIZE * sizeof(float));
                env.actions[0] = (float)actions[i];
                c_step_gameplay(&env);
                wrote++;
                if (env.terminals[0] != 0.0f) {
                    break;
                }
            }

            c_close(&env);
            free(env.observations);
            free(env.actions);
            free(env.rewards);
            free(env.terminals);
            return wrote;
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
    lib = ctypes.CDLL(str(so))
    lib._tmp = tmp
    lib.craftax_mini_anchor_obs_size.restype = ctypes.c_int32
    lib.craftax_mini_anchor_collect.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.craftax_mini_anchor_collect.restype = ctypes.c_int32
    return lib


def load_episodes(
    limit: int | None = None,
    successful_only: bool = True,
) -> list[tuple[int, int, np.ndarray]]:
    import polars as pl

    df = pl.read_parquet(HUMAN_DATAFRAME_PATH)
    if successful_only:
        df = df.filter(pl.col("success") > 0)
    rows = df.select(["world", "task_object_id", "actions"]).iter_rows(named=True)
    episodes = []
    for row in rows:
        actions = parse_actions(row["actions"])
        if actions.size == 0:
            continue
        episodes.append((int(row["world"]), int(row["task_object_id"]), actions))
        if limit is not None and len(episodes) >= limit:
            break
    return episodes


def collect_observations(
    lib: ctypes.CDLL,
    episodes: list[tuple[int, int, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    obs_size = lib.craftax_mini_anchor_obs_size()

    out = []
    for seed, goal, actions in episodes:
        obs = np.empty((len(actions), obs_size), dtype=np.float32)
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        count = lib.craftax_mini_anchor_collect(
            ctypes.c_uint32(seed),
            ctypes.c_int32(goal),
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(len(actions)),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if count > 0:
            out.append((obs[:count].copy(), actions[:count].copy()))
    return out


class AnchorPolicy(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.encoder = nn.Linear(obs_size, hidden_size, bias=False)
        self.decoder = nn.Linear(hidden_size, NUM_ACTIONS + 1, bias=False)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            for _ in range(num_layers)
        ])

    @staticmethod
    def _g(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    @staticmethod
    def _log_g(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for layer in self.layers:
            hidden, gate, proj = layer(h).chunk(3, dim=-1)
            log_coeffs = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self._log_g(hidden)
            a_star = log_coeffs.cumsum(dim=1)
            out = (a_star + (log_values - a_star).logcumsumexp(dim=1)).exp()
            p = proj.sigmoid()
            h = p * out + (1.0 - p) * h
        return self.decoder(h)[..., :NUM_ACTIONS]

    def native_weights(self) -> np.ndarray:
        parts = [
            self.encoder.weight.detach().cpu().numpy().astype(np.float32).ravel(),
            self.decoder.weight.detach().cpu().numpy().astype(np.float32).ravel(),
        ]
        parts.extend(layer.weight.detach().cpu().numpy().astype(np.float32).ravel() for layer in self.layers)
        return np.concatenate(parts)


def make_batch(dataset, obs_size: int, batch_size: int, max_len: int, device: torch.device):
    idx = np.random.randint(0, len(dataset), size=batch_size)
    obs = torch.zeros((batch_size, max_len, obs_size), dtype=torch.float32, device=device)
    act = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    for b, i in enumerate(idx):
        ep_obs, ep_act = dataset[i]
        if len(ep_act) > max_len:
            start = np.random.randint(0, len(ep_act) - max_len + 1)
            ep_obs = ep_obs[start:start + max_len]
            ep_act = ep_act[start:start + max_len]
        n = len(ep_act)
        obs[b, :n] = torch.from_numpy(ep_obs).to(device)
        act[b, :n] = torch.from_numpy(ep_act.astype(np.int64)).to(device)
    return obs, act


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    lib = build_replay_lib()
    obs_size = lib.craftax_mini_anchor_obs_size()
    episodes = load_episodes(args.limit_episodes, successful_only=not args.include_failures)
    dataset = collect_observations(lib, episodes)
    if not dataset:
        raise RuntimeError("No training episodes collected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnchorPolicy(obs_size, args.hidden_size, args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for step in range(1, args.epochs + 1):
        obs, act = make_batch(dataset, obs_size, args.batch_size, args.max_len, device)
        logits = model(obs)
        loss = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), act.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0:
            pred = logits.argmax(dim=-1)
            mask = act != -100
            acc = (pred[mask] == act[mask]).float().mean().item()
            print(f"step={step} loss={loss.item():.4f} acc={acc:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.native_weights().tofile(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
