#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pufferlib.craftax_mini_human_data import ROOT


ANCHOR_PATH = ROOT / "resources" / "craftax_mini" / "craftax_mini_anchor_policy.bin"
OUT_DIR = ROOT / "resources" / "craftax_mini"
NUM_ACTIONS = 5
HIDDEN = 32
LAYERS = 1


def build_lib():
    tmp = tempfile.TemporaryDirectory()
    src = Path(tmp.name) / "craftax_mini_rl_ablation.c"
    so = Path(tmp.name) / "craftax_mini_rl_ablation.so"
    src.write_text(
        r'''
        #include <stdint.h>
        #include <stdlib.h>
        #include <string.h>
        #include "ocean/craftax_mini/craftax_mini.h"

        int32_t rl_obs_size(void) { return CRAFTAX_OBS_SIZE; }

        Craftax* rl_create(uint32_t seed, int32_t goal_block, int32_t max_steps) {
            g_craftax_mini_config_goal_block = goal_block;
            g_craftax_mini_max_timesteps = max_steps;
            g_craftax_mini_use_human_maps = true;
            craftax_set_reset_pool_size(0);
            Craftax* env = (Craftax*)calloc(1, sizeof(Craftax));
            env->num_agents = 1;
            env->rng = seed;
            env->seed = seed;
            env->observations = (float*)calloc(CRAFTAX_OBS_SIZE, sizeof(float));
            env->actions = (float*)calloc(1, sizeof(float));
            env->rewards = (float*)calloc(1, sizeof(float));
            env->terminals = (float*)calloc(1, sizeof(float));
            c_init(env);
            c_step_encode(env);
            return env;
        }

        void rl_observe(Craftax* env, float* obs_out) {
            c_step_encode(env);
            memcpy(obs_out, env->observations, CRAFTAX_OBS_SIZE * sizeof(float));
        }

        float rl_step(Craftax* env, int32_t action, int32_t* done_out) {
            env->actions[0] = (float)action;
            c_step_gameplay(env);
            *done_out = env->terminals[0] != 0.0f;
            return env->rewards[0];
        }

        void rl_close(Craftax* env) {
            if (!env) return;
            c_close(env);
            free(env->observations);
            free(env->actions);
            free(env->rewards);
            free(env->terminals);
            free(env);
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
    lib.rl_obs_size.restype = ctypes.c_int32
    lib.rl_create.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32]
    lib.rl_create.restype = ctypes.c_void_p
    lib.rl_observe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.rl_step.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    lib.rl_step.restype = ctypes.c_float
    lib.rl_close.argtypes = [ctypes.c_void_p]
    return lib


class NativePolicy(nn.Module):
    def __init__(self, obs_size: int):
        super().__init__()
        self.encoder = nn.Linear(obs_size, HIDDEN, bias=False)
        self.decoder = nn.Linear(HIDDEN, NUM_ACTIONS + 1, bias=False)
        self.layers = nn.ModuleList([nn.Linear(HIDDEN, 3 * HIDDEN, bias=False) for _ in range(LAYERS)])

    @staticmethod
    def g(x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    @staticmethod
    def log_g(x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def initial_state(self, batch: int, device):
        return [torch.zeros(batch, HIDDEN, device=device) for _ in range(LAYERS)]

    def forward_eval(self, obs, state):
        h = self.encoder(obs)
        next_state = []
        for i, layer in enumerate(self.layers):
            hidden, gate, proj = layer(h).chunk(3, dim=-1)
            out = torch.lerp(state[i], self.g(hidden), gate.sigmoid())
            p = proj.sigmoid()
            h = p * out + (1.0 - p) * h
            next_state.append(out)
        dec = self.decoder(h)
        return dec[:, :NUM_ACTIONS], dec[:, NUM_ACTIONS], next_state

    def forward_sequence(self, obs):
        h = self.encoder(obs)
        for layer in self.layers:
            hidden, gate, proj = layer(h).chunk(3, dim=-1)
            log_coeffs = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self.log_g(hidden)
            a_star = log_coeffs.cumsum(dim=1)
            out = (a_star + (log_values - a_star).logcumsumexp(dim=1)).exp()
            p = proj.sigmoid()
            h = p * out + (1.0 - p) * h
        dec = self.decoder(h)
        return dec[..., :NUM_ACTIONS], dec[..., NUM_ACTIONS]

    def load_native(self, path: Path):
        flat = np.fromfile(path, dtype=np.float32)
        offset = 0
        for param in [self.encoder.weight, self.decoder.weight, *[layer.weight for layer in self.layers]]:
            n = param.numel()
            param.data.copy_(torch.from_numpy(flat[offset:offset + n].reshape(tuple(param.shape))))
            offset += n
        if offset != flat.size:
            raise RuntimeError(f"Consumed {offset} floats from {path}, file has {flat.size}")

    def save_native(self, path: Path):
        parts = [self.encoder.weight, self.decoder.weight, *[layer.weight for layer in self.layers]]
        flat = torch.cat([p.detach().cpu().flatten() for p in parts]).numpy().astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat.tofile(path)


def make_envs(lib, num_envs: int, max_episode_steps: int):
    envs = []
    for i in range(num_envs):
        goal = [10, 21, 22][i % 3]
        envs.append(lib.rl_create(ctypes.c_uint32(i), ctypes.c_int32(goal), ctypes.c_int32(max_episode_steps)))
    return envs


def close_envs(lib, envs):
    for env in envs:
        lib.rl_close(env)


def observe(lib, envs, obs_size):
    obs = np.empty((len(envs), obs_size), dtype=np.float32)
    for i, env in enumerate(envs):
        lib.rl_observe(env, obs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return obs


def train_one(args, use_reg: bool):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    lib = build_lib()
    obs_size = lib.rl_obs_size()
    envs = make_envs(lib, args.num_envs, args.max_episode_steps)
    device = torch.device("cpu")
    policy = NativePolicy(obs_size).to(device)
    anchor = None
    if use_reg:
        anchor = NativePolicy(obs_size).to(device)
        anchor.load_native(args.anchor_path)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    returns_window = []
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    obs_np = observe(lib, envs, obs_size)
    state = policy.initial_state(args.num_envs, device)

    for update in range(1, args.updates + 1):
        obs_buf, act_buf, logp_buf, reward_buf, done_buf, value_buf, entropy_buf = [], [], [], [], [], [], []
        kl_buf = []
        for _ in range(args.horizon):
            obs_t = torch.from_numpy(obs_np).to(device)
            logits, value, next_state = policy.forward_eval(obs_t, state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            entropy = dist.entropy()
            if anchor is not None:
                with torch.no_grad():
                    anchor_logits, _, _ = anchor.forward_eval(obs_t, anchor.initial_state(args.num_envs, device))
                    log_probs = torch.log_softmax(logits, dim=-1)
                    anchor_log_probs = torch.log_softmax(anchor_logits, dim=-1)
                    kl_buf.append((log_probs.exp() * (log_probs - anchor_log_probs)).sum(dim=-1))

            rewards = np.empty(args.num_envs, dtype=np.float32)
            dones = np.empty(args.num_envs, dtype=np.float32)
            for i, env in enumerate(envs):
                done = ctypes.c_int32(0)
                rewards[i] = lib.rl_step(env, ctypes.c_int32(int(action[i])), ctypes.byref(done))
                dones[i] = float(done.value)
                episode_returns[i] += rewards[i]
                if done.value:
                    returns_window.append(float(episode_returns[i]))
                    episode_returns[i] = 0.0

            obs_buf.append(obs_t)
            act_buf.append(action)
            logp_buf.append(logp)
            reward_buf.append(torch.from_numpy(rewards).to(device))
            done_buf.append(torch.from_numpy(dones).to(device))
            value_buf.append(value)
            entropy_buf.append(entropy)

            state = [s.detach() * torch.from_numpy(1.0 - dones).to(device).unsqueeze(-1) for s in next_state]
            obs_np = observe(lib, envs, obs_size)

        with torch.no_grad():
            _, next_value, _ = policy.forward_eval(torch.from_numpy(obs_np).to(device), state)
            returns = []
            running = next_value
            for rew, done in zip(reversed(reward_buf), reversed(done_buf)):
                running = rew + args.gamma * running * (1.0 - done)
                returns.append(running)
            returns.reverse()

        values = torch.stack(value_buf)
        returns_t = torch.stack(returns)
        logps = torch.stack(logp_buf)
        entropy = torch.stack(entropy_buf)
        adv = returns_t - values.detach()
        pg_loss = -(adv * logps).mean()
        vf_loss = 0.5 * (returns_t - values).pow(2).mean()
        ent = entropy.mean()
        loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * ent
        reg_kl = torch.tensor(0.0)
        if kl_buf:
            reg_kl = torch.stack(kl_buf).mean()
            loss = loss + args.reg_coef * reg_kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        opt.step()

        if update == 1 or update % args.log_interval == 0:
            recent = returns_window[-100:]
            mean_return = float(np.mean(recent)) if recent else 0.0
            print(
                f"{'reg' if use_reg else 'no_reg'} update={update} "
                f"steps={update * args.horizon * args.num_envs} "
                f"mean_return={mean_return:.4f} loss={loss.item():.4f} "
                f"reg_kl={float(reg_kl):.4f}"
            )

    out = args.out_dir / ("craftax_mini_rl_reg.bin" if use_reg else "craftax_mini_rl_no_reg.bin")
    policy.save_native(out)
    close_envs(lib, envs)
    recent = returns_window[-100:]
    return out, float(np.mean(recent)) if recent else 0.0, len(returns_window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-path", type=Path, default=ANCHOR_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--reg-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--log-interval", type=int, default=8)
    args = parser.parse_args()

    no_reg = train_one(args, use_reg=False)
    reg = train_one(args, use_reg=True)
    print(f"no_reg_out={no_reg[0]} mean_return={no_reg[1]:.6f} episodes={no_reg[2]}")
    print(f"reg_out={reg[0]} mean_return={reg[1]:.6f} episodes={reg[2]}")


if __name__ == "__main__":
    main()
