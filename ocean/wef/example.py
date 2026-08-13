#!/usr/bin/env python3
"""Convert native raw PufferNet .bin checkpoints to torch .pt state_dicts.

Supports discrete (logit heads) and continuous (Gaussian means + logstd) policies.

Weight file order (matches CUDA weights_create / puffercpu.h):
  encoder weight          (hidden, obs)
  decoder weight          (atn_sum + 1, hidden)   # last row is value
  decoder logstd          (num_actions,)          # continuous only
  mingru[layer] weight    (3 * hidden, hidden)    # per layer

Usage:
  # WEF continuous (default act heads are 1,1,1,1):
  python ocean/wef/example.py CHECKPOINT.bin out.pt \\
      --obs-size 110 --hidden-size 128 --num-layers 3 --continuous

  # Load + act:
  from ocean.wef.example import load_policy
  policy = load_policy('out.pt', obs_size=110, hidden_size=128, num_layers=3,
                       continuous=True, num_actions=4)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_act_sizes(raw: str) -> list[int]:
    return [int(x) for x in raw.split(',') if x]


def _align8(idx: int) -> int:
    """Match puffercpu.h get_weights_aligned: pad cursor to multiple of 8 floats."""
    return (idx + 7) & ~7


def take(flat: torch.Tensor, idx: int, n: int, shape: tuple[int, ...], align: bool = True):
    """Consume n floats from flat[idx:], optionally 8-align the cursor afterward.

    GPU checkpoints are loaded by puffercpu with get_weights_aligned. After each
    block the index is rounded up to a multiple of 8, which *skips* 0–7 floats in
    the file. Those skips must be reproduced here or every subsequent tensor is
    shifted (MinGRU garbage → saturated actions → near-random return).
    """
    end = idx + n
    if end > flat.numel():
        # puffercpu allocates +7 trailing floats so a final align can read zeros
        need = end - flat.numel()
        if need <= 7:
            pad = torch.zeros(need, dtype=flat.dtype)
            chunk = torch.cat([flat[idx:], pad]).reshape(shape).clone()
            idx = end
            if align:
                idx = _align8(idx)
            return chunk, idx
        raise ValueError(f'checkpoint ended early at {idx}; need {n} more floats')
    chunk = flat[idx:end].reshape(shape).clone()
    idx = end
    if align:
        idx = _align8(idx)
    return chunk, idx


def convert(
    bin_path: str,
    pt_path: str,
    obs_size: int,
    hidden_size: int,
    num_layers: int,
    act_sizes: list[int],
    continuous: bool = False,
):
    """Convert flat fp32 .bin → OrderedDict state_dict saved as .pt.

    Weight order (matches CUDA weights_create + puffercpu load):
      encoder (H, obs)           # 8-align after
      decoder ((A+1), H)         # 8-align after; last row = value
      logstd (1, A) if continuous  # 8-align after  ← often skips 4 floats
      mingru[i] (3H, H) each     # 8-align after
    """
    flat = torch.from_numpy(np.fromfile(bin_path, dtype=np.float32))
    atn_sum = sum(act_sizes)
    num_actions = len(act_sizes) if continuous else atn_sum
    if continuous:
        if any(a != 1 for a in act_sizes):
            raise ValueError(
                f'continuous policies expect act_sizes all 1s, got {act_sizes}')
        num_actions = len(act_sizes)
        atn_sum = num_actions

    state = OrderedDict()
    idx = 0

    state['encoder.encoder.weight'], idx = take(
        flat, idx, hidden_size * obs_size, (hidden_size, obs_size))
    state['encoder.encoder.bias'] = torch.zeros(hidden_size, dtype=torch.float32)

    decoder, idx = take(
        flat, idx, (atn_sum + 1) * hidden_size, (atn_sum + 1, hidden_size))
    state['decoder.decoder.weight'] = decoder[:atn_sum].clone()
    state['decoder.decoder.bias'] = torch.zeros(atn_sum, dtype=torch.float32)
    state['decoder.value_function.weight'] = decoder[atn_sum:].clone()
    state['decoder.value_function.bias'] = torch.zeros(1, dtype=torch.float32)

    if continuous:
        logstd, idx = take(flat, idx, num_actions, (1, num_actions))
        state['decoder.log_std'] = logstd.clone()

    for layer in range(num_layers):
        key = f'network.layers.{layer}.weight'
        state[key], idx = take(
            flat, idx, 3 * hidden_size * hidden_size,
            (3 * hidden_size, hidden_size))

    # Cursor may sit slightly past file end due to final align into +7 pad.
    if idx < flat.numel():
        # Leftover only OK if it is the 0–7 skip region past last real weight
        # (should not happen if last take already aligned).
        raise ValueError(
            f'unused floats after conversion: cursor {idx}, file has {flat.numel()} '
            f'(continuous={continuous}, act_sizes={act_sizes}, H={hidden_size}, L={num_layers})')
    if idx > flat.numel() + 7:
        raise ValueError(
            f'cursor {idx} far past file size {flat.numel()} — bad arch dims?')

    meta = {
        'obs_size': obs_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'act_sizes': act_sizes,
        'continuous': continuous,
        'num_actions': num_actions if continuous else atn_sum,
        'bin_path': str(bin_path),
    }
    torch.save({'state_dict': state, 'meta': meta}, pt_path)
    print(f'wrote {pt_path} ({idx} floats, continuous={continuous})')
    return state, meta


# ---------------------------------------------------------------------------
# PyTorch policy matching CUDA / puffercpu MinGRU architecture
# ---------------------------------------------------------------------------

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


class MinGRULayer(nn.Module):
    """Single-step MinGRU matching algo.cu mingru_gate / puffercpu.h mingru.

    Weight is bias-free Linear(hidden → 3*hidden) packed as (3H, H):
      combined = [hidden_cand, gate, highway_proj]
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x, state: (B, H)
        H = self.hidden_size
        combined = F.linear(x, self.weight, bias=None)  # (B, 3H)
        hidden = combined[:, :H]
        gate = combined[:, H:2 * H]
        hw = combined[:, 2 * H:3 * H]

        z = _sigmoid(gate)
        h_tilde = torch.where(hidden >= 0, hidden + 0.5, _sigmoid(hidden))
        h_out = state + z * (h_tilde - state)  # lerp(state, h_tilde, z)
        s = _sigmoid(hw)
        out = s * h_out + (1.0 - s) * x
        return out, h_out


class PufferNetContinuous(nn.Module):
    """Linear encoder → N×MinGRU → Linear decoder (means) + log_std + value.

    State dict keys match convert() output.
    """

    def __init__(
        self,
        obs_size: int,
        hidden_size: int,
        num_layers: int,
        num_actions: int,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.continuous = True

        self.encoder = nn.Module()
        self.encoder.encoder = nn.Linear(obs_size, hidden_size, bias=True)

        self.network = nn.Module()
        self.network.layers = nn.ModuleList(
            [MinGRULayer(hidden_size) for _ in range(num_layers)])

        self.decoder = nn.Module()
        self.decoder.decoder = nn.Linear(hidden_size, num_actions, bias=True)
        self.decoder.value_function = nn.Linear(hidden_size, 1, bias=True)
        self.decoder.log_std = nn.Parameter(torch.zeros(1, num_actions))

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device)

    def forward_eval(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ):
        """obs: (B, obs_size). state: (L, B, H) or None.

        Returns mean (or sample), value, next_state.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        if state is None:
            state = self.initial_state(B, obs.device)
        # Encoder is pure linear in CUDA (bias stored as zeros).
        h = F.linear(obs.float(), self.encoder.encoder.weight, self.encoder.encoder.bias)

        new_states = []
        for i, layer in enumerate(self.network.layers):
            h, s = layer(h, state[i])
            new_states.append(s)
        next_state = torch.stack(new_states, dim=0)

        mean = F.linear(h, self.decoder.decoder.weight, self.decoder.decoder.bias)
        value = F.linear(h, self.decoder.value_function.weight, self.decoder.value_function.bias)
        if deterministic:
            action = mean
        else:
            std = torch.exp(self.decoder.log_std.expand_as(mean))
            action = torch.distributions.Normal(mean, std).sample()
        return action, value, next_state

    def act_numpy(
        self,
        obs: np.ndarray,
        state: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numpy convenience: obs (B, O) or (O,), returns actions, value, state."""
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        st = None
        if state is not None:
            st = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, value, next_state = self.forward_eval(obs_t, st, deterministic)
        return (
            action.cpu().numpy(),
            value.cpu().numpy(),
            next_state.cpu().numpy(),
        )


def load_policy(
    pt_path: str,
    obs_size: Optional[int] = None,
    hidden_size: Optional[int] = None,
    num_layers: Optional[int] = None,
    continuous: Optional[bool] = None,
    num_actions: Optional[int] = None,
    act_sizes: Optional[list[int]] = None,
    device: str = 'cpu',
) -> PufferNetContinuous:
    payload = torch.load(pt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and 'state_dict' in payload:
        state = payload['state_dict']
        meta = payload.get('meta', {})
    else:
        state = payload
        meta = {}

    obs_size = obs_size or meta.get('obs_size')
    hidden_size = hidden_size or meta.get('hidden_size')
    num_layers = num_layers or meta.get('num_layers')
    continuous = meta.get('continuous', True) if continuous is None else continuous
    if not continuous:
        raise NotImplementedError('discrete load_policy not implemented in this helper')
    if num_actions is None:
        num_actions = meta.get('num_actions')
    if num_actions is None and act_sizes is not None:
        num_actions = len(act_sizes)
    if None in (obs_size, hidden_size, num_layers, num_actions):
        raise ValueError('missing arch dims; pass explicitly or use convert() .pt with meta')

    policy = PufferNetContinuous(obs_size, hidden_size, num_layers, num_actions)
    # Keys from convert(): encoder.encoder.*, decoder.*, network.layers.i.weight
    missing, unexpected = policy.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f'load_state_dict mismatch missing={missing} unexpected={unexpected}')
    policy.eval()
    return policy.to(device)


def convert_from_ini(ini_path: str, pt_path: Optional[str] = None, continuous: bool = True):
    """Convert using a best_policy.ini / sweep log that has load_model_path + arch."""
    vals = {}
    text = Path(ini_path).read_text(encoding='utf-8', errors='replace')
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        vals[k.strip()] = v.strip()

    bin_path = vals.get('load_model_path')
    if not bin_path:
        raise ValueError(f'no load_model_path in {ini_path}')
    hidden_size = int(vals.get('hidden_size', vals.get('policy.hidden_size', 128)))
    num_layers = int(vals.get('num_layers', vals.get('policy.num_layers', 3)))
    obs_size = int(vals.get('obs_size', 110))
    act_sizes = parse_act_sizes(vals.get('act_sizes', '1,1,1,1'))
    if pt_path is None:
        pt_path = str(Path(bin_path).with_suffix('.pt'))
    return convert(bin_path, pt_path, obs_size, hidden_size, num_layers, act_sizes, continuous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bin_path', nargs='?', help='raw .bin checkpoint')
    parser.add_argument('pt_path', nargs='?', help='output .pt path')
    parser.add_argument('--ini', type=str, default=None, help='policy ini with load_model_path')
    parser.add_argument('--obs-size', type=int, default=110)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument(
        '--act-sizes', type=parse_act_sizes, default=parse_act_sizes('1,1,1,1'),
        help='per-head sizes; continuous uses all-1s (default 1,1,1,1 for wef)')
    parser.add_argument(
        '--continuous', action='store_true', default=False,
        help='include decoder logstd and treat act_sizes as continuous heads')
    parser.add_argument(
        '--discrete', action='store_true', default=False,
        help='force discrete (no logstd); default if neither flag is continuous=False')
    args = parser.parse_args()

    continuous = args.continuous and not args.discrete
    # WEF default: continuous when act_sizes are all ones
    if not args.continuous and not args.discrete:
        continuous = all(a == 1 for a in args.act_sizes)

    if args.ini:
        convert_from_ini(args.ini, args.pt_path, continuous=continuous)
        return

    if not args.bin_path or not args.pt_path:
        parser.error('bin_path and pt_path required unless --ini is set')
    convert(
        args.bin_path, args.pt_path, args.obs_size, args.hidden_size,
        args.num_layers, args.act_sizes, continuous=continuous)


if __name__ == '__main__':
    main()
