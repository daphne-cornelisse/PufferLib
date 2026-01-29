## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import sys
import glob
import ast
import time
import random
import shutil
import argparse
import importlib
import matplotlib
import configparser
from threading import Thread
from collections import defaultdict, deque

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.models
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME
)
# Assume advantage kernel has been built if torch has been compiled with CUDA or HIP support
# and can find CUDA or HIP in the system
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)

def show_reconstruction(mb_obs_nxt, mb_obs_next_pred, prediction_error, logger, step,
                        vision=5, cont_obs_size=1, num_tile_types=10):
    
    import wandb
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    
    # Compute observation size (vision is configured as N x N).
    obs_size = vision
    if obs_size % 2 == 0:
        obs_size += 1
    grid_size = obs_size * obs_size
    mb_obs_nxt = mb_obs_nxt.detach().cpu().numpy()
    mb_obs_next_pred = mb_obs_next_pred.detach().cpu().numpy()
    prediction_error = prediction_error.detach().cpu().numpy()
    
    wandb_images = []
    
    batch_idx = 0
    time_idx = 0
    
    # Get flattened observations. Slice off any continuous features.
    actual_grid = mb_obs_nxt[batch_idx, time_idx, :grid_size].reshape(obs_size, obs_size)
    predicted_grid = mb_obs_next_pred[batch_idx, time_idx, :grid_size].reshape(obs_size, obs_size)
    error = prediction_error[batch_idx, time_idx]

    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    
    # Actual observation
    im0 = axes[0].imshow(actual_grid, vmin=0, vmax=num_tile_types - 1,
                         interpolation='nearest', cmap='cividis')
    axes[0].set_title(f'Actual next observation ({obs_size}x{obs_size})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im0, ax=axes[0], label='Tile Type')
    
    # Predicted observation
    im1 = axes[1].imshow(predicted_grid, vmin=0, vmax=num_tile_types - 1,
                         interpolation='nearest', cmap='cividis')
    axes[1].set_title(f'Pred next observation ({obs_size}x{obs_size})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[1], label='Tile Type')
    
    # Absolute difference
    diff_grid = np.abs(actual_grid - predicted_grid)
    im2 = axes[2].imshow(diff_grid, cmap='Reds', interpolation='nearest', vmin=0, vmax=2)
    axes[2].set_title(f'Abs Difference\nMSE: {error:.4f}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[2], label='Error')
    
    plt.suptitle(f'Batch {batch_idx}, Time {time_idx}', fontsize=14)
    plt.tight_layout()
    
    # Convert figure to image array
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    image_array = np.array(image)
    buf.close()
    
    wandb_images.append(
        wandb.Image(
            image_array, 
            caption=f"Batch {batch_idx}, t={time_idx}, MSE={error:.4f}"
        )
    )
    
    plt.close(fig)
    
    # Log to wandb
    logger.wandb.log({
        'world_model/reconstruction': wandb_images,
        'world_model/mean_reconstruction_error': prediction_error.mean(),
        'world_model/max_reconstruction_error': prediction_error.max(),
        'world_model/min_reconstruction_error': prediction_error.min(),
    }, step=step)

class PuffeRL:
    def __init__(self, config, full_config, vecenv, policy, world_model, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = config['torch_deterministic']
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config['seed']
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)

        # Vecenv info
        self.full_config = full_config
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents
        
        self.solved_at_step = None
        self.last_gif_step = 0
        self.last_map_eval_step = 0
        self.last_coverage_step = 0
        self.cumulative_solved = 0
        self.map_eval_env = None
        self.map_eval_env_test = None

        # Experience
        if config['batch_size'] == 'auto' and config['bptt_horizon'] == 'auto':
            raise pufferlib.APIUsageError('Must specify batch_size or bptt_horizon')
        elif config['batch_size'] == 'auto':
            config['batch_size'] = total_agents * config['bptt_horizon']
        elif config['bptt_horizon'] == 'auto':
            config['bptt_horizon'] = config['batch_size'] // total_agents

        batch_size = config['batch_size']
        horizon = config['bptt_horizon']
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f'Total agents {total_agents} <= segments {segments}'
            )

        device = config['device']
        self.observations = torch.zeros(segments, horizon, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents

        # LSTM
        if config['use_rnn']:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
            self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}

        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']
        max_minibatch_size = config['max_minibatch_size']
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f'minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly')

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(
                f'batch_size {batch_size} must be >= minibatch_size {minibatch_size}'
            )

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config['update_epochs'] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon 
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f'minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}'
            )

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        self.world_model = world_model.to(device)
        if config['compile']:
            self.policy = torch.compile(policy, mode=config['compile_mode'])
            self.policy.forward_eval = torch.compile(policy, mode=config['compile_mode'])
            pufferlib.pytorch.sample_logits = torch.compile(pufferlib.pytorch.sample_logits, mode=config['compile_mode'])

        # Optimizers
        def build_optimizer(params):
            if config['optimizer'] == 'adam':
                return torch.optim.Adam(
                    params,
                    lr=config['learning_rate'],
                    betas=(config['adam_beta1'], config['adam_beta2']),
                    eps=config['adam_eps'],
                )
            if config['optimizer'] == 'muon':
                import heavyball
                from heavyball import ForeachMuon
                warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
                heavyball.utils.compile_mode = "default"

                # # optionally a little bit better/faster alternative to newtonschulz iteration
                # import heavyball.utils
                # heavyball.utils.zeroth_power_mode = 'thinky_polar_express'

                # heavyball_momentum=True introduced in heavyball 2.1.1
                # recovers heavyball-1.7.2 behaviour - previously swept hyperparameters work well
                return ForeachMuon(
                    params,
                    lr=config['learning_rate'],
                    betas=(config['adam_beta1'], config['adam_beta2']),
                    eps=config['adam_eps'],
                    heavyball_momentum=True,
                )
            raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

        self.optimizer = build_optimizer(self.policy.parameters())
        self.world_model_optimizer = build_optimizer(self.world_model.parameters())

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Learning rate scheduler
        epochs = config['total_timesteps'] // config['batch_size']
        eta_min = config['learning_rate'] * config['min_lr_ratio']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=eta_min)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config['precision']
        self.amp_context = contextlib.nullcontext()
        if config.get('amp', True) and config['device'] == 'cuda':
            self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
        if precision not in ('float32', 'bfloat16'):
            raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']

        if config['use_rnn']:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t # TODO: Handle truncations separately
            self.global_step += int(mask.sum())

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)#, non_blocking=True)
            r = torch.as_tensor(r).to(device)#, non_blocking=True)
            d = torch.as_tensor(d).to(device)#, non_blocking=True)

            profile('eval_forward', epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config['use_rnn']:
                    state['lstm_h'] = self.lstm_h[env_id.start]
                    state['lstm_c'] = self.lstm_c[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

            profile('eval_copy', epoch)
            with torch.no_grad():
                if config['use_rnn']:
                    self.lstm_h[env_id.start] = state['lstm_h']
                    self.lstm_c[env_id.start] = state['lstm_c']

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1+self.ep_indices[env_id.stop - 1].item())

                if config['cpu_offload']:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r            
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l+1 >= config['bptt_horizon']:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config['device']).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile('eval_misc', epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile('env', epoch)
            self.vecenv.send(action)

        profile('eval_misc', epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        profile('train_misc', epoch, nest=True)
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        wm_clip = config['wm_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1

        wm_pred_error_mean = 0.0
        wm_intrinsic_mean = 0.0
    
        for mb in range(self.total_minibatches):
            profile('train_misc', epoch)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(self.values, self.rewards,
                self.terminals, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta

            profile('train_copy', epoch)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)
                       
            mb_obs_cur = mb_obs[:, :-1]
            mb_obs_nxt = mb_obs[:, 1:] 
            
            # Tiny world model predicts next observation given action taken
            categorical_logits, continuous_pred = self.world_model(mb_obs_cur, mb_actions[:, :-1])

            # Split target observations
            mb_obs_nxt_cat = mb_obs_nxt[:, :, :self.world_model.cat_obs_size].long()
            mb_obs_nxt_cont = mb_obs_nxt[:, :, self.world_model.cat_obs_size:]

            batch_size, seq_len, num_positions = mb_obs_nxt_cat.shape

            # Categorical loss (cross-entropy)
            categorical_loss = torch.nn.functional.cross_entropy(
                categorical_logits.reshape(-1, self.world_model.num_tile_types),  # [B*S*P, 10]
                mb_obs_nxt_cat.reshape(-1),                                # [B*S*P]
                reduction='none'
            )  # [B*S*P]

            # Reshape back and average over positions
            cat_loss = categorical_loss.reshape(batch_size, seq_len, num_positions).mean(dim=-1)  # [B, S]

            # Continuous loss (optional)
            if self.world_model.cont_obs_size > 0:
                cont_loss = ((continuous_pred - mb_obs_nxt_cont)**2).mean(dim=-1)
                next_obs_pred_error = cat_loss + cont_loss
            else:
                next_obs_pred_error = cat_loss
            next_obs_pred_error_clipped = torch.clamp(next_obs_pred_error, 0, config['wm_clip_coef'])
            wm_pred_error_mean += next_obs_pred_error.mean().item() / self.total_minibatches

            # Reconstruct predictions for visualization
            # Convert logits to predicted classes
            categorical_pred = categorical_logits.argmax(dim=-1)  # Shape: (batch, seq, categorical_obs_size)

            # Concatenate categorical predictions with continuous predictions
            if self.world_model.cont_obs_size > 0:
                mb_obs_next_pred = torch.cat([
                    categorical_pred.float(),
                    continuous_pred
                ], dim=-1)
            else:
                mb_obs_next_pred = categorical_pred.float()
            
            profile('train_misc', epoch)
            if mb % 100 == 0 and hasattr(self.logger, 'wandb') and epoch % 100 == 0:
                env_cfg = self.full_config.get('env', {})
                show_reconstruction(
                    mb_obs_nxt,
                    mb_obs_next_pred,
                    next_obs_pred_error,
                    self.logger,
                    self.global_step,
                    vision=env_cfg['vision'],
                    cont_obs_size=self.world_model.cont_obs_size,
                    num_tile_types=self.world_model.num_tile_types,
                )
            
            # Normalize prediction error and reward above-average errors.
            error_mean = next_obs_pred_error_clipped.mean()
            error_std = next_obs_pred_error_clipped.std(unbiased=False)
            error_std = error_std + 1e-8
            normalized_error = (next_obs_pred_error_clipped - error_mean) / error_std
            normalized_error = torch.relu(normalized_error)
            intrinsic_reward = normalized_error.pow(2) * config['wm_reward_coef']
            wm_intrinsic_mean += intrinsic_reward.mean().item() / self.total_minibatches
            
            # Note: we're skipping one element every batch (the last one) to match shapes
            mb_rewards[:, :-1] += intrinsic_reward
                
            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            adv = advantages[idx]
            adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
                 ratio, adv, config['gamma'], config['gae_lambda'],
                 config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Weight advantages by priority and normalize
            # adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()
        
            entropy_loss = entropy.mean() #std()

            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
            self.amp_context.__enter__() # TODO: AMP needs some debugging
            
            # Aggregate batch losses
            wm_loss = next_obs_pred_error_clipped.mean()

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()
            
            # Logging
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.item() / self.total_minibatches
            losses['value_loss'] += v_loss.item() / self.total_minibatches
            losses['entropy'] += entropy_loss.item() / self.total_minibatches
            losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
            losses['approx_kl'] += approx_kl.item() / self.total_minibatches
            losses['clipfrac'] += clipfrac.item() / self.total_minibatches
            losses['importance'] += ratio.mean().item() / self.total_minibatches
            losses['wm_loss'] += wm_loss.item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile('learn', epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # World model loss backward and step
            wm_loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), config['max_grad_norm'])
                self.world_model_optimizer.step()
                self.world_model_optimizer.zero_grad()

        # Reprioritize experience
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var
        self.wm_pred_error_mean = wm_pred_error_mean
        self.wm_intrinsic_mean = wm_intrinsic_mean

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        
        if done_training and self.solved_at_step is None:
            self.solved_at_step = self.global_step
            print(f'Env not solved, marking solved at final step {self.solved_at_step}.')
        
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        gif_interval_steps = config.get('gif_interval_steps', 0)
        gif_interval_epochs = config.get('gif_interval', 0)
        if gif_interval_steps:
            if (self.global_step - self.last_gif_step) >= gif_interval_steps or done_training:
                self.record_gif(policy=self.uncompiled_policy)
                self.last_gif_step = self.global_step
        elif gif_interval_epochs and (self.epoch % gif_interval_epochs == 0 or done_training):
            self.record_gif(policy=self.uncompiled_policy)

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'
                
        return logs

    def mean_and_log(self):
        import wandb
        config = self.config
        weighted_stats = {}
        weights = np.asarray(self.stats.get('n', []), dtype=np.float32)
        total_weight = float(weights.sum()) if weights.size else 0.0
        if total_weight > 0.0:
            def weighted_mean(key):
                vals = np.asarray(self.stats.get(key, []), dtype=np.float32)
                if vals.size != weights.size:
                    return None
                return float(np.sum(vals * weights) / total_weight)

            for key in (
                'perf',
                'score',
                'solved',
                'episode_return',
                'episode_length',
            ):
                value = weighted_mean(key)
                if value is not None:
                    weighted_stats[key] = value

            weighted_stats['episodes'] = total_weight
            solved_vals = np.asarray(self.stats.get('solved', []), dtype=np.float32)
            if solved_vals.size == weights.size:
                weighted_stats['solved_episodes'] = float(np.sum(solved_vals * weights))

        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        if weighted_stats:
            self.stats.update(weighted_stats)
    
        device = config['device']
        agent_steps = int(dist_sum(self.global_step, device))
        
        # Check if agent has solved the task
        score_key = 'score'
        if (self.solved_at_step is None and 
            score_key in self.stats and 
            self.stats[score_key] >= 0.999):
            self.solved_at_step = agent_steps
            print(f'solved at step {self.solved_at_step}!')
            
        logs = {
            'SPS': dist_sum(self.sps, device),
            'agent_steps': agent_steps,
            'uptime': time.time() - self.start_time,
            'epoch': int(dist_sum(self.epoch, device)),
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v['elapsed'] for k, v in self.profile},
            #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        episode_length = self.stats.get('episode_length', 0.0)
        episode_return = self.stats.get('episode_return', 0.0)
        if episode_length > 0:
            logs['environment/return_per_step'] = episode_return / episode_length

        if 'solved' in self.stats:
            logs['environment/solve_rate'] = self.stats['solved']

        logs['world_model/pred_error_mean'] = self.wm_pred_error_mean
        logs['world_model/intrinsic_reward_mean'] = self.wm_intrinsic_mean

        solved_eps = self.stats.get('solved_episodes', None)
        if solved_eps is not None:
            self.cumulative_solved += float(solved_eps)
            logs['environment/cumulative_solved'] = self.cumulative_solved

        # Keep solved_at_step for early stop, but avoid noisy charts.

        map_eval_logs = self.evaluate_maps(policy=self.uncompiled_policy)
        if map_eval_logs:
            logs.update(map_eval_logs)

        coverage_logs = self.evaluate_coverage(policy=self.uncompiled_policy)
        if coverage_logs:
            logs.update(coverage_logs)

        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               self.logger.log(logs, agent_steps)
               return logs
           else:
               return None

        self.logger.log(logs, agent_steps)
        return logs

    def evaluate_coverage(self, policy=None):
        config = self.config
        interval_steps = int(config.get('coverage_interval_steps', 0))
        if interval_steps <= 0:
            return {}
        if (self.global_step - self.last_coverage_step) < interval_steps:
            return {}

        logs = self._coverage_from_live_env()

        if not logs:
            return {}

        coverage_dir = config.get('coverage_dir', 'experiments/coverage')
        if coverage_dir:
            os.makedirs(coverage_dir, exist_ok=True)
            from PIL import Image
            Image.fromarray(logs['__img_goal']).save(
                os.path.join(coverage_dir, f'coverage_goal_step{self.global_step:09d}.png')
            )
            Image.fromarray(logs['__img_agent']).save(
                os.path.join(coverage_dir, f'coverage_agent_step{self.global_step:09d}.png')
            )

        if hasattr(self.logger, "wandb") and self.logger.wandb:
            import wandb
            logs['environment/coverage_heatmap'] = wandb.Image(
                logs['__img_goal'],
                caption=f"goal coverage @ {self.global_step} steps",
            )
            logs['environment/coverage_agent_heatmap'] = wandb.Image(
                logs['__img_agent'],
                caption=f"agent coverage @ {self.global_step} steps",
            )

        logs.pop('__img_goal', None)
        logs.pop('__img_agent', None)

        self.last_coverage_step = self.global_step
        return logs

    def _coverage_from_live_env(self):
        from pufferlib.ocean.pushworld import binding as pw_binding

        env = self.vecenv.driver_env
        full_map = np.asarray(pw_binding.vec_full_map(env.c_envs, 0))
        goal_visits = np.asarray(pw_binding.vec_get_coverage_goal(env.c_envs, 0))
        agent_visits = np.asarray(pw_binding.vec_get_coverage_agent(env.c_envs, 0))

        base_map = self._coverage_base_map(full_map)
        img_goal, img_agent = self._render_coverage(base_map, goal_visits, agent_visits)

        if not self.config.get('coverage_accumulate', False):
            pw_binding.vec_clear_coverage(env.c_envs, 0)

        return {
            '__img_goal': img_goal,
            '__img_agent': img_agent,
        }

    @staticmethod
    def _coverage_base_map(full_map):
        base_map = np.zeros_like(full_map)
        base_map[full_map == 1] = 1
        base_map[full_map == 2] = 2
        base_map[(full_map == 3) | (full_map == 6)] = 3
        return base_map

    @staticmethod
    def _render_coverage(base_map, goal_visits, agent_visits):
        height, width = base_map.shape

        def _normalize(counts):
            counts = counts.astype(np.float32)
            maxv = float(np.max(counts)) if counts.size else 0.0
            if maxv <= 0:
                return np.zeros_like(counts, dtype=np.float32)
            norm = np.log1p(counts) / np.log1p(maxv)
            norm = np.where(counts > 0, 0.2 + 0.8 * norm, 0.0)
            return norm

        def _render_heatmap(overlay_norm, color):
            img = np.zeros((height, width, 3), dtype=np.float32)
            img[:] = 0.05
            wall = base_map == 1
            agent_wall = base_map == 2
            goal = base_map == 3
            img[wall] = [0.12, 0.12, 0.12]
            img[agent_wall] = [0.22, 0.22, 0.22]
            img[goal] = [0.1, 0.35, 0.1]
            img[..., 0] = np.clip(img[..., 0] + overlay_norm * color[0], 0.0, 1.0)
            img[..., 1] = np.clip(img[..., 1] + overlay_norm * color[1], 0.0, 1.0)
            img[..., 2] = np.clip(img[..., 2] + overlay_norm * color[2], 0.0, 1.0)
            return (img * 255.0).astype(np.uint8)

        goal_norm = _normalize(goal_visits)
        agent_norm = _normalize(agent_visits)
        img_goal = _render_heatmap(goal_norm, (0.95, 0.0, 0.0))
        img_agent = _render_heatmap(agent_norm, (0.0, 0.4, 0.95))
        return img_goal, img_agent

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        return model_path

    def record_gif(self, policy=None):
        config = self.config
        gif_frames = int(config.get('gif_frames', 0))
        if gif_frames <= 0:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        gif_fps = float(config.get('gif_fps', 12))
        gif_dir = config.get('gif_dir', os.path.join(config['data_dir'], 'gifs'))
        gif_tile_size = int(config.get('gif_tile_size', 16))
        os.makedirs(gif_dir, exist_ok=True)

        from copy import deepcopy
        eval_args = deepcopy(self.full_config)
        eval_args['vec'] = dict(eval_args['vec'])
        eval_args['env'] = dict(eval_args['env'])
        eval_args['vec']['backend'] = 'Serial'
        eval_args['vec']['num_envs'] = 1
        eval_args['vec'].pop('num_workers', None)
        eval_args['vec']['batch_size'] = 1
        eval_args['env']['num_envs'] = 1
        eval_args['env']['render_mode'] = 'rgb_array'
        eval_args['env']['render_tile_size'] = gif_tile_size
        eval_args['env']['render_full_map'] = bool(config.get('gif_full_map', True))

        env_name = eval_args['env_name']
        eval_env = load_env(env_name, eval_args)
        driver = eval_env.driver_env
        device = config['device']

        gif_seed = config.get('gif_seed', None)
        gif_randomize = config.get('gif_randomize', False)
        if gif_seed is None and gif_randomize:
            gif_seed = int(time.time_ns() % 2147483647)

        gif_map_idx = config.get('gif_map_idx', None)
        gif_map_random = config.get('gif_map_random', gif_randomize)
        from pufferlib.ocean.pushworld import binding as pw_binding
        from pufferlib.ocean.pushworld import pushworld as pw_env
        num_puzzles = int(pw_binding.vec_num_puzzles(driver.c_envs))
        if num_puzzles > 0:
            if gif_map_idx is None and not gif_map_random:
                train_map_idx = eval_args['env'].get('train_map_idx', -1)
                train_map = eval_args['env'].get('train_map', None)
                coverage_map_idx = config.get('coverage_map_idx', -1)
                coverage_map = config.get('coverage_map', None)
                if isinstance(train_map, str) and train_map.strip() == "":
                    train_map = None
                if isinstance(coverage_map, str) and coverage_map.strip() == "":
                    coverage_map = None

                if train_map_idx is not None and int(train_map_idx) >= 0:
                    gif_map_idx = int(train_map_idx)
                elif train_map is not None:
                    gif_map_idx = pw_env._resolve_puzzle_index(
                        os.path.expanduser(eval_args['env'].get('puzzle_dir', '')),
                        train_map,
                        None,
                        int(eval_args['env'].get('max_puzzles', 0)),
                    )
                elif coverage_map_idx is not None and int(coverage_map_idx) >= 0:
                    gif_map_idx = int(coverage_map_idx)
                elif coverage_map is not None:
                    gif_map_idx = pw_env._resolve_puzzle_index(
                        os.path.expanduser(eval_args['env'].get('puzzle_dir', '')),
                        coverage_map,
                        None,
                        int(eval_args['env'].get('max_puzzles', 0)),
                    )
            if gif_map_idx is None and gif_map_random:
                gif_map_idx = int(np.random.randint(0, num_puzzles))
            if gif_map_idx is not None:
                gif_map_idx = int(gif_map_idx)
                if gif_map_idx < 0 or gif_map_idx >= num_puzzles:
                    gif_map_idx = int(np.random.randint(0, num_puzzles))
                pw_binding.vec_set_puzzle_indices(
                    driver.c_envs,
                    [gif_map_idx] * driver.num_agents,
                )

        if gif_seed is not None:
            obs, _ = eval_env.reset(seed=gif_seed)
        else:
            obs, _ = eval_env.reset()

        policy = policy or self.uncompiled_policy
        state = {}
        if eval_args['train']['use_rnn']:
            state = dict(
                lstm_h=torch.zeros(eval_env.num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(eval_env.num_agents, policy.hidden_size, device=device),
            )

        frames = []
        was_training = policy.training
        policy.eval()
        with torch.no_grad():
            while len(frames) < gif_frames:
                frame = driver.render()
                if frame is not None:
                    frames.append(frame)

                obs_t = torch.as_tensor(obs).to(device)
                logits, _ = policy.forward_eval(obs_t, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
                action = action.cpu().numpy().reshape(eval_env.action_space.shape)
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)

                obs, _, terminals, truncations, _ = eval_env.step(action)
                if np.any(terminals) or np.any(truncations):
                    obs, _ = eval_env.reset()
                    if eval_args['train']['use_rnn']:
                        state['lstm_h'].zero_()
                        state['lstm_c'].zero_()

        if was_training:
            policy.train()
        eval_env.close()

        if not frames:
            return

        import imageio
        gif_path = os.path.join(gif_dir, f'{env_name}_ep{self.epoch:06d}.gif')
        imageio.mimsave(gif_path, frames, fps=gif_fps, loop=0)

        if hasattr(self.logger, "wandb") and self.logger.wandb:
            import wandb
            self.logger.wandb.log(
                {'render/gif': wandb.Video(gif_path, fps=gif_fps, format="gif")},
                step=self.global_step
            )

    def evaluate_maps(self, policy=None):
        config = self.config
        interval_steps = int(config.get('map_eval_interval_steps', 0))
        interval_epochs = int(config.get('map_eval_interval', 0))
        if interval_steps <= 0 and interval_epochs <= 0:
            return {}

        if interval_steps > 0:
            if (self.global_step - self.last_map_eval_step) < interval_steps:
                return {}
        elif interval_epochs > 0 and (self.epoch % interval_epochs) != 0:
            return {}

        env_name = self.config['env']
        from copy import deepcopy
        from pufferlib.ocean.pushworld import binding as pw_binding

        eval_batch = int(config.get('map_eval_batch', 64))
        eval_batch = max(1, eval_batch)

        def build_eval_env(puzzle_dir=None):
            eval_args = deepcopy(self.full_config)
            eval_args['vec'] = dict(backend='PufferEnv', num_envs=1)
            eval_args['env'] = dict(eval_args['env'])
            eval_args['env']['num_envs'] = eval_batch
            eval_args['env']['render_mode'] = 'None'
            eval_args['env']['render_full_map'] = False
            eval_args['env'].pop('train_map', None)
            eval_args['env'].pop('train_map_idx', None)
            if puzzle_dir:
                eval_args['env']['puzzle_dir'] = puzzle_dir
            return load_env(env_name, eval_args)

        if self.map_eval_env is None or self.map_eval_env.num_agents != eval_batch:
            self.map_eval_env = build_eval_env()

        eval_env = self.map_eval_env
        num_puzzles = int(pw_binding.vec_num_puzzles(eval_env.c_envs))

        policy = policy or self.uncompiled_policy
        device = config['device']
        max_steps = int(self.full_config['env'].get('max_episode_length', 500))
        def run_eval(eval_env, num_puzzles):
            if num_puzzles <= 0:
                return 0, 0

            batch = eval_env.num_agents
            puzzle_indices = np.full(batch, -1, dtype=np.int32)
            pending = np.zeros(batch, dtype=bool)

            state = {}
            if config['use_rnn']:
                state = dict(
                    lstm_h=torch.zeros(batch, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(batch, policy.hidden_size, device=device),
                )

            total = 0
            solved = 0
            for start in range(0, num_puzzles, batch):
                count = min(batch, num_puzzles - start)
                puzzle_indices[:count] = np.arange(start, start + count, dtype=np.int32)
                puzzle_indices[count:] = -1
                pending[:] = False
                pending[:count] = True

                pw_binding.vec_set_puzzle_indices(eval_env.c_envs, puzzle_indices.tolist())
                obs, _ = eval_env.reset()

                steps = 0
                while pending.any() and steps < max_steps:
                    obs_t = torch.as_tensor(obs).to(device)
                    if config['use_rnn']:
                        logits, _ = policy.forward_eval(obs_t, state)
                    else:
                        logits, _ = policy.forward_eval(obs_t)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits)
                    action = action.cpu().numpy().reshape(eval_env.action_space.shape)
                    if isinstance(logits, torch.distributions.Normal):
                        action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)

                    obs, _, terminals, truncations, _ = eval_env.step(action)
                    terminals = np.asarray(terminals) | np.asarray(truncations)
                    last_solved = np.asarray(pw_binding.vec_last_solved(eval_env.c_envs)).astype(bool)

                    newly_done = pending & terminals
                    if np.any(newly_done):
                        total += int(np.sum(newly_done))
                        solved += int(np.sum(newly_done & last_solved))
                        pending[newly_done] = False
                    steps += 1

            return solved, total

        was_training = policy.training
        policy.eval()
        logs = {}
        with torch.no_grad():
            solved, total = run_eval(eval_env, num_puzzles)
            if total > 0:
                logs['environment/map_eval_solved_rate'] = solved / total
                logs['environment/map_eval_puzzles'] = total

            test_dir = config.get('map_eval_test_dir', '')
            if test_dir:
                test_dir = os.path.expanduser(test_dir)
                if os.path.isdir(test_dir):
                    if self.map_eval_env_test is None or self.map_eval_env_test.num_agents != eval_batch:
                        self.map_eval_env_test = build_eval_env(puzzle_dir=test_dir)
                    eval_env_test = self.map_eval_env_test
                    num_puzzles_test = int(pw_binding.vec_num_puzzles(eval_env_test.c_envs))
                    solved_test, total_test = run_eval(eval_env_test, num_puzzles_test)
                    if total_test > 0:
                        logs['environment/map_eval_test_solved_rate'] = solved_test / total_test

        if was_training:
            policy.train()

        if logs:
            self.last_map_eval_step = self.global_step
        return logs

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[dim default]', b1='[bright_cyan]', b2='[default]'):
        config = self.config
        sps = dist_sum(self.sps, config['device'])
        agent_steps = dist_sum(self.global_step, config['device'])
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True,
            show_header=False, border_style='bright_cyan')
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f'{b1}PufferLib {b2}3.0 {idx[0]*" "}:blowfish:',
            f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
            f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
            f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
            f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
        )
        idx[0] = (idx[0] - 1) % 10
            
        s = Table(box=None, expand=True)
        remaining = f'{b2}A hair past a freckle{c2}'
        if sps != 0:
            remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{b2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{b2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{b2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{b2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{b2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{b2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{b2}Remaining', remaining)

        delta = profile.eval['buffer'] + profile.train['buffer']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf('  Env', b2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf('  Learn', b2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.train_misc, b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version.'''

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return f'{b2}{num}{c2}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}K'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}M'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}B'
    else:
        return f'{b2}{num/1e12:.2f}{c2}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()

class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]['start'] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile['start']
        profile['delta'] += delta
        # Multiply delta by freq to account for skipped epochs
        profile['elapsed'] += delta * self.frequency

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof['delta'] > 0:
                prof['buffer'] = prof['delta']
                prof['delta'] = 0

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                   time.sleep(self.delay)
                   continue

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100*(total-free)/total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]

class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100*time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path, early_stop):
        pass

class NeptuneLogger:
    def __init__(self, args, load_id=None, mode='async'):
        import neptune as nept
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune['model'].track_files(model_path)

    def close(self, model_path, early_stop):
        self.neptune['early_stop'] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination='artifacts')
        return f'artifacts/{self.run_id}.pt'
 
class WandbLogger:
    def __init__(self, args, load_id=None, resume='allow'):
        import wandb
        run_name = args.get('wandb_run_name', None)
        wandb.init(
            id=load_id or wandb.util.generate_id(),
            name=run_name,
            project=args['wandb_project'],
            group=args['wandb_group'],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags = [args['tag']] if args['tag'] is not None else [],
            settings=wandb.Settings(console="off"),  # stop sending dashboard to wandb
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path, early_stop):
        self.wandb.run.summary['early_stop'] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'

def train(env_name, args=None, vecenv=None, policy=None, logger=None, early_stop_fn=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print("World size", world_size)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)
    
    wm_cont_obs = args['train'].get('wm_continuous_obs_size', 1)
    wm_num_tiles = args['train'].get('wm_num_tile_types', 10)
    world_model = pufferlib.models.TinyWorldModel(
        observation_size=np.prod(vecenv.single_observation_space.shape),
        action_size=vecenv.single_action_space.n,
        continuous_obs_size=wm_cont_obs,
        num_tile_types=wm_num_tiles,
    )
    
    if 'LOCAL_RANK' in os.environ:
        args['train']['device'] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend='nccl', world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, 'lstm'):
            #model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)

    train_config = { **args['train'], 'env': env_name }
    pufferl = PuffeRL(train_config, args, vecenv, policy, world_model, logger)

    # Sweep needs data for early stopped runs, so send data when steps > 100M
    logging_threshold = min(0.20*train_config['total_timesteps'], 100_000_000)
    all_logs = []

    while pufferl.global_step < train_config['total_timesteps']:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
                        
            should_stop_early = False
            if early_stop_fn is not None:
                should_stop_early = early_stop_fn(logs)
                # This is hacky, but need to see if threshold looks reasonable
                if 'early_stop_threshold' in logs:
                    pufferl.logger.log({'environment/early_stop_threshold': logs['early_stop_threshold']}, logs['agent_steps'])

            stop_on_solve = train_config.get('stop_on_solve', True)
            min_solve_steps = train_config.get('min_solve_steps', 0)
            if stop_on_solve and pufferl.solved_at_step is not None and pufferl.global_step >= min_solve_steps:
                should_stop_early = True
                # Log last video at solve time

            if pufferl.global_step > logging_threshold:
                all_logs.append(logs)

            if should_stop_early:
                model_path = pufferl.close()
                pufferl.logger.close(model_path, early_stop=True)
                return all_logs

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    for i in range(128):  # Run eval for at least 32, but put a hard stop at 128.
        stats = pufferl.evaluate()
        if i >= 32 and stats:
            break

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs

def controlled_exp(env_name, args=None):
    """Run experiments with all combinations of specified parameter values."""
    import itertools
    from copy import deepcopy

    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Targeted experiments require either wandb or neptune")

    # Check if controlled_exp config exists
    if "controlled_exp" not in args:
        raise pufferlib.APIUsageError("No [controlled_exp.*] sections found in config")

    # Extract parameters from controlled_exp namespace
    params = {}
    for section, section_config in args["controlled_exp"].items():
        if isinstance(section_config, dict):
            for param, param_config in section_config.items():
                if isinstance(param_config, dict) and "values" in param_config:
                    params[f"{section}.{param}"] = param_config["values"]

    if not params:
        raise pufferlib.APIUsageError("No parameters with 'values' lists found in [controlled_exp.*] sections")

    # Generate all combinations
    keys = list(params.keys())
    combinations = list(itertools.product(*[params[k] for k in keys]))

    print(f"Running a total of {len(combinations)} experiments with parameters: {keys}")

    # Run each combination
    for i, combo in enumerate(combinations, 1):
        exp_args = deepcopy(args)

        # Set parameters
        for key, value in zip(keys, combo):
            section, param = key.split(".")
            exp_args[section][param] = value

        run_name_parts = [f"{key.split('.')[-1]}={value}" for key, value in zip(keys, combo)]
        exp_name = "_".join(run_name_parts)
        exp_args['wandb_run_name'] = f"{exp_name}"

        print(f"\nExperiment {i}/{len(combinations)}: {dict(zip(keys, combo))}")

        # Train
        train(env_name, args=exp_args)

    print(f"\n✓ Completed all {len(combinations)} experiments")

def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args['vec']['backend']
    if backend != 'PufferEnv':
        backend = 'Serial'

    args['vec'] = dict(backend=backend, num_envs=1)
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args['train']['device']

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    while True:
        render = driver.render()
        if len(frames) < args['save_frames']:
            frames.append(render)

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1/args['fps'])
        elif driver.render_mode == 'rgb_array':
            pass
            #import cv2
            #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #cv2.imshow('frame', render)
            #cv2.waitKey(1)
            #time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if len(frames) > 0 and len(frames) == args['save_frames']:
            import imageio
            imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
            print(f'Saved {len(frames)} frames to {args["gif_path"]}')

def stop_if_loss_nan(logs):
    return any("losses/" in k and np.isnan(v) for k, v in logs.items())

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args['wandb'] and not args['neptune']:
        raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')
    args['no_model_upload'] = True  # Uploading trained model during sweep crashed wandb

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'
    running_target_buffer = deque(maxlen=30)

    def stop_if_perf_below(logs):
        if stop_if_loss_nan(logs):
            logs['is_loss_nan'] = True
            return True

        if method != 'Protein':
            return False

        if ('uptime' in logs and target_key in logs):
            metric_val, cost = logs[target_key], logs['uptime']
            running_target_buffer.append(metric_val)
            target_running_mean = np.mean(running_target_buffer)
            
            # If metric distribution is percentile, threshold is also logit transformed
            threshold = sweep.get_early_stop_threshold(cost)
            logs['early_stop_threshold'] = max(threshold, -5)  # clipping for visualization

            if sweep.should_stop(max(target_running_mean, metric_val), cost):
                logs['is_loss_nan'] = False
                return True
        return False

    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # In the first run, skip sweep and use the train args specified in the config
        if i > 0:
            sweep.suggest(args)

        all_logs = train(env_name, args=args, early_stop_fn=stop_if_perf_below)
        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

        is_final_loss_nan = all_logs[-1].get('is_loss_nan', False)
        if is_final_loss_nan:
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args['train'], env=args['env_name'], tag=args['tag'])

    pufferl = PuffeRL(train_config, args, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    prof.export_chrome_trace("trace.json")

def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    args['vec'] = dict(backend='Serial', num_envs=1)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)
    weights.tofile(path)
    print(f'Saved {len(weights)} weights to {path}')

def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    env_name = args['env_name']
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
 
def load_env(env_name, args):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])

def load_policy(args, vecenv, env_name=''):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    device = args['train']['device']
    policy_cls = getattr(env_module.torch, args['policy_name'])
    policy = policy_cls(vecenv.driver_env, **args['policy'])

    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['rnn'])

    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['neptune']:
            path = NeptuneLogger(args, load_id, mode='read-only').download()
        elif args['wandb']:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        #state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        #optim_state = torch.load(state_path)['optimizer_state_dict']
        #pufferl.optimizer.load_state_dict(optim_state)

    return policy

def load_config(env_name, parser=None):
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

    return process_config(p, parser=parser)

def load_config_file(file_path, fill_in_default=True, parser=None):
    if not os.path.exists(file_path):
        raise pufferlib.APIUsageError('No config file found')

    config_paths = [file_path]

    if fill_in_default:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
        # Process the puffer defaults first
        config_paths.insert(0, os.path.join(puffer_dir, 'config/default.ini'))

    p = configparser.ConfigParser()
    p.read(config_paths)

    return process_config(p, parser=parser)

def make_parser():
    '''Creates the argument parser with default PufferLib arguments.'''
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
    parser.add_argument('--neptune-name', type=str, default='pufferai')
    parser.add_argument('--neptune-project', type=str, default='ablations')
    parser.add_argument('--no-model-upload', action='store_true', help='Do not upload models to wandb or neptune')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    return parser

def process_config(config, parser=None):
    if parser is None:
        parser = make_parser()

    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    def auto_type(value):
        """Type inference for numeric args that use 'auto' as a default value"""
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        return float(value)

    for section in config.sections():
        for key in config[section]:
            try:
                value = ast.literal_eval(config[section][key])
            except:
                value = config[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=value,
                type=auto_type if value == 'auto' else type(value)
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['env'] = args['env_name'] or ''  # for trainer dashboard
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = "Usage: puffer [train, eval, sweep, controlled_exp, autotune, profile, export, sanity] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        eval(env_name=env_name)
    elif mode == "sweep":
        sweep(env_name=env_name)
    elif mode == "controlled_exp":
        controlled_exp(env_name=env_name)
    elif mode == "autotune":
        autotune(env_name=env_name)
    elif mode == "profile":
        profile(env_name=env_name)
    elif mode == "export":
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == '__main__':
    main()
