from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces

class TinyWorldModel(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=128,
                 continuous_obs_size=1, num_tile_types=10):
        super().__init__()

        self.cont_obs_size = int(continuous_obs_size)
        if self.cont_obs_size < 0:
            raise ValueError("continuous_obs_size must be >= 0")
        self.cat_obs_size = observation_size - self.cont_obs_size
        if self.cat_obs_size < 0:
            raise ValueError("categorical obs size must be >= 0")
        self.num_tile_types = int(num_tile_types)
        self.input_size = self.cat_obs_size * self.num_tile_types + self.cont_obs_size + action_size
        self.num_actions = action_size
        
        # Separate heads for categorical and continuous predictions
        self.shared = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.input_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        
        # Output logits for each categorical position
        self.categorical_head = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, self.cat_obs_size * self.num_tile_types)
        )
        
        # Output continuous values (optional)
        self.continuous_head = None
        if self.cont_obs_size > 0:
            self.continuous_head = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, self.cont_obs_size)
            )

    def forward(self, observations, actions):
        '''
        Predict next observations given current observations and actions.
        '''
        
        categorical_obs = observations[:, :, :self.cat_obs_size]
        continuous_obs = None
        if self.cont_obs_size > 0:
            continuous_obs = observations[:, :, self.cat_obs_size:]
        
        feature_dtype = self.shared[0].weight.dtype
        categorical_obs_onehot = torch.nn.functional.one_hot(
            categorical_obs.long(),
            num_classes=self.num_tile_types
        ).to(feature_dtype)
        
        categorical_obs_flat = categorical_obs_onehot.flatten(start_dim=-2)
        actions_onehot = torch.nn.functional.one_hot(
            actions.long(), num_classes=self.num_actions
        ).to(feature_dtype)
        if continuous_obs is not None:
            continuous_obs = continuous_obs.to(feature_dtype)
        
        if continuous_obs is None:
            x = torch.cat([categorical_obs_flat, actions_onehot], dim=-1)
        else:
            x = torch.cat([categorical_obs_flat, continuous_obs, actions_onehot], dim=-1)
        features = self.shared(x)
        
        # Reshape categorical logits: (batch, seq, positions, classes)
        categorical_logits = self.categorical_head(features).reshape(
            -1, observations.shape[1], self.cat_obs_size, self.num_tile_types
        )
        continuous_pred = None
        if self.continuous_head is not None:
            continuous_pred = self.continuous_head(features)
        
        return categorical_logits, continuous_pred

class BTDUnlockPotential(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_size,
        latent_dim=64,
        encoder_type="mlp",
        encoder_hidden=256,
        conv_channels=32,
        grid_size=0,
        action_embed_dim=16,
        dynamics_hidden=256,
        phi_hidden=128,
        latent_norm="layernorm",
        spread_eps=1e-6,
        branching_metric="log",
    ):
        super().__init__()
        self.num_actions = int(action_size)
        self.latent_dim = int(latent_dim)
        self.encoder_type = encoder_type
        self.grid_size = int(grid_size)
        self.latent_norm_type = latent_norm
        self.spread_eps = float(spread_eps)
        self.branching_metric = branching_metric

        obs_dim = int(np.prod(obs_shape))
        if self.encoder_type == "conv":
            if self.grid_size <= 0:
                self.grid_size = int(np.sqrt(obs_dim))
            conv = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Conv2d(1, conv_channels, 3, padding=1)
                ),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(
                    nn.Conv2d(conv_channels, conv_channels, 3, stride=2, padding=1)
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, self.grid_size, self.grid_size)
                conv_out = conv(dummy).shape[-1]
            self.encoder = nn.Sequential(
                conv,
                pufferlib.pytorch.layer_init(nn.Linear(conv_out, self.latent_dim)),
            )
        else:
            self.encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(obs_dim, encoder_hidden)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Linear(encoder_hidden, encoder_hidden)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Linear(encoder_hidden, self.latent_dim)),
            )

        if self.latent_norm_type == "layernorm":
            self.latent_norm = nn.LayerNorm(self.latent_dim)
        else:
            self.latent_norm = None

        self.action_embed = nn.Embedding(self.num_actions, action_embed_dim)
        self.dynamics = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.latent_dim + action_embed_dim, dynamics_hidden)
            ),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(dynamics_hidden, self.latent_dim)),
        )

        self.phi = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.latent_dim, phi_hidden)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(phi_hidden, 1), std=0.01),
        )
        self.phi_target = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.latent_dim, phi_hidden)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(phi_hidden, 1), std=0.01),
        )
        self.phi_target.load_state_dict(self.phi.state_dict())
        for p in self.phi_target.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "action_ids",
            torch.arange(self.num_actions, dtype=torch.long),
            persistent=False,
        )

    def _normalize_latent(self, z):
        if self.latent_norm is not None:
            return self.latent_norm(z)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6)

    def encode(self, observations):
        x = observations.float()
        if self.encoder_type == "conv":
            b = x.shape[0]
            x = x.view(b, 1, self.grid_size, self.grid_size)
        else:
            x = x.view(x.shape[0], -1)
        z = self.encoder(x)
        return self._normalize_latent(z)

    def predict_next(self, z, actions):
        a_emb = self.action_embed(actions)
        delta = self.dynamics(torch.cat([z, a_emb], dim=-1))
        return self._normalize_latent(z + delta)

    def branching(self, z):
        batch, dim = z.shape
        z_rep = z.unsqueeze(1).expand(batch, self.num_actions, dim).reshape(-1, dim)
        a_rep = self.action_ids.repeat(batch)
        z_next = self.predict_next(z_rep, a_rep).view(batch, self.num_actions, dim)
        mu = z_next.mean(dim=1)
        diff = z_next - mu[:, None, :]
        spread = diff.pow(2).sum(dim=-1).mean(dim=1)
        if self.branching_metric == "sqrt":
            return torch.sqrt(spread + self.spread_eps)
        return torch.log(spread + self.spread_eps)

    def update_target(self, tau):
        for tgt, src in zip(self.phi_target.parameters(), self.phi.parameters()):
            tgt.data.mul_(tau).add_(src.data, alpha=1 - tau)

class Default(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            num_obs = np.prod(env.single_observation_space.shape) #1211
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
                nn.GELU(),
            )
            
        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, num_atns), std=0.01)
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_atns), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)

            # # Note: temp for grid env only
            # categorical_obs = observations[:, :121]
            # continuous_obs = observations[:, 121:]
            
            # categorical_obs_onehot = torch.nn.functional.one_hot(
            #     categorical_obs.long(), 
            #     num_classes=10,
            # ) 
            
            # categorical_obs_flat = categorical_obs_onehot.flatten(start_dim=-2) 
            
            # observations = torch.cat([categorical_obs_flat, continuous_obs], dim=-1).view(batch_size, -1)
            
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values

class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128):
        '''Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example.'''
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if 'layer_norm' in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        #self.pre_layernorm = nn.LayerNorm(hidden_size)
        #self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward_eval(self, observations, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        hidden = self.policy.encode_observations(observations, state=state)
        h = state['lstm_h']
        c = state['lstm_c']

        # TODO: Don't break compile
        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0], 'LSTM state must be (h, c)'
            lstm_state = (h, c)
        else:
            lstm_state = None

        #hidden = self.pre_layernorm(hidden)
        hidden, c = self.cell(hidden, lstm_state)
        #hidden = self.post_layernorm(hidden)
        state['hidden'] = hidden
        state['lstm_h'] = hidden
        state['lstm_c'] = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x = observations
        lstm_h = state['lstm_h']
        lstm_c = state['lstm_c']

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if lstm_h is not None:
            assert lstm_h.shape[1] == lstm_c.shape[1] == B, 'LSTM state must be (h, c)'
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        x = x.reshape(B*TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)

        hidden = hidden.transpose(0, 1)
        #hidden = self.pre_layernorm(hidden)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        hidden = hidden.float()
 
        #hidden = self.post_layernorm(hidden)
        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(B*TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        #state.batch_logits = logits.reshape(B, TT, -1)
        state['hidden'] = hidden
        state['lstm_h'] = lstm_h.detach()
        state['lstm_c'] = lstm_c.detach()
        return logits, values

class Convolutional(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample

        #TODO: Remove these from required params
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class ProcgenResnet(nn.Module):
    '''Procgen baseline from the AICrowd NeurIPS 2020 competition
    Based on the ResNet architecture that was used in the Impala paper.'''
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden
 
    def decode_actions(self, hidden):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
