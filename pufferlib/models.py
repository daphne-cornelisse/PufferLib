from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces

class TinyWorldModel(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=128, cat_threshold=0.1, cont_threshold=0.01):
        super().__init__()
        
        self.cat_obs_size = observation_size - 3 # Grid elements are classes
        self.cont_obs_size = 3   # Agent direction is continuous
        self.num_tile_types = 10  # EMPTY through WALL_4 (0-9)
        self.input_size = self.cat_obs_size * self.num_tile_types + self.cont_obs_size + action_size
        self.num_actions = action_size
        self.hidden_size = hidden_size
        self.cat_threshold = cat_threshold
        self.cont_threshold = cont_threshold
        
        # Separate heads for categorical and continuous predictions
        self.shared = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.input_size, hidden_size)),
            nn.LayerNorm(hidden_size), 
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size), 
            nn.ReLU(),
        )
        
        # GRU for WM memory — nn.GRU for training, GRUCell for eval (shared weights)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.gru_cell.weight_ih = self.gru.weight_ih_l0
        self.gru_cell.weight_hh = self.gru.weight_hh_l0
        self.gru_cell.bias_ih   = self.gru.bias_ih_l0
        self.gru_cell.bias_hh   = self.gru.bias_hh_l0

        # Output logits for each categorical position
        self.categorical_head = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size // 2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size // 2, self.cat_obs_size * self.num_tile_types))
        )
        
        self.continuous_head = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size // 2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size // 2, 1))
        )

    def forward(self, observations, actions, gru_h=None):
        '''
        Training forward — processes full sequences using nn.GRU.

        Args:
            observations : (batch, seq, obs_size)
            actions      : (batch, seq)
            gru_h        : (1, batch, hidden) or None

        Returns:
            categorical_logits : (batch, seq, cat_obs_size, num_tile_types)
            continuous_pred    : (batch, seq, 1)
            gru_h              : (1, batch, hidden)
        '''
        batch_size, seq_len = observations.shape[:2]

        categorical_obs = observations[:, :, :self.cat_obs_size]
        continuous_obs  = observations[:, :, self.cat_obs_size:].clone()
        continuous_obs[:, :, 1:] = 0  
        
        categorical_obs_onehot = torch.nn.functional.one_hot(
            categorical_obs.long(), num_classes=self.num_tile_types
        )
        categorical_obs_flat = categorical_obs_onehot.flatten(start_dim=-2).float()
        actions_onehot = torch.nn.functional.one_hot(
            actions.long(), num_classes=self.num_actions
        ).float()
        
        x = torch.cat([categorical_obs_flat, continuous_obs, actions_onehot], dim=-1)
        features = self.shared(x.reshape(batch_size * seq_len, -1))
        features = features.reshape(batch_size, seq_len, self.hidden_size)

        if gru_h is None:
            gru_h = torch.zeros(1, batch_size, self.hidden_size, device=observations.device)

        out, gru_h = self.gru(features, gru_h)  # out: (batch, seq, hidden)

        categorical_logits = self.categorical_head(out).reshape(
            batch_size, seq_len, self.cat_obs_size, self.num_tile_types
        )
        continuous_pred = self.continuous_head(out).reshape(batch_size, seq_len, 1)
        
        return categorical_logits, continuous_pred, gru_h

    def forward_step(self, observation, action, gru_h=None):
        '''
        Single-step eval forward — uses GRUCell, faster than nn.GRU.

        Args:
            observation : (batch, obs_size)
            action      : (batch,)
            gru_h       : (batch, hidden) or None

        Returns:
            categorical_logits : (batch, cat_obs_size, num_tile_types)
            continuous_pred    : (batch, 1)
            gru_h              : (batch, hidden)
        '''
        batch_size = observation.shape[0]

        cat_obs  = observation[:, :self.cat_obs_size]
        cont_obs = observation[:, self.cat_obs_size:].clone()
        cont_obs[:, 1:] = 0

        cat_onehot = torch.nn.functional.one_hot(
            cat_obs.long(), num_classes=self.num_tile_types
        ).flatten(start_dim=-2).float()
        act_onehot = torch.nn.functional.one_hot(
            action.long(), num_classes=self.num_actions
        ).float()

        x = torch.cat([cat_onehot, cont_obs, act_onehot], dim=-1)
        features = self.shared(x)  # (batch, hidden)

        if gru_h is None:
            gru_h = torch.zeros(batch_size, self.hidden_size, device=observation.device)

        gru_h = self.gru_cell(features, gru_h)  # (batch, hidden)

        categorical_logits = self.categorical_head(gru_h).reshape(
            batch_size, self.cat_obs_size, self.num_tile_types
        )
        continuous_pred = self.continuous_head(gru_h)

        return categorical_logits, continuous_pred, gru_h

    def compute_prediction_error(self, observations, actions, next_observations,
                              gru_h=None, clip_value=None):
        if observations.ndim == 2:
            observations      = observations.unsqueeze(1)
            actions           = actions.unsqueeze(1)
            next_observations = next_observations.unsqueeze(1)

        # Ensure gru_h is (1, batch, hidden) for nn.GRU
        if gru_h is not None and gru_h.ndim == 2:
            gru_h = gru_h.unsqueeze(0)

        categorical_logits, continuous_pred, gru_h = self.forward(
            observations, actions, gru_h
        )
        
        next_obs_cat  = next_observations[:, :, :self.cat_obs_size].long()
        next_obs_cont = next_observations[:, :, self.cat_obs_size:]
        
        batch_size, seq_len, num_positions = next_obs_cat.shape
        
        categorical_loss = torch.nn.functional.cross_entropy(
            categorical_logits.reshape(-1, self.num_tile_types),
            next_obs_cat.reshape(-1),
            reduction='none'
        )
        cat_loss  = categorical_loss.reshape(batch_size, seq_len, num_positions).mean(dim=-1)
        cont_loss = ((continuous_pred - next_obs_cont[:, :, :1]) ** 2).mean(dim=-1)
        
        # Margin: zero out "good enough" predictions
        # cat threshold: cross-entropy of a confident correct prediction is ~0.1
        #                uniform over 10 classes is ~2.3, so 0.1 is already quite good
        # cont threshold: MSE below 0.01 is essentially perfect for a direction signal
        cat_loss  = torch.relu(cat_loss  - self.cat_threshold)
        cont_loss = torch.relu(cont_loss - self.cont_threshold)

        prediction_error = cat_loss + cont_loss
        
        if clip_value is not None:
            prediction_error = torch.clamp(prediction_error, 0, clip_value)
            
        return prediction_error, gru_h
    
    def compute_intrinsic_reward(self, observations, actions, next_observations,
                                  gru_h=None, reward_coef=1.0, clip_value=None, normalize=False):
        prediction_error, gru_h = self.compute_prediction_error(
            observations, actions, next_observations,
            gru_h=gru_h, clip_value=clip_value
        )
        
        if normalize:
            error_mean = prediction_error.mean()
            error_std  = prediction_error.std() + 1e-8
            normalized_error = (prediction_error - error_mean) / error_std
            shifted = normalized_error - normalized_error.min()
            intrinsic_reward = torch.log(1 + shifted) * reward_coef
        else:
            upper = torch.quantile(prediction_error, 0.95)
            intrinsic_reward = torch.clamp(prediction_error, 0, upper) * reward_coef
            
        return intrinsic_reward, gru_h
    
    def get_predictions_for_logging(self, observations, actions, next_observations, gru_h=None, clip_value=None):
        with torch.no_grad():
            categorical_logits, continuous_pred, gru_h = self.forward(
                observations, actions, gru_h
            )
            categorical_pred = categorical_logits.argmax(dim=-1)
            predictions = torch.cat([categorical_pred.float(), continuous_pred], dim=-1)
            errors, gru_h = self.compute_prediction_error(
                observations, actions, next_observations,
                gru_h=gru_h, clip_value=clip_value
            )
            return predictions, errors, gru_h
    
    @torch.no_grad()
    def visualize_prediction(self, observations, actions, next_observations, gru_h=None, idx=0):
        categorical_logits, continuous_pred, gru_h = self.forward(
            observations, actions, gru_h
        )
        categorical_pred = categorical_logits.argmax(dim=-1)
        predicted = torch.cat([categorical_pred.float(), continuous_pred], dim=-1)
        error, _ = self.compute_prediction_error(
            observations, actions, next_observations, gru_h=gru_h
        )
        return {
            'actual':    next_observations[idx].cpu().numpy(),
            'predicted': predicted[idx].cpu().numpy(),
            'error':     error[idx].cpu().numpy(),
        }


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
