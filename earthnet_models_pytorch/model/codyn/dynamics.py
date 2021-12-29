


from typing import Optional, Union

import abc

from functools import partial

import torch

import torch.distributions as distrib
from torch import nn
import torch.nn.functional as F

from earthnet_models_pytorch.model.codyn.base import MLP, init_weight


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=-1, eps=1e-8):
    """
    Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int
        Dimensions of raw_params so that the first half corresponds to the mean, and the second half to the scale.
    eps : float
        Minimum possible value of the final scale parameter.

    Returns
    -------
    torch.distributions.Normal
        Normal distributions with the input mean and eps + softplus(raw scale) * scale_stddev as scale.
    """
    loc, raw_scale = torch.chunk(raw_params, 2, dim)
    assert loc.shape[dim] == raw_scale.shape[dim]
    scale = F.softplus(raw_scale) + eps
    normal = distrib.Normal(loc, scale * scale_stddev)
    return normal


def rsample_normal(raw_params, scale_stddev=1):
    """
    Samples from a normal distribution with given parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    """
    normal = make_normal_from_raw_params(raw_params, scale_stddev=scale_stddev)
    sample = normal.rsample()
    return sample


class BaseDynamics(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encodings: list, pred_start: int = 0, n_preds: Optional[int] = None):
        
        # Initial hidden state

        state = self.extract_state(encodings, pred_start = pred_start)
        if isinstance(state, tuple):
            state, state_params = state
        else:
            state_params = None
        # Infer parameters: Memory in predictors, possible variational inference

        params = [inf(enc, pred_start = pred_start, n_preds = n_preds) for inf, enc, use in zip(self.infer_preds, encodings, self.use_encs) if use]

        # Update state

        outs, aux_loss_pars = self.update_state(params, state, pred_start = pred_start, n_preds = n_preds)

        if state_params is not None:
            aux_loss_pars['state_params'] = state_params

        return outs, aux_loss_pars

class DeterministicMLPStateExtraction(nn.Module):
    def __init__(self, use_n_steps: int, input_frequencies: list, input_channels: list, hidden_fusion_channels: list, hidden_channels: int, output_channels: int, n_layers: int, activation: list = [None,'relu']):
        super().__init__()

        self.use_n_steps = use_n_steps
        self.input_frequencies = input_frequencies

        self.fuse_mlps = nn.ModuleList([MLP(f*c, 0, h, 1, activation = activation[0]) for c,h,f in zip(input_channels, hidden_fusion_channels, input_frequencies)])

        self.mlp = MLP(use_n_steps*sum(hidden_fusion_channels), hidden_channels, output_channels, n_layers-1, activation = activation[1:])

    def forward(self, encodings, pred_start: int = 0):

        assert(pred_start >= self.use_n_steps or self.training)

        encs = []
        for i, (enc, f) in enumerate(zip(encodings, self.input_frequencies)):
            enc = enc[:,:self.use_n_steps*f,...]
            b, t, c = enc.shape
            enc = enc.reshape(-1, c*f)
            enc = self.fuse_mlps[i](enc)
            encs.append(enc.reshape(b,-1,enc.shape[-1]))
        encs = torch.cat(encs, dim = -1)
        b, t, c = encs.shape
        return self.mlp(encs.reshape(b,t*c))

class StochasticMLPStateExtraction(nn.Module):
    def __init__(self, use_n_steps: int, input_frequencies: list, input_channels: list, hidden_fusion_channels: list, hidden_channels: int, output_channels: int, n_layers: int, activation: list = [None,'relu']):
        super().__init__()

        self.use_n_steps = use_n_steps
        self.input_frequencies = input_frequencies

        self.fuse_mlps = nn.ModuleList([MLP(f*c, 0, h, 1, activation = activation[0]) for c,h,f in zip(input_channels, hidden_fusion_channels, input_frequencies)])

        self.mlp = MLP(use_n_steps*sum(hidden_fusion_channels), hidden_channels, 2*output_channels, n_layers-1, activation = activation[1:])

    def forward(self, encodings, pred_start: int = 0):

        assert(pred_start >= self.use_n_steps or self.training)

        encs = []
        for i, (enc, f) in enumerate(zip(encodings, self.input_frequencies)):
            enc = enc[:,:self.use_n_steps*f,...]
            b, t, c = enc.shape
            enc = enc.reshape(-1, c*f)
            enc = self.fuse_mlps[i](enc)
            encs.append(enc.reshape(b,-1,enc.shape[-1]))
        encs = torch.cat(encs, dim = -1)
        b, t, c = encs.shape
        params = self.mlp(encs.reshape(b,t*c))
        return rsample_normal(params), params


STATE_EXTRACTIONS = {"stochasticmlp": StochasticMLPStateExtraction, "deterministicmlp": DeterministicMLPStateExtraction}


class DeterministicLSTMPredInference(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, aggregation_frequency: int = 1, n_mlp_layers: int = 1, mlp_activation = None):
        super().__init__()

        self.lstm = nn.LSTM(input_channels, hidden_channels, 1)

        self.aggregation_frequency = aggregation_frequency

        self.mlp = MLP(aggregation_frequency*hidden_channels, (n_mlp_layers != 1) * output_channels, output_channels, n_mlp_layers, mlp_activation)

    def forward(self, encodings, pred_start: int = 0, n_preds: Optional[int] = None):

        encs = self.lstm(encodings)[0]
        b, t, c = encs.shape
        encs = encs.reshape(-1, c*self.aggregation_frequency)
        outs = self.mlp(encs)
        return outs.reshape(b,t//self.aggregation_frequency,outs.shape[-1])

class StochasticLSTMPredInference(DeterministicLSTMPredInference):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, aggregation_frequency: int = 1, n_mlp_layers: int = 1, mlp_activation = None):
        super().__init__(input_channels = input_channels, hidden_channels = hidden_channels, output_channels = output_channels, aggregation_frequency = aggregation_frequency, n_mlp_layers = n_mlp_layers, mlp_activation = mlp_activation)

        self.mlp = MLP(aggregation_frequency*hidden_channels, (n_mlp_layers != 1) * output_channels, 2*output_channels, n_mlp_layers, mlp_activation)

class ThroughpassPredInference(nn.Module):
    def __init__(self, context_length: int = 10, aggregation_frequency: int = 1):
        super().__init__()
        self.aggregation_frequency = aggregation_frequency
        self.context_length = context_length
    def forward(self,  encodings, pred_start: int = 0, n_preds: Optional[int] = None):
        pred_start = self.context_length if self.training else pred_start
        encodings = encodings[:,pred_start*self.aggregation_frequency:,:]
        b, t, c = encodings.shape
        return encodings.reshape(b,t//self.aggregation_frequency,c*self.aggregation_frequency)

PRED_INFERENCES = {"stochasticlstm": StochasticLSTMPredInference, "deterministiclstm": DeterministicLSTMPredInference, "throughpass": ThroughpassPredInference}

class LSTMUpdate(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_channels, hidden_channels, num_layers = num_layers, batch_first = True)
        self.num_layers = num_layers

    def forward(self, params, state, pred_start: int = 0, n_preds: Optional[int] = None):
        b, c = state.shape
        h_0, c_0 = torch.chunk(state.reshape(b, self.num_layers, c//self.num_layers).permute(1,0,2), 2, dim = 2)
        out, (h_n, c_n) = self.lstm(params[0], (h_0.contiguous(), c_0.contiguous()))
        return out, {}

class FirstOrderUpdate(nn.Module):
    def __init__(self, sample_params: list, input_frequencies: list, state_channels: int, prior_hidden_channels: int, pred_channels: list, prior_layers: int, prior_activation: list, update_hidden_channels: int, update_layers: int, update_activation: list, init_args: dict = {"init_type": 'orthogonal', "init_gain": 1.01}):
        super().__init__()
        self.sample_params = sample_params

        self.input_frequencies = input_frequencies

        self.prior_mlps = nn.ModuleList([MLP(state_channels, prior_hidden_channels, 2*pred_channels[i], prior_layers, prior_activation) for i in range(len(sample_params)) if sample_params[i]])

        self.update_mlp = MLP(sum(pred_channels), update_hidden_channels, state_channels, update_layers, update_activation)
        update_init = partial(init_weight, **init_args)
        self.update_mlp.apply(update_init)

    def forward(self, params, state, pred_start: int = 0, n_preds: Optional[int] = None):
        if n_preds is None: n_preds = params[0].shape[1]
        if self.training:
            states = [state]
            n_preds = n_preds - 1
        else:
            states = []
        
        infer_q_params = []
        for i in range(len(params)):
            if self.sample_params[i]:
                infer_q_params.append(params[i].permute(1,0,2)[self.training:,...])
                params[i] = rsample_normal(params[i])
            params[i] = params[i][:,:(params[i].shape[1]//self.input_frequencies[i])*self.input_frequencies[i],...]
        
        infer_p_params = [[] for _ in [s for s in self.sample_params if s]]
        updates = []
        for i in range(pred_start, pred_start+n_preds):
            inputs = []
            for j in range(len(self.sample_params)):
                t_pars = params[j].shape[1]//self.input_frequencies[j]
                if self.sample_params[j]:
                    if self.training or i >= t_pars:
                        p_params = self.prior_mlps[j](state)
                        if self.training:
                            infer_p_params[j].append(p_params)
                        if i >= t_pars:
                            params[j] = torch.cat([params[j]]+[rsample_normal(p_params).unsqueeze(1) for _ in range(self.input_frequencies[j])], dim = 1)
                inputs.append(params[j][:,t_pars:(t_pars + self.input_frequencies[j]),...].sum(1))
            inputs = torch.cat(inputs, -1)
            
            update = self.update_mlp(inputs)
            updates.append(update)
            state = state + update
            states.append(state)

        if self.training:
            infer_p_params = [torch.stack(p) for p in infer_p_params]

            return torch.stack(states, dim = 1), {"infer_q_params": infer_q_params, "infer_p_params": infer_p_params, "updates": torch.stack(updates, dim = 1)}
        else:
            return torch.stack(states, dim = 1), {"updates": torch.stack(updates, dim = 1)}

STATE_UPDATES = {"firstorder": FirstOrderUpdate, "lstm": LSTMUpdate}

class SimpleDynamics(BaseDynamics):

    def __init__(self, state_extraction_args, pred_inference_args, state_update_args):
        super().__init__()

        self.extract_state = STATE_EXTRACTIONS[state_extraction_args["name"]](**state_extraction_args["args"])

        self.use_encs = [s is not None for s in pred_inference_args]
        self.infer_preds = nn.ModuleList([MLP(1,0,1,1) if s is None else PRED_INFERENCES[s["name"]](**s["args"]) for s in pred_inference_args])

        self.update_state = STATE_UPDATES[state_update_args["name"]](**state_update_args["args"])




ALL_DYNAMICS = {"simple": SimpleDynamics}

def setup_dynamics(setting):
    return ALL_DYNAMICS[setting["name"]](**setting["args"])