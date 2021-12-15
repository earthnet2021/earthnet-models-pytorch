"""RNN pixelwise
"""

from typing import Optional, Union

import argparse
import ast

import timm
import torch
import torchvision

import segmentation_models_pytorch as smp

from torch import nn

from earthnet_models_pytorch.utils import str2bool



Activations = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh }

Activation_Default_Args = {"relu": {"inplace": True}, "leaky_relu": {"negative_slope": 0.2, "inplace": True}, "elu": {"inplace": True}, "sigmoid": {}, "tanh": {}}


def make_lin_block(ninp, nout, activation):
    """
    Creates a linear block formed by an activation function and a linear operation.

    Parameters
    ----------
    ninp : int
        Input dimension.
    nout : int
        Output dimension.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function before
        the linear operation.
    """
    modules = []
    if activation is not None:
        modules.append(Activations[activation](**Activation_Default_Args[activation]))
    modules.append(nn.Linear(ninp, nout))
    return nn.Sequential(*modules)


class MLP(nn.Module):
    """
    Module implementing an MLP.
    """
    def __init__(self, ninp, nhid, nout, nlayers, activation='relu'):
        """
        Parameters
        ----------
        ninp : int
            Input dimension.
        nhid : int
            Number of dimensions in intermediary layers.
        nout : int
            Output dimension.
        nlayers : int
            Number of layers in the MLP.
        activation : str
            'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function before the
            linear operation.
        """
        super().__init__()
        assert nhid == 0 or nlayers > 1
        if isinstance(activation, str) or activation is None:
            activation = [activation for i in range(nlayers)]
        if 'none' in activation:
            activation = [(act if act != 'none' else None) for act in activation]
        modules = [
            make_lin_block(
                ninp=ninp if il == 0 else nhid,
                nout=nout if il == nlayers - 1 else nhid,
                activation=activation[il],
            ) for il in range(nlayers)
        ]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


class RNN(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg", "en21x", "en21x-px"])

        self.state_mlp = MLP(**self.hparams.state_mlp_args)
        self.update_mlp = MLP(**self.hparams.update_mlp_args)

        self.rnn = torch.nn.GRU(input_size = self.hparams.update_mlp_args["nout"], hidden_size = self.hparams.state_mlp_args["nout"], num_layers = 1, batch_first = True)

        self.head = torch.nn.Linear(in_features = self.hparams.state_mlp_args["nout"], out_features = 1)

        self.sigmoid = nn.Sigmoid()



    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--state_mlp_args", type = ast.literal_eval, default = '{"ninp": 310, "nhid": 384, "nout": 192, "nlayers": 3, "activation": "relu"}')
        parser.add_argument("--update_mlp_args", type = ast.literal_eval, default = '{"ninp": 30, "nhid": 128, "nout": 96, "nlayers": 2, "activation": "relu"}')
        parser.add_argument("--setting", type = str, default = "en21x-px")
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        parser.add_argument("--lc_onehot", type = str2bool, default = False)
        parser.add_argument("--use_dem", type = str2bool, default = True)
        parser.add_argument("--use_soilgrids", type = str2bool, default = True)

        return parser


    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        
        n_preds = 0 if n_preds is None else n_preds

        c_l = self.hparams.context_length if self.training else pred_start

        if self.training or len(data["dynamic"][0].shape) <= 3:
            hr_dynamic_inputs = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]

            last_dynamic_input = hr_dynamic_inputs[:,-1,...]

            b, t, c = hr_dynamic_inputs.shape

            hr_dynamic_inputs = hr_dynamic_inputs.reshape(b, t*c)

            static_inputs = data["static"][0]        
        else:
            hr_dynamic_inputs = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]

            b, t, c, h, w = hr_dynamic_inputs.shape

            hr_dynamic_inputs = hr_dynamic_inputs.permute(0,3,4,1,2).reshape(b*h*w, t, c)

            last_dynamic_input = hr_dynamic_inputs[:,-1,...]

            #b, t, c = hr_dynamic_inputs.shape

            hr_dynamic_inputs = hr_dynamic_inputs.reshape(b*h*w, t*c)

            static_inputs = data["static"][0]

            _, c_s, _, _ = static_inputs.shape

            static_inputs = static_inputs.permute(0,2,3,1).reshape(b*h*w, c_s)

        if self.hparams.use_dem and self.hparams.use_soilgrids:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs), dim = 1)
        elif self.hparams.use_dem:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs[:,0][:,None]), dim = 1)
        elif self.hparams.use_soilgrids:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs[:,1:]), dim = 1)
        else:
            state_inputs = hr_dynamic_inputs
        state = self.state_mlp(state_inputs)

        state[:,0] = last_dynamic_input[:,0]

        state = state[None,:,:]

        if len(data["dynamic"][0].shape) <= 3:
            meso_dynamic_inputs = data["dynamic"][1][:,c_l:,...]

            _, t_m, c_m = meso_dynamic_inputs.shape

            if self.hparams.lc_onehot:
                lc = torch.stack([data["landcover"] == x for x in [82, 83, 102, 103, 104]], dim = 2).repeat(1,t_m, 1)
            else:
                lc = data["landcover"] / 100

                lc = lc[:,None,:].repeat(1,t_m, 1)

            dem = static_inputs[:,0]

            dem = dem[:,None,None].repeat(1,t_m, 1)

        else:
            meso_dynamic_inputs = data["dynamic"][1][:,c_l:,...]
            
            if len(meso_dynamic_inputs.shape) == 5:
                _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

                meso_dynamic_inputs = meso_dynamic_inputs[:,:,:,(h_m//2- 1):(h_m//2),(w_m//2- 1):(w_m//2)].mean((3,4))[:,None,None,...].repeat(1,h, w, 1, 1).reshape(b*h*w, t_m, c_m)
            else:
                _, t_m, c_m = meso_dynamic_inputs.shape
                meso_dynamic_inputs = meso_dynamic_inputs[:,None,None,...].repeat(1,h, w, 1, 1).reshape(b*h*w, t_m, c_m)

            if self.hparams.lc_onehot:
                lc = torch.cat([data["landcover"].reshape(b*h*w, 1, 1) == x for x in [82, 83, 102, 103, 104]], dim = 2).repeat(1,t_m, 1)
            else:
                lc = data["landcover"] / 100

                lc = lc.reshape(b*h*w, 1, 1).repeat(1, t_m, 1)

            dem = static_inputs[:,0,...]

            dem = dem.reshape(b*h*w, 1, 1).repeat(1, t_m, 1)


        update_inputs = torch.cat([meso_dynamic_inputs, dem, lc], axis = -1)

        update = self.update_mlp(update_inputs)


        out, _ = self.rnn(update.contiguous(), state.contiguous())

        out = self.sigmoid(self.head(out))

        if not (self.training or len(data["dynamic"][0].shape) <= 3):
            _, t_o, c_o = out.shape

            out = out.reshape(b,h,w,t_o, c_o).permute(0,3,4,1,2)

        return out, {}

