"""Hybrid GSI
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
from earthnet_models_pytorch.model.rnn import MLP


class HybridGSI(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        assert self.hparams.encoder_args["classes"] == 16

        self.encoder = getattr(smp, self.hparams.encoder_name)(**self.hparams.encoder_args)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()



    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--encoder_name", type = str, default = "FPN")
        parser.add_argument("--encoder_args", type = ast.literal_eval, default = '{"encoder_name": "timm-efficientnet-b4", "encoder_weights": "noisy-student", "in_channels": 191, "classes": 16}')
        parser.add_argument("--train_npixels", type = int, default = 256)
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        parser.add_argument("--use_dem", type = str2bool, default = True)
        parser.add_argument("--lc_min", type = int, default = 2)
        parser.add_argument("--lc_max", type = int, default = 6)
        parser.add_argument("--val_n_splits", type = int, default = 20)

        return parser


    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        
        n_preds = 0 if n_preds is None else n_preds

        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]

        b, t, c, h, w = hr_dynamic_inputs.shape

        hr_dynamic_inputs = hr_dynamic_inputs.reshape(b, t*c, h, w)

        static_inputs = data["static"][0]

        if self.hparams.use_dem:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs), dim = 1)
        else:
            state_inputs = hr_dynamic_inputs

        state_params = self.encoder(state_inputs)

        

        meso_dynamic_inputs = data["dynamic"][1][:,c_l:,...]

        if len(meso_dynamic_inputs.shape) == 5:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape
        else:
            _, t_m, c_m = meso_dynamic_inputs.shape

        # Need t2m, ssr, sm_rootzone, pev

        t2m = meso_dynamic_inputs[:,:,28,...][:,:,None,...]
        ssr = meso_dynamic_inputs[:,:,22,...][:,:,None,...]
        sm_root = meso_dynamic_inputs[:,:,13,...][:,:,None,...]
        pev = meso_dynamic_inputs[:,:,10,...][:,:,None,...]
        water = torch.clamp(sm_root/torch.clamp(pev, min = 1e-4), min = 0, max = 1000)

        meteo = torch.cat([t2m, -t2m, ssr, water], dim = 2) # cold heat light water
        if len(meteo.shape) == 5:
                meteo = meteo[:,:,:,(h_m//2- 1):(h_m//2),(w_m//2- 1):(w_m//2)].mean((3,4))
        meteo = meteo.reshape(b, 1, 1, t_m, 4).repeat(1, h, w, 1, 1).reshape(b*h*w,t_m, 4) 

        state = self.sigmoid(state_params[:,:4,...]).permute(0,2,3,1).reshape(b*h*w, 4)

        taos = self.sigmoid(state_params[:,4:8,...]).permute(0,2,3,1).reshape(b*h*w, 4)
        slopes = self.softplus(state_params[:,8:12,...]).permute(0,2,3,1).reshape(b*h*w, 4)
        biases = (state_params[:,12:,...]).permute(0,2,3,1).reshape(b*h*w, 4)

        out_arr = []
        for i in range(t_m):
            state = (1 - taos) * state + taos * (1 / (1 + torch.exp(-slopes * (meteo[:,i,:] - biases))))
            out_arr.append(state.prod(dim = -1))

        out = torch.stack(out_arr, dim = 1) # bhw, t_m

        out = out.reshape(b,h*w,t_m,1).reshape(b,h,w,t_m,1).permute(0,3,4,1,2)

        out = torch.tanh(torch.pow(out,2))


        #f = (1-tao) * f + (1/(1+exp(-sl*(T-base)))) * tao

        # if self.hparams.update_encoder_name == "MLP":
        #     if len(meso_dynamic_inputs.shape) == 5:
        #         meso_dynamic_inputs = meso_dynamic_inputs[:,:,:,(h_m//2- 1):(h_m//2),(w_m//2- 1):(w_m//2)].mean((3,4))
        #     meso_dynamic_inputs = meso_dynamic_inputs.reshape(b, 1, 1, t_m, c_m).repeat(1, h, w, 1, 1).reshape(b*h*w,t_m, c_m)

        #     location_input = state[:,None,:64,:,:].repeat(1,t_m, 1, 1,1).permute(0,3,4,1,2).reshape(b*h*w, t_m, 64)

        #     update_inputs = torch.cat([meso_dynamic_inputs,location_input], dim = -1)

        #     state = state.reshape(b, c_s, h * w).transpose(1,2).reshape(1, b*h*w, c_s)[:,:,64:]

        #     if self.training:
        #         idxs = torch.randperm(b*h*w).type_as(state).long()
        #         lc = data["landcover"].reshape(-1)
        #         idxs = idxs[(lc >= self.hparams.lc_min).byte() & (lc <= self.hparams.lc_max).byte()][:self.hparams.train_lstm_npixels*b]
        #         # idxs = torch.randint(low = 0, high = b*h*w, size = (self.hparams.train_lstm_npixels*b, )).type_as(update).long()
        #         if len(idxs) == 0:
        #             print(f"Detected cube without vegetation: {data['cubename']}")
        #             idxs = [1,2,3,4]

        #         update_inputs = update_inputs[idxs, :, :]

        #         state = state[:,idxs,:]

        #     update = self.update_encoder(update_inputs)


        # else:

        #     meso_dynamic_inputs = meso_dynamic_inputs.reshape(b*t_m, c_m, h_m, w_m)

        #     update = self.update_encoder(meso_dynamic_inputs)

        #     update = torch.cat([update, meso_dynamic_inputs[:,:,(h_m//2- 1):(h_m//2),(w_m//2- 1):(w_m//2)].mean((2,3))], dim = 1)

        #     _, c_u = update.shape

        #     update = update.reshape(b,t_m,c_u).unsqueeze(1).repeat(1,h*w, 1, 1).reshape(b*h*w,t_m,c_u)

        #     state = state.reshape(b, c_s, h * w).transpose(1,2).reshape(1, b*h*w, c_s)

        #     if self.training:
        #         idxs = torch.randperm(b*h*w).type_as(update).long()
        #         lc = data["landcover"].reshape(-1)
        #         idxs = idxs[(lc >= self.hparams.lc_min).byte() & (lc <= self.hparams.lc_max).byte()][:self.hparams.train_lstm_npixels*b]
        #         # idxs = torch.randint(low = 0, high = b*h*w, size = (self.hparams.train_lstm_npixels*b, )).type_as(update).long()
        #         if len(idxs) == 0:
        #             print(f"Detected cube without vegetation: {data['cubename']}")
        #             idxs = [1,2,3,4]

        #         state = state[:,idxs,:]
        #         update = update[idxs,:,:]

        # if self.training:
        #     out, _ = self.rnn(update.contiguous(), state.contiguous())
        # else:
        #     out_arr = []
        #     states = torch.chunk(state, self.hparams.val_n_splits, dim = 1)
        #     updates = torch.chunk(update, self.hparams.val_n_splits, dim = 0)
        #     for i in range(self.hparams.val_n_splits):
        #         out_arr.append(self.rnn(updates[i].contiguous(),states[i].contiguous())[0])
        #     out = torch.cat(out_arr, dim = 0)

        # if not self.hparams.update_encoder_name == "MLP":
        #     out = torch.cat([out,state.repeat(t_m,1,1).permute(1,0,2)], dim = -1)

        # out = self.sigmoid(self.head(out))

        # _, _, c_o = out.shape

        # if self.training:

        #     tmp_out = -torch.ones((b*h*w, t_m, c_o)).type_as(out)
        #     tmp_out[idxs, :, :] = out

        #     out = tmp_out

        # out = out.reshape(b,h*w,t_m,c_o).reshape(b,h,w,t_m,c_o).permute(0,3,4,1,2)

        return out, {}















