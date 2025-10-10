


from typing import Optional, Union

import argparse
import ast

import torch
import torch.nn as nn

from earthnet_models_pytorch.utils import str2bool
from earthnet_models_pytorch.model.enc_dec_layer import get_encoder_decoder, ACTIVATIONS
from earthnet_models_pytorch.model.conditioning_layer import Identity_Conditioning,Cat_Conditioning,Cat_Project_Conditioning,CrossAttention_Conditioning,FiLMBlock, MLP2d
from earthnet_models_pytorch.model.layer_utils import inverse_permutation


class UNetLSTM(nn.Module):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams

        self.n_tokens_c = 8

        enc_in_channels = self.hparams.n_hr_inputs * self.hparams.context_length + self.hparams.n_static_inputs

        self.enc, self.dec = get_encoder_decoder("PatchMerge", enc_in_channels, self.hparams.hid_channels, self.hparams.hid_channels * 2 * self.hparams.n_lstm_layers, down_factor=2**self.hparams.enc_depth, filter_size = self.hparams.filter_size, skip_connection=True, norm = self.hparams.norm, act = self.hparams.act, bias = self.hparams.bias, readout_act=None,conditioning = None)

        self.embed_weather = MLP2d(self.hparams.n_weather_inputs, self.hparams.hid_channels, self.hparams.hid_channels, act = self.hparams.act)

        if self.hparams.use_gru:
            self.gru = nn.GRU(self.hparams.hid_channels, self.hparams.hid_channels, self.hparams.n_lstm_layers, batch_first = True)
        else:
            self.lstm = nn.LSTM(self.hparams.hid_channels, self.hparams.hid_channels, self.hparams.n_lstm_layers, batch_first = True)

        self.spatial_head = nn.Conv2d(self.hparams.hid_channels * 2 * self.hparams.n_lstm_layers, self.hparams.hid_channels, 1, stride=1, padding=0, bias=False)

        self.readout = nn.Conv2d(self.hparams.hid_channels, 1, 1, stride=1, padding=0, bias=False)
        if self.hparams.readout_act:
            self.readout_act = ACTIVATIONS[self.hparams.readout_act]()
        else:
            self.readout_act = nn.Identity()


    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)
        
        parser.add_argument("--setting", type = str, default = "en21-std")
        parser.add_argument("--context_length", type = int, default = 10)
        parser.add_argument("--target_length", type = int, default = 20)
        parser.add_argument("--enc_depth", type = int, default = 5)
        parser.add_argument("--hid_channels", type = int, default = 128)
        parser.add_argument("--n_lstm_layers", type = int, default = 2)
        parser.add_argument("--filter_size", type = int, default = 3)
        parser.add_argument("--bias", type = str2bool, default = True)
        parser.add_argument("--norm", type = str, default = None)
        parser.add_argument("--act", type = str, default = None)
        parser.add_argument("--readout_act", type = str, default = "tanh")
        parser.add_argument("--n_hr_inputs", type = int, default = 5)
        parser.add_argument("--n_static_inputs", type = int, default = 3)
        parser.add_argument("--n_weather_inputs", type = int, default = 24)
        parser.add_argument("--spatial_shuffle", type = str2bool, default = False)
        parser.add_argument("--lc_min", type = int, default = 10)
        parser.add_argument("--lc_max", type = int, default = 40)     
        parser.add_argument("--train_lstm_npixels", type = int, default = 512)
        parser.add_argument("--val_n_splits", type = int, default = 32)
        parser.add_argument("--use_gru", type = str2bool, default = False)

        return parser

    def forward(self, data, pred_start: int = 0, preds_length: Optional[int] = None):
    
        preds_length = 0 if preds_length is None else preds_length

        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0][:, :c_l, ...]

        B, T, C, H, W = hr_dynamic_inputs.shape

        static_inputs = data["static"][0][:, :self.hparams.n_static_inputs, ...]


        meso_dynamic_inputs = data["dynamic"][1]
        if len(meso_dynamic_inputs.shape) == 3:

            _, t_m, c_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m, c_m, 1, 1)#.repeat(1, 1, 1, H, W)

        else:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m//5, 5, c_m, h_m, w_m).mean(2)[:,:,:,39:41,39:41]#.repeat(1, 1, 1, H, W)
            t_m = t_m//5


        if self.hparams.spatial_shuffle:
            perm = torch.randperm(B*H*W, device=hr_dynamic_inputs.device)
            invperm = inverse_permutation(perm)

            meso_dynamic_inputs
            if meso_dynamic_inputs.shape[-1] == 1:
                meso_dynamic_inputs = meso_dynamic_inputs.expand(-1, -1, -1, H, W)
            else:
                meso_dynamic_inputs = nn.functional.interpolate(meso_dynamic_inputs, size = (H, W), mode='nearest-exact')

            hr_dynamic_inputs = hr_dynamic_inputs.permute(1, 2, 0, 3, 4).reshape(T, C, B*H*W)[:, :, perm].reshape(T, C, B, H, W).permute(2, 0, 1, 3, 4)
            meso_dynamic_inputs = meso_dynamic_inputs.permute(1, 2, 0, 3, 4).reshape(t_m, c_m, B*H*W)[:, :, perm].reshape(t_m, c_m, B, H, W).permute(2, 0, 1, 3, 4).contiguous()
            static_inputs = static_inputs.permute(1, 0, 2, 3).reshape(self.hparams.n_static_inputs, B*H*W)[:, perm].reshape(self.hparams.n_static_inputs, B, H, W).permute(1, 0, 2, 3)
            c = meso_dynamic_inputs.reshape(B, t_m, c_m//self.n_tokens_c, self.n_tokens_c, H, W).permute(0, 3, 1, 2, 4, 5).reshape(B*t_m, c_m, H, W)
        else:
            c = meso_dynamic_inputs.reshape(B, t_m, c_m//self.n_tokens_c, self.n_tokens_c).permute(0, 3, 1, 2).reshape(B*t_m,c_m, 1, 1)


        x = torch.cat([hr_dynamic_inputs.reshape(B, T*C, H, W), static_inputs], dim = 1)

        x, skips = self.enc(x)
        embed_spatial = self.dec(x, skips)

        h0, c0 = torch.split(embed_spatial.permute(0, 2, 3, 1).reshape(B * H * W, self.hparams.n_lstm_layers, -1).permute(1,0,2), self.hparams.hid_channels, dim = -1)

        c = self.embed_weather(c)

        if c.shape[-1] == 1:
            c = c.expand(-1, -1, H, W)
        elif c.shape[-1] != W:
            c = nn.functional.interpolate(c, size = (H, W), mode='nearest-exact')


        c = c.reshape(B,t_m,self.hparams.hid_channels, H, W).permute(0, 3, 4, 1, 2).reshape(B*H*W, t_m, self.hparams.hid_channels).contiguous()
        h0 = h0.contiguous()
        c0 = c0.contiguous()

        if self.training:
            idxs = torch.randperm(B*H*W).type_as(c).long()
            lc = data["landcover"].reshape(-1)
            idxs = idxs[(lc >= self.hparams.lc_min).bool() & (lc <= self.hparams.lc_max).bool()][:self.hparams.train_lstm_npixels*B]
            if len(idxs) == 0:
                print(f"Detected cube without vegetation: {data['cubename']}")
                idxs = [1,2,3,4]

            c = c[idxs,...]
            h0 = h0[:, idxs, ...]
            c0 = c0[:, idxs, ...]
            
            if self.hparams.use_gru:
                output, _  = self.gru(c, h0)
            else:
                output, (hn, cn) = self.lstm(c, (h0, c0))

            tmp_out = -2*torch.ones((B*H*W, t_m, self.hparams.hid_channels)).type_as(output)
            tmp_out[idxs, :, :] = output
            output = tmp_out

        else:
            out_arr = []
            h0 = torch.chunk(h0, self.hparams.val_n_splits, dim = 1)
            c0 = torch.chunk(c0, self.hparams.val_n_splits, dim = 1)
            c = torch.chunk(c, self.hparams.val_n_splits, dim = 0)
            for i in range(self.hparams.val_n_splits):
                if self.hparams.use_gru:
                    out_arr.append(self.gru(c[i].contiguous(),h0[i].contiguous())[0])
                else:
                    out_arr.append(self.lstm(c[i].contiguous(),(h0[i].contiguous(), c0[i].contiguous()))[0])
            output = torch.cat(out_arr, dim = 0)

        preds_tmp = output.reshape(B, H, W, t_m, self.hparams.hid_channels).permute(0, 3, 4, 1, 2)[:, -self.hparams.target_length:,...].reshape(B*self.hparams.target_length, self.hparams.hid_channels, H, W)

        embed_spatial = self.spatial_head(embed_spatial).unsqueeze(1).expand(-1, self.hparams.target_length, self.hparams.hid_channels, H, W).reshape(B*self.hparams.target_length, self.hparams.hid_channels, H, W)

        preds = preds_tmp + embed_spatial

        preds = self.readout(preds)
        preds = self.readout_act(preds)
        
        if self.training:
            preds = torch.where(preds_tmp[:, :1, ...] != -2, preds, -2*torch.ones_like(preds).type_as(preds))

        preds = preds.reshape(B,self.hparams.target_length, 1, H, W)

        if self.hparams.spatial_shuffle:
            preds = preds.permute(1, 2, 0, 3, 4).reshape(self.hparams.target_length, 1, B*H*W)[:, :, invperm].reshape(self.hparams.target_length, 1, B, H, W).permute(2, 0, 1, 3, 4)

        return preds, {}