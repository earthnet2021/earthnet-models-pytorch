"""
Adapted from https://github.com/chengtan9907/SimVPv2
"""
from typing import Optional, Union

import argparse
import ast

import torch
import torch.nn as nn

from earthnet_models_pytorch.utils import str2bool

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * g_t
        return h_new, c_new

class ConvLSTM(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        if self.hparams.conv_on_input:
            self.conv_input1 = nn.Conv2d(self.hparams.num_inputs, self.hparams.num_hidden // 2,
                                         self.hparams.filter_size,
                                         stride=2, padding=self.hparams.filter_size // 2, bias=False)
            self.conv_input2 = nn.Conv2d(self.hparams.num_hidden // 2, self.hparams.num_hidden, self.hparams.filter_size, stride=2,
                                         padding=self.hparams.filter_size // 2, bias=False)
            self.relu = nn.ReLU()
            self.action_conv_input2 = nn.Conv2d(self.hparams.num_hidden // 2, self.hparams.num_hidden, 1, stride=1,
                                                padding=0, bias=False)
            self.deconv_output1 = nn.ConvTranspose2d(self.hparams.num_hidden, self.hparams.num_hidden // 2,
                                                     self.hparams.filter_size, stride=2, padding=self.hparams.filter_size // 2,
                                                     bias=False)
            self.deconv_output2 = nn.ConvTranspose2d(self.hparams.num_hidden // 2, self.hparams.num_outputs,
                                                     self.hparams.filter_size, stride=2, padding=self.hparams.filter_size // 2,
                                                     bias=False)
            height = 32
        else:
            height = 128
            self.conv_last = nn.Conv2d(self.hparams.num_hidden, self.hparams.num_outputs, kernel_size=1, stride=1, padding=0, bias=False)

        cell_list = []
        for i in range(self.hparams.num_layers):
            in_channel = self.hparams.num_inputs if (i == 0) and not self.hparams.conv_on_input else self.hparams.num_hidden
            cell_list.append(
                ConvLSTMCell(in_channel, self.hparams.num_hidden, height, height, self.hparams.filter_size, self.hparams.stride, self.hparams.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--setting", type = str, default = "en21x")
        parser.add_argument("--context_length", type = int, default = 10)
        parser.add_argument("--target_length", type = int, default = 20)
        parser.add_argument("--num_inputs", type = int, default = 5+3+24)
        parser.add_argument("--num_hidden", type = int, default = 128)
        parser.add_argument("--num_outputs", type = int, default = 1)
        parser.add_argument("--num_layers", type = int, default = 4)
        parser.add_argument("--filter_size", type = int, default = 5)
        parser.add_argument("--stride", type = int, default = 1)
        parser.add_argument("--layer_norm", type = str2bool, default = False)
        parser.add_argument("--conv_on_input", type = str2bool, default = True)
        parser.add_argument("--res_on_conv", type = str2bool, default = True)

        return parser

    
    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):

        n_preds = 0 if n_preds is None else n_preds

        c_l = self.hparams.context_length if self.training else pred_start

        frames = data["dynamic"][0]

        B, T, C, H, W = frames.shape

        static_inputs = data["static"][0][:, :3, ...]

        meso_dynamic_inputs = data["dynamic"][1]

        if len(meso_dynamic_inputs.shape) == 3:

            _, t_m, c_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m, c_m, 1, 1).repeat(1, 1, 1, H, W)

        else:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(B, t_m//5, 5, c_m, h_m, w_m).mean(2)[:,:,:,39:41,39:41].repeat(1, 1, 1, H, W)


        

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.hparams.num_layers):
            if self.hparams.conv_on_input:
                zeros = torch.zeros([B, self.hparams.num_hidden, H//4, W//4]).type_as(frames)
            else:
                zeros = torch.zeros([B, self.hparams.num_hidden, H, W]).type_as(frames)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(c_l + self.hparams.target_length - 1):
            
            if t < c_l:
                frame = frames[:, t]
            else:
                frame = x_gen

            net = torch.cat([frame, static_inputs, meso_dynamic_inputs[:, t]], dim = 1).contiguous()

            if self.hparams.conv_on_input:
                net_shape1 = net.size()
                net = self.conv_input1(net)
                if self.hparams.res_on_conv:
                    input_net1 = net
                net_shape2 = net.size()
                net = self.conv_input2(self.relu(net))
                if self.hparams.res_on_conv:
                    input_net2 = net

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.hparams.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            if self.hparams.conv_on_input:
                if self.hparams.res_on_conv:
                    x_gen = self.deconv_output1(h_t[self.hparams.num_layers - 1] + input_net2, output_size=net_shape2)
                    x_gen = self.deconv_output2(self.relu(x_gen + input_net1), output_size=net_shape1)
                else:
                    x_gen = self.deconv_output1(h_t[self.hparams.num_layers - 1], output_size=net_shape2)
                    x_gen = self.deconv_output2(self.relu(x_gen), output_size=net_shape1)
            else:
                x_gen = self.conv_last(h_t[self.hparams.num_layers - 1])

            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()[:, -self.hparams.target_length:, :1, ...]

        return next_frames, {}