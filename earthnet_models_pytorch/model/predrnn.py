"""
Code adapted from https://github.com/thuml/predrnn-pytorch
"""

from typing import Optional, Union
import argparse
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F

from earthnet_models_pytorch.utils import str2bool

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_a = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_a = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, h_t, c_t, m_t, a_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        a_concat = self.conv_a(a_t)
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat * a_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m



class PredRNN(nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        if self.hparams.conv_on_input == 1:
            self.conv_input1 = nn.Conv2d(self.hparams.num_inputs, self.hparams.num_hidden // 2,
                                         self.hparams.filter_size,
                                         stride=2, padding=self.hparams.filter_size // 2, bias=False)
            self.conv_input2 = nn.Conv2d(self.hparams.num_hidden // 2, self.hparams.num_hidden, self.hparams.filter_size, stride=2,
                                         padding=self.hparams.filter_size // 2, bias=False)
            self.action_conv_input1 = nn.Conv2d(self.hparams.num_weather, self.hparams.num_hidden // 2,
                                                1,
                                                stride=1, padding=0, bias=False)
            self.action_relu = nn.ReLU()
            self.action_conv_input2 = nn.Conv2d(self.hparams.num_hidden // 2, self.hparams.num_hidden, 1, stride=1,
                                                padding=0, bias=False)
            self.deconv_output1 = nn.ConvTranspose2d(self.hparams.num_hidden, self.hparams.num_hidden // 2,
                                                     self.hparams.filter_size, stride=2, padding=self.hparams.filter_size // 2,
                                                     bias=False)
            self.deconv_output2 = nn.ConvTranspose2d(self.hparams.num_hidden // 2, self.hparams.num_inputs,
                                                     self.hparams.filter_size, stride=2, padding=self.hparams.filter_size // 2,
                                                     bias=False)

        cell_list = []

        for i in range(self.hparams.num_layers):
            if i == 0:
                in_channel = self.hparams.num_inputs + self.hparams.num_weather if self.hparams.conv_on_input == 0 else self.hparams.num_hidden
            else:
                in_channel = self.hparams.num_hidden
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.hparams.num_hidden, 32,
                                       self.hparams.filter_size, self.hparams.stride, self.hparams.layer_norm)
            )

        self.cell_list = nn.ModuleList(cell_list)

        if self.hparams.conv_on_input == 0:
            self.conv_last = nn.Conv2d(self.hparams.num_hidden, self.hparams.num_inputs, 1, stride=1,
                                       padding=0, bias=False)
        self.adapter = nn.Conv2d(self.hparams.num_hidden, self.hparams.num_hidden, 1, stride=1, padding=0,
                                 bias=False)

    
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
        parser.add_argument("--num_inputs", type = int, default = 5)
        parser.add_argument("--num_weather", type = int, default = 24)
        parser.add_argument("--num_hidden", type = int, default = 128)
        parser.add_argument("--num_layers", type = int, default = 4)
        parser.add_argument("--filter_size", type = int, default = 5)
        parser.add_argument("--stride", type = int, default = 1)
        parser.add_argument("--layer_norm", type = str2bool, default = False)
        parser.add_argument("--conv_on_input", type = str2bool, default = True)
        parser.add_argument("--res_on_conv", type = str2bool, default = True)

        return parser

    def forward(self, data, pred_start = 0, n_preds = None):

        n_preds = 0 if n_preds is None else n_preds
        
        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0]#[:,(c_l - self.hparams.context_length):c_l,...]

        b, t, c, h, w = hr_dynamic_inputs.shape

        static_inputs = data["static"][0]

        meso_dynamic_inputs = data["dynamic"][1]

        if len(meso_dynamic_inputs.shape) == 3:

            _, t_m, c_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(b, t_m, c_m, 1, 1).repeat(1, 1, 1, h//4, w//4)

        else:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(b, t_m//5, 5, c_m, h_m, w_m).mean(2)[:,:,:,39:41,39:41].repeat(1, 1, 1, h//4, w//4)

        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        #frames = all_frames.permute(0, 1, 4, 2, 3).contiguous()
        input_frames = hr_dynamic_inputs.contiguous()#frames[:, :, :self.hparams.num_inputs, :, :]
        input_actions = meso_dynamic_inputs.contiguous()#frames[:, :, self.hparams.num_inputs:, :, :]
        #mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.hparams.num_layers):
            zeros = torch.zeros(
                [b, self.hparams.num_hidden, h//4, w//4]).type_as(input_frames)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        decouple_loss = []
        memory = torch.zeros([b, self.hparams.num_hidden, h//4, w//4]).type_as(input_frames)

        for i in range(t - 1):
            if i < c_l:
                net = input_frames[:, i]
            else:
                net = x_gen
            #else:
                
                #net = mask_true[:, t - 1] * input_frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen

            action = input_actions[:, i+1]

            if self.hparams.conv_on_input == 1:
                net_shape1 = net.size()
                net = self.conv_input1(net)
                if self.hparams.res_on_conv == 1:
                    input_net1 = net
                net_shape2 = net.size()
                net = self.conv_input2(net)
                if self.hparams.res_on_conv == 1:
                    input_net2 = net
                action = self.action_conv_input1(action)
                action = self.action_relu(action)
                action = self.action_conv_input2(action)

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory, action)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for j in range(1, self.hparams.num_layers):
                h_t[j], c_t[j], memory, delta_c, delta_m = self.cell_list[j](h_t[j - 1], h_t[j], c_t[j], memory, action)
                delta_c_list[j] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[j] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for j in range(0, self.hparams.num_layers):
                decouple_loss.append(torch.mean(torch.abs(
                    torch.cosine_similarity(delta_c_list[j], delta_m_list[j], dim=2))))
            if self.hparams.conv_on_input == 1:
                if self.hparams.res_on_conv == 1:
                    x_gen = self.deconv_output1(h_t[self.hparams.num_layers - 1] + input_net2, output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen + input_net1, output_size=net_shape1)
                else:
                    x_gen = self.deconv_output1(h_t[self.hparams.num_layers - 1], output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen, output_size=net_shape1)
            else:
                x_gen = self.conv_last(h_t[self.hparams.num_layers - 1])
            next_frames.append(x_gen)

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        # loss = self.MSE_criterion(next_frames, all_frames[:, 1:, :, :, :next_frames.shape[4]]) + self.beta * decouple_loss
        # next_frames = next_frames[:, :, :, :, :self.hparams.num_inputs]
        
        return next_frames, {"decouple_loss": decouple_loss}