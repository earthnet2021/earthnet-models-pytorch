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
from earthnet_models_pytorch.model.enc_dec_layer import get_encoder_decoder, ACTIVATIONS
from earthnet_models_pytorch.model.conditioning_layer import Identity_Conditioning,Cat_Conditioning,Cat_Project_Conditioning,CrossAttention_Conditioning,FiLMBlock
from earthnet_models_pytorch.model.layer_utils import inverse_permutation

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm, conditioning = None, condition_x_not_h = False, c_channels = None,n_tokens_c = 8):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.conditioning = conditioning
        self.condition_x_not_h = condition_x_not_h
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
            if conditioning == "action":
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
            #x_channels = num_hidden * 7 if condition_x_not_h else num_hidden * 4 
            x_channels = in_channel if condition_x_not_h else num_hidden
            if conditioning == "action":
                self.conv_a = nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                )
            elif conditioning == "cat":
                self.cond = Cat_Project_Conditioning(x_channels, c_channels)
            elif conditioning == "FiLM":
                self.cond = FiLMBlock(c_channels, num_hidden, x_channels)
            elif conditioning == "xAttn":
                self.cond = CrossAttention_Conditioning(x_channels, c_channels, n_tokens_c=n_tokens_c, act = "gelu", hidden_dim = 8, n_heads=8)
            else:
                self.cond = Identity_Conditioning()

            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, h_t, c_t, m_t, a_t):

                    
        m_concat = self.conv_m(m_t)

        if self.conditioning == "action":
            x_concat = self.conv_x(x_t)
            h_concat = self.conv_h(h_t)
            a_concat = self.conv_a(a_t)
            h_concat = h_concat * a_concat
        else:
            if self.condition_x_not_h:
                x_t = self.cond(x_t, a_t)
            else:
                h_t = self.cond(h_t, a_t)

            x_concat = self.conv_x(x_t)
            h_concat = self.conv_h(h_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
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

        self.n_tokens_c = 8
        self.width = 128

        enc_in_channels = self.hparams.num_inputs+ 3*(self.hparams.use_static_inputs)
        c_channels = self.hparams.num_weather

        if self.hparams.encdec_act:
            act = self.hparams.encdec_act
        else:
            act = "relu" if self.hparams.relu_on_conv else None

        if self.hparams.weather_conditioning_loc == "all":
            enc_dec_proc_conditioning = self.hparams.weather_conditioning
            self.early_conditioning = Identity_Conditioning()
        else:
            enc_dec_proc_conditioning = None
            if self.hparams.weather_conditioning_loc == "early":
                x_channels = enc_in_channels
            
                if self.hparams.weather_conditioning == "cat":
                    self.early_conditioning = Cat_Conditioning()#Cat_Project_Conditioning(x_channels, c_channels)
                    enc_in_channels = x_channels + c_channels
                elif self.hparams.weather_conditioning == "FiLM":
                    self.early_conditioning = FiLMBlock(c_channels, self.hparams.num_hidden, x_channels)
                elif self.hparams.weather_conditioning == "xAttn":
                    self.early_conditioning = CrossAttention_Conditioning(x_channels, c_channels, n_tokens_c=self.n_tokens_c, act = act, hidden_dim = 16, n_heads=16, mlp_after_attn=self.hparams.mlp_after_attn)
                else:
                    self.early_conditioning = Identity_Conditioning()
            else:
                self.early_conditioning = Identity_Conditioning()

        if self.hparams.conv_on_input:
            if self.hparams.encdec_norm:
                norm = self.hparams.encdec_norm
            else:
                norm = "group" if self.hparams.norm_on_conv else None
            
            
            self.width = 32




            self.enc, self.dec = get_encoder_decoder(
                self.hparams.encoder, enc_in_channels, self.hparams.num_hidden, self.hparams.num_inputs, down_factor = 4, filter_size= self.hparams.filter_size, skip_connection = self.hparams.res_on_conv, norm = norm, act = act, readout_act=self.hparams.encdec_readoutact,conditioning = enc_dec_proc_conditioning, c_channels_enc = c_channels,c_channels_dec = c_channels, n_tokens_c = self.n_tokens_c
            )

            # if self.hparams.encoder == "PredRNN":
            #     self.enc = PredRNNEncoder(in_channels, self.hparams.num_hidden // 2, self.hparams.num_hidden, self.hparams.filter_size, norm = self.hparams.norm_on_conv, relu = self.hparams.relu_on_conv)
            #     self.dec = PredRNNDecoder(self.hparams.num_hidden, self.hparams.num_hidden // 2, self.hparams.num_inputs, self.hparams.filter_size, norm = self.hparams.norm_on_conv, relu = self.hparams.relu_on_conv, residual = self.hparams.res_on_conv)
            # elif self.hparams.encoder == "SimVP":
            #     self.enc = Encoder(in_channels, self.hparams.num_hidden, 4)
            #     self.dec = Decoder(self.hparams.num_hidden, self.hparams.num_inputs, 4)
            # self.conv_input1 = nn.Conv2d(self.hparams.num_inputs+ 3*(self.hparams.use_static_inputs), self.hparams.num_hidden // 2,
            #                              self.hparams.filter_size,
            #                              stride=2, padding=self.hparams.filter_size // 2, bias=False)
            # self.conv_input2 = nn.Conv2d(self.hparams.num_hidden // 2, self.hparams.num_hidden, self.hparams.filter_size, stride=2,
            #                              padding=self.hparams.filter_size // 2, bias=False)
            # self.deconv_output1 = nn.ConvTranspose2d(self.hparams.num_hidden, self.hparams.num_hidden // 2,
            #                                          self.hparams.filter_size, stride=2, padding=self.hparams.filter_size // 2,
            #                                          bias=False)
            # self.deconv_output2 = nn.ConvTranspose2d(self.hparams.num_hidden // 2, self.hparams.num_inputs,
            #                                          self.hparams.filter_size, stride=2, padding=self.hparams.filter_size // 2,
            #                                          bias=False)
        
        if self.hparams.weather_conditioning == "action":
            self.action_conv_input1 = nn.Conv2d(self.hparams.num_weather, self.hparams.num_hidden // 2,
                                                1,
                                                stride=1, padding=0, bias=False)
            self.relu = nn.ReLU()
            self.action_conv_input2 = nn.Conv2d(self.hparams.num_hidden // 2, self.hparams.num_hidden, 1, stride=1,
                                                padding=0, bias=False)
        #elif self.hparams.weather_conditioning == "FiLM":
        #elif self.hparams.weather_conditioning == "cat":

        lstm_weather_cond = self.hparams.weather_conditioning if self.hparams.weather_conditioning_loc in ["latent", "all"] else None

        cell_list = []

        for i in range(self.hparams.num_layers):
            if i == 0:
                in_channel = self.hparams.num_hidden if self.hparams.conv_on_input else enc_in_channels 
            else:
                in_channel = self.hparams.num_hidden
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.hparams.num_hidden, self.width,
                                       self.hparams.filter_size, self.hparams.stride, self.hparams.layer_norm, conditioning = lstm_weather_cond, condition_x_not_h = self.hparams.condition_x_not_h, c_channels = self.hparams.num_weather,n_tokens_c = 8)
            )

        self.cell_list = nn.ModuleList(cell_list)

        if self.hparams.conv_on_input == 0:
            self.readout = nn.Conv2d(self.hparams.num_hidden, self.hparams.num_inputs, 1, stride=1,
                                       padding=0, bias=False)
            if self.hparams.encdec_readoutact:
                self.readout_act = ACTIVATIONS[self.hparams.encdec_readoutact]()
            else:
                self.readout_act = nn.Identity()
            
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
        parser.add_argument("--relu_on_conv", type = str2bool, default = False)
        parser.add_argument("--norm_on_conv", type = str2bool, default = False)
        parser.add_argument("--use_static_inputs", type = str2bool, default = False)
        parser.add_argument("--encoder", type = str, default = "PredRNN")
        parser.add_argument("--weather_conditioning", type = str, default = "action")
        parser.add_argument("--condition_x_not_h", type = str2bool, default = False)
        parser.add_argument("--weather_conditioning_loc", type = str, default = "latent")
        parser.add_argument("--encdec_norm", type = str, default = None)
        parser.add_argument("--encdec_act", type = str, default = None)
        parser.add_argument("--encdec_readoutact", type = str, default = None)
        parser.add_argument("--mlp_after_attn", type = str2bool, default = False)
        parser.add_argument("--spatial_shuffle", type = str2bool, default = False)

        return parser

    def forward(self, data, pred_start = 0, preds_length = None, sampling = None):

        preds_length = 0 if preds_length is None else preds_length
        
        c_l = self.hparams.context_length if self.training else pred_start

        hr_dynamic_inputs = data["dynamic"][0]#[:,(c_l - self.hparams.context_length):c_l,...]

        b, t, c, h, w = hr_dynamic_inputs.shape

        static_inputs = data["static"][0][:,:3,...]

        meso_dynamic_inputs = data["dynamic"][1]

        if len(meso_dynamic_inputs.shape) == 3:

            _, t_m, c_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(b, t_m, c_m, 1, 1)#.repeat(1, 1, 1, self.width, self.width)

        else:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape

            meso_dynamic_inputs = meso_dynamic_inputs.reshape(b, t_m//5, 5, c_m, h_m, w_m).mean(2)[:,:,:,39:41,39:41]#.repeat(1, 1, 1, self.width, self.width)


        if self.hparams.spatial_shuffle:
            perm = torch.randperm(b*h*w, device=hr_dynamic_inputs.device)
            invperm = inverse_permutation(perm)

            if meso_dynamic_inputs.shape[-1] == 1:
                meso_dynamic_inputs = meso_dynamic_inputs.expand(-1, -1, -1, h, w)
            else:
                meso_dynamic_inputs = nn.functional.interpolate(meso_dynamic_inputs, size = (h, w), mode='nearest-exact')

            hr_dynamic_inputs = hr_dynamic_inputs.permute(1, 2, 0, 3, 4).reshape(t,c, b*h*w)[:, :, perm].reshape(t, c, b, h, w).permute(2, 0, 1, 3, 4)
            meso_dynamic_inputs = meso_dynamic_inputs.permute(1, 2, 0, 3, 4).reshape(t_m, c_m, b*h*w)[:, :, perm].reshape(t_m, c_m, b,h,w).permute(2, 0, 1, 3, 4).contiguous()
            static_inputs = static_inputs.permute(1, 0, 2, 3).reshape(3, b*h*w)[:, perm].reshape(3, b,h,w).permute(1, 0, 2, 3)


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
                [b, self.hparams.num_hidden, self.width, self.width]).type_as(input_frames)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        decouple_loss = []
        memory = torch.zeros([b, self.hparams.num_hidden, self.width, self.width]).type_as(input_frames)

        for i in range(self.hparams.context_length + self.hparams.target_length - 1):
            if i < c_l:
                if sampling and i > 0:
                    proba = (torch.rand(b) < sampling[0]).type_as(input_frames)[:, None, None, None]
                    net = proba * input_frames[:, i] + (1 - proba) * x_gen
                else:
                    net = input_frames[:, i]
            else:
                if sampling:
                    proba = (torch.rand(b) < sampling[1]).type_as(input_frames)[:, None, None, None]
                    net = proba * input_frames[:, i] + (1 - proba) * x_gen
                else:
                    net = x_gen

            if self.hparams.use_static_inputs:
                net = torch.cat([net, static_inputs], dim = 1).contiguous()
            #else:
                
                #net = mask_true[:, t - 1] * input_frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen

            action = input_actions[:, i+1]

            if self.hparams.weather_conditioning == "action":
                action = self.action_conv_input1(action.expand(-1, -1, h, w))
                action = self.relu(action)
                action = self.action_conv_input2(action)

            net = self.early_conditioning(net, action)

            if self.hparams.conv_on_input:

                net, skips = self.enc(net, action)

                # net_shape1 = net.size()
                # net = self.conv_input1(net)
                # if self.hparams.res_on_conv:
                #     input_net1 = net
                # net_shape2 = net.size()
                # if self.hparams.relu_on_conv:
                #     net = self.relu(net)
                # net = self.conv_input2(net)
                # if self.hparams.res_on_conv:
                #     input_net2 = net
            
            # else:
            #     action = None

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
            if self.hparams.conv_on_input:
                x_gen = self.dec(h_t[self.hparams.num_layers - 1], skips, action)
                # if self.hparams.res_on_conv:
                #     x_gen = self.deconv_output1(h_t[self.hparams.num_layers - 1] + input_net2, output_size=net_shape2)
                #     if self.hparams.relu_on_conv:
                #         x_gen = self.deconv_output2(self.relu(x_gen + input_net1), output_size=net_shape1)
                #     else:
                #         x_gen = self.deconv_output2(x_gen + input_net1, output_size=net_shape1)
                # else:
                #     x_gen = self.deconv_output1(h_t[self.hparams.num_layers - 1], output_size=net_shape2)
                #     if self.hparams.relu_on_conv:
                #         x_gen = self.relu(x_gen)
                #     x_gen = self.deconv_output2(x_gen, output_size=net_shape1)
            else:
                x_gen = self.readout(h_t[self.hparams.num_layers - 1])
                x_gen = self.readout_act(x_gen)
            next_frames.append(x_gen)

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()[:, -self.hparams.target_length:, :1, ...]
        # loss = self.MSE_criterion(next_frames, all_frames[:, 1:, :, :, :next_frames.shape[4]]) + self.beta * decouple_loss
        # next_frames = next_frames[:, :, :, :, :self.hparams.num_inputs]

        if self.hparams.spatial_shuffle:
            next_frames = next_frames.permute(1, 2, 0, 3, 4).reshape(self.hparams.target_length, 1, b*h*w)[:, :, invperm].reshape(self.hparams.target_length, 1, b, h, w).permute(2, 0, 1, 3, 4)

        return next_frames, {"decouple_loss": decouple_loss}