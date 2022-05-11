from operator import length_hint
from typing import Optional, Union, List

import argparse
import ast
import torch.nn as nn
import torch 
import sys
from earthnet_models_pytorch.utils import str2bool
from torchsummary import summary
import icecream

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2

        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,     # 4 for the 4 split in the ConvLSTM
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              stride = 1,
                              bias=self.bias)
   
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class UnetConvLSTM(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        self.hidden_dim_input = [24+2, 8, 16, 32, 32*4*4, 64, 64, 32, 16] #[2 + 24, 32, 64, 128, 4096, 512, 256, 128, 64]
        self.hidden_dim_output = [8, 16, 32, 32, 32*4*4, 32, 16, 8, 8] # [32, 64, 128, 256, 4096, 128, 64, 32, 32]
        self.hidden_size = [128, 64, 32, 16, 1, 16, 32, 64, 128]
        self.num_layers = len(self.hidden_size)
        
        cell_list = []     
        for i in range(self.num_layers):
            cell_list.append(
                ConvLSTMCell(input_dim=self.hidden_dim_input[i],
                                          hidden_dim=self.hidden_dim_output[i],
                                          kernel_size=3,
                                          bias=self.hparams.bias))
        self.cell_list = nn.ModuleList(cell_list)

        self.pooling = nn.AvgPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv = nn.Conv2d(in_channels=self.hidden_dim_output[-1],
                            out_channels=1,     
                            kernel_size=self.hparams.kernel_size,
                            padding=1,
                            bias=self.hparams.bias)
        #self.MLP = nn.Linear(self.hidden_dim_output[int(self.num_layers / 2)], self.hidden_dim_output[int(self.num_layers / 2)])
        
        self.activation_output = nn.Sigmoid()

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                (torch.zeros(batch_size, self.hidden_dim_output[i], self.hidden_size[i], self.hidden_size[i], device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim_output[i], self.hidden_size[i], self.hidden_size[i], device=self.conv.weight.device))
                )
        return init_states
        
    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        # TODO remove the useless features
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)
        parser.add_argument("--setting", type = str, default = "en22")
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--bias", type=str2bool, default=True)
        parser.add_argument("--input", type = str, default = None)
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        parser.add_argument("--skip_connections", type = str2bool, default=False)
        return parser
    

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        
        c_l = self.hparams.context_length if self.training else pred_start
        # Data
        hr_dynamics = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]
        target = hr_dynamics[:,:,0,...].unsqueeze(2)
        weather = data["dynamic"][1]
        topology = data["static"][0]

        # Shape
        b, _, _, _, _ = data["dynamic"][0].shape

        # initialize hidden states
        hidden_state = self.init_hidden(batch_size=b)

        output = []
        # Context data
        for t in range(self.hparams.context_length):
            # type of input
            if self.hparams.input =='NDVI+T':
                x = torch.cat((target[:, t,...], topology), dim=1)
            elif self.hparams.input == 'NDVI+T+W':
                weather_t = weather[:,t,...].unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)  
                x = torch.cat((target[:,t,...], topology), dim = 1)
                x = torch.cat((x, weather_t), dim = 1)
            else:
                x = target[:, t, :, :]

            # generation of the prediction
            if t < self.hparams.context_length -1:
                x = self.u_net(x, hidden_state)

            # last output
            else:
                x_previous = torch.clone(target[:,-1,...])
                x = self.u_net(x, hidden_state)
                if self.hparams.skip_connections:
                    x = x + x_previous
                x = self.activation_output(x)

        # Prediction
        for t in range(self.hparams.target_length):
            # type of input
            x_previous = torch.clone(x)
            if self.hparams.input =='NDVI+T':
                x = torch.cat((x, topology), dim=1)
            elif self.hparams.input == 'NDVI+T+W':
                weather_t = weather[:,c_l + t,...].unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)  
                x = torch.cat((x, topology), dim = 1)
                x = torch.cat((x, weather_t), dim = 1)
            
            x = self.u_net(x, hidden_state)

            if self.hparams.skip_connections:
                x = x + x_previous
                x = self.activation_output(x)

            output += [x]
        output = torch.cat(output, dim=1).unsqueeze(2)
        return output, {}

    def u_net(self, x, hidden_state):
        batch, _, _, _ = x.shape

        for layer_i in range(int(self.num_layers/2)):
            h, c = hidden_state[layer_i] 
            h, c = self.cell_list[layer_i](input_tensor=x,
                                                cur_state=[h, c])
            hidden_state[layer_i] = (h, c)

            x = self.pooling(h)
            x = self.relu(x)
            
        # Flatten
        x = self.pooling(x)
        x = self.flatten(x)
        #x = self.MLP(x)
        x = self.relu(x).unsqueeze(2).unsqueeze(3)

        h, c = hidden_state[layer_i + 1] 
        h, c = self.cell_list[layer_i + 1](input_tensor=x,
                                            cur_state=[h, c])
        h, c = hidden_state[layer_i + 1] = (h, c)

        x = x.reshape(batch, self.hidden_dim_output[int(self.num_layers / 2) + 1], 4, 4)
        x = self.upsample(x)
        x = self.upsample(x)

        # Decoding
        skip_i = int(self.num_layers/2)
        for layer_i in range(int(self.num_layers/2) + 1, self.num_layers):
            skip_i -= 1
            # skip connnection
            x = torch.cat((x, hidden_state[skip_i][0]), dim=1)

            h, c = hidden_state[layer_i] 
            h, c = self.cell_list[layer_i](input_tensor=x,
                                                cur_state=[h, c])
            hidden_state[layer_i] = (h, c)

            if layer_i < self.num_layers - 1:
                x = self.upsample(h)
            else:
                x = self.conv(h)
        return x



    
