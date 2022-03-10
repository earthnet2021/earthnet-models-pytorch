"""ContextConvLSTM
"""
from typing import Optional, Union, List

import argparse
import ast
from grpc import dynamic_ssl_server_credentials
from pip import main
import torch.nn as nn
import torch
import sys
from earthnet_models_pytorch.utils import str2bool

class ContextConvLSTMCell(nn.Module):

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


class ContextConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W
    Output:

    """

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams
        
        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg", "en21x", "en22"])

        self.encoder_1_convlstm = ContextConvLSTMCell(input_dim=6,  # TODO find a better solution
                                               hidden_dim=self.hparams.hidden_dim[0],
                                               kernel_size=self.hparams.kernel_size,
                                               bias=self.hparams.bias)
        
        self.encoder_2_convlstm = ContextConvLSTMCell(input_dim=self.hparams.hidden_dim[0],
                                               hidden_dim=self.hparams.hidden_dim[1],
                                               kernel_size=self.hparams.kernel_size,
                                               bias=self.hparams.bias)
        self.decoder_1_convlstm = ContextConvLSTMCell(input_dim=self.hparams.hidden_dim[1],
                                               hidden_dim=self.hparams.hidden_dim[2],
                                               kernel_size=self.hparams.kernel_size,
                                               bias=self.hparams.bias)
        self.decoder_2_convlstm = ContextConvLSTMCell(input_dim=self.hparams.hidden_dim[2],
                                               hidden_dim=self.hparams.hidden_dim[3],
                                               kernel_size=self.hparams.kernel_size,
                                               bias=self.hparams.bias)
        padding = self.hparams.kernel_size // 2, self.hparams.kernel_size // 2
        self.conv = nn.Conv2d(in_channels=self.hparams.hidden_dim[3],
                              out_channels=1,     
                              kernel_size=self.hparams.kernel_size,
                              padding=padding,
                              bias=self.hparams.bias)
    
        
    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        # TODO remove the useless features
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)
        
        parser.add_argument("--input_dim", type=int, default=6)
        parser.add_argument("--hidden_dim", type=ast.literal_eval, default=[64, 64, 64, 64])  # TODO find a better type ? list(int)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--bias", type=str2bool, default=True)
        parser.add_argument("--return_all_layers", type=str2bool, default=False)
        parser.add_argument("--state_encoder_name", type = str, default = "FPN")
        parser.add_argument("--state_encoder_args", type = ast.literal_eval, default = '{"encoder_name": "timm-efficientnet-b4", "encoder_weights": "noisy-student", "in_channels": 191, "classes": 256}')
        parser.add_argument("--update_encoder_name", type = str, default = "efficientnet_b1")
        parser.add_argument("--update_encoder_inchannels", type = int, default = 28)
        parser.add_argument("--update_encoder_nclasses", type = int, default = 128)
        parser.add_argument("--train_lstm_npixels", type = int, default = 256)
        parser.add_argument("--setting", type = str, default = "en21x")
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        parser.add_argument("--use_dem", type = str2bool, default = True)
        parser.add_argument("--use_soilgrids", type = str2bool, default = True)
        parser.add_argument("--lc_min", type = int, default = 82)
        parser.add_argument("--lc_max", type = int, default = 104)
        parser.add_argument("--val_n_splits", type = int, default = 20)
        # TODO add argument in the base.yaml
        return parser
    

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        # TODO concat meteo with hidden
        input_tensor, meteo = self.input_data(data, pred_start, n_preds)
        
        b, t, c, h, w = input_tensor.shape
        
        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, height=h, width=w)
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, height=h, width=w)
        
        output = []
        
        # encoding network
        for t in range(self.hparams.context_length):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=input_tensor[:, t, :, :],
                                               cur_state=[h_t, c_t])  
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  
            
        # forecasting network
        for t in range(self.hparams.target_length):
            h_t, c_t = self.decoder_1_convlstm(input_tensor=h_t2,
                                                 cur_state=[h_t, c_t]) 
            h_t2, c_t2 = self.decoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])
            pred = self.conv(h_t2)
            output += [pred]
            
        output = torch.cat(output, dim=0).unsqueeze(0)
        
        return output, {}


    
    def input_data(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        n_preds = 0 if n_preds is None else n_preds
        
        c_l = self.hparams.context_length if self.training else pred_start   # first element to predict
        
        # Dynamic inputs (kndvi, blue, green, red, near-red) only to be used from t 0 to t context_lenght
        hr_dynamic_inputs = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]   # [1, 45, 5, 128, 128]
        
        # batch, time, channels, height, width
        _, t, _, _, _ = hr_dynamic_inputs.shape
        
        meso_dynamic_inputs = data["dynamic"][1][:,c_l:,...].repeat(1, 1, 1, 128, 128)  # meteo  [1, 45, 33] -> [1, 45, 33, 128, 128] ?
        static_inputs = data["static"][0].unsqueeze(1).repeat(1, t, 1, 1, 1)  # dem [1, 9, 1, 128, 128]

        # TODO select the data with a sufficient quantity of interesting landcover (not building, not cloud)
        # lc = data["landcover"][(lc >= self.hparams.lc_min).byte() & (lc <= self.hparams.lc_max).byte()]  # [1, 1, 128, 128]


        state_inputs = torch.cat((hr_dynamic_inputs, static_inputs), dim = 2)
        return state_inputs, meso_dynamic_inputs

    
