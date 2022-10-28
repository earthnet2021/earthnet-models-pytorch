"""ConvLSTM_LSTM
"""

from typing import Optional, Union, List

import argparse
import ast
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

class MLP(nn.Module):
    def __init__(self, dim_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_features[0], dim_features[1]),
            nn.ReLU(),
            nn.Linear(dim_features[1], dim_features[2]),
            nn.ReLU(), 
            nn.Linear(dim_features[2], dim_features[3]),
            nn.ReLU()
        )

    def forward(self, weather):
        return self.model(weather)


class ConvLSTMLSTM(nn.Module):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams
        
        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg", "en21x", "en22"])

        
        input_encoder = 1  # # TODO find a better solution nb of channel 39 = 5 r,b,g, nr, ndvi + 1 dem + 33 weather
        input_decoder = 1  # 33 weather

        if hparams.method == 'MLP':
            dim_features = [64, 24, 10, 1]
            self.MLP = MLP(dim_features)
            input_encoder = 2
            input_decoder = 2
        
        if self.hparams.input == 'NDVI+T':
            input_encoder = 2
            input_decoder = 2
        elif self.hparams.input == 'NDVI+T+W':
            input_encoder = 1 + 1 + 24
            input_decoder = 1 + 1 + 24
        elif self.hparams.input == 'RGBNR':  
            input_encoder = 4
            input_decoder = 4

        
        self.encoder_1_convlstm = ContextConvLSTMCell(input_dim=input_encoder,  
                                               hidden_dim=self.hparams.hidden_dim[0],
                                               kernel_size=self.hparams.kernel_size,
                                               bias=self.hparams.bias)
                                            
        self.encoder_2_convlstm = ContextConvLSTMCell(input_dim=self.hparams.hidden_dim[0],
                                               hidden_dim=self.hparams.hidden_dim[1],
                                               kernel_size=self.hparams.kernel_size,
                                               bias=self.hparams.bias)
                                               
        self.decoder_1_convlstm = ContextConvLSTMCell(input_dim=input_decoder, 
                                               hidden_dim=self.hparams.hidden_dim[0],
                                               kernel_size=self.hparams.kernel_size,
                                               bias=self.hparams.bias)
        
        self.decoder_2_convlstm = ContextConvLSTMCell(input_dim=self.hparams.hidden_dim[0],
                                               hidden_dim=self.hparams.hidden_dim[1],
                                               kernel_size=self.hparams.kernel_size,
                                               bias=self.hparams.bias)
        
        padding = self.hparams.kernel_size // 2, self.hparams.kernel_size // 2
        """
        self.conv1 = nn.Conv2d(in_channels=input_encoder,
                            out_channels=64,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)
        self.conv2 = nn.Conv2d(in_channels=64,
                            out_channels=64,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)
        self.conv3 = nn.Conv2d(in_channels=64,
                            out_channels=64,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)
        self.conv4 = nn.Conv2d(in_channels=64,
                            out_channels=64,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)
        """
        self.conv = nn.Conv2d(in_channels=64,
                            out_channels=1,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)

        #self.lstm = nn.LSTM(24,64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(24, 64, 2, batch_first=True)


        self.conv = nn.Conv2d(in_channels=self.hparams.hidden_dim[1],
                            out_channels=1,     
                            kernel_size=self.hparams.kernel_size,
                            padding=padding,
                            bias=self.hparams.bias)

        self.activation_output = nn.Sigmoid()

        
    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        # TODO remove the useless features
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        # parser.add_argument("--input_dim", type=int, default=6)
        parser.add_argument("--hidden_dim", type=ast.literal_eval, default=[64, 64, 64, 64])  # TODO find a better type ? list(int)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--bias", type=str2bool, default=True)
        parser.add_argument("--return_all_layers", type=str2bool, default=False)
        parser.add_argument("--setting", type = str, default = "en22")
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        parser.add_argument("--method", type = str, default = None)
        parser.add_argument("--input", type = str, default = None)
        parser.add_argument("--skip_connections", type = str2bool, default=False)
        parser.add_argument("--add_conv", type = str2bool, default=False)
        return parser
    

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        c_l = self.hparams.context_length if self.training else pred_start
        # Data
        hr_dynamics = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]
        target = hr_dynamics[:,:,0,...].unsqueeze(2)
        weather = data["dynamic"][1].unsqueeze(3).unsqueeze(4)
        topology = data["static"][0]

        # Shape
        b, t, _, h, w = data["dynamic"][0].shape

        # initialize hidden states
        if self.hparams.method != "CNN":
            h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, height=h, width=w)
            h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, height=h, width=w)

        #h_lstm, c_lstm = torch.zeros(1, b,self.hparams.hidden_dim[1]).to(self.conv.weight.device), torch.zeros(1, b, self.hparams.hidden_dim[1]).to(self.conv.weight.device)
        h_lstm2, c_lstm2 = torch.zeros(2, b,self.hparams.hidden_dim[1]).to(self.conv.weight.device), torch.zeros(2, b, self.hparams.hidden_dim[1]).to(self.conv.weight.device)

        output = []

        # encoding network
        for t in range(self.hparams.context_length):
            if self.hparams.input =='NDVI+T':
                input = torch.cat((target[:, t,...], topology), dim=1)
                if self.hparams.method == 'forecast':
                    weather_t = weather[:, t + 1,...].unsqueeze(1).squeeze(4).squeeze(3)
                else:
                    weather_t = weather[:, t,...].unsqueeze(1).squeeze(4).squeeze(3)
            else:
                input = target[:, t, :, :]
                
            if self.hparams.method == "CNN":
                pred = self.conv1(input)
                pred = nn.ReLU()(pred)
                pred = self.conv2(pred)
                pred = nn.ReLU()(pred)
                pred = self.conv3(pred)
                pred = nn.ReLU()(pred)
                h_t2 = self.conv4(pred)

            else:
                # First block
                h_t, c_t = self.encoder_1_convlstm(input_tensor=input,
                                                cur_state=[h_t, c_t])  
                # out , (h_lstm, c_lstm) = self.lstm(weather_t, (h_lstm, c_lstm))
                # if self.hparams.method != "MLP":
                #    h_t = h_t + out.squeeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)                                 

                # second block
                h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                    cur_state=[h_t2, c_t2]) 

            out , (h_lstm2, c_lstm2) = self.lstm2(weather_t, (h_lstm2, c_lstm2))

            h_t2 = h_t2 + out.squeeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)

                

        # First prediction        
        pred = self.conv(h_t2)

        if self.hparams.skip_connections:
            pred = pred + target[:,-1,...]

        if self.hparams.method == 'MLP':
            pred = pred + self.MLP(out.squeeze(1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128) 
            # pred = pred + self.MLP(weather[:,c_l + t,...].squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128) 
                            
        pred = self.activation_output(pred)


        # forecasting network
        for t in range(self.hparams.target_length):
            # for skip connection
            pred_previous = torch.clone(pred)

            # Input
            if self.hparams.input =='NDVI+T':
                pred = torch.cat((pred, topology), dim=1)
                if self.hparams.method == 'forecast':
                    weather_t = weather[:,c_l + t + 1,...].unsqueeze(1).squeeze(4).squeeze(3)
                else:
                    weather_t = weather[:,c_l + t,...].unsqueeze(1).squeeze(4).squeeze(3)

            elif self.hparams.input == 'NDVI+T+W':
                weather_t = weather[:,c_l + t,...].repeat(1, 1, 128, 128)  
                pred = torch.cat((pred, topology), dim = 1)
                pred = torch.cat((pred, weather_t), dim = 1)

            if self.hparams.method == "CNN":
                pred = self.conv1(input)
                pred = nn.ReLU()(pred)
                pred = self.conv2(pred)
                pred = nn.ReLU()(pred)
                pred = self.conv3(pred)
                pred = nn.ReLU()(pred)
                h_t2 = self.conv4(pred)

            else:
                #first block
                h_t, c_t = self.decoder_1_convlstm(input_tensor=pred,
                                                    cur_state=[h_t, c_t]) 
                #out , (h_lstm, c_lstm) = self.lstm(weather_t, (h_lstm, c_lstm))
                #if self.hparams.method != "MLP":
                #    h_t = h_t + out.squeeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)   
                # Second block
                h_t2, c_t2 = self.decoder_2_convlstm(input_tensor=h_t,
                                                    cur_state=[h_t2, c_t2])
                

            out , (h_lstm2, c_lstm2) = self.lstm2(weather_t, (h_lstm2, c_lstm2))

            h_t2 = h_t2 + out.squeeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
            
            pred = self.conv(h_t2)

            if self.hparams.skip_connections:
                pred = pred + pred_previous
            if self.hparams.method == 'MLP':
                pred = pred + self.MLP(out.squeeze(1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128) 
  
            pred = self.activation_output(pred)
            output += [pred]
            
        output = torch.cat(output, dim=1).unsqueeze(2)
        return output, {}