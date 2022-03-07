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
                              out_channels=4 * self.hidden_dim,
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

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
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
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams
        self.ndvi_pred = (hparams.setting in ["en21-veg", "europe-veg", "en21x", "en22"])

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        print(self.hparams.hidden_dim, self.hparams.num_layers)
        if not len(self.hparams.hidden_dim) == self.hparams.num_layers:
            raise ValueError('Inconsistent list length.')

        cell_list = []
        for i in range(0, self.hparams.num_layers):
            cur_input_dim = self.hparams.input_dim if i == 0 else self.hparams.hidden_dim[i - 1]

            cell_list.append(ContextConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hparams.hidden_dim[i],
                                          kernel_size=self.hparams.kernel_size,
                                          bias=self.hparams.bias))

        self.cell_list = nn.ModuleList(cell_list)
      
      
        
    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)
        
        parser.add_argument("--input_dim", type=int, default=28)
        parser.add_argument("--hidden_dim", type=ast.literal_eval, default=[64, 64,128])  # TODO find a better type ? list(int)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--num_layers", type=int, default=3)
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
        
        print(data["dynamic"][0].shape)
        print(data["dynamic"][1].shape)
        print(data["static"][0].shape)
        print(data["landcover"].shape)
        
        n_preds = 0 if n_preds is None else n_preds
        
        c_l = self.hparams.context_length if self.training else pred_start   # first element to predict
        
         # High resolution [0] dynamic inputs only to be used from t 0 to t context_lenght
        hr_dynamic_inputs = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]   # [0] for kndvi, [1] for eobs

        last_dynamic_input = hr_dynamic_inputs[:,-1,...]

        # batch, time, channels, height, width 
        b, t, c, h, w = hr_dynamic_inputs.shape

        hr_dynamic_inputs = hr_dynamic_inputs.reshape(b, t*c, h, w)  # why reshape t and c ?

        static_inputs = data["static"][0]
                
        # Concatenates DEM and Soil to High-res dynamic
        if self.hparams.use_dem and self.hparams.use_soilgrids:  
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs), dim = 1)
        elif self.hparams.use_dem:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs[:,0,...][:,None,...]), dim = 1)
        elif self.hparams.use_soilgrids:
            state_inputs = torch.cat((hr_dynamic_inputs, static_inputs[:,1:,...]), dim = 1)
        else:
            state_inputs = hr_dynamic_inputs 

        meso_dynamic_inputs = data["dynamic"][1][:,c_l:,...]  # eobs data
       
        # Determine whether the dataset low-res dynamic variables [1] *meteo* are spatial (5 dims) or scalar (3 dims)
        if len(meso_dynamic_inputs.shape) == 5:
            _, t_m, c_m, h_m, w_m = meso_dynamic_inputs.shape
        else:
            _, t_m, c_m = meso_dynamic_inputs.shape
            
        meso_dynamic_inputs = meso_dynamic_inputs.reshape(b*t_m, c_m, h_m, w_m)

        _, c_u = update.shape

        update = update.reshape(b,t_m,c_u).unsqueeze(1).repeat(1,h*w, 1, 1).reshape(b*h*w,t_m,c_u)

        state = state.reshape(b, c_s, h * w).transpose(1,2).reshape(1, b*h*w, c_s)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
        
                                         image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
    def is_vegetation():
        '''
        if self.training:
            idxs = torch.randperm(b*h*w).type_as(update).long()
                lc = data["landcover"].reshape(-1)  #flatten the landcover data
                idxs = idxs[(lc >= self.hparams.lc_min).byte() & (lc <= self.hparams.lc_max).byte()][:self.hparams.train_lstm_npixels*b]
                # idxs = torch.randint(low = 0, high = b*h*w, size = (self.hparams.train_lstm_npixels*b, )).type_as(update).long()
                if len(idxs) == 0:
                    print(f"Detected cube without vegetation: {data['cubename']}")
                    idxs = [1,2,3,4]

                state = state[:,idxs,:]
                update = update[idxs,:,:]
        '''
        # TODO
        return
    
    def input_data():
        # TODO
        return

    
