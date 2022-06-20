import numpy as np  
from glob import glob 
from tqdm import tqdm 

from turtle import forward
from typing import Optional, Union, List
import argparse
import ast
from grpc import dynamic_ssl_server_credentials
from pip import main
import torch.nn as nn
import torch 
import sys
from earthnet_models_pytorch.utils import str2bool

import torch
import numpy as np
import pandas as pd 
from dataset_preprocessing import data_loading_for_years, torch_data 
from glob import glob 
from tqdm import tqdm 
from torch import optim 
import torch.nn as nn 
from feature_loss import SSIM
#from feature_loss import SSIM

class self_attention_memory_module(nn.Module): #SAM 
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        # h(hidden)를 위한 layer q, k, v 
        # m(memory)을 위한 layer k2, v2 
        #layer z, m은 attention_h와 attention_m concat 후의 layer.  
        self.layer_q = nn.Conv2d(input_dim, hidden_dim ,1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim,1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim,1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_z = nn.Conv2d(input_dim *2, input_dim*2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        #feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H*W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H*W)
        Q_h = Q_h.transpose(1, 2)
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim = -1) #batch_size, H*W, H*W
        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H*W)
        Z_h = torch.matmul(A_h, V_h.permute(0,2,1))

        ###### memory m attention #####
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H*W)
        V_m = V_m.view(batch_size, self.input_dim, H*W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim = -1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0,2,1))
        Z_h = Z_h.transpose(1,2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1,2).view(batch_size, self.input_dim, H, W)

        ### attention으로 구한 Z_h와 Z_m concat 후 계산 ####
        W_z = torch.cat([Z_h , Z_m], dim = 1)
        Z = self.layer_z(W_z)
        ## Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim = 1)) # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim = 1)
        ### 논문의 수식과 같습니다(figure)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m 

        return  new_h, new_m 
        
class SA_Convlstm_cell(nn.Module):
    def __init__(self, params):
        super().__init__()
        #hyperparrams 
        self.input_channels = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.kernel_size= params['kernel_size']
        self.padding = params['padding']
        self.device = params['device']
        self.attention_layer = self_attention_memory_module(params['hidden_dim'], params['att_hidden_dim'], self.device)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = self.input_channels + self.hidden_dim, out_channels = 4 * self.hidden_dim, kernel_size= self.kernel_size, padding = self.padding)
                     ,nn.GroupNorm(4* self.hidden_dim, 4* self.hidden_dim ))   #(num_groups, num_channels)     

    def forward(self, x, hidden):
        h, c, m = hidden
        combined = torch.cat([x, h], dim = 1) #combine 해서 (batch_size, input_dim + hidden_dim, img_size[0], img_size[1])
        combined_conv = self.conv2d(combined)# conv2d 후 (batch_size, 4 * hidden_dim, img_size[0], img_size[1])
        i, f, o ,g =torch.split(combined_conv, self.hidden_dim, dim =1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i*g
        h_next = o * torch.tanh(c_next)
        #### 위까지는 보통의 convlstm과 같음. 
        ### attention 해주기 
        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, (h_next, c_next, m_next)
    
    def init_hidden(self, batch_size, img_size): #h, c, m initalize
        h, w = img_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
               torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
               torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device))

class Sa_convlstm(nn.Module): #self-attention convlstm for spatiotemporal prediction model
    def __init__(self, params):
        super(Sa_convlstm, self).__init__()
        #hyperparams 

        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.cells, self.bns = [], []
        self.n_layers = params['n_layers']
        self.input_window_size = params['input_window_size']
        for i in range(params['n_layers']):
            params['input_dim'] = params['input_dim'] if i == 0 else params['hidden_dim']
            params['hidden_dim'] = params['hidden_dim'] if i != params['n_layers']-1 else 1 
            self.cells.append(SA_Convlstm_cell(params))
            self.bns.append(nn.LayerNorm((params['hidden_dim'], 56, 38))) #layernorm 사용
        self.cells = nn.ModuleList(self.cells)
        self.bns = nn.ModuleList(self.bns)

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        # TODO remove the useless features
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)
    #params= {'input_dim': 1 ,'batch_size' : BATCH_SIZE, 'padding':1, 'lr' : lr, 'device':device, 'att_hidden_dim':att_hid_dim, 'kernel_size':3, 'img_size':img_size, 'hidden_dim': 16 , 'n_layers': 3, 'output_dim': output, 'input_window_size':input_window_size, 'loss':loss}
        '''if torch.cuda.is_available() :
            device = torch.device('cuda')
        else : 
            device = torch.device('cpu')
        torch.cuda.empty_cache() 
        set_device = 4
        torch.cuda.set_device(set_device)
        print(set_device)
        torch.manual_seed(42)
        BATCH_SIZE= 8 
        img_size = (448,304)
        new_size = (56, 38)
        input_window_size, output  = 12,12 '''

        # parser.add_argument("--input_dim", type=int, default=6)
        parser.add_argument("--hidden_dim", type=ast.literal_eval, default=[64, 64, 64, 64])  
        
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--bias", type=str2bool, default=True)
        parser.add_argument("--return_all_layers", type=str2bool, default=False)
        parser.add_argument("--setting", type = str, default = "en22")
        parser.add_argument("--context_length", type = int, default = 9)
        parser.add_argument("--target_length", type = int, default = 36)
        parser.add_argument("--lc_min", type = int, default = 82)
        parser.add_argument("--lc_max", type = int, default = 104)
        parser.add_argument("--method", type = str, default = None)
        parser.add_argument("--input", type = str, default = None)
        parser.add_argument("--skip_connections", type = str2bool, default=False)
        parser.add_argument("--att_hid_dim", type = int, default=16)
        return parser


    def forward(self, data, hidden = None, pred_start: int = 0):
        c_l = self.hparams.context_length if self.training else pred_start

        # Data
        hr_dynamics = data["dynamic"][0][:,(c_l - self.hparams.context_length):c_l,...]
        target = hr_dynamics[:,:,0,...].unsqueeze(2)
        weather = data["dynamic"][1].unsqueeze(3).unsqueeze(4)
        topology = data["static"][0]
        landcover = data["landcover"]
        b, t, _, h, w = data["dynamic"][0].shape

        if hidden == None:
            hidden = self.init_hidden(batch_size = self.batch_size, img_size = self.img_size)
        
        input_x = x
    
        for i, layer in enumerate(self.cells):
            out_layer = []
            hid = hidden[i]
            for t in range(self.input_window_size):
                input_x = x[:,t,:,:,:]
                out, hid = layer(input_x, hid)
                out = self.bns[i](out)
                out_layer.append(out)
            out_layer = torch.stack(out_layer, dim = 1) #output predictions들 저장. 
            x = out_layer
        return out_layer 

    def init_hidden(self, batch_size, img_size):
        states = [] 
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))
        return states 






