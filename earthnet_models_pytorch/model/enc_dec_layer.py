
import math

import torch
import torch.nn as nn

from earthnet_models_pytorch.model.layer_utils import ACTIVATIONS
from earthnet_models_pytorch.model.conditioning_layer import Identity_Conditioning, FiLMBlock, Cat_Project_Conditioning, CrossAttention_Conditioning

class PatchMergeDownsample(nn.Module):

    def __init__(self, channels, down_factor = 2, norm = None):
        super().__init__()

        self.down_factor = down_factor
    
        if norm == "group":
            self.norm = nn.GroupNorm(16, channels * down_factor**2)
        elif norm == "layer":
            self.norm = nn.LayerNorm(channels * down_factor**2)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(channels * down_factor**2)
        else:
            self.norm == nn.Identity()

        self.reduction = nn.Conv2d(channels * down_factor**2, channels, 1, stride = 1, padding = 0, bias = False)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.reshape(B, C, H//self.down_factor, self.down_factor, W//self.down_factor, self.down_factor).permute(0, 1, 3, 5, 2, 4).reshape(B, C * self.down_factor**2, H//self.down_factor, W//self.down_factor)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchMergeEncoder(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, down_factor = 4, filter_size = 5, norm = None, act = None, bias = True, conditioning = None, c_channels = None, n_tokens_c = 8):
        """
            down_factors // 2 downsample layers
        x -> conv -> down -> ... -> down -> conv -> x
        """
        super().__init__()

        if norm:
            bias = False

        conditioners = []
        layers = []
        downsamplers = []
        for i in range(int(math.sqrt(down_factor))):
            layer = []
            curr_in_channels = in_channels if i == 0 else hid_channels

            layer.append(
                nn.Conv2d(curr_in_channels, hid_channels, filter_size, stride = 1, padding = filter_size//2, bias = bias)
            )

            if conditioning == "cat":
                conditioners.append(Cat_Project_Conditioning(curr_in_channels, c_channels))
            elif conditioning == "FiLM":
                conditioners.append(FiLMBlock(c_channels, hid_channels, curr_in_channels))
            elif conditioning == "xAttn":
                n_heads = int(2**(math.log2(hid_channels)//2))
                conditioners.append(CrossAttention_Conditioning(curr_in_channels, c_channels, n_tokens_c=n_tokens_c, act = act, hidden_dim = hid_channels//n_heads, n_heads=n_heads))
            else:
                conditioners.append(Identity_Conditioning())

            if norm:
                if norm == "group":
                    layer.append(
                        nn.GroupNorm(16, hid_channels)
                    )
                elif norm == "batch":
                    layer.append(
                        nn.BatchNorm2d(hid_channels)
                    )
                elif norm == "layer":
                    layer.append(
                        nn.LayerNorm(hid_channels)
                    )
                else:
                    print("Norm {norm} not supported in PatchMergeEncoder")

            if act:
                layer.append(ACTIVATIONS[act]())
        
            downsamplers.append(PatchMergeDownsample(hid_channels, down_factor = 2, norm = norm))

            
            layers.append(nn.Sequential(*layer))

        self.conditioners = nn.ModuleList(conditioners)
        self.layers = nn.ModuleList(layers)
        self.downsamplers = nn.ModuleList(downsamplers)

        self.last_layer = nn.Sequential(
            nn.Conv2d(hid_channels, out_channels, filter_size, stride = 1, padding = filter_size//2, bias = bias)
        )
        if norm == "group":
            self.last_norm = nn.GroupNorm(16, out_channels)
        elif norm == "batch":
            self.last_norm = nn.BatchNorm2d(out_channels)
        elif norm == "layer":
            self.last_norm = nn.LayerNorm(out_channels)
        else:
            self.last_norm = nn.Identity()
        
        if act:
            self.last_act = ACTIVATIONS[act]()
        else:
            self.last_act = nn.Identity()

    def forward(self, x, c = None):

        skips = []
        for i in range(len(self.layers)):
            x = self.conditioners[i](x, c)
            if i != 0:
                x = x + self.layers[i](x)
            else:
                x = self.layers[i](x)
            skips.append(x)
            x = self.downsamplers[i](x)
        x = x + self.last_layer(x)
        x = self.last_norm(x)
        x = self.last_act(x)
        skips.append(x)

        return x, skips

class Up2dDecoder(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, up_factor = 4, filter_size = 5, skip_connection = True, norm = None, act = None, bias = True, readout_act = None, conditioning = None, c_channels = None, n_tokens_c = 8):
        """
            up_factors // 2 upsample layers
        conv(x+skip) -> up -> conv(x+skip) -> ... -> up -> conv(x+skip) -> x
        """
        super().__init__()

        self.skip_connection = skip_connection

        if norm:
            bias = False

        conditioners = []
        layers = []
        upsamplers = []
        for i in range(int(math.sqrt(up_factor))):
            layer = []
            curr_in_channels = in_channels if i == 0 else hid_channels

            layer.append(
                nn.Conv2d(curr_in_channels, hid_channels, filter_size, stride = 1, padding = filter_size//2, bias = bias)
            )


            if conditioning == "cat":
                conditioners.append(Cat_Project_Conditioning(curr_in_channels, c_channels))
            elif conditioning == "FiLM":
                conditioners.append(FiLMBlock(c_channels, hid_channels, curr_in_channels))
            elif conditioning == "xAttn":
                n_heads = int(2**(math.log2(hid_channels)//2))
                conditioners.append(CrossAttention_Conditioning(curr_in_channels, c_channels, n_tokens_c=n_tokens_c, act = act, hidden_dim = hid_channels//n_heads, n_heads=n_heads))
            else:
                conditioners.append(Identity_Conditioning())

            if norm:
                if norm == "group":
                    layer.append(
                        nn.GroupNorm(16, hid_channels)
                    )
                elif norm == "batch":
                    layer.append(
                        nn.BatchNorm2d(hid_channels)
                    )
                elif norm == "layer":
                    layer.append(
                        nn.LayerNorm(hid_channels)
                    )
                else:
                    print("Norm {norm} not supported in PatchMergeEncoder")
            
            if act:
                layer.append(ACTIVATIONS[act]())
        
            upsamplers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
            layers.append(nn.Sequential(*layer))
        
        self.conditioners = nn.ModuleList(conditioners)
        self.layers = nn.ModuleList(layers)
        self.upsamplers = nn.ModuleList(upsamplers)

        self.last_layer = nn.Sequential(
            nn.Conv2d(hid_channels, hid_channels, filter_size, stride = 1, padding = filter_size//2, bias = bias)
        )
        if norm == "group":
            self.last_norm = nn.GroupNorm(16, hid_channels)
        elif norm == "batch":
            self.last_norm = nn.BatchNorm2d(hid_channels)
        elif norm == "layer":
            self.last_norm = nn.LayerNorm(hid_channels)
        else:
            self.last_norm = nn.Identity()
        
        if act:
            self.last_act = ACTIVATIONS[act]()
        else:
            self.last_act = nn.Identity()

        self.readout = nn.Conv2d(hid_channels, out_channels, 1, stride = 1, padding = 0, bias = False)
        if readout_act:
            self.readout_act = ACTIVATIONS[readout_act]()
        else:
            self.readout_act = nn.Identity()
    
    def forward(self, x, skips, c = None):

        for i in range(len(self.layers)):
            if self.skip_connection:
                x = x + skips[-(1+i)]
            x = self.conditioners[i](x, c)
            x = x + self.layers[i](x)
            x = self.upsamplers[i](x)
        if self.skip_connection:
            x = x + skips[0]
        x = x + self.last_layer(x)
        x = self.last_norm(x)
        x = self.last_act(x)

        x = self.readout(x)
        x = self.readout_act(x)

        return x



class PredRNNEncoder(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, filter_size = 5, norm = False, relu = False, bias = False, conditioning = None, c_channels = None, n_tokens_c = 8):
        super().__init__()


        if conditioning == "cat":
            self.cond1 = Cat_Project_Conditioning(in_channels, c_channels)
            self.cond2 = Cat_Project_Conditioning(hid_channels, c_channels)
        elif conditioning == "FiLM":
            self.cond1 = FiLMBlock(c_channels, hid_channels, in_channels)
            self.cond2 = FiLMBlock(c_channels, hid_channels, hid_channels)
        elif conditioning == "xAttn":
            n_heads = int(2**(math.log2(hid_channels)//2))
            self.cond1 = CrossAttention_Conditioning(in_channels, c_channels, n_tokens_c=n_tokens_c, act = "relu", hidden_dim = hid_channels//n_heads, n_heads=n_heads)
            self.cond2 = CrossAttention_Conditioning(hid_channels, c_channels, n_tokens_c=n_tokens_c, act = "relu", hidden_dim = hid_channels//n_heads, n_heads=n_heads)
        else:
            self.cond1 = Identity_Conditioning()
            self.cond2 = Identity_Conditioning()

        self.conv1 = nn.Conv2d(in_channels, hid_channels, filter_size, stride = 2, padding = filter_size//2, bias = bias)

        self.conv2 = nn.Conv2d(hid_channels, out_channels, filter_size, stride = 2, padding = filter_size//2, bias = bias)
        
        self.norm_on_conv = norm
        if self.norm_on_conv:
            self.norm1 = nn.GroupNorm(2, hid_channels)
            self.norm2 = nn.GroupNorm(2, out_channels)
        self.relu_on_conv = relu
        if self.relu_on_conv:
            self.relu = nn.ReLU()

    def forward(self, x, c = None):

        x = self.cond1(x, c)
        net = self.conv1(x)

        if self.norm_on_conv:
            net = self.norm1(net)
        if self.relu_on_conv:
            net = self.relu(net)

        input_net1 = net

        net = self.cond2(net, c)
        net = self.conv2(net)

        if self.norm_on_conv:
            net = self.norm2(net)
        if self.relu_on_conv:
            net = self.relu(net)

        input_net2 = net

        return net, [x, input_net1, input_net2]

class PredRNNDecoder(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, filter_size = 5, residual = True, norm = False, relu = False, bias = False, readout_act = None, conditioning = None, c_channels = None, n_tokens_c = 8):
        super().__init__()

        if conditioning == "cat":
            self.cond1 = Cat_Project_Conditioning(in_channels, c_channels)
            self.cond2 = Cat_Project_Conditioning(hid_channels, c_channels)
        elif conditioning == "FiLM":
            self.cond1 = FiLMBlock(c_channels, hid_channels, in_channels)
            self.cond2 = FiLMBlock(c_channels, hid_channels, hid_channels)
        elif conditioning == "xAttn":
            n_heads = int(2**(math.log2(hid_channels)//2))
            self.cond1 = CrossAttention_Conditioning(in_channels, c_channels, n_tokens_c=n_tokens_c, act = "relu", hidden_dim = hid_channels//n_heads, n_heads=n_heads)
            self.cond2 = CrossAttention_Conditioning(hid_channels, c_channels, n_tokens_c=n_tokens_c, act = "relu", hidden_dim = hid_channels//n_heads, n_heads=n_heads)
        else:
            self.cond1 = Identity_Conditioning()
            self.cond2 = Identity_Conditioning()
        
        self.deconv1 = nn.ConvTranspose2d(in_channels, hid_channels, filter_size, stride = 2, padding = filter_size//2, bias = bias)

        self.deconv2 = nn.ConvTranspose2d(hid_channels, out_channels, filter_size, stride = 2, padding = filter_size//2, bias = bias)
        
        self.norm_on_conv = norm
        if self.norm_on_conv:
            self.norm1 = nn.GroupNorm(2, hid_channels)
        self.res_on_conv = residual
        self.relu_on_conv = relu
        if self.relu_on_conv:
            self.relu = nn.ReLU()
        if readout_act:
            self.readout_act = ACTIVATIONS[readout_act]()
        else:
            self.readout_act = nn.Identity()

    def forward(self, x, skips, c = None):
        if self.res_on_conv:
            x = self.cond1(skips[2] + x, c)
            x = self.deconv1(x, output_size = skips[1].size())

            if self.norm_on_conv:
                x = self.norm1(x)
            if self.relu_on_conv:
                x = self.relu(x)

            x = self.cond2(skips[1] + x, c)
            x = self.deconv2(x, output_size = skips[0].size())

        else:
            x = self.cond1(x, c)
            x = self.deconv1(x, output_size = skips[1].size())

            if self.norm_on_conv:
                x = self.norm1(x)
            if self.relu_on_conv:
                x = self.relu(x)

            x = self.cond2(x, c)
            x = self.deconv2(x, output_size = skips[0].size())

        x = self.readout_act(x)
        return x



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class SimVPEncoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S, conditioning = None, c_channels = None, n_tokens_c = 8):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

        conditioners = []
        for i, s in enumerate(strides):
            
            if (i > 0) & (s == 2):
                conditioners.append(Identity_Conditioning())
                continue

            curr_in_channels = C_in if i == 0 else C_hid
            

            if conditioning == "cat":
                conditioners.append(Cat_Project_Conditioning(curr_in_channels, c_channels))
            elif conditioning == "FiLM":
                conditioners.append(FiLMBlock(c_channels, C_hid, curr_in_channels))
            elif conditioning == "xAttn":
                n_heads = 2**(math.log2(C_hid)//2)
                conditioners.append(CrossAttention_Conditioning(curr_in_channels, c_channels, n_tokens_c=n_tokens_c, act = "relu", hidden_dim = C_hid//n_heads, n_heads=n_heads))
            else:
                conditioners.append(Identity_Conditioning())
        
        self.conditioners = nn.ModuleList(conditioners)
    
    def forward(self,x, c = None):# B*4, 3, 128, 128
        x = self.conditioners[0](x, c)
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.conditioners[i](latent, c)
            latent = self.enc[i](latent)
        return latent,enc1


class SimVPDecoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S, readout_act = None, conditioning = None, c_channels = None, n_tokens_c = 8):
        super().__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        if readout_act:
            self.readout_act = ACTIVATIONS[readout_act]()
        else:
            self.readout_act = nn.Identity()

        
        conditioners = []
        for i, s in enumerate(strides):
            
            if (i > 0) & (s == 2):
                conditioners.append(Identity_Conditioning())
                continue

            curr_in_channels = 2*C_hid if i == len(strides)-1 else C_hid
            
            if conditioning == "cat":
                conditioners.append(Cat_Project_Conditioning(curr_in_channels, c_channels))
            elif conditioning == "FiLM":
                conditioners.append(FiLMBlock(c_channels, C_hid, curr_in_channels))
            elif conditioning == "xAttn":
                n_heads = 2**(math.log2(C_hid)//2)
                conditioners.append(CrossAttention_Conditioning(curr_in_channels, c_channels, n_tokens_c=n_tokens_c, act = "relu", hidden_dim = C_hid//n_heads, n_heads=n_heads))
            else:
                conditioners.append(Identity_Conditioning())
        
        self.conditioners = nn.ModuleList(conditioners)
    
    def forward(self, hid, enc1=None, c = None):
        for i in range(0,len(self.dec)-1):
            hid = self.conditioners[i](hid, c)
            hid = self.dec[i](hid)
        hid = self.conditioners[-1](torch.cat([hid, enc1], dim=1), c)
        Y = self.dec[-1](hid)
        Y = self.readout(Y)
        Y = self.readout_act(Y)
        return Y


def get_encoder_decoder(type, in_channels, hid_channels, out_channels, down_factor = 4, filter_size = 5, skip_connection = True, norm = "group", act = "leakyrelu", bias = True, readout_act = "tanh", conditioning = None, c_channels_enc = None, c_channels_dec = None, n_tokens_c = 8):

    if type == "PatchMerge":
        encoder = PatchMergeEncoder(
            in_channels, hid_channels, hid_channels, down_factor=down_factor, filter_size = filter_size, norm = norm, act = act, bias = bias, conditioning = conditioning, c_channels = c_channels_enc, n_tokens_c = n_tokens_c
        )
        decoder = Up2dDecoder(
            hid_channels, hid_channels, out_channels, up_factor=down_factor, filter_size = filter_size, skip_connection = skip_connection, norm = norm, act = act, bias = bias, readout_act= readout_act, conditioning = conditioning, c_channels = c_channels_dec, n_tokens_c = n_tokens_c
        )

    elif type == "PredRNN":
        encoder = PredRNNEncoder(
            in_channels, hid_channels//2, hid_channels, filter_size=filter_size, norm = (norm is not None) , relu = (act is not None), bias = bias, conditioning = conditioning, c_channels = c_channels_enc, n_tokens_c = n_tokens_c
        )
        decoder = PredRNNDecoder(
            hid_channels, hid_channels//2, out_channels, filter_size=filter_size, residual=skip_connection, norm = (norm is not None), relu = (act is not None), bias = bias, conditioning = conditioning, c_channels = c_channels_dec, n_tokens_c = n_tokens_c
        )
    
    elif type == "SimVP":
        encoder = SimVPEncoder(in_channels, hid_channels, int(2 * math.sqrt(down_factor)), conditioning = conditioning, c_channels = c_channels_enc, n_tokens_c = n_tokens_c)
        decoder = SimVPDecoder(hid_channels, out_channels, int(2 * math.sqrt(down_factor)), conditioning = conditioning, c_channels = c_channels_dec, n_tokens_c = n_tokens_c)

    return encoder, decoder