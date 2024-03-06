



import torch
import torch.nn as nn
from earthnet_models_pytorch.model.layer_utils import ACTIVATIONS

class MLP2d(nn.Module):

    def __init__(self, n_in, n_hid, n_out, act = "relu", groups = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(n_in, n_hid, 1, groups = groups)
        self.act = ACTIVATIONS[act]()
        self.conv2 = nn.Conv2d(n_hid, n_out, 1, groups = groups)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class FiLM(nn.Module):

    def __init__(self, n_c, n_hid, n_x, act = "relu"):
        super().__init__()

        self.beta = MLP2d(n_c, n_hid, n_x, act = act)
        self.gamma = MLP2d(n_c, n_hid, n_x, act = act)

    def forward(self, x, c):
        
        if len(c.shape) == 2:
            c = c[:, :, None, None]

        beta = self.beta(c)
        gamma = self.gamma(c)

        B, C, H, W = x.shape
        B, C_, H_, W_ = beta.shape

        if H_ == 1:
            beta = beta.expand(-1, -1, H, W)
            gamma = gamma.expand(-1, -1, H, W)
        
        elif H!=H_:
            beta = nn.functional.interpolate(beta, size = (H, W), mode='nearest-exact')
            gamma = nn.functional.interpolate(gamma, size = (H, W), mode='nearest-exact')


        return beta + gamma * x

class FiLMBlock(nn.Module):

    def __init__(self, n_c, n_hid, n_x, norm = "group", act = "leakyrelu"):
        super().__init__()
        self.FiLM = FiLM(n_c, n_hid, n_x, act = act)
        self.conv = nn.Conv2d(n_x, n_x, 1)
        if norm == "group":
            groups = 16 if (n_x > 16) and (n_x%16 == 0) else 2
            self.norm = nn.GroupNorm(groups, n_x)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(n_x)
        elif norm == "layer":
            self.norm = nn.LayerNorm(n_x)
        else:
            self.norm = nn.Identity()
        self.act = ACTIVATIONS[act]()

    def forward(self, x, c):
        x2 = self.conv(x)
        x2 = self.norm(x2)
        x2 = self.FiLM(x2, c)
        x2 = self.act(x2)
        return x + x2

class Cat_Conditioning(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, c):
        
        if len(c.shape) == 2:
            c = c[:, :, None, None]
        
        B, C, H, W = x.shape
        B, C_, H_, W_ = c.shape

        if H_ == 1:
            c = c.expand(-1, -1, H, W)
        elif H!=H_:
            c = nn.functional.interpolate(c, size = (H, W), mode='nearest-exact')

        return torch.cat([x, c], dim = 1).contiguous()
    
class Cat_Project_Conditioning(nn.Module):
    def __init__(self, x_channels, c_channels, out_channels = None):
        super().__init__()

        if out_channels is None:
            out_channels = x_channels
        
        self.conv = nn.Conv2d(x_channels+c_channels, out_channels, 1, bias = False)

    def forward(self, x, c):

        if len(c.shape) == 2:
            c = c[:, :, None, None]
        
        B, C, H, W = x.shape
        B, C_, H_, W_ = c.shape

        if H_ == 1:
            c = c.expand(-1, -1, H, W)
        
        elif H!=H_:
            c = nn.functional.interpolate(c, size = (H, W), mode='nearest-exact')

        return self.conv(torch.cat([x, c], dim = 1).contiguous())
    

class Identity_Conditioning(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, c):
        return x

class CrossAttention_Conditioning(nn.Module):

    def __init__(self, x_channels, c_channels, n_tokens_c, hidden_dim = 16, n_heads = 16, act = "leakyrelu", norm = "layer", mlp_after_attn = False):
        super().__init__()

        self.embed_dim = hidden_dim*n_heads
        self.n_tokens_c = n_tokens_c

        self.project_x = nn.Conv2d(x_channels, self.embed_dim, 1, bias = False)

        self.embed_c = MLP2d(c_channels, n_tokens_c*hidden_dim, n_tokens_c*self.embed_dim, act = act, groups = n_tokens_c)

        self.attn = nn.MultiheadAttention(self.embed_dim, n_heads, batch_first=True)

        if norm == "group":
            self.norm = nn.GroupNorm(n_heads, self.embed_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(self.embed_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.norm = nn.Identity()

        if mlp_after_attn:
            self.mlp_after_attn = MLP2d(self.embed_dim, self.embed_dim//2, self.embed_dim, act = act)
        else:
            self.mlp_after_attn = None

        self.project_h = nn.Conv2d(self.embed_dim, x_channels, 1, bias = False)

    def forward(self, x, c):

        B, C, H, W = x.shape

        x_e = self.project_x(x)

        if len(c.shape) == 2:
            c = c[:, :, None, None]

        c_e = self.embed_c(c)

        B, C_, H_, W_ = c_e.shape

        if H_ == 1:
            c_e = c_e.expand(-1, -1, H, W)        
        elif H!=H_:
            c_e = nn.functional.interpolate(c_e, size = (H, W), mode='nearest-exact')

        x_e = x_e.permute(0, 2, 3, 1).reshape(-1, 1, self.embed_dim)

        c_e = c_e.permute(0, 2, 3, 1).reshape(-1, self.n_tokens_c, self.embed_dim)

        h = self.attn(x_e, c_e, c_e, need_weights = False)[0]

        h = self.norm(x_e + h)

        h = h.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        if self.mlp_after_attn:
            h = h + self.mlp_after_attn(h)
            if isinstance(self.norm, nn.LayerNorm):
                h = self.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                h = self.norm(h)

        x = x + self.project_h(h)

        return x