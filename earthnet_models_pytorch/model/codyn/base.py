


"""
Shapes API

What do I need?

Very flexible...

So: 

"""

import torch
from torch import nn

class Shape():

    def __init__(self, shape, order: str):
        assert(len(order) == len(shape))
        self.order = order
        for idx, o in enumerate(order):
            setattr(self, o, shape[idx])

    def __ne__(self, other):
        order = set([o for o in self.order]).intersection(set([o for o in other.order]))
        equal = True
        for o in order:
            if getattr(self, o) != getattr(other, o):
                equal = False
        return not equal

    def __eq__(self, other):
        return not self.__ne__(other)

    def __str__(self):
        outstring = "("
        for o in self.order:
            outstring += "({}, {})".format(o, getattr(self, o))
        outstring += ")"
        return outstring

class Shapes():

    def __init__(self, **kwargs):
        all_shapes = {}
        for attribute, value in kwargs.items():
            setattr(self, attribute, Shape(**value))
            all_shapes[attribute] = getattr(self, attribute)
        self.all_shapes = all_shapes

    def __str__(self):
        outstring = "Shapes instance with attributes: \n"
        for shape in self.all_shapes:
            outstring += "\t {}, dimensions {}\n".format(shape, str(self.all_shapes[shape]))
        return outstring

class InstanceNorm(nn.Module):

    def __init__(self, channels):
        super(InstanceNorm, self).__init__()

        self.norm = nn.InstanceNorm2d(channels, affine = True)
    
    def forward(self, input):
        return self.norm(input)

Activations = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh }

Activation_Default_Args = {"relu": {"inplace": True}, "leaky_relu": {"negative_slope": 0.2, "inplace": True}, "elu": {"inplace": True}, "sigmoid": {}, "tanh": {}}

Norms = {"bn": nn.BatchNorm2d, "in": InstanceNorm}



def make_lin_block(ninp, nout, activation):
    """
    Creates a linear block formed by an activation function and a linear operation.

    Parameters
    ----------
    ninp : int
        Input dimension.
    nout : int
        Output dimension.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function before
        the linear operation.
    """
    modules = []
    if activation is not None:
        modules.append(Activations[activation](**Activation_Default_Args[activation]))
    modules.append(nn.Linear(ninp, nout))
    return nn.Sequential(*modules)


class MLP(nn.Module):
    """
    Module implementing an MLP.
    """
    def __init__(self, ninp, nhid, nout, nlayers, activation='relu'):
        """
        Parameters
        ----------
        ninp : int
            Input dimension.
        nhid : int
            Number of dimensions in intermediary layers.
        nout : int
            Output dimension.
        nlayers : int
            Number of layers in the MLP.
        activation : str
            'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function before the
            linear operation.
        """
        super().__init__()
        assert nhid == 0 or nlayers > 1
        if isinstance(activation, str) or activation is None:
            activation = [activation for i in range(nlayers)]
        if 'none' in activation:
            activation = [(act if act != 'none' else None) for act in activation]
        modules = [
            make_lin_block(
                ninp=ninp if il == 0 else nhid,
                nout=nout if il == nlayers - 1 else nhid,
                activation=activation[il],
            ) for il in range(nlayers)
        ]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)




class Conv_Block(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = None, bias = True, norm = None, activation = None, activation_kwargs = None):
        super(Conv_Block, self).__init__()

        if padding is None:
            if kernel_size%2 == 1:
                padding = kernel_size // 2
            else:
                pad_big = kernel_size//2
                pad_small = (kernel_size-1)//2
                self.add_module("pad", nn.ZeroPad2d((pad_small,pad_big,pad_small,pad_big)))
                padding = 0

        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = bias))

        if norm is not None:
            self.add_module("norm", Norms[norm](out_channels))
        
        if activation is not None:
            if activation_kwargs is None:
                activation_kwargs = Activation_Default_Args[activation]
            self.add_module("activation", Activations[activation](**activation_kwargs))

class Up_Conv_Block(Conv_Block):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = None, bias = True, norm = None, activation = None, activation_kwargs = None):   
        super(Up_Conv_Block, self).__init__(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias, norm = norm, activation = activation, activation_kwargs = activation_kwargs)
        
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = bias)


class Mask_Block(nn.Module):
    def __init__(self, kernel_sizes: list, out_channels: int, strict: bool = False):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.strict = strict

    def forward(self, masks):
        if self.strict:
            masks = 1.*(masks<1)

        masks = masks.min(1).values.unsqueeze(1)

        for kernel_size in self.kernel_sizes:
            dilation_kernel = torch.ones(1,1,kernel_size,kernel_size).type_as(masks)/(kernel_size*kernel_size)
            if kernel_size%2 == 1:
                padding = kernel_size // 2
            else:
                pad_big = kernel_size//2
                pad_small = (kernel_size-1)//2
                masks = nn.functional.pad(masks, (pad_small,pad_big,pad_small,pad_big))
                padding = 0
            masks = nn.functional.conv2d(masks, dilation_kernel, stride = 1, padding=padding)

        if self.strict:
            return (1-1.*(masks > 0)).type_as(masks).repeat(1,self.out_channels,1,1)
        else:
            return (masks > 0).type_as(masks).repeat(1,self.out_channels,1,1)





def init_weight(m, init_type='normal', init_gain=0.02):
    """
    Initializes the input module with the given parameters.

    Parameters
    ----------
    m : torch.nn.Module
        Module to initialize.
    init_type : str
        'normal', 'xavier', 'kaiming', or 'orthogonal'. Orthogonal initialization types for convolutions and linear
        operations.
    init_gain : float
        Gain to use for the initialization.
    """
    classname = m.__class__.__name__
    if classname in ('Conv2d', 'ConvTranspose2d', 'Linear'):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname == 'BatchNorm2d':
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
