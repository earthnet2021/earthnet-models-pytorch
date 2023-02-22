
import torch
import torch.nn as nn


ACTIVATIONS = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "hardsigmoid": nn.Hardsigmoid,
    "hardtanh": nn.Hardtanh
}



def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv