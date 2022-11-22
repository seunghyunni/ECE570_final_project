import datetime
import os, sys
import random
import argparse
import numpy as np

from torch.autograd import Variable

from torch.utils.data import DataLoader
import torch
from torch import nn

import torch.nn.functional as F
import pdb
import cv2

import time

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
import math
import torch.nn.utils.weight_norm as WeightNorm
# from memcnn.models.revop import InvertibleModuleWrapper, create_coupling
# import memcnn
import pdb


def norm(dim):
    return nn.GroupNorm(min(36, dim), dim)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class DCFfunc(nn.Module):

    def __init__(self, dim):
        super(DCFfunc, self).__init__()
        self.norm1 = norm(dim)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Sigmoid()
        self.conv1 = ConcatConv2d(dim, dim, 1, 1, 0)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 1, 1, 0)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


# 
BIAS = True
# BIAS = False

std = 1.0

NUM_BASES = 8
# NUM_BASES = 16
# NUM_BASES = 128

# NUM_BASES = 14
# NUM_BASES = 12

# INTER_SIZE = 512
# INTER_SIZE = 128
INTER_SIZE = 64

# torch.manual_seed(1)

class norm_layer(nn.Module):
    def __init__(self):
        super(norm_layer, self).__init__()
        pass
    def forward(self, x):
        x = x - x.mean()
        x = x / x.std()
        return x


class rand_base_generator_dcfnet(nn.Module):
    def __init__(self, num_layers, num_bases, kernel_size, transpose=False, fix=False):
        super(rand_base_generator_dcfnet, self).__init__()
        self.fix = fix

        dim = num_layers*kernel_size*kernel_size*num_bases

        # self.random_vec = Parameter(torch.randn((1, dim, 1, 1)), requires_grad=False)
        self.random_vec = Parameter(torch.randn((1, dim, 1, 1)))


        self.dcf = DCFfunc(dim)

        self.num_bases = num_bases
        self.kernel_size = kernel_size
        if transpose:
            self.view_shape = (num_layers, 1, self.num_bases, self.kernel_size, self.kernel_size)
        else:
            self.view_shape = (num_layers, self.num_bases, self.kernel_size, self.kernel_size)

        self.random_vec.data.normal_(0.0, std)
        self.termination = None
        
    def forward(self):
        t = torch.zeros(2).cuda()
        t[1].uniform_(0.2, 1.0) # for stochastic training

        return self.dcf(self.random_vec, t[1]).view(self.view_shape) 

rand_base_generator = rand_base_generator_dcfnet

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class _ConvTransposeMixin(object):

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])


class ConvTranspose2dr(_ConvTransposeMixin, _ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2dr, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        self.weight.data.normal_(0.0, 0.1)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class Conv_DCFre(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
        num_bases=NUM_BASES, bias=True,  base_grad=False, initializer='random', fix=False):
        super(Conv_DCFre, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.edge = int((kernel_size-stride)/2)
        self.stride = stride
        self.padding = padding
        self.kernel_list = {}
        self.num_bases = num_bases
        self.output_padding = output_padding

        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels, num_bases))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.num_bases = num_bases

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv) #Normal works better, working on more robust initializations
        if self.bias is not None:
            self.bias.data.zero_()


    def forward(self, input):
        # pdb.set_trace()
        # self.bases = self.bases_generator().view(self.num_bases, self.kernel_size*self.kernel_size)
        # pdb.set_trace()
        rec_kernel = torch.einsum('cvb, bkl->cvkl', self.weight, self.bases)
        self.rec_filter = rec_kernel
        
        feature = F.conv2d(input, rec_kernel,
            self.bias, self.stride, self.padding, dilation=1)
        self.feature = feature
         
        return feature


if __name__ == '__main__':
    net = rev_block(32).cuda()
    data = torch.randn(10, 32).cuda()
    
