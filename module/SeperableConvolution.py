import torch as tt
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn import init
from torch import optim
from functools import partial
from torch.nn import init


def _pair(*args):
    if len(args) == 1:
        return args * 2
    else:
        return args


class SeparableConv(nn.Module):
    def __init__(self, in_channels, num_kernels, kernel_size=3, padding=1):
        super(SeparableConv, self).__init__()
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.out_channels = num_kernels * in_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.bias = False

        # weight_shape = num_kernels * kernel_size^2
        # the weight's shape should be (c_o, c_i, kH, kW), here we treat each input plane separately, so
        # c_i == 1
        self.weight = nn.Parameter(data=tt.ones(num_kernels, 1, kernel_size, kernel_size))
        # for ii in range(num_kernels):
        #     self.weight.data[ii].fill_(ii)  # only for test
        self.init_weights()

    def init_weights(self):
        for param_name, param_value in self.named_parameters():
            if 'weight' in param_name:
                init.xavier_normal(param_value.data)
            if 'bias' in param_name:
                param_value.data.fill_(0.0)

    def forward(self, x):
        c_i = self.in_channels

        # the weight's shape should be (c_o, c_i, kH, kW)
        weight = self.weight.repeat(c_i, 1, 1, 1)
        padding = self.padding
        groups = c_i
        x = F.conv2d(x, weight=weight, bias=None, stride=1, padding=padding, dilation=1, groups=groups)
        del weight
        return x


if __name__ == '__main__':
    sep_conv = SeparableConv(3, 5)
    var_fake_data = Variable(tt.zeros(2, 3, 64, 64))
    for ii in range(3):
        var_fake_data[:, ii] = 1
    var_out = sep_conv(var_fake_data)
    print(var_out.size())
    print(var_out[0, :, 10, 10])
