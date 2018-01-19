import torch as tt
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn import init
from torch import optim
from functools import partial

F.conv2d()


class SeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(SeperableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = False

    def forward(self, x):
        weight =
        padding = self.padding
        groups =
        F.conv2d(x, weight=, bias=None, stride=1, padding=padding, dilation=1, groups=groups)
        del weight
