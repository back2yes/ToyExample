import torch as tt
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable


class IOUtils(object):
    def __init__(self, is_cuda):
        self.set_default_tensor_type(is_cuda)

    def set_default_tensor_type(self, is_cuda=False):
        self.is_cuda = is_cuda
        if is_cuda:
            tt.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            tt.set_default_tensor_type('torch.FloatTensor')

    def to_var(self, x):
        if self.is_cuda:
            return Variable(x).cuda()
        else:
            return Variable(x)
