import torch as tt
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn import init
from torch import optim
from functools import partial


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Coder(nn.Module):
    def __init__(self, coder_name, is_encoder=True):
        super(Coder, self).__init__()
        self.coder = None
        self.build_coder(coder_name, is_encoder=is_encoder)
        # self.build_tiny_coder()

    def forward(self, x):
        return self.coder(x)

    def init_weight(self):
        for param_name, param_value in self.named_parameters():
            # print(param_name)
            if 'weight' in param_name:
                init.xavier_normal(param_value)
            if 'bias' in param_name:
                param_value.data.fill_(0.0)

    def build_general_coder(self, in_channels, filter_nums, filter_size, block_sugar=None, final_act=None, extra=None):
        self.coder = None
        self.coder = nn.Sequential()
        num_layers = len(filter_nums)
        assert filter_size % 2 == 1, '**filter size must be odd**'
        padding = filter_size // 2
        filter_nums = (in_channels,) + tuple(filter_nums)
        if block_sugar is None:
            block_sugar = Identity()

        for ii, _ in enumerate(filter_nums[1:], 1):
            # print(ii, _)
            i_c, o_c = filter_nums[ii - 1], filter_nums[ii]
            conv = nn.Conv2d(i_c, o_c, filter_size, padding=padding, bias=False)
            self.coder.add_module('conv.{}'.format(ii), conv)
            if ii < num_layers:
                self.coder.add_module('sugar.{}'.format(ii), block_sugar)
            else:
                if final_act is None:
                    final_act = Identity()
                if extra is not None:
                    for ee in range(1, extra + 1):
                        c_extra = filter_nums[-1]
                        conv = nn.Conv2d(c_extra, c_extra, filter_size, padding=padding, bias=False)
                        relu = nn.ReLU(inplace=True)
                        self.coder.add_module('extra.conv.{}'.format(ee), conv)
                        self.coder.add_module('extra.relu.{}'.format(ee), relu)
                self.coder.add_module('final_act', final_act)
            # setattr(self.coder, 'conv.{}'.format(ii), conv)
        self.init_weight()

    @staticmethod
    def downsampling_sugar():
        pool = nn.MaxPool2d(3, 2, 1)
        relu = nn.ReLU(inplace=True)
        sugar = nn.Sequential(pool, relu)
        return sugar

    @staticmethod
    def upsampling_sugar():
        pool = nn.Upsample(scale_factor=2)
        relu = nn.ReLU(inplace=True)
        sugar = nn.Sequential(pool, relu)
        return sugar

    def build_encoder(self, num_channels, filter_size, extra):
        sugar = self.downsampling_sugar()
        self.build_general_coder(num_channels[0], num_channels[1:], filter_size, block_sugar=sugar, extra=extra)

    def build_decoder(self, num_channels, filter_size, extra):
        sugar = self.upsampling_sugar()
        self.build_general_coder(num_channels[-1], num_channels[-2::-1], filter_size, block_sugar=sugar, extra=extra)

    def build_tiny(self, is_encoder=True):
        # [16, 64, 64, 128, 128] are the output channels of each kernel
        num_channels = [3, 16, 64, 128, 256]
        if is_encoder:
            self.build_encoder(num_channels, 3, 2)
        else:
            self.build_decoder(num_channels, 3, 2)

    def build_tinier(self, is_encoder=True):
        # [16, 64, 64, 128, 128] are the output channels of each kernel
        num_channels = [3, 16, 64, 128]
        if is_encoder:
            self.build_encoder(num_channels, 3, 1)
        else:
            self.build_decoder(num_channels, 3, 1)

    def build_coder(self, name, is_encoder=True):
        build_name = 'build_{}'.format(name)
        if hasattr(self, build_name):
            build_function = getattr(self, build_name)
        else:
            raise NotImplementedError
        build_function(is_encoder=is_encoder)
    # def build_tiny_coder(self):
    #     self.build_general_coder(3, [4, 8], 3)


class ToySeparableAE(nn.Module):
    def __init__(self, ae_id, lr=1e-5):
        super(ToySeparableAE, self).__init__()
        self.lr = lr
        self.build_ae(ae_id)

    def build_ae(self, ae_id):
        self.encoder = Coder(ae_id, is_encoder=True)
        self.decoder = Coder(ae_id, is_encoder=False)
        self.build_optimizer()
        self._l1_criterion = nn.L1Loss()
        self._l2_criterion = nn.MSELoss()
        # self.train_process = self.train_tiny
        self.train_process = self.train_sparse_tiny

    # def build_tiny_ae(self):
    #     self.encoder = Encoder('tinier')
    #     self.decoder = Decoder('tinier')
    #     self.build_optimizer()
    #     self._l1_criterion = nn.L1Loss()
    #     # self.train_process = self.train_tiny
    #     self.train_process = self.train_sparse_tiny

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return code, recon

    # def train_tiny(self, x, criterion):
    #     self.zero_grad()
    #     code, recon = self.forward(x)
    #     if criterion is None:
    #         loss = self._l1_criterion(input=recon, target=x)
    #     else:
    #         loss = criterion(input=recon, target=x)
    #     loss.backward()
    #     self._optimizer.step()
    #     del code, recon
    #     return loss

    def train_sparse_tiny(self, x, criterion, lamb_sparsity, weight_vmin, weight_vmax):
        self.zero_grad()
        code, recon = self.forward(x)
        if criterion is None:
            loss_quality = self._l1_criterion(input=recon, target=x)
            # loss_quality = self._l2_criterion(input=recon, target=x)
        else:
            loss_quality = criterion(input=recon, target=x)
        loss_sparsity = code.norm(1) * lamb_sparsity
        loss = loss_quality + loss_sparsity
        loss.backward()
        self._optimizer.step()
        del code, recon
        self.weight_clipping(weight_vmin, weight_vmax)
        return loss_quality, loss_sparsity

    def step(self, x, criterion=None, lamb_sparsity=None):
        return self.train_process(x, criterion, lamb_sparsity, -0.5, 0.5)

    def build_optimizer(self):
        self._optimizer = optim.Adam(params=self.parameters(), lr=self.lr)

    def weight_clipping(self, vmin, vmax):
        for param in self.parameters():
            param.data.clamp_(vmin, vmax)
