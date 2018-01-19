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
    def __init__(self):
        super(Coder, self).__init__()
        self.coder = None
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

    # def build_tiny_coder(self):
    #     self.build_general_coder(3, [4, 8], 3)


class Encoder(Coder):
    def __init__(self, encoder_id):
        super(Encoder, self).__init__()
        if encoder_id == 'tiny':
            self.build_tiny_encoder()
        elif encoder_id == 'tinier':
            self.build_tinier_encoder()
        elif encoder_id == 'large':
            self.build_large_encoder()
        elif encoder_id == 'extra':
            self.build_extra_encoder()
        else:
            raise NotImplementedError

    def build_encoder(self, in_channels, filter_nums, filter_size, extra=None):
        pool = nn.MaxPool2d(3, 2, 1)
        relu = nn.ReLU(inplace=True)
        sugar = nn.Sequential(pool, relu)

        self.build_general_coder(in_channels, filter_nums, filter_size, block_sugar=sugar, extra=extra)

    def build_tiny_encoder(self):
        # [16, 64, 64, 128, 128] are the output channels of each kernel
        self.build_encoder(3, [16, 64, 128, 256], 3)

    def build_tinier_encoder(self):
        # [16, 64, 64, 128, 128] are the output channels of each kernel
        self.build_encoder(3, [16, 64, 128], 3)

    def build_large_encoder(self):
        # [16, 64, 64, 128, 128] are the output channels of each kernel
        self.build_encoder(3, [16, 16, 32, 32, 64, 128, 256], 3)

    def build_extra_encoder(self):
        # self.build_encoder(3, [16, 64, 64, 128, 128], 3)
        # [3, 32, 64, 128]
        self.build_encoder(3, [32, 64, 128], 3, extra=6)


class Decoder(Coder):
    def __init__(self, decoder_id):
        super(Decoder, self).__init__()
        if decoder_id == 'tiny':
            self.build_tiny_decoder()
        elif decoder_id == 'tinier':
            self.build_tinier_decoder()
        elif decoder_id == 'large':
            self.build_large_decoder()
        elif decoder_id == 'extra':
            self.build_extra_decoder()
        else:
            raise NotImplementedError

    def build_decoder(self, in_channels, filter_nums, filter_size, extra=None):
        # pool = nn.MaxPool2d(3, 2, 1)
        upsampling = nn.Upsample(scale_factor=2)
        relu = nn.ReLU(inplace=True)
        sugar = nn.Sequential(upsampling, relu)

        self.build_general_coder(in_channels, filter_nums, filter_size, block_sugar=sugar, extra=extra)

    def build_tiny_decoder(self):
        # self.build_encoder(3, [16, 64, 64, 128, 128], 3)
        self.build_decoder(256, [3, 16, 64, 128][::-1], 3)

    def build_tinier_decoder(self):
        # self.build_encoder(3, [16, 64, 64, 128, 128], 3)
        self.build_decoder(128, [3, 16, 64][::-1], 3)

    def build_large_decoder(self):
        # self.build_encoder(3, [16, 64, 64, 128, 128], 3)
        self.build_decoder(256, [3, 16, 16, 32, 32, 64, 128][::-1], 3)

    def build_extra_decoder(self):
        # self.build_encoder(3, [16, 64, 64, 128, 128], 3)
        # [3, 32, 64, 128]
        self.build_decoder(128, [3, 32, 64][::-1], 3, extra=2)


class ToyAE(nn.Module):
    def __init__(self, ae_id, lr=1e-5):
        super(ToyAE, self).__init__()
        self.lr = lr
        self.build_ae(ae_id)

    def build_ae(self, ae_id):
        self.encoder = Encoder(ae_id)
        self.decoder = Decoder(ae_id)
        self.build_optimizer()
        self._l1_criterion = nn.L1Loss()
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
