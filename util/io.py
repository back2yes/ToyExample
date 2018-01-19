import torch as tt
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image


class IOUtils(object):
    def __init__(self, is_cuda=False):
        # self.set_default_tensor_type(is_cuda)
        self.is_cuda = is_cuda

    # def set_default_tensor_type(self, is_cuda=False):
    #     self.is_cuda = is_cuda
    #     if is_cuda:
    #         tt.set_default_tensor_type('torch.cuda.FloatTensor')
    #     else:
    #         tt.set_default_tensor_type('torch.FloatTensor')

    def to_var(self, x, is_cuda=None):
        if is_cuda is None:
            is_cuda = self.is_cuda
        if is_cuda:
            return Variable(x).cuda()
        else:
            return Variable(x)

    def load_image(self, fp, size=None, as_var=True, is_cuda=None):
        pil_img = Image.open(fp)

        if size is not None:
            if isinstance(size, int):
                original_iw, original_ih = pil_img.size()
                target_iw = size
                target_ih = int(original_ih // original_iw * target_iw)
                size = (target_iw, target_ih)
            pil_img = pil_img.resize(size)

        tsr_img = transforms.ToTensor()(pil_img)
        if not as_var:
            return tsr_img.cpu().permute(1, 2, 0)
        return self.to_var(tsr_img[None], is_cuda)

    def show_single_image(self, x):
        plt.ion()
        tsr_show_data = x.data.cpu()[0].permute(1, 2, 0)
        tsr_show_data.clamp_(0.0, 1.0)
        plt.imshow(tsr_show_data, vmin=0.0, vmax=1.0)
        plt.waitforbuttonpress(0.1)
        # print(tsr_show_data)
        del tsr_show_data

    def show_trainval_image(self, *args):

        # plt.ion()
        plt.figure(0)
        # tsr_train = train.data.cpu()[0].permute(1, 2, 0).clamp_(0.0, 1.0)
        # tsr_valid = valid.data.cpu()[0].permute(1, 2, 0).clamp_(0.0, 1.0)
        try:
            self.fig_counter += 1
        except:
            self.fig_counter = 0

        num_subplots = len(args)
        for ii, arg in enumerate(args, 1):
            # tsr_data = arg.data.cpu()[0].permute(1, 2, 0).clamp_(0.0, 1.0)
            tsr_data = arg.data.cpu()[0].permute(1, 2, 0)
            vmax = tsr_data.max()
            vmin = tsr_data.min()
            tsr_data.sub_(vmin).div_(vmax - vmin)
            # print(vmax)
            plt.subplot(1, num_subplots, ii)
            plt.imshow(tsr_data)
            plt.axis('off')
            plt.savefig('viz/recon/fig_{:05d}.jpg'.format(self.fig_counter))
        # plt.waitforbuttonpress(0.01)
        # plt.subplot(1, 2, 2)
        # plt.imshow(tsr_valid)
        # plt.axis('off')
        # plt.waitforbuttonpress(0.1)
        # # print(tsr_show_data)
        # del tsr_train, tsr_valid

    def save_single_image(self, var_x, fp, nrow=16, normalize=True, scale_each=True):
        tsr_x = var_x[:1].data.permute(1, 0, 2, 3) * 20.0
        save_image(tsr_x, fp, nrow=nrow, normalize=normalize, scale_each=scale_each)
