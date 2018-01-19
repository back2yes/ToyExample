from module.ToyAE import Identity, Coder, Encoder, Decoder, ToyAE
from util.io import IOUtils
import matplotlib.pyplot as plt
import numpy as np

toolkit = IOUtils(is_cuda=True)
fp = '/home/x/data/HIMYM/01.jpg'
val_list = ['/home/x/data/HIMYM/{:02d}.jpg'.format(ii) for ii in range(2, 11)]
var_train_img = toolkit.load_image(fp, size=(256, 256))
# encoder = Encoder()
# decoder = Decoder()
#
# code = encoder(var_img)
# recon = decoder(code)
#
# print(var_img.size(), code.size(), recon.size())

# ae = ToyAE(ae_id='large')
ae = ToyAE(ae_id='extra')
ae.cuda()
code, recon = ae(var_train_img)
print(code.size(), recon.size())
import torch as tt

for ii in range(20000):
    if ii < 2000:
        loss_quality, loss_sparsity = ae.step(var_train_img, lamb_sparsity=0.0)
    else:
        loss_quality, loss_sparsity = ae.step(var_train_img, lamb_sparsity=1e-5)
    if ii % 100 == 0:
        # print(loss)
        var_valid_img = toolkit.load_image(val_list[(ii // 100) % len(val_list)], size=(256, 256))
        var_train_code, var_train_recon = ae.forward(var_train_img)
        var_valid_code, var_valid_recon = ae.forward(var_valid_img)
        var_random_code = tt.normal(tt.zeros_like(var_valid_code), tt.ones_like(var_valid_code))
        var_random_recon = ae.decoder.forward(var_random_code)
        # toolkit.show_image(var_img)
        # print(var_recon.max(), var_recon.min())
        print('ii: {:06d}, L_qual={:.5f}, L_sp={:.5f}'.format(ii, loss_quality.data[0], loss_sparsity.data[0]))
        toolkit.save_single_image(var_train_code, 'viz/code/{:04d}.jpg'.format(ii))
        # toolkit.show_trainval_image(var_train_recon, var_valid_recon)

        # interesting result showing over-fitting derived from both the encoder and the decoder
        toolkit.show_trainval_image(var_train_recon, var_valid_recon, var_valid_img, var_random_recon)
