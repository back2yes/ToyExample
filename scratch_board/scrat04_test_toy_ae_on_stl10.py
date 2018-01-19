import torch as tt
from module.ToyAE import Identity, Coder, Encoder, Decoder, ToyAE
from util.io import IOUtils
import matplotlib.pyplot as plt
import numpy as np
from data.stl10 import STL10
from torch.utils.data import DataLoader

# ae = ToyAE(ae_id='extra')
desc = 'tiny'
toolkit = IOUtils(is_cuda=True, desc=desc)
ae = ToyAE(ae_id=desc, lr=4e-5)
print(ae)
ae.cuda()
ds = STL10()
dl = DataLoader(ds, batch_size=32, shuffle=True, pin_memory=True, )
ds_test = STL10(subsets=('test',))
# dl_test = DataLoader(ds_test, batch_size=1, shuffle=True, pin_memory=True)
counter = 0
for epoch in range(100):
    for ii, xs in enumerate(dl):
        var_train_img = toolkit.to_var(xs)

        if epoch < 5:
            loss_quality, loss_sparsity = ae.step(var_train_img, lamb_sparsity=0.0)
        else:
            # loss_quality, loss_sparsity = ae.step(var_train_img, lamb_sparsity=1e-5)
            loss_quality, loss_sparsity = ae.step(var_train_img, lamb_sparsity=3e-5)

        if ii == 0:
            # print(loss)
            counter += 1
            var_valid_img = toolkit.to_var(ds_test[counter % len(ds_test)][None])
            var_train_code, var_train_recon = ae.forward(var_train_img)
            var_valid_code, var_valid_recon = ae.forward(var_valid_img)
            var_random_code = tt.normal(tt.zeros_like(var_valid_code), tt.ones_like(var_valid_code))
            var_random_recon = ae.decoder.forward(var_random_code)
            # toolkit.show_image(var_img)
            # print(var_recon.max(), var_recon.min())
            print(
                'epoch: {:06d}, L_qual={:.5f}, L_sp={:.5f}'.format(epoch, loss_quality.data[0], loss_sparsity.data[0]))
            toolkit.save_single_image(var_train_code, '{}/code/{:04d}.jpg'.format(toolkit.viz_dir, counter))
            toolkit.show_trainval_image('{}/recon/{:04d}.jpg'.format(toolkit.viz_dir, counter),
                                        var_train_img, var_train_recon,
                                        var_valid_recon, var_valid_img,
                                        var_random_recon)

# for ii in range(20000):
#     if ii < 2000:
#         loss_quality, loss_sparsity = ae.step(var_train_img, lamb_sparsity=0.0)
#     else:
#         loss_quality, loss_sparsity = ae.step(var_train_img, lamb_sparsity=1e-5)
#     if ii % 100 == 0:
#         # print(loss)
#         var_valid_img = toolkit.load_image(val_list[(ii // 100) % len(val_list)], size=(256, 256))
#         var_train_code, var_train_recon = ae.forward(var_train_img)
#         var_valid_code, var_valid_recon = ae.forward(var_valid_img)
#         var_random_code = tt.normal(tt.zeros_like(var_valid_code), tt.ones_like(var_valid_code))
#         var_random_recon = ae.decoder.forward(var_random_code)
#         # toolkit.show_image(var_img)
#         # print(var_recon.max(), var_recon.min())
#         print('ii: {:06d}, L_qual={:.5f}, L_sp={:.5f}'.format(ii, loss_quality.data[0], loss_sparsity.data[0]))
#         toolkit.save_single_image(var_train_code, 'viz/code/{:04d}.jpg'.format(ii))
#         # toolkit.show_trainval_image(var_train_recon, var_valid_recon)
#
#         # interesting result showing over-fitting derived from both the encoder and the decoder
#         toolkit.show_trainval_image(var_train_recon, var_valid_recon, var_valid_img, var_random_recon)
