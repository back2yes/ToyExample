import torchvision  as tv
import torch as tt
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class STL10(Dataset):
    def __init__(self, root_dir='/home/x/data/STL10/stl10_binary', subsets=('train',)):
        # subsets: {'train', 'test', 'unlabeled'}
        pjoin = os.path.join
        super(STL10, self).__init__()
        tmp_blobs = []
        for subset in subsets:
            tmp_blobs.append(self.load_stl10_file(pjoin(root_dir, subset + '_X.bin')))
        self.stl_blob = tt.from_numpy(np.concatenate(tmp_blobs, axis=0))

    def load_stl10_file(self, fp):
        blob = np.fromfile(fp, dtype=np.uint8).reshape(-1, 3, 96, 96).astype('float32').transpose(0, 1, 3, 2)
        blob /= np.cast[np.float32](255.0)
        # return tt.from_numpy(blob)
        return tt.from_numpy(blob)

    def __getitem__(self, index):
        return self.stl_blob[index]

    def __len__(self):
        return len(self.stl_blob)


if __name__ == '__main__':
    fp = '/home/x/data/STL10/stl10_binary'
    print(os.listdir(fp))
    a = np.fromfile(fp + '/train_X.bin', dtype=np.uint8).reshape(-1, 3, 96, 96) / np.cast[np.float32](255.0)
    a = tt.from_numpy(a)
    # a = np.fromfile(fp + '/unlabeled_X.bin', dtype=np.uint8)
    print(a.size())
    print(a.type())
    print(a.max())
    print(a.min())

    ds = STL10()
    dl = DataLoader(ds, batch_size=1, pin_memory=True)
    import matplotlib.pyplot as plt

    for ii, xs in enumerate(dl):
        plt.ion()
        plt.imshow(xs[0].permute(1, 2, 0))
        plt.waitforbuttonpress(0.2)
        print(ii, xs.size())
