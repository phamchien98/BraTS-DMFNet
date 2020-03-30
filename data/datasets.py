import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import time
from .data_utils import pkload
from .rand import *
from .transforms import *
sys.path.append('..')
from preprocess import process_f32

class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', for_train=False, transforms=''):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)

        self.names = names
        self.paths = paths
        self.transforms = eval(transforms or 'Identity()')
        print("{} with {} samples.".format('Train' if for_train else 'Valid', len(names)))

    def __getitem__(self, index):
        path = self.paths[index]
        if False and os.path.exists(path + 'data_f32.pkl'):
            start_load = time.time()
            x, y = pkload(path + 'data_f32.pkl')
            print("It takes {:.2f} s to load pkl file".format(time.time()-start_load))
            print("Load:", x.shape, y.shape)
        else:
            # start_convert = time.time()
            x, y = process_f32(path, save=False)            
            # print("It takes {:.2f} s to proccess miss data".format(time.time()-start_convert))
            # print("Convert:", x.shape, y.shape)
        # print(x.shape, y.shape)#(240, 240, 155, 4) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print("After None", x.shape, y.shape)  # (1, 240, 240, 155, 4) (1, 240, 240, 155)
        x, y = self.transforms([x, y])
        # print("After transform, ", x.shape, y.shape)  After transform,  (1, 128, 128, 128, 4) (1, 128, 128, 128)
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        # print("After ascontiguous:", x.shape, y.shape)  After ascontiguous: (1, 4, 128, 128, 128) (1, 128, 128, 128)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
