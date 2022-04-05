import os
import torch

from torchvision.datasets import MNIST
from data import DATA_PATH


class UnsupervisedMNIST(MNIST):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False, label=False):
        super(UnsupervisedMNIST, self).__init__(root,
                                                train=train,
                                                transform=transform,
                                                download=download)
        self.label = label

    def __getitem__(self, index):
        if self.label:
            return super(UnsupervisedMNIST, self).__getitem__(index)
        else:
            return super(UnsupervisedMNIST, self).__getitem__(index)[0]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')
