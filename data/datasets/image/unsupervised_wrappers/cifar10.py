import torch

from torchvision.datasets import CIFAR10
from data import DATA_PATH


class UnsupervisedCIFAR10(CIFAR10):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False, label=False):
        super(UnsupervisedCIFAR10, self).__init__(root,
                                                  train=train,
                                                  transform=transform,
                                                  download=download)
        self.label = label
        
    def __getitem__(self, index):
        if self.label:
            return super(UnsupervisedCIFAR10, self).__getitem__(index)
        else:
            return super(UnsupervisedCIFAR10, self).__getitem__(index)[0]
