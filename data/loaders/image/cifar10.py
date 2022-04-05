from torchvision.transforms import Compose, ToTensor
from data.datasets.image import UnsupervisedCIFAR10
from data.transforms import Quantize
from data import TrainTestLoader, DATA_PATH


class CIFAR10(TrainTestLoader):
    '''
    The CIFAR10 dataset of (Krizhevsky & Hinton, 2009):
    https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[], label=False):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor()]
        trans_test = [ToTensor()]
    
        # Load data
        self.train = UnsupervisedCIFAR10(root, train=True, transform=Compose(trans_train), download=download, label=label)
        self.test = UnsupervisedCIFAR10(root, train=False, transform=Compose(trans_test), label=label)
