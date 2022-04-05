from torchvision.transforms import Compose, ToTensor
from data.datasets.image import UnsupervisedFashionMNIST
from data.transforms import Quantize
from data import TrainTestLoader, DATA_PATH


class FashionMNIST(TrainTestLoader):
    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[], label=False):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor()]
        trans_test = [ToTensor()]

        # Load data
        self.train = UnsupervisedFashionMNIST(root, train=True, transform=Compose(trans_train), download=download, label=label)
        self.test = UnsupervisedFashionMNIST(root, train=False, transform=Compose(trans_test), label=label)
