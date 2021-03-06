from torchvision.transforms import Compose, ToTensor
from data.datasets.image import ImageNet32Dataset
from data.transforms import Quantize
from data import TrainTestLoader, DATA_PATH


class ImageNet32(TrainTestLoader):
    '''
    The ImageNet dataset of
    (Russakovsky et al., 2015): https://arxiv.org/abs/1409.0575
    downscaled to 32x32, as used in
    (van den Oord et al., 2016): https://arxiv.org/abs/1601.06759
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[]):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]

        # Load data
        self.train = ImageNet32Dataset(root, train=True, transform=Compose(trans_train), download=download)
        self.test = ImageNet32Dataset(root, train=False, transform=Compose(trans_test))
