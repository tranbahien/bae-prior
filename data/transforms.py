import torch


class Flatten():
    def __call__(self, image):
        return image.view(-1)


class StaticBinarize():
    def __call__(self, image):
        return image.round().long()


class DynamicBinarize():
    def __call__(self, image):
        return image.bernoulli().long()


class Quantize():
    def __init__(self, num_bits=8):
        self.num_bits = num_bits

    def __call__(self, image):
        image = image * 255 # [0, 1] -> [0, 255]
        if self.num_bits != 8:
            image = torch.floor(image / 2 ** (8 - self.num_bits)) # [0, 255] -> [0, 2**num_bits - 1]
        return image.long()
    