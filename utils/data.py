import pickle
import numpy as np
import torch
import torchvision.utils as vutils

from torch.utils.data import DataLoader, TensorDataset


def load_freyfaces(TRAIN=1000, TEST=956, batch_size=64, test_batch_size=64, seed=0, **kwargs):
    np.random.seed(seed)

    # start processing
    with open('datasets/FreyFaces/freyfaces.pkl', 'rb') as f:
        data = pickle.load(f)

    data = data / 255.

    # shuffle data:
    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN]
    # test images
    x_test = data[(TRAIN):(TRAIN + TEST)]

    # pytorch data loader
    train = TensorDataset(torch.from_numpy(x_train).float().unsqueeze(1))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
    
    test = TensorDataset(torch.from_numpy(x_test).float().unsqueeze(1))
    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_yalefaces(TRAIN=2319, TEST=1, batch_size=64, test_batch_size=64, seed=0, **kwargs):
    np.random.seed(seed)

    # start processing
    with open('datasets/YaleFaces/yalefaces.pkl', 'rb') as f:
        data = pickle.load(f)

    data = data / 255.

    # shuffle data:
    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN]
    # test images
    x_test = data[(TRAIN):(TRAIN + TEST)]

    # pytorch data loader
    train = TensorDataset(torch.from_numpy(x_train).float().unsqueeze(1))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
    
    test = TensorDataset(torch.from_numpy(x_test).float().unsqueeze(1))
    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader