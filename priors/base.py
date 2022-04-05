import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class Prior(nn.Module):
    def __init__(self):
        super(Prior, self).__init__()
        self.hyperprior = False

    def forward(self, net):
        return -self.log_prob(net)

    def log_prob(self, net):
        raise NotImplementedError

    def sample(self, name, param):
        raise NotImplementedError
