import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from learnable_priors.distributions import PriorDistribution
from utils import inv_softplus


class GaussianDistribution(PriorDistribution):

    def __init__(self, *shape, mean_init=0., log_var_init=-6,
                 mean_trainable=True, std_trainable=False):
        super(GaussianDistribution, self).__init__()
        self.shape = shape      
        self.mean = nn.Parameter(
            torch.ones(*shape) * mean_init,
            requires_grad=True)

        self.log_var = nn.Parameter(
            torch.ones(*shape) * log_var_init,
            requires_grad=True)

    @property
    def std(self):
        return torch.sqrt(self.var)
        

    @property
    def var(self):
        return torch.exp(self.log_var)

    def sample(self, *shape):
        eps = torch.randn(*shape, *self.shape, device=self.mean.device,
                          requires_grad=False)

        samples = self.mean + self.std * eps
        return samples

    def log_prob(self, inputs):
        return torch.sum(-0.5 * ((self.mean - inputs) ** 2 / self.var))


class FactorizedGaussianDistribution(nn.Module):
    def __init__(self, *shape):
        super(FactorizedGaussianDistribution, self).__init__()
        self.shape = shape
        self.mean = nn.Parameter(torch.zeros(*shape), requires_grad=False)
        self.logvars = nn.Parameter(torch.zeros(*shape), requires_grad=True)

    def sample(self, *shape):
        epsilon_for_samples = torch.randn(*shape, *self.shape,
                                          device=self.mean.device,
                                          requires_grad=False)

        samples = self.mean + (self.logvars/2).exp() * epsilon_for_samples
        return samples

    def log_prob(self, inputs):
        return torch.sum(-0.5 * (np.log(2 * np.pi) + self.logvars +\
            (self.mean - inputs) ** 2 / self.logvars.exp()))