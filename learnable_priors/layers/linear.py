import numpy as np
import torch
import torch.nn as nn

from nn.layers import Linear

from learnable_priors.distributions import *


class LinearPrior(nn.Module):
    
    def __init__(self, n_in, n_out, W_prior_dist, b_prior_dist,
                 W_prior_params={}, b_prior_params={}, bias=True):
        super(LinearPrior, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        
        self.W_prior = globals()[W_prior_dist](n_out, n_in, **W_prior_params)
        if self.bias:
            self.b_prior = globals()[b_prior_dist](n_out, **b_prior_params)
            
    def forward(self, X):
        W = torch.squeeze(self.W_prior.sample(1))
        if self.bias:
            b = torch.squeeze(self.b_prior.sample(1))
        else:
            b = 0
            
        return F.linear(X, W, b) 

    def resample_hyper_prior(self, module):
        if self.W_prior.has_hyper_prior:
            self.W_prior.resample_hyper_prior(module.weight)
        if self.b_prior.has_hyper_prior:
            self.b_prior.resample_hyper_prior(module.bias)

    def log_prob(self, module):
        if not isinstance(module, Linear):
            raise ValueError

        total_log_prob = self.W_prior.log_prob(module.weight)
        if self.bias:
            total_log_prob += self.b_prior.log_prob(module.bias)

        return total_log_prob
