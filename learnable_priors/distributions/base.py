import torch
import torch.nn as nn

class PriorDistribution(nn.Module):

    has_hyper_prior = False

    def __init__(self):
        super(PriorDistribution, self).__init__()
    
    def sample(self, *shape):
        raise NotImplementedError
    
    def log_prob(self, inputs):
        raise NotImplementedError

    
