import torch
import numpy as np

from torch.distributions.continuous_bernoulli import ContinuousBernoulli


def sample_autoencoder_prior(X, encoder_prior, decoder_prior, n_samples):
    samples = []
    
    for _ in range(n_samples):
        z = encoder_prior.sample(X, 1)
        probs = decoder_prior.sample(z, 1)
        dist = ContinuousBernoulli(probs=probs)        
        sample = dist.rsample([1]).squeeze()

        samples.append(sample)
        
    return torch.cat(samples)