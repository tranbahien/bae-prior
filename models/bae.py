import os
import torch
import numpy as np

from torch import nn
from torch.distributions import kl_divergence
from tqdm import tqdm

from utils import get_all_files


class BAE(nn.Module):

    def __init__(self, decoder, encoder, encoder_prior, decoder_prior):
        super(BAE, self).__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.encoder_prior = encoder_prior
        self.decoder_prior = decoder_prior

        self.decoder_samples = None
        self.encoder_samples = None
        self.density_estimator = None
        self.loaded_samples = False

    def predict(self, x, randomness=False, get_mean=True):
        x_pred = None
        if randomness:
            x_pred = []
            for i in range(len(self.encoder_samples)):
                encoder_params, decoder_params = self.load_samples(i)
                self.encoder_params = encoder_params
                self.decoder_params = decoder_params
                
                x_pred.append(self.decoder(self.encoder(x)))

            x_pred = torch.stack(x_pred, dim=0)
            if not get_mean:
                return x_pred
            
            x_pred = torch.mean(x_pred, dim=0)
        else:
            z = self.encoder(x)
            x_pred = self.decoder(z)
        
        return x_pred

    def encode(self, x, randomness=False, return_mean=True):
        z = None
        if randomness:
            z = []
            for i in range(len(self.encoder_samples)):
                encoder_params, _ = self.load_samples(i)
                self.encoder_params = encoder_params
                
                z.append(self.encoder(x))

            z = torch.stack(z, dim=0)
            if return_mean:
                z = torch.mean(z, dim=0)
        else:
            z = self.encoder(x)
        
        return z

    def decode(self, z, randomness=False, return_std=False):
        x = None
        if randomness:
            x = []
            for i in range(len(self.decoder_samples)):
                _, decoder_params = self.load_samples(i)
                self.decoder_params = decoder_params

                x.append(self.decoder(z))
            
            x = torch.stack(x, dim=0)
            x_std = torch.std(x, dim=0)
            x = torch.mean(x, dim=0)
            
            if return_std:
                return x, x_std
        else:
            x = self.decoder.net(z)

        return x

    def set_density_estimator(self, density_estimator):
        self.density_estimator = density_estimator

    def sample(self, n_samples, randomness=True, return_std=False):
        z = self.density_estimator.sample(n_samples)
        z = torch.tensor(z, device=next(self.parameters()).device).float()

        return self.decode(z, randomness, return_std)

    def forward(self, x, n_data, likelihood_temp=1., prior_temp=1.):
        n_batch = x.shape[0]
        z = self.encode(x)
        log_lik = self.decoder.log_prob(x, context=z)
        log_lik = torch.sum(log_lik) / n_batch

        log_prior = 0.
        log_prior = torch.sum(self.encoder_prior.log_prob(self.encoder)) / n_data
        log_prior += torch.sum(self.decoder_prior.log_prob(self.decoder.net)) / n_data

        log_prob = (1. / likelihood_temp) * log_lik + (1. / prior_temp) * log_prior

        return log_prob, log_lik, log_prior

    def log_likelihood(self, x):
        x_pred = self.predict(x, randomness=True)

        return self.decoder.log_prob_wihout_context(x, x_pred)

    def load_samples(self, idx):
        encoder_params = None
        decoder_params = None

        if self.loaded_samples:
            encoder_params = self.encoder_samples[idx]
            decoder_params = self.decoder_samples[idx]
        else:
            encoder_params =  torch.load(self.encoder_samples[idx])
            decoder_params =  torch.load(self.decoder_samples[idx])

        return encoder_params, decoder_params

    def set_samples(self, sample_dir, cache=False):
        encoder_files = get_all_files(os.path.join(sample_dir, "encoder*"))
        decoder_files = get_all_files(os.path.join(sample_dir, "decoder*"))

        if cache:
            self.encoder_samples = []
            self.decoder_samples = []

            for i in tqdm(range(len(encoder_files))):
                self.encoder_samples.append(
                    torch.load(encoder_files[i]))
                self.decoder_samples.append(
                    torch.load(decoder_files[i]))
        else:
            self.encoder_samples = encoder_files
            self.decoder_samples = decoder_files
        
        self.loaded_samples = cache
        
    def save_sample(self, sample_dir, idx):
        torch.save(self.encoder_params,
            os.path.join(sample_dir, "encoder_{:03d}.pt".format(idx)))
        torch.save(self.decoder_params,
            os.path.join(sample_dir, "decoder_{:03d}.pt".format(idx)))
    
    @property
    def params(self):
        return self.state_dict()

    @params.setter
    def params(self, params):
        self.load_state_dict(params)

    @property
    def encoder_params(self):
        return self.encoder.state_dict()

    @encoder_params.setter
    def encoder_params(self, params):
        self.encoder.load_state_dict(params)

    @property
    def decoder_params(self):
        return self.decoder.net.state_dict()

    @decoder_params.setter
    def decoder_params(self, params):
        self.decoder.net.load_state_dict(params)