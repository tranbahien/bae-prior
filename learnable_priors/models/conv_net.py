import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.layers import LambdaLayer
from nn.layers import act_module
from nn.nets import ConvNet, InvConvNet
from nn.layers import Linear, Conv2d, ConvTranspose2d
from learnable_priors.layers import LinearPrior, Conv2dPrior, ConvTranspose2dPrior
from torch.nn import BatchNorm2d, Flatten, ConstantPad2d
from collections import OrderedDict


class ConvNetPrior(nn.Module):
    def __init__(self, input_size, output_size, n_channels,
                 kernel_sizes, strides, paddings,
                 W_prior_dist, b_prior_dist, W_prior_params={}, b_prior_params={},
                 activation='relu', in_lambda=None, out_lambda=None):
        super(ConvNetPrior, self).__init__()
        
        prior_params = {
            'W_prior_dist': W_prior_dist,
            'b_prior_dist': b_prior_dist,
            'W_prior_params': W_prior_params,
            'b_prior_params': b_prior_params
        }
        
        conv_layers = []
        if in_lambda: conv_layers.append(LambdaLayer(in_lambda))
        
        idxs = list(range(len(n_channels)))
        for idx in range(len(n_channels)):
            if idx == 0:
                in_size, out_size = input_size, n_channels[0]
            else:
                in_size, out_size = n_channels[idx-1], n_channels[idx]
            conv_layers.append(Conv2dPrior(in_size, out_size,
                                      kernel_size=kernel_sizes[idx-1],
                                      padding=paddings[idx-1],
                                      stride=strides[idx-1],
                                      **prior_params))
            conv_layers.append(act_module(activation))
        
        fc_layers = []
        fc_layers.append(LinearPrior(n_channels[-1], output_size, **prior_params))
        if out_lambda: fc_layers.append(LambdaLayer(out_lambda))

        self.convs = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*fc_layers)

    def sample(self, x, n_samples=1):
        samples = []
        
        for _ in range(n_samples):
            sample = self.convs(x)
            sample = sample.squeeze()
            if len(sample.shape) == 1:
                sample = sample.unsqueeze(0)
            sample = self.fc(sample)
            
            samples.append(sample)
        
        return torch.cat(samples, dim=0)

    def _get_modules(self, net):
        if not isinstance(net, ConvNet):
            raise ValueError

        prior_nm = dict(filter(lambda kv: isinstance(kv[1], (LinearPrior, Conv2dPrior, ConvTranspose2dPrior)),
                               self.named_modules()))
        net_nm = dict(filter(lambda kv: isinstance(kv[1], (Linear, Conv2d, ConvTranspose2d)),
                             net.named_modules()))

        return prior_nm, net_nm 
    
    def sample_params(self):
        prior_nm = dict(filter(lambda kv: isinstance(kv[1], (LinearPrior, Conv2dPrior, ConvTranspose2dPrior)),
                               self.named_modules()))
        sample = {}

        for n in prior_nm.keys():
            sample[n+'.weight'] = prior_nm[n].W_prior.sample(1).squeeze(0).detach()
            sample[n+'.bias'] = prior_nm[n].b_prior.sample(1).squeeze(0).detach()

        return OrderedDict(sample)
        
    
    def init_mean(self, net):
        prior_nm, net_nm = self._get_modules(net)
        for n in prior_nm.keys():
            prior_nm[n].W_prior.mean = net_nm[n].weight
            prior_nm[n].b_prior.mean = net_nm[n].bias

    def log_prob(self, net):
        prior_nm, net_nm = self._get_modules(net)

        total_log_prob = 0.0
        for n in prior_nm.keys():
            total_log_prob += prior_nm[n].log_prob(net_nm[n])

        return total_log_prob


class InvConvNetPrior(nn.Module):
    def __init__(self, input_size, output_size, n_hiddens, n_channels,
                 kernel_sizes, strides, paddings,
                 W_prior_dist, b_prior_dist, W_prior_params={}, b_prior_params={},
                 activation='relu', in_lambda=None, out_lambda=None,
                 mid_lambda=None):
        super(InvConvNetPrior, self).__init__()
        
        prior_params = {
            'W_prior_dist': W_prior_dist,
            'b_prior_dist': b_prior_dist,
            'W_prior_params': W_prior_params,
            'b_prior_params': b_prior_params
        }

        fc_layers = []
        if in_lambda: fc_layers.append(LambdaLayer(in_lambda))
        for in_size, out_size in zip([input_size] + n_hiddens[:-1], n_hiddens):
            fc_layers.append(LinearPrior(in_size, out_size, **prior_params))

            fc_layers.append(act_module(activation))
        if mid_lambda: fc_layers.append(LambdaLayer(mid_lambda))

        conv_layers = []
        
        for idx in range(len(n_channels)-1):
            in_size, out_size = n_channels[idx], n_channels[idx+1]
            conv_layers.append(ConvTranspose2dPrior(in_size, out_size,
                                               kernel_size=kernel_sizes[idx],
                                               stride=strides[idx],
                                               padding=paddings[idx],
                                               **prior_params))
            conv_layers.append(act_module(activation))

        conv_layers.append(ConvTranspose2dPrior(n_channels[-1], output_size,
                                                kernel_size=kernel_sizes[-1],
                                                stride=strides[-1],
                                                padding=paddings[-1],
                                                **prior_params))
        if out_lambda: conv_layers.append(LambdaLayer(out_lambda))

        self.fc = nn.Sequential(*fc_layers)
        self.convs = nn.Sequential(*conv_layers)

    def sample(self, x, n_samples=1):
        samples = []
        
        for _ in range(n_samples):
            sample = self.fc(x)
            sample = self.convs(sample)
            
            samples.append(sample)

        return torch.cat(samples, dim=0)
    
    def sample_params(self):
        prior_nm = dict(filter(lambda kv: isinstance(kv[1], (LinearPrior, Conv2dPrior, ConvTranspose2dPrior)),
                               self.named_modules()))
        sample = {}

        for n in prior_nm.keys():
            sample[n+'.weight'] = prior_nm[n].W_prior.sample(1).squeeze(0).detach()
            sample[n+'.bias'] = prior_nm[n].b_prior.sample(1).squeeze(0).detach()

        return OrderedDict(sample)

    def _get_modules(self, net):
        if not isinstance(net, InvConvNet):
            raise ValueError

        prior_nm = dict(filter(lambda kv: isinstance(kv[1], (LinearPrior, Conv2dPrior, ConvTranspose2dPrior)),
                               self.named_modules()))
        net_nm = dict(filter(lambda kv: isinstance(kv[1], (Linear, Conv2d, ConvTranspose2d)),
                             net.named_modules()))

        return prior_nm, net_nm
    
    def init_mean(self, net):
        prior_nm, net_nm = self._get_modules(net)
        for n in prior_nm.keys():
            prior_nm[n].W_prior.mean = net_nm[n].weight
            prior_nm[n].b_prior.mean = net_nm[n].bias

    def log_prob(self, net):
        prior_nm, net_nm = self._get_modules(net)

        total_log_prob = 0.0
        for n in prior_nm.keys():
            total_log_prob += prior_nm[n].log_prob(net_nm[n])

        return total_log_prob
