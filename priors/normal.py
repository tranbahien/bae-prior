import torch

from priors import Prior


class PriorNormal(Prior):

    def __init__(self, mu=0.0, std=1.0, device="cpu"):
        super(PriorNormal, self).__init__()
        self.mu = mu
        self.std = std

    def sample(self, name, param):
        mu, std = self._get_params_by_name(name)

        if (mu is None) and (std is None):
            return None

        return (mu + std * torch.randn_like(param)).to(param.device)

    def _get_params_by_name(self, name):
        if not (('.weight' in name) or ('.bias' in name)):
            return None, None
        else:
            return self.mu, self.std

    def log_prob(self, net):
        res = 0.
        for name, param in net.named_parameters():
            mu, std = self._get_params_by_name(name)
            if (mu is None) and (std is None):
                continue
            var = std ** 2
            res -= torch.sum(((param - mu) ** 2) / (2 * var))
        return res

