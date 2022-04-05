import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import os
import time
import pickle
import pandas as pd
import torch.nn as nn
import absl.app
import numpy as np
import torchvision.utils as vutils

from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm 

from learnable_priors.models.conv_net import ConvNetPrior, InvConvNetPrior
from nn.nets import ConvNet, InvConvNet
from data.loaders.image import MNIST
from models.bae import BAE
from distributions import DiagonalNormal, ConditionalNormal, ConditionalBernoulli
from samplers import AdaptiveSGHMC, SGHMC
from priors import PriorNormal
from utils import ensure_dir, set_seed, DensityEstimator, inf_loop, sample_autoencoder_prior
from utils import load_freyfaces, load_yalefaces
from utils.logger.logger import setup_logging
from wasserstein_dist import TransformNet, distributional_sliced_wasserstein_distance

import warnings
warnings.filterwarnings("ignore")


FLAGS = absl.app.flags.FLAGS

f = absl.app.flags
f.DEFINE_integer("seed", 123, "The random seed for reproducibility")
f.DEFINE_string("out_dir", "./exp/conv_face/bae_1000", "The path to the directory containing the experimental results")
f.DEFINE_string("prior_dir", "./exp/conv_face/priors", "The path to the directory containing optimized priors")
f.DEFINE_string("prior_type", "optim", "The type of the prior")
f.DEFINE_integer("batch_size", 64, "The mini-batch size")
f.DEFINE_integer("latent_size", 50, "The size of latent space")
f.DEFINE_integer("n_samples", 32, "The number of prior samples for each forward pass")
f.DEFINE_integer("n_iters_prior", 1000, "The  number of epochs used to optimize the priors")
f.DEFINE_integer("n_epochs_pretrain", 200, "The number of epochs used to pretrain model")
f.DEFINE_integer("training_size", 1000, "The size of training data")
f.DEFINE_float("lr", 0.003, "The learning rate for the sampler")
f.DEFINE_float("mdecay", 0.05, "The momentum for the sampler")
f.DEFINE_integer("num_burn_in_steps", 2000, "The number of burn-in steps of the sampler")
f.DEFINE_integer("iter_start_collect", 6000, "Which iteration to start collecting samples")
f.DEFINE_integer("n_mcmc_samples", 32, "The number of collected MCMC samples")
f.DEFINE_integer("keep_every", 1000, "The thinning interval")
f.DEFINE_bool("train", True, "Whether or not training a new model")
f.DEFINE_bool("optimize_prior", True, "Whether or not optimizing the prior")
f.DEFINE_bool("pretrain_model", False, "Whether or not pretraining the model")
f.DEFINE_bool("combined_data", False, "Whether or not combining data for inference")
f.DEFINE_integer("seed_chain", 0, "The random seed for the sampling chain")


FLAGS(sys.argv)
set_seed(FLAGS.seed)

#########
# SETUP #
#########
LOG_DIR = os.path.join(FLAGS.out_dir, "logs")
RESULT_DIR = os.path.join(FLAGS.out_dir, "results")
PRIOR_DIR = FLAGS.prior_dir
ensure_dir(LOG_DIR)
ensure_dir(RESULT_DIR)
ensure_dir(PRIOR_DIR)
logger = setup_logging(LOG_DIR)

logger.info("====="*20)
for k, v in FLAGS.flag_values_dict().items():
    logger.info(">> {}: {}".format(k, v))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info("Using device: {}".format(device))

####################################
# CREATE A SUBSET OF TRAINING DATA #
####################################
logger.info("Creating a subset of training data")

set_seed(FLAGS.seed)
dataloader_subset, _ = load_freyfaces(TRAIN=1955, TEST=1, batch_size=64)

latent_size = FLAGS.latent_size

set_seed(FLAGS.seed)

# Initialize the network
encoder_net = ConvNet(1, latent_size,
                  n_channels=[64, 128, 128, 256],
                  kernel_sizes=[4, 4, 4, 4],
                  strides=[2, 2, 2, 2],
                  paddings=[1, 1, 1, 1],
                  activation='leaky_relu',
                  in_lambda=None)

decoder_net = InvConvNet(
        latent_size, 1, n_hiddens=[7*7*256], n_channels=[256, 128, 128],
        kernel_sizes=[4, 4, 4],
        strides=[1, 1, 2],
        paddings=[0, 0, 0],
        activation='leaky_relu',
        mid_lambda=lambda x: x.view(x.shape[0], 256, 7, 7),
        out_lambda=lambda x: torch.sigmoid(x.view(x.shape[0], 1, 28, 28)))

encoder_net = encoder_net.to(device)
decoder_net = decoder_net.to(device)

if FLAGS.pretrain_model or FLAGS.optimize_prior:
    if os.path.exists(os.path.join(RESULT_DIR, "pretrained", "encoder.pt")):
        logger.info("The pretrained networks exist!")
    else:
        params = list(encoder_net.parameters()) + list(decoder_net.parameters())
        optimizer = optim.Adam(params, lr=0.001)

        N_EPOCHS_AE = FLAGS.n_epochs_pretrain
        for epoch in range(N_EPOCHS_AE):
            l = 0.0
            for i, x in enumerate(dataloader_subset):
                x = x[0].float().to(device)
                x_pred = decoder_net(encoder_net(x))
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(x, x_pred)
                loss.backward()
                optimizer.step()

                l += loss.detach().cpu().item()
            logger.info('Epoch: {}/{}, Loss: {:.5f}'.format(epoch+1, N_EPOCHS_AE, l/(i+1)))

        # Save the pretrained networks
        ensure_dir(os.path.join(RESULT_DIR, "pretrained"))
        torch.save(encoder_net.state_dict(), os.path.join(RESULT_DIR, "pretrained", "encoder.pt"))
        torch.save(decoder_net.state_dict(), os.path.join(RESULT_DIR, "pretrained", "decoder.pt"))

#####################
# INITIALIZE PRIORS #
#####################
if FLAGS.prior_type == "optim":
    set_seed(FLAGS.seed)
    encoder_prior = ConvNetPrior(
        1, latent_size,
        n_channels=[64, 128, 128, 256],
        kernel_sizes=[4, 4, 4, 4],
        strides=[2, 2, 2, 2],
        paddings=[1, 1, 1, 1],
        activation='leaky_relu',
        in_lambda=None,
        W_prior_dist="GaussianDistribution",
        b_prior_dist="GaussianDistribution")

    decoder_prior = InvConvNetPrior(
        latent_size, 1, n_hiddens=[7*7*256], n_channels=[256, 128, 128],
        kernel_sizes=[4, 4, 4],
        strides=[1, 1, 2],
        paddings=[0, 0, 0],
        activation='leaky_relu',
        mid_lambda=lambda x: x.view(x.shape[0], 256, 7, 7),
        out_lambda=lambda x: torch.sigmoid(x.view(x.shape[0], 1, 28, 28)),
        W_prior_dist="GaussianDistribution",
        b_prior_dist="GaussianDistribution")

    encoder_prior = encoder_prior.to(device)
    decoder_prior = decoder_prior.to(device)

###################
# OPTIMIZE PRIORS #
###################
prior_exist = False
if os.path.exists(os.path.join(PRIOR_DIR, "encoder_prior.pt")):
    logger.info("The optimized priors exist!")
    prior_exist = True

if FLAGS.optimize_prior and not(prior_exist) and (FLAGS.prior_type == "optim"):
    encoder_prior.init_mean(encoder_net)
    decoder_prior.init_mean(decoder_net)

    logger.info("Optimizing the priors")
    params = list(encoder_prior.parameters()) + list(decoder_prior.parameters())
    optimizer = optim.Adam(params, lr=0.001)

    transform_net = TransformNet(784).cuda()
    op_trannet = optim.Adam(transform_net.parameters(), lr=0.0005)

    N_SAMPLES = FLAGS.n_samples
    i = 0
    for data in inf_loop(dataloader_subset):
        if i > FLAGS.n_iters_prior:
            break
        x = data[0].float().to(device)

        optimizer.zero_grad()
        x_pred = sample_autoencoder_prior(x, encoder_prior, decoder_prior, N_SAMPLES)

        dist = distributional_sliced_wasserstein_distance(
                x_pred.view(-1, np.prod(x_pred.shape[1:])),
                x.repeat([N_SAMPLES] + [1]*(len(x.shape)-1)).view(-1, np.prod(x.shape[1:])),
                num_projections=1000, f=transform_net,
                f_op=op_trannet,  p=2, max_iter=30, lam=100, device='cuda')

        if i > 50:
            dist.backward()
            optimizer.step()
        
        logger.info("Iter #{} : Wasserstein distance: {:.5f}".format(i, dist.item()))
        i += 1

    torch.save(encoder_prior.state_dict(), os.path.join(PRIOR_DIR, "encoder_prior.pt"))
    torch.save(decoder_prior.state_dict(), os.path.join(PRIOR_DIR, "decoder_prior.pt"))

########################
# CREATE TRAINING DATA #
########################
logger.info("Creating a training data")
set_seed(FLAGS.seed)

train_loader, test_loader = load_yalefaces(TRAIN=1000, TEST=1320, batch_size=1000, test_batch_size=64)
data = next(iter(train_loader))
x_subset = data[0][:FLAGS.training_size]

BATCH_SIZE = FLAGS.batch_size
dataset_subset = TensorDataset(x_subset)
train_loader = DataLoader(dataset_subset, batch_size=BATCH_SIZE, shuffle=True)

###################################
# INITIALIZE BAYESIAN AUTOENCODER #
###################################
set_seed(FLAGS.seed + FLAGS.seed_chain)

# Initialize encoder and decoder networks
encoder_net = ConvNet(1, latent_size,
                  n_channels=[64, 128, 128, 256],
                  kernel_sizes=[4, 4, 4, 4],
                  strides=[2, 2, 2, 2],
                  paddings=[1, 1, 1, 1],
                  activation='leaky_relu',
                  in_lambda=None)

decoder_net = InvConvNet(
        latent_size, 1, n_hiddens=[7*7*256], n_channels=[256, 128, 128],
        kernel_sizes=[4, 4, 4],
        strides=[1, 1, 2],
        paddings=[0, 0, 0],
        activation='leaky_relu',
        mid_lambda=lambda x: x.view(x.shape[0], 256, 7, 7),
        out_lambda=lambda x: torch.sigmoid(x.view(x.shape[0], 1, 28, 28)))

encoder_net = encoder_net.to(device)
decoder_net = decoder_net.to(device)

# Load the pretrain models
if FLAGS.pretrain_model:
    logger.info("Loading pretrained model")
    encoder_net.load_state_dict(torch.load(os.path.join(RESULT_DIR, "pretrained", "encoder.pt")))
    decoder_net.load_state_dict(torch.load(os.path.join(RESULT_DIR, "pretrained", "decoder.pt"))) 

# Load the priors
if FLAGS.optimize_prior and (FLAGS.prior_type == "optim"):
    logger.info("Loading optimized prior")
    encoder_prior.load_state_dict(torch.load(os.path.join(PRIOR_DIR, "encoder_prior.pt")))
    decoder_prior.load_state_dict(torch.load(os.path.join(PRIOR_DIR, "decoder_prior.pt")))

    # Freeze the priors
    encoder_prior = encoder_prior.to(device).requires_grad_(False)
    decoder_prior = decoder_prior.to(device).requires_grad_(False)
    
elif FLAGS.prior_type == "std":
    logger.info("Using standard Gaussian prior")
    encoder_prior = PriorNormal(mu=0.0, std=1)
    decoder_prior = PriorNormal(mu=0.0, std=1)

# Initialize the Bayesian Autoencoder
encoder = encoder_net
decoder = ConditionalBernoulli(decoder_net)

model = BAE(encoder=encoder, decoder=decoder,
            encoder_prior=encoder_prior, decoder_prior=decoder_prior)
model = model.to(device)

#################
# SETUP SAMPLER #
#################

# Configure the sampler
n_data = len(train_loader.dataset)
sampler_config = {
    "lr": FLAGS.lr,
    "mdecay": FLAGS.mdecay,
    "num_burn_in_steps": FLAGS.num_burn_in_steps,
    "scale_grad": n_data
}

keep_every = FLAGS.keep_every
iter_start_collect = FLAGS.iter_start_collect
n_iters = FLAGS.n_mcmc_samples * keep_every + iter_start_collect + 1

SAMPLES_DIR = os.path.join(RESULT_DIR, "param_samples")
ensure_dir(SAMPLES_DIR)

# Initialize the sampler
set_seed(FLAGS.seed + FLAGS.seed_chain)
params = list(encoder.parameters()) + list(decoder.net.parameters())
sampler = AdaptiveSGHMC(params, **sampler_config)

####################
# PERFORM SAMPLING #
####################
if FLAGS.train:
    logger.info("Perform sampling")
    set_seed(FLAGS.seed + FLAGS.seed_chain)

    model.train()
    sample_idx = 0

    if FLAGS.combined_data:
        logger.info("Combining data for inference")
        combined_dataset = torch.utils.data.ConcatDataset([dataloader_subset.dataset, train_loader.dataset])
        train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

    l = 0.0
    i = 0
    for x in inf_loop(train_loader):
        if i > n_iters:
            break

        x = x[0].float().to(device)
        sampler.zero_grad()
        log_prob, log_lik, log_prior = model.forward(x, n_data)
        loss = -log_prob
        loss.backward()
        sampler.step()

        log_lik = log_lik.detach().cpu().item()
        log_prior = log_prior.detach().cpu().item()

        l += loss.detach().cpu().item()
        if i % 1000 == 0:
            logger.info('Iter: {}/{}, Loss: {:.3f}, log_lik: {:.3f}, log_prior: {:.3f}'.format(
                i+1, n_iters, l/(i+1), log_lik, log_prior))

        if i > iter_start_collect:
            if i % keep_every == 0:
                logger.info("Saving #{} sample".format(sample_idx))
                model.save_sample(SAMPLES_DIR, sample_idx)
                sample_idx += 1
        i += 1

set_seed(FLAGS.seed)

model.set_samples(SAMPLES_DIR, cache=True)
model.eval()

with torch.no_grad():
    results = {}

    ll = []
    for i, x in tqdm(enumerate(test_loader)):
        x = x[0].float().to(device)
        ll.append(model.log_likelihood(x))
    ll = torch.cat(ll).detach().cpu().numpy()
    ll = ll.mean()
    logger.info('NLL: {:.3f}'.format(-ll))