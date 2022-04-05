import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import os
import time
import torch.nn as nn
import absl.app
import pandas as pd
import numpy as np
import torchvision.utils as vutils

from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm 

from learnable_priors.models.conv_net import ConvNetPrior, InvConvNetPrior
from nn.nets import ConvNet, InvConvNet
from data.loaders.image import CelebA
from models.bae import BAE
from distributions import DiagonalNormal, ConditionalNormal, ConditionalBernoulli, ConditionalMeanNormal
from samplers import AdaptiveSGHMC, SGHMC
from priors import PriorNormal
from utils import ensure_dir, set_seed, DensityEstimator, inf_loop, sample_autoencoder_prior, save_images_to_npy
from utils.logger.logger import setup_logging
from wasserstein_dist import TransformNet, distributional_sliced_wasserstein_distance
from metrics import calculate_fid_given_paths

import warnings
warnings.filterwarnings("ignore")


FLAGS = absl.app.flags.FLAGS

f = absl.app.flags
f.DEFINE_integer("seed", 123, "The random seed for reproducibility")
f.DEFINE_string("out_dir", "./exp/conv_celeba/bae_1000", "The path to the directory containing the experimental results")
f.DEFINE_string("prior_dir", "./exp/conv_celeba/priors", "The path to the directory containing the priors")
f.DEFINE_string("prior_type", "optim", "The type of the prior")
f.DEFINE_integer("batch_size", 64, "The mini-batch size")
f.DEFINE_integer("latent_size", 50, "The size of latent space")
f.DEFINE_integer("n_samples", 16, "The number of prior samples for each forward pass")
f.DEFINE_integer("n_iters_prior", 2000, "The  number of epochs used to optimize the priors")
f.DEFINE_integer("n_epochs_pretrain", 200, "The number of epochs used to pretrain model")
f.DEFINE_integer("training_size", 1000, "The size of training data")
f.DEFINE_float("lr", 0.003, "The learning rate for the sampler")
f.DEFINE_float("mdecay", 0.05, "The momentum for the sampler")
f.DEFINE_integer("num_burn_in_steps", 2000, "The number of burn-in steps of the sampler")
f.DEFINE_integer("iter_start_collect", 6000, "Which iteration to start collecting samples")
f.DEFINE_integer("n_mcmc_samples", 32, "The number of collected MCMC samples")
f.DEFINE_integer("keep_every", 1000, "The thinning interval")
f.DEFINE_bool("optimize_prior", True, "Whether or not optimizing the prior")
f.DEFINE_bool("train", True, "Whether or not training a new model")
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

ensure_dir(PRIOR_DIR)
ensure_dir(LOG_DIR)
ensure_dir(RESULT_DIR)
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
# Create dataloader for the subset of training data
dataset = CelebA()
train_loader, _, _ = dataset.get_data_loaders(1000)
x_subset = next(iter(train_loader))

dataset_subset = TensorDataset(x_subset)
dataloader_subset = DataLoader(dataset_subset, batch_size=32,
                               shuffle=True, pin_memory=True, num_workers=1)

latent_size = FLAGS.latent_size

set_seed(FLAGS.seed)

# Initialize the network
encoder_net = ConvNet(3, latent_size,
                  n_channels=[64, 128, 256, 512],
                  kernel_sizes=[5, 5, 5, 5],
                  strides=[2, 2, 2, 2],
                  paddings=[0, 0, 0, 0],
                  activation='leaky_relu',
                  in_lambda=None)

decoder_net = InvConvNet(
        latent_size, 3, n_hiddens=[8*8*512], n_channels=[512, 256, 128],
        kernel_sizes=[5, 5, 4],
        strides=[2, 2, 2],
        paddings=[2, 1, 0],
        activation='leaky_relu',
        mid_lambda=lambda x: x.view(x.shape[0], 512, 8, 8),
        out_lambda=lambda x: torch.sigmoid(x.view(x.shape[0], 3, 64, 64)))

encoder_net = encoder_net.to(device)
decoder_net = decoder_net.to(device)

params = list(encoder_net.parameters()) + list(decoder_net.parameters())
optimizer = optim.Adam(params, lr=0.001)

if FLAGS.pretrain_model or FLAGS.optimize_prior:
    if os.path.exists(os.path.join(RESULT_DIR, "pretrained", "encoder.pt")):
        logger.info("The pretrained networks exist!")
    else:
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
        3, latent_size,
        n_channels=[64, 128, 256, 512],
        kernel_sizes=[5, 5, 5, 5],
        strides=[2, 2, 2, 2],
        paddings=[0, 0, 0, 0],
        activation='leaky_relu',
        in_lambda=None,
        W_prior_dist="GaussianDistribution",
        b_prior_dist="GaussianDistribution")

    decoder_prior = InvConvNetPrior(
        latent_size, 3, n_hiddens=[8*8*512], n_channels=[512, 256, 128],
        kernel_sizes=[5, 5, 4],
        strides=[2, 2, 2],
        paddings=[2, 1, 0],
        activation='leaky_relu',
        mid_lambda=lambda x: x.view(x.shape[0], 512, 8, 8),
        out_lambda=lambda x: torch.sigmoid(x.view(x.shape[0], 3, 64, 64)),
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
    encoder_net.load_state_dict(torch.load(os.path.join(RESULT_DIR, "pretrained", "encoder.pt")))
    decoder_net.load_state_dict(torch.load(os.path.join(RESULT_DIR, "pretrained", "decoder.pt"))) 
    
    encoder_prior.init_mean(encoder_net)
    decoder_prior.init_mean(decoder_net)

    logger.info("Optimizing the priors")
    params = list(encoder_prior.parameters()) + list(decoder_prior.parameters())
    optimizer = optim.Adam(params, lr=0.001)

    transform_net = TransformNet(3*64*64).cuda()
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

        if i % 200 == 0:
            torch.save(encoder_prior.state_dict(), os.path.join(PRIOR_DIR, "encoder_prior_iter_{}.pt".format(i)))
            torch.save(decoder_prior.state_dict(), os.path.join(PRIOR_DIR, "decoder_prior_iter_{}.pt".format(i)))

        i += 1

    torch.save(encoder_prior.state_dict(), os.path.join(PRIOR_DIR, "encoder_prior.pt"))
    torch.save(decoder_prior.state_dict(), os.path.join(PRIOR_DIR, "decoder_prior.pt"))

########################
# CREATE TRAINING DATA #
########################
BATCH_SIZE = FLAGS.batch_size
logger.info("Creating a training and test data")
# dataset_subset = TensorDataset(x_subset)
# train_loader = DataLoader(dataset_subset_2, batch_size=BATCH_SIZE, shuffle=True,
#                           num_workers=1, pin_memory=True)

set_seed(FLAGS.seed)
train_loader, _, _ = dataset.get_data_loaders(FLAGS.training_size)
iterator = iter(train_loader)
x_train = next(iterator) # To avoid getting data already used to tune prior
x_train = next(iterator)
x_train = next(iterator)

dataset_train = TensorDataset(x_train)
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                          shuffle=True, pin_memory=True, num_workers=1)


test_loader = DataLoader(dataset.test, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=1, pin_memory=True)

if FLAGS.combined_data:
    logger.info("Combining data for inference")
    combined_dataset = torch.utils.data.ConcatDataset([dataset_subset, train_loader.dataset])
    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

###################################
# INITIALIZE BAYESIAN AUTOENCODER #
###################################
set_seed(FLAGS.seed + FLAGS.seed_chain)

# Initialize encoder and decoder networks
encoder_net = ConvNet(3, latent_size,
                  n_channels=[64, 128, 256, 512],
                  kernel_sizes=[5, 5, 5, 5],
                  strides=[2, 2, 2, 2],
                  paddings=[0, 0, 0, 0],
                  activation='leaky_relu',
                  in_lambda=None)

decoder_net = InvConvNet(
        latent_size, 3, n_hiddens=[8*8*512], n_channels=[512, 256, 128],
        kernel_sizes=[5, 5, 4],
        strides=[2, 2, 2],
        paddings=[2, 1, 0],
        activation='leaky_relu',
        mid_lambda=lambda x: x.view(x.shape[0], 512, 8, 8),
        out_lambda=lambda x: torch.sigmoid(x.view(x.shape[0], 3, 64, 64)))

encoder_net = encoder_net.to(device)
decoder_net = decoder_net.to(device)

# Load the pretrain models
if FLAGS.pretrain_model:
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
    ll = []
    for i, x in tqdm(enumerate(test_loader)):
        x = x.float().to(device)
        ll.append(model.log_likelihood(x))
    ll = torch.cat(ll).detach().cpu().numpy()
    ll = ll.mean()
    logger.info('NLL: {:.3f}'.format(-ll))
    
    # Generate latent codes
    z = []
    for i, x in tqdm(enumerate(train_loader)):
        x = x[0].to(device)
        z.append(model.encode(x, randomness=True))
    z = torch.cat(z).detach().cpu().numpy()
    np.save(os.path.join(RESULT_DIR, "train_z.npy"), z)
    
    # Estimate the density of the latent space
    density_estimator = DensityEstimator(method="gmm_dirichlet", n_components=20)
    z = np.load(os.path.join(RESULT_DIR, "train_z.npy"))
    density_estimator.fit(z)
    density_estimator.save_model(os.path.join(RESULT_DIR, "gmm_dirichlet.pkl"))
    density_estimator.load_model(os.path.join(RESULT_DIR, "gmm_dirichlet.pkl"))
    model.set_density_estimator(density_estimator)
    
    # Generate samples
    samples = []
    for i in tqdm(range(10)):
        samples.append(model.sample(1000).detach().cpu().float())
    samples = torch.cat(samples, dim=0).numpy()

    sampled_imgs_file = os.path.join(RESULT_DIR, "sampled_imgs.npy")
    save_images_to_npy(sampled_imgs_file, samples)
    
    # Compute FID score
    # TODO: configure path to the image file
    test_imgs_file = "./datasets/celeba/test_imgs.npy"
    sampled_imgs_file = os.path.join(RESULT_DIR, "sampled_imgs.npy")

    fid = calculate_fid_given_paths([test_imgs_file, sampled_imgs_file])
    logger.info('FID score: {:.3f}'.format(fid))