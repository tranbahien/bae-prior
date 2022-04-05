import torch
import torch.nn as nn

from wasserstein_dist import *


def distributional_sliced_wasserstein_distance(
        first_samples, second_samples, num_projections,
        f, f_op, p=2, max_iter=10, lam=1, device='cuda'):
    
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
    
        encoded_projections = first_samples_detach.matmul(
            projections.transpose(0, 1))
        distribution_projections = (second_samples_detach.matmul(
            projections.transpose(0, 1)))
    
        wasserstein_distance = torch.abs((
            torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.pow(torch.sum(
            torch.pow(wasserstein_distance, p), dim=1), 1. / p)
        wasserstein_distance = torch.pow(torch.pow(
            wasserstein_distance, p).mean(), 1. / p)
    
        loss = reg - wasserstein_distance
    
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
    
    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    
    distribution_projections = (second_samples.matmul(
        projections.transpose(0, 1)))
    
    wasserstein_distance = torch.abs((torch.sort(
        encoded_projections.transpose(0, 1), dim=1)[0] -
        torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(
        wasserstein_distance, p), dim=1), 1. / p)
    wasserstein_distance = torch.pow(torch.pow(
        wasserstein_distance, p).mean(), 1. / p)
    
    return wasserstein_distance


def distributional_generalized_sliced_wasserstein_distance(
        first_samples, second_samples, num_projections, f, f_op,
        g_function, r, p=2, max_iter=10, lam=1, device='cuda'):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    
    for _ in range(max_iter):
        projections = f(pro)
        reg = lam * cosine_distance_torch(projections, projections)
        wasserstein_distance = g_function(
            first_samples, second_samples, projections, r, p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
    
    projections = f(pro)
    wasserstein_distance = g_function(first_samples, second_samples,
                                      projections, r, p)
    
    return wasserstein_distance
