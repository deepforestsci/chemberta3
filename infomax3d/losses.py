import itertools
import math

import dgl
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss
import numpy as np
import torch.nn.functional as F


class NTXent(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.1, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXent, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentMultiplePositives(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0,
                 conformer_variance_reg=0) -> None:
        super(NTXentMultiplePositives, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.conformer_variance_reg = conformer_variance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        '''
        :param z1: batchsize, metric dim
        :param z2: batchsize*num_conformers, metric dim
        '''
        batch_size, metric_dim = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        sim_matrix = torch.einsum('ik,juk->iju', z1, z2)  # [batch_size, batch_size, num_conformers]

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=2)
            sim_matrix = sim_matrix / torch.einsum('i,ju->iju', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)  # [batch_size, batch_size, num_conformers]

        sim_matrix = sim_matrix.sum(dim=2)  # [batch_size, batch_size]
        pos_sim = torch.diagonal(sim_matrix)  # [batch_size]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.conformer_variance_reg > 0:
            std = torch.sqrt(z2.var(dim=1) + 1e-04)
            std_conf_loss = torch.mean(torch.relu(1 - std))
            loss += self.conformer_variance_reg * std_conf_loss
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss
