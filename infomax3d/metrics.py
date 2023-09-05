import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MAE(nn.Module):

    def __init__(self,):
        super().__init__()

    def forward(self, preds, targets):
        loss = F.l1_loss(preds, targets)
        return loss


class ContrastiveAccuracy(nn.Module):

    def __init__(self, threshold=0.5) -> None:
        super(ContrastiveAccuracy, self).__init__()
        self.threshold = threshold

    def forward(self,
                x1: Tensor,
                x2: Tensor,
                pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        preds: Tensor = (sim_matrix + 1) / 2 > self.threshold
        if pos_mask == None:  # if we are comparing global with global
            pos_mask = torch.eye(batch_size, device=x1.device)
        neg_mask = 1 - pos_mask

        num_positives = len(x1)
        num_negatives = len(x1) * (len(x2) - 1)
        true_positives = num_positives - (
            (preds.long() - pos_mask) * pos_mask).count_nonzero()
        true_negatives = num_negatives - ((
            (~preds).long() - neg_mask) * neg_mask).count_nonzero()
        return (true_positives / num_positives +
                true_negatives / num_negatives) / 2


class PositiveSimilarity(nn.Module):
    """
        https://en.wikipedia.org/wiki/Cosine_similarity
    """

    def __init__(self) -> None:
        super(PositiveSimilarity, self).__init__()

    def forward(self,
                x1: Tensor,
                x2: Tensor,
                pos_mask: Tensor = None) -> Tensor:
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:len(x1)]

        if pos_mask != None:  # if we are comparing local with global
            batch_size, _ = x1.size()
            sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

            x1_abs = x1.norm(dim=1)
            x2_abs = x2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
            pos_sim = (pos_mask * sim_matrix).sum(dim=1)
        else:  # if we are comparing global with global
            pos_sim = F.cosine_similarity(x1, x2)
        pos_sim = (pos_sim + 1) / 2
        return pos_sim.mean(dim=0)


class NegativeSimilarity(nn.Module):

    def __init__(self) -> None:
        super(NegativeSimilarity, self).__init__()

    def forward(self,
                x1: Tensor,
                x2: Tensor,
                pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        if pos_mask != None:  # if we are comparing local with global
            pos_sim = (pos_mask * sim_matrix).sum(dim=1)
        else:  # if we are comparing global with global
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        neg_sim = (sim_matrix.sum(dim=1) - pos_sim) / (batch_size - 1)
        neg_sim = (neg_sim + 1) / 2
        return neg_sim.mean(dim=0)


class Alignment(nn.Module):

    def __init__(self, alpha=2) -> None:
        super(Alignment, self).__init__()
        self.alpha = alpha

    def forward(self,
                x1: Tensor,
                x2: Tensor,
                pos_mask: Tensor = None) -> Tensor:
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:len(x1)]
        return (x1 - x2).norm(dim=1).pow(self.alpha).mean()


class TruePositiveRate(nn.Module):

    def __init__(self, threshold=0.5) -> None:
        super(TruePositiveRate, self).__init__()
        self.threshold = threshold

    def forward(self,
                x1: Tensor,
                x2: Tensor,
                pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        preds: Tensor = (sim_matrix + 1) / 2 > self.threshold
        if pos_mask == None:  # if we are comparing global with global
            pos_mask = torch.eye(batch_size, device=x1.device)

        num_positives = len(x1)
        true_positives = num_positives - (
            (preds.long() - pos_mask) * pos_mask).count_nonzero()

        return true_positives / num_positives


class TrueNegativeRate(nn.Module):

    def __init__(self, threshold=0.5) -> None:
        super(TrueNegativeRate, self).__init__()
        self.threshold = threshold

    def forward(self,
                x1: Tensor,
                x2: Tensor,
                pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:  # if we have noisy samples our x2 has them appended at the end so we just take the non noised ones to calculate the similaritiy
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        preds: Tensor = (sim_matrix + 1) / 2 > self.threshold
        if pos_mask == None:  # if we are comparing global with global
            pos_mask = torch.eye(batch_size, device=x1.device)
        neg_mask = 1 - pos_mask

        num_negatives = len(x1) * (len(x2) - 1)
        true_negatives = num_negatives - ((
            (~preds).long() - neg_mask) * neg_mask).count_nonzero()

        return true_negatives / num_negatives
