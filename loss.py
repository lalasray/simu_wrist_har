import torch
import torch.nn as nn
import torch.nn.functional as F

# Define mean instance discrimination (Infonce) loss
class InfonceLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfonceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k):

        # Normalize representations
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Calculate cosine similarity
        sim_matrix = torch.matmul(q, k.t())
        sim_matrix = sim_matrix / torch.norm(q, dim=-1, keepdim=True)
        sim_matrix = sim_matrix / torch.norm(k, dim=-1, keepdim=True).t()

        # Calculate logits
        logits = sim_matrix / self.temperature

        # Positive pairs: diagonal elements
        diag = torch.diag(logits)
        # Negative pairs: off-diagonal elements
        negatives = logits - torch.diag(torch.diagonal(logits))

        # Calculate cross-entropy loss
        exp_logits = torch.exp(logits - diag.unsqueeze(1))
        numerator = torch.sum(exp_logits, dim=-1)
        denominator = exp_logits.sum(dim=-1).unsqueeze(1) + torch.sum(torch.exp(negatives), dim=-1)
        loss = -torch.log(numerator / denominator)

        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize the feature vectors
        z_i = F.normalize(z_i, p=2, dim=-1)
        z_j = F.normalize(z_j, p=2, dim=-1)

        # Remove one data point if the number of data points is odd
        if len(z_i) % 2 != 0:
            z_i = z_i[:-1]
        
        # Concatenate and compute similarity matrix
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(z, z.t()) / self.temperature

        # Exclude diagonal elements from similarity matrix
        mask = torch.eye(len(z), dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix[~mask].view(len(z), -1)

        # Compute logits
        logits = torch.cat([sim_matrix, sim_matrix.t()], dim=1)

        # Compute contrastive loss
        labels = torch.arange(len(z), device=sim_matrix.device)
        loss = F.cross_entropy(logits, labels)

        return loss

class InfonceLossForClustering(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfonceLossForClustering, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, labels):
        # Calculate cosine similarity
        sim_matrix = torch.matmul(q, k.t())
        sim_matrix = sim_matrix / torch.norm(q, dim=-1, keepdim=True)
        sim_matrix = sim_matrix / torch.norm(k, dim=-1, keepdim=True).t()

        # Calculate logits
        logits = sim_matrix / self.temperature

        # Positive pairs: diagonal elements
        diag = torch.diag(logits)
        # Negative pairs: off-diagonal elements
        negatives = logits - torch.diag(torch.diagonal(logits))

        # Calculate cross-entropy loss
        exp_logits = torch.exp(logits - diag.unsqueeze(1))
        numerator = torch.sum(exp_logits, dim=-1)
        denominator = exp_logits.sum(dim=-1).unsqueeze(1) + torch.sum(torch.exp(negatives), dim=-1)
        loss = -torch.log(numerator / denominator)

        # Mask out the losses for samples in different clusters
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        loss = loss * mask

        # Normalize the loss by the number of positive pairs
        num_positive_pairs = torch.sum(mask) - len(labels)
        loss = loss.sum() / num_positive_pairs

        return loss