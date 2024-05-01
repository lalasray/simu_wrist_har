import torch
import torch.nn as nn

# Define mean instance discrimination (Infonce) loss
class InfonceLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfonceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k):
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
