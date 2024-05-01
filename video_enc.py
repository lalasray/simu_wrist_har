import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(VideoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

