import torch.nn as nn

pose_type = "embeddings"
pose_encoder_type = "fc"
if pose_type == "embeddings":
    if pose_encoder_type == "transformer":
        print("ToDo")
    elif pose_encoder_type == "CNN":
        print("ToDo")
    elif pose_encoder_type == "lstm":
        print("ToDo")
    else:
        class PoseEncoder(nn.Module):
            def __init__(self, input_dim, embedding_dim):
                super(PoseEncoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim)
                )

            def forward(self, x):
                return self.encoder(x)
