import torch.nn as nn

video_encoder_type = "fc"

if video_encoder_type == "transformer":

    print("ToDo")

elif video_encoder_type == "cnn":

    print("ToDo")

else:

    class VideoEncoder(nn.Module):
        def __init__(self, input_dim = (30*1024*12*16), embedding_dim = 512):
            super(VideoEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim*2),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )

        def forward(self, x):
            batch_size = x.size(0)
            return self.encoder(x.view(batch_size, -1))
