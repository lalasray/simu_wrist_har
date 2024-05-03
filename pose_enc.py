import torch.nn as nn
import torch

pose_type = "pose"
pose_encoder_type = "cnn"

if pose_type == "pose":

    if pose_encoder_type == "spatiotemporal":

        class PoseEncoder(nn.Module):
            def __init__(self, input_dim=(30, 156), embedding_dim=512, num_heads=8, num_layers=6):
                super(PoseEncoder, self).__init__()
                self.embedding_dim = embedding_dim
                self.pos_encoding = PositionalEncoding(embedding_dim, input_dim[1])
                self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1)
                self.layer_norm = nn.LayerNorm(embedding_dim)
                
            def forward(self, x):
                x = self.pos_encoding(x)
                x = x.permute(1, 0, 2)
                attn_output, _ = self.self_attention(x, x, x)
                attn_output = attn_output.permute(1, 0, 2) 
                attn_output = self.layer_norm(attn_output)
                context = torch.mean(attn_output, dim=1)
                
                return context

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=1000):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=0.1)
                
                position = torch.arange(0, max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
                pe = torch.zeros(max_len, d_model)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:x.size(0), :]
                return self.dropout(x)

    elif pose_encoder_type == "cnn":

        class PoseEncoder(nn.Module):
            def __init__(self, input_dim=(30,156), embedding_dim=512):
                super(PoseEncoder, self).__init__()
                self.conv1 = nn.Conv1d(in_channels=input_dim[0], out_channels=32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
                self.pool = nn.MaxPool1d(kernel_size=3)
                self.fc1 = nn.Linear(256 * (input_dim[1] // 27), embedding_dim) 
                self.fc2 = nn.Linear(embedding_dim, embedding_dim)
                self.dropout = nn.Dropout(p=0.3)  

            def forward(self, x):
                x = x.view(-1, x.size(1), 156)
                x = self.conv1(x)
                x = torch.relu(x)
                x = self.pool(x)
                x = self.dropout(x)  
                x = self.conv2(x)
                x = torch.relu(x)
                x = self.pool(x)
                x = self.dropout(x)  
                x = self.conv3(x)
                x = torch.relu(x)
                x = self.pool(x)
                x = self.dropout(x)  
                x = torch.flatten(x, start_dim=1)
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                
                return x

    else:

        class PoseEncoder(nn.Module):
            def __init__(self, input_dim=30*156, embedding_dim=512):
                super(PoseEncoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, embedding_dim*2),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(embedding_dim*2, embedding_dim),
                    nn.Dropout(p=0.3)
                )

            def forward(self, x):
                batch_size = x.size(0)
                return self.encoder(x.view(batch_size, -1))
                    
elif pose_type == "embedding":

    if pose_encoder_type == "attention":

        print("ToDo")

    elif pose_encoder_type == "cnn":

        print("ToDo")
  
    else:

        print("ToDo")