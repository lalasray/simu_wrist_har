import torch
import torch.nn as nn
import torch.nn.functional as F

text_encoder_type = "fc"

if text_encoder_type == "cnn":

    class TextEncoder(nn.Module):
        def __init__(self, input_dim=(768), embedding_dim=512):
            super(TextEncoder, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim[1], out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=3)
            self.fc1 = nn.Linear(7168, embedding_dim*2)
            self.fc2 = nn.Linear(embedding_dim*2, embedding_dim)


        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            
            return x


elif text_encoder_type == "attention": #ToFix

    class SelfAttention(nn.Module):
        def __init__(self, embed_dim = 512, num_heads = 4):
            super(SelfAttention, self).__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
            
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)
            
            self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
            
        def forward(self, query, key, value, mask=None):
            batch_size = query.shape[0]
            
            Q = self.query(query)
            K = self.key(key)
            V = self.value(value)
            print(Q.shape,K.shape,V.shape)
            
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
            
            if mask is not None:
                energy = energy.masked_fill(mask == 0, float('-inf'))
            
            attention = F.softmax(energy, dim=-1)
            
            x = torch.matmul(attention, V)
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(batch_size, -1, self.embed_dim)
            
            return x

    class TextEncoder(nn.Module):
        def __init__(self, input_dim = (768), embedding_dim =512, num_heads = 4):
            super(TextEncoder, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim[1], out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=3)
            self.attention = SelfAttention(embed_dim=256, num_heads=num_heads)
            self.fc1 = nn.Linear(7168, embedding_dim*2)
            self.fc2 = nn.Linear(embedding_dim*2, embedding_dim)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.attention(x, x, x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            
            return x

else:

    class TextEncoder(nn.Module):
        def __init__(self, input_dim= 768, embedding_dim = 512):
            super(TextEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim*2),
                nn.ReLU(),
                nn.Linear(embedding_dim*2, embedding_dim)
            )

        def forward(self, x):
            batch_size = x.size(0)
            return self.encoder(x.view(batch_size, -1))