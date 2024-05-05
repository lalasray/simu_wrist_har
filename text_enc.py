import torch
import torch.nn as nn
import torch.nn.functional as F
import config
text_encoder_type = config.text_encoder_type

if text_encoder_type == "cnn":

   class TextEncoder(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=512):
        super(TextEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, embedding_dim * 2) 
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(p=0.3)  

    def forward(self, x):
        x = self.conv1(x.unsqueeze(2))
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
        x = self.dropout(x) 
        x = self.fc2(x)
        
        return x

elif text_encoder_type == "res":

    class TextEncoder(nn.Module):
        def __init__(self, input_dim=768, embedding_dim=512):
            super(TextEncoder, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=3)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
            self.pool = nn.MaxPool1d(kernel_size=3, padding=1)
            self.fc1 = nn.Linear(input_dim, embedding_dim * 2) 
            self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
            self.dropout = nn.Dropout(p=0.3)  

        def forward(self, x):
            residual = x
            x = self.conv1(x.unsqueeze(2))
            x = torch.relu(x)
            x = self.pool(x)
            x = self.dropout(x)  
            x = x + residual.view(x.shape[0], x.shape[1], -1)
            residual = x
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.dropout(x)  
            x = x + residual.view(x.shape[0], x.shape[1], -1)
            residual = x
            x = self.conv3(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.dropout(x)  
            x = x + residual.view(x.shape[0], x.shape[1], -1)
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.dropout(x) 
            x = self.fc2(x)
            
            return x

elif text_encoder_type == "spatial":

    class TextEncoder(nn.Module):
        def __init__(self, input_dim=768, embedding_dim=1024, num_heads=8):
            super(TextEncoder, self).__init__()
            self.embedding_dim = embedding_dim
            self.upsample = nn.Linear(input_dim, embedding_dim)
            self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1)
            self.layer_norm = nn.LayerNorm(embedding_dim)
                    
        def forward(self, x):
            x = x.unsqueeze(0)
            x = self.upsample(x)
            x = x.permute(1, 0, 2)
            attn_output, _ = self.self_attention(x, x, x)
            attn_output = attn_output.permute(1, 0, 2) 
            attn_output = self.layer_norm(attn_output)
            context = torch.mean(attn_output, dim=1)
            return context


else:

    class TextEncoder(nn.Module):
        def __init__(self, input_dim=768, embedding_dim=512):
            super(TextEncoder, self).__init__()
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

#batch_size = 16
#input_tensor = torch.randn(batch_size, 768)
#model = TextEncoder(embedding_dim = 128)
#print("input shape:", input_tensor.shape)
#output = model(input_tensor)
#print("Output shape:", output.shape)