import torch
import torch.nn as nn
import torch.nn.functional as F

text_encoder_type = "cnn"

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
#model = TextEncoder(embedding_dim = 1024)
#print("input shape:", input_tensor.shape)
#output = model(input_tensor)
#print("Output shape:", output.shape)