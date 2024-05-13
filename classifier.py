import torch
import torch.nn as nn
import torch.nn.functional as F
from config import classifer_type

decoder = classifer_type

if decoder == "multihead":

    class ClassifierDecoder(nn.Module):
        def __init__(self, input_size, num_classes, cnn_channels=128, cnn_kernel_size=1, num_heads=8):
            super(ClassifierDecoder, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=cnn_kernel_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )
            
            self.multihead_attention = nn.MultiheadAttention(embed_dim=cnn_channels, num_heads=num_heads)
            
            self.fc = nn.Linear(cnn_channels, num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)
            features = self.cnn(x)
            features = features.permute(2, 0, 1)  
            attended_features, _ = self.multihead_attention(features, features, features)
            attended_features = attended_features.permute(1, 2, 0)  
            output = self.fc(attended_features.mean(dim=2))
            return output
else: 
    class ClassifierDecoder(nn.Module):
        def __init__(self, input_size, num_classes):
            super(ClassifierDecoder, self).__init__()
            self.fc1 = nn.Linear(input_size, 512)
            self.fc2 = nn.Linear(512, num_classes)
        
        def forward(self, imp):
            imp_flat = imp.view(imp.size(0), -1) 
            #text_flat = text.view(text.size(0), -1)  
            #combined = torch.cat((imp_flat, text_flat), dim=1) 
            out = F.relu(self.fc1(imp_flat)) 
            out = self.fc2(out)
            return out


#batch_size = 32
#input_size = 2048
#random_input = torch.randn(batch_size,input_size)
#print("Shape of random_input:", random_input.shape)  # Verify the shape

#num_classes = 10
#model = ClassifierDecoder(input_size=input_size, num_classes=num_classes)

# Now, let's check the forward pass
#output = model(random_input)
#print("Output shape:", output.shape)
