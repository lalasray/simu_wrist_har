import torch
import torch.nn as nn
import torch.nn.functional as F
from config import classifer_type,embedding_dim,batch_size_class,classes

decoder = classifer_type

if decoder == "multihead":

    class ClassifierDecoder(nn.Module):
        def __init__(self, input_size = embedding_dim, num_classes = 11, cnn_channels=32, cnn_kernel_size=1, num_heads=8):
            super(ClassifierDecoder, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels= input_size, out_channels=cnn_channels, kernel_size=cnn_kernel_size),
                nn.LeakyReLU(),
            )
            self.multihead_attention = nn.MultiheadAttention(embed_dim=cnn_channels, num_heads=num_heads)
            self.fc = nn.Linear(cnn_channels, num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)
            x = x.permute(0, 2, 1)
            features = self.cnn(x)
            features = features.permute(2, 0, 1)  
            attended_features, _ = self.multihead_attention(features, features, features)
            attended_features = attended_features.permute(1, 2, 0)  
            output = self.fc(attended_features.mean(dim=2))
            return output

elif decoder == "i_multihead":

    class ClassifierDecoder(nn.Module):
        def __init__(self, input_size=embedding_dim, num_classes=11, linear_size=11*batch_size_class, num_heads=8):
            super(ClassifierDecoder, self).__init__()
            
            self.linear1 = nn.Linear(embedding_dim, linear_size)
            self.multihead_attention1 = nn.MultiheadAttention(embed_dim=linear_size, num_heads=num_heads)
            self.linear_norm = nn.Linear(linear_size, linear_size)
            self.linear2 = nn.Linear(linear_size, linear_size)
            self.multihead_attention2 = nn.MultiheadAttention(embed_dim=linear_size, num_heads=num_heads)
            self.norm2 = nn.LayerNorm(linear_size)
            self.fc = nn.Linear(linear_size, num_classes)
            self.leakyrelu = nn.LeakyReLU()

        def forward(self, x):
            features = self.linear1(x)  
            features = self.linear_norm(features)
            features = self.leakyrelu(features)
            features = features.unsqueeze(1)
            attended_features1, _ = self.multihead_attention1(features, features, features)
            attended_features1 = self.norm2(attended_features1)
            attended_features1 = self.leakyrelu(attended_features1)
            attended_features2 = self.linear2(attended_features1)
            attended_features2 = self.norm2(attended_features2)
            attended_features2 = self.leakyrelu(attended_features2)
            attended_features2, _ = self.multihead_attention2(attended_features2, attended_features2, attended_features2)
            attended_features2 = self.norm2(attended_features2)
            attended_features2 = self.leakyrelu(attended_features2)
            attended_features2 = attended_features2.view(attended_features2.shape[0], 11, -1)
            output = attended_features2.mean(dim=2)
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
            out = F.leaky_relu(self.fc1(imp_flat)) 
            out = self.fc2(out)
            return out

if __name__ == '__main__':
    batch_size = 32
    input_size = 256
    random_input = torch.randn(batch_size,input_size)
    print("Shape of random_input:", random_input.shape)  # Verify the shape

    num_classes = 11
    model = ClassifierDecoder(input_size=input_size, num_classes=num_classes)

    # Now, let's check the forward pass
    output = model(random_input)
    print("Output shape:", output.shape)
