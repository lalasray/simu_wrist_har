import torch
import torch.nn as nn
import torch.nn.functional as F
import config

imu_decoder_type = config.imu_decoder_type

if imu_decoder_type == "cnn":

    class IMUDecoder(nn.Module):
        def __init__(self, embedding_dim, output_dim=12):
            super(IMUDecoder, self).__init__()
            self.embedding_dim = embedding_dim
            self.output_dim = output_dim
            
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(16)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(32)
            self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(64)
            
            self.fc1 = nn.Linear(64 * embedding_dim, 128)
            self.bn_fc1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, output_dim)
            
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
            
        def forward(self, x):
            x = x.unsqueeze(1) 
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.leaky_relu(x) 
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.leaky_relu(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.leaky_relu(x)
            
            x = x.view(x.size(0), -1) 
            x = self.fc1(x)
            x = self.bn_fc1(x)
            x = self.leaky_relu(x) 
            
            x = self.fc2(x)
            
            return x
        
elif imu_decoder_type == "attention":

    class MultiHeadAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        def forward(self, x):
            # x shape: (batch_size, 3, embed_dim)
            # MultiheadAttention expects shape: (3, batch_size, embed_dim)
            x = x.permute(1, 0, 2)  # (3, batch_size, embed_dim)
            attn_output, _ = self.multihead_attn(x, x, x)
            attn_output = attn_output.permute(1, 0, 2)  # (batch_size, 3, embed_dim)
            return attn_output

    class IMUDecoder(nn.Module):
        def __init__(self, input_dim=256, num_heads=4, num_classes=10):
            super(IMUDecoder, self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            
            # Multi-head attention
            self.multihead_attn = MultiHeadAttention(embed_dim=128, num_heads=num_heads)
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(128)
            
            # Fully connected layers
            self.fc1 = nn.Linear(3 * 128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)

        def forward(self, x):
            # Convolutional layers
            x = x.permute(0, 2, 1)  # (batch_size, 256, 3)
            x = F.relu(self.conv1(x))  # (batch_size, 64, 3)
            x = F.relu(self.conv2(x))  # (batch_size, 128, 3)
            x = x.permute(0, 2, 1)  # (batch_size, 3, 128)
            
            # Multi-head attention
            x = self.multihead_attn(x)
            
            # Layer normalization
            x = self.layer_norm(x)
            
            # Flatten the input
            x = x.view(x.size(0), -1)  # (batch_size, 3 * 128)
            
            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
            return x
        
elif imu_decoder_type == "d_attention":

    class MultiHeadAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        def forward(self, x):
            # x shape: (batch_size, seq_length, embed_dim)
            # MultiheadAttention expects shape: (seq_length, batch_size, embed_dim)
            x = x.permute(1, 0, 2)  # (seq_length, batch_size, embed_dim)
            attn_output, _ = self.multihead_attn(x, x, x)
            attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_length, embed_dim)
            return attn_output

    class IMUDecoder(nn.Module):
        def __init__(self, input_dim=6, conv_out_channels=128, num_heads=4, num_classes=10):
            super(IMUDecoder, self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=conv_out_channels, kernel_size=3, padding=1)
            
            # First multi-head attention
            self.multihead_attn1 = MultiHeadAttention(embed_dim=conv_out_channels, num_heads=num_heads)
            
            # First fully connected layer followed by layer normalization
            self.fc1 = nn.Linear(conv_out_channels, conv_out_channels)
            self.layer_norm1 = nn.LayerNorm(conv_out_channels)
            
            # Second multi-head attention
            self.multihead_attn2 = MultiHeadAttention(embed_dim=conv_out_channels, num_heads=num_heads)
            
            # Fully connected layers for final classification
            self.fc2 = nn.Linear(256 * conv_out_channels, 512)
            self.fc3 = nn.Linear(512, 256)
            self.fc4 = nn.Linear(256, num_classes)

        def forward(self, x):
            # Convolutional layers
            x = x.permute(0, 2, 1)  # (batch_size, 6, 256)
            x = F.relu(self.conv1(x))  # (batch_size, 64, 256)
            x = F.relu(self.conv2(x))  # (batch_size, 128, 256)
            x = x.permute(0, 2, 1)  # (batch_size, 256, 128)
            
            # First multi-head attention
            x = self.multihead_attn1(x)
            
            # Fully connected layer followed by layer normalization
            x = F.relu(self.fc1(x))
            x = self.layer_norm1(x)
            
            # Second multi-head attention
            x = self.multihead_attn2(x)
            
            # Flatten the input
            x = x.reshape(x.size(0), -1)  # (batch_size, 256 * 128)
            
            # Fully connected layers for final classification
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            
            return x
            
if __name__ == '__main__':
    batch_size = 32
    embedding = torch.randn(batch_size, 6, 256)
    decoder = IMUDecoder(input_dim=256, num_heads=4, num_classes=10)
    output = decoder(embedding)
    print(output.shape)  # Should be (batch_size, 10)