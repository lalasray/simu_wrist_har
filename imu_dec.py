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
            
if __name__ == '__main__':
    batch_size = config.batch_size
    embedding_dim = config.embedding_dim
    model = IMUDecoder(embedding_dim=embedding_dim)
    input_tensor = torch.randn(batch_size, embedding_dim)
    output_tensor = model(input_tensor)
    print(f"Output shape: {output_tensor.shape}")