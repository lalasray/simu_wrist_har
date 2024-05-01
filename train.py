import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from video_enc import VideoEncoder
from pose_enc import PoseEncoder
from loss import InfonceLoss
from model import MultiModalModel

# Generate some random data for demonstration
num_samples = 1000
text_input_dim = 768
imu_input_dim = 30*12
video_input_dim = 30*1024*12*16
pose_input_dim = 30*1024*25
embedding_dim = 512
num_epochs = 10
batch_size = 32

text_data = torch.randn(num_samples, text_input_dim)
imu_data = torch.randn(num_samples, imu_input_dim)
video_data = torch.randn(num_samples, video_input_dim)
pose_data = torch.randn(num_samples, pose_input_dim)

# Create DataLoader
dataset = TensorDataset(text_data, imu_data, video_data, pose_data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage
text_encoder = TextEncoder(text_input_dim, embedding_dim)
imu_encoder = ImuEncoder(imu_input_dim, embedding_dim)
video_encoder = VideoEncoder(video_input_dim, embedding_dim)
pose_encoder = PoseEncoder(pose_input_dim, embedding_dim)

model = MultiModalModel(text_encoder, imu_encoder, video_encoder, pose_encoder)
print(model)
#criterion = InfonceLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
#for epoch in range(num_epochs):
#    total_loss = 0.0
#    for batch_data in data_loader:
#        text_input, imu_input, video_input, pose_input = batch_data
#        optimizer.zero_grad()
#        text_output, imu_output, video_output, pose_output = model(text_input, imu_input, video_input, pose_input)
        # Calculate Infonce loss for each pair of modalities
#        loss = criterion(text_output, imu_output) + criterion(text_output, video_output) + \
#               criterion(text_output, pose_output) + criterion(imu_output, video_output) + \
#               criterion(imu_output, pose_output) + criterion(video_output, pose_output)
#        loss.backward()
#        optimizer.step()
#        total_loss += loss.item()
#    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")
