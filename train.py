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

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
embedding_dim = 512
num_epochs = 3

text_data = torch.randn(100,10,768).to(device)
imu_data = torch.randn(100,12,60).to(device)
video_data = torch.randn(100,30,1024,12,16).to(device)
pose_data = torch.randn(100,30,1024,25).to(device)

# Create DataLoader
dataset = TensorDataset(text_data, imu_data, video_data, pose_data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#text_encoder = TextEncoder(embedding_dim=embedding_dim).to(device)
imu_encoder = ImuEncoder(embedding_dim = embedding_dim).to(device)

for batch in data_loader:
    text_batch, imu_batch, video_batch, pose_batch = batch
    #print("Text Data (first element):", text_batch[0].shape)
    #output = text_encoder(text_batch)
    #print(output.shape)

    #print("IMU Data (first element):", imu_batch.shape)
    output = imu_encoder(imu_batch)
    print(output.shape)

    #print("Video Data (first element):", video_batch[0].shape)
    #print("Pose Data (first element):", pose_batch[0].shape)
    #break

#video_encoder = VideoEncoder(embedding_dim).to(device)
#pose_encoder = PoseEncoder(embedding_dim).to(device)

#model = MultiModalModel(text_encoder, imu_encoder, video_encoder, pose_encoder).to(device)
#criterion = InfonceLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
#for epoch in range(num_epochs):
#    total_loss = 0.0
#    for batch_data in data_loader:
#        text_input, imu_input, video_input, pose_input = [x.to(device) for x in batch_data]
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
