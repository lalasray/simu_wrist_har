import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from video_enc import VideoEncoder
from pose_enc import PoseEncoder
from loss import InfonceLoss
from model import TriModalModel,QuadModalModel
from dataloader import TriDataset,get_data_files
import os
from torch.utils.data import ConcatDataset
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
embedding_dim = 512
num_epochs = 100
#parent = os.path.abspath(sys.argv[0])
parent = "c:/Users/lalas/Documents/GitHub/simu_wrist_har/"
val_path = parent + '/data/how2sign/val/tensors'
test_path = parent + '/data/how2sign/test/tensors'
train_path = parent + '/data/how2sign/val/tensors'

dataset_val = TriDataset(get_data_files(val_path))
dataset_train = TriDataset(get_data_files(train_path))
dataset_test = TriDataset(get_data_files(test_path))
combined_dataset = ConcatDataset([dataset_train, dataset_test])

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

text_encoder = TextEncoder(embedding_dim=embedding_dim).to(device)
imu_encoder = ImuEncoder(embedding_dim = embedding_dim).to(device)
pose_encoder = PoseEncoder(embedding_dim = embedding_dim).to(device)

model = TriModalModel(text_encoder, imu_encoder, pose_encoder).to(device)
criterion = InfonceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 5
best_val_loss = float('inf')
counter = 0
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train() 
    for pose, imu, text in train_loader:
        optimizer.zero_grad()
        imu = imu.permute(0, 2, 1)
        #imu = imu.unsqueeze(3) # only for lstm
        text_output, imu_output, pose_output = model(text, imu, pose)
        loss = criterion(text_output, imu_output) + criterion(text_output, pose_output) + criterion(imu_output, pose_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {total_loss}")

    model.eval()  
    val_loss = 0.0
    with torch.no_grad():
        for pose, imu, text in val_loader:
            imu = imu.permute(0, 2, 1)
            #imu = imu.unsqueeze(3) # only for lstm
            text_output, imu_output, pose_output = model(text, imu, pose)
            loss = criterion(text_output, imu_output) + criterion(text_output, pose_output) + criterion(imu_output, pose_output)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth') 
    else:
        counter += 1
        if counter >= patience:
            print("Validation loss hasn't decreased for", patience, "epochs. Early stopping...")
            break 