import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from model import TriModalModel, QuadModalModel
from dataloader import TriDataset, get_data_files
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = config.batch_size
embedding_dim = config.embedding_dim
num_epochs = config.num_epochs

parent = config.parent
val_path = os.path.join(parent, 'data/how2sign/val/tensors')
test_path = os.path.join(parent, 'data/how2sign/test/tensors')
train_path = os.path.join(parent, 'data/how2sign/train/tensors')

dataset_val = TriDataset(get_data_files(val_path))
dataset_train = TriDataset(get_data_files(train_path))
dataset_test = TriDataset(get_data_files(test_path))
combined_dataset = ConcatDataset([dataset_train, dataset_test])

train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

text_encoder = TextEncoder(embedding_dim=embedding_dim).to(device)
imu_encoderL = ImuEncoder(embedding_dim=embedding_dim).to(device)
imu_encoderR = ImuEncoder(embedding_dim=embedding_dim).to(device)
pose_encoder = PoseEncoder(embedding_dim=embedding_dim).to(device)

model = QuadModalModel(text_encoder, imu_encoderL, imu_encoderR, pose_encoder).to(device)
criterion = config.loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = config.patience
best_val_loss = float('inf')
counter = 0
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=(config.patience)/2)

local_log_dir = "local_logs"
hyperparameters = {"embedding_dim": embedding_dim, "batch_size": batch_size}

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train() 
    for pose, imu, text in train_loader:
        optimizer.zero_grad()
        text_output, imu_outputL, imu_outputR, pose_output = model(text, imu, pose)
        
        t_iL_loss = criterion(text_output, imu_outputL)
        t_iR_loss = criterion(text_output, imu_outputR)
        t_p_loss = criterion(text_output, pose_output)
        iL_p_loss = criterion(imu_outputL, pose_output)
        iR_p_loss = criterion(imu_outputR, pose_output)
        iR_p_loss = criterion(imu_outputR, pose_output)
        iR_iL_loss = criterion(imu_outputR, imu_outputL)

        loss = t_iL_loss + t_iR_loss + t_p_loss+iL_p_loss+iR_p_loss+iR_p_loss+iR_iL_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    total_loss /= len(train_loader)
        
    model.eval()  
    val_loss = 0.0
    with torch.no_grad():
        for pose, imu, text in val_loader:
            text_output, imu_output, pose_output = model(text, imu, pose)
            loss = criterion(text_output, imu_output) + criterion(text_output, pose_output) + criterion(imu_output, pose_output)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Train Loss: {total_loss}")
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model_mc.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Validation loss hasn't decreased for ", patience, " epochs. Early stopping...")
            break 
