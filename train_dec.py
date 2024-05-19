import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from text_enc import TextEncoder
from imu_enc import ImuEncoder
from pose_enc import PoseEncoder
from model import TriModalDecModel
from dataloader import TriDataset, get_data_files
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config
from imu_dec import IMUDecoder
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = config.batch_size
embedding_dim = config.embedding_dim
num_epochs = config.num_epochs

parent = config.parent
val_path = os.path.join(parent, 'data/how2sign/val/tensors')
test_path = os.path.join(parent, 'data/how2sign/test/tensors')
train_path = os.path.join(parent, 'data/how2sign/val/tensors')

dataset_val = TriDataset(get_data_files(val_path))
dataset_train = TriDataset(get_data_files(val_path))
dataset_test = TriDataset(get_data_files(test_path))
combined_dataset = ConcatDataset([dataset_train, dataset_test])

train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

text_encoder = TextEncoder(embedding_dim=embedding_dim).to(device)
imu_encoder = ImuEncoder(embedding_dim=embedding_dim).to(device)
pose_encoder = PoseEncoder(embedding_dim=embedding_dim).to(device)
imu_decoder_i = IMUDecoder(embedding_dim=embedding_dim).to(device)
imu_decoder_p = IMUDecoder(embedding_dim=embedding_dim).to(device)

model = TriModalDecModel(text_encoder, imu_encoder, pose_encoder, imu_decoder_i,imu_decoder_p).to(device)
criterion = config.loss
mse_loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = config.patience
best_val_loss = float('inf')
counter = 0
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=(config.patience)/2)

local_log_dir = "local_logs"
hyperparameters = {"embedding_dim": embedding_dim, "batch_size": batch_size}

for epoch in range(num_epochs):
    total_loss = 0.0
    t_i_loss_total = 0
    t_p_loss_total = 0
    i_p_loss_total = 0
    model.train() 
    for pose, imu, text in train_loader:
        optimizer.zero_grad()
        text_output, imu_output, pose_output, imu_decoder_i, imu_decoder_p = model(text, imu, pose)
        imu_gt = F.normalize(imu[:, 30, :], p=2, dim=-1)
        t_i_loss = criterion(text_output, imu_output)
        t_p_loss = criterion(text_output, pose_output)
        i_p_loss = criterion(imu_output, pose_output)
        imu_loss_i = mse_loss(imu_gt, F.normalize(imu_decoder_i, p=2, dim=-1)).mean()
        imu_loss_p = mse_loss(imu_gt, F.normalize(imu_decoder_p, p=2, dim=-1)).mean()



        loss = t_i_loss + t_p_loss + i_p_loss + imu_loss_i + imu_loss_p

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    total_loss /= len(train_loader)
        
    model.eval()  
    val_loss = 0.0
    with torch.no_grad():
        for pose, imu, text in val_loader:
            text_output, imu_output, pose_output, imu_decoder_i, imu_decoder_p = model(text, imu, pose)
            imu_gt = imu[:, 30, :]
            loss = criterion(text_output, imu_output) + criterion(text_output, pose_output) + criterion(imu_output, pose_output) + mse_loss(imu_gt, F.normalize(imu_decoder_i, p=2, dim=-1)).mean() + mse_loss(imu_gt, F.normalize(imu_decoder_p, p=2, dim=-1)).mean()
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Train Loss: {total_loss}")
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model_fc.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Validation loss hasn't decreased for ", patience, " epochs. Early stopping...")
            break 
