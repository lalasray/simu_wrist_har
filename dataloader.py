import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TriDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        pose, imu, text = torch.load(self.data_files[idx])
        pose = (pose.to(device)).float()
        imu = (imu.to(device)).float()
        text = text.to(device)
        #print(pose.shape, imu.shape, text.shape)
        return pose, imu, text

def get_data_files(data_path):
    data_files = []
    for file in os.listdir(data_path):
        if file.endswith('.pt'):
            data_files.append(os.path.join(data_path, file))
    return data_files

class QaudDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        pose, imu, text = torch.load(self.data_files[idx])
        pose = (pose.to(device)).float()
        imuL = (imu[:,:6].to(device)).float()
        imuR = (imu[:,6:].to(device)).float()
        text = text.to(device)
        #print(pose.shape, imu.shape, text.shape)
        return pose, imuL,imuR, text

def get_data_files(data_path):
    data_files = []
    for file in os.listdir(data_path):
        if file.endswith('.pt'):
            data_files.append(os.path.join(data_path, file))
    return data_files

if __name__ == '__main__':
    batch_size = config.batch_size
    val_path = os.path.join(config.parent, 'data/how2sign/val/tensors')
    dataset = QaudDataset(get_data_files(val_path))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for pose, imuL, imuR, text in data_loader:
        print(pose.shape, imuL.shape, imuR.shape, text.shape)
        break
