import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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

#batch_size = 32
#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#for pose, imu, text in data_loader:
    #print(pose.shape, imu.shape, text.shape)
    #break
