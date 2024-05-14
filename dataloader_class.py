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
        imu, label_data = torch.load(self.data_files[idx])
        label = torch.argmax(label_data).unsqueeze(0)
        return imu, label

def get_data_files(data_path):
    data_files = []
    for file in os.listdir(data_path):
        if file.endswith('.pt'):
            data_files.append(os.path.join(data_path, file))
    return data_files

#parent = config.parent
#train_path = parent + 'data/openpack_uni/tensors' 
#train_dataset = TriDataset(get_data_files(train_path))
#data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#for imu, label_data in data_loader:
#    print(imu.type, label_data[0])
#    break