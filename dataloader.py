import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, pose_files, imu_files, text_files, pose_window_size, pose_stride, imu_window_size, imu_stride):
        self.pose_files = pose_files
        self.imu_files = imu_files
        self.text_files = text_files
        self.pose_window_size = pose_window_size
        self.pose_stride = pose_stride
        self.imu_window_size = imu_window_size
        self.imu_stride = imu_stride

    def __len__(self):
        return len(self.pose_files)

    def __getitem__(self, idx):
        pose_data = np.load(self.pose_files[idx])
        imu_data = np.load(self.imu_files[idx])
        text_data = np.load(self.text_files[idx])
        print(pose_data.shape, imu_data.shape, text_data.shape)
        #pose_windows = [pose_data[i:i+self.pose_window_size] for i in range(0, len(pose_data) - self.pose_window_size + 1, self.pose_stride)]
        #imu_windows = [imu_data[i:i+self.imu_window_size] for i in range(0, len(imu_data) - self.imu_window_size + 1, self.imu_stride)]
        
        # Convert pose and IMU windows to tensors
        #pose_tensors = [torch.tensor(window) for window in pose_windows]
        #imu_tensors = [torch.tensor(window) for window in imu_windows]

        # Pad sequences to the same length
        #padded_pose = pad_sequence(pose_tensors, batch_first=True)
        #padded_imu = pad_sequence(imu_tensors, batch_first=True)

        # Combine into a single data point
        #combined_data_point = {
        #    'pose': padded_pose,
        #    'imu': padded_imu,
        #    'text': torch.tensor(text_data)
        #}

        return 0

# Directory paths for pose, IMU, and text data
pose_path = '/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/val/pose_processed/'
imu_path = '/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/val/imu_processed/'
text_path = '/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/val/text_processed/'

pose_files = []
imu_files = []
text_files = []
for file in os.listdir(pose_path):
    if file.endswith('.npy') and os.path.exists(os.path.join(imu_path, file)) and os.path.exists(os.path.join(text_path, file)):
        pose_files.append(os.path.join(pose_path, file))
        imu_files.append(os.path.join(imu_path, file))
        text_files.append(os.path.join(text_path, file))

print(len(pose_files), len(imu_files), len(text_files))

pose_window_size = 30
pose_stride = 15
imu_window_size = 60
imu_stride = 30

# Create dataset instance
dataset = CustomDataset(pose_files, imu_files, text_files, pose_window_size, pose_stride, imu_window_size, imu_stride)

# Create data loader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate over data loader in training loop
for batch in data_loader:
    # Training code here
    #print(batch)
    break
