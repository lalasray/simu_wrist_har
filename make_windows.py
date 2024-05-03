import os
import numpy as np
import torch

path = '/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/train/'
pose_path = path+'pose_processed/'
imu_path = path+'imu_processed/'
text_path = path+'text_processed/'

pose_files = []
imu_files = []
text_files = []
pose_window_size = 30 
pose_stride = 15
for file in os.listdir(pose_path):
    if file.endswith('.npy') and os.path.exists(os.path.join(imu_path, file)) and os.path.exists(os.path.join(text_path, file)):
        pose_files.append(os.path.join(pose_path, file))
        imu_files.append(os.path.join(imu_path, file))
        text_files.append(os.path.join(text_path, file))

for idx in range(len(pose_files)):
    pose_data = np.load(pose_files[idx])
    imu_data = np.load(imu_files[idx])
    text_data = np.load(text_files[idx])
    #print(pose_data.shape, imu_data.shape, text_data.shape)
    pose_windows = [pose_data[i:i+pose_window_size] for i in range(0, len(pose_data) - pose_window_size + 1, pose_stride)]
    imu_windows = [imu_data[i:i+(pose_window_size*2)] for i in range(0, len(imu_data) - (pose_window_size*2) + 1, (pose_stride*2))]
    #print(len(pose_windows),len(imu_windows))
    for window, (pose_window, imu_window) in enumerate(zip(pose_windows, imu_windows)):
        pose = torch.tensor(pose_window)
        imu = torch.tensor(imu_window)
        text = torch.tensor(text_data)
        # Save tensors
        save_path = path+'tensors/'
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.splitext(os.path.basename(pose_files[idx]))[0]
        torch.save((pose, imu, text), os.path.join(save_path, f"{filename}_window{window}.pt"))
