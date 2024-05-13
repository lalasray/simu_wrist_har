import os
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

data = "openpack"

if data == "asl":
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

elif data == "mmfit":
    
    print("ToDo")

elif data == "openpack":
    directory = '/media/lala/Seagate/Dataset/Meta/imuwrist/preprocessed-IMU-with-operation-labels/imuWithOperationLabel/'
    pose_window_size = 60
    pose_stride = 30

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            columns_of_interest = ['atr01/acc_x', 'atr01/acc_y', 'atr01/acc_z', 'atr01/gyro_x', 'atr01/gyro_y', 'atr01/gyro_z',
                            'atr02/acc_x', 'atr02/acc_y', 'atr02/acc_z', 'atr02/gyro_x', 'atr02/gyro_y', 'atr02/gyro_z']

            imu = df[columns_of_interest].to_numpy()
            columns_of_interest_2 = ['operation']
            label = df[columns_of_interest_2].to_numpy()

            for idx in range(0, imu.shape[0] - pose_window_size, pose_stride):
                imu_window = imu[idx:idx+pose_window_size]
                label_window = label[idx:idx+pose_window_size]
                imu_window_data = torch.tensor(imu_window)
                label_window_data = torch.tensor(label_window)
                save_path = '/home/lala/other/Repos/git/simu_wrist_har/data/openpack_uni/tensors/'
                os.makedirs(save_path, exist_ok=True)
                name, ext = os.path.splitext(filename)
                window_filename = f"{name}_window{idx}.pt"
                label_window_flat = label_window_data.flatten()
                majority_label = torch.mode(label_window_flat, dim=0).values.item()
                num_classes = 11
                label_window_class = F.one_hot(torch.tensor(majority_label), num_classes=num_classes)
                torch.save((imu_window_data, label_window_class), os.path.join(save_path, window_filename))
                

elif data == "mmfit_multi":

    print("ToDo")

elif data == "openpack_multi":
    
    df = pd.read_csv('/media/lala/Seagate/Dataset/Meta/imuwrist/preprocessed-IMU-with-operation-labels/imuWithOperationLabel/U0101-S0100.csv')
    #print(df)
    columns_of_interest = ['atr01/acc_x', 'atr01/acc_y', 'atr01/acc_z', 'atr01/gyro_x', 'atr01/gyro_y', 'atr01/gyro_z',
                       'atr02/acc_x', 'atr02/acc_y', 'atr02/acc_z', 'atr02/gyro_x', 'atr02/gyro_y', 'atr02/gyro_z']

    imu = df[columns_of_interest].to_numpy()
    columns_of_interest_2 = ['operation']
    label = df[columns_of_interest_2].to_numpy()
    print(label.shape)

    