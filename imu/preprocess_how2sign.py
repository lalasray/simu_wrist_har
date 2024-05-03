import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import glob


directory = '/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/train/npz'

npz_files = glob.glob(os.path.join(directory, '**/*.npz'), recursive=True)

for npz_file in npz_files:

    file_path = npz_file
    filename = os.path.basename(file_path)
    #print(filename)
    filename_no_extension = os.path.splitext(filename)[0]
    print(filename_no_extension+ "is being processed.")
    #print(filename_no_extension)
    imu_directory = directory.replace('npz', 'imu/')
    left = imu_directory+filename_no_extension+"_Left_wrist.csv"
    df_left = pd.read_csv(left)
    np_array_left = df_left.to_numpy()
    #print(np_array_left.shape)
    data_left = np_array_left[:,1:7]
    #print(data_left.shape)
    right = imu_directory+filename_no_extension+"_Right_wrist.csv"
    df_right = pd.read_csv(right)
    np_array_right = df_right.to_numpy()
    #print(np_array_right.shape)
    data_right = np_array_right[:,1:7]
    #print(data_right.shape)

    data_mb = np.concatenate((data_left, data_right), axis=1)
    #print(data_mb.shape)

    new_indices = np.linspace(0, data_mb.shape[0] - 1, int(data_mb.shape[0])*4)
    interpolated_functions = [interp1d(np.arange(data_mb.shape[0]), data_mb[:, i], kind='cubic') for i in range(data_mb.shape[1])]
    interpolated_columns = [func(new_indices) for func in interpolated_functions]
    interpolated_array = np.column_stack(interpolated_columns)
    #print(interpolated_array.shape)

    #def moving_average_smoothing_2d(data, window_size):

        #smoothed_data = np.zeros_like(data, dtype=float)
        #for i in range(data.shape[0]):
        #    smoothed_data[i, :] = moving_average_smoothing(data[i, :], window_size)

        #return smoothed_data

    #def moving_average_smoothing(data, window_size):

        #kernel = np.ones(window_size) / window_size
        #smoothed_data = np.convolve(data, kernel, mode='same')
        #return smoothed_data


    #plt.figure(figsize=(20, 4))

    #plt.subplot(1, 3, 2)
    #plt.plot(data_mb)
    #plt.xlabel('Index')
    #plt.ylabel('Value')
    #plt.title('Original')

    #plt.subplot(1, 3, 1)
    #plt.plot(interpolated_array)
    #plt.xlabel('Index')
    #plt.ylabel('Value')
    #plt.title('Interpolated')

    #plt.subplot(1, 3, 3)
    #plt.plot(moving_average_smoothing_2d(interpolated_array,4))
    #plt.xlabel('Index')
    #plt.ylabel('Value')
    #plt.title('Original')

    # Adjust layout and display plots
    #plt.tight_layout()
    #plt.show()
    new_directory = directory.replace('npz', 'imu_processed/')
    np.save(new_directory+filename_no_extension+".npy",interpolated_array)
    print(filename_no_extension + "is done.")