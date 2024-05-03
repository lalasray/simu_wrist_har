import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import glob


directory = '/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/val/npz'

npz_files = glob.glob(os.path.join(directory, '**/*.npz'), recursive=True)

for npz_file in npz_files:

    file_path = npz_file
    filename = os.path.basename(file_path)
    #print(filename)
    filename_no_extension = os.path.splitext(filename)[0]
    print(filename_no_extension+ "is being processed.")
    #print(filename_no_extension)
    np_array = np.load(file_path)
    root = np_array["smplx_root_pose"]
    body = np_array["smplx_body_pose"]
    lhand = np_array["smplx_lhand_pose"]
    rhand = np_array["smplx_rhand_pose"]
    #print(root.shape, body.shape, lhand.shape, rhand.shape)
    data_mb = np.concatenate((root, body, lhand, rhand), axis=1)
    #print(data_mb.shape)

    new_indices = np.linspace(0, data_mb.shape[0] - 1, int(data_mb.shape[0])*2)
    interpolated_functions = [interp1d(np.arange(data_mb.shape[0]), data_mb[:, i], kind='cubic') for i in range(data_mb.shape[1])]
    interpolated_columns = [func(new_indices) for func in interpolated_functions]
    interpolated_array = np.column_stack(interpolated_columns)
    #print(interpolated_array.shape)

    new_directory = directory.replace('npz', 'pose_processed/')
    np.save(new_directory+filename_no_extension+".npy",interpolated_array)
    print(filename_no_extension + "is done.")