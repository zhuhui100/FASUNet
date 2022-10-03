import os
import numpy as np


data_dir = "../dataset_2D/train_big_3D/image/"
save_dir = "../dataset_2D/train_304/image/"

count = 0
for name in os.listdir(data_dir):
    data_3d = np.load(data_dir + name)

    for i in range(data_3d.shape[-1]):
        name_2d = name.replace(".npy", "_" + str(i)+".npy")
        np.save(save_dir + name_2d, data_3d[:,:,i])
        count += 1

print(data_3d.shape, count)