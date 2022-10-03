import random
import numpy as np
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate
import cv2

from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
from torchvision import transforms


class MyDataset(data.Dataset):


    def __init__(self, data_root_dir, txt_path, shuffle=True, transform=None):
        with open(txt_path, 'r') as f:
            data_path = [x.strip("\n") for x in f.readlines()]
        data_path.sort()

        
        self.imgList = [data_root_dir + x.split(",")[0] for x in data_path]
        self.labelList = [data_root_dir + x.split(",")[1] for x in data_path]
        self.transform = transform
    
    def __getitem__(self, index):
        
        img = np.load(self.imgList[index])

        label = np.load(self.labelList[index])

        if self.transform:
            [img, label] = self.transform(img, label)


        img = np.expand_dims(img, axis=0)           
        img = torch.from_numpy(img).float()
        y_2 = zoom(label, (1/2, 1/2), order=0)
        y_4 = zoom(label, (1/4, 1/4), order=0)
        y_8 = zoom(label, (1/8, 1/8), order=0)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        name = self.imgList[index].split("/")[-1]

        return (img, label, y_2, y_4, y_8, name)
    
    def __len__(self):
        return len(self.imgList)


if __name__ == "__main__":
    from myTransforms import (MinMaxNormalize, PadZ_8, Compose)
    import cv2
    from draw import save_3D_result, gray_to_color
    import os


    train_tranforms = None
    dataset = MyDataset(
                    ".",
                    "../dataset_2D/txt/33.txt",
                    shuffle = False,
                    transform=train_tranforms,
                    )
    data_loader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=1)

    save_dir = "./temp/"
    txt = open("./temp.txt", "w")
    os.makedirs(save_dir, exist_ok=True)
    for x, y, y_2, y_4, y_8, name in data_loader:
        print(n, list(x.size()), list(y.size()), y.max(), x.max())
        print(name, file=txt)
    txt.close()


