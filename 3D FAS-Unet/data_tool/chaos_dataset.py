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
    """
        example:
        txt_path = ../data/txt/train.txt
    """

    def __init__(self, data_root_dir, txt_path, shuffle=True, transform=None):
        with open(txt_path, 'r') as f:
            data_path = [x.strip("\n") for x in f.readlines()]
            data_path = sorted(data_path)
        if shuffle:
            random.shuffle(data_path)

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

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        name = self.imgList[index].split("/")[-1]

        return (img, label,  name)
    
    def __len__(self):
        return len(self.imgList)


if __name__ == "__main__":

    train_tranforms = None

    dataset = MyDataset(
                    ".",
                    "../dataset/overlap_patch_from_512/val_128x128x64.txt",
                    shuffle = False,
                    transform=train_tranforms,
                    )
    data_loader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=1)

    for x, y, name in data_loader:
        # print(name, list(x.size()), x.max(), "标签：", list(y.size()), y.max())
        print(name)

    print(data_loader.__len__())        

