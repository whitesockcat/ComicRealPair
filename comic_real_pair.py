import random
from PIL import Image
import torch
import os
from torch.utils.data import DataLoader,Dataset
import numpy as np

class ComicRealPairDataset(Dataset):
    
    def __init__(self,
                pic_dir,
                ann_dir,
                transform=None):
        self.pic_dir = pic_dir 
        self.ann_dir = ann_dir  
        self.transform = transform
        comic_num = 100
        real_num = 150
        # 选一个数量多的
        self.ids = [i for i in range(real_num)]

    def _get_path1(self, idx, isSame = True):
        path1 = None
        return path1

    def _get_path0(self, idx):
        name = 'abc.jpg' # TODO
        path0 = self.pic_dir + name
        return path0

    def _get_target(self, isSame):
        target = torch.from_numpy(np.array([isSame])) # TODO
        return target

    def __getitem__(self,index):
        img_id = self.ids[index]
        is_same_class = random.randint(0,1)

        path0 = self._get_path0(img_id)
        path1 = self._get_path1(img_id, is_same_class)

        img0 = Image.open(path0).convert('RGB')
        img1 = Image.open(path1).convert('RGB')
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        target = self._get_target(is_same_class)

        return img0, img1, target
    
    def __len__(self):
        return len(self.ids)

