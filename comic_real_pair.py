
import os
import pandas as pd
from PIL import Image

pair_path = 'pair_list.csv'
pair_list = pd.read_csv(pair_path, sep=',', header=0)

import random
from PIL import Image
import torch
import os
from torch.utils.data import DataLoader,Dataset
import numpy as np

class ComicRealPairDataset(Dataset):
    
    def __init__(self,
                 pair_list,
                 is_training = True,
                 pic_dir = 'train_mix/',
                 transform=None):
        self.pic_dir = pic_dir
        self.transform = transform
        self.is_training = is_training
        self.pair_list = pair_list
        real_num = len(self.pair_list)
        self.ids = [i for i in range(real_num)]# 选real作总数

    def __getitem__(self,index):
        idx = self.ids[index]
        
        real_path = self.pic_dir + self.pair_list.iat[idx, 0]
        comi_path = self.pic_dir + self.pair_list.iat[idx, 1]
        
        real_img = Image.open(real_path).convert('L')#RGB
        comi_img = Image.open(comi_path).convert('L')#RGB
        
        if self.transform is not None:
            real_img = self.transform(real_img)
            comi_img = self.transform(comi_img)
            
        if self.is_training:
            target = int(self.pair_list.iat[idx, 2])
            return real_img, comi_img, target
        else:
            return real_img, comi_img
    
    def __len__(self):
        return len(self.ids)
