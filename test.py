
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from comic_real_pair import ComicRealPairDataset

# root = 'af2019-ksyun-testA-20190416/'
# outpath = 'testA.csv'
root = 'af2019-ksyun-testB-20190424/'
outpath = 'testB.csv'
ann_path = root + 'list.csv'
df = pd.read_csv(ann_path, sep=',', header=0)
pair_list = df.ix[:, 1:]
pic_dir = root + 'images/'

pair_dataset = ComicRealPairDataset(pair_list,
                                    pic_dir = pic_dir,
                                    is_training= False,
                                    transform=transforms.Compose([transforms.Resize((100,100)),#TODO
                                                                  transforms.ToTensor()
                                                                  ]),
                                    )
from network import SiameseNetwork
net = SiameseNetwork()
net.load_state_dict(torch.load('final.th'))

id_confidence = []

test_dataloader = DataLoader(pair_dataset,num_workers=0,batch_size=1,shuffle=False)
for i, data in enumerate(test_dataloader):
    x0, x1 = data
    x0, x1 = torch.tensor(x0), torch.tensor(x1)
    output1,output2 = net(x0,x1)
    euclidean_distance = F.pairwise_distance(output1, output2).item()
    score = max(0 ,1- euclidean_distance)
    id_confidence.append([i, score])# 分数可以加调节参数改
save = pd.DataFrame(id_confidence)
save.to_csv(outpath,index=False,sep=',', header=['group_id', 'confidence'])