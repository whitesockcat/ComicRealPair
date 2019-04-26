import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import random
from PIL import Image
import torch
import os
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd

class Config():
    training_dir = "train_mix/"
    testing_dir = "./data/faces/testing/"# TODO
    train_batch_size = 4 # 64
    train_number_epochs = 5 #100

from comic_real_pair import ComicRealPairDataset
from network import SiameseNetwork
from loss import ContrastiveLoss

pair_path = 'pair_list.csv'
pair_list = pd.read_csv(pair_path, sep=',', header=0)

pair_dataset = ComicRealPairDataset(pair_list,
                transform=transforms.Compose([transforms.Resize((100,100)),#TODO
                                            transforms.ToTensor()
                                            ]),
                                    )

train_dataloader = DataLoader(pair_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
#         img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        img0, img1 , label = torch.tensor(img0, requires_grad=True), torch.tensor(img1, requires_grad=True), torch.tensor(label, requires_grad=True)
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 100 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            # iteration_number +=10
            # counter.append(iteration_number)
            # loss_history.append(loss_contrastive.item())
# show_plot(counter,loss_history)

torch.save(net.state_dict(), '/userhome/comic/final.th')
