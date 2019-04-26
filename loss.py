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

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        label = label.type(torch.cuda.FloatTensor)
        x = (1-label)
        y = torch.pow(euclidean_distance, 2).type(torch.cuda.FloatTensor)
        a = x * y
        b = (label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)).type(torch.cuda.FloatTensor)
        # loss_contrastive = torch.mean((a + b).float())
        loss_contrastive = (a+b).mean()
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
