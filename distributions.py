"""This cell is train_cifar0.py
This file is using CIFAR100, 
Resnet18 or shufflenet, which can be changed in the load model section
"""
#%% load dependency
import os
import numpy as np
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
from datetime import datetime

from modules import *
from utils import *
from torch.utils.data import Dataset, DataLoader

print('starting date time ', datetime.now())




#%%
loga = torch.tensor(0)
beta, gamma, zeta, eps = torch.tensor([2/3, -0.1, 1.1, 1e-20])
u = torch.arange(0,1,1e-3)
s = torch.sigmoid((torch.log(u+eps) - torch.log(1 - u+eps) + loga) / beta)
sbar = s * (zeta - gamma) + gamma
z = hard_sigmoid(sbar)
#%%
qs = beta*loga.exp()*s**(-beta-1)*(1-s)**(-beta-1)\
    /(loga.exp()*s**(-beta)+(1-s)**(-beta))**2
qs_bar = 1/(zeta-gamma).abs()