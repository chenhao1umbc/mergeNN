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

#%% Concrete pdf and cdf
def pdf_qs(x):
    qs = beta*loga.exp()*x**(-beta-1)*(1-x)**(-beta-1)\
    /(loga.exp()*x**(-beta)+(1-x)**(-beta))**2
    return qs
def cdf_Qs(x):
    Qs = ((x.log()-(1-x).log())*beta-loga).sigmoid()
    return Qs
qs = pdf_qs(s)
Qs = cdf_Qs(s)
plt.plot(u, qs)
plt.plot(u, Qs)

#%% Hard concrete pdf and cdf
def pdf_qs_bar(x):
    return 1/(zeta-gamma).abs()*pdf_qs((x-gamma)/(zeta-gamma))

def cdf_Qs_bar(x):
    return cdf_Qs((x-gamma)/(zeta-gamma))

def delta(x):
    a = x.clone()
    a[a!=0] = 1
    a = 1-a
    return a

qz1 = cdf_Qs_bar(0)*delta(z) + (1-cdf_Qs_bar(1))*delta(z-1) + \
    (cdf_Qs_bar(1)-cdf_Qs_bar(0))*pdf_qs_bar(z)
qz = z.clone()
qz[z==0] = cdf_Qs_bar(0)
qz[z==1] = (1-cdf_Qs_bar(1))
idx = torch.logical_or(z==0, z==1)
qz[~idx] = (cdf_Qs_bar(1)-cdf_Qs_bar(0))*\
    pdf_qs_bar(z[~idx])

