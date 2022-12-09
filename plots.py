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
file1 = open('./modulelevel3_log.txt', 'r')
lines = file1.readlines()
start = len('after validation tensor(')
res = {i:[] for i in range(6)}
for ii, line in enumerate(lines):
    if line[:start] == 'after validation tensor(':
        data = line[start:].strip()
        exec('a='+data[:-1])
        for i in range(6):
            res[i].append(a[i])
plt.rcParams['figure.dpi'] = 500

plt.figure()
marker = ['-x', '-o', '-^', '-d', '-v', '->']
for i in range(6):
    plt.plot(res[i], marker[i], markevery=20)
plt.legend([f'log alpha_{i}' for i in range(1,7)])
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.savefig('module_loga.png')

#%%
h = [328.9614458622783,
 331.01736405892507,
 328.14458622783,
 330.736405892507,
 329.9614458622783,
 274.6798730743995,
 272.42975964690595,
 273.6798730743995,
 239.56801906769252,
 238.56801906769252,
 238.56801906769252]
plt.figure(figsize=(10,5))
for i in range(4):
    rgb = torch.rand(3).tolist()
    if i == 0: plt.bar([str(ii) for ii in range(1, 6)], h[:5], width=0.6, color=[rgb])
    if i == 1: plt.bar([str(ii) for ii in range(6, 9)], h[5:8], width=0.6, color=[rgb])
    if i == 2: plt.bar([str(9), str(10)], h[8:10], width=0.6, color=[rgb])
    if i == 3: plt.bar('Merged', h[10], width=0.6, color=[rgb])
plt.xlabel('Model index')
plt.ylabel('PPL')
plt.savefig('bar.png')