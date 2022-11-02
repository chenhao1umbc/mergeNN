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
batch_size = 256
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

#%%
id0 = 'resnet18_mnist'
def get_random_model(seed, load=False):
    torch.manual_seed(seed)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # original
    if load : model.load_state_dict(torch.load(f'teachers/{id0}.pt'))
    for param in model.parameters():
        param.requires_grad = False
    return model
t0 = get_random_model(0, load=True) # t0 is trained, t1-t3 are not trained
for i in range(1,4):
    exec(f"t{i} = get_random_model({i})")

"swap t1 1st half, t2 second half with trained model"
l = 60 #len(list(t0.parameters()))//2
temp_dict = t1.state_dict().copy()
for i, k in enumerate(t0.state_dict().keys()):
    if i < 60:
        if k in temp_dict.keys():
            temp_dict[k] = t0.state_dict()[k]
t1.load_state_dict(temp_dict, strict=False)

temp_dict = t2.state_dict().copy()
for i, k in enumerate(t0.state_dict().keys()):
    if i >= 60:
        if k in temp_dict.keys():
            temp_dict[k] = t0.state_dict()[k]
t2.load_state_dict(temp_dict, strict=False)

for i in range(1,4):
    exec(f"t{i} = t{i}.cuda()")

"cut 3 models into 6 modules"
import copy
def split_model(m):
    h1 = copy.deepcopy(m)
    h2 = copy.deepcopy(m)
    h1.layer3 = nn.Sequential()
    h1.layer4 = nn.Sequential()
    h1.avgpool = nn.Sequential()
    h1.fc = nn.Sequential()

    h2.layer1 = nn.Sequential()
    h2.layer2 = nn.Sequential()
    h2.conv1 = nn.Sequential()
    h2.bn1 = nn.Sequential()
    h2.relu = nn.Sequential()
    h2.maxpool = nn.Sequential()
    return h1, h2
m11, m12 = split_model(t1)
m21, m22 = split_model(t2)
m31, m32 = split_model(t3)
def replace_relu(m):
    m.layer3[0].relu = nn.ReLU()
    m.layer3[1].relu = nn.ReLU()
    m.layer4[0].relu = nn.ReLU()
    m.layer4[1].relu = nn.ReLU()
    return m
m12 = replace_relu(m12)
m22 = replace_relu(m22)
m32 = replace_relu(m32)

#%%
d = torch.rand(5,1,28,28)
r1 = t0(d)
r2 = m22(m11(d.cuda()).reshape(5,128,7,7)).detach().cpu()
print((r1-r2).abs().sum())