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

#%% plot pdf
def plot_pdf(loga):
    beta, gamma, zeta, eps = torch.tensor([0.5, -0.1, 1.1, 1e-20])
    u = torch.arange(0,1,1e-3)+eps
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + loga) / beta)
    sbar = s * (zeta - gamma) + gamma
    z = hard_sigmoid(sbar)

    "Concrete pdf and cdf"
    def pdf_qs(x):
        qs = beta*loga.exp()*x**(-beta-1)*(1-x)**(-beta-1)\
        /(loga.exp()*x**(-beta)+(1-x)**(-beta))**2
        return qs
    def cdf_Qs(x):
        Qs = ((x.log()-(1-x).log())*beta-loga).sigmoid()
        return Qs
    qs = pdf_qs(s)
    Qs = cdf_Qs(s)

    "Hard concrete pdf and cdf"
    def pdf_qs_bar(x):
        return 1/(zeta-gamma).abs()*pdf_qs((x-gamma)/(zeta-gamma))

    def cdf_Qs_bar(x):
        return cdf_Qs((x-gamma)/(zeta-gamma))

    qz = z.clone()
    qz[z==0] = cdf_Qs_bar(0)
    qz[z==1] = (1-cdf_Qs_bar(1))
    idx = torch.logical_or(z==0, z==1)
    qz[~idx] = (cdf_Qs_bar(1)-cdf_Qs_bar(0)) * pdf_qs_bar(z[~idx])
    "l0 norm paper fig.2 was wrong as below, since it did not multiply the coeff."
    # qz[~idx] = pdf_qs_bar(z[~idx]) 

    # s0 = max((z==0).sum()/z.shape[0]*20, 1)
    # s1 = max((z==1).sum()/z.shape[0]*20, 1)
    s0 = 5
    s1 = 5
    plt.plot(s, qs, '--', color="#1f77b4") 
    plt.plot(sbar, pdf_qs_bar(sbar), '-.', color="#2ca02c")
    plt.plot(z[~idx], qz[~idx], color="#ff7f0e")
    plt.plot(1,  1-cdf_Qs_bar(1), 'o', markersize=s1, color="#ff7f0e")
    plt.plot(0, cdf_Qs_bar(0), 'o', markersize=s0, color="#ff7f0e")
    plt.grid()
    plt.ylim([0,2.5])
    plt.xlabel('g', fontsize=16)
    plt.ylabel('p(g)', fontsize=16)
    plt.legend(['Concrete', 'Streched concrete', 'Hard concrete', \
        'Hard concrete p(g=0) and p(g=1) '],prop={'size': 14}) #g is z
    plt.savefig('loga_n3.png')
    plt.show()

loga = torch.tensor(-3)
plot_pdf(loga)

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