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
plt.rcParams['figure.dpi'] = 150
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
file1 = open('./modulelevel3_log.out', 'r')
lines = file1.readlines()
start = len('after validation tensor(')
res = {i:[] for i in range(6)}
for ii, line in enumerate(lines):
    if line[:start] == 'after validation tensor(':
        data = line[start:].strip()
        exec('a='+data[:-1])
        for i in range(6):
            res[i].append(a[i])

plt.figure(figsize=(4,3), dpi=300)
marker = ['-x', '-o', '-^', '-d', '-v', '->']
for i in range(6):
    plt.plot(res[i], marker[i], markevery=20)
plt.legend([f'log alpha_{i}' for i in range(1,7)],fontsize=10)
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Value',fontsize=12)
plt.tight_layout() # otherwise the text will be chopped off
plt.savefig('module_loga.png')

#%% 10 model performance barchart
h = [0.1606, 0.1744, 0.5225, 0.5387, 0.5587, 0.8438, 0.8456, 0.8506, 0.9413, 0.9581, 0.9581]
plt.figure(figsize=(10,5))
plt.bar([str(i) for i in range(1, 11)], h[:10])
plt.bar('Merged', h[10], color='#ff7f0e')
plt.xlabel('Model index')
plt.ylabel('Accuracy')
plt.savefig('bar.png')

#%% vae example plots, attach the code to the end of the traning_vae file
model.load_state_dict(torch.load('saves/model_vae_K2.pt'))
from PIL import Image, ImageColor, ImageDraw, ImageFont
model.eval()
test_losses = torch.zeros(3)
epoch = 1
beta = min(1.0,(epoch)/min(args.epochs,args.warm_up)) * args.beta_max
with torch.no_grad():
    for i, (data,_) in enumerate(test_loader):
        data = mix_data(data.to(device))[0].view(-1,dimx)

        recon_y, mu_z, logvar_z, recons = model(data)
        loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)

        test_losses[0] += loss.item()
        test_losses[1] += ELL.item()
        test_losses[2] += KLD.item()
        break

    n = min(data.size(0), 8)
    ncols = 4
    comparison = torch.zeros(n*ncols,1,28,28)
    comparison[:n] = data.view(data.size(0), 1, 28, 28)[:n]
    comparison[n:2*n] = recon_y.view(data.size(0), 1, 28, 28)[:n]
    comparison[2*n:3*n] = recons[:,0].view(data.size(0), 1, 28, 28)[:n]
    comparison[3*n:4*n] = recons[:,1].view(data.size(0), 1, 28, 28)[:n]

    grid = make_grid(comparison,nrow=n)
    
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save('vae_examples.png')