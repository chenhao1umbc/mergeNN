#%%
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from src.model import *
from src.argparser import *
from src.utils import *
from src.dataloader import *

#%%
class Arg():
    def __init__(self) -> None:
        pass
args = Arg()
args.seed = 1
args.num_workers = 4
args.data_directory = 'data'
args.batch_size = 64
args.dimz = 20
args.sources = 2 # number of sources
args.variational = True
args.learning_rate = 2e-4
args.decay = 0.9998
args.epochs = 10
args.save_interval = 1
args.prior = 0.5
args.scale = 'gauss'
args.warm_up = 50
args.beta_max = 0.5
args.log_interval = 10

#%%
class Container(nn.Module):
    def __init__(self, dimx=28*28, dimz=20, n_sources=2, device='cuda',variational=True):
        super().__init__()
        oc_sd = 0.01
        n_layer2merge = 2
        n_models = 2
        self.loga = nn.Parameter((torch.randn(n_models*n_layer2merge)*oc_sd))

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.device = device
        self.variational = variational

        chans = (700, 600, 500, 400, 300)

        self.out_z = nn.Linear(chans[-1],2*self.n_sources*self.dimz).requires_grad_(False),
        self.out_w = nn.Linear(chans[-1],self.n_sources).requires_grad_(False),

        self.Encoder = nn.Sequential(
            LinearBlock(self.dimx, chans[0]).requires_grad_(False),
            LinearBlock(chans[0],chans[1]).requires_grad_(False),
            LinearBlock(chans[1],chans[2]).requires_grad_(False),
            LinearBlock(chans[2],chans[3]).requires_grad_(False),
            LinearBlock(chans[3],chans[4]).requires_grad_(False),
            )

        self.Decoder = nn.Sequential(
            LinearBlock(self.dimz, chans[4]).requires_grad_(False),
            LinearBlock(chans[4],chans[3]).requires_grad_(False),
            LinearBlock(chans[3],chans[2]).requires_grad_(False),
            LinearBlock(chans[2],chans[1]).requires_grad_(False),
            LinearBlock(chans[1],chans[0]).requires_grad_(False),
            LinearBlock(chans[0],self.dimx,activation=False).requires_grad_(False),
            )
        
        self.alt_layer1 = nn.Linear(chans[0],chans[1]).requires_grad_(False)
        self.alt_layer2 = nn.Linear(chans[3],chans[2]).requires_grad_(False)

    def hard_concrete(self, loga, batch_size=128):
        beta = 0.1
        gamma, zeta, eps = -0.1, 1.1, 1e-20
        u = torch.rand(batch_size, loga.shape[0], device=loga.device)
        s = torch.sigmoid((torch.log(u+eps) - torch.log(1 - u+eps) + loga) / beta)
        sbar = s * (zeta - gamma) + gamma
        z = self.hard_sigmoid(sbar)
        return z

    def hard_sigmoid(self, x):
        return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

    def encode(self, x):
        # d = self.Encoder[1].block[0]
        d = self.Encoder(x)
        dz = self.out_z(d)
        mu = dz[:,::2]
        logvar = dz[:,1::2]
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        d = self.Decoder(z.view(-1,self.dimz))
        recon_separate = torch.sigmoid(d).view(-1,self.n_sources,self.dimx)
        recon_x = recon_separate.sum(1) 
        return recon_x, recon_separate

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dimx))
        z = self.reparameterize(mu, logvar)
        recon_x, recons = self.decode(z)

        return recon_x, mu, logvar, recons

#%%
model_secondary = VAE(dimx=28*28,dimz=20,n_sources=2).cuda()
model_secondary.load_state_dict(torch.load('saves/pretrained/model_vae_K2.pt'))
for p in model_secondary.parameters():
    p.requires_grad_(False)
model = Container().cuda()
model.load_state_dict(torch.load('saves/model_vae_K2.pt'), strict=False)
model.alt_layer1.weight = model_secondary.Encoder[1].block[0].weight
model.alt_layer1.bias = model_secondary.Encoder[1].block[0].bias
model.alt_layer2.weight = model_secondary.Decoder[2].block[0].weight
model.alt_layer2.bias = model_secondary.Decoder[2].block[0].bias

#%%
device = 'cuda'
dimx = 28*28
kwargs = {'num_workers': 4, 'pin_memory': True} if True else {}
test_losses = torch.zeros(3)
train_loader, test_loader = get_data_loaders('data', 64, kwargs)
loss_function = Loss(sources=args.sources,likelihood='laplace',variational=args.variational,prior=args.prior,scale=args.scale)

epoch = 1
beta = min(1.0,(epoch)/min(args.epochs,args.warm_up)) * args.beta_max

#%%
for i, (data,_) in enumerate(test_loader):
    data = mix_data(data.to(device))[0].view(-1,dimx)

    recon_y, mu_z, logvar_z, recons = model(data)
    loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)

    test_losses[0] += loss.item()
    test_losses[1] += ELL.item()
    test_losses[2] += KLD.item()
    break
