#%%
"This code is made to perform model level merge of the pretrained transformers"
import math
from typing import Tuple
from utils import *
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from src import dataloader, models
import time, datetime

prep_data = False
if prep_data:
    from src import prep_esc50

#%%
class Arg():
    def __init__(self) -> None:
        fold = 1
        self.model = 'ast'
        self.dataset = 'esc50'
        self.imagenet_pretrain = True
        self.audioset_pretrain = True
        self.data_train =  f'./data/datafiles/esc_train_data_{fold}.json'
        self.data_val =  f'./data/datafiles/esc_eval_data_{fold}.json'
        self.label_csv  =  f'./src/esc_class_labels_indices.csv'
        self.bal = 'none'
        self.freqm = 24
        self.timem = 96
        self.mixup = 0
        self.n_epochs = 3
        self.batch_size = 16
        self.fstride = 10
        self.tstride = 10

        self.dataset_mean = -6.6268077
        self.dataset_std = 5.358466
        self.audio_length = 512
        self.noise = False

        self.metrics = 'acc'
        self.loss = 'CE'
        self.warmup = False
        self.lrscheduler_start = 5
        self.lrscheduler_step = 1
        self.lrscheduler_decay = 0.85
        self.n_class = 50
        self.lr = 1e-5

args = Arg()

audio_conf = {'num_mel_bins': 128, 
                'target_length': args.audio_length, 
                'freqm': args.freqm, 
                'timem': args.timem, 
                'mixup': args.mixup, 
                'dataset': args.dataset, 
                'mode':'train', 'mean':args.dataset_mean, 
                'std':args.dataset_std,
                'noise':args.noise}

val_audio_conf = {'num_mel_bins': 128, 
            'target_length': args.audio_length, 
            'freqm': 0, 
            'timem': 0, 
            'mixup': 0, 
            'dataset': args.dataset, 
            'mode':'evaluation', 
            'mean':args.dataset_mean, 
            'std':args.dataset_std, 
            'noise':False}

if prep_data:
    train_loader = Data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    val_loader = Data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, pin_memory=True)
    
    d, l = [], [] # doing the second wrap is because audo processing is slow, reading raw data and processing
    # is redundant process. So the second wrap will significantly speed up the training and validation process
    for i, (audio_input, labels) in enumerate(train_loader):
        d.append(audio_input)
        l.append(labels)
    tr_d, tr_l = torch.cat(d), torch.cat(l)
    torch.save([tr_d, tr_l], 'src/tr_data_label_ESC50.pt')
    d, l = [], []
    for i, (audio_input, labels) in enumerate(val_loader):
        d.append(audio_input)
        l.append(labels)
    val_d, val_l = torch.cat(d), torch.cat(l)
    torch.save([tr_d, tr_l], 'src/val_data_label_ESC50.pt')

xtr, ytr = torch.load('src/tr_data_label_ESC50.pt')
xval, yval = torch.load('src/tr_data_label_ESC50.pt')
data = Data.TensorDataset(xtr, ytr)
tr = Data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
data = Data.TensorDataset(xval, yval)
val = Data.DataLoader(data, batch_size=args.batch_size*2, shuffle=True)

model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, 
                    input_fdim=128,input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                    audioset_pretrain=args.audioset_pretrain, model_size='base384')


#%% evaluate each of the model
names = [f'./data/transformers/tranf_model_{i}_{j}.pt' for i in range(1,4) for j in range(600, 2401, 600)] 
the10 = names[2:]
res = []
models = []
for name in the10:
    model = torch.load(name)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    models.append(model)
    test_loss = evaluate(model, test_data)
    test_ppl = math.exp(test_loss)
    res.append(test_ppl)

#%% merging the models
oc_sd=0.01
lamb = 5
loga = (torch.randn(10)*oc_sd).cuda().requires_grad_()
print('initial loga', loga)
def hard_concrete(loga, batch_size=128):
    beta, gamma, zeta, eps = 2/3, -0.1, 1.1, 1e-20
    u = torch.rand(batch_size, loga.shape[0], device=loga.device)
    s = torch.sigmoid((torch.log(u+eps) - torch.log(1 - u+eps) + loga) / beta)
    sbar = s * (zeta - gamma) + gamma
    z = hard_sigmoid(sbar)
    return z

optimizer = torch.optim.RAdam([loga],
                lr= 1e-3,
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)

