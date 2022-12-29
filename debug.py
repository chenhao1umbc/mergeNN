#%%
"Theses code is based on AST github https://github.com/YuanGongND/ast"
import math
from typing import Tuple

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
        self.batch_size = 48
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


train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
    batch_size=args.batch_size, shuffle=True, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size*2, shuffle=False, pin_memory=True)

audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                audioset_pretrain=args.audioset_pretrain, model_size='base384')
#%%
from torch.cuda.amp import autocast,GradScaler
# def train(audio_model, train_loader, test_loader, args):
if True:
    device = 'cuda'
    audio_model = audio_model.cuda()
    
    # Set up the optimizer
    global_step, epoch = 0, 1
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # dataset specific settings
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    warmup = args.warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
        list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
        gamma=args.lrscheduler_decay)
    args.loss_fn = loss_fn
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    # for amp
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = torch.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                audio_output = audio_model(audio_input)
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss


        print('start validation')
        # stats, valid_loss = validate(audio_model, test_loader, args, epoch)

# train(audio_model, train_loader, val_loader, args)