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


train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
    batch_size=args.batch_size, shuffle=True, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size*2, shuffle=False, pin_memory=True)

model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, 
                    input_fdim=128,input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                    audioset_pretrain=args.audioset_pretrain, model_size='base384')
#%%
from torch.cuda.amp import autocast,GradScaler
def get_acc(x, xhat):
    """x shape of [batch_size, n_class], xhat has the same shape as x
    x is one-hot encoding

    Args:
        x (matrix): ground truth
        xhat (matrix): predictions
    """
    _, ind1 = x.max(dim=1)
    _, ind2 = xhat.max(dim=1)
    res = ind1 - ind2
    return (res == 0).sum()/res.numel()

def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = torch.tensor(A_loss).mean().item
        acc = get_acc(target, audio_output)

    return acc, loss

def train(model, train_loader, test_loader, args):
    device = 'cuda'
    model = model.cuda()
    
    # Set up the optimizer
    global_step, epoch = 0, 1
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # dataset specific settings
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()

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

    model.train()
    while epoch < args.n_epochs + 1:
        model.train()
        print('---------------')
        print(datetime.datetime.now())

        for i, (audio_input, labels) in enumerate(train_loader):
            print(f"current #iter={i}")
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                audio_output = model(audio_input)
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print('start validation')
        acc, loss = validate(model, test_loader, args, epoch)
        print("acc, loss, epoch", acc, loss, epoch)
    torch.save(model,'ESC.pt')
    print('done')

train(model, train_loader, val_loader, args)

#%%
validate(model, val_loader, args, 0)