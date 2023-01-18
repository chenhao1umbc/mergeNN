#%%
"Theses code is based on AST github https://github.com/YuanGongND/ast"
import math, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.utils.data as Data

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
        self.audioset_pretrain = True
        self.imagenet_pretrain = self.audioset_pretrain or True
        self.data_train =  f'./data/datafiles/esc_train_data_{fold}.json'
        self.data_val =  f'./data/datafiles/esc_eval_data_{fold}.json'
        self.label_csv  =  f'./src/esc_class_labels_indices.csv'
        self.bal = 'none'
        self.freqm = 24
        self.timem = 96
        self.mixup = 0
        self.n_epochs = 10
        self.batch_size = 16
        self.fstride = 10
        self.tstride = 10

        self.dataset_mean = -6.6268077
        self.dataset_std = 5.358466
        self.audio_length = 512
        self.noise = False

        self.metrics = 'acc'
        self.loss = 'CE'
        self.loss_fn = nn.CrossEntropyLoss()
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

def validate(audio_model, val_loader, args):
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
        loss_mean = torch.tensor(A_loss).mean().item()
        acc = get_acc(target, audio_output)

    return acc, loss_mean

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

    args.loss_fn = loss_fn
    scaler = GradScaler()
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    model.train()
    while epoch < args.n_epochs + 1:
        model.train()
        print('---------------')
        print(datetime.datetime.now())

        for i, (audio_input, labels) in enumerate(train_loader):
            model.train()
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
            acc, loss = validate(model, test_loader, args)
            print("acc, loss, iter", acc, loss, i)
            torch.save(model,f'ESC{i}_mid.pt')
            model.train()
        print(stop)
        print('start validation')
        acc, loss = validate(model, test_loader, args)
        print("acc, loss, epoch", acc, loss, epoch)
        
        epoch += 1 
    print('done')

#%%
train(model, tr, val, args)
# validate(model, val, args)