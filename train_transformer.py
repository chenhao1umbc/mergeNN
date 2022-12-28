#%%
"Theses code is based on AST github https://github.com/YuanGongND/ast"
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from src import dataloader

prep_data = True
if prep_data:
    from src import prep_esc50

#%%

train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                audioset_pretrain=args.audioset_pretrain, model_size='base384')