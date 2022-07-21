#%% loading dependency
import os
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import stft 
from scipy import stats
import itertools

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import torch_optimizer as optim

# from torch.utils.tensorboard import SummaryWriter
"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')

def plot_history(history, indices, names, save_path):
    plt.clf()
    for index, name in zip(indices, names):
        losses = [x[index] for x in history]
        plt.plot(losses, label=name)
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(save_path)


def plot_gate_values(gate_values, student_num):
    # epoch x layer x gate
    for layer_num in range(len(gate_values[0])):
        layer_gates_by_epoch = [x[layer_num] for x in gate_values]
        arr = torch.stack(layer_gates_by_epoch).detach().cpu().numpy()
        plt.clf()
        plt.plot(arr)
        plt.title('layer number {}'.format(layer_num))
        plt.savefig('students/gates{}_{}.png'.format(student_num, layer_num))


