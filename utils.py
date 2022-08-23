#%% loading dependency
import os
import h5py 
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import torch_optimizer as optim
Tensor = torch.Tensor

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


def compress_bn(bn, planes, indices):
    out = nn.BatchNorm2d(planes)
    out.weight.data = bn.weight.data[indices]
    out.bias.data = bn.bias.data[indices]
    return out


def compress_conv2d(conv: nn.Conv2d, in_importance_indices: Tensor, out_important_indices: Tensor) -> nn.Conv2d:
    compressed = nn.Conv2d(len(in_importance_indices), len(out_important_indices), conv.kernel_size,
                           stride=conv.stride, padding=conv.padding, bias=False)
    compressed.weight.data = conv.weight.data[out_important_indices][:, in_importance_indices].detach().clone()
    return compressed


def connect_final_linear(lin1: nn.Linear, lin2: nn.Linear) -> nn.Linear:
    in_features = lin1.in_features
    lin = nn.Linear(in_features * 2, lin1.out_features)
    lin.weight.data *= 0
    lin.weight.data[:, :in_features] = lin1.weight.data.detach().clone() / 2
    lin.weight.data[:, in_features:] = lin2.weight.data.detach().clone() / 2
    lin.bias.data = lin1.bias.data.detach().clone() / 2 + lin2.bias.data.detach().clone() / 2
    return lin


def compress_final_linear(layer: nn.Linear, in_important_indices: Tensor) -> nn.Linear:
    compressed = nn.Linear(len(in_important_indices), layer.out_features)
    compressed.weight.data = layer.weight.data[:, in_important_indices].detach().clone()
    compressed.bias.data = layer.bias.data.detach().clone()
    return compressed


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))



def connect_middle_conv(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    """middle conv is conv in layers/blocks"""
    in_channels = conv1.in_channels
    out_channels = conv1.out_channels
    conv = nn.Conv2d(in_channels*2, out_channels*2, kernel_size=conv1.kernel_size, stride=conv1.stride,
                     padding=conv1.padding, bias=False)
    conv.weight.data *= 0
    conv.weight.data[:out_channels, :in_channels] = conv1.weight.data.detach().clone()
    conv.weight.data[out_channels:, in_channels:] = conv2.weight.data.detach().clone()
    return conv


