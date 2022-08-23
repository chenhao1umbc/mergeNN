#%%
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from modules import *

from utils import *
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import sys

teacher0 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
teacher0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
teacher0.load_state_dict(torch.load(f'teachers/teacher{11}.pt'))

teacher1 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
teacher1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
teacher1.load_state_dict(torch.load(f'teachers/teacher{10}.pt'))

#%%
def connect_first_conv2d(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    """first conv is the conv at the beginning of resnet (before layers) that takes 3 channels"""
    ker_size = conv1.kernel_size
    n_stride = conv1.stride
    out_channels = conv1.out_channels
    n_padding = conv1.padding
    conv = nn.Conv2d(1, out_channels*2, ker_size, n_stride, n_padding, bias=False)
    conv.weight.data[:out_channels] = conv1.weight.data.detach().clone()
    conv.weight.data[out_channels:] = conv2.weight.data.detach().clone()
    return conv

def connect_bn(planes, bn1, bn2):
    out = nn.BatchNorm2d(planes)
    out.weight.data[:planes//2] = bn1.weight.data.clone()
    out.weight.data[planes//2:] = bn2.weight.data.clone()
    out.bias.data[:planes//2] = bn1.bias.data.clone()
    out.bias.data[planes//2:] = bn2.bias.data.clone()
    return out


class Student(nn.Module):
    def __init__(self, teacher0, teacher1):
        super(self).__init__()
        self.conv1 = connect_first_conv2d(teacher0.conv1, teacher1.conv1)
        self.bn1 = connect_bn(64 * 2, teacher0.bn1, teacher1.bn1)
        self.relu = nn.ReLU()

        self.layer1 = L0Layer(teacher0.layer1[0], teacher1.layer1[0])
        self.gate = self.layer1.main_gate
        self.layer2 = L0Layer(teacher0.layer2, teacher1.layer2)
        self.layer3 = L0Layer(teacher0.layer3, teacher1.layer3)

        self.layers = [self.layer1, self.layer2, self.layer3]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = connect_final_linear(teacher0.fc, teacher1.fc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        mask = self.gate.mask(x)
        x = self.gate(x, mask)

        x = self.layers[0](x, mask)
        for layer in self.layers[1:]:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def l0_loss(self):
        l0_loss = 0#self.gate.l0_loss()
        for layer in self.layers:
            l0_loss += layer.l0_loss()
        return l0_loss

    def compress(self):
        importance_indices = self.gate.important_indices()
        self.conv1 = compress_conv2d(self.conv1, torch.ones(3, dtype=torch.bool), importance_indices)
        self.bn1 = compress_bn(self.bn1, 16, importance_indices)
        for layer in self.layers:
            importance_indices = layer.compress(importance_indices)
        self.fc = compress_final_linear(self.fc, importance_indices)

        delattr(self, 'gate')
        self.forward = types.MethodType(new_forward, self)

    def gate_parameters(self):
        parameters = []# [self.gate.parameters()]
        for layer in self.layers:
            parameters.append(layer.gate_parameters())
        return chain.from_iterable(parameters)

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(), self.bn1.parameters(), self.fc.parameters()]
        for layer in self.layers:
            parameters.append(layer.non_gate_parameters())
        return chain.from_iterable(parameters)

    def gate_values(self):
        values = []#[self.gate.importance_of_features()]
        for layer in self.layers:
            values += layer.gate_values()

        return values

student = Student(teacher0, teacher1)