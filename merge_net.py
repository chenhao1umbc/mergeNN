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

#%%
teacher0 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
teacher0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
teacher0.load_state_dict(torch.load(f'teachers/teacher{11}.pt'))

teacher1 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
teacher1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
teacher1.load_state_dict(torch.load(f'teachers/teacher{10}.pt'))

#%%
class Connect_conv(nn.Module):
    def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
        super().__init__()
        """middle conv is conv in layers/blocks"""
        in_channels = conv1.in_channels
        out_channels = conv1.out_channels
        self.conv = nn.Conv2d(in_channels*2, out_channels*2, kernel_size=conv1.kernel_size, stride=conv1.stride,
                        padding=conv1.padding, bias=False)
        self.conv.weight.data *= 0
        self.conv.weight.data[:out_channels, :in_channels] = conv1.weight.data.detach().clone()
        self.conv.weight.data[out_channels:, in_channels:] = conv2.weight.data.detach().clone()
    def forward(self, x):
        return self.conv

class Connect_bn(nn.Module):
    def __init__(self, n_channel, bn1, bn2):
        super().__init__()
        self.out = nn.BatchNorm2d(n_channel)
        self.out.weight.data[:n_channel//2] = bn1.weight.data.clone()
        self.out.weight.data[n_channel//2:] = bn2.weight.data.clone()
        self.out.bias.data[:n_channel//2] = bn1.bias.data.clone()
        self.out.bias.data[n_channel//2:] = bn2.bias.data.clone()
    def forward(self, x):    
        return self.out

class Layer_l0(nn.Module):
    def __init__(self, tch0_layer, tch1_layer):
        super().__init__()
        self.relu = nn.ReLU()

        n_channel = tch0_layer.conv1.out_channels * 2
        self.main_gate = L0GateLayer2d(n_channel)

        # downsample
        if tch0_layer.downsample_conv is not None:
            self.downsample_conv = connect_middle_conv(tch0_layer.downsample_conv, tch1_layer.downsample_conv)
            self.downsample_bn = nn.BatchNorm2d(n_channel)
        else:
            self.downsample_conv = None

        # first block
        self.block0 = nn.Sequential(
            Connect_conv(tch0_layer[0].conv1, tch1_layer[0].conv1), #self.conv1
            Connect_bn(n_channel, tch0_layer[0].bn1, tch1_layer[0].bn1), #self.bn1
            nn.ReLU(inplace=True),
            L0GateLayer2d(n_channel), #self.gate1
            Connect_conv(tch0_layer[0].conv2, tch1_layer[0].conv2), # self.conv2
            Connect_bn(n_channel, tch0_layer[0].bn2, tch1_layer[0].bn2) #self.bn2
        )
        self.gate1 = self.block0[3]
        # second block
        self.block1 = nn.Sequential(
            Connect_conv(tch0_layer[1].conv1, tch1_layer[1].conv1), #self.conv3 =
            Connect_bn(n_channel, tch0_layer[1].bn1, tch1_layer[1].bn1), # self.bn3
            nn.ReLU(inplace=True),
            L0GateLayer2d(n_channel), # self.gate2            
            Connect_conv(tch0_layer[1].conv2, tch1_layer[1].conv2), # self.conv4
            Connect_bn(n_channel, tch0_layer[1].bn4, tch1_layer[1].bn2) # self.bn4
        )
        self.gate2 = self.block1[3]

    def forward(self, x, main_mask=None):
        if main_mask is None:
            main_mask = self.main_gate.mask(x)

        # downsample
        identity = x
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        # first block
        x = self.block0(x)
        x += identity
        x = self.main_gate(x, main_mask)

        # second block
        identity = x
        x = self.block0(x)
        x += identity
        x = self.main_gate(x, main_mask)

        return x

    def l0_loss(self):
        return self.main_gate.l0_loss() + self.gate1.l0_loss() + self.gate2.l0_loss()

    def compress(self, in_importance_indices):
        out_importance_indices = self.main_gate.important_indices().detach()
        planes = len(out_importance_indices)

        # downsample
        if self.downsample_conv is not None:
            self.downsample_conv = compress_conv2d(self.downsample_conv, in_importance_indices, out_importance_indices)
            self.downsample_bn = nn.BatchNorm2d(planes)

        # first block
        important_indices_in_block = self.gate1.important_indices()
        self.conv1 = compress_conv2d(self.conv1, in_importance_indices, important_indices_in_block)
        self.bn1 = compress_bn(self.bn1, planes, important_indices_in_block)
        self.conv2 = compress_conv2d(self.conv2, important_indices_in_block, out_importance_indices)
        self.bn2 = compress_bn(self.bn2, planes, out_importance_indices)

        # second block
        important_indices_in_block = self.gate2.important_indices()
        self.conv3 = compress_conv2d(self.conv3, out_importance_indices, important_indices_in_block)
        self.bn3 = compress_bn(self.bn3, planes, important_indices_in_block)
        self.conv4 = compress_conv2d(self.conv4, important_indices_in_block, out_importance_indices)
        self.bn4 = compress_bn(self.bn4, planes, out_importance_indices)

        important_indices_in_block = self.gate3.important_indices()
        self.conv5 = compress_conv2d(self.conv5, out_importance_indices, important_indices_in_block)
        self.bn5 = compress_bn(self.bn5, planes, important_indices_in_block)
        self.conv6 = compress_conv2d(self.conv6, important_indices_in_block, out_importance_indices)
        self.bn6 = compress_bn(self.bn6, planes, out_importance_indices)

        delattr(self, 'main_gate')
        delattr(self, 'gate1')
        delattr(self, 'gate2')
        delattr(self, 'gate3')
        self.forward = types.MethodType(new_forward, self)
        return out_importance_indices

    def gate_parameters(self):
        return chain(self.main_gate.parameters(), self.gate1.parameters(), self.gate2.parameters(), self.gate3.parameters())

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(),
                      self.bn1.parameters(),
                      self.conv2.parameters(),
                      self.bn2.parameters(),
                      self.conv3.parameters(),
                      self.bn3.parameters(),
                      self.conv4.parameters(),
                      self.bn4.parameters(),
                      self.conv5.parameters(),
                      self.bn5.parameters(),
                      self.conv6.parameters(),
                      self.bn6.parameters()]
        if self.downsample_conv is not None:
            parameters += [self.downsample_conv.parameters(),
                           self.downsample_bn.parameters()]

        return chain.from_iterable(parameters)

    def gate_values(self):
        """used only for plots"""
        return [self.main_gate.importance_of_features(), self.gate1.importance_of_features(),
                self.gate2.importance_of_features()]

class Student(nn.Module):
    def __init__(self, teacher0, teacher1):
        super().__init__()
        self.conv1 = connect_first_conv2d(teacher0.conv1, teacher1.conv1)
        self.bn1 = Connect_bn(64 * 2, teacher0.bn1, teacher1.bn1)
        self.relu = nn.ReLU()

        self.layer1 = Layer_l0(teacher0.layer1, teacher1.layer1)
        self.gate = self.layer1.main_gate
        self.layer2 = Layer_l0(teacher0.layer2, teacher1.layer2)
        self.layer3 = Layer_l0(teacher0.layer3, teacher1.layer3)

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
#%%
