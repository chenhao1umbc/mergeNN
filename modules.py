import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
from itertools import chain
import types

from utils import *
import math

"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')

#%%
class L0GateLayer(nn.Module):
    def __init__(self, n_channels, loc_mean=1, loc_sd=0.01, beta=2 / 3, \
            gamma=-0.1, zeta=1.1, fix_temp=True, _3d=False):
        super(L0GateLayer, self).__init__()
        self.n_channels = n_channels
        self.loc = nn.Parameter(torch.zeros(n_channels).normal_(loc_mean, loc_sd))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros((1, n_channels)))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        self._3d = _3d

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.mask(x)
        if not self._3d :
            x = x.permute(2, 3, 0, 1)
        else:
            x = x.permute(2, 3, 4, 0, 1)
        x = x * mask
        if not self._3d:
            x = x.permute(2, 3, 0, 1)
        else:
            x = x.permute(3,4, 0, 1,2)
        return x

    def mask(self, x):
        batch_size = x.shape[0]
        if batch_size > self.uniform.shape[0]:
            self.uniform = torch.zeros((batch_size, self.n_channels), device=self.uniform.device)
        self.uniform.uniform_()
        u = self.uniform[:batch_size]
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
        s = s * (self.zeta - self.gamma) + self.gamma
        return hard_sigmoid(s)

    def l0_loss(self):
        mean_probability_of_nonzero_gate = torch.mean(torch.sigmoid(self.loc - self.gamma_zeta_ratio * self.temp))
        loss = torch.square(mean_probability_of_nonzero_gate - 0.5)
        return loss

    def importance_of_features(self):
        s = torch.sigmoid(self.loc * (self.zeta - self.gamma) + self.gamma)
        return hard_sigmoid(s)

    def important_indices(self):
        importance = self.importance_of_features().detach()
        important_indices = torch.argsort(importance, descending=True)[:len(importance) // 2]
        return important_indices


class Connect_1stconv(nn.Module):
    def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d, _3d=False):
        super().__init__()
        """first conv is the conv at the beginning of resnet"""
        ker_size = conv1.kernel_size
        n_stride = conv1.stride
        out_channels = conv1.out_channels
        n_padding = conv1.padding
        if not _3d:
            self.conv = nn.Conv2d(1, out_channels*2, ker_size, n_stride, n_padding, bias=False)
        else:
            self.conv = nn.Conv3d(1, out_channels*2, ker_size, n_stride, n_padding, bias=False)
        self.conv.weight.data[:out_channels] = conv1.weight.data.detach().clone()
        self.conv.weight.data[out_channels:] = conv2.weight.data.detach().clone()
        

class Connect_conv(nn.Module):
    def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d, _3d=False):
        super().__init__()
        """middle conv is conv in layers/blocks"""
        in_channels = conv1.in_channels
        out_channels = conv1.out_channels
        if not _3d:
            self.conv = nn.Conv2d(in_channels*2, out_channels*2, kernel_size=conv1.kernel_size, stride=conv1.stride,
                            padding=conv1.padding, bias=False)
        else:
            self.conv = nn.Conv3d(in_channels*2, out_channels*2, kernel_size=conv1.kernel_size, stride=conv1.stride,
                padding=conv1.padding, bias=False)
        self.conv.weight.data *= 0
        self.conv.weight.data[:out_channels, :in_channels] = conv1.weight.data.detach().clone()
        self.conv.weight.data[out_channels:, in_channels:] = conv2.weight.data.detach().clone()


class Connect_bn(nn.Module):
    def __init__(self, n_channel, bn1, bn2, _3d=False):
        super().__init__()
        if not _3d:
            self.bn = nn.BatchNorm2d(n_channel)
        else:
            self.bn = nn.BatchNorm3d(n_channel)
        self.bn.weight.data[:n_channel//2] = bn1.weight.data.clone()
        self.bn.weight.data[n_channel//2:] = bn2.weight.data.clone()
        self.bn.bias.data[:n_channel//2] = bn1.bias.data.clone()
        self.bn.bias.data[n_channel//2:] = bn2.bias.data.clone()


class Connect_fc(nn.Module):
    def __init__(self, lin1: nn.Linear, lin2: nn.Linear) -> nn.Linear:
        super().__init__()
        in_features = lin1.in_features
        self.fc = nn.Linear(in_features * 2, lin1.out_features)
        self.fc.weight.data *= 0
        self.fc.weight.data[:, :in_features] = lin1.weight.data.detach().clone() / 2
        self.fc.weight.data[:, in_features:] = lin2.weight.data.detach().clone() / 2
        self.fc.bias.data = lin1.bias.data.detach().clone() / 2 + lin2.bias.data.detach().clone() / 2


class Prune_conv(nn.Module):
    def __init__(self, conv, in_index: Tensor, out_index: Tensor, _3d=False):
        super().__init__()
        if not _3d:
            self.conv = nn.Conv2d(len(in_index), len(out_index), conv.kernel_size,
                            stride=conv.stride, padding=conv.padding, bias=False)
        else:
            self.conv = nn.Conv3d(len(in_index), len(out_index), conv.kernel_size,
                            stride=conv.stride, padding=conv.padding, bias=False)    
        self.conv.weight.data = conv.weight.data[out_index][:, in_index].detach().clone()


class Prune_bn(nn.Module):
    def __init__(self, bn, n_channel, indices, _3d=False):
        super().__init__()
        if not _3d:
            self.bn = nn.BatchNorm2d(n_channel)
        else:
            self.bn = nn.BatchNorm3d(n_channel)
        self.bn.weight.data = bn.weight.data[indices]
        self.bn.bias.data = bn.bias.data[indices]


class Layer_l0(nn.Module):
    def __init__(self, cl0_layer, cl1_layer):
        super().__init__()
        self.relu = nn.ReLU()

        n_channel = cl0_layer[0].conv1.out_channels * 2
        self.main_gate = L0GateLayer(n_channel)

        # downsample
        if cl0_layer[0].downsample is not None:
            self.downsample_conv = Connect_conv(cl0_layer[0].downsample[0], \
                                    cl1_layer[0].downsample[0]).conv
            self.downsample_bn = nn.BatchNorm2d(n_channel)
        else:
            self.downsample_conv = None

        # first block
        self.block0 = nn.Sequential(
            Connect_conv(cl0_layer[0].conv1, cl1_layer[0].conv1).conv, #self.conv1
            Connect_bn(n_channel, cl0_layer[0].bn1, cl1_layer[0].bn1).bn, #self.bn1
            nn.ReLU(inplace=True),
            L0GateLayer(n_channel), #self.gate1
            Connect_conv(cl0_layer[0].conv2, cl1_layer[0].conv2).conv, # self.conv2
            Connect_bn(n_channel, cl0_layer[0].bn2, cl1_layer[0].bn2).bn #self.bn2
        )
        self.gate1 = self.block0[3]
        # second block
        self.block1 = nn.Sequential(
            Connect_conv(cl0_layer[1].conv1, cl1_layer[1].conv1).conv, #self.conv3 =
            Connect_bn(n_channel, cl0_layer[1].bn1, cl1_layer[1].bn1).bn, # self.bn3
            nn.ReLU(inplace=True),
            L0GateLayer(n_channel), # self.gate2            
            Connect_conv(cl0_layer[1].conv2, cl1_layer[1].conv2).conv, # self.conv4
            Connect_bn(n_channel, cl0_layer[1].bn2, cl1_layer[1].bn2).bn # self.bn4
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
        x = self.block1(x)
        x += identity
        x = self.main_gate(x, main_mask)

        return x

    def l0_loss(self):
        return self.main_gate.l0_loss() + self.gate1.l0_loss() + self.gate2.l0_loss()

    def compress(self, in_index):
        out_index = self.main_gate.important_indices().detach()
        n_outchannels = len(out_index)

        # downsample
        if self.downsample_conv is not None:
            self.downsample_conv = compress_conv2d(self.downsample_conv, in_index, out_index)
            self.downsample_bn = nn.BatchNorm2d(n_outchannels)

        # first block
        block_index = self.gate1.important_indices()
        self.conv1 = Prune_conv(self.conv1, in_index, block_index).conv
        self.bn1 = Prune_bn(self.bn1, n_outchannels, block_index).bn
        self.conv2 = Prune_conv(self.conv2, block_index, out_index).conv
        self.bn2 = Prune_bn(self.bn2, n_outchannels, out_index).bn

        # second block
        block_index = self.gate2.important_indices()
        self.conv3 = Prune_conv(self.conv3, out_index, block_index).conv
        self.bn3 = Prune_bn(self.bn3, n_outchannels, block_index).bn
        self.conv4 = Prune_conv(self.conv4, block_index, out_index).conv
        self.bn4 = Prune_bn(self.bn4, n_outchannels, out_index).bn

        delattr(self, 'main_gate')
        delattr(self, 'gate1')
        delattr(self, 'gate2')
        self.forward = types.MethodType(new_forward, self)
        return out_index

    def gate_parameters(self):
        return chain(self.main_gate.parameters(), self.gate1.parameters(), self.gate2.parameters())

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(),
                      self.bn1.parameters(),
                      self.conv2.parameters(),
                      self.bn2.parameters(),
                      self.conv3.parameters(),
                      self.bn3.parameters(),
                      self.conv4.parameters(),
                      self.bn4.parameters()]
        if self.downsample_conv is not None:
            parameters += [self.downsample_conv.parameters(),
                           self.downsample_bn.parameters()]

        return chain.from_iterable(parameters)

    def gate_values(self):
        """used only for plots"""
        return [self.main_gate.importance_of_features(), self.gate1.importance_of_features(),
                self.gate2.importance_of_features()]


class Judge(nn.Module):
    def __init__(self, teacher0, teacher1):
        super().__init__()
        self.conv1 = Connect_1stconv(teacher0.conv1, teacher1.conv1).conv
        self.bn1 = Connect_bn(64 * 2, teacher0.bn1, teacher1.bn1).bn
        self.relu = nn.ReLU()

        self.layer1 = Layer_l0(teacher0.layer1, teacher1.layer1)
        self.gate = self.layer1.main_gate
        self.layer2 = Layer_l0(teacher0.layer2, teacher1.layer2)
        self.layer3 = Layer_l0(teacher0.layer3, teacher1.layer3)
        self.layer4 = Layer_l0(teacher0.layer4, teacher1.layer4)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Connect_fc(teacher0.fc, teacher1.fc).fc

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


class BasicBlock2D(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: int = None,

    ):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes,\
            kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, \
            kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Downsample(nn.Module):
    def __init__(self, n_in, n_out) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(n_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    def forward(self, x):
        return self.downsample(x)


class Resnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        #(maxpool): Cifar does not have it.
        self.layer1 = nn.Sequential(
            BasicBlock2D(16, 16),
            BasicBlock2D(16, 16),
            BasicBlock2D(16, 16),
            )
        self.layer2 = nn.Sequential(
            BasicBlock2D(16, 32, stride=2, downsample=Downsample(16, 32)),
            BasicBlock2D(32, 32),
            BasicBlock2D(32, 32)
            )
        self.layer3 = nn.Sequential(
            BasicBlock2D(32, 64, stride=2, downsample=Downsample(32, 64)),
            BasicBlock2D(64, 64),
            BasicBlock2D(64, 64)
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 100)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


#%%
class Layer_l03d(nn.Module):
    def __init__(self, cl0_layer, cl1_layer):
        super().__init__()
        self.relu = nn.ReLU()

        n_channel = cl0_layer[0].conv1.out_channels * 2
        self.main_gate = L0GateLayer(n_channel, _3d=True)

        # downsample
        if cl0_layer[0].downsample is not None:
            self.downsample_conv = Connect_conv(cl0_layer[0].downsample.downsample[0], \
                                    cl1_layer[0].downsample.downsample[0], _3d=True).conv
            self.downsample_bn = nn.BatchNorm3d(n_channel)
        else:
            self.downsample_conv = None

        # first block
        self.block0 = nn.Sequential(
            Connect_conv(cl0_layer[0].conv1, cl1_layer[0].conv1, _3d=True).conv, #self.conv1
            Connect_bn(n_channel, cl0_layer[0].bn1, cl1_layer[0].bn1, _3d=True).bn, #self.bn1
            nn.ReLU(inplace=True),
            L0GateLayer(n_channel, _3d=True), #self.gate1
            Connect_conv(cl0_layer[0].conv2, cl1_layer[0].conv2, _3d=True).conv, # self.conv2
            Connect_bn(n_channel, cl0_layer[0].bn2, cl1_layer[0].bn2, _3d=True).bn #self.bn2
        )
        self.gate1 = self.block0[3]
        # second block
        self.block1 = nn.Sequential(
            Connect_conv(cl0_layer[1].conv1, cl1_layer[1].conv1, _3d=True).conv, #self.conv3 =
            Connect_bn(n_channel, cl0_layer[1].bn1, cl1_layer[1].bn1, _3d=True).bn, # self.bn3
            nn.ReLU(inplace=True),
            L0GateLayer(n_channel, _3d=True), # self.gate2            
            Connect_conv(cl0_layer[1].conv2, cl1_layer[1].conv2, _3d=True).conv, # self.conv4
            Connect_bn(n_channel, cl0_layer[1].bn2, cl1_layer[1].bn2, _3d=True).bn # self.bn4
        )
        self.gate2 = self.block1[3]
        # third block
        self.block2 = nn.Sequential(
            Connect_conv(cl0_layer[1].conv1, cl1_layer[1].conv1, _3d=True).conv, #self.conv5 =
            Connect_bn(n_channel, cl0_layer[1].bn1, cl1_layer[1].bn1, _3d=True).bn, # self.bn5
            nn.ReLU(inplace=True),
            L0GateLayer(n_channel, _3d=True), # self.gate3            
            Connect_conv(cl0_layer[1].conv2, cl1_layer[1].conv2, _3d=True).conv, # self.conv6
            Connect_bn(n_channel, cl0_layer[1].bn2, cl1_layer[1].bn2, _3d=True).bn # self.bn6
        )
        self.gate3 = self.block2[3]

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
        x = self.block1(x)
        x += identity
        x = self.main_gate(x, main_mask)

        # third block
        identity = x
        x = self.block2(x)
        x += identity
        x = self.main_gate(x, main_mask)


        return x

    def l0_loss(self):
        return self.main_gate.l0_loss() + self.gate1.l0_loss() \
            + self.gate2.l0_loss() + self.gate3.l0_loss()

    def compress(self, in_index):
        out_index = self.main_gate.important_indices().detach()
        n_outchannels = len(out_index)

        # downsample
        if self.downsample_conv is not None:
            self.downsample_conv = compress_conv2d(self.downsample_conv, in_index, out_index)
            self.downsample_bn = nn.BatchNorm2d(n_outchannels)

        # first block
        block_index = self.gate1.important_indices()
        self.conv1 = Prune_conv(self.conv1, in_index, block_index).conv
        self.bn1 = Prune_bn(self.bn1, n_outchannels, block_index).bn
        self.conv2 = Prune_conv(self.conv2, block_index, out_index).conv
        self.bn2 = Prune_bn(self.bn2, n_outchannels, out_index).bn

        # second block
        block_index = self.gate2.important_indices()
        self.conv3 = Prune_conv(self.conv3, out_index, block_index).conv
        self.bn3 = Prune_bn(self.bn3, n_outchannels, block_index).bn
        self.conv4 = Prune_conv(self.conv4, block_index, out_index).conv
        self.bn4 = Prune_bn(self.bn4, n_outchannels, out_index).bn

        delattr(self, 'main_gate')
        delattr(self, 'gate1')
        delattr(self, 'gate2')
        self.forward = types.MethodType(new_forward, self)
        return out_index

    def gate_parameters(self):
        return chain(self.main_gate.parameters(), self.gate1.parameters(), self.gate2.parameters())

    def non_gate_parameters(self):
        parameters = [self.conv1.parameters(),
                      self.bn1.parameters(),
                      self.conv2.parameters(),
                      self.bn2.parameters(),
                      self.conv3.parameters(),
                      self.bn3.parameters(),
                      self.conv4.parameters(),
                      self.bn4.parameters()]
        if self.downsample_conv is not None:
            parameters += [self.downsample_conv.parameters(),
                           self.downsample_bn.parameters()]

        return chain.from_iterable(parameters)

    def gate_values(self):
        """used only for plots"""
        return [self.main_gate.importance_of_features(), self.gate1.importance_of_features(),
                self.gate2.importance_of_features()]


class BasicBlock3D(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: int = None,
    ):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv3d(inplanes, planes,\
            kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, \
            kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Downsample3D(nn.Module):
    def __init__(self, n_in, n_out) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(n_in, n_out, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(n_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    def forward(self, x):
        return self.downsample(x)


class Resnet3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        #(maxpool): Cifar does not have it.
        self.layer1 = nn.Sequential(
            BasicBlock3D(16, 16),
            BasicBlock3D(16, 16),
            BasicBlock3D(16, 16),
            )
        self.layer2 = nn.Sequential(
            BasicBlock3D(16, 32, stride=2, downsample=Downsample3D(16, 32)),
            BasicBlock3D(32, 32),
            BasicBlock3D(32, 32)
            )
        self.layer3 = nn.Sequential(
            BasicBlock3D(32, 64, stride=2, downsample=Downsample3D(32, 64)),
            BasicBlock3D(64, 64),
            BasicBlock3D(64, 64)
            )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, 100)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Judge3D(nn.Module):
    def __init__(self, client0, client1):
        super().__init__()
        self.conv1 = Connect_1stconv(client0.conv1, client1.conv1, _3d=True).conv
        self.bn1 = Connect_bn(16 * 2, client0.bn1, client1.bn1, _3d=True).bn
        self.relu = nn.ReLU()

        self.layer1 = Layer_l03d(client0.layer1, client1.layer1)
        self.gate = self.layer1.main_gate
        self.layer2 = Layer_l03d(client0.layer2, client1.layer2)
        self.layer3 = Layer_l03d(client0.layer3, client1.layer3)
        self.layers = [self.layer1, self.layer2, self.layer3]

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = Connect_fc(client0.fc, client1.fc).fc

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

