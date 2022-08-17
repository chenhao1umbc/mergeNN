import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import torch_optimizer as optim

from utils import *
import math

"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')

#%%
class Conv3x3(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, \
            padding=1, groups=1, bias=False, dilation=1)
    
    def forward(self, x):
        x = self.layer(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.layer = nn.Conv2d(in_planes, out_planes, kernel_size=1,\
             stride=stride, bias=False)
    
    def forward(self, x):
        x = self.layer(x)
        return x


class Layer(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super(Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # downsample
        if stride != 1 or inplanes != planes:
            self.downsample_conv = Conv1x1(inplanes, planes, stride)
            self.downsample_bn = nn.BatchNorm2d(planes)
        else:
            self.downsample_conv = None

        # first block
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # second block
        self.conv3 = Conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = Conv3x3(planes, planes)
        self.bn4 = nn.BatchNorm2d(planes)

        # third block
        self.conv5 = Conv3x3(planes, planes)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = Conv3x3(planes, planes)
        self.bn6 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        # first block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        # second block
        identity = out
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        out += identity
        out = self.relu(out)

        identity = out
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.bn6(out)

        out += identity
        out = self.relu(out)
        return out


class L0GateLayer2d(nn.Module):
    def __init__(self, n_channels, loc_mean=1, loc_sd=0.01, beta=2 / 3, gamma=-0.1, zeta=1.1, fix_temp=True):
        super(L0GateLayer2d, self).__init__()
        self.n_channels = n_channels
        self.loc = nn.Parameter(torch.zeros(n_channels).normal_(loc_mean, loc_sd))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros((1, n_channels)))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.mask(x)
        x = x.permute(2, 3, 0, 1)
        x = x * mask
        x = x.permute(2, 3, 0, 1)
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


class L0Layer(nn.Module):
    def __init__(self, layer1, layer2):
        super(L0Layer, self).__init__()
        self.relu = nn.ReLU()

        planes = layer1.conv1.out_channels * 2
        self.main_gate = L0GateLayer2d(planes)

        # downsample
        if layer1.downsample_conv is not None:
            self.downsample_conv = connect_middle_conv(layer1.downsample_conv, layer2.downsample_conv)
            self.downsample_bn = nn.BatchNorm2d(planes)
        else:
            self.downsample_conv = None

        # first block
        self.conv1 = connect_middle_conv(layer1.conv1, layer2.conv1)
        self.gate1 = L0GateLayer2d(planes)
        self.bn1 = connect_bn(planes, layer1.bn1, layer2.bn1)
        self.conv2 = connect_middle_conv(layer1.conv2, layer2.conv2)
        self.bn2 = connect_bn(planes, layer1.bn2, layer2.bn2)

        # second block
        self.conv3 = connect_middle_conv(layer1.conv3, layer2.conv3)
        self.gate2 = L0GateLayer2d(planes)
        self.bn3 = connect_bn(planes, layer1.bn3, layer2.bn3)
        self.conv4 = connect_middle_conv(layer1.conv4, layer2.conv4)
        self.bn4 = connect_bn(planes, layer1.bn4, layer2.bn4)

        # second block
        self.conv5 = connect_middle_conv(layer1.conv5, layer2.conv5)
        self.gate3 = L0GateLayer2d(planes)
        self.bn5 = connect_bn(planes, layer1.bn5, layer2.bn5)
        self.conv6 = connect_middle_conv(layer1.conv6, layer2.conv6)
        self.bn6 = connect_bn(planes, layer1.bn6, layer2.bn6)

    def forward(self, x, main_mask=None):
        if main_mask is None:
            main_mask = self.main_gate.mask(x)

        # downsample
        identity = x
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        # first block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gate1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)
        x = self.main_gate(x, main_mask)

        # second block
        identity = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.gate2(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x += identity
        x = self.relu(x)
        x = self.main_gate(x, main_mask)

        # second block
        identity = x
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.gate3(x)

        x = self.conv6(x)
        x = self.bn6(x)

        x += identity
        x = self.relu(x)
        x = self.main_gate(x, main_mask)

        return x

    def l0_loss(self):
        return self.main_gate.l0_loss() + self.gate1.l0_loss() + self.gate2.l0_loss() + self.gate3.l0_loss()

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
                self.gate2.importance_of_features(), self.gate3.importance_of_features()]


class Teacher(nn.Module):
    def __init__(self) -> None:
        super(Teacher, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = Layer(16, 16, stride=1)
        self.layer2 = Layer(16, 32, stride=2)
        self.layer3 = Layer(32, 64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 100)

        # initialization of weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
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


class Student(nn.Module):
    def __init__(self, teacher0, teacher1):
        super(Student, self).__init__()
        self.conv1 = connect_first_conv2d(teacher0.conv1, teacher1.conv1)
        self.bn1 = connect_bn(16 * 2, teacher0.bn1, teacher1.bn1)
        self.relu = nn.ReLU()

        self.layer1 = L0Layer(teacher0.layer1, teacher1.layer1)
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

