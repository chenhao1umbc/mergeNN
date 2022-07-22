from turtle import forward
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

    def forward(self, x: Tensor) -> Tensor:
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

