#%% load dependency
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
from datetime import datetime
print('starting date time ', datetime.now())

class Teacher(nn.Module):
    def __init__(self) -> None:
        super(Teacher, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Conv2d(16, 16,kernel_size=3, padding=1,stride=1)
        self.layer2 = nn.Conv2d(16, 1,kernel_size=3, padding=1,stride=1)

        self.layers = nn.Sequential(
                    self.conv1,
                    self.bn1,
                    self.relu,
                    self.layer1,
                    self.layer2,
                    )

        # initialization of weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return out
#%%
t = Teacher()
x = torch.rand(64,3, 16, 16)
xhat = t(x)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#%%

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: int = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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

class Downsample(nn.Modules):
    def __init__(self, n_in, n_out) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(n_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    def forward(self, x):
        return self.downsample(x)

class Resnet(nn.Modules):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        #(maxpool): Cifar does not have it.
        self.layer1 = nn.Sequential(
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            )
        self.layer2 = nn.Sequential(
            BasicBlock(16, 32, stride=2, downsample=Downsample(16, 16)),
            BasicBlock(32, 32),
            BasicBlock(32, 32)
            )
        self.layer3 = nn.Sequential(
            BasicBlock(32, 64, stride=2, downsample=Downsample(16, 16)),
            BasicBlock(64, 64, stride=2),
            BasicBlock(64, 64)
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