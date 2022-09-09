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


#%%

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


class Downsample(nn.Module):
    def __init__(self, n_in, n_out) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(n_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    def forward(self, x):
        return self.downsample(x)


class Downsample3D(nn.Module):
    def __init__(self, n_in, n_out) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(n_in, n_out, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(n_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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



#%% 
x = torch.rand(32, 1, 32,64,64)
m = Resnet3D()
xh = m(x)