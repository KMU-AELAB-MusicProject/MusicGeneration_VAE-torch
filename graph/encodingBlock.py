import torch
import torch.nn as nn

from graph.cbam import CBAM
from graph.weights_initializer import weights_init


class TimePitchModule(nn.Module):
    def __init__(self):
        super(TimePitchModule, self).__init__()

        self.time = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 1), stride=(2, 1), padding=[1, 0],
                               bias=False)
        self.pitch = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 4), stride=(1, 2), padding=[0, 1],
                               bias=False)

        self.bn = nn.BatchNorm2d(32, eps=1e-5, momentum=0.01, affine=True)

        self.cbam = CBAM(32)

        self.leaky = nn.LeakyReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        out = self.time(x)
        out = self.pitch(out)
        out = self.bn(out)
        out = self.cbam(out)
        out = self.leaky(out)

        return out


class PitchTimeModule(nn.Module):
    def __init__(self):
        super(PitchTimeModule, self).__init__()

        self.pitch = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4), stride=(1, 2), padding=[0, 1],
                               bias=False)
        self.time = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), stride=(2, 1), padding=[1, 0],
                              bias=False)

        self.bn = nn.BatchNorm2d(32, eps=1e-5, momentum=0.01, affine=True)

        self.cbam = CBAM(32)

        self.leaky = nn.LeakyReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        out = self.pitch(x)
        out = self.time(out)
        out = self.bn(out)
        out = self.cbam(out)
        out = self.leaky(out)

        return out


class ResidualModule(nn.Module):
    def __init__(self, channel):
        super(ResidualModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(channel, eps=1e-5, momentum=0.01, affine=True)
        self.bn2 = nn.BatchNorm2d(channel, eps=1e-5, momentum=0.01, affine=True)

        self.cbam = CBAM(channel)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        out = x + out

        out = self.relu(out)

        return out


class PoolingModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PoolingModule, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)

        self.cbam = CBAM(out_channel)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.cbam(out)
        out = self.relu(out)

        return out