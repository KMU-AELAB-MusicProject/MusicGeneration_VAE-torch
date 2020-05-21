import torch
import torch.nn as nn

from graph.cbam import CBAM
from graph.weights_initializer import weights_init


class TimePitchModule(nn.Module):
    def __init__(self):
        super(TimePitchModule, self).__init__()

        self.time = nn.ConvTranspose2d(in_channels=2304, out_channels=1024, kernel_size=(6, 1), stride=(6, 1),
                                        bias=False)
        self.pitch = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=(1, 3), stride=(1, 3),
                                        bias=False)

        self.bn = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.01, affine=True)

        self.cbam = CBAM(1024)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        out = self.time(x)
        out = self.relu(out)

        out = self.pitch(out)
        out = self.bn(out)

        out = out + self.cbam(out)

        out = self.relu(out)

        return out


class PitchTimeModule(nn.Module):
    def __init__(self):
        super(PitchTimeModule, self).__init__()

        self.pitch = nn.ConvTranspose2d(in_channels=2304, out_channels=1024, kernel_size=(1, 3), stride=(1, 3),
                                        bias=False)
        self.time = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=(6, 1), stride=(6, 1),
                                       bias=False)

        self.bn = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.01, affine=True)
        self.cbam = CBAM(1024)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        out = self.pitch(x)
        out = self.relu(out)

        out = self.time(out)
        out = self.bn(out)

        out = out + self.cbam(out)

        out = self.relu(out)

        return out


class DeConvModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeConvModule, self).__init__()

        self.deConv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)

        self.deConv2 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=True)

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)
        self.bn3 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)

        self.cbam = CBAM(out_channel)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        out1 = self.deConv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.deConv2(x)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out = torch.cat((out1, out2), dim=1)

        out = self.conv(out)
        out = self.bn3(out)

        out = out + self.cbam(out)

        out = self.relu(out)

        return out


class DeConvPitchPadding(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeConvPitchPadding, self).__init__()

        self.deConv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2,
                                          padding=1, output_padding=(0, 1), bias=True)

        self.deConv2 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2,
                                          padding=1, output_padding=(0, 1), bias=True)

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)
        self.bn3 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)

        self.cbam1 = CBAM(out_channel)
        self.cbam2 = CBAM(out_channel)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        out1 = self.deConv1(x)
        out1 = self.bn2(out1)
        out1 = out1 + self.cbam1(out1)
        out1 = self.relu(out1)

        out2 = self.deConv2(x)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out = torch.cat((out1, out2), dim=1)

        out = self.conv(out)
        out = self.bn3(out)

        out = out + self.cbam2(out)

        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, layers): # [1024, 512, 256, 128, 64]
        super().__init__()

        self.leaky = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.time = TimePitchModule()
        self.pitch = PitchTimeModule()

        self.fit1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.01, affine=True)

        self.fit2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, bias=False)

        self.layers = []
        for i in range(1, len(layers)):
            if i < 3:
                self.layers.append(DeConvPitchPadding(layers[i - 1], layers[i]))
            else:
                self.layers.append(DeConvModule(layers[i - 1], layers[i]))
        self.layers = nn.ModuleList(self.layers)

        self.cbam = CBAM(1024)

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 2304, 1, 1)
        pitch = self.pitch(x)
        time = self.time(x)

        out = torch.cat((pitch, time), dim=1)
        
        out = self.fit1(out)
        out = self.bn(out)

        out = out + self.cbam(out)

        out = self.relu(out)
        
        for layer in self.layers:
            out = layer(out)
            
        logits = self.sigmoid(self.fit2(out))

        return logits
