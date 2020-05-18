import torch
import torch.nn as nn

from graph.weights_initializer import weights_init


class TimePitchModule(nn.Module):
    def __init__(self):
        super(TimePitchModule, self).__init__()

        self.time = nn.ConvTranspose2d(in_channels=2304, out_channels=1024, kernel_size=(6, 1), stride=(6, 1),
                                        bias=False)
        self.pitch = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=(1, 6), stride=(1, 6),
                                        bias=False)

        self.bn = nn.BatchNorm2d(32, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.time(x)
        out = self.pitch(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


class PitchTimeModule(nn.Module):
    def __init__(self):
        super(PitchTimeModule, self).__init__()

        self.pitch = nn.ConvTranspose2d(in_channels=2304, out_channels=1024, kernel_size=(1, 6), stride=(1, 6),
                                        bias=False)
        self.time = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=(6, 1), stride=(6, 1),
                                       bias=False)

        self.bn = nn.BatchNorm2d(32, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pitch(x)
        out = self.time(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


class DeConvModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeConvModule, self).__init__()

        self.deConv1 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=False)

        self.deConv2 = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=False)

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=1,
                              bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)
        self.bn3 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.deConv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.deConv2(x)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out = torch.cat((out1, out2), dim=1)

        out = self.conv2(out)
        out = self.bn2(out)

        out = x + out

        out = self.conv(out)
        out = self.bn3(out)
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


        self.fit = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, bias=False)
        self.fit2 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(32, eps=1e-5, momentum=0.01, affine=True)

        self.layers = []

        for i in range(1, len(layers)):
            self.layers.append(DeConvModule(layers[i - 1], layers[i]))

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 2304, 1, 1)
        pitch = self.pitch(x)
        time = self.time(x)

        out = torch.cat((pitch, time), dim=1)

        out = self.fit(out)
        out = self.bn(out)
        out = self.relu(out)

        for layer in self.layers:
            out = layer(out)

        logits = self.sigmoid(self.fit3(out))

        return logits
