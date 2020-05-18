import torch
import torch.nn as nn

from graphs.weights_initializer import weights_init


class ChordFeature(nn.Module):
    def __init__(self):
        super(ChordFeature, self).__init__()

        self.chord_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 12), stride=[2, 0], padding=[1, 0],
                                     bias=False)
        self.chord_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 12), stride=[2, 0], padding=[1, 0],
                                     bias=False)
        self.chord_fit = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, bias=False)
        self.chord_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.chord_conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)
        self.batch_norm2 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.01, affine=True)
        self.batch_norm3 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.01, affine=True)
        self.batch_norm4 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.01, affine=True)
        self.batch_norm5 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)

        self.avg = nn.AvgPool2d(kernel_size=(12, 3))

    def forward(self, x):
        chord_x = x.view(-1, 5, 96 * 2, 12)
        chord_x = torch.sum(chord_x, 1, keepdim=True)  # 1,192,12

        out = self.chord_conv1(chord_x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.chord_conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.chord_fit(out)
        out = self.batch_norm3(out)
        out = self.relu(out)

        out = self.chord_conv3(out)
        out = self.batch_norm4(out)
        out = self.relu(out)

        out = self.chord_conv4(out)
        out = self.batch_norm5(out)
        out = self.relu(out)

        out = self.avg(out)

        return out


class OnOffFeature(nn.Module):
    def __init__(self):
        super(OnOffFeature, self).__init__()

        self.onoff_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=(2, 1), padding=1,
                                     bias=False)
        self.onoff_conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=(2, 1), padding=1,
                                     bias=False)
        self.batch_norm2 = nn.BatchNorm2d(8)

        self.onoff_conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=(2, 1), padding=1,
                                     bias=False)
        self.onoff_conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(2, 1), padding=1,
                                     bias=False)
        self.onoff_fit = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.onoff_conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=(2, 1), padding=1,
                                     bias=False)

        self.onoff_avg = nn.AvgPool2d(kernel_size=(6, 1))

    def forward(self, x):
        onoff_x = torch.nn.functional.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        onoff_x = torch.sum(x - onoff_x, 3, keepdim=True)  # 384,1,1

        onoff_output = self.relu(self.onoff_conv1(onoff_x))
        onoff_output = self.relu(self.onoff_conv2(onoff_output))
        onoff_output = self.batch_norm2(onoff_output)

        onoff_output = self.relu(self.onoff_conv3(onoff_output))
        onoff_output = self.relu(self.onoff_conv4(onoff_output))
        onoff_output = self.relu(self.onoff_fit(onoff_output))
        onoff_output = self.relu(self.onoff_conv5(onoff_output))

        onoff_output = self.onoff_avg(onoff_output)

        return x


class ConvModule(nn.Module):
    def __init__(self, in_channel, out_channel, isBasic=True):
        super(ConvModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(in_channel, eps=1e-5, momentum=0.01, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)

        self.isBasic = isBasic

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.isBasic:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
        else:
            out = x

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class BasicFeature(nn.Module):
    def __init__(self, layers):
        super(BasicFeature, self).__init__()

        self.pitch1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 4), stride=(1, 2), padding=[0, 1],
                                bias=False)
        self.pitch2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 1), stride=(2, 1), padding=[1, 0],
                                bias=False)

        self.time1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4, 1), stride=(2, 1), padding=[1, 0],
                               bias=False)
        self.time2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 4), stride=(1, 2), padding=[0, 1],
                               bias=False)

        self.fit = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm2d(8, eps=1e-5, momentum=0.01, affine=True)

        self.avg = nn.AvgPool2d(kernel_size=(12, 4))

        self.layers = []

        for i in range(1, len(layers)):
            self.layers.append(ConvModule(layers[i - 1], layers[i], False if i < 3 else True))

    def forward(self, x):
        pitch = self.relu(self.pitch1(x))
        pitch = self.relu(self.pitch2(pitch))

        time = self.relu(self.time1(x))
        time = self.relu(self.time2(time))

        out = torch.cat((pitch, time), dim=1)

        out = self.fit(out)
        out = self.bn(out)
        out = self.relu(out)

        for layer in self.layers:
            out = layer(out)

        out = self.avg(out)

        return out


class BarDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ###########################
        self.chord = ChordFeature()
        self.on_off = OnOffFeature()
        self.basic = BasicFeature([8, 16, 32, 64])

        self.linear = nn.Linear(64 * 3, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 1, 96 * 2, 60)

        # chord feature extraction
        chord_output = self.chord(x)

        # chord feature extraction
        onoff_output = self.onoff_avg(x)

        # normal feature extraction
        out = self.basic(x)

        out = torch.cat((chord_output, onoff_output, out), dim=1)
        out = out.view(-1, 64 * 3)

        outputs = self.sigmoid(self.linear1(out))

        return outputs
