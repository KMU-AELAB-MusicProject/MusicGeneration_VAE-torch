import torch
import torch.nn as nn

from graphs.weights_initializer import weights_init


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        ###########################
        self.chord_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=(2, 1), padding=1,
                                     bias=False)
        self.chord_conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=(2, 1), padding=1,
                                     bias=False)
        self.batch_norm1 = nn.BatchNorm2d(8)

        self.chord_conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=(2, 1), padding=1,
                                     bias=False)
        self.chord_conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.chord_fit = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.chord_conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)

        self.avg = nn.AvgPool2d(kernel_size=(12, 3))

        ###########################
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

        self.onoff_avg = nn.AvgPool2d(kernel_size=(12, 1))

        ###########################
        self.pitch1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 4), stride=(1, 2), padding=[0, 1],
                                bias=False)
        self.pitch2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 1), stride=(2, 1), padding=[1, 0],
                                bias=False)

        self.time1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4, 1), stride=(2, 1), padding=[1, 0],
                               bias=False)
        self.time2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 4), stride=(1, 2), padding=[0, 1],
                               bias=False)

        self.fit1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1,  bias=False)
        self.batch_norm3 = nn.BatchNorm2d(8)

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.fit2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)

        self.linear1 = nn.Linear(64 * 3, 128, bias=True)
        self.linear2 = nn.Linear(128, 1, bias=True)

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 1, 384, 96)

        # chord feature extraction
        chord_x = x.view(-1, 8, 384, 12)
        chord_x = torch.sum(chord_x, 1, keepdim=True)   # 384,12,1

        chord_output = self.relu(self.chord_conv1(chord_x))
        chord_output = self.relu(self.chord_conv2(chord_output))
        chord_output = self.batch_norm1(chord_output)

        chord_output = self.relu(self.chord_conv3(chord_output))
        chord_output = self.relu(self.chord_conv4(chord_output))
        chord_output = self.relu(self.chord_fit(chord_output))
        chord_output = self.relu(self.chord_conv5(chord_output))

        chord_output = self.avg(chord_output)

        # chord feature extraction
        onoff_x = torch.nn.functional.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        onoff_x = torch.sum(x - onoff_x, 3, keepdim=True) # 384,1,1

        onoff_output = self.relu(self.onoff_conv1(onoff_x))
        onoff_output = self.relu(self.onoff_conv2(onoff_output))
        onoff_output = self.batch_norm2(onoff_output)

        onoff_output = self.relu(self.onoff_conv3(onoff_output))
        onoff_output = self.relu(self.onoff_conv4(onoff_output))
        onoff_output = self.relu(self.onoff_fit(onoff_output))
        onoff_output = self.relu(self.onoff_conv5(onoff_output))

        onoff_output = self.onoff_avg(onoff_output)

        # normal feature extraction
        pitch = self.relu(self.pitch1(x))
        pitch = self.relu(self.pitch2(pitch))

        time = self.relu(self.time1(x))
        time = self.relu(self.time2(time))

        out = torch.cat((pitch, time), dim=1)

        out = self.relu(self.fit1(out))
        out = self.batch_norm3(out)

        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))

        out = self.relu(self.fit2(out))
        out = self.batch_norm4(out)

        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.avg(out)

        out = torch.cat((chord_output, onoff_output, out), dim=1)
        out = out.view(-1, 64 * 3)

        out = self.relu(self.linear1(out))
        logits = self.linear2(out)

        return logits
