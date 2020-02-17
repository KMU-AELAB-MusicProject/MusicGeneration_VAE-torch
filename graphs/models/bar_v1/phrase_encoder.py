import torch
import torch.nn as nn

from graphs.weights_initializer import weights_init


class PhraseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.leaky = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.pitch1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4), stride=(1, 2),
                                padding=[0, 1], bias=False)
        self.pitch2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), stride=(2, 1),
                                padding=[1, 0], bias=False)

        self.time1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 1), stride=(2, 1),
                               padding=[1, 0], bias=False)
        self.time2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 4), stride=(1, 2),
                               padding=[0, 1], bias=False)

        self.reduce1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.fit1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(128)

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        self.reduce2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.fit2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False)

        self.batch_norm2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.reduce3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.fit3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=False)

        self.batch_norm3 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)

        self.reduce4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.fit4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, bias=False)

        self.batch_norm4 = nn.BatchNorm2d(1024)

        self.avg = nn.AvgPool2d(kernel_size=(12, 3))

        self.mean = nn.Linear(1024, 1152, bias=True)
        self.var = nn.Linear(1024, 1152, bias=True)

        self.apply(weights_init)

    def reparameterize(self, mean, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        pitch_out = self.leaky(self.pitch1(x))
        pitch_out = self.leaky(self.pitch2(pitch_out))

        time_out = self.leaky(self.time1(x))
        time_out = self.leaky(self.time2(time_out))

        out = torch.cat((pitch_out, time_out), dim=1)

        out = self.leaky(self.reduce1(out))
        out = self.fit1(out)
        out = self.batch_norm1(out)
        out1 = self.leaky(out)

        out = self.leaky(self.conv1(out1))
        out = self.leaky(self.conv2(out))

        out = self.leaky(self.reduce2(out + out1))
        out = self.fit2(out)
        out = self.batch_norm2(out)
        out2 = self.leaky(out)

        out = self.leaky(self.conv3(out2))
        out = self.leaky(self.conv4(out))

        out = self.leaky(self.reduce3(out + out2))
        out = self.leaky(self.fit3(out))
        out = self.batch_norm3(out)
        out3 = self.relu(out)

        out = self.relu(self.conv5(out3))
        out = self.relu(self.conv6(out))

        out = self.relu(self.reduce4(out + out3))
        out = self.relu(self.fit4(out))
        out4 = self.avg(out)

        mean = self.mean(out4.view(-1, 1024))
        var = self.var(out4.view(-1, 1024))

        z = self.reparameterize(mean, var)

        return z, mean, var
