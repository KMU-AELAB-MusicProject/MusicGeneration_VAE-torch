import torch
import torch.nn as nn

from graph.weights_initializer import weights_init


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // 16, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // 16, channel, 1, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        out = self.sigmoid(out)

        return x * out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)

        out = self.sigmoid(out)
        return x * out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

        self.apply(weights_init)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)

        return out
