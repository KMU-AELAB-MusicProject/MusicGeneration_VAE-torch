import torch
import torch.nn as nn

from graphs.weights_initializer import weights_init


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.leaky = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.pitch1 = nn.ConvTranspose2d(in_channels=510, out_channels=256, kernel_size=(1, 6), stride=(1, 6),
                                         bias=False)
        self.pitch2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(24, 1), stride=(24, 1),
                                         bias=False)

        self.time1 = nn.ConvTranspose2d(in_channels=510, out_channels=256, kernel_size=(24, 1), stride=(24, 1),
                                        bias=False)
        self.time2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(1, 6), stride=(1, 6),
                                        bias=False)

        self.fit1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(256)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=True)

        self.fit2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=True)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=True)

        self.fit3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, bias=False)

        self.lstm = nn.LSTM(input_size=96, hidden_size=96, num_layers=2, batch_first=True)

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 510, 1, 1)
        pitch_out = self.leaky(self.pitch1(x))
        pitch_out = self.leaky(self.pitch2(pitch_out))

        time_out = self.leaky(self.time1(x))
        time_out = self.leaky(self.time2(time_out))

        out = torch.cat((pitch_out, time_out), dim=1)

        out = self.leaky(self.fit1(out))
        out = self.batch_norm1(out)

        out = self.leaky(self.deconv1(out))
        out = self.leaky(self.deconv2(out))

        out = self.leaky(self.fit2(out))
        out = self.batch_norm2(out)

        out = self.leaky(self.deconv3(out))
        out = self.leaky(self.deconv4(out))

        out = self.sigmoid(self.fit3(out))

        self.lstm.flatten_parameters()
        out = self.sigmoid(self.lstm(out.view(-1, 384, 96))[0])

        logits = out.view(-1, 1, 384, 96)

        return logits
