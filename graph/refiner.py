import torch
import torch.nn as nn

from graph.weights_initializer import weights_init


class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=4, padding=2),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(.2),
            nn.MaxPool2d(kernel_size=2)
        )   # [96, 60] -> [48, 30]

        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, padding=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(.2),
            nn.MaxPool2d(kernel_size=2)
        )   # [48, 30] -> [24, 15]

        self.layer3 = torch.nn.Sequential(
            nn.Linear(2880, 1024),
            nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            nn.Linear(1024, 2880),
            nn.ReLU()
        )

        self.layer5 = torch.nn.Sequential(
            nn.ConvTranspose2d(8, 2, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )

        self.layer6 = torch.nn.Sequential(
            nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, x):   # x [batch, 1, 96, 60]
        x_2 = self.layer1(x)
        x_8 = self.layer2(x_2)
        x_feature = self.layer3(x_8.view(-1, 2880))
        x_feature = self.layer4(x_feature)
        x_8_t = x_8 + x_feature.view(-1, 8, 24, 15)
        x_2_t = x_2 + self.layer5(x_8_t)
        x_1_t = (x + self.layer6(x_2_t)) * 0.5

        return x_1_t
