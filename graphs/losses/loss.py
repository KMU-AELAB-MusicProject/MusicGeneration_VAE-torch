import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels, mean, var, gan_loss):
        loss1 = self.loss(logits, labels)
        loss2 = self.loss(torch.gt(logits, 0.35).type('torch.cuda.FloatTensor'), labels)

        # reconstruction error + KLD + gan_loss
        return 0.5 * (0.2 * loss1 + 0.8 * loss2) + \
               (-0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())) + \
               gan_loss


class DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, target):
        return self.loss(logits, target)
