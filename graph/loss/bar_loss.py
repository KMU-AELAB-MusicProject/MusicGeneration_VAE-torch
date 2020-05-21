import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels, gan_loss):
        recon_loss = self.loss(logits, labels)
        out = torch.gt(logits, 0.35).type('torch.cuda.FloatTensor')
        additional_loss = (torch.gt(labels - out, 0.0001).type('torch.cuda.FloatTensor')).sum()

        return (recon_loss + additional_loss * 0.01) + gan_loss


class DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)
