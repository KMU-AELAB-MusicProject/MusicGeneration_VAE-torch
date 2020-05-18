import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels, mean, var, pre_mean, pre_var):
        recon_loss = self.loss(logits, labels)
        elbo = (torch.sum(1 + var - mean.pow(2) - var.exp()) + torch.sum(1 + var - pre_mean.pow(2) - pre_var.exp())) / 2

        out = torch.gt(logits, 0.35).type('torch.cuda.FloatTensor')
        additional_loss = (torch.gt(labels - out, 0.0001).type('torch.cuda.FloatTensor')).mean()

        # reconstruction error + KLD + gan_loss
        return (recon_loss + additional_loss * 0.8) - (0.5 * elbo) # - torch.log(gan_loss).mean()


class PhraseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels, mean, var):
        recon_loss = self.loss(logits, labels)

        out = torch.gt(logits, 0.35).type('torch.cuda.FloatTensor')
        additional_loss = (torch.gt(labels - out, 0.0001).type('torch.cuda.FloatTensor')).mean()

        # reconstruction error + KLD + gan_loss
        return (recon_loss + additional_loss * 0.8) - (0.5 * torch.sum(1 + var - mean.pow(2) - var.exp()))


class DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, f_logits, r_logits):
        return -((torch.log(r_logits).mean()) + torch.log(1 - f_logits).mean())
