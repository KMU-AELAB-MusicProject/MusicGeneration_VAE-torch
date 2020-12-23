import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels, mean, var, gan_loss):
        loss1 = self.loss(logits, labels)
        loss2 = self.loss(torch.gt(logits, 0.3).type('torch.cuda.FloatTensor'), labels)

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

class WAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, gen_note, note, f_logits, fz_logits):
        loss1 = self.loss(gen_note, note)
        loss2 = torch.log(f_logits).mean()
        loss3 = torch.log(fz_logits).mean()

        return loss1 - (loss2 * 10.) - (loss3 * 8.)
