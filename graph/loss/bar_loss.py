import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, logits, labels, gan_loss):
        recon_loss = self.loss(logits, labels)
        out = torch.gt(logits, 0.35).type('torch.cuda.FloatTensor')
        additional_loss = (torch.gt(labels - out, 0.0001).type('torch.cuda.FloatTensor')).mean()

        return (recon_loss + additional_loss) + gan_loss


class DLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target=0):
        if target == 1:
            return torch.log(1. - outputs).mean()
        else:
            return torch.log(outputs).mean()
