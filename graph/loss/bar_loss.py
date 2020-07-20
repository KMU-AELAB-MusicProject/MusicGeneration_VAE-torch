import torch
import numpy as np
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        distribution = np.array(
            [0.0079033, 0.00712255, 0.01189558, 0.00953322, 0.01102056, 0.01156428, 0.01136433, 0.01637716, 0.01211462,
             0.01776168, 0.01644157, 0.0171948, 0.01922302, 0.01582762, 0.02385192, 0.02001634, 0.02312213, 0.02348127,
             0.02263083, 0.0268141, 0.02373071, 0.02942328, 0.0272045, 0.0304963, 0.03032582, 0.02782333, 0.03458292,
             0.03230801, 0.03388906, 0.03283811, 0.03093611, 0.03616363, 0.03006419, 0.03296618, 0.02867032, 0.02654072,
             0.02609579, 0.01954488, 0.02251165, 0.01813882, 0.01599178, 0.01313839, 0.01104167, 0.01169814, 0.00756204,
             0.00793332, 0.00601032, 0.00540243, 0.00512497, 0.00286655, 0.00308927, 0.00260029, 0.00184589, 0.00166959,
             0.00103728, 0.00112497, 0.00071164, 0.00052543, 0.00072274, 0.00038808], dtype=np.float32) * 0.08

        self.loss = nn.BCELoss()
        self.distribution_smoothing = torch.from_numpy(distribution).cuda()
        self.default_smoothing = torch.Tensor(np.array([0.1 / 60], dtype=np.float32)).cuda()

    def forward(self, logits, labels, is_pretraining=False):
        smoothed_labels = (labels * 0.82) + self.default_smoothing + self.distribution_smoothing
        recon_loss = self.loss(logits, smoothed_labels)

        if is_pretraining:
            out = torch.gt(logits, 0.35).type('torch.cuda.FloatTensor')
            additional_loss = (torch.gt(labels - out, 0.0001).type('torch.cuda.FloatTensor')).sum() * 0.0008
            return recon_loss + additional_loss * 0.0008

        return recon_loss


class DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)
