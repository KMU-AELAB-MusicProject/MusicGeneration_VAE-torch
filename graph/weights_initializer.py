import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2') != -1:
        m.weight.data.normal_(0.0, 0.02)

        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)

        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

        if m.bias is not None:
            m.bias.data.zero_()