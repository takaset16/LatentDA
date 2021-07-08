# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import util

from RandAugment.networks.shakeshake.shakeshake import ShakeShake
from RandAugment.networks.shakeshake.shakeshake import Shortcut


class ShakeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(in_ch, out_ch, stride=stride)

        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNet(nn.Module):
    def __init__(self, depth, w_base, label, num_channel, n_data, n_layer, n_aug, multi_gpu=0):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6

        in_chs = [16, w_base, w_base * 2, w_base * 4]
        self.in_chs = in_chs
        self.n_data = n_data
        self.n_layer = n_layer
        self.n_aug = n_aug
        self.num_classes = label

        if multi_gpu == 1:  # マルチGPU
            self.c_in = nn.Conv2d(num_channel, in_chs[0], 3, padding=1)
            self.c_in = nn.DataParallel(self.c_in)
            self.layer1 = self._make_layer(n_units, in_chs[0], in_chs[1])
            self.layer1 = nn.DataParallel(self.layer1)
            self.layer2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
            self.layer2 = nn.DataParallel(self.layer2)
            self.layer3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)
            self.layer3 = nn.DataParallel(self.layer3)
            self.fc_out = nn.Linear(in_chs[3], self.num_classes)
            self.fc_out = nn.DataParallel(self.fc_out)
        else:  # シングルGPU
            self.c_in = nn.Conv2d(num_channel, in_chs[0], 3, padding=1)
            self.layer1 = self._make_layer(n_units, in_chs[0], in_chs[1])
            self.layer2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
            self.layer3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)
            self.fc_out = nn.Linear(in_chs[3], self.num_classes)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.in_chs[3])
        h = self.fc_out(h)
        return h
