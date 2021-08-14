# coding: utf-8
import torch.nn as nn
import numpy as np
import augmentation as aug
import util


class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPNet, self).__init__()
        self.num_classes = num_classes
        self.num_unit = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, y, n_aug=0, layer_aug=0, flag_dropout=0, layer_drop=0, flag_track=1):
        if n_aug >= 1 and layer_aug == 0:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes)
        x = self.fc1(x)
        x = self.relu(x)

        if n_aug >= 1 and layer_aug == 1:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 1:
            x = self.dropout(x)
        x = self.fc2(x)

        return x, y  # all the outputs must be the type of tensor
