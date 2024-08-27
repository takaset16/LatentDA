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
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, y, num_layer=0, n_aug=0, layer_aug=0, param_aug=0):
        if num_layer == 2:
            if x.ndim > 2:
                x = x.reshape(x.size(0), -1)

            if n_aug >= 1 and layer_aug == 0:
                x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
            x = self.fc1(x)
            x = self.relu(x)

            if n_aug >= 1 and layer_aug == 1:
                x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
            x = self.fc3(x)

        elif num_layer == 3:
            if x.ndim > 2:
                x = x.reshape(x.size(0), -1)

            if n_aug >= 1 and layer_aug == 0:
                x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
            x = self.fc1(x)
            x = self.relu(x)

            if n_aug >= 1 and layer_aug == 1:
                x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
            x = self.fc2(x)
            x = self.relu(x)

            if n_aug >= 1 and layer_aug == 2:
                x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
            x = self.fc3(x)

        return x, y  # all the outputs must be the type of tensor
