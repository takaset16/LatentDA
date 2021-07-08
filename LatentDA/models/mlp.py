# coding: utf-8
import torch.nn as nn
import numpy as np
import augmentation as aug
import util


class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layer, n_aug):
        super(MLPNet, self).__init__()
        self.num_classes = num_classes
        self.num_unit = hidden_size  # mixupを行う直前のレイヤーのユニット数（特徴の数）
        self.n_layer = n_layer
        self.n_aug = n_aug

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, y, flag_aug, flag_var, n_layer_var=1):
        h = None
        if flag_var == 1:
            x = self.fc1(x)
            x = self.relu(x)
            h = x
        else:
            if flag_aug == 1:
                if self.n_layer == 10000:
                    layer = np.random.randint(2)
                else:
                    layer = self.n_layer

                if layer == 0:
                    x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
                x = self.fc1(x)
                x = self.relu(x)

                if layer == 1:
                    h = x
                    x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
                x = self.fc2(x)
            else:
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)

        return x, y, h
