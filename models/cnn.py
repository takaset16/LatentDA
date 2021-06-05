# coding: utf-8
import torch
import torch.nn as nn
import cv2
import numpy as np
import util


class ConvNet(nn.Module):
    def __init__(self, num_classes, num_channel, size_after_cnn, n_aug):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.size_after_cnn = size_after_cnn
        self.n_aug = n_aug

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc = nn.Linear(size_after_cnn * size_after_cnn * 64, num_classes)
        self.dropout = nn.Dropout(inplace=False)

    def forward(self, x, y, flag_aug=0, flag_dropout=0, flag_var=0, layer_aug=0, layer_drop=0, layer_var=1):
        if flag_aug == 1 and layer_aug == 0:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        x = self.layer1(x)

        if flag_var == 1 and layer_var == 1:
            return x, y
        if flag_aug == 1 and layer_aug == 1:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 1:
            x = self.dropout(x)
        x = self.layer2(x)

        if flag_var == 1 and layer_var == 2:
            return x, y
        if flag_aug == 1 and layer_aug == 2:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 2:
            x = self.dropout(x)
        x = self.layer3(x)

        if flag_var == 1 and layer_var == 3:
            return x, y
        if flag_aug == 1 and layer_aug == 3:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 3:
            x = self.dropout(x)
        x = self.layer4(x)

        if flag_var == 1 and layer_var == 4:
            return x, y
        if flag_aug == 1 and layer_aug == 4:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 4:
            x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        if flag_var == 1 and layer_var == 5:
            return x, y
        if flag_aug == 1 and layer_aug == 5:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 5:
            x = self.dropout(x)

        return x, y  # all the outputs must be the type of tensor
