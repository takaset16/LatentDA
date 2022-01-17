# coding: utf-8
import torch.nn as nn
import numpy as np
import util


class ConvNet(nn.Module):
    def __init__(self, num_classes, num_channel, size_after_cnn):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.size_after_cnn = size_after_cnn

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
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.fc = nn.Linear(size_after_cnn * size_after_cnn * 128, num_classes)

    def forward(self, x, y, n_aug=0, layer_aug=0, flag_save_images=0, n_parameter=0):
        if n_aug >= 1 and layer_aug == 0:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images, n_parameter)
        x = self.layer1(x)

        if n_aug >= 1 and layer_aug == 1:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images, n_parameter)
        x = self.layer2(x)

        if n_aug >= 1 and layer_aug == 2:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images, n_parameter)
        x = self.layer3(x)

        if n_aug >= 1 and layer_aug == 3:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images, n_parameter)
        x = self.layer4(x)

        if n_aug >= 1 and layer_aug == 4:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images, n_parameter)
        x = self.layer5(x)

        if n_aug >= 1 and layer_aug == 5:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images, n_parameter)
        x = self.layer6(x)

        if n_aug >= 1 and layer_aug == 6:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images, n_parameter)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x, y
