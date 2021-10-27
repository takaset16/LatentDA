# coding: utf-8
import torch
import torch.nn as nn
import cv2
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
        """
        self.layer1_notrack = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU())
        self.layer2_notrack = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU())
        self.layer3_notrack = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU())
        self.layer4_notrack = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU())
        """
        self.fc = nn.Linear(size_after_cnn * size_after_cnn * 64, num_classes)
        self.dropout = nn.Dropout(inplace=False)

    def forward(self, x, y, n_aug=0, layer_aug=0, flag_dropout=0, layer_drop=0, flag_track=1, flag_save_images=0):
        if n_aug >= 1 and layer_aug == 0:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images)
        if flag_track == 1:
            x = self.layer1(x)
        else:
            x = self.layer1_notrack(x)

        if n_aug >= 1 and layer_aug == 1:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images)
        if flag_dropout == 1 and layer_drop == 1:
            x = self.dropout(x)
        if flag_track == 1:
            x = self.layer2(x)
        else:
            x = self.layer2_notrack(x)

        if n_aug >= 1 and layer_aug == 2:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images)
        if flag_dropout == 1 and layer_drop == 2:
            x = self.dropout(x)
        if flag_track == 1:
            x = self.layer3(x)
        else:
            x = self.layer3_notrack(x)

        if n_aug >= 1 and layer_aug == 3:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images)
        if flag_dropout == 1 and layer_drop == 3:
            x = self.dropout(x)
        if flag_track == 1:
            x = self.layer4(x)
        else:
            x = self.layer4_notrack(x)

        if n_aug >= 1 and layer_aug == 4:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images)
        if flag_dropout == 1 and layer_drop == 4:
            x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        if n_aug >= 1 and layer_aug == 5:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, flag_save_images)
        if flag_dropout == 1 and layer_drop == 5:
            x = self.dropout(x)

        return x, y  # all the outputs must be the type of tensor
