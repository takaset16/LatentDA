# coding: utf-8
# https://qiita.com/tchih11/items/377cbf9162e78a639958
import torch
import torch.nn as nn
import util


class block(nn.Module):
    def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1):
        """
        ??????????????
        Args:
            first_conv_in_channels : 1???conv??1×1??input channel?
            first_conv_out_channels : 1???conv??1×1??output channel?
            identity_conv : channel?????conv?
            stride : 3×3conv?????stide??size??????????2???
        """
        super(block, self).__init__()

        # 1???conv??1×1?
        self.conv1 = nn.Conv2d(
            first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(first_conv_out_channels)

        # 2???conv??3×3?
        # ????3???size?????????stride???
        self.conv2 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(first_conv_out_channels)

        # 3???conv??1×1?
        # output channel?input channel?4????
        self.conv3 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels*4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(first_conv_out_channels*4)
        self.relu = nn.ReLU()

        # identity?channel???????????conv??1×1???????????None
        self.identity_conv = identity_conv

    def forward(self, x):

        identity = x.clone()  # ???????

        x = self.conv1(x)  # 1×1?????
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # 3×3??????????3???stride?2?????????size???????
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # 1×1?????
        x = self.bn3(x)

        # ??????conv??1×1?????identity?channel??????????
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, num_classes, num_channel=3):
        super(ResNet, self).__init__()
        self.num_classes = num_classes

        # conv1???????????????
        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x??????????????stride?1
        self.conv2_x = self._make_layer(block, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1)

        # conv3_x????????????????????stride?2
        self.conv3_x = self._make_layer(block, 4, res_block_in_channels=256,  first_conv_out_channels=128, stride=2)
        self.conv4_x = self._make_layer(block, 6, res_block_in_channels=512,  first_conv_out_channels=256, stride=2)
        self.conv5_x = self._make_layer(block, 3, res_block_in_channels=1024, first_conv_out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
        layers = []

        # 1???????????channel?????size???????
        # identify?????1×1?conv??????????????????stride?2???
        identity_conv = nn.Conv2d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1,stride=stride)
        layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

        # 2?????input_channel??1???output_channel?4?
        in_channels = first_conv_out_channels*4

        # channel???size???????????identity_conv?None?stride?1
        for i in range(num_res_blocks - 1):
            layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x, y, num_layer=0, n_aug=0, layer_aug=0, param_aug=0):
        if n_aug >= 1 and layer_aug == 0:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)

        x = self.conv1(x)   # in:(3,224*224)?out:(64,112*112)
        x = self.bn1(x)     # in:(64,112*112)?out:(64,112*112)
        x = self.relu(x)    # in:(64,112*112)?out:(64,112*112)
        x = self.maxpool(x) # in:(64,112*112)?out:(64,56*56)

        if n_aug >= 1 and layer_aug == 1:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
        x = self.conv2_x(x)  # in:(64,56*56)  ?out:(256,56*56)

        if n_aug >= 1 and layer_aug == 2:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
        x = self.conv3_x(x)  # in:(256,56*56) ?out:(512,28*28)

        if n_aug >= 1 and layer_aug == 3:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
        x = self.conv4_x(x)  # in:(512,28*28) ?out:(1024,14*14)

        if n_aug >= 1 and layer_aug == 4:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
        x = self.conv5_x(x)  # in:(1024,14*14)?out:(2048,7*7)

        if n_aug >= 1 and layer_aug == 5:
            x, y = util.run_n_aug(x, y, n_aug, self.num_classes, param_aug)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x, y
