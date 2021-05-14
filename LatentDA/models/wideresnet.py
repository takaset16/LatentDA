# coding: utf-8
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import util
import sys


_bn_momentum = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=_bn_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, num_channel, n_aug, temp):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.n_aug = n_aug
        self.temp = temp

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(num_channel, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=_bn_momentum)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.dropout = nn.Dropout(inplace=False)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, y, flag_random_layer=0, flag_aug=0, flag_dropout=0, flag_var=0, layer_aug=0, layer_drop=0, layer_var=1):
        if flag_random_layer == 1:
            layer_aug = np.random.randint(layer_aug + 1)
            layer_drop = np.random.randint(layer_drop + 1)
            layer_var = np.random.randint(layer_var + 1)

        if flag_aug == 1 and layer_aug == 0:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        x = self.conv1(x)

        if flag_var == 1 and layer_var == 1:
            return x, y
        if flag_aug == 1 and layer_aug == 1:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 1:
            x = self.dropout(x)
        x = self.layer1(x)

        if flag_var == 1 and layer_var == 2:
            return x, y
        if flag_aug == 1 and layer_aug == 2:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 2:
            x = self.dropout(x)
        x = self.layer2(x)

        if flag_var == 1 and layer_var == 3:
            return x, y
        if flag_aug == 1 and layer_aug == 3:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 3:
            x = self.dropout(x)
        x = self.layer3(x)

        if flag_var == 1 and layer_var == 4:
            return x, y
        if flag_aug == 1 and layer_aug == 4:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 4:
            x = self.dropout(x)
        x = F.relu(self.bn1(x))

        if flag_var == 1 and layer_var == 5:
            return x, y
        if flag_aug == 1 and layer_aug == 5:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 5:
            x = self.dropout(x)
        # x = F.avg_pool2d(x, 8)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        if flag_var == 1 and layer_var == 6:
            return x, y
        if flag_aug == 1 and layer_aug == 6:
            x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
        if flag_dropout == 1 and layer_drop == 6:
            x = self.dropout(x)

        return x, y

    def forward_snn(self, x, y, flag_myaug=0, flag_snnloss=0, snnloss=None, flag_tSNE=0):
        if flag_myaug == 1:
            if self.n_layer == 10000:
                layer = np.random.randint(6)
            else:
                layer = self.n_layer

            if layer == 0:
                x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
                # x, y = util.run_n_aug_random(x, y, self.n_aug, self.num_classes)
            out = self.conv1(x)
            if flag_snnloss == 1:
                snnloss[0] = pytorch_snn.snnLoss_broadcast(out, y, T=self.temp, num_classes=self.num_classes, ndim=4)
                # snnloss[0] = numpy_snn.snnLoss(out, y, T=self.temp)
                # snnloss[0] = pytorch_snn.snnLoss(out, y, T=self.temp)

            if layer == 1:
                out, y = util.run_n_aug(out, y, self.n_aug, self.num_classes)
                # out, y = util.run_n_aug_random(out, y, self.n_aug, self.num_classes)
            out = self.layer1(out)
            if flag_snnloss == 1:
                snnloss[1] = pytorch_snn.snnLoss_broadcast(out, y, T=self.temp, num_classes=self.num_classes, ndim=4)
                # snnloss[1] = numpy_snn.snnLoss(out, y, T=self.temp)
                # snnloss[1] = pytorch_snn.snnLoss(out, y, T=self.temp)

            if layer == 2:
                out, y = util.run_n_aug(out, y, self.n_aug, self.num_classes)
                # out, y = util.run_n_aug_random(out, y, self.n_aug, self.num_classes)
            out = self.layer2(out)
            if flag_snnloss == 1:
                snnloss[2] = pytorch_snn.snnLoss_broadcast(out, y, T=self.temp, num_classes=self.num_classes, ndim=4)
                # snnloss[2] = numpy_snn.snnLoss(out, y, T=self.temp)
                # snnloss[2] = pytorch_snn.snnLoss(out, y, T=self.temp)

            if layer == 3:
                out, y = util.run_n_aug(out, y, self.n_aug, self.num_classes)
                # out, y = util.run_n_aug_random(out, y, self.n_aug, self.num_classes)
            out = self.layer3(out)
            if flag_snnloss == 1:
                snnloss[3] = pytorch_snn.snnLoss_broadcast(out, y, T=self.temp, num_classes=self.num_classes, ndim=4)
                # snnloss[3] = numpy_snn.snnLoss(out, y, T=self.temp)
                # snnloss[3] = pytorch_snn.snnLoss(out, y, T=self.temp)

            if layer == 4:
                out, y = util.run_n_aug(out, y, self.n_aug, self.num_classes)
                # out, y = util.run_n_aug_random(out, y, self.n_aug, self.num_classes)
            out = F.relu(self.bn1(out))
            if flag_snnloss == 1:
                snnloss[4] = pytorch_snn.snnLoss_broadcast(out, y, T=self.temp, num_classes=self.num_classes, ndim=4)
                # snnloss[4] = numpy_snn.snnLoss(out, y, T=self.temp)
                # snnloss[4] = pytorch_snn.snnLoss(out, y, T=self.temp)

            if layer == 5:
                out, y = util.run_n_aug(out, y, self.n_aug, self.num_classes)
                # out, y = util.run_n_aug_random(out, y, self.n_aug, self.num_classes)
            # out = F.avg_pool2d(out, 8)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            if flag_snnloss == 1:
                snnloss[5] = pytorch_snn.snnLoss_broadcast(out, y, T=self.temp, num_classes=self.num_classes, ndim=2)
                # snnloss[5] = numpy_snn.snnLoss(out, y, T=self.temp)
                # snnloss[5] = pytorch_snn.snnLoss(out, y, T=self.temp)
        else:
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = F.relu(self.bn1(x))
            # x = F.avg_pool2d(x, 8)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.linear(x)

        return x, y, snnloss
