# coding: utf-8
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import sklearn.utils
import augmentation
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional
import sklearn.datasets
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from RandAugment import RandAugment


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x)


def to_var_grad(x):
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x, requires_grad=True)


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def shuffle(x, i):

    return sklearn.utils.shuffle(x, random_state=i)


def make_training_data(x, num, seed):  # 訓練データ作成
    x = shuffle(x, seed)
    x_training = x[0:num]

    return x_training


def make_training_test_data(x, num, seed):  # 訓練データとテストデータ作成
    x = shuffle(x, seed)
    x_test = x[0:num]
    x_training = x[num:]

    return x_training, x_test


def run_n_aug(x, y, n_aug, num_classes):
    """
    # x = augmentation.random_noise(x)
    x = augmentation.horizontal_flip(x)
    # x = augmentation.vertical_flip(x)
    x = augmentation.random_crop(x)
    # x = augmentation.random_transfer(x)
    # x = augmentation.random_transfer(x)
    # x = augmentation.random_rotation(x)
    # x, y = augmentation.mixup(image=x, label=y, num_classes=num_classes, alpha=alpha)
    # x = augmentation.cutout(x)
    # x = augmentation.random_erasing(x)
    # x, y = augmentation.ricap(image_batch=x, label_batch=y, num_classes=num_classes)
    """

    # n_augで指定する場合
    if n_aug == 1:
        x = augmentation.horizontal_flip(x)
        # x = augmentation.vertical_flip(x)
    elif n_aug == 2:
        x = augmentation.random_crop(x)
    elif n_aug == 3:
        x = augmentation.random_transfer(x)
    elif n_aug == 4:
        x = augmentation.random_rotation(x)
    elif n_aug == 5:  # one-hotなラベルが返される
        x, y = augmentation.mixup(image=x, label=y, num_classes=num_classes)
    elif n_aug == 6:
        x = augmentation.cutout(x)
    elif n_aug == 7:
        x = augmentation.random_erasing(x)
    elif n_aug == 8:  # one-hotなラベルが返される
        x, y = augmentation.ricap(image_batch=x, label_batch=y, num_classes=num_classes)
        x = to_var(x)
        y = to_var(y)
    elif n_aug == 9:
        x = augmentation.random_noise(x)

    return x, y


def save_images(images):
    resize = 128
    n, c, h, w = images.shape

    for i in range(n):
        # print(images.data.cpu()[i])
        image = images[i].view(1, c, h, w)  # bilinearモードのために、4次元にする
        image = torch.nn.functional.upsample(image, size=(resize, resize), mode="bilinear", align_corners=True)
        image = image.view(c, resize, resize)  # 3次元に戻す

        img_rgb = np.round(np.array(image.data.cpu() * 255.0, np.int32))
        random_images_reshape = img_rgb.transpose(1, 2, 0)  # チャネルを最後に移動
        if c == 1:
            img_bgr = random_images_reshape  # 色の順番を変更しない
        else:
            img_bgr = random_images_reshape[:, :, [2, 1, 0]]  # 色の順番をRGBからBGRに変更

        cv2.imwrite('images/image_%d.png' % i, img_bgr)  # 画像を出力


def save_images_ch(images, n_layer):  # チャンネルごとに画像保存
    resize = 128
    n, c, h, w = images.shape
    num_save_channels = 3  # 保存するチャネル数

    for i in range(n):
        for j in range(num_save_channels):
            img = images[i, j].view(1, 1, h, w)  # bilinearモードのために、4次元にする
            img = torch.nn.functional.upsample(img, size=(resize, resize), mode="bilinear", align_corners=True)
            img = img.view(1, resize, resize)  # 3次元に戻す

            img_rgb = np.round(np.array(img.data.cpu() * 255, np.int32))
            random_images_reshape = img_rgb.transpose(1, 2, 0)  # チャネルを最後に移動

            cv2.imwrite('images/feature/image_%d_ch_%d_layer_%d.png' % (i, j, n_layer), random_images_reshape)


def save_tSNE(x, y, n_layer, layer, flag_test):
    x = np.array(x.data.cpu())
    y = np.array(y.data.cpu())
    x = x.reshape([x.shape[0], -1])

    digits2d = TSNE(n_components=2, random_state=0).fit_transform(x)

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(10):
        target = digits2d[y == i]
        ax.scatter(x=target[:, 0], y=target[:, 1], label=str(i), alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    if flag_test == 0:
        plt.savefig('images/tSNE/tSNE_n_layer_%d_layer_%d_training.png' % (n_layer, layer))
    else:
        plt.savefig('images/tSNE/tSNE_n_layer_%d_layer_%d_test.png' % (n_layer, layer))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def calcurate_var(features):
    return np.sum(np.var(features, axis=0))


def calcurate_var_cnn(features, labels, num_classes):
    var = np.zeros(num_classes)
    for i in range(num_classes):
        index = np.where(labels == i)
        x_i = features[index]
        var[i] = np.sum(np.var(x_i, axis=0))

    return np.mean(var)

