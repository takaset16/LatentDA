# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def calcDistance(x_target, x_without_target, T):
    term = 0
    for i in range(len(x_without_target)):
        se = torch.sum((x_target - x_without_target[i]) ** 2)
        term = term + torch.exp(((-1) * se) / T)

    term = term - 1

    return term


def snnLoss(x, y, T):
    res = 0

    for i in range(len(x)):
        term1 = calcDistance(x[i], x[y == y[i]], T)
        term2 = calcDistance(x[i], x, T)

        res = res + torch.log(term1 / term2 + 1e-16)

    res = (-1) * (res / len(x))

    return res


def calcDistance_broadcast(x1, x2, T, ndim):
    se = None
    if ndim == 4:
        se = torch.sum((x1 - x2) ** 2, dim=(2, 3, 4))  # shape=(n, n, c, h, w) (n, n)
    elif ndim == 2:
        se = torch.sum((x1 - x2) ** 2, dim=2)  # shape=(n, n, c, h, w) (n, n)
    # print(se.data)
    # se_max = torch.max(se)
    # print(se_max.data)
    # print((1e4 + (-1) * se).data)
    term = torch.sum(torch.exp((1e4 + (-1) * se) / T), dim=1) - 1  # 各サンプルの値から自分自身との距離1をひいておく shape=(n)
    # print(term.data)

    return term


def snnLoss_broadcast(x, y, T, num_classes, ndim):
    x1 = x.clone()
    x2 = x.clone()

    n = 0
    if ndim == 4:
        n, c, h, w = x.shape

        x1 = x1.view(n, 1, c, h, w)  # 次元を追加
        x2 = x2.view(1, n, c, h, w)  # 次元を追加
    elif ndim == 2:
        n, v = x.shape

        x1 = x1.view(n, 1, v)  # 次元を追加
        x2 = x2.view(1, n, v)  # 次元を追加

    """同じクラス内の距離を計算"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    term1_classes = torch.ones((num_classes, n)).to(device)
    term2_classes = term1_classes.clone()

    for i in range(num_classes):
        term1_classes[i, 0:x1[y == i].shape[0]] = calcDistance_broadcast(x1[y == i], x2[:, y == i], T, ndim)  # 同じクラスのサンプルの配列を取り出して、すべての組み合わせの距離を計算 shape=(n, n, c, h, w)

    """サンプルのすべての組み合わせの距離を計算"""
    term2 = calcDistance_broadcast(x1, x2, T, ndim)

    for i in range(num_classes):
        term2_classes[i, 0:x1[y == i].shape[0]] = term2[y == i]

    res = torch.sum(torch.log(term1_classes / term2_classes + 1e-16))
    # print(res.data)
    ans = (-1) * (1 / n) * res  # すべて掛け合わせ、各クラスの値を足し shape=(n)
    # print(ans.data)

    return ans
