# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def calcDistance(x_target, x_without_target, T):
    term = 0
    for i in range(len(x_without_target)):
        se = np.sum((x_target - x_without_target[i]) ** 2)
        term += np.exp(((-1) * se) / T)

    term = term - 1

    return term


def snnLoss(x, y, T):
    x = np.array(x.data.cpu())
    y = np.array(y.data.cpu())

    res = 0
    for i in range(x.shape[0]):
        term1 = calcDistance(x[i], x[y == y[i]], T)
        term2 = calcDistance(x[i], x, T)

        res += np.log(term1/term2 + 1e-16)

    res = (-1) * (res / x.shape[0])

    return res


def makeData(bias):
    np.random.seed(9)

    data1 = np.random.randn(100, 2) + [0, 0]
    data2 = np.random.randn(100, 2) + [bias, 0]
    data3 = np.random.randn(100, 2) + [0, bias]
    data4 = np.random.randn(100, 2) + [bias, bias]

    plt.scatter(data1[:, 0], data1[:, 1])
    plt.scatter(data2[:, 0], data2[:, 1])
    plt.scatter(data3[:, 0], data3[:, 1])
    plt.scatter(data4[:, 0], data4[:, 1])
    plt.show()

    x = np.r_[data1, data2, data3, data4]
    y = np.zeros(400)
    y[100:200] = 1
    y[200:300] = 2
    y[300:] = 3

    return x, y


"""
T = 100
x, y = makeData(1)
res = snnLoss(x.copy(), y.copy(), T)

print(res)
"""
