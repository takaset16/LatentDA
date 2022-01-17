# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss and optimizer
class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        log_likelihood = - F.log_softmax(inputs, dim=1)
        loss = torch.sum(torch.mul(log_likelihood, target)) / inputs.shape[0]  # サンプル1個あたりのlossを返さないと学習が失敗する

        return loss


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, size_average=True):
        super(SmoothCrossEntropyLoss. self).__init__()
        self.label_smoothing = label_smoothing
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        if self.label_smoothing > 0.0:
            smooth = torch.zeros_like(target)
            smooth = smooth + s_by_c
            target = target * (1. - self.label_smoothing) + smooth

        return cross_entropy(input, target, self.size_average)


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
