# -*- coding: utf-8 -*-

"""
@date: 2023/9/30 上午11:20
@file: t_scheduler.py
@author: zj
@description: 
"""

import numpy as np

import torch
from torch.optim import lr_scheduler

from wr2 import wR2

model = wR2(num_classes=4)

lr = 0.01
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

epochs = 100
lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01  # linear

lrf = 0.01
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

iterations = 100
nw = max(round(3 * iterations), 100)

is_warmup = True
warmup_epoch = 5


def adjust_learning_rate(lr, warmup_epoch, optimizer, epoch: int, step: int, len_epoch: int) -> None:
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    # Warmup
    if epoch < warmup_epoch:
        lr = lr * float(1 + step + epoch * len_epoch) / (warmup_epoch * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(epochs):

    for ni in range(iterations):
        # Warmup
        if is_warmup and epoch < warmup_epoch:
            adjust_learning_rate(0.01, warmup_epoch, optimizer, epoch, ni, 100)

        optimizer.step()

    # Scheduler
    lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
    print(lr)
    # print(lr, optimizer.param_groups[0]['lr'])

    scheduler.step()

# print("*************")
# xi = [0, 100]
#
# for ni in range(100):
#     print(np.interp(ni, xi, [0.001, 0.03]))
