# -*- coding: utf-8 -*-

"""
@date: 2023/9/27 下午4:56
@file: loss.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

provNum, alphaNum, adNum = 38, 25, 35


class wR2Loss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, outputs, targets):
        assert len(outputs) == len(targets)
        assert len(targets[0]) == 11

        loss_xy = self.loss(outputs[:, :2], targets[:, :2])
        loss_wh = self.loss(outputs[:, 2:], targets[:, 2:4])
        total_loss = 0.8 * loss_xy + 0.2 * loss_wh

        return total_loss


class RPNetLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_detect = nn.L1Loss()
        self.loss_classify = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        assert len(outputs) == len(targets)
        assert len(targets[0]) == 11

        loss_xy = self.loss_detect(outputs[:, :2], targets[:, :2])
        loss_wh = self.loss_detect(outputs[:, 2:4], targets[:, 2:4])
        total_loss = 0.8 * loss_xy + 0.2 * loss_wh

        total_loss += self.loss_classify(
            outputs[:, 4:(4 + provNum)], targets[:, 4].long())
        total_loss += self.loss_classify(
            outputs[:, (4 + provNum):(4 + provNum + alphaNum)], targets[:, 5].long())
        total_loss += self.loss_classify(
            outputs[:, (4 + provNum + alphaNum):(4 + provNum + alphaNum + adNum)], targets[:, 6].long())
        total_loss += self.loss_classify(
            outputs[:, (4 + provNum + alphaNum + adNum):(4 + provNum + alphaNum + adNum * 2)], targets[:, 7].long())
        total_loss += self.loss_classify(
            outputs[:, (4 + provNum + alphaNum + adNum * 2):(4 + provNum + alphaNum + adNum * 3)], targets[:, 8].long())
        total_loss += self.loss_classify(
            outputs[:, (4 + provNum + alphaNum + adNum * 3):(4 + provNum + alphaNum + adNum * 4)], targets[:, 9].long())
        total_loss += self.loss_classify(
            outputs[:, (4 + provNum + alphaNum + adNum * 4):(4 + provNum + alphaNum + adNum * 5)],
            targets[:, 10].long())

        return total_loss
