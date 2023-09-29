# -*- coding: utf-8 -*-

"""
@date: 2023/9/28 下午2:46
@file: rpnet.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wr2 import wR2

provNum, alphaNum, adNum = 38, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def roi_pooling(features, rois, size=(7, 7)):
    """
    每张图片执行单个ROI区域截取
    """
    assert len(features) == len(rois)
    assert rois.dim() == 2 and rois.size(1) == 4
    assert features.dim() == 4

    output = []
    for feature, roi in zip(features, rois.long()):
        x1, y1, x2, y2 = roi
        roi_feature = feature[:, y1:(y2 + 1), x1:(x2 + 1)]
        output.append(F.adaptive_max_pool2d(roi_feature, size))

    return torch.stack(output)


class RPNet(nn.Module):

    def __init__(self, device, wr2_pretrained=None):
        super().__init__()
        self.wR2 = wR2(num_classes=4)
        if wr2_pretrained is not None:
            print(f"Loading wR2 pretrained: {wr2_pretrained}")
            self.wR2.load_state_dict(torch.load(wr2_pretrained, map_location='cpu'))

        self.c1 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, provNum),
        )
        self.c2 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphaNum),
        )
        self.c3 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.c4 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.c5 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.c6 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.c7 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )

        self.postfix = torch.FloatTensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]).to(device)
        self.device = device

    def _forward(self, x):
        x = self.wR2.s1(x)
        x2 = self.wR2.s2(x)
        x = self.wR2.s3(x2)
        x4 = self.wR2.s4(x)
        x = self.wR2.s5(x4)
        x6 = self.wR2.s6(x)
        x = self.wR2.s7(x6)
        x = self.wR2.s8(x)
        x = self.wR2.s9(x)
        x = self.wR2.s10(x)

        x = x.view(x.size(0), -1)
        box_xywh = self.wR2.classifier(x)

        return box_xywh, x2, x4, x6

    def forward(self, x):
        box_xywh, x2, x4, x6 = self._forward(x)

        # [x_c, y_c, box_w, box_h] -> [x1, y1, x2, y2]
        assert box_xywh.size(1) == 4
        box_xyxy = box_xywh.mm(self.postfix).clamp(min=0, max=1)

        p1 = torch.FloatTensor(
            [[x2.size(2), 0, 0, 0], [0, x2.size(3), 0, 0], [0, 0, x2.size(2), 0], [0, 0, 0, x2.size(3)]]).to(
            self.device)
        p2 = torch.FloatTensor(
            [[x4.size(2), 0, 0, 0], [0, x4.size(3), 0, 0], [0, 0, x4.size(2), 0], [0, 0, 0, x4.size(3)]]).to(
            self.device)
        p3 = torch.FloatTensor(
            [[x6.size(2), 0, 0, 0], [0, x6.size(3), 0, 0], [0, 0, x6.size(2), 0], [0, 0, 0, x6.size(3)]]).to(
            self.device)

        pooling1 = roi_pooling(x2, box_xyxy.mm(p1), size=(16, 8))
        pooling2 = roi_pooling(x4, box_xyxy.mm(p2), size=(16, 8))
        pooling3 = roi_pooling(x6, box_xyxy.mm(p3), size=(16, 8))
        rois = torch.cat((pooling1, pooling2, pooling3), dim=1)
        rois = rois.view(rois.size(0), -1)

        o1 = self.c1(rois)
        o2 = self.c2(rois)
        o3 = self.c3(rois)
        o4 = self.c4(rois)
        o5 = self.c5(rois)
        o6 = self.c6(rois)
        o7 = self.c7(rois)

        # box_xywh: [N, 4]
        # o1: [N, provNum]
        # o2: [N, alphaNum]
        # o3: [N, adNum]
        # o4: [N, adNum]
        # o5: [N, adNum]
        # o6: [N, adNum]
        # o7: [N, adNum]
        return torch.cat((box_xywh, o1, o2, o3, o4, o5, o6, o7), dim=1)
        # return box_xywh, [o1, o2, o3, o4, o5, o6, o7]
