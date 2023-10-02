# -*- coding: utf-8 -*-

"""
@date: 2023/9/28 下午6:01
@file: ccpdevaluator.py
@author: zj
@description: 
"""

import torch
from torch import Tensor

provNum, alphaNum, adNum = 38, 25, 35


def bboxes_iou(bboxes_a: Tensor, bboxes_b: Tensor, xyxy=True) -> Tensor:
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    # bboxes_a: [N_a, 4] bboxes_b: [N_b, 4]
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        # xyxy: x_top_left, y_top_left, x_bottom_right, y_bottom_right
        # 计算交集矩形的左上角坐标
        # torch.max([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        # torch.max: 双重循环
        #   第一重循环 for i in range(N_a)，遍历boxes_a, 获取边界框i，大小为[2]
        #       第二重循环　for j in range(N_b)，遍历bboxes_b，获取边界框j，大小为[2]
        #           分别比较i[0]/j[0]和i[1]/j[1]，获取得到最大的x/y
        #   遍历完成后，获取得到[N_a, N_b, 2]
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        # 计算交集矩形的右下角坐标
        # torch.min([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # 计算bboxes_a的面积
        # x_bottom_right/y_bottom_right - x_top_left/y_top_left = w/h
        # prod([N, w/h], 1) = [N], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # x_center/y_center -> x_top_left, y_top_left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        # x_center/y_center -> x_bottom_right/y_bottom_right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # prod([N_a, w/h], 1) = [N_a], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # 判断符合条件的结果：x_top_left/y_top_left < x_bottom_right/y_bottom_right
    # [N_a, N_b, 2] < [N_a, N_b, 2] = [N_a, N_b, 2]
    # prod([N_a, N_b, 2], 2) = [N_a, N_b], 数值为1/0
    en = (tl < br).type(tl.type()).prod(dim=2)
    # 首先计算交集w/h: [N_a, N_b, 2] - [N_a, N_b, 2] = [N_a, N_b, 2]
    # 然后计算交集面积：prod([N_a, N_b, 2], 2) = [N_a, N_b]
    # 然后去除不符合条件的交集面积
    # [N_a, N_b] * [N_a, N_b](数值为1/0) = [N_a, N_b]
    # 大小为[N_a, N_b]，表示bboxes_a的每个边界框与bboxes_b的每个边界框之间的IoU
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    # 计算IoU
    # 首先计算所有面积
    # area_a[:, None] + area_b - area_i =
    # [N_a, 1] + [N_b] - [N_a, N_b] = [N_a, N_b]
    # 然后交集面积除以所有面积，计算IoU
    # [N_a, N_b] / [N_a, N_b] = [N_a, N_b]
    return area_i / (area_a[:, None] + area_b - area_i)


class CCPDEvaluator:

    def __init__(self, only_det=False, iou_thresh=0.7):
        self.iou_thresh = iou_thresh
        self.only_det = only_det

        self.det_correct_num = 0.
        self.classify_correct_num = 0.
        self.total_num = 0.

    def reset(self):
        self.det_correct_num = 0.
        self.classify_correct_num = 0.
        self.total_num = 0.

    def update(self, outputs, targets):
        assert len(outputs) == len(targets)

        det_correct_num = 0.
        classify_correct_num = 0.
        total_num = len(outputs)

        for output, target in zip(outputs, targets):
            pred_xywh = output[:4]
            target_xywh = target[:4]

            iou = bboxes_iou(pred_xywh.unsqueeze(0), target_xywh.unsqueeze(0), xyxy=False)
            # print(pred_xywh, target_xywh, iou)
            if iou > self.iou_thresh:
                det_correct_num += 1
        self.det_correct_num += det_correct_num
        self.total_num += total_num

        ap = det_correct_num / total_num
        if self.only_det:
            return ap, 0.

        for output, target in zip(outputs, targets):
            lp_name = list()
            lp_name.append(torch.argmax(output[4:(4 + provNum)]).item())
            lp_name.append(torch.argmax(output[(4 + provNum):(4 + provNum + alphaNum)]).item())
            lp_name.append(torch.argmax(output[(4 + provNum + alphaNum):(4 + provNum + alphaNum + adNum)]).item())
            lp_name.append(
                torch.argmax(output[(4 + provNum + alphaNum + adNum):(4 + provNum + alphaNum + adNum * 2)]).item())
            lp_name.append(
                torch.argmax(output[(4 + provNum + alphaNum + adNum * 2):(4 + provNum + alphaNum + adNum * 3)]).item())
            lp_name.append(
                torch.argmax(output[(4 + provNum + alphaNum + adNum * 3):(4 + provNum + alphaNum + adNum * 4)]).item())
            lp_name.append(
                torch.argmax(output[(4 + provNum + alphaNum + adNum * 4):(4 + provNum + alphaNum + adNum * 5)]).item())

            # print(lp_name, target[4:])
            if lp_name == target[4:].tolist():
                classify_correct_num += 1
        self.classify_correct_num += classify_correct_num
        # print(det_correct_num, classify_correct_num, total_num)

        accuracy = classify_correct_num / total_num
        return ap, accuracy

    def result(self):
        ap = self.det_correct_num / self.total_num
        accuracy = self.classify_correct_num / self.total_num

        return ap, accuracy
