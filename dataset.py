# -*- coding: utf-8 -*-

"""
@date: 2023/9/27 下午5:04
@file: dataset.py
@author: zj
@description: 
"""

import os
import cv2

import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset


def data_preprocess(image):
    image = cv2.resize(image, (480, 480))

    # HWC -> CHW
    data = torch.from_numpy(image).permute((2, 0, 1)).float()
    data /= 255.0

    return data


def load_data(data_root, pattern='*.json'):
    assert os.path.isdir(data_root)

    data_list = list()

    p = Path(data_root)
    for path in tqdm(p.rglob(pattern)):
        data_list.append(str(path).strip())

    return data_list


def build_data(img_path):
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, (480, 480))

    # HWC -> CHW
    data = torch.from_numpy(img).permute((2, 0, 1)).float()
    data /= 255.0

    return data, (img_h, img_w)


def build_target(img_path, img_h, img_w):
    # img_name: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    img_name = os.path.basename(img_path)

    # ['025', '95_113', '154&383_386&473', '386&473_177&454_154&383_363&402', '0_0_22_27_27_33_16', '37', '15']
    all_infos = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    # print(f"all_infos: {all_infos}")

    # left-top / right-bottom
    # [[x1, y1], [x2, y2]]
    box_xyxy = [[int(eel) for eel in el.split('&')] for el in all_infos[2].split('_')]
    x1, y1 = box_xyxy[0]
    x2, y2 = box_xyxy[1]

    x_c = (x1 + x2) / 2.
    y_c = (y1 + y2) / 2.
    box_w = x2 - x1
    box_h = y2 - y1
    label = [x_c / img_w, y_c / img_h, box_w / img_w, box_h / img_h]

    label.extend([int(x) for x in all_infos[4].split("_")])

    return torch.from_numpy(np.array(label, dtype=float))


class CCPD(Dataset):

    def __init__(self, data_root, only_lp=False):
        self.data_root = data_root
        self.only_lp = only_lp

        print(f"Get Data: {data_root}")
        self.data_list = load_data(data_root, pattern="*.jpg")
        print(f"Dataset len: {len(self.data_list)}")

    def __getitem__(self, index):
        img_path = self.data_list[index]

        image, (img_h, img_w) = build_data(img_path)
        label = build_target(img_path, img_h, img_w)

        return image, label

    def __len__(self):
        return len(self.data_list)
