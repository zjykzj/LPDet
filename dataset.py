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
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.logger import LOGGER
from transform import Transform


def data_preprocess(image, target_size=480):
    image = cv2.resize(image, (target_size, target_size))

    # HWC -> CHW
    data = torch.from_numpy(image).permute((2, 0, 1)).float()
    data /= 255.0

    return data


def load_data(data_root, pattern='*.json'):
    assert os.path.isdir(data_root)

    data_list = list()

    p = Path(data_root)
    # for path in tqdm(p.rglob(pattern)):
    for path in p.rglob(pattern):
        data_list.append(str(path).strip())

    return data_list


def parse_name(img_path, img_h, img_w):
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

    # return torch.from_numpy(np.array(label, dtype=float))
    return np.array(label, dtype=float)


class CCPD(Dataset):

    def __init__(self, data_root, target_size=480, is_train=True):
        self.data_root = data_root
        self.target_size = target_size
        self.is_train = is_train

        LOGGER.info(f"Get {'train' if is_train else 'val'} data: {data_root}")
        self.data_list = load_data(data_root, pattern="*.jpg")
        LOGGER.info(f"Dataset len: {len(self.data_list)}")

        self.transform = Transform(is_train=is_train)

    def __getitem__(self, index):
        img_path = self.data_list[index]

        data, target = self.build_data(img_path)
        for i in range(1000):
            if data is None:
                index = np.random.choice(1000)
                img_path = self.data_list[index]
                data, target = self.build_data(img_path)
            else:
                break

        return data, target

    def __len__(self):
        return len(self.data_list)

    def build_data(self, img_path):
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        target = parse_name(img_path, img_h, img_w)

        # labels = np.array([[0, *target[:4]]])
        # img, labels, _ = self.transform([img], [labels], self.target_size)
        # if len(labels) == 0:
        #     return None, None
        #
        # assert len(labels) == 1 and labels.shape[1] == 5, f"{img_path} {labels}"
        # target[:4] = labels[0][1:]

        data = data_preprocess(img, target_size=self.target_size)
        return data, torch.from_numpy(target)
