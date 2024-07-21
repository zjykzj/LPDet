# -*- coding: utf-8 -*-

"""
@Time    : 2024/7/20 16:54
@File    : ccpd2yolo.py
@Author  : zj
@Description:

Batch processing, generating YOLO format data

./images/
    train/
        file1.jpg
        file2.jpg
        ...
    val/
        file1.jpg
        file2.jpg
        ...
    test/
./labels
    train/
        file1.txt
        file2.txt
    ...
    val/
    test/

"""

import os
import shutil

import cv2
import numpy as np

from tqdm import tqdm


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
    return np.array(label, dtype=float), all_infos


def show_image():
    img_path = "../datasets/ccpd/CCPD2020/0230045572917-78_82-212&434_403&555-403&512_217&555_212&468_399&434-1_0_5_30_27_25_25_24-183-25.jpg"
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    img_name = os.path.basename(img_path)
    target, all_infos = parse_name(img_path, img_h, img_w)

    # left-top / right-bottom
    # [[x1, y1], [x2, y2]]
    box_xyxy = [[int(eel) for eel in el.split('&')] for el in all_infos[2].split('_')]
    x1, y1 = box_xyxy[0]
    x2, y2 = box_xyxy[1]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    kps_xy = [[int(eel) for eel in el.split('&')] for el in all_infos[3].split('_')]
    assert len(kps_xy) == 4 and len(kps_xy[0]) == 2, kps_xy
    for x, y in kps_xy:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def main():
    # data_root = "../datasets/ccpd/CCPD2020/ccpd_green/train"
    # cls_name = "train"
    # data_root = "../datasets/ccpd/CCPD2020/ccpd_green/val"
    # cls_name = "val"
    data_root = "../datasets/ccpd/CCPD2020/ccpd_green/test"
    cls_name = "test"

    dst_root = "../datasets/ccpd/CCPD2020/ccpd_green/yololike"
    dst_image_root = os.path.join(dst_root, 'images', cls_name)
    dst_label_root = os.path.join(dst_root, 'labels', cls_name)
    if not os.path.exists(dst_image_root):
        os.makedirs(dst_image_root)
    if not os.path.exists(dst_label_root):
        os.makedirs(dst_label_root)

    for img_name in tqdm(os.listdir(data_root)):
        img_path = os.path.join(data_root, img_name)
        assert os.path.isfile(img_path), img_path

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        target, all_infos = parse_name(img_path, img_h, img_w)

        kps_xy = [[int(eel) for eel in el.split('&')] for el in all_infos[3].split('_')]
        kps_xy = np.array(kps_xy, dtype=float).reshape(-1, 2) / np.array([img_w, img_h])
        kps_xy = kps_xy.reshape(-1).tolist()

        dst_img_path = os.path.join(dst_image_root, img_name)
        shutil.copy(img_path, dst_img_path)
        dst_label_path = dst_img_path.replace('images', 'labels').replace('.jpg', '.txt')
        np.savetxt(dst_label_path, [[0, *kps_xy]], delimiter=' ', fmt='%s')


if __name__ == '__main__':
    main()
