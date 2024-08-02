# -*- coding: utf-8 -*-

"""
@File    : ccpd2yolo.py
@Author  : zj
@Time    : 2024/7/20 16:54
@Description:

Download the CCPD2019 and CCPD2020 datasets and store them in the following format:

```text
.
├── CCPD2019
│   ├── ccpd_base
│   ├── ccpd_blur
│   ├── ccpd_challenge
│   ├── ccpd_db
│   ├── ccpd_fn
│   ├── ccpd_np
│   ├── ccpd_rotate
│   ├── ccpd_tilt
│   ├── ccpd_weather
│   ├── LICENSE
│   ├── README.md
│   └── splits
├── CCPD2020
│   └── ccpd_green
```

Convert CCPD data files to YOLO format. Note: The 4 key points of the license plate are used:

```text
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
```

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
    img_path = "./assets/ccpd_green/02625-94_253-242&460_494&565-494&565_256&530_242&460_485&480-0_0_3_24_24_29_25_32-76-47.jpg"
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


def save_to_dst(img_path, img_name, dst_image_root):
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


def process_ccpd2019(data_root, dst_root):
    for name in ['splits/train.txt', 'splits/val.txt', 'splits/test.txt']:
        txt_path = os.path.join(data_root, name)
        assert os.path.isfile(txt_path), txt_path
        print('*' * 100)
        print(f"Getting {txt_path} data...")

        cls_name = os.path.basename(name).split('.')[0]
        dst_image_root = os.path.join(dst_root, 'images', cls_name)
        dst_label_root = os.path.join(dst_root, 'labels', cls_name)
        if not os.path.exists(dst_image_root):
            os.makedirs(dst_image_root)
        if not os.path.exists(dst_label_root):
            os.makedirs(dst_label_root)
        print(f"Save to {dst_image_root}")

        with open(txt_path, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                if line == '':
                    continue

                img_path = os.path.join(data_root, line)
                assert os.path.isfile(img_path), img_path
                assert img_path.endswith('.jpg'), img_path

                save_to_dst(img_path, os.path.basename(img_path), dst_image_root)


def process_ccpd2020(data_root, dst_root):
    for name in ['train', 'val', 'test']:
        data_dir = os.path.join(data_root, name)
        assert os.path.isdir(data_dir), data_dir
        print('*' * 100)
        print(f"Getting {data_dir} data...")

        dst_image_root = os.path.join(dst_root, 'images', name)
        dst_label_root = os.path.join(dst_root, 'labels', name)
        if not os.path.exists(dst_image_root):
            os.makedirs(dst_image_root)
        if not os.path.exists(dst_label_root):
            os.makedirs(dst_label_root)
        print(f"Save to {dst_image_root}")

        for img_name in tqdm(os.listdir(data_dir)):
            img_path = os.path.join(data_dir, img_name)
            assert os.path.isfile(img_path), img_path
            assert img_path.endswith('.jpg'), img_path

            save_to_dst(img_path, img_name, dst_image_root)


def main():
    dst_root = "../datasets/chinese_license_plate/det"
    data_root = "../datasets/ccpd"

    ccpd2019_root = os.path.join(data_root, "CCPD2019")
    if os.path.isdir(ccpd2019_root):
        print(f"Process {ccpd2019_root}")
        process_ccpd2019(ccpd2019_root, os.path.join(dst_root, "CCPD2019"))

    ccpd2020_root = os.path.join(data_root, "CCPD2020", "ccpd_green")
    if os.path.isdir(ccpd2020_root):
        print(f"Process {ccpd2020_root}")
        process_ccpd2020(ccpd2020_root, os.path.join(dst_root, "CCPD2020"))


if __name__ == '__main__':
    main()
