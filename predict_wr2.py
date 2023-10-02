# -*- coding: utf-8 -*-

"""
@date: 2023/9/28 下午5:43
@file: predict_wr2.py
@author: zj
@description:

Usage - Predict wR2:
    $

"""
import os.path
import cv2
import torch
import argparse

from wr2 import wR2
from dataset import data_preprocess


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict wR2')
    parser.add_argument('wr2', metavar='wR2', type=str, default="./runs/wR2-e45.pth",
                        help='path to pretrained path')
    parser.add_argument('image', metavar='IMAGE', type=str, default="./assets/1.jpg",
                        help='path to image')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


if __name__ == '__main__':
    args = parse_opt()

    # img_path = "assets/4.jpg"
    img_path = args.image
    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]
    data = data_preprocess(image)

    model = wR2(num_classes=4)
    # wr2_pretrained = "runs/wR2-e45.pth"
    wr2_pretrained = args.wr2
    print(f"Loading wR2 pretrained: {wr2_pretrained}")
    ckpt = torch.load(wr2_pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        outputs = model(data.unsqueeze(0).to(device)).cpu()
    print(outputs.shape)

    xc, yc, box_w, box_h = outputs[0]
    x_min = (xc - box_w / 2) * img_w
    y_min = (yc - box_h / 2) * img_h
    x_max = (xc + box_w / 2) * img_w
    y_max = (yc + box_h / 2) * img_h

    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
    cv2.imshow("det", image)
    cv2.waitKey(0)

    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    save_path = f"runs/{img_name}_wr2.jpg"
    print(f"Save to {save_path}")
    cv2.imwrite(save_path, image)
