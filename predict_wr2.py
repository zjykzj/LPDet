# -*- coding: utf-8 -*-

"""
@date: 2023/9/28 下午5:43
@file: predict_wr2.py
@author: zj
@description: 
"""

import cv2

import torch

from wr2 import wR2


def data_preprocess(image):
    image = cv2.resize(image, (480, 480))

    # HWC -> CHW
    data = torch.from_numpy(image).permute((2, 0, 1)).float()
    data /= 255.0

    return data


if __name__ == '__main__':
    img_path = "assets/4.jpg"
    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]
    data = data_preprocess(image)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = wR2(num_classes=4)
    wr2_pretrained = "runs/wR2-e45.pth"
    print(f"Loading wR2 pretrained: {wr2_pretrained}")
    model.load_state_dict(torch.load(wr2_pretrained, map_location='cpu'))
    model = model.to(device)
    model.eval()

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
