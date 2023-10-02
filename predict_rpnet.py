# -*- coding: utf-8 -*-

"""
@date: 2023/9/29 上午11:41
@file: predict_rpnet.py
@author: zj
@description: 
"""

import os
import cv2
import torch
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ttf_path = "assets/fonts/HanYiZhongJianHei-2.ttf"
ttf = ImageFont.truetype(ttf_path, 30)

from rpnet import RPNet, provNum, alphaNum, adNum, provinces, alphabets, ads
from dataset import data_preprocess


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict RPNet')
    parser.add_argument('rpnet', metavar='RPNet', type=str, default="./runs/RPNet-e60.pth",
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RPNet(device, wr2_pretrained=None)
    # rpnet_pretrained = "runs/RPNet-e60.pth"
    rpnet_pretrained = args.rpnet
    print(f"Loading RPNet pretrained: {rpnet_pretrained}")
    ckpt = torch.load(rpnet_pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(data.unsqueeze(0).to(device)).cpu()
    print(outputs.shape)

    xc, yc, box_w, box_h = outputs[0][:4]
    x_min = (xc - box_w / 2) * img_w
    y_min = (yc - box_h / 2) * img_h
    x_max = (xc + box_w / 2) * img_w
    y_max = (yc + box_h / 2) * img_h

    lp_idxs = list()
    lp_idxs.append(torch.argmax(outputs[0][4:(4 + provNum)]).long().item())
    lp_idxs.append(torch.argmax(outputs[0][(4 + provNum):(4 + provNum + alphaNum)]).long().item())
    lp_idxs.append(torch.argmax(outputs[0][(4 + provNum + alphaNum):(4 + provNum + alphaNum + adNum)]).long().item())
    lp_idxs.append(
        torch.argmax(outputs[0][(4 + provNum + alphaNum + adNum):(4 + provNum + alphaNum + adNum * 2)]).long().item())
    lp_idxs.append(
        torch.argmax(
            outputs[0][(4 + provNum + alphaNum + adNum * 2):(4 + provNum + alphaNum + adNum * 3)]).long().item())
    lp_idxs.append(
        torch.argmax(
            outputs[0][(4 + provNum + alphaNum + adNum * 3):(4 + provNum + alphaNum + adNum * 4)]).long().item())
    lp_idxs.append(
        torch.argmax(
            outputs[0][(4 + provNum + alphaNum + adNum * 4):(4 + provNum + alphaNum + adNum * 5)]).long().item())

    lp_name = list()
    lp_name.append(provinces[lp_idxs[0]])
    lp_name.append(alphabets[lp_idxs[1]])
    lp_name.append(ads[lp_idxs[2]])
    lp_name.append(ads[lp_idxs[3]])
    lp_name.append(ads[lp_idxs[4]])
    lp_name.append(ads[lp_idxs[5]])
    lp_name.append(ads[lp_idxs[6]])
    lp_name = ''.join(lp_name)
    print(f"lp_name: {lp_name}")

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_draw = ImageDraw.Draw(image)
    img_draw.text((int(x_min), int(y_min) - 40), lp_name, font=ttf, fill=(255, 0, 0))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
    # cv2.putText(image, lp_name, (int(x_min), int(y_min) - 20), cv2.FONT_ITALIC, 1, (0, 0, 255))
    cv2.imshow("det", image)
    cv2.waitKey(0)

    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    save_path = f"runs/{img_name}_rpnet.jpg"
    print(f"Save to {save_path}")
    cv2.imwrite(save_path, image)
