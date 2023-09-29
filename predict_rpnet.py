# -*- coding: utf-8 -*-

"""
@date: 2023/9/29 上午11:41
@file: predict_rpnet.py
@author: zj
@description: 
"""
import cv2

import torch

from rpnet import RPNet, provNum, alphaNum, adNum, provinces, alphabets, ads


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

    model = RPNet(device, wr2_pretrained="runs/wR2-e45.pth")
    rpnet_pretrained = "runs/RPNet-e60.pth"
    print(f"Loading RPNet pretrained: {rpnet_pretrained}")
    model.load_state_dict(torch.load(rpnet_pretrained, map_location='cpu'))
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

    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
    cv2.putText(image, lp_name, (int(x_min), int(y_min) - 20), cv2.FONT_ITALIC, 1, (0, 0, 255))
    cv2.imshow("det", image)
    cv2.waitKey(0)
