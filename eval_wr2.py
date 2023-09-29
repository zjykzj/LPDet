# -*- coding: utf-8 -*-

"""
@date: 2023/9/28 下午10:14
@file: eval_wr2.py
@author: zj
@description: 
"""

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from wr2 import wR2
from dataset import CCPD
from evaluator import CCPDEvaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Training wR2')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(val_root):
    model = wR2(num_classes=4)
    wr2_pretrained = "runs/wR2-e10.pth"
    print(f"Loading wR2 pretrained: {wr2_pretrained}")
    model.load_state_dict(torch.load(wr2_pretrained, map_location="cpu"))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    val_dataset = CCPD(val_root)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False,
                                pin_memory=True)

    ccpd_evaluator = CCPDEvaluator(only_det=True)

    pbar = tqdm(val_dataloader)
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(images)

        ap, _ = ccpd_evaluator.update(outputs, targets)
        info = "Idx:%d AP:%f" % (idx, ap)
        pbar.set_description(info)
    ap, _ = ccpd_evaluator.result()
    print(f"AP: {ap}")


def main():
    args = parse_opt()

    val(args.val_root)


if __name__ == '__main__':
    main()
