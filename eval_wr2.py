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
    parser = argparse.ArgumentParser(description='Eval wR2')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, default="runs/wR2-e45.pth",
                        help='path to pretrained model')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(val_root, wr2_pretrained):
    model = wR2(num_classes=4)
    # wr2_pretrained = "runs/wR2-e45.pth"
    print(f"Loading wR2 pretrained: {wr2_pretrained}")
    ckpt = torch.load(wr2_pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    val_dataset = CCPD(val_root, target_size=480, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False,
                                pin_memory=True)

    ccpd_evaluator = CCPDEvaluator(only_det=True)

    pbar = tqdm(val_dataloader)
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images).cpu()

        ap, _ = ccpd_evaluator.update(outputs, targets)
        info = f"Batch:{idx} AP:{ap * 100:.3f}"
        pbar.set_description(info)
    ap, _ = ccpd_evaluator.result()
    print(f"AP:{ap * 100:.3f}")


def main():
    args = parse_opt()

    val(args.val_root, args.pretrained)


if __name__ == '__main__':
    main()
