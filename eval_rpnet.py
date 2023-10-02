# -*- coding: utf-8 -*-

"""
@date: 2023/9/29 下午12:15
@file: eval_rpnet.py
@author: zj
@description: 
"""

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from rpnet import RPNet
from dataset import CCPD
from evaluator import CCPDEvaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Eval RPNet')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, default="runs/wR2-e45.pth",
                        help='path to pretrained model')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(val_root, rpnet_pretrained):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = RPNet(device, wr2_pretrained="runs/wR2-e45.pth")
    # rpnet_pretrained = "runs/RPNet-e60.pth"
    model = RPNet(device, wr2_pretrained=None)
    print(f"Loading RPNet pretrained: {rpnet_pretrained}")
    ckpt = torch.load(rpnet_pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

    val_dataset = CCPD(val_root)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False,
                                pin_memory=True)

    ccpd_evaluator = CCPDEvaluator(only_det=False)

    pbar = tqdm(val_dataloader)
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images).cpu()

        ap, acc = ccpd_evaluator.update(outputs, targets)
        info = f"Batch:{idx} AP:{ap * 100:.3f} ACC: {acc * 100:.3f}"
        pbar.set_description(info)
    ap, acc = ccpd_evaluator.result()
    print(f"AP:{ap * 100:.3f} ACC: {acc * 100:.3f}")


def main():
    args = parse_opt()

    val(args.val_root, args.pretrained)


if __name__ == '__main__':
    main()
