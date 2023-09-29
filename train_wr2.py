# -*- coding: utf-8 -*-

"""
@date: 2023/9/27 下午5:00
@file: train.py
@author: zj
@description: 
"""

import argparse
import math
import os.path

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from wr2 import wR2
from loss import wR2Loss
from dataset import CCPD
from evaluator import CCPDEvaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Training wR2')
    parser.add_argument('train_root', metavar='DIR', type=str, help='path to train dataset')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')
    parser.add_argument('output', metavar='OUTPUT', type=str, help='path to output')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


def train(train_root, val_root, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = wR2(num_classes=4).to(device)
    criterion = wR2Loss().to(device)

    learn_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90])

    train_dataset = CCPD(train_root)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=False,
                                  pin_memory=True)
    val_dataset = CCPD(val_root)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False,
                                pin_memory=True)

    ccpd_evaluator = CCPDEvaluator(only_det=True)

    print("Start training")
    epochs = 100
    warmup_epoch = 5
    for epoch in range(1, epochs + 1):
        # epoch: start from 1
        model.train()

        pbar = tqdm(train_dataloader)
        for idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            info = f"Epoch:{epoch} Batch:{idx} LR:{lr:.6f} Loss:{loss:.6f}"
            pbar.set_description(info)

        if epoch % 1 == 0 and epoch > 0:
            model.eval()
            save_path = os.path.join(output, f"wR2-e{epoch}.pth")
            print(f"Save to {save_path}")
            torch.save(model.state_dict(), save_path)

            pbar = tqdm(val_dataloader)
            for idx, (images, targets) in enumerate(pbar):
                images = images.to(device)
                with torch.no_grad():
                    outputs = model(images).cpu()

                ap, _ = ccpd_evaluator.update(outputs, targets)
                info = f"Batch:{idx} AP:{ap * 100:.3f}"
                pbar.set_description(info)
            ap, _ = ccpd_evaluator.result()
            print(f"AP: {ap}")
        scheduler.step()


def main():
    args = parse_opt()

    output = args.output
    if not os.path.exists(output):
        os.makedirs(output)

    train(args.train_root, args.val_root, output)


if __name__ == '__main__':
    main()
