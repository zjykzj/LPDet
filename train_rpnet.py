# -*- coding: utf-8 -*-

"""
@date: 2023/9/27 下午5:00
@file: train.py
@author: zj
@description:

Usage - Single-GPU training:
    $ python train_rpnet.py ../datasets/CCPD2019/ccpd_base ../datasets/CCPD2019/ccpd_weather runs

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 32532 train_rpnet.py --device 4,5,6,7 --wr2-pretrained runs/wR2-e45.pth ../datasets/CCPD2019/ccpd_base ../datasets/CCPD2019/ccpd_weather runs

"""

import argparse
import os.path
import time

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed

from rpnet import RPNet
from loss import RPNetLoss
from dataset import CCPD
from evaluator import CCPDEvaluator
from utils.torchutil import select_device
from utils.ddputil import smart_DDP
from utils.logger import LOGGER

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt():
    parser = argparse.ArgumentParser(description='Training wR2')
    parser.add_argument('train_root', metavar='DIR', type=str, help='path to train dataset')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')
    parser.add_argument('output', metavar='OUTPUT', type=str, help='path to output')

    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--wr2-pretrained', type=str, default="runs/wR2-e45.pth", help='path to wR2 pretrained')

    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")
    return args


def train(train_root, val_root, batch_size, output, device, wr2_pretrained):
    LOGGER.info("=> Create Model")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RPNet(device=device, wr2_pretrained=wr2_pretrained).to(device)
    criterion = RPNetLoss().to(device)

    learn_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90])

    LOGGER.info("=> Load data")
    train_dataset = CCPD(train_root)
    sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True and sampler is None,
                                  sampler=sampler,
                                  num_workers=4,
                                  drop_last=False,
                                  pin_memory=True)
    if RANK in {-1, 0}:
        val_dataset = CCPD(val_root)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False,
                                    pin_memory=True)

        LOGGER.info("=> Load evaluator")
        ccpd_evaluator = CCPDEvaluator(only_det=False)

    LOGGER.info("=> Start training")
    t0 = time.time()

    # DDP mode
    cuda = device.type != 'cpu'
    if cuda and RANK != -1:
        model = smart_DDP(model)

    epochs = 100
    start_epoch = 1
    warmup_epoch = 5
    for epoch in range(start_epoch, epochs + start_epoch):
        # epoch: start from 1
        model.train()
        if RANK != -1:
            train_dataloader.sampler.set_epoch(epoch)

        pbar = train_dataloader
        if LOCAL_RANK in {-1, 0}:
            pbar = tqdm(pbar)
        for idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss = criterion(outputs, targets)
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if RANK in {-1, 0}:
                lr = optimizer.param_groups[0]["lr"]
                info = f"Epoch:{epoch} Batch:{idx} LR:{lr:.6f} Loss:{loss:.6f}"
                pbar.set_description(info)

        if RANK in {-1, 0} and epoch % 5 == 0 and epoch > 0:
            model.eval()
            save_path = os.path.join(output, f"RPNet-e{epoch}.pth")
            LOGGER.info(f"Save to {save_path}")
            torch.save(model.state_dict(), save_path)

            pbar = tqdm(val_dataloader)
            for idx, (images, targets) in enumerate(pbar):
                images = images.to(device)
                with torch.no_grad():
                    outputs = model(images).cpu()

                ap, acc = ccpd_evaluator.update(outputs, targets)
                info = f"Batch:{idx} AP:{ap * 100:.3f} ACC: {acc * 100:.3f}"
                pbar.set_description(info)
            ap, acc = ccpd_evaluator.result()
            LOGGER.info(f"AP:{ap * 100:.3f} ACC: {acc * 100:.3f}")
        scheduler.step()
    LOGGER.info(f'\n{epochs} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')


def main():
    opt = parse_opt()

    output = opt.output
    if not os.path.exists(output):
        os.makedirs(output)

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with LPDet Multi-GPU DDP training'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    train(opt.train_root, opt.val_root, opt.batch_size, output, device, opt.wr2_pretrained)


if __name__ == '__main__':
    main()
