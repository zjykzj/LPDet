# -*- coding: utf-8 -*-

"""
@date: 2023/9/28 下午1:55
@file: t_dataset.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader

from dataset import CCPD

data_set = CCPD("/home/zj/data/CCPD/CCPD2019/ccpd_base")
print(len(data_set))

img, box, lbl_name = data_set.__getitem__(100)
print(img.shape, box.shape, lbl_name)

dataloader = DataLoader(data_set, batch_size=4, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
imgs, boxs, lbl_names = iter(dataloader).__next__()

print(imgs.shape, boxs.shape, lbl_names.__len__())

for box, lbl_name in zip(boxs, lbl_names):
    print(box, lbl_name)
