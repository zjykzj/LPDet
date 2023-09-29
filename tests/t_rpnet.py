# -*- coding: utf-8 -*-

"""
@date: 2023/9/28 下午4:44
@file: t_rpnet.py
@author: zj
@description: 
"""

import torch

from rpnet import RPNet

device = torch.device('cpu')
model = RPNet(device)
print(model)

torch.manual_seed(5)
data = torch.randn(3, 3, 480, 480)
res = model(data)

print(res.shape)
