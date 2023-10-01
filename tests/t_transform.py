# -*- coding: utf-8 -*-

"""
@date: 2023/9/30 下午10:06
@file: t_transform.py
@author: zj
@description: 
"""

import cv2
import numpy as np

img_path = "../assets/train/00934147509579-90_87-188&500_376&561-368&558_186&556_184&498_366&500-0_0_31_23_27_32_33-84-18.jpg"

x_min, y_min, x_max, y_max = 188, 500, 376, 561

image = cv2.imread(img_path)
img_h, img_w = image.shape[:2]
cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

x_c = (x_max + x_min) / 2. / img_w
y_c = (y_max + y_min) / 2. / img_h
box_w = x_max - x_min
box_h = y_max - y_min

label = np.array([[0, x_c, y_c, box_w / img_w, box_h / img_h]])

import numpy as np
from transform import Transform

transform = Transform(is_train=False)
image_v2, labels, shapes = transform([image], [label], 480)

x_c, y_c, box_w, box_h = labels[0][1:] * 480
x_min = x_c - box_w / 2.
y_min = y_c - box_h / 2.
x_max = x_c + box_w / 2.
y_max = y_c + box_h / 2.
cv2.rectangle(image_v2, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

cv2.imshow("image", image)
assert isinstance(image_v2, np.ndarray), type(image_v2)
cv2.imshow("transform", image_v2.astype(np.uint8))
cv2.waitKey(0)
