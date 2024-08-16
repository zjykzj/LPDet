# -*- coding: utf-8 -*-

"""
@Time    : 2024/8/11 16:08
@File    : gradio-ledet.py
@Author  : zj
@Description: 
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import gradio as gr
from datetime import datetime

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

from crnn_ctc.predict_plate import predict_crnn

save_root = "./runs/"
if not os.path.exists(save_root):
    os.makedirs(save_root)

# Load model
device = select_device("cpu")
model = DetectMultiBackend("./yolov5n-seg_plate.onnx", device=device, dnn=False, data=None, fp16=False)

model_recog = DetectMultiBackend("./crnn_tiny-plate.onnx", device=device, dnn=False, data=None, fp16=False)


@smart_inference_mode()
def run(
        im0,
        model,
        model_recog,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
):
    # Load model
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    ## Load CRNN-CTC Model
    recog_time = 0
    recog_num = 0

    # Run inference
    bs = 1
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    img_size = 640
    stride = 32
    auto = False
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred, proto = model(im, augment=augment, visualize=False)[:2]

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1

        imc = im0  # for save_crop
        # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        annotator = Annotator(im0, line_width=line_thickness, example=str('#京沪津'))
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Mask plotting
            annotator.masks(masks,
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=im[i])

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                # print(xyxy)
                x_min, y_min, x_max, y_max = torch.stack(xyxy).detach().cpu().numpy().astype(int)
                crop_img = imc[y_min:y_max, x_min:x_max]
                # cv2.imwrite(f"crop_{j}.jpg", crop_img)

                plate, predict_time = predict_crnn(image=crop_img, model=model_recog, device=device)
                recog_time += predict_time
                recog_num += 1

                c = int(cls)  # integer class
                # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                label = str(plate)
                annotator.box_label(xyxy, label, color=colors(c, True))
                # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

        # Stream results
        im0 = annotator.result()

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Detect+Seg Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    LOGGER.info(f"Recog Speed: {recog_time / recog_num:.1f}ms per image at shape {(1, 3, 48, 168)} ")

    return im0


def predict(inp):
    # 获取当前日期和时间
    now = datetime.now()
    # 格式化为字符串，例如 "2024-08-16_21-37-00"
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    inp.save(os.path.join(save_root, f"{formatted_time}.jpg"))

    image = np.array(inp)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = run(image, model, model_recog, device=device)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


if __name__ == '__main__':
    gr.Interface(fn=predict,
                 inputs=gr.Image(type="pil"),
                 outputs="image",
                 examples=list(glob.glob("./assets/ccpd/*.jpg"))).launch()
