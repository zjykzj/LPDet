
# TRAIN

## TRAIN

```shell
$ python segment/train.py --data ChineseLicensePlate-seg.yaml --weights yolov5n-seg.pt --img 640 --device 0 --epoch 10
...
...
      Epoch    GPU_mem   box_loss   seg_loss   obj_loss   cls_loss  Instances       Size
        9/9      2.58G   0.006408   0.005511   0.002205          0         27        640: 100%|██████████| 12923/12923 35:22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 3300/3300 15:07
                   all     105585     105585      0.993      0.994      0.992       0.85      0.993      0.994      0.992      0.823

10 epochs completed in 8.513 hours.
Optimizer stripped from runs/train-seg/exp2/weights/last.pt, 4.1MB
Optimizer stripped from runs/train-seg/exp2/weights/best.pt, 4.1MB

Validating runs/train-seg/exp2/weights/best.pt...
Fusing layers...
Model summary: 165 layers, 1879750 parameters, 0 gradients, 6.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 3300/3300 16:04
                   all     105585     105585      0.993      0.994      0.992       0.85      0.993      0.994      0.992      0.823
Results saved to runs/train-seg/exp2
```

## EVAL

```shell
$ python segment/val.py --weights yolov5n-seg_plate.pt --data ChineseLicensePlate-seg.yaml --img 640 --device 0
...
...
Model summary: 165 layers, 1879750 parameters, 0 gradients, 6.7 GFLOPs
val: Scanning /workdir/datasets/chinese_license_plate/det/CCPD2019/labels/test.cache... 105585 images, 0 backgrounds, 0 corrupt: 100%|██████████| 105585/105585 00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 3300/3300 12:24
                   all     105585     105585      0.993      0.994      0.992       0.85      0.993      0.994      0.992      0.828
Speed: 0.1ms pre-process, 0.7ms inference, 0.7ms NMS per image at shape (32, 3, 640, 640)
Results saved to runs/val-seg/exp
```

## PREDICT

### Det+Seg

```shell
$ python segment/predict.py --weights yolov5n-seg_plate.pt --source ./assets/ccpd/
segment/predict: weights=['yolov5n-seg_plate.pt'], source=./assets/ccpd/, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/predict-seg, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1, retina_masks=False
requirements: /data/zj/face/LPDet/requirements.txt not found, check failed.
YOLOv5 🚀 2024-7-22 Python-3.8.18 torch-1.10.0+cu111 CUDA:0 (NVIDIA GeForce RTX 3090, 24260MiB)

Fusing layers...
Model summary: 165 layers, 1879750 parameters, 0 gradients, 6.7 GFLOPs
image 1/4 /data/zj/face/LPDet/assets/ccpd/0290-8_4-462&542_677&655-677&626_481&655_462&571_658&542-0_0_17_32_33_25_6-96-25.jpg: 640x416 1 plate, 10.3ms
image 2/4 /data/zj/face/LPDet/assets/ccpd/03905411877394636-92_250-173&509_520&612-520&612_197&592_173&509_501&525-0_0_3_29_29_33_33_33-101-53.jpg: 640x416 1 plate, 6.6ms
image 3/4 /data/zj/face/LPDet/assets/ccpd/30475-102_79-197&428_501&586-500&586_207&515_197&428_501&492-0_0_5_24_30_31_32_33-122-444.jpg: 640x416 1 plate, 6.6ms
image 4/4 /data/zj/face/LPDet/assets/ccpd/3124-7_17-0&287_719&650-704&558_0&650_36&379_719&287-0_0_20_31_8_33_33-78-187.jpg: 640x416 1 plate, 6.6ms
Speed: 0.3ms pre-process, 7.5ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)
```

### Det+Seg+Recog

```shell
$ CUDA_VISIBLE_DEVICES=1 python3 segment/predict_plate.py --weights yolov5n-seg_plate.pt --w-for-recog crnn_tiny-plate-b512-e100.pth --source ./assets/ccpd
segment/predict_plate: weights=['yolov5n-seg_plate.pt'], source=./assets/ccpd, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/predict-seg, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1, retina_masks=False, w_for_recog=['crnn_tiny-plate-b512-e100.pth']
requirements: /data/zj/face/LPDet/requirements.txt not found, check failed.
YOLOv5 🚀 2024-7-22 Python-3.8.18 torch-1.10.0+cu111 CUDA:0 (NVIDIA GeForce RTX 3090, 24260MiB)

Fusing layers...
Model summary: 165 layers, 1879750 parameters, 0 gradients, 6.7 GFLOPs
Loading CRNN pretrained: crnn_tiny-plate-b512-e100.pth
crnn_tiny-plate-b512-e100 summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 皖A·T891G - Predict time: 7.8 ms
image 1/4 /data/zj/face/LPDet/assets/ccpd/0290-8_4-462&542_677&655-677&626_481&655_462&571_658&542-0_0_17_32_33_25_6-96-25.jpg: 640x416 1 plate, 10.3ms
Pred: 皖A·D55999 - Predict time: 7.5 ms
image 2/4 /data/zj/face/LPDet/assets/ccpd/03905411877394636-92_250-173&509_520&612-520&612_197&592_173&509_501&525-0_0_3_29_29_33_33_33-101-53.jpg: 640x416 1 plate, 6.9ms
Pred: 皖A·F06789 - Predict time: 7.3 ms
image 3/4 /data/zj/face/LPDet/assets/ccpd/30475-102_79-197&428_501&586-500&586_207&515_197&428_501&492-0_0_5_24_30_31_32_33-122-444.jpg: 640x416 1 plate, 6.9ms
Pred: 皖A·W7J99 - Predict time: 7.4 ms
image 4/4 /data/zj/face/LPDet/assets/ccpd/3124-7_17-0&287_719&650-704&558_0&650_36&379_719&287-0_0_20_31_8_33_33-78-187.jpg: 640x416 1 plate, 6.9ms
Detect+Seg Speed: 0.4ms pre-process, 7.7ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)
Recog Speed: 7.5ms per image at shape (1, 3, 48, 168)
Results saved to runs/predict-seg/exp
```