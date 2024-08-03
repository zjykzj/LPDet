
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