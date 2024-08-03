<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

<div align="center"><a title="" href="https://github.com/zjykzj/LPDet"><img align="center" src="assets/icons/LPDet.svg" alt=""></a></div>

<p align="center">
  «LPDet» provides a complete License Plate Detection and Recognition algorithm
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

Implementing license plate detection, segmentation, and recognition functions based on [YOLOv5-v7.0](https://github.com/ultralytics/yolov5/releases/tag/v7.0) and [CRNN-CTC](https://github.com/zjykzj/crnn-ctc)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [News🚀🚀🚀](#news)
- [Background](#background)
- [Usage](#usage)
  - [Detect + Seg](#detect--seg)
  - [Recog](#recog)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## News🚀🚀🚀

| Version                                                       | Release Date | Major Updates                                                                                                                           |
|---------------------------------------------------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [v1.1.0](https://github.com/zjykzj/LPDet/releases/tag/v1.1.0) | 2024/08/04   | Optimize license plate segmentation and recognition algorithms.                                                                         |
| [v1.0.0](https://github.com/zjykzj/LPDet/releases/tag/v1.0.0) | 2024/07/21   | Implementing license plate detection, segmentation, and recognition functions.                                                          |
| [v0.3.0](https://github.com/zjykzj/LPDet/releases/tag/v0.3.0) | 2023/10/03   | Support for Automatic Mixed Precision (AMP) training.                                                                                   |
| [v0.2.0](https://github.com/zjykzj/LPDet/releases/tag/v0.2.0) | 2023/10/02   | Support for distributed training with multi-GPUs.                                                                                       |
| [v0.1.0](https://github.com/zjykzj/LPDet/releases/tag/v0.1.0) | 2023/09/29   | Reconstruct the [872699467/CCPD_CNN](https://github.com/872699467/CCPD_CNN) implementation to adapt to interfaces after Pytorch v1.0.0. |

## Background

This warehouse provides a complete license plate detection and recognition algorithm, with the goal of perfectly detecting and recognizing all license plates and license plate information.

## Usage

### Detect + Seg

* ChineseLicensePlate: [Baidu Drive](https://pan.baidu.com/s/1fQh0E9c6Z4satvrEthKevg)(ad7l)

```shell
# Train
$ python segment/train.py --data ChineseLicensePlate-seg.yaml --weights yolov5n-seg.pt --img 640 --epoch 10
# Eval
$ python segment/val.py --weights yolov5n-seg_plate.pt --data ChineseLicensePlate-seg.yaml --img 640
# Predict
$ python segment/predict.py --weights yolov5n-seg_plate.pt --source ./assets/ccpd_green/
```

<img src="assets/results/predict/02625-94_253-242&460_494&565-494&565_256&530_242&460_485&480-0_0_3_24_24_29_25_32-76-47.jpg" alt="Image1" style="width: 200px;"> <img src="assets/results/predict/03521267361111111-104_252-253&406_497&551-493&551_257&476_253&406_497&474-0_0_3_27_33_33_31_24-135-132.jpg" alt="Image 2" style="width: 200px;">

### Recog

Train license plate recognition algorithm using [zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc)

```shell
$ git submodule init
$ git submodule update
```

Then predicting license plates

```shell
$ python3 segment/predict_plate.py --weights yolov5n-seg_plate.pt --w-for-recog crnn_tiny-plate-b512-e100.pth --source ./assets/ccpd_green/
```

<img src="assets/results/recog/02625-94_253-242&460_494&565-494&565_256&530_242&460_485&480-0_0_3_24_24_29_25_32-76-47.jpg" alt="Image 1" style="width: 200px;"> <img src="assets/results/recog/03521267361111111-104_252-253&406_497&551-493&551_257&476_253&406_497&474-0_0_3_27_33_33_31_24-135-132.jpg" alt="Image 2" style="width: 200px;">

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [detectRecog/CCPD](https://github.com/detectRecog/CCPD)
* [872699467/CCPD_CNN](https://github.com/872699467/CCPD_CNN)
* [zjykzj/FastestDet](https://github.com/zjykzj/FastestDet)
* [zjykzj/YOLOv5](https://github.com/zjykzj/YOLOv5)
* [zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/LPDet/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj