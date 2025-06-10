# HRNet Pose Estimation Model Adapter

## Introduction

This repo is a model integration between HRNet for human pose estimation and [Dataloop](https://dataloop.ai/).

HRNet (High-Resolution Network) is a state-of-the-art deep neural network for human pose estimation that maintains high-resolution representations throughout the entire process. It produces accurate keypoint detection for single and multiple persons by connecting high-to-low resolution convolutions in parallel and repeatedly exchanging information across resolutions.

## Model Available

- [HRNet-W48 (384×288)](https://drive.google.com/uc?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS) - High accuracy pose estimation model
- [HRNet-W32 (256×192)](https://drive.google.com/uc?id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38) - Balanced accuracy and speed model  
- [HRNet-W32 (256×256)](https://drive.google.com/uc?id=1_wn2ifmoQprBrFvUCDedjPON4Y6jsN-v) - Square input format model

## Requirements

- dtlpy
- torch>=1.9.0
- torchvision>=0.10.0
- opencv-python>=4.5.0
- numpy>=1.21.0
- ultralytics>=8.3.152
- matplotlib>=3.3.0
- ffmpeg-python>=0.2.0
- munkres>=1.1.4
- scipy>=1.7.0
- Pillow>=8.3.0
- gdown>=4.6.0
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the HRNet model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform.

## Deployment

After installing the pretrained model, it is necessary to deploy it, so it can be used for prediction.

## Sources and Further Reading

- [HRNet Documentation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- [Simple-HRNet Implementation](https://github.com/stefanopini/simple-HRNet)
- [COCO Keypoint Detection](https://cocodataset.org/#keypoints-2020)

## Acknowledgements

The original models paper and codebase can be found here:
- HRNet paper on [arXiv](https://arxiv.org/abs/1902.09212) and codebase on [GitHub](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
- Simple-HRNet wrapper on [GitHub](https://github.com/stefanopini/simple-HRNet).

We appreciate their efforts in advancing the field and making their work accessible to the broader community.