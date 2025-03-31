# Depth to Pressure Translation Repository

This repository contains code, data, and models for research on body pressure estimation using deep learning. The repository is organized into several directories containing data sets, trained models, and supporting documentation.

```bash

DPTranslation
├── assets
│   ├── test_predictions
│   ├── test_results   
│   └── training_predictions
│
├── config
│    ├── __init__.py
│    ├── attnfnet_config.py
│    ├── unet_config.py
│    ├── test_config.py
│    └── paths.py
│
├── datasets
│       └── ttv
│            └── depth2bp_cleaned_no_KPa
│                ├── x_ttv.npz
│                ├── y_ttv.npz
│                ├── weight_measurements.csv
│                ├── test_press_calib_scale.npy
│                ├── train_press_calib_scale.npy
│                └── val_press_calib_scale.npy       
│
├── losses
│   ├── __init__.py
│   ├── GANLoss.py
│   ├── GANSSIML2Loss.py
│   └── SSIML2Loss.py   
│
├── metrics
│   ├── __init__.py
│   .
│   .
│   └── MeanMPerPixelAcc.py
│     
├── model_checkpoints
│   ├── attnfnet
│   └── unet
│
├── models
│   ├── AttnFnet
│   ├── discriminator
│   └── Unet
│
├── pretrained_checkpoints
│   └── sam_vit_b_01ec64.pth
│
├── runs
├── scripts
├── util
├── train_attnfnet_ddp.py
├── train_attnfnet.py
└── train_unet_ddp.py
```

## AttnFnet
Neevkumar Manavar, Hanno Gerd Meyer, Joachim Waßmuth, Barbara Hammer, Axel Schneider

[Paper] [[Project](https://www.sail.nrw/project/care-bed-robotics/)] [[Dataset](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/)] [[Dataset Cleaned](https://doi.org/10.7910/DVN/ZS7TQS)] [[Synthetic Dataset](https://doi.org/10.7910/DVN/C6J1SP)]

<p align="center"><img src="https://github.com/neevmanvar/AttnFnet/blob/main/assets/figures/encoding_feature_representation.gif" alt="encoding_token_representation"/></p>

### Overview

The **Attention Feature Network (AttnFnet)** is a transformer-based deep neural network designed for translating single-depth images into pressure distribution maps, with particular applications in medical monitoring.

Pressure injuries significantly impact bedridden patients, leading to severe health complications. Timely monitoring and accurate prediction of pressure distribution can prevent these injuries. AttnFnet effectively generates precise pressure maps from depth images, providing an essential tool for real-time patient monitoring.

<img src="https://github.com/neevmanvar/AttnFnet/blob/main/assets/figures/AttnFnet_architecture.png" alt="attnfnet architecture"/>

AttnFnet employs transformer layers integrated with convolutional projections to capture global context and local feature representations. The architecture includes skip connections between encoder and decoder to retain contextual features. The depth image is divided into patches, and conventional projections are obtained and sinosoidal postion embedding is added, all patches then pass through 12 transformer layers with convolutional feed forward, at last encodings pass though bottleneck to reduce computational complexity and decoder up convolute to image space to get pressure projection.
<br/><br/>
<p align="center"> <img src="https://github.com/neevmanvar/AttnFnet/blob/main/assets/figures/PatchGAN_architecture.png" alt="patchgan" width="800"/></p>
<br/><br/>
This work utilizes cGAN loss with mixed domain loss to translate depth representation into pressure representation; structural-similarity-index loss is used with L2 norm loss as a mixed domain loss. This work uses the PatchGAN architecture described by Isola et al. to distinguish fake and real images.
<br/><br/>

```bash
@article{mirza2014conditional,
  title={Conditional Generative Adversarial Nets},
  author={Mirza, Mehdi and Osindero, Simon},
  journal={arXiv preprint arXiv:1411.1784},
  year={2014}
}

@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1125--1134},
  year={2017}
}
```
## Dataset
- The model is trained and evaluated on a publicly available multimodal lying pose dataset, consisting of depth and pressure images from 102 subjects in diverse lying postures.
- Dataset details available at:
  - [Original dataset](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/)
  - [Cleaned dataset](https://doi.org/10.7910/DVN/ZS7TQS)
  - [Synthetic dataset](https://doi.org/10.7910/DVN/C6J1SP)

