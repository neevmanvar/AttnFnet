# Depth to Pressure Translation Repository

This repository contains code, data, and models for research on body pressure estimation using deep learning. The repository is organized into several directories containing data sets, trained models, and supporting documentation.

```bash

AttnFnet
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

AttnFnet follows an encoder–decoder structure with skip connections (inspired by U-Net architectures) to preserve spatial details. Uniquely, AttnFNet integrates transformer layers into the network to capture global context. The depth image is first processed by convolutional layers to extract low-level features and progressively encode contextual spatial information. At the transformer block, these features are passed through a self-attention module and a convolutional projection. This transformer block enables the model to learn long-range relationships in the image (for example, relating distant body parts or overall body shape to pressure distribution). A bottleneck operation is performed to reduce computational complexity, then decoder projects encoded features back to image space.

<br/><br/>
<p align="center"> <img src="https://github.com/neevmanvar/AttnFnet/blob/main/assets/figures/PatchGAN_architecture.png" alt="patchgan" width="800"/></p>
<br/><br/>

The model is trained adversarially with a PatchGAN discriminator proposed by Isola et al., as used in conditional GAN frameworks for image translation. Method uses 62x62 patch to distinguish between real and fake probabilities maps. Training the depth-to-pressure model involves a composite loss function that balances adversarial learning with structural and pixel-wise accuracy. The AttnFNet generator $G$ and PatchGAN discriminator $D$ are optimized in a conditional GAN (cGAN) framework​. This work uses cGAN loss with mixed domain loss to translate depth into pressure images. cGAN loss comes from the original conditional GAN formulation (Mirza and Osindero, 2014) and proposed SSIML2 loss is used along with adversarial loss (more information in the paper).
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


## Installation
The code requires ```python>=3.9``` and ```pytorch>=2.6```. Please follow the instructions here to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Clone this repository to get started ```git clone https://github.com/neevmanvar/AttnFnet.git```.

go to AttnFnet repository

```cd AttnFnet```

use command ``` pip install -r requirements.txt ``` to install dependencies

### Dataset Requrements
- The model is trained and evaluated on a publicly available multimodal lying pose dataset, consisting of depth and pressure images from 102 subjects in diverse lying postures.
- Dataset details available at:
  - [Original dataset](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/)
  - [Cleaned dataset](https://doi.org/10.7910/DVN/ZS7TQS)
  - [Synthetic dataset](https://doi.org/10.7910/DVN/C6J1SP)
- Download Cleaned depth images created by Henry Clever and use depth_uncover_cleaned_0to102.npy for this project, you can include more cover images too.
- download original dataset with pressure images, go to its main directory and use script ``` xxxxxx.py ``` to get pre-processed pressure images.
- current impliment doesn't include body-mass normalization but if you want you can use it.

## Training

## Testing
