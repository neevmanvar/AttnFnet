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

Clone this repository to get started ```git clone https://github.com/neevmanvar/AttnFnet.git```

Change directory ```cd AttnFnet```

use command ``` pip install -r requirements.txt ``` to install dependencies

### Dataset Requrements
- The model is trained and evaluated on a publicly available multimodal lying pose dataset, consisting of depth and pressure images from 102 subjects in diverse lying postures.
    - Dataset details available at:
        - [Original dataset](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/)
        - [Cleaned dataset](https://doi.org/10.7910/DVN/ZS7TQS)
        - [Synthetic dataset](https://doi.org/10.7910/DVN/C6J1SP)
    - Download Cleaned depth images created by Henry Clever and use depth_uncover_cleaned_0to102.npy for this project, you can include more cover images too.
    - Download original dataset with pressure images, go to its main directory and use script ``` xxxxxx.py ``` to get pre-processed pressure images.
    - Current implimentation doesn't include body-mass normalization but if you want you can use it.
    - now put files ```x_ttv.npz, y_ttv.npz, weight_measurements.csv, test_press_calib_scale.npy, train_press_calib_scale.npy,``` and ```val_press_calib_scale.npy``` into ``` dataset/ttv/depth2bp_cleaned_no_KPa/ ``` directory

- In the end pressure values and depth values must be normalized between ```0-1``` and should have 60:20:20 training, validation and test partition with 2745 training images, 900 validation and 945 test images.

## Training
#### Distributed Data Parallel Training
Attnfnet's Distributed data parallel training was built for ```torchrun``` to setup distributed environment variables from the PyTorch. Trining ``` Attnfnet ``` network requires ```attnfnet_config.py``` file provided in ```config``` directory, you can adjust hyperparameters from there. 

Run command ``` torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS train_attnfnet_ddp.py ``` to train network with distributed data parallel on a single node multi-worker. or ``` torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=$NUM_TRAINERS train_attnfnet_ddp.py ``` to train network with stacked single-node multi-worker

You can overwrite any hyperparameter value provided in ``` attnfnet_config.py ``` file by just writing ``` ++ class.parameter_name ``` for example ``` torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS train_attnfnet_ddp.py ++optimizer.learning_rate=0.0002 ```

Use same commands for ```train_unet_ddp.py``` to train ```U-Net``` network.

#### Single node Single Worker Training
Similar to DDP, you can run ```train_attnfnet.py``` using torchrun by specifying ```$NUM_TRAINERS=1``` or use command ```python3 train_attnfnet.py```

## Testing
Testing requires ```test_config.py``` file to run any code. Before metric evaluation, run ```python3 -m scripts.predict``` and ```python3 -m scripts.predict ++data.model_name=unet``` to save model predictions. predictions must be numpy array with shape (945, 1, 27, 64) with values ranging ```0-1```.

Run following commands one by one to save metric scores and calbrated scores (in KPa).

```python3 -m scripts.evaluate``` and ```python3 -m scripts.evaluate ++data.model_name=unet```

```python3 -m scripts.evaluate_depth2bp``` and ```python3 -m scripts.evaluate_depth2bp ++data.model_name=unet```

it will generate metric scores as well as model's best and worst predictions with metric scores. To evaluate both U-Net and AttnFnet together, you need to use ```evaluate_methods.py``` file.

Current repository does not include predictions from BPBnet or BPWnet, but ```evaluate_methods.py``` file contains code to use that networks too. When you don't want to compare BPBnet and BPWnet, remove all BPBnet and BPWnet variables from ```evaluate_methods.py```, it should be obvious in code. If you still struggle then I will include seperate scripts for that too.

If one wants to use BPBnet and BPWnet for comparison, you have to follow instruction on [BodyPressure](https://github.com/Healthcare-Robotics/BodyPressure) repository and train both networks.  You have to save all model predictions by going to ```BodyPressure/networks/evaluate_depthreal_slp.py --> code line 850``` and saving all test results as .npy file. output file must contain pressure values with shape ```(945, 1, 27, 64)``` in absolute pressure 0-100 KPa range. 

If one wants direct arrays then contact me to get direct predictions as well as body-mass normalized test images. save those arrays into ```assets/model_predictions/bpbnet/depth2bp_no_KPa/y_test.npz``` and ```assets/model_predictions/bpbnet/depth2bp_no_KPa/y_pred.npz```, similarly for ```BPWnet```.

All results are saved in ```assets/test_results``` directory.

