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

[Paper] [Project](https://www.sail.nrw/project/care-bed-robotics/) [Dataset](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/) [Dataset Cleaned](https://doi.org/10.7910/DVN/ZS7TQS) [Sunthetic Dataset](https://doi.org/10.7910/DVN/C6J1SP)

The **BodyPressure** project is focused on:

- **Data Processing & Simulation:**  
  Processing both real and synthetic datasets for body pressure mapping. The `data_BP` folder contains raw and preprocessed data in various formats (e.g., `.npy`, `.p`).

- **Deep Learning Models:**  
  Training and evaluation of convolutional neural network models to estimate body pressure distributions from input data. The `convnets` subdirectory stores several pre-trained models (saved as `.pt` files).

- **SLP & SMPL Integration:**  
  Integration of SLP (Surface Pressure) data with SMPL (Skinned Multi-Person Linear) body models. The repository includes fitted SMPL data (see `SLP_SMPL_fits`) as well as supporting SMPL model files in the `smpl` directory.

- **Synthetic Data Generation:**  
  Generation of synthetic training datasets using both standard and depth-imaging techniques. Relevant files are organized under `synth` and `synth_depth`.

- **Results & Evaluation:**  
  Results from experiments and model evaluations are stored under the `results` folder.

---
