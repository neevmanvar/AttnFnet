# AttnFnet
Attention Feature Network, feature aware depth to pressure translation using cGAN training with mixed loss.
# BodyPressure Repository

This repository contains code, data, and models for research on body pressure estimation using deep learning and SMPL-based body modeling. The repository is organized into several directories that hold data sets, trained convolutional network models, SMPL models, and supporting documentation.

```bash

DPTranslation
├── assets
│   ├── test_predictions
│   ├── test_results   
│   ├── training_predictions
│   
├── config
│    ├── __init__.py
│    ├── attnfnet_config.py
│    ├── unet_config.py
│    ├── test_config.py
│    ├── paths.py
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
│   └── __init__.py
│   │   ├── GANLoss.py
│   │   ├── GANSSIML2Loss.py
│   │   └── SSIML2Loss.py
│   │   
│   ├── synth
│   │   ├── train_slp_lay_f_1to40_8549.p
│   │   .
│   │   └── train_slp_rside_m_71to80_1939.p
│   │   
│   ├── synth_depth
│   │   ├── train_slp_lay_f_1to40_8549_depthims.p
│   │   .
│   │   └── train_slp_rside_m_71to80_1939_depthims.p
│   │   
│   └── synth_meshes
│
├── docs
.
.
└── smpl
    ├── models
    ├── smpl_webuser
    └── smpl_webuser3
```

## Overview

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
