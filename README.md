# AttnFnet
Attention Feature Network, feature aware depth to pressure translation using cGAN training with mixed loss.
# BodyPressure Repository

This repository contains code, data, and models for research on body pressure estimation using deep learning and SMPL-based body modeling. The repository is organized into several directories that hold data sets, trained convolutional network models, SMPL models, and supporting documentation.

---

## Repository Structure

BodyPressure
├── data_BP
│   ├── convnets
│   │   ├── CAL_10665ct_128b_500e_0.0001lr.pt
│   │   ├── betanet_108160ct_128b_volfrac_500e_0.0001lr.pt
│   │   ├── resnet34_1_anglesDC_108160ct_128b_x1pm_rgangs_lb_slpb_dpns_rt_100e_0.0001lr.pt
│   │   └── resnet34_2_anglesDC_108160ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_rgangs_lb_lv2v_slpb_dpns_rt_40e_0.0001lr.pt
│   │
│   ├── mod1est_real
│   ├── mod1est_synth
│   ├── results
│   ├── SLP
│   │   └── danaLab
│   │       ├── 00001
│   │       .
│   │       └── 00102
│   │   
│   ├── slp_real_cleaned
│   │   ├── depth_uncover_cleaned_0to102.npy
│   │   ├── depth_cover1_cleaned_0to102.npy
│   │   ├── depth_cover2_cleaned_0to102.npy
│   │   ├── depth_onlyhuman_0to102.npy
│   │   ├── O_T_slp_0to102.npy
│   │   ├── slp_T_cam_0to102.npy
│   │   ├── pressure_recon_Pplus_gt_0to102.npy
│   │   └── pressure_recon_C_Pplus_gt_0to102.npy
│   │   
│   ├── SLP_SMPL_fits
│   │   └── fits
│   │       ├── p001
│   │       .
│   │       └── p102
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
---

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
