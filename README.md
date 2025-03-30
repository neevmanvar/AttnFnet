# AttnFnet
Attention Feature Network, feature aware depth to pressure translation using cGAN training with mixed loss.

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
│   └── *(Documentation files and additional resources)*
│
└── smpl
    ├── models
    ├── smpl_webuser
    └── smpl_webuser3

