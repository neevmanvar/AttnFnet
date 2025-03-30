from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from torch import nn
from typing import Type, Tuple
import os

@dataclass
class Data:
    data_name:str = "depth2bp_cleaned_no_KPa"
    path: str = os.path.join("datasets/ttv", data_name).replace("\\","/")
    trainsplit: float = 0.6
    model_name: str = "attnfnet"
    train_pred_batch_idx: int = 1# 28
    train_pred_epoch_dir:str = os.path.join("assets/training_predictions", model_name, data_name, "on_epoch_predictions").replace("\\","/")
    train_pred_batch_dir:str = os.path.join("assets/training_predictions", model_name, data_name, "on_batch_predictions").replace("\\","/")
    clear:bool = False

@dataclass
class GenModel:
    in_chans: int = 1
    use_mlp: bool = False
    embed_dim: int = 768
    depth: int = 12        
    num_heads: int = 12
    global_attn_indexes: Tuple = (2, 5, 8, 11) # (0,1,2,3,4,5,6,7,8,9,10,11,12) # (2, 5, 8, 11)
    skip_connection_numbers: Tuple = (6, 8, 10) # 6,8,10 normally 7,9,11
    image_size: int = 512
    patch_size: int = 16
    mlp_ratio: float = 4.0
    out_chans: int = 256
    act_layer: str = "GELU"
    use_rel_pos: bool = True
    window_size: int = 0
    decoder_in_chans: int = 256
    decoder_out_chans: int = 1
    final_act: str = "Sigmoid"
    use_skip_connections: bool = True
    input_image_size: Tuple = (54, 128)
    target_image_size: Tuple = (27, 64)

@dataclass
class DiscModel:
    gen_in_shape: Tuple = (1, 54, 128)
    gen_out_shape: Tuple = (1, 27, 64)
    kernel: int = 3
    patch_in_size: Tuple = (512, 512)
    features: Tuple = (64, 128, 256, 512)

@dataclass
class Optimizer:
    learning_rate: float =  0.0001
    beta_1: float = 0.5

@dataclass
class Trainer:
    max_epochs: int =  93
    batchsize: int =  1
    val_batchsize:int = 16
    save_every: int = 1
    pretrained: bool = True
    pretrained_ckpt: str = "sam_vit_b_01ec64.pth"
    snapshot_path: str = os.path.join("model_checkpoints", Data.model_name, Data.data_name, "snapshot.pth").replace("\\","/")

@dataclass
class Config:
    gen_model: GenModel = field(default_factory=GenModel)
    data: Data = field(default_factory=Data)
    disc_model: DiscModel = field(default_factory=DiscModel)
    optimizer: Optimizer = field(default_factory=Optimizer)
    trainer: Trainer = field(default_factory=Trainer)
    
cs = ConfigStore.instance()
cs.store(name="config", node=Config)