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
    clear: bool = False
    ckpt_path:str = os.path.join("model_checkpoints", "${data.model_name}", "${data.data_name}", "snapshot.pth").replace("\\","/")
    batch_size: int = 32

@dataclass
class AttnfnetModel:
    in_chans: int = 1
    use_mlp: bool = False
    embed_dim: int = 768
    depth: int = 12        
    num_heads: int = 12
    global_attn_indexes: Tuple = (0,1,2,3,4,5,6,7,8,9,10,11,12) # (2, 5, 8, 11)
    skip_connection_numbers: Tuple = (3, 6, 9)
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
class UnetModel:
    kernel: int = 4
    in_shape: Tuple = (54, 128, 1)
    out_shape: Tuple = (27, 64, 1)
    features: int = 64
    act: str = "sigmoid"

@dataclass
class Optimizer:
    learning_rate: float =  0.0001
    beta_1: float = 0.5

@dataclass
class Config:
    attnfnet_model: AttnfnetModel = field(default_factory=AttnfnetModel)
    unet_model: UnetModel = field(default_factory=UnetModel)
    data: Data = field(default_factory=Data)
    optimizer: Optimizer = field(default_factory=Optimizer)

cs = ConfigStore.instance()
cs.store(name="test_config", node=Config)