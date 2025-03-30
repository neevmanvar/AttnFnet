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
    model_name: str = "unet"
    train_pred_batch_idx: int = 28
    train_pred_epoch_dir:str = os.path.join("assets/training_predictions", model_name, data_name, "on_epoch_predictions").replace("\\","/")
    train_pred_batch_dir:str = os.path.join("assets/training_predictions", model_name, data_name, "on_batch_predictions").replace("\\","/")

@dataclass
class GenModel:
    kernel: int = 4
    in_shape: Tuple = (54, 128, 1)
    out_shape: Tuple = (27, 64, 1)
    features: int = 64
    act: str = "sigmoid"

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
    save_every: int = 1
    snapshot_path: str = os.path.join("model_checkpoints", Data.model_name, Data.data_name, "snapshot.pth").replace("\\","/")

@dataclass
class Config:
    gen_model: GenModel = field(default_factory=GenModel)
    data: Data = field(default_factory=Data)
    disc_model: DiscModel = field(default_factory=DiscModel)
    optimizer: Optimizer = field(default_factory=Optimizer)
    trainer: Trainer = field(default_factory=Trainer)
    
cs = ConfigStore.instance()
cs.store(name="unet_config", node=Config)