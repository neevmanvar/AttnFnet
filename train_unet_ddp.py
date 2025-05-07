from torch import nn
import torch.distributed as dist
import torch.optim as optim
import hydra
from config.unet_config import Config
from models.Unet.unet import Unet
from models.discriminator.patchgan import PatchGAN
from util.prepare_dataset import SLPDataset
from util.prepare_dataloader import prepare_dataloader
from torch.utils.data.distributed import DistributedSampler
from util.model_trainer import DDPTrainer
from losses.GANSSIML2Loss import GANSSIML2Loss
from losses.GANLoss import GANLoss
from util.load_sam_weights import LoadSAMWeights
from util.handle_dirs import HandleTrainingDir
from util.model_callbacks import ModelEncodingToken
import torch


def ddp_setup():
    """
    Initializes the distributed training process group using the NCCL backend.
    
    This function sets up the necessary communication backend for distributed data 
    parallel (DDP) training across multiple GPUs.
    """
    dist.init_process_group(backend="nccl")


def load_train_objs(cfg: Config):
    """
    Loads and initializes training objects including datasets, models, and optimizers.
    
    Parameters:
        cfg (Config): Configuration object containing parameters for data paths, model settings,
                      optimizer settings, and training parameters.
    
    Returns:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        gen_model: Generator model (Unet) instance.
        disc_model: Discriminator model (PatchGAN) instance.
        gen_optimizer: Optimizer for the generator model.
        disc_optimizer: Optimizer for the discriminator model.
    """
    # Create training and validation datasets using SLPDataset.
    train_ds = SLPDataset(dataset_path=cfg.data.path, partition="train")
    val_ds = SLPDataset(dataset_path=cfg.data.path, partition="val")

    # Initialize the generator model (Unet) with configuration parameters.
    gen_model = Unet(
        in_shape=cfg.gen_model.in_shape, 
        out_shape=cfg.gen_model.out_shape,
        kernel=cfg.gen_model.kernel,
        features=cfg.gen_model.features,
        act=cfg.gen_model.act
    )

    # Initialize the discriminator model (PatchGAN) with configuration parameters.
    disc_model = PatchGAN(
        gen_in_shape=cfg.disc_model.gen_in_shape,
        gen_out_shape=cfg.disc_model.gen_out_shape,
        patch_in_size=cfg.disc_model.patch_in_size,
        kernel=cfg.disc_model.kernel,
        features=cfg.disc_model.features
    )

    # Create Adam optimizers for both the generator and discriminator models.
    gen_optimizer = optim.Adam(
        params=gen_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta_1, 0.999)
    )
    disc_optimizer = optim.Adam(
        params=disc_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta_1, 0.999)
    )

    return train_ds, val_ds, gen_model, disc_model, gen_optimizer, disc_optimizer


@hydra.main(version_base=None, config_name="unet_config")
def my_hydra_app(cfg: Config) -> None:
    """
    Main entry point for the distributed training application using Hydra for configuration management.
    
    Steps:
        1. Prepares the training directories.
        2. Clears the CUDA cache.
        3. Sets up distributed training.
        4. Loads datasets, models, and optimizers.
        5. Prepares distributed dataloaders for training and validation.
        6. Initializes a token predictor callback to monitor specific layers of the generator.
        7. Trains the models using a distributed data parallel trainer.
        8. Cleans up the distributed process group after training.
    
    Parameters:
        cfg (Config): Configuration object provided by Hydra containing settings for data, models,
                      training parameters, and more.
    """
    # Set up the training directories as specified in the configuration.
    hd = HandleTrainingDir(cfg, clear_dir=cfg.data.clear_dir)
    
    # Clear CUDA cache to free up memory.
    torch.cuda.empty_cache()
    
    # Initialize the distributed training process group.
    ddp_setup()
    
    # Load datasets, models, and optimizers using the configuration.
    train_ds, val_ds, gen_model, disc_model, gen_optimizer, disc_optimizer = load_train_objs(cfg)
    
    # Create distributed samplers for training and validation datasets.
    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds)
    
    # Prepare distributed dataloaders for training and validation.
    train_data = prepare_dataloader(
        dataset=train_ds, 
        batch_size=cfg.trainer.batchsize, 
        is_distributed=True, 
        sampler=train_sampler
    )
    val_data = prepare_dataloader(
        dataset=val_ds, 
        batch_size=1, 
        is_distributed=True, 
        sampler=val_sampler
    )

    # Initialize a token predictor callback to monitor specific layers of the generator.
    token_predictor = ModelEncodingToken(
        val_data=val_data,
        train_data=train_data,
        use_val_data=True,
        # List of layers from which to extract tokens.
        layer_list=[gen_model.d1_act, gen_model.d3, gen_model.d6],
        # Corresponding names for the layers for clarity.
        block_name_list=["gen_model.d1", "gen_model.d3", "gen_model.d6"],
        batch_index=0,
        save_dir=hd.MODEL_INTERMEDIATE_TOKEN_PREDICION_DIR,
        model_name="unet",
        channel_dim=1
    )

    # Initialize the distributed data parallel trainer with all necessary components.
