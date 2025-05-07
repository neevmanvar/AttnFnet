# Import necessary modules from PyTorch and other libraries
from torch import nn
import torch.distributed as dist  # For distributed training using NCCL backend
import torch.optim as optim       # Optimizers for model training
import hydra                     # For configuration management
from config.attnfnet_config import Config  # Custom configuration class for model and training parameters

# Import custom model definitions
from models.AttnFnet.attnfnet import AttnFnet     # Generator model architecture (Attention FNet)
from models.discriminator.patchgan import PatchGAN  # Discriminator model architecture (PatchGAN)

# Import dataset and dataloader preparation utilities
from util.prepare_dataset import SLPDataset       # Custom dataset loader for the specific dataset
from util.prepare_dataloader import prepare_dataloader  # Helper function to create dataloaders

# Import distributed sampler to use with distributed training
from torch.utils.data.distributed import DistributedSampler

# Import the trainer class for distributed data parallel training
from util.model_trainer import DDPTrainer

# Import custom loss functions for GAN training
from losses.GANSSIML2Loss import GANSSIML2Loss
from losses.GANLoss import GANLoss

# Import helper function to load pretrained SAM weights into the generator model
from util.load_sam_weights import LoadSAMWeights

# Import utility to handle training directory (e.g., cleaning or preparing folders)
from util.handle_dirs import HandleTrainingDir

# Import a callback to monitor or modify model encoding tokens during training
from util.model_callbacks import ModelEncodingToken

import torch  # Additional torch functionalities if needed

def ddp_setup():
    """
    Initializes the process group for distributed training using the NCCL backend.
    This setup is necessary for multi-GPU training.
    """
    dist.init_process_group(backend="nccl")

def load_train_objs(cfg: Config):
    """
    Loads and initializes training objects including datasets, models, and optimizers.
    
    Parameters:
        cfg (Config): The configuration object containing parameters for data, models, and training.
    
    Returns:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        gen_model: Generator model (AttnFnet) instance.
        disc_model: Discriminator model (PatchGAN) instance.
        gen_optimizer: Optimizer for the generator model.
        disc_optimizer: Optimizer for the discriminator model.
    """
    # Load the training and validation datasets using a custom dataset class
    train_ds = SLPDataset(dataset_path=cfg.data.path, partition="train")
    val_ds = SLPDataset(dataset_path=cfg.data.path, partition="val")

    # Initialize the generator model (AttnFnet) with configuration parameters
    gen_model = AttnFnet(
        image_size=cfg.gen_model.image_size,
        input_image_size=cfg.gen_model.input_image_size,
        in_chans=cfg.gen_model.in_chans,
        patch_size=cfg.gen_model.patch_size,
        embed_dim=cfg.gen_model.embed_dim,
        depth=cfg.gen_model.depth,
        num_heads=cfg.gen_model.num_heads,
        mlp_ratio=cfg.gen_model.mlp_ratio,
        out_chans=cfg.gen_model.out_chans,
        act_layer=getattr(nn, cfg.gen_model.act_layer),
        use_rel_pos=cfg.gen_model.use_rel_pos,
        window_size=cfg.gen_model.window_size,
        global_attn_indexes=cfg.gen_model.global_attn_indexes,
        skip_connection_numbers=cfg.gen_model.skip_connection_numbers,
        use_mlp=cfg.gen_model.use_mlp,
        decoder_in_chans=cfg.gen_model.decoder_in_chans,
        decoder_out_chans=cfg.gen_model.decoder_out_chans,
        target_image_size=cfg.gen_model.target_image_size,
        final_act=getattr(nn, cfg.gen_model.final_act),
        use_skip_connections=cfg.gen_model.use_skip_connections
    )
    
    # Optionally load pretrained weights into the generator model if specified in the config
    if cfg.trainer.pretrained:
        gen_model = LoadSAMWeights(gen_model, pretrained_checkpoint_name=cfg.trainer.pretrained_ckpt)()

    # Initialize the discriminator model (PatchGAN) with configuration parameters
    disc_model = PatchGAN(
        gen_in_shape=cfg.disc_model.gen_in_shape,
        gen_out_shape=cfg.disc_model.gen_out_shape,
        patch_in_size=cfg.disc_model.patch_in_size,
        kernel=cfg.disc_model.kernel,
        features=cfg.disc_model.features
    )

    # Create optimizers for both the generator and discriminator models
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

    # Optionally, one could compile the models with torch.compile for performance optimizations (currently commented out)
    # gen_model = torch.compile(gen_model)
    # disc_model = torch.compile(disc_model)

    return train_ds, val_ds, gen_model, disc_model, gen_optimizer, disc_optimizer

@hydra.main(version_base=None, config_name="config")
def my_hydra_app(cfg: Config) -> None:
    """
    Main entry point for the training application using Hydra for configuration management.
    
    Steps:
    1. Prepares training directories and clears them if necessary.
    2. Sets up distributed training.
    3. Loads datasets, models, and optimizers.
    4. Prepares distributed dataloaders.
    5. Initializes a token predictor callback for model analysis.
    6. Trains the models using a distributed data parallel (DDP) trainer.
    7. Cleans up the distributed process group after training.
    
    Parameters:
        cfg (Config): The configuration object loaded by Hydra.
    """
    # Set up and manage the training directories based on the configuration
    hd = HandleTrainingDir(cfg, clear_dirs=cfg.data.clear_dir)
    
    # Clear CUDA cache to free up GPU memory
    torch.cuda.empty_cache()
    
    # Initialize distributed data parallel (DDP) process group
    ddp_setup()
    
    # Load training and validation datasets, models, and optimizers using the provided configuration
    train_ds, val_ds, gen_model, disc_model, gen_optimizer, disc_optimizer = load_train_objs(cfg)
    
    # Create distributed samplers for training and validation datasets to ensure proper data splitting
    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds)
    
    # Prepare the dataloaders using the custom function with distributed samplers
    train_data = prepare_dataloader(
        dataset=train_ds,
        batch_size=cfg.trainer.batchsize,
        is_distributed=True,
        sampler=train_sampler
    )
    val_data = prepare_dataloader(
        dataset=val_ds,
        batch_size=cfg.trainer.val_batchsize,
        is_distributed=True,
        sampler=val_sampler
    )
    
    # Initialize a callback to predict or log encoding tokens from specific layers of the generator's image encoder.
    # This may be used for monitoring intermediate outputs or for further analysis.
    token_predictor = ModelEncodingToken(
        val_data=val_data,
        train_data=train_data,
        use_val_data=True,
        # List of specific feedforward network layers from different transformer blocks to monitor
        layer_list=[
            gen_model.image_encoder.transformer_block[1].feedforward.ff_net, 
            gen_model.image_encoder.transformer_block[5].feedforward.ff_net,
            gen_model.image_encoder.transformer_block[11].feedforward.ff_net
        ],
        # Corresponding names for each of the transformer block layers
        block_name_list=[
            "gen_model.image_encoder.transformer_block.1.ff_net",
            "gen_model.image_encoder.transformer_block.5.ff_net",
            "gen_model.image_encoder.transformer_block.11.ff_net"
        ],
        batch_index=0,  # Index of the batch to use for encoding token prediction
        save_dir=hd.MODEL_INTERMEDIATE_TOKEN_PREDICION_DIR,  # Directory to save token predictions
        model_name="attnfnet",  # Name of the model used in token predictions
        channel_dim=1  # The dimension representing channels in the token predictions
    )

    # Initialize the distributed data parallel trainer with all training components
    trainer = DDPTrainer(
        gen_model,            # Generator model
        disc_model,           # Discriminator model
        gen_optimizer,        # Generator optimizer
        disc_optimizer,       # Discriminator optimizer
        GANSSIML2Loss(weight_ssim=100),  # Custom loss function combining GAN, SSIM, and L2 losses
        GANLoss(label_smoothing_factors=[0.9, 0.1]),  # GAN loss with label smoothing
        train_data,           # Training dataloader
        val_data,             # Validation dataloader
        cfg,                  # Configuration object
        token_predictor       # Callback for model encoding token prediction
    )
    
    # Enable anomaly detection to help identify any issues during backpropagation
    with torch.autograd.set_detect_anomaly(True):
        # Start the training process for a maximum number of epochs as specified in the configuration
        trainer.train(cfg.trainer.max_epochs)
    
    # Clean up and destroy the distributed process group after training is complete
    dist.destroy_process_group()

# Entry point for the Hydra-based training application.
my_hydra_app()
