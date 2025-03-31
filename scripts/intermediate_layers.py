import hydra
from config.test_config import Config
from util.handle_dirs import HandleTestDir
import sys
import os
from util.prepare_dataset import SLPDataset
from util.prepare_dataloader import prepare_dataloader
from models.AttnFnet.attnfnet import AttnFnet
from models.Unet.unet import Unet
from torch import nn
from util.manage_snapshot import load_gen_model
import torch
from alive_progress import alive_bar
import numpy as np
import matplotlib.pyplot as plt
from util.generate_token import GenerateToken

def load_model(cfg: Config):
    """
    Load the generator model based on the configuration.

    Depending on the model name provided in the configuration, this function instantiates either
    an AttnFnet or a Unet model with parameters from the configuration. Then, it loads the model
    weights from the checkpoint specified in cfg.data.ckpt_path using the utility function load_gen_model.

    Args:
        cfg (Config): Hydra configuration object containing model parameters and checkpoint path.

    Returns:
        tuple: A tuple (epoch_run, gen_model) where epoch_run is the epoch number at which the checkpoint was saved,
               and gen_model is the loaded generator model.
    """
    if cfg.data.model_name == "attnfnet":
        # Instantiate AttnFnet with parameters from the configuration.
        gen_model = AttnFnet(
            image_size=cfg.attnfnet_model.image_size,
            input_image_size=cfg.attnfnet_model.input_image_size,
            in_chans=cfg.attnfnet_model.in_chans,
            patch_size=cfg.attnfnet_model.patch_size,
            embed_dim=cfg.attnfnet_model.embed_dim,
            depth=cfg.attnfnet_model.depth,
            num_heads=cfg.attnfnet_model.num_heads,
            mlp_ratio=cfg.attnfnet_model.mlp_ratio,
            out_chans=cfg.attnfnet_model.out_chans,
            act_layer=getattr(nn, cfg.attnfnet_model.act_layer),
            use_rel_pos=cfg.attnfnet_model.use_rel_pos,
            window_size=cfg.attnfnet_model.window_size,
            global_attn_indexes=cfg.attnfnet_model.global_attn_indexes,
            skip_connection_numbers=cfg.attnfnet_model.skip_connection_numbers,
            use_mlp=cfg.attnfnet_model.use_mlp,
            decoder_in_chans=cfg.attnfnet_model.decoder_in_chans,
            decoder_out_chans=cfg.attnfnet_model.decoder_out_chans,
            target_image_size=cfg.attnfnet_model.target_image_size,
            final_act=getattr(nn, cfg.attnfnet_model.final_act),
            use_skip_connections=cfg.attnfnet_model.use_skip_connections
        )
    else:
        # Instantiate Unet with parameters from the configuration.
        gen_model = Unet(
            in_shape=cfg.unet_model.in_shape, 
            out_shape=cfg.unet_model.out_shape,
            kernel=cfg.unet_model.kernel,
            features=cfg.unet_model.features,
            act=cfg.unet_model.act
        )

    # Load the model checkpoint (generator state) from the provided checkpoint path.
    epoch_run, gen_model = load_gen_model(cfg.data.ckpt_path, gen_model)

    return epoch_run, gen_model


@hydra.main(version_base=None, config_name="test_config")
def my_hydra_app(cfg: Config) -> None:
    """
    Main function to generate and save intermediate tokens from the generator model.

    Workflow:
      1. Initialize directories and load the test dataset.
      2. Prepare the test dataloader.
      3. Load the generator model (AttnFnet or Unet) and move it to the appropriate device.
      4. Based on the model type, select a target layer for token extraction.
      5. Use the GenerateToken utility to compute an intermediate token from the target layer,
         save the generated token visualization, and then reset the token state.

    Args:
        cfg (Config): Hydra configuration object containing dataset, model, and training parameters.
    """
    # Initialize directories using the provided configuration.
    hd = HandleTestDir(cfg=cfg, clear_dirs=cfg.data.clear)
    
    # Load the test dataset.
    slptestset = SLPDataset(dataset_path=cfg.data.path, partition='test')
    # Prepare a dataloader for the test dataset (batch_size set to 1 for inference).
    test_data = prepare_dataloader(dataset=slptestset, batch_size=1, is_distributed=False)
    
    # Load the generator model and the checkpoint epoch.
    epoch_run, gen_model = load_model(cfg)

    # Check if CUDA is available and print device information.
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("Number of GPUs available:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("GPU", i, ":", torch.cuda.get_device_name(i))
    else:
        print("No GPUs available, CPU will be used.")
    
    # Set the device to CUDA if available, otherwise use CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # Move the model to the device and set it to evaluation mode.
    gen_model = gen_model.to(device).eval()

    # Preallocate an array for predictions based on the shape of the ground truth from the first batch.
    y_test_smpl = (next(iter(test_data))[1]).shape
    y_preds = np.zeros(shape=(1, y_test_smpl[1], y_test_smpl[2], y_test_smpl[3]))

    # Select the target layer from which to generate tokens based on model type.
    if cfg.data.model_name == "attnfnet":
        block_num = 11  # Specify the transformer block index.
        target_layer = gen_model.image_encoder.transformer_block[block_num].feedforward.ff_net
    elif cfg.data.model_name == "unet":
        target_layer = gen_model.d2  # For Unet, choose layer d2.
    else:
        raise ValueError("use model name attnfnet or unet")

    # Initialize the token generator utility.
    gt = GenerateToken(model_name_list=["attnfnet"], channel_dim=1)

    # Inference loop: Process only the first batch to generate and save the token visualization.
    with torch.no_grad():
        with alive_bar(len(test_data)) as bar:
            for i, tdata in enumerate(test_data):
                x, y = tdata  # Unpack input and ground truth.
                x = x.to(device)
                y = y.to(device)
                if i == 0:
                    # Update the token generator's state using the target layer and current batch.
                    gt.update_state([gen_model], target_layer, [y], [x])
                    # Determine the path to save the generated token figure.
                    path = os.path.join(hd.INTERMEDIATE_TOKEN_DIR, target_layer._get_name()).replace("\\", "/")
                    # Save the figure.
                    gt.save_figure(path=path)
                    # Reset the token generator state.
                    gt.reset_state()
                    break  # Process only the first batch.
                bar()

        print("prediction shape", y_preds.shape)


my_hydra_app()
