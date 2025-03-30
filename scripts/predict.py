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

def load_model(cfg: Config):
    """
    Load and initialize the generator model based on the configuration.

    Depending on the configuration setting for data.model_name, this function
    instantiates either an AttnFnet model or a Unet model with parameters specified
    in the configuration. It then loads the model weights from a checkpoint using
    load_gen_model and returns the last run epoch and the loaded model.

    Args:
        cfg (Config): Configuration object containing model and checkpoint parameters.

    Returns:
        tuple: A tuple (epoch_run, gen_model) where epoch_run indicates the last trained epoch,
               and gen_model is the loaded generator model.
    """
    if cfg.data.model_name == "attnfnet":
        gen_model = AttnFnet(image_size=cfg.attnfnet_model.image_size,
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
                        use_skip_connections=cfg.attnfnet_model.use_skip_connections)
    else:
        gen_model = Unet(in_shape=cfg.unet_model.in_shape, 
                        out_shape=cfg.unet_model.out_shape,
                        kernel=cfg.unet_model.kernel,
                        features=cfg.unet_model.features,
                        act=cfg.unet_model.act)

    epoch_run, gen_model = load_gen_model(cfg.data.ckpt_path, gen_model)

    return epoch_run, gen_model


@hydra.main(version_base=None, config_name="test_config")
def my_hydra_app(cfg: Config) -> None:
    """
    Main application entry point for testing the generator model predictions.

    This function sets up the test directory, loads the test dataset, and prepares the dataloader.
    It then loads the generator model (AttnFnet or Unet) from a checkpoint, prints the available CUDA
    devices, and runs inference on the test dataset while tracking progress with an alive progress bar.
    Finally, the predictions are saved in compressed NumPy format.

    Args:
        cfg (Config): Configuration object loaded by Hydra.
    """
    # Set up the directory structure for test predictions.
    hd = HandleTestDir(cfg=cfg, clear_dirs=cfg.data.clear)
    prediction_array_path = hd.get_model_predictions_path()

    # Prepare the test dataset and dataloader.
    slptestset = SLPDataset(dataset_path=cfg.data.path, partition='test')
    test_data = prepare_dataloader(dataset=slptestset, batch_size=cfg.data.batch_size, is_distributed=False)

    # Load the generator model from the specified checkpoint.
    epoch_run, gen_model = load_model(cfg)

    # Display CUDA availability and device information.
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("Number of GPUs available:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("GPU", i, ":", torch.cuda.get_device_name(i))
    else:
        print("No GPUs available, CPU will be used.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # Move the generator model to the appropriate device and set it to evaluation mode.
    gen_model = gen_model.to(device).eval()

    # Prepare an empty array for storing predictions. The shape is determined by one sample from the test dataloader.
    y_test_smpl = (next(iter(test_data))[1]).shape
    y_preds = np.zeros(shape=(1, y_test_smpl[1], y_test_smpl[2], y_test_smpl[3]))

    # Inference loop with no gradient computation.
    with torch.no_grad():
        with alive_bar(len(test_data)) as bar:
            for i, tdata in enumerate(test_data):    
                x, y = tdata
                x = x.to(device)
                y = y.to(device)
                y_pred = gen_model(x)
                # Concatenate the new predictions with the stored predictions.
                y_preds = np.concatenate((y_preds, y_pred.cpu().numpy()), axis=0)
                bar()
        print("prediction shape", y_preds.shape)
        # Save the predictions (excluding the initial empty array) in a compressed NPZ file.
        np.savez_compressed(prediction_array_path, y_pred=y_preds[1:])

# Run the Hydra application.
my_hydra_app()
