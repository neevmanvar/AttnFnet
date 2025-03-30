import torch
from typing import Dict, List, Tuple, Type
import time
import pickle
import os
from torch.utils.data import DataLoader
from util.prepare_dataset import SLPDataset
from util.prepare_dataloader import prepare_dataloader
from torch.nn import Module
from util.generate_image import GenerateImage
from util.generate_token import GenerateToken

class ModelSanityCheck:
    """
    A class to perform a sanity check on model predictions during training.
    
    This class uses a validation DataLoader to forward a single batch (specified by batch_index)
    through the provided model. It then visualizes the input image, ground truth, and the model's
    prediction using the GenerateImage class, saving the result as an image file.
    
    Attributes:
        epoch_dir (str): Directory where epoch-level prediction images will be saved.
        batch_dir (str): Directory where batch-level prediction images will be saved.
        val (DataLoader): Validation DataLoader.
        device: Device on which to run the model (e.g., 'cuda' or 'cpu').
        batch_index (int): Index of the batch to be visualized.
        gi (GenerateImage): Instance of GenerateImage for visualization.
    """
    def __init__(self, val_data: DataLoader, batch_index: int, save_epoch_dir: str, save_batch_dir: str, device):
        self.epoch_dir = save_epoch_dir
        self.batch_dir = save_batch_dir
        self.val = val_data
        self.device = device
        self.batch_index = batch_index
        self.gi = GenerateImage()

    def forward(self, model: Module, step: int, epoch: int, save_batch_dir=True):
        """
        Forward a single batch through the model and save the prediction image.
        
        The method iterates through the validation DataLoader until it reaches the batch 
        specified by self.batch_index. It then runs the model on the first sample of that batch,
        visualizes the input, prediction, and ground truth using GenerateImage, and saves the image.
        
        Parameters:
            model (Module): The model to be evaluated.
            step (int): The current training step.
            epoch (int): The current epoch number.
            save_batch_dir (bool): If True, saves the image in the batch predictions directory,
                                   otherwise in the epoch predictions directory.
        
        Side Effects:
            Saves an image file with the model's prediction.
        """
        with torch.no_grad():
            batch_count = 0
            for x, y in self.val:
                x, y = x.to(self.device), y.to(self.device)
                # Use only the first sample if batch size > 1.
                if x.shape[0] > 1:
                    x = x[0:1]
                    y = y[0:1]
                if batch_count == self.batch_index:
                    pred = model(x).detach()
                    self.gi.update_state([y], [x], [pred])
                    if save_batch_dir:
                        path = os.path.join(self.batch_dir, f"step_{step}_epoch_{epoch}.png").replace("\\", "/")
                        print("model save path: ", path)
                    else:
                        path = os.path.join(self.epoch_dir, f"step_{step}_epoch_{epoch}.png").replace("\\", "/")
                        print("model save path: ", path)
                    self.gi.save_figure(path)
                    print(f"model prediction is saved on step {step} and epoch {epoch}")
                    self.gi.reset_state()
                    break
                batch_count += 1

class ModelEncodingToken:
    """
    A class to extract and visualize intermediate token outputs (encodings) from specified layers of a model.
    
    This class uses a DataLoader (either validation or training data) to obtain a single batch of data.
    It then iterates over a list of layers (and corresponding block names) to extract token outputs via a forward hook.
    The token outputs are processed and visualized using GenerateToken.
    
    Attributes:
        save_dir (str): Directory where the token prediction images will be saved.
        ds (DataLoader): The DataLoader used for inference (validation or training).
        layer_list (Tuple[torch.nn.Module, ...]): Tuple of layers from which tokens are to be extracted.
        block_name_list (Tuple[str, ...]): Tuple of block names corresponding to each layer.
        device: Device to perform inference.
        batch_index (int): Batch index to use for extracting tokens.
        gt (GenerateToken): Instance of GenerateToken for visualizing token outputs.
    """
    def __init__(self, val_data: DataLoader, train_data: DataLoader = None,
                 use_val_data: bool = True, 
                 layer_list: Tuple[torch.nn.Module, ...] = [],
                 block_name_list: Tuple[str, ...] = [], 
                 batch_index: int = 0, 
                 save_dir: str = "",
                 model_name: str = "model",
                 channel_dim: int = 1,
                 device = 0):
        
        self.save_dir = save_dir
        self.ds = val_data if use_val_data else train_data
        self.layer_list = layer_list
        self.block_name_list = block_name_list
        self.device = device
        self.batch_index = batch_index
        self.gt = GenerateToken(model_name_list=[model_name], channel_dim=channel_dim)

        if len(block_name_list) != len(layer_list):
            raise ValueError("length of the block names and layer length must be same")

    def forward(self, model: Module, epoch: int):
        """
        Forward a single batch through the model and extract intermediate token outputs.
        
        For each layer in the layer_list, the method registers a forward hook (via GenerateToken)
        to capture the token output from that layer and then visualizes and saves the token image.
        
        Parameters:
            model (Module): The model from which tokens are to be extracted.
            epoch (int): The current epoch number (used in the saved filename).
        
        Side Effects:
            Saves token prediction images for each specified layer.
        """
        with torch.no_grad():
            batch_count = 0
            for x, y in self.ds:
                x, y = x.to(self.device), y.to(self.device)
                # Use only the first sample if batch size > 1.
                if x.shape[0] > 1:
                    x = x[0:1]
                    y = y[0:1]
                if batch_count == self.batch_index:
                    layer_count = 0
                    for layer in self.layer_list:
                        # For each layer, update the state of GenerateToken to capture the token output.
                        self.gt.update_state([model], layer, [y], [x])
                        path = os.path.join(self.save_dir, f"{self.block_name_list[layer_count]}_batch_{self.batch_index}_epoch_{epoch}.png").replace("\\", "/")
                        self.gt.save_figure(path)
                        print(f"model token is saved on epoch {epoch}")
                        self.gt.reset_state()
                        layer_count += 1
                    break
                batch_count += 1

def test():
    """
    Test function for the ModelSanityCheck class.
    
    This function creates a validation dataset and dataloader using SLPDataset and prepare_dataloader.
    It then instantiates an AttnFnet model and runs the ModelSanityCheck on a specific batch index.
    The generated predictions are saved (if directories are provided) and printed to the console.
    """
    val_ds = SLPDataset("datasets/ttv/depth2bp_cleaned_no_KPa", partition="val")
    val_data = prepare_dataloader(val_ds, batch_size=1, is_distributed=False)
    from models.AttnFnet.attnfnet import AttnFnet
    model = AttnFnet(window_size=2, global_attn_indexes=[2, 5, 8, 11],
                     in_chans=1,
                     use_skip_connections=True, 
                     skip_connection_numbers=[3, 6, 9], 
                     use_mlp=False, 
                     target_image_size=(27,64), 
                     input_image_size=(54, 128))
    schk = ModelSanityCheck(val_data, batch_index=15, save_epoch_dir="", save_batch_dir="")
    for i in range(3):
        schk.forward(model, step=i, epoch=1)

if __name__ == "__main__":
    test()
