import os
from config import paths
import warnings
import glob
import re
from config.attnfnet_config import Config
import shutil
import torch
from torch import nn

class LoadSAMWeights:
    """
    A helper class to load and transfer pre-trained SAM weights into a given model.
    
    This class loads a pre-trained checkpoint and selectively copies the weights into the model's
    state dictionary, including weights for the patch encoder, normalization layers, attention layers,
    feedforward network, and the neck (bottleneck) layers. After loading and mapping the weights, it
    updates the model's state dictionary.

    Args:
        model (nn.Module): The model instance whose weights are to be updated.
        pretrained_checkpoint_name (str, optional): The filename of the pre-trained checkpoint.
            Defaults to an empty string.

    Attributes:
        sd (dict): The state dictionary of the provided model.
        model (nn.Module): The model instance.
        trained_sd (dict): The state dictionary loaded from the pre-trained checkpoint.
    """
    def __init__(self, model: nn.Module, pretrained_checkpoint_name=""):
        checkpoint_path = os.path.join(paths.dirs.PRERAINED_CHECKPOINTS_DIR, pretrained_checkpoint_name).replace("\\", "/")
        self.sd = model.state_dict()
        self.model = model
        # Load the pre-trained weights; assume the checkpoint contains only weights.
        self.trained_sd = torch.load(checkpoint_path, weights_only=True)

    def load_patch_encoder_weights(self):
        """
        Load weights for the patch encoder from the pre-trained state dictionary.
        
        This method iterates over the pre-trained weights, selects those related to the image encoder's
        patch embedding (projection), and updates the corresponding weights in the model's state dictionary.
        If the target projection weight has only 3 input channels, it is assigned directly; otherwise, the weight
        is unsqueezed appropriately.
        """
        for k, v in self.trained_sd.items():
            if "image_encoder" in k and "patch_embed" in k:
                if "weight" in k:
                    # Check the shape to decide if unsqueeze is needed.
                    if self.sd['image_encoder.patch_encoder.projection.weight'].shape[1] == 3:
                        self.sd['image_encoder.patch_encoder.projection.weight'] = v
                    else:
                        self.sd['image_encoder.patch_encoder.projection.weight'] = torch.unsqueeze(v[:, 0, :, :], dim=1)
                else:
                    self.sd['image_encoder.patch_encoder.projection.bias'] = v
        print("Finished setting up total 1 patch encoder weights....")
        
    def load_normalization_weights(self):
        """
        Load weights for normalization layers from the pre-trained state dictionary.
        
        This method gathers all normalization weights and biases from the pre-trained weights for transformer blocks,
        and then assigns them in order to the corresponding normalization layers in the model's state dictionary.
        """
        norm_weight_list = []
        norm_bias_list = []        
        for k, v in self.trained_sd.items():
            if ("image_encoder" in k and "norm" in k and "block" in k):
                if "weight" in k:
                    norm_weight_list.append(v)
                else:
                    norm_bias_list.append(v)
        
        count = 0
        for k, v in self.sd.items():
            if "image_encoder" in k and "norm_layer" in k and "transformer_block" in k:
                if "weight" in k:
                    self.sd[k] = norm_weight_list[count]
                else:
                    self.sd[k] = norm_bias_list[count]
                    count += 1
        print("Finished setting up total %d normalize layer weights...." % count)

    def load_attention_weights(self):
        """
        Load weights for attention layers from the pre-trained state dictionary.
        
        This method collects the weights and biases for both the query/key/value projections (in_proj)
        and the output projection (out_proj) of the attention mechanism in the transformer blocks. It then
        maps these weights to the corresponding layers in the model's state dictionary.
        """
        attn_in_weights_list = []
        attn_in_biases_list = []
        attn_out_weights_list = []
        attn_out_biases_list = []

        for k, v in self.trained_sd.items():
            if ("image_encoder" in k and "attn" in k and "qkv" in k):
                if "weight" in k:
                    attn_in_weights_list.append(v)
                else:
                    attn_in_biases_list.append(v)
                    
            if ("image_encoder" in k and "attn" in k and "proj" in k):
                if "weight" in k:
                    attn_out_weights_list.append(v)
                else:
                    attn_out_biases_list.append(v)
        
        count = 0
        for k, v in self.sd.items():
            if ("image_encoder" in k and "self_attn" in k and "in_proj" in k):
                if "weight" in k:
                    self.sd[k] = attn_in_weights_list[count]
                else:
                    self.sd[k] = attn_in_biases_list[count]
            if ("image_encoder" in k and "self_attn" in k and "out_proj" in k):
                if "weight" in k:
                    self.sd[k] = attn_out_weights_list[count]
                else:
                    self.sd[k] = attn_out_biases_list[count]
                    count += 1
        print("Finished setting up total %d attention head weights...." % count)

    def load_neck_weights(self):
        """
        Load weights for the neck (bottleneck) layers from the pre-trained state dictionary.
        
        This method maps the weights and biases of the neck's convolutional and normalization layers
        from the pre-trained state dictionary to the corresponding entries in the model's state dictionary.
        """
        for k, v in self.trained_sd.items():
            if ("image_encoder" and "neck.0" in k):
                self.sd['image_encoder.neck.0.weight'] = v
            if ("image_encoder" and "neck.1" in k):
                if "weight" in k:
                    self.sd['image_encoder.neck.1.weight'] = v
                else:
                    self.sd['image_encoder.neck.1.bias'] = v
            if ("image_encoder" and "neck.2" in k):
                self.sd['image_encoder.neck.2.weight'] = v
            if ("image_encoder" and "neck.3" in k):
                if "weight" in k:
                    self.sd['image_encoder.neck.3.weight'] = v
                else:
                    self.sd['image_encoder.neck.3.bias'] = v
        print("Finished setting up total 4 bottleneck weights....")

    def load_feedforward_weights(self):
        """
        Load weights for the feedforward (MLP) network layers from the pre-trained state dictionary.
        
        This method collects all the weights and biases associated with the feedforward layers
        (denoted by 'mlp' in the key) and maps them sequentially into the model's state dictionary.
        """
        mlp_weights_list = []
        mlp_bias_list = []
        for k, v in self.trained_sd.items():
            if ("image_encoder" in k and "mlp" in k and "weight" in k):
                mlp_weights_list.append(v)
            if ("image_encoder" in k and "mlp" in k and "bias" in k):
                mlp_bias_list.append(v)

        count = 0
        for k, v in self.sd.items():
            if "image_encoder" in k and "mlp" in k and "weight" in k:
                self.sd[k] = mlp_weights_list[count]
            elif "image_encoder" in k and "mlp" in k and "bias" in k:
                self.sd[k] = mlp_bias_list[count]
                count += 1
        print("Finished setting up total %d feedforward network weights...." % count)

    def __call__(self):
        """
        Load all SAM weights into the model and update the model's state dictionary.
        
        The method sequentially loads weights for the patch encoder, normalization layers,
        attention layers, feedforward network, and neck layers, then updates the model with these weights.
        
        Returns:
            nn.Module: The model with updated weights.
        """
        self.load_patch_encoder_weights()
        self.load_normalization_weights()
        self.load_attention_weights()
        self.load_feedforward_weights()
        self.load_neck_weights()

        self.model.load_state_dict(state_dict=self.sd)
        # Uncomment the following lines for debugging if needed:
        # print(self.sd.keys())
        # print(self.trained_sd.keys())
        # print(self.trained_sd['image_encoder.neck.0.weight'].shape)
        # print(self.sd['image_encoder.neck.0.weight'].shape)
        # print(self.trained_sd['image_encoder.neck.2.weight'].shape)
        # print(self.sd['image_encoder.neck.2.weight'].shape)
        return self.model
    

def test():
    """
    Test the LoadSAMWeights class by loading pre-trained weights into an AttnFnet model.
    
    This function instantiates an AttnFnet model with specific parameters, creates a LoadSAMWeights instance
    with a given checkpoint filename, and then calls the instance to load the weights.
    """
    from models.AttnFnet.attnfnet import AttnFnet
    model = AttnFnet(in_chans=1, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11],
                     skip_connection_numbers=[3, 6, 9], use_skip_connections=True, use_mlp=False)
    lw = LoadSAMWeights(model, "sam_vit_b_01ec64.pth")
    lw()

if __name__ == "__main__":
    test()
