import os
import torch
import numpy as np
from torch import nn
import sys
from typing import Tuple, Type
from torchsummary import summary
import math

class PatchEncoder(nn.Module):
    """
    Encodes an image into patches and adds positional encodings.

    This module splits an input image into non-overlapping patches using a convolutional projection.
    It then adds positional encoding to each patch. The positional encoding can be computed using either
    a sine-cosine formulation (relative positional encoding) or a simple increasing index.

    Args:
        image_size (int): The size of the input image (assumed square). Default is 512.
        patch_size (int): The size of each patch. Default is 16.
        num_patches (int): Total number of patches in the image. Default is 1024.
        embed_dim (int): Dimensionality of the embedding for each patch. Default is 768.
        use_rel_pos (bool): If True, use sine-cosine relative positional encoding. Otherwise, use simple encoding.
        in_chans (int): Number of input channels in the image. Default is 3.
    """
    def __init__(self, image_size: int = 512, 
                 patch_size: int = 16, 
                 num_patches: int = 1024, 
                 embed_dim: int = 768, 
                 use_rel_pos: bool = True,
                 in_chans: int = 3):
        super(PatchEncoder, self).__init__()
        self.image_size = image_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.use_rel_pos = use_rel_pos
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.encoding_size = image_size // patch_size
        self.num_patches_H = int(np.sqrt(num_patches))
        self.num_patches_W = int(np.sqrt(num_patches))
        self.projection = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, 
                                    kernel_size=(patch_size, patch_size), 
                                    stride=(patch_size, patch_size), padding=0)
        
        if self.use_rel_pos:
            pe = self._create_relative_pos_encoding()
        else:
            pe = self._create_simple_pos_encoding()

        self.register_buffer("positional_encoding", pe)

    def _create_relative_pos_encoding(self) -> torch.Tensor:
        """
        Compute sine-cosine based relative positional encodings for patches.

        Returns:
            torch.Tensor: Positional encodings of shape [1, num_patches_H, num_patches_W, embed_dim].
        """
        num_total_patches = self.num_patches_H * self.num_patches_W
        # Create a column vector of patch indices: shape [num_total_patches, 1]
        position_ids = torch.arange(num_total_patches, dtype=torch.float32).unsqueeze(1)
        # Compute the scaling factors for each dimension (only for even indices)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32) * 
                             (-math.log(10000.0) / self.embed_dim))
        # Allocate encoding tensor: shape [num_total_patches, embed_dim]
        pe = torch.zeros(num_total_patches, self.embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position_ids * div_term)
        pe[:, 1::2] = torch.cos(position_ids * div_term)
        # Reshape to [1, num_patches_H, num_patches_W, embed_dim]
        pe = pe.view(1, self.num_patches_H, self.num_patches_W, self.embed_dim)
        return pe

    def _create_simple_pos_encoding(self) -> torch.Tensor:
        """
        Compute a simple increasing index positional encoding for patches.

        Returns:
            torch.Tensor: Positional encodings of shape [1, num_patches_H, num_patches_W, embed_dim].
        """
        num_total_patches = self.num_patches_H * self.num_patches_W
        pe = torch.arange(num_total_patches, dtype=torch.float32).view(1, self.num_patches_H, self.num_patches_W, 1)
        pe = pe.expand(1, self.num_patches_H, self.num_patches_W, self.embed_dim)
        return pe

    def forward(self, image: torch.Tensor) -> Tuple:
        """
        Forward pass for PatchEncoder.

        Projects the input image into patches and adds positional encoding.

        Args:
            image (torch.Tensor): Input image tensor of shape [B, in_chans, image_size, image_size].

        Returns:
            Tuple: A tuple containing:
                - encoded (torch.Tensor): The patch embeddings with positional encoding added, 
                                          shape [B, num_patches_H, num_patches_W, embed_dim].
                - projected_patches (torch.Tensor): The patch embeddings before adding positional encoding.
        """
        device = image.device
        projected_patches = self.projection(image).to(device)
        # B C H W -> B H W C
        projected_patches = projected_patches.permute(0, 2, 3, 1)
        encoded = projected_patches + self.positional_encoding
        return encoded, projected_patches


class ImageEncoder(nn.Module):
    """
    Encodes an image using patch encoding followed by a series of transformer blocks.

    This module uses a PatchEncoder to split the image into patches and embed them,
    then processes the embeddings with transformer blocks. It optionally collects skip connections
    for later use in a decoder.

    Args:
        image_size (int): The size of the input image. Default is 512.
        input_image_size (Tuple): The original size of the input image. Default is [512, 512].
        in_chans (int): Number of input image channels. Default is 3.
        patch_size (int): Size of each patch. Default is 16.
        embed_dim (int): Embedding dimension for the patches. Default is 768.
        depth (int): Number of transformer blocks. Default is 24.
        num_heads (int): Number of attention heads in each transformer block. Default is 12.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Default is 4.0.
        out_chans (int): Number of output channels after encoding. Default is 256.
        act_layer (Type[nn.Module]): Activation layer to use. Default is nn.GELU.
        use_rel_pos (bool): If True, use relative positional encoding in PatchEncoder.
        window_size (int): Window size for local attention. Default is 0.
        global_attn_indexes (Tuple[int, ...]): Indexes of transformer blocks that use global attention.
        skip_connection_numbers (Tuple[int, ...]): Transformer block indexes from which to extract skip connections.
        use_mlp (bool): If True, use MLP based feedforward network in transformer blocks.
    """
    def __init__(self,
                 image_size: int = 512,
                 input_image_size: Tuple = [512, 512],
                 in_chans: int = 3, 
                 patch_size: int = 16, 
                 embed_dim: int = 768, 
                 depth: int = 24,        
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 out_chans: int = 256,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 global_attn_indexes: Tuple[int, ...] = (),
                 skip_connection_numbers: Tuple[int, ...] = [],
                 use_mlp: bool = True,
                 ):
        super(ImageEncoder, self).__init__()
        self.image_size = image_size
        self.input_image_size = input_image_size
        self.in_chans = in_chans
        self.depth = depth                             
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.transformer_units = [
            int(embed_dim * mlp_ratio),
            embed_dim,
        ]   # Size of the MLP layers
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.out_chans = out_chans
        self.act_layer = act_layer
        self.global_attn_indexes = global_attn_indexes
        self.skip_connection_numbers = skip_connection_numbers  # original [10, 15, 20, 24]
        # self.skip_connection_list = []
        self.head_dims = embed_dim // num_heads
        self.use_mlp = use_mlp

        self.patch_encoder = PatchEncoder(image_size=self.image_size, 
                                          patch_size=self.patch_size, 
                                          num_patches=self.num_patches, 
                                          embed_dim=self.embed_dim, 
                                          use_rel_pos=self.use_rel_pos, 
                                          in_chans=self.in_chans)

        self.transformer_block = nn.ModuleList()

        for i in range(self.depth):
            block = TransformerBlock(window_size=window_size if i not in global_attn_indexes else 0,
                                        head_dims=self.head_dims,
                                        num_heads=num_heads,
                                        transformer_units=self.transformer_units,
                                        act_layer=act_layer,
                                        dim=embed_dim,
                                        use_mlp=use_mlp)  # here sim is embeded dim
            self.transformer_block.append(block)
            
            self.neck = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),  # use from segment anything implementation
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            )

        self.resizer = nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the ImageEncoder.

        Resizes the input if necessary, applies patch encoding, processes patches through transformer blocks,
        and applies a neck (convolutional layers) to produce the final encoded output along with skip connections.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            Tuple[torch.Tensor, list]: A tuple containing:
                - x: Encoded image tensor after neck processing.
                - skip_connection_list: List of intermediate features for skip connections.
        """
        skip_connection_list = []
        if self.input_image_size[1:] != [self.image_size, self.image_size, self.in_chans]:
            x = self.resizer(x)
        x, _ = self.patch_encoder(x)
        for i, blk in enumerate(self.transformer_block):
            x = blk(x)
            if i + 1 in self.skip_connection_numbers:
                skip_connection_list.append(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x, skip_connection_list


class LayerNorm2d(nn.Module):
    """
    Implements Layer Normalization for 2D feature maps.

    Normalizes across the channel dimension for each spatial location.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D layer normalization.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block that applies multi-head self-attention and feedforward network.

    It also supports window-based partitioning of the input for local self-attention.
    """
    def __init__(self,
                 dim: int, 
                 window_size: int, 
                 head_dims: int,
                 num_heads: int,
                 transformer_units: list,
                 act_layer: Type[nn.Module],
                 use_mlp: bool):
        super(TransformerBlock, self).__init__()
        self.window_size = window_size
        self.head_dims = head_dims
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.act_layer = act_layer
        self.norm_layer1 = nn.LayerNorm(dim)
        self.norm_layer2 = nn.LayerNorm(dim)        
        
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.1, batch_first=True)

        self.feedforward = FeedForward(use_mlp=use_mlp,
                                       hidden_units=transformer_units,
                                       embed_dim=dim,
                                       act_layer=act_layer)

    def forward(self, patch):
        """
        Forward pass for the transformer block.

        Applies layer normalization, multi-head self-attention (with optional window partitioning),
        and a feedforward network. Uses residual connections throughout.

        Args:
            patch (torch.Tensor): Input tensor of shape [B, H, W, C].

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        x1 = self.norm_layer1(patch)
        B, H, W, C = x1.shape
        
        if self.window_size > 0:
            x1, pad_hw = window_partition(x1, self.window_size)
            x1 = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2], x1.shape[-1])
            attn_out, attn_out_weights = self.self_attn(x1, x1, x1)
            attn_out = window_unpartition(attn_out, self.window_size, pad_hw, (H, W))
        else:
            x1 = x1.reshape(B, H * W, C)
            attn_out, attn_out_weights = self.self_attn(x1, x1, x1)
            attn_out = attn_out.reshape(B, H, W, C)

        x2 = attn_out + patch
        x3 = self.norm_layer2(x2)
        x3 = self.feedforward(x3)
        encoded_patch = x3 + x2

        return encoded_patch

# Source: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
# Function: window_partition
# License: Apache 2.0
# Copied from Facebook Research's Segment Anything project
# Copyright (c) Meta Platforms, Inc. and affiliates.
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition the input tensor into non-overlapping windows with padding if needed.

    Args:
        x (torch.Tensor): Input tensor with shape [B, H, W, C].
        window_size (int): Size of the window.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]:
            - windows: Tensor of shape [B * num_windows, window_size, window_size, C].
            - (Hp, Wp): Tuple containing the padded height and width.
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

# Source: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
# Function: window_partition
# License: Apache 2.0
# Copied from Facebook Research's Segment Anything project
# Copyright (c) Meta Platforms, Inc. and affiliates.
def window_unpartition(windows: torch.Tensor, 
                        window_size: int, 
                        pad_hw: Tuple[int, int], 
                        hw: Tuple[int, int]) -> torch.Tensor:
    """
    Reverse the window partition operation to reconstruct the original tensor.

    Args:
        windows (torch.Tensor): Windows tensor with shape [B * num_windows, window_size, window_size, C].
        window_size (int): Size of each window.
        pad_hw (Tuple[int, int]): Padded height and width (Hp, Wp).
        hw (Tuple[int, int]): Original height and width (H, W) before padding.

    Returns:
        torch.Tensor: Reconstructed tensor of shape [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class FeedForward(nn.Module):
    """
    Implements a feedforward (MLP) block for transformer blocks.

    Depending on the use_mlp flag, the feedforward network is either implemented using
    fully-connected layers (MLP) or convolutional layers.
    """
    def __init__(self, 
                 use_mlp: bool, 
                 hidden_units: list,
                 embed_dim: int,
                 act_layer: nn.Module):
        super(FeedForward, self).__init__()
        self.use_mlp = use_mlp
        self.hidden_units = hidden_units
        self.embed_dim = embed_dim

        if use_mlp:
            self.ff_net = nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_units[0]),
                act_layer(),
                nn.Linear(self.hidden_units[0], self.hidden_units[-1]),
            )
        else:
            self.ff_net = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=4, stride=1, bias=False, padding=(0, 0)),
                nn.Dropout(p=0.1),
                nn.ZeroPad2d((2, 1, 2, 1))
            )

    def forward(self, x):
        """
        Forward pass for the feedforward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the feedforward network.
        """
        if not self.use_mlp:
            x = x.permute(0, 3, 1, 2)
        x1 = self.ff_net(x)
        if not self.use_mlp:
            x1 = x1.permute(0, 2, 3, 1)

        return x1


class Decoder(nn.Module):
    """
    Decoder network that reconstructs an image from encoded features.

    The decoder uses upscaling convolution blocks and optionally integrates skip connections
    from the encoder to refine the reconstructed image.

    Args:
        image_size (int): Size of the input feature map (assumed square). Default is 512.
        target_image_size (Tuple): Desired output image size (height, width). Default is (512, 512).
        patch_size (int): Size of the patch used in the encoder. Default is 16.
        in_chans (int): Number of input channels to the decoder. Default is 256.
        out_chans (int): Number of output channels of the reconstructed image. Default is 1.
        act (Type[nn.Module]): Activation layer used in the decoder. Default is nn.GELU.
        final_act (Type[nn.Module]): Final activation function applied to the output. Default is nn.Sigmoid.
        use_skip_connections (bool): If True, integrates skip connections from the encoder. Default is False.
        embed_dim (int): Embedding dimension from the encoder, used if skip connections are employed.
    """
    def __init__(self, image_size: int = 512,
                 target_image_size: Tuple = (512, 512),
                 patch_size: int = 16,
                 in_chans: int = 256,
                 out_chans: int = 1,
                 act: Type[nn.Module] = nn.GELU,
                 final_act: Type[nn.Module] = nn.Sigmoid,
                 use_skip_connections: bool = False,
                 embed_dim: int = 768):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoding_size = image_size // patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.act = act
        self.final_act = final_act
        self.use_skip_connections = use_skip_connections
        self.target_image_size = target_image_size

        # skip connection conv layers
        if self.use_skip_connections:
            self.skip_convs = nn.ModuleList([
                nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=1, stride=1, padding=0)
            ])
        else:
            self.skip_convs = nn.ModuleList()

        H = W = 32
        self.skip_upscale = nn.ModuleList([
            nn.Upsample(size=[H * 2, W * 2], align_corners=False, mode="bilinear"),
            nn.Upsample(size=[H * 4, W * 4], align_corners=False, mode="bilinear"),
            nn.Upsample(size=[H * 8, W * 8], align_corners=False, mode="bilinear")
        ])

        ######## calculate layer input features after reshape or interpolate
        # use only in_chans for interpolate
        up2_skip_reshape_features = in_chans // 4    # in_chans |  (H*2)/H * (W*2)/W in_chans//4 (when use reshape in skip)
        up3_skip_reshape_features = in_chans // 16   # in_chans |  (H*4)/H * (W*4)/W in_chans//16 (when use reshape in skip)
        final_skip_reshape_features = in_chans // 64 # in_chans |  (H*8)/H * (W*8)/W in_chans//64 (when use reshape in skip)

        # up scaling layers
        self.up1 = self.ConvBlock(in_chans, in_chans // 2, transpose=False) 
        self.up2 = self.ConvBlock(up2_skip_reshape_features + in_chans // 2, in_chans // 4, transpose=False) if use_skip_connections and len(self.skip_convs) > 0 else self.ConvBlock(in_chans // 2, in_chans // 4, transpose=False)
        self.up3 = self.ConvBlock(up3_skip_reshape_features + in_chans // 4, in_chans // 8, transpose=False) if use_skip_connections and len(self.skip_convs) > 1 else self.ConvBlock(in_chans // 4, in_chans // 8, transpose=False)
        
        # self.final_up = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=self.in_chans // 4, out_channels=self.out_chans, kernel_size=1, stride=2, padding=0, bias=False) 
        #     if use_skip_connections and len(self.skip_convs) > 2 else 
        #     nn.ConvTranspose2d(in_channels=self.in_chans // 8, out_channels=self.out_chans, kernel_size=1, stride=2, padding=0, bias=False),
        #     self.final_act()
        # )
        
        self.final_up = nn.Sequential(
            nn.Conv2d(in_channels= final_skip_reshape_features + in_chans // 8, out_channels=self.out_chans, kernel_size=1, stride=1, padding=0, bias=False)
            if use_skip_connections and len(self.skip_convs) > 2 else
            nn.Conv2d(in_channels= in_chans // 8, out_channels=self.out_chans, kernel_size=1, stride=1, padding=0, bias=False),
            self.final_act(),
            nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        )

        self.resizer = nn.Upsample(size=tuple(target_image_size), mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, skip_connections: torch.Tensor = None):
        """
        Forward pass for the Decoder.

        Reconstructs the image from encoded features, optionally using skip connections from the encoder.

        Args:
            x (torch.Tensor): Encoded feature map from the encoder.
            skip_connections (torch.Tensor, optional): Skip connection features from the encoder. 
                                                         Required if use_skip_connections is True.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        if self.use_skip_connections and skip_connections is None:
            raise ValueError("No skip connections found but use_skip_connections is set to True")

        if self.use_skip_connections:
            skip_connections_list = skip_connections  
        else:
            skip_connections_list = []
        
        up1 = self.up1(x)
        if len(skip_connections_list) > 0:
            up1 = self.skip_connection_append(up1, skip_connections_list[-1], self.skip_convs[0], self.skip_upscale[0])
        
        up2 = self.up2(up1)
        if len(skip_connections_list) > 1:
            up2 = self.skip_connection_append(up2, skip_connections_list[-2], self.skip_convs[1], self.skip_upscale[1])
        
        up3 = self.up3(up2)
        if len(skip_connections_list) > 2:
            up3 = self.skip_connection_append(up3, skip_connections_list[-3], self.skip_convs[2], self.skip_upscale[2])
        
        up4 = self.final_up(up3)
        if up4.shape[2:] != self.target_image_size:
            up4 = self.resizer(up4)

        # up4 = up4.permute(0, 2, 3, 1)
        return up4

    def ConvBlock(self, in_features, out_features, transpose):
        """
        Create a convolutional block used in upscaling.

        Depending on the 'transpose' flag, it either creates a transposed convolution block
        or a standard convolution block with bilinear upsampling.

        Args:
            in_features: Number of input channels.
            out_features: Number of output channels.
            transpose (bool): If True, use transposed convolution.

        Returns:
            nn.Sequential: The convolutional block.
        """
        if transpose:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                self.act()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                self.act(),
                nn.UpsamplingBilinear2d(scale_factor=(2, 2))
            )
        
    def skip_connection_append(self, layer: torch.Tensor, skip: torch.Tensor, conv_layer: nn.Conv2d, upscale_layer: nn.Upsample = None) -> torch.Tensor:
        """
        Append a skip connection to a layer.

        This method reshapes and processes the skip connection feature map before concatenating it
        with the current layer.

        Args:
            layer (torch.Tensor): The current feature map.
            skip (torch.Tensor): The skip connection feature map.
            conv_layer (nn.Conv2d): Convolution layer to process the skip connection.
            upscale_layer (nn.Upsample, optional): Upsampling layer if needed.

        Returns:
            torch.Tensor: The concatenated feature map.
        """
        ### you can directly reshape layer and concatinate to the decoder and it will work fine 
        ### or use layer upscaling to change patch size and directly feed past features to decoder
        skip = skip.permute(0, 3, 1, 2)
        skip = conv_layer(skip)   # remove for more computational complexity but adjust up layers accordingly
        if upscale_layer is None:   # Not
            reshape_skip = upscale_layer(skip) # nn.Upsample(size=[layer.shape[2], layer.shape[3]], align_corners=True, mode="bilinear")(skip)
        else:
            reshape_skip = torch.reshape(skip, (layer.shape[0], -1, layer.shape[2], layer.shape[3]))

        concate = torch.cat([layer, reshape_skip], dim=1)
        return concate


class AttnFnet(nn.Module):
    """
    A complete attention-based FNet architecture combining an image encoder and a decoder.

    The encoder processes the input image through patch encoding and transformer blocks,
    while the decoder reconstructs the image from the encoded features. Optionally, skip
    connections from the encoder are used to improve reconstruction.

    Args:
        image_size (int): Size of the input image. Default is 512.
        input_image_size (Tuple): Original input image size. Default is [512, 512].
        in_chans (int): Number of input image channels. Default is 3.
        patch_size (int): Patch size used in the encoder. Default is 16.
        embed_dim (int): Embedding dimension in the encoder. Default is 768.
        depth (int): Number of transformer blocks in the encoder. Default is 24.
        num_heads (int): Number of attention heads in transformer blocks. Default is 12.
        mlp_ratio (float): MLP ratio for transformer blocks. Default is 4.0.
        out_chans (int): Output channels from the encoder. Default is 256.
        act_layer (Type[nn.Module]): Activation layer. Default is nn.GELU.
        use_rel_pos (bool): If True, use relative positional encoding. Default is False.
        window_size (int): Window size for local attention. Default is 0.
        global_attn_indexes (Tuple[int, ...]): Transformer block indexes using global attention.
        skip_connection_numbers (Tuple[int, ...]): Transformer block indexes for skip connections.
        use_mlp (bool): If True, use MLP feedforward network in transformer blocks. Default is True.
        decoder_in_chans (int): Input channels for the decoder. Default is 256.
        decoder_out_chans (int): Output channels for the decoder. Default is 1.
        target_image_size (Tuple): Desired output image size. Default is (512, 512).
        final_act (Type[nn.Module]): Final activation function in the decoder. Default is nn.Sigmoid.
        use_skip_connections (bool): If True, use skip connections in the decoder. Default is False.
    """
    def __init__(self,
                 image_size: int = 512,
                 input_image_size: Tuple = [512, 512],
                 in_chans: int = 3, 
                 patch_size: int = 16, 
                 embed_dim: int = 768, 
                 depth: int = 24,        
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 out_chans: int = 256,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 global_attn_indexes: Tuple[int, ...] = (),
                 skip_connection_numbers: Tuple[int, ...] = [],
                 use_mlp: bool = True,
                 decoder_in_chans: int = 256,
                 decoder_out_chans: int = 1,
                 target_image_size: Tuple = (512, 512),
                 final_act: Type[nn.Module] = nn.Sigmoid,
                 use_skip_connections: bool = False):
        super(AttnFnet, self).__init__()

        self.image_encoder = ImageEncoder(image_size=image_size,
                                          input_image_size=input_image_size,
                                          in_chans=in_chans,
                                          patch_size=patch_size,
                                          embed_dim=embed_dim,
                                          depth=depth,
                                          num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,
                                          out_chans=out_chans,
                                          act_layer=act_layer,
                                          use_rel_pos=use_rel_pos,
                                          window_size=window_size,
                                          global_attn_indexes=global_attn_indexes,
                                          skip_connection_numbers=skip_connection_numbers,
                                          use_mlp=use_mlp)
        
        self.decoder = Decoder(image_size=image_size,
                               target_image_size=target_image_size,
                               patch_size=patch_size,
                               in_chans=decoder_in_chans,
                               out_chans=decoder_out_chans,
                               act=act_layer,
                               final_act=final_act,
                               use_skip_connections=use_skip_connections,
                               embed_dim=embed_dim)
    
    def forward(self, x):
        """
        Forward pass for the AttnFnet.

        Encodes the input image and then decodes it to reconstruct the output image.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Reconstructed output image.
        """
        x, skip_connections = self.image_encoder(x)
        x = self.decoder(x, skip_connections)
        return x
    

def test():
    """
    Test function to verify the functionality of the AttnFnet and its associated modules.

    It creates random input data, initializes the model, discriminator, and loss functions,
    and then performs a few training steps, printing the time taken per step and the overall epoch time.
    """
    x = np.random.uniform(0, 1, (1, 1, 54, 128))
    x = torch.tensor(x, dtype=torch.float32)  # .to(0)
    y = np.random.uniform(0, 1, (1, 1, 27, 64))
    y = torch.tensor(y, dtype=torch.float32)  # .to(0)
    loss = nn.MSELoss()
    from losses.GANLoss import GANLoss
    disc_loss_fn = GANLoss()
    from losses.GANSSIML2Loss import GANSSIML2Loss
    loss = GANSSIML2Loss()
    from models.discriminator.patchgan import PatchGAN
    from torch import optim
    y = PatchEncoder()(x)
    print("Patch encoder out shape ", y[0].shape)
    
    model = AttnFnet(window_size=4,
                     embed_dim=768,
                     depth=12,
                     num_heads=12,
                     use_rel_pos=True,
                     in_chans=1, 
                     global_attn_indexes=[2, 5, 8, 11],  # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12    # [2, 5, 8, 11]
                     use_skip_connections=True, 
                     skip_connection_numbers=[3, 6, 9], 
                     use_mlp=False,
                     target_image_size=(27, 64),
                     input_image_size=(54, 128))  # .to(0)
    disc = PatchGAN(gen_in_shape=(1, 54, 128), gen_out_shape=(1, 27, 64))  # .to(0)
    summary(model, (1, 54, 128))
    from torchview import draw_graph
    model = AttnFnet(window_size=2, global_attn_indexes=[2, 5, 8, 11], use_skip_connections=True, skip_connection_numbers=[1, 2, 3], depth=4)
    batch_size = 2
    model_graph = draw_graph(model, input_size=(batch_size, 3, 512, 512), device='cpu', expand_nested=True, save_graph=True, depth=2)
    model_graph.visual_graph

if __name__ == "__main__":
    test()
