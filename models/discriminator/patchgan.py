# Discriminator architecture based on PatchGAN
# Source: "Image-to-Image Translation with Conditional Adversarial Networks" (Isola et al., 2017)
# https://arxiv.org/abs/1611.07004

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CNNBlock(nn.Module):
    """
    A convolutional block that applies convolution, batch normalization, and LeakyReLU activation.

    This block is used as a building block in the PatchGAN discriminator to progressively
    extract features. It maintains spatial dimensions when stride is 1 by using appropriate padding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the convolution.
        kernel_size (int): Size of the convolution kernel.
    """
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super().__init__()
        # Calculate padding to maintain spatial dimensions when stride=1.
        padding = (kernel_size - 1) // 2  
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=False)
        
    def forward(self, x):
        """
        Forward pass for the CNNBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution, batch normalization, and activation.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x  # Alternatively: return self.act(self.bn(self.conv(x)))

class PatchGAN(nn.Module):
    """
    PatchGAN discriminator network.

    This discriminator is designed to classify whether overlapping patches in an image are real or fake.
    It takes as input a concatenation of the generator input and output (or real image) and processes them
    through a series of convolutional layers, ultimately producing a probability map over patches.

    Args:
        gen_in_shape (Tuple[int, int, int]): Shape of the generator input image (C, H, W).
        gen_out_shape (Tuple[int, int, int]): Shape of the generator output image (C, H, W).
        patch_in_size (Tuple[int, int]): Target spatial dimensions (H, W) to which the input is resized.
        kernel (int): Kernel size used in the convolution layers.
        features (Tuple[int, ...]): Number of features for each layer. Default is (64, 128, 256, 512).
    """
    def __init__(
        self,
        gen_in_shape: Tuple[int, int, int] = (3, 512, 512),  # (C, H, W)
        gen_out_shape: Tuple[int, int, int] = (1, 512, 512),
        patch_in_size: Tuple[int, int] = (512, 512),
        kernel: int = 3,
        features: Tuple[int, ...] = (64, 128, 256, 512)
    ):
        super().__init__()
        self.gen_in_shape = gen_in_shape
        self.gen_out_shape = gen_out_shape
        self.kernel = kernel
        self.features = features
        self.patch_in_size = patch_in_size

        # Calculate the number of input channels after concatenating generator input and output.
        input_channels = gen_in_shape[0] + gen_out_shape[0]

        # Initial convolution layer with stride 2 to downsample the input.
        self.initial_conv = nn.Conv2d(
            input_channels,
            features[0],
            kernel_size=kernel,
            stride=2,
            padding=(kernel - 1) // 2,
            bias=False
        )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)

        # Create intermediate blocks using CNNBlock.
        self.blocks = nn.ModuleList()
        in_channels = features[0]
        for i, feature in enumerate(features[1:]):
            # Use stride 1 for the last block, otherwise stride=2 for downsampling.
            stride = 1 if i == len(features[1:]) - 1 else 2
            self.blocks.append(
                CNNBlock(in_channels, feature, stride=stride, kernel_size=kernel)
            )
            in_channels = feature

        # Final output convolution layer producing a single-channel output.
        self.final_conv = nn.Conv2d(
            features[-1],
            1,
            kernel_size=kernel,
            stride=1,
            padding=(kernel - 1) // 2,
            bias=False
        )

        # Resizer to upscale inputs to the desired patch input size.
        self.resizer = nn.Upsample(size=(patch_in_size[0], patch_in_size[1]), mode='bilinear')

        # Weight initialization can be applied here if needed.
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for convolutional layers with a normal distribution.

        Args:
            module (nn.Module): A module to potentially initialize.
        """
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x1, x2):
        """
        Forward pass for the PatchGAN discriminator.

        Args:
            x1 (torch.Tensor): Generator input image tensor.
            x2 (torch.Tensor): Generator output or real image tensor.

        Returns:
            torch.Tensor: Discriminator output (patch-based probability map).
        """
        # Convert inputs to float32.
        x1 = x1.to(dtype=torch.float32)  # e.g., permute(0, 3, 1, 2) if needed.
        x2 = x2.to(dtype=torch.float32)  # e.g., permute(0, 3, 1, 2) if needed.
        
        # Resize inputs to the specified patch size.
        x1 = self.resizer(x1)
        x2 = self.resizer(x2)

        # Concatenate along the channel dimension.
        x = torch.cat([x1, x2], dim=1)
        # Forward pass through initial convolution and activation.
        x = self.leaky_relu(self.initial_conv(x))
        # Pass through intermediate CNN blocks.
        for block in self.blocks:
            x = block(x)
        # Final convolution to produce the output.
        return self.final_conv(x)  # Optionally, permute output if needed.

    
def test():
    """
    Test function to verify the functionality of the PatchGAN discriminator.

    It creates an instance of PatchGAN, generates random input tensors, performs a forward pass,
    and visualizes the model graph using torchview.
    """
    # Create discriminator with specified generator input and output shapes.
    discriminator = PatchGAN(
        gen_in_shape=(3, 256, 512),  # (C, H, W)
        gen_out_shape=(1, 512, 512),
        kernel=3,
        patch_in_size=(512, 512)
    )

    # Example inputs: x_real is the generator input and x_fake is the generator output.
    x_real = torch.randn(1, 3, 27, 512)  # Generator input shape.
    x_fake = torch.randn(1, 1, 512, 512)  # Generator output (or real image) shape.

    # Forward pass through the discriminator.
    output = discriminator(x_real, x_fake)
    # print(output.shape)  # Expected output shape, e.g., torch.Size([1, 1, 32, 64]) 

    # Visualize the model graph using torchview.
    from torchview import draw_graph

    model = PatchGAN(
        gen_in_shape=(3, 256, 512),  # (C, H, W)
        gen_out_shape=(1, 512, 512),
        kernel=3,
        patch_in_size=(512, 512)
    )

    # device='meta' ensures no memory is consumed during visualization.
    model_graph = draw_graph(model, input_data=[x_real, x_fake], device='cpu', expand_nested=True, save_graph=True, depth=2)
    model_graph.visual_graph

if __name__ == "__main__":
    test()
