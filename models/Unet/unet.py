import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Type

class Unet(nn.Module):
    """
    U-Net architecture for image-to-image translation tasks.

    The network consists of an encoder (downsampling path) and a decoder (upsampling path) with skip connections.
    It supports input padding to ensure square dimensions and optional resizing to a target output shape.

    Args:
        in_shape (Tuple[int, int, int]): Input image shape as (H, W, C). Default is (512, 512, 3).
        out_shape (Tuple[int, int, int]): Output image shape as (H, W, C_out). Default is (512, 512, 1).
        kernel (int): Kernel size for the initial convolution. Default is 4.
        features (int): Base number of feature maps. Default is 64.
        act (str): Final activation type, either "tanh" or any other value for Sigmoid. Default is "sigmoid".
    """
    def __init__(self, 
                 in_shape: Tuple[int, int, int] = (512, 512, 3),
                 out_shape: Tuple[int, int, int] = (512, 512, 1),
                 kernel: int = 4,
                 features: int = 64,
                 act: str = "sigmoid"):
        super().__init__()
        self.in_shape = in_shape  # (H, W, C)
        self.out_shape = out_shape  # (H, W, C_out)
        self.kernel = kernel
        self.features = features
        
        # Calculate maximum dimension for input padding
        self.max_dim = max(in_shape[0], in_shape[1])
        self.resize_needed = (out_shape[0] != self.max_dim) or (out_shape[1] != self.max_dim)
        
        # Initial convolution layer for downsampling the input image
        self.initial_conv = nn.Conv2d(in_shape[2], features, kernel, stride=2, padding=1, bias=False)
        self.d1_act = nn.LeakyReLU(0.2)
        
        # Downsampling blocks
        self.d2 = Block(features, features * 2, down=True, act="leaky")
        self.d3 = Block(features * 2, features * 4, down=True, act="leaky")
        self.d4 = Block(features * 4, features * 8, down=True, act="leaky")
        self.d5 = Block(features * 8, features * 16, down=True, act="leaky")
        self.d6 = Block(features * 16, features * 16, down=True, act="leaky")
        
        # Bottleneck block
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 16, features * 16, kernel, stride=2, padding=1, bias=False),
            nn.ReLU()
        )
        
        # Upsampling blocks with skip connections from the encoder
        self.up1 = Block(features * 16, features * 16, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 32, features * 16, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 32, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 16, features * 4, down=False, act="relu")
        self.up5 = Block(features * 8, features * 2, down=False, act="relu")
        self.up6 = Block(features * 4, features, down=False, act="relu")
        
        # Final layer: uses a transposed convolution to produce the desired output channels.
        self.final_conv = nn.ConvTranspose2d(features * 2, out_shape[2], kernel, stride=2, padding=1, bias=False)
        self.activation = nn.Tanh() if act == "tanh" else nn.Sigmoid()
        
        # Upsample the output if the target output shape is different from the maximum padded dimension.
        if self.resize_needed:
            self.resizer = nn.Upsample(size=(out_shape[0], out_shape[1]), mode='nearest')
        
        # Initialize weights for convolution layers
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Custom weight initialization for Conv2d and ConvTranspose2d layers.
        Weights are initialized from a normal distribution and biases are set to zero.
        """
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        """
        Forward pass for the U-Net.

        This method pads the input to a square if needed, processes the image through the encoder,
        applies the bottleneck, and then reconstructs the image through the decoder while
        concatenating skip connections.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of the reconstructed image.
        """
        # Input padding to square if needed
        _, _, h, w = x.size()
        if h != w:
            pad_h = self.max_dim - h
            pad_w = self.max_dim - w
            # Pad (left, right, top, bottom)
            x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, 
                          pad_h // 2, pad_h - pad_h // 2))
            
        # Encoder path
        d1 = self.d1_act(self.initial_conv(x))
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        # Bottleneck
        b = self.bottleneck(d6)
        # Decoder path with skip connections
        up1 = self.up1(b)
        up2 = self.up2(torch.cat([up1, d6], 1))
        up3 = self.up3(torch.cat([up2, d5], 1))
        up4 = self.up4(torch.cat([up3, d4], 1))
        up5 = self.up5(torch.cat([up4, d3], 1))
        up6 = self.up6(torch.cat([up5, d2], 1))
        
        # Final output layer with skip connection from the first encoder layer
        out = self.activation(self.final_conv(torch.cat([up6, d1], 1)))
        
        # Resize output to target dimensions if necessary
        if self.resize_needed:
            out = self.resizer(out)
            
        return out


class Block(nn.Module):
    """
    A building block used in the U-Net encoder and decoder paths.

    This block can perform downsampling (using Conv2d) or upsampling (using ConvTranspose2d),
    followed by batch normalization, an activation function, and optional dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        down (bool): If True, perform downsampling using Conv2d. If False, perform upsampling using ConvTranspose2d.
        act (str): Activation type to use; "relu" uses ReLU and any other value uses LeakyReLU with negative slope 0.2.
        use_dropout (bool): If True, apply dropout with probability 0.5 after activation.
    """
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        # Select the appropriate convolution operation based on the down flag.
        self.conv = nn.Conv2d if down else nn.ConvTranspose2d
        self.layer = self.conv(
            in_channels, out_channels, 4, 2, 1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        
    def forward(self, x):
        """
        Forward pass for the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying convolution, batch normalization,
                          activation, and optionally dropout.
        """
        x = self.layer(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    

def test():
    """
    Test function to verify the Unet implementation.

    It initializes a Unet model, passes sample data through it, prints the output shape,
    performs a few training steps, and displays the model summary.
    """
    # Example usage with input shape (54, 128, 1) and output shape (27, 64, 1)
    model = Unet(in_shape=(54, 128, 1), out_shape=(27, 64, 1), kernel=4, act="sgmoid")
    x = torch.randn(1, 1, 54, 128)  # Input tensor (N, C, H, W)
    out = model(x)
    print(out.shape)  # Expected output shape might be resized to (N, C_out, H_out, W_out)
  
    # Print a summary of the model architecture
    from torchsummary import summary
    summary(model, (1, 54, 128))
    
    # Uncomment the lines below to visualize the model graph using torchview.
    # from torchview import draw_graph
    # # device='meta' -> no memory is consumed for visualization
    # model_graph = draw_graph(model, input_size=(1, 3, 27, 512), device='cpu', expand_nested=True, save_graph=True)
    # model_graph.visual_graph

if __name__ == "__main__":
    test()
