import torch
from torch.nn import Module
from typing import Tuple, Type, List
from matplotlib import pyplot as plt
from warnings import warn
import numpy as np

class GenerateToken:
    """
    A class for visualizing token outputs from a specific layer of multiple models.
    
    This class registers a forward hook on a given token layer to capture its output and then 
    displays the corresponding input image, the predicted token (aggregated along a channel dimension), 
    and the ground truth image side by side.
    
    Attributes:
        model_name_list (Tuple[Module, ...] or List[str]): A list or tuple containing model names.
        channel_dim (int): The channel dimension along which to average the token output.
        fig (plt.Figure): The matplotlib figure for visualization.
        y_pred (List): List to store predictions.
        axes (List[plt.Axes]): List of subplot axes used for visualization.
        titles (List[str]): Titles for the columns of the visualization.
        model_cols (int): Number of models (columns) to be visualized.
    """
    def __init__(self, model_name_list: Tuple[Module, ...] = None, channel_dim: int = 1) -> None:
        """
        Initialize the GenerateToken visualization object.
        
        Args:
            model_name_list (Tuple[Module, ...], optional): Tuple of model names. If not provided,
                defaults to a single model named "model".
            channel_dim (int): The channel dimension along which to compute the mean token output.
        """
        self.model_name_list = model_name_list
        self.channel_dim = channel_dim
        self.fig = plt.figure()
        self.y_pred = []
        self.axes: List[Type[plt.subplot]] = []
        plt.ion()  # Enable interactive mode
        # Titles for the columns: input image, predicted token, ground truth.
        self.titles = ['Input Image', 'Predicted Token', 'Ground Truth']

        if self.model_name_list is not None:
            # Create subplots for each model (each model uses 3 subplots)
            for i in range(len(model_name_list) * 3):
                self.axes.append(self.fig.add_subplot(len(model_name_list), 3, i + 1))
        else:
            self.model_name_list = ["model"]
            for i in range(3):
                self.axes.append(self.fig.add_subplot(1, 3, i + 1))
        
        if self.model_name_list is not None:
            self.model_cols = len(self.model_name_list)
        else:
            self.model_cols = 1

    def update_state(self, model_list: List[torch.nn.Module], token_layer: torch.nn.Module, 
                     y_true: List[torch.Tensor], x_inputs: List[torch.Tensor]) -> Type[plt.figure]:
        """
        Update the visualization state with new images and token outputs.
        
        This method registers a forward hook on the given token_layer to capture its output.
        Then for each model, it computes the prediction (token output), averages it along the specified
        channel dimension, and visualizes the input image, the predicted token output, and the ground truth.
        
        Args:
            model_list (List[torch.nn.Module]): List of models to generate predictions.
            token_layer (torch.nn.Module): The layer from which to capture the token output.
            y_true (List[torch.Tensor]): List containing ground truth images (only the first element is used).
            x_inputs (List[torch.Tensor]): List of input images for each model.
        
        Returns:
            plt.figure: The updated matplotlib figure with visualizations.
        
        Raises:
            ValueError: If the length of x_inputs does not match the number of models.
        """
        feature = {}
        def get_feature(name):
            def hook(model, input, output):
                feature[name] = output.detach()
            return hook
        
        token_layer.register_forward_hook(get_feature('target_layer_output'))

        if len(model_list) != len(self.model_name_list):
            raise ValueError("length of model_names and model_list are not same")
        
        y_true = y_true[0]
        if len(x_inputs) != self.model_cols and self.model_name_list is not None:
            diff = self.model_cols - len(x_inputs)
            # Pad the input list by appending copies if necessary.
            for i in range(diff):
                x_inputs.append(x_inputs[0])
            warn("length of inputs and models are not same; input list has been padded", category=UserWarning)

        j = 0
        for i in range(self.model_cols):
            with torch.no_grad():
                y_pred = model_list[i](x_inputs[i]).detach()
            if 'target_layer_output' in feature:
                print("Intermediate token output shape:", feature['target_layer_output'].shape)
                token_output = feature['target_layer_output'].detach().cpu()
                # Aggregate the token output by computing the mean along the specified channel dimension.
                token_output = torch.mean(token_output, dim=self.channel_dim, keepdim=True)
            
            # Squeeze extra dimensions for proper visualization.
            y_true, x_input, token_output = self.sqeeze_dim([y_true, x_inputs[i], token_output])
            x_input = x_input.cpu()
            y_true = y_true.cpu()

            # Plot input image.
            self.axes[j].imshow(x_input)
            self.axes[j].set_title(self.titles[0])
            self.axes[j].set_axis_off()
            # Plot predicted token output.
            self.axes[j+1].imshow(token_output)
            self.axes[j+1].set_title(self.titles[1])
            self.axes[j+1].set_axis_off()
            # Plot ground truth.
            self.axes[j+2].imshow(y_true)
            self.axes[j+2].set_title(self.titles[2])
            self.axes[j+2].set_axis_off()
            j = j + 3

        return self.fig

    def show(self):
        """
        Display the current figure.
        """
        plt.show()

    def save_figure(self, path):
        """
        Save the current figure to the specified file path.
        
        Args:
            path (str): File path to save the figure.
        """
        self.fig.savefig(path)
    
    def get_figure(self):
        """
        Retrieve the current matplotlib figure.
        
        Returns:
            plt.Figure: The current figure.
        """
        return self.fig
    
    def close(self):
        """
        Close the current figure and all open matplotlib windows.
        """
        plt.close(self.fig)
        plt.close("all")
    
    def reset_state(self):
        """
        Clear all axes in the current figure.
        """
        for axx in self.axes:
            axx.clear()

    def sqeeze_dim(self, X: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Squeeze and permute tensor dimensions for proper image display.
        
        This function ensures that if the tensor has a channel dimension in the wrong position,
        it will be permuted to (H, W, C). It also removes the batch dimension if it is 1.
        
        Args:
            X (Tuple[torch.Tensor]): Tuple of tensors to process.
        
        Returns:
            Tuple[torch.Tensor, ...]: Processed tensors with proper dimensions.
        """
        Y = []
        for x in X:
            if x.shape[1] < x.shape[-1] and len(x.shape) == 4:
                x = x.permute(0, 2, 3, 1)
            if (len(x.shape) > 3 and x.shape[0] == 1) or (len(x.shape) < 3 and x.shape[0] == 1):
                x = torch.squeeze(x, axis=0)
            Y.append(x)
        return tuple(Y)
