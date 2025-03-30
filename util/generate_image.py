import torch
from torch.nn import Module
from typing import Tuple, Type, List
from matplotlib import pyplot as plt
from warnings import warn

class GenerateImage:
    """
    A class for visualizing model input images, predictions, and ground truth images.
    
    The GenerateImage class is designed to display images in a grid layout using matplotlib.
    It supports updating the visualization state by running predictions from provided models
    or using pre-computed predictions.
    
    Attributes:
        model_list (Tuple[Module, ...] or None): Tuple of PyTorch models. If provided, each model's
            predictions will be displayed. If None, only a single set of images will be shown.
        fig (plt.Figure): The matplotlib figure object.
        y_pred (List): List to store predictions.
        axes (List[plt.Axes]): List of subplot axes used for visualization.
        titles (List[str]): Titles for the image columns. Defaults to ['Input Image', 'Predicted Image', 'Ground Truth'].
        model_cols (int): Number of models (columns) to be visualized. Defaults to 1 if no models are provided.
    """
    def __init__(self, model_list: Tuple[Module, ...] = None) -> None:
        self.model_list = model_list
        self.fig = plt.figure()
        self.y_pred = []
        self.axes: List[Type[plt.subplot]] = []
        plt.ion()  # Enable interactive mode
        
        self.titles = ['Input Image', 'Predicted Image', 'Ground Truth']

        if self.model_list is not None:
            # Create a grid with one row per model; each model occupies 3 columns.
            for i in range(len(model_list) * 3):
                self.axes.append(self.fig.add_subplot(len(model_list), 3, i + 1))
        else:
            # If no model list is provided, use a single row with 3 subplots.
            for i in range(3):
                self.axes.append(self.fig.add_subplot(1, 3, i + 1))
        
        if self.model_list is not None:
            self.model_cols = len(self.model_list)
        else:
            self.model_cols = 1

    def update_state(self, y_true: List[torch.Tensor], x_inputs: List[torch.Tensor], 
                     y_preds: List[torch.Tensor] = None) -> Type[plt.figure]:
        """
        Update the visualization with new images.
        
        For each model, the method displays:
            - The input image.
            - The predicted image (either computed on the fly from the model or provided).
            - The ground truth image.
        
        Args:
            y_true (List[torch.Tensor]): List containing the ground truth image(s). Only the first element is used.
            x_inputs (List[torch.Tensor]): List of input images corresponding to each model.
            y_preds (List[torch.Tensor], optional): List of predicted images. If None, predictions are computed by
                running each model on its corresponding input image.
        
        Returns:
            plt.figure: The matplotlib figure object with the updated subplots.
        
        Warns:
            UserWarning: If the length of x_inputs does not match the number of models.
        """
        y_true = y_true[0]
        if len(x_inputs) != self.model_cols and self.model_list is not None:
            diff = self.model_cols - len(x_inputs)
            # Attempt to pad x_inputs list if lengths do not match.
            for i in range(diff):
                x_inputs.append(x_inputs[0])
            warn("Length of inputs and models is not the same; input list has been padded.", category=UserWarning)

        j = 0
        for i in range(self.model_cols):
            if y_preds is None:
                y_pred = self.model_list[i](x_inputs[i]).cpu()
            else:
                y_pred = y_preds[i].cpu()

            # Squeeze dimensions to get proper image format.
            y_true, x_input, y_pred = self.sqeeze_dim([y_true, x_inputs[i], y_pred])
            
            x_input = x_input.cpu()
            y_true = y_true.cpu()

            # Plot input image.
            self.axes[j].imshow(x_input)
            self.axes[j].set_title(self.titles[0])
            self.axes[j].set_axis_off()
            # Plot prediction.
            self.axes[j+1].imshow(y_pred)
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
            path (str): Path to save the figure.
        """
        self.fig.savefig(path)

    def get_figure(self):
        """
        Get the current matplotlib figure.
        
        Returns:
            plt.Figure: The current figure.
        """
        return self.fig

    def close(self):
        """
        Close the current figure and all related windows.
        """
        plt.close(self.fig)
        plt.close("all")

    def reset_state(self):
        """
        Clear all subplots in the current figure.
        """
        for ax in self.axes:
            ax.clear()

    def sqeeze_dim(self, X: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Adjust tensor dimensions for proper image visualization.
        
        This function permutes and squeezes tensors if necessary to ensure the image is in the
        format (H, W, C) and that any singleton dimensions are removed.
        
        Args:
            X (Tuple[torch.Tensor]): Tuple of tensors to adjust.
        
        Returns:
            Tuple[torch.Tensor, ...]: Tuple of processed tensors.
        """
        Y = []
        for x in X:
            # If channel dimension is at index 1 and width is greater than channels, permute to (B, H, W, C)
            if x.shape[1] < x.shape[-1] and len(x.shape) == 4:
                x = x.permute(0, 2, 3, 1)
            # Remove singleton batch dimension if present.
            if (len(x.shape) > 3 and x.shape[0] == 1) or (len(x.shape) < 3 and x.shape[0] == 1):
                x = torch.squeeze(x, axis=0)
            Y.append(x)
        return tuple(Y)
