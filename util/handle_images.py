import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
import torch
from typing import Tuple

class HistogramComparison:
    """
    A class to compute and compare histograms of images in a specified color space.

    The class supports images in LAB, RGB, or GRAY color spaces. For LAB, an RGB-to-YUV
    conversion is applied. It computes normalized histograms for each channel and provides
    methods to plot and compare histograms of a ground truth image and one or more predicted images.
    
    Attributes:
        color_space (str): Color space used for histogram computation ('LAB', 'RGB', or 'GRAY').
    """
    def __init__(self, color_space='LAB'):
        """
        Initialize the HistogramComparison class.
        
        Parameters:
            color_space (str): 'LAB', 'RGB', or 'GRAY'. Determines the color space for the histogram computation.
        """
        self.color_space = color_space.upper()

    def calculate_histogram(self, image, channels=None, bins=256, value_range=(0.0, 1.0)):
        """
        Calculate histograms for the image channels using PyTorch.
        
        Parameters:
            image: Input image in either RGB, LAB, or grayscale, with values in range [0, 1].
                   Expected shape (H, W) for grayscale or (H, W, C) for multi-channel.
            channels: Number of channels in the image. If None, it is determined automatically.
            bins (int): Number of bins for the histogram.
            value_range (tuple): The range of values to compute the histogram over, default is (0.0, 1.0).
        
        Returns:
            List[np.array]: A list of normalized histograms for each channel.
        """
        image = torch.tensor(image)
        if channels is None:
            if len(image.shape) == 2:
                channels = 1
            else:
                channels = image.shape[-1]

        # If using LAB color space and image has 3 channels, perform an RGB to YUV conversion.
        if self.color_space == 'LAB' and channels == 3:
            # Ensure the image is in (H, W, 3) format.
            if image.dim() == 3 and image.size(2) != 3:
                image = image.permute(1, 2, 0)
            
            # RGB to YUV conversion matrix (from TensorFlow's implementation).
            kernel = torch.tensor([
                [0.299, -0.14714119, 0.61497538],
                [0.587, -0.28886916, -0.51496512],
                [0.114, 0.43601035, -0.10001026]
            ], dtype=image.dtype, device=image.device)
            
            bias = torch.tensor([0.0, 0.5, 0.5], dtype=image.dtype, device=image.device)
            image = torch.matmul(image, kernel) + bias

        histograms = []
        for i in range(channels):
            # Extract data for the i-th channel.
            if channels > 1:
                channel_data = image[:, :, i]
            else:
                channel_data = image

            # Compute histogram using torch.histc.
            hist = torch.histc(channel_data, bins=bins, min=value_range[0], max=value_range[1])
            
            # Normalize the histogram so that the sum of bins is 1.
            if hist.sum() > 0:
                hist = hist / hist.sum()
            
            # Convert to numpy array and append to list.
            histograms.append(hist.cpu().numpy())

        return histograms
    
    def plot_histograms(self, histograms, legends, titles, main_title, save_path):
        """
        Plot and save histograms for each channel.
        
        Parameters:
            histograms: List of histograms for each prediction set. The first element should correspond to the ground truth.
            legends: List of legend labels for the plots.
            titles: List of titles for each subplot (one per channel).
            main_title: Main title for the entire figure.
            save_path: Path where the plot image will be saved.
        """
        # Determine channel labels based on the chosen color space.
        channels = ['L', 'a', 'b'] if self.color_space == 'LAB' else ['R', 'G', 'B']
        if self.color_space == 'GRAY':
            channels = ['Gray']

        plt.figure(figsize=(15, 5))
        plt.suptitle(main_title)
        
        # Create a subplot for each channel.
        for i, channel in enumerate(channels):
            plt.subplot(1, len(channels), i + 1)
            # Plot the histogram curve for each set.
            for hist in histograms:
                plt.plot(hist[i])
            plt.yscale('log')
            plt.title(titles[i])
            plt.legend(legends)
        
        plt.savefig(save_path)
        plt.close()  # Close the plot to free memory

    def compare(self, y_true, *y_preds, legends, main_title='Histogram Comparison', titles=None, save_path='comparison.png'):
        """
        Compare histograms for a ground truth image and one or more predicted images.
        
        Parameters:
            y_true: Ground truth image.
            y_preds: One or more predicted images.
            legends: List of legend labels for the plots.
            main_title: Main title for the entire figure.
            titles: List of titles for each subplot. If None, defaults to ['L', 'a', 'b'] for LAB,
                    ['R', 'G', 'B'] for RGB, or ['Gray'] for grayscale.
            save_path: Path where the plot image will be saved.
        """
        # Squeeze dimensions using the helper function.
        y_true = sqeeze_dim([y_true])[0]
        y_preds = sqeeze_dim(y_preds)

        # Determine number of channels.
        channels = 1 if len(y_true.shape) == 2 else y_true.shape[-1]

        # Calculate histograms for ground truth and each prediction.
        hist_true = self.calculate_histogram(y_true, channels=channels)
        hist_preds = [self.calculate_histogram(y_pred, channels=channels) for y_pred in y_preds]

        # Combine into a list for plotting.
        histograms = [hist_true] + hist_preds

        # Default subplot titles.
        if titles is None:
            if self.color_space == 'LAB':
                titles = ['L', 'a', 'b']
            elif self.color_space == 'RGB':
                titles = ['R', 'G', 'B']
            else:
                titles = ['Gray']

        # Generate the plot.
        self.plot_histograms(histograms, legends, titles, main_title, save_path)


class ImageComparison:
    """
    A class for visualizing and comparing images and plots.
    
    Provides methods to display image comparisons (input, prediction, ground truth) and to generate
    various comparison plots including scatter plots, error distributions, bar plots (e.g., MAE, R2 score),
    and combined visualizations.
    """
    def __init__(self):
        pass  # Placeholder for future functionality such as image rescaling.

    def show_images(self, image_list, save_path, main_title='Image Comparison', titles=None):
        """
        Display and save comparison images.
        
        Parameters:
            image_list: List containing [input image, predicted image, ground truth image].
            save_path (str): Path where the comparison image will be saved.
            main_title (str): Title for the entire figure.
            titles (List[str], optional): Titles for each subplot. Defaults to ['Input Image', 'Predicted Image', 'Ground Truth'].
        """
        if titles is None:
            titles = ['Input Image', 'Predicted Image', 'Ground Truth']

        # Ensure all images have the same dimensions using sqeeze_dim.
        x_input, y_true, y_predict = sqeeze_dim(image_list)

        fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey='none')
        fig.suptitle(main_title)

        axs[0].imshow(x_input, cmap='gray' if x_input.ndim == 2 else None)
        axs[0].set_title(titles[0])
        axs[1].imshow(y_predict, vmin=np.min(y_predict), vmax=np.max(y_predict), cmap='gray' if y_predict.ndim == 2 else None)
        axs[1].set_title(titles[1])
        axs[2].imshow(y_true, vmin=np.min(y_true), vmax=np.max(y_true), cmap='gray' if y_true.ndim == 2 else None)
        axs[2].set_title(titles[2])

        fig.savefig(save_path)
        plt.close(fig)
        plt.close('all')

    def plot_comparisons(self, x_values, y_values, save_path='plot.png', main_title='Main Title', subtitles=None, subplot=True, x_labels=None, y_labels=None):
        """
        Plot and save multiple comparison plots or subplots.
        
        Parameters:
            x_values (List): List of x-axis data for each plot.
            y_values (List): List of y-axis data for each plot.
            save_path (str): Base path to save the plot(s). For individual plots, the name will be appended with an index.
            main_title (str): Main title for the entire figure.
            subtitles (List[str], optional): Titles for each subplot.
            subplot (bool): If True, all plots are arranged in subplots; if False, each plot is saved individually.
            x_labels (List[str], optional): Labels for the x-axis.
            y_labels (List[str], optional): Labels for the y-axis.
        """
        num_plots = len(x_values)

        if not subplot:
            # Plot and save each plot separately.
            for i in range(num_plots):
                plt.figure(figsize=(6, 4))
                plt.plot(x_values[i], y_values[i])

                min_x_limit = [min(x_values[i]) if not np.isnan(min(x_values[i])) and not np.isinf(min(x_values[i])) else 0][0]
                max_x_limit = [max(x_values[i]) if not np.isinf(max(x_values[i])) and not np.isnan(max(x_values[i])) else 1][0]
                min_y_limit = [min(y_values[i]) if not np.isnan(min(y_values[i])) and not np.isinf(min(y_values[i])) else 0][0]
                max_y_limit = [max(y_values[i]) if not np.isinf(max(y_values[i])) and not np.isnan(max(y_values[i])) else 1][0]

                plt.xlim(min_x_limit, max_x_limit)
                plt.ylim(min_y_limit, max_y_limit)

                plt.title(subtitles[i] if subtitles else f"Plot {i+1}")
                plt.xlabel(x_labels[i] if x_labels and len(x_labels) > 1 else (x_labels[0] if x_labels else 'X-axis'))
                plt.ylabel(y_labels[i] if y_labels and len(y_labels) > 1 else (y_labels[0] if y_labels else 'Y-axis'))

                plt.savefig(f"{save_path}_plot_{subtitles[i]}.png")
                plt.close()
        else:
            # Plot all plots as subplots in one figure.
            fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
            fig.suptitle(main_title)

            for i in range(num_plots):
                axs[i].plot(x_values[i], y_values[i])

                min_x_limit = [min(x_values[i]) if not np.isnan(min(x_values[i])) or not np.isinf(min(x_values[i])) else 0][0]
                max_x_limit = [max(x_values[i]) if not np.isinf(max(x_values[i])) or not np.isnan(max(x_values[i])) else 1][0]
                min_y_limit = [min(y_values[i]) if not np.isnan(min(y_values[i])) or not np.isinf(min(y_values[i])) else 0][0]
                max_y_limit = [max(y_values[i]) if not np.isinf(max(y_values[i])) or not np.isnan(max(y_values[i])) else 1][0]

                axs[i].set_xlim(min_x_limit, max_x_limit)
                axs[i].set_ylim(min_y_limit, max_y_limit)
                axs[i].set_title(subtitles[i] if subtitles else f"Plot {i+1}")
                if x_labels:
                    axs[i].set_xlabel(x_labels[i] if len(x_labels) > 1 else x_labels[0])
                if y_labels:
                    axs[i].set_ylabel(y_labels[i] if len(y_labels) > 1 else y_labels[0])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(save_path)
            plt.close(fig)

def sqeeze_dim(X: Tuple[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """
    Squeeze and transpose numpy arrays to ensure proper image format for visualization.
    
    This function rearranges dimensions of the input arrays so that images are in (H, W, C) format,
    and removes any unnecessary singleton dimensions.
    
    Args:
        X (Tuple[np.ndarray]): Tuple of numpy arrays.
    
    Returns:
        Tuple[np.ndarray, ...]: Tuple of processed numpy arrays.
    """
    Y = []
    for x in X:
        if x.shape[1] < x.shape[-1] and len(x.shape) == 4:
            x = np.transpose(x, (0, 2, 3, 1))
        elif x.shape[0] < x.shape[-1] and len(x.shape) == 3:
            x = np.transpose(x, (1, 2, 0))
        if (len(x.shape) > 3 and x.shape[0] == 1) or (len(x.shape) < 3 and x.shape[0] == 1):
            x = np.squeeze(x, axis=0)
        elif len(x.shape) < 3:
            x = np.expand_dims(x, axis=len(x.shape))
        Y.append(x)
    return tuple(Y)

def compare_weights_visualizations(measured_weights, calculated_ground_truth_weights, calculated_predicted_weights, save_path):
    """
    Generate various visualizations comparing measured weights with calculated weights.

    This function creates several plots:
      1. Scatter plots comparing ground truth vs measured weights and predicted vs measured weights.
      2. Histograms of the error distributions.
      3. Bar plots comparing the Mean Absolute Error (MAE).
      4. Bar plots of R-squared scores.
      5. A combined subplot with scatter, error distribution, and MAE comparison.
      
    Parameters:
        measured_weights: List or array of measured weight values.
        calculated_ground_truth_weights: List or array of calculated ground truth weights.
        calculated_predicted_weights: List or array of calculated predicted weights.
        save_path (str): Directory where the generated plots will be saved.
    """
    # Ensure the save directory exists.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert weight lists to numpy arrays.
    measured_weights = np.array(measured_weights)
    calculated_ground_truth_weights = np.array(calculated_ground_truth_weights)
    calculated_predicted_weights = np.array(calculated_predicted_weights)

    # 1. Scatter Plot: Ground Truth vs Measured, and Predicted vs Measured.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(measured_weights, calculated_ground_truth_weights, color='blue', label='Ground Truth', alpha=0.7)
    plt.plot([min(measured_weights), max(measured_weights)],
             [min(measured_weights), max(measured_weights)], 'r--')
    plt.xlabel('Measured Weights')
    plt.ylabel('Calculated Weights (Ground Truth)')
    plt.title('Ground Truth vs Measured')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(measured_weights, calculated_predicted_weights, color='green', label='Predicted', alpha=0.7)
    plt.plot([min(measured_weights), max(measured_weights)],
             [min(measured_weights), max(measured_weights)], 'r--')
    plt.xlabel('Measured Weights')
    plt.ylabel('Calculated Weights (Predicted)')
    plt.title('Predicted vs Measured')
    plt.legend()

    plt.tight_layout()
    scatter_plot_path = os.path.join(save_path, 'scatter_comparison.png')
    plt.savefig(scatter_plot_path)
    plt.show()

    # 2. Error Distribution histograms.
    error_ground_truth = measured_weights - calculated_ground_truth_weights
    error_predicted = measured_weights - calculated_predicted_weights

    plt.figure(figsize=(10, 6))
    plt.hist(error_ground_truth, bins=10, alpha=0.5, label='Ground Truth Error', color='blue')
    plt.hist(error_predicted, bins=10, alpha=0.5, label='Predicted Error', color='green')
    plt.xlabel('Error (Measured - Calculated)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    error_dist_path = os.path.join(save_path, 'error_distribution.png')
    plt.savefig(error_dist_path)
    plt.show()

    # 3. Bar Plot: Mean Absolute Error (MAE) comparison.
    mae_ground_truth = mean_absolute_error(measured_weights, calculated_ground_truth_weights)
    mae_predicted = mean_absolute_error(measured_weights, calculated_predicted_weights)

    plt.figure(figsize=(6, 6))
    plt.bar(['Ground Truth', 'Predicted'], [mae_ground_truth, mae_predicted], color=['blue', 'green'])
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE Comparison')
    mae_plot_path = os.path.join(save_path, 'mae_comparison.png')
    plt.savefig(mae_plot_path)
    plt.show()

    # 4. Bar Plot: R-squared Score comparison.
    r2_ground_truth = r2_score(measured_weights, calculated_ground_truth_weights)
    r2_predicted = r2_score(measured_weights, calculated_predicted_weights)

    plt.figure(figsize=(6, 6))
    plt.bar(['Ground Truth', 'Predicted'], [r2_ground_truth, r2_predicted], color=['blue', 'green'])
    plt.ylabel('R-squared Score')
    plt.title('R-squared Score Comparison')
    r2_plot_path = os.path.join(save_path, 'r2_comparison.png')
    plt.savefig(r2_plot_path)
    plt.show()

    # 5. Combined Subplot: Scatter plots, error distribution, and MAE comparison.
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    axs[0, 0].scatter(measured_weights, calculated_ground_truth_weights, color='blue', label='Ground Truth', alpha=0.7)
    axs[0, 0].plot([min(measured_weights), max(measured_weights)], [min(measured_weights), max(measured_weights)], 'r--')
    axs[0, 0].set_xlabel('Measured Weights')
    axs[0, 0].set_ylabel('Calculated Weights (Ground Truth)')
    axs[0, 0].set_title('Ground Truth vs Measured')
    axs[0, 0].legend()

    axs[0, 1].scatter(measured_weights, calculated_predicted_weights, color='green', label='Predicted', alpha=0.7)
    axs[0, 1].plot([min(measured_weights), max(measured_weights)], [min(measured_weights), max(measured_weights)], 'r--')
    axs[0, 1].set_xlabel('Measured Weights')
    axs[0, 1].set_ylabel('Calculated Weights (Predicted)')
    axs[0, 1].set_title('Predicted vs Measured')
    axs[0, 1].legend()

    axs[1, 0].hist(error_ground_truth, bins=10, alpha=0.5, label='Ground Truth Error', color='blue')
    axs[1, 0].hist(error_predicted, bins=10, alpha=0.5, label='Predicted Error', color='green')
    axs[1, 0].set_xlabel('Error (Measured - Calculated)')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Error Distribution')
    axs[1, 0].legend()

    axs[1, 1].bar(['Ground Truth', 'Predicted'], [mae_ground_truth, mae_predicted], color=['blue', 'green'])
    axs[1, 1].set_ylabel('Mean Absolute Error')
    axs[1, 1].set_title('MAE Comparison')

    plt.tight_layout()
    combined_plot_path = os.path.join(save_path, 'combined_visualization.png')
    plt.savefig(combined_plot_path)
    plt.show()

    print(f"Plots saved in: {save_path}")


def plot_deviation_map(y_test, y_pred, title, cmap_label, block_size, save_path=""):
    """
    Plot deviation map and region-wise deviation between y_test and y_pred.
    
    This function computes the absolute deviation between the ground truth (y_test) and prediction (y_pred)
    images, and then displays two plots:
        1. A deviation map showing the per-pixel absolute deviation.
        2. A region-wise deviation plot showing the mean deviation within blocks of a specified size.
    
    Parameters:
        y_test: Array of ground truth values (e.g., an image or a set of images).
        y_pred: Array of predicted values corresponding to y_test.
        title (str): Title for the deviation map plot.
        cmap_label (str): Label for the colorbar in the plots.
        block_size (int): Block size used to compute region-wise (block) average deviations.
        save_path (str): File path where the generated plot will be saved.
    """
    # Step 1: Compute the absolute deviation between y_test and y_pred.
    deviation = np.expand_dims(np.mean(np.abs(y_test - y_pred), axis=0), axis=0)
    
    # Set the colormap scale limits based on deviation statistics.
    vmax = np.mean(deviation) + 3 * np.std(deviation)
    vmin = 0

    # Step 2: Plot the deviation map.
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(deviation[0, :, :, 0], cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label=cmap_label)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Step 3: Compute region-wise deviation (mean deviation in blocks).
    rows, cols = deviation.shape[1:3]  # Assuming shape is (batch, height, width, channels)
    num_blocks_row = rows // block_size
    num_blocks_col = cols // block_size

    region_deviation = np.zeros((num_blocks_row, num_blocks_col))
    # Compute the average deviation in each block.
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block = deviation[0, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, 0]
            region_deviation[i, j] = np.mean(block)
            
    vmax = np.max(region_deviation)
    vmin = 0
    
    # Step 4: Plot the region-wise deviation.
    plt.subplot(1, 2, 2)
    plt.imshow(region_deviation, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(label=cmap_label)
    plt.title(f"Region-wise Deviation (Block size: {block_size})")
    plt.xlabel('Region X-axis')
    plt.ylabel('Region Y-axis')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
