import matplotlib as mlp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import collections.abc
import torch

class compare_model_predictions():
    """
    A class to compare predictions from different models against inputs and ground truth images.
    
    This class facilitates visualization of input images, predictions from multiple models, and ground truth 
    images side by side. It also supports plotting best/worst predictions based on metric scores, histogram 
    comparisons, scatter plots for weight estimation, deviation maps, and training history plots.
    
    Attributes:
        x_tests (np.array): Array of input images.
        y_tests (np.array): Array of ground truth images.
        y_preds (List): List of numpy arrays containing predictions from different models.
        model_list (List): List of model names corresponding to the predictions.
        n_cols (int): Number of columns for the visualization grid.
        fig_initialized (bool): Flag indicating whether a figure has been initialized.
    """
    def __init__(self, 
                 x_tests: np.array = np.zeros((4, 512, 512, 1)),
                 y_tests: np.array = np.zeros((4, 512, 512, 1)),
                 y_preds: List = [np.zeros((4, 512, 512, 1)), np.zeros((4, 512, 512, 1))],
                 model_list = ["unet-gan", "VT-gan"] ) -> None:
        """
        Initialize the compare_model_predictions object.
        
        Args:
            x_tests (np.array): Numpy array of input images.
            y_tests (np.array): Numpy array of ground truth images.
            y_preds (List): List of numpy arrays of predictions from different models.
            model_list (List): List of names of the models used for prediction.
            
        Raises:
            ValueError: If y_preds is not a list or if the length of y_preds does not match model_list.
        """
        mlp.rcParams['font.family'] = ['sans-serif']
        self.x_tests = x_tests
        self.y_tests = y_tests
        self.y_preds = y_preds
        self.model_list = model_list
        # Two extra columns: one for the input and one for the ground truth.
        self.n_cols = len(self.model_list) + 2  
        self.fig_initialized = False
        try:
            len(y_preds)
        except:
            raise ValueError("y_preds must be list of one or multiple elements, predictions from different models")

        if len(y_preds) != len(model_list):
            raise ValueError("length of y_preds list and model_list must be same")
            
    def build_figure(self, n_rows: int = 4, n_cols = None, **kwargs) -> None:
        """
        Build a matplotlib figure with a grid of subplots.
        
        Args:
            n_rows (int): Number of rows in the subplot grid.
            n_cols (int): Number of columns in the subplot grid. If None, uses self.n_cols.
            **kwargs: Additional keyword arguments for plt.subplots.
        """
        self.fig_initialized = True
        if n_cols is None:
            n_cols = self.n_cols
        
        gridspec_kw = kwargs.get('gridspec_kw', {'wspace': 0.01, 'hspace': 0.05})
        kwargs['gridspec_kw'] = gridspec_kw

        # Calculate figure size based on grid dimensions.
        figure_length = 1.5 * n_cols  # inches
        figure_width = 1.5 * n_rows  # inches

        figsize = kwargs.get('figsize', (figure_length, figure_width))
        kwargs['figsize'] = figsize        

        self.fig, self.axes = plt.subplots(nrows=n_rows, ncols=n_cols, **kwargs)
    
    def get_random_preds(self, n_rows: int = 4, index_list: List = None):
        """
        Build a figure showing random samples from the input, predictions, and ground truth.
        
        Args:
            n_rows (int): Number of rows (samples) to display.
            index_list (List): Optional list of specific indices to display; must be same length as n_rows.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
            
        Raises:
            ValueError: If index_list is provided and its length does not equal n_rows.
        """
        if self.fig_initialized:
            self.close_fig()
        grid_spec = dict(wspace=-0.3, hspace=0.05)
        self.build_figure(n_rows=n_rows, figsize=(5, n_rows * 1.7), gridspec_kw=grid_spec, dpi=300)
        if index_list is not None and len(index_list) != n_rows:
            raise ValueError("length of index list must be same as number of rows")
        
        for i in range(n_rows):
            if index_list is None:
                index = np.random.randint(0, self.x_tests.shape[0])
                print(index)
            else:
                index = index_list[i]
            # Order: Input image, predictions from each model, ground truth.
            element_list = [self.x_tests[index]] + [y_pred[index] for y_pred in self.y_preds] + [self.y_tests[index]]
            sub_title_list = ["Depth Input"] + [model_name + " pred." for model_name in self.model_list] + ["Ref. PImg"]
            for j in range(self.n_cols):
                self.axes[i, j].imshow(element_list[j], cmap='jet')
                self.axes[i, j].set_axis_off()
                if i < 1:
                    vertical_position = 1.0
                    self.axes[i, j].set_title(sub_title_list[j], y=vertical_position, fontsize=6)
        return self.fig
    
    def show_fig(self):
        """
        Display the current figure.
        """
        self.fig.show()
        plt.show()
    
    def save_fig(self, path):
        """
        Save the current figure to the specified path and close all figures.
        
        Args:
            path (str): File path to save the figure.
        """
        self.fig.savefig(path)
        plt.close('all')
    
    def close_fig(self):
        """
        Close all matplotlib figures.
        """
        plt.close('all')
    
    def get_bestpreds_images(self, 
                             metric_scores_list: List = [{}, {}],
                             best_metric_score_index_list: List = [{}, {}],
                             metric_names: List = ["MeanSquaredError", "MPPA"]):
        """
        Plot images corresponding to the best metric scores for each model.
        
        Args:
            metric_scores_list (List): List of dictionaries containing metric scores for each model.
            best_metric_score_index_list (List): List of dictionaries with the best metric score indices.
            metric_names (List): List of metric names to plot. Use "all" to plot all metrics.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if "all" in metric_names:
            metric_names = metric_scores_list[0].keys()
        
        n_rows = len(metric_names)
        n_cols = 3 * len(self.model_list)
        if self.fig_initialized:
            self.close_fig()
        grid_spec = dict(wspace=0.2, hspace=0.1)
        self.build_figure(n_rows=n_rows, n_cols=n_cols, figsize=(11, 4), gridspec_kw=grid_spec, dpi=300)
    
        for i, metric_name in zip(range(n_rows), metric_names):
            # Shorten common metric names for display.
            if "FrechetInceptionDistance" in metric_name:
                short_metric_name = "FID"
            elif "MeanPerPixelAcc" in metric_name:
                short_metric_name = "PPA"
            elif "RootMeanSquared" in metric_name:
                short_metric_name = "RMSESL"
            elif "Absolute" in metric_name:
                short_metric_name = "MAE"
            elif "Squared" in metric_name:
                short_metric_name = "MSE"
            elif "PeakSignalNoiseRatio" in metric_name:
                short_metric_name = "PSNR"
            elif "StructuralSimilarityIndexMeasure" in metric_name:
                short_metric_name = "SSIM"
            else:
                short_metric_name = metric_name
    
            best_index_list = [best_metric_score_index[metric_name] for best_metric_score_index in best_metric_score_index_list]
    
            element_list = [self.x_tests[index] for index in best_index_list] + \
                           [y_pred[index] for index, y_pred in zip(best_index_list, self.y_preds)] + \
                           [self.y_tests[index] for index in best_index_list]
            sub_title_list = ["Input" + model_name for model_name in self.model_list] + \
                             [model_name + "_pred" for model_name in self.model_list] + \
                             [model_name + " Ref." for model_name in self.model_list]
            if "Error" not in metric_name and "FID" not in metric_name and "PSNR" not in metric_name:
                best_metric_score_list = [np.max(metric_scores[metric_name]) for metric_scores in metric_scores_list]
            else:
                best_metric_score_list = [np.min(metric_scores[metric_name]) for metric_scores in metric_scores_list]
    
            for j in range(n_cols):
                if element_list[j].ndim == 4:
                    element_list[j] = np.squeeze(element_list[j], axis=0)
                self.axes[i, j].imshow(element_list[j], cmap='jet')
                self.axes[i, j].set_axis_off()
                if "pred" in sub_title_list[j]:
                    idx = j - len(self.model_list)
                    self.axes[i, j].text(-0.5, 0.5, short_metric_name + ": " + str(round(best_metric_score_list[idx], 4)),
                                         va='center', 
                                         transform=self.axes[i, j].transAxes,
                                         bbox={'facecolor': 'white', 'pad': 2}, 
                                         fontsize=8, 
                                         rotation='vertical')
                if i < 1:
                    vertical_position = 1.0
                    self.axes[i, j].set_title(sub_title_list[j], y=vertical_position, fontsize=8)
        return self.fig
    
    def get_worstpreds_images(self, 
                              metric_scores_list: List = [{}, {}],
                              worst_metric_score_index_list: List = [{}, {}],
                              metric_names: List = ["MeanSquaredError", "MPPA"]):
        """
        Plot images corresponding to the worst metric scores for each model.
        
        Args:
            metric_scores_list (List): List of dictionaries containing metric scores.
            worst_metric_score_index_list (List): List of dictionaries with worst metric score indices.
            metric_names (List): List of metric names to plot.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if "all" in metric_names:
            metric_names = metric_scores_list[0].keys()
        
        n_rows = len(metric_names)
        n_cols = 3 * len(self.model_list)
        if self.fig_initialized:
            self.close_fig()
        grid_spec = dict(wspace=0.2, hspace=0.1)
        self.build_figure(n_rows=n_rows, n_cols=n_cols, figsize=(11, 4), gridspec_kw=grid_spec, dpi=300)
    
        for i, metric_name in zip(range(n_rows), metric_names):
            if "FrechetInceptionDistance" in metric_name:
                short_metric_name = "FID"
            elif "MeanPerPixelAcc" in metric_name:
                short_metric_name = "PPA"
            elif "RootMeanSquared" in metric_name:
                short_metric_name = "RMSESL"
            elif "Absolute" in metric_name:
                short_metric_name = "MAE"
            elif "Squared" in metric_name:
                short_metric_name = "MSE"
            elif "PeakSignalNoiseRatio" in metric_name:
                short_metric_name = "PSNR"
            elif "StructuralSimilarityIndexMeasure" in metric_name:
                short_metric_name = "SSIM"
            else:
                short_metric_name = metric_name
    
            worst_index_list = [worst_metric_score_index[metric_name] for worst_metric_score_index in worst_metric_score_index_list]
    
            element_list = [self.x_tests[index] for index in worst_index_list] + \
                           [y_pred[index] for index, y_pred in zip(worst_index_list, self.y_preds)] + \
                           [self.y_tests[index] for index in worst_index_list]
            sub_title_list = ["Input" + model_name for model_name in self.model_list] + \
                             [model_name + "_pred" for model_name in self.model_list] + \
                             [model_name + " Ref." for model_name in self.model_list]
            if "Error" not in metric_name and "FID" not in metric_name and "PSNR" not in metric_name:
                worst_metric_score_list = [np.min(metric_scores[metric_name]) for metric_scores in metric_scores_list]
            else:
                worst_metric_score_list = [np.max(metric_scores[metric_name]) for metric_scores in metric_scores_list]
    
            for j in range(n_cols):
                if element_list[j].ndim == 4:
                    element_list[j] = np.squeeze(element_list[j], axis=0)
                self.axes[i, j].imshow(element_list[j], cmap='jet')
                self.axes[i, j].set_axis_off()
                if "pred" in sub_title_list[j]:
                    idx = j - len(self.model_list)
                    self.axes[i, j].text(-0.5, 0.5, short_metric_name + ": " + str(round(worst_metric_score_list[idx], 4)),
                                         va='center', 
                                         transform=self.axes[i, j].transAxes,
                                         bbox={'facecolor': 'white', 'pad': 2}, 
                                         fontsize=8, 
                                         rotation='vertical')
                if i < 1:
                    vertical_position = 1.0
                    self.axes[i, j].set_title(sub_title_list[j], y=vertical_position, fontsize=8)
        return self.fig
    
    def get_best_worst_hist(self,
                            metric_scores_list: List = [{}, {}],
                            best_metric_score_index_list: List = [{}, {}],
                            worst_metric_score_index_list: List = [{}, {}],
                            metric_names: List = ["MeanSquaredError", "MeanPerPixelAccuracy"]
                            ):
        """
        Plot histograms for the best and worst metric scores for each model.
        
        Args:
            metric_scores_list (List): List of dictionaries containing metric scores.
            best_metric_score_index_list (List): List of dictionaries with best metric score indices.
            worst_metric_score_index_list (List): List of dictionaries with worst metric score indices.
            metric_names (List): List of metric names to include. Use "all" to include all metrics.
        
        Returns:
            matplotlib.figure.Figure: The generated figure with histograms.
        """
        best_metric_score_index_list_initial = best_metric_score_index_list
        worst_metric_score_index_list_initial = worst_metric_score_index_list

        if len(best_metric_score_index_list) < len(self.y_preds):
            for i in range(len(self.y_preds) - len(best_metric_score_index_list)):
                best_metric_score_index_list += best_metric_score_index_list
                worst_metric_score_index_list += worst_metric_score_index_list

        if len(best_metric_score_index_list) != len(worst_metric_score_index_list):
            raise ValueError("length of best and worst metrix score indexes must be same found %d and %d" %
                             (len(best_metric_score_index_list), len(worst_metric_score_index_list)))

        if "all" in metric_names:
            metric_names = metric_scores_list[0].keys()
        
        n_rows = len(metric_names) * 2
        n_cols = len(self.model_list) + 2
        if self.fig_initialized:
            self.close_fig()
        
        width_ratios = [1 if i + 1 != n_cols - 1 else 1.5 for i in range(n_cols)]
        grid_spec = dict(wspace=0.3, hspace=0.2, width_ratios=width_ratios)
        self.build_figure(n_rows=n_rows, n_cols=n_cols, figsize=(7.2, n_rows * 1.75), gridspec_kw=grid_spec, dpi=300)
        i = 0
        for metric_name in metric_names:
            if "FrechetInceptionDistance" in metric_name:
                short_metric_name = "FID"
            elif "MeanPerPixelAcc" in metric_name:
                short_metric_name = "PPA"
            elif "RootMeanSquared" in metric_name:
                short_metric_name = "RMSESL"
            elif "Absolute" in metric_name:
                short_metric_name = "MAE"
            elif "Squared" in metric_name:
                short_metric_name = "MSE"
            elif "PeakSignalNoiseRatio" in metric_name:
                short_metric_name = "PSNR"
            elif "StructuralSimilarityIndexMeasure" in metric_name:
                short_metric_name = "SSIM"
            else:
                short_metric_name = metric_name

            best_index_list = [best_metric_score_index[metric_name] for best_metric_score_index in best_metric_score_index_list]
            worst_index_list = [worst_metric_score_index[metric_name] for worst_metric_score_index in worst_metric_score_index_list]
            best_index_list_initial = [best_metric_score_index_init[metric_name] for best_metric_score_index_init in best_metric_score_index_list_initial][0]
            worst_index_list_initial = [worst_metric_score_index_init[metric_name] for worst_metric_score_index_init in worst_metric_score_index_list_initial][0]

            best_element_list = [ 
                                    y_pred[index] for index, y_pred in zip(best_index_list, self.y_preds)
                                 ] + [
                                    self.calculate_histogram(y_pred[index])[0] for index, y_pred in zip(best_index_list, self.y_preds)
                                 ] + [
                                    self.calculate_histogram(self.y_tests[index])[0] for index in best_index_list_initial
                                 ] + [
                                    self.y_tests[index] for index in best_index_list_initial
                                 ]
            worst_element_list = [ 
                                    y_pred[index] for index, y_pred in zip(worst_index_list, self.y_preds)
                                 ] + [
                                    self.calculate_histogram(y_pred[index])[0] for index, y_pred in zip(worst_index_list, self.y_preds)
                                 ] + [
                                    self.calculate_histogram(self.y_tests[index])[0] for index in worst_index_list_initial
                                 ] + [
                                    self.y_tests[index] for index in worst_index_list_initial
                                 ]
            
            sub_title_list = [model_name for model_name in self.model_list] + ["Histogram Plots"] + ["Reference"]

            if "Error" not in metric_name and "FID" not in metric_name and "PSNR" not in metric_name:
                worst_metric_score_list = [np.min(metric_scores[metric_name]) for metric_scores in metric_scores_list]
                best_metric_score_list = [np.max(metric_scores[metric_name]) for metric_scores in metric_scores_list]
            else:
                worst_metric_score_list = [np.max(metric_scores[metric_name]) for metric_scores in metric_scores_list]
                best_metric_score_list = [np.min(metric_scores[metric_name]) for metric_scores in metric_scores_list]

            lengends = ["" for model_name in self.model_list] + [self.model_list[i] for i in range(len(self.model_list))] + ["Reference"]

            j = 0
            while j < len(best_element_list):
                if best_element_list[j].ndim == 4:
                    best_element_list[j] = np.squeeze(best_element_list[j], axis=0)
                if worst_element_list[j].ndim == 4:
                    worst_element_list[j] = np.squeeze(worst_element_list[j], axis=0)

                if j < len(self.y_preds):
                    self.axes[i, j].imshow(best_element_list[j], cmap='jet')
                    self.axes[i, j].set_axis_off()
                    self.axes[i + 1, j].imshow(worst_element_list[j], cmap='jet')
                    self.axes[i + 1, j].set_axis_off()
                    self.axes[0, j].set_title(sub_title_list[j], fontsize=8)
                    self.axes[i, j].text(-0.5, 0.5, short_metric_name + ": " + str(round(best_metric_score_list[j], 4)),
                                         va='center', 
                                         transform=self.axes[i, j].transAxes,
                                         bbox={'facecolor': 'white', 'pad': 2}, 
                                         fontsize=8, 
                                         rotation='vertical')
                    self.axes[i + 1, j].text(-0.5, 0.5, short_metric_name + ": " + str(round(worst_metric_score_list[j], 4)),
                                         va='center', 
                                         transform=self.axes[i + 1, j].transAxes,
                                         bbox={'facecolor': 'white', 'pad': 2}, 
                                         fontsize=8, 
                                         rotation='vertical')
                elif j == len(best_element_list) - 1:
                    self.axes[i, n_cols - 1].imshow(best_element_list[j], cmap='jet')
                    self.axes[i, n_cols - 1].set_axis_off()
                    self.axes[i + 1, n_cols - 1].imshow(worst_element_list[j], cmap='jet')
                    self.axes[i + 1, n_cols - 1].set_axis_off()
                    self.axes[0, n_cols - 1].set_title(sub_title_list[n_cols - 1], fontsize=10)
                else:
                    if "reference" in lengends[j].lower():
                        self.axes[i, len(self.y_preds)].plot(best_element_list[j], linestyle='-', color='gray', label=lengends[j], linewidth=1, alpha=0.6)
                        self.axes[i + 1, len(self.y_preds)].plot(worst_element_list[j], linestyle='-', color='gray', label=lengends[j], linewidth=1, alpha=0.6)   
                    else:    
                        self.axes[i, len(self.y_preds)].plot(best_element_list[j], linestyle='-', label=lengends[j], linewidth=1, markersize=5, alpha=1)
                        self.axes[i + 1, len(self.y_preds)].plot(worst_element_list[j], linestyle='-', label=lengends[j], linewidth=1, markersize=5, alpha=1)
                    
                    self.axes[0, len(self.y_preds)].set_title(sub_title_list[len(self.y_preds)], fontsize=8)
                    self.axes[i, len(self.y_preds)].set_yscale('log')
                    self.axes[i + 1, len(self.y_preds)].set_yscale('log')
                    self.axes[i, len(self.y_preds)].legend(loc='upper right', fontsize=6)
                    self.axes[i + 1, len(self.y_preds)].legend(loc='upper right', fontsize=6)
                    self.axes[i, len(self.y_preds)].set_ylim(0, 10e-3)
                    self.axes[i + 1, len(self.y_preds)].set_ylim(0, 10e-3)
                    self.axes[n_rows - 1, len(self.y_preds)].set_xlabel('fixed width bins')
                    self.axes[n_rows // 2, len(self.y_preds)].text(1.1, 1.0, 'frequency (log scale)', va='center', rotation='vertical',
                                                    transform=self.axes[n_rows // 2, len(self.y_preds)].transAxes,
                                                    fontsize=8, weight='ultralight')
                    self.axes[n_rows // 2, len(self.y_preds)].yaxis.set_label_position("right")
                    self.axes[i + 1, len(self.y_preds)].yaxis.set_label_position("right")
                j += 1
            i += 2
        return self.fig
    
    def get_box_plots(self, metric_scores_list: List = [{}, {}],
                      metric_names: List = ["MeanSquaredError", "MeanPerPixelAccuracy"]):
        """
        Plot box plots of metric scores for different models.
        
        Args:
            metric_scores_list (List): List of dictionaries containing metric scores.
            metric_names (List): List of metric names to plot.
            
        Raises:
            ValueError: If the length of metric_names is not even.
        """
        if "all" in metric_names:
            metric_names = metric_scores_list[0].keys()
        
        n_cols = len(metric_names)
        n_rows = 1
        if (n_rows * n_cols) != len(metric_names):
            raise ValueError("Length of the metric_names must be even, found odd")

        self.close_fig()
        self.build_figure(n_rows=n_rows, n_cols=n_cols, figsize=(n_cols * 1.8, n_rows * 2.8), dpi=300, gridspec_kw={'wspace': 1.0, 'hspace': 2.0})
        i = 0
        j = 0
        for metric_name in metric_names:
            if "FrechetInceptionDistance" in metric_name:
                short_metric_name = "FID"
            elif "MeanPerPixelAcc" in metric_name:
                short_metric_name = "PPA"
            elif "RootMeanSquared" in metric_name:
                short_metric_name = "RMSESL"
            elif "Absolute" in metric_name:
                short_metric_name = "MAE"
            elif "Squared" in metric_name:
                short_metric_name = "MSE"
            elif "PeakSignalNoiseRatio" in metric_name:
                short_metric_name = "PSNR"
            elif "StructuralSimilarityIndexMeasure" in metric_name:
                short_metric_name = "SSIM"
            else:
                short_metric_name = metric_name

            x_axis_titles = [self.model_list[i] for i in range(len(self.model_list))]
            score_list = [metric_score[metric_name] for metric_score in metric_scores_list]
            if j > n_cols - 1:
                j = 0
                i += 1
            box = self.axes[j].boxplot(score_list, patch_artist=True)
            for patch in box['boxes']:
                patch.set(facecolor='white')    
            self.axes[j].set_ylabel(short_metric_name + " (unit)")
            self.axes[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            self.axes[j].set_xticks(range(1, len(score_list) + 1), x_axis_titles, rotation=45, ha='right')
            j += 1
        plt.subplots_adjust(bottom=0.25)
    
    def calculate_histogram(self, image, channels=None, bins=180, value_range=(0.0, 1.0)):
        """
        Calculate histograms for the image channels.
        
        Parameters:
            image (array-like): Input image with values in the range [0, 1].
            channels (int): Number of channels in the image. If None, inferred from image shape.
            bins (int): Number of bins for the histogram.
            value_range (tuple): The range of values for the histogram.
            
        Returns:
            List[np.array]: List of histograms for each channel.
        """
        image = torch.tensor(image)
        if channels is None:
            channels = 1 if len(image.shape) == 2 else image.shape[-1]
        histograms = []
        for i in range(channels):
            channel_data = image[:, :, i] if channels > 1 else image
            hist = torch.histc(channel_data, bins=bins, min=value_range[0], max=value_range[1])
            if hist.sum() > 0:
                hist = hist / hist.sum()
            histograms.append(hist.cpu().numpy())
        return histograms

    def weight_scatter_plot(self,
                            measured_weights: List = [],
                            calculated_ground_truth_weights: List = [],
                            predicted_weights_list: List = [[], []],
                            splot=True):
        """
        Plot scatter plots comparing measured weights, calculated ground truth weights,
        and predicted weights from different models.
        
        Args:
            measured_weights (List): List of measured weight values.
            calculated_ground_truth_weights (List): List of ground truth weight values.
            predicted_weights_list (List): List of predicted weight lists from different models.
            splot (bool): If True, create separate subplots for each model.
        """
        if len(predicted_weights_list) != len(self.y_preds):
            raise ValueError("length of predicted weight list must be equal to length of model list")
        
        n_rows = 1
        n_cols = len(predicted_weights_list) if splot else 1
        title_list = [self.model_list[i] for i in range(len(self.model_list))]
        lengends = [self.model_list[i] + "_pred" for i in range(len(self.model_list))]
        fig_H = n_cols * 2.3 if n_cols > 1 else 5

        self.build_figure(n_rows=n_rows, n_cols=n_cols, figsize=(fig_H, 3), sharey=True, dpi=300, gridspec_kw={'wspace': 0.3, 'hspace': 0.1})
        
        j = 0
        for i in range(len(predicted_weights_list)):
            fit = np.polyfit(measured_weights, predicted_weights_list[i], 1)
            fit_fn = np.poly1d(fit)
            if splot:
                self.axes[j].scatter(measured_weights, calculated_ground_truth_weights, color='black', alpha=0.7, label="calculated", s=2)
                self.axes[j].plot([min(measured_weights), max(measured_weights)], [min(measured_weights), max(measured_weights)], 'r--', label="measured")
                self.axes[j].scatter(measured_weights, predicted_weights_list[i], alpha=0.4, label=lengends[i], s=2)
                self.axes[j].plot(measured_weights, fit_fn(measured_weights), '--', linewidth=1, dashes=(1, 5))
                self.axes[0].set_ylabel('predicted weights (kg)')
                self.axes[j].set_xlabel('measured weights (kg)')
                self.axes[j].legend(loc='upper left', fontsize=6)
            else:
                if j < 1:
                    self.axes.scatter(measured_weights, calculated_ground_truth_weights, color='black', alpha=0.7, label="calculated", s=2)
                    self.axes.plot([min(measured_weights), max(measured_weights)], [min(measured_weights), max(measured_weights)], 'r--', label="measured")
                self.axes.scatter(measured_weights, predicted_weights_list[i], alpha=0.7, label=lengends[i], s=2)
                self.axes.plot(measured_weights, fit_fn(measured_weights), '--', linewidth=1, dashes=(1, 5))
                self.axes.set_ylabel('predicted weights (kg)')
                self.axes.set_xlabel('measured weights (kg)')
                self.axes.legend(loc='upper left', fontsize=6)
            plt.subplots_adjust(bottom=0.25)
            j += 1

    def get_deviation_plots(self, press_calibrated_scale=None, colorbar_label="unit", Filter_block_size=None, calibrated_y_preds: List = None, calibrated_y_tests: List = None):
        """
        Compute and plot deviation maps for different postures based on predictions and ground truth.
        
        The function scales predictions and ground truth images using a calibration scale if provided,
        then calculates the deviation per pixel in different posture categories (supine, left side, right side)
        using a helper function get_posture_arrays. It then plots deviation maps for each model.
        
        Args:
            press_calibrated_scale: Scale factor for calibration. Can be an ndarray, sequence, or int.
            colorbar_label (str): Label for the colorbar.
            Filter_block_size: If provided, computes deviations in blocks.
            calibrated_y_preds (List): Calibrated predictions, if available.
            calibrated_y_tests (List): Calibrated ground truth images, if available.
        
        Returns:
            Tuple: Three lists containing deviation maps for supine, left side, and right side postures.
        """
        if isinstance(press_calibrated_scale, np.ndarray):
            y_test = (self.y_tests) * press_calibrated_scale.reshape((-1, 1, 1, 1))
            y_preds = [(y_pred) * press_calibrated_scale.reshape((-1, 1, 1, 1)) for y_pred in self.y_preds]
        elif isinstance(press_calibrated_scale, collections.abc.Sequence):
            y_test = self.y_tests * press_calibrated_scale[0]
            y_preds = [y_pred * press_calibrated_scale[-1] for y_pred in self.y_preds]
        elif isinstance(press_calibrated_scale, int):
            y_test = self.y_tests * press_calibrated_scale
            y_preds = [y_pred * press_calibrated_scale for y_pred in self.y_preds]
        elif calibrated_y_preds is None:
            y_preds = self.y_preds
            y_test = self.y_tests
            print("using 0-1 scale")
        elif calibrated_y_preds is not None and calibrated_y_tests is None:
            y_preds = calibrated_y_preds
            y_test = (self.y_tests) * press_calibrated_scale.reshape((-1, 1, 1, 1))
        else:
            y_preds = calibrated_y_preds
            y_test = calibrated_y_tests

        suppine_deviations_list = []
        leftside_deviations_list = []
        rightside_deviations_list = []

        suppine_y_test, leftside_y_test, rightside_y_test = self.get_posture_arrays(y_test)
        count = 0
        for y_pred in y_preds:
            s, l, r = self.get_posture_arrays(y_pred)
            
            if calibrated_y_tests is not None:
                suppine_y_test, leftside_y_test, rightside_y_test = self.get_posture_arrays(calibrated_y_tests[count])

            sd = np.mean(np.abs(suppine_y_test - s), axis=0)
            ld = np.mean(np.abs(leftside_y_test - l), axis=0)
            rd = np.mean(np.abs(rightside_y_test - r), axis=0)

            if Filter_block_size is not None:
                rows, cols = y_test.shape[1:3]
                num_blocks_row = rows // Filter_block_size
                num_blocks_col = cols // Filter_block_size
                
                region_sd = np.zeros((num_blocks_row, num_blocks_col))
                region_ld = np.zeros((num_blocks_row, num_blocks_col))
                region_rd = np.zeros((num_blocks_row, num_blocks_col))

                for i in range(num_blocks_row):
                    for j in range(num_blocks_col):
                        block = sd[i * Filter_block_size:(i + 1) * Filter_block_size, j * Filter_block_size:(j + 1) * Filter_block_size, 0]
                        region_sd[i, j] = np.mean(block)
                        
                        block = ld[i * Filter_block_size:(i + 1) * Filter_block_size, j * Filter_block_size:(j + 1) * Filter_block_size, 0]
                        region_ld[i, j] = np.mean(block)
                        
                        block = rd[i * Filter_block_size:(i + 1) * Filter_block_size, j * Filter_block_size:(j + 1) * Filter_block_size, 0]
                        region_rd[i, j] = np.mean(block)
                
                suppine_deviations_list.append(region_sd)
                leftside_deviations_list.append(region_ld)
                rightside_deviations_list.append(region_rd)
            else:    
                suppine_deviations_list.append(sd)
                leftside_deviations_list.append(ld)
                rightside_deviations_list.append(rd)
            count += 1
        n_rows = 1
        n_cols = 3
        self.fig = plt.figure(constrained_layout=False, figsize=(8, 2.5), dpi=300)

        left_pos = 0.05
        right_pos = (8 - 0.2) / len(self.model_list)
        
        vmax = 1.5
        im = None
        i = 0
        while i < len(self.model_list):
            gs = self.fig.add_gridspec(nrows=n_rows, ncols=n_cols, wspace=0, left=left_pos, right=left_pos + (right_pos / 9))

            self.ax1 = self.fig.add_subplot(gs[:, 0])
            self.ax2 = self.fig.add_subplot(gs[:, 1])
            self.ax3 = self.fig.add_subplot(gs[:, 2])

            print("maximum deviation limit for colorbar", vmax)

            im1 = self.ax1.imshow(suppine_deviations_list[i], cmap='jet', vmin=0, vmax=vmax)
            im2 = self.ax2.imshow(leftside_deviations_list[i], cmap='jet', vmin=0, vmax=vmax)
            im3 = self.ax3.imshow(rightside_deviations_list[i], cmap='jet', vmin=0, vmax=vmax)

            im = im3 if im is None else im

            self.ax1.set_title("Suppine", fontsize=8)
            self.ax2.set_title("Leftside", fontsize=8)
            self.ax3.set_title("Rightside", fontsize=8)

            mid_pos = (left_pos + left_pos + (right_pos / 9)) / 2
            if i > 0:
                mid_pos -= 0.05
            
            self.fig.text(mid_pos, 0.1, self.model_list[i], ha='center', va='center', fontsize=10)
            self.ax1.set_axis_off()
            self.ax2.set_axis_off()
            self.ax3.set_axis_off()
            left_pos = left_pos + (right_pos / 9) + 0.01
            i += 1
        
        cbar = self.fig.colorbar(im, ax=self.fig.axes, location='right', fraction=0.02, pad=0.05)
        cbar.set_label(colorbar_label)

        return suppine_deviations_list, leftside_deviations_list, rightside_deviations_list

    def get_labeled_deviation_plots(self, press_calibrated_scale=None, colorbar_label="unit", Filter_block_size=None):
        """
        Create deviation plots with labeled regions.
        
        This function uses get_deviation_plots to compute deviation maps, then zeroes out specific regions
        corresponding to the pelvis and head areas to compute and display labeled deviation information.
        
        Args:
            press_calibrated_scale: Scale factor for calibration.
            colorbar_label (str): Label for the colorbar.
            Filter_block_size: Block size to filter deviations.
        """
        suppine_deviations_list, leftside_deviations_list, rightside_deviations_list = self.get_deviation_plots(
            press_calibrated_scale=press_calibrated_scale, colorbar_label=colorbar_label, Filter_block_size=Filter_block_size)
        i = 0
        vmax = np.mean(rightside_deviations_list[0]) + 4 * np.std(rightside_deviations_list[0])
        while i < len(self.model_list):
            suppine_hip = suppine_deviations_list[i][14:17, 5:8]
            lefiside_hip = leftside_deviations_list[i][15:19, 5:7]
            rightside_hip = rightside_deviations_list[i][13:17, 6:10]

            suppine_head = suppine_deviations_list[i][3:6, 5:8]
            lefiside_head = leftside_deviations_list[i][5:8, 7:9]
            rightside_head = rightside_deviations_list[i][4:7, 3:7]

            print("average deviation in pelvic area of suppine posture with " + self.model_list[i] + ": ", np.mean(suppine_hip))
            print("average deviation in pelvic area of leftside posture with " + self.model_list[i] + ": ", np.mean(lefiside_hip))
            print("average deviation in pelvic area of rightside posture with " + self.model_list[i] + ": ", np.mean(rightside_hip))
            print("average deviation in head area of suppine posture with " + self.model_list[i] + ": ", np.mean(suppine_head))
            print("average deviation in head area of leftside posture with " + self.model_list[i] + ": ", np.mean(lefiside_head))
            print("average deviation in head area of rightside posture with " + self.model_list[i] + ": ", np.mean(rightside_head))
            print('\n')

            suppine_deviations_list[i][14:17, 5:8] = 0
            leftside_deviations_list[i][15:19, 5:7] = 0
            rightside_deviations_list[i][13:17, 6:10] = 0

            suppine_deviations_list[i][3:6, 5:8] = 0
            leftside_deviations_list[i][5:8, 7:9] = 0
            rightside_deviations_list[i][4:7, 3:7] = 0

            im1 = self.ax1.imshow(suppine_deviations_list[i], cmap='jet', vmin=0, vmax=vmax)
            im2 = self.ax2.imshow(leftside_deviations_list[i], cmap='jet', vmin=0, vmax=vmax)
            im3 = self.ax3.imshow(rightside_deviations_list[i], cmap='jet', vmin=0, vmax=vmax)
            i += 1

    def get_posture_arrays(self, arr):
        """
        Split the input array into posture-specific arrays.
        
        Assumes each subject has 45 images arranged as:
            - First 15: supine
            - Next 15: left side
            - Last 15: right side
        
        Args:
            arr: Input array with shape (total_images, height, width, channels).
        
        Returns:
            Tuple: (supine_postures, left_side_postures, right_side_postures) concatenated along axis 0.
        """
        supine_postures = []
        left_side_postures = []
        right_side_postures = []

        # Assume number of subjects = total_images // 45
        for i in range(self.y_tests.shape[0] // 45):
            start_index = i * 45
            end_index = start_index + 45
            supine_postures.append(arr[start_index:start_index + 15])
            left_side_postures.append(arr[start_index + 15:start_index + 30])
            right_side_postures.append(arr[start_index + 30:end_index])

        supine_postures = np.concatenate(supine_postures, axis=0)
        left_side_postures = np.concatenate(left_side_postures, axis=0)
        right_side_postures = np.concatenate(right_side_postures, axis=0)
        
        return supine_postures, left_side_postures, right_side_postures
    
    def get_training_history(self, training_history_list, metric_name="ssiml2", ylim: Tuple = None, legend_loc=None, sci_limits: Tuple = None):  
        """
        Plot training history curves for a given metric.
        
        Args:
            training_history_list (List): List of dictionaries containing training history for different models.
            metric_name (str): Metric to plot (e.g., "ssiml2", "mse", "ssim").
            ylim (Tuple): Y-axis limits.
            legend_loc: Location for the legend.
            sci_limits (Tuple): Limits for scientific notation on the y-axis.
        """
        if len(training_history_list) != len(self.model_list):
            raise ValueError("length of predicted weight list must be equal to length of model list")
        
        n_rows = 1
        n_cols = 1

        lengends = [self.model_list[i] for i in range(len(self.model_list))]
        fig_W = 5

        self.build_figure(n_rows=n_rows, n_cols=n_cols, figsize=(fig_W, 2.5), sharey=True, dpi=300, gridspec_kw={'wspace': 0, 'hspace': 0})

        min_y_limit = []
        max_y_limit = []
        for i in range(len(training_history_list)):
            for k, v in training_history_list[i].items():
                if "ssim" in k.lower() and "val" in k.lower():
                    val_ssim = np.asarray(v)
                elif "loss" in k.lower() and "val" in k.lower():
                    val_mse = np.asarray(v)
            val_ssiml2 = val_mse + (1 - val_ssim)

            if "mse" in metric_name.lower():
                loss = val_mse
            elif metric_name.lower() == "ssim":
                loss = val_ssim
            else:
                loss = val_ssiml2
 
            epochs = list(range(len(loss)))
            
            self.axes.plot(epochs, loss, label=lengends[i], linewidth=1)
            min_y_limit.append(min(loss))
            max_y_limit.append(max(loss))

        min_x_limit = min(epochs)
        max_x_limit = max(epochs)
        min_y_limit = min(min_y_limit)
        max_y_limit = max(max_y_limit)
        
        self.axes.set_xlim(min_x_limit, max_x_limit)
        if ylim is None:
            self.axes.set_ylim(min_y_limit, max_y_limit)
        else:
            self.axes.set_ylim(ylim[0], ylim[-1])

        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel(metric_name.upper())
        if sci_limits is None:
            sci_limits = (-4, 0)
        self.axes.ticklabel_format(axis='y', style='sci', scilimits=sci_limits)

        if legend_loc is None:
            self.axes.legend(loc='upper left', fontsize=6)
        else:
            self.axes.legend(loc=legend_loc, fontsize=6)
        plt.subplots_adjust(bottom=0.25)

def test():
    comp = compare_model_predictions(x_tests=np.ones((5, 428, 512, 1)), 
                                     y_preds=[np.ones((5, 84, 192, 1)), np.ones((5, 84, 192, 1))], 
                                     y_tests=np.ones((5, 84, 192, 1)))
    comp.show_fig()

if __name__ == "__main__":
    test()