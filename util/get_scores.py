import torchmetrics
import numpy as np
import torch

def get_metric_scores(metric: torchmetrics.Metric, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute per-sample metric scores using a torchmetrics.Metric instance.
    
    For each sample in y_true and y_pred, the function updates the metric,
    computes the score, and resets the metric for the next sample.
    
    Special handling is provided for metrics named "FrechetInceptionDistance", which 
    require at least 2 samples and additional processing (e.g., repeating channels).

    Args:
        metric (torchmetrics.Metric): A torchmetrics metric instance.
        y_true (np.ndarray): Ground truth images as a numpy array.
        y_pred (np.ndarray): Predicted images as a numpy array.

    Returns:
        np.ndarray: An array of metric scores computed for each sample.
    """
    metric_score_list = []
    for i in range(y_true.shape[0]):
        if "FrechetInceptionDistance" not in metric._get_name():
            yp = torch.tensor(y_pred[i:i+1], dtype=torch.float32)
            yt = torch.tensor(y_true[i:i+1], dtype=torch.float32)
            metric.update(yp, yt)
        else:
            # For FID, process two samples at a time and repeat channels if necessary.
            yp = torch.tensor(y_pred[i:i+2])
            yt = torch.tensor(y_true[i:i+2])
            if yt.shape[0] < 2:
                yp = yp.repeat(2, 1, 1, 1)
                yt = yt.repeat(2, 1, 1, 1)
            # FID update requires specifying real/fake.
            metric.update(yp.repeat(1, 3, 1, 1), real=False)
            metric.update(yt.repeat(1, 3, 1, 1), real=True)
        metric_score_list.append(metric.compute().item())
        metric.reset()
    metric_score_arr = np.asarray(metric_score_list)
    return metric_score_arr

def get_loss_scores(loss: torchmetrics.Metric, y_true, y_pred):
    """
    Compute per-sample loss scores using a loss metric function.
    
    For each sample in y_true and y_pred, the function calculates the loss value
    and collects these into an array.

    Args:
        loss (torchmetrics.Metric): A loss function or metric that can be called directly.
        y_true: Ground truth data.
        y_pred: Predicted data.

    Returns:
        np.ndarray: An array of loss values computed for each sample.
    """
    metric_loss_list = []
    for i in range(y_true.shape[0]):
        metric_loss_list.append(loss(y_true[[i]], y_pred[[i]]))
    metric_loss_arr = np.asarray(metric_loss_list)
    return metric_loss_arr

def get_avg_metric_scores(metric: torchmetrics.Metric, y_true, y_pred):
    """
    Compute the average metric score over all samples.
    
    Args:
        metric (torchmetrics.Metric): A torchmetrics metric instance.
        y_true: Ground truth data.
        y_pred: Predicted data.
    
    Returns:
        float: The average metric score.
    """
    metric_score_arr = get_metric_scores(metric, y_true, y_pred)
    return np.mean(metric_score_arr)

def get_avg_loss_scores(loss: torchmetrics.Metric, y_true, y_pred):
    """
    Compute the average loss over all samples.
    
    Args:
        loss (torchmetrics.Metric): A loss function or metric.
        y_true: Ground truth data.
        y_pred: Predicted data.
    
    Returns:
        float: The average loss.
    """
    metric_loss_arr = get_loss_scores(loss, y_true, y_pred)
    return np.mean(metric_loss_arr)

def get_std_metric_scores(metric: torchmetrics.Metric, y_true, y_pred):
    """
    Compute the standard deviation of the metric scores over all samples.
    
    Args:
        metric (torchmetrics.Metric): A torchmetrics metric instance.
        y_true: Ground truth data.
        y_pred: Predicted data.
    
    Returns:
        float: The standard deviation of metric scores.
    """
    metric_score_arr = get_metric_scores(metric, y_true, y_pred)
    return np.std(metric_score_arr)

def get_std_loss_scores(loss: torchmetrics.Metric, y_true, y_pred):
    """
    Compute the standard deviation of the loss values over all samples.
    
    Args:
        loss (torchmetrics.Metric): A loss function or metric.
        y_true: Ground truth data.
        y_pred: Predicted data.
    
    Returns:
        float: The standard deviation of loss values.
    """
    metric_loss_arr = get_loss_scores(loss, y_true, y_pred)
    return np.std(metric_loss_arr)
