import hydra
from config.test_config import Config
from util.handle_dirs import HandleTestDir
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, gaussian_filter
import glob
from alive_progress import alive_bar   
from util.prepare_dataset import SLPDataset, load_model_predictions
import numpy as np

def filter_large_blobs(binary_mask, area_thresh):
    """
    Keeps only the connected components ("blobs") in a binary mask that exceed a specified area threshold.
    
    This function uses connected-component labeling to detect blobs in the input binary mask
    (where nonzero values indicate the presence of a component). Only blobs with an area larger than
    the provided threshold are retained in the output mask.
    
    Args:
        binary_mask (numpy.ndarray): 2D array (binary image) representing pressure distribution,
                                     where 1 indicates pressure above a threshold and 0 otherwise.
        area_thresh (int): Minimum number of pixels required for a blob to be kept.
    
    Returns:
        numpy.ndarray: A binary mask of the same size as `binary_mask` with only the large blobs retained.
    """
    # Label connected components (blobs) in the binary mask.
    labeled_array, num_features = label(binary_mask)
    
    # Initialize an output mask, same shape as binary_mask, to accumulate large blobs.
    filtered_mask = np.zeros_like(binary_mask)
    
    # Loop through each detected blob (blob numbering starts at 1)
    for blob_num in range(1, num_features + 1):
        # Create a mask for the current blob.
        blob = (labeled_array == blob_num).astype(np.float32)
        
        # Retain only blobs that exceed the area threshold.
        if np.sum(blob) > area_thresh:
            filtered_mask += blob  # Add the blob to the output mask.
    
    # Ensure the output mask is binary (True where any blob is present).
    return (filtered_mask > 0).astype(np.float32)

def calculate_iou(y_true, y_pred, threshold=0.1, filter_blobs=False, area_thresh=100):
    """
    Calculates the Intersection over Union (IoU) between ground truth and predicted pressure distributions.
    
    The pressure maps are first binarized using a provided threshold. Optionally, the resulting binary images 
    can be filtered to keep only large connected components ("blobs") that exceed an area threshold. 
    The IoU is calculated per sample and then averaged over the batch.
    
    Args:
        y_true (numpy.ndarray): Ground truth pressure distribution array with shape [batch_size, height, width, 1].
        y_pred (numpy.ndarray): Predicted pressure distribution array with shape [batch_size, height, width, 1].
        threshold (float, optional): Threshold to binarize the pressure distribution (default: 0.1).
        filter_blobs (bool, optional): If True, filter the binary masks to retain only blobs above the area threshold.
        area_thresh (int, optional): Minimum pixel area for a blob to be retained (default: 100).
    
    Returns:
        float: Mean IoU score across the batch.
    """
    # Binarize ground truth and prediction using the threshold.
    y_true_binary = (y_true > threshold).astype(np.float32)
    y_pred_binary = (y_pred > threshold).astype(np.float32)

    # If filtering is enabled, apply the blob filtering function to the first sample of the batch.
    if filter_blobs:
        y_true_binary = filter_large_blobs(y_true_binary[0, :, :, 0], area_thresh)[np.newaxis, :, :, np.newaxis]
        y_pred_binary = filter_large_blobs(y_pred_binary[0, :, :, 0], area_thresh)[np.newaxis, :, :, np.newaxis]

    # Compute the intersection and union for each sample along the spatial dimensions.
    intersection = np.sum(np.logical_and(y_true_binary, y_pred_binary), axis=(1, 2, 3))
    union = np.sum(np.logical_or(y_true_binary, y_pred_binary), axis=(1, 2, 3))

    # Compute the IoU score per sample, with a small epsilon added to avoid division by zero.
    iou_scores = intersection / (union + 1e-16)

    # Return the mean IoU score across all samples.
    return np.mean(iou_scores)

def visualize_iou(y_true, y_pred, threshold=0.1, sample_index=0, filter_blobs=False, area_thresh=100, save_path=""):
    """
    Visualizes the IoU computation for a given sample from the batch.
    
    This function displays four images side by side:
      1. Ground truth pressure distribution.
      2. Predicted pressure distribution.
      3. Intersection mask (overlap between the binarized ground truth and prediction).
      4. Difference mask (areas where the ground truth and prediction differ).
    
    Optionally, both the ground truth and prediction can be filtered to only include significant blobs.
    The resulting plot is saved to the specified path.
    
    Args:
        y_true (numpy.ndarray): Ground truth pressure distribution array, shape [batch_size, height, width, 1].
        y_pred (numpy.ndarray): Predicted pressure distribution array, shape [batch_size, height, width, 1].
        threshold (float, optional): Threshold to binarize the images (default: 0.1).
        sample_index (int, optional): Index of the sample to be visualized (default: 0).
        filter_blobs (bool, optional): If True, filter out small blobs based on the area threshold.
        area_thresh (int, optional): Minimum area required for a blob to be kept (default: 100).
        save_path (str, optional): File path to save the generated figure.
    """
    # Extract the sample (removing the channel dimension for visualization purposes).
    true_sample = y_true[sample_index, :, :, 0]
    pred_sample = y_pred[sample_index, :, :, 0]

    # Binarize the samples using the given threshold.
    true_binary = (true_sample > threshold).astype(np.float32)
    pred_binary = (pred_sample > threshold).astype(np.float32)

    # Optionally apply blob filtering to remove small connected components.
    if filter_blobs:
        true_binary = filter_large_blobs(true_binary, area_thresh)
        pred_binary = filter_large_blobs(pred_binary, area_thresh)

    # Compute the intersection (logical AND) and difference (logical XOR) of the two binary masks.
    intersection = np.logical_and(true_binary, pred_binary).astype(np.float32)
    difference = np.logical_xor(true_binary, pred_binary).astype(np.float32)

    # Set up the plotting canvas with 4 subplots.
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    # Plot the ground truth image.
    ax[0].imshow(true_sample, cmap='seismic')
    ax[0].set_title('Ground Truth (Filtered)' if filter_blobs else 'Ground Truth')
    ax[0].axis('off')

    # Plot the prediction image.
    ax[1].imshow(pred_sample, cmap='seismic')
    ax[1].set_title('Prediction (Filtered)' if filter_blobs else 'Prediction')
    ax[1].axis('off')

    # Plot the intersection image, highlighting overlap regions.
    ax[2].imshow(intersection, cmap='Greens')
    ax[2].set_title('Intersection (IoU)')
    ax[2].axis('off')

    # Plot the difference image, highlighting mismatches between ground truth and prediction.
    ax[3].imshow(difference, cmap='Reds')
    ax[3].set_title('Difference (Error)')
    ax[3].axis('off')

    # Save the generated figure to the specified path.
    fig.savefig(save_path)

@hydra.main(version_base=None, config_name="test_config")
def my_hydra_app(cfg: Config) -> None:
    """
    Main execution function for testing pressure distribution prediction and evaluation.
    
    This function, which is managed by Hydra for configuration, performs the following steps:
      1. Sets up the test directories for storing results.
      2. Checks for the existence of a prediction file; if missing, instructs the user to run the prediction script.
      3. Loads the test dataset and model predictions.
      4. Applies dataset-specific calibration on both the ground truth and predictions.
      5. Computes the mean Intersection over Union (IoU) score.
      6. Visualizes and saves specific test samples with overlayed ground truth, predictions, and error maps.
    
    Args:
        cfg (Config): A configuration object loaded by Hydra that contains parameters for paths, model names,
                      dataset names, and calibration values.
    """
    # Initialize the HandleTestDir helper to manage directories for predictions and results.
    hd = HandleTestDir(cfg=cfg, clear_dirs=False)
    
    # Set the paths for the prediction array and results directory.
    prediction_array_path = hd.get_model_predictions_path()
    result_dir = hd.get_model_results_dir()
    
    # Create a directory for saving pose visualizations if it doesn't exist.
    pose_dirs = os.path.join(result_dir, "subject_pose").replace("\\", "/")
    if not os.path.exists(pose_dirs):
        os.makedirs(pose_dirs)
    
    # Remove any previous visualization files in the directory.
    [os.remove(f) for f in glob.glob(os.path.join(pose_dirs, "*.png").replace("\\", "/"))]

    # If the prediction file does not exist, prompt the user to run the prediction script.
    if not os.path.isfile(prediction_array_path):
        raise ValueError("prediction file not found, run python3 -m scrips.prediction --model %s --dataset %s to get predictions array" %
                         (cfg.data.model_name, cfg.data.data_name))

    # Load the test dataset using the custom SLPDataset class.
    slptestset = SLPDataset(dataset_path=cfg.data.path, partition='test')
    x_test, y_test = slptestset._get_arrays()
    
    # Obtain the pressure calibration scale for the test partition.
    press_calib_scale = slptestset._get_pressure_calibration(partition='test')
    
    # Load the predicted pressure distribution array.
    y_pred = load_model_predictions(prediction_path=prediction_array_path)

    # Calibrate the predictions and ground truth based on model type and dataset specifics.
    if cfg.data.model_name != "bpbnet" and cfg.data.model_name != "bpwnet":
        if cfg.data.data_name == "depth2bp_cleaned":
            # Log min/max values for debugging.
            print("min max values x_input: ", np.min(x_test), np.max(x_test))
            print("min max values y_test: ", np.min(y_test), np.max(y_test))
            print("min max values y_pred: ", np.min(y_pred), np.max(y_pred))
            # Apply calibration constants specific to the cleaned depth2bp dataset.
            calibrated_y_test = (y_test) * 101.52622953166929
            calibrated_y_pred = (y_pred) * 72.7732209260577
            # When ground truth is zero, set prediction to zero.
            calibrated_y_pred = np.where(y_test == 0, 0, calibrated_y_pred)

        elif "no_KPa" in cfg.data.data_name:
            # Log min/max values for debugging.
            print("min max values x_input: ", np.min(x_test), np.max(x_test))
            print("min max values y_test: ", np.min(y_test), np.max(y_test))
            print("min max values y_pred: ", np.min(y_pred), np.max(y_pred))
            # Calibrate using a pressure scale factor.
            calibrated_y_test = ((y_test * 171.82)) * press_calib_scale.reshape((-1, 1, 1, 1))
            calibrated_y_pred = ((y_pred * 171.82)) * press_calib_scale.reshape((-1, 1, 1, 1))
        else:
            # Log the input and output ranges.
            print("min max values x_input: ", np.min(x_test), np.max(x_test))
            print("min max values y_test: ", np.min(y_test), np.max(y_test))
            print("min max values y_pred: ", np.min(y_pred), np.max(y_pred))
            if cfg.data.data_name == "depth2bp":
                # Normalize the depth2bp dataset inputs and ground truth.
                x_test = (x_test + 1) / 2
                y_test = (y_test + 1) / 2

            # Calibrate with a multiplication factor and reshape the pressure calibration scale.
            calibrated_y_test = ((y_test * 255.0)) * press_calib_scale.reshape((-1, 1, 1, 1))
            calibrated_y_pred = ((y_pred * 255.0)) * press_calib_scale.reshape((-1, 1, 1, 1))
    else:
        # For bpbnet/bpwnet models, directly load the ground truth data.
        print("using bpb/bpwnet .........", cfg.data.model_name)
        print("min max values x_input: ", np.min(x_test), np.max(x_test))
        print("min max values y_test: ", np.min(y_test), np.max(y_test))
        print("min max values y_pred: ", np.min(y_pred), np.max(y_pred))
        
        calibrated_y_test = np.load(hd.MODEL_PREDICTIONS_DIR + "/y_test.npy")
        calibrated_y_pred = y_pred

    # Transpose the calibration arrays to match the expected shape [batch_size, height, width, channels].
    calibrated_y_test = np.transpose(calibrated_y_test, (0, 2, 3, 1))
    calibrated_y_pred = np.transpose(calibrated_y_pred, (0, 2, 3, 1))
    
    # Set threshold and area threshold parameters.
    th = 0.1
    area_th = 10

    # Calculate the mean IoU score for the test batch with blob filtering applied.
    iou_score = calculate_iou(calibrated_y_test, calibrated_y_pred, threshold=th, filter_blobs=True, area_thresh=area_th)
    print("Mean IoU Score with filled pose:", iou_score)

    # List of sample indices for which pose visualization will be generated.
    pose_indexes = [540, 781, 801, 815, 939]

    # For each selected sample index, generate and save the IoU visualization.
    for i in range(len(pose_indexes)):
        save_path = os.path.join(pose_dirs, "Depth_pose_Frame_" + str(pose_indexes[i]) + ".png").replace("\\", "/")
        title = "Frame_" + str(pose_indexes[i])
        visualize_iou(calibrated_y_test, calibrated_y_pred, threshold=th, sample_index=pose_indexes[i],
                      filter_blobs=True, area_thresh=area_th, save_path=save_path)

# Run the application when executed as a script.
if __name__ == "__main__":
    my_hydra_app()
