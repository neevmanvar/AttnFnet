import os
import matplotlib.pyplot as plt
import torchmetrics.image
import torchmetrics.image.psnr
from util.handle_dirs import HandleTestDir
from util.get_scores import get_metric_scores, get_avg_metric_scores, get_std_metric_scores
import torch
import numpy as np
from util.handle_images import HistogramComparison, ImageComparison

import pickle
from alive_progress import alive_bar 
import hydra
from config.test_config import Config
import torchmetrics
from util.prepare_dataset import SLPDataset, load_model_predictions
from metrics.MeanPerPixelAcc import MeanPerPixelAcc

@hydra.main(version_base=None, config_name="test_config")
def my_hydra_app(cfg: Config) -> None:
    """
    Main function to compute various image quality metrics on test predictions.
    
    This function performs the following steps:
      1. Initializes directories and loads the prediction array.
      2. Loads the test dataset and corresponding ground truth labels.
      3. If using specific models ('bpbnet' or 'bpwnet'), normalizes the prediction and ground truth data.
      4. Computes several image quality metrics (e.g., FID, PSNR, SSIM, MAE, MSE, etc.) for each prediction.
      5. Determines the best and worst performing predictions based on each metric.
      6. Generates comparison images and histograms for the best and worst cases.
      7. Saves average and standard deviation scores, as well as indices for best/worst predictions, to disk.
      
    Args:
        cfg (Config): Configuration object loaded by Hydra containing dataset, model, and directory paths.
    """
    
    # Set up directories for saving results and handling predictions using the configuration.
    hd = HandleTestDir(cfg=cfg, clear_dirs=True)
    prediction_array_path = hd.get_model_predictions_path()
    result_dir = hd.get_model_results_dir()
    best_worst_metric_score_dir = hd.get_best_worst_metric_score_dir()
    random_pred_dir = hd.get_random_pred_dir()

    # Check if prediction array file exists. If not, raise an error instructing to run the prediction script.
    if not os.path.isfile(prediction_array_path):
        raise ValueError("prediction file not found, run python3 -m scrips.prediction --model %s --dataset %s to get predictions array" %
                         (cfg.data.model_name, cfg.data.data_name))

    # Load the test dataset and corresponding arrays.
    slptestset = SLPDataset(dataset_path=cfg.data.path, partition='test')
    x_test, y_test = slptestset._get_arrays()
    y_pred = load_model_predictions(prediction_path=prediction_array_path)

    # If using bpbnet or bpwnet, perform normalization on both predictions and ground truth.
    if cfg.data.model_name == "bpbnet" or cfg.data.model_name == "bpwnet":
        print("using bpbnet")
        # Determine the maximum value between y_test and y_pred.
        max_val = np.max(y_test) if np.max(y_test) > np.max(y_pred) else np.max(y_pred)
        # Load ground truth again from file and normalize both arrays.
        y_test = np.load(hd.MODEL_PREDICTIONS_DIR + "/y_test.npy")
        y_pred = (np.min(y_pred) + y_pred) / (max_val + np.min(y_pred))
        y_test = (np.min(y_test) + y_test) / (max_val + np.min(y_test))

    # Print basic statistics and shapes for debugging.
    print("min-max pred: ", np.min(y_pred), np.max(y_pred))
    print("predictions shape: ", y_pred.shape)
    print("min-max y_test: ", np.min(y_test), np.max(y_test))
    print("y_test shape: ", y_test.shape)
    print("min-max x_test: ", np.min(x_test), np.max(x_test))
    print("x_test shape: ", x_test.shape)

    # Initialize dictionaries to store indices, predictions, and scores for best and worst metric values.
    best_metric_index = {}
    worst_metric_index = {}
    best_metric_pred = {}
    best_metric_score = {}
    worst_metric_pred = {}
    worst_metric_score = {}

    # Dictionaries to store the scores computed for each metric.
    metrics_scores = {}
    avg_metrics_scores = {}
    std_metrics_scores = {}

    # Define a list of metric objects to compute.
    metrics = [
        # For FID, we adjust the feature size and input image dimensions.
        torchmetrics.image.FrechetInceptionDistance(feature=192, normalize=True, input_img_size=(x_test.shape[1]*3, x_test.shape[2], x_test.shape[3])), 
        MeanPerPixelAcc(),
        torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0),
        torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0),
        torchmetrics.MeanAbsoluteError(),
        torchmetrics.MeanSquaredError(),
        torchmetrics.image.RootMeanSquaredErrorUsingSlidingWindow(window_size=4)
    ]

    print("starting metric computation......")
    # Compute metrics scores using an alive progress bar for visual feedback.
    with alive_bar(len(metrics)) as bar:
        for metric in metrics:
            metric_name = metric._get_name()  # Get the name of the metric.
            # Compute raw metric scores between ground truth and prediction.
            metrics_scores[metric_name] = get_metric_scores(metric, y_test, y_pred)
            
            # Special handling for FrechetInceptionDistance: update metric with repeated 3-channel predictions.
            if metric_name == "FrechetInceptionDistance":
                metric.update(torch.tensor(y_pred).repeat(1, 3, 1, 1), real=False)
                metric.update(torch.tensor(y_pred).repeat(1, 3, 1, 1), real=True)
                avg_metrics_scores[metric_name] = np.asarray(metric.compute().item()).tolist()
            else:
                # For other metrics, compute the average score.
                avg_metrics_scores[metric_name] = np.mean(metrics_scores[metric_name]).tolist()
            
            # Compute the standard deviation of the metric scores.
            std_metrics_scores[metric_name] = get_std_metric_scores(metric, y_test, y_pred).tolist()
            bar()  # Update progress bar.
    print("finished metric computation.....")

    # Initialize helper classes for image and histogram comparisons.
    imagecomparision = ImageComparison()
    histogramcomparison = HistogramComparison(color_space='gray')

    # Determine the best and worst predictions based on each metric.
    with alive_bar(len(metrics)) as bar:
        for metric in metrics:
            metric_name = metric._get_name()
            
            # For most metrics, a higher score indicates better performance,
            # except for error metrics or FID (where lower is better).
            if "Error" not in metric_name and "FrechetInceptionDistance" not in metric_name and "FID" not in metric_name:
                # Best prediction is where the metric score is maximum.
                best_pred_index = (np.where(metrics_scores[metric_name] == np.max(metrics_scores[metric_name]))[0][0])
                best_metric_score[metric_name] = np.max(metrics_scores[metric_name])
                best_metric_pred[metric_name] = [x_test[best_pred_index], y_test[best_pred_index], y_pred[best_pred_index]]
                
                # Worst prediction is where the metric score is minimum.
                worst_pred_index = (np.where(metrics_scores[metric_name] == np.min(metrics_scores[metric_name]))[0][0])
                worst_metric_pred[metric_name] = [x_test[worst_pred_index], y_test[worst_pred_index], y_pred[worst_pred_index]]
                worst_metric_score[metric_name] = np.min(metrics_scores[metric_name])
            else:
                # For error metrics and FID, lower scores are considered better.
                best_pred_index = (np.where(metrics_scores[metric_name] == np.min(metrics_scores[metric_name]))[0][0])
                # Note: There is an update to best_metric_score twice in the code; the final value will be np.min.
                best_metric_score[metric_name] = np.min(metrics_scores[metric_name])
                best_metric_pred[metric_name] = [x_test[best_pred_index], y_test[best_pred_index], y_pred[best_pred_index]]
                
                worst_pred_index = (np.where(metrics_scores[metric_name] == np.max(metrics_scores[metric_name]))[0][0])
                worst_metric_pred[metric_name] = [x_test[worst_pred_index], y_test[worst_pred_index], y_pred[worst_pred_index]]
                worst_metric_score[metric_name] = np.max(metrics_scores[metric_name])

            # Save the best and worst indices for later reference.
            best_metric_index[metric_name] = best_pred_index
            worst_metric_index[metric_name] = worst_pred_index
            print("using metric with indexes", (metric_name, best_pred_index, type(best_pred_index)))
            print("using metric with indexes", (metric_name, worst_pred_index, type(worst_pred_index)))
            
            # Save histogram comparison images for best prediction.
            save_path = os.path.join(best_worst_metric_score_dir, "best_" + metric_name + "_" + str(best_metric_score[metric_name]) + "_histogram.png").replace("\\", "/")
            histogramcomparison.compare(y_test[best_pred_index], y_pred[best_pred_index],
                                         legends=["ground truth", "prediction"],
                                         titles=["best_" + metric_name + "=" + str(best_metric_score[metric_name])],
                                         save_path=save_path)
            # Save histogram comparison images for worst prediction.
            save_path = os.path.join(best_worst_metric_score_dir, "worst_" + metric_name + "_" + str(worst_metric_score[metric_name]) + "_histogram.png").replace("\\", "/")
            histogramcomparison.compare(y_test[worst_pred_index], y_pred[worst_pred_index],
                                         legends=["ground truth", "prediction"],
                                         titles=["worst_" + metric_name + "=" + str(worst_metric_score[metric_name])],
                                         save_path=save_path)
            bar()  # Update progress bar.

    # Save visual comparisons (images) for both best and worst predictions.
    for metric in metrics:
        metric_name = metric._get_name()
        # Generate save path and title for best predictions.
        save_path = os.path.join(best_worst_metric_score_dir, "best_" + metric_name + "_" + str(best_metric_score[metric_name]) + ".png").replace("\\", "/")
        title = "best_ " + metric_name + ": " + str(best_metric_score[metric_name])
        imagecomparision.show_images(best_metric_pred[metric_name], save_path, title)

        # Generate save path and title for worst predictions.
        save_path = os.path.join(best_worst_metric_score_dir, "worst_" + metric_name + "_" + str(worst_metric_score[metric_name]) + ".png").replace("\\", "/")
        title = "worst " + metric_name + ": " + str(worst_metric_score[metric_name])
        imagecomparision.show_images(worst_metric_pred[metric_name], save_path, title)

    # Generate and save comparison images for 15 random test frames.
    for i in range(15):
        index = np.random.randint(0, y_test.shape[0])
        save_path = os.path.join(random_pred_dir, "Frame_" + str(index) + ".png").replace("\\", "/")
        title = "Frame_" + str(index)
        imagecomparision.show_images([x_test[index], y_test[index], y_pred[index]], save_path, title)

    # Save the average metric scores as a JSON file.
    import json
    file_name = "average_metric_scores.txt"
    text_file = os.path.join(result_dir, file_name).replace("\\", "/")
    with open(text_file, 'w') as file:
        file.write(json.dumps(avg_metrics_scores))

    # Save the standard deviation of metric scores as a JSON file.
    file_name = "std_metric_scores.txt"
    text_file = os.path.join(result_dir, file_name).replace("\\", "/")
    with open(text_file, 'w') as file:
        file.write(json.dumps(std_metrics_scores))

    # Save indices of best metric scores using pickle.
    file_name = "best_metric_scores_index"
    file_path = os.path.join(result_dir, file_name).replace("\\", "/")
    with open(file_path, "wb") as file_pi:
        pickle.dump(best_metric_index, file_pi) 

    # Save indices of worst metric scores using pickle.
    file_name = "worst_metric_scores_index"
    file_path = os.path.join(result_dir, file_name).replace("\\", "/")
    with open(file_path, "wb") as file_pi:
        pickle.dump(worst_metric_index, file_pi)

    # Save all raw metric scores using pickle.
    file_name = "metric_scores"
    file_path = os.path.join(result_dir, file_name).replace("\\", "/")
    with open(file_path, "wb") as file_pi:
        pickle.dump(metrics_scores, file_pi)

# Run the hydra app.
my_hydra_app()
