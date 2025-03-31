from util.handle_dirs import HandleTestDir
from alive_progress import alive_bar
from util.prepare_dataset import SLPDataset, load_model_predictions
import numpy as np
import os
import pickle
from util.compare_model_images import compare_model_predictions
import sys
import json
import hydra
from config.test_config import Config
import torchmetrics

@hydra.main(version_base=None, config_name="test_config")
def my_hydra_app(cfg: Config) -> None:
    """
    Main function to compare predictions from multiple models and generate visual comparisons.

    Workflow:
      1. Load metric scores, best/worst indices, and prediction arrays for several models:
         - Attnfnet, Unet, BPBnet, and BPWnet.
      2. Load the test dataset arrays.
      3. Transpose the arrays into image format (height, width, channels).
      4. Load additional ground truth (bpxnet_y_test) for later comparison.
      5. Create a comparison object from the utility function 'compare_model_predictions'.
      6. Generate and save visualizations:
         - Random prediction examples.
         - Best and worst predictions based on various metrics (e.g., FID, PSNR, MSE, MAE, SSIM, etc.).
         - Histograms and box plots of metric scores.
         - Scatter plots comparing measured weights against calculated and predicted weights.
         - Deviation plots of the calibrated pressure maps.
      7. (Optional) Code for comparing training history is provided but commented out.

    Args:
        cfg (Config): Configuration object provided by Hydra containing model, data, and directory paths.
    """
    
    # ------------------ Attnfnet Model ------------------
    # Initialize directories for the current model configuration.
    hd = HandleTestDir(cfg=cfg, clear_dirs=cfg.data.clear)
    attnfnet_prediction_array_path = hd.get_model_predictions_path()
    attnfnet_result_dir = hd.get_model_results_dir()

    # Load precomputed metric scores for attnfnet.
    metric_file_name = "metric_scores"
    metric_file_path = os.path.join(attnfnet_result_dir, metric_file_name).replace("\\", "/")
    with open(metric_file_path, "rb") as file_pi:
        attnfnet_metrics_scores = pickle.load(file_pi)

    # Load best metric scores indices.
    best_metric_name = "best_metric_scores_index"
    best_metric_file_path = os.path.join(attnfnet_result_dir, best_metric_name).replace("\\", "/")
    with open(best_metric_file_path, "rb") as file_pi:
        attnfnet_best_metric_scores_index = pickle.load(file_pi)

    # Load worst metric scores indices.
    worst_metric_name = "worst_metric_scores_index"
    worst_metric_file_path = os.path.join(attnfnet_result_dir, worst_metric_name).replace("\\", "/")
    with open(worst_metric_file_path, "rb") as file_pi:
        attnfnet_worst_metric_scores_index = pickle.load(file_pi)

    # Load test dataset and predictions for attnfnet.
    slptestset = SLPDataset(dataset_path=cfg.data.path, partition='test')
    x_test, y_test = slptestset._get_arrays()
    attnfnet_y_pred = load_model_predictions(prediction_path=attnfnet_prediction_array_path)

    # ------------------ Unet Model ------------------
    # Change model name in configuration to load Unet predictions.
    cfg.data.model_name = "unet"
    hd = HandleTestDir(cfg=cfg, clear_dirs=cfg.data.clear)
    unet_prediction_array_path = hd.get_model_predictions_path()
    unet_result_dir = hd.get_model_results_dir()

    # Load Unet metric scores.
    metric_file_name = "metric_scores"
    metric_file_path = os.path.join(unet_result_dir, metric_file_name).replace("\\", "/")
    with open(metric_file_path, "rb") as file_pi:
        unet_metrics_scores = pickle.load(file_pi)

    # Load Unet best metric scores indices.
    best_metric_name = "best_metric_scores_index"
    best_metric_file_path = os.path.join(unet_result_dir, best_metric_name).replace("\\", "/")
    with open(best_metric_file_path, "rb") as file_pi:
        unet_best_metric_scores_index = pickle.load(file_pi)

    # Load Unet worst metric scores indices.
    worst_metric_name = "worst_metric_scores_index"
    worst_metric_file_path = os.path.join(unet_result_dir, worst_metric_name).replace("\\", "/")
    with open(worst_metric_file_path, "rb") as file_pi:
        unet_worst_metric_scores_index = pickle.load(file_pi)

    # Load Unet predictions.
    unet_y_pred = load_model_predictions(prediction_path=unet_prediction_array_path)

    # ------------------ BPBnet Model ------------------
    # Change model name in configuration to load BPBnet predictions.
    cfg.data.model_name = "bpbnet"
    hd = HandleTestDir(cfg=cfg, clear_dirs=cfg.data.clear)
    bpbnet_prediction_array_path = hd.get_model_predictions_path()
    bpbnet_result_dir = hd.get_model_results_dir()

    # Load BPBnet metric scores.
    metric_file_name = "metric_scores"
    metric_file_path = os.path.join(bpbnet_result_dir, metric_file_name).replace("\\", "/")
    with open(metric_file_path, "rb") as file_pi:
        bpbnet_metrics_scores = pickle.load(file_pi)

    # Load BPBnet best metric scores indices.
    best_metric_name = "best_metric_scores_index"
    best_metric_file_path = os.path.join(bpbnet_result_dir, best_metric_name).replace("\\", "/")
    with open(best_metric_file_path, "rb") as file_pi:
        bpbnet_best_metric_scores_index = pickle.load(file_pi)

    # Load BPBnet worst metric scores indices.
    worst_metric_name = "worst_metric_scores_index"
    worst_metric_file_path = os.path.join(bpbnet_result_dir, worst_metric_name).replace("\\", "/")
    with open(worst_metric_file_path, "rb") as file_pi:
        bpbnet_worst_metric_scores_index = pickle.load(file_pi)

    # Load BPBnet predictions.
    bpbnet_y_pred = load_model_predictions(prediction_path=bpbnet_prediction_array_path)

    # ------------------ BPWnet Model ------------------
    # Change model name in configuration to load BPWnet predictions.
    cfg.data.model_name = "bpwnet"
    hd = HandleTestDir(cfg=cfg, clear_dirs=cfg.data.clear)
    bpwnet_prediction_array_path = hd.get_model_predictions_path()
    bpwnet_result_dir = hd.get_model_results_dir()

    # Load BPWnet metric scores.
    metric_file_name = "metric_scores"
    metric_file_path = os.path.join(bpwnet_result_dir, metric_file_name).replace("\\", "/")
    with open(metric_file_path, "rb") as file_pi:
        bpwnet_metrics_scores = pickle.load(file_pi)

    # Load BPWnet best metric scores indices.
    best_metric_name = "best_metric_scores_index"
    best_metric_file_path = os.path.join(bpwnet_result_dir, best_metric_name).replace("\\", "/")
    with open(best_metric_file_path, "rb") as file_pi:
        bpwnet_best_metric_scores_index = pickle.load(file_pi)

    # Load BPWnet worst metric scores indices.
    worst_metric_name = "worst_metric_scores_index"
    worst_metric_file_path = os.path.join(bpwnet_result_dir, worst_metric_name).replace("\\", "/")
    with open(worst_metric_file_path, "rb") as file_pi:
        bpwnet_worst_metric_scores_index = pickle.load(file_pi)

    # Load BPWnet predictions.
    bpwnet_y_pred = load_model_predictions(prediction_path=bpwnet_prediction_array_path)

    # ------------------ Data Preparation ------------------
    # Transpose test inputs and labels to (height, width, channels) image format.
    x_test = np.transpose(x_test, (0, 2, 3, 1))
    y_test = np.transpose(y_test, (0, 2, 3, 1))
    unet_y_pred = np.transpose(unet_y_pred, (0, 2, 3, 1))
    attnfnet_y_pred = np.transpose(attnfnet_y_pred, (0, 2, 3, 1))
    bpbnet_y_pred = np.transpose(bpbnet_y_pred, (0, 2, 3, 1))
    bpwnet_y_pred = np.transpose(bpwnet_y_pred, (0, 2, 3, 1))

    # Load additional ground truth (bpxnet_y_test) from disk and transpose it.
    bpxnet_y_test = np.load(hd.MODEL_PREDICTIONS_DIR + "/y_test.npy")
    bpxnet_y_test = np.transpose(bpxnet_y_test, (0, 2, 3, 1))

    # Print shapes for verification.
    print("predictions shape: ", unet_y_pred.shape, bpbnet_y_pred.shape, bpwnet_y_pred.shape, attnfnet_y_pred.shape)
    print("input shape: ", x_test.shape)

    # ------------------ Model Comparison ------------------
    # Get directory to save comparison results.
    comparision_result_dir = hd.get_result_comparision_dir()

    # Define the list of models to be compared.
    model_list = ['Unet', 'Attnfnet', 'BPBnet', 'BPWnet']

    # Create a comparison object that will generate the visualizations.
    comp_preds = compare_model_predictions(
        x_tests=np.rot90(x_test, k=1, axes=(1, 2)),
        y_tests=np.rot90(y_test, k=1, axes=(1, 2)),
        y_preds=[
            np.rot90(unet_y_pred, k=1, axes=(1, 2)),
            np.rot90(attnfnet_y_pred, k=1, axes=(1, 2)),
            np.rot90(bpbnet_y_pred, k=1, axes=(1, 2)),
            np.rot90(bpwnet_y_pred, k=1, axes=(1, 2))
        ],
        model_list=model_list
    )

    # ------------------ Generate and Save Visualizations ------------------
    # Get random predictions based on a specified list of indices.
    comp_preds.get_random_preds(n_rows=7, index_list=[334, 16, 765, 11, 695, 932, 933])
    save_path = os.path.join(comparision_result_dir, "random_predictions.png").replace("\\", "./")
    comp_preds.save_fig(save_path)
    comp_preds.close_fig()

    # Generate best prediction images for selected metrics and save the figure.
    comp_preds.get_bestpreds_images(
        metric_scores_list=[unet_metrics_scores, attnfnet_metrics_scores, bpbnet_metrics_scores, bpwnet_metrics_scores],
        best_metric_score_index_list=[unet_best_metric_scores_index, attnfnet_best_metric_scores_index, bpbnet_best_metric_scores_index, bpwnet_best_metric_scores_index],
        metric_names=["FrechetInceptionDistance", "PeakSignalNoiseRatio", "MeanPerPixelAcc"]
    )
    save_path = os.path.join(comparision_result_dir, "best_predictions_FID_PSNR_PPA.png").replace("\\", "/")
    comp_preds.save_fig(save_path)

    # Generate worst prediction images for selected metrics and save the figure.
    comp_preds.get_worstpreds_images(
        metric_scores_list=[unet_metrics_scores, attnfnet_metrics_scores, bpbnet_metrics_scores, bpwnet_metrics_scores],
        worst_metric_score_index_list=[unet_worst_metric_scores_index, attnfnet_worst_metric_scores_index, bpbnet_worst_metric_scores_index, bpwnet_worst_metric_scores_index],
        metric_names=["FrechetInceptionDistance", "PeakSignalNoiseRatio", "MeanPerPixelAcc"]
    )
    save_path = os.path.join(comparision_result_dir, "worst_predictions_FID_PSNR_MPPA.png").replace("\\", "/")
    comp_preds.save_fig(save_path)

    # Generate best prediction images for error metrics (MSE, MAE, SSIM) and save the figure.
    comp_preds.get_bestpreds_images(
        metric_scores_list=[unet_metrics_scores, attnfnet_metrics_scores, bpbnet_metrics_scores, bpwnet_metrics_scores],
        best_metric_score_index_list=[unet_best_metric_scores_index, attnfnet_best_metric_scores_index, bpbnet_best_metric_scores_index, bpwnet_best_metric_scores_index],
        metric_names=["MeanSquaredError", "MeanAbsoluteError", "StructuralSimilarityIndexMeasure"]
    )
    save_path = os.path.join(comparision_result_dir, "best_predictions_MSE_MAE_SSIM.png").replace("\\", "/")
    comp_preds.save_fig(save_path)
    
    # Generate worst prediction images for error metrics (MSE, MAE, SSIM) and save the figure.
    comp_preds.get_worstpreds_images(
        metric_scores_list=[unet_metrics_scores, attnfnet_metrics_scores, bpbnet_metrics_scores, bpwnet_metrics_scores],
        worst_metric_score_index_list=[unet_worst_metric_scores_index, attnfnet_worst_metric_scores_index, bpbnet_worst_metric_scores_index, bpwnet_worst_metric_scores_index],
        metric_names=["MeanSquaredError", "MeanAbsoluteError", "StructuralSimilarityIndexMeasure"]
    )
    save_path = os.path.join(comparision_result_dir, "worst_predictions_MSE_MAE_SSIM.png").replace("\\", "/")
    comp_preds.save_fig(save_path)
    
    # Generate histograms for best/worst predictions based on FID and MSE.
    comp_preds.get_best_worst_hist(
        metric_scores_list=[unet_metrics_scores, attnfnet_metrics_scores, bpbnet_metrics_scores, bpwnet_metrics_scores],
        best_metric_score_index_list=[attnfnet_best_metric_scores_index],
        worst_metric_score_index_list=[attnfnet_worst_metric_scores_index],
        metric_names=["FrechetInceptionDistance", "MeanSquaredError"]
    )
    save_path = os.path.join(comparision_result_dir, "histograms_FID_MSE.png").replace("\\", "/")
    comp_preds.save_fig(save_path)
    
    # Generate histograms for best/worst predictions based on SSIM and PSNR.
    comp_preds.get_best_worst_hist(
        metric_scores_list=[unet_metrics_scores, attnfnet_metrics_scores, bpbnet_metrics_scores, bpwnet_metrics_scores],
        best_metric_score_index_list=[attnfnet_best_metric_scores_index],
        worst_metric_score_index_list=[attnfnet_worst_metric_scores_index],
        metric_names=["StructuralSimilarityIndexMeasure", "PeakSignalNoiseRatio"]
    )
    save_path = os.path.join(comparision_result_dir, "histograms_SSIM_PSNR.png").replace("\\", "/")
    comp_preds.save_fig(save_path)

    # Generate box plots for overall metric score distributions.
    comp_preds.get_box_plots(
        metric_scores_list=[unet_metrics_scores, attnfnet_metrics_scores, bpbnet_metrics_scores, bpwnet_metrics_scores],
        metric_names=["FrechetInceptionDistance", "MeanSquaredError", "MeanPerPixelAcc", "StructuralSimilarityIndexMeasure"]
    )
    save_path = os.path.join(comparision_result_dir, "metric_scores_box_plots.png").replace("\\", "/")
    comp_preds.save_fig(save_path)

    # ------------------ Weight Comparison ------------------
    # Load weight calculation JSON files for the different models.
    weight_calculations_unet_path = os.path.join(unet_result_dir, "average_weights_calculations.json")
    weight_calculations_attnfnet_path = os.path.join(attnfnet_result_dir, "average_weights_calculations.json")
    weight_calculations_bpbnet_path = os.path.join(bpbnet_result_dir, "average_weights_calculations.json")
    weight_calculations_bpwnet_path = os.path.join(bpwnet_result_dir, "average_weights_calculations.json")

    # For Unet.
    with open(weight_calculations_unet_path) as f:
        data = json.load(f)
    measured_weights = data["measured_weights"]
    predicted_weights_unet = data["predicted_weights"]

    # For Attnfnet.
    with open(weight_calculations_attnfnet_path) as f:
        data = json.load(f)
    predicted_weights_attnfnet = data["predicted_weights"]
    calculated_weights = data["calculated_weights"]

    # For BPBnet.
    with open(weight_calculations_bpbnet_path) as f:
        data = json.load(f)
    predicted_weights_bpbnet = data["predicted_weights"]

    # For BPWnet.
    with open(weight_calculations_bpwnet_path) as f:
        data = json.load(f)
    predicted_weights_bpwnet = data["predicted_weights"]

    # Generate a scatter plot comparing measured weights, calculated ground truth weights, and predicted weights.
    comp_preds.weight_scatter_plot(
        measured_weights=measured_weights,
        calculated_ground_truth_weights=calculated_weights,
        predicted_weights_list=[predicted_weights_unet, predicted_weights_attnfnet, predicted_weights_bpbnet, predicted_weights_bpwnet],
        splot=False
    )
    save_path = os.path.join(comparision_result_dir, "weight_comparision.png").replace("\\", "/")
    comp_preds.save_fig(save_path)

    # ------------------ Pressure Calibration and Deviation Plots ------------------
    # Obtain pressure calibration scale.
    press_calib_scale = slptestset._get_pressure_calibration("test")
    if cfg.data.data_name == "depth2bp_cleaned":
        press_calib_scale = [101.52622953166929, 72.7732209260577]
        print("using pressure tuple scale...")
    elif cfg.data.data_name == "depth2bp_cleaned_no_KPa":
        press_calib_scale = press_calib_scale * 171.82
        print("using pressure 171.82*press scale...")
    else:
        press_calib_scale = press_calib_scale * 255.0
        print("using pressure 255.0*press scale...")
    
    # Prepare calibrated predictions for deviation plots.
    calibrated_y_preds = [
        np.rot90(unet_y_pred * press_calib_scale.reshape((-1, 1, 1, 1)), k=1, axes=(1, 2)),
        np.rot90(attnfnet_y_pred * press_calib_scale.reshape((-1, 1, 1, 1)), k=1, axes=(1, 2)),
        np.rot90(bpbnet_y_pred, k=1, axes=(1, 2)),
        np.rot90(bpwnet_y_pred, k=1, axes=(1, 2))
    ]

    # Close any open figure before generating deviation plots.
    comp_preds.close_fig()
    comp_preds.get_deviation_plots(
        press_calibrated_scale=press_calib_scale,
        colorbar_label="Pressure (kPa)",
        calibrated_y_preds=calibrated_y_preds,
        calibrated_y_tests=[
            np.rot90(y_test, k=1, axes=(1, 2)),
            np.rot90(y_test, k=1, axes=(1, 2)),
            np.rot90(bpxnet_y_test, k=1, axes=(1, 2)),
            np.rot90(bpxnet_y_test, k=1, axes=(1, 2))
        ]
    )
    save_path = os.path.join(comparision_result_dir, "deviations.png").replace("\\", "/")
    comp_preds.save_fig(save_path)
    
    comp_preds.close_fig()
    comp_preds.get_deviation_plots(
        press_calibrated_scale=press_calib_scale,
        Filter_block_size=2,
        colorbar_label="Pressure (kPa)",
        calibrated_y_preds=calibrated_y_preds
    )
    save_path = os.path.join(comparision_result_dir, "region_deviations.png").replace("\\", "/")
    comp_preds.save_fig(save_path)
    
my_hydra_app()
