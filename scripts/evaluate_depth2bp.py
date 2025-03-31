# import tensorflow as tf
import hydra
from config.test_config import Config
import os
from util.handle_dirs import HandleTestDir
import numpy as np
from util.handle_images import ImageComparison, compare_weights_visualizations, plot_deviation_map
from alive_progress import alive_bar   
from util.prepare_dataset import SLPDataset, load_model_predictions
import sys
import torch
import torchmetrics
import torchvision
from util.get_scores import get_avg_metric_scores, get_metric_scores


@hydra.main(version_base=None, config_name="test_config")
def my_hydra_app(cfg: Config) -> None:
    """
    Main function to calibrate predictions, compute evaluation metrics, and visualize weight estimations and deviations.
    
    Workflow:
      1. Initialize directories and load dataset splits (train, validation, test) along with predictions.
      2. Depending on the model and data configuration, calibrate the ground truth and predicted pressure values.
         - Different calibration formulas are applied based on the dataset name and model type.
      3. Display random frame visualizations from the test set for inspection.
      4. Compute Mean Absolute Error (MAE) and Mean Squared Error (MSE) metrics on the calibrated data.
      5. Resize the calibrated images if needed.
      6. Estimate weights per subject:
         - For each frame in the test set, compute taxel weights based on the pressure values.
         - Aggregate weights over every 45 frames to get per-person weight estimates.
      7. Compare the predicted and calculated weights with measured weights.
      8. Save average calibrated metric scores and weight comparisons to disk.
      9. Compile the calibrated pressure maps into different posture groups (supine, left side, right side).
     10. Generate and save deviation maps for each posture group.
     
    Args:
        cfg (Config): Configuration object provided by Hydra containing model, data, and directory paths.
    """
    
    # Initialize directories and configuration-related paths.
    hd = HandleTestDir(cfg=cfg, clear_dirs=cfg.data.clear)
    dataset_dir = hd.get_dataset_dir()
    train_config_file = hd.get_train_config()
    prediction_array_path = hd.get_model_predictions_path()
    result_dir = hd.get_model_results_dir()
    random_pred_dir = hd.get_random_pred_dir()

    # Ensure the prediction array file exists; otherwise, prompt to run the prediction script.
    if not os.path.isfile(prediction_array_path):
        raise ValueError("prediction file not found, run python3 -m scrips.prediction --model %s --dataset %s to get predictions array" %
                         (cfg.data.model_name, cfg.data.data_name))

    # Load the test dataset and its associated arrays.
    slptestset = SLPDataset(dataset_path=cfg.data.path, partition='test')
    x_test, y_test = slptestset._get_arrays()
    press_calib_scale = slptestset._get_pressure_calibration(partition='test')
    weights_frame = slptestset._get_weights_frame()
    y_pred = load_model_predictions(prediction_path=prediction_array_path)

    # Load the training and validation sets for calibration purposes.
    slptrainset = SLPDataset(dataset_path=cfg.data.path, partition='train')
    x_train, y_train = slptrainset._get_arrays()
    p_train_scale = slptrainset._get_pressure_calibration(partition="train")
    
    slpvalset = SLPDataset(dataset_path=cfg.data.path, partition='val')
    x_val, y_val = slpvalset._get_arrays()
    p_val_scale = slpvalset._get_pressure_calibration(partition="val")

    # Calibrate the predictions and ground truth based on model and dataset type.
    if cfg.data.model_name != "bpbnet" and cfg.data.model_name != "bpwnet":
        if cfg.data.data_name == "depth2bp_cleaned":
            print("min max values x_input: ", np.min(x_test), np.max(x_test))
            print("min max values y_test: ", np.min(y_test), np.max(y_test))
            print("min max values y_pred: ", np.min(y_pred), np.max(y_pred))
            # Apply calibration constants specific to the cleaned depth2bp dataset.
            calibrated_y_test = (y_test) * 101.52622953166929
            calibrated_y_pred = (y_pred) * 72.7732209260577
            # Set predictions to 0 where ground truth is 0.
            calibrated_y_pred = np.where(y_test == 0, 0, calibrated_y_pred)

            calibrated_y_train = (y_train) * 101.52622953166929
            calibrated_y_val = (y_val) * 101.52622953166929

        elif "no_KPa" in cfg.data.data_name:
            print("min max values x_input: ", np.min(x_test), np.max(x_test))
            print("min max values y_test: ", np.min(y_test), np.max(y_test))
            print("min max values y_pred: ", np.min(y_pred), np.max(y_pred))
            # Apply calibration using pressure scale factors.
            calibrated_y_test = ((y_test * 171.82)) * press_calib_scale.reshape((-1, 1, 1, 1))
            calibrated_y_pred = ((y_pred * 171.82)) * press_calib_scale.reshape((-1, 1, 1, 1))

            calibrated_y_train = ((y_train * 196.78)) * p_train_scale.reshape((-1, 1, 1, 1))
            calibrated_y_val = ((y_val * 168.52)) * p_val_scale.reshape((-1, 1, 1, 1))
        else:
            print("min max values x_input: ", np.min(x_test), np.max(x_test))
            print("min max values y_test: ", np.min(y_test), np.max(y_test))
            print("min max values y_pred: ", np.min(y_pred), np.max(y_pred))
            if cfg.data.data_name == "depth2bp":
                # Normalize inputs for depth2bp dataset.
                x_test = (x_test + 1) / 2
                y_test = (y_test + 1) / 2

            # Calibrate using a multiplication factor and pressure calibration scale.
            calibrated_y_test = ((y_test * 255.0)) * press_calib_scale.reshape((-1, 1, 1, 1))
            calibrated_y_pred = ((y_pred * 255.0)) * press_calib_scale.reshape((-1, 1, 1, 1))

            calibrated_y_train = ((y_train * 255.0)) * p_train_scale.reshape((-1, 1, 1, 1))
            calibrated_y_val = ((y_val * 255.0)) * p_val_scale.reshape((-1, 1, 1, 1))
    else:
        # For bpbnet/bpwnet models, load ground truth from file and use direct predictions.
        print("using bpb/bpwnet .........", cfg.data.model_name)
        print("min max values x_input: ", np.min(x_test), np.max(x_test))
        print("min max values y_test: ", np.min(y_test), np.max(y_test))
        print("min max values y_pred: ", np.min(y_pred), np.max(y_pred))
        
        calibrated_y_test = np.load(hd.MODEL_PREDICTIONS_DIR + "/y_test.npy")
        calibrated_y_pred = y_pred

        calibrated_y_train = ((y_train * 196.78)) * p_train_scale.reshape((-1, 1, 1, 1))
        calibrated_y_val = ((y_val * 168.52)) * p_val_scale.reshape((-1, 1, 1, 1))

    # Print calibrated statistics for debugging.
    print("calibrated min max values x_input: ", np.min(x_test), np.max(x_test))
    print("calibrated min max values y_train: ", np.min(calibrated_y_train), np.max(calibrated_y_train))
    print("calibrated min max values y_val: ", np.min(calibrated_y_val), np.max(calibrated_y_val))
    print("calibrated min max values y_test: ", np.min(calibrated_y_test), np.max(calibrated_y_test))
    print("calibrated min max values y_pred: ", np.min(calibrated_y_pred), np.max(calibrated_y_pred))
    print("shape of y_test ", y_test.shape)
    print("shape of y_pred ", y_pred.shape)
    print("input shape of X_test: ", x_test.shape)

    # Visualize 15 random frames from the test set.
    imagecomparision = ImageComparison()
    for i in range(15):
        index = np.random.randint(0, y_test.shape[0])
        save_path = os.path.join(random_pred_dir, "Calibrated_Frame_" + str(index) + ".png").replace("\\", "/")
        title = "Frame_" + str(index)
        imagecomparision.show_images([x_test[index], y_test[index], y_pred[index]], save_path, title)

    # Compute evaluation metrics (Mean Absolute Error and Mean Squared Error) on calibrated data.
    metrics_scores = {}
    avg_metrics_scores = {}
    metrics = [torchmetrics.MeanAbsoluteError(), torchmetrics.MeanSquaredError()]
    print("starting metric computation......")
    with alive_bar(len(metrics)) as bar:
        for metric in metrics:
            key = "calibrated" + metric._get_name()
            # Compute metric scores between calibrated test and predicted values.
            metrics_scores[key] = get_metric_scores(metric, calibrated_y_test, calibrated_y_pred)
            # Calculate average score.
            avg_metrics_scores[key] = get_avg_metric_scores(metric, calibrated_y_test, calibrated_y_pred).tolist()
            # Compute standard deviation and standard error.
            avg_metrics_scores[key + " std"] = np.std(metrics_scores[key]).tolist()
            avg_metrics_scores[key + " std_error"] = (np.std(metrics_scores[key]) / np.sqrt(metrics_scores[key].shape[0])).tolist()
            bar()
    print("finished metric computation......")
    print(avg_metrics_scores)

    # Save average calibrated metric scores as a JSON file.
    import json
    file_name = "average_calibrated_metric_scores.txt"
    text_file = os.path.join(result_dir, file_name).replace("\\", "/")
    with open(text_file, 'w') as file:
        file.write(json.dumps(avg_metrics_scores))

    # Resize calibrated images if dimensions do not match the expected size (84,192).
    if (calibrated_y_pred.shape[1], calibrated_y_test.shape[2]) != (84, 192):
        resizer = torchvision.transforms.Resize(size=(84, 192), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        calibrated_y_test = np.asarray(resizer(torch.tensor(calibrated_y_test)))
        calibrated_y_pred = np.asarray(resizer(torch.tensor(calibrated_y_pred)))

    # Calculate predicted weights using calibrated predictions.
    measured_weights = weights_frame["measured weights (kg)"][81:]
    measured_weights = np.asarray(measured_weights)
    measured_weights_with_frames = np.repeat(measured_weights, 45)
    SENSEL_AREA = 1.03226 / 10000  # in square meters
    g = 9.80  # gravitational acceleration in m/s^2

    predicted_weights = []
    weight_per_person_list = []
    with alive_bar(calibrated_y_pred.shape[0]) as bar:
        for i in range(calibrated_y_pred.shape[0]):
            taxel_weight_list = []
            # For each pixel (taxel), compute the mass if the calibrated value is greater than zero.
            for p in np.nditer(calibrated_y_pred[i]):
                if p > 0.0:
                    mass = p * 1000 * SENSEL_AREA / g
                    taxel_weight_list.append(mass)
            weight_per_frame = np.sum(taxel_weight_list)
            weight_per_person_list.append(weight_per_frame)
            # Aggregate weight every 45 frames (assumed per subject).
            if (i + 1) % 45 == 0:
                average_per_person_weight = np.mean(weight_per_person_list)
                print("prediccted weight per 45 frame", average_per_person_weight)
                weight_per_person_list = []
                predicted_weights.append(average_per_person_weight.tolist())
            bar()

    # Calculate weights using the calibrated ground truth.
    calculated_weights = []
    weight_per_person_list = []
    with alive_bar(calibrated_y_pred.shape[0]) as bar:
        for i in range(calibrated_y_pred.shape[0]):
            taxel_weight_list = []
            for p in np.nditer(calibrated_y_test[i]):
                if p > 0.0:
                    mass = p * 1000 * SENSEL_AREA / g
                    taxel_weight_list.append(mass)
            weight_per_frame = np.sum(taxel_weight_list)
            weight_per_person_list.append(weight_per_frame)
            if (i + 1) % 45 == 0:
                average_per_person_weight = np.mean(weight_per_person_list)
                print("calculated weight per 45 frame", average_per_person_weight)
                weight_per_person_list = []
                calculated_weights.append(average_per_person_weight)
            bar()

    calculated_weights = np.asarray(calculated_weights)
    # Print comparisons between predicted, calculated, and measured weights.
    print("predicted weights vs calibrated weights vs measured weights: ")
    print("predicted weights: ", predicted_weights)
    print("calculated weights: ", calculated_weights)
    print("measured weights: ", measured_weights)
    print("difference between calculated and predicted weights: ", calculated_weights - predicted_weights)
    print("average difference calc-pred: ", np.mean(calculated_weights - measured_weights))
    print("difference between measured and predicted weights: ", measured_weights - predicted_weights)
    print("average difference measured-pred: ", np.mean(predicted_weights - measured_weights))

    # Compute absolute differences between predicted and measured weights.
    difference_pred_measured = np.subtract(measured_weights, predicted_weights)
    difference_calc_measured = np.subtract(calculated_weights, predicted_weights)

    # Compile weight comparison metrics into a dictionary.
    avg_weights = {}
    avg_weights["measured_weights"] = measured_weights.tolist()
    avg_weights["calculated_weights"] = calculated_weights.tolist()
    avg_weights["predicted_weights"] = predicted_weights
    avg_weights["absolute_difference_pred_measured"] = list(np.absolute(difference_pred_measured))
    avg_weights["absolute_difference_calc_measured"] = list(np.absolute(difference_calc_measured))
    avg_weights["avg_difference_pred_measured"] = str(np.mean(np.absolute(difference_pred_measured)))
    avg_weights["avg_difference_calc_measured"] = str(np.mean(np.absolute(difference_calc_measured)))

    for k, v in avg_weights.items():
        print(k, type(v))

    # Save the weight comparison metrics to a JSON file.
    file_name = "average_weights_calculations.json"
    text_file = os.path.join(result_dir, file_name).replace("\\", "/")
    with open(text_file, 'w') as file:
        file.write(json.dumps(avg_weights))

    # Visualize and save weight comparisons.
    save_path = os.path.join(result_dir, "subject_weights_plots").replace("\\", "/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    compare_weights_visualizations(measured_weights, calculated_weights, predicted_weights, save_path=save_path)

    print("compile arrays according to postures....")

    # Initialize empty lists to split calibrated data into posture groups.
    supine_postures_y_test = []
    left_side_postures_y_test = []
    right_side_postures_y_test = []

    supine_postures_y_pred = []
    left_side_postures_y_pred = []
    right_side_postures_y_pred = []

    # Split the calibrated test and prediction arrays into three posture groups per subject.
    for i in range(calibrated_y_pred.shape[0] // 45):  # assume 21 subjects if there are 45 frames per subject
        start_index = i * 45
        end_index = start_index + 45

        # For ground truth: first 15 frames are supine, next 15 left side, last 15 right side.
        supine_postures_y_test.append(calibrated_y_test[start_index:start_index + 15])
        left_side_postures_y_test.append(calibrated_y_test[start_index + 15:start_index + 30])
        right_side_postures_y_test.append(calibrated_y_test[start_index + 30:end_index])

        # For predictions, split in the same manner.
        supine_postures_y_pred.append(calibrated_y_pred[start_index:start_index + 15])
        left_side_postures_y_pred.append(calibrated_y_pred[start_index + 15:start_index + 30])
        right_side_postures_y_pred.append(calibrated_y_pred[start_index + 30:end_index])

    # Convert the lists to numpy arrays and transpose them into image format.
    supine_postures_y_test = np.transpose(np.concatenate(supine_postures_y_test, axis=0), (0, 2, 3, 1))
    left_side_postures_y_test = np.transpose(np.concatenate(left_side_postures_y_test, axis=0), (0, 2, 3, 1))
    right_side_postures_y_test = np.transpose(np.concatenate(right_side_postures_y_test, axis=0), (0, 2, 3, 1))

    supine_postures_y_pred = np.transpose(np.concatenate(supine_postures_y_pred, axis=0), (0, 2, 3, 1))
    left_side_postures_y_pred = np.transpose(np.concatenate(left_side_postures_y_pred, axis=0), (0, 2, 3, 1))
    right_side_postures_y_pred = np.transpose(np.concatenate(right_side_postures_y_pred, axis=0), (0, 2, 3, 1))

    # Print output shapes for verification.
    print(f"Supine Postures y_test Shape: {supine_postures_y_test.shape}")
    print(f"Left Side Postures y_test Shape: {left_side_postures_y_test.shape}")
    print(f"Right Side Postures y_test Shape: {right_side_postures_y_test.shape}")

    print(f"Supine Postures y_pred Shape: {supine_postures_y_pred.shape}")
    print(f"Left Side Postures y_pred Shape: {left_side_postures_y_pred.shape}")
    print(f"Right Side Postures y_pred Shape: {right_side_postures_y_pred.shape}")

    # Create directory for saving deviation maps if it does not exist.
    deviation_map_save_dir = os.path.join(result_dir, "deviation_plots").replace("\\", "/")
    if not os.path.exists(deviation_map_save_dir):
        os.makedirs(deviation_map_save_dir)

    # Plot deviation maps for each posture group and save the images.
    save_path = os.path.join(deviation_map_save_dir, "avg_deviations_in_suppine_postures.png").replace("\\", "/")
    plot_deviation_map(y_test=supine_postures_y_test, y_pred=supine_postures_y_pred, 
                       title="Avg Deviations in Suppine postures",
                       cmap_label="kPa", 
                       block_size=2,
                       save_path=save_path)

    save_path = os.path.join(deviation_map_save_dir, "avg_deviations_in_leftside_postures.png").replace("\\", "/")
    plot_deviation_map(y_test=left_side_postures_y_test, y_pred=left_side_postures_y_pred, 
                       title="Avg Deviations in leftside postures",
                       cmap_label="kPa", 
                       block_size=2,
                       save_path=save_path)

    save_path = os.path.join(deviation_map_save_dir, "avg_deviations_in_rightside_postures.png").replace("\\", "/")
    plot_deviation_map(y_test=right_side_postures_y_test, y_pred=right_side_postures_y_pred, 
                       title="Avg Deviations in rightside postures",
                       cmap_label="kPa", 
                       block_size=2,
                       save_path=save_path)

# Execute the hydra application.
my_hydra_app()
