import os
from config.paths import dirs
import warnings
import glob
import re
from config.attnfnet_config import Config
import shutil

class HandleTrainingDir():
    """
    A class to manage training directories for datasets, model checkpoints, predictions, and validation runs.
    
    This class sets up the required directories for training:
      - The dataset directory.
      - The model checkpoint directory.
      - Directories for saving on-epoch predictions, on-batch predictions, and intermediate token predictions.
      - The TensorBoard validation runs directory.
    
    If the clear_dirs flag is set to True, existing files in these directories are removed.
    
    Args:
        cfg (Config): Configuration object containing dataset and model parameters.
        clear_dirs (bool): If True, clear (delete) existing files in the directories.
        
    Raises:
        ValueError: If the dataset directory does not exist.
    """
    def __init__(self, cfg: Config, clear_dirs: bool = False) -> None:
        # Set the dataset directory based on configuration.
        self.DATASET_DIR = os.path.join(dirs.TTV_DATASET_DIR, cfg.data.data_name).replace("\\", "/")
        if not os.path.exists(self.DATASET_DIR):
            raise ValueError("dataset directory %s does not exist" % (self.DATASET_DIR))

        # Use dataset name from config.
        use_dataset_name = cfg.data.data_name

        # Set up the checkpoint directory.
        self.CHECKPOINT_DIR = os.path.join(dirs.MODEL_CHECKPOINTS_DIR, cfg.data.model_name, use_dataset_name).replace("\\", "/")
        if not os.path.exists(self.CHECKPOINT_DIR):
            warnings.warn("model checkpoint directory does not exist, making new directory to save checkpoints")
            os.makedirs(self.CHECKPOINT_DIR)

        # Clear the checkpoint directory if requested.
        if clear_dirs == True:
            warnings.warn("clear is True removing all model checkpoints..")
            [os.remove(f) for f in glob.glob(os.path.join(self.CHECKPOINT_DIR, "*").replace("\\", "/"))]

        # Set up directories for training predictions.
        TRAINING_PREDICTION_DIR = os.path.join(dirs.TRAINING_PREDICTION_DIR, cfg.data.model_name, use_dataset_name).replace("\\", "/")
        self.MODEL_ON_EPOCH_PREDICTION_DIR = os.path.join(TRAINING_PREDICTION_DIR, "on_epoch_predictions").replace("\\", "/")
        self.MODEL_ON_BATCH_PREDICTION_DIR = os.path.join(TRAINING_PREDICTION_DIR, "on_batch_predictions").replace("\\", "/")
        self.MODEL_INTERMEDIATE_TOKEN_PREDICION_DIR = os.path.join(TRAINING_PREDICTION_DIR, "intermediate_token_predictions").replace("\\", "/")

        # Create prediction directories if they don't exist.
        if not os.path.exists(self.MODEL_ON_EPOCH_PREDICTION_DIR):
            warnings.warn("model epoch predictions directory is not exist, making new directory to save training samples")
            os.makedirs(self.MODEL_ON_EPOCH_PREDICTION_DIR)

        if not os.path.exists(self.MODEL_ON_BATCH_PREDICTION_DIR):
            warnings.warn("model batch predictions directory is not exist, making new directory to save training samples")
            os.makedirs(self.MODEL_ON_BATCH_PREDICTION_DIR)
        
        if not os.path.exists(self.MODEL_INTERMEDIATE_TOKEN_PREDICION_DIR):
            warnings.warn("model intermediate token predictions directory is not exist, making new directory to save training tokens")
            os.makedirs(self.MODEL_INTERMEDIATE_TOKEN_PREDICION_DIR)

        # Clear prediction directories if requested.
        if clear_dirs == True:
            warnings.warn("clear is True removing all model batch and epoch predictions...")
            [os.remove(f) for f in glob.glob(os.path.join(self.MODEL_ON_BATCH_PREDICTION_DIR, "*.png").replace("\\", "/"))]
            [os.remove(f) for f in glob.glob(os.path.join(self.MODEL_ON_EPOCH_PREDICTION_DIR, "*.png").replace("\\", "/"))]
            [os.remove(f) for f in glob.glob(os.path.join(self.MODEL_INTERMEDIATE_TOKEN_PREDICION_DIR, "*.png").replace("\\", "/"))]

        # Set up the TensorBoard validation runs directory.
        self.VALIDATION_RUNS_DIR = os.path.join(dirs.TENSORBOARD_RUNS_DIR, cfg.data.model_name, use_dataset_name).replace("\\", "/")
 
        if clear_dirs == True:
            shutil.rmtree(self.VALIDATION_RUNS_DIR + "/")
        
        if not os.path.exists(self.VALIDATION_RUNS_DIR):
            os.makedirs(self.VALIDATION_RUNS_DIR)
        
    def get_dataset_dir(self):
        """Return the dataset directory."""
        return self.DATASET_DIR
    
    def get_checkpoint_dir(self):
        """Return the model checkpoint directory."""
        return self.CHECKPOINT_DIR
    
    def get_epoch_prediction_dir(self):
        """Return the directory for on-epoch predictions."""
        return self.MODEL_ON_EPOCH_PREDICTION_DIR
    
    def get_batch_prediction_dir(self):
        """Return the directory for on-batch predictions."""
        return self.MODEL_ON_BATCH_PREDICTION_DIR

    def get_validation_runs_dir(self):
        """Return the TensorBoard validation runs directory."""
        return self.VALIDATION_RUNS_DIR

class HandleTestDir():
    """
    A class to manage directories for testing and evaluation results.
    
    This class sets up directories for:
      - The dataset directory.
      - Model checkpoints.
      - Test predictions (both as images and numpy arrays).
      - Test results (including best/worst metric score images, random predictions, and training history results).
      - Comparison results.
      - Intermediate token predictions.
    
    It creates the directories if they do not exist, and if clear_dirs is True, it clears contents of certain directories.
    
    Args:
        cfg (Config): Configuration object containing dataset and model parameters.
        clear_dirs (bool): If True, clear the contents of specific directories.
    """
    def __init__(self, cfg: Config, clear_dirs: bool = False) -> None:
        self.DATASET_DIR = os.path.join(dirs.TTV_DATASET_DIR, cfg.data.data_name).replace("\\", "/")
        use_dataset_name = cfg.data.data_name
        
        self.CHECKPOINT_DIR = os.path.join(dirs.MODEL_CHECKPOINTS_DIR, cfg.data.model_name, use_dataset_name).replace("\\", "/")

        self.MODEL_PREDICTIONS_DIR = os.path.join(dirs.TEST_PREDICTIONS_DIR, cfg.data.model_name, use_dataset_name).replace("\\", "/")
        self.MODEL_PREDICTIONS_ARRAY_PATH = os.path.join(self.MODEL_PREDICTIONS_DIR, "y_pred.npz").replace("\\", "/")

        self.MODEL_RESULTS_DIR = os.path.join(dirs.TEST_RESULTS_DIR, cfg.data.model_name, use_dataset_name).replace("\\", "/")
        self.BEST_WORST_METRIC_SCORE_IMAGES_DIR = os.path.join(self.MODEL_RESULTS_DIR, "best_worst_metric_score_images").replace("\\", "/")
        self.RANDOM_PEDICTIONS_DIR = os.path.join(self.MODEL_RESULTS_DIR, "random_predictions").replace("\\", "/")
        training_history_results_dir = os.path.join(self.MODEL_RESULTS_DIR, "training_history_results").replace("\\", "/")
        self.TRAINING_HISTORY_RESULTS_DIR = training_history_results_dir

        self.COMRISIONSDIR = os.path.join(dirs.TEST_RESULTS_DIR, "comparisions", use_dataset_name).replace("\\", "/")

        self.INTERMEDIATE_TOKEN_DIR = os.path.join(self.MODEL_RESULTS_DIR, "intermediate_token_predictions")

        # Create necessary directories if they don't exist.
        if not os.path.exists(self.INTERMEDIATE_TOKEN_DIR):
            warnings.warn("model prediction dir doesn't exist creating new dir")
            os.makedirs(self.INTERMEDIATE_TOKEN_DIR)
        if not os.path.exists(self.MODEL_PREDICTIONS_DIR):
            warnings.warn("model prediction dir doesn't exist creating new dir")
            os.makedirs(self.MODEL_PREDICTIONS_DIR)              
        if not os.path.exists(self.MODEL_RESULTS_DIR):
            warnings.warn("model results dir doesn't exist creating new dir")
            os.makedirs(self.MODEL_RESULTS_DIR)
        if not os.path.exists(self.BEST_WORST_METRIC_SCORE_IMAGES_DIR):
            warnings.warn("model %s dir doesn't exist creating new dir" % (self.BEST_WORST_METRIC_SCORE_IMAGES_DIR))
            os.makedirs(self.BEST_WORST_METRIC_SCORE_IMAGES_DIR)
        if not os.path.exists(self.RANDOM_PEDICTIONS_DIR):
            warnings.warn("model %s dir doesn't exist creating new dir" % (self.RANDOM_PEDICTIONS_DIR))
            os.makedirs(self.RANDOM_PEDICTIONS_DIR)        
        if not os.path.exists(training_history_results_dir):
            os.makedirs(training_history_results_dir)
        if not os.path.exists(self.COMRISIONSDIR):
            warnings.warn("model %s dir doesn't exist creating new dir" % (self.COMRISIONSDIR))
            os.makedirs(self.COMRISIONSDIR)

        # Clear directories if requested.
        if clear_dirs == True:
            warnings.warn("clear is True removing all image contents from %s" % (self.BEST_WORST_METRIC_SCORE_IMAGES_DIR))
            [os.remove(f) for f in glob.glob(os.path.join(self.BEST_WORST_METRIC_SCORE_IMAGES_DIR, "*.png").replace("\\", "/"))]
        if clear_dirs == True:
            warnings.warn("clear is True removing all image contents from %s" % (self.RANDOM_PEDICTIONS_DIR))
            [os.remove(f) for f in glob.glob(os.path.join(self.RANDOM_PEDICTIONS_DIR, "*.png").replace("\\", "/"))]
        if clear_dirs == True:
            warnings.warn("clear is True removing all image contents from %s" % (self.TRAINING_HISTORY_RESULTS_DIR))
            [os.remove(f) for f in glob.glob(os.path.join(self.TRAINING_HISTORY_RESULTS_DIR, "*.png").replace("\\", "/"))]

    def get_dataset_dir(self):
        """Return the dataset directory."""
        return self.DATASET_DIR
    
    def get_checkpoint_dir(self):
        """Return the model checkpoint directory."""
        return self.CHECKPOINT_DIR
    
    def get_minloss_model_checkpoint(self, ckpt_name=""):
        """
        Get the checkpoint file with the minimum loss.
        
        If ckpt_name is not provided, the method searches the checkpoint directory for files 
        (excluding those with 'disc_model' in the name) and extracts the loss value from the filename.
        It then returns the checkpoint with the lowest loss.
        
        Args:
            ckpt_name (str): Optional specific checkpoint filename.
        
        Returns:
            str: Full path to the selected checkpoint file.
        """
        if ckpt_name == "":
            loss_list = []
            filename = os.listdir(self.CHECKPOINT_DIR)[0]
            file_extension = os.path.splitext(filename)[1]
            for ckpt in glob.glob(os.path.join(self.CHECKPOINT_DIR, "*" + file_extension)):
                if "disc_model" not in ckpt:
                    loss = ckpt.split("loss")[-1].split(file_extension)[0]
                    if not loss.isnumeric():
                        loss = re.findall(r'[\d]*[.][\d]+', loss)[0]
                    loss_list.append(float(loss))
            ckpt_list = os.listdir(self.CHECKPOINT_DIR)
            min_loss = min(loss_list)
            print("min loss: ", min_loss)
            ckpt_name = [i for i in ckpt_list if str(min_loss) in i][0]
        self.MODEL_CHECKPOINT = os.path.join(self.CHECKPOINT_DIR, ckpt_name)
        return self.MODEL_CHECKPOINT

    def get_pred_dir(self):
        """Return the directory for model predictions."""
        return self.MODEL_PREDICTIONS_DIR    

    def get_model_predictions_path(self):
        """Return the file path for saving model predictions as a numpy array."""
        return self.MODEL_PREDICTIONS_ARRAY_PATH
    
    def get_model_results_dir(self):
        """Return the directory for model results."""
        return self.MODEL_RESULTS_DIR
    
    def get_best_worst_metric_score_dir(self):
        """Return the directory for best/worst metric score images."""
        return self.BEST_WORST_METRIC_SCORE_IMAGES_DIR
    
    def get_random_pred_dir(self):
        """Return the directory for random predictions."""
        return self.RANDOM_PEDICTIONS_DIR
    
    def get_training_history_results_dir(self):
        """Return the directory for training history results."""
        return self.TRAINING_HISTORY_RESULTS_DIR
    
    def get_result_comparision_dir(self):
        """Return the directory for result comparisons."""
        return self.COMRISIONSDIR
