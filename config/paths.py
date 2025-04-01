import os

class dirs:
    ASSET_DIR = "assets"
    TRAINING_PREDICTION_DIR = os.path.join(ASSET_DIR, "training_predictions").replace("\\","/")
    TEST_PREDICTIONS_DIR = os.path.join(ASSET_DIR, "test_predictions").replace("\\","/")
    TEST_RESULTS_DIR = os.path.join(ASSET_DIR, "test_results").replace("\\","/")
    TENSORBOARD_RUNS_DIR = "runs"

    LOSSES_DIR = "losses"
    METRIC_DIR = "metrics"

    DATASET_DIR = "datasets"
    TTV_DATASET_DIR = os.path.join(DATASET_DIR, "ttv").replace("\\","/")
    NPZ_DATASET_DIR = os.path.join(DATASET_DIR, "npz").replace("\\","/")

    MODELS_DIR = "models"

    PRERAINED_CHECKPOINTS_DIR = "pretrained_checkpoints"
    SCRIPTS_DIR = "scrips"
    TRAINING_SCRIPTS_DIR = "training"
    UTIL_DIR = "util"
    
