import os

class dirs:
    ASSET_DIR = "assets"
    TRAINING_PREDICTION_DIR = os.path.join(ASSET_DIR, "training_predictions").replace("\\","/")
    TRAINING_HISTORY_DIR = os.path.join(ASSET_DIR, "training_history").replace("\\","/")
    MODEL_SUMMARY_DIR = os.path.join(ASSET_DIR, "model_summary").replace("\\","/")
    MODEL_PLOTS_DIR = os.path.join(ASSET_DIR, "model_plots").replace("\\","/")
    TEST_PREDICTIONS_DIR = os.path.join(ASSET_DIR, "test_predictions").replace("\\","/")
    TEST_EVALUATION_DIR = os.path.join(ASSET_DIR, "test_evaluation").replace("\\","/")
    TEST_RESULTS_DIR = os.path.join(ASSET_DIR, "test_results").replace("\\","/")
    TENSORBOARD_RUNS_DIR = "runs"

    LOSSES_DIR = "losses"
    METRIC_DIR = "metrics"

    MODEL_CHECKPOINTS_DIR = "model_checkpoints"
    MODEL_SIGNATURES_DIR = "model_signatures"

    DATASET_DIR = "datasets"
    TTV_DATASET_DIR = os.path.join(DATASET_DIR, "ttv").replace("\\","/")
    NPZ_DATASET_DIR = os.path.join(DATASET_DIR, "npz").replace("\\","/")

    MODELS_DIR = "models"

    PRERAINED_CHECKPOINTS_DIR = "pretrained_checkpoints"
    SCRIPTS_DIR = "scrips"
    TRAINING_SCRIPTS_DIR = "training"
    UTIL_DIR = "util"
    
