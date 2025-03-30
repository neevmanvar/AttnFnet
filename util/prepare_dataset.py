import torch
import numpy as np
from torch.utils.data import Dataset
import time
import glob
import os
from typing import Tuple, Type
import pandas as pd

class SLPDataset(Dataset):
    """
    A PyTorch Dataset class for loading SLP (presumably "Sensor/Lab Pressure" or similar) data.

    The dataset is expected to be stored in a compressed (.npz) format containing arrays for inputs ("x")
    and targets ("y") for different partitions (e.g., "train", "test", "val"). This class loads the data,
    transposes the axes to have channels first, and provides additional helper methods for pressure calibration
    and weight measurements.

    Attributes:
        dataset_path (str): Path to the dataset directory.
        x (np.ndarray): Input data loaded from the .npz files.
        y (np.ndarray): Target data loaded from the .npz files.
    """
    def __init__(self, dataset_path: str, partition: str = "train"):
        """
        Initialize the SLPDataset.

        Parameters:
            dataset_path (str): Directory where the dataset .npz files are stored.
            partition (str): Partition to load ("train", "test", or "val"). Raises a ValueError if an invalid partition is provided.
        
        Side Effects:
            Loads data from .npz files and prints the time taken for loading.
        """
        super(SLPDataset, self).__init__()
        self.dataset_path = dataset_path
        if partition not in ["train", "test", "val"]:
            raise ValueError("no existing partition name %s exist in key with name ['train', 'test', 'val']" % (partition))
        start_time = time.time()
        for file in glob.glob(os.path.join(dataset_path, "*.npz").replace("\\", "/")):
            npz_arr = np.load(file)
            for k, v in npz_arr.items():
                if partition in k and "x" in k:
                    self.x = v
                elif partition in k and "y" in k:
                    self.y = v
        elapsed_time = time.time()
        print("time taken to load train and validation compressed data: %.3f sec" % (elapsed_time - start_time))
        # Transpose to have channels first: (N, C, H, W)
        self.x = np.transpose(self.x, (0, 3, 1, 2))
        self.y = np.transpose(self.y, (0, 3, 1, 2)).astype(np.float32)
        # Uncomment to use a subset of data
        # self.x = self.x[:5]
        # self.y = self.y[:5]

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.x)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the sample at the given index.

        Parameters:
            index (int or Tensor): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input (x) and target (y) tensors.
        """
        if torch.is_tensor(index):
            index = index.tolist()
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])

    def _get_pressure_calibration(self, partition: str = "test") -> np.ndarray:
        """
        Load and return the pressure calibration scale for a given partition.

        Parameters:
            partition (str): The partition name (default "test").

        Returns:
            np.ndarray: The pressure calibration scale array.
        """
        test_pressure_calibration = np.load(self.dataset_path + "/" + partition + "_press_calib_scale.npy")
        return test_pressure_calibration

    def _get_weights_frame(self) -> pd.DataFrame:
        """
        Load and return a DataFrame containing weight measurements.

        Returns:
            pd.DataFrame: DataFrame read from the 'weight_measurements.csv' file.
        """
        df = pd.read_csv(self.dataset_path + "/weight_measurements.csv")
        return df

    def _get_arrays(self):
        """
        Get the input and target arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the input array (x) and target array (y).
        """
        return self.x, self.y

def load_model_predictions(prediction_path: str):
    """
    Load model predictions from a compressed .npz file.

    Parameters:
        prediction_path (str): Path to the .npz file containing predictions.

    Returns:
        np.ndarray: The predictions array stored under the key "y_pred".
    """
    npz_arr = np.load(prediction_path)
    y_pred = npz_arr["y_pred"]
    return y_pred

def test():
    """
    Test function for the SLPDataset class.

    Instantiates an SLPDataset, prints the type and shape of a sample target tensor,
    and displays the pressure calibration scale and the first few rows of the weight measurements.
    """
    slpset = SLPDataset("datasets/ttv/depth2bp_cleaned_no_KPa")
    print(type(slpset.__getitem__(5)[1]))
    print(slpset.__getitem__(5)[1].shape)
    print(slpset._get_pressure_calibration().shape)
    print(slpset._get_weights_frame().head())

if __name__ == "__main__":
    test()
