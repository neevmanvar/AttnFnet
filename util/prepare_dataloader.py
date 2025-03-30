from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

def prepare_dataloader(dataset: Dataset, batch_size: int = 1, is_distributed: bool = True, sampler: DistributedSampler = None):
    """
    Prepare a DataLoader for a given dataset, optionally for distributed training.

    Parameters:
        dataset (Dataset): The dataset to load.
        batch_size (int): The number of samples per batch.
        is_distributed (bool): If True, the DataLoader is set up for distributed training.
        sampler (DistributedSampler): A DistributedSampler instance; must not be None if is_distributed is True.

    Returns:
        DataLoader: A DataLoader configured with the specified batch size and sampler.
        
    Raises:
        ValueError: If is_distributed is True and sampler is None.
    """
    if is_distributed:
        if sampler is None:
            raise ValueError("sampler cannot be a None object when using DDP")
        return DataLoader(dataset, batch_size, shuffle=False, pin_memory=True, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size, shuffle=False, pin_memory=True)
