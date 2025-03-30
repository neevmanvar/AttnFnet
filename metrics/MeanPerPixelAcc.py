import torch
import torchmetrics

class MeanPerPixelAcc(torchmetrics.Metric):
    """
    Computes the mean per-pixel accuracy between predictions and targets.
    This metric scales inputs by 255, rounds the values, and then calculates the ratio
    of correctly predicted pixels to the total number of pixels.
    
    Attributes:
        total_correct (torch.Tensor): Cumulative count of correct predictions.
        total_pixels (torch.Tensor): Cumulative count of total pixels.
    """
    def __init__(self, name='mean_per_pixel_accuracy', **kwargs):
        super(MeanPerPixelAcc, self).__init__(**kwargs)
        self.add_state("total_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the metric state using the given predictions and targets.
        
        Parameters:
            preds (torch.Tensor): Predictions tensor, expected in range [0, 1].
            target (torch.Tensor): Target tensor, expected in range [0, 1].
        
        Raises:
            ValueError: If the shapes of preds and target do not match.
        """
        # Scale the tensors to [0, 255] and round to integer values.
        preds, target = preds * 255.0, target * 255.0
        
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        # Compute the number of correct pixel predictions.
        correct_predictions = torch.sum(torch.round(target) == torch.round(preds))
        # Total number of pixels.
        num_pixels = torch.numel(target)

        self.total_correct += correct_predictions
        self.total_pixels += num_pixels

    def compute(self) -> torch.Tensor:
        """
        Computes the mean per-pixel accuracy.
        
        Returns:
            torch.Tensor: The ratio of correctly predicted pixels to total pixels.
        """
        return self.total_correct.float() / self.total_pixels


def test():
    # Create random predictions and targets for testing
    x = torch.rand((4, 3, 300, 300))
    y = torch.rand((4, 3, 300, 300))
    
    # Initialize the MeanPerPixelAcc metric
    ppa = MeanPerPixelAcc()
    
    # Update the metric with identical predictions and targets to simulate perfect accuracy
    ppa.update(x, x)
    score = ppa.compute()
    
    # Print the computed accuracy score
    print(score)
    
    # Reset the metric state
    ppa.reset()


if __name__ == "__main__":
    test()
