from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class SSIML2Loss(nn.Module):
    """
    A loss module that combines a structural similarity (SSIM) based loss and an L2 (MSE) loss.
    The SSIM component encourages perceptual similarity between images, while the L2 component
    enforces pixel-wise accuracy.
    
    Parameters:
        weight_ssim (float): Weight for the SSIM loss component.
        weight_l2 (float): Weight for the L2 loss component.
        datarange (float): The value range of the input images (default is 1.0).
        *args, **kwargs: Additional arguments for underlying loss functions.
    """
    def __init__(self, weight_ssim: float = 100.0, weight_l2: float = 1.0, datarange: float = 1.0, *args, **kwargs):
        super(SSIML2Loss, self).__init__(*args, **kwargs)
        self.alpha = weight_ssim  # Weight for the SSIM loss
        self.beta = weight_l2     # Weight for the L2 loss
        self.l2 = nn.MSELoss(**kwargs)
        self.ssim = SSIM(data_range=datarange, **kwargs)
    
    def forward(self, y_pred, y_target):
        """
        Computes the combined loss between the predicted and target images.
        
        The loss is computed as a weighted sum of:
            - SSIM loss: 1 - SSIM value between the images.
            - L2 loss: Mean Squared Error between the images.
        
        Parameters:
            y_pred (torch.Tensor): The predicted/generated images.
            y_target (torch.Tensor): The target/ground truth images.
        
        Returns:
            torch.Tensor: The combined SSIM and L2 loss.
        """
        # Ensure the SSIM module is on the same device as the predictions.
        self.ssim.to(y_pred.device)
        
        # Calculate SSIM loss as (1 - SSIM value).
        ssim_loss = (1 - self.ssim(y_pred, y_target))
        # Calculate L2 (MSE) loss.
        l2_loss = self.l2(y_pred, y_target)
        
        # Combine both losses using the respective weights.
        ssiml2 = self.alpha * ssim_loss + self.beta * l2_loss
        
        return ssiml2
