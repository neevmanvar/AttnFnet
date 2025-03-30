from torch import nn
import torch
from losses.SSIML2Loss import SSIML2Loss

class GANSSIML2Loss(nn.Module):
    """
    A custom loss module that combines adversarial loss from a discriminator with a composite image quality loss,
    specifically the SSIML2Loss. This loss is designed for GAN-based image generation tasks, where both the adversarial
    loss and the perceptual similarity (measured via SSIM and L2 distance) are important.

    Parameters:
        LAMDA (int): Scaling factor for the SSIML2 loss component. Defaults to 100.
        weight_ssim (float): Weight factor for the SSIM component within the SSIML2 loss. Defaults to 100.0.
        weight_l2 (float): Weight factor for the L2 component within the SSIML2 loss. Defaults to 1.0.
        *args, **kwargs: Additional arguments to be passed to the underlying loss functions.
    """
    def __init__(self, LAMDA: int = 100, weight_ssim: float = 100.0, weight_l2: float = 1.0, *args, **kwargs):
        super(GANSSIML2Loss, self).__init__(*args, **kwargs)
        # Binary Cross Entropy with Logits loss for the adversarial component.
        self.gan = nn.BCEWithLogitsLoss(**kwargs)
        # SSIML2Loss combines structural similarity (SSIM) and L2 loss for image quality.
        self.ssiml2 = SSIML2Loss(weight_ssim=weight_ssim, weight_l2=weight_l2, **kwargs)
        self.LAMDA = LAMDA
    
    def forward(self, y_preds, y_target, y_discriminator):
        """
        Forward pass to compute the combined GAN and SSIML2 loss.

        This method computes:
            1. The adversarial loss by comparing the discriminator's output on generated images (y_discriminator)
               against a tensor of ones.
            2. The SSIML2 loss by comparing the generated images (y_preds) with the ground truth (y_target).
        The final loss is a sum of the GAN loss and a scaled SSIML2 loss.

        Parameters:
            y_preds (torch.Tensor): Generated images (predictions) from the generator.
            y_target (torch.Tensor): Ground truth images.
            y_discriminator (torch.Tensor): Discriminator output for the generated images.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The combined loss.
                - gan_loss (torch.Tensor): The adversarial (GAN) loss.
                - ssiml2_loss (torch.Tensor): The SSIML2 loss.
        """
        # Create a tensor of ones with the same shape as the discriminator output to serve as the target labels.
        labels = torch.ones_like(y_discriminator)
        
        # Compute the GAN loss (adversarial loss).
        gan_loss = self.gan(y_discriminator, labels)
        
        # Compute the SSIML2 loss comparing the generated images to the target images.
        ssiml2_loss = self.ssiml2(y_preds, y_target)
        
        # Combine the losses: adversarial loss + scaled SSIML2 loss.
        loss = gan_loss + self.LAMDA * ssiml2_loss

        return loss, gan_loss, ssiml2_loss
