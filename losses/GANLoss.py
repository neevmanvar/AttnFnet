from torch import nn
import torch

class GANLoss(nn.Module):
    """
    A custom loss module for Generative Adversarial Networks (GANs) that computes the combined loss for the
    discriminator's output on both real and generated (fake) data. It applies label smoothing to the targets
    and calculates the Binary Cross Entropy with Logits loss for both parts.

    Parameters:
        disc_weight (float): Weight factor for the discriminator loss. The computed loss for real and fake samples
                             is multiplied by this factor.
        label_smoothing_factors (list): A list containing two values for label smoothing.
                                        The first value is used as the target for real samples,
                                        and the last value is used as the target for fake samples.
        *args, **kwargs: Additional arguments passed to the superclass or the loss function.
    """
    def __init__(self, disc_weight: float = 0.5, label_smoothing_factors: list = [1, 0], *args, **kwargs):
        super(GANLoss, self).__init__(*args, **kwargs)
        self.disc_weight = disc_weight
        self.labels_true = label_smoothing_factors[0]
        self.labels_fake = label_smoothing_factors[-1]
        # The BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        self.loss_object = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, y_generated_disc, y_true_disc):
        """
        Forward pass to compute the GAN loss.

        This method creates label tensors for real and fake data using label smoothing,
        computes the BCEWithLogitsLoss for both the true and generated discriminator outputs,
        and returns a weighted sum of these losses along with the individual losses.

        Parameters:
            y_generated_disc (torch.Tensor): The discriminator output for generated (fake) data.
            y_true_disc (torch.Tensor): The discriminator output for real data.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The combined discriminator loss, weighted by disc_weight.
                - real_loss (torch.Tensor): The loss computed for the real data.
                - fake_loss (torch.Tensor): The loss computed for the fake (generated) data.
        """
        # Create label tensors matching the shapes of the input tensors
        labels_true = torch.full_like(y_true_disc, fill_value=self.labels_true, device=y_generated_disc.device)
        labels_fake = torch.full_like(y_generated_disc, fill_value=self.labels_fake, device=y_generated_disc.device)

        # Calculate the loss for real and fake data using the BCEWithLogitsLoss
        real_loss = self.loss_object(y_true_disc, labels_true)
        fake_loss = self.loss_object(y_generated_disc, labels_fake)

        # Compute the combined loss weighted by the discriminator weight factor
        loss = self.disc_weight * (real_loss + fake_loss)

        return loss, real_loss, fake_loss
