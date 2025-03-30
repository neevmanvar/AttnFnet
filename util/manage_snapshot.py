import torch

def load_snapshot(snapshot_path, gen_model, disc_model, gen_opt, disc_opt):
    """
    Load a training snapshot and restore the state dictionaries of the generator model, discriminator model,
    and their respective optimizers.

    Parameters:
        snapshot_path (str): Path to the snapshot file.
        gen_model (torch.nn.Module): Generator model instance.
        disc_model (torch.nn.Module): Discriminator model instance.
        gen_opt (torch.optim.Optimizer): Optimizer for the generator.
        disc_opt (torch.optim.Optimizer): Optimizer for the discriminator.

    Returns:
        tuple: A tuple containing:
            - epochs_run (int): The number of epochs already run.
            - gen_model (torch.nn.Module): Generator model with restored weights.
            - disc_model (torch.nn.Module): Discriminator model with restored weights.
            - gen_opt (torch.optim.Optimizer): Generator optimizer with restored state.
            - disc_opt (torch.optim.Optimizer): Discriminator optimizer with restored state.
    
    """
    snapshot = torch.load(snapshot_path, weights_only=True)
    gen_model.load_state_dict(snapshot["GEN_MODEL_STATE"])
    disc_model.load_state_dict(snapshot["DISC_MODEL_STATE"])
    gen_opt.load_state_dict(snapshot["GEN_OPT_STATE"])
    disc_opt.load_state_dict(snapshot["DISC_OPT_STATE"])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {epochs_run}")
    return epochs_run, gen_model, disc_model, gen_opt, disc_opt


def save_snapshot(snapshot_path, epoch, gen_model_state_dict, disc_model_state_dict, gen_opt_state_dict, disc_opt_state_dict):
    """
    Save a training snapshot that includes the state dictionaries of the generator model, discriminator model,
    and their corresponding optimizers, as well as the current epoch count.

    Parameters:
        snapshot_path (str): Path where the snapshot will be saved.
        epoch (int): The current epoch number (0-indexed). The saved epoch will be epoch + 1.
        gen_model_state_dict (dict): State dictionary of the generator model.
        disc_model_state_dict (dict): State dictionary of the discriminator model.
        gen_opt_state_dict (dict): State dictionary of the generator optimizer.
        disc_opt_state_dict (dict): State dictionary of the discriminator optimizer.

    Side Effects:
        Saves the snapshot to disk and prints a message indicating that the checkpoint has been saved.
    """
    snapshot = {}
    snapshot["GEN_MODEL_STATE"] = gen_model_state_dict  # Could use gen_model.module.state_dict() if needed.
    snapshot["DISC_MODEL_STATE"] = disc_model_state_dict  # Could use disc_model.module.state_dict() if needed.
    snapshot["GEN_OPT_STATE"] = gen_opt_state_dict
    snapshot["DISC_OPT_STATE"] = disc_opt_state_dict
    snapshot["EPOCHS_RUN"] = epoch + 1
    torch.save(snapshot, snapshot_path)
    print(f"Epoch {epoch} | Training checkpoint saved at {snapshot_path}")


def load_gen_model(snapshot_path, gen_model):
    """
    Load the generator model state from a snapshot.

    Parameters:
        snapshot_path (str): Path to the snapshot file.
        gen_model (torch.nn.Module): Generator model instance to be updated.

    Returns:
        tuple: A tuple containing:
            - epochs_run (int): The number of epochs already run.
            - gen_model (torch.nn.Module): Generator model with restored weights.

    """
    snapshot = torch.load(snapshot_path, weights_only=True)
    gen_model.load_state_dict(snapshot["GEN_MODEL_STATE"])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Loading snapshot from Epoch {epochs_run}")
    return epochs_run, gen_model
