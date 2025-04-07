from torch import nn
import torch.optim as optim
import hydra
from config.attnfnet_config import Config
from models.Unet.unet import Unet
from models.discriminator.patchgan import PatchGAN
from util.prepare_dataset import SLPDataset
from util.prepare_dataloader import prepare_dataloader

from util.model_trainer import Trainer
from losses.GANSSIML2Loss import GANSSIML2Loss
from losses.GANLoss import GANLoss
import torch


def load_train_objs(cfg: Config, device='cuda'):
    train_ds = SLPDataset(dataset_path=cfg.data.path, partition="train")
    val_ds = SLPDataset(dataset_path=cfg.data.path, partition="val")

    gen_model = Unet(in_shape=cfg.gen_model.in_shape, 
                     out_shape=cfg.gen_model.out_shape,
                     kernel=cfg.gen_model.kernel,
                     features=cfg.gen_model.features,
                     act=cfg.gen_model.act)


    disc_model = PatchGAN(gen_in_shape=cfg.disc_model.gen_in_shape,
                          gen_out_shape=cfg.disc_model.gen_out_shape,
                          patch_in_size=cfg.disc_model.patch_in_size,
                          kernel=cfg.disc_model.kernel,
                          features=cfg.disc_model.features)


    gen_optimizer = optim.Adam(params=gen_model.parameters(), lr=cfg.optimizer.learning_rate, betas=(cfg.optimizer.beta_1, 0.999))
    disc_optimizer = optim.Adam(params=disc_model.parameters(), lr=cfg.optimizer.learning_rate, betas=(cfg.optimizer.beta_1, 0.999))

    # gen_model = torch.compile(gen_model)
    # disc_model = torch.compile(disc_model)

    return train_ds, val_ds, gen_model, disc_model, gen_optimizer, disc_optimizer

@hydra.main(version_base=None, config_name="config")
def my_hydra_app(cfg: Config)->None:
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("Number of GPUs available:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("GPU", i, ":", torch.cuda.get_device_name(i))
    else:
        print("No GPUs available, CPU will be used.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")
    train_ds, val_ds, gen_model, disc_model, gen_optimizer, disc_optimizer = load_train_objs(cfg, device = device)
    train_data = prepare_dataloader(dataset=train_ds, is_distributed=False, batch_size=cfg.trainer.batchsize)
    val_data = prepare_dataloader(dataset=val_ds, is_distributed=False, batch_size=1)
    trainer = Trainer(gen_model, disc_model, gen_optimizer, disc_optimizer, GANSSIML2Loss(), GANLoss(label_smoothing_factors=[0.9, 0.1]), train_data, val_data, device, cfg)
    trainer.train(cfg.trainer.max_epochs)

my_hydra_app()