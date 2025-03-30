import torch
from torch import nn, optim
import torch.distributed
from torch.utils.data import DataLoader
from config.attnfnet_config import Config
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
from util.manage_snapshot import load_snapshot, save_snapshot
import sys
from util.model_callbacks import ModelSanityCheck, SaveModelHistory, ModelEncodingToken
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
from losses.SSIML2Loss import SSIML2Loss

class DDPTrainer:
    """
    Distributed Data Parallel (DDP) trainer for GAN models.

    This class encapsulates the training loop for a generator and a discriminator in a distributed
    training environment. It handles:
      - Loading from a snapshot if available.
      - Wrapping the models in DistributedDataParallel.
      - Running the training batches and epochs.
      - Logging metrics to TensorBoard.
      - Performing periodic sanity checks (saving sample predictions) and token visualizations.
      - Running validation on the validation DataLoader and aggregating metrics.
      - Saving training snapshots periodically.

    Attributes:
        gpu_id (int): ID of the current GPU (from the LOCAL_RANK environment variable).
        gen_model (nn.Module): Generator model placed on the assigned GPU.
        disc_model (nn.Module): Discriminator model placed on the assigned GPU.
        train_data (DataLoader): Training dataset loader.
        val_data (DataLoader): Validation dataset loader.
        gen_optimizer (optim.Optimizer): Optimizer for the generator.
        disc_optimizer (optim.Optimizer): Optimizer for the discriminator.
        save_every (int): Frequency (in epochs) at which snapshots are saved.
        epochs_run (int): Number of epochs already run (loaded from snapshot if available).
        gen_loss_fn (nn.Module): Loss function for the generator.
        disc_loss_fn (nn.Module): Loss function for the discriminator.
        snapshot_path (str): File path to the training snapshot.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        start (torch.cuda.Event): CUDA event marking the start of a timed operation.
        end (torch.cuda.Event): CUDA event marking the end of a timed operation.
        sanity (ModelSanityCheck): Callback for performing sanity checks (e.g., saving sample predictions).
        token_predictor (ModelEncodingToken or None): Optional callback for visualizing intermediate token outputs.
        ssiml2 (SSIML2Loss): Module for computing SSIML2 loss.
        ssim (StructuralSimilarityIndexMeasure): Metric for computing SSIM.
        mse (MeanSquaredError): Metric for computing Mean Squared Error.
    """
    def __init__(
            self,
            gen_model: nn.Module,
            disc_model: nn.Module,
            gen_optimizer: optim.Optimizer,
            disc_optimizer: optim.Optimizer,
            gen_loss: nn.Module,
            disc_loss: nn.Module,
            train_data: DataLoader,
            val_data: DataLoader = None,
            cfg: Config = None,
            token_predictor: ModelEncodingToken = None):
        
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.gen_model = gen_model.to(self.gpu_id)
        self.disc_model = disc_model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.save_every = cfg.trainer.save_every
        self.epochs_run = 0
        self.gen_loss_fn = gen_loss
        self.disc_loss_fn = disc_loss
        self.snapshot_path = cfg.trainer.snapshot_path

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.writer = SummaryWriter('runs/'+ cfg.data.model_name+'/{}/exp_{}'.format(cfg.data.data_name, timestamp))

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.sanity = ModelSanityCheck(val_data=val_data, 
                                       batch_index=cfg.data.train_pred_batch_idx, 
                                       save_epoch_dir=cfg.data.train_pred_epoch_dir, 
                                       save_batch_dir=cfg.data.train_pred_batch_dir, device=self.gpu_id)
        
        self.token_predictor = token_predictor

        self.ssiml2 = SSIML2Loss().to(self.gpu_id)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.gpu_id)
        self.mse = MeanSquaredError().to(self.gpu_id)

        if os.path.exists(self.snapshot_path):
            print("loading snapshot")
            self.epochs_run, self.gen_model, self.disc_model, self.gen_optimizer, self.disc_optimizer = load_snapshot(
                self.snapshot_path, 
                self.gen_model, 
                self.disc_model, 
                self.gen_optimizer, 
                self.disc_optimizer
            )

        self.gen_model = DDP(self.gen_model, device_ids=[self.gpu_id], broadcast_buffers=False)
        self.disc_model = DDP(self.disc_model, device_ids=[self.gpu_id], broadcast_buffers=False)

    def _run_batch(self, source, targets):
        # Generator forward pass and loss computation.
        self.gen_optimizer.zero_grad()
        gen_output = self.gen_model(source)
        disc_fake_output = self.disc_model(source, gen_output)
        gen_loss, gan_loss, ssiml2_loss = self.gen_loss_fn(gen_output, targets, disc_fake_output)
        gen_loss.backward()
        self.gen_optimizer.step()

        # Discriminator forward pass and loss computation.
        self.disc_optimizer.zero_grad()
        disc_real_output = self.disc_model(source, targets)
        disc_fake_output = self.disc_model(source, gen_output.detach())
        disc_loss, real_loss, fake_loss = self.disc_loss_fn(disc_fake_output, disc_real_output)
        disc_loss.backward()
        self.disc_optimizer.step()

        ssim_metric = self.ssim(targets, gen_output)

        self.running_gen_loss += gen_loss.detach().item()
        self.running_disc_loss += disc_loss.detach().item()
        self.running_ssiml2_loss += ssiml2_loss.detach().item()
        self.running_gen_gan_loss += gan_loss.detach().item()
        self.running_ssim += ssim_metric.detach().item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        step_count = 0
        self.running_gen_loss = 0
        self.running_disc_loss = 0
        self.running_ssiml2_loss = 0
        self.running_gen_gan_loss = 0
        self.running_ssim = 0
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
            if step_count % 200 == 0 and self.gpu_id == 0:
                self.sanity.forward(self.gen_model, step=step_count, epoch=epoch, save_batch_dir=True)

            last_gen_loss = self.running_gen_loss / b_sz  # Loss per batch.
            last_disc_loss = self.running_disc_loss / b_sz
            last_ssiml2_loss = self.running_ssiml2_loss / b_sz
            last_gen_gan_loss = self.running_gen_gan_loss / b_sz
            last_ssim = self.running_ssim / b_sz

            print('batch {} gen_loss: {}, disc_loss: {}, ssiml2_loss: {}, gen_gan_loss: {}'.format(
                step_count + 1, last_gen_loss, last_disc_loss, last_ssiml2_loss, last_gen_gan_loss))
            tb_x = epoch * len(self.train_data) + step_count + 1
            self.writer.add_scalar('Loss/train_gen', last_gen_loss, tb_x)
            self.writer.add_scalar('Loss/train_disc', last_disc_loss, tb_x)
            self.writer.add_scalar('Loss/train_ssiml2', last_ssiml2_loss, tb_x)
            self.writer.add_scalar('Loss/train_gen_gan', last_gen_gan_loss, tb_x)
            self.writer.add_scalar('Metric/train_ssim', last_ssim, tb_x)

            self.running_gen_loss = 0
            self.running_disc_loss = 0
            self.running_ssiml2_loss = 0
            self.running_gen_gan_loss = 0
            self.running_ssim = 0
            step_count += 1

    def _test(self, epoch: int):
        self.start.record()
        self.gen_model.eval()
        with torch.no_grad():
            vssiml2 = torch.tensor(0.0).to(self.gpu_id)
            vssim = torch.tensor(0.0).to(self.gpu_id)
            vmse = torch.tensor(0.0).to(self.gpu_id)
            total_samples = torch.tensor(0.0).to(self.gpu_id)

            for i, vdata in enumerate(self.val_data):
                st = time.time()
                x, y = vdata
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)
                y_pred = self.gen_model(x).detach()
                vssiml2 += self.ssiml2(y, y_pred)
                vssim += self.ssim(y, y_pred).to(self.gpu_id)
                vmse += self.mse(y, y_pred)
                total_samples += x.size(0)
                
            tb_x = epoch + 1
            print("test time: ", time.time() - st)
            torch.distributed.all_reduce(vssiml2, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(vssim, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(vmse, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)

            if self.gpu_id == 0:
                self.writer.add_scalar('Loss/val_ssiml2', vssiml2 / len(self.val_data), tb_x)
                self.writer.add_scalar('Loss/val_mse', vmse / len(self.val_data), tb_x)
                self.writer.add_scalar('Metric/val_ssim', vssim / len(self.val_data), tb_x)
            
        self.end.record()
        torch.cuda.synchronize()
        print("time taken to validate data: ", round((self.start.elapsed_time(self.end)) / 1000, 2))

    def train(self, max_epochs: int):
        """
        Train the GAN model using Distributed Data Parallel.
        
        This training loop iterates over epochs (starting from a loaded checkpoint if available) up to max_epochs.
        In each epoch, it runs the training batches, performs validation, executes sanity checks (saving sample
        predictions and token visualizations if available), saves snapshots at specified intervals, and logs metrics
        to TensorBoard.

        Parameters:
            max_epochs (int): Maximum number of epochs for training.
        """
        epoch_time = time.time()
        self.start.record()
        for epoch in range(self.epochs_run, max_epochs):
            self.gen_model.train(True)
            self.disc_model.train(True)
            self._run_epoch(epoch)
            self.end.record()
            torch.cuda.synchronize()
            print("time per epoch: ", round((self.start.elapsed_time(self.end)) / 1000, 2))
            
            torch.distributed.barrier()
            self._test(epoch)
            torch.distributed.barrier()

            if self.gpu_id == 0:
                self.sanity.forward(self.gen_model, step=0, epoch=epoch, save_batch_dir=False)
            
            if self.gpu_id == 0 and self.token_predictor is not None:
                self.token_predictor.forward(self.gen_model, epoch=epoch)

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                save_snapshot(self.snapshot_path, epoch, 
                              self.gen_model.module.state_dict(), 
                              self.disc_model.module.state_dict(),
                              self.gen_optimizer.state_dict(),
                              self.disc_optimizer.state_dict())
            self.writer.flush()

        print("Total training time: ", time.time() - epoch_time)


class Trainer:
    """
    Trainer for standard (non-distributed) training of GAN models.

    This class provides a complete training loop for a generator and discriminator model without using
    DistributedDataParallel. It performs forward passes on training batches, computes losses, updates models,
    logs metrics to TensorBoard, validates on a validation set, and saves snapshots periodically.

    Attributes:
        device (str): Device used for training ("cuda" or "cpu").
        gen_model (nn.Module): Generator model placed on the specified device.
        disc_model (nn.Module): Discriminator model placed on the specified device.
        train_data (DataLoader): DataLoader for training data.
        val_data (DataLoader): DataLoader for validation data.
        gen_optimizer (optim.Optimizer): Optimizer for the generator.
        disc_optimizer (optim.Optimizer): Optimizer for the discriminator.
        save_every (int): Frequency (in epochs) to save training snapshots.
        epochs_run (int): Number of epochs already completed (loaded from snapshot if available).
        gen_loss_fn (nn.Module): Loss function for the generator.
        disc_loss_fn (nn.Module): Loss function for the discriminator.
        snapshot_path (str): Path for saving/loading training snapshots.
        sanity (ModelSanityCheck): Callback for saving sample predictions as a sanity check.
        token_predictor (ModelEncodingToken or None): Optional callback for visualizing intermediate token outputs.
        ssiml2 (SSIML2Loss): Module for computing SSIML2 loss.
        ssim (StructuralSimilarityIndexMeasure): SSIM metric.
        mse (MeanSquaredError): MSE metric.
        writer (SummaryWriter): TensorBoard writer for logging training metrics.
        start (torch.cuda.Event): CUDA event marking the start of a timed operation.
        end (torch.cuda.Event): CUDA event marking the end of a timed operation.
    """
    def __init__(
            self,
            gen_model: nn.Module,
            disc_model: nn.Module,
            gen_optimizer: optim.Optimizer,
            disc_optimizer: optim.Optimizer,
            gen_loss: nn.Module,
            disc_loss: nn.Module,
            train_data: DataLoader,
            val_data: DataLoader = None,
            device: str = "cuda",
            cfg: Config = None,
            token_predictor: ModelEncodingToken = None):
        
        self.device = device
        self.gen_model = gen_model.to(device)
        self.disc_model = disc_model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.save_every = cfg.trainer.save_every
        self.epochs_run = 0
        self.gen_loss_fn = gen_loss
        self.disc_loss_fn = disc_loss
        self.snapshot_path = cfg.trainer.snapshot_path
        self.sanity = ModelSanityCheck(val_data=val_data,
                                       batch_index=cfg.data.train_pred_batch_idx, 
                                       save_epoch_dir=cfg.data.train_pred_epoch_dir, 
                                       save_batch_dir=cfg.data.train_pred_batch_dir, device=device)

        self.token_predictor = token_predictor

        self.ssiml2 = SSIML2Loss().to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.mse = MeanSquaredError().to(device)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.writer = SummaryWriter('runs/attnfnet/{}/exp_{}'.format(cfg.data.data_name, timestamp))
        
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        
        if os.path.exists(self.snapshot_path):
            print("loading snapshot")
            self.epochs_run, self.gen_model, self.disc_model, self.gen_optimizer, self.disc_optimizer = load_snapshot(
                self.snapshot_path, 
                self.gen_model, 
                self.disc_model, 
                self.gen_optimizer, 
                self.disc_optimizer
            )

    def _run_batch(self, source, targets):
        # Generator forward pass and loss computation.
        self.gen_optimizer.zero_grad()
        gen_output = self.gen_model(source)
        disc_fake_output = self.disc_model(source, gen_output)
        gen_loss, gan_loss, ssiml2_loss = self.gen_loss_fn(gen_output, targets, disc_fake_output)
        gen_loss.backward()
        self.gen_optimizer.step()

        # Discriminator forward pass and loss computation.
        self.disc_optimizer.zero_grad()
        disc_real_output = self.disc_model(source, targets)
        disc_fake_output = self.disc_model(source, gen_output.detach())  # Detach gen_output
        disc_loss, real_loss, fake_loss = self.disc_loss_fn(disc_fake_output, disc_real_output)
        disc_loss.backward()
        self.disc_optimizer.step()

        ssim_metric = self.ssim(targets, gen_output)

        self.running_gen_loss += gen_loss.item()
        self.running_disc_loss += disc_loss.item()
        self.running_ssiml2_loss += ssiml2_loss.item()
        self.running_gen_gan_loss += gan_loss.item()
        self.running_ssim += ssim_metric.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[Device{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        step_count = 0
        self.running_gen_loss = 0
        self.running_disc_loss = 0
        self.running_ssiml2_loss = 0
        self.running_gen_gan_loss = 0
        self.running_ssim = 0
        
        for source, targets in self.train_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            self._run_batch(source, targets)
            if step_count % 200 == 0:
                self.sanity.forward(self.gen_model, step=step_count, epoch=epoch, save_batch_dir=True)
            
            last_gen_loss = self.running_gen_loss / b_sz  # loss per batch
            last_disc_loss = self.running_disc_loss / b_sz
            last_ssiml2_loss = self.running_ssiml2_loss / b_sz
            last_gen_gan_loss = self.running_gen_gan_loss / b_sz
            last_ssim = self.running_ssim / b_sz

            print('batch {} gen_loss: {}, disc_loss: {}, ssiml2_loss: {}, gen_gan_loss: {}'.format(
                step_count + 1, last_gen_loss, last_disc_loss, last_ssiml2_loss, last_gen_gan_loss))
            tb_x = epoch * len(self.train_data) + step_count + 1
            self.writer.add_scalar('Loss/train_gen', last_gen_loss, tb_x)
            self.writer.add_scalar('Loss/train_disc', last_disc_loss, tb_x)
            self.writer.add_scalar('Loss/train_ssiml2', last_ssiml2_loss, tb_x)
            self.writer.add_scalar('Loss/train_gen_gan', last_gen_gan_loss, tb_x)
            self.writer.add_scalar('Metric/train_ssim', last_ssim, tb_x)

            self.running_gen_loss = 0
            self.running_disc_loss = 0
            self.running_ssiml2_loss = 0
            self.running_gen_gan_loss = 0
            self.running_ssim = 0
            step_count += 1

    def _test(self, epoch: int):
        self.start.record()
        self.gen_model.eval()
        with torch.no_grad():
            vssiml2 = torch.tensor(0.0).to(self.device)
            vssim = torch.tensor(0.0).to(self.device)
            vmse = torch.tensor(0.0).to(self.device)
            for i, vdata in enumerate(self.val_data):
                st = time.time()
                x, y = vdata
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.gen_model(x).detach()
                vssiml2 += self.ssiml2(y, y_pred)
                vssim += self.ssim(y, y_pred)
                vmse += self.mse(y, y_pred)
            tb_x = epoch + 1
            print("test time: ", time.time() - st)
            self.writer.add_scalar('Loss/val_ssiml2', vssiml2 / len(self.val_data), tb_x)
            self.writer.add_scalar('Loss/val_mse', vmse / len(self.val_data), tb_x)
            self.writer.add_scalar('Metric/val_ssim', vssim / len(self.val_data), tb_x)
            
        self.end.record()
        torch.cuda.synchronize()
        print("time taken to validate data: ", round((self.start.elapsed_time(self.end)) / 1000, 2))
        
    def train(self, max_epochs: int):
        """
        Train the GAN model using standard training (with Distributed Data Parallel wrapping).
        
        The training loop iterates from the current epoch (possibly loaded from a snapshot) to max_epochs.
        For each epoch, it:
          - Runs training batches and accumulates loss and metric values.
          - Performs periodic sanity checks by saving sample predictions.
          - Runs token visualization if a token predictor is provided.
          - Runs validation on the validation dataset.
          - Saves training snapshots periodically.
          - Logs all metrics to TensorBoard.
        
        Parameters:
            max_epochs (int): The maximum number of epochs to train.
        """
        epoch_time = time.time()
        self.start.record()
        for epoch in range(self.epochs_run, max_epochs):
            self.gen_model.train(True)
            self.disc_model.train(True)
            self._run_epoch(epoch)
            self.end.record()
            torch.cuda.synchronize()
            print("time per epoch: ", round((self.start.elapsed_time(self.end)) / 1000, 2))
            self.sanity.forward(self.gen_model, step=0, epoch=epoch, save_batch_dir=False)
            
            if self.token_predictor is not None:
                self.token_predictor.forward(self.gen_model, epoch=epoch)

            self._test(epoch)
            self.writer.flush()
            if epoch % self.save_every == 0:
                save_snapshot(self.snapshot_path, epoch, 
                              self.gen_model.state_dict(), 
                              self.disc_model.state_dict(),
                              self.gen_optimizer.state_dict(),
                              self.disc_optimizer.state_dict())
        print("Total training time: ", time.time() - epoch_time)
