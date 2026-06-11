import sys
sys.dont_write_bytecode = True
import pytorch_lightning as pl
import torch
import os
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

from .ema import ExponentialMovingAverage
from .utils import create_model, is_main_process
from .losses import get_loss
from .sde_lib import get_sde
from .optimizers import get_optimizer
from .lr_schedulers import get_lr_scheduler
#====================================================================
class ScoreModelLightningModule(pl.LightningModule):
    def __init__(self, config, train_loss_fn, val_loss_fn):
        super().__init__()
        self.config = config

        logger.info(" >> >> INSIDE lightningModule config.deterministic %s, sde: %s", config.deterministic, config.training.sde)

        base_model = create_model(config)
        self._use_channels_last = False
        if getattr(config.training, 'pytorch2_speedup', False):
            from .pytorch2_speedup_utils import detect_hardware, setup_model_speedups
            _caps = detect_hardware()
            self.model, self._use_channels_last = setup_model_speedups(base_model, _caps, config)
        else:
            self.model = base_model

        #self.model = create_model(config)
        self.sde   = get_sde(config)
        self.ema   = ExponentialMovingAverage(self.model.parameters(),
                                              decay=config.model.ema_rate)

        print(" >> >> INSIDE lightningModule", type(self.sde))

        self.train_loss_fn = train_loss_fn #get_loss(self.sde, True, config)
        self.val_loss_fn   = val_loss_fn #get_loss(self.sde, False, config)
        self.batch_size = config.training.batch_size
        self.val_losses = []
        self.train_losses = []
        self._batch_counter = 0

    def forward(self, x, cond, time_cond):
        # Check for NaNs. May get desynced if multi-GPU training. Comment out if this happens
        assert not torch.any(torch.isnan(x)).item(), ' < < < ERROR > > > Target data has NaNs'
        assert not torch.any(torch.isnan(cond)).item(), ' < < < ERROR > > > Input data has NaNs'
        assert not torch.any(torch.isnan(time_cond)).item(), ' < < < ERROR > > > time_cond has NaNs'

        return self.model.forward(x, cond, time_cond)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update(self.model.parameters())

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        cond, target, time = batch
        cond   = cond.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        #time   = time.to(device, non_blocking=True)
        if self._use_channels_last:
            try:
                if cond.ndim == 4:
                    cond   = cond.to(memory_format=torch.channels_last)
                if target.ndim == 4:
                    target = target.to(memory_format=torch.channels_last)
            except Exception:
                self._use_channels_last = False
        return cond, target, time

    def training_step(self, batch, batch_idx):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._batch_counter += 1
        #if (self._batch_counter % 50) == 0:
        #    logger.info(" >> INSIDE lightningModuleEMA training_step [rank %d] processed %d batches so far (batch_idx=%s)", rank, self._batch_counter, batch_idx)
        
        cond, target, time = batch

        train_loss = self.train_loss_fn(self.model, target, cond)
        self.train_losses.append(train_loss.detach())
        #self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        return train_loss

    def on_train_epoch_end(self):
        # Log average training loss
        if self.train_losses:
            avg_train_loss = torch.stack(self.train_losses).mean()
            if self.trainer.global_rank == 0:
                self.log("train_loss", avg_train_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False, rank_zero_only=True)
                print(f" >> >> Epoch {self.current_epoch} - train_loss: {avg_train_loss}")
            self.train_losses.clear()

        # Log the learning rate at the end of each epoch
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        lr = optimizer.param_groups[0]['lr']
        if self.trainer.global_rank == 0:
            self.log("lr", lr, prog_bar=True, logger=True, on_epoch=True, on_step=False, rank_zero_only=True)
            print(f" >> >> Epoch {self.current_epoch} - lr: {lr}")

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        logger.info(" >> INSIDE lightningModuleEMA on_train_epoch_end [rank %d] EPOCH END: processed %d batches this epoch", rank, getattr(self, "_batch_counter", -1))
        
    def validation_step(self, batch, batch_idx):
        #if is_main_process():
            #logger.info("))))))))))))))))))))))))) VALIDATION STEP ((((((((((((((((((((((((((((") 
            #logger.info("))))))))))))))))))))))))) VALIDATION STEP ((((((((((((((((((((((((((((") 
            #logger.info("))))))))))))))))))))))))) VALIDATION STEP ((((((((((((((((((((((((((((") 
            #logger.info("))))))))))))))))))))))))) VALIDATION STEP ((((((((((((((((((((((((((((") 
        
        cond, target, time = batch
        val_loss = self.val_loss_fn(self.model, target, cond)
        self.val_losses.append(val_loss.detach())
        return val_loss

    def on_validation_epoch_start(self):
        # Apply EMA weights before validation steps so val_loss reflects EMA model quality
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())

    def on_validation_epoch_end(self):
        # Compute and log average validation loss (collected under EMA weights)
        if self.val_losses:
            avg_val_loss = torch.stack(self.val_losses).mean()
            if self.trainer.global_rank == 0:
                self.log("val_loss", avg_val_loss, prog_bar=True, sync_dist=True, logger=True, on_epoch=True, on_step=False, rank_zero_only=True)

        # Restore original weights
        self.ema.restore(self.model.parameters())
        # Clear the buffer
        self.val_losses.clear()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
        checkpoint["step"] = self.global_step
        checkpoint["epoch"] = self.current_epoch
        #checkpoint["location_params"] = self.config.location_params.state_dict()
        # Strip _orig_mod prefix inserted by torch.compile so checkpoints are
        # portable between compiled and uncompiled runs.
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            if any("._orig_mod." in k for k in sd):
                checkpoint["state_dict"] = {
                    k.replace("._orig_mod", ""): v for k, v in sd.items()
                }

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"], device=self.device)
        if "location_params" in checkpoint:
            self.config.location_params.load_state_dict(checkpoint["location_params"])
        # on_save_checkpoint strips _orig_mod so checkpoints are portable.
        # Re-insert it here when loading into a compiled model.
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            model_sd = self.state_dict()
            if (any("._orig_mod." in k for k in model_sd)
                    and not any("._orig_mod." in k for k in sd)):
                checkpoint["state_dict"] = {
                    k.replace("model.", "model._orig_mod.", 1): v
                    for k, v in sd.items()
                }

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.model.parameters())
        scheduler = get_lr_scheduler(optimizer, self.config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_start(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._batch_counter = 0
        logger.info(" >> INSIDE lightningModuleEMA on_train_epoch_start [rank %d pid %d] on_train_epoch_start", rank, os.getpid())

    # def on_train_start(self):
    #     rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    #     logger.info(" >> INSIDE lightningModuleEMA on_train_start [rank %d pid %d] cuda_available=%s cuda_count=%d cuda_current=%s CUDA_VISIBLE_DEVICES=%s",
    #                 rank, os.getpid(), torch.cuda.is_available(), torch.cuda.device_count(),
    #                 torch.cuda.current_device() if torch.cuda.is_available() else None,
    #                 os.environ.get("CUDA_VISIBLE_DEVICES"))

    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    #     logger.info(" >> INSIDE lightningModuleEMA on_train_batch_start [rank %d pid %d] on_train_batch_start batch_idx=%s", rank, os.getpid(), batch_idx)

    # def on_before_zero_grad(self, optimizer):
    #     rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    #     logger.info(" >> INSIDE lightningModuleEMA on_before_zero_grad [rank %d pid %d] on_before_zero_grad", rank, os.getpid())

    # def on_after_backward(self):
    #     rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    #     # sync_cuda is optional but may help show memory usage
    #     if torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #     logger.info(" >> INSIDE lightningModuleEMA on_after_backward [rank %d pid %d] on_after_backward cuda_mem=%d", rank, os.getpid(), torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)