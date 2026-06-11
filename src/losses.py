# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Modifications copyright 2024 Henry Addison
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications to the original work have been made by Henry Addison
# to allow for sampling

"""All functions related to loss computation and optimization.
"""

import os
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None
import absl

logger = logging.getLogger(__name__)

from .utils import get_score_fn, get_model_fn, is_main_process, get_BCE_loss_weight
from .data_scripts.data_utils import count_precip_above_threshold
#====================================================================
VESDE,VPSDE = None,None # dummy to keep code running won't be used
#====================================================================
def get_BCEWithLogitsLoss(train, criterion, threshold, reduce_mean=True):
    logger.info(" >> >> INSIDE get_BCEWithLogitsLoss")
    
    def loss_fn(model, batch, cond, generator=None):
        """Compute the loss function for a deterministic run.

        Args:
        model: A score model.
        batch: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.
        generator: An optional random number generator so can control the timesteps and initial noise samples used by loss function [ignored in train mode]

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        # for deterministic model, do not use the time or target inputs - set to 0 always
        x = torch.zeros_like(batch)
        t = torch.zeros(batch.shape[0])#, device=batch.device)
        pred = model(x, cond, t)

        # CREATE 1s and 0s GROUND TRUTH DATA
        batch_binary = (batch >= threshold).to(torch.float32)
        
        #if (torch.sum(batch_binary) > 0):
        #    if is_main_process():
        #        logger.info(f" >> >> INSIDE get_BCEWithLogitsLoss | mean pred {torch.mean(pred)}")
        #        logger.info(f" >> >> INSIDE get_BCEWithLogitsLoss | sum batch_binary {torch.sum(batch_binary)}")
        
        loss = criterion(pred, batch_binary)

        #if is_main_process():
        #   logger.info(f" >> >> INSIDE get_deterministic_loss_fn loss {loss}")
        #   logger.info("_____________________________________________________")
        return loss

    return loss_fn
#====================================================================
def get_mse_loss_fn(sde, train, reduce_mean=True):
    def loss_fn(model, batch, cond, generator=None):
        """Compute the MSE loss function for a deterministic model.

        Args:
        model: A score model.
        batch: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.
        generator: An optional random number generator so can control the timesteps and initial noise samples used by loss function [ignored in train mode]

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        # for deterministic model, do not use the time or target inputs - set to 0 always
        x = torch.zeros_like(batch)
        t = torch.zeros(batch.shape[0])#, device=batch.device)
        pred = model(x, cond, t)

        loss = F.mse_loss(pred, batch)

        #if is_main_process():
        #   logger.info(f" >> >> INSIDE get_mse_loss_fn loss {loss}")
        #   logger.info("_____________________________________________________")
        return loss
    return loss_fn
#====================================================================
def get_loss(sde, train, config, zarr_path):
    print(" >> >> INSIDE losses.get_loss sde", type(sde), ", config.deterministic", config.deterministic, type(config.deterministic))
    logger.info(" >> >> INSIDE losses.get_loss sde %s, config.deterministic %s %s", type(sde), config.deterministic, type(config.deterministic))
    
    if (config.deterministic or (config.deterministic == 'True')):
        assert config.data.predictands.target_transform_keys[0] == "none", (
            f" <<ERROR>> ASYM loss requires physical-unit targets (target_transform_keys must be 'none'). "
            f"Got {config.data.predictands.target_transform_keys}"
        )
        assert (config.training.det_loss_type in ['BCE', 'MSE', 'DUAL']), (
            f" <<ERROR>> config.training.det_loss_type must be BCE, MSE, or DUAL. Got {config.training.det_loss_type}"
        )

        if (config.training.det_loss_type == 'BCE'):
            weight = get_BCE_loss_weight(config, zarr_path)
            criterion = nn.BCEWithLogitsLoss(weight=weight, reduction=config.training.reduction)
        
            threshold = np.float32(config.training.precip_threshold)
            threshold_tensor = torch.tensor(threshold)
        
            loss_fn = get_BCEWithLogitsLoss(train, criterion, threshold_tensor)
        
        elif (config.training.det_loss_type == 'MSE'):
            loss_fn = get_mse_loss_fn(sde, train, reduce_mean=config.training.reduce_mean)
        
        elif (config.training.det_loss_type == 'DUAL'):
            weight = get_BCE_loss_weight(config, zarr_path)
            criterion = nn.BCEWithLogitsLoss(weight=weight, reduction=config.training.reduction)
            threshold_tensor = torch.tensor(np.float32(config.training.precip_threshold))
            _bce = get_BCEWithLogitsLoss(train, criterion, threshold_tensor)
            _mse = get_mse_loss_fn(sde, train, reduce_mean=config.training.reduce_mean)

            if config.training.balance_losses:
                ema = [1.0, 1.0]  # [ema_mse, ema_bce]
                alpha = 0.99

                def loss_fn(model, batch, cond, generator=None):
                    l_mse = _mse(model, batch, cond, generator)
                    l_bce = _bce(model, batch, cond, generator)
                    ema[0] = alpha * ema[0] + (1 - alpha) * l_mse.item()
                    ema[1] = alpha * ema[1] + (1 - alpha) * l_bce.item()
                    
                    l_total = l_mse + (ema[0] / (ema[1] + 1e-8)) * l_bce
                    
                    l_mse_ = l_mse.item()
                    l_bce_ = l_bce.item()
                    l_total_ = l_total.item()
                    #if is_main_process():
                    #    logger.info(f" >> >> INSIDE get_loss DUAL BALANCED | l_mse={l_mse_:.5f}, l_bce={l_bce_:.5f}, l_total={l_total_:.5f}, ema[0]={ema[0]:.5f}, ema[1]={ema[1]:.5f}")
                    
                    return l_total
            else:
                def loss_fn(model, batch, cond, generator=None):
                    l_mse = _mse(model, batch, cond, generator)
                    l_bce = _bce(model, batch, cond, generator)

                    l_mse_ = l_mse.item()
                    l_bce_ = l_bce.item()
                    l_total_ = l_mse_+l_bce_
                    #if is_main_process():
                    #    logger.info(f" >> >> INSIDE get_loss DUAL | l_mse={l_mse_:.5f}, l_bce={l_bce_:.5f}, l_total={l_total_:.5f}")
                    
                    return l_mse + l_bce
       
    else:
        if config.training.continuous:
            loss_fn = get_sde_loss_fn(sde, 
                                      train,
                                      reduce_mean=config.training.reduce_mean,
                                      continuous=True,
                                      likelihood_weighting=config.training.likelihood_weighting)
        else:
            assert not config.training.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, VESDE):
                loss_fn = get_smld_loss_fn(sde, train, reduce_mean=config.training.reduce_mean)
            elif isinstance(sde, VPSDE):
                loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=config.training.reduce_mean)
            else:
                raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")
    return loss_fn
#====================================================================

def get_sde_loss_fn(sde, train, reduce_mean=True, continous=True, likelihood_weighting=False):
    # Dummy function to preserve sde and keep code running, even though it won't be used
    return 0

def get_smld_loss_fn(sde, train, reduce_mean=True):
    # Dummy function to preserve sde and keep code running, even though it won't be used
    return 0

def get_smld_loss_fn(sde, train, reduce_mean=True):
    # Dummy function to preserve sde and keep code running, even though it won't be used
    return 0

def get_ddpm_loss_fn(sde, train, reduce_mean=True):
    # Dummy function to preserve sde and keep code running, even though it won't be used
    return 0