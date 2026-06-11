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
    step = 0
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

        pred = pred.float()

        # CREATE 1s and 0s GROUND TRUTH DATA
        # Per-sample: shape (batch_size,)
        batch_binary = (batch >= threshold).any(dim=tuple(range(1, batch.ndim))).float()
        
        if step % 1000 == 0 and is_main_process():
            logger.info(f" >> >> INSIDE get_BCEWithLogitsLoss | pred {pred.detach().cpu().numpy()}")
            logger.info(f" >> >> INSIDE get_BCEWithLogitsLoss | batch_binary {batch_binary.detach().cpu().numpy()}")
            
        loss = criterion(pred, batch_binary)
        step += 1
        #if is_main_process():
        #   logger.info(f" >> >> INSIDE get_deterministic_loss_fn loss {loss}")
        #   logger.info("_____________________________________________________")
        return loss

    return loss_fn
#====================================================================

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