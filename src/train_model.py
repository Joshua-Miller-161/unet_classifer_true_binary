"""Training for score-based generative models. """

import sys
sys.dont_write_bytecode = True
from collections import defaultdict
import os
from absl import flags
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from dotenv import load_dotenv
import logging
import torch
from torchinfo import summary
#torch.set_float32_matmul_precision('medium')
logger = logging.getLogger(__name__)
#===================================================================
FLAGS = flags.FLAGS
#====================================================================
from src.deterministic_models import cncsnpp
from src.lightningModuleEMA import ScoreModelLightningModule
from src.utils import LossOnlyProgressBar, setup_checkpoint, check_saved_checkpoint, is_main_process, create_model, save_config
#from src.data_scripts.collate_np.data_module import LightningDataModule
from src.data_scripts.collate_np_per_var.data_module import LightningDataModule
from src.losses import get_loss
from src.sde_lib import get_sde
from src.data_scripts.data_utils import datafile_path
#====================================================================
def train(config, filename, val_filename):
    # PyTorch 2 global speedups — only applied when explicitly enabled.
    # When False, training behaviour is 100% unchanged from the original.
    _trainer_extra_kwargs = {}
    if getattr(config.training, 'pytorch2_speedup', False):
        from src.pytorch2_speedup_utils import (
            detect_hardware,
            log_capabilities,
            apply_global_speedups,
            apply_inductor_settings,
            get_trainer_precision,
        )
        _pt2_caps = detect_hardware()
        log_capabilities(_pt2_caps)
        apply_global_speedups(_pt2_caps)
        apply_inductor_settings(_pt2_caps)
        _trainer_extra_kwargs["precision"] = get_trainer_precision(_pt2_caps)
    
    if ((config.deterministic == 'True') or (config.deterministic == 'true') or (config.deterministic == True) or (config.deterministic == 1)):
        config.deterministic = True
        #config.model.name = config.model.name
        config.model.name = 'det_'+config.model.name
    else:
        config.deterministic = False

    if not config.data.input_transform_dataset or str(config.data.input_transform_dataset).lower() in {"none", "null", ""}:
        config.data.input_transform_dataset = config.data.dataset_name

    
    if is_main_process():
        print(" >> INSIDE train_model.py: got run_config")
        print(" >> INSIDE train_model.py folder:", str(os.path.join(os.getenv('DERIVED_DATA'), config.data.dataset_name, config.experiment_name)))
    logger.info(" >> INSIDE train_model.py: got run_config")
    logger.info(" >> INSIDE train_model.py folder: %s", str(os.path.join(os.getenv('DERIVED_DATA'), config.data.dataset_name, config.experiment_name, config.data.input_transform_dataset)))

    logger.info(" >> INSIDE train_model.py config.deterministic %s, sde: %s", config.deterministic, config.training.sde)

    target_xfm_keys = defaultdict(lambda: config.data.target_transform_key) | dict(config.data.target_transform_overrides)

    data_module = LightningDataModule(
        config=config,
        active_dataset_name=config.data.dataset_name,
        model_src_dataset_name=config.data.dataset_name,
        input_transform_dataset_name=config.data.input_transform_dataset,
        transform_dir=os.path.join(os.getenv('WORK_DIR'), 'transforms', 'unet_classifier'),
        batch_size=config.training.batch_size,
        filename=filename,
        val_filename=val_filename,
        include_time_inputs=False,
        evaluation=False,
        shuffle=True,
        num_workers=3,
        prefetch_factor=3
    )

    train_loss_fn = get_loss(get_sde(config), True, config, datafile_path(config.data.dataset_name, filename))
    val_loss_fn = get_loss(get_sde(config), False, config, datafile_path(config.data.dataset_name, filename))
    
    model = ScoreModelLightningModule(config, train_loss_fn, val_loss_fn)

    pbar = LossOnlyProgressBar()

    checkpoint_cb, checkpoint_path = setup_checkpoint(config, os.getenv('WORK_DIR'))

    save_config(config, os.path.join(checkpoint_path, "config.yml"))

    resume_checkpoint_path = check_saved_checkpoint(checkpoint_path)

    trainer = Trainer(
        default_root_dir=os.path.join("lightning_logs", config.data.dataset_name),
        max_epochs=config.training.n_epochs,
        accelerator="gpu",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=False) if int(os.environ.get("SLURM_GPUS_ON_NODE", 1)) > 1 else "auto", # Orig. DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
        use_distributed_sampler=True,
        log_every_n_steps=10,
        val_check_interval=1.0, # Run validation at the end of every epoch
        callbacks=[pbar, checkpoint_cb],
        **_trainer_extra_kwargs
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_checkpoint_path
    )