import sys
sys.dont_write_bytecode = True
from torch.optim import Adam, SGD
#====================================================================
def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if (config.optim.optimizer == 'Adam'):
    adam_kwargs = dict(
        lr=config.optim.lr,
        betas=(config.optim.beta1, 0.999),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )
    # Level 8: fused Adam — single CUDA kernel for param updates (~5-20% speedup).
    # Only when pytorch2_speedup=True; when False, uses standard Adam unchanged.
    if getattr(getattr(config, 'training', None), 'pytorch2_speedup', False):
        try:
            from .pytorch2_speedup_utils import detect_hardware, get_fused_adam_kwargs
            _caps = detect_hardware()
            adam_kwargs.update(get_fused_adam_kwargs(_caps))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Fused Adam setup failed: %s", exc)
    optimizer = Adam(params, **adam_kwargs)

  elif (config.optim.optimizer == 'SGD'):
    optimizer = SGD(params,
                    lr=config.optim.lr,
                    momentum=0.9,
                    weight_decay=config.optim.weight_decay)
  
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer