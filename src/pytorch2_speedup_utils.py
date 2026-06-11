"""
PyTorch 2 training speedup utilities for A100 HPC environments.

All speedups are ONLY active when config.training.pytorch2_speedup = True.
When False, zero changes are made — original training behavior is preserved exactly.

Hierarchy (each level is gated on hardware/version compatibility via try/except):
  Level 1: TF32 + matmul precision=medium  (Ampere+ cc>=8.0)
  Level 2: cuDNN benchmark auto-tune       (any CUDA)
  Level 3: FP16 reduced-precision accum.   (Ampere+)
  Level 4: Flash / mem-efficient SDPA      (Ampere+, PyTorch 2.0+)
  Level 5: torch._inductor compiler opts   (PyTorch 2.0+)
  Level 6: channels_last memory format     (Ampere+, conv-heavy models)
  Level 7: torch.compile                   (PyTorch 2.0+, hierarchical mode fallback)
  Level 8: Fused Adam optimizer kwargs     (PyTorch 2.0+, CUDA)
  Trainer: Lightning precision=bf16-mixed  (A100 native BF16)

Hardware target: NVIDIA A100-SXM4-40GB x4, AMD EPYC 7452, CUDA 13.2, PyTorch 2.x
"""

import logging
import inspect
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Hardware Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_hardware():
    """
    Probe GPU/CUDA/PyTorch capabilities without raising exceptions.
    Returns a flat dict consumed by all other functions in this module.
    All entries default to a safe/disabled value so callers can do
    caps.get("feature") without key-error checks.
    """
    caps = {
        "cuda_available":              False,
        "gpu_count":                   0,
        "compute_capability":          (0, 0),
        "compute_capability_float":    0.0,
        "is_ampere_or_newer":          False,   # A100, RTX 30xx — cc >= 8.0
        "is_hopper_or_newer":          False,   # H100              cc >= 9.0
        "bf16_native":                 False,   # Native BF16 maths (Ampere+)
        "tf32_supported":              False,   # TF32 matmul       (Ampere+)
        "flash_sdp_available":         False,   # SDPA flash-attn   (Ampere+, PT2+)
        "compile_available":           False,   # torch.compile     (PT2+)
        "triton_available":            False,   # Triton backend
        "cuda_version":                None,
        "torch_version":               torch.__version__,
        "gpu_names":                   [],
    }

    if not torch.cuda.is_available():
        logger.info("detect_hardware: no CUDA device found.")
        return caps

    caps["cuda_available"] = True

    try:
        caps["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        logger.warning("Could not query GPU count: %s", exc)

    try:
        caps["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    try:
        major, minor = torch.cuda.get_device_capability(0)
        caps["compute_capability"]       = (major, minor)
        caps["compute_capability_float"] = major + minor / 10.0
        caps["is_ampere_or_newer"]       = (major >= 8)
        caps["is_hopper_or_newer"]       = (major >= 9)
    except Exception as exc:
        logger.warning("Could not get device capability: %s", exc)

    caps["bf16_native"]    = caps["is_ampere_or_newer"]
    caps["tf32_supported"] = caps["is_ampere_or_newer"]

    try:
        caps["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(caps["gpu_count"])
        ]
    except Exception:
        pass

    # Flash SDPA: needs PyTorch 2.0+ and Ampere+
    try:
        caps["flash_sdp_available"] = (
            caps["is_ampere_or_newer"]
            and hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and hasattr(torch.backends.cuda, "enable_flash_sdp")
        )
    except Exception:
        pass

    # torch.compile: PyTorch 2.0+
    try:
        torch_major = int(torch.__version__.split(".")[0])
        caps["compile_available"] = (torch_major >= 2) and caps["cuda_available"]
    except Exception:
        pass

    # Triton (needed for inductor / max-autotune)
    if caps["compile_available"]:
        try:
            import triton  # noqa: F401
            caps["triton_available"] = True
        except ImportError:
            caps["triton_available"] = False

    return caps


def log_capabilities(caps, rank=0):
    """Print a capability summary to logger and stdout (rank-0 only)."""
    if rank != 0:
        return
    lines = [
        "=" * 72,
        "  PyTorch 2 Speedup — Detected Hardware Capabilities",
        "=" * 72,
        f"  PyTorch version      : {caps['torch_version']}",
        f"  CUDA available       : {caps['cuda_available']}",
        f"  CUDA version         : {caps['cuda_version']}",
        f"  GPU count            : {caps['gpu_count']}",
        f"  GPU names            : {caps['gpu_names']}",
        f"  Compute capability   : {caps['compute_capability']}"
        f"  (Ampere = cc (8,0), A100 = cc (8,0))",
        f"  Ampere+ (TF32/BF16)  : {caps['is_ampere_or_newer']}",
        f"  BF16 native          : {caps['bf16_native']}",
        f"  TF32 supported       : {caps['tf32_supported']}",
        f"  Flash SDPA available : {caps['flash_sdp_available']}",
        f"  torch.compile avail. : {caps['compile_available']}",
        f"  Triton available     : {caps['triton_available']}",
        "=" * 72,
    ]
    for line in lines:
        logger.info(line)
    print("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Level 1-4: global PyTorch settings
# ──────────────────────────────────────────────────────────────────────────────

def apply_global_speedups(caps, rank=0):
    """
    Apply global PyTorch performance flags.
    Each level is guarded by a try/except — failures are warn-logged and skipped.
    Returns list of successfully applied setting names (for logging/diagnostics).
    """
    applied = []

    if not caps.get("cuda_available"):
        logger.info("apply_global_speedups: no CUDA — skipping.")
        return applied

    # Level 1: TF32 + matmul precision (Ampere+ only)
    # 'medium' routes float32 matmuls through BF16 accumulation on A100,
    # giving ~2-3x TFLOPS with negligible accuracy loss for this model class.
    if caps.get("tf32_supported"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("medium")
            applied.append("TF32 + matmul_precision=medium")
        except Exception as exc:
            logger.warning("TF32 setup failed: %s", exc)

    # Level 2: cuDNN benchmark — auto-selects fastest conv algorithm per shape
    try:
        torch.backends.cudnn.benchmark = True
        applied.append("cuDNN benchmark=True")
    except Exception as exc:
        logger.warning("cuDNN benchmark failed: %s", exc)

    # Level 3: FP16 reduced-precision reduction (Ampere+ — safe for BF16 AMP runs)
    if caps.get("is_ampere_or_newer"):
        try:
            if hasattr(torch.backends.cuda, "allow_fp16_reduced_precision_reduction"):
                torch.backends.cuda.allow_fp16_reduced_precision_reduction = True
                applied.append("FP16 reduced-precision reduction")
        except Exception as exc:
            logger.warning("FP16 reduced-precision reduction failed: %s", exc)

    # Level 4: Flash Attention + memory-efficient SDPA (Ampere+, PyTorch 2.0+)
    # Gives 4-8x speedup on attention blocks (AttnBlockpp in this model).
    if caps.get("flash_sdp_available"):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            applied.append("Flash SDP")
        except Exception as exc:
            logger.warning("Flash SDP enable failed: %s", exc)
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            applied.append("Memory-efficient SDP")
        except Exception as exc:
            logger.warning("Memory-efficient SDP enable failed: %s", exc)

    if rank == 0:
        logger.info("Global speedups applied: %s", applied)
    return applied


# ──────────────────────────────────────────────────────────────────────────────
# Level 5: torch._inductor compiler settings
# ──────────────────────────────────────────────────────────────────────────────

def _try_set(obj, attr, val, applied_list, label):
    """Safely set obj.attr = val; append label to applied_list on success."""
    try:
        if hasattr(obj, attr):
            setattr(obj, attr, val)
            applied_list.append(label)
    except Exception as exc:
        logger.warning("Could not set %s: %s", label, exc)


def apply_inductor_settings(caps, rank=0):
    """
    Configure torch._inductor for maximum A100 throughput.
    Only relevant when torch.compile is used (Level 7).
    Returns list of applied setting names.
    """
    applied = []
    if not caps.get("compile_available"):
        return applied

    try:
        import torch._inductor.config as ind

        # Coordinate-descent tuning: iteratively searches kernel launch params.
        # Adds ~10-20 min first-epoch compilation; amortises over long training runs.
        _try_set(ind, "coordinate_descent_tuning", True, applied,
                 "coordinate_descent_tuning")

        # Epilogue fusion: fuses pointwise ops (activations, bias add) after matmul.
        _try_set(ind, "epilogue_fusion", True, applied, "epilogue_fusion")

        # Exhaustive GEMM algorithm search (similar to cudnn benchmark for matmuls).
        _try_set(ind, "max_autotune_gemm", True, applied, "max_autotune_gemm")

        # Unique Triton kernel names: no accuracy impact, helps profiling.
        try:
            if hasattr(ind, "triton"):
                _try_set(ind.triton, "unique_kernel_names", True, applied,
                         "triton.unique_kernel_names")
        except Exception:
            pass

    except ImportError:
        logger.warning("torch._inductor not importable; skipping inductor config.")
    except Exception as exc:
        logger.warning("Inductor config error: %s", exc)

    if rank == 0:
        logger.info("Inductor settings applied: %s", applied)
    return applied


# ──────────────────────────────────────────────────────────────────────────────
# Level 6: Channels-last memory format
# ──────────────────────────────────────────────────────────────────────────────

def apply_channels_last(model, caps):
    """
    Convert model weights to NHWC (channels_last) layout.
    A100 cuDNN NHWC convolution kernels are 10-30% faster than NCHW.
    Apply BEFORE torch.compile so the compiler can optimise for the layout.

    Returns True if conversion succeeded (so callers can also convert input
    batches to channels_last in transfer_batch_to_device).
    """
    if not caps.get("is_ampere_or_newer"):
        logger.info("apply_channels_last: not Ampere+ — skipping.")
        return False
    try:
        model.to(memory_format=torch.channels_last)
        logger.info("channels_last memory format applied to model.")
        return True
    except Exception as exc:
        logger.warning("channels_last conversion failed (%s) — skipping.", exc)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Level 7: torch.compile
# ──────────────────────────────────────────────────────────────────────────────

def compile_model(base_model, caps, config, rank=0):
    """
    Compile model with torch.compile using the best available mode.

    Hierarchy:
      1. max-autotune    — exhaustive Triton kernel search; best A100 throughput
                           (~10-20 min first-epoch warm-up, amortised over training)
      2. reduce-overhead — CUDA graphs; minimal Python overhead per step
      3. default         — balanced; supports dynamic shapes
      4. uncompiled      — safety fallback if all torch.compile calls throw

    Returns (model_or_compiled, mode_description_string).
    """
    if not caps.get("compile_available"):
        if rank == 0:
            logger.info("compile_model: torch.compile not available — uncompiled.")
        return base_model, "none (torch.compile unavailable)"

    candidates = [
        {"mode": "max-autotune",    "fullgraph": False, "dynamic": False},
        {"mode": "reduce-overhead", "fullgraph": False, "dynamic": False},
        {"mode": "default",         "fullgraph": False, "dynamic": True},
    ]

    for cand in candidates:
        mode   = cand["mode"]
        kwargs = {k: v for k, v in cand.items() if k != "mode"}
        try:
            compiled = torch.compile(base_model, mode=mode, **kwargs)
            if rank == 0:
                logger.info("torch.compile: mode='%s' succeeded.", mode)
            return compiled, f"torch.compile(mode={mode!r})"
        except Exception as exc:
            if rank == 0:
                logger.warning(
                    "torch.compile(mode='%s') failed: %s — trying next mode.", mode, exc
                )

    if rank == 0:
        logger.warning("compile_model: all modes failed — using uncompiled model.")
    return base_model, "none (all compile modes failed)"


# ──────────────────────────────────────────────────────────────────────────────
# Level 8: Fused Adam optimizer kwargs
# ──────────────────────────────────────────────────────────────────────────────

def get_fused_adam_kwargs(caps):
    """
    Return {'fused': True} when PyTorch 2+ fused Adam is available on CUDA.
    Fused Adam executes the update step in a single CUDA kernel, reducing
    memory bandwidth and Python overhead (~5-20% optimizer step speedup).
    Returns {} when unsupported so callers can do Adam(params, **kwargs) safely.
    """
    if not caps.get("cuda_available"):
        return {}
    try:
        torch_major = int(torch.__version__.split(".")[0])
        if torch_major >= 2:
            from torch.optim import Adam as _Adam
            if "fused" in inspect.signature(_Adam.__init__).parameters:
                return {"fused": True}
    except Exception as exc:
        logger.warning("get_fused_adam_kwargs: check failed: %s", exc)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Lightning Trainer precision recommendation
# ──────────────────────────────────────────────────────────────────────────────

def get_trainer_precision(caps):
    """
    Return the recommended Lightning Trainer precision= string for this hardware.

    Hierarchy:
      bf16-mixed  — A100 native BF16; ~2x vs FP32, no loss scaling needed
      16-mixed    — FP16 + GradScaler; fallback for Volta (cc 7.0)
      32-true     — Full FP32; safe fallback when CUDA unavailable or checks fail
    """
    if not caps.get("cuda_available"):
        return "32-true"

    # BF16 native: Ampere+ (A100 is cc 8.0)
    if caps.get("bf16_native"):
        try:
            t = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
            del t
            return "bf16-mixed"
        except Exception as exc:
            logger.warning("BF16 smoke-test failed: %s", exc)

    # FP16: Volta+ (cc >= 7.0)
    if caps.get("compute_capability_float", 0.0) >= 7.0:
        try:
            t = torch.zeros(1, dtype=torch.float16, device="cuda")
            del t
            return "16-mixed"
        except Exception as exc:
            logger.warning("FP16 smoke-test failed: %s", exc)

    return "32-true"


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper used by lightningModuleEMA.py
# ──────────────────────────────────────────────────────────────────────────────

def setup_model_speedups(base_model, caps, config, rank=0):
    """
    Apply model-level speedups (channels_last + torch.compile) and return
    (compiled_model, use_channels_last_bool).

    Called from ScoreModelLightningModule.__init__ when pytorch2_speedup=True.
    Global settings (TF32, cuDNN, SDPA, inductor) are handled separately in
    train_model.py before the Trainer is constructed.
    """
    # Level 6: channels_last BEFORE compile — lets the compiler optimise for NHWC
    use_cl = apply_channels_last(base_model, caps)

    # Level 7: torch.compile
    model, cmode = compile_model(base_model, caps, config, rank=rank)

    if rank == 0:
        logger.info(
            "setup_model_speedups: channels_last=%s | compile=%s", use_cl, cmode
        )
    return model, use_cl