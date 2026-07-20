#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified RPM training loop using model-owned metric updates.

Assumes these files are available on the Python path:
    util_refac.py
    logger_refac.py
    rpm_refac.py
    speed_refac.py

The metric-update rule is owned by RPM. The training loop always uses:

    model.forward(...)
    model.train_regressor(...)
    model.backward(...)
    model.update_metrics(...)

The logger stores only:
    - epoch
    - per-epoch metrics implemented in util_refac.py
    - RPM layer metric matrices
    - model weights
    - condition number of the training-set U matrix
"""

import os
import sys

# Ensure local package modules are loaded from this script's directory first.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _SCRIPT_DIR:
    try:
        sys.path.remove(_SCRIPT_DIR)
    except ValueError:
        pass
    sys.path.insert(0, _SCRIPT_DIR)
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dspTools as dt

from config_refac import Config
from logger_refac import Logger
from rpm_refac import RPM, RPMLayer
from speed_refac import (
    MahalanobisRBFKernel,
    MahalanobisLaplacianKernel,
)
from util_refac import (
    arithmetic_fisher,
    covariance_effective_dim,
    excitation_alignment,
    feature_covariance,
    geometric_fisher,
    mse,
    normalized_mse,
    target_alignment,
)
from device_utils import resolve_device


# =============================================================================
# Data / model helpers
# =============================================================================

def _prepare_runtime_config(config: Config) -> str:
    """Resolve the requested device and optionally route output to a device subdir."""
    resolved_device = resolve_device(config.get("device", "auto"))
    config.device = resolved_device

    base_experiment_dir = config.get("_base_experiment_dir", None)
    if base_experiment_dir is None:
        current_experiment_dir = config.experiment_dir
        if isinstance(current_experiment_dir, str):
            basename = os.path.basename(os.path.normpath(current_experiment_dir))
            if basename.startswith("device_"):
                base_experiment_dir = os.path.dirname(current_experiment_dir)
            else:
                base_experiment_dir = current_experiment_dir
        else:
            base_experiment_dir = current_experiment_dir

    config._base_experiment_dir = base_experiment_dir

    if bool(config.get("use_device_subdir", False)):
        device_label = resolved_device.replace(":", "_")
        target_dir = os.path.join(base_experiment_dir, f"device_{device_label}")
        if os.path.normpath(config.experiment_dir) != os.path.normpath(target_dir):
            config.experiment_dir = target_dir
    else:
        config.experiment_dir = base_experiment_dir

    return resolved_device


def make_mackey_glass_splits(
    config: Config,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/val/test splits matching the old Mackey-Glass setup.
    """
    device = torch.device(_prepare_runtime_config(config))

    system_info = {
        "System": "MackeyGlassdB",
        "Ntest": config.num_test_samples,
        "N": config.num_train_samples + config.num_val_samples,
        "L": config.num_lags,
        "wgnVar": config.wgn_var,
        "pH": config.prediction_horizon,
        "align": False,
        "sdim": False,
        "testRatio": config.test_ratio,
        "AlphaStable": False,
        "ImpNoise": False,
        "D": None,
        "Stable_Alpha": 1.4,
        "Stable_Scale": 0.1,
    }

    x_all, x_test, y_all, y_test = dt.createInputs(system_info, seed=seed)

    x_val = x_all[-config.num_val_samples:]
    y_val = y_all[-config.num_val_samples:]

    x_train = x_all[:-config.num_val_samples]
    y_train = y_all[:-config.num_val_samples]

    x_train = torch.as_tensor(x_train, dtype=config.dtype, device=device)
    y_train = torch.as_tensor(y_train, dtype=config.dtype, device=device).reshape(-1)

    x_val = torch.as_tensor(x_val, dtype=config.dtype, device=device)
    y_val = torch.as_tensor(y_val, dtype=config.dtype, device=device).reshape(-1)

    x_test = torch.as_tensor(x_test, dtype=config.dtype, device=device)
    y_test = torch.as_tensor(y_test, dtype=config.dtype, device=device).reshape(-1)

    return x_train, y_train, x_val, y_val, x_test, y_test


def _as_list(value, *, name: str):
    """
    Convert a scalar config value to a list.

    Lists and tuples are preserved.
    """
    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def _require_same_length(values, *, reference_length: int, name: str):
    """
    Require a per-layer config list to match the number of RPM layers.
    """
    if len(values) != reference_length:
        raise ValueError(
            f"{name} must have length {reference_length}. "
            f"Got length {len(values)}."
        )

    return values



def _broadcast_to_length(value, *, reference_length: int, name: str):
    """
    Convert scalar config entries to a per-layer list.

    If value is a scalar, it is broadcast to every layer. If it is a list, it
    must already have length reference_length.
    """
    values = _as_list(value, name=name)

    if len(values) == 1:
        return values * reference_length

    return _require_same_length(
        values,
        reference_length=reference_length,
        name=name,
    )


def _canonical_kernel_type(kernel_type: str) -> str:
    """
    Normalize kernel family names used in the config.
    """
    kernel_type = str(kernel_type).lower()

    aliases = {
        "rbf": "rbf",
        "gaussian": "rbf",
        "mahalanobis_rbf": "rbf",
        "laplacian": "laplacian",
        "laplace": "laplacian",
        "mahalanobis_laplacian": "laplacian",
    }

    if kernel_type not in aliases:
        raise ValueError(
            "kernel_type must be one of {'rbf', 'gaussian', 'laplacian', "
            "'laplace', 'mahalanobis_rbf', 'mahalanobis_laplacian'}. "
            f"Got {kernel_type!r}."
        )

    return aliases[kernel_type]


def make_mahalanobis_kernel(
    *,
    kernel_type: str,
    sigma: float,
    metric_a: torch.Tensor,
    r_eps: float,
):
    """
    Construct the configured Mahalanobis radial kernel.
    """
    kernel_type = _canonical_kernel_type(kernel_type)

    if kernel_type == "rbf":
        return MahalanobisRBFKernel(
            sigma=sigma,
            metric_a=metric_a,
            r_eps=r_eps,
        )

    if kernel_type == "laplacian":
        return MahalanobisLaplacianKernel(
            sigma=sigma,
            metric_a=metric_a,
            r_eps=r_eps,
        )

    raise RuntimeError(f"Unhandled kernel_type={kernel_type!r}.")


def make_rpm(
    input_dim: int,
    config: Config,
) -> RPM:
    """
    Construct an RPM from config.

    Supports both single-layer and multilayer RPMs.

    Required config entries
    -----------------------
    output_dim : list[int]
        Output dimension for each RPM layer.

        Examples:
            output_dim: [500]
            output_dim: [500, 250, 100]

    alphas : list[float]
        Metric update rate for each RPM layer.

        Examples:
            alphas: [0.5]
            alphas: [0.5, 0.25, 0.1]

    kernel_sigma : list[float]
        Kernel bandwidth/length scale for each RPM layer.

    kernel_type : str or list[str]
        Kernel family. Supported values are "rbf" and "laplacian".
        A scalar is broadcast to all layers; a list gives per-layer kernels.

        Examples:
            kernel_sigma: [0.75]
            kernel_sigma: [0.75, 0.5, 0.25]

    Shared across layers
    --------------------
    The following settings are shared across all layers:
        approx_method
        nystrom_m
        nystrom_mode
        nystrom_seed
        chunk_size
        metric_clamp
        metric_rank_eps
        store_full_centers

    This function only creates the model. It does not set centers, build
    spectral features, train the readout, compute metrics, or update the metric.
    """
    _prepare_runtime_config(config)

    config.require(
        "device",
        "dtype",
        "output_dim",
        "alphas",
        "kernel_sigma",
        "metric_rank_eps",
        "chunk_size",
        "metric_clamp",
        "approx_method",
        "nystrom_m",
        "nystrom_mode",
        "nystrom_seed",
        "store_full_centers",
    )

    output_dims = [
        int(dim)
        for dim in _as_list(config.output_dim, name="output_dim")
    ]

    num_layers = len(output_dims)

    alphas = [
        float(alpha)
        for alpha in _require_same_length(
            _as_list(config.alphas, name="alphas"),
            reference_length=num_layers,
            name="alphas",
        )
    ]

    kernel_sigmas = [
        float(sigma)
        for sigma in _require_same_length(
            _as_list(config.kernel_sigma, name="kernel_sigma"),
            reference_length=num_layers,
            name="kernel_sigma",
        )
    ]

    kernel_types = [
        _canonical_kernel_type(kernel_type)
        for kernel_type in _broadcast_to_length(
            config.get("kernel_type", "rbf"),
            reference_length=num_layers,
            name="kernel_type",
        )
    ]

    device = torch.device(config.device)

    layers = []
    layer_input_dim = int(input_dim)

    for output_dim, alpha, kernel_sigma, kernel_type in zip(
        output_dims,
        alphas,
        kernel_sigmas,
        kernel_types,
    ):
        initial_metric = torch.eye(
            layer_input_dim,
            dtype=config.dtype,
            device=device,
        )

        kernel = make_mahalanobis_kernel(
            kernel_type=kernel_type,
            sigma=kernel_sigma,
            metric_a=initial_metric,
            r_eps=config.metric_rank_eps,
        )

        layer = RPMLayer(
            kernel=kernel,
            output_dim=output_dim,
            alpha=alpha,
            chunk_size=config.chunk_size,
            metric_clamp=config.metric_clamp,
            approx_method=config.approx_method,
            nystrom_m=config.nystrom_m,
            nystrom_mode=config.nystrom_mode,
            nystrom_seed=config.nystrom_seed,
            freeze=False,
            copy_kernel=True,
            full=config.store_full_centers,
        )

        layers.append(layer)

        # The next layer operates on this layer's feature representation.
        layer_input_dim = output_dim

    update_config = {
        # AGOP options.
        "center_grads": config.get("center_grads", True),
        "metric_norm_type": config.get("metric_norm_type", "Trace"),
        "metric_norm_mult": float(config.get("metric_norm_mult", 1.0)),
        "metric_eps": float(config.get("metric_eps", 1e-12)),

        # Gradient-descent options. These are ignored when
        # metric_update_rule == "agop".
        "metric_gradient_objective": config.get(
            "metric_gradient_objective",
            "fixed_readout",
        ),
        "metric_gradient_lrs": config.get("metric_gradient_lrs", None),
        "metric_gradient_optimizer": config.get(
            "metric_gradient_optimizer",
            "sgd",
        ),
        "metric_gradient_adam_beta1": float(
            config.get("metric_gradient_adam_beta1", 0.9)
        ),
        "metric_gradient_adam_beta2": float(
            config.get("metric_gradient_adam_beta2", 0.999)
        ),
        "metric_gradient_adam_eps": float(
            config.get("metric_gradient_adam_eps", 1e-8)
        ),
        "metric_gradient_loss": config.get("metric_gradient_loss", "mse"),
        "metric_gradient_eps": float(config.get("metric_gradient_eps", 1e-12)),
        "metric_gradient_project_psd": bool(
            config.get("metric_gradient_project_psd", True)
        ),
        "metric_gradient_max_norm": float(
            config.get("metric_gradient_max_norm", 1.0)
        ),
        "metric_gradient_diag_load": config.get(
            "metric_gradient_diag_load",
            1e-8,
        ),
        "metric_gradient_use_current_readout": bool(
            config.get("metric_gradient_use_current_readout", True)
        ),
        "metric_gradient_store_differentiable_readout": bool(
            config.get("metric_gradient_store_differentiable_readout", False)
        ),

        # FOOF options. These are ignored unless metric_update_rule == "foof".
        "foof_optimizer": config.get(
            "foof_optimizer",
            config.get("metric_gradient_optimizer", "sgd"),
        ),
        "foof_lrs": config.get(
            "foof_lrs",
            config.get("metric_gradient_lrs", None),
        ),
        "foof_gamma": config.get("foof_gamma", 1e-6),
        "foof_loss": config.get(
            "foof_loss",
            config.get("metric_gradient_loss", "mse"),
        ),
        "foof_eps": float(config.get("foof_eps", config.get("metric_eps", 1e-12))),
        "foof_covariance_normalize": bool(
            config.get("foof_covariance_normalize", False)
        ),
        "foof_factor_rank": config.get("foof_factor_rank", None),
        "foof_max_direction_norm": config.get(
            "foof_max_direction_norm",
            config.get("metric_gradient_max_norm", 1.0),
        ),
        "foof_project_psd": bool(
            config.get(
                "foof_project_psd",
                config.get("metric_gradient_project_psd", True),
            )
        ),
        "foof_diag_load": config.get(
            "foof_diag_load",
            config.get("metric_gradient_diag_load", 0.0),
        ),
        "foof_eigenclip_max": config.get("foof_eigenclip_max", None),
        "foof_use_current_readout": bool(
            config.get(
                "foof_use_current_readout",
                config.get("metric_gradient_use_current_readout", True),
            )
        ),
        "foof_adam_beta1": float(
            config.get(
                "foof_adam_beta1",
                config.get("metric_gradient_adam_beta1", 0.9),
            )
        ),
        "foof_adam_beta2": float(
            config.get(
                "foof_adam_beta2",
                config.get("metric_gradient_adam_beta2", 0.999),
            )
        ),
        "foof_adam_eps": float(
            config.get(
                "foof_adam_eps",
                config.get("metric_gradient_adam_eps", 1e-8),
            )
        ),

        # Needed by full-MMSE gradient mode.
        "reg_param": config.get("reg_param", 1e-16),
        "use_onehot_labels": config.get("use_onehot_labels", False),
    }

    return RPM(
        layers=layers,
        metric_update_rule=config.get(
            "metric_update_rule",
            config.get(
                "metric_update_method",
                config.get("optimization", "agop"),
            ),
        ),
        normalize=bool(
            config.get(
                "normalize_features",
                config.get("normalize", False),
            )
        ),
        update_config=update_config,
    )

def validate_metric_update_config(config: Config) -> None:
    """
    Validate fields needed by the unified model-owned update interface.
    """
    output_dims = _as_list(config.output_dim, name="output_dim")
    alphas = _as_list(config.alphas, name="alphas")
    kernel_sigmas = _as_list(config.kernel_sigma, name="kernel_sigma")

    num_layers = len(output_dims)

    # kernel_type can be scalar or per-layer list.
    for kernel_type in _broadcast_to_length(
        config.get("kernel_type", "rbf"),
        reference_length=num_layers,
        name="kernel_type",
    ):
        _canonical_kernel_type(kernel_type)

    _require_same_length(
        alphas,
        reference_length=num_layers,
        name="alphas",
    )

    _require_same_length(
        kernel_sigmas,
        reference_length=num_layers,
        name="kernel_sigma",
    )

    update_rule = config.get(
        "metric_update_rule",
        config.get(
            "metric_update_method",
            config.get("optimization", "agop"),
        ),
    )
    update_rule = str(update_rule).lower()

    if update_rule in {"gradient", "gd"}:
        update_rule = "gradient_descent"

    if update_rule in {
        "foof_gd",
        "foof_sgd",
        "foof_adam",
        "operator",
        "operator_descent",
        "operator_level",
    }:
        update_rule = "foof"

    if update_rule not in {"agop", "gradient_descent", "foof"}:
        raise ValueError(
            "metric_update_rule must be one of "
            "{'agop', 'gradient_descent', 'foof'} or aliases "
            "{'gradient', 'gd', 'foof_gd', 'foof_sgd', 'foof_adam'}. "
            f"Got {update_rule!r}."
        )

    metric_gradient_lrs = config.get("metric_gradient_lrs", None)

    if metric_gradient_lrs is not None:
        if isinstance(metric_gradient_lrs, (list, tuple)):
            _require_same_length(
                list(metric_gradient_lrs),
                reference_length=num_layers,
                name="metric_gradient_lrs",
            )

    foof_lrs = config.get("foof_lrs", None)

    if foof_lrs is not None:
        if isinstance(foof_lrs, (list, tuple)):
            _require_same_length(
                list(foof_lrs),
                reference_length=num_layers,
                name="foof_lrs",
            )

    foof_gamma = config.get("foof_gamma", None)

    if foof_gamma is not None:
        if isinstance(foof_gamma, (list, tuple)):
            _require_same_length(
                list(foof_gamma),
                reference_length=num_layers,
                name="foof_gamma",
            )

    foof_factor_rank = config.get("foof_factor_rank", None)

    if foof_factor_rank is not None:
        if isinstance(foof_factor_rank, (list, tuple)):
            _require_same_length(
                list(foof_factor_rank),
                reference_length=num_layers,
                name="foof_factor_rank",
            )

    objective = config.get("metric_gradient_objective", "fixed_readout")
    if objective not in {"fixed_readout", "full_mmse"}:
        raise ValueError(
            "metric_gradient_objective must be one of "
            "{'fixed_readout', 'full_mmse'}. "
            f"Got {objective!r}."
        )

    optimizer = str(config.get("metric_gradient_optimizer", "sgd")).lower()
    if optimizer not in {"sgd", "adam"}:
        raise ValueError(
            "metric_gradient_optimizer must be one of {'sgd', 'adam'}. "
            f"Got {optimizer!r}."
        )

    foof_optimizer = str(config.get("foof_optimizer", optimizer)).lower()
    if foof_optimizer == "gd":
        foof_optimizer = "sgd"

    if foof_optimizer not in {"sgd", "adam"}:
        raise ValueError(
            "foof_optimizer must be one of {'sgd', 'gd', 'adam'}. "
            f"Got {foof_optimizer!r}."
        )

    return None


def layer_metric_matrices(model: RPM) -> List[Optional[torch.Tensor]]:
    """
    Return a detached copy of each RPM layer metric matrix.
    """
    metric_matrices = []

    for layer in model.layers:
        if layer.metric is None:
            metric_matrices.append(None)
        else:
            metric_matrices.append(layer.metric.detach().clone())

    return metric_matrices


# =============================================================================
# Plotting helpers
# =============================================================================

def _plotting_names(config: Config) -> List[str]:
    """
    Return the requested plot names from config.plotting.

    Supported config forms
    ----------------------
    plotting: null
        No plots are created.

    plotting:
      names: null
        No plots are created.

    plotting:
      names:
        - mse_train
        - mse_val
        - m_matrices
        - m_eigvals
        Requested plots are created.

    The special names m_matrices and m_eigvals refer to layer metric matrices
    and their eigenvalues. All other names are interpreted as scalar metrics
    stored in the Logger.
    """
    plotting = config.get("plotting", None)

    if plotting is None:
        return []

    if isinstance(plotting, dict):
        names = plotting.get("names", None)
    else:
        names = plotting

    if names is None:
        return []

    if isinstance(names, str):
        names = [names]

    return [str(name) for name in names if name is not None]


def _plotting_option(config: Config, name: str, default):
    """
    Read an optional plotting setting from config.plotting.
    """
    plotting = config.get("plotting", None)

    if isinstance(plotting, dict):
        return plotting.get(name, default)

    return default


def _sanitize_filename(name: str) -> str:
    """
    Convert a metric name into a filesystem-friendly filename stem.
    """
    safe = []

    for char in str(name):
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")

    return "".join(safe)


def _as_scalar_float(value) -> Optional[float]:
    """
    Convert a logged scalar into a Python float.

    Returns None for non-scalar values such as matrices or weight vectors.
    """
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() != 1:
            return None
        return float(value.reshape(-1)[0])

    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return float(value.reshape(-1)[0])

    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)

    return None


def _plots_root(config: Config) -> str:
    """
    Return the runPlots directory for this experiment.
    """
    root_dir_name = _plotting_option(config, "root_dir", "runPlots")
    return os.path.join(config.experiment_dir, str(root_dir_name))


def _plot_scalar_metric_history(
    *,
    logger: Logger,
    config: Config,
    trial_idx: int,
    metric_name: str,
) -> None:
    """
    Plot one scalar metric history for one trial.

    The plot is overwritten at every epoch so the file always contains the
    latest history available for that trial.
    """
    if metric_name not in logger.data:
        print(f"[plotting] Skipping unknown metric {metric_name!r}.")
        return

    if trial_idx >= len(logger.data[metric_name]):
        return

    raw_values = logger.data[metric_name][trial_idx]
    values = []

    for value in raw_values:
        scalar = _as_scalar_float(value)
        if scalar is not None:
            values.append(scalar)

    if len(values) == 0:
        print(
            f"[plotting] Skipping metric {metric_name!r} because it does not "
            "contain scalar values."
        )
        return

    out_dir = os.path.join(_plots_root(config), "metricPlots")
    os.makedirs(out_dir, exist_ok=True)

    epochs = np.arange(len(values))
    dpi = int(_plotting_option(config, "dpi", 150))
    file_format = str(_plotting_option(config, "format", "png"))
    filename = (
        f"trial_{trial_idx:03d}_{_sanitize_filename(metric_name)}."
        f"{file_format}"
    )
    path = os.path.join(out_dir, filename)

    plt.figure()
    plt.plot(epochs, values, marker="o")
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} | trial {trial_idx:03d}")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _plot_metric_matrix_heatmaps(
    *,
    config: Config,
    trial_idx: int,
    epoch_idx: int,
    metric_matrices: List[Optional[torch.Tensor]],
) -> None:
    """
    Save one heatmap per layer metric matrix for the current epoch.
    """
    out_dir = os.path.join(_plots_root(config), "m_matrices")
    os.makedirs(out_dir, exist_ok=True)

    dpi = int(_plotting_option(config, "dpi", 150))
    file_format = str(_plotting_option(config, "format", "png"))

    for layer_idx, matrix in enumerate(metric_matrices):
        if matrix is None:
            continue

        matrix_np = matrix.detach().cpu().numpy()
        path = os.path.join(
            out_dir,
            (
                f"trial_{trial_idx:03d}_epoch_{epoch_idx:03d}_"
                f"layer_{layer_idx:02d}_m_matrix.{file_format}"
            ),
        )

        plt.figure()
        image = plt.imshow(matrix_np, aspect="auto")
        plt.colorbar(image)
        plt.xlabel("column")
        plt.ylabel("row")
        plt.title(
            f"M matrix | trial {trial_idx:03d} | "
            f"epoch {epoch_idx:03d} | layer {layer_idx:02d}"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()


def _plot_metric_matrix_eigenvalues(
    *,
    config: Config,
    trial_idx: int,
    epoch_idx: int,
    metric_matrices: List[Optional[torch.Tensor]],
) -> None:
    """
    Save one stem plot of metric eigenvalues per layer for the current epoch.
    """
    out_dir = os.path.join(_plots_root(config), "m_eigvals")
    os.makedirs(out_dir, exist_ok=True)

    dpi = int(_plotting_option(config, "dpi", 150))
    file_format = str(_plotting_option(config, "format", "png"))

    for layer_idx, matrix in enumerate(metric_matrices):
        if matrix is None:
            continue

        matrix = matrix.detach().cpu()
        matrix = 0.5 * (matrix + matrix.T)
        eigvals = torch.linalg.eigvalsh(matrix).flip(0).numpy()
        indices = np.arange(eigvals.shape[0])

        path = os.path.join(
            out_dir,
            (
                f"trial_{trial_idx:03d}_epoch_{epoch_idx:03d}_"
                f"layer_{layer_idx:02d}_m_eigvals.{file_format}"
            ),
        )

        plt.figure()
        plt.stem(indices, eigvals)
        plt.xlabel("eigenvalue index")
        plt.ylabel("eigenvalue")
        plt.title(
            f"M eigenvalues | trial {trial_idx:03d} | "
            f"epoch {epoch_idx:03d} | layer {layer_idx:02d}"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()


def make_epoch_plots(
    *,
    logger: Logger,
    config: Config,
    trial_idx: int,
    epoch_idx: int,
    metric_matrices: List[Optional[torch.Tensor]],
) -> None:
    """
    Create all configured plots for the current epoch.

    Special plot names
    ------------------
    m_matrices:
        Save heatmaps of each layer's metric matrix at the current epoch.

    m_eigvals:
        Save stem plots of each layer metric matrix's eigenvalues at the
        current epoch.

    Any other requested name is treated as a scalar logger metric and plotted
    as a history over logged epochs.
    """
    names = _plotting_names(config)

    if len(names) == 0:
        return

    requested = set(names)

    if "m_matrices" in requested:
        _plot_metric_matrix_heatmaps(
            config=config,
            trial_idx=trial_idx,
            epoch_idx=epoch_idx,
            metric_matrices=metric_matrices,
        )

    if "m_eigvals" in requested:
        _plot_metric_matrix_eigenvalues(
            config=config,
            trial_idx=trial_idx,
            epoch_idx=epoch_idx,
            metric_matrices=metric_matrices,
        )

    for metric_name in names:
        if metric_name in {"m_matrices", "m_eigvals"}:
            continue

        _plot_scalar_metric_history(
            logger=logger,
            config=config,
            trial_idx=trial_idx,
            metric_name=metric_name,
        )


# =============================================================================
# Metric computation
# =============================================================================

def compute_epoch_metrics(
    *,
    model: RPM,
    train_features: torch.Tensor,
    y_train: torch.Tensor,
    val_features: torch.Tensor,
    y_val: torch.Tensor,
    test_features: torch.Tensor,
    y_test: torch.Tensor,
    train_covariance: torch.Tensor,
    reg_param: float,
    metric_eps: float,
    effective_dim_eps: float,
    nmse_eps: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute all per-epoch logged metrics.

    All metrics here are implemented in util_refac.py except cond_num,
    which is logged separately because it is explicitly requested.
    """
    feature_dim = train_covariance.shape[0]

    eye = torch.eye(
        feature_dim,
        dtype=train_covariance.dtype,
        device=train_covariance.device,
    )

    train_covariance_inv = torch.linalg.inv(train_covariance + reg_param * eye)

    val_covariance = feature_covariance(val_features)
    test_covariance = feature_covariance(test_features)

    yhat_train = model.predict(train_features)
    yhat_val = model.predict(val_features)
    yhat_test = model.predict(test_features)

    metrics = {
        # MSE / NMSE.
        "mse_train": mse(yhat_train, y_train),
        "mse_val": mse(yhat_val, y_val),
        "mse_test": mse(yhat_test, y_test),
        "nmse_train": normalized_mse(yhat_train, y_train, eps=nmse_eps),
        "nmse_val": normalized_mse(yhat_val, y_val, eps=nmse_eps),
        "nmse_test": normalized_mse(yhat_test, y_test, eps=nmse_eps),

        # Target alignment.
        "target_align_train": target_alignment(
            train_features,
            y_train,
            train_features,
            y_train,
            train_covariance_inv=train_covariance_inv,
        ),
        "target_align_val": target_alignment(
            train_features,
            y_train,
            val_features,
            y_val,
            train_covariance_inv=train_covariance_inv,
        ),
        "target_align_test": target_alignment(
            train_features,
            y_train,
            test_features,
            y_test,
            train_covariance_inv=train_covariance_inv,
        ),

        # Excitation alignment.
        "excite_align_train": excitation_alignment(
            train_features,
            model.W,
            eval_covariance=train_covariance,
        ),
        "excite_align_val": excitation_alignment(
            val_features,
            model.W,
            eval_covariance=val_covariance,
        ),
        "excite_align_test": excitation_alignment(
            test_features,
            model.W,
            eval_covariance=test_covariance,
        ),

        # Fisher misalignment.
        # Val/test target moments are evaluated in the training covariance geometry.
        "fisher_arith_train": arithmetic_fisher(
            train_features,
            y_train,
            covariance=train_covariance,
            reg_param=reg_param,
            eps=metric_eps,
        ),
        "fisher_arith_val": arithmetic_fisher(
            val_features,
            y_val,
            covariance=train_covariance,
            reg_param=reg_param,
            eps=metric_eps,
        ),
        "fisher_arith_test": arithmetic_fisher(
            test_features,
            y_test,
            covariance=train_covariance,
            reg_param=reg_param,
            eps=metric_eps,
        ),
        "fisher_geom_train": geometric_fisher(
            train_features,
            y_train,
            covariance=train_covariance,
            reg_param=reg_param,
            eps=metric_eps,
        ),
        "fisher_geom_val": geometric_fisher(
            val_features,
            y_val,
            covariance=train_covariance,
            reg_param=reg_param,
            eps=metric_eps,
        ),
        "fisher_geom_test": geometric_fisher(
            test_features,
            y_test,
            covariance=train_covariance,
            reg_param=reg_param,
            eps=metric_eps,
        ),

        # Effective dimension.
        "eff_dim_train": covariance_effective_dim(
            train_covariance,
            eps=effective_dim_eps,
        ),
        "eff_dim_val": covariance_effective_dim(
            val_covariance,
            eps=effective_dim_eps,
        ),
        "eff_dim_test": covariance_effective_dim(
            test_covariance,
            eps=effective_dim_eps,
        ),
    }

    return metrics


def evaluate_current_geometry(
    *,
    logger: Logger,
    model: RPM,
    trial_idx: int,
    epoch_idx: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    config: Config,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate and log the current RPM geometry.

    This function does not update the RPM metric matrix. It only:
        1. sets centers for the current metric,
        2. builds spectral features,
        3. trains the ridge/MMSE readout in the current RKHS,
        4. computes the refactored metrics,
        5. logs the current layer metrics and readout weights.
    """
    model.set_centers(
        x_train,
        q_thresh=config.center_quant_thresh,
    )

    train_features = model.forward(x_train)
    val_features = model.forward(x_val)
    test_features = model.forward(
        x_test[:config.num_test_samples],
    )

    y_test_eval = y_test[:config.num_test_samples]

    train_covariance, _ = model.train_regressor(
        train_features,
        y_train,
        rcond=config.reg_param,
        onehot=config.use_onehot_labels,
        return_u=True,
    )

    metrics = compute_epoch_metrics(
        model=model,
        train_features=train_features,
        y_train=y_train,
        val_features=val_features,
        y_val=y_val,
        test_features=test_features,
        y_test=y_test_eval,
        train_covariance=train_covariance,
        reg_param=config.reg_param,
        metric_eps=config.metric_eps,
        effective_dim_eps=config.effective_dim_eps,
        nmse_eps=config.nmse_eps,
    )

    cond_num = torch.linalg.cond(train_covariance)
    metric_matrices = layer_metric_matrices(model)

    logger.log_epoch(
        epoch=epoch_idx,
        **metrics,
        metric_matrices=metric_matrices,
        weights=model.W.detach().clone(),
        cond_num=cond_num,
    )

    make_epoch_plots(
        logger=logger,
        config=config,
        trial_idx=trial_idx,
        epoch_idx=epoch_idx,
        metric_matrices=metric_matrices,
    )

    return metrics


# =============================================================================
# Training
# =============================================================================

def run_trial(
    *,
    trial_idx: int,
    logger: Logger,
    config: Config,
) -> None:
    """
    Run one RPM trial using the model-owned metric update rule.
    """
    validate_metric_update_config(config)
    x_train, y_train, x_val, y_val, x_test, y_test = make_mackey_glass_splits(
        config,
        seed=trial_idx,
    )

    model = make_rpm(
        input_dim=x_train.shape[1],
        config=config,
    )

    logger.start_trial(
        trial_idx=trial_idx,
        config=config,
        seed=trial_idx,
    )

    # ------------------------------------------------------------------
    # Explicit epoch 0 baseline
    # ------------------------------------------------------------------
    # make_rpm(...) only constructs the model. The baseline is evaluated here
    # as an explicit training-loop step.
    #
    # Since the model metric is still A = I and no AGOP update has occurred,
    # this is the Gaussian RKHS baseline:
    #     - build Gaussian RKHS features using A = I,
    #     - train the ridge/MMSE readout,
    #     - evaluate all metrics,
    #     - log epoch 0.
    metrics = evaluate_current_geometry(
        logger=logger,
        model=model,
        trial_idx=trial_idx,
        epoch_idx=0,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        config=config,
    )

    if config.enable_epoch_print:
        print(
            f"trial={trial_idx:03d} epoch=000 "
            f"mse_train={float(metrics['mse_train']):.6g} "
            f"mse_val={float(metrics['mse_val']):.6g} "
            f"mse_test={float(metrics['mse_test']):.6g}"
        )

    train_indices = np.arange(x_train.shape[0])

    for epoch_idx in range(1, config.train_num_epochs + 1):
        # Each epoch performs one RPM metric-learning update before logging.
        # Thus epoch 1 is the first post-update geometry.
        model.clear_pending_metric_updates()
        model.reset_agop()

        batches = np.random.choice(
            train_indices,
            size=(config.num_batches, config.batch_size),
            replace=True,
        )

        for batch_indices in batches:
            batch_indices = torch.as_tensor(
                batch_indices,
                dtype=torch.long,
                device=x_train.device,
            )

            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            model.set_centers(
                x_batch,
                q_thresh=config.center_quant_thresh,
            )

            batch_features = model.forward(x_batch)

            model.train_regressor(
                batch_features,
                y_batch,
                rcond=config.reg_param,
                onehot=config.use_onehot_labels,
                return_u=False,
            )

            # The RPM owns the update rule. This call computes and stores a
            # pending update; it does not mutate the metric matrices.
            model.backward(x_batch, y_batch)

        # Apply the pending updates accumulated by backward(...).
        model.update_metrics()

        metrics = evaluate_current_geometry(
            logger=logger,
            model=model,
            trial_idx=trial_idx,
            epoch_idx=epoch_idx,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            config=config,
        )

        if config.enable_epoch_print:
            print(
                f"trial={trial_idx:03d} epoch={epoch_idx:03d} "
                f"mse_train={float(metrics['mse_train']):.6g} "
                f"mse_val={float(metrics['mse_val']):.6g} "
                f"mse_test={float(metrics['mse_test']):.6g}"
            )

    logger.end_trial()


def run_experiment(config: Config) -> Logger:
    """
    Run all trials and save the logger after each trial.
    """
    _prepare_runtime_config(config)
    os.makedirs(config.experiment_dir, exist_ok=True)

    logger = Logger.from_config(
        config,
        metric_names=config.metric_names,
        strict=True,
        move_to_cpu=True,
    )

# %%
    logger.save_run_meta(

        os.path.join(
            config.experiment_dir,
            config.get("logger_run_meta_filename", "rpm_unified_run_meta.yaml"),
        )
    )

    for trial_idx in range(config.num_trials):
        print(f"TRIAL: {trial_idx}")
        run_trial(
            trial_idx=trial_idx,
            logger=logger,
            config=config,
        )
        logger.save()

    return logger


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config_agop.yaml"
    config = Config.from_file(config_path)
    run_experiment(config)
