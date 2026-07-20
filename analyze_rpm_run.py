#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-run RPM analysis utilities.

This script reconstructs an RPM at a requested logged epoch using:
    - metric_matrices stored by Logger
    - weights stored by Logger

It then rebuilds the spectral feature objects from the original training data
and can save plots for:
    1. one spectral eigenfunction/feature coordinate from any layer,
    2. GP prior samples induced by any layer's current kernel,
    3. GP posterior mean/std/samples induced by any layer's current kernel,
    4. frequency spectra of GP-prior samples,
    5. frequency spectrum of the desired signal.

Expected run artifacts
----------------------
The training script should have saved a Logger object containing at least:
    logger.data["metric_matrices"][trial][epoch]
    logger.data["weights"][trial][epoch]

The current unified training scripts satisfy this convention.

Example
-------
python analyze_rpm_run.py \
    --config mg_rpm_unified_step2_new_names_plots.yaml \
    --epoch 10 \
    --trial 0 \
    --layer 0 \
    --eigenfunction 3 \
    --plot-eigenfunction \
    --plot-prior \
    --plot-posterior \
    --num-points 200 \
    --gp-max-train 300
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure local package modules are loaded from this script's directory first.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _SCRIPT_DIR:
    try:
        sys.path.remove(_SCRIPT_DIR)
    except ValueError:
        pass
    sys.path.insert(0, _SCRIPT_DIR)

from config_refac import Config
from logger_refac import Logger
from device_utils import resolve_device


# =============================================================================
# Small helpers
# =============================================================================

@dataclass
class ReconstructedRun:
    config: Config
    logger: Logger
    model: object
    trial_idx: int
    epoch_idx: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize_filename(text: str) -> str:
    out = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out)


def _to_runtime_tensor(x, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _resolve_runtime_device(config: Config, device_override: Optional[str] = None) -> str:
    resolved_device = resolve_device(device_override or config.get("device", "auto"))
    config.device = resolved_device
    return resolved_device


def _resolve_logger_path(config: Config, logger_path: Optional[str]) -> str:
    if logger_path is not None:
        return logger_path

    filename = config.get("logger_filename", None)
    if filename is None:
        raise ValueError(
            "logger path was not supplied and config.logger_filename is missing. "
            "Pass --logger explicitly or add logger_filename to the config."
        )

    return os.path.join(config.experiment_dir, filename)


def _resolve_epoch_index(logger: Logger, trial_idx: int, epoch: int) -> int:
    if "metric_matrices" not in logger.data:
        raise KeyError("Logger does not contain metric_matrices.")

    num_epochs = len(logger.data["metric_matrices"][trial_idx])

    if epoch < 0:
        epoch = num_epochs + epoch

    if epoch < 0 or epoch >= num_epochs:
        raise IndexError(
            f"epoch={epoch} is out of range for trial {trial_idx}. "
            f"Available epoch indices are 0 to {num_epochs - 1}."
        )

    return int(epoch)


def _select_split(
    run: ReconstructedRun,
    split: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    split = split.lower()

    if split == "train":
        return run.x_train, run.y_train
    if split == "val":
        return run.x_val, run.y_val
    if split == "test":
        return run.x_test, run.y_test

    raise ValueError("split must be one of {'train', 'val', 'test'}.")


def _slice_points(
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    *,
    start: int,
    num_points: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray]:
    n = x.shape[0]
    start = int(start)

    if start < 0 or start >= n:
        raise IndexError(f"start={start} is out of range for {n} points.")

    if num_points is None or num_points <= 0:
        stop = n
    else:
        stop = min(n, start + int(num_points))

    x_sel = x[start:stop]
    y_sel = None if y is None else y[start:stop]
    point_indices = np.arange(start, stop)

    return x_sel, y_sel, point_indices


def _layer_input(model, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Return the input representation entering layer_idx.

    layer_idx = 0 returns the original x.
    layer_idx = 1 returns layer 0 features, etc.
    """
    if layer_idx < 0:
        layer_idx = len(model.layers) + layer_idx

    if layer_idx < 0 or layer_idx >= len(model.layers):
        raise IndexError(
            f"layer_idx={layer_idx} is out of range for {len(model.layers)} layers."
        )

    z = x
    normalize = model.normalize

    for idx in range(layer_idx):
        z = model.layers[idx].get_features(z, normalize=normalize)

    return z


def _layer_features(model, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Evaluate all spectral features/eigenfunctions of one layer at x.
    """
    z = _layer_input(model, x, layer_idx)
    layer = model.layers[layer_idx]
    return layer.get_features(z, normalize=model.normalize)


def _kernel_matrix(layer, x: torch.Tensor, y: torch.Tensor, *, chunksize: int) -> torch.Tensor:
    return layer.kernel(
        x,
        y,
        gram=True,
        chunksize=chunksize,
    )


def _safe_cholesky(matrix: torch.Tensor, *, jitter: float = 1e-8, max_tries: int = 8):
    """
    Cholesky with increasing jitter. Falls back to eigenvalue projection.
    """
    matrix = 0.5 * (matrix + matrix.T)
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)

    for attempt in range(max_tries):
        try:
            amount = float(jitter) * (10.0 ** attempt)
            return torch.linalg.cholesky(matrix + amount * eye)
        except torch._C._LinAlgError:
            continue
        except RuntimeError:
            continue

    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=float(jitter))
    projected = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
    projected = 0.5 * (projected + projected.T)
    return torch.linalg.cholesky(projected + float(jitter) * eye)


def _sample_gaussian(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    *,
    num_samples: int,
    jitter: float,
) -> torch.Tensor:
    """
    Draw samples from N(mean, covariance). Returns shape (num_samples, N).
    """
    if num_samples <= 0:
        return torch.empty(0, mean.numel(), device=mean.device, dtype=mean.dtype)

    mean = mean.reshape(-1)
    covariance = 0.5 * (covariance + covariance.T)
    L = _safe_cholesky(covariance, jitter=jitter)
    noise = torch.randn(
        num_samples,
        mean.numel(),
        device=mean.device,
        dtype=mean.dtype,
    )
    return mean.unsqueeze(0) + noise @ L.T


def _subsample_training_points(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    max_train: Optional[int],
    mode: str,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_train is None or max_train <= 0 or max_train >= x_train.shape[0]:
        return x_train, y_train

    max_train = int(max_train)
    mode = str(mode).lower()

    if mode == "first":
        indices = torch.arange(max_train, device=x_train.device)
    elif mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices_cpu = torch.randperm(x_train.shape[0], generator=generator)[:max_train]
        indices = indices_cpu.to(x_train.device)
    else:
        raise ValueError("gp_train_mode must be one of {'first', 'random'}.")

    return x_train[indices], y_train[indices]


# =============================================================================
# Reconstruction
# =============================================================================

def load_reconstructed_run(
    *,
    config_path: Optional[str],
    logger_path: Optional[str],
    trial_idx: int,
    epoch: int,
    device_override: Optional[str] = None,
) -> ReconstructedRun:
    """
    Reconstruct the RPM at one logged epoch.
    """
    if config_path is not None:
        config = Config.from_file(config_path)
        _resolve_runtime_device(config, device_override=device_override)
        resolved_logger_path = _resolve_logger_path(config, logger_path)
        logger = Logger.load(resolved_logger_path, map_location="cpu")
    else:
        if logger_path is None:
            raise ValueError("Either --config or --logger must be supplied.")
        logger = Logger.load(logger_path, map_location="cpu")
        config = Config.from_dict(logger.run_meta)
        resolved_logger_path = logger_path

    if device_override is not None:
        _resolve_runtime_device(config, device_override=device_override)
    else:
        _resolve_runtime_device(config)

    if trial_idx < 0 or trial_idx >= len(logger.data["metric_matrices"]):
        raise IndexError(
            f"trial={trial_idx} is out of range. "
            f"Logger has {len(logger.data['metric_matrices'])} trials."
        )

    epoch_idx = _resolve_epoch_index(logger, trial_idx, epoch)

    # Import lazily so `python analyze_rpm_run.py --help` does not require
    # the dataset dependencies used by dspTools.
    from train_rpm import make_mackey_glass_splits, make_rpm

    x_train, y_train, x_val, y_val, x_test, y_test = make_mackey_glass_splits(
        config,
        seed=trial_idx,
    )

    model = make_rpm(input_dim=x_train.shape[1], config=config)

    device = torch.device(config.device)
    dtype = config.dtype

    metric_matrices = logger.data["metric_matrices"][trial_idx][epoch_idx]
    weights = logger.data["weights"][trial_idx][epoch_idx]

    if len(metric_matrices) != len(model.layers):
        raise RuntimeError(
            f"Logger epoch has {len(metric_matrices)} metric matrices, but "
            f"model has {len(model.layers)} layers. Check config/model mismatch."
        )

    for layer, metric in zip(model.layers, metric_matrices):
        if metric is None:
            layer.metric = None
        else:
            layer.metric = _to_runtime_tensor(metric, device=device, dtype=dtype)

    # Build spectral objects using the loaded metrics. This is required before
    # eigenfunction evaluation and layer-wise GP analysis.
    model.set_centers(
        x_train,
        q_thresh=config.center_quant_thresh,
    )

    model.W = _to_runtime_tensor(weights, device=device, dtype=dtype)

    print(
        f"Loaded run from {resolved_logger_path}\n"
        f"  trial={trial_idx}\n"
        f"  epoch={epoch_idx}\n"
        f"  layers={len(model.layers)}"
    )

    return ReconstructedRun(
        config=config,
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
    )


# =============================================================================
# Plotting functions
# =============================================================================

def plot_layer_eigenfunction(
    run: ReconstructedRun,
    *,
    layer_idx: int,
    eigenfunction_idx: int,
    split: str,
    start: int,
    num_points: Optional[int],
    out_dir: str,
    show_target: bool = False,
) -> str:
    """
    Plot one spectral feature/eigenfunction coordinate from one layer.
    """
    x, y = _select_split(run, split)
    x_eval, y_eval, point_indices = _slice_points(
        x,
        y,
        start=start,
        num_points=num_points,
    )

    features = _layer_features(run.model, x_eval, layer_idx)

    if eigenfunction_idx < 0:
        eigenfunction_idx = features.shape[1] + eigenfunction_idx

    if eigenfunction_idx < 0 or eigenfunction_idx >= features.shape[1]:
        raise IndexError(
            f"eigenfunction={eigenfunction_idx} is out of range. "
            f"Layer {layer_idx} has {features.shape[1]} features."
        )

    values = features[:, eigenfunction_idx].detach().cpu().numpy()

    _ensure_dir(out_dir)
    filename = (
        f"trial_{run.trial_idx:03d}_epoch_{run.epoch_idx:03d}_"
        f"layer_{layer_idx:02d}_eigfunc_{eigenfunction_idx:04d}_"
        f"{_sanitize_filename(split)}.png"
    )
    path = os.path.join(out_dir, filename)

    plt.figure()
    plt.plot(point_indices, values, marker="o", markersize=3)

    if show_target and y_eval is not None:
        target = y_eval.detach().cpu().numpy().reshape(-1)
        target_std = np.std(target)
        value_std = np.std(values)
        if target_std > 0 and value_std > 0:
            target_scaled = (target - np.mean(target)) / target_std
            target_scaled = target_scaled * value_std + np.mean(values)
            plt.plot(point_indices, target_scaled, linestyle="--")

    plt.xlabel(f"{split} sample index")
    plt.ylabel(f"layer {layer_idx} eigenfunction {eigenfunction_idx}")
    plt.title(
        f"Layer {layer_idx} eigenfunction {eigenfunction_idx} | "
        f"trial {run.trial_idx} epoch {run.epoch_idx}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_layer_gp_prior(
    run: ReconstructedRun,
    *,
    layer_idx: int,
    split: str,
    start: int,
    num_points: Optional[int],
    num_samples: int,
    jitter: float,
    out_dir: str,
) -> str:
    """
    Plot GP prior samples induced by one layer's kernel at evaluation points.
    """
    x, y = _select_split(run, split)
    x_eval, _, point_indices = _slice_points(
        x,
        y,
        start=start,
        num_points=num_points,
    )

    layer = run.model.layers[layer_idx]
    z_eval = _layer_input(run.model, x_eval, layer_idx)

    K_eval = _kernel_matrix(
        layer,
        z_eval,
        z_eval,
        chunksize=int(run.config.chunk_size),
    )

    mean = torch.zeros(K_eval.shape[0], device=K_eval.device, dtype=K_eval.dtype)
    samples = _sample_gaussian(
        mean,
        K_eval,
        num_samples=num_samples,
        jitter=jitter,
    ).detach().cpu().numpy()

    _ensure_dir(out_dir)
    filename = (
        f"trial_{run.trial_idx:03d}_epoch_{run.epoch_idx:03d}_"
        f"layer_{layer_idx:02d}_gp_prior_{_sanitize_filename(split)}.png"
    )
    path = os.path.join(out_dir, filename)

    plt.figure()
    for sample_idx in range(samples.shape[0]):
        plt.plot(point_indices, samples[sample_idx], linewidth=1.0)

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel(f"{split} sample index")
    plt.ylabel("f(x)")
    plt.title(
        f"GP prior samples | layer {layer_idx} | "
        f"trial {run.trial_idx} epoch {run.epoch_idx}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path



def _fft_spectrum(
    values: np.ndarray,
    *,
    sample_spacing: float = 1.0,
    subtract_mean: bool = True,
    window: Optional[str] = None,
    scale: str = "amplitude",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a one-sided FFT spectrum for a real-valued sequence.

    Parameters
    ----------
    values:
        One-dimensional signal values.

    sample_spacing:
        Spacing between adjacent samples. The default gives normalized
        frequency in cycles/sample.

    subtract_mean:
        If True, remove the DC mean before computing the FFT.

    window:
        Optional window name. Currently supports None, "none", and "hann".

    scale:
        "amplitude", "power", or "db". The "db" scale uses amplitude dB.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    n = values.size

    if n < 2:
        raise ValueError("FFT plots require at least two points.")

    if subtract_mean:
        values = values - np.mean(values)

    if window is not None:
        window_name = str(window).lower()
    else:
        window_name = "none"

    if window_name in {"none", "null", "false"}:
        pass
    elif window_name in {"hann", "hanning"}:
        values = values * np.hanning(n)
    else:
        raise ValueError("window must be one of {None, 'none', 'hann'}.")

    freqs = np.fft.rfftfreq(n, d=float(sample_spacing))
    spectrum = np.abs(np.fft.rfft(values)) / float(n)

    scale = str(scale).lower()
    if scale == "amplitude":
        plotted = spectrum
    elif scale == "power":
        plotted = spectrum ** 2
    elif scale in {"db", "amplitude_db"}:
        plotted = 20.0 * np.log10(spectrum + float(eps))
    else:
        raise ValueError("scale must be one of {'amplitude', 'power', 'db'}.")

    return freqs, plotted


def _spectrum_ylabel(scale: str) -> str:
    scale = str(scale).lower()
    if scale == "amplitude":
        return "FFT amplitude"
    if scale == "power":
        return "FFT power"
    if scale in {"db", "amplitude_db"}:
        return "FFT amplitude (dB)"
    return "FFT magnitude"


def plot_layer_gp_prior_frequency(
    run: ReconstructedRun,
    *,
    layer_idx: int,
    split: str,
    start: int,
    num_points: Optional[int],
    num_samples: int,
    jitter: float,
    out_dir: str,
    sample_spacing: float = 1.0,
    subtract_mean: bool = True,
    window: Optional[str] = None,
    scale: str = "amplitude",
    average_across_samples: bool = True,
    plot_individual_samples: bool = False,
) -> str:
    """
    Plot the frequency spectrum of GP prior samples from one layer's kernel.

    By default this computes one FFT spectrum per sampled prior realization,
    averages those spectra across samples, and plots only the mean spectrum.
    This is usually easier to interpret than overlaying many individual prior
    spectra.

    Notes
    -----
    The averaging is performed after applying the requested ``scale``. Thus for
    ``scale='db'``, the plotted curve is the average dB spectrum across prior
    samples. For ``scale='amplitude'`` or ``scale='power'``, it is the average
    amplitude or power spectrum.
    """
    x, y = _select_split(run, split)
    x_eval, _, _ = _slice_points(
        x,
        y,
        start=start,
        num_points=num_points,
    )

    layer = run.model.layers[layer_idx]
    z_eval = _layer_input(run.model, x_eval, layer_idx)

    K_eval = _kernel_matrix(
        layer,
        z_eval,
        z_eval,
        chunksize=int(run.config.chunk_size),
    )

    mean = torch.zeros(K_eval.shape[0], device=K_eval.device, dtype=K_eval.dtype)
    samples = _sample_gaussian(
        mean,
        K_eval,
        num_samples=num_samples,
        jitter=jitter,
    ).detach().cpu().numpy()

    spectra = []
    freqs = None
    for sample_idx in range(samples.shape[0]):
        freqs_i, spectrum_i = _fft_spectrum(
            samples[sample_idx],
            sample_spacing=sample_spacing,
            subtract_mean=subtract_mean,
            window=window,
            scale=scale,
        )

        if freqs is None:
            freqs = freqs_i

        spectra.append(spectrum_i)

    spectra = np.stack(spectra, axis=0)
    mean_spectrum = np.mean(spectra, axis=0)

    _ensure_dir(out_dir)
    filename = (
        f"trial_{run.trial_idx:03d}_epoch_{run.epoch_idx:03d}_"
        f"layer_{layer_idx:02d}_gp_prior_frequency_mean_"
        f"{_sanitize_filename(split)}.png"
    )
    path = os.path.join(out_dir, filename)

    plt.figure()

    if plot_individual_samples:
        for sample_idx in range(spectra.shape[0]):
            plt.plot(freqs, spectra[sample_idx], linewidth=0.7, alpha=0.35)

    if average_across_samples:
        plt.plot(freqs, mean_spectrum, linewidth=2.0)
    else:
        # Backward-compatible behavior: plot individual spectra only.
        for sample_idx in range(spectra.shape[0]):
            plt.plot(freqs, spectra[sample_idx], linewidth=1.0)

    plt.xlabel("Frequency (cycles/sample)")
    plt.ylabel(_spectrum_ylabel(scale))

    if average_across_samples:
        title_prefix = f"Mean GP prior frequency over {num_samples} samples"
    else:
        title_prefix = "GP prior frequency samples"

    plt.title(
        f"{title_prefix} | layer {layer_idx} | "
        f"trial {run.trial_idx} epoch {run.epoch_idx}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_desired_signal_frequency(
    run: ReconstructedRun,
    *,
    split: str,
    start: int,
    num_points: Optional[int],
    out_dir: str,
    sample_spacing: float = 1.0,
    subtract_mean: bool = True,
    window: Optional[str] = None,
    scale: str = "amplitude",
) -> str:
    """
    Plot the frequency spectrum of the desired signal y on a selected split.
    """
    x, y = _select_split(run, split)
    if y is None:
        raise RuntimeError(f"Selected split {split!r} has no target signal.")

    _, y_eval, _ = _slice_points(
        x,
        y,
        start=start,
        num_points=num_points,
    )

    y_np = y_eval.detach().cpu().numpy().reshape(-1)
    freqs, spectrum = _fft_spectrum(
        y_np,
        sample_spacing=sample_spacing,
        subtract_mean=subtract_mean,
        window=window,
        scale=scale,
    )

    _ensure_dir(out_dir)
    filename = (
        f"trial_{run.trial_idx:03d}_epoch_{run.epoch_idx:03d}_"
        f"desired_frequency_{_sanitize_filename(split)}.png"
    )
    path = os.path.join(out_dir, filename)

    plt.figure()
    plt.plot(freqs, spectrum, linewidth=1.5)
    plt.xlabel("Frequency (cycles/sample)")
    plt.ylabel(_spectrum_ylabel(scale))
    plt.title(
        f"Desired signal frequency | {split} | "
        f"trial {run.trial_idx} epoch {run.epoch_idx}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_layer_gp_posterior(
    run: ReconstructedRun,
    *,
    layer_idx: int,
    split: str,
    start: int,
    num_points: Optional[int],
    noise_var: float,
    gp_max_train: Optional[int],
    gp_train_mode: str,
    gp_seed: int,
    num_samples: int,
    jitter: float,
    out_dir: str,
    show_eval_targets: bool = True,
) -> str:
    """
    Plot GP posterior mean/std induced by one layer's kernel.

    The posterior uses training targets y_train as observations and the layer's
    current kernel as the GP covariance.
    """
    x_eval_all, y_eval_all = _select_split(run, split)
    x_eval, y_eval, point_indices = _slice_points(
        x_eval_all,
        y_eval_all,
        start=start,
        num_points=num_points,
    )

    x_gp_train, y_gp_train = _subsample_training_points(
        run.x_train,
        run.y_train,
        max_train=gp_max_train,
        mode=gp_train_mode,
        seed=gp_seed,
    )

    layer = run.model.layers[layer_idx]

    z_train = _layer_input(run.model, x_gp_train, layer_idx)
    z_eval = _layer_input(run.model, x_eval, layer_idx)

    chunksize = int(run.config.chunk_size)

    K_tt = _kernel_matrix(layer, z_train, z_train, chunksize=chunksize)
    K_te = _kernel_matrix(layer, z_train, z_eval, chunksize=chunksize)
    K_ee = _kernel_matrix(layer, z_eval, z_eval, chunksize=chunksize)

    n_train = K_tt.shape[0]
    eye = torch.eye(n_train, device=K_tt.device, dtype=K_tt.dtype)
    L = _safe_cholesky(K_tt + float(noise_var) * eye, jitter=jitter)

    y_vec = y_gp_train.reshape(-1, 1).to(device=K_tt.device, dtype=K_tt.dtype)
    alpha = torch.cholesky_solve(y_vec, L)

    posterior_mean = (K_te.T @ alpha).reshape(-1)
    v = torch.cholesky_solve(K_te, L)
    posterior_cov = K_ee - K_te.T @ v
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)

    posterior_var = torch.clamp(torch.diag(posterior_cov), min=0.0)
    posterior_std = torch.sqrt(posterior_var)

    posterior_samples = _sample_gaussian(
        posterior_mean,
        posterior_cov,
        num_samples=num_samples,
        jitter=jitter,
    )

    mean_np = posterior_mean.detach().cpu().numpy()
    std_np = posterior_std.detach().cpu().numpy()
    samples_np = posterior_samples.detach().cpu().numpy()

    _ensure_dir(out_dir)
    filename = (
        f"trial_{run.trial_idx:03d}_epoch_{run.epoch_idx:03d}_"
        f"layer_{layer_idx:02d}_gp_posterior_{_sanitize_filename(split)}.png"
    )
    path = os.path.join(out_dir, filename)

    plt.figure()
    plt.plot(point_indices, mean_np, linewidth=2.0, label="posterior mean")
    plt.fill_between(
        point_indices,
        mean_np - 2.0 * std_np,
        mean_np + 2.0 * std_np,
        alpha=0.25,
        label="±2 std",
    )

    for sample_idx in range(samples_np.shape[0]):
        plt.plot(point_indices, samples_np[sample_idx], linewidth=1.0, alpha=0.75)

    if show_eval_targets and y_eval is not None:
        y_np = y_eval.detach().cpu().numpy().reshape(-1)
        plt.scatter(point_indices, y_np, s=12, label=f"{split} targets")

    plt.xlabel(f"{split} sample index")
    plt.ylabel("f(x)")
    plt.title(
        f"GP posterior | layer {layer_idx} | "
        f"trial {run.trial_idx} epoch {run.epoch_idx} | "
        f"train N={n_train}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path



# =============================================================================
# YAML-driven analysis entry point
# =============================================================================

def _cfg_get(config: Config, name: str, default=None):
    """Small wrapper so the analysis YAML can stay flexible."""
    return config.get(name, default)


def _cfg_section(config: Config, name: str) -> dict:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise TypeError(f"{name} must be a mapping/dict in the analysis YAML.")
    return section


def _section_get(section: dict, name: str, default=None):
    return section.get(name, default)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load_analysis_config(path: str) -> Config:
    """
    Load the YAML file that controls post-run analysis.

    This is intentionally the same style as the training scripts:

        config_path = sys.argv[1] if provided else ./analyze_rpm_run_config.yaml
        config = Config.from_file(config_path)
        run_analysis(config)
    """
    return Config.from_file(path)


def _resolve_analysis_output_root(run: ReconstructedRun, analysis_config: Config) -> str:
    out_dir = analysis_config.get("output_root", None)
    if out_dir is None:
        out_dir = os.path.join(run.config.experiment_dir, "postRunAnalysis")
    return out_dir


def run_analysis(analysis_config: Config) -> List[str]:
    """
    Run post-run model analysis from a YAML config.

    Required analysis YAML keys
    ---------------------------
    training_config_path : str
        The YAML config used by the training script. This is used to rebuild
        the RPM architecture and recreate the train/val/test splits.

    Optional keys
    -------------
    logger_path : str or null
        If null, the logger is loaded from
        training_config.experiment_dir / training_config.logger_filename.

    trial_idx : int
    epoch_idx : int
        Epoch may be negative; -1 means the last logged epoch.

    device : str or null
        Optional device override, usually "cpu" for post-run analysis.

    output_root : str or null
        If null, outputs go to <experiment_dir>/postRunAnalysis.

    plots : dict
        Contains optional sections: eigenfunctions, gp_prior, gp_prior_frequency, desired_frequency, gp_posterior.
    """
    training_config_path = analysis_config.get("training_config_path", None)
    if training_config_path is None:
        # Alias that reads naturally in some project configs.
        training_config_path = analysis_config.get("run_config_path", None)

    if training_config_path is None:
        raise ValueError(
            "analysis YAML must define training_config_path, pointing to the "
            "YAML used by the training script."
        )

    logger_path = analysis_config.get("logger_path", None)
    trial_idx = int(analysis_config.get("trial_idx", 0))
    epoch_idx_requested = int(analysis_config.get("epoch_idx", analysis_config.get("epoch", -1)))
    device_override = analysis_config.get("device", None)

    run = load_reconstructed_run(
        config_path=training_config_path,
        logger_path=logger_path,
        trial_idx=trial_idx,
        epoch=epoch_idx_requested,
        device_override=device_override,
    )

    out_root = _resolve_analysis_output_root(run, analysis_config)
    epoch_root = _ensure_dir(
        os.path.join(
            out_root,
            f"trial_{run.trial_idx:03d}_epoch_{run.epoch_idx:03d}",
        )
    )

    plots = _cfg_section(analysis_config, "plots")
    saved_paths: List[str] = []

    # ------------------------------------------------------------------
    # Eigenfunction plots
    # ------------------------------------------------------------------
    eig_cfg = _section_get(plots, "eigenfunctions", None)
    if eig_cfg is not None:
        if not isinstance(eig_cfg, dict):
            raise TypeError("plots.eigenfunctions must be a mapping/dict.")

        if bool(eig_cfg.get("enabled", True)):
            layer_idx = int(eig_cfg.get("layer_idx", 0))
            split = eig_cfg.get("split", "test")
            start = int(eig_cfg.get("start", 0))
            num_points = eig_cfg.get("num_points", 200)
            if num_points is not None:
                num_points = int(num_points)
            show_target = bool(eig_cfg.get("show_target", False))

            eigenfunction_indices = _as_list(
                eig_cfg.get(
                    "eigenfunction_indices",
                    eig_cfg.get("eigenfunction_idx", eig_cfg.get("index", 0)),
                )
            )

            out_dir = _ensure_dir(os.path.join(epoch_root, "eigenfunctions"))
            for eig_idx in eigenfunction_indices:
                path = plot_layer_eigenfunction(
                    run,
                    layer_idx=layer_idx,
                    eigenfunction_idx=int(eig_idx),
                    split=split,
                    start=start,
                    num_points=num_points,
                    out_dir=out_dir,
                    show_target=show_target,
                )
                saved_paths.append(path)

    # ------------------------------------------------------------------
    # GP prior plots
    # ------------------------------------------------------------------
    prior_cfg = _section_get(plots, "gp_prior", None)
    if prior_cfg is not None:
        if not isinstance(prior_cfg, dict):
            raise TypeError("plots.gp_prior must be a mapping/dict.")

        if bool(prior_cfg.get("enabled", True)):
            layer_idx = int(prior_cfg.get("layer_idx", 0))
            split = prior_cfg.get("split", "test")
            start = int(prior_cfg.get("start", 0))
            num_points = prior_cfg.get("num_points", 200)
            if num_points is not None:
                num_points = int(num_points)
            num_samples = int(prior_cfg.get("num_samples", 5))
            jitter = float(prior_cfg.get("jitter", analysis_config.get("gp_jitter", 1e-8)))

            path = plot_layer_gp_prior(
                run,
                layer_idx=layer_idx,
                split=split,
                start=start,
                num_points=num_points,
                num_samples=num_samples,
                jitter=jitter,
                out_dir=_ensure_dir(os.path.join(epoch_root, "gp_prior")),
            )
            saved_paths.append(path)

    # ------------------------------------------------------------------
    # GP prior frequency-domain plots
    # ------------------------------------------------------------------
    prior_freq_cfg = _section_get(plots, "gp_prior_frequency", None)
    if prior_freq_cfg is not None:
        if not isinstance(prior_freq_cfg, dict):
            raise TypeError("plots.gp_prior_frequency must be a mapping/dict.")

        if bool(prior_freq_cfg.get("enabled", True)):
            layer_idx = int(prior_freq_cfg.get("layer_idx", 0))
            split = prior_freq_cfg.get("split", "test")
            start = int(prior_freq_cfg.get("start", 0))
            num_points = prior_freq_cfg.get("num_points", 200)
            if num_points is not None:
                num_points = int(num_points)
            num_samples = int(prior_freq_cfg.get("num_samples", 5))
            jitter = float(prior_freq_cfg.get("jitter", analysis_config.get("gp_jitter", 1e-8)))

            path = plot_layer_gp_prior_frequency(
                run,
                layer_idx=layer_idx,
                split=split,
                start=start,
                num_points=num_points,
                num_samples=num_samples,
                jitter=jitter,
                out_dir=_ensure_dir(os.path.join(epoch_root, "gp_prior_frequency")),
                sample_spacing=float(prior_freq_cfg.get("sample_spacing", 1.0)),
                subtract_mean=bool(prior_freq_cfg.get("subtract_mean", True)),
                window=prior_freq_cfg.get("window", None),
                scale=prior_freq_cfg.get("scale", "amplitude"),
                average_across_samples=bool(
                    prior_freq_cfg.get("average_across_samples", True)
                ),
                plot_individual_samples=bool(
                    prior_freq_cfg.get("plot_individual_samples", False)
                ),
            )
            saved_paths.append(path)

    # ------------------------------------------------------------------
    # Desired-signal frequency-domain plots
    # ------------------------------------------------------------------
    desired_freq_cfg = _section_get(plots, "desired_frequency", None)
    if desired_freq_cfg is not None:
        if not isinstance(desired_freq_cfg, dict):
            raise TypeError("plots.desired_frequency must be a mapping/dict.")

        if bool(desired_freq_cfg.get("enabled", True)):
            split = desired_freq_cfg.get("split", "test")
            start = int(desired_freq_cfg.get("start", 0))
            num_points = desired_freq_cfg.get("num_points", 200)
            if num_points is not None:
                num_points = int(num_points)

            path = plot_desired_signal_frequency(
                run,
                split=split,
                start=start,
                num_points=num_points,
                out_dir=_ensure_dir(os.path.join(epoch_root, "desired_frequency")),
                sample_spacing=float(desired_freq_cfg.get("sample_spacing", 1.0)),
                subtract_mean=bool(desired_freq_cfg.get("subtract_mean", True)),
                window=desired_freq_cfg.get("window", None),
                scale=desired_freq_cfg.get("scale", "amplitude"),
            )
            saved_paths.append(path)

    # ------------------------------------------------------------------
    # GP posterior plots
    # ------------------------------------------------------------------
    posterior_cfg = _section_get(plots, "gp_posterior", None)
    if posterior_cfg is not None:
        if not isinstance(posterior_cfg, dict):
            raise TypeError("plots.gp_posterior must be a mapping/dict.")

        if bool(posterior_cfg.get("enabled", True)):
            layer_idx = int(posterior_cfg.get("layer_idx", 0))
            split = posterior_cfg.get("split", "test")
            start = int(posterior_cfg.get("start", 0))
            num_points = posterior_cfg.get("num_points", 200)
            if num_points is not None:
                num_points = int(num_points)

            gp_max_train = posterior_cfg.get("gp_max_train", 300)
            if gp_max_train is not None:
                gp_max_train = int(gp_max_train)
                if gp_max_train <= 0:
                    gp_max_train = None

            path = plot_layer_gp_posterior(
                run,
                layer_idx=layer_idx,
                split=split,
                start=start,
                num_points=num_points,
                noise_var=float(posterior_cfg.get("noise_var", 1e-6)),
                gp_max_train=gp_max_train,
                gp_train_mode=posterior_cfg.get("gp_train_mode", "first"),
                gp_seed=int(posterior_cfg.get("gp_seed", 0)),
                num_samples=int(posterior_cfg.get("num_samples", 0)),
                jitter=float(posterior_cfg.get("jitter", analysis_config.get("gp_jitter", 1e-8))),
                out_dir=_ensure_dir(os.path.join(epoch_root, "gp_posterior")),
                show_eval_targets=bool(posterior_cfg.get("show_eval_targets", True)),
            )
            saved_paths.append(path)

    # ------------------------------------------------------------------
    # Optional generic task list for more than one layer/plot type.
    # This lets you run several analyses from one YAML without adding new
    # top-level sections.
    # ------------------------------------------------------------------
    tasks = analysis_config.get("plot_tasks", None)
    if tasks is not None:
        if not isinstance(tasks, list):
            raise TypeError("plot_tasks must be a list of dictionaries.")

        for task_idx, task in enumerate(tasks):
            if not isinstance(task, dict):
                raise TypeError(f"plot_tasks[{task_idx}] must be a dictionary.")

            if not bool(task.get("enabled", True)):
                continue

            plot_type = str(task.get("type", task.get("plot_type", ""))).lower()
            layer_idx = int(task.get("layer_idx", 0))
            split = task.get("split", "test")
            start = int(task.get("start", 0))
            num_points = task.get("num_points", 200)
            if num_points is not None:
                num_points = int(num_points)

            if plot_type in {"eigenfunction", "eigenfunctions", "eigfunc"}:
                eig_indices = _as_list(task.get("eigenfunction_indices", task.get("eigenfunction_idx", 0)))
                out_dir = _ensure_dir(os.path.join(epoch_root, "eigenfunctions"))
                for eig_idx in eig_indices:
                    saved_paths.append(
                        plot_layer_eigenfunction(
                            run,
                            layer_idx=layer_idx,
                            eigenfunction_idx=int(eig_idx),
                            split=split,
                            start=start,
                            num_points=num_points,
                            out_dir=out_dir,
                            show_target=bool(task.get("show_target", False)),
                        )
                    )

            elif plot_type in {"gp_prior", "prior"}:
                saved_paths.append(
                    plot_layer_gp_prior(
                        run,
                        layer_idx=layer_idx,
                        split=split,
                        start=start,
                        num_points=num_points,
                        num_samples=int(task.get("num_samples", 5)),
                        jitter=float(task.get("jitter", analysis_config.get("gp_jitter", 1e-8))),
                        out_dir=_ensure_dir(os.path.join(epoch_root, "gp_prior")),
                    )
                )

            elif plot_type in {"gp_prior_frequency", "prior_frequency", "gp_prior_fft", "prior_fft"}:
                saved_paths.append(
                    plot_layer_gp_prior_frequency(
                        run,
                        layer_idx=layer_idx,
                        split=split,
                        start=start,
                        num_points=num_points,
                        num_samples=int(task.get("num_samples", 5)),
                        jitter=float(task.get("jitter", analysis_config.get("gp_jitter", 1e-8))),
                        out_dir=_ensure_dir(os.path.join(epoch_root, "gp_prior_frequency")),
                        sample_spacing=float(task.get("sample_spacing", 1.0)),
                        subtract_mean=bool(task.get("subtract_mean", True)),
                        window=task.get("window", None),
                        scale=task.get("scale", "amplitude"),
                        average_across_samples=bool(
                            task.get("average_across_samples", True)
                        ),
                        plot_individual_samples=bool(
                            task.get("plot_individual_samples", False)
                        ),
                    )
                )

            elif plot_type in {"desired_frequency", "desired_fft", "target_frequency", "target_fft"}:
                saved_paths.append(
                    plot_desired_signal_frequency(
                        run,
                        split=split,
                        start=start,
                        num_points=num_points,
                        out_dir=_ensure_dir(os.path.join(epoch_root, "desired_frequency")),
                        sample_spacing=float(task.get("sample_spacing", 1.0)),
                        subtract_mean=bool(task.get("subtract_mean", True)),
                        window=task.get("window", None),
                        scale=task.get("scale", "amplitude"),
                    )
                )

            elif plot_type in {"gp_posterior", "posterior"}:
                gp_max_train = task.get("gp_max_train", 300)
                if gp_max_train is not None:
                    gp_max_train = int(gp_max_train)
                    if gp_max_train <= 0:
                        gp_max_train = None

                saved_paths.append(
                    plot_layer_gp_posterior(
                        run,
                        layer_idx=layer_idx,
                        split=split,
                        start=start,
                        num_points=num_points,
                        noise_var=float(task.get("noise_var", 1e-6)),
                        gp_max_train=gp_max_train,
                        gp_train_mode=task.get("gp_train_mode", "first"),
                        gp_seed=int(task.get("gp_seed", 0)),
                        num_samples=int(task.get("num_samples", 0)),
                        jitter=float(task.get("jitter", analysis_config.get("gp_jitter", 1e-8))),
                        out_dir=_ensure_dir(os.path.join(epoch_root, "gp_posterior")),
                        show_eval_targets=bool(task.get("show_eval_targets", True)),
                    )
                )

            else:
                raise ValueError(
                    f"Unknown plot task type {plot_type!r}. Use one of "
                    "{'eigenfunction', 'gp_prior', 'gp_posterior', 'gp_prior_frequency', 'desired_frequency'}."
                )

    if len(saved_paths) == 0:
        print(
            "No analysis plots requested. Add sections under plots or entries "
            "under plot_tasks in the analysis YAML."
        )
        return []

    print("Saved analysis plots:")
    for path in saved_paths:
        print(f"  {path}")

    return saved_paths


def main() -> None:
    # Same style as the training scripts: pass one YAML path, or edit the
    # default config file and run the script directly from an IDE.
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./analyze_rpm_run_config.yaml"
    analysis_config = load_analysis_config(config_path)
    run_analysis(analysis_config)


if __name__ == "__main__":
    main()
