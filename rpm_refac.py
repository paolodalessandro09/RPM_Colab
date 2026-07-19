#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:45:21 2026

@author: benjamin
"""

import os
import sys

# Ensure local package modules are loaded from this file's directory first.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _MODULE_DIR:
    try:
        sys.path.remove(_MODULE_DIR)
    except ValueError:
        pass
    sys.path.insert(0, _MODULE_DIR)
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import dspTools as dt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering,KMeans
import torch.distributions as tdist
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.neighbors import KNeighborsRegressor
from scipy.signal import stft,freqz,butter,impulse, firwin,chirp
from scipy.fft import fft, fftfreq, fftshift
import scipy.io as io
import functools 
import torch.nn.functional as F
from sklearn import linear_model
import torch
from typing import Optional, Tuple, Dict, Any
import copy
from speed_refac import TorchSpecNystrom

from util_refac import psd_project

def _as_layer_list(value, num_layers: int, *, name: str):
    """
    Convert a scalar or list-like config value into a per-layer list.
    """
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value for _ in range(num_layers)]

    if len(values) != num_layers:
        raise ValueError(
            f"{name} must be scalar or have length {num_layers}. "
            f"Got length {len(values)}."
        )

    return values


def _trace_normalize_psd_metric(
    metric: torch.Tensor,
    *,
    target_trace: Optional[float] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Normalize a PSD metric by trace.

    If target_trace is None, use the matrix dimension.
    """
    if target_trace is None:
        target_trace = float(metric.shape[0])

    trace = torch.trace(metric).clamp_min(eps)
    return metric * (target_trace / trace)

def _kernel_type_from_kernel(kernel) -> str:
    """
    Infer which radial Mahalanobis kernel family is being used.

    Kernel classes can set kernel.kernel_type explicitly. If that attribute is
    absent, we fall back to the class name so this also works with subclasses.
    """
    kernel_type = getattr(kernel, "kernel_type", None)

    if kernel_type is None:
        name = type(kernel).__name__.lower()
        if "laplac" in name:
            kernel_type = "laplacian"
        else:
            kernel_type = "rbf"

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
            "Unsupported kernel_type for differentiable metric-gradient "
            f"features: {kernel_type!r}."
        )

    return aliases[kernel_type]


def _mahalanobis_kernel_cross_direct(
    centers: torch.Tensor,
    x: torch.Tensor,
    metric: torch.Tensor,
    sigma: float,
    *,
    kernel_type: str = "rbf",
    chunksize: int = 500,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute K(centers, x) directly from a Mahalanobis metric matrix.

    Supported kernel_type values:
        - "rbf":       exp(-d_M^2 / (2 sigma^2))
        - "laplacian": exp(-d_M / sigma)

    Returns shape:
        (num_centers, num_points)

    This avoids metric_factor/eigh and is therefore safe for gradients with
    respect to metric in the gradient-descent update path.
    """
    sigma = float(sigma)
    sigma_safe = max(sigma, eps)
    kernel_type = str(kernel_type).lower()

    num_centers = centers.shape[0]
    num_points = x.shape[0]

    out = torch.empty(
        num_centers,
        num_points,
        device=x.device,
        dtype=x.dtype,
    )

    centers = centers.to(device=x.device, dtype=x.dtype)
    metric = metric.to(device=x.device, dtype=x.dtype)

    center_quad = torch.sum((centers @ metric) * centers, dim=1)

    for start in range(0, num_points, chunksize):
        xb = x[start:start + chunksize]

        x_quad = torch.sum((xb @ metric) * xb, dim=1)
        cross = centers @ metric @ xb.T

        squared_dist = center_quad[:, None] + x_quad[None, :] - 2.0 * cross
        squared_dist = torch.clamp(squared_dist, min=0.0)

        if kernel_type == "rbf":
            values = torch.exp(-squared_dist / (2.0 * sigma_safe * sigma_safe))

        elif kernel_type == "laplacian":
            dist = torch.sqrt(torch.clamp(squared_dist, min=eps))
            values = torch.exp(-dist / sigma_safe)

        else:
            raise ValueError(
                "kernel_type must be one of {'rbf', 'laplacian'}. "
                f"Got {kernel_type!r}."
            )

        out[:, start:start + xb.shape[0]] = values

    return out


# Backward-compatible name for code that may still reference the old helper.
def _mahalanobis_rbf_cross_direct(
    centers: torch.Tensor,
    x: torch.Tensor,
    metric: torch.Tensor,
    sigma: float,
    *,
    chunksize: int = 500,
    eps: float = 1e-12,
) -> torch.Tensor:
    return _mahalanobis_kernel_cross_direct(
        centers,
        x,
        metric,
        sigma,
        kernel_type="rbf",
        chunksize=chunksize,
        eps=eps,
    )


def _spec_features_metric_grad_direct(
    layer,
    x: torch.Tensor,
    metric: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute TorchSpecNystrom features while differentiating directly through
    metric.

    This reuses the already-built spectral eigvecs/eigvals as constants, but
    computes the cross-kernel with a direct Mahalanobis formula instead of
    calling layer.kernel(...). That avoids gradients through metric_factor/eigh.
    """
    spec = layer.spec

    if spec is None:
        raise RuntimeError("Layer spec is None. Call set_centers first.")

    if spec.approx_mode is None:
        raise RuntimeError("Layer spec has no approx_mode. Call make_kernel first.")

    num_features = min(layer.output_dim, spec.eigvals.numel())
    chunksize = layer.chunk_size

    sigma = float(layer.kernel.sigma)
    kernel_type = _kernel_type_from_kernel(layer.kernel)

    # ------------------------------------------------------------------
    # Nyström mode
    # ------------------------------------------------------------------
    if spec.approx_mode == "nystrom":
        landmarks = spec.nystrom_centers.to(device=x.device, dtype=x.dtype)

        eigvecs = spec.nystrom_eigvecs[:, :num_features].to(
            device=x.device,
            dtype=x.dtype,
        )

        eigvals = spec.nystrom_eigvals[:num_features].to(
            device=x.device,
            dtype=x.dtype,
        )

        sqrt_landmark_weights = spec.nystrom_sqrt_weights.to(
            device=x.device,
            dtype=x.dtype,
        ).view(1, -1)

        scale = spec.nystrom_scale.to(
            device=x.device,
            dtype=x.dtype,
        )

        inv_sqrt_eigvals = (
            1.0 / torch.sqrt(eigvals.clamp_min(1e-12))
        ).view(1, -1)

        cross_kernel = _mahalanobis_kernel_cross_direct(
            landmarks,
            x,
            metric,
            sigma,
            kernel_type=kernel_type,
            chunksize=chunksize,
        ).T

        weighted_cross_kernel = cross_kernel * sqrt_landmark_weights

        features = scale * (
            (weighted_cross_kernel @ eigvecs) * inv_sqrt_eigvals
        )

    # ------------------------------------------------------------------
    # Full mode
    # ------------------------------------------------------------------
    elif spec.approx_mode == "full":
        centers = spec.centers.to(device=x.device, dtype=x.dtype)

        sqrt_weights = torch.sqrt(
            torch.clamp(
                spec.weights.to(device=x.device, dtype=x.dtype),
                min=1e-12,
            )
        ).view(1, -1)

        eigvecs = spec.eigvecs[:, :num_features].to(
            device=x.device,
            dtype=x.dtype,
        )

        eigvals = spec.eigvals[:num_features].to(
            device=x.device,
            dtype=x.dtype,
        )

        inv_sqrt_eigvals = (
            1.0 / torch.sqrt(eigvals.clamp_min(1e-12))
        ).view(1, -1)

        cross_kernel = _mahalanobis_kernel_cross_direct(
            centers,
            x,
            metric,
            sigma,
            kernel_type=kernel_type,
            chunksize=chunksize,
        ).T

        weighted_cross_kernel = cross_kernel * sqrt_weights

        features = (weighted_cross_kernel @ eigvecs) * inv_sqrt_eigvals

    else:
        raise RuntimeError(f"Unknown approx_mode={spec.approx_mode}")

    if normalize:
        features = spec._normalize_features(features)

    return features


# =============================================================================
# FOOF helpers
# =============================================================================

def _safe_solve(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Solve C X = R with a pinv fallback for ill-conditioned FOOF systems.
    """
    try:
        return torch.linalg.solve(c, r)
    except RuntimeError:
        return torch.linalg.pinv(c) @ r
    except torch._C._LinAlgError:
        return torch.linalg.pinv(c) @ r


def _foof_metric_factor_from_metric(
    metric: torch.Tensor,
    *,
    rank: Optional[int] = None,
    eps: float = 1e-12,
    max_eval: Optional[float] = None,
) -> torch.Tensor:
    """
    Return W with shape (r, d) such that W.T @ W approximates metric.

    rank=None keeps all d eigen-directions. rank<d gives a low-rank factor
    using the largest eigenvalues.
    """
    metric = torch.nan_to_num(
        metric,
        nan=0.0,
        posinf=1.0 / eps,
        neginf=-1.0 / eps,
    )
    metric = 0.5 * (metric + metric.T)

    evals, evecs = torch.linalg.eigh(metric)
    evals = torch.nan_to_num(
        evals,
        nan=eps,
        posinf=1.0 / eps,
        neginf=eps,
    )
    evals = torch.clamp(evals, min=eps)

    if max_eval is not None:
        evals = torch.clamp(evals, max=float(max_eval))

    dim = metric.shape[0]
    if rank is None:
        rank = dim
    rank = int(min(max(rank, 1), dim))

    if rank < dim:
        keep = torch.argsort(evals, descending=True)[:rank]
        evals = evals[keep]
        evecs = evecs[:, keep]

    # Rows of W are sqrt(lambda_i) q_i.T, so W.T @ W = Q diag(lambda) Q.T.
    return torch.sqrt(evals).unsqueeze(1) * evecs.T


def _radial_kernel_cross_transformed(
    centers_s: torch.Tensor,
    s: torch.Tensor,
    sigma: float,
    *,
    kernel_type: str = "rbf",
    chunksize: int = 500,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute K(centers_s, s) in transformed S-space.

    Returns shape (num_centers, num_points). This is an ordinary radial kernel
    in S-space, equivalent to the Mahalanobis kernel in the original Z-space.
    """
    sigma_safe = max(float(sigma), eps)
    kernel_type = str(kernel_type).lower()

    num_centers = centers_s.shape[0]
    num_points = s.shape[0]

    centers_s = centers_s.to(device=s.device, dtype=s.dtype)

    out = torch.empty(
        num_centers,
        num_points,
        device=s.device,
        dtype=s.dtype,
    )
    center_norms = torch.sum(centers_s * centers_s, dim=1)

    for start in range(0, num_points, chunksize):
        sb = s[start:start + chunksize]
        s_norms = torch.sum(sb * sb, dim=1)
        cross = centers_s @ sb.T
        squared_dist = center_norms[:, None] + s_norms[None, :] - 2.0 * cross
        squared_dist = torch.clamp(squared_dist, min=0.0)

        if kernel_type == "rbf":
            values = torch.exp(-squared_dist / (2.0 * sigma_safe * sigma_safe))

        elif kernel_type == "laplacian":
            dist = torch.sqrt(torch.clamp(squared_dist, min=eps))
            values = torch.exp(-dist / sigma_safe)

        else:
            raise ValueError(
                "kernel_type must be one of {'rbf', 'laplacian'}. "
                f"Got {kernel_type!r}."
            )

        out[:, start:start + sb.shape[0]] = values

    return out


def _spec_features_foof_transformed(
    layer,
    s: torch.Tensor,
    factor_w: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute TorchSpecNystrom features from transformed inputs S = Z @ W.T.

    The spectral eigensystem is the one already built in layer.spec from the
    equivalent metric kernel on Z. During the FOOF backward pass, the
    cross-kernel is recomputed directly in S-space so autograd gives gradients
    with respect to S, not directly with respect to M.
    """
    spec = layer.spec

    if spec is None:
        raise RuntimeError("Layer spec is None. Call set_centers first.")

    if spec.approx_mode is None:
        raise RuntimeError("Layer spec has no approx_mode. Call make_kernel first.")

    chunksize = layer.chunk_size
    sigma = float(layer.kernel.sigma)
    kernel_type = _kernel_type_from_kernel(layer.kernel)
    factor_w = factor_w.to(device=s.device, dtype=s.dtype)

    # ------------------------------------------------------------------
    # Nyström mode
    # ------------------------------------------------------------------
    if spec.approx_mode == "nystrom":
        num_features = min(layer.output_dim, spec.nystrom_eigvals.numel())

        landmarks = spec.nystrom_centers.to(device=s.device, dtype=s.dtype)
        landmarks_s = landmarks @ factor_w.T

        eigvecs = spec.nystrom_eigvecs[:, :num_features].to(
            device=s.device,
            dtype=s.dtype,
        )
        eigvals = spec.nystrom_eigvals[:num_features].to(
            device=s.device,
            dtype=s.dtype,
        )

        sqrt_landmark_weights = spec.nystrom_sqrt_weights.to(
            device=s.device,
            dtype=s.dtype,
        ).view(1, -1)

        scale = spec.nystrom_scale.to(device=s.device, dtype=s.dtype)
        inv_sqrt_eigvals = (
            1.0 / torch.sqrt(eigvals.clamp_min(1e-12))
        ).view(1, -1)

        cross_kernel = _radial_kernel_cross_transformed(
            landmarks_s,
            s,
            sigma,
            kernel_type=kernel_type,
            chunksize=chunksize,
        ).T

        weighted_cross_kernel = cross_kernel * sqrt_landmark_weights
        features = scale * (
            (weighted_cross_kernel @ eigvecs) * inv_sqrt_eigvals
        )

    # ------------------------------------------------------------------
    # Full mode
    # ------------------------------------------------------------------
    elif spec.approx_mode == "full":
        num_features = min(layer.output_dim, spec.eigvals.numel())

        centers = spec.centers.to(device=s.device, dtype=s.dtype)
        centers_s = centers @ factor_w.T

        sqrt_weights = torch.sqrt(
            torch.clamp(
                spec.weights.to(device=s.device, dtype=s.dtype),
                min=1e-12,
            )
        ).view(1, -1)

        eigvecs = spec.eigvecs[:, :num_features].to(
            device=s.device,
            dtype=s.dtype,
        )
        eigvals = spec.eigvals[:num_features].to(
            device=s.device,
            dtype=s.dtype,
        )
        inv_sqrt_eigvals = (
            1.0 / torch.sqrt(eigvals.clamp_min(1e-12))
        ).view(1, -1)

        cross_kernel = _radial_kernel_cross_transformed(
            centers_s,
            s,
            sigma,
            kernel_type=kernel_type,
            chunksize=chunksize,
        ).T

        weighted_cross_kernel = cross_kernel * sqrt_weights
        features = (weighted_cross_kernel @ eigvecs) * inv_sqrt_eigvals

    else:
        raise RuntimeError(f"Unknown approx_mode={spec.approx_mode!r}.")

    if normalize:
        features = spec._normalize_features(features)

    return features


def _foof_direction(
    z: torch.Tensor,
    d_s: torch.Tensor,
    *,
    gamma: float,
    covariance_normalize: bool,
    eps: float,
    max_direction_norm: Optional[float],
) -> torch.Tensor:
    """
    Compute Q = solve(Z.T Z + gamma I, Z.T D).T.

    Q has the same shape as W, so the raw FOOF/GD update is W <- W - lr * Q.
    """
    z = z.detach()
    d_s = d_s.detach()

    n, dim = z.shape
    eye = torch.eye(dim, device=z.device, dtype=z.dtype)

    if covariance_normalize:
        c = (z.T @ z) / max(n, 1) + float(gamma) * eye
        r = (z.T @ d_s) / max(n, 1)
    else:
        c = z.T @ z + float(gamma) * eye
        r = z.T @ d_s

    sol = _safe_solve(c, r)
    direction = sol.T
    direction = torch.nan_to_num(
        direction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if max_direction_norm is not None and max_direction_norm > 0:
        direction_norm = torch.linalg.norm(direction)
        if torch.isfinite(direction_norm) and direction_norm > max_direction_norm:
            direction = direction * (
                float(max_direction_norm) / direction_norm.clamp_min(eps)
            )

    return direction.detach()

#==================RPM CLASSES=================================================
# =============================================================================
# RPMLayer
# =============================================================================

class RPMLayer:
    def __init__(
        self,
        kernel,
        output_dim,
        alpha=1.0,
        chunk_size=1000,
        metric_clamp=1e-12,
        approx_method="full",
        nystrom_m=512,
        nystrom_mode="uniform",
        nystrom_seed=0,
        freeze=False,
        copy_kernel=True,
        full=True,
    ):
        """
        Single Recursive Parsimony Machine layer.

        Feature normalization is handled internally by TorchSpecNystrom.
        This layer defaults to returning normalized features, but get_features
        allows normalize=False as an override.

        Parameters
        ----------
        kernel : object
            Kernel object compatible with TorchSpecNystrom.

        output_dim : int
            Number of spectral features.

        alpha : float
            Metric update rate.

        chunk_size : int
            Chunk size for kernel and feature computations.

        metric_clamp : float
            Eigenvalue floor used when projecting AGOP to PSD.

        approx_method : {"full", "nystrom"}
            Approximation method.

        nystrom_m : int
            Number of Nyström landmarks.

        nystrom_mode : {"uniform", "weighted", "first"}
            Nyström sampling mode.

        nystrom_seed : int or None
            Random seed for Nyström sampling.

        freeze : bool
            If True, metric updates are skipped.

        copy_kernel : bool
            If True, deep-copy kernel so each layer owns its metric.

        full : bool
            Passed to TorchSpecNystrom.
        """
        self.kernel = copy.deepcopy(kernel) if copy_kernel else kernel

        self.output_dim = int(output_dim)
        self.alpha = float(alpha)
        self.chunk_size = int(chunk_size)
        self.metric_clamp = metric_clamp

        self.approx_method = approx_method
        self.nystrom_m = int(nystrom_m)
        self.nystrom_mode = nystrom_mode
        self.nystrom_seed = nystrom_seed

        self.freeze = freeze
        self.full = full

        self.centers = None
        self.spec = None

        self.agops = []

    @property
    def metric(self):
        return self.kernel.metric_a

    @metric.setter
    def metric(self, value):
        self.kernel.metric_a = value

    @property
    def A(self):
        return self.metric

    @A.setter
    def A(self, value):
        self.metric = value

    @property
    def K(self):
        return self.spec

    @property
    def M(self):
        return self.output_dim

    @property
    def PSDclamp(self):
        return self.metric_clamp

    def set_centers(self, x, q_thresh=None):
        """
        Set centers and initialize the internal TorchSpecNystrom object.
        """
        self.centers = x

        if self.metric is None:
            self.metric = torch.eye(
                x.shape[1],
                device=x.device,
                dtype=x.dtype,
            )

        self.spec = TorchSpecNystrom(
            kernel=self.kernel,
            centers=x,
            full=self.full,
            num_eigs=self.output_dim,
        )

        if q_thresh is not None:
            self.spec.quantize_centers_mahalanobis(
                q_thresh,
                chunksize=self.chunk_size,
            )

        return self

    def make_kernel(self):
        """
        Build the internal spectral kernel representation.

        TorchSpecNystrom.make_kernel(...) always fits feature normalization
        constants.
        """
        if self.spec is None:
            raise RuntimeError("set_centers must be called before make_kernel.")

        self.spec.make_kernel(
            chunksize=self.chunk_size,
            method=self.approx_method,
            nystrom_m=self.nystrom_m,
            nystrom_mode=self.nystrom_mode,
            nystrom_seed=self.nystrom_seed,
            normalizer_feature_chunksize=self.chunk_size,
            normalizer_kernel_chunksize=self.chunk_size,
        )

        return self

    def get_features(self, x, normalize=True):
        """
        Compute layer features.

        normalize=True by default.
        Use normalize=False to override and return raw features.
        """
        if self.spec is None:
            raise RuntimeError(
                "set_centers and make_kernel must be called before get_features."
            )

        return self.spec.get_features(
            x,
            num_features=self.output_dim,
            feature_chunksize=self.chunk_size,
            kernel_chunksize=self.chunk_size,
            normalize=normalize,
        )

    def update_metric(self, new_metric=None):
        """
        Update the kernel metric and clear stale spectral state.
        """
        if self.freeze:
            return self

        if new_metric is not None:
            self.metric = new_metric

        if self.spec is not None:
            self.spec.kernel = self.kernel
            self.spec._clear_spectral_state()

        return self

    def store_agop(self, agop):
        self.agops.append(agop)

    def reset_agop(self):
        self.agops = []

# =============================================================================
# RPM
# =============================================================================

class RPM:
    def __init__(
        self,
        layers,
        *,
        metric_update_rule="agop",
        normalize=False,
        update_config=None,
    ):
        """
        Recursive Parsimony Machine.

        This is a wrapper around RPMLayer objects.

        The RPM owns the metric-update rule. Training code should not branch
        between AGOP and gradient descent directly. Instead, the standard
        update cycle is:

            features = model.forward(x)
            model.train_regressor(features, y)
            model.backward(x, y)
            model.update_metrics()

        Parameters
        ----------
        layers : list[RPMLayer]
            RPM layers.

        metric_update_rule : {"agop", "gradient_descent", "foof"}
            Internal rule used by backward(...). Aliases "gradient" and "gd"
            are accepted for "gradient_descent". Aliases "foof_gd",
            "foof_sgd", "foof_adam", and "operator_descent" are accepted
            for "foof".

        normalize : bool
            Default feature-normalization behavior for set_centers(...),
            forward(...), backward(...), and AGOP/gradient/FOOF update internals.
            Individual method calls may still override this by passing
            normalize=True/False explicitly.

        update_config : dict or None
            Optional dictionary of update hyperparameters. This allows the
            training script to pass config values once at construction time.
        """
        self.layers = layers
        self.weight = None

        self.metric_update_rule = self._canonical_metric_update_rule(
            metric_update_rule
        )
        self.normalize = bool(normalize)
        self.update_config = dict(update_config or {})

        # Pending metric candidates produced by backward(...). These are
        # applied only by update_metrics(...), so backward never mutates M.
        self.pending_metric_updates = [[] for _ in self.layers]
        self.pending_readout = None

        # Optimizer state for gradient_descent metric updates. AGOP does not
        # use this state. Each layer gets its own Adam moments because each
        # metric matrix may have a different shape.
        self.metric_optimizer_state = [
            {"step": 0, "m": None, "v": None}
            for _ in self.layers
        ]

        # Optimizer state for FOOF updates to the metric factor W. This is
        # separate from metric_optimizer_state because FOOF optimizes W-like
        # rectangular directions, while gradient_descent optimizes square M.
        self.foof_optimizer_state = [
            {"step": 0, "m": None, "v": None}
            for _ in self.layers
        ]

    @property
    def W(self):
        return self.weight

    @W.setter
    def W(self, value):
        self.weight = value

    @staticmethod
    def _canonical_metric_update_rule(rule):
        """
        Normalize metric-update rule names.
        """
        rule = str(rule).lower()

        aliases = {
            "agop": "agop",
            "gradient": "gradient_descent",
            "gd": "gradient_descent",
            "gradient_descent": "gradient_descent",
            "foof": "foof",
            "foof_gd": "foof",
            "foof_sgd": "foof",
            "foof_adam": "foof",
            "operator": "foof",
            "operator_descent": "foof",
            "operator_level": "foof",
        }

        if rule not in aliases:
            raise ValueError(
                "metric_update_rule must be one of "
                "{'agop', 'gradient_descent', 'foof', 'gradient', 'gd', "
                "'foof_gd', 'foof_sgd', 'foof_adam'}. "
                f"Got {rule!r}."
            )

        return aliases[rule]

    def set_metric_update_rule(self, rule):
        """
        Change the metric-update rule used by backward(...).
        """
        self.metric_update_rule = self._canonical_metric_update_rule(rule)
        return self

    def _cfg(self, name, default=None):
        """
        Read an update hyperparameter from self.update_config.
        """
        return self.update_config.get(name, default)

    def _resolve_normalize(self, normalize):
        """
        Use the model-level normalization default unless explicitly overridden.
        """
        if normalize is None:
            return self.normalize

        return bool(normalize)

    def clear_pending_metric_updates(self):
        """
        Clear metric updates accumulated by backward(...).
        """
        self.pending_metric_updates = [[] for _ in self.layers]
        self.pending_readout = None
        return self

    def set_centers(self, x, q_thresh=None, normalize=None):
        """
        Set centers and build kernels for every layer.

        For layer 0, centers are x.
        For layer i > 0, centers are the features from layer i - 1.

        normalize=True by default, so multilayer center propagation uses
        normalized intermediate features unless explicitly overridden.
        """
        normalize = self._resolve_normalize(normalize)
        z = x

        for layer_idx, layer in enumerate(self.layers):
            layer.set_centers(z, q_thresh=q_thresh)
            layer.make_kernel()

            if layer_idx < len(self.layers) - 1:
                z = layer.get_features(z, normalize=normalize)

        return self

    def forward(
        self,
        x,
        normalize=None,
        return_layer_features=False,
    ):
        """
        Forward pass through all layers.

        Parameters
        ----------
        x:
            Input data.

        normalize:
            normalize=True by default.
            Use normalize=False to return raw unnormalized features.

        return_layer_features:
            If False, return only the final-layer features.

            If True, return:

                final_features, layer_features

            where layer_features is a list containing the output features from
            each RPM layer.

        Notes
        -----
        The default behavior is unchanged, so existing training scripts that use

            features = model.forward(x, normalize=True)

        will continue to work.
        """
        normalize = self._resolve_normalize(normalize)
        z = x
        layer_features = []

        for layer in self.layers:
            z = layer.get_features(z, normalize=normalize)

            if return_layer_features:
                layer_features.append(z)

        if return_layer_features:
            return z, layer_features

        return z

    def train_regressor(
        self,
        features,
        targets,
        rcond=1e-16,
        onehot=False,
        return_u=False,
    ):
        """
        Train final linear regressor using ridge/MMSE.
        """
        num_samples = features.shape[0]
        feature_dim = features.shape[1]

        U = features.T @ features / num_samples

        if onehot:
            P = torch.mean(
                features.T.unsqueeze(2) * targets.unsqueeze(0),
                dim=1,
            )

            self.weight = torch.linalg.pinv(U, rcond=rcond) @ P

        else:
            targets = targets.reshape(-1)
            P = torch.mean(features.T * targets, dim=1)

            ridge = rcond * torch.eye(
                feature_dim,
                device=features.device,
                dtype=features.dtype,
            )

            self.weight = torch.linalg.solve(U + ridge, P)

        if return_u:
            return U, P

        return None

    def predict(self, features):
        """
        Evaluate the linear readout on already-computed features.
        """
        if self.weight is None:
            raise RuntimeError("train_regressor must be called before predict.")

        return features @ self.weight

    def calc_agop(
        self,
        x,
        label_weights=None,
        center_grads=True,
        projector=None,
        normalize=None,
    ):
        """
        Compute and store AGOP for every layer.

        normalize=True by default, so AGOP uses the same normalized feature
        pipeline as training unless explicitly overridden.
        """
        if self.weight is None:
            raise RuntimeError("train_regressor must be called before calc_agop.")

        normalize = self._resolve_normalize(normalize)
        inputs = [x.detach().clone().requires_grad_(True)]

        for layer in self.layers:
            h = layer.get_features(inputs[-1], normalize=normalize)

            if projector is not None:
                h = h @ projector

            inputs.append(h)

        predictions = inputs[-1] @ self.weight

        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(1)

        batch_size, output_dim = predictions.shape
        device = predictions.device
        dtype = predictions.dtype

        if label_weights is None:
            label_weights = torch.ones(
                output_dim,
                device=device,
                dtype=dtype,
            )
        else:
            label_weights = label_weights.to(device=device, dtype=dtype)

        for layer_idx, layer in enumerate(self.layers):
            layer_input = inputs[layer_idx]
            input_dim = layer_input.shape[1]

            agop = torch.zeros(
                input_dim,
                input_dim,
                device=device,
                dtype=dtype,
            )

            for output_idx in range(output_dim):
                objective = (
                    label_weights[output_idx] * predictions[:, output_idx]
                ).sum()

                grad = torch.autograd.grad(
                    outputs=objective,
                    inputs=layer_input,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                if center_grads:
                    grad = grad - grad.mean(dim=0, keepdim=True)

                agop = agop + (grad.T @ grad) / batch_size

            layer.store_agop(agop)

        return self
    
    def backward(self, x, y, **kwargs):
        """
        Compute metric-update information using the model's internal rule.

        This method intentionally does not mutate layer metric matrices. It
        only accumulates pending metric updates that are later applied by
        update_metrics(...).
        """
        rule = self.metric_update_rule

        if rule == "agop":
            return self._backward_agop(x, y, **kwargs)

        if rule == "gradient_descent":
            return self._backward_gradient_descent(x, y, **kwargs)

        if rule == "foof":
            return self._backward_foof(x, y, **kwargs)

        raise RuntimeError(f"Unsupported metric_update_rule={rule!r}.")

    def backwards(self, x, y, **kwargs):
        """
        Alias for backward(...), matching the training-loop spelling sometimes
        used in notes.
        """
        return self.backward(x, y, **kwargs)

    def _backward_agop(self, x, y=None, **kwargs):
        """
        AGOP backward pass.

        The AGOP matrices are stored on each layer and consumed by
        update_metrics(...). The target y is accepted for API symmetry but is
        not directly used because the current readout already encodes the
        trained regression function.
        """
        center_grads = kwargs.get(
            "center_grads",
            self._cfg("center_grads", True),
        )
        projector = kwargs.get("projector", self._cfg("projector", None))
        label_weights = kwargs.get(
            "label_weights",
            self._cfg("label_weights", None),
        )
        normalize = kwargs.get("normalize", None)

        self.calc_agop(
            x,
            label_weights=label_weights,
            center_grads=center_grads,
            projector=projector,
            normalize=normalize,
        )

        return self

    def _backward_gradient_descent(self, x, y, **kwargs):
        """
        Gradient-descent backward pass.

        Computes raw metric gradients and stores them in
        self.pending_metric_updates. update_metrics(...) is responsible for
        applying the selected optimizer, either SGD or Adam, and then doing
        PSD projection / normalization.
        """
        objective = kwargs.get(
            "metric_gradient_objective",
            self._cfg("metric_gradient_objective", "fixed_readout"),
        )

        common = dict(
            loss=kwargs.get(
                "loss",
                self._cfg("metric_gradient_loss", "mse"),
            ),
            normalize=kwargs.get("normalize", None),
        )

        if objective == "fixed_readout":
            gradients = self._compute_metric_gradients_fixed_readout(
                x,
                y,
                use_current_readout=bool(
                    kwargs.get(
                        "use_current_readout",
                        self._cfg("metric_gradient_use_current_readout", True),
                    )
                ),
                **common,
            )

        elif objective == "full_mmse":
            gradients, readout = self._compute_metric_gradients_full_mmse(
                x,
                y,
                reg_param=float(
                    kwargs.get(
                        "reg_param",
                        self._cfg("reg_param", 1e-16),
                    )
                ),
                onehot=bool(
                    kwargs.get(
                        "onehot",
                        self._cfg("use_onehot_labels", False),
                    )
                ),
                store_differentiable_readout=bool(
                    kwargs.get(
                        "store_differentiable_readout",
                        self._cfg(
                            "metric_gradient_store_differentiable_readout",
                            False,
                        ),
                    )
                ),
                **common,
            )
            if readout is not None:
                self.pending_readout = readout.detach()

        else:
            raise ValueError(
                "metric_gradient_objective must be one of "
                "{'fixed_readout', 'full_mmse'}. "
                f"Got {objective!r}."
            )

        for layer_idx, grad in enumerate(gradients):
            if grad is not None:
                self.pending_metric_updates[layer_idx].append(grad.detach())

        return self

    def _backward_foof(self, x, y, **kwargs):
        """
        FOOF backward pass.

        For each layer, the current PSD metric M_l is factored as

            M_l = W_l.T @ W_l,

        and the layer is evaluated as

            Z_{l-1} -> S_l = Z_{l-1} @ W_l.T -> Phi_l(S_l).

        Autograd gives D_l = d loss / d S_l. This method stores the
        covariance-corrected FOOF direction

            Q_l = solve(Z.T @ Z + gamma I, Z.T @ D_l).T

        in self.pending_metric_updates. The actual W and M update is applied
        by update_metrics(...), matching the AGOP and gradient_descent paths.
        """
        if self.weight is None:
            raise RuntimeError("train_regressor must be called before backward(...).")

        loss = kwargs.get(
            "loss",
            self._cfg("foof_loss", self._cfg("metric_gradient_loss", "mse")),
        )
        if loss != "mse":
            raise ValueError("Only loss='mse' is currently supported for FOOF.")

        normalize = self._resolve_normalize(kwargs.get("normalize", None))
        use_current_readout = bool(
            kwargs.get(
                "use_current_readout",
                self._cfg(
                    "foof_use_current_readout",
                    self._cfg("metric_gradient_use_current_readout", True),
                ),
            )
        )

        gamma_values = _as_layer_list(
            kwargs.get("gamma", self._cfg("foof_gamma", 1e-6)),
            len(self.layers),
            name="foof_gamma",
        )

        factor_ranks = _as_layer_list(
            kwargs.get("factor_rank", self._cfg("foof_factor_rank", None)),
            len(self.layers),
            name="foof_factor_rank",
        )

        eps = float(
            kwargs.get(
                "eps",
                self._cfg("foof_eps", self._cfg("metric_eps", 1e-12)),
            )
        )
        covariance_normalize = bool(
            kwargs.get(
                "covariance_normalize",
                self._cfg("foof_covariance_normalize", False),
            )
        )
        max_direction_norm = kwargs.get(
            "max_direction_norm",
            self._cfg(
                "foof_max_direction_norm",
                self._cfg("metric_gradient_max_norm", 1.0),
            ),
        )
        if max_direction_norm is not None:
            max_direction_norm = float(max_direction_norm)

        eigenclip_max = kwargs.get(
            "eigenclip_max",
            self._cfg("foof_eigenclip_max", None),
        )

        z = x
        layer_inputs = []
        s_vars = []

        for layer_idx, layer in enumerate(self.layers):
            if layer.freeze:
                z = layer.get_features(z, normalize=normalize)
                continue

            if layer.metric is None:
                raise RuntimeError(
                    "Layer metric is None. Call set_centers before backward(...)."
                )

            metric = layer.metric.to(device=z.device, dtype=z.dtype)
            factor_w = _foof_metric_factor_from_metric(
                metric,
                rank=factor_ranks[layer_idx],
                eps=max(float(layer.metric_clamp or eps), eps),
                max_eval=eigenclip_max,
            ).detach()

            layer_inputs.append((layer_idx, z.detach()))

            s = z @ factor_w.T
            if not s.requires_grad:
                s = s.detach().requires_grad_(True)

            s_vars.append((layer_idx, s))
            z = _spec_features_foof_transformed(
                layer,
                s,
                factor_w,
                normalize=normalize,
            )

        if len(s_vars) == 0:
            return self

        weight = self.weight.detach() if use_current_readout else self.weight
        predictions = z @ weight
        if predictions.shape != y.shape:
            predictions = predictions.reshape_as(y)

        objective = torch.mean((predictions - y) ** 2)

        active_s_vars = [item[1] for item in s_vars]
        s_grads = torch.autograd.grad(
            objective,
            active_s_vars,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        z_by_layer = {layer_idx: z_value for layer_idx, z_value in layer_inputs}

        for (layer_idx, _s), d_s in zip(s_vars, s_grads):
            direction = _foof_direction(
                z_by_layer[layer_idx],
                d_s,
                gamma=float(gamma_values[layer_idx]),
                covariance_normalize=covariance_normalize,
                eps=eps,
                max_direction_norm=max_direction_norm,
            )
            self.pending_metric_updates[layer_idx].append(direction)

        return self

    def _adam_foof_step(
        self,
        *,
        layer_idx,
        direction,
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
    ):
        """
        Return Adam's bias-corrected update direction for one FOOF factor W.
        """
        state = self.foof_optimizer_state[layer_idx]

        if (
            state["m"] is None
            or state["m"].shape != direction.shape
            or state["m"].device != direction.device
            or state["m"].dtype != direction.dtype
        ):
            state["m"] = torch.zeros_like(direction)
            state["v"] = torch.zeros_like(direction)
            state["step"] = 0

        state["step"] += 1
        state["m"] = beta1 * state["m"] + (1.0 - beta1) * direction
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * (direction * direction)

        step_idx = state["step"]
        m_hat = state["m"] / (1.0 - beta1 ** step_idx)
        v_hat = state["v"] / (1.0 - beta2 ** step_idx)

        return m_hat / (torch.sqrt(v_hat) + adam_eps)

    def _postprocess_foof_metric(
        self,
        metric,
        *,
        layer,
        norm="Trace",
        mult=1.0,
        eps=1e-12,
        project_psd=True,
        diag_load=0.0,
        eigenclip_max=None,
    ):
        """
        Stabilize M = W.T @ W after a FOOF optimizer step.
        """
        metric = torch.nan_to_num(
            metric,
            nan=0.0,
            posinf=1.0 / eps,
            neginf=-1.0 / eps,
        )
        metric = 0.5 * (metric + metric.T)

        if diag_load is not None and float(diag_load) > 0:
            eye = torch.eye(
                metric.shape[0],
                device=metric.device,
                dtype=metric.dtype,
            )
            metric = metric + float(diag_load) * eye

        if project_psd:
            clamp_eps = layer.metric_clamp if layer.metric_clamp is not None else eps
            metric = psd_project(metric, eps=clamp_eps)

        if eigenclip_max is not None:
            evals, evecs = torch.linalg.eigh(0.5 * (metric + metric.T))
            evals = torch.clamp(evals, min=eps, max=float(eigenclip_max))
            metric = (evecs * evals.unsqueeze(0)) @ evecs.T
            metric = 0.5 * (metric + metric.T)

        metric = self._normalize_metric_update(
            metric,
            norm=norm,
            mult=mult,
            eps=eps,
        )

        return metric.detach()

    def _apply_pending_foof_updates(self):
        """
        Apply pending FOOF directions to W, reconstruct M = W.T @ W, and
        stabilize the resulting metric.

        The optimizer acts on the covariance-corrected FOOF direction Q, not on
        the square metric matrix directly.
        """
        optimizer = str(
            self._cfg(
                "foof_optimizer",
                self._cfg("metric_gradient_optimizer", "sgd"),
            )
        ).lower()
        if optimizer == "gd":
            optimizer = "sgd"

        if optimizer not in {"sgd", "adam"}:
            raise ValueError(
                "foof_optimizer must be one of {'sgd', 'gd', 'adam'}. "
                f"Got {optimizer!r}."
            )

        num_layers = len(self.layers)
        lrs = self._cfg("foof_lrs", self._cfg("metric_gradient_lrs", None))
        if lrs is None:
            lrs = [layer.alpha for layer in self.layers]
        else:
            lrs = _as_layer_list(lrs, num_layers, name="foof_lrs")

        factor_ranks = _as_layer_list(
            self._cfg("foof_factor_rank", None),
            num_layers,
            name="foof_factor_rank",
        )

        norm = self._cfg("metric_norm_type", "Trace")
        mult = float(self._cfg("metric_norm_mult", 1.0))
        eps = float(self._cfg("foof_eps", self._cfg("metric_eps", 1e-12)))
        project_psd = bool(
            self._cfg(
                "foof_project_psd",
                self._cfg("metric_gradient_project_psd", True),
            )
        )

        diag_load = self._cfg(
            "foof_diag_load",
            self._cfg("metric_gradient_diag_load", 0.0),
        )
        if diag_load is not None:
            diag_load = float(diag_load)

        eigenclip_max = self._cfg("foof_eigenclip_max", None)

        beta1 = float(
            self._cfg(
                "foof_adam_beta1",
                self._cfg("metric_gradient_adam_beta1", 0.9),
            )
        )
        beta2 = float(
            self._cfg(
                "foof_adam_beta2",
                self._cfg("metric_gradient_adam_beta2", 0.999),
            )
        )
        adam_eps = float(
            self._cfg(
                "foof_adam_eps",
                self._cfg("metric_gradient_adam_eps", 1e-8),
            )
        )

        for layer_idx, layer in enumerate(self.layers):
            if layer.freeze:
                continue

            directions = self.pending_metric_updates[layer_idx]
            if len(directions) == 0:
                raise RuntimeError(
                    "No pending FOOF update for a layer. "
                    "Call backward(...) before update_metrics(...)."
                )

            direction = torch.mean(torch.stack(directions, dim=0), dim=0)
            direction = torch.nan_to_num(
                direction,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            current_metric = layer.metric
            if current_metric is None:
                raise RuntimeError(
                    "Layer metric is None. Call set_centers before "
                    "update_metrics(...)."
                )

            current_metric = current_metric.to(
                device=direction.device,
                dtype=direction.dtype,
            )
            factor_w = _foof_metric_factor_from_metric(
                current_metric,
                rank=factor_ranks[layer_idx],
                eps=max(float(layer.metric_clamp or eps), eps),
                max_eval=eigenclip_max,
            ).detach()

            if factor_w.shape != direction.shape:
                raise RuntimeError(
                    f"FOOF factor shape {factor_w.shape} does not match "
                    f"pending direction shape {direction.shape} for "
                    f"layer {layer_idx}."
                )

            if optimizer == "sgd":
                step = direction

            elif optimizer == "adam":
                step = self._adam_foof_step(
                    layer_idx=layer_idx,
                    direction=direction,
                    beta1=beta1,
                    beta2=beta2,
                    adam_eps=adam_eps,
                )

            else:
                raise RuntimeError(f"Unsupported foof optimizer={optimizer!r}.")

            new_factor_w = factor_w - float(lrs[layer_idx]) * step
            new_metric = new_factor_w.T @ new_factor_w
            new_metric = self._postprocess_foof_metric(
                new_metric,
                layer=layer,
                norm=norm,
                mult=mult,
                eps=eps,
                project_psd=project_psd,
                diag_load=diag_load,
                eigenclip_max=eigenclip_max,
            )

            layer.update_metric(new_metric.detach())

        return self

    def update_metrics(
        self,
        *,
        norm=None,
        mult=None,
        eps=None,
        clear=True,
    ):
        """
        Apply metric matrices accumulated by backward(...).

        For AGOP, this averages stored AGOP matrices and applies the same
        convex metric update as average_agop(...). For gradient descent, this
        averages pending candidate metric matrices and writes them into each
        layer.
        """
        norm = self._cfg("metric_norm_type", "Trace") if norm is None else norm
        mult = float(self._cfg("metric_norm_mult", 1.0) if mult is None else mult)
        eps = float(self._cfg("metric_eps", 1e-12) if eps is None else eps)

        if self.metric_update_rule == "agop":
            self._apply_pending_agop_updates(norm=norm, mult=mult, eps=eps)

        elif self.metric_update_rule == "gradient_descent":
            self._apply_pending_gradient_updates()

        elif self.metric_update_rule == "foof":
            self._apply_pending_foof_updates()

        else:
            raise RuntimeError(
                f"Unsupported metric_update_rule={self.metric_update_rule!r}."
            )

        if self.pending_readout is not None:
            self.weight = self.pending_readout.detach()
            self.pending_readout = None

        if clear:
            self.clear_pending_metric_updates()
            if self.metric_update_rule == "agop":
                self.reset_agop()

        return self

    def update_matrices(self, **kwargs):
        """
        Alias for update_metrics(...).
        """
        return self.update_metrics(**kwargs)

    def _apply_pending_agop_updates(self, norm="Trace", mult=1.0, eps=1e-12):
        """
        Convert stored layer AGOPs into metric matrices and apply them.
        """
        for layer in self.layers:
            if len(layer.agops) == 0:
                raise RuntimeError(
                    "No AGOPs stored for a layer. "
                    "Call backward(...) before update_metrics(...)."
                )

            batch_agops = torch.stack(layer.agops, dim=0)
            avg_agop = torch.mean(batch_agops, dim=0)

            metric_update = psd_project(
                avg_agop,
                eps=layer.metric_clamp,
            )

            metric_update = self._normalize_metric_update(
                metric_update,
                norm=norm,
                mult=mult,
                eps=eps,
            )

            current_metric = layer.metric

            if current_metric is None:
                new_metric = metric_update
            else:
                current_metric = current_metric.to(
                    device=metric_update.device,
                    dtype=metric_update.dtype,
                )

                new_metric = (
                    (1.0 - layer.alpha) * current_metric
                    + layer.alpha * metric_update
                )

            new_metric = self._normalize_metric_update(
                new_metric,
                norm=norm,
                mult=mult,
                eps=eps,
            )

            layer.update_metric(new_metric.detach())

        return self

    @staticmethod
    def _normalize_metric_update(metric, *, norm="Trace", mult=1.0, eps=1e-12):
        """
        Normalize a metric matrix using the selected convention.
        """
        if norm == "Trace":
            trace = torch.trace(metric).clamp_min(eps)
            return metric * (float(mult) * metric.shape[0] / trace)

        if norm == "Frob":
            frob = torch.sqrt(torch.sum(metric ** 2)).clamp_min(eps)
            return metric / frob

        if norm is None:
            return metric

        raise ValueError("norm must be one of {'Trace', 'Frob', None}.")

    def _apply_pending_gradient_updates(self):
        """
        Apply pending gradient-descent updates using SGD or Adam.

        backward(...) stores raw gradients. This method averages any gradients
        accumulated since the previous update, applies the optimizer step to the
        current metric matrix, then applies optional diagonal loading, PSD
        projection, and metric normalization.
        """
        optimizer = str(self._cfg("metric_gradient_optimizer", "sgd")).lower()
        if optimizer not in {"sgd", "adam"}:
            raise ValueError(
                "metric_gradient_optimizer must be one of {'sgd', 'adam'}. "
                f"Got {optimizer!r}."
            )

        num_layers = len(self.layers)
        lrs = self._cfg("metric_gradient_lrs", None)
        if lrs is None:
            lrs = [layer.alpha for layer in self.layers]
        else:
            lrs = _as_layer_list(lrs, num_layers, name="metric_gradient_lrs")

        norm = self._cfg("metric_norm_type", "Trace")
        mult = float(self._cfg("metric_norm_mult", 1.0))
        eps = float(self._cfg("metric_gradient_eps", 1e-12))
        project_psd = bool(self._cfg("metric_gradient_project_psd", True))
        max_grad_norm = float(self._cfg("metric_gradient_max_norm", 1.0))
        diag_load = self._cfg("metric_gradient_diag_load", 1e-8)
        if diag_load is not None:
            diag_load = float(diag_load)

        beta1 = float(self._cfg("metric_gradient_adam_beta1", 0.9))
        beta2 = float(self._cfg("metric_gradient_adam_beta2", 0.999))
        adam_eps = float(self._cfg("metric_gradient_adam_eps", 1e-8))

        for layer_idx, layer in enumerate(self.layers):
            grads = self.pending_metric_updates[layer_idx]

            if layer.freeze:
                continue

            if len(grads) == 0:
                raise RuntimeError(
                    "No pending gradient update for a layer. "
                    "Call backward(...) before update_metrics(...)."
                )

            grad = torch.mean(torch.stack(grads, dim=0), dim=0)
            grad = self._prepare_metric_gradient(
                grad,
                eps=eps,
                max_grad_norm=max_grad_norm,
            )

            current_metric = layer.metric
            if current_metric is None:
                raise RuntimeError(
                    "Layer metric is None. Call set_centers before "
                    "update_metrics(...)."
                )

            current_metric = current_metric.to(
                device=grad.device,
                dtype=grad.dtype,
            )
            current_metric = 0.5 * (current_metric + current_metric.T)

            lr = float(lrs[layer_idx])

            if optimizer == "sgd":
                step = grad

            elif optimizer == "adam":
                step = self._adam_metric_step(
                    layer_idx=layer_idx,
                    grad=grad,
                    beta1=beta1,
                    beta2=beta2,
                    adam_eps=adam_eps,
                )

            else:
                raise RuntimeError(f"Unsupported optimizer={optimizer!r}.")

            new_metric = current_metric - lr * step
            new_metric = self._postprocess_metric_after_optimizer_step(
                new_metric,
                layer=layer,
                norm=norm,
                mult=mult,
                eps=eps,
                project_psd=project_psd,
                diag_load=diag_load,
            )

            layer.update_metric(new_metric.detach())

        return self

    @staticmethod
    def _prepare_metric_gradient(grad, *, eps=1e-12, max_grad_norm=1.0):
        """
        Sanitize, symmetrize, and optionally clip a raw metric gradient.
        """
        grad = torch.nan_to_num(
            grad,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        grad = 0.5 * (grad + grad.T)

        if max_grad_norm is not None and max_grad_norm > 0:
            grad_norm = torch.linalg.norm(grad)
            if torch.isfinite(grad_norm) and grad_norm > max_grad_norm:
                grad = grad * (float(max_grad_norm) / grad_norm.clamp_min(eps))

        return grad

    def _adam_metric_step(
        self,
        *,
        layer_idx,
        grad,
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
    ):
        """
        Return Adam's bias-corrected update direction for one metric matrix.
        """
        state = self.metric_optimizer_state[layer_idx]

        if (
            state["m"] is None
            or state["m"].shape != grad.shape
            or state["m"].device != grad.device
            or state["m"].dtype != grad.dtype
        ):
            state["m"] = torch.zeros_like(grad)
            state["v"] = torch.zeros_like(grad)
            state["step"] = 0

        state["step"] += 1
        state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)

        step_idx = state["step"]
        m_hat = state["m"] / (1.0 - beta1 ** step_idx)
        v_hat = state["v"] / (1.0 - beta2 ** step_idx)

        step = m_hat / (torch.sqrt(v_hat) + adam_eps)
        step = 0.5 * (step + step.T)
        return step

    def _postprocess_metric_after_optimizer_step(
        self,
        metric,
        *,
        layer,
        norm="Trace",
        mult=1.0,
        eps=1e-12,
        project_psd=True,
        diag_load=1e-8,
    ):
        """
        Apply diagonal loading, PSD projection, and normalization after an
        optimizer step has produced a candidate metric matrix.
        """
        metric = torch.nan_to_num(
            metric,
            nan=0.0,
            posinf=1.0 / eps,
            neginf=-1.0 / eps,
        )
        metric = 0.5 * (metric + metric.T)

        if diag_load is not None and diag_load > 0:
            eye = torch.eye(
                metric.shape[0],
                device=metric.device,
                dtype=metric.dtype,
            )
            metric = metric + float(diag_load) * eye

        if project_psd:
            clamp_eps = layer.metric_clamp
            if clamp_eps is None:
                clamp_eps = eps

            metric = psd_project(metric, eps=clamp_eps)

        metric = self._normalize_metric_update(
            metric,
            norm=norm,
            mult=mult,
            eps=eps,
        )

        return metric.detach()

    def _prepare_metric_vars(self, *, caller):
        """
        Create differentiable metric variables for every non-frozen layer.
        """
        metric_vars = []

        for layer in self.layers:
            if layer.freeze:
                metric_vars.append(None)
                continue

            if layer.metric is None:
                raise RuntimeError(
                    f"Layer metric is None. Call set_centers before {caller}."
                )

            metric_var = layer.metric.detach().clone()
            metric_var = 0.5 * (metric_var + metric_var.T)
            metric_var.requires_grad_(True)
            metric_vars.append(metric_var)

        return metric_vars

    def _features_with_metric_vars(self, x, metric_vars, *, normalize):
        """
        Forward pass using differentiable metric variables.
        """
        z = x

        for layer, metric_var in zip(self.layers, metric_vars):
            if metric_var is None:
                z = layer.get_features(z, normalize=normalize)
            else:
                z = _spec_features_metric_grad_direct(
                    layer,
                    z,
                    metric_var,
                    normalize=normalize,
                )

        return z

    def _postprocess_gradient_metric(
        self,
        *,
        layer,
        metric_var,
        grad,
        lr,
        norm,
        mult,
        eps,
        project_psd,
        max_grad_norm,
        diag_load,
    ):
        """
        Convert a raw metric gradient into a candidate next metric matrix.
        """
        grad = torch.nan_to_num(
            grad,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        grad = 0.5 * (grad + grad.T)

        grad_norm = torch.linalg.norm(grad)

        if torch.isfinite(grad_norm) and grad_norm > max_grad_norm:
            grad = grad * (float(max_grad_norm) / grad_norm.clamp_min(eps))

        updated_metric = metric_var.detach() - float(lr) * grad
        updated_metric = 0.5 * (updated_metric + updated_metric.T)

        if diag_load is not None and diag_load > 0:
            eye = torch.eye(
                updated_metric.shape[0],
                device=updated_metric.device,
                dtype=updated_metric.dtype,
            )
            updated_metric = updated_metric + float(diag_load) * eye

        if project_psd:
            clamp_eps = layer.metric_clamp
            if clamp_eps is None:
                clamp_eps = eps

            updated_metric = psd_project(
                updated_metric,
                eps=clamp_eps,
            )

        if norm == "Trace":
            updated_metric = _trace_normalize_psd_metric(
                updated_metric,
                target_trace=mult * updated_metric.shape[0],
                eps=eps,
            )

        elif norm == "Frob":
            frob = torch.sqrt(torch.sum(updated_metric ** 2)).clamp_min(eps)
            updated_metric = updated_metric / frob

        elif norm is None:
            pass

        else:
            raise ValueError("norm must be one of {'Trace', 'Frob', None}.")

        return updated_metric.detach()

    def _compute_metric_gradients_fixed_readout(
        self,
        x,
        y,
        *,
        loss="mse",
        normalize=None,
        use_current_readout=True,
    ):
        """
        Compute raw metric gradients using the fixed-readout objective.

        The current readout is held fixed by default. No optimizer step, PSD
        projection, or metric normalization is applied here.
        """
        if self.weight is None:
            raise RuntimeError(
                "train_regressor must be called before backward(...)."
            )

        if loss != "mse":
            raise ValueError("Only loss='mse' is currently supported.")

        normalize = self._resolve_normalize(normalize)
        metric_vars = self._prepare_metric_vars(
            caller="backward(..., metric_update_rule='gradient_descent')"
        )

        features = self._features_with_metric_vars(
            x,
            metric_vars,
            normalize=normalize,
        )

        weight = self.weight.detach() if use_current_readout else self.weight
        predictions = features @ weight

        if predictions.shape != y.shape:
            predictions = predictions.reshape_as(y)

        objective = torch.mean((predictions - y) ** 2)
        active_metric_vars = [v for v in metric_vars if v is not None]

        grads = torch.autograd.grad(
            objective,
            active_metric_vars,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        gradients = [None for _ in self.layers]
        grad_iter = iter(grads)

        for layer_idx, metric_var in enumerate(metric_vars):
            if metric_var is None:
                continue
            gradients[layer_idx] = next(grad_iter).detach()

        return gradients

    def _compute_metric_gradients_full_mmse(
        self,
        x,
        y,
        *,
        reg_param=1e-16,
        onehot=False,
        loss="mse",
        normalize=None,
        store_differentiable_readout=False,
    ):
        """
        Compute raw metric gradients using the full-MMSE reduced objective.

        This differentiates through the closed-form ridge/MMSE readout solve.
        No optimizer step, PSD projection, or metric normalization is applied
        here.
        """
        if loss != "mse":
            raise ValueError("Only loss='mse' is currently supported.")

        normalize = self._resolve_normalize(normalize)
        metric_vars = self._prepare_metric_vars(
            caller="backward(..., metric_update_rule='gradient_descent')"
        )

        features = self._features_with_metric_vars(
            x,
            metric_vars,
            normalize=normalize,
        )

        num_samples = features.shape[0]
        feature_dim = features.shape[1]
        U = features.T @ features / num_samples

        ridge = float(reg_param) * torch.eye(
            feature_dim,
            device=features.device,
            dtype=features.dtype,
        )

        if onehot:
            targets = y

            if targets.ndim == 1:
                raise ValueError(
                    "onehot=True requires y to have shape (N, C), "
                    "but y is one-dimensional."
                )

            P = torch.mean(
                features.T.unsqueeze(2) * targets.unsqueeze(0),
                dim=1,
            )
            readout = torch.linalg.solve(U + ridge, P)
            predictions = features @ readout

        else:
            targets = y.reshape(-1)
            P = torch.mean(features.T * targets, dim=1)
            readout = torch.linalg.solve(U + ridge, P)
            predictions = features @ readout

        if predictions.shape != y.shape:
            predictions = predictions.reshape_as(y)

        objective = torch.mean((predictions - y) ** 2)
        active_metric_vars = [v for v in metric_vars if v is not None]

        grads = torch.autograd.grad(
            objective,
            active_metric_vars,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        gradients = [None for _ in self.layers]
        grad_iter = iter(grads)

        for layer_idx, metric_var in enumerate(metric_vars):
            if metric_var is None:
                continue
            gradients[layer_idx] = next(grad_iter).detach()

        maybe_readout = readout.detach() if store_differentiable_readout else None
        return gradients, maybe_readout

    def update_metric_error_gradient(
        self,
        x,
        y,
        *,
        lrs=None,
        loss="mse",
        normalize=True,
        norm="Trace",
        mult=1.0,
        eps=1e-12,
        project_psd=True,
        max_grad_norm=1.0,
        diag_load=1e-8,
        use_current_readout=True,
    ):
        """
        Update each layer metric matrix by direct gradient descent on prediction
        error.
    
        This version avoids differentiating through kernel.metric_factor(...) or
        eigendecompositions of the metric. Instead, during this update only, it
        computes the Mahalanobis RBF cross-kernel directly as
    
            d_M(c, x) = (c - x)^T M (c - x)
    
        so gradients with respect to M are stable even when M starts as identity.
        """
        if self.weight is None:
            raise RuntimeError(
                "train_regressor must be called before "
                "update_metric_error_gradient."
            )
    
        if loss != "mse":
            raise ValueError("Only loss='mse' is currently supported.")
    
        num_layers = len(self.layers)
    
        if lrs is None:
            lrs = [layer.alpha for layer in self.layers]
        else:
            lrs = _as_layer_list(lrs, num_layers, name="lrs")
    
        metric_vars = []
    
        for layer in self.layers:
            if layer.freeze:
                metric_vars.append(None)
                continue
    
            if layer.metric is None:
                raise RuntimeError(
                    "Layer metric is None. Call set_centers before "
                    "update_metric_error_gradient."
                )
    
            metric_var = layer.metric.detach().clone()
            metric_var = 0.5 * (metric_var + metric_var.T)
            metric_var.requires_grad_(True)
    
            metric_vars.append(metric_var)
    
        z = x
    
        for layer, metric_var in zip(self.layers, metric_vars):
            if metric_var is None:
                z = layer.get_features(z, normalize=normalize)
            else:
                z = _spec_features_metric_grad_direct(
                    layer,
                    z,
                    metric_var,
                    normalize=normalize,
                )
    
        weight = self.weight.detach() if use_current_readout else self.weight
    
        predictions = z @ weight
    
        if predictions.shape != y.shape:
            predictions = predictions.reshape_as(y)
    
        objective = torch.mean((predictions - y) ** 2)
    
        active_metric_vars = [
            metric_var for metric_var in metric_vars
            if metric_var is not None
        ]
    
        grads = torch.autograd.grad(
            objective,
            active_metric_vars,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    
        grad_iter = iter(grads)
    
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                metric_var = metric_vars[layer_idx]
    
                if metric_var is None:
                    continue
    
                grad = next(grad_iter)
    
                grad = torch.nan_to_num(
                    grad,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
    
                grad = 0.5 * (grad + grad.T)
    
                grad_norm = torch.linalg.norm(grad)
    
                if torch.isfinite(grad_norm) and grad_norm > max_grad_norm:
                    grad = grad * (
                        max_grad_norm / grad_norm.clamp_min(eps)
                    )
    
                lr = float(lrs[layer_idx])
    
                updated_metric = metric_var.detach() - lr * grad
                updated_metric = 0.5 * (updated_metric + updated_metric.T)
    
                if diag_load is not None and diag_load > 0:
                    eye = torch.eye(
                        updated_metric.shape[0],
                        device=updated_metric.device,
                        dtype=updated_metric.dtype,
                    )
                    updated_metric = updated_metric + float(diag_load) * eye
    
                if project_psd:
                    clamp_eps = layer.metric_clamp
                    if clamp_eps is None:
                        clamp_eps = eps
    
                    updated_metric = psd_project(
                        updated_metric,
                        eps=clamp_eps,
                    )
    
                if norm == "Trace":
                    updated_metric = _trace_normalize_psd_metric(
                        updated_metric,
                        target_trace=mult * updated_metric.shape[0],
                        eps=eps,
                    )
    
                elif norm == "Frob":
                    frob = torch.sqrt(
                        torch.sum(updated_metric ** 2)
                    ).clamp_min(eps)
    
                    updated_metric = updated_metric / frob
    
                elif norm is None:
                    pass
    
                else:
                    raise ValueError(
                        "norm must be one of {'Trace', 'Frob', None}."
                    )
    
                layer.update_metric(updated_metric.detach())
    
        return self

    
    def update_metric_error_gradient_full_mmse(
        self,
        x,
        y,
        *,
        lrs=None,
        reg_param=1e-16,
        onehot=False,
        loss="mse",
        normalize=True,
        norm="Trace",
        mult=1.0,
        eps=1e-12,
        project_psd=True,
        max_grad_norm=1.0,
        diag_load=1e-8,
        store_differentiable_readout=False,
    ):
        """
        Update each layer metric matrix using the gradient of the reduced
        ridge/MMSE objective.

        This differs from update_metric_error_gradient(...), which holds the
        current readout fixed. Here the readout is recomputed inside the
        differentiable graph:

            Phi_M = features under metric M

            w_star(M) = argmin_w mean(||Phi_M w - y||^2)
                        + reg_param ||w||^2

            loss(M) = mean(||Phi_M w_star(M) - y||^2)

        The metric gradient is then

            d loss(M) / d M_l

        for each layer l.

        Important approximation
        -----------------------
        This method accounts for the dependence of the closed-form readout on
        the metric. It still reuses the already-built spectral eigvecs/eigvals
        and feature-normalization constants as fixed objects. This avoids
        unstable gradients through eigendecompositions of the kernel matrix.
        """
        if loss != "mse":
            raise ValueError("Only loss='mse' is currently supported.")

        num_layers = len(self.layers)

        if lrs is None:
            lrs = [layer.alpha for layer in self.layers]
        else:
            lrs = _as_layer_list(lrs, num_layers, name="lrs")

        metric_vars = []

        for layer in self.layers:
            if layer.freeze:
                metric_vars.append(None)
                continue

            if layer.metric is None:
                raise RuntimeError(
                    "Layer metric is None. Call set_centers before "
                    "update_metric_error_gradient_full_mmse."
                )

            metric_var = layer.metric.detach().clone()
            metric_var = 0.5 * (metric_var + metric_var.T)
            metric_var.requires_grad_(True)

            metric_vars.append(metric_var)

        # Differentiable feature construction under the current metric vars.
        z = x

        for layer, metric_var in zip(self.layers, metric_vars):
            if metric_var is None:
                z = layer.get_features(z, normalize=normalize)
            else:
                z = _spec_features_metric_grad_direct(
                    layer,
                    z,
                    metric_var,
                    normalize=normalize,
                )

        features = z
        num_samples = features.shape[0]
        feature_dim = features.shape[1]

        # Differentiable closed-form ridge/MMSE solve.
        U = features.T @ features / num_samples

        ridge = float(reg_param) * torch.eye(
            feature_dim,
            device=features.device,
            dtype=features.dtype,
        )

        if onehot:
            targets = y

            if targets.ndim == 1:
                raise ValueError(
                    "onehot=True requires y to have shape (N, C), "
                    "but y is one-dimensional."
                )

            P = torch.mean(
                features.T.unsqueeze(2) * targets.unsqueeze(0),
                dim=1,
            )

            readout = torch.linalg.solve(U + ridge, P)
            predictions = features @ readout

        else:
            targets = y.reshape(-1)
            P = torch.mean(features.T * targets, dim=1)

            readout = torch.linalg.solve(U + ridge, P)
            predictions = features @ readout

        if predictions.shape != y.shape:
            predictions = predictions.reshape_as(y)

        objective = torch.mean((predictions - y) ** 2)

        active_metric_vars = [
            metric_var for metric_var in metric_vars
            if metric_var is not None
        ]

        grads = torch.autograd.grad(
            objective,
            active_metric_vars,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        grad_iter = iter(grads)

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                metric_var = metric_vars[layer_idx]

                if metric_var is None:
                    continue

                grad = next(grad_iter)

                grad = torch.nan_to_num(
                    grad,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                grad = 0.5 * (grad + grad.T)

                grad_norm = torch.linalg.norm(grad)

                if torch.isfinite(grad_norm) and grad_norm > max_grad_norm:
                    grad = grad * (
                        float(max_grad_norm) / grad_norm.clamp_min(eps)
                    )

                lr = float(lrs[layer_idx])

                updated_metric = metric_var.detach() - lr * grad
                updated_metric = 0.5 * (updated_metric + updated_metric.T)

                if diag_load is not None and diag_load > 0:
                    eye = torch.eye(
                        updated_metric.shape[0],
                        device=updated_metric.device,
                        dtype=updated_metric.dtype,
                    )
                    updated_metric = updated_metric + float(diag_load) * eye

                if project_psd:
                    clamp_eps = layer.metric_clamp
                    if clamp_eps is None:
                        clamp_eps = eps

                    updated_metric = psd_project(
                        updated_metric,
                        eps=clamp_eps,
                    )

                if norm == "Trace":
                    updated_metric = _trace_normalize_psd_metric(
                        updated_metric,
                        target_trace=mult * updated_metric.shape[0],
                        eps=eps,
                    )

                elif norm == "Frob":
                    frob = torch.sqrt(
                        torch.sum(updated_metric ** 2)
                    ).clamp_min(eps)

                    updated_metric = updated_metric / frob

                elif norm is None:
                    pass

                else:
                    raise ValueError(
                        "norm must be one of {'Trace', 'Frob', None}."
                    )

                layer.update_metric(updated_metric.detach())

            if store_differentiable_readout:
                self.weight = readout.detach()

        return self

    def average_agop(
        self,
        norm="Trace",
        mult=1.0,
        eps=1e-12,
    ):
        """
        Average stored AGOPs and update each layer metric.
        """
        for layer in self.layers:
            if len(layer.agops) == 0:
                raise RuntimeError(
                    "No AGOPs stored for a layer. "
                    "Call calc_agop before average_agop."
                )

            batch_agops = torch.stack(layer.agops, dim=0)
            avg_agop = torch.mean(batch_agops, dim=0)

            metric_update = psd_project(
                avg_agop,
                eps=layer.metric_clamp,
            )

            if norm == "Trace":
                trace = torch.trace(metric_update).clamp_min(eps)
                metric_update = metric_update * (
                    mult * metric_update.shape[0] / trace
                )

            elif norm == "Frob":
                frob = torch.sqrt(torch.sum(metric_update ** 2)).clamp_min(eps)
                metric_update = metric_update / frob

            elif norm is None:
                pass

            else:
                raise ValueError("norm must be one of {'Trace', 'Frob', None}.")

            current_metric = layer.metric

            if current_metric is None:
                new_metric = metric_update
            else:
                current_metric = current_metric.to(
                    device=metric_update.device,
                    dtype=metric_update.dtype,
                )

                new_metric = (
                    (1.0 - layer.alpha) * current_metric
                    + layer.alpha * metric_update
                )

            if norm == "Trace":
                trace = torch.trace(new_metric).clamp_min(eps)
                new_metric = new_metric * (
                    mult * new_metric.shape[0] / trace
                )

            layer.update_metric(new_metric)

        return self

    def reset_agop(self):
        for layer in self.layers:
            layer.reset_agop()

        return self
    
    def summary(self):
        """
        Print a concise summary of the RPM architecture and layer metrics.
        """
        num_layers = len(self.layers)
        alphas = [layer.alpha for layer in self.layers]
    
        print("=" * 80)
        print("RPM Summary")
        print("=" * 80)
        print(f"number of layers: {num_layers}")
        print(f"alphas: {alphas}")
        print("-" * 80)
    
        for layer_idx, layer in enumerate(self.layers):
            metric = layer.metric
    
            if metric is None:
                input_dim = None
            else:
                input_dim = metric.shape[0]
    
            output_dim = layer.output_dim
            kernel_size = layer.kernel.sigma
    
            print(f"Layer[{layer_idx}]:")
            print(f"    input dim: {input_dim}")
            print(f"    output dim: {output_dim}")
            print(f"    kernel size: {kernel_size}")
    
        print("=" * 80)
    
    def metric_projected_features(
        self,
        x,
        *,
        layer_idx=None,
        normalize=True,
        descending=True,
        eps=0.0,
        ):
        """
        Project each layer input into that layer's metric eigenvectors.
    
        This is a diagnostic method for studying representational collapse.
    
        For layer l, let z_l be the input to that layer and M_l be the metric
        matrix. This method computes:
    
            projected_l = z_l @ Q_l
    
        where
    
            M_l = Q_l diag(lambda_l) Q_l.T
    
        The projected features are NOT scaled or normalized by eigenvalues.
        The corresponding eigenvalues are returned separately.
    
        Parameters
        ----------
        x:
            Input data.
    
        layer_idx:
            If None, return projections/eigenvalues for every layer.
    
            If an int, return only the diagnostic projection for that layer.
    
        normalize:
            Controls the normal RPM feature propagation between layers.
            This should usually be True so that layer inputs match the normal
            training/evaluation pipeline.
    
        descending:
            If True, sort eigenvalues/eigenvectors from largest to smallest.
    
        eps:
            Optional eigenvalue floor for reporting only. Use eps=0.0 to report
            raw eigenvalues.
    
        Returns
        -------
        If layer_idx is None:
    
            projected_features_by_layer, eigenvalues_by_layer
    
            where each is a list with one entry per layer.
    
        If layer_idx is an int:
    
            projected_features, eigenvalues
    
        Notes
        -----
        For the first layer, z_l is just the original input x.
    
        For deeper layers, z_l is the usual RPM feature representation produced
        by the previous layer. This means the diagnostic projection is applied
        to the actual representation that enters each layer.
        """
        if layer_idx is not None:
            if layer_idx < 0:
                layer_idx = len(self.layers) + layer_idx
    
            if layer_idx < 0 or layer_idx >= len(self.layers):
                raise IndexError(
                    f"layer_idx={layer_idx} is out of range for "
                    f"{len(self.layers)} layers."
                )
    
        z = x
    
        projected_features_by_layer = []
        eigenvalues_by_layer = []
    
        for current_layer_idx, layer in enumerate(self.layers):
            metric = layer.metric
    
            if metric is None:
                raise RuntimeError(
                    f"Layer {current_layer_idx} has no metric matrix."
                )
    
            metric = metric.to(device=z.device, dtype=z.dtype)
            metric = 0.5 * (metric + metric.T)
    
            eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    
            if descending:
                order = torch.arange(
                    eigenvalues.numel() - 1,
                    -1,
                    -1,
                    device=eigenvalues.device,
                )
    
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
    
            if eps is not None and eps > 0:
                eigenvalues = torch.clamp(eigenvalues, min=eps)
    
            projected_features = z @ eigenvectors
    
            projected_features_by_layer.append(projected_features)
            eigenvalues_by_layer.append(eigenvalues)
    
            # Propagate normally to get the actual input to the next RPM layer.
            if current_layer_idx < len(self.layers) - 1:
                z = layer.get_features(z, normalize=normalize)
    
        if layer_idx is not None:
            return (
                projected_features_by_layer[layer_idx],
                eigenvalues_by_layer[layer_idx],
            )
    
        return projected_features_by_layer, eigenvalues_by_layer
    def plot_layer_prior_realizations(
        self,
        train_mean=None,
        train_std=None,
        *,
        sample_points=None,
        layer_idx=0,
        num_points=200,
        num_realizations=5,
        normalize=True,
        jitter=1e-6,
        chunksize=None,
        seed=None,
        figsize=(8, 4),
        title=None,
        plot_mean=False,
        show=True,
        ax=None,
        return_samples=True,
    ):
        """
        Plot GP prior realizations induced by one RPM layer's Mahalanobis kernel.
    
        This is a diagnostic method.
    
        There are two ways to choose the GP input locations:
    
        1. Synthetic locations:
               Provide train_mean and train_std.
               The method samples x ~ N(train_mean, train_std^2).
    
        2. Actual locations:
               Provide sample_points.
               These are used directly as the GP input locations.
    
        For layer_idx > 0, the chosen original-space locations are propagated
        through layers 0, ..., layer_idx - 1 before sampling from the prior of
        the requested layer.
    
        Parameters
        ----------
        train_mean:
            Mean of the original training inputs. Used only when
            sample_points is None.
    
        train_std:
            Standard deviation of the original training inputs. Used only when
            sample_points is None.
    
        sample_points:
            Optional actual input locations. Shape:
    
                (num_points, original_input_dim)
    
            If provided, these replace the synthetic train_mean/train_std
            sampling mode.
    
        layer_idx:
            Which RPM layer's prior to sample from.
    
        num_points:
            Number of synthetic input locations to sample. Ignored when
            sample_points is provided.
    
        num_realizations:
            Number of independent prior functions to draw.
    
        normalize:
            Whether to use normalized RPM features when propagating to deeper
            layers. This should usually match training/evaluation.
    
        jitter:
            Diagonal loading added to the Gram matrix before sampling.
    
        chunksize:
            Chunk size for kernel evaluation. If None, uses the selected
            layer's chunk_size.
    
        seed:
            Optional random seed.
    
        figsize:
            Matplotlib figure size.
    
        title:
            Optional plot title.
    
        plot_mean:
            If False, plot individual prior realizations.
    
            If True, plot the empirical mean function across sampled
            realizations and shade +/- one empirical standard deviation using
            plt.fill_between.
    
        show:
            If True, call plt.show().
    
        ax:
            Optional matplotlib axis. If None, a new figure/axis is created.
    
        return_samples:
            If True, return a dictionary containing samples, locations, and K.
    
        Returns
        -------
        If return_samples=True, returns a dict with:
    
            samples:
                Tensor of shape (num_realizations, num_points).
    
            x_locations:
                Original-space input locations used to generate the layer
                inputs. Shape (num_points, original_input_dim).
    
            layer_inputs:
                Tensor of shape (num_points, input_dim_to_requested_layer).
    
            gram_matrix:
                Kernel Gram matrix of shape (num_points, num_points).
    
            eigenvalues:
                Eigenvalues of the stabilized Gram matrix.
    
            fig, ax:
                Matplotlib figure and axis.
        """
        import matplotlib.pyplot as plt
    
        if layer_idx < 0:
            layer_idx = len(self.layers) + layer_idx
    
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError(
                f"layer_idx={layer_idx} is out of range for "
                f"{len(self.layers)} layers."
            )
    
        layer = self.layers[layer_idx]
    
        if layer.metric is None:
            raise RuntimeError(
                f"Layer {layer_idx} has no metric matrix."
            )
    
        device = layer.metric.device
        dtype = layer.metric.dtype
    
        generator = None
    
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
    
        # ------------------------------------------------------------------
        # Choose original-space input locations.
        # ------------------------------------------------------------------
        if sample_points is not None:
            x_locations = torch.as_tensor(
                sample_points,
                device=device,
                dtype=dtype,
            )
    
            if x_locations.ndim != 2:
                raise ValueError(
                    "sample_points must have shape "
                    "(num_points, original_input_dim). "
                    f"Got shape {tuple(x_locations.shape)}."
                )
    
            num_points = x_locations.shape[0]
            input_dim = x_locations.shape[1]
    
        else:
            if train_mean is None or train_std is None:
                raise ValueError(
                    "Either sample_points must be provided, or both "
                    "train_mean and train_std must be provided."
                )
    
            train_mean = torch.as_tensor(
                train_mean,
                device=device,
                dtype=dtype,
            ).reshape(-1)
    
            train_std = torch.as_tensor(
                train_std,
                device=device,
                dtype=dtype,
            ).reshape(-1)
    
            if train_mean.shape != train_std.shape:
                raise ValueError(
                    f"train_mean and train_std must have the same shape. "
                    f"Got {tuple(train_mean.shape)} and "
                    f"{tuple(train_std.shape)}."
                )
    
            input_dim = train_mean.numel()
    
            standard_normal = torch.randn(
                num_points,
                input_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
    
            x_locations = (
                train_mean.view(1, -1)
                + standard_normal * train_std.view(1, -1)
            )
    
        # Validate original input dimension against the first layer metric.
        if self.layers[0].metric is not None:
            expected_input_dim = self.layers[0].metric.shape[0]
    
            if input_dim != expected_input_dim:
                raise ValueError(
                    f"Input locations have input_dim={input_dim}, "
                    f"but layer 0 metric expects input_dim={expected_input_dim}."
                )
    
        # ------------------------------------------------------------------
        # Propagate original-space locations to the requested layer input space.
        # ------------------------------------------------------------------
        z = x_locations
    
        for prev_layer_idx in range(layer_idx):
            prev_layer = self.layers[prev_layer_idx]
    
            if prev_layer.spec is None:
                raise RuntimeError(
                    f"Layer {prev_layer_idx} has no spectral object. "
                    "Call model.set_centers(...) before this diagnostic method."
                )
    
            z = prev_layer.get_features(z, normalize=normalize)
    
        layer_inputs = z
    
        # ------------------------------------------------------------------
        # Build the GP prior covariance from the selected layer kernel.
        # ------------------------------------------------------------------
        if chunksize is None:
            chunksize = layer.chunk_size
    
        gram_matrix = layer.kernel(
            layer_inputs,
            layer_inputs,
            gram=True,
            chunksize=chunksize,
        )
    
        gram_matrix = 0.5 * (gram_matrix + gram_matrix.T)
    
        eye = torch.eye(
            num_points,
            device=device,
            dtype=dtype,
        )
    
        covariance = gram_matrix + float(jitter) * eye
    
        # Prefer Cholesky, fall back to eigendecomposition if needed.
        try:
            factor = torch.linalg.cholesky(covariance)
            eigenvalues = torch.linalg.eigvalsh(covariance)
    
        except torch._C._LinAlgError:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            eigenvalues = torch.clamp(eigenvalues, min=float(jitter))
    
            covariance = (
                eigenvectors * eigenvalues.unsqueeze(0)
            ) @ eigenvectors.T
    
            covariance = 0.5 * (covariance + covariance.T)
    
            factor = torch.linalg.cholesky(covariance)
    
        # ------------------------------------------------------------------
        # Draw GP prior realizations.
        # ------------------------------------------------------------------
        standard_functions = torch.randn(
            num_realizations,
            num_points,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    
        samples = standard_functions @ factor.T
    
        # ------------------------------------------------------------------
        # Plot.
        # ------------------------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
    
        sample_index = torch.arange(num_points).detach().cpu().numpy()
    
        if plot_mean:
            mean_function = samples.mean(dim=0)
            std_function = samples.std(dim=0, unbiased=False)
    
            mean_np = mean_function.detach().cpu().numpy()
            std_np = std_function.detach().cpu().numpy()
    
            ax.plot(
                sample_index,
                mean_np,
                linewidth=2.0,
                label="Mean prior realization",
            )
    
            ax.fill_between(
                sample_index,
                mean_np - std_np,
                mean_np + std_np,
                alpha=0.25,
                label="+/- 1 std",
            )
    
            ax.legend()
    
        else:
            samples_np = samples.detach().cpu().numpy()
    
            for realization_idx in range(num_realizations):
                ax.plot(
                    sample_index,
                    samples_np[realization_idx],
                    linewidth=1.5,
                    alpha=0.85,
                )
    
        if title is None:
            if sample_points is not None:
                location_mode = "training locations"
            else:
                location_mode = "synthetic locations"
    
            if plot_mean:
                title = (
                    f"Layer {layer_idx} GP prior mean +/- std "
                    f"on {location_mode}"
                )
            else:
                title = (
                    f"Layer {layer_idx} GP prior samples "
                    f"on {location_mode}"
                )
    
        ax.set_title(title)
        ax.set_xlabel("Input location index")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
    
        fig.tight_layout()
    
        if show:
            plt.show()
    
        if return_samples:
            return {
                "samples": samples,
                "x_locations": x_locations,
                "layer_inputs": layer_inputs,
                "gram_matrix": gram_matrix,
                "eigenvalues": eigenvalues,
                "fig": fig,
                "ax": ax,
            }
    
        return None
    
    def plot_layer_posterior_realizations(
        self,
        x_train,
        y_train,
        *,
        sample_points=None,
        layer_idx=0,
        noise_var=1e-6,
        num_realizations=5,
        normalize=True,
        jitter=1e-6,
        chunksize=None,
        seed=None,
        figsize=(8, 4),
        title=None,
        plot_mean=False,
        show=True,
        ax=None,
        return_samples=True,
    ):
        """
        Plot GP posterior realizations induced by one RPM layer's Mahalanobis
        kernel, conditioned on training data.
    
        This is a diagnostic method.
    
        For a selected RPM layer l, this method:
            1. propagates x_train into the input space of layer l,
            2. propagates sample_points into the input space of layer l,
               or uses x_train as sample_points if sample_points is None,
            3. builds the GP posterior using the selected layer's kernel,
            4. samples posterior functions at the sample locations,
            5. plots either individual posterior realizations or posterior
               mean +/- one posterior standard deviation.
    
        Important
        ---------
        This does NOT use the RPM linear readout. It treats the selected layer's
        Mahalanobis kernel as a standalone GP covariance and performs ordinary
        GP regression:
    
            f ~ GP(0, k_l)
    
            y_train = f(z_train) + noise
    
        where z_train is x_train propagated to the input space of layer l.
    
        Parameters
        ----------
        x_train:
            Training inputs in the original RPM input space.
            Shape: (num_train, original_input_dim).
    
        y_train:
            Training targets. Currently intended for scalar regression.
            Shape: (num_train,) or (num_train, 1).
    
        sample_points:
            Optional points in the original RPM input space where posterior
            samples are drawn.
    
            If None, posterior samples are drawn at x_train.
    
        layer_idx:
            RPM layer whose Mahalanobis kernel defines the GP posterior.
    
        noise_var:
            Observation noise variance added to K_train.
    
        num_realizations:
            Number of posterior functions to sample.
    
        normalize:
            Whether to use normalized RPM features when propagating through
            earlier layers. This should usually match training/evaluation.
    
        jitter:
            Numerical diagonal loading added to covariance matrices.
    
        chunksize:
            Chunk size for kernel evaluation. If None, uses the selected
            layer's chunk_size.
    
        seed:
            Optional random seed.
    
        figsize:
            Matplotlib figure size.
    
        title:
            Optional plot title.
    
        plot_mean:
            If False, plot individual posterior realizations.
    
            If True, plot posterior mean with +/- one posterior standard
            deviation using fill_between.
    
        show:
            If True, call plt.show().
    
        ax:
            Optional matplotlib axis. If None, a new figure/axis is created.
    
        return_samples:
            If True, return a dictionary containing posterior samples,
            posterior mean/covariance, layer inputs, and figure objects.
    
        Returns
        -------
        If return_samples=True, returns a dict with:
    
            samples:
                Tensor of shape (num_realizations, num_sample_points).
    
            posterior_mean:
                Tensor of shape (num_sample_points,).
    
            posterior_covariance:
                Tensor of shape (num_sample_points, num_sample_points).
    
            posterior_std:
                Tensor of shape (num_sample_points,).
    
            train_layer_inputs:
                x_train propagated to the selected layer input space.
    
            sample_layer_inputs:
                sample_points propagated to the selected layer input space.
    
            fig, ax:
                Matplotlib figure and axis.
        """
        import matplotlib.pyplot as plt
    
        if layer_idx < 0:
            layer_idx = len(self.layers) + layer_idx
    
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError(
                f"layer_idx={layer_idx} is out of range for "
                f"{len(self.layers)} layers."
            )
    
        layer = self.layers[layer_idx]
    
        if layer.metric is None:
            raise RuntimeError(
                f"Layer {layer_idx} has no metric matrix."
            )
    
        device = layer.metric.device
        dtype = layer.metric.dtype
    
        x_train = torch.as_tensor(
            x_train,
            device=device,
            dtype=dtype,
        )
    
        y_train = torch.as_tensor(
            y_train,
            device=device,
            dtype=dtype,
        ).reshape(-1)
    
        if x_train.ndim != 2:
            raise ValueError(
                f"x_train must have shape (N, D). Got {tuple(x_train.shape)}."
            )
    
        if y_train.ndim != 1:
            raise ValueError(
                "y_train must be scalar regression targets with shape (N,) "
                "or broadcastable to (N,)."
            )
    
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"x_train and y_train have incompatible first dimensions: "
                f"{x_train.shape[0]} and {y_train.shape[0]}."
            )
    
        if sample_points is None:
            x_sample = x_train
        else:
            x_sample = torch.as_tensor(
                sample_points,
                device=device,
                dtype=dtype,
            )
    
            if x_sample.ndim != 2:
                raise ValueError(
                    "sample_points must have shape "
                    "(num_sample_points, original_input_dim). "
                    f"Got {tuple(x_sample.shape)}."
                )
    
            if x_sample.shape[1] != x_train.shape[1]:
                raise ValueError(
                    f"sample_points input dimension {x_sample.shape[1]} "
                    f"does not match x_train input dimension {x_train.shape[1]}."
                )
    
        if self.layers[0].metric is not None:
            expected_input_dim = self.layers[0].metric.shape[0]
    
            if x_train.shape[1] != expected_input_dim:
                raise ValueError(
                    f"x_train has input_dim={x_train.shape[1]}, "
                    f"but layer 0 metric expects input_dim={expected_input_dim}."
                )
    
        # --------------------------------------------------------------
        # Propagate original-space inputs to the selected layer input space.
        # --------------------------------------------------------------
        z_train = x_train
        z_sample = x_sample
    
        for prev_layer_idx in range(layer_idx):
            prev_layer = self.layers[prev_layer_idx]
    
            if prev_layer.spec is None:
                raise RuntimeError(
                    f"Layer {prev_layer_idx} has no spectral object. "
                    "Call model.set_centers(...) before this diagnostic method."
                )
    
            z_train = prev_layer.get_features(z_train, normalize=normalize)
            z_sample = prev_layer.get_features(z_sample, normalize=normalize)
    
        train_layer_inputs = z_train
        sample_layer_inputs = z_sample
    
        num_train = train_layer_inputs.shape[0]
        num_sample = sample_layer_inputs.shape[0]
    
        if chunksize is None:
            chunksize = layer.chunk_size
    
        # --------------------------------------------------------------
        # Build GP posterior.
        # --------------------------------------------------------------
        k_train = layer.kernel(
            train_layer_inputs,
            train_layer_inputs,
            gram=True,
            chunksize=chunksize,
        )
    
        k_cross = layer.kernel(
            sample_layer_inputs,
            train_layer_inputs,
            gram=True,
            chunksize=chunksize,
        )
    
        k_sample = layer.kernel(
            sample_layer_inputs,
            sample_layer_inputs,
            gram=True,
            chunksize=chunksize,
        )
    
        k_train = 0.5 * (k_train + k_train.T)
        k_sample = 0.5 * (k_sample + k_sample.T)
    
        train_eye = torch.eye(
            num_train,
            device=device,
            dtype=dtype,
        )
    
        sample_eye = torch.eye(
            num_sample,
            device=device,
            dtype=dtype,
        )
    
        k_train_noisy = (
            k_train
            + float(noise_var) * train_eye
            + float(jitter) * train_eye
        )
    
        try:
            train_chol = torch.linalg.cholesky(k_train_noisy)
    
        except torch._C._LinAlgError:
            eigvals, eigvecs = torch.linalg.eigh(k_train_noisy)
            eigvals = torch.clamp(eigvals, min=float(jitter))
    
            k_train_noisy = (
                eigvecs * eigvals.unsqueeze(0)
            ) @ eigvecs.T
    
            k_train_noisy = 0.5 * (k_train_noisy + k_train_noisy.T)
    
            train_chol = torch.linalg.cholesky(k_train_noisy)
    
        alpha = torch.cholesky_solve(
            y_train.reshape(-1, 1),
            train_chol,
        )
    
        posterior_mean = (k_cross @ alpha).reshape(-1)
    
        solved_cross = torch.cholesky_solve(
            k_cross.T,
            train_chol,
        )
    
        posterior_covariance = k_sample - k_cross @ solved_cross
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.T
        )
    
        posterior_covariance = posterior_covariance + float(jitter) * sample_eye
    
        try:
            posterior_chol = torch.linalg.cholesky(posterior_covariance)
            posterior_eigenvalues = torch.linalg.eigvalsh(posterior_covariance)
    
        except torch._C._LinAlgError:
            posterior_eigenvalues, posterior_eigenvectors = torch.linalg.eigh(
                posterior_covariance
            )
    
            posterior_eigenvalues = torch.clamp(
                posterior_eigenvalues,
                min=float(jitter),
            )
    
            posterior_covariance = (
                posterior_eigenvectors * posterior_eigenvalues.unsqueeze(0)
            ) @ posterior_eigenvectors.T
    
            posterior_covariance = 0.5 * (
                posterior_covariance + posterior_covariance.T
            )
    
            posterior_chol = torch.linalg.cholesky(posterior_covariance)
    
        posterior_std = torch.sqrt(
            torch.diagonal(posterior_covariance).clamp_min(0.0)
        )
    
        # --------------------------------------------------------------
        # Draw posterior function samples.
        # --------------------------------------------------------------
        generator = None
    
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
    
        standard_functions = torch.randn(
            num_realizations,
            num_sample,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    
        samples = posterior_mean.view(1, -1) + standard_functions @ posterior_chol.T
    
        # --------------------------------------------------------------
        # Plot.
        # --------------------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
    
        sample_index = torch.arange(num_sample).detach().cpu().numpy()
    
        if plot_mean:
            mean_np = posterior_mean.detach().cpu().numpy()
            std_np = posterior_std.detach().cpu().numpy()
    
            ax.plot(
                sample_index,
                mean_np,
                linewidth=2.0,
                label="Posterior mean",
            )
    
            ax.fill_between(
                sample_index,
                mean_np - std_np,
                mean_np + std_np,
                alpha=0.25,
                label="+/- 1 posterior std",
            )
    
            ax.legend()
    
        else:
            samples_np = samples.detach().cpu().numpy()
    
            for realization_idx in range(num_realizations):
                ax.plot(
                    sample_index,
                    samples_np[realization_idx],
                    linewidth=1.5,
                    alpha=0.85,
                )
    
        if title is None:
            if sample_points is None:
                location_mode = "training locations"
            else:
                location_mode = "sample locations"
    
            if plot_mean:
                title = (
                    f"Layer {layer_idx} GP posterior mean +/- std "
                    f"on {location_mode}"
                )
            else:
                title = (
                    f"Layer {layer_idx} GP posterior samples "
                    f"on {location_mode}"
                )
    
        ax.set_title(title)
        ax.set_xlabel("Input location index")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
    
        fig.tight_layout()
    
        if show:
            plt.show()
    
        if return_samples:
            return {
                "samples": samples,
                "posterior_mean": posterior_mean,
                "posterior_covariance": posterior_covariance,
                "posterior_std": posterior_std,
                "posterior_eigenvalues": posterior_eigenvalues,
                "train_layer_inputs": train_layer_inputs,
                "sample_layer_inputs": sample_layer_inputs,
                "k_train": k_train,
                "k_cross": k_cross,
                "k_sample": k_sample,
                "fig": fig,
                "ax": ax,
            }
    
        return None
#============END RPM Classes===============