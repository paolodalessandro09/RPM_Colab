#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utility functions for RPM experiments/classes.

This module contains reusable helper functions that are not class methods:
    - tensor/logging helpers
    - PSD projection
    - feature covariance and target-feature moments
    - alignment observables
    - Fisher observables
    - effective-dimension observables
    - predictive-weighted parameter variance
    - MSE / NMSE metrics
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import asdict, dataclass, field, is_dataclass
import torch


# =============================================================================
# Tensor / logging helpers
# =============================================================================

def to_cpu_detached(x: Any) -> Any:
    """
    Convert common torch / numpy-ish objects into CPU python-friendly objects
    for saving/logging, while preserving structure where reasonable.
    """
    if torch.is_tensor(x):
        return x.detach().cpu()

    if isinstance(x, (float, int, str, bool)) or x is None:
        return x

    if isinstance(x, dict):
        return {k: to_cpu_detached(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu_detached(v) for v in x)

    return x

def config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Convert a config object into a torch-saveable dictionary.
    """
    if config is None:
        return {}

    if is_dataclass(config):
        out = asdict(config)
    elif isinstance(config, dict):
        out = dict(config)
    else:
        out = dict(vars(config))

    return to_cpu_detached(out)


# =============================================================================
# Linear algebra helpers
# =============================================================================

# def psd_project(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     """
#     Project a symmetric matrix onto the PSD cone with eigenvalue floor eps.
#     """
#     matrix = 0.5 * (matrix + matrix.T)

#     eigvals, eigvecs = torch.linalg.eigh(matrix + torch.eye(matrix.size(0))*1e-3)
#     eigvals = torch.clamp(eigvals, min=eps)

#     return eigvecs @ torch.diag(eigvals) @ eigvecs.T

def psd_project(
    matrix: torch.Tensor,
    eps: float = 1e-12,
    jitter: float = 1e-10,
    max_tries: int = 6,
    fallback: str = "diagonal",
) -> torch.Tensor:
    """
    Project a symmetric matrix onto the PSD cone with eigenvalue floor eps.

    This version is more defensive than a raw torch.linalg.eigh call:
        1. removes NaN/Inf values,
        2. symmetrizes,
        3. adds increasing diagonal jitter if eigh fails,
        4. falls back to a positive diagonal matrix if projection still fails.

    Parameters
    ----------
    matrix:
        Square matrix to project.

    eps:
        Minimum eigenvalue after projection.

    jitter:
        Initial diagonal jitter used only if eigendecomposition fails.

    max_tries:
        Number of increasingly large jitter attempts.

    fallback:
        "diagonal" or "identity".

    Returns
    -------
    torch.Tensor
        PSD-projected matrix.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"psd_project expects a square matrix. Got shape {matrix.shape}."
        )

    dim = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype

    matrix = torch.nan_to_num(
        matrix,
        nan=0.0,
        posinf=1.0 / eps,
        neginf=-1.0 / eps,
    )

    matrix = 0.5 * (matrix + matrix.T)

    eye = torch.eye(dim, device=device, dtype=dtype)

    for attempt in range(max_tries):
        try:
            if attempt == 0:
                matrix_try = matrix
            else:
                jitter_value = jitter * (10.0 ** (attempt - 1))
                matrix_try = matrix + jitter_value * eye

            eigvals, eigvecs = torch.linalg.eigh(matrix_try)

            eigvals = torch.nan_to_num(
                eigvals,
                nan=eps,
                posinf=1.0 / eps,
                neginf=eps,
            )

            eigvals = torch.clamp(eigvals, min=eps)

            projected = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
            projected = 0.5 * (projected + projected.T)

            return projected

        except torch._C._LinAlgError:
            continue
        except RuntimeError:
            continue

    if fallback == "diagonal":
        diagonal = torch.diag(matrix)
        diagonal = torch.nan_to_num(
            diagonal,
            nan=eps,
            posinf=1.0 / eps,
            neginf=eps,
        )
        diagonal = torch.clamp(diagonal, min=eps)

        return torch.diag(diagonal)

    if fallback == "identity":
        return eps * eye

    raise RuntimeError(
        "PSD projection failed and no valid fallback was selected."
    )


# =============================================================================
# Feature statistics
# =============================================================================

def feature_covariance(features: torch.Tensor) -> torch.Tensor:
    """
    Compute C = E[phi(x) phi(x)^T].
    """
    return features.T @ features / features.shape[0]


def target_feature_moment(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute rho = E[phi(x) y].

    Parameters
    ----------
    features : torch.Tensor
        Feature matrix Phi with shape (N, M).

    targets : torch.Tensor
        Target vector with shape (N,), (N, 1), or compatible flattenable shape.

    Returns
    -------
    rho : torch.Tensor
        Vector with shape (M,).
    """
    targets = targets.reshape(-1).to(
        device=features.device,
        dtype=features.dtype,
    )

    if targets.numel() != features.shape[0]:
        raise ValueError(
            f"targets has {targets.numel()} entries, but features has "
            f"{features.shape[0]} rows."
        )

    return torch.mean(features.T * targets, dim=1)


# =============================================================================
# Alignment / MSE-decomposition metrics
# =============================================================================

def target_alignment(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    eval_features: torch.Tensor,
    eval_targets: torch.Tensor,
    *,
    train_covariance_inv: Optional[torch.Tensor] = None,
    train_covariance: Optional[torch.Tensor] = None,
    reg_param: float = 1e-16,
) -> torch.Tensor:
    """
    Target-alignment term:

        rho_eval^T (U + lambda I)^(-1) rho_train

    where:
        U         = E_train[phi phi^T]
        rho_train = E_train[phi y]
        rho_eval  = E_eval[phi y]
    """
    rho_train = target_feature_moment(train_features, train_targets)
    rho_eval = target_feature_moment(eval_features, eval_targets)

    if train_covariance_inv is not None:
        train_covariance_inv = train_covariance_inv.to(
            device=train_features.device,
            dtype=train_features.dtype,
        )

        return (
            rho_eval.reshape(1, -1)
            @ train_covariance_inv
            @ rho_train.reshape(-1, 1)
        ).squeeze()

    if train_covariance is None:
        train_covariance = feature_covariance(train_features)

    eye = torch.eye(
        train_covariance.shape[0],
        device=train_covariance.device,
        dtype=train_covariance.dtype,
    )

    solved = torch.linalg.solve(
        train_covariance + reg_param * eye,
        rho_train.reshape(-1, 1),
    )

    return (rho_eval.reshape(1, -1) @ solved).squeeze()


def excitation_alignment(
    eval_features: torch.Tensor,
    weights: torch.Tensor,
    *,
    eval_covariance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Excitation-alignment term:

        w^T C_eval w

    where:
        C_eval = E_eval[phi phi^T]
    """
    if eval_covariance is None:
        eval_covariance = feature_covariance(eval_features)

    weights = weights.to(
        device=eval_features.device,
        dtype=eval_features.dtype,
    )

    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)

    return (weights.T @ eval_covariance @ weights).squeeze()


# =============================================================================
# Fisher observables
# =============================================================================

def predictive_eigen_weights(
    covariance: torch.Tensor,
    rho: torch.Tensor,
    *,
    eps: float = 1e-12,
    symmetrize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute predictive eigendirection weights:

        a_i = u_i^T rho
        p_i = a_i^2 / sum_j a_j^2

    Returns p_i and covariance eigenvalues sorted descending.
    """
    if symmetrize:
        covariance = 0.5 * (covariance + covariance.T)

    eigvals, eigvecs = torch.linalg.eigh(covariance)

    idx = torch.argsort(eigvals, descending=True)
    eigvals = torch.clamp(eigvals[idx], min=0.0)
    eigvecs = eigvecs[:, idx]

    rho = rho.reshape(-1).to(
        device=covariance.device,
        dtype=covariance.dtype,
    )

    a = eigvecs.T @ rho
    a2 = a ** 2
    p = a2 / torch.sum(a2).clamp_min(eps)

    return p, eigvals


def arithmetic_fisher(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    covariance: Optional[torch.Tensor] = None,
    rho: Optional[torch.Tensor] = None,
    reg_param: float = 1e-16,
    eps: float = 1e-12,
    symmetrize: bool = True,
) -> torch.Tensor:
    """
    Arithmetic Fisher misalignment:

        sum_i p_i / (lambda_i + reg_param)
    """
    if covariance is None:
        covariance = feature_covariance(features)

    if rho is None:
        rho = target_feature_moment(features, targets)

    p, eigvals = predictive_eigen_weights(
        covariance,
        rho,
        eps=eps,
        symmetrize=symmetrize,
    )

    return torch.sum(p / (eigvals + reg_param))


def geometric_fisher(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    covariance: Optional[torch.Tensor] = None,
    rho: Optional[torch.Tensor] = None,
    reg_param: float = 1e-16,
    eps: float = 1e-12,
    symmetrize: bool = True,
) -> torch.Tensor:
    """
    Geometric Fisher misalignment:

        exp(-sum_i p_i log(lambda_i + reg_param))
    """
    if covariance is None:
        covariance = feature_covariance(features)

    if rho is None:
        rho = target_feature_moment(features, targets)

    p, eigvals = predictive_eigen_weights(
        covariance,
        rho,
        eps=eps,
        symmetrize=symmetrize,
    )

    return torch.exp(-torch.sum(p * torch.log(eigvals + reg_param)))


def fisher_metrics(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    covariance: Optional[torch.Tensor] = None,
    rho: Optional[torch.Tensor] = None,
    reg_param: float = 1e-16,
    eps: float = 1e-12,
    symmetrize: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper for the Fisher observables.
    """
    if covariance is None:
        covariance = feature_covariance(features)

    if rho is None:
        rho = target_feature_moment(features, targets)

    p, eigvals = predictive_eigen_weights(
        covariance,
        rho,
        eps=eps,
        symmetrize=symmetrize,
    )

    fisher_arith = torch.sum(p / (eigvals + reg_param))
    fisher_geom = torch.exp(-torch.sum(p * torch.log(eigvals + reg_param)))

    return {
        "fisher_arith": fisher_arith,
        "fisher_geom": fisher_geom,
        "predictive_weights": p,
        "cov_eigvals": eigvals,
        "rho_norm2": torch.sum(rho ** 2),
    }


# =============================================================================
# Effective dimension observables
# =============================================================================

def covariance_effective_dim(
    covariance: torch.Tensor,
    *,
    eps: float = 1e-12,
    symmetrize: bool = True,
) -> torch.Tensor:
    """
    Effective dimension of a covariance matrix:

        q_i = lambda_i / sum_j lambda_j
        d_eff = exp(-sum_i q_i log(q_i))
    """
    if symmetrize:
        covariance = 0.5 * (covariance + covariance.T)

    eigvals = torch.linalg.eigvalsh(covariance)
    eigvals = torch.clamp(eigvals, min=0.0)

    total = torch.sum(eigvals)

    if total <= eps:
        return torch.zeros((), device=covariance.device, dtype=covariance.dtype)

    q = torch.clamp(eigvals / total, min=eps)

    return torch.exp(-torch.sum(q * torch.log(q)))


def feature_effective_dim(
    features: torch.Tensor,
    *,
    covariance: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    symmetrize: bool = True,
) -> torch.Tensor:
    """
    Effective dimension of a feature representation.
    """
    if covariance is None:
        covariance = feature_covariance(features)

    return covariance_effective_dim(
        covariance,
        eps=eps,
        symmetrize=symmetrize,
    )


# =============================================================================
# Predictive-weighted parameter variance
# =============================================================================

def predictive_weighted_param_variance(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    covariance: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    unbiased: bool = False,
    symmetrize: bool = True,
) -> torch.Tensor:
    """
    Return per-direction predictive-weighted parameter variance:

        p_i Var(theta_i)

    where theta_i is the learned weight projected onto covariance
    eigendirection u_i.

    Returns
    -------
    pi_var_theta : torch.Tensor
        Tensor with shape (num_features,).
    """
    if covariance is None:
        covariance = feature_covariance(train_features)

    if symmetrize:
        covariance = 0.5 * (covariance + covariance.T)

    eigvals, eigvecs = torch.linalg.eigh(covariance)

    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]

    rho = target_feature_moment(train_features, train_targets).to(
        device=covariance.device,
        dtype=covariance.dtype,
    )

    a = eigvecs.T @ rho
    p = (a ** 2) / torch.sum(a ** 2).clamp_min(eps)

    weights = weights.to(
        device=covariance.device,
        dtype=covariance.dtype,
    )

    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)

    if weights.ndim != 2:
        raise ValueError(
            "weights must have shape (num_trials, num_features) "
            "or (num_trials, num_features, 1)."
        )

    if weights.shape[1] != covariance.shape[0]:
        raise ValueError(
            f"weights has {weights.shape[1]} features, but covariance has "
            f"{covariance.shape[0]} features."
        )

    theta_eig = weights @ eigvecs
    var_theta = torch.var(theta_eig, dim=0, unbiased=unbiased)

    return p * var_theta


# =============================================================================
# Performance metrics
# =============================================================================

def mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared error:

        mean((y - yhat)^2)
    """
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1).to(
        device=predictions.device,
        dtype=predictions.dtype,
    )

    if predictions.numel() != targets.numel():
        raise ValueError(
            f"predictions has {predictions.numel()} entries, but targets has "
            f"{targets.numel()} entries."
        )

    return torch.mean((targets - predictions) ** 2)


def normalized_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-12,
    unbiased_var: bool = False,
) -> torch.Tensor:
    """
    Normalized mean squared error:

        MSE / Var(y)
    """
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1).to(
        device=predictions.device,
        dtype=predictions.dtype,
    )

    if predictions.numel() != targets.numel():
        raise ValueError(
            f"predictions has {predictions.numel()} entries, but targets has "
            f"{targets.numel()} entries."
        )

    target_var = torch.var(targets, unbiased=unbiased_var).clamp_min(eps)

    return torch.mean((targets - predictions) ** 2) / target_var


