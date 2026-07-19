#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:32:06 2026

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

#============SPEED APPROXIMATION===================
class MahalanobisRBFKernel:
    def __init__(self, sigma=1.0, metric_a=None, r_eps=None):
        """
        Mahalanobis RBF kernel.

        Parameters
        ----------
        sigma : float
            RBF bandwidth.

        metric_a : torch.Tensor or None
            Either:
                - a metric matrix A with shape (D, D), or
                - a precomputed metric factor F with shape (D, r).

            The distance is:

                ||x - y||_A^2 = ||xF - yF||_2^2

        r_eps : float or None
            If metric_a is PSD / low-rank, r_eps controls the eigenvalue cutoff.
            If r_eps is None, the metric matrix is treated as positive definite
            and Cholesky is used.
        """
        self.sigma = sigma
        self.metric_a = metric_a
        self.r_eps = r_eps
        self.kernel_type = "rbf"

    @staticmethod
    def prepare_metric_factor(metric_a, r_eps=None):
        """
        Precompute factor F such that:

            ||x - y||_A^2 = ||xF - yF||_2^2

        Returns
        -------
        F : torch.Tensor or None
            Shape (D, r), or None if the metric is numerically zero.
        """
        if r_eps is None:
            return torch.linalg.cholesky(metric_a)

        eigvals, eigvecs = torch.linalg.eigh(metric_a)
        eigvals = torch.clamp(eigvals, min=0.0)

        max_eigval = torch.max(eigvals)

        if max_eigval <= 0:
            return None

        threshold = r_eps * max_eigval
        keep = eigvals >= threshold

        if not torch.any(keep):
            keep = torch.zeros_like(eigvals, dtype=torch.bool)
            keep[torch.argmax(eigvals)] = True

        kept_eigvals = eigvals[keep]
        kept_eigvecs = eigvecs[:, keep]

        return kept_eigvecs * torch.sqrt(kept_eigvals).unsqueeze(0)

    def metric_factor(self, r_eps=None):
        """
        Return the current metric factor F.

        If self.metric_a is already a rectangular factor, it is returned directly.
        If self.metric_a is square, it is factored.
        """
        if self.metric_a is None:
            return None

        if r_eps is None:
            r_eps = self.r_eps

        metric_a = self.metric_a

        if metric_a.dim() == 2 and metric_a.shape[0] == metric_a.shape[1]:
            return self.prepare_metric_factor(metric_a, r_eps=r_eps)

        return metric_a

    def transform(self, x, r_eps=None):
        """
        Apply the metric factor to x.

        If the metric is None or numerically zero, returns None.
        """
        factor = self.metric_factor(r_eps=r_eps)

        if factor is None:
            return None

        return x @ factor

    def distance(
        self,
        x,
        y,
        gram=False,
        chunksize=None,
        r_eps=None,
    ):
        """
        Squared Mahalanobis distance.

        Parameters
        ----------
        x : torch.Tensor
            Shape (Bx, D).

        y : torch.Tensor
            Shape (By, D).

        gram : bool
            If False, compute paired rowwise distances.
            If True, compute all pairwise distances.

        Returns
        -------
        distances : torch.Tensor
            If gram=False: shape (Bx,)
            If gram=True: shape (Bx, By)
        """
        factor = self.metric_factor(r_eps=r_eps)

        if factor is None:
            if gram:
                return torch.zeros(
                    x.size(0),
                    y.size(0),
                    device=x.device,
                    dtype=x.dtype,
                )

            return torch.zeros(
                x.size(0),
                device=x.device,
                dtype=x.dtype,
            )

        x_metric = x @ factor
        y_metric = y @ factor

        num_x = x.size(0)
        num_y = y.size(0)

        if not gram:
            if num_x != num_y:
                raise ValueError(
                    "When gram=False, x and y must have the same number of rows."
                )

            return torch.sum((x_metric - y_metric) ** 2, dim=1)

        y_norms = (y_metric ** 2).sum(dim=1, keepdim=True).T

        if chunksize is None:
            x_norms = (x_metric ** 2).sum(dim=1, keepdim=True)
            xy = x_metric @ y_metric.T

            return torch.clamp(x_norms + y_norms - 2 * xy, min=0.0)

        out = torch.empty(
            num_x,
            num_y,
            device=x.device,
            dtype=x.dtype,
        )

        for start in range(0, num_x, chunksize):
            x_block = x_metric[start:start + chunksize]

            x_norms = (x_block ** 2).sum(dim=1, keepdim=True)
            xy = x_block @ y_metric.T

            out[start:start + x_block.size(0)] = torch.clamp(
                x_norms + y_norms - 2 * xy,
                min=0.0,
            )

        return out

    def __call__(
        self,
        x,
        y,
        sigma=None,
        gram=True,
        chunksize=None,
        r_eps=None,
    ):
        """
        Compute the Mahalanobis RBF kernel.

        Parameters
        ----------
        x : torch.Tensor
            Shape (Bx, D).

        y : torch.Tensor
            Shape (By, D).

        sigma : float or None
            Optional override for self.sigma.

        gram : bool
            Included for compatibility with previous TorchSpec code.
            The kernel always returns the pairwise kernel matrix.

        Returns
        -------
        kernel_matrix : torch.Tensor
            Shape (Bx, By).
        """
        if sigma is None:
            sigma = self.sigma

        squared_distances = self.distance(
            x,
            y,
            gram=True,
            chunksize=chunksize,
            r_eps=r_eps,
        )

        return torch.exp(-squared_distances / (2 * sigma ** 2))
    

class MahalanobisLaplacianKernel(MahalanobisRBFKernel):
    def __init__(self, sigma=1.0, metric_a=None, r_eps=None):
        """
        Mahalanobis Laplacian kernel with the same constructor and call
        interface as MahalanobisRBFKernel.

        The kernel is

            k(x, y) = exp(-||x - y||_M / sigma)

        where

            ||x - y||_M = sqrt((x - y)^T M (x - y)).

        Parameters
        ----------
        sigma : float
            Laplacian length scale.

        metric_a : torch.Tensor or None
            Either a PSD metric matrix M with shape (D, D), or a rectangular
            metric factor F with shape (D, r). This matches the RBF class.

        r_eps : float or None
            Eigenvalue cutoff used when factorizing a PSD metric matrix.
        """
        super().__init__(sigma=sigma, metric_a=metric_a, r_eps=r_eps)
        self.kernel_type = "laplacian"

    def __call__(
        self,
        x,
        y,
        sigma=None,
        gram=True,
        chunksize=None,
        r_eps=None,
    ):
        """
        Compute the Mahalanobis Laplacian kernel.

        Parameters are intentionally identical to MahalanobisRBFKernel.__call__.
        The kernel always returns the pairwise kernel matrix with shape
        (x.shape[0], y.shape[0]).
        """
        if sigma is None:
            sigma = self.sigma

        squared_distances = self.distance(
            x,
            y,
            gram=True,
            chunksize=chunksize,
            r_eps=r_eps,
        )

        sigma = float(sigma)
        eps = 1e-12
        distances = torch.sqrt(torch.clamp(squared_distances, min=eps))
        return torch.exp(-distances / max(sigma, eps))


# Alias using the other natural word order.
LaplacianMahalanobisKernel = MahalanobisLaplacianKernel

# =============================================================================
# TorchSpecNystrom
# =============================================================================

class TorchSpecNystrom:
    def __init__(
        self,
        kernel,
        centers,
        full=True,
        num_eigs=128,
    ):
        """
        Spectral kernel feature wrapper.

        Supported approximation modes:
            - "full"
            - "nystrom"

        LOBPCG is intentionally not supported.

        Feature normalization is handled internally:
            - make_kernel(...) always fits feature normalization statistics
              from self.centers.
            - get_features(..., normalize=True) applies them.
            - normalize=True is the default.

        Parameters
        ----------
        kernel : object
            Kernel object with interface:

                kernel(x, y, gram=True, chunksize=None, r_eps=None)
                kernel.distance(x, y, gram=True, chunksize=None, r_eps=None)
                kernel.metric_factor(r_eps=None)

            The kernel owns the metric through kernel.metric_a.

        centers : torch.Tensor
            Centers/data matrix of shape (N, D).

        full : bool
            If True, stores original centers in self.full_centers when
            quantization is applied.

        num_eigs : int
            Number of spectral features/eigenpairs to keep.
        """
        self.kernel = kernel
        self.centers = centers

        self.weights = torch.ones(
            centers.size(0),
            device=centers.device,
            dtype=centers.dtype,
        )

        self.full = full
        self.num_eigs = int(num_eigs)

        self.device = centers.device
        self.dtype = centers.dtype

        self.approx_mode = None

        # Feature normalization state.
        self.feature_mean = None
        self.feature_scale = None

        # Full-mode spectral state.
        self.kernel_matrix = None
        self.eigvals = None
        self.eigvecs = None

        # Nyström spectral state.
        self.nystrom_indices = None
        self.nystrom_centers = None
        self.nystrom_eigvecs = None
        self.nystrom_eigvals = None
        self.nystrom_scale = None
        self.nystrom_weights = None
        self.nystrom_sqrt_weights = None

        self.quantized = False

    @staticmethod
    def _ensure_kernel_shape(num_centers, num_points, kernel_matrix):
        """
        Kernel(C, X) should be either:

            (num_centers, num_points)

        or:

            (num_points, num_centers)

        This returns the matrix in shape:

            (num_centers, num_points)
        """
        if kernel_matrix.shape == (num_centers, num_points):
            return kernel_matrix

        if kernel_matrix.shape == (num_points, num_centers):
            return kernel_matrix.T

        raise RuntimeError(
            f"Unexpected kernel shape {kernel_matrix.shape}, "
            f"expected {(num_centers, num_points)} or {(num_points, num_centers)}."
        )

    @staticmethod
    def _compute_feature_scale(features, eps=1e-12):
        """
        Compute scalar feature scale:

            scale = sqrt((1 / D) * trace(Phi.T @ Phi / N))

        This matches the previous training-loop normalization.
        """
        num_features = features.shape[1]

        scale = torch.sqrt(
            torch.trace(features.T @ features / features.shape[0])
            / num_features
        )

        return scale.clamp_min(eps)

    def _fit_feature_normalizer(
        self,
        num_features=None,
        feature_chunksize=500,
        kernel_chunksize=500,
        eps=1e-12,
        r_eps=None,
    ):
        """
        Fit feature normalization constants from self.centers.

        Stores:
            self.feature_mean  : shape (1, M)
            self.feature_scale : scalar tensor
        """
        if self.eigvals is None:
            raise RuntimeError("Cannot fit normalizer before make_kernel finishes.")

        if num_features is None:
            num_features = self.eigvals.numel()

        features = self.get_features(
            self.centers,
            num_features=num_features,
            feature_chunksize=feature_chunksize,
            kernel_chunksize=kernel_chunksize,
            r_eps=r_eps,
            normalize=False,
        )

        self.feature_mean = torch.mean(features, dim=0, keepdim=True)
        self.feature_scale = self._compute_feature_scale(features, eps=eps)

        return self.feature_mean, self.feature_scale

    def _normalize_features(self, features):
        """
        Apply stored feature normalization.
        """
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError(
                "Feature normalizer has not been fit. "
                "Call make_kernel(...) before get_features(..., normalize=True)."
            )

        mean = self.feature_mean.to(device=features.device, dtype=features.dtype)
        scale = self.feature_scale.to(device=features.device, dtype=features.dtype)

        return (features - mean) / scale

    def make_kernel(
        self,
        return_kernel=False,
        chunksize=None,
        *,
        method="full",
        eps=1e-12,
        nystrom_m=512,
        nystrom_seed=0,
        nystrom_mode="uniform",
        normalizer_feature_chunksize=500,
        normalizer_kernel_chunksize=None,
        r_eps=None,
    ):
        """
        Build the spectral kernel representation.

        This always fits feature normalization constants after building the
        spectral representation.

        Parameters
        ----------
        return_kernel : bool
            If True, returns kernel/eigenpairs.

        chunksize : int or None
            Chunk size passed to the kernel.

        method : {"full", "nystrom"}
            Approximation mode.

        eps : float
            Numerical floor.

        nystrom_m : int
            Number of Nyström landmarks.

        nystrom_seed : int or None
            Random seed for Nyström landmark selection.

        nystrom_mode : {"uniform", "weighted", "first"}
            Landmark sampling strategy.

        normalizer_feature_chunksize : int
            Feature chunk size used to fit normalization constants.

        normalizer_kernel_chunksize : int or None
            Kernel chunk size used to fit normalization constants.
            If None, defaults to chunksize.

        r_eps : float or None
            Optional metric-rank cutoff passed through to the kernel.
        """
        method = method.lower()

        if method not in {"full", "nystrom"}:
            raise ValueError("method must be either 'full' or 'nystrom'.")

        centers = self.centers
        num_centers = centers.shape[0]

        if num_centers < 1:
            raise ValueError("centers must contain at least one row.")

        num_features = min(self.num_eigs, max(num_centers - 1, 1))

        # Reset state.
        self.approx_mode = method

        self.feature_mean = None
        self.feature_scale = None

        self.kernel_matrix = None
        self.eigvals = None
        self.eigvecs = None

        self.nystrom_indices = None
        self.nystrom_centers = None
        self.nystrom_eigvecs = None
        self.nystrom_eigvals = None
        self.nystrom_scale = None
        self.nystrom_weights = None
        self.nystrom_sqrt_weights = None

        # ---------------------------------------------------------------------
        # Nyström mode
        # ---------------------------------------------------------------------
        if method == "nystrom":
            nystrom_m = min(int(nystrom_m), num_centers)

            if nystrom_m < 2:
                return self.make_kernel(
                    return_kernel=return_kernel,
                    chunksize=chunksize,
                    method="full",
                    eps=eps,
                    normalizer_feature_chunksize=normalizer_feature_chunksize,
                    normalizer_kernel_chunksize=normalizer_kernel_chunksize,
                    r_eps=r_eps,
                )

            if nystrom_mode == "uniform":
                generator = torch.Generator(device="cpu")
                if nystrom_seed is not None:
                    generator.manual_seed(int(nystrom_seed))

                indices_cpu = torch.randperm(
                    num_centers,
                    generator=generator,
                )[:nystrom_m]

                indices = indices_cpu.to(centers.device)

            elif nystrom_mode == "weighted":
                generator = torch.Generator(device="cpu")
                if nystrom_seed is not None:
                    generator.manual_seed(int(nystrom_seed))

                probs = self.weights.detach().cpu().double()
                probs = torch.clamp(probs, min=0.0)
                total = probs.sum()

                if total <= 0:
                    indices_cpu = torch.randperm(
                        num_centers,
                        generator=generator,
                    )[:nystrom_m]
                else:
                    probs = probs / total
                    indices_cpu = torch.multinomial(
                        probs,
                        num_samples=nystrom_m,
                        replacement=False,
                        generator=generator,
                    )

                indices = indices_cpu.to(centers.device)

            elif nystrom_mode == "first":
                indices = torch.arange(nystrom_m, device=centers.device)

            else:
                raise ValueError(
                    "nystrom_mode must be one of: 'uniform', 'weighted', 'first'."
                )

            landmarks = centers[indices]

            landmark_kernel = self.kernel(
                landmarks,
                landmarks,
                gram=True,
                chunksize=chunksize,
            )

            landmark_kernel = self._ensure_kernel_shape(
                nystrom_m,
                nystrom_m,
                landmark_kernel,
            )

            landmark_kernel = 0.5 * (landmark_kernel + landmark_kernel.T)

            eigvals_small, eigvecs_small = torch.linalg.eigh(landmark_kernel)

            eigvals_small = eigvals_small.flip(0)
            eigvecs_small = eigvecs_small[
                :,
                torch.arange(nystrom_m - 1, -1, -1, device=centers.device),
            ]

            num_features_eff = min(num_features, nystrom_m)

            eigvals_small = torch.clamp(
                eigvals_small[:num_features_eff],
                min=eps,
            )

            eigvecs_small = eigvecs_small[:, :num_features_eff]

            full_eigvals = (num_centers / nystrom_m) * eigvals_small

            landmark_weights = self.weights[indices].to(
                device=centers.device,
                dtype=centers.dtype,
            )

            self.approx_mode = "nystrom"

            self.eigvals = full_eigvals
            self.eigvecs = None
            self.kernel_matrix = None

            self.nystrom_indices = indices
            self.nystrom_centers = landmarks
            self.nystrom_eigvecs = eigvecs_small
            self.nystrom_eigvals = eigvals_small
            self.nystrom_weights = landmark_weights
            self.nystrom_sqrt_weights = torch.sqrt(
                torch.clamp(landmark_weights, min=eps)
            )
            self.nystrom_scale = torch.sqrt(
                torch.tensor(
                    nystrom_m / num_centers,
                    device=centers.device,
                    dtype=centers.dtype,
                )
            )

            self._fit_feature_normalizer(
                num_features=self.eigvals.numel(),
                feature_chunksize=normalizer_feature_chunksize,
                kernel_chunksize=normalizer_kernel_chunksize or chunksize,
                eps=eps,
                r_eps=r_eps,
            )

            if return_kernel:
                return None, self.eigvals, None

            return None

        # ---------------------------------------------------------------------
        # Full mode
        # ---------------------------------------------------------------------
        sqrt_weights = torch.sqrt(
            torch.clamp(
                self.weights.to(device=centers.device, dtype=centers.dtype),
                min=eps,
            )
        )

        kernel_matrix = self.kernel(
            centers,
            centers,
            gram=True,
            chunksize=chunksize,
        )

        kernel_matrix = self._ensure_kernel_shape(
            num_centers,
            num_centers,
            kernel_matrix,
        )

        kernel_matrix = (
            sqrt_weights[:, None]
            * kernel_matrix
            * sqrt_weights[None, :]
        )

        kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.T)

        eigvals, eigvecs = torch.linalg.eigh(kernel_matrix)

        eigvals = eigvals.flip(0)
        eigvecs = eigvecs[
            :,
            torch.arange(num_centers - 1, -1, -1, device=centers.device),
        ]

        if num_features < num_centers:
            eigvals = eigvals[:num_features]
            eigvecs = eigvecs[:, :num_features]

        eigvals = torch.clamp(eigvals, min=eps)

        self.approx_mode = "full"
        self.kernel_matrix = kernel_matrix
        self.eigvals = eigvals
        self.eigvecs = eigvecs

        self._fit_feature_normalizer(
            num_features=self.eigvals.numel(),
            feature_chunksize=normalizer_feature_chunksize,
            kernel_chunksize=normalizer_kernel_chunksize or chunksize,
            eps=eps,
            r_eps=r_eps,
        )

        if return_kernel:
            return kernel_matrix, eigvals, eigvecs

        return None

    def get_features(
        self,
        x,
        num_features,
        feature_chunksize=500,
        kernel_chunksize=500,
        r_eps=None,
        normalize=True,
    ):
        """
        Compute spectral kernel features.

        normalize=True by default.

        Use normalize=False to get raw spectral features.
        """
        if self.approx_mode is None:
            raise RuntimeError("make_kernel must be called before get_features.")

        if self.eigvals is None:
            raise RuntimeError("No eigenvalues found. Did make_kernel finish successfully?")

        num_features = min(int(num_features), self.eigvals.numel())
        batch_size = x.size(0)

        out = torch.empty(
            batch_size,
            num_features,
            device=x.device,
            dtype=x.dtype,
        )

        # ---------------------------------------------------------------------
        # Nyström mode
        # ---------------------------------------------------------------------
        if self.approx_mode == "nystrom":
            landmarks = self.nystrom_centers

            eigvecs = self.nystrom_eigvecs[:, :num_features].to(
                device=x.device,
                dtype=x.dtype,
            )

            eigvals = self.nystrom_eigvals[:num_features].to(
                device=x.device,
                dtype=x.dtype,
            )

            sqrt_landmark_weights = self.nystrom_sqrt_weights.to(
                device=x.device,
                dtype=x.dtype,
            ).view(1, -1)

            scale = self.nystrom_scale.to(
                device=x.device,
                dtype=x.dtype,
            )

            inv_sqrt_eigvals = (
                1.0 / torch.sqrt(eigvals)
            ).view(1, -1)

            for start in range(0, batch_size, feature_chunksize):
                xb = x[start:start + feature_chunksize]

                cross_kernel = self.kernel(
                    landmarks,
                    xb,
                    gram=True,
                    chunksize=kernel_chunksize,
                    r_eps=r_eps,
                )

                cross_kernel = self._ensure_kernel_shape(
                    landmarks.shape[0],
                    xb.shape[0],
                    cross_kernel,
                ).T

                weighted_cross_kernel = cross_kernel * sqrt_landmark_weights

                out[start:start + xb.size(0)] = scale * (
                    (weighted_cross_kernel @ eigvecs) * inv_sqrt_eigvals
                )

                del cross_kernel, weighted_cross_kernel

            if normalize:
                out = self._normalize_features(out)

            return out

        # ---------------------------------------------------------------------
        # Full mode
        # ---------------------------------------------------------------------
        if self.approx_mode == "full":
            sqrt_weights = torch.sqrt(
                torch.clamp(
                    self.weights.to(device=x.device, dtype=x.dtype),
                    min=1e-12,
                )
            ).view(1, -1)

            eigvecs = self.eigvecs[:, :num_features].to(
                device=x.device,
                dtype=x.dtype,
            )

            inv_sqrt_eigvals = (
                1.0 / torch.sqrt(self.eigvals[:num_features])
            ).to(device=x.device, dtype=x.dtype).view(1, -1)

            for start in range(0, batch_size, feature_chunksize):
                xb = x[start:start + feature_chunksize]

                cross_kernel = self.kernel(
                    self.centers,
                    xb,
                    gram=True,
                    chunksize=kernel_chunksize,
                    r_eps=r_eps,
                )

                cross_kernel = self._ensure_kernel_shape(
                    self.centers.shape[0],
                    xb.shape[0],
                    cross_kernel,
                ).T

                weighted_cross_kernel = cross_kernel * sqrt_weights

                out[start:start + xb.size(0)] = (
                    weighted_cross_kernel @ eigvecs
                ) * inv_sqrt_eigvals

                del cross_kernel, weighted_cross_kernel

            if normalize:
                out = self._normalize_features(out)

            return out

        raise RuntimeError(f"Unknown approx_mode={self.approx_mode}")

    def quantize_centers(
        self,
        q_thresh,
        return_centers=False,
        chunksize=2000,
        r_eps=None,
    ):
        """
        Greedy center quantization using self.kernel.distance(...).
        """
        centers = self.centers
        device = centers.device
        dtype = centers.dtype
        num_centers = centers.shape[0]

        alive = torch.ones(num_centers, device=device, dtype=torch.bool)

        representative_indices = []
        counts = []

        while alive.any():
            index = torch.nonzero(alive, as_tuple=False)[0, 0].item()
            center = centers[index:index + 1]

            alive_indices = torch.nonzero(alive, as_tuple=False).squeeze(1)
            num_alive = alive_indices.numel()

            close = torch.zeros(num_alive, device=device, dtype=torch.bool)

            for start in range(0, num_alive, chunksize):
                idx = alive_indices[start:start + chunksize]

                distances = self.kernel.distance(
                    center,
                    centers[idx],
                    gram=True,
                    chunksize=None,
                    r_eps=r_eps,
                ).squeeze()

                close[start:start + idx.numel()] = distances < q_thresh

            covered_indices = alive_indices[close]

            representative_indices.append(index)
            counts.append(float(covered_indices.numel()))

            alive[covered_indices] = False

        representative_indices = torch.tensor(
            representative_indices,
            device=device,
            dtype=torch.long,
        )

        quantized_centers = centers[representative_indices]

        if self.full:
            self.full_centers = self.centers

        self.centers = quantized_centers
        self.weights = torch.tensor(
            counts,
            device=device,
            dtype=dtype,
        )

        self.quantized = True
        self._clear_spectral_state()

        if return_centers:
            return quantized_centers, counts

        return None

    def quantize_centers_mahalanobis(
        self,
        q_thresh,
        return_centers=False,
        chunksize=2000,
        r_eps=None,
    ):
        """
        Greedy center quantization using the kernel's internal metric factor.
        """
        with torch.inference_mode():
            centers = self.centers.contiguous()
            device = centers.device
            dtype = centers.dtype
            num_centers = centers.shape[0]

            metric_factor = self.kernel.metric_factor(r_eps=r_eps)

            if metric_factor is None:
                return self.quantize_centers(
                    q_thresh,
                    return_centers=return_centers,
                    chunksize=chunksize,
                    r_eps=r_eps,
                )

            metric_factor = metric_factor.to(device=device, dtype=dtype)

            transformed_centers = (centers @ metric_factor).contiguous()
            transformed_norms = (transformed_centers ** 2).sum(dim=1)

            alive_indices = torch.arange(num_centers, device=device)

            representative_indices = []
            counts = []

            while alive_indices.numel() > 0:
                index = alive_indices[0]

                xi = transformed_centers[index]
                xi_norm = transformed_norms[index]

                num_alive = alive_indices.numel()
                keep_mask = torch.ones(
                    num_alive,
                    device=device,
                    dtype=torch.bool,
                )

                covered_count = 0

                for start in range(0, num_alive, chunksize):
                    idx = alive_indices[start:start + chunksize]
                    y = transformed_centers[idx]

                    distances = torch.clamp(
                        transformed_norms[idx] + xi_norm - 2 * (y @ xi),
                        min=0.0,
                    )

                    is_close = distances < q_thresh
                    keep_mask[start:start + idx.numel()] = ~is_close
                    covered_count += int(is_close.sum())

                representative_indices.append(index)
                counts.append(covered_count)

                alive_indices = alive_indices[keep_mask]

            representative_indices = torch.stack(representative_indices)
            quantized_centers = centers[representative_indices]

            weights = torch.tensor(counts, device=device, dtype=dtype)

            if self.full:
                self.full_centers = self.centers

            self.centers = quantized_centers
            self.weights = weights
            self.quantized = True

            self._clear_spectral_state()

            if return_centers:
                return quantized_centers, counts

            return None

    def _get_transformed_centers_for_quantization(self, r_eps=None):
        centers = self.centers.contiguous()
        device = centers.device
        dtype = centers.dtype

        metric_factor = self.kernel.metric_factor(r_eps=r_eps)

        if metric_factor is None:
            transformed_centers = centers
            transformed_norms = (transformed_centers ** 2).sum(dim=1)
            return transformed_centers, transformed_norms

        metric_factor = metric_factor.to(device=device, dtype=dtype)

        transformed_centers = (centers @ metric_factor).contiguous()
        transformed_norms = (transformed_centers ** 2).sum(dim=1)

        return transformed_centers, transformed_norms

    @staticmethod
    def _quantize_count_from_transformed_centers(
        transformed_centers,
        transformed_norms,
        q_thresh,
        chunksize=2000,
    ):
        device = transformed_centers.device
        num_centers = transformed_centers.shape[0]

        alive_indices = torch.arange(num_centers, device=device)

        num_representatives = 0
        counts = []

        while alive_indices.numel() > 0:
            index = alive_indices[0]

            xi = transformed_centers[index]
            xi_norm = transformed_norms[index]

            num_alive = alive_indices.numel()
            keep_mask = torch.ones(
                num_alive,
                device=device,
                dtype=torch.bool,
            )

            covered_count = 0

            for start in range(0, num_alive, chunksize):
                idx = alive_indices[start:start + chunksize]
                y = transformed_centers[idx]

                distances = torch.clamp(
                    transformed_norms[idx] + xi_norm - 2 * (y @ xi),
                    min=0.0,
                )

                is_close = distances < q_thresh

                keep_mask[start:start + idx.numel()] = ~is_close
                covered_count += int(is_close.sum())

            num_representatives += 1
            counts.append(covered_count)

            alive_indices = alive_indices[keep_mask]

        return num_representatives, counts

    @staticmethod
    def _estimate_q_thresh_range_from_transformed_centers(
        transformed_centers,
        num_pairs=20000,
        low_q=0.02,
        high_q=0.90,
        seed=0,
    ):
        device = transformed_centers.device
        num_centers = transformed_centers.shape[0]

        if num_centers < 2:
            return 1e-16, 1.01e-16

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        i = torch.randint(
            0,
            num_centers,
            (num_pairs,),
            generator=generator,
        ).to(device)

        j = torch.randint(
            0,
            num_centers,
            (num_pairs,),
            generator=generator,
        ).to(device)

        differences = transformed_centers[i] - transformed_centers[j]
        squared_distances = (differences * differences).sum(dim=1)

        q_low = torch.quantile(squared_distances, low_q).item()
        q_high = torch.quantile(squared_distances, high_q).item()

        q_low = max(q_low, 1e-16)
        q_high = max(q_high, q_low * 1.01)

        return q_low, q_high

    @classmethod
    def _find_q_thresh_for_target_count(
        cls,
        transformed_centers,
        transformed_norms,
        target_count,
        q_low,
        q_high,
        chunksize=2000,
        max_iter=20,
        tol_count=0,
    ):
        best_q_thresh = None
        best_count = None

        low = q_low
        high = q_high

        for _ in range(max_iter):
            mid = 0.5 * (low + high)

            count_mid, _ = cls._quantize_count_from_transformed_centers(
                transformed_centers,
                transformed_norms,
                mid,
                chunksize=chunksize,
            )

            if best_count is None:
                best_q_thresh = mid
                best_count = count_mid
            elif abs(count_mid - target_count) < abs(best_count - target_count):
                best_q_thresh = mid
                best_count = count_mid

            if abs(count_mid - target_count) <= tol_count:
                break

            if count_mid > target_count:
                low = mid
            else:
                high = mid

        return best_q_thresh, best_count

    def choose_quantization_thresholds_for_targets(
        self,
        target_counts,
        subset_size=3000,
        num_pairs=20000,
        chunksize=2000,
        r_eps=None,
        seed=0,
        max_iter=20,
        low_q=0.02,
        high_q=0.99,
        tol_count=0,
    ):
        """
        Choose quantization thresholds that approximately yield target counts.
        """
        with torch.inference_mode():
            transformed_centers_full, transformed_norms_full = (
                self._get_transformed_centers_for_quantization(r_eps=r_eps)
            )

            num_centers = transformed_centers_full.shape[0]
            device = transformed_centers_full.device

            if num_centers < 1:
                raise ValueError("Cannot choose thresholds with zero centers.")

            subset_size = min(int(subset_size), num_centers)

            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

            subset_indices = torch.randperm(
                num_centers,
                generator=generator,
            )[:subset_size].to(device)

            transformed_centers = transformed_centers_full[subset_indices]
            transformed_norms = transformed_norms_full[subset_indices]

            q_low, q_high = self._estimate_q_thresh_range_from_transformed_centers(
                transformed_centers,
                num_pairs=num_pairs,
                low_q=low_q,
                high_q=high_q,
                seed=seed,
            )

            q_low = 0.0

            thresholds = []
            approximate_counts = []

            for target_count in target_counts:
                target_count_subset = max(
                    1,
                    int(round(target_count * subset_size / num_centers)),
                )

                threshold, approximate_count = self._find_q_thresh_for_target_count(
                    transformed_centers,
                    transformed_norms,
                    target_count_subset,
                    q_low,
                    q_high,
                    chunksize=chunksize,
                    max_iter=max_iter,
                    tol_count=tol_count,
                )

                thresholds.append(threshold)
                approximate_counts.append(approximate_count)

            return thresholds, approximate_counts

    def _clear_spectral_state(self):
        """
        Clear stale kernel/eigendecomposition/normalization state.
        """
        self.approx_mode = None

        self.feature_mean = None
        self.feature_scale = None

        self.kernel_matrix = None
        self.eigvals = None
        self.eigvecs = None

        self.nystrom_indices = None
        self.nystrom_centers = None
        self.nystrom_eigvecs = None
        self.nystrom_eigvals = None
        self.nystrom_scale = None
        self.nystrom_weights = None
        self.nystrom_sqrt_weights = None

    def clone(self):
        clone = TorchSpecNystrom(
            kernel=self.kernel,
            centers=self.centers,
            full=self.full,
            num_eigs=self.num_eigs,
        )

        clone.weights = self.weights
        clone.approx_mode = self.approx_mode

        clone.feature_mean = self.feature_mean
        clone.feature_scale = self.feature_scale

        clone.kernel_matrix = self.kernel_matrix
        clone.eigvals = self.eigvals
        clone.eigvecs = self.eigvecs

        clone.nystrom_indices = self.nystrom_indices
        clone.nystrom_centers = self.nystrom_centers
        clone.nystrom_eigvecs = self.nystrom_eigvecs
        clone.nystrom_eigvals = self.nystrom_eigvals
        clone.nystrom_scale = self.nystrom_scale
        clone.nystrom_weights = self.nystrom_weights
        clone.nystrom_sqrt_weights = self.nystrom_sqrt_weights

        clone.quantized = self.quantized

        if hasattr(self, "full_centers"):
            clone.full_centers = self.full_centers

        return clone

    # -------------------------------------------------------------------------
    # Compatibility aliases
    # -------------------------------------------------------------------------

    def makeKernel(self, *args, **kwargs):
        return self.make_kernel(*args, **kwargs)

    def getFeatures(
        self,
        x,
        M,
        chunksize=None,
        normalize=True,
        r_eps=None,
    ):
        return self.get_features(
            x,
            num_features=M,
            feature_chunksize=chunksize or 500,
            kernel_chunksize=chunksize or 500,
            r_eps=r_eps,
            normalize=normalize,
        )

    def quantizeCenters(self, qThresh, returnCenters=False, chunksize=2000):
        return self.quantize_centers(
            q_thresh=qThresh,
            return_centers=returnCenters,
            chunksize=chunksize,
        )

    def quantizeCenters_mahalanobis(
        self,
        qThresh,
        returnCenters=False,
        chunksize=2000,
        r_eps=None,
    ):
        return self.quantize_centers_mahalanobis(
            q_thresh=qThresh,
            return_centers=returnCenters,
            chunksize=chunksize,
            r_eps=r_eps,
        )
#==================END SPEED APPROXIMATION=====================================



