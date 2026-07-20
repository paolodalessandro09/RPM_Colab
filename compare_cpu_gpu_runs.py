#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two saved RPM logger runs and print a markdown table."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional

import torch

from logger_refac import Logger


DEFAULT_METRICS = [
    "mse_train",
    "mse_val",
    "mse_test",
    "nmse_train",
    "nmse_val",
    "nmse_test",
    "target_align_train",
    "target_align_val",
    "target_align_test",
    "excite_align_train",
    "excite_align_val",
    "excite_align_test",
    "fisher_geom_train",
    "fisher_geom_val",
    "fisher_geom_test",
    "eff_dim_train",
    "eff_dim_val",
    "eff_dim_test",
]


def _resolve_logger_path(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "rpm_metrics.pt")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"No logger found in directory: {path}")

    if os.path.isfile(path):
        return path

    raise FileNotFoundError(f"Logger path does not exist: {path}")


def _last_scalar(logger: Logger, metric: str) -> Optional[float]:
    if metric not in logger.data:
        return None

    trial_values = logger.data[metric][0]
    if not trial_values:
        return None

    value = trial_values[-1]

    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() != 1:
            return None
        return float(value.reshape(-1)[0])

    if isinstance(value, (float, int)):
        return float(value)

    return None


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6e}"


def build_table(cpu_logger: Logger, gpu_logger: Logger, metrics: Iterable[str]) -> str:
    lines = [
        "| Metric | CPU | GPU | Abs diff |",
        "|---|---:|---:|---:|",
    ]

    for metric in metrics:
        cpu_value = _last_scalar(cpu_logger, metric)
        gpu_value = _last_scalar(gpu_logger, metric)

        if cpu_value is None or gpu_value is None:
            lines.append(f"| {metric} | {_format_value(cpu_value)} | {_format_value(gpu_value)} | n/a |")
            continue

        diff = abs(gpu_value - cpu_value)
        lines.append(f"| {metric} | {_format_value(cpu_value)} | {_format_value(gpu_value)} | {diff:.6e} |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU and GPU RPM run logs")
    parser.add_argument("--cpu", dest="cpu_path", required=True, help="Path to CPU logger (.pt file or run directory)")
    parser.add_argument("--gpu", dest="gpu_path", required=True, help="Path to GPU logger (.pt file or run directory)")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS, help="Metrics to compare")
    args = parser.parse_args()

    cpu_path = _resolve_logger_path(args.cpu_path)
    gpu_path = _resolve_logger_path(args.gpu_path)

    cpu_logger = Logger.load(cpu_path, map_location="cpu")
    gpu_logger = Logger.load(gpu_path, map_location="cpu")

    print(f"CPU logger: {cpu_path}")
    print(f"GPU logger: {gpu_path}")
    print()
    print(build_table(cpu_logger, gpu_logger, args.metrics))


if __name__ == "__main__":
    main()
