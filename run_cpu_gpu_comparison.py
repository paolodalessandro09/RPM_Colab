#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run CPU and GPU RPM experiments into separate directories and compare them."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from config_refac import Config
from train_rpm import run_experiment
from compare_cpu_gpu_runs import build_table, _resolve_logger_path
from logger_refac import Logger


def _make_device_config(base_config: Config, *, device: str) -> Config:
    cfg = Config.from_dict(base_config.to_dict())
    cfg.device = device
    cfg.use_device_subdir = True

    base_dir = os.path.abspath(base_config.experiment_dir)
    cfg._base_experiment_dir = base_dir

    device_label = device.replace(":", "_")
    cfg.experiment_dir = os.path.join(base_dir, f"device_{device_label}")

    cfg.logger_filename = "rpm_metrics.pt"
    cfg.logger_run_meta_filename = "rpm_metrics_run_meta.yaml"

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare CPU/GPU RPM experiments")
    parser.add_argument("--config", required=True, help="Base YAML config to use")
    parser.add_argument("--cpu-only", action="store_true", help="Run only the CPU experiment")
    args = parser.parse_args()

    base_cfg = Config.from_file(args.config)

    cpu_cfg = _make_device_config(base_cfg, device="cpu")
    cpu_cfg.save(os.path.join(os.getcwd(), "config_cpu.yaml"))

    print(f"Running CPU experiment -> {cpu_cfg.experiment_dir}")
    run_experiment(cpu_cfg)

    if args.cpu_only:
        print("CPU-only mode enabled; skipping GPU run.")
        return

    gpu_cfg = _make_device_config(base_cfg, device="cuda")
    gpu_cfg.save(os.path.join(os.getcwd(), "config_gpu.yaml"))

    print(f"Running GPU experiment -> {gpu_cfg.experiment_dir}")
    run_experiment(gpu_cfg)

    cpu_logger_path = _resolve_logger_path(cpu_cfg.experiment_dir)
    gpu_logger_path = _resolve_logger_path(gpu_cfg.experiment_dir)

    cpu_logger = Logger.load(cpu_logger_path, map_location="cpu")
    gpu_logger = Logger.load(gpu_logger_path, map_location="cpu")

    print("\nCPU vs GPU comparison")
    print(build_table(cpu_logger, gpu_logger, [
        "mse_train",
        "mse_val",
        "mse_test",
        "nmse_train",
        "nmse_val",
        "nmse_test",
        "target_align_test",
        "excite_align_test",
        "fisher_geom_test",
        "eff_dim_test",
    ]))


if __name__ == "__main__":
    main()
