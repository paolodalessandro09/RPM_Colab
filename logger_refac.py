#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger class for RPM experiments.

The logger is intentionally separate from configuration. It records:
    - run_meta: a flat copy of the config that produced the run,
    - trial_meta: per-trial settings/seed/runtime,
    - data: tracked metrics organized as metric -> trial -> epoch.
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
import time
from typing import Any, Dict, Iterable, List, Optional, Union
from dataclasses import dataclass, field

import torch
import yaml

from util_refac import to_cpu_detached


def object_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a config-like object to a plain dictionary.
    """
    if obj is None:
        return {}

    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())

    if isinstance(obj, dict):
        return dict(obj)

    return dict(vars(obj))


@dataclass
class Logger:
    """
    Trial/Epoch logger.

    Structure
    ---------
    run_meta : dict
        Flat dictionary containing run-level settings.

    trial_meta : list[dict]
        One flat dictionary per trial. Each dictionary stores trial-specific
        settings such as seed and runtime, and may also store a copy of the
        config used for that trial.

    data : dict
        data[name] = [trial0_epochs, trial1_epochs, ...]
    """

    names: Iterable[str]
    save_dir: str
    filename: str = "logger.pt"
    config: Optional[Any] = None
    strict: bool = False
    move_to_cpu: bool = True

    data: Dict[str, List[List[Any]]] = field(default_factory=dict)
    trial_meta: List[Dict[str, Any]] = field(default_factory=list)
    run_meta: Dict[str, Any] = field(default_factory=dict)

    _active_trial: Optional[int] = None
    _active_epoch: Optional[int] = None
    _trial_start_time: Optional[float] = None

    def __post_init__(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        for name in self.names:
            self.register(name)

        if self.config is not None:
            self.set_config(self.config)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        metric_names: Optional[Iterable[str]] = None,
        strict: bool = True,
        move_to_cpu: bool = True,
    ) -> "Logger":
        """
        Construct a logger from any config object.

        The config must provide:
            config.experiment_dir
            config.logger_filename

        Metric names can be passed explicitly or read from:
            config.metric_names
        """
        if metric_names is None:
            metric_names = config.metric_names

        return cls(
            names=metric_names,
            save_dir=config.experiment_dir,
            filename=config.logger_filename,
            config=config,
            strict=strict,
            move_to_cpu=move_to_cpu,
        )

    @property
    def path(self) -> str:
        return os.path.join(self.save_dir, self.filename)

    @property
    def run_meta_path(self) -> str:
        root, _ = os.path.splitext(self.filename)
        return os.path.join(self.save_dir, f"{root}_run_meta.yaml")

    def set_config(self, config: Any) -> None:
        """
        Store run-level settings as a flat dictionary.
        """
        self.config = config
        self.run_meta = object_to_dict(config)
        self.run_meta["config_class"] = type(config).__name__
        self.run_meta["metric_names"] = list(self.data.keys())

    def save_run_meta(self, path: Optional[str] = None) -> str:
        """
        Save only run_meta as a separate YAML file.

        This file is the lightweight reproduction recipe:
            config = TrainingConfig.from_file(path)
        """
        path = path or self.run_meta_path

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(
                self.run_meta,
                f,
                sort_keys=False,
            )

        return path

    def register(self, name: str) -> None:
        if name not in self.data:
            self.data[name] = []

    def start_trial(
        self,
        trial_idx: Optional[int] = None,
        *,
        config: Optional[Any] = None,
        seed: Optional[int] = None,
        **meta: Any,
    ) -> int:
        """
        Start a new trial.

        The trial metadata stores trial-specific information, including the seed.
        If a trial-specific config is supplied, that config is stored in the
        trial metadata. Otherwise the logger's run-level config is used.
        """
        if self._active_trial is not None:
            raise RuntimeError("A trial is already active. Call end_trial() first.")

        if trial_idx is None:
            trial_idx = len(self.trial_meta)

        for name in self.data:
            while len(self.data[name]) <= trial_idx:
                self.data[name].append([])

        while len(self.trial_meta) <= trial_idx:
            self.trial_meta.append({})

        if config is None:
            config = self.config

        trial_record: Dict[str, Any] = {}

        if config is not None:
            trial_record.update(object_to_dict(config))

        trial_record["trial_idx"] = trial_idx

        if seed is not None:
            trial_record["seed"] = seed

        trial_record.update(meta)

        self._trial_start_time = time.perf_counter()
        trial_record["t_start_perf_counter"] = float(self._trial_start_time)

        if self.move_to_cpu:
            trial_record = to_cpu_detached(trial_record)

        self.trial_meta[trial_idx].update(trial_record)

        self._active_trial = trial_idx
        self._active_epoch = -1

        return trial_idx

    def end_trial(self) -> None:
        if self._active_trial is None:
            raise RuntimeError("No active trial to end.")

        t_end = time.perf_counter()
        t_start = self._trial_start_time

        runtime_sec = None
        if t_start is not None:
            runtime_sec = float(t_end - t_start)

        idx = self._active_trial

        self.trial_meta[idx]["t_end_perf_counter"] = float(t_end)

        if runtime_sec is not None:
            self.trial_meta[idx]["runtime_sec"] = runtime_sec

        self._active_trial = None
        self._active_epoch = None
        self._trial_start_time = None

    def log_epoch(
        self,
        epoch: Optional[int] = None,
        **metrics: Any,
    ) -> None:
        """
        Log per-epoch metrics for the active trial.
        """
        if self._active_trial is None:
            raise RuntimeError("No active trial. Call start_trial() first.")

        if epoch is None:
            epoch = (self._active_epoch + 1) if self._active_epoch is not None else 0

        self._active_epoch = max(self._active_epoch or -1, epoch)

        for key, value in metrics.items():
            if key not in self.data:
                if self.strict:
                    raise KeyError(
                        f"Unknown metric '{key}'. Register it or set strict=False."
                    )

                self.register(key)

                while len(self.data[key]) <= self._active_trial:
                    self.data[key].append([])

            value = to_cpu_detached(value) if self.move_to_cpu else value
            self.data[key][self._active_trial].append(value)

    def log_trial(self, **metrics: Any) -> None:
        """
        Log end-of-trial values into the same data structure.
        """
        if self._active_trial is None:
            raise RuntimeError("No active trial. Call start_trial() first.")

        for key, value in metrics.items():
            if key not in self.data:
                if self.strict:
                    raise KeyError(
                        f"Unknown metric '{key}'. Register it or set strict=False."
                    )

                self.register(key)

                while len(self.data[key]) <= self._active_trial:
                    self.data[key].append([])

            value = to_cpu_detached(value) if self.move_to_cpu else value
            self.data[key][self._active_trial].append(value)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "trial_meta": self.trial_meta,
            "run_meta": self.run_meta,
            "names": list(self.data.keys()),
            "strict": self.strict,
            "move_to_cpu": self.move_to_cpu,
        }

    def save(
        self,
        path: Optional[str] = None,
        *,
        save_run_meta: bool = False,
    ) -> str:
        """
        Save the full logger state.

        If save_run_meta=True, also save a separate YAML reproduction file.
        """
        path = path or self.path

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(self.state_dict(), path)

        if save_run_meta:
            self.save_run_meta()

        return path

    @classmethod
    def load(
        cls,
        path: str,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "Logger":
        payload = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        save_dir = os.path.dirname(path)
        filename = os.path.basename(path)

        logger = cls(
            names=payload.get("names", []),
            save_dir=save_dir,
            filename=filename,
            config=None,
            strict=payload.get("strict", False),
            move_to_cpu=payload.get("move_to_cpu", True),
        )

        logger.data = payload["data"]
        logger.trial_meta = payload.get("trial_meta", [])
        logger.run_meta = payload.get("run_meta", {})

        logger._active_trial = None
        logger._active_epoch = None
        logger._trial_start_time = None

        return logger

    @property
    def config_dict(self) -> Dict[str, Any]:
        return self.run_meta

    def get(
        self,
        name: str,
        trial: Optional[int] = None,
    ) -> Any:
        if name not in self.data:
            raise KeyError(f"Unknown metric '{name}'.")

        if trial is None:
            return self.data[name]

        return self.data[name][trial]

    def summary(self) -> Dict[str, Any]:
        return {
            "n_trials": max((len(v) for v in self.data.values()), default=0),
            "metrics": {
                key: [len(epoch_list) for epoch_list in trials]
                for key, trials in self.data.items()
            },
            "run_meta": self.run_meta,
            "trial_meta": self.trial_meta,
        }
