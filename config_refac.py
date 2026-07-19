#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flexible configuration object for RPM experiments.

This module is intentionally separate from the logger. The config object owns
experiment settings. The logger only records a copy of those settings.
"""

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import yaml


DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "float64": torch.float64,
}

DTYPE_TO_STRING = {
    torch.float32: "float32",
    torch.float64: "float64",
}


class Config:
    """
    Flexible experiment configuration.

    This class stores arbitrary key/value pairs and exposes them as attributes:

        config.kernel_sigma
        config.feature_output_dim
        config.new_hyperparameter

    New keys can be added to the YAML file without changing this class.
    """

    def __init__(
        self,
        values: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ):
        values = {} if values is None else dict(values)
        values.update(overrides)

        super().__setattr__("_values", {})

        for key, value in values.items():
            self[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = self._normalize_value(value)

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __getattr__(self, key: str) -> Any:
        if key in self._values:
            return self._values[key]

        raise AttributeError(
            f"Config has no attribute '{key}'. "
            "Check the config YAML file or add a default with config.setdefault(...)."
        )

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_values":
            super().__setattr__(key, value)
            return

        self._values[key] = self._normalize_value(value)

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._values.keys()))
        return f"{type(self).__name__}({keys})"

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def values(self):
        return self._values.values()

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        if key not in self._values:
            self[key] = default

        return self._values[key]

    def update(
        self,
        values: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "Config":
        if values is not None:
            for key, value in values.items():
                self[key] = value

        for key, value in kwargs.items():
            self[key] = value

        return self

    def require(self, *keys: str) -> None:
        """
        Raise a clear error if required keys are missing.
        """
        missing = [key for key in keys if key not in self._values]

        if missing:
            raise KeyError(
                "Missing required config keys: "
                + ", ".join(missing)
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a YAML/torch-saveable dictionary.
        """
        out = deepcopy(self._values)

        for key, value in list(out.items()):
            out[key] = self._serialize_value(value)

        return out

    def save(self, path: str) -> str:
        """
        Save the config as a YAML file.
        """
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                sort_keys=False,
            )

        return path

    @classmethod
    def from_file(
        cls,
        path: str,
        **overrides: Any,
    ) -> "Config":
        """
        Load a config from a YAML file.

        Keyword overrides are applied after loading the file.
        """
        with open(path, "r") as f:
            values = yaml.safe_load(f)

        if values is None:
            values = {}

        values["source_config_path"] = path
        values.update(overrides)

        return cls(values)

    @classmethod
    def from_dict(
        cls,
        values: Dict[str, Any],
        **overrides: Any,
    ) -> "Config":
        """
        Build a config from a dictionary.
        """
        values = dict(values)
        values.update(overrides)

        return cls(values)

    @classmethod
    def from_run_meta(
        cls,
        path: str,
        **overrides: Any,
    ) -> "Config":
        """
        Load a config from a run_meta YAML file saved by Logger.save_run_meta(...).
        """
        return cls.from_file(path, **overrides)

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """
        Convert YAML-friendly values into runtime-friendly values.
        """
        if isinstance(value, str) and value in DTYPE_FROM_STRING:
            return DTYPE_FROM_STRING[value]

        return value

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """
        Convert runtime values into YAML-friendly values.
        """
        if isinstance(value, torch.dtype):
            return DTYPE_TO_STRING.get(value, str(value).replace("torch.", ""))

        if isinstance(value, dict):
            return {
                key: Config._serialize_value(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                Config._serialize_value(val)
                for val in value
            ]

        return value


