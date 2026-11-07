import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pickle
import numpy as np


from rmc import CosineSchedule, LinearSchedule, QuadraticSchedule


@dataclass
class ConfigMA:
    targets: List[str]
    method: str
    path_configs: Path
    path_out: Path


def load_json(config_path: str) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dict."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON value must be an object")

    # Check required keys
    
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing config keys: {', '.join(missing)}")
        
    return data

def validate_required_keys(values: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate that all required keys are present in the values dict."""
    missing = [key for key in required_keys if key not in values]
    if missing:
        raise KeyError(f"Missing config keys: {', '.join(missing)}")


def validate_ConfigMA(values: Dict[str, Any]) -> ConfigMA:
    """Validate types and build a Config object."""
    # Validate types
    targets = values["targets"]
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise TypeError("'targets' must be a list of strings")

    method = values["method"]
    if not isinstance(method, str):
        raise TypeError("'method' must be a string")
       
    for key in ("path_configs", "path_out"):
        if not isinstance(values[key], (str, Path)):
            raise TypeError(f"'{key}' must be a string (file path)")
        
    return ConfigMA(
        targets=targets,
        method=method,
        path_configs=Path(values["path_configs"]),
        path_out=Path(values["path_out"]),
    )    

def load_ConfigMA(config_path: str) -> ConfigMA:
    """Load and validate a JSON configuration file for meta-analysis."""
    data = load_json(config_path)
    required = ["targets", "method", "path_configs", "path_out"]
    validate_required_keys(data, required)  
    return validate_ConfigMA(data)


def get_activation_function(name: str):
    """Return the activation function based on the provided name."""
    if name == "relu":
        return nnx.relu
    elif name == "leaky_relu":
        return nnx.leaky_relu
    elif name == "tanh":
        return nnx.tanh
    elif name == "silu":
        return nnx.silu
    elif name == "celu":
        return nnx.celu
    elif name == "elu":
        return nnx.elu
    elif name == "gelu":
        return nnx.gelu
    elif name == "sigmoid":
        return nnx.sigmoid
    elif name == "softplus":
        return nnx.softplus
    elif name == "identity":
        return nnx.identity
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_schedule_function(name: str):
    if name == "cosine":
        return CosineSchedule()
    elif name == "linear":
        return LinearSchedule()
    elif name == "quadratic":
        return QuadraticSchedule()
    else:
        raise ValueError(f"Unknown schedule function: {name}")


def pad_with_const(x):
    """Pad array with row of ones.

    Args:
        x: Array to process.
    Returns:
        Processed array.
    """
    extra = np.ones((x.shape[0], 1))
    return np.hstack([extra, x])


def standardize_and_pad(x):
    """Standardize data and pad with constant value.

    Args:
        x: Array to process.
    Returns:
        Processed array.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std
    return pad_with_const(x)


def load_data(name):
    """Load data from pickle file and preprocess.

    Args:
        name: Name of file to read.
    Returns:
        Arrays of (x, y) data read.
    """
    with open(name, mode="rb") as f:
        x, y = pickle.load(f)
    y = (y + 1) // 2
    x = standardize_and_pad(x)
    return x, y