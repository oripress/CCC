"""Continual Test Time Adaptation models."""
from typing import List

import torch.nn as nn

__model_registry = {}


def _clear_registry():
    """Clear the entries in the registry."""
    global __model_registry
    __model_registry.clear()


def register(name: str):
    """Decorator to register a new model"""
    global __model_registry

    def _register(cls):
        if name in __model_registry:
            raise ValueError(f"Duplicate name added to registry: {name}.")
        __model_registry[name] = cls
        return cls

    return _register


def init(name, *args, **kwargs):
    """Init the specified model using the given arguments."""
    global __model_registry
    if name not in __model_registry:
        raise ValueError("Model with name {name} not registered.")
    return __model_registry[name](*args, **kwargs)


def get_options() -> List[str]:
    """Return a list of all registered models."""
    global __model_registry
    return tuple(__model_registry.keys())


class AdaptiveModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError()
