"""Dataset adapters. Add one file per benchmark and register it below."""

from .base import DatasetAdapter, Sequence
from .uav123 import UAV123Adapter

ADAPTERS = {
    "uav123": UAV123Adapter,
}


def get(name: str) -> type:
    if name not in ADAPTERS:
        raise KeyError(
            f"unknown dataset {name!r}; known: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[name]


__all__ = ["DatasetAdapter", "Sequence", "get", "ADAPTERS"]
