"""Utilities for exporting YOLO validation curves."""

__all__ = ["export_model_curves"]


def __getattr__(name: str):
    if name == "export_model_curves":
        from .core import export_model_curves

        return export_model_curves
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
