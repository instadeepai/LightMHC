"""Module for creating models."""
import inspect
from typing import Any

from torch import nn

from lightmhc.model.gnn import attn_gnn


def model_class_factory(model_name: str) -> nn.Module:
    """Returns model class for a given model name.

    Args:
        model_name: Name of model.

    Returns:
        model_class: Model class for model_name.
    """
    try:
        return MODEL_NAME2MODEL_CLS[model_name]
    except KeyError as keyerror:
        raise NotImplementedError(
            f"The class {model_name} is not implemented."
            f"Available Classes: {list(MODEL_NAME2MODEL_CLS)}"
        ) from keyerror


def is_model_cls(x: Any) -> bool:
    """List of model helper function."""
    return (
        inspect.isclass(x)
        and issubclass(x, nn.Module)
        and not inspect.isabstract(x)
        and "lightmhc.model" in str(x)
    )


# GNN models
MODEL_NAME2MODEL_CLS = dict(inspect.getmembers(attn_gnn, predicate=is_model_cls))
