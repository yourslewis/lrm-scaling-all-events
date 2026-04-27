# Copyright (c) Meta Platforms, Inc. and affiliates.
# Registry pattern for config-driven module construction.
#
# Usage:
#   @register("encoder", "HSTU")
#   class HSTU(...): ...
#
#   module = build("encoder", "HSTU", embedding_dim=256, ...)
#
# Categories: encoder, embedding, loss, sampler, preprocessor,
#             postprocessor, kernel, interest, experiment

from typing import Any, Dict, List, Optional, Type

_REGISTRY: Dict[str, Dict[str, Type]] = {}


def register(category: str, name: str):
    """
    Class decorator that registers a module under ``category/name``.

    Example::

        @register("encoder", "HSTU")
        class HSTU(torch.nn.Module):
            ...
    """
    def decorator(cls: Type) -> Type:
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        if name in _REGISTRY[category]:
            existing = _REGISTRY[category][name]
            # Allow re-registration of the same class (e.g. module reload)
            if existing is not cls:
                raise ValueError(
                    f"Registry conflict: {category}/{name} is already "
                    f"registered to {existing.__module__}.{existing.__qualname__}"
                )
        _REGISTRY[category][name] = cls
        return cls
    return decorator


def build(category: str, name: str, **kwargs) -> Any:
    """
    Instantiate a registered module by ``category`` and ``name``.

    All remaining ``**kwargs`` are forwarded to the constructor.

    Raises ``KeyError`` if the category/name pair is unknown.
    """
    if category not in _REGISTRY or name not in _REGISTRY[category]:
        available = list_registered(category)
        raise KeyError(
            f"No module registered as {category}/{name}. "
            f"Available: {available}"
        )
    cls = _REGISTRY[category][name]
    return cls(**kwargs)


def get_class(category: str, name: str) -> Type:
    """Return the registered class without instantiating it."""
    if category not in _REGISTRY or name not in _REGISTRY[category]:
        available = list_registered(category)
        raise KeyError(
            f"No module registered as {category}/{name}. "
            f"Available: {available}"
        )
    return _REGISTRY[category][name]


def list_registered(category: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List registered modules.

    If ``category`` is given, returns a flat list of names.
    Otherwise returns ``{category: [names]}`` for all categories.
    """
    if category is not None:
        return list(_REGISTRY.get(category, {}).keys())  # type: ignore[return-value]
    return {cat: list(entries.keys()) for cat, entries in _REGISTRY.items()}


def is_registered(category: str, name: str) -> bool:
    """Check if a module is registered."""
    return category in _REGISTRY and name in _REGISTRY[category]
