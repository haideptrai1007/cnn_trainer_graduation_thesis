"""
models — Lightweight CNN model zoo for multi-class classification.

All models are built on `timm` and optimized for low-parameter,
single-channel (grayscale) inputs with 4-class output.

Usage
-----
>>> from models import get_model, list_models
>>> model = get_model("mobileone_s0")
>>> list_models()
"""

import timm
from torch import nn


# ──────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────
_REGISTRY = {}


def register(name: str):
    """Decorator to register a model factory function."""
    def wrapper(fn):
        _REGISTRY[name] = fn
        return fn
    return wrapper


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────

@register("edgenext_xxs")
def edgenext_xxs(num_classes: int = 4, in_chans: int = 1, pretrained: bool = False, **kw) -> nn.Module:
    """
    EdgeNeXt XX-Small  (~1.33M params)

    Hybrid CNN-Transformer with split depth-wise transposed attention.
    Good balance of accuracy and latency on edge devices.
    """
    return timm.create_model(
        "edgenext_xx_small.in1k",
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=0.1,
        drop_path_rate=0.1,
        **kw,
    )



@register("mobileone_s0")
def mobileone_s0(num_classes: int = 4, in_chans: int = 1, pretrained: bool = False, **kw) -> nn.Module:
    """
    MobileOne S0 (width=0.5)  (~1.13M params)

    Re-parameterizable branches that fold into a plain CNN at inference.
    Extremely fast on mobile hardware.
    """
    return timm.create_model(
        "mobileone_s0",
        pretrained=pretrained,
        width_factor=0.5,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=kw.pop("drop_rate", 0.05),
        drop_path_rate=kw.pop("drop_path_rate", 0.05),
        **kw,
    )


@register("ghostnetv3")
def ghostnetv3(num_classes: int = 4, in_chans: int = 1, pretrained: bool = False, **kw) -> nn.Module:
    """
    GhostNetV3 0.50x  (~2.12M params)

    Cheap linear operations to generate ghost feature maps,
    reducing compute while maintaining representational capacity.
    """
    return timm.create_model(
        "ghostnetv3_050",
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=kw.pop("drop_rate", 0.4),
        **kw,
    )


@register("mobilenetv4")
def mobilenetv4(num_classes: int = 4, in_chans: int = 1, pretrained: bool = False, **kw) -> nn.Module:
    """
    MobileNetV4 Conv-Small 0.50x  (~0.96M params)

    Universal inverted bottleneck with optimized NAS-searched architecture.
    Smallest model in the zoo — ideal for extreme latency constraints.
    """
    return timm.create_model(
        "mobilenetv4_conv_small_050",
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=kw.pop("drop_rate", 0.3),
        drop_path_rate=kw.pop("drop_path_rate", 0.1),
        **kw,
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def get_model(name: str, **kw) -> nn.Module:
    """
    Build a model by name.

    Parameters
    ----------
    name : str
        One of: edgenext_xxs, mobileone_s0, ghostnetv3, mobilenetv4
    **kw :
        Forwarded to the factory (num_classes, in_chans, pretrained, etc.)

    Returns
    -------
    nn.Module
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Choose from: {available}")
    return _REGISTRY[name](**kw)


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(_REGISTRY.keys())