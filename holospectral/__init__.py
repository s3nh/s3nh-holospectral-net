"""
HoloSpectralNet: A lightweight holographic-spectral hybrid architecture.
"""

from .model import HoloSpectralNet
from .layers import (
    HolographicMixer,
    SpectralGate,
    LiteChannelMix,
    HoloSpectralBlock,
    RotaryEmbedding,
)
from .config import hsn_tiny, hsn_small, hsn_base
from .utils import fft_circular_conv, fft_circular_corr

__version__ = "0.1.0"

__all__ = [
    "HoloSpectralNet",
    "HolographicMixer",
    "SpectralGate",
    "LiteChannelMix",
    "HoloSpectralBlock",
    "RotaryEmbedding",
    "hsn_tiny",
    "hsn_small",
    "hsn_base",
    "fft_circular_conv",
    "fft_circular_corr",
]
