"""
Predefined model configurations for HoloSpectralNet.
"""

from .model import HoloSpectralNet


def hsn_tiny(vocab_size=32000):
    """
    Tiny HoloSpectralNet configuration (~400K parameters).
    
    Args:
        vocab_size: Size of the vocabulary
        
    Returns:
        HoloSpectralNet model instance
    """
    return HoloSpectralNet(vocab_size=vocab_size, dim=128, depth=4, rank=16)


def hsn_small(vocab_size=32000):
    """
    Small HoloSpectralNet configuration (~1.5M parameters).
    
    Args:
        vocab_size: Size of the vocabulary
        
    Returns:
        HoloSpectralNet model instance
    """
    return HoloSpectralNet(vocab_size=vocab_size, dim=256, depth=6, rank=32)


def hsn_base(vocab_size=32000):
    """
    Base HoloSpectralNet configuration (~8M parameters).
    
    Args:
        vocab_size: Size of the vocabulary
        
    Returns:
        HoloSpectralNet model instance
    """
    return HoloSpectralNet(vocab_size=vocab_size, dim=512, depth=8, rank=64)
