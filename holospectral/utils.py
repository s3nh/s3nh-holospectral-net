"""
Utility functions for holographic operations.
"""

import torch


def fft_circular_conv(x, y):
    """
    Holographic binding via circular convolution.
    
    Args:
        x: Input tensor of shape (..., dim)
        y: Input tensor of shape (dim,) or broadcastable shape
        
    Returns:
        Circular convolution result (real part)
    """
    fx = torch.fft.fft(x, dim=-1)
    fy = torch.fft.fft(y, dim=-1)
    return torch.fft.ifft(fx * fy, dim=-1).real


def fft_circular_corr(x, y):
    """
    Holographic unbinding via circular correlation.
    
    Args:
        x: Input tensor of shape (..., dim)
        y: Input tensor of shape (dim,) or broadcastable shape
        
    Returns:
        Circular correlation result (real part)
    """
    fx = torch.fft.fft(x, dim=-1)
    fy = torch.fft.fft(y, dim=-1).conj()
    return torch.fft.ifft(fx * fy, dim=-1).real
