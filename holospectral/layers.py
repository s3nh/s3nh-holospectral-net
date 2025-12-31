"""
Core layers for HoloSpectralNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import fft_circular_conv, fft_circular_corr


class HolographicMixer(nn.Module):
    """
    Holographic token mixing layer using circular convolution.
    
    Binds tokens with learned key via circular convolution, creates hologram
    via cumulative bundling (cumsum), and unbinds to retrieve mixed representation.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.bind_key = nn.Parameter(torch.randn(dim) * 0.02)
        self.unbind_key = nn.Parameter(torch.randn(dim) * 0.02)
        self.gate = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # x: (batch, seq_len, dim)
        bound = fft_circular_conv(x, self.bind_key)
        hologram = torch.cumsum(bound, dim=1)
        retrieved = fft_circular_corr(hologram, self.unbind_key)
        return x + self.gate * retrieved


class SpectralGate(nn.Module):
    """
    Spectral gating layer that learns which frequency components matter.
    
    Applies learned gate in frequency domain via FFT.
    """
    
    def __init__(self, max_seq_len=512):
        super().__init__()
        self.freq_gate = nn.Parameter(torch.ones(max_seq_len // 2 + 1))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        seq_len = x.shape[1]
        x_freq = torch.fft.rfft(x, dim=1)
        gate = self.freq_gate[:x_freq.shape[1]].unsqueeze(0).unsqueeze(-1)
        x_gated = x_freq * torch.sigmoid(gate)
        return torch.fft.irfft(x_gated, dim=1, n=seq_len)


class LiteChannelMix(nn.Module):
    """
    Lightweight channel mixing layer.
    
    Replaces heavy FFN (no 4x expansion) with factorized projection:
    dim → rank → dim
    """
    
    def __init__(self, dim, rank=32):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        return x + self.scale * self.up(F.gelu(self.down(x)))


class RotaryEmbedding(nn.Module):
    """
    Parameter-free rotary positional embedding (RoPE).
    """
    
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim) with rotary embeddings applied
        """
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
                         x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]]], dim=-1)


class HoloSpectralBlock(nn.Module):
    """
    Complete HoloSpectral block combining holographic mixing, spectral gating,
    and lightweight channel mixing.
    """
    
    def __init__(self, dim, max_seq_len=512, rank=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.holo_mixer = HolographicMixer(dim)
        self.spectral_gate = SpectralGate(max_seq_len)
        self.channel_mix = LiteChannelMix(dim, rank)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        x = x + self.holo_mixer(self.norm1(x))
        x = x + self.spectral_gate(self.norm2(x))
        x = self.channel_mix(self.norm3(x))
        return x
