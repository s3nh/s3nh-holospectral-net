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
    via exponential moving average (replacing cumsum for better gradient flow),
    and unbinds to retrieve mixed representation.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.bind_key = nn.Parameter(torch.randn(dim) * 0.02)
        self.unbind_key = nn.Parameter(torch.randn(dim) * 0.02)
        self.gate = nn.Parameter(torch.ones(1))
        self.decay = nn.Parameter(torch.tensor(0.9))  # Learnable decay
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # x: (batch, seq_len, dim)
        bound = fft_circular_conv(x, self.bind_key)
        
        # Exponential moving average instead of cumsum
        B, T, D = bound.shape
        decay = torch.sigmoid(self.decay)
        
        # Create decay weights for each position
        positions = torch.arange(T, device=x.device).float()
        # For each position t, weight for position i is decay^(t-i)
        decay_matrix = decay ** (positions.unsqueeze(1) - positions.unsqueeze(0))
        # Make causal (zero out future positions)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        decay_matrix = decay_matrix * causal_mask
        
        # Apply weighted sum: hologram[t] = sum(decay^(t-i) * bound[i] for i <= t)
        hologram = torch.einsum('ts,bsd->btd', decay_matrix, bound)
        
        retrieved = fft_circular_corr(hologram, self.unbind_key)
        return x + self.gate * retrieved


class SpectralGate(nn.Module):
    """Causal spectral gating using chunked local FFT.
    
    Applies learned gate in frequency domain via FFT to local chunks,
    ensuring causality for autoregressive language modeling.
    """
    
    def __init__(self, max_seq_len=512, chunk_size=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.freq_gate = nn.Parameter(torch.ones(chunk_size // 2 + 1))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        B, T, D = x.shape
        chunk_size = min(self.chunk_size, T)
        
        # Pad sequence to be divisible by chunk_size
        pad_len = (chunk_size - T % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Reshape into chunks
        x_chunked = x.view(B, -1, chunk_size, D)
        
        # Apply FFT to each chunk independently (causal within chunk)
        x_freq = torch.fft.rfft(x_chunked, dim=2)
        gate = self.freq_gate[:x_freq.shape[2]].view(1, 1, -1, 1)
        x_gated = x_freq * torch.sigmoid(gate)
        x_out = torch.fft.irfft(x_gated, dim=2, n=chunk_size)
        
        # Reshape back
        x_out = x_out.reshape(B, -1, D)
        
        # Remove padding
        return x_out[:, :T]


class LiteChannelMix(nn.Module):
    """
    Lightweight channel mixing layer.
    
    Replaces heavy FFN (no 4x expansion) with factorized projection:
    dim → hidden → dim where hidden = dim * expansion
    """
    
    def __init__(self, dim, rank=32, expansion=2):
        super().__init__()
        hidden = dim * expansion
        self.down = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(hidden, dim, bias=False)
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Remove internal residual since block handles it
        return self.scale * self.up(F.gelu(self.down(x)))


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
        self.spectral_gate = SpectralGate(max_seq_len, chunk_size=32)
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
        x = x + self.channel_mix(self.norm3(x))  # Add residual connection
        return x
