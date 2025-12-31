"""
Main HoloSpectralNet model.
"""

import torch
import torch.nn as nn

from .layers import HoloSpectralBlock, RotaryEmbedding


class HoloSpectralNet(nn.Module):
    """
    HoloSpectralNet: A novel, lightweight neural architecture that replaces
    attention mechanisms with holographic binding and spectral gating.
    
    Args:
        vocab_size: Size of the vocabulary
        dim: Model dimension
        depth: Number of HoloSpectralBlocks
        max_seq_len: Maximum sequence length
        rank: Rank for LiteChannelMix factorization
        tie_weights: Whether to tie embedding and output weights
    """
    
    def __init__(
        self,
        vocab_size=32000,
        dim=256,
        depth=6,
        max_seq_len=512,
        rank=32,
        tie_weights=True
    ):
        super().__init__()
        self.dim = dim
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.rotary = RotaryEmbedding(dim)
        
        self.blocks = nn.ModuleList([
            HoloSpectralBlock(dim, max_seq_len, rank)
            for _ in range(depth)
        ])
        
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        if tie_weights:
            self.head.weight = self.token_emb.weight
            
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for the model."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input token indices of shape (batch, seq_len)
            
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        h = self.token_emb(x)
        h = self.rotary(h)
        
        for block in self.blocks:
            h = block(h)
            
        h = self.norm_out(h)
        logits = self.head(h)
        
        return logits
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
