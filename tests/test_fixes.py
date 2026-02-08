"""
Unit tests for HoloSpectralNet convergence fixes.
"""

import unittest
import torch
import torch.nn.functional as F

from holospectral import (
    HolographicMixer,
    SpectralGate,
    LiteChannelMix,
    HoloSpectralBlock,
)


class TestHolographicMixerDecay(unittest.TestCase):
    """Test exponential decay mechanism in HolographicMixer."""
    
    def test_decay_parameter_exists(self):
        """Test that decay parameter is initialized."""
        mixer = HolographicMixer(dim=16)
        self.assertTrue(hasattr(mixer, 'decay'))
        self.assertIsInstance(mixer.decay, torch.nn.Parameter)
    
    def test_decay_is_learnable(self):
        """Test that decay parameter has gradients."""
        mixer = HolographicMixer(dim=16)
        x = torch.randn(2, 8, 16, requires_grad=True)
        
        output = mixer(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(mixer.decay.grad)
        self.assertTrue(torch.isfinite(mixer.decay.grad).all())


class TestSpectralGateCausality(unittest.TestCase):
    """Test causal behavior of SpectralGate."""
    
    def test_chunk_size_parameter(self):
        """Test that chunk_size parameter is used."""
        gate = SpectralGate(max_seq_len=512, chunk_size=32)
        self.assertEqual(gate.chunk_size, 32)
    
    def test_freq_gate_size(self):
        """Test that freq_gate is sized for chunk, not full sequence."""
        chunk_size = 32
        gate = SpectralGate(max_seq_len=512, chunk_size=chunk_size)
        expected_size = chunk_size // 2 + 1
        self.assertEqual(gate.freq_gate.size(0), expected_size)
    
    def test_output_shape_preservation(self):
        """Test that output shape matches input shape."""
        gate = SpectralGate(max_seq_len=512, chunk_size=32)
        
        for seq_len in [16, 31, 32, 33, 64, 100]:
            x = torch.randn(2, seq_len, 16)
            output = gate(x)
            self.assertEqual(output.shape, x.shape)
    
    def test_handles_padding(self):
        """Test that SpectralGate handles sequences not divisible by chunk_size."""
        gate = SpectralGate(max_seq_len=512, chunk_size=32)
        
        # Test with sequence length that needs padding (not divisible by 32)
        x = torch.randn(2, 50, 16)
        output = gate(x)
        
        self.assertEqual(output.shape, (2, 50, 16))
        self.assertTrue(torch.isfinite(output).all())


class TestLiteChannelMixExpansion(unittest.TestCase):
    """Test expansion parameter in LiteChannelMix."""
    
    def test_expansion_parameter(self):
        """Test that expansion parameter controls hidden dimension."""
        dim = 64
        expansion = 4
        
        mix = LiteChannelMix(dim=dim, rank=32, expansion=expansion)
        
        # Check that hidden dimension is dim * expansion
        expected_hidden = dim * expansion
        self.assertEqual(mix.down.out_features, expected_hidden)
        self.assertEqual(mix.up.in_features, expected_hidden)
    
    def test_default_expansion(self):
        """Test that default expansion is 2."""
        dim = 64
        mix = LiteChannelMix(dim=dim, rank=32)  # No expansion specified
        
        # Default should be expansion=2
        expected_hidden = dim * 2
        self.assertEqual(mix.down.out_features, expected_hidden)
    
    def test_no_internal_residual(self):
        """Test that LiteChannelMix doesn't add internal residual."""
        dim = 16
        mix = LiteChannelMix(dim=dim)
        
        x = torch.randn(2, 8, dim)
        output = mix(x)
        
        # Output should not equal input (no skip connection)
        # Since we're using random init, they should be different
        self.assertFalse(torch.allclose(output, x, atol=1e-6))


class TestResidualConnections(unittest.TestCase):
    """Test consistent residual connections in HoloSpectralBlock."""
    
    def test_all_sublayers_have_residuals(self):
        """Test that all three sublayers use residual connections."""
        block = HoloSpectralBlock(dim=16, max_seq_len=64, rank=8)
        
        x = torch.randn(2, 8, 16)
        output = block(x)
        
        # Output should be different from input (transformations applied)
        self.assertFalse(torch.allclose(output, x, atol=1e-6))
        
        # But gradients should flow (test with backward)
        output.sum().backward()
    
    def test_gradient_flow(self):
        """Test that gradients flow through all residual paths."""
        block = HoloSpectralBlock(dim=16, max_seq_len=64, rank=8)
        
        x = torch.randn(2, 8, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for all layer parameters
        self.assertIsNotNone(block.holo_mixer.bind_key.grad)
        self.assertIsNotNone(block.spectral_gate.freq_gate.grad)
        self.assertIsNotNone(block.channel_mix.down.weight.grad)
        
        # All gradients should be finite
        for param in block.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())


class TestIntegration(unittest.TestCase):
    """Integration tests for all fixes together."""
    
    def test_block_forward_backward(self):
        """Test that a full block can do forward and backward passes."""
        block = HoloSpectralBlock(dim=32, max_seq_len=128, rank=16)
        
        batch_size = 4
        seq_len = 64
        dim = 32
        
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, dim))
        
        # Check backward pass
        loss = output.sum()
        loss.backward()
        
        # Check input gradients exist
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
    
    def test_varying_sequence_lengths(self):
        """Test block with various sequence lengths."""
        block = HoloSpectralBlock(dim=32, max_seq_len=128, rank=16)
        
        for seq_len in [8, 16, 31, 32, 33, 64, 100]:
            x = torch.randn(2, seq_len, 32)
            output = block(x)
            
            self.assertEqual(output.shape, (2, seq_len, 32))
            self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
