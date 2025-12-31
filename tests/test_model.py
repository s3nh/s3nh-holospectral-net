"""
Unit tests for HoloSpectralNet.
"""

import unittest
import torch

from holospectral import (
    HoloSpectralNet,
    HolographicMixer,
    SpectralGate,
    LiteChannelMix,
    HoloSpectralBlock,
    RotaryEmbedding,
    hsn_tiny,
    hsn_small,
    hsn_base,
    fft_circular_conv,
    fft_circular_corr,
)


class TestHolographicOperations(unittest.TestCase):
    """Test holographic operations."""
    
    def test_fft_circular_conv(self):
        """Test circular convolution."""
        x = torch.randn(2, 8, 16)
        y = torch.randn(16)
        result = fft_circular_conv(x, y)
        
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isfinite(result).all())
    
    def test_fft_circular_corr(self):
        """Test circular correlation."""
        x = torch.randn(2, 8, 16)
        y = torch.randn(16)
        result = fft_circular_corr(x, y)
        
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isfinite(result).all())


class TestLayers(unittest.TestCase):
    """Test individual layers."""
    
    def test_holographic_mixer(self):
        """Test HolographicMixer layer."""
        batch_size, seq_len, dim = 2, 8, 16
        x = torch.randn(batch_size, seq_len, dim)
        
        mixer = HolographicMixer(dim)
        output = mixer(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_spectral_gate(self):
        """Test SpectralGate layer."""
        batch_size, seq_len, dim = 2, 8, 16
        x = torch.randn(batch_size, seq_len, dim)
        
        gate = SpectralGate(max_seq_len=16)
        output = gate(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_lite_channel_mix(self):
        """Test LiteChannelMix layer."""
        batch_size, seq_len, dim = 2, 8, 16
        rank = 8
        x = torch.randn(batch_size, seq_len, dim)
        
        mix = LiteChannelMix(dim, rank)
        output = mix(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_rotary_embedding(self):
        """Test RotaryEmbedding layer."""
        batch_size, seq_len, dim = 2, 8, 16
        x = torch.randn(batch_size, seq_len, dim)
        
        rotary = RotaryEmbedding(dim)
        output = rotary(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_holospectral_block(self):
        """Test HoloSpectralBlock."""
        batch_size, seq_len, dim = 2, 8, 16
        x = torch.randn(batch_size, seq_len, dim)
        
        block = HoloSpectralBlock(dim, max_seq_len=16, rank=8)
        output = block(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, dim))
        self.assertTrue(torch.isfinite(output).all())


class TestModelConfigs(unittest.TestCase):
    """Test model configurations."""
    
    def test_hsn_tiny(self):
        """Test tiny model configuration."""
        vocab_size = 1000
        model = hsn_tiny(vocab_size=vocab_size)
        
        # Test instantiation
        self.assertIsInstance(model, HoloSpectralNet)
        
        # Test parameter count (should be around 400K)
        num_params = model.count_parameters()
        self.assertGreater(num_params, 0)
        self.assertLess(num_params, 1_000_000)  # Should be < 1M
        
        print(f"Tiny model parameters: {num_params:,}")
    
    def test_hsn_small(self):
        """Test small model configuration."""
        vocab_size = 1000
        model = hsn_small(vocab_size=vocab_size)
        
        # Test instantiation
        self.assertIsInstance(model, HoloSpectralNet)
        
        # Test parameter count (should be around 1.5M)
        num_params = model.count_parameters()
        self.assertGreater(num_params, 500_000)
        self.assertLess(num_params, 3_000_000)
        
        print(f"Small model parameters: {num_params:,}")
    
    def test_hsn_base(self):
        """Test base model configuration."""
        vocab_size = 1000
        model = hsn_base(vocab_size=vocab_size)
        
        # Test instantiation
        self.assertIsInstance(model, HoloSpectralNet)
        
        # Test parameter count (should be around 8M)
        num_params = model.count_parameters()
        self.assertGreater(num_params, 3_000_000)
        self.assertLess(num_params, 15_000_000)
        
        print(f"Base model parameters: {num_params:,}")


class TestModel(unittest.TestCase):
    """Test main model."""
    
    def test_forward_pass(self):
        """Test forward pass shape correctness."""
        vocab_size = 1000
        batch_size = 2
        seq_len = 16
        
        model = hsn_small(vocab_size=vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logits = model(input_ids)
        
        # Check output shape
        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
        self.assertTrue(torch.isfinite(logits).all())
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        vocab_size = 1000
        batch_size = 2
        seq_len = 16
        
        model = hsn_small(vocab_size=vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}")
    
    def test_weight_tying(self):
        """Test weight tying between embedding and output head."""
        vocab_size = 1000
        
        # Test with weight tying enabled
        model_tied = HoloSpectralNet(vocab_size=vocab_size, tie_weights=True)
        self.assertTrue(model_tied.token_emb.weight is model_tied.head.weight)
        
        # Test with weight tying disabled
        model_untied = HoloSpectralNet(vocab_size=vocab_size, tie_weights=False)
        self.assertFalse(model_untied.token_emb.weight is model_untied.head.weight)
    
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        vocab_size = 1000
        batch_size = 2
        
        model = hsn_small(vocab_size=vocab_size)
        
        for seq_len in [8, 16, 32, 64]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits = model(input_ids)
            
            self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
