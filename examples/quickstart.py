"""
Quick start example for HoloSpectralNet.
"""

import torch
from holospectral import hsn_tiny, hsn_small, hsn_base


def main():
    print("HoloSpectralNet Quick Start Example")
    print("=" * 50)
    
    # Create models with different configurations
    vocab_size = 1000
    
    print(f"\nCreating models with vocab_size={vocab_size}...")
    
    # Tiny model
    model_tiny = hsn_tiny(vocab_size=vocab_size)
    params_tiny = model_tiny.count_parameters()
    print(f"\nTiny model parameters: {params_tiny:,}")
    
    # Small model
    model_small = hsn_small(vocab_size=vocab_size)
    params_small = model_small.count_parameters()
    print(f"Small model parameters: {params_small:,}")
    
    # Base model
    model_base = hsn_base(vocab_size=vocab_size)
    params_base = model_base.count_parameters()
    print(f"Base model parameters: {params_base:,}")
    
    # Run a forward pass
    print("\n" + "=" * 50)
    print("Running forward pass with small model...")
    
    batch_size = 2
    seq_len = 32
    
    # Create random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model_small(input_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: (batch={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    
    # Verify output shape
    assert logits.shape == (batch_size, seq_len, vocab_size), "Output shape mismatch!"
    print("\n✓ Forward pass successful!")
    
    print("\n" + "=" * 50)
    print("Parameter Count Summary:")
    print(f"  Tiny:  {params_tiny:,} parameters")
    print(f"  Small: {params_small:,} parameters")
    print(f"  Base:  {params_base:,} parameters")
    print("=" * 50)


if __name__ == "__main__":
    main()
