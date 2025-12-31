"""
Simple training script for HoloSpectralNet with synthetic data.
"""

import torch
import torch.nn.functional as F
from holospectral import hsn_small


def generate_synthetic_data(batch_size, seq_len, vocab_size):
    """
    Generate synthetic data for demonstration purposes.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        vocab_size: Size of the vocabulary
        
    Returns:
        tuple: (input_ids, target_ids)
    """
    # Random token sequences
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Targets are shifted inputs (simple language modeling)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, target_ids


def train_step(model, input_ids, target_ids, optimizer):
    """
    Perform a single training step.
    
    Args:
        model: HoloSpectralNet model
        input_ids: Input token indices
        target_ids: Target token indices
        optimizer: Optimizer
        
    Returns:
        Loss value
    """
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss (flatten for cross-entropy)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    # Configuration
    vocab_size = 1000  # Small vocab for demo
    batch_size = 4
    seq_len = 64
    num_steps = 100
    learning_rate = 3e-4
    
    # Initialize model
    print("Initializing HoloSpectralNet (small)...")
    model = hsn_small(vocab_size=vocab_size)
    
    # Count and print parameters
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    for step in range(num_steps):
        # Generate batch
        input_ids, target_ids = generate_synthetic_data(batch_size, seq_len, vocab_size)
        
        # Training step
        loss = train_step(model, input_ids, target_ids, optimizer)
        
        # Log progress
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{num_steps} - Loss: {loss:.4f}")
    
    print("\nTraining complete!")
    print(f"Final loss: {loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "holospectral_model.pt")
    print("Model saved to holospectral_model.pt")


if __name__ == "__main__":
    main()
