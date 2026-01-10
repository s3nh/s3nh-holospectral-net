"""
Tiny Shakespeare training script for HoloSpectralNet. 

Downloads the Tiny Shakespeare dataset and trains a character-level language model
using the HoloSpectralNet architecture. 

Usage:
    python train_shakespeare. py
"""

import os
import math
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from holospectral import HoloSpectralNet


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration for Tiny Shakespeare."""
    # Data
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path: str = "data/tinyshakespeare. txt"
    train_split: float = 0.9
    
    # Model
    dim: int = 256
    depth: int = 6
    rank: int = 32
    max_seq_len: int = 256
    
    # Training
    batch_size: int = 128 
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 50000
    eval_interval: int = 5000
    eval_iters: int = 200
    warmup_iters: int = 100
    min_lr: float = 1e-5
    
    # Generation
    gen_max_tokens: int = 500
    gen_temperature: float = 0.8
    gen_top_k: int = 40
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints/shakespeare_holospectral. pt"


# =============================================================================
# Dataset
# =============================================================================

class CharDataset(Dataset):
    """Character-level dataset for Tiny Shakespeare."""
    
    def __init__(self, data:  str, block_size: int):
        self.block_size = block_size
        
        # Build vocabulary from data
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode the entire dataset
        self.data = torch.tensor([self.stoi[ch] for ch in data], dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode a string into token indices."""
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token indices into a string."""
        return ''.join([self.itos[t. item()] for t in tokens])


def download_dataset(url: str, path: str) -> str:
    """Download Tiny Shakespeare dataset if not present."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if not os.path.exists(path):
        print(f"Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded to {path}")
    else:
        print(f"Dataset already exists at {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def create_dataloaders(
    text: str, 
    block_size: int, 
    batch_size: int, 
    train_split: float
) -> Tuple[DataLoader, DataLoader, CharDataset]:
    """Create train and validation dataloaders."""
    
    # Create dataset
    dataset = CharDataset(text, block_size)
    
    # Split into train and validation
    n = int(train_split * len(dataset))
    train_data = torch.utils.data.Subset(dataset, range(n))
    val_data = torch.utils.data. Subset(dataset, range(n, len(dataset)))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset


# =============================================================================
# Training Utilities
# =============================================================================

def get_lr(it: int, config: TrainConfig) -> float:
    """Learning rate scheduler with warmup and cosine decay."""
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    
    # Cosine decay
    if it > config.max_iters:
        return config.min_lr
    
    decay_ratio = (it - config. warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math. cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    eval_iters: int,
    device: str
) -> dict:
    """Estimate loss on train and validation sets."""
    model.eval()
    out = {}
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        loader_iter = iter(loader)
        
        for _ in range(eval_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits. view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        
        out[split] = sum(losses) / len(losses)
    
    model.train()
    return out


@torch.no_grad()
def generate(
    model: nn.Module,
    dataset: CharDataset,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = "cpu"
) -> str:
    """Generate text from a prompt using the trained model."""
    model.eval()
    
    # Encode prompt
    idx = dataset.encode(prompt).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        # Crop to max sequence length if needed
        idx_cond = idx if idx.size(1) <= model.blocks[0].spectral_gate.freq_gate.size(0) * 2 - 2 else idx[:, -256:]
        
        # Get predictions
        logits = model(idx_cond)
        logits = logits[:, -1, : ] / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits. size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    
    return dataset.decode(idx[0])


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainConfig):
    """Main training function."""
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print(f"Using device: {config.device}")
    print("=" * 60)
    
    # Download and prepare data
    text = download_dataset(config.data_url, config. data_path)
    print(f"Dataset size: {len(text):,} characters")
    
    # Create dataloaders
    train_loader, val_loader, dataset = create_dataloaders(
        text, config.max_seq_len, config.batch_size, config.train_split
    )
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("=" * 60)
    
    # Initialize model
    model = HoloSpectralNet(
        vocab_size=dataset.vocab_size,
        dim=config.dim,
        depth=config.depth,
        max_seq_len=config.max_seq_len,
        rank=config.rank,
        tie_weights=True
    ).to(config.device)
    
    num_params = model.count_parameters()
    print(f"Model:  HoloSpectralNet")
    print(f"  - dim: {config.dim}")
    print(f"  - depth:  {config.depth}")
    print(f"  - rank: {config.rank}")
    print(f"  - max_seq_len: {config.max_seq_len}")
    print(f"  - Parameters: {num_params:,}")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Training loop
    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    
    print(f"Starting training for {config.max_iters} iterations...")
    print()
    
    for iter_num in range(config.max_iters):
        # Update learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(config.device), y.to(config.device)
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn. utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Evaluation
        if (iter_num + 1) % config.eval_interval == 0 or iter_num == 0:
            losses = estimate_loss(
                model, train_loader, val_loader, 
                config.eval_iters, config.device
            )
            
            print(f"Iter {iter_num + 1:5d} | "
                  f"Train Loss: {losses['train']:.4f} | "
                  f"Val Loss: {losses['val']:.4f} | "
                  f"LR: {lr:.2e}")
            
            # Save checkpoint if best
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                os.makedirs(os.path.dirname(config. checkpoint_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'vocab':  {'stoi': dataset.stoi, 'itos': dataset.itos}
                }, config.checkpoint_path)
                print(f"  └── New best model saved!  (val_loss:  {best_val_loss:.4f})")
    
    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    # Generate sample text
    print("\n📜 Sample Generation:")
    print("-" * 60)
    prompt = "ROMEO:"
    generated = generate(
        model, dataset, prompt,
        max_new_tokens=config.gen_max_tokens,
        temperature=config.gen_temperature,
        top_k=config.gen_top_k,
        device=config.device
    )
    print(generated)
    print("-" * 60)
    
    return model, dataset


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__": 
    config = TrainConfig()
    model, dataset = train(config)