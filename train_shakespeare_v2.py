"""
Tiny Shakespeare training script for HoloSpectralNet (v2 - with fixes for repetition/collapse).

Key fixes:
1. Label smoothing to prevent overconfidence
2. Dropout for regularization  
3. Higher temperature during training
4. Repetition penalty during generation
5. Better hyperparameters

Usage:
    python train_shakespeare_v2.py
"""

import os
import math
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch. nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from holospectral import HoloSpectralNet


@dataclass
class TrainConfig: 
    """Training configuration - tuned to prevent mode collapse."""
    # Data
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path: str = "data/tinyshakespeare. txt"
    train_split: float = 0.9
    
    # Model
    dim: int = 256
    depth: int = 6
    rank: int = 32
    max_seq_len: int = 256
    dropout: float = 0.1  # ADD DROPOUT
    
    # Training - ADJUSTED
    batch_size: int = 32  # Smaller batch for more gradient noise
    learning_rate: float = 1e-3  # Higher LR initially
    weight_decay: float = 0.01  # Less aggressive weight decay
    max_iters: int = 50000  # Fewer iters to prevent overfitting
    eval_interval: int = 1500
    eval_iters: int = 100
    warmup_iters: int = 200
    min_lr: float = 1e-4
    label_smoothing: float = 0.1  # LABEL SMOOTHING
    
    # Generation - ADJUSTED
    gen_max_tokens: int = 500
    gen_temperature: float = 1.0  # Higher temperature
    gen_top_k: int = 50
    gen_top_p: float = 0.9  # Nucleus sampling
    repetition_penalty: float = 1.2  # Penalize repetition
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints/shakespeare_holospectral_v2.pt"


class CharDataset(Dataset):
    """Character-level dataset for Tiny Shakespeare."""
    
    def __init__(self, data: str, block_size: int):
        self.block_size = block_size
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = torch.tensor([self.stoi[ch] for ch in data], dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi. get(ch, 0) for ch in text], dtype=torch. long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        return ''.join([self.itos.get(t.item(), '?') for t in tokens])


def download_dataset(url: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(url, path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def create_dataloaders(text, block_size, batch_size, train_split):
    dataset = CharDataset(text, block_size)
    n = int(train_split * len(dataset))
    train_data = torch.utils.data. Subset(dataset, range(n))
    val_data = torch.utils.data.Subset(dataset, range(n, len(dataset)))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, dataset


def get_lr(it: int, config: TrainConfig) -> float:
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.max_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math. pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    model.eval()
    out = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        loader_iter = iter(loader)
        for _ in range(min(eval_iters, len(loader))):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                break
            x, y = x. to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits. view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if losses else 0
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
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    device: str = "cpu"
) -> str:
    """Generate text with repetition penalty and nucleus sampling."""
    model.eval()
    
    idx = dataset.encode(prompt).unsqueeze(0).to(device)
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Crop context
        idx_cond = idx[: , -256:] if idx.size(1) > 256 else idx
        
        # Forward pass
        logits = model(idx_cond)
        logits = logits[:, -1, :]. clone()
        
        # Apply repetition penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for token_id in set(generated_tokens[-50:]):  # Look at last 50 tokens
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
        
        # Temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[: , 1:] = sorted_indices_to_remove[:, :-1]. clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        generated_tokens.append(idx_next. item())
        idx = torch.cat([idx, idx_next], dim=1)
    
    return dataset.decode(idx[0])


def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print(f"Device: {config.device}")
    print("=" * 60)
    
    # Data
    text = download_dataset(config.data_url, config. data_path)
    print(f"Dataset:  {len(text):,} characters")
    
    train_loader, val_loader, dataset = create_dataloaders(
        text, config.max_seq_len, config.batch_size, config.train_split
    )
    print(f"Vocab size: {dataset.vocab_size}")
    print("=" * 60)
    
    # Model
    model = HoloSpectralNet(
        vocab_size=dataset.vocab_size,
        dim=config.dim,
        depth=config.depth,
        max_seq_len=config.max_seq_len,
        rank=config.rank,
        tie_weights=True
    ).to(config.device)
    
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Label smoothing: {config.label_smoothing}")
    print(f"Dropout: {config. dropout}")
    print("=" * 60)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.99)  # Slightly higher beta2
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # Training
    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    
    print(f"Training for {config.max_iters} iterations...")
    print()
    
    for iter_num in range(config.max_iters):
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x. to(config.device), y.to(config.device)
        
        # Forward with label smoothing
        logits = model(x)
        loss = criterion(logits. view(-1, logits.size(-1)), y.view(-1))
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Eval
        if (iter_num + 1) % config.eval_interval == 0 or iter_num == 0:
            losses = estimate_loss(model, train_loader, val_loader, config.eval_iters, config. device)
            
            print(f"Iter {iter_num + 1:5d} | Train:  {losses['train']:.4f} | Val: {losses['val']:.4f} | LR: {lr:.2e}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                os.makedirs(os.path.dirname(config. checkpoint_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'vocab':  {'stoi': dataset.stoi, 'itos': dataset.itos}
                }, config.checkpoint_path)
                print(f"  └── Best model saved!")
    
    print()
    print("=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    # Generate with improved sampling
    print("\n📜 Sample Generation (with repetition penalty + nucleus sampling):")
    print("-" * 60)
    
    for prompt in ["ROMEO:", "JULIET:", "First Citizen:"]:
        print(f"\n[Prompt: {prompt}]")
        generated = generate(
            model, dataset, prompt,
            max_new_tokens=200,
            temperature=config.gen_temperature,
            top_k=config.gen_top_k,
            top_p=config.gen_top_p,
            repetition_penalty=config. repetition_penalty,
            device=config.device
        )
        print(generated[: 300])
        print()
    
    print("-" * 60)
    return model, dataset


if __name__ == "__main__":
    config = TrainConfig()
    train(config)