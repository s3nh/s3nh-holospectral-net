"""
Text generation script for trained HoloSpectralNet on Tiny Shakespeare.

Usage:
    python generate_shakespeare. py --prompt "ROMEO:" --max_tokens 500
"""

import argparse
import torch
import torch. nn. functional as F
from typing import Optional

from holospectral import HoloSpectralNet


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    
    # Reconstruct model
    model = HoloSpectralNet(
        vocab_size=len(vocab['stoi']),
        dim=config.dim,
        depth=config.depth,
        max_seq_len=config.max_seq_len,
        rank=config.rank,
        tie_weights=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    vocab: dict,
    prompt: str,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    device: str = "cpu"
) -> str:
    """Generate text from a prompt."""
    stoi, itos = vocab['stoi'], vocab['itos']
    
    # Encode prompt
    idx = torch.tensor([stoi[ch] for ch in prompt], dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        # Crop to max sequence length
        idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
        
        # Get predictions
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k is not None: 
            v, _ = torch. topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch. cat([idx, idx_next], dim=1)
    
    # Decode
    return ''.join([itos[t.item()] for t in idx[0]])


def main():
    parser = argparse.ArgumentParser(description="Generate Shakespeare text with HoloSpectralNet")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/shakespeare_holospectral.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="ROMEO:",
                        help="Prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (0 for no filtering)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, vocab = load_model(args.checkpoint, args.device)
    print(f"Model loaded.  Vocabulary size: {len(vocab['stoi'])}")
    
    print(f"\n📜 Generating from prompt: '{args.prompt}'")
    print("-" * 60)
    
    generated = generate(
        model, vocab, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        device=args.device
    )
    
    print(generated)
    print("-" * 60)


if __name__ == "__main__":
    main()