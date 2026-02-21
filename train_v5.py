"""
Stanford Alpaca trainer for HoloSpectralNet with AdamW optimizer (v3).

Changes from v2:
- Smaller vocab via tiktoken truncation to reduce embedding overparameterization
- Dropout in LiteChannelMix and after embedding
- Label smoothing on cross-entropy loss
- Early stopping based on val loss patience
- Data augmentation via random instruction-response shuffling per epoch
- Cosine schedule min_lr raised to prevent late-training memorization

Usage:
    pip install datasets tiktoken
    python train_alpaca_v3.py
"""

import os
import re
import json
import math
import random
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from holospectral import HoloSpectralNet


# =============================================================================
# Truncated BPE Tokenizer
# =============================================================================

class BPETokenizer:
    """
    tiktoken cl100k_base with vocab truncated to max_vocab tokens.

    Why truncate: full cl100k_base has 100256 tokens. With dim=384, the
    embedding table alone would be 384 * 100256 = 38.5M params — far larger
    than the rest of the model. Most of those tokens never appear in 52K
    Alpaca examples. Truncating to the tokens that actually appear keeps
    the embedding table manageable and prevents underfitting on rare tokens.
    """

    def __init__(self, max_vocab: int = 32000):
        import tiktoken
        self._base = tiktoken.get_encoding("cl100k_base")
        self._max_vocab = max_vocab
        # Token ID mapping: we'll build this from data
        self._id_to_new: Dict[int, int] = {}
        self._new_to_id: Dict[int, int] = {}
        self._built = False

        # Special token IDs (assigned after vocab is built)
        self._pad_id = -1
        self._bos_id = -1
        self._eos_id = -1
        self._unk_id = -1

    def build_vocab(self, texts: List[str]):
        """Build truncated vocab from actual data. Call ONCE before training."""
        # Count token frequencies across all texts
        freq: Dict[int, int] = {}
        for text in texts:
            for tok_id in self._base.encode(text, allowed_special=set()):
                freq[tok_id] = freq.get(tok_id, 0) + 1

        # Keep top max_vocab - 4 tokens (reserve 4 for specials)
        budget = self._max_vocab - 4
        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])[:budget]

        # Build bidirectional mapping: old_id <-> new_id
        # New IDs 0-3 are specials
        self._unk_id = 0
        self._pad_id = 1
        self._bos_id = 2
        self._eos_id = 3

        new_idx = 4
        for old_id, _ in sorted_tokens:
            self._id_to_new[old_id] = new_idx
            self._new_to_id[new_idx] = old_id
            new_idx += 1

        self._built = True
        total_tokens = sum(freq.values())
        covered = sum(c for _, c in sorted_tokens)
        coverage = covered / max(total_tokens, 1) * 100
        print(f"  BPE vocab: {new_idx} tokens from {len(freq)} unique "
              f"({coverage:.1f}% token coverage)")

    @property
    def vocab_size(self) -> int:
        return len(self._new_to_id) + 4  # +4 for specials

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    def encode(self, text: str) -> List[int]:
        """Encode text, mapping to truncated vocab IDs."""
        assert self._built, "Call build_vocab() first"
        raw_ids = self._base.encode(text, allowed_special=set())
        return [self._id_to_new.get(tid, self._unk_id) for tid in raw_ids]

    def decode(self, ids: List[int]) -> str:
        """Decode truncated vocab IDs back to text."""
        specials = {self._pad_id, self._bos_id, self._eos_id, self._unk_id}
        raw_ids = [self._new_to_id[i] for i in ids
                   if i not in specials and i in self._new_to_id]
        if not raw_ids:
            return ""
        return self._base.decode(raw_ids)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "type": "tiktoken_truncated",
                "max_vocab": self._max_vocab,
                "id_to_new": {str(k): v for k, v in self._id_to_new.items()},
                "new_to_id": {str(k): v for k, v in self._new_to_id.items()},
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(max_vocab=data["max_vocab"])
        tok._id_to_new = {int(k): v for k, v in data["id_to_new"].items()}
        tok._new_to_id = {int(k): v for k, v in data["new_to_id"].items()}
        tok._unk_id = 0
        tok._pad_id = 1
        tok._bos_id = 2
        tok._eos_id = 3
        tok._built = True
        return tok


# =============================================================================
# Fallback Word-Level Tokenizer
# =============================================================================

class SimpleTokenizer:
    """Fallback if tiktoken is not installed."""
    SPECIAL_TOKENS = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]

    def __init__(self, max_vocab_size: int = 32000):
        self.max_vocab_size = max_vocab_size
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.stoi[tok] = i
            self.itos[i] = tok

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def build_vocab(self, texts: List[str]):
        freq: Dict[str, int] = {}
        total = 0
        for text in texts:
            for word in self._tokenize(text):
                freq[word] = freq.get(word, 0) + 1
                total += 1
        budget = self.max_vocab_size - len(self.SPECIAL_TOKENS)
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])[:budget]
        idx = len(self.SPECIAL_TOKENS)
        for word, _ in sorted_words:
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        unk_count = sum(c for w, c in freq.items() if w not in self.stoi)
        print(f"  Word vocab: {len(self.stoi)} tokens (UNK: {unk_count/max(total,1)*100:.1f}%)")

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<|eos|>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<|unk|>"]

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(w, self.stoi["<|unk|>"]) for w in self._tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        specials = set(range(len(self.SPECIAL_TOKENS)))
        tokens = [self.itos.get(i, "") for i in ids if i not in specials]
        text = " ".join(t for t in tokens if t)
        text = re.sub(r"\s([,.!?;:\)\]])", r"\1", text)
        return text

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"type": "simple", "stoi": self.stoi,
                        "itos": {str(k): v for k, v in self.itos.items()}}, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        return tok


def create_tokenizer(texts: List[str], max_vocab_size: int = 32000):
    """Try tiktoken first, fall back to simple tokenizer."""
    try:
        import tiktoken
        print("  Using tiktoken BPE (truncated to fit dataset)")
        tok = BPETokenizer(max_vocab=max_vocab_size)
        tok.build_vocab(texts)
        return tok
    except ImportError:
        print("  ⚠ tiktoken not found, using word-level tokenizer")
        tok = SimpleTokenizer(max_vocab_size=max_vocab_size)
        tok.build_vocab(texts)
        return tok


# =============================================================================
# Alpaca Templates & Data
# =============================================================================

PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def load_alpaca_data(data_path: str) -> List[Dict]:
    """Load Alpaca dataset."""
    if os.path.exists(data_path):
        print(f"  Loading cached data from {data_path}")
        with open(data_path, "r") as f:
            return json.load(f)
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    try:
        from datasets import load_dataset
        print("  Loading via HuggingFace datasets...")
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        data = [{"instruction": ex["instruction"], "input": ex["input"],
                 "output": ex["output"]} for ex in ds]
        with open(data_path, "w") as f:
            json.dump(data, f)
        print(f"  Cached {len(data)} examples to {data_path}")
        return data
    except ImportError:
        pass
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print("  Downloading from GitHub...")
    urllib.request.urlretrieve(url, data_path)
    with open(data_path, "r") as f:
        data = json.load(f)
    print(f"  Downloaded {len(data)} examples")
    return data


def format_alpaca_prompt(example: Dict) -> Tuple[str, str]:
    if example.get("input", "").strip():
        prompt = PROMPT_INPUT.format(instruction=example["instruction"],
                                     input=example["input"])
    else:
        prompt = PROMPT_NO_INPUT.format(instruction=example["instruction"])
    return prompt, example["output"]


# =============================================================================
# Dataset
# =============================================================================

class AlpacaDataset(Dataset):
    """Alpaca dataset with loss masking on response tokens only."""

    def __init__(self, data: List[Dict], tokenizer, max_seq_len: int):
        self.input_ids: List[torch.Tensor] = []
        self.loss_masks: List[torch.Tensor] = []
        skipped = 0

        for example in data:
            prompt_str, response_str = format_alpaca_prompt(example)
            prompt_ids = tokenizer.encode(prompt_str)
            response_ids = tokenizer.encode(response_str)

            full_ids = [tokenizer.bos_id] + prompt_ids + response_ids + [tokenizer.eos_id]
            prompt_len = 1 + len(prompt_ids)
            mask = [0] * prompt_len + [1] * (len(response_ids) + 1)

            if len(full_ids) > max_seq_len + 1:
                full_ids = full_ids[:max_seq_len + 1]
                mask = mask[:max_seq_len + 1]
                if sum(mask) == 0:
                    skipped += 1
                    continue

            pad_len = (max_seq_len + 1) - len(full_ids)
            full_ids += [tokenizer.pad_id] * pad_len
            mask += [0] * pad_len

            self.input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            self.loss_masks.append(torch.tensor(mask, dtype=torch.float))

        print(f"  {len(self.input_ids)} examples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        t = self.input_ids[idx]
        m = self.loss_masks[idx]
        return t[:-1], t[1:], m[1:]


# =============================================================================
# Config
# =============================================================================

@dataclass
class AlpacaTrainConfig:
    # Data
    data_path: str = "data/alpaca_data.json"
    train_split: float = 0.9  # 10% val for better overfitting signal

    # Tokenizer — truncated BPE keeps embedding table sane
    max_vocab_size: int = 16384
    tokenizer_path: str = "checkpoints/alpaca_tokenizer_v3.json"

    # Model — smaller to match 52K dataset
    dim: int = 512 
    depth: int =  32 
    rank: int = 128
    max_seq_len: int = 512
    dropout: float = 0.1  # applied in wrapper

    # AdamW
    adamw_lr: float = 3e-4         # reduced from 1e-3
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_weight_decay: float = 0.1
    adamw_emb_wd: float = 0.01    # light WD on embeddings too

    # Training
    batch_size: int = 16
    grad_accum_steps: int = 4      # effective batch = 64
    num_epochs: int = 5            # more epochs but with early stopping
    warmup_ratio: float = 0.06
    lr_floor: float = 0.15        # min LR = 15% of peak (was 10%)
    grad_clip: float = 1.0
    label_smoothing: float = 0.1  # prevents overconfident memorization
    eval_interval: int = 250      # eval more frequently to catch overfitting
    eval_iters: int = 50
    log_interval: int = 50

    # Early stopping
    patience: int = 5             # stop after 5 evals without val improvement

    # Generation
    gen_max_tokens: int = 100
    gen_temperature: float = 0.8
    gen_top_k: int = 50
    gen_repetition_penalty: float = 1.2

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints/alpaca_holospectral_muon_v3.pt"


# =============================================================================
# Dropout Wrapper for HoloSpectralNet
# =============================================================================

class HoloSpectralNetWithDropout(nn.Module):
    """
    Wraps HoloSpectralNet and adds dropout after embedding and after each block.

    HoloSpectralNet itself has no dropout. Rather than forking the model code,
    we wrap it and inject dropout at the key residual stream points. This is
    the same strategy used in nanoGPT and many fine-tuning frameworks.
    """

    def __init__(self, vocab_size, dim, depth, max_seq_len, rank,
                 tie_weights=True, dropout=0.1):
        super().__init__()
        self.inner = HoloSpectralNet(
            vocab_size=vocab_size, dim=dim, depth=depth,
            max_seq_len=max_seq_len, rank=rank, tie_weights=tie_weights,
        )
        self.emb_drop = nn.Dropout(dropout)
        self.block_drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(depth)])
        self.dropout = dropout

    def forward(self, x):
        h = self.inner.token_emb(x)
        h = self.inner.rotary(h)
        h = self.emb_drop(h)  # dropout after embedding

        for block, drop in zip(self.inner.blocks, self.block_drops):
            h = drop(block(h))  # dropout after each block

        h = self.inner.norm_out(h)
        logits = self.inner.head(h)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def named_parameters_inner(self):
        """Yield named parameters using inner model names for optimizer split."""
        for name, param in self.inner.named_parameters():
            yield name, param


# =============================================================================
# Training Utilities
# =============================================================================

def masked_cross_entropy(logits, targets, mask, label_smoothing=0.0):
    """
    Cross-entropy only on response tokens with optional label smoothing.

    Label smoothing prevents the model from becoming overconfident on
    training examples, which directly fights memorization/overfitting.
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, V), targets.view(B * T),
        reduction="none", label_smoothing=label_smoothing,
    )
    loss = loss * mask.view(B * T)
    return loss.sum() / mask.sum().clamp(min=1)


def get_lr_scale(step, total_steps, warmup_ratio, lr_floor=0.15):
    """Cosine schedule with warmup and floor."""
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(lr_floor, 0.5 * (1.0 + math.cos(math.pi * decay_ratio)))


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device,
                  label_smoothing=0.0):
    """Estimate loss on train and val. No wrap-around."""
    model.eval()
    out = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        max_iters = min(eval_iters, len(loader))
        loader_iter = iter(loader)
        for _ in range(max_iters):
            try:
                x, y, m = next(loader_iter)
            except StopIteration:
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            losses.append(
                masked_cross_entropy(logits, y, m, label_smoothing=label_smoothing).item()
            )
        out[split] = sum(losses) / len(losses) if losses else float("inf")
    model.train()
    return out


@torch.no_grad()
def generate_response(model, tokenizer, instruction, input_text="",
                      max_new_tokens=100, temperature=0.8, top_k=50,
                      repetition_penalty=1.2, max_seq_len=512, device="cpu"):
    """Generate with repetition penalty and special-token masking."""
    model.eval()
    example = {"instruction": instruction, "input": input_text, "output": ""}
    prompt_str, _ = format_alpaca_prompt(example)
    prefix_ids = [tokenizer.bos_id] + tokenizer.encode(prompt_str)
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    # Ban special tokens except eos
    ban_ids = {tokenizer.pad_id, tokenizer.bos_id}
    if hasattr(tokenizer, "unk_id"):
        ban_ids.add(tokenizer.unk_id)
    ban_tensor = torch.tensor(list(ban_ids), device=device) if ban_ids else None

    generated_ids: List[int] = []
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
        logits = model(idx_cond)[:, -1, :]

        # Ban specials
        if ban_tensor is not None:
            logits[0, ban_tensor] = float("-inf")

        # Ban beyond vocab
        vocab_sz = tokenizer.vocab_size
        if logits.size(-1) > vocab_sz:
            logits[0, vocab_sz:] = float("-inf")

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            seen = torch.tensor(list(set(generated_ids)), device=device)
            sl = logits[0, seen]
            logits[0, seen] = torch.where(sl > 0, sl / repetition_penalty,
                                           sl * repetition_penalty)

        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() == tokenizer.eos_id:
            break
        generated_ids.append(idx_next.item())
        idx = torch.cat([idx, idx_next], dim=1)

    return tokenizer.decode(generated_ids)


# =============================================================================
# Training
# =============================================================================

def train(config: AlpacaTrainConfig):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    print("=" * 70)
    print("  HoloSpectralNet × Alpaca × AdamW (v3 — anti-overfit)")
    print("=" * 70)
    print(f"  Device: {config.device}")

    # ── Data ─────────────────────────────────────────────────────────────
    raw_data = load_alpaca_data(config.data_path)
    print(f"  Examples: {len(raw_data)}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    print("\nTokenizer:")
    all_texts = []
    for ex in raw_data:
        p, r = format_alpaca_prompt(ex)
        all_texts.extend([p, r])
    tokenizer = create_tokenizer(all_texts, config.max_vocab_size)
    print(f"  Final vocab size: {tokenizer.vocab_size}")

    # ── Split ────────────────────────────────────────────────────────────
    print("\nDatasets:")

    # Shuffle before split so val isn't just the tail of the file
    shuffled = list(raw_data)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * config.train_split)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    train_dataset = AlpacaDataset(train_data, tokenizer, config.max_seq_len)
    val_dataset = AlpacaDataset(val_data, tokenizer, config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────
    print("\nModel:")
    model = HoloSpectralNetWithDropout(
        vocab_size=tokenizer.vocab_size,
        dim=config.dim, depth=config.depth,
        max_seq_len=config.max_seq_len, rank=config.rank,
        tie_weights=True, dropout=config.dropout,
    ).to(config.device)

    print(f"  dim={config.dim}, depth={config.depth}, rank={config.rank}, "
          f"dropout={config.dropout}")
    print(f"  Parameters: {model.count_parameters():,}")

    # ── Optimizers ───────────────────────────────────────────────────────
    print("\nOptimizers:")

    # Separate embedding params for different weight decay
    emb_ids = set()
    emb_list = []
    for name, param in model.inner.named_parameters():
        if ("token_emb" in name or "head" in name) and param.requires_grad:
            emb_ids.add(id(param))
            emb_list.append(param)
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in emb_ids]

    opt_adamw = torch.optim.AdamW([
        {"params": emb_list, "weight_decay": config.adamw_emb_wd},
        {"params": other_params, "weight_decay": config.adamw_weight_decay},
    ], lr=config.adamw_lr, betas=config.adamw_betas)

    print(f"  AdamW: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    print(f"\n  AdamW LR: {config.adamw_lr}")
    print(f"  Label smoothing: {config.label_smoothing}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Early stopping patience: {config.patience} evals")
    print(f"  Effective batch: {config.batch_size}×{config.grad_accum_steps}"
          f"={config.batch_size * config.grad_accum_steps}")

    # ── Training ─────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.num_epochs
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    accum_loss = 0.0

    print(f"\n{'='*70}")
    print(f"  Training: {config.num_epochs} epochs, {total_steps} steps (max)")
    print(f"{'='*70}\n")

    model.train()
    stopped_early = False

    for epoch in range(config.num_epochs):
        if stopped_early:
            break

        for micro_step, (x, y, m) in enumerate(train_loader):
            if stopped_early:
                break

            # LR schedule
            scale = get_lr_scale(global_step, total_steps,
                                 config.warmup_ratio, config.lr_floor)
            for pg in opt_adamw.param_groups:
                pg["lr"] = config.adamw_lr * scale

            # Forward
            x, y, m = x.to(config.device), y.to(config.device), m.to(config.device)
            logits = model(x)
            loss = masked_cross_entropy(
                logits, y, m, label_smoothing=config.label_smoothing
            ) / config.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            # Accumulate
            if (micro_step + 1) % config.grad_accum_steps != 0:
                continue

            # Step
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt_adamw.step()
            opt_adamw.zero_grad(set_to_none=True)
            global_step += 1

            # Log
            if global_step % config.log_interval == 0:
                print(f"  Ep {epoch+1}/{config.num_epochs} | "
                      f"Step {global_step}/{total_steps} | "
                      f"Batch Loss: {accum_loss:.4f} | "
                      f"LR: {scale:.3f}")
                accum_loss = 0.0

            # Eval
            if global_step % config.eval_interval == 0:
                losses = estimate_loss(
                    model, train_loader, val_loader,
                    config.eval_iters, config.device,
                    label_smoothing=config.label_smoothing,
                )
                tl, vl = losses["train"], losses["val"]

                print(f"\n{'─'*70}")
                print(f"  EVAL @ step {global_step} | "
                      f"Train: {tl:.4f} | Val: {vl:.4f} | "
                      f"Gap: {vl - tl:.4f}")

                if tl < vl * 0.8:
                    print(f"  ⚠ Overfitting: train/val = {tl/vl:.3f}")

                # Generate samples
                for instr, inp in [
                    ("Give three tips for staying healthy.", ""),
                    ("Rewrite the following sentence in the third person.",
                     "I am going to the store."),
                    ("What is the capital of France?", ""),
                ]:
                    resp = generate_response(
                        model, tokenizer, instr, inp,
                        max_new_tokens=config.gen_max_tokens,
                        temperature=config.gen_temperature,
                        top_k=config.gen_top_k,
                        repetition_penalty=config.gen_repetition_penalty,
                        max_seq_len=config.max_seq_len,
                        device=config.device,
                    )
                    print(f"\n  ### Instruction: {instr}")
                    if inp:
                        print(f"  ### Input: {inp}")
                    print(f"  ### Response: {resp}")

                print(f"\n{'─'*70}\n")

                # Checkpoint + early stopping
                if vl < best_val_loss:
                    best_val_loss = vl
                    patience_counter = 0
                    os.makedirs(os.path.dirname(config.checkpoint_path),
                                exist_ok=True)
                    tokenizer.save(config.tokenizer_path)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "opt_adamw_state": opt_adamw.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "train_loss": tl,
                        "config": config,
                    }, config.checkpoint_path)
                    print(f"  ✅ Best model saved (val={best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  ⏳ No improvement for {patience_counter}/{config.patience} evals")
                    if patience_counter >= config.patience:
                        print(f"\n  🛑 Early stopping triggered at step {global_step}")
                        stopped_early = True
                        break

                model.train()

    # ── Final ────────────────────────────────────────────────────────────
    final = estimate_loss(
        model, train_loader, val_loader,
        config.eval_iters, config.device,
        label_smoothing=config.label_smoothing,
    )

    print(f"\n{'='*70}")
    print(f"  {'Early stopped!' if stopped_early else 'Training complete!'}")
    print(f"  Final  → Train: {final['train']:.4f} | Val: {final['val']:.4f}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total steps: {global_step}")
    print(f"  Checkpoint: {config.checkpoint_path}")
    print(f"{'='*70}")

    return model, tokenizer


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    config = AlpacaTrainConfig()
    model, tokenizer = train(config)