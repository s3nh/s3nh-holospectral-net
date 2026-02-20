"""
Stanford Alpaca trainer for HoloSpectralNet with Muon optimizer (v2).

Fixes from v1:
- Replaced naive word tokenizer with tiktoken BPE (cl100k_base)
- Added repetition penalty to generation
- Fixed special token ID handling for tiktoken decode
- Gradient accumulation for effective larger batch size
- Proper LR schedule for Muon + AdamW dual optimizer

Usage:
    pip install datasets tiktoken
    python train_alpaca_v2.py
"""

import os
import re
import json
import math
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from holospectral import HoloSpectralNet


# =============================================================================
# Muon Optimizer (from KellerJordan/Muon, single-GPU)
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# =============================================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the
    slope at zero. Runs stably in bfloat16 on GPU.

    From: https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    # Work with the smaller dimension for efficiency
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize spectral norm to at most 1 (convergence requirement)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Quintic Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-schulz.

    For each 2D parameter, Muon:
    1. Maintains a Nesterov momentum buffer
    2. Orthogonalizes the update via Newton-Schulz iteration
    3. Scales by sqrt(max(nrows, ncols) / min(nrows, ncols))

    Only for 2D hidden weight matrices. Embeddings, biases, LayerNorm,
    1D params, and output heads must use AdamW.

    Args:
        lr: Learning rate in spectral norm units (default: 0.02)
        momentum: Nesterov momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        weight_decay: Decoupled weight decay (default: 0.0)
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]

                # Momentum: buf = beta * buf + (1 - beta) * grad
                buf.lerp_(g, 1 - momentum)

                # Nesterov: use grad + beta * buf
                if nesterov:
                    update = g.lerp_(buf, momentum)
                else:
                    update = buf

                # Orthogonalize via Newton-Schulz
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Scale by aspect ratio
                update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                # Decoupled weight decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                # Apply update
                p.data.add_(update.to(p.data.dtype), alpha=-lr)


# =============================================================================
# Parameter Split for Muon + AdamW
# =============================================================================

def split_params_for_muon(model: HoloSpectralNet):
    """
    Split HoloSpectralNet parameters:
    - Muon: 2D hidden weight matrices (channel_mix.down.weight, channel_mix.up.weight)
    - AdamW: Everything else (embeddings, LayerNorm, 1D params, head)
    """
    muon_params, adamw_params = [], []
    muon_names, adamw_names = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_hidden_2d = (
            param.ndim == 2
            and "channel_mix" in name
            and "weight" in name
            and "scale" not in name
        )

        if is_hidden_2d:
            muon_params.append(param)
            muon_names.append(name)
        else:
            adamw_params.append(param)
            adamw_names.append(name)

    muon_total = sum(p.numel() for p in muon_params)
    adamw_total = sum(p.numel() for p in adamw_params)
    print(f"  Muon:  {len(muon_params)} tensors, {muon_total:,} params")
    print(f"    → {', '.join(muon_names[:6])}{'...' if len(muon_names) > 6 else ''}")
    print(f"  AdamW: {len(adamw_params)} tensors, {adamw_total:,} params")
    print(f"    → {', '.join(adamw_names[:6])}{'...' if len(adamw_names) > 6 else ''}")

    return muon_params, adamw_params


# =============================================================================
# BPE Tokenizer (tiktoken cl100k_base)
# =============================================================================

class BPETokenizer:
    """
    Wraps tiktoken's cl100k_base with special tokens for instruction-following.

    cl100k_base has 100256 base tokens (IDs 0–100255).
    We add 3 special tokens at IDs 100256, 100257, 100258.
    Total vocab = 100259.

    The model's output head has 100259 logits. During generation we mask
    special token logits so they cannot be sampled. During decode we filter
    to base-only IDs so tiktoken never sees an unknown ID.
    """

    def __init__(self):
        import tiktoken
        self._base = tiktoken.get_encoding("cl100k_base")
        self._n_base = self._base.n_vocab  # 100256

        self.SPECIAL_TOKENS = {
            "<|pad|>": self._n_base,      # 100256
            "<|bos|>": self._n_base + 1,  # 100257
            "<|eos|>": self._n_base + 2,  # 100258
        }

    @property
    def vocab_size(self) -> int:
        return self._n_base + len(self.SPECIAL_TOKENS)  # 100259

    @property
    def n_base(self) -> int:
        """Number of base (decodable) tokens."""
        return self._n_base

    @property
    def pad_id(self) -> int:
        return self.SPECIAL_TOKENS["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.SPECIAL_TOKENS["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.SPECIAL_TOKENS["<|eos|>"]

    def encode(self, text: str) -> List[int]:
        """Encode text to BPE token IDs (base tokens only, 0 to n_base-1)."""
        return self._base.encode(text, allowed_special=set())

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text. Filters out any non-base IDs safely."""
        filtered = [i for i in ids if 0 <= i < self._n_base]
        if not filtered:
            return ""
        return self._base.decode(filtered)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"type": "tiktoken_cl100k_base"}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        return cls()


# =============================================================================
# Fallback Word-Level Tokenizer (if tiktoken unavailable)
# =============================================================================

class SimpleTokenizer:
    """Fallback word-level tokenizer if tiktoken is not installed."""

    SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]

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
        total_tokens = 0
        for text in texts:
            for word in self._tokenize(text):
                freq[word] = freq.get(word, 0) + 1
                total_tokens += 1
        budget = self.max_vocab_size - len(self.SPECIAL_TOKENS)
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])[:budget]
        idx = len(self.SPECIAL_TOKENS)
        for word, _ in sorted_words:
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        unk_count = sum(c for w, c in freq.items() if w not in self.stoi)
        unk_rate = unk_count / max(total_tokens, 1) * 100
        print(f"Tokenizer: {len(self.stoi)} tokens (UNK rate: {unk_rate:.1f}%)")

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def n_base(self) -> int:
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
        special_set = set(range(len(self.SPECIAL_TOKENS)))
        tokens = [self.itos.get(i, "") for i in ids if i not in special_set]
        text = " ".join(t for t in tokens if t)
        text = re.sub(r"\s([,.!?;:\)\]])", r"\1", text)
        text = re.sub(r"([\(\[\"'])\s", r"\1", text)
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
        print("Using tiktoken BPE tokenizer (cl100k_base)")
        return BPETokenizer()
    except ImportError:
        print("⚠ tiktoken not found, falling back to word-level tokenizer")
        print("  Install tiktoken for much better results: pip install tiktoken")
        tok = SimpleTokenizer(max_vocab_size=max_vocab_size)
        tok.build_vocab(texts)
        return tok


# =============================================================================
# Alpaca Templates & Data Loading
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
    """Load Alpaca dataset via HuggingFace datasets or raw JSON fallback."""
    if os.path.exists(data_path):
        print(f"Loading cached Alpaca data from {data_path}")
        with open(data_path, "r") as f:
            return json.load(f)

    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Loading Alpaca dataset via HuggingFace datasets...")
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        data = [{"instruction": ex["instruction"], "input": ex["input"],
                 "output": ex["output"]} for ex in ds]
        with open(data_path, "w") as f:
            json.dump(data, f)
        print(f"Loaded {len(data)} examples, cached to {data_path}")
        return data
    except ImportError:
        pass

    # Fallback: raw JSON from GitHub
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print(f"Downloading Alpaca dataset from GitHub...")
    urllib.request.urlretrieve(url, data_path)
    with open(data_path, "r") as f:
        data = json.load(f)
    print(f"Downloaded {len(data)} examples to {data_path}")
    return data


def format_alpaca_prompt(example: Dict) -> Tuple[str, str]:
    """Format an Alpaca example into (prompt, response) using original templates."""
    if example.get("input", "").strip():
        prompt = PROMPT_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
        )
    else:
        prompt = PROMPT_NO_INPUT.format(
            instruction=example["instruction"],
        )
    return prompt, example["output"]


# =============================================================================
# Alpaca Dataset
# =============================================================================

class AlpacaDataset(Dataset):
    """
    Alpaca instruction-following dataset with loss masking.

    Each example becomes:
        <bos> [prompt tokens] [response tokens] <eos> [padding]

    Loss mask is 0 for prompt, 1 for response+eos, 0 for padding.
    Works with both BPETokenizer and SimpleTokenizer.
    """

    def __init__(self, data: List[Dict], tokenizer, max_seq_len: int):
        self.input_ids: List[torch.Tensor] = []
        self.loss_masks: List[torch.Tensor] = []

        skipped = 0
        for example in data:
            prompt_str, response_str = format_alpaca_prompt(example)
            prompt_ids = tokenizer.encode(prompt_str)
            response_ids = tokenizer.encode(response_str)

            # Full sequence: <bos> + prompt + response + <eos>
            full_ids = [tokenizer.bos_id] + prompt_ids + response_ids + [tokenizer.eos_id]

            # Loss mask: 0 for bos+prompt, 1 for response+eos
            prompt_len = 1 + len(prompt_ids)
            mask = [0] * prompt_len + [1] * (len(response_ids) + 1)

            # Truncate
            if len(full_ids) > max_seq_len + 1:
                full_ids = full_ids[:max_seq_len + 1]
                mask = mask[:max_seq_len + 1]
                if sum(mask) == 0:
                    skipped += 1
                    continue

            # Pad
            pad_len = (max_seq_len + 1) - len(full_ids)
            full_ids += [tokenizer.pad_id] * pad_len
            mask += [0] * pad_len

            self.input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            self.loss_masks.append(torch.tensor(mask, dtype=torch.float))

        print(f"  Dataset: {len(self.input_ids)} examples loaded, {skipped} skipped (prompt too long)")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        tokens = self.input_ids[idx]
        mask = self.loss_masks[idx]
        x = tokens[:-1]    # input:  positions 0..N-1
        y = tokens[1:]     # target: positions 1..N
        m = mask[1:]       # mask aligned with target
        return x, y, m


# =============================================================================
# Training Config
# =============================================================================

@dataclass
class AlpacaTrainConfig:
    """Training config for Alpaca + HoloSpectralNet + Muon."""

    # Data
    data_path: str = "data/alpaca_data.json"
    train_split: float = 0.98

    # Tokenizer
    max_vocab_size: int = 32000  # only used for SimpleTokenizer fallback
    tokenizer_path: str = "checkpoints/alpaca_tokenizer_v2.json"

    # Model
    dim: int = 512
    depth: int = 8
    rank: int = 64
    max_seq_len: int = 512

    # Muon optimizer (hidden 2D weights: channel_mix.down.weight, channel_mix.up.weight)
    muon_lr: float = 0.02           # in spectral-norm units
    muon_momentum: float = 0.95     # Nesterov momentum
    muon_ns_steps: int = 5          # Newton-Schulz iterations

    # AdamW optimizer (embeddings, LayerNorm, 1D params, head)
    adamw_lr: float = 1e-3
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_weight_decay: float = 0.1
    adamw_emb_wd: float = 0.0       # no weight decay on embeddings

    # Training schedule
    batch_size: int = 16
    grad_accum_steps: int = 4        # effective batch = 64
    num_epochs: int = 3
    warmup_ratio: float = 0.06       # longer warmup helps Muon
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 50

    # Generation
    gen_max_tokens: int = 100
    gen_temperature: float = 0.8
    gen_top_k: int = 50
    gen_repetition_penalty: float = 1.2

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints/alpaca_holospectral_muon_v2.pt"


# =============================================================================
# Training Utilities
# =============================================================================

def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy only on response tokens (where mask=1)."""
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, V), targets.view(B * T), reduction="none"
    )
    loss = loss * mask.view(B * T)
    return loss.sum() / mask.sum().clamp(min=1)


def get_lr_scale(step: int, total_steps: int, warmup_ratio: float) -> float:
    """
    Cosine schedule with linear warmup. Returns multiplier in [0.1, 1.0].
    Applied to both Muon and AdamW learning rates.
    Floor of 0.1 prevents training collapse at end of schedule.
    """
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * decay_ratio)))


@torch.no_grad()
def estimate_loss(model: nn.Module, loader: DataLoader,
                  eval_iters: int, device: str) -> float:
    """Estimate masked loss on a data loader."""
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(eval_iters):
        try:
            x, y, m = next(it)
        except StopIteration:
            it = iter(loader)
            x, y, m = next(it)
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x)
        losses.append(masked_cross_entropy(logits, y, m).item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


@torch.no_grad()
def generate_response(
    model: nn.Module,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    max_seq_len: int = 512,
    device: str = "cpu",
) -> str:
    """
    Generate a response with repetition penalty and special-token masking.

    Special tokens (pad, bos) are masked to -inf so they can never be sampled.
    EOS is kept unmasked so the model can learn to stop.
    Repetition penalty divides logits of already-seen tokens.
    """
    model.eval()

    # Build prompt
    example = {"instruction": instruction, "input": input_text, "output": ""}
    prompt_str, _ = format_alpaca_prompt(example)
    prefix_ids = [tokenizer.bos_id] + tokenizer.encode(prompt_str)
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    # Compute which IDs to ban from sampling (special tokens except eos)
    special_ids = set()
    if hasattr(tokenizer, "SPECIAL_TOKENS"):
        if isinstance(tokenizer.SPECIAL_TOKENS, dict):
            special_ids = set(tokenizer.SPECIAL_TOKENS.values())
        elif isinstance(tokenizer.SPECIAL_TOKENS, list):
            if hasattr(tokenizer, "stoi"):
                special_ids = {tokenizer.stoi[t] for t in tokenizer.SPECIAL_TOKENS
                               if t in tokenizer.stoi}
    # Keep eos_id available — model needs it to stop generation
    special_ids.discard(tokenizer.eos_id)
    ban_tensor = torch.tensor(list(special_ids), device=device) if special_ids else None

    generated_ids: List[int] = []

    for _ in range(max_new_tokens):
        # Crop context to max_seq_len
        idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]

        logits = model(idx_cond)[:, -1, :]  # (1, vocab_size)

        # Ban special tokens from being sampled
        if ban_tensor is not None:
            logits[0, ban_tensor] = float("-inf")

        # Ban any ID beyond vocab_size (safety net)
        vocab_sz = tokenizer.vocab_size
        if logits.size(-1) > vocab_sz:
            logits[0, vocab_sz:] = float("-inf")

        # Repetition penalty: penalize tokens already generated
        if repetition_penalty != 1.0 and generated_ids:
            seen = torch.tensor(list(set(generated_ids)), device=device)
            seen_logits = logits[0, seen]
            logits[0, seen] = torch.where(
                seen_logits > 0,
                seen_logits / repetition_penalty,
                seen_logits * repetition_penalty,
            )

        # Temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Stop on EOS
        if idx_next.item() == tokenizer.eos_id:
            break

        generated_ids.append(idx_next.item())
        idx = torch.cat([idx, idx_next], dim=1)

    return tokenizer.decode(generated_ids)


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: AlpacaTrainConfig):
    """Main training function: Alpaca + HoloSpectralNet + Muon/AdamW."""

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    print("=" * 70)
    print("  HoloSpectralNet × Stanford Alpaca × Muon Optimizer (v2)")
    print("=" * 70)
    print(f"  Device: {config.device}")

    # ── 1. Load Alpaca Data ──────────────────────────────────────────────
    raw_data = load_alpaca_data(config.data_path)
    print(f"  Alpaca examples: {len(raw_data)}")

    # ── 2. Build Tokenizer ───────────────────────────────────────────────
    print("\nBuilding tokenizer...")
    all_texts = []
    for ex in raw_data:
        p, r = format_alpaca_prompt(ex)
        all_texts.extend([p, r])
    tokenizer = create_tokenizer(all_texts, config.max_vocab_size)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── 3. Train/Val Split & Datasets ────────────────────────────────────
    print("\nBuilding datasets...")
    split_idx = int(len(raw_data) * config.train_split)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    train_dataset = AlpacaDataset(train_data, tokenizer, config.max_seq_len)
    val_dataset = AlpacaDataset(val_data, tokenizer, config.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ── 4. Initialize Model ─────────────────────���────────────────────────
    print("\nInitializing model...")
    model = HoloSpectralNet(
        vocab_size=tokenizer.vocab_size,
        dim=config.dim,
        depth=config.depth,
        max_seq_len=config.max_seq_len,
        rank=config.rank,
        tie_weights=True,
    ).to(config.device)

    print(f"  HoloSpectralNet: dim={config.dim}, depth={config.depth}, rank={config.rank}")
    print(f"  Parameters: {model.count_parameters():,}")

    # ── 5. Split Optimizers: Muon + AdamW ────────────────────────────────
    print("\nSetting up optimizers...")
    muon_params, adamw_params = split_params_for_muon(model)

    # Separate embedding params for zero weight decay
    emb_param_ids = set()
    emb_params_list = []
    for name, param in model.named_parameters():
        if ("token_emb" in name or "head" in name) and param.requires_grad:
            emb_param_ids.add(id(param))
            emb_params_list.append(param)

    other_adamw_params = [p for p in adamw_params if id(p) not in emb_param_ids]

    opt_muon = Muon(
        muon_params,
        lr=config.muon_lr,
        momentum=config.muon_momentum,
        ns_steps=config.muon_ns_steps,
    )
    opt_adamw = torch.optim.AdamW(
        [
            {"params": emb_params_list, "weight_decay": config.adamw_emb_wd},
            {"params": other_adamw_params, "weight_decay": config.adamw_weight_decay},
        ],
        lr=config.adamw_lr,
        betas=config.adamw_betas,
    )

    print(f"\n  Muon  LR: {config.muon_lr} (momentum={config.muon_momentum}, ns_steps={config.muon_ns_steps})")
    print(f"  AdamW LR: {config.adamw_lr} (betas={config.adamw_betas})")
    print(f"  Effective batch size: {config.batch_size} × {config.grad_accum_steps} = {config.batch_size * config.grad_accum_steps}")

    # ── 6. Training Loop ─────────────────────────────────────────────────
    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.num_epochs
    global_step = 0
    best_val_loss = float("inf")
    accum_loss = 0.0

    print(f"\n{'='*70}")
    print(f"  Training: {config.num_epochs} epochs, {total_steps} optimizer steps")
    print(f"  ({len(train_loader) * config.num_epochs} micro-batches total)")
    print(f"{'='*70}\n")

    model.train()
    for epoch in range(config.num_epochs):
        for micro_step, (x, y, m) in enumerate(train_loader):

            # ── LR schedule (both optimizers) ────────────────────────
            scale = get_lr_scale(global_step, total_steps, config.warmup_ratio)
            for pg in opt_muon.param_groups:
                pg["lr"] = config.muon_lr * scale
            for pg in opt_adamw.param_groups:
                pg["lr"] = config.adamw_lr * scale

            # ── Forward + loss ───────────────────────────────────────
            x, y, m = x.to(config.device), y.to(config.device), m.to(config.device)

            logits = model(x)
            loss = masked_cross_entropy(logits, y, m) / config.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            # ── Accumulate gradients ─────────────────────────────────
            if (micro_step + 1) % config.grad_accum_steps != 0:
                continue

            # ── Gradient clipping + optimizer step ───────────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt_muon.step()
            opt_adamw.step()
            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)

            global_step += 1

            # ── Logging ──────────────────────────────────────────────
            if global_step % config.log_interval == 0:
                muon_lr = opt_muon.param_groups[0]["lr"]
                adamw_lr = opt_adamw.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch+1}/{config.num_epochs} | "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {accum_loss:.4f} | "
                    f"Muon LR: {muon_lr:.4e} | "
                    f"AdamW LR: {adamw_lr:.4e}"
                )
                accum_loss = 0.0

            # ── Evaluation ───────────────────────────────────────────
            if global_step % config.eval_interval == 0:
                val_loss = estimate_loss(
                    model, val_loader, config.eval_iters, config.device
                )
                print(f"\n{'─'*70}")
                print(f"  EVAL @ step {global_step} | Val Loss: {val_loss:.4f}")

                # Sample generations
                test_instructions = [
                    ("Give three tips for staying healthy.", ""),
                    (
                        "Rewrite the following sentence in the third person.",
                        "I am going to the store.",
                    ),
                    ("What is the capital of France?", ""),
                ]

                for instr, inp in test_instructions:
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

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
                    tokenizer.save(config.tokenizer_path)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "opt_muon_state": opt_muon.state_dict(),
                            "opt_adamw_state": opt_adamw.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "best_val_loss": best_val_loss,
                            "config": config,
                        },
                        config.checkpoint_path,
                    )
                    print(f"  ✅ New best model saved (val_loss={best_val_loss:.4f})")

                model.train()

    # ── Done ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {config.checkpoint_path}")
    print(f"  Tokenizer:  {config.tokenizer_path}")
    print(f"{'='*70}")

    return model, tokenizer


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    config = AlpacaTrainConfig()
    model, tokenizer = train(config)