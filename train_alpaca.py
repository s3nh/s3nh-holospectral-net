"""
Stanford Alpaca training script for HoloSpectralNet.

Downloads the 52K Alpaca instruction-following dataset from tatsu-lab/stanford_alpaca
and trains HoloSpectralNet as an instruction-following model.

Alpaca data format (per entry):
  - instruction: str  — the task description
  - input: str        — optional context (empty string if none)
  - output: str       — the target response

The prompt templates follow the original Alpaca format exactly:
  WITH input:    "Below is an instruction ... ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
  WITHOUT input: "Below is an instruction ... ### Instruction:\n{instruction}\n\n### Response:\n"

Loss is masked so the model only learns to predict the ### Response portion.

Usage:
    pip install datasets
    python train_alpaca.py
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
# Alpaca Prompt Templates (verbatim from tatsu-lab/stanford_alpaca)
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


# =============================================================================
# Word-Level Tokenizer with Special Tokens
# =============================================================================

class AlpacaTokenizer:
    """
    Word-level tokenizer built from the Alpaca corpus.

    Why word-level instead of char-level:
    - Your current CharDataset in train_shakespeare.py uses ~65 char tokens.
      That forces the model to learn spelling before semantics.
    - Word-level tokens carry meaning from token #1. With 52K Alpaca examples,
      you get a rich vocab (~16K–32K unique words) which is much closer to
      how real LLMs operate.

    Special tokens give the model structural awareness:
    - <|pad|>  — padding (ignored in loss)
    - <|unk|>  — out-of-vocabulary fallback
    - <|bos|>  — beginning of sequence
    - <|eos|>  — end of sequence (model learns when to STOP)
    """

    SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]

    def __init__(self, max_vocab_size: int = 32000):
        self.max_vocab_size = max_vocab_size
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._init_special()

    def _init_special(self):
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.stoi[tok] = i
            self.itos[i] = tok

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Split into words + punctuation. Lowercase for consistency."""
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from all Alpaca text fields."""
        freq: Dict[str, int] = {}
        for text in texts:
            for word in self._tokenize(text):
                freq[word] = freq.get(word, 0) + 1

        budget = self.max_vocab_size - len(self.SPECIAL_TOKENS)
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])[:budget]

        idx = len(self.SPECIAL_TOKENS)
        for word, _ in sorted_words:
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"Tokenizer: {len(self.stoi)} tokens "
              f"({len(self.SPECIAL_TOKENS)} special + {len(self.stoi) - len(self.SPECIAL_TOKENS)} words)")

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
        """Encode text → token IDs."""
        return [self.stoi.get(w, self.unk_id) for w in self._tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs → text."""
        tokens = []
        for i in ids:
            tok = self.itos.get(i, "<|unk|>")
            if tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        text = " ".join(tokens)
        # Clean up punctuation spacing
        text = re.sub(r"\s([,.!?;:\)\]])", r"\1", text)
        text = re.sub(r"([\(\[\"'])\s", r"\1", text)
        return text

    def save(self, path: str):
        """Save tokenizer vocab to JSON."""
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}, f)

    @classmethod
    def load(cls, path: str) -> "AlpacaTokenizer":
        """Load tokenizer vocab from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        return tok


# =============================================================================
# Alpaca Dataset
# =============================================================================

def load_alpaca_data(data_path: str) -> List[Dict]:
    """
    Load the Alpaca dataset. Tries HuggingFace `datasets` first (recommended),
    falls back to downloading the raw JSON from GitHub.
    """
    if os.path.exists(data_path):
        print(f"Loading cached Alpaca data from {data_path}")
        with open(data_path, "r") as f:
            return json.load(f)

    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)

    # Try HuggingFace datasets first
    try:
        from datasets import load_dataset
        print("Loading Alpaca dataset via HuggingFace datasets...")
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        data = [{"instruction": ex["instruction"], "input": ex["input"], "output": ex["output"]} for ex in ds]
        with open(data_path, "w") as f:
            json.dump(data, f)
        print(f"Loaded {len(data)} examples, cached to {data_path}")
        return data
    except ImportError:
        pass

    # Fallback: download raw JSON from GitHub
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print(f"Downloading Alpaca dataset from GitHub...")
    urllib.request.urlretrieve(url, data_path)
    with open(data_path, "r") as f:
        data = json.load(f)
    print(f"Downloaded {len(data)} examples to {data_path}")
    return data


def format_alpaca_prompt(example: Dict) -> Tuple[str, str]:
    """
    Format a single Alpaca example into (prompt, response) using the
    exact templates from tatsu-lab/stanford_alpaca README.

    Returns:
        prompt: The instruction/input portion (model should NOT be trained on this)
        response: The output portion (model IS trained on this)
    """
    if example.get("input", "").strip():
        prompt = PROMPT_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
        )
    else:
        prompt = PROMPT_NO_INPUT.format(
            instruction=example["instruction"],
        )
    response = example["output"]
    return prompt, response


class AlpacaDataset(Dataset):
    """
    Dataset for Alpaca instruction-following training.

    Key design decisions vs your current CharDataset:

    1. Each example is a COMPLETE instruction+response pair (not a sliding window)
    2. Loss masking: the prompt portion has mask=0, only the response has mask=1.
       This is exactly what tatsu-lab/stanford_alpaca does with IGNORE_INDEX=-100.
    3. Sequences are: <|bos|> [prompt tokens] [response tokens] <|eos|> [padding]
    4. This means the model learns to ANSWER instructions, not parrot them back.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: AlpacaTokenizer,
        max_seq_len: int,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.input_ids: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.loss_masks: List[torch.Tensor] = []

        skipped = 0
        for example in data:
            prompt_str, response_str = format_alpaca_prompt(example)

            prompt_ids = tokenizer.encode(prompt_str)
            response_ids = tokenizer.encode(response_str)

            # Full sequence: <bos> + prompt + response + <eos>
            full_ids = (
                [tokenizer.bos_id]
                + prompt_ids
                + response_ids
                + [tokenizer.eos_id]
            )

            # Loss mask: 0 for bos+prompt, 1 for response+eos
            prompt_len = 1 + len(prompt_ids)  # bos + prompt
            mask = [0] * prompt_len + [1] * (len(response_ids) + 1)  # response + eos

            # Truncate if too long (keep max_seq_len + 1 for input/target shift)
            if len(full_ids) > max_seq_len + 1:
                full_ids = full_ids[: max_seq_len + 1]
                mask = mask[: max_seq_len + 1]
                # If truncation killed all response tokens, skip
                if sum(mask) == 0:
                    skipped += 1
                    continue

            # Pad to max_seq_len + 1
            pad_len = (max_seq_len + 1) - len(full_ids)
            full_ids = full_ids + [tokenizer.pad_id] * pad_len
            mask = mask + [0] * pad_len

            self.input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            self.loss_masks.append(torch.tensor(mask, dtype=torch.float))

        print(f"AlpacaDataset: {len(self.input_ids)} examples loaded, {skipped} skipped (prompt too long)")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        tokens = self.input_ids[idx]
        mask = self.loss_masks[idx]
        x = tokens[:-1]   # input:  positions 0..N-1
        y = tokens[1:]     # target: positions 1..N (shifted by 1)
        m = mask[1:]       # mask aligned with target
        return x, y, m


# =============================================================================
# Training Config
# =============================================================================

@dataclass
class AlpacaTrainConfig:
    """Training configuration strictly for Alpaca + HoloSpectralNet."""

    # Data
    data_path: str = "data/alpaca_data.json"
    train_split: float = 0.98  # 52K examples, keep ~1K for validation

    # Tokenizer
    max_vocab_size: int = 32000
    tokenizer_path: str = "checkpoints/alpaca_tokenizer.json"

    # Model (matches your existing HoloSpectralNet API)
    dim: int = 384
    depth: int = 8
    rank: int = 96
    max_seq_len: int = 512  # Alpaca used 512 for LLaMA fine-tuning

    # Training (following Alpaca's own hyperparameters where applicable)
    batch_size: int = 32
    learning_rate: float = 2e-5  # Same as Alpaca LLaMA-7B
    weight_decay: float = 0.0    # Same as Alpaca (no weight decay)
    num_epochs: int = 3          # Same as Alpaca LLaMA-7B
    warmup_ratio: float = 0.03   # Same as Alpaca
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    eval_interval: int = 1000
    eval_iters: int = 50
    log_interval: int = 100

    # Generation
    gen_max_tokens: int = 100
    gen_temperature: float = 0.7
    gen_top_k: int = 50

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints/alpaca_holospectral.pt"


# =============================================================================
# Training Utilities
# =============================================================================

def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy computed ONLY on response tokens.

    This replaces F.cross_entropy(logits.view(-1, V), y.view(-1)) from
    your train_shakespeare.py. The key difference: loss on prompt tokens is
    zeroed out, so the model is only rewarded for generating good responses.

    This is equivalent to what tatsu-lab/stanford_alpaca does by setting
    label[:source_len] = IGNORE_INDEX (-100) in their preprocess() function.
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), reduction="none")
    loss = loss * mask.view(B * T)
    return loss.sum() / mask.sum().clamp(min=1)


def get_lr(step: int, total_steps: int, config: AlpacaTrainConfig) -> float:
    """
    Cosine schedule with linear warmup.
    Matches Alpaca's lr_scheduler_type="cosine" + warmup_ratio=0.03.
    """
    warmup_steps = int(total_steps * config.warmup_ratio)
    if step < warmup_steps:
        return config.learning_rate * step / max(warmup_steps, 1)
    if step >= total_steps:
        return config.min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    loader: DataLoader,
    eval_iters: int,
    device: str,
) -> float:
    """Estimate masked loss on a loader."""
    model.eval()
    losses = []
    loader_iter = iter(loader)
    for _ in range(eval_iters):
        try:
            x, y, m = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y, m = next(loader_iter)
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x)
        loss = masked_cross_entropy(logits, y, m)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


@torch.no_grad()
def generate_response(
    model: nn.Module,
    tokenizer: AlpacaTokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cpu",
) -> str:
    """
    Generate a response to an Alpaca-formatted instruction.

    Constructs the prompt using the exact Alpaca template, encodes it,
    then autoregressively samples until <|eos|> or max_new_tokens.
    """
    model.eval()

    # Build prompt using Alpaca template
    example = {"instruction": instruction, "input": input_text}
    prompt_str, _ = format_alpaca_prompt({**example, "output": ""})

    prefix_ids = [tokenizer.bos_id] + tokenizer.encode(prompt_str)
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop to max_seq_len
        idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() == tokenizer.eos_id:
            break

        idx = torch.cat([idx, idx_next], dim=1)

    generated_ids = idx[0, len(prefix_ids):].tolist()
    return tokenizer.decode(generated_ids)


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: AlpacaTrainConfig):
    """Main training function for Alpaca + HoloSpectralNet."""

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    print("=" * 70)
    print("  HoloSpectralNet × Stanford Alpaca Trainer")
    print("=" * 70)
    print(f"Device: {config.device}")

    # ── 1. Load Alpaca Data ──────────────────────────────────────────────
    raw_data = load_alpaca_data(config.data_path)
    print(f"Total Alpaca examples: {len(raw_data)}")

    # ── 2. Build Tokenizer from ALL text in the dataset ──────────────────
    all_texts = []
    for ex in raw_data:
        prompt_str, response_str = format_alpaca_prompt(ex)
        all_texts.append(prompt_str)
        all_texts.append(response_str)

    tokenizer = AlpacaTokenizer(max_vocab_size=config.max_vocab_size)
    tokenizer.build_vocab(all_texts)

    # ── 3. Split into train/val ──────────────────────────────────────────
    split_idx = int(len(raw_data) * config.train_split)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── 4. Create Datasets & Loaders ─────────────────────────────────────
    train_dataset = AlpacaDataset(train_data, tokenizer, config.max_seq_len)
    val_dataset = AlpacaDataset(val_data, tokenizer, config.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── 5. Initialize Model ──────────────────────────────────────────────
    model = HoloSpectralNet(
        vocab_size=tokenizer.vocab_size,
        dim=config.dim,
        depth=config.depth,
        max_seq_len=config.max_seq_len,
        rank=config.rank,
        tie_weights=True,
    ).to(config.device)

    num_params = model.count_parameters()
    print(f"\nModel: HoloSpectralNet")
    print(f"  dim={config.dim}, depth={config.depth}, rank={config.rank}")
    print(f"  max_seq_len={config.max_seq_len}, vocab_size={tokenizer.vocab_size}")
    print(f"  Parameters: {num_params:,}")
    print("=" * 70)

    # ── 6. Optimizer (matching Alpaca: AdamW, no weight decay) ───────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── 7. Training Loop ─────────────────────────────────────────────────
    total_steps = len(train_loader) * config.num_epochs
    global_step = 0
    best_val_loss = float("inf")

    print(f"Training for {config.num_epochs} epochs ({total_steps} steps)")
    print()

    model.train()
    for epoch in range(config.num_epochs):
        for batch_idx, (x, y, m) in enumerate(train_loader):
            # LR schedule
            lr = get_lr(global_step, total_steps, config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y, m = x.to(config.device), y.to(config.device), m.to(config.device)

            # Forward
            logits = model(x)
            loss = masked_cross_entropy(logits, y, m)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            global_step += 1

            # ── Logging ──────────────────────────────────────────────
            if global_step % config.log_interval == 0:
                print(f"  Epoch {epoch+1}/{config.num_epochs} | "
                      f"Step {global_step}/{total_steps} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e}")

            # ── Evaluation ───────────────────────────────────────────
            if global_step % config.eval_interval == 0:
                val_loss = estimate_loss(model, val_loader, config.eval_iters, config.device)
                print(f"\n{'─'*70}")
                print(f"  EVAL @ step {global_step} | Val Loss: {val_loss:.4f}")

                # Sample generations
                test_instructions = [
                    ("Give three tips for staying healthy.", ""),
                    ("Rewrite the following sentence in the third person.",
                     "I am going to the store."),
                    ("What is the capital of France?", ""),
                ]
                for instr, inp in test_instructions:
                    response = generate_response(
                        model, tokenizer, instr, inp,
                        max_new_tokens=config.gen_max_tokens,
                        temperature=config.gen_temperature,
                        top_k=config.gen_top_k,
                        device=config.device,
                    )
                    print(f"\n  ### Instruction: {instr}")
                    if inp:
                        print(f"  ### Input: {inp}")
                    print(f"  ### Response: {response}")

                print(f"{'─'*70}\n")

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
                    tokenizer.save(config.tokenizer_path)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "config": config,
                        "tokenizer_path": config.tokenizer_path,
                    }, config.checkpoint_path)
                    print(f"  �� New best model saved (val_loss={best_val_loss:.4f})")

                model.train()

    print("\n" + "=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Tokenizer:  {config.tokenizer_path}")
    print("=" * 70)

    return model, tokenizer


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    config = AlpacaTrainConfig()
    model, tokenizer = train(config)