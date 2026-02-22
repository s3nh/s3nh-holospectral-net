"""
HoloSpectral Knowledge Distillation from 7B LLM Teacher
Cross-GPU teacher-student training with hidden state and logits distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import copy

# Import your HoloSpectralNet
try:
    from holospectral import HoloSpectralNet, hsn_small, hsn_base
except ImportError:
    # Fallback implementation if not installed
    import sys
    sys.path.append('.')
    from model import HoloSpectralNet, hsn_small, hsn_base


@dataclass
class DistillationConfig:
    """Configuration for teacher-student distillation"""
    # Temperature for soft targets
    temperature: float = 2.0
    
    # Loss weights
    alpha_logits: float = 0.5      # Weight for KL divergence on logits
    alpha_hidden: float = 0.3      # Weight for hidden state MSE
    alpha_hard: float = 0.2        # Weight for ground truth CE loss
    
    # Hidden state matching
    hidden_matching: str = "linear"  # "linear", "mlp", or "none"
    teacher_dim: int = 4096          # Teacher hidden dimension (e.g., Llama-7B)
    student_dim: int = 512           # Your HSN dimension
    
    # Training
    max_seq_len: int = 512
    gradient_accumulation_steps: int = 4
    
    # Teacher on different GPU
    teacher_gpu_id: int = 1          # GPU where 7B teacher resides
    student_gpu_id: int = 0          # GPU where HSN student trains


class HiddenStateProjector(nn.Module):
    """
    Project teacher hidden states to student dimension for distillation.
    Maps from large teacher dim (4096) to small student dim (512).
    """
    def __init__(
        self, 
        teacher_dim: int, 
        student_dim: int,
        projection_type: str = "linear"
    ):
        super().__init__()
        self.projection_type = projection_type
        
        if projection_type == "linear":
            # Simple linear projection
            self.projector = nn.Linear(teacher_dim, student_dim, bias=False)
        elif projection_type == "mlp":
            # Two-layer MLP with bottleneck
            intermediate_dim = (teacher_dim + student_dim) // 2
            self.projector = nn.Sequential(
                nn.Linear(teacher_dim, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, student_dim)
            )
        elif projection_type == "none":
            self.projector = nn.Identity()
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
    
    def forward(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            teacher_hidden: (batch, seq_len, teacher_dim)
        Returns:
            projected: (batch, seq_len, student_dim)
        """
        return self.projector(teacher_hidden)


class HoloSpectralWithHiddenStates(HoloSpectralNet):
    """
    Extended HoloSpectralNet that returns intermediate hidden states
    for distillation purposes.
    """
    def __init__(self, *args, return_hidden_states: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_hidden_states = return_hidden_states
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_hidden_states: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass with optional hidden state extraction.
        
        Returns:
            If return_hidden_states=False: logits (batch, seq_len, vocab_size)
            If return_hidden_states=True: (logits, hidden_states_list)
        """
        return_hs = return_hidden_states if return_hidden_states is not None else self.return_hidden_states
        
        batch_size, seq_len = x.shape
        
        # Embedding
        x = self.token_embedding(x)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        hidden_states = []
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x)
            if return_hs:
                # Store post-block hidden states (before final norm)
                hidden_states.append(x.clone())
        
        # Final norm and head
        x = self.norm(x)
        logits = self.head(x)
        
        if return_hs:
            return logits, hidden_states
        return logits


class TeacherStudentDistiller:
    """
    Manages teacher-student distillation with cross-GPU communication.
    Teacher (7B LLM) on GPU 1, Student (HSN) on GPU 0.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,  # Your 7B LLM (e.g., Llama-7B)
        student_model: HoloSpectralNet,
        config: DistillationConfig,
        teacher_tokenizer=None,    # Teacher tokenizer
        student_tokenizer=None,    # Student tokenizer (might be same)
    ):
        self.config = config
        self.teacher = teacher_model
        self.student = student_model
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        
        # Move models to respective GPUs
        self.device_teacher = torch.device(f"cuda:{config.teacher_gpu_id}")
        self.device_student = torch.device(f"cuda:{config.student_gpu_id}")
        
        self.teacher.to(self.device_teacher)
        self.student.to(self.device_student)
        
        # Set teacher to eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Hidden state projector (on student GPU)
        if config.hidden_matching != "none":
            self.hidden_projector = HiddenStateProjector(
                config.teacher_dim,
                config.student_dim,
                config.hidden_matching
            ).to(self.device_student)
        else:
            self.hidden_projector = None
        
        # Temperature for soft targets
        self.temperature = config.temperature
        
    def get_teacher_outputs(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get teacher predictions and hidden states.
        Runs on teacher GPU, returns dict with logits and hidden states.
        """
        with torch.no_grad():
            input_ids = input_ids.to(self.device_teacher)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device_teacher)
            
            # Assuming teacher is a HuggingFace model
            outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            teacher_logits = outputs.logits  # (batch, seq_len, vocab_size_teacher)
            teacher_hidden = outputs.hidden_states[-1]  # Last layer hidden states
            
            # Move outputs to CPU then to student GPU to avoid GPU0-GPU1 direct transfer issues
            return {
                'logits': teacher_logits.cpu(),
                'hidden_states': teacher_hidden.cpu(),
                'attention_mask': attention_mask.cpu() if attention_mask is not None else None
            }
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_hidden: List[torch.Tensor],
        teacher_hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss:
        - KL divergence on softened distributions
        - MSE on hidden states
        - Optional CE on hard targets
        """
        batch_size, seq_len, student_vocab = student_logits.shape
        _, _, teacher_vocab = teacher_logits.shape
        
        # Move teacher outputs to student device
        teacher_logits = teacher_logits.to(student_logits.device)
        teacher_hidden = teacher_hidden.to(student_logits.device)
        
        # Create mask for valid positions (ignore padding)
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=student_logits.device, dtype=torch.bool)
        else:
            mask = mask.to(student_logits.device)
        
        # 1. Logits Distillation (KL Divergence)
        # Match vocabulary sizes - project teacher to student vocab or vice versa
        if teacher_vocab != student_vocab:
            # Simple approach: use teacher's predictions on student vocab
            # Better: learn a projection matrix (not implemented here for simplicity)
            teacher_logits = teacher_logits[:, :, :student_vocab]
        
        # Soften distributions
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence (teacher || student)
        kl_loss = F.kl_div(
            student_soft.view(-1, student_vocab),
            teacher_soft.view(-1, student_vocab),
            reduction='none'
        ).sum(dim=-1)
        
        # Apply mask and average
        kl_loss = (kl_loss * mask.view(-1)).sum() / mask.sum()
        kl_loss = kl_loss * (self.temperature ** 2)  # Scale by T^2
        
        # 2. Hidden State Distillation
        hidden_loss = torch.tensor(0.0, device=student_logits.device)
        if self.hidden_projector is not None and len(student_hidden) > 0:
            # Project teacher hidden to student dimension
            teacher_hidden_proj = self.hidden_projector(teacher_hidden)
            
            # Match last student layer with projected teacher
            student_last_hidden = student_hidden[-1]  # (batch, seq_len, student_dim)
            
            # MSE loss
            hidden_diff = (student_last_hidden - teacher_hidden_proj) ** 2
            hidden_loss = (hidden_diff * mask.unsqueeze(-1)).sum() / mask.sum()
        
        # 3. Hard Target Loss (if labels provided)
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None:
            labels = labels.to(student_logits.device)
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_vocab),
                labels.view(-1),
                reduction='none'
            )
            hard_loss = (hard_loss * mask.view(-1)).sum() / mask.sum()
        
        # Combined loss
        total_loss = (
            self.config.alpha_logits * kl_loss +
            self.config.alpha_hidden * hidden_loss +
            self.config.alpha_hard * hard_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'kl': kl_loss.item(),
            'hidden': hidden_loss.item(),
            'hard': hard_loss.item() if labels is not None else 0.0
        }
        
        return total_loss, loss_dict
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Single training step with teacher-student distillation.
        """
        input_ids = batch['input_ids']
        labels = batch.get('labels')
        attention_mask = batch.get('attention_mask')
        
        # 1. Get teacher outputs (on teacher GPU)
        teacher_outputs = self.get_teacher_outputs(input_ids, attention_mask)
        
        # 2. Student forward pass (on student GPU)
        student_input_ids = input_ids.to(self.device_student)
        student_labels = labels.to(self.device_student) if labels is not None else None
        student_mask = attention_mask.to(self.device_student) if attention_mask is not None else None
        
        # Enable hidden state return for distillation
        student_logits, student_hidden = self.student(
            student_input_ids, 
            return_hidden_states=True
        )
        
        # 3. Compute loss
        loss, loss_dict = self.compute_distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_outputs['logits'],
            student_hidden=student_hidden,
            teacher_hidden=teacher_outputs['hidden_states'],
            labels=student_labels,
            mask=student_mask
        )
        
        # 4. Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        if self.hidden_projector is not None:
            torch.nn.utils.clip_grad_norm_(self.hidden_projector.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        
        return loss_dict


class DataCollatorForDistillation:
    """
    Collates batches for distillation with proper padding and masking.
    Handles tokenization alignment between teacher and student if needed.
    """
    
    def __init__(
        self,
        teacher_tokenizer,
        student_tokenizer=None,
        max_length: int = 512,
        pad_to_multiple_of: int = 8
    ):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer or teacher_tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of examples into a batch.
        Each example should have 'text' or 'input_ids' field.
        """
        # Tokenize texts
        texts = [ex['text'] for ex in examples]
        
        # Teacher tokenization
        teacher_encoded = self.teacher_tokenizer(
            texts,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        # For causal LM, labels are same as input_ids (shifted inside model)
        # But for distillation, we want next-token prediction
        input_ids = teacher_encoded['input_ids']
        attention_mask = teacher_encoded['attention_mask']
        
        # Create labels by shifting input_ids
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_student_from_teacher(
    teacher_config,
    target_params: str = "1.5M",  # "400K", "1.5M", "8M"
    vocab_size: Optional[int] = None
) -> HoloSpectralNet:
    """
    Create a HoloSpectralNet student with appropriate size.
    """
    vocab = vocab_size or teacher_config.vocab_size
    
    if target_params == "400K":
        student = hsn_tiny(vocab_size=vocab)
    elif target_params == "1.5M":
        student = hsn_small(vocab_size=vocab)
    elif target_params == "8M":
        student = hsn_base(vocab_size=vocab)
    else:
        # Custom config
        student = HoloSpectralNet(
            vocab_size=vocab,
            dim=512,
            depth=8,
            max_seq_len=teacher_config.max_position_embeddings,
            rank=64
        )
    
    return student


def load_teacher_model(model_name: str = "meta-llama/Llama-2-7b-hf"):
    """
    Load 7B teacher model with optimizations for inference.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 1},  # Force to GPU 1
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    return model, tokenizer


def main():
    """
    Example training setup.
    """
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
    # Configuration
    config = DistillationConfig(
        temperature=2.0,
        alpha_logits=0.5,
        alpha_hidden=0.3,
        alpha_hard=0.2,
        teacher_dim=4096,      # Llama-7B hidden size
        student_dim=512,       # HSN base dimension
        teacher_gpu_id=1,
        student_gpu_id=0
    )
    
    # Load teacher (7B LLM) on GPU 1
    print("Loading teacher model on GPU 1...")
    teacher, teacher_tokenizer = load_teacher_model("meta-llama/Llama-2-7b-hf")
    
    # Create student (HoloSpectralNet) on GPU 0
    print("Creating student model on GPU 0...")
    student = HoloSpectralWithHiddenStates(
        vocab_size=teacher_tokenizer.vocab_size,
        dim=config.student_dim,
        depth=8,
        max_seq_len=512,
        rank=64,
        return_hidden_states=True
    )
    
    print(f"Student parameters: {student.count_parameters():,}")
    
    # Setup distiller
    distiller = TeacherStudentDistiller(
        teacher_model=teacher,
        student_model=student,
        config=config,
        teacher_tokenizer=teacher_tokenizer
    )
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Create dataloader
    collator = DataCollatorForDistillation(
        teacher_tokenizer=teacher_tokenizer,
        max_length=512
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + 
        list(distiller.hidden_projector.parameters() if distiller.hidden_projector else []),
        lr=3e-4,
        weight_decay=0.01
    )
    
    # Training loop
    student.train()
    for epoch in range(3):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            loss_dict = distiller.training_step(batch, optimizer)
            total_loss += loss_dict['total']
            
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}: {loss_dict}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'student_state_dict': student.state_dict(),
            'projector_state_dict': distiller.hidden_projector.state_dict() if distiller.hidden_projector else None,
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"hsn_distilled_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()