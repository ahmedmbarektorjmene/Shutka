"""
RWKV-X: RNN with sparse attention mechanism
Combines time-mixing and channel-mixing with sparse attention for long-context modeling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return self.weight * x / (norm + self.eps)


class SparseAttention(nn.Module):
    """
    Sparse attention mechanism for RWKV-X
    Uses top-k selection to maintain linear complexity
    """
    def __init__(self, d_model: int, attn_size: int = 64, topk: int = 32):
        super().__init__()
        self.d_model = d_model
        self.attn_size = attn_size
        self.topk = topk
        
        self.q_proj = nn.Linear(d_model, attn_size)
        self.k_proj = nn.Linear(d_model, attn_size)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, attn_size)
        K = self.k_proj(x)  # (batch, seq_len, attn_size)
        V = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.attn_size)  # (batch, seq_len, seq_len)
        
        # Sparse attention: select top-k keys for each query
        if self.topk < seq_len:
            topk_scores, topk_indices = torch.topk(scores, k=self.topk, dim=-1)
            # Create sparse attention mask
            sparse_scores = torch.full_like(scores, float('-inf'))
            sparse_scores.scatter_(-1, topk_indices, topk_scores)
            attn_weights = F.softmax(sparse_scores, dim=-1)
        else:
            attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_output = torch.bmm(attn_weights, V)  # (batch, seq_len, d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class TimeMixBlock(nn.Module):
    """
    Time-mixing block: processes sequential dependencies through recurrence
    """
    def __init__(self, d_model: int, attn_size: int = 64, topk: int = 32):
        super().__init__()
        self.d_model = d_model
        
        # Time-mixing parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
        # Projections
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        # Sparse attention
        self.sparse_attn = SparseAttention(d_model, attn_size, topk)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Time-mixing: mix current and previous states
        xx = torch.cat([x[:, 0:1, :], x[:, :-1, :]], dim=1)  # Shift right
        
        # Mix with learnable parameters
        k = self.key(x * self.time_mix_k + xx * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + xx * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + xx * (1 - self.time_mix_r))
        
        # Apply sparse attention
        attn_out = self.sparse_attn(x)
        
        # Combine with time-mixed values
        rkv = torch.sigmoid(r) * (attn_out + v)
        
        # Output
        output = self.output(rkv)
        
        return output


class ChannelMixBlock(nn.Module):
    """
    Channel-mixing block: handles local transformations
    """
    def __init__(self, d_model: int, expand: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        
        # Channel-mixing parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
        # Projections
        self.key = nn.Linear(d_model, self.d_inner, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        # Time-mixing
        xx = torch.cat([x[:, 0:1, :], x[:, :-1, :]], dim=1)  # Shift right
        
        # Mix
        k = self.key(x * self.time_mix_k + xx * (1 - self.time_mix_k))
        r = self.receptance(x * self.time_mix_r + xx * (1 - self.time_mix_r))
        
        # Activation and value projection
        k = torch.square(torch.relu(k))  # Squared ReLU
        kv = self.value(k)
        
        # Output
        output = torch.sigmoid(r) * kv
        
        return output


class RWKVXBlock(nn.Module):
    """
    RWKV-X block: combines time-mixing and channel-mixing with residual connections
    """
    def __init__(self, d_model: int, attn_size: int = 64, topk: int = 32, expand: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.time_mix = TimeMixBlock(d_model, attn_size, topk)
        self.norm2 = RMSNorm(d_model)
        self.channel_mix = ChannelMixBlock(d_model, expand)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.time_mix(self.norm1(x))
        x = x + self.channel_mix(self.norm2(x))
        return x


class RWKVXModel(nn.Module):
    """
    RWKV-X Model for sequence modeling
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        attn_size: int = 64,
        sparse_topk: int = 32,
        expand: int = 4,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # RWKV-X blocks
        self.blocks = nn.ModuleList([
            RWKVXBlock(d_model, attn_size, sparse_topk, expand)
            for _ in range(n_layers)
        ])
        
        # Final norm and output projection
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        """
        input_ids: (batch, seq_len)
        labels: (batch, seq_len) optional
        """
        # Embedding
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        
        # RWKV-X blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            # Ignore padding tokens (token_id = 0)
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Check for NaN and replace with a small value if needed
            if torch.isnan(loss):
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 1.0, top_p: float = 0.95):
        """
        Generate tokens autoregressively
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
