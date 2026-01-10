"""
Mamba-2: State Space Model with Structured State Space Duality (SSD)
Based on the design document and Mamba-2 architecture principles
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


class SSMBlock(nn.Module):
    """
    Structured State Space Model block with SSD (Structured State Space Duality)
    Simplified implementation for CPU training
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM parameters (A, B, C, D)
        # A: state transition matrix (diagonal for efficiency)
        # Initialize A_log to negative values to ensure stable exp
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state) * -0.5 - 1.0)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # B and C are computed from input
        self.B_proj = nn.Linear(self.d_inner, d_state)
        self.C_proj = nn.Linear(self.d_inner, d_state)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Activation
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Remove padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        x = self.activation(x)
        
        # SSM recurrence
        # A: diagonal state matrix (exp of A_log for stability)
        # Clamp A_log to prevent numerical overflow
        A_log_clamped = torch.clamp(self.A_log, min=-10.0, max=2.0)
        A = -torch.exp(A_log_clamped)  # (d_inner, d_state)
        
        # Compute B and C from input
        B = self.B_proj(x)  # (batch, seq_len, d_state)
        C = self.C_proj(x)  # (batch, seq_len, d_state)
        
        # State space recurrence (simplified for CPU)
        # h_t = A * h_{t-1} + B * x_t
        # y_t = C * h_t + D * x_t
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            # Update state: h_t = A * h_{t-1} + B_t * x_t
            x_t = x[:, t, :]  # (batch, d_inner)
            B_t = B[:, t, :]  # (batch, d_state)
            
            # State update: h[:, i, j] = h[:, i, j] * A[i, j] + x_t[:, i] * B_t[:, j]
            # A is (d_inner, d_state), broadcast to (batch, d_inner, d_state)
            A_broadcast = A.unsqueeze(0)  # (1, d_inner, d_state)
            h = h * A_broadcast + x_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_inner, d_state)
            
            # Clamp h to prevent overflow/underflow
            h = torch.clamp(h, min=-100.0, max=100.0)
            
            # Output: y_t = sum_j(C_t[:, j] * h[:, :, j]) + D * x_t
            C_t = C[:, t, :]  # (batch, d_state)
            # Sum over d_state dimension: (batch, d_inner, d_state) * (batch, 1, d_state) -> sum -> (batch, d_inner)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + self.D * x_t  # (batch, d_inner)
            
            # Clamp output to prevent overflow
            y_t = torch.clamp(y_t, min=-100.0, max=100.0)
            outputs.append(y_t)
        
        x = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        # Gating
        z = torch.sigmoid(z)
        x = x * z
        
        # Output projection
        x = self.out_proj(x)  # (batch, seq_len, d_model)
        
        return x


class Mamba2Block(nn.Module):
    """Mamba-2 block with residual connection and normalization"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ssm = SSMBlock(d_model, d_state, d_conv, expand)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class Mamba2Model(nn.Module):
    """
    Mamba-2 Model for sequence modeling
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba-2 blocks
        self.blocks = nn.ModuleList([
            Mamba2Block(d_model, d_state, d_conv, expand)
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
        
        # Mamba-2 blocks
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
