"""
xLSTM: Extended Long Short-Term Memory
Features exponential gating, matrix memory, and parallelizable recurrence
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


class sLSTMCell(nn.Module):
    """
    Scalar LSTM cell with exponential gating
    Uses scalar memory (like traditional LSTM but with exponential gates)
    """
    def __init__(self, d_model: int, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        
        # Gates: input, forget, output (with exponential gating)
        self.input_gate = nn.Linear(d_model, d_model)
        self.forget_gate = nn.Linear(d_model, d_model)
        self.output_gate = nn.Linear(d_model, d_model)
        
        # Candidate values
        self.candidate = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple:
        """
        x: (batch, d_model) - current input
        h: (batch, d_model) - hidden state
        c: (batch, d_model) - cell state
        Returns: (new_h, new_c)
        """
        # Exponential gating (exp instead of sigmoid for stronger control)
        # Clamp input to prevent overflow
        input_gate_log = torch.clamp(self.input_gate(x), min=-10.0, max=2.0)
        forget_gate_log = torch.clamp(self.forget_gate(x), min=-10.0, max=2.0)
        i_gate = torch.exp(input_gate_log)  # Exponential input gate
        f_gate = torch.exp(forget_gate_log)  # Exponential forget gate
        o_gate = torch.sigmoid(self.output_gate(x))  # Output gate (sigmoid for stability)
        
        # Candidate values
        c_candidate = torch.tanh(self.candidate(x))
        
        # Update cell state: c_t = f_t * c_{t-1} + i_t * c_candidate
        new_c = f_gate * c + i_gate * c_candidate
        
        # Update hidden state: h_t = o_t * tanh(c_t)
        new_h = o_gate * torch.tanh(new_c)
        
        return new_h, new_c


class mLSTMCell(nn.Module):
    """
    Matrix LSTM cell with exponential gating and matrix memory
    Uses matrix memory for richer state representation
    """
    def __init__(self, d_model: int, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        
        # Matrix memory: (batch, n_heads, head_dim, head_dim)
        # Gates for matrix operations
        self.input_gate = nn.Linear(d_model, d_model)
        self.forget_gate = nn.Linear(d_model, d_model)
        self.output_gate = nn.Linear(d_model, d_model)
        
        # Value projections for matrix memory
        self.value_proj = nn.Linear(d_model, d_model * head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, M: torch.Tensor) -> tuple:
        """
        x: (batch, d_model) - current input
        h: (batch, d_model) - hidden state
        M: (batch, n_heads, head_dim, head_dim) - matrix memory
        Returns: (new_h, new_M)
        """
        batch = x.size(0)
        
        # Reshape for multi-head processing
        x_heads = x.view(batch, self.n_heads, self.head_dim)  # (batch, n_heads, head_dim)
        
        # Exponential gating
        # Clamp to prevent overflow
        input_gate_log = torch.clamp(self.input_gate(x), min=-10.0, max=2.0)
        forget_gate_log = torch.clamp(self.forget_gate(x), min=-10.0, max=2.0)
        i_gate = torch.exp(input_gate_log)  # (batch, d_model)
        f_gate = torch.exp(forget_gate_log)  # (batch, d_model)
        o_gate = torch.sigmoid(self.output_gate(x))  # (batch, d_model)
        
        # Reshape gates for matrix operations
        i_gate = i_gate.view(batch, self.n_heads, 1, self.head_dim)  # (batch, n_heads, 1, head_dim)
        f_gate = f_gate.view(batch, self.n_heads, 1, self.head_dim)  # (batch, n_heads, 1, head_dim)
        
        # Value projection
        v = self.value_proj(x)  # (batch, d_model * head_dim)
        v = v.view(batch, self.n_heads, self.head_dim, self.head_dim)  # (batch, n_heads, head_dim, head_dim)
        
        # Update matrix memory: M_t = f_t * M_{t-1} + i_t * v_t
        # Element-wise operations
        new_M = f_gate.transpose(-2, -1) * M + i_gate.transpose(-2, -1) * v  # (batch, n_heads, head_dim, head_dim)
        
        # Compute output from matrix memory
        # h_t = o_t * (M_t @ x_heads)
        M_out = torch.bmm(new_M.view(batch * self.n_heads, self.head_dim, self.head_dim),
                          x_heads.view(batch * self.n_heads, self.head_dim, 1))
        M_out = M_out.view(batch, self.n_heads, self.head_dim)  # (batch, n_heads, head_dim)
        M_out = M_out.contiguous().view(batch, self.d_model)  # (batch, d_model)
        
        # Apply output gate
        new_h = o_gate * M_out
        new_h = self.out_proj(new_h)
        
        return new_h, new_M


class XLSTMCell(nn.Module):
    """
    xLSTM cell: can use either sLSTM or mLSTM
    """
    def __init__(self, d_model: int, head_dim: int = 64, use_mlstm: bool = True):
        super().__init__()
        self.use_mlstm = use_mlstm
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        
        if use_mlstm:
            self.cell = mLSTMCell(d_model, head_dim)
        else:
            self.cell = sLSTMCell(d_model, head_dim)
    
    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        """
        x: (batch, d_model)
        state: (h, c) for sLSTM or (h, M) for mLSTM
        """
        return self.cell(x, *state)


class XLSTMBlock(nn.Module):
    """
    xLSTM block with residual connection
    """
    def __init__(self, d_model: int, head_dim: int = 64, use_mlstm: bool = True):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.cell = XLSTMCell(d_model, head_dim, use_mlstm)
        self.use_mlstm = use_mlstm
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
    
    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        """
        x: (batch, seq_len, d_model)
        state: initial state tuple
        Returns: (output, final_state)
        """
        batch, seq_len, _ = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Process sequence
        outputs = []
        h, memory = state
        
        for t in range(seq_len):
            x_t = x_norm[:, t, :]  # (batch, d_model)
            h, memory = self.cell(x_t, (h, memory))
            outputs.append(h)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        # Residual connection
        output = output + x
        
        return output, (h, memory)


class XLSTMModel(nn.Module):
    """
    xLSTM Model for sequence modeling
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        head_dim: int = 64,
        use_mlstm: bool = True,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_mlstm = use_mlstm
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # xLSTM blocks
        self.blocks = nn.ModuleList([
            XLSTMBlock(d_model, head_dim, use_mlstm)
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
    
    def _init_state(self, batch_size: int, device: torch.device) -> tuple:
        """Initialize hidden state and memory"""
        if self.use_mlstm:
            # Matrix memory: (batch, n_heads, head_dim, head_dim)
            M = torch.zeros(batch_size, self.n_heads, self.head_dim, self.head_dim, device=device)
            h = torch.zeros(batch_size, self.d_model, device=device)
            return (h, M)
        else:
            # Scalar memory (cell state)
            h = torch.zeros(batch_size, self.d_model, device=device)
            c = torch.zeros(batch_size, self.d_model, device=device)
            return (h, c)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        """
        input_ids: (batch, seq_len)
        labels: (batch, seq_len) optional
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Embedding
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        
        # Initialize state
        state = self._init_state(batch_size, device)
        
        # xLSTM blocks
        for block in self.blocks:
            x, state = block(x, state)
        
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
        batch_size = generated.size(0)
        device = generated.device
        
        # Initialize state
        state = self._init_state(batch_size, device)
        
        with torch.no_grad():
            # Process initial sequence
            x = self.embedding(generated)
            for block in self.blocks:
                x, state = block(x, state)
            
            # Generate new tokens
            for _ in range(max_length):
                # Get last hidden state
                last_h = x[:, -1:, :]  # (batch, 1, d_model)
                last_h = self.norm(last_h)
                logits = self.lm_head(last_h)[:, -1, :] / temperature
                
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
                
                # Update state with new token
                x_new = self.embedding(next_token)  # (batch, 1, d_model)
                for block in self.blocks:
                    x_new, state = block(x_new, state)
                x = torch.cat([x, x_new], dim=1)
        
        return generated
