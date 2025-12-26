"""
Sequence model implementations with registration.
All sequence models inherit from BaseSequenceModel.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Optional, TYPE_CHECKING, Any

from .base import BaseSequenceModel
from ..registry import REGISTRY
from ...config import (
    TransformerEncoderConfig, 
    TransformerDecoderConfig, 
    RNNConfig, 
    BiLSTMConfig
)

Tensor = torch.Tensor

# ============================================================================
# UTILITIES
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.offset = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (self.d_model ** -0.5)
        return x / (norm + self.eps) * self.scale + self.offset

class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_fn: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        
        if act_fn == "gelu":
            self.act = nn.GELU()
        elif act_fn == "relu":
            self.act = nn.ReLU()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W_out(self.dropout(self.act(self.W_in(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, attention_dir: str = "bidirectional", dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.attention_dir = attention_dir
        
        self.W_Q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = d_head ** -0.5

    def forward(self, x_q, x_k, x_v, mask=None, is_cross_attention=False):
        # x_q: [batch, seq_q, d_model]
        # x_k: [batch, seq_k, d_model]
        # x_v: [batch, seq_k, d_model]
        
        B, Sq, _ = x_q.size()
        Bk, Sk, _ = x_k.size()
        
        Q = self.W_Q(x_q).view(B, Sq, self.n_heads, self.d_head).transpose(1, 2) # [B, H, Sq, Dh]
        K = self.W_K(x_k).view(Bk, Sk, self.n_heads, self.d_head).transpose(1, 2) # [B, H, Sk, Dh]
        V = self.W_V(x_v).view(Bk, Sk, self.n_heads, self.d_head).transpose(1, 2) # [B, H, Sk, Dh]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale # [B, H, Sq, Sk]
        
        if mask is not None:
            # mask should be broadcastable to [B, H, Sq, Sk]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        if self.attention_dir == "causal" and not is_cross_attention:
            causal_mask = torch.triu(torch.ones(Sq, Sk, device=x_q.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.matmul(attn_probs, V) # [B, H, Sq, Dh]
        out = out.transpose(1, 2).contiguous().view(B, Sq, self.n_heads * self.d_head)
        return self.W_O(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.d_head, 
            attention_dir="bidirectional", 
            dropout=cfg.attn_dropout
        )
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_mlp, cfg.act_fn, cfg.dropout)

    def forward(self, x):
        resid = x
        x = self.ln1(x)
        x = self.attn(x, x, x)
        x = resid + x
        
        resid = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = resid + x
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.d_head, 
            attention_dir="bidirectional", # Queries attend to queries fully? Usually causal context for generation.
            # But here user said "Transformer Decoder stack (Self-Attention + Cross-Attention). Used for DETR-style decoding where 'x' are queries and 'encoder_out' is visual features."
            # DETR queries attend to each other, full bidirectional usually.
            dropout=cfg.attn_dropout
        )
        
        self.ln2 = RMSNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.d_head, 
            attention_dir="bidirectional", 
            dropout=cfg.attn_dropout
        )
        
        self.ln3 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_mlp, cfg.act_fn, cfg.dropout)

    def forward(self, x, encoder_out):
        # Self Attention
        resid = x
        x = self.ln1(x)
        x = self.self_attn(x, x, x, is_cross_attention=False)
        x = resid + x
        
        # Cross Attention
        resid = x
        x = self.ln2(x)
        x = self.cross_attn(x, encoder_out, encoder_out, is_cross_attention=True)
        x = resid + x
        
        # MLP
        resid = x
        x = self.ln3(x)
        x = self.mlp(x)
        x = resid + x
        return x


# ============================================================================
# MODELS
# ============================================================================

class TransformerEncoderModel(BaseSequenceModel):
    """
    Standard Transformer Encoder stack (Self-Attention only).
    """
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.cfg = cfg
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(cfg) for _ in range(cfg.n_layers)
        ])

    @property
    def requires_cross_attention(self) -> bool:
        return False

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        for block in self.blocks:
            x = block(x)
        return x


class TransformerDecoderModel(BaseSequenceModel):
    """
    Transformer Decoder stack (Self-Attention + Cross-Attention).
    Used for DETR-style decoding where 'x' are queries and 'encoder_out' is visual features.
    """
    def __init__(self, cfg: TransformerDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(cfg) for _ in range(cfg.n_layers)
        ])

    @property
    def requires_cross_attention(self) -> bool:
        return True

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        
        if encoder_out is None:
            raise ValueError("TransformerDecoderModel requires encoder_out for cross-attention")
            
        for block in self.blocks:
            x = block(x, encoder_out)
        return x


class RNNSequenceModel(BaseSequenceModel):
    """
    Standard RNN Sequence Model.
    """
    def __init__(self, cfg: RNNConfig):
        super().__init__()
        self.cfg = cfg
        
        # RNNConfig.hidden_size corresponds to d_model in our pipeline sync logic
        d_model = cfg.hidden_size
        
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=False # Standard RNN
        )
        self.act = nn.ReLU() # Often helpful after RNN

    @property
    def requires_cross_attention(self) -> bool:
        return False

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        
        # RNN returns (output, h_n)
        out, _ = self.rnn(x)
        return out


class BiLSTMSequenceModel(BaseSequenceModel):
    """
    Bidirectional LSTM Sequence Model.
    Projects output back to d_model size.
    """
    def __init__(self, cfg: BiLSTMConfig):
        super().__init__()
        self.cfg = cfg
        
        # BiLSTMConfig.hidden_size is d_model // 2
        d_model = cfg.hidden_size * 2
            
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=cfg.hidden_size, 
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0
        )

    @property
    def requires_cross_attention(self) -> bool:
        return False

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        
        out, _ = self.lstm(x)
        return out


# ============================================================================
# REGISTRATION
# ============================================================================

REGISTRY.register_sequence_model(
    name="transformer_encoder",
    cls=TransformerEncoderModel,
    description="Stack of Transformer Blocks (Self-Attention only)",
    requires_cross_attention=False
)

REGISTRY.register_sequence_model(
    name="transformer_decoder",
    cls=TransformerDecoderModel,
    description="Stack of Transformer Decoder Blocks (Self + Cross Attention)",
    requires_cross_attention=True
)

REGISTRY.register_sequence_model(
    name="rnn",
    cls=RNNSequenceModel,
    description="Standard Recurrent Neural Network",
    requires_cross_attention=False
)

REGISTRY.register_sequence_model(
    name="bilstm",
    cls=BiLSTMSequenceModel,
    description="Bidirectional LSTM (output dim = d_model)",
    requires_cross_attention=False
)
