from dataclasses import dataclass, field
from typing import List, Optional
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

@dataclass
class CaptchaConfig(HookedTransformerConfig):
    """
    Configuration class for the Captcha architecture.
    """
    model_type: str = 'asymmetric-convnext-transformer'
    
    # --- Modular Architecture Components ---
    # These control the UniversalCaptchaModel assembly
    encoder_type: str = 'asymmetric_convnext'   # Options: 'asymmetric_convnext', 'legacy_cnn'
    encoder_dim_out: int = 512                  # Output dimension of the chosen encoder
    
    projector_type: str = 'linear'              # Options: 'linear', 'mlp', 'identity'
    
    sequence_model_type: str = 'transformer_encoder' # Options: 'transformer_encoder', 'transformer_decoder_detr'
    
    head_type: str = 'ctc'                      # Options: 'ctc', 'linear' (for simple/DETR)
    
    # --- Image / Processor Configuration ---
    input_height: int = 80
    input_width_divisor: int = 4                # Width must be divisible by this (e.g. 4 for convnext, 28 for legacy)
    
    # --- Loss Configuration ---
    loss_type: str = 'ctc'                      # Options: 'ctc', 'cross_entropy'

    # --- CNN Encoder Parameters (Legacy) ---
    cnn_filter_sizes: List[int] = field(default_factory=lambda: [7, 5])
    cnn_strides: List[int] = field(default_factory=lambda: [1, 1])
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32])
    
    # --- Transformer Defaults ---
    n_layers: int = 4           
    d_model: int = 256          
    n_heads: int = 8
    d_head: int = 32            
    d_mlp: int = 1024           
    n_ctx: int = 384            
    d_vocab: int = 62           
    act_fn: str = "gelu"        
    normalization_type: str = "RMS"
    
    # --- RoPE Configuration ---
    positional_embedding_type: str = "rotary" 
    rotary_dim: Optional[int] = d_head
    
    attention_dir: str = "causal"
    seed: Optional[int] = None

    def __post_init__(self):
        # 1. Handle Legacy Model Types by mapping them to explicit component configs
        if self.model_type == 'asymmetric-convnext-transformer':
            self.encoder_type = 'asymmetric_convnext'
            self.sequence_model_type = 'transformer_encoder'
            self.head_type = 'ctc'
            self.loss_type = 'ctc'
            self.input_height = 80
            self.input_width_divisor = 4
            self.encoder_dim_out = 512
            
        elif self.model_type == 'cnn-transformer-detr':
            self.encoder_type = 'legacy_cnn'
            self.sequence_model_type = 'transformer_decoder_detr'
            self.head_type = 'linear'
            self.loss_type = 'cross_entropy'
            self.input_height = 70
            self.input_width_divisor = 28
            self.encoder_dim_out = 3136 # Output of flattened patches
            
        # 2. Derived defaults
        if self.d_head is None and self.d_model is not None and self.n_heads is not None:
            self.d_head = self.d_model // self.n_heads
            
        if self.rotary_dim is None and self.positional_embedding_type == "rotary":
            self.rotary_dim = self.d_head

        super().__post_init__()