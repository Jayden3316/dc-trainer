from typing import Dict, Any
from src.config.config import (
    ExperimentConfig, DatasetConfig, TrainingConfig, ModelConfig,
    AsymmetricConvNextEncoderConfig, LegacyCNNEncoderConfig,
    LinearProjectorConfig, MLPProjectorConfig, IdentityProjectorConfig, BottleneckProjectorConfig, ResidualProjectorConfig,
    TransformerEncoderConfig, TransformerDecoderConfig, RNNConfig, BiLSTMConfig,
    LinearHeadConfig, MLPHeadConfig, ClassificationHeadConfig
)

def hydrate_config(data: Dict[str, Any]) -> ExperimentConfig:
    """Manually hydrate dictionary into ExperimentConfig hierarchy.
    Shared logic extracted from cli.py.
    """
    
    # 1. Dataset Config
    ds_data = data.get('dataset_config', data.get('dataset', {}))
    dataset_config = DatasetConfig(**ds_data)
    
    # 2. Training Config
    tr_data = data.get('training_config', data.get('training', {}))
    training_config = TrainingConfig(**tr_data)
    
    # 3. Model Config
    mc_data = data.get('model_config', data.get('model', {}))
    
    # Helper to selecting config class based on type name
    def get_config_obj(type_name, config_dict, mapping, default_cls):
        # If type_name is missing, we revert to default class
        if not type_name:
             return default_cls()
        
        # If type_name is present, we must use that class
        cls = mapping.get(type_name, default_cls)
        
        # If config_dict is None, treat as empty dict
        if config_dict is None:
            config_dict = {}

        # Filter keys that valid for the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_args = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_args)

    # Encoders
    encoder_type = mc_data.get('encoder_type')
    encoder_cls_map = {
        'asymmetric_convnext': AsymmetricConvNextEncoderConfig,
        'legacy_cnn': LegacyCNNEncoderConfig
    }
    encoder_config = get_config_obj(encoder_type, mc_data.get('encoder_config'), encoder_cls_map, AsymmetricConvNextEncoderConfig)

    # Projectors
    proj_type = mc_data.get('projector_type')
    proj_cls_map = {
        'linear': LinearProjectorConfig,
        'mlp': MLPProjectorConfig,
        'identity': IdentityProjectorConfig,
        'bottleneck': BottleneckProjectorConfig,
        'residual': ResidualProjectorConfig
    }
    projector_config = get_config_obj(proj_type, mc_data.get('projector_config'), proj_cls_map, LinearProjectorConfig)
    
    # Sequence Models
    seq_type = mc_data.get('sequence_model_type')
    seq_cls_map = {
        'transformer_encoder': TransformerEncoderConfig,
        'transformer_decoder': TransformerDecoderConfig, 
        'transformer_decoder_detr': TransformerDecoderConfig, # Alias
        'rnn': RNNConfig,
        'bilstm': BiLSTMConfig
    }
    sequence_model_config = get_config_obj(seq_type, mc_data.get('sequence_model_config'), seq_cls_map, TransformerEncoderConfig)

    # Heads
    head_type = mc_data.get('head_type')
    head_cls_map = {
        'linear': LinearHeadConfig, # and 'ctc' shares this
        'ctc': LinearHeadConfig,
        'mlp': MLPHeadConfig,
        'classification': ClassificationHeadConfig
    }
    head_config = get_config_obj(head_type, mc_data.get('head_config'), head_cls_map, LinearHeadConfig)
    
    # Model Config Args
    # We construct args map to pass only what's in YAML, avoiding Loader-level defaults
    # defaulting is handled by ModelConfig.__post_init__ or field defaults
    mc_args = {
        'encoder_type': encoder_type,
        'encoder_config': encoder_config,
        'projector_type': proj_type,
        'projector_config': projector_config,
        'sequence_model_type': seq_type,
        'sequence_model_config': sequence_model_config,
        'head_type': head_type,
        'head_config': head_config,
    }
    
    # Optional scalar fields
    for field in ['d_model', 'd_vocab', 'loss_type', 'task_type']:
        if field in mc_data:
            mc_args[field] = mc_data[field]

    model_config = ModelConfig(**mc_args)
    
    
    # Construct ExperimentConfig with optional fields
    # We only pass fields that are present in the data to allow dataclass defaults to take effect
    config_args = {
        'experiment_name': data.get('experiment_name', 'custom_run'),
        'dataset_config': dataset_config,
        'training_config': training_config,
        'model_config': model_config,
    }
    
    # Optional fields that should only be passed if present in YAML
    optional_fields = [
        'metadata_path', 
        'train_metadata_path', 
        'val_metadata_path', 
        'image_base_dir', 
        'seed'
    ]
    
    for field_name in optional_fields:
        if field_name in data:
            config_args[field_name] = data[field_name]

    return ExperimentConfig(**config_args)
