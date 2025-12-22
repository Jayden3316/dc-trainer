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
    ds_data = data.get('dataset_config', {})
    dataset_config = DatasetConfig(**ds_data)
    
    # 2. Training Config
    tr_data = data.get('training_config', {})
    training_config = TrainingConfig(**tr_data)
    
    # 3. Model Config
    mc_data = data.get('model_config', {})
    
    # Helper to selecting config class based on type name
    def get_config_obj(type_name, config_dict, mapping, default_cls):
        if not type_name or not config_dict:
            return default_cls()
        cls = mapping.get(type_name, default_cls)
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
    
    model_config = ModelConfig(
        encoder_type=encoder_type,
        encoder_config=encoder_config,
        projector_type=proj_type,
        projector_config=projector_config,
        sequence_model_type=seq_type,
        sequence_model_config=sequence_model_config,
        head_type=head_type,
        head_config=head_config,
        d_model=mc_data.get('d_model', 256),
        d_vocab=mc_data.get('d_vocab', 62),
        loss_type=mc_data.get('loss_type', 'ctc')
    )
    
    return ExperimentConfig(
        experiment_name=data.get('experiment_name', 'custom_run'),
        dataset_config=dataset_config,
        training_config=training_config,
        model_config=model_config,
        metadata_path=data.get('metadata_path', 'data/metadata.json'),
        image_base_dir=data.get('image_base_dir', 'data/images')
    )
