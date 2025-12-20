# Captcha OCR: Modular Experimentation Framework

A flexible, modular, and config-driven framework for researching and breaking text-based CAPTCHAs. This project is designed to enable combinatorial experimentation with different encoders, sequence models, and decoders.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone github.com/Jayden3316/dc-training.git
cd captcha_ocr

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Dataset

Create a `dataset_config.yaml` file to define your dataset parameters.

```yaml
# dataset_config.yaml
width: 200
height: 80
width_divisor: 4
width_bias: 0
min_chars: 4
max_chars: 8
fonts: ["fonts/arial.ttf"]
max_fonts_per_family: 2
```

### 3. Generate Data

Generate a synthetic dataset using the config.

```bash
python -m captcha_ocr.cli generate --config-file dataset_config.yaml --count 1000 --output-dir data/train --word-file data/words.tsv
```

### 4. Configure Experiment

Create an `experiment_config.yaml` for your model and training loop.

```yaml
# experiment_config.yaml
experiment_name: "convnext_transformer_ctc"
seed: 42
image_base_dir: "data/train"
metadata_path: "data/train/metadata.json"

# Note: Dataset config here determines model input shapes
dataset_config:
  width: 200
  height: 80
  width_divisor: 4
  width_bias: 0

model_config:
  encoder_type: "asymmetric_convnext"
  encoder_config:
    dims: [64, 128, 256, 512]
    stage_block_counts: [2, 2, 6, 2]
  
  projector_type: "linear"
  
  sequence_model_type: "transformer_encoder"
  sequence_model_config:
    d_model: 256
    n_layers: 4
    d_head: 64
    
  head_type: "ctc"
  head_config:
    d_model: 256
    d_vocab: 62
  
  d_model: 256
  d_vocab: 62
  loss_type: "ctc"

training_config:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  optimizer_type: "adamw"
  checkpoint_dir: "checkpoints"
```

### 5. Run Experiment

Run the training command with your config.

```bash
python -m captcha_ocr.cli train --config-file experiment_config.yaml
```

---

## Configuration Reference

The framework enforces strict configuration via YAML files.

-   **Generation (`dataset_config.yaml`)**: Maps directly to `DatasetConfig` fields.
-   **Training (`experiment_config.yaml`)**: Maps to `ExperimentConfig` hierarchy.

### Component Details

#### Encoders
-   **`asymmetric_convnext`**: A modern ConvNeXt-based encoder.
-   **`legacy_cnn`**: A traditional CNN (requires `width_divisor=28`, `width_bias=14`).

#### Sequence Models
-   **`transformer_encoder`**: Standard BERT-like encoder.
-   **`transformer_decoder`**: Decoder with cross-attention.
-   **`rnn` / `bilstm`**: Classic recurrent layers.

#### Heads
-   **`ctc`**: CTC Loss head.
-   **`classification`**: Simple classification head.
