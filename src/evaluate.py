import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Any

from src.config.config import ExperimentConfig
from src.architecture.model import CaptchaModel
from src.processor import CaptchaProcessor
from src.train import CaptchaDataset, collate_fn
from src.utils import calculate_metrics
from src.decoding import decode_simple

from src.config.loader import hydrate_config

def evaluate(
    checkpoint_path: str,
    metadata_path: str,
    batch_size: int = 32,
    num_workers: int = 4
):
    # 1. Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain configuration. Please use a checkpoint trained with the updated train.py.")
    
    config_dict = checkpoint['config']
    config = hydrate_config(config_dict)
    
    # 2. Setup Device
    device = torch.device(config.training_config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Load Model
    print(f"Initializing model: {config.model_config.encoder_type} + {config.model_config.sequence_model_type}...")
    model = CaptchaModel(config.model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # 4. Setup Data
    print(f"Loading evaluation dataset from {metadata_path}...")
    # 4. Setup Data
    print(f"Loading evaluation dataset from {metadata_path}...")
    
    # 3. Setup Processor
    vocab = checkpoint.get('vocab')
    vocab_path = None
    
    if vocab is None:
        # Fallback for legacy checkpoints
        if config.train_metadata_path:
            p = Path(config.train_metadata_path)
            if p.exists():
                print(f"Loading vocabulary from training metadata: {p}")
                vocab_path = str(p)
            else:
                 print(f"Warning: train_metadata_path {p} found in config but file does not exist. Using default/fallback vocabulary.")
        else:
             print("Warning: No vocab in checkpoint and no train_metadata_path in config. Using fallback vocabulary.")
    else:
        print(f"Loaded vocabulary of size {len(vocab)} from checkpoint.")

    processor = CaptchaProcessor(config=config, metadata_path=vocab_path, vocab=vocab)
    
    # Create dataset
    dataset = CaptchaDataset(
        metadata_path=metadata_path, 
        processor=processor, 
        base_dir=config.image_base_dir
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=num_workers
    )
    
    print(f"Evaluation samples: {len(dataset)}")
    
    # 5. Run Evaluation
    total_samples = 0
    val_preds = []
    val_targets = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None: continue
            
            images = batch["pixel_values"].to(device)
            targets = batch["input_ids"].to(device) # [B, S] (padded)
            target_lengths = batch["target_lengths"].to(device)
            
            logits = model(images) # [B, S, V]
            preds = logits.argmax(dim=-1) # [B, S]
            
            for i in range(len(targets)):
                pred_str = processor.decode(preds[i])
                if targets.dim() == 2:
                    # Sequence Task (Generation/OCR)
                    target_ids = targets[i][:target_lengths[i]].tolist()
                    target_str = decode_simple(target_ids, processor.idx_to_char)
                else:
                    # Classification Task
                    target_str = processor.decode(targets[i])
                
                val_preds.append(pred_str)
                val_targets.append(target_str)
                    
            total_samples += images.size(0)
            
    # 6. Report & Calculate Metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {total_samples}")
    
    results = {}
    metrics_to_compute = config.training_config.metrics
    
    # Default metrics if none specified
    if not metrics_to_compute:
        print("No metrics specified in config. Using default set.")
        # Infer defaults based on task type ideally, but let's do a safe set
        if config.model_config.task_type == 'classification':
             metrics_to_compute = ['accuracy']
        else:
             metrics_to_compute = ['character_accuracy', 'exact_match']

    if metrics_to_compute:
        # 1. OCR-style metrics
        ocr_keys = ['character_accuracy', 'edit_distance', 'exact_match']
        needed_ocr = [m for m in metrics_to_compute if m in ocr_keys]
        
        if needed_ocr:
            total_edit_dist = 0.0
            total_char_acc = 0.0
            total_exact = 0
            
            for t, p in zip(val_targets, val_preds):
                m = calculate_metrics(t, p)
                total_edit_dist += m['edit_distance']
                total_char_acc += m['character_accuracy']
                if m['exact_match']:
                    total_exact += 1
            
            if 'edit_distance' in needed_ocr:
                val = total_edit_dist / total_samples if total_samples > 0 else 0
                results['edit_distance'] = val
                print(f"Edit Distance:      {val:.4f}")
            if 'character_accuracy' in needed_ocr:
                val = total_char_acc / total_samples if total_samples > 0 else 0
                results['character_accuracy'] = val
                print(f"Character Accuracy: {val:.4f}")
            if 'exact_match' in needed_ocr:
                val = total_exact / total_samples if total_samples > 0 else 0
                results['exact_match'] = val
                print(f"Exact Match (Word): {val:.4f}")

        # 2. Classification-style metrics
        cls_keys = ['f1', 'precision', 'recall', 'accuracy']
        needed_cls = [m for m in metrics_to_compute if m in cls_keys]
        
        if needed_cls:
            from src.utils import calculate_classification_metrics
            cls_results = calculate_classification_metrics(val_targets, val_preds, needed_cls)
            results.update(cls_results)
            for k, v in cls_results.items():
                 print(f"{k.capitalize()}: {v:.4f}")

    print("="*50 + "\n")
    
    return results
