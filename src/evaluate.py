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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
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
    # Initialize processor with config
    processor = CaptchaProcessor(config=config)
    
    # Create dataset
    # We assume image base dir is usually related to where the script is run or config, 
    # but strictly speaking, evaluation might happen on a different machine with different paths.
    # The ExperimentConfig has image_base_dir. We might want to allow override if needed, 
    # but for now let's use what's in config or current dir if relative.
    # Actually, CaptchaDataset takes base_dir. 
    # Let's trust the config's image_base_dir or use '.' if relative handling is needed.
    # Ideally, we should allow overriding image_dir.
    # But evaluating usually implies we have the data.
    # Let's stick to config.image_base_dir for now, as per plan. 
    # If users need override, they can change the config or we add arg later.
    
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
    total_edit_distance = 0.0
    total_char_accuracy = 0.0
    exact_matches = 0
    total_samples = 0
    
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
                
                metrics = calculate_metrics(target_str, pred_str)
                
                total_edit_distance += metrics['edit_distance']
                total_char_accuracy += metrics['character_accuracy']
                if metrics['exact_match']:
                    exact_matches += 1
                    
            total_samples += images.size(0)
            
    # 6. Report
    avg_char_acc = total_char_accuracy / total_samples if total_samples > 0 else 0.0
    avg_edit_dist = total_edit_distance / total_samples if total_samples > 0 else 0.0
    exact_match_acc = exact_matches / total_samples if total_samples > 0 else 0.0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {total_samples}")
    print(f"Character Accuracy: {avg_char_acc:.4f}")
    print(f"Edit Distance:      {avg_edit_dist:.4f}")
    print(f"Exact Match (Word): {exact_match_acc:.4f}")
    print("="*50 + "\n")
    
    return {
        "char_accuracy": avg_char_acc,
        "edit_distance": avg_edit_dist,
        "exact_match": exact_match_acc
    }
