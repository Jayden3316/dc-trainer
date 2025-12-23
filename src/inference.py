import torch
from pathlib import Path
from typing import List, Union, Tuple
from PIL import Image
from tqdm import tqdm

from src.config.loader import hydrate_config
from src.architecture.model import CaptchaModel
from src.processor import CaptchaProcessor
from src.utils import upsample_image

def run_inference(
    checkpoint_path: str,
    image_paths: List[str],
    image_base_dir: str = None
) -> List[Tuple[str, str]]:
    
    # 1. Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain configuration.")
        
    config = hydrate_config(checkpoint['config'])
    
    # 2. Setup Device & Model
    device = torch.device(config.training_config.device if torch.cuda.is_available() else "cpu")
    model = CaptchaModel(config.model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # 3. Setup Processor
    vocab = checkpoint.get('vocab')
    metadata_path = None
    
    if vocab is None:
        # Fallback for legacy checkpoints
        if config.train_metadata_path:
            p = Path(config.train_metadata_path)
            if p.exists():
                print(f"Loading vocabulary from training metadata: {p}")
                metadata_path = str(p)
            else:
                 print(f"Warning: train_metadata_path {p} found in config but file does not exist. Using default/fallback vocabulary.")
        else:
             print("Warning: No vocab in checkpoint and no train_metadata_path in config. Using fallback vocabulary.")
    else:
        print(f"Loaded vocabulary of size {len(vocab)} from checkpoint.")

    processor = CaptchaProcessor(config=config, metadata_path=metadata_path, vocab=vocab)
    
    results = []
    
    print(f"Running inference on {len(image_paths)} images...")
    
    with torch.no_grad():
        for img_path_str in image_paths:
            p = Path(img_path_str)
            if image_base_dir:
                 p = Path(image_base_dir) / p
            
            if not p.exists():
                print(f"Warning: Image not found at {p}")
                results.append((str(p), "<FILE_NOT_FOUND>"))
                continue
                
            try:
                # Load and Process Image manually since we aren't using Dataset
                # We need to replicate what Processor.__call__ does but maybe simpler or just use it.
                # CaptchaProcessor call expects (image_path, text). Text is dummy for inference.
                # But Processor reads file internally if path is passed.
                
                processed = processor(str(p)) 
                if processed is None:
                     results.append((str(p), "<CORRUPT_IMAGE>"))
                     continue
                     
                # Add batch dimension
                image_tensor = processed['pixel_values'].unsqueeze(0).to(device)
                
                # Forward
                logits = model(image_tensor)
                pred_ids = logits.argmax(dim=-1).squeeze(0)
                
                # Decode
                pred_str = processor.decode(pred_ids)
                
                results.append((str(p), pred_str))
                print(f"{p}: {pred_str}")
                
            except Exception as e:
                print(f"Error processing {p}: {e}")
                results.append((str(p), f"<ERROR: {e}>"))
                
    return results
