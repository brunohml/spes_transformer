import os
import sys
# Add the src directory to the Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_dir)

import logging
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import argparse

from src.datasets.data import DynamicSPESData
from src.datasets.dataset import DynamicImputationDataset
from src.models.ts_transformer import model_factory

def load_config(model_dir):
    """Load config from model_dir/configuration.json"""
    config_path = os.path.join(model_dir, 'configuration.json')
    print(f"Looking for configuration file at: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def plot_imputation_grid(original_data, masked_data, predictions, example_id, output_path, mean_mask_length, masking_ratio, best_metric):
    """Create 36x1 grid visualization of imputation results"""
    print(f"Plotting with reconstruction accuracy: {best_metric:.4f}")
    
    fig = plt.figure(figsize=(10, 85))
    gs = fig.add_gridspec(37, 1)
    
    title_ax = fig.add_subplot(gs[0])
    title = f'Imputation Results: Original (blue) vs Predicted (orange)\n' \
            f'Mask Length: {mean_mask_length}, Masking Ratio: {masking_ratio}\n' \
            f'Best Reconstruction Accuracy: {best_metric:.4f}'
    title_ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=16)
    title_ax.axis('off')
    
    axes = [fig.add_subplot(gs[i+1]) for i in range(36)]
    
    example_data = original_data[0].T  # (36, 487)
    example_pred = predictions[0].T    # (36, 487)
    example_mask = masked_data[0].T    # (36, 487)
    
    for row, ax in enumerate(axes):
        # Plot original signal
        ax.plot(example_data[row], 'b-', linewidth=2, label='Original')
        
        # Plot predicted signal
        ax.plot(example_pred[row], 'orange', linewidth=1, alpha=0.8, label='Predicted')
        
        # Highlight masked regions
        mask = ~example_mask[row]  # Invert mask since True means "keep"
        for start, length in get_mask_segments(mask):
            ax.axvspan(start, start + length - 1, color='purple', alpha=0.2)
        
        # Set labels and limits
        if row == 35:
            ax.set_xlabel('Time')
        ax.set_ylabel(f'Ch {row}')
        ax.set_xlim(0, 486)
        
        # Add legend only to first subplot
        if row == 0:
            ax.plot([], [], color='purple', alpha=0.2, label='Masked', linewidth=10)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("\n=== Plotting Complete ===")

def get_mask_segments(mask):
    """Convert boolean mask array into list of (start_idx, length) tuples
    for consecutive True values"""
    segments = []
    start = None
    
    for i, val in enumerate(mask):
        if val and start is None:  # Start of new masked segment
            start = i
        elif not val and start is not None:  # End of masked segment
            segments.append((start, i - start))
            start = None
    
    # Handle case where mask ends with True
    if start is not None:
        segments.append((start, len(mask) - start))
    
    return segments

def get_best_metric(model_dir):
    """Load best reconstruction accuracy from metrics_.xls"""
    metrics_path = os.path.join(model_dir, 'metrics_.xls')
    print(f"\nLooking for metrics file at: {metrics_path}")
    
    try:
        df = pd.read_excel(metrics_path)
        print("\nAvailable columns in metrics file:", df.columns.tolist())
        
        if 'recon_acc' not in df.columns:
            print("Warning: 'recon_acc' column not found in metrics file")
            return float('nan')
        
        best_metric = df['recon_acc'].iloc[-1]  # Get last value
        print(f"Found reconstruction accuracy: {best_metric:.4f}")
        
        if pd.isna(best_metric):
            print("Warning: Retrieved value is NaN")
            return float('nan')
            
        return best_metric
        
    except Exception as e:
        print(f"Error reading metrics file: {str(e)}")
        return float('nan')

def main(args):
    print("\nStarting visualization script...")
    
    # Load config from model directory
    config = load_config(args.model_path)
    print("Loaded config")
    
    # Get model checkpoint path
    model_checkpoint = os.path.join(args.model_path, 'checkpoints', 'model_best.pth')
    print(f"Will load model from: {model_checkpoint}")
    
    # Get best reconstruction accuracy from metrics file
    recon_acc = get_best_metric(args.model_path)  # Store in a separate variable
    
    # Load data
    data = DynamicSPESData(config['data_dir'], pattern='train', config=config)
    print(f"Loaded data with {len(data.all_IDs)} examples")
    
    # Select one random example
    example_id = random.choice(data.all_IDs)
    print(f"Selected example: {example_id}")
    
    # Create dataset with single example
    dataset = DynamicImputationDataset(data, [example_id], 
                                     mean_mask_length=config['mean_mask_length'],
                                     masking_ratio=config['masking_ratio'],
                                     mode=config['mask_mode'],
                                     distribution=config['mask_distribution'])
    print("Created DynamicImputationDataset")
    
    loader = DataLoader(dataset=dataset,
                       batch_size=1,
                       shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model_factory(config, data)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    best_metric = checkpoint.get('best_metric', float('nan'))  # Get best metric from checkpoint
    model.to(device)
    model.eval()
    print("Model loaded and ready")
    
    with torch.no_grad():
        batch = next(iter(loader))
        X = batch[0].float().to(device)
        mask = batch[1].to(device)
        padding_mask = torch.ones(X.shape[0], X.shape[1], dtype=torch.bool, device=device)
        
        print(f"\nInput shapes:")
        print(f"X: {X.shape}")
        print(f"mask: {mask.shape}")
        print(f"padding_mask: {padding_mask.shape}")
        
        predictions = model(X, padding_mask)
        print(f"Predictions shape: {predictions.shape}")
    
    plot_imputation_grid(X.cpu().numpy(), 
                        mask.cpu().numpy(),
                        predictions.cpu().numpy(),
                        example_id,
                        args.output_path,
                        mean_mask_length=config['mean_mask_length'],
                        masking_ratio=config['masking_ratio'],
                        best_metric=recon_acc)  # Pass the stored recon_acc value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model directory (e.g., spes_output/pretrained/model_name)')
    args = parser.parse_args()
    
    # Set output path in same directory as model
    args.output_path = os.path.join(args.model_path, 'imputation_vis.png')
    
    main(args)
