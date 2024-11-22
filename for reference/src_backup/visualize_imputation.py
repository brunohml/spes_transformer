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
import torch.nn as nn
import argparse

# Import DynamicSPESData directly
from src.datasets.data import DynamicSPESData
from src.models.ts_transformer import model_factory
from src.datasets.dataset import DynamicImputationDataset

def load_config(model_path):
    """Load config from the model directory"""
    config_path = os.path.join(model_path, 'configuration.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def select_random_examples(data, n_soz=1, n_non_soz=1):
    """Randomly select n_soz SOZ examples and n_non_soz non-SOZ examples"""
    soz_examples = data.labels_df[data.labels_df['soz'] == 1].index.tolist()
    non_soz_examples = data.labels_df[data.labels_df['soz'] == 0].index.tolist()
    
    selected_soz = random.sample(soz_examples, n_soz)
    selected_non_soz = random.sample(non_soz_examples, n_non_soz)
    
    return selected_soz + selected_non_soz

def plot_imputation_grid(original_data, masked_data, predictions, example_id, output_path, mean_mask_length, masking_ratio):
    """Create 36x1 grid visualization of imputation results"""
    # Add debugging prints
    print(f"\nDebugging Plot Data:")
    print(f"Original data shape: {original_data.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Original data non-zero timesteps: {np.sum(original_data[0, :, 0] != 0)}")
    print(f"Predictions non-zero timesteps: {np.sum(predictions[0, :, 0] != 0)}")
    
    fig, axes = plt.subplots(36, 1, figsize=(10, 40))
    title = f'Imputation Results: Original (blue) vs Predicted (orange)\nMask Length: {mean_mask_length}, Masking Ratio: {masking_ratio}'
    fig.suptitle(title, fontsize=16, y=0.99)
    
    # Get the data for this example
    example_data = original_data[0].T  # (36, 487)
    example_pred = predictions[0].T     # (36, 487)
    example_mask = masked_data[0].T     # (36, 487)
    
    for row in range(36):
        ax = axes[row]
        
        # Plot original signal with thicker line
        ax.plot(range(487), example_data[row], 'b-', linewidth=2, label='Original')
        
        # Plot predicted signal with thinner line
        ax.plot(range(487), example_pred[row], 'orange', linewidth=1, alpha=0.8, label='Predicted')
        
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
            ax.legend()
            # Add a purple patch to legend for masked regions
            ax.fill_between([], [], [], color='purple', alpha=0.2, label='Masked')
            ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path)
    plt.close()

def get_mask_segments(mask):
    """
    Convert boolean mask array into list of (start_idx, length) tuples
    for consecutive True values
    """
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

def main(args):
    print("\nStarting visualization script...")
    
    # Get model checkpoint path and load config
    model_checkpoint = os.path.join(args.model_path, 'checkpoints', 'model_best.pth')
    config = load_config(args.model_path)
    print("Loaded config")
    
    # Load data using DynamicSPESData
    data = DynamicSPESData(config['data_dir'], pattern='train', config=config)
    print(f"Loaded test data with {len(data.all_IDs)} examples")
    
    # Select a random example
    example_id = random.choice(data.all_IDs)
    print(f"Selected example ID: {example_id}")
    
    # Create dataset with single example
    dataset = DynamicImputationDataset(data, [example_id], 
                                     mean_mask_length=config['mean_mask_length'],
                                     masking_ratio=config['masking_ratio'],
                                     mode=config['mask_mode'],
                                     distribution=config['mask_distribution'])
    print("Created DynamicImputationDataset")
    
    loader = DataLoader(dataset=dataset,
                       batch_size=1,  # Only one example
                       shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model_factory(config, data)
    checkpoint = torch.load(model_checkpoint, map_location=device)

        # Add debugging for model state
    print("\nModel checkpoint info:")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best metric: {checkpoint.get('best_metric', 'unknown')}")
    
    # Check if BatchNorm statistics exist
    sample_bn = next(layer for layer in model.modules() if isinstance(layer, nn.BatchNorm1d))
    print("\nBatchNorm running stats before loading:")
    print(f"Mean: {sample_bn.running_mean[:5]}")
    print(f"Var: {sample_bn.running_var[:5]}")
    
    
    print("\nBatchNorm running stats after loading:")
    print(f"Mean: {sample_bn.running_mean[:5]}")
    print(f"Var: {sample_bn.running_var[:5]}")

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.train()
    with torch.no_grad():
        batch = next(iter(loader))
        X = batch[0].float().to(device)
        mask = batch[1].to(device)
        padding_mask = torch.ones(X.shape[0], X.shape[1], dtype=torch.bool, device=device)
        
        print(f"\nDebugging Data:")
        print(f"Input tensor shape: {X.shape}")
        print(f"Mask tensor shape: {mask.shape}")
        print(f"Number of non-zero values in input: {torch.sum(X != 0)}")
        print(f"Number of True values in mask: {torch.sum(mask)}")
        
        # Add debugging prints for intermediate activations
        def hook_fn(module, input, output):
            print(f"\n{module.__class__.__name__} output stats:")
            print(f"Mean: {output.mean().item():.4f}")
            print(f"Std: {output.std().item():.4f}")
            print(f"Min: {output.min().item():.4f}")
            print(f"Max: {output.max().item():.4f}")
        
        # Register hooks
        hooks = []
        hooks.append(model.project_inp.projection.conv_layers.register_forward_hook(hook_fn))  # Conv output
        hooks.append(model.project_inp.projection.fc_layer[-1].register_forward_hook(hook_fn))  # LayerNorm
        hooks.append(model.transformer_encoder.layers[-1].norm2.register_forward_hook(hook_fn))  # Last BatchNorm
        hooks.append(model.output_layer.register_forward_hook(hook_fn))  # Final output
        
        predictions = model(X, padding_mask)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Get the original data from the dataset
    original_data = X.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    plot_imputation_grid(original_data, 
                        mask.cpu().numpy(),
                        predictions,
                        example_id,
                        args.output_path,
                        mean_mask_length=config['mean_mask_length'],
                        masking_ratio=config['masking_ratio'])

    # After loading the model
    print("\nOutput layer weight stats:")
    output_weights = model.output_layer.weight.data
    print(f"Weight mean: {output_weights.mean():.4f}")
    print(f"Weight std: {output_weights.std():.4f}")
    print(f"Weight min: {output_weights.min():.4f}")
    print(f"Weight max: {output_weights.max():.4f}")

    # Also check input range
    print("\nInput data stats:")
    print(f"Input mean: {X.mean():.4f}")
    print(f"Input std: {X.std():.4f}")
    print(f"Input min: {X.min():.4f}")
    print(f"Input max: {X.max():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model directory containing checkpoints/model_best.pth and configuration.json')
    parser.add_argument('--output_path', type=str, default='imputation_visualization.png',
                       help='Where to save the visualization')
    args = parser.parse_args()
    main(args)
