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

# Import SPESData directly
from datasets.data import SPESData
from datasets.dataset import ImputationDataset
from models.ts_transformer import model_factory

def load_config(model_dir):
    """Load config from the model directory"""
    config_path = os.path.join(model_dir, 'configuration.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def select_random_examples(data, n_soz=1, n_non_soz=1):
    """Randomly select n_soz SOZ examples and n_non_soz non-SOZ examples"""
    soz_examples = data.labels_df[data.labels_df['soz'] == 1].index.tolist()
    non_soz_examples = data.labels_df[data.labels_df['soz'] == 0].index.tolist()
    
    selected_soz = random.sample(soz_examples, n_soz)
    selected_non_soz = random.sample(non_soz_examples, n_non_soz)
    
    return selected_soz + selected_non_soz

def plot_imputation_grid(original_data, masked_data, predictions, example_ids, output_path):
    """Create 36x2 grid visualization of imputation results"""
    print("\n=== DEBUG INFO ===")
    print("Shapes:")
    print(f"Original data shape: {original_data.shape}")
    print(f"Masked data shape: {masked_data.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Example IDs: {example_ids}")
    
    fig, axes = plt.subplots(36, 2, figsize=(10, 80))
    fig.suptitle('Imputation Results: Original (blue), Masked (light blue), Predicted (orange)', fontsize=16)
    
    for col, example_id in enumerate(example_ids):
        print(f"\nProcessing example {col + 1} (ID: {example_id}):")
        example_data = original_data.loc[example_id]
        
        for row in range(36):
            ax = axes[row, col]
            channel = f'channel_{row}'
            
            # Debug first channel of each example
            if row == 0:
                print(f"\nChannel {row}:")
                print(f"Original data range: [{example_data[channel].min():.2f}, {example_data[channel].max():.2f}]")
                mask = masked_data[col][row]
                masked_indices = np.where(~mask)[0]
                print(f"Number of masked points: {len(masked_indices)}")
                print(f"Masked indices (first 10): {masked_indices[:10]}")
                print(f"Last masked index: {masked_indices[-1] if len(masked_indices) > 0 else 'None'}")
                
                if len(masked_indices) > 0:
                    print("\nSample of values:")
                    print("Original:", example_data[channel].values[masked_indices[:5]])
                    print("Predicted:", predictions[col][row][masked_indices[:5]])
            
            # Plot original signal
            ax.plot(example_data[channel].values, 'b-', linewidth=1)
            
            # Plot masked and predicted points
            mask = masked_data[col][row]
            masked_indices = np.where(~mask)[0]
            
            if len(masked_indices) > 0:
                ax.scatter(masked_indices, 
                          example_data[channel].values[masked_indices],
                          color='lightblue', alpha=0.6, label='Masked' if row == 0 else None)
                
                ax.scatter(masked_indices, 
                          predictions[col][row][masked_indices],
                          color='orange', alpha=0.8, label='Predicted' if row == 0 else None)
            
            # Set labels and limits
            if row == 35:
                ax.set_xlabel('Time')
            if col == 0:
                ax.set_ylabel(f'Ch {row}')
            ax.set_xlim(0, 486)
            
            # Add SOZ status and legend to first row
            if row == 0:
                soz_status = "SOZ" if data.labels_df.loc[example_id]['soz'] == 1 else "Non-SOZ"
                ax.set_title(f"Example {col+1} ({soz_status})")
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("\n=== Plotting Complete ===")

def main(args):
    print("\nStarting visualization script...")
    
    # Load config and model
    config = load_config(os.path.dirname(os.path.dirname(args.model_path)))
    print("Loaded config")
    
    # Load data
    global data
    data = SPESData(config['data_dir'], pattern='TEST', config=config)
    print(f"Loaded test data with {len(data.all_IDs)} examples")
    
    # Select examples
    example_ids = select_random_examples(data, n_soz=1, n_non_soz=1)
    print(f"Selected examples: {example_ids}")
    
    dataset = ImputationDataset(data, example_ids, 
                               mean_mask_length=config['mean_mask_length'],
                               masking_ratio=config['masking_ratio'],
                               mode=config['mask_mode'],
                               distribution=config['mask_distribution'])
    print("Created ImputationDataset")
    
    loader = DataLoader(dataset=dataset,
                       batch_size=len(example_ids),
                       shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model_factory(config, data)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
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
    
    plot_imputation_grid(data.feature_df, 
                        mask.cpu().numpy(),
                        predictions.cpu().numpy(),
                        example_ids,
                        args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model_best.pth')
    parser.add_argument('--output_path', type=str, default='imputation_visualization.png',
                        help='Where to save the visualization')
    args = parser.parse_args()
    main(args)
