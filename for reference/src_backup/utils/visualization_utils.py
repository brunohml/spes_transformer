import os
import torch
import logging
from torch.utils.data import DataLoader
from datasets.dataset import ImputationDataset
from datasets.datasplit import split_dataset

def create_imputation_visualization(model, data, device, config, output_dir):
    """Create visualization of imputation results using best model"""
    logger = logging.getLogger('visualization')
    
    # Get test set indices using the same split function used in training
    _, test_indices = split_dataset(data, val_ratio=0, test_pattern=config['val_pattern'])
    
    # Filter labels to only include test set
    test_labels_df = data.labels_df.loc[test_indices]
    
    # Select random examples from test set
    soz_examples = test_labels_df[test_labels_df['soz'] == 1].index.tolist()
    non_soz_examples = test_labels_df[test_labels_df['soz'] == 0].index.tolist()
    
    if not soz_examples or not non_soz_examples:
        logger.warning("Cannot create visualization: missing SOZ or non-SOZ examples in test set")
        return
        
    example_ids = [soz_examples[0], non_soz_examples[0]]
    
    # Create dataset and loader for selected examples
    dataset = ImputationDataset(
        data, 
        example_ids,
        mean_mask_length=config['mean_mask_length'],
        masking_ratio=config['masking_ratio'],
        mode=config['mask_mode'],
        distribution=config['mask_distribution']
    )
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=len(example_ids),
        shuffle=False
    )
    
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(loader))
        X = batch[0].float().to(device)
        mask = batch[1].to(device)
        padding_mask = torch.ones(X.shape[0], X.shape[1], dtype=torch.bool, device=device)
        predictions = model(X, padding_mask)
    
    # Import visualization function here to avoid circular imports
    from visualize_imputation import plot_imputation_grid
    
    # Create visualization
    plot_imputation_grid(
        data.feature_df,
        mask.cpu().numpy(),
        predictions.cpu().numpy(),
        example_ids,
        os.path.join(output_dir, 'imputation_visualization.png')
    ) 