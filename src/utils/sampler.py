import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(labels, target_ratio=0.5):
    """
    Create a weighted sampler to achieve desired SOZ to non-SOZ ratio
    Args:
        labels: numpy array of binary labels
        target_ratio: desired ratio of SOZ (1) samples
    Returns:
        weights for WeightedRandomSampler
    """
    n_samples = len(labels)
    n_soz = (labels == 1).sum()
    n_non_soz = n_samples - n_soz
    
    # Calculate weights to achieve target ratio
    soz_weight = target_ratio / (n_soz / n_samples)
    non_soz_weight = (1 - target_ratio) / (n_non_soz / n_samples)
    
    # Assign weights to samples
    weights = torch.FloatTensor([
        soz_weight if label == 1 else non_soz_weight 
        for label in labels
    ])
    
    return weights