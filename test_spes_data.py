import os
import logging
import pandas as pd
import numpy as np
from spes_data_class import SPESData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_spes_data():
    """Test the SPESData class functionality"""
    
    # Initialize the data class
    root_dir = "spes_trial_pickles"  # adjust path as needed
    logger.info(f"Testing SPESData with directory: {root_dir}")
    
    try:
        spes_data = SPESData(root_dir)
        
        # Basic information
        logger.info(f"Number of trials: {len(spes_data.all_IDs)}")
        logger.info(f"Total DataFrame shape: {spes_data.all_df.shape}")
        logger.info(f"Expected shape: {len(spes_data.all_IDs)} trials * {spes_data.seq_len} timepoints x {spes_data.n_electrodes} channels")
        
        # Check single trial structure
        first_trial_id = spes_data.all_IDs[0]
        first_trial_data = spes_data.all_df.loc[first_trial_id]
        logger.info(f"\nShape of first trial: {first_trial_data.shape}")
        
        # Check metadata
        logger.info(f"Stimulation channel for first trial: {spes_data.trial_stim_channels[first_trial_id]}")
        logger.info(f"Number of selected electrodes: {len(spes_data.trial_electrodes[first_trial_id])}")
        
        # Data validation
        n_missing = spes_data.all_df.isna().sum().sum()
        logger.info(f"Number of missing values: {n_missing}")
        
        # Value ranges
        logger.info(f"Data range: [{spes_data.all_df.values.min():.3f}, {spes_data.all_df.values.max():.3f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing SPESData: {str(e)}")
        raise e

if __name__ == "__main__":
    test_spes_data()
