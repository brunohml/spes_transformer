import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from spes_data_class import SPESData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_patient_summary(spes_data):
    """Save detailed patient-level summary to a text file in run_logs directory"""
    # Create run_logs directory if it doesn't exist
    log_dir = "run_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logger.info(f"Created directory: {log_dir}")
    
    # Create filename with current datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"data_summary_{current_time}.txt")
    
    with open(filename, 'w') as f:
        f.write("="*50 + "\n")
        f.write("SPES DATASET PATIENT-LEVEL SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Get unique patients from metadata
        all_patients = spes_data.metadata_df['patient_id'].unique()
        f.write(f"Total unique patients: {len(all_patients)}\n")
        
        # For each patient
        for patient in sorted(all_patients):
            patient_trials = spes_data.metadata_df[spes_data.metadata_df['patient_id'] == patient]
            unique_stim_pairs = patient_trials['stim_pair'].unique()
            
            f.write(f"\nPatient {patient}:\n")
            f.write(f"• Total trials: {len(patient_trials)}\n")
            f.write(f"• Unique stimulation pairs: {len(unique_stim_pairs)}\n")
            f.write(f"• Trials per stim pair:\n")
            for stim_pair in unique_stim_pairs:
                stim_pair_count = len(patient_trials[patient_trials['stim_pair'] == stim_pair])
                f.write(f"  - {stim_pair}: {stim_pair_count} trials\n")
        
        f.write("\n" + "="*50 + "\n")
    
    logger.info(f"\nPatient-level summary saved to: {filename}")

def print_dataset_summary(spes_data):
    """Print high-level dataset summary to terminal"""
    logger.info("\n" + "="*50)
    logger.info("SPES DATASET SUMMARY")
    logger.info("="*50)
    
    # Main DataFrames Summary
    logger.info("\n=== DataFrame Dimensions ===")
    logger.info("\nall_df (Training Data):")
    logger.info(f"• Shape: {spes_data.all_df.shape} (timesteps × channels)")
    logger.info(f"• Number of trials: {len(spes_data.all_df.index.unique())}")
    logger.info(f"• Timesteps per trial: {spes_data.seq_len}")
    logger.info(f"• Channels per trial: {spes_data.n_electrodes}")
    
    # Training/Test Split
    logger.info("\n=== Training/Test Split ===")
    logger.info(f"Test Patient: {spes_data.test_patient_id}")
    logger.info("\nTraining Set:")
    train_patients = spes_data.metadata_df[spes_data.metadata_df.index.isin(
        [f'trial_{i}' for i in spes_data.train_df.index.unique()]
    )]['patient_id'].unique()
    logger.info(f"• Number of patients: {len(train_patients)}")
    logger.info(f"• Total trials: {len(spes_data.train_df.index.unique())}")
    logger.info(f"• SOZ distribution: {dict(spes_data.train_labels['soz'].value_counts())}")
    
    logger.info("\nTest Set:")
    logger.info(f"• Total trials: {len(spes_data.test_df.index.unique())}")
    logger.info(f"• SOZ distribution: {dict(spes_data.test_labels['soz'].value_counts())}")
    
    # Data Properties
    logger.info("\n=== Data Properties ===")
    logger.info(f"• Sampling rate: {spes_data.sampling_rate} Hz")
    logger.info(f"• Sequence length: {spes_data.seq_len} timesteps")
    logger.info(f"• Feature dimension: {spes_data.feature_dim}")
    logger.info(f"• Data range: [{spes_data.all_df.values.min():.3f}, {spes_data.all_df.values.max():.3f}]")
    
    logger.info("\n" + "="*50)

def test_spes_data():
    """Test the SPESData class functionality"""
    
    # Initialize the data class
    root_dir = "spes_trial_pickles"
    test_patient = "Epat26"
    logger.info(f"\nInitializing SPESData:")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Test patient: {test_patient}")
    
    try:
        spes_data = SPESData(root_dir, test_patient)
        
        # Print summary to terminal
        print_dataset_summary(spes_data)
        
        # Save patient-level summary to file
        save_patient_summary(spes_data)
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing SPESData: {str(e)}")
        raise e

if __name__ == "__main__":
    test_spes_data()
