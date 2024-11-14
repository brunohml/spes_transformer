import os
import pickle
import random
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_file(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single pickle file and create a training example
    
    Args:
        args: Tuple containing (filepath, soz_df)
            filepath: Path to pickle file
            soz_df: DataFrame containing SOZ labels
            
    Returns:
        Tuple of (feature_df, label_df) or (None, None) if file should be skipped
        - feature_df: (487, 36) DataFrame containing channel data
        - label_df: (1, 1) DataFrame containing SOZ label
    """
    filepath, soz_df = args
    
    try:
        # Load pickle file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Extract response matrix and channel names
        response_matrix = data[0]  # (n_channels, seq_len)
        available_channels = data[4]  # channel names array
        n_available = len(available_channels)
        
        # Verify we have enough channels
        if n_available < 36:
            logger.warning(f"File {filepath} has only {n_available} channels. Minimum 36 required. Skipping...")
            return None, None
            
        # Get metadata from filename
        patient_id = os.path.basename(os.path.dirname(filepath))
        parts = os.path.basename(filepath).replace('.pickle', '').split('_')
        stim_pair = parts[1]
        
        # Randomly select exactly 36 channels
        selected_channels = random.sample(list(available_channels), 36)
        channel_indices = [list(available_channels).index(ch) for ch in selected_channels]
        selected_data = response_matrix[channel_indices, :].T  # Shape: (487, 36)
        
        # Get SOZ label
        soz_match = soz_df[(soz_df['Pt'] == patient_id) & 
                          (soz_df['Lead'] == stim_pair)]
        if len(soz_match) == 0:
            logger.warning(f"No SOZ label found for patient {patient_id}, stim_pair {stim_pair}. Skipping...")
            return None, None
        soz_label = int(soz_match.iloc[0]['SOZ'])
        
        # Create DataFrames
        feature_df = pd.DataFrame(
            selected_data,  # Shape: (487, 36)
            columns=[f"channel_{i}" for i in range(36)]
        )
        
        label_df = pd.DataFrame({
            'soz': [soz_label]
        })
        
        return feature_df, label_df
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None, None

def create_split_summary(train_features_df, train_labels_df, test_features_df, test_labels_df, output_dir):
    """Create summary of data split and save to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate statistics
    train_examples = len(train_labels_df)
    test_examples = len(test_labels_df)
    
    train_soz_count = train_labels_df['soz'].sum()
    train_non_soz_count = len(train_labels_df) - train_soz_count
    train_soz_ratio = train_soz_count / len(train_labels_df) if len(train_labels_df) > 0 else 0
    
    test_soz_count = test_labels_df['soz'].sum()
    test_non_soz_count = len(test_labels_df) - test_soz_count
    test_soz_ratio = test_soz_count / len(test_labels_df) if len(test_labels_df) > 0 else 0
    
    # Create summary text
    summary = [
        "SPES Dataset Split Summary",
        "========================\n",
        "Training Set:",
        f"Total trials: {train_examples}",
        f"SOZ trials: {train_soz_count}",
        f"Non-SOZ trials: {train_non_soz_count}",
        f"SOZ ratio: {train_soz_ratio:.2%}",
        f"Sequence length: 487",
        f"Number of electrodes: 36",
        "\nTest Set:",
        f"Total trials: {test_examples}",
        f"SOZ trials: {test_soz_count}",
        f"Non-SOZ trials: {test_non_soz_count}",
        f"SOZ ratio: {test_soz_ratio:.2%}",
        f"Sequence length: 487",
        f"Number of electrodes: 36",
        "\nTotal Dataset:",
        f"Total trials: {train_examples + test_examples}",
        f"Total SOZ trials: {train_soz_count + test_soz_count}",
        f"Total non-SOZ trials: {train_non_soz_count + test_non_soz_count}",
        f"Overall SOZ ratio: {(train_soz_count + test_soz_count)/(train_examples + test_examples):.2%}"
    ]
    
    # Write summary to file
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    try:
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary))
        logger.info(f"Created split summary file at {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save split summary: {str(e)}")

def prepare_spes_data(root_dir: str, soz_labels_path: str, test_patient: str = None) -> None:
    """
    Process all SPES pickle files and create training/test datasets
    
    Args:
        root_dir: Path to directory containing patient subdirectories with pickle files
        soz_labels_path: Path to SOZ labels CSV file
        test_patient: Patient ID to use for test set (e.g., 'Epat26')
    """
    # Load SOZ labels
    soz_df = pd.read_csv(soz_labels_path)
    
    # Get all pickle files and split by patient
    train_files = []
    test_files = []
    
    for patient_dir in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_dir)
        if os.path.isdir(patient_path):
            pickle_files = [
                os.path.join(patient_path, f) 
                for f in os.listdir(patient_path) 
                if f.endswith('.pickle')
            ]
            if patient_dir == test_patient:
                test_files.extend(pickle_files)
            else:
                train_files.extend(pickle_files)
    
    logger.info(f"Found {len(train_files)} training and {len(test_files)} test files")
    
    # Process training files
    logger.info("Processing training files...")
    train_features = []
    train_labels = []
    train_counter = 0
    
    for filepath in tqdm(train_files, desc="Processing training files"):
        feature_df, label_df = process_single_file((filepath, soz_df))
        if feature_df is not None:
            feature_df.index = [train_counter] * 487
            label_df.index = [train_counter]
            train_features.append(feature_df)
            train_labels.append(label_df)
            train_counter += 1
    
    # Process test files
    logger.info("Processing test files...")
    test_features = []
    test_labels = []
    test_counter = 0
    
    for filepath in tqdm(test_files, desc="Processing test files"):
        feature_df, label_df = process_single_file((filepath, soz_df))
        if feature_df is not None:
            feature_df.index = [test_counter] * 487
            label_df.index = [test_counter]
            test_features.append(feature_df)
            test_labels.append(label_df)
            test_counter += 1
    
    # Combine and save training data
    train_features_df = pd.concat(train_features)
    train_labels_df = pd.concat(train_labels)
    
    train_features_df.to_pickle('processed_data/train_features.pkl')
    train_labels_df.to_pickle('processed_data/train_labels.pkl')
    
    # Combine and save test data
    test_features_df = pd.concat(test_features)
    test_labels_df = pd.concat(test_labels)
    
    test_features_df.to_pickle('processed_data/test_features.pkl')
    test_labels_df.to_pickle('processed_data/test_labels.pkl')
    
    logger.info(f"\nCreated {train_counter} training and {test_counter} test trials")
    logger.info(f"Training features shape: {train_features_df.shape}")
    logger.info(f"Test features shape: {test_features_df.shape}")
    
    # After saving the data files, add:
    os.makedirs('classification_datasets/SPES', exist_ok=True)
    create_split_summary(
        train_features_df, train_labels_df,
        test_features_df, test_labels_df,
        'classification_datasets/SPES'
    )
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    root_dir = "spes_trial_pickles"
    soz_labels_path = "spes_trial_metadata/labels_SOZ.csv"
    test_patient = "Epat26"
    
    prepare_spes_data(root_dir, soz_labels_path, test_patient)
