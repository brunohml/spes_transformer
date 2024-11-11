import os
import pickle
import random
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_response_curves(df: pd.DataFrame, example_id: int, output_path: str, soz_status: str):
    """Create visualization of channel responses for a single example"""
    example_data = df[df.index == example_id]
    
    # Create a figure with vertically stacked subplots
    fig, axes = plt.subplots(36, 1, figsize=(15, 50))
    fig.suptitle(f'Channel Responses for {soz_status} Example', fontsize=16)
    
    # Plot each channel on its own subplot
    for i in range(36):
        ax = axes[i]
        ax.plot(example_data[f'channel_{i}'].values, 'b-', linewidth=1)
        ax.set_ylabel(f'Ch {i}')
        ax.set_xlim(0, 486)  # Set consistent x-axis limits
        
        # Only show x-axis label for bottom plot
        if i == 35:
            ax.set_xlabel('Time')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_single_file(args) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Process a single pickle file and create a training example"""
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
            return None, None, None
            
        # Get metadata from filename and normalize format
        patient_id = os.path.basename(os.path.dirname(filepath)).lower()
        parts = os.path.basename(filepath).replace('.pickle', '').split('_')
        stim_pair = parts[1].replace(' ', '').replace('--', '-')
        
        # Randomly select exactly 36 channels
        selected_channels = random.sample(list(available_channels), 36)
        channel_indices = [list(available_channels).index(ch) for ch in selected_channels]
        selected_data = response_matrix[channel_indices, :].T  # Shape: (487, 36)
        
        # Get SOZ label
        soz_match = soz_df[(soz_df['Pt'] == patient_id) & 
                          (soz_df['Lead'] == stim_pair)]
        if len(soz_match) == 0:
            if patient_id not in missing_by_patient:
                missing_by_patient[patient_id] = set()
            missing_by_patient[patient_id].add(stim_pair)
            return None, None, None
        soz_label = int(soz_match.iloc[0]['SOZ'])
        
        # Create DataFrames (keep these simple for the model)
        feature_df = pd.DataFrame(
            selected_data,  # Shape: (487, 36) 
            columns=[f"channel_{i}" for i in range(36)]
        )
        
        # Create label DataFrame with same label repeated 487 times
        label_df = pd.DataFrame({
            'soz': [soz_label] * 487  # Repeat label for each timepoint
        })
        
        # Return metadata separately
        metadata = {
            'patient_id': patient_id,
            'stim_pair': stim_pair,
            'soz': soz_label
        }
        
        return feature_df, label_df, metadata
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None, None, None

def create_split_summary(train_metadata: List[dict], test_metadata: List[dict], 
                        train_count: int, test_count: int, output_dir: str):
    """Create summary using metadata without modifying DataFrames"""
    
    # Calculate training set statistics
    train_soz_count = sum(1 for m in train_metadata if m['soz'] == 1)
    train_non_soz_count = len(train_metadata) - train_soz_count
    
    # Count unique stim pairs
    train_pairs = set((m['patient_id'], m['stim_pair']) for m in train_metadata)
    train_soz_pairs = set((m['patient_id'], m['stim_pair']) 
                         for m in train_metadata if m['soz'] == 1)
    train_non_soz_pairs = set((m['patient_id'], m['stim_pair']) 
                             for m in train_metadata if m['soz'] == 0)
    
    # Similar calculations for test set
    test_soz_count = sum(1 for m in test_metadata if m['soz'] == 1)
    test_non_soz_count = len(test_metadata) - test_soz_count
    
    test_pairs = set((m['patient_id'], m['stim_pair']) for m in test_metadata)
    test_soz_pairs = set((m['patient_id'], m['stim_pair']) 
                        for m in test_metadata if m['soz'] == 1)
    test_non_soz_pairs = set((m['patient_id'], m['stim_pair']) 
                            for m in test_metadata if m['soz'] == 0)
    
    # Create summary text
    summary = [
        "SPES Dataset Split Summary",
        "========================\n",
        "Training Set:",
        f"Total trials: {train_count}",
        f"SOZ trials: {train_soz_count}",
        f"Non-SOZ trials: {train_non_soz_count}",
        f"SOZ ratio: {train_soz_count / train_count:.2%}",
        f"Unique SOZ stim pairs: {len(train_soz_pairs)}",
        f"Unique non-SOZ stim pairs: {len(train_non_soz_pairs)}",
        f"Average trials per stim pair: {train_count / len(train_pairs):.2f}",
        f"Sequence length: 487",
        f"Number of electrodes: 36",
        "\nTest Set:",
        f"Total trials: {test_count}",
        f"SOZ trials: {test_soz_count}",
        f"Non-SOZ trials: {test_non_soz_count}",
        f"SOZ ratio: {test_soz_count / test_count:.2%}",
        f"Unique SOZ stim pairs: {len(test_soz_pairs)}",
        f"Unique non-SOZ stim pairs: {len(test_non_soz_pairs)}",
        f"Average trials per stim pair: {test_count / len(test_pairs):.2f}",
        f"Sequence length: 487",
        f"Number of electrodes: 36",
        "\nTotal Dataset:",
        f"Total trials: {train_count + test_count}",
        f"Total SOZ trials: {train_soz_count + test_soz_count}",
        f"Total non-SOZ trials: {train_non_soz_count + test_non_soz_count}",
        f"Overall SOZ ratio: {(train_soz_count + test_soz_count)/(train_count + test_count):.2%}",
        f"Total unique SOZ stim pairs: {len(train_soz_pairs) + len(test_soz_pairs)}",
        f"Total unique non-SOZ stim pairs: {len(train_non_soz_pairs) + len(test_non_soz_pairs)}",
        f"Overall average trials per stim pair: {(train_count + test_count)/(len(train_pairs) + len(test_pairs)):.2f}"
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
    """Process all SPES pickle files and create training/test datasets"""
    # Create output directory
    output_dir = 'classification_datasets/SPES'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SOZ labels and normalize format
    soz_df = pd.read_csv(soz_labels_path)
    soz_df['Lead'] = soz_df['Lead'].str.replace(' ', '').str.replace('--', '-')
    soz_df['Pt'] = soz_df['Pt'].str.lower()
    
    # Track mismatches by patient
    missing_by_patient = {}
    
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
    
    # Process files in parallel
    logger.info("Processing training files...")
    with ProcessPoolExecutor() as executor:
        train_results = list(tqdm(
            executor.map(process_single_file, zip(train_files, repeat(soz_df))),
            total=len(train_files),
            desc="Processing training files"
        ))
    
    train_features = []
    train_labels = []
    train_metadata = []
    train_counter = 0
    
    for feature_df, label_df, metadata in train_results:
        if feature_df is not None:
            # Use same indices for both features and labels
            indices = [train_counter] * 487
            feature_df.index = indices
            label_df.index = indices
            train_features.append(feature_df)
            train_labels.append(label_df)
            train_metadata.append(metadata)
            train_counter += 1
    
    logger.info("Processing test files...")
    with ProcessPoolExecutor() as executor:
        test_results = list(tqdm(
            executor.map(process_single_file, zip(test_files, repeat(soz_df))),
            total=len(test_files),
            desc="Processing test files"
        ))
    
    test_features = []
    test_labels = []
    test_metadata = []
    test_counter = 0
    
    for feature_df, label_df, metadata in test_results:
        if feature_df is not None:
            # Use same indices for both features and labels
            indices = [test_counter] * 487
            feature_df.index = indices
            label_df.index = indices  # Make sure labels have same indices as features
            test_features.append(feature_df)
            test_labels.append(label_df)
            test_metadata.append(metadata)
            test_counter += 1
    
    # Combine data and save
    train_features_df = pd.concat(train_features)
    train_labels_df = pd.concat(train_labels)
    test_features_df = pd.concat(test_features)
    test_labels_df = pd.concat(test_labels)
    
    # Verify label alignment before saving
    verify_label_alignment(train_labels_df, train_metadata, soz_df, "train")
    verify_label_alignment(test_labels_df, test_metadata, soz_df, "test")
    
    # Create visualizations
    try:
        # Find SOZ and non-SOZ examples
        soz_example = train_features_df[train_labels_df['soz'] == 1].index.unique()[0]
        non_soz_example = train_features_df[train_labels_df['soz'] == 0].index.unique()[0]
        
        # Create plots
        plot_response_curves(train_features_df, soz_example, 
                           os.path.join(output_dir, 'soz_example.png'), 'SOZ')
        plot_response_curves(train_features_df, non_soz_example, 
                           os.path.join(output_dir, 'non_soz_example.png'), 'Non-SOZ')
    except Exception as e:
        logger.error(f"Failed to create visualizations: {str(e)}")
    
    # Save data only if verification passes
    train_features_df.to_csv(os.path.join(output_dir, 'SPES_TRAIN.csv'), index=True)
    train_labels_df.to_csv(os.path.join(output_dir, 'SPES_TRAIN_labels.csv'), index=True)
    test_features_df.to_csv(os.path.join(output_dir, 'SPES_TEST.csv'), index=True)
    test_labels_df.to_csv(os.path.join(output_dir, 'SPES_TEST_labels.csv'), index=True)
    
    logger.info(f"\nCreated {train_counter} training and {test_counter} test trials")
    logger.info(f"Training features shape: {train_features_df.shape}")
    logger.info(f"Training labels shape: {train_labels_df.shape}")
    logger.info(f"Test features shape: {test_features_df.shape}")
    logger.info(f"Test labels shape: {test_labels_df.shape}")
    
    # Create split summary with metadata
    create_split_summary(
        train_metadata, test_metadata,
        train_counter, test_counter,
        output_dir
    )
    
    logger.info("Data preparation complete!")

def count_soz_labels(soz_labels_path: str) -> Tuple[int, int]:
    """Count the number of SOZ and non-SOZ labels in the labels file"""
    soz_df = pd.read_csv(soz_labels_path)
    soz_count = (soz_df['SOZ'] == 1).sum()
    non_soz_count = (soz_df['SOZ'] == 0).sum()
    return soz_count, non_soz_count

def verify_label_alignment(labels_df, metadata_list, original_soz_df, split_name="train"):
    """Verify that processed labels match original metadata"""
    logger.info(f"\nVerifying {split_name} set label alignment...")
    
    mismatches = []
    total_sequences = len(metadata_list)
    verified_count = 0
    
    # Debug prints
    logger.info(f"Labels DataFrame shape: {labels_df.shape}")
    logger.info(f"Number of metadata entries: {len(metadata_list)}")
    
    for idx, metadata in enumerate(metadata_list):
        stim_pair = metadata['stim_pair']
        patient_id = metadata['patient_id']
        
        try:
            # Get label we assigned (get first row of each sequence)
            soz_value = labels_df.iloc[idx * 487]['soz']  # Use iloc instead of loc
            
            # Get expected label from original metadata
            expected_soz = original_soz_df[
                (original_soz_df['Pt'] == patient_id) & 
                (original_soz_df['Lead'] == stim_pair)
            ]['SOZ'].iloc[0]
            
            if int(soz_value) != int(expected_soz):
                mismatches.append({
                    'patient': patient_id,
                    'stim_pair': stim_pair,
                    'assigned_label': int(soz_value),
                    'expected_label': int(expected_soz)
                })
            else:
                verified_count += 1
                
        except Exception as e:
            logger.error(f"Error processing sequence {idx}:")
            logger.error(f"Patient: {patient_id}, Stim pair: {stim_pair}")
            logger.error(f"Error: {str(e)}")
            raise
    
    # Log results
    logger.info(f"Verified {verified_count}/{total_sequences} sequences")
    
    if mismatches:
        logger.error(f"Found {len(mismatches)} label mismatches:")
        for mismatch in mismatches:
            logger.error(
                f"Patient {mismatch['patient']}, "
                f"Stim pair {mismatch['stim_pair']}: "
                f"got {mismatch['assigned_label']}, "
                f"expected {mismatch['expected_label']}"
            )
        raise ValueError("Label verification failed!")
    else:
        logger.info("✓ All labels verified successfully!")

if __name__ == "__main__":
    root_dir = "spes_trial_pickles"
    soz_labels_path = "spes_trial_metadata/labels_SOZ.csv"
    test_patient = "Epat26"
    
    # Count SOZ labels in CSV
    soz_count, non_soz_count = count_soz_labels(soz_labels_path)
    logger.info("\nSOZ Labels in CSV file:")
    logger.info(f"SOZ stim pairs (1): {soz_count}")
    logger.info(f"Non-SOZ stim pairs (0): {non_soz_count}")
    logger.info(f"Total stim pairs: {soz_count + non_soz_count}")
    logger.info("-" * 50)
    
    # Process data
    prepare_spes_data(root_dir, soz_labels_path, test_patient)
