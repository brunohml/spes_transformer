import os
import pickle
import random
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import argparse
import math

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

def process_single_file(args) -> List[Tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Process a single pickle file and create one or more training examples"""
    filepath, soz_df, n_resamples, global_idx = args
    results = []
    
    try:
        # Load pickle file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        response_matrix = data[2]  # (n_channels, seq_len) 0 = normalized, 2 = unnormalized
        available_channels = data[4]
        
        if len(available_channels) < 36:
            logger.warning(f"File {filepath} has insufficient channels. Skipping...")
            return []
            
        # Get metadata from filepath
        patient_id = os.path.basename(os.path.dirname(filepath)).lower()
        parts = os.path.basename(filepath).replace('.pickle', '').split('_')
        stim_pair = parts[1].replace(' ', '').replace('--', '-')
        
        # Get SOZ label
        soz_match = soz_df[(soz_df['Pt'] == patient_id) & (soz_df['Lead'] == stim_pair)]
        if len(soz_match) == 0:
            return []
        soz_label = int(soz_match.iloc[0]['SOZ'])
        
        # Create n_resamples examples with different channel combinations
        for i in range(n_resamples):
            # Randomly select 36 channels
            selected_channels = random.sample(list(available_channels), 36)
            channel_indices = [list(available_channels).index(ch) for ch in selected_channels]
            selected_data = response_matrix[channel_indices, :].T  # Shape: (487, 36)
            
            # Create unique global index for this example
            example_index = [global_idx + i] * 487  # Use global_idx to ensure uniqueness
            
            feature_df = pd.DataFrame(
                selected_data,
                columns=[f"channel_{i}" for i in range(36)],
                index=example_index
            )
            
            label_df = pd.DataFrame({
                'soz': [soz_label] * 487
            }, index=example_index)
            
            metadata = {
                'patient_id': patient_id,
                'stim_pair': stim_pair,
                'soz': soz_label,
                'n_resamples': n_resamples
            }
            
            results.append((feature_df, label_df, metadata))
            
        return results
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return []

def create_split_summary(train_metadata: List[dict], test_metadata: List[dict], 
                        train_count: int, test_count: int, output_dir: str):
    """Create summary using metadata without modifying DataFrames"""
    
    # Calculate training set statistics
    train_soz_count = sum(1 for m in train_metadata if m['soz'] == 1)
    train_non_soz_count = len(train_metadata) - train_soz_count
    
    # Track SOZ and non-SOZ pairs separately
    train_pairs = set()
    train_soz_pairs = set()
    train_non_soz_pairs = set()
    train_soz_resamples = {}
    train_non_soz_resamples = {}
    
    for m in train_metadata:
        pair = (m['patient_id'], m['stim_pair'])
        train_pairs.add(pair)
        
        if m['soz'] == 1:
            train_soz_pairs.add(pair)
            train_soz_resamples[pair] = m['n_resamples']
        else:
            train_non_soz_pairs.add(pair)
            train_non_soz_resamples[pair] = m['n_resamples']
    
    # Calculate averages
    avg_train_soz_resamples = sum(train_soz_resamples.values()) / len(train_soz_pairs) if train_soz_pairs else 0
    avg_train_non_soz_resamples = sum(train_non_soz_resamples.values()) / len(train_non_soz_pairs) if train_non_soz_pairs else 0
    
    # Similar calculations for test set
    test_soz_count = sum(1 for m in test_metadata if m['soz'] == 1)
    test_non_soz_count = len(test_metadata) - test_soz_count
    
    test_pairs = set()
    test_soz_pairs = set()
    test_non_soz_pairs = set()
    test_soz_resamples = {}
    test_non_soz_resamples = {}
    
    for m in test_metadata:
        pair = (m['patient_id'], m['stim_pair'])
        test_pairs.add(pair)
        
        if m['soz'] == 1:
            test_soz_pairs.add(pair)
            test_soz_resamples[pair] = m['n_resamples']
        else:
            test_non_soz_pairs.add(pair)
            test_non_soz_resamples[pair] = m['n_resamples']
    
    avg_test_soz_resamples = sum(test_soz_resamples.values()) / len(test_soz_pairs) if test_soz_pairs else 0
    avg_test_non_soz_resamples = sum(test_non_soz_resamples.values()) / len(test_non_soz_pairs) if test_non_soz_pairs else 0
    
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
        f"Average resamples per SOZ trial: {avg_train_soz_resamples:.2f}",
        f"Average resamples per non-SOZ trial: {avg_train_non_soz_resamples:.2f}",
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
        f"Average resamples per SOZ trial: {avg_test_soz_resamples:.2f}",
        f"Average resamples per non-SOZ trial: {avg_test_non_soz_resamples:.2f}",
        f"Sequence length: 487",
        f"Number of electrodes: 36",
        "\nTotal Dataset:",
        f"Total trials: {train_count + test_count}",
        f"Total SOZ trials: {train_soz_count + test_soz_count}",
        f"Total non-SOZ trials: {train_non_soz_count + test_non_soz_count}",
        f"Overall SOZ ratio: {(train_soz_count + test_soz_count)/(train_count + test_count):.2%}",
        f"Total unique SOZ stim pairs: {len(train_soz_pairs) + len(test_soz_pairs)}",
        f"Total unique non-SOZ stim pairs: {len(train_non_soz_pairs) + len(test_non_soz_pairs)}",
        f"Overall average trials per stim pair: {(train_count + test_count)/(len(train_pairs) + len(test_pairs)):.2f}",
        f"Overall average resamples per SOZ trial: {(sum(train_soz_resamples.values()) + sum(test_soz_resamples.values())) / (len(train_soz_pairs) + len(test_soz_pairs)):.2f}",
        f"Overall average resamples per non-SOZ trial: {(sum(train_non_soz_resamples.values()) + sum(test_non_soz_resamples.values())) / (len(train_non_soz_pairs) + len(test_non_soz_pairs)):.2f}"
    ]
    
    # Write summary to file
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    try:
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary))
        logger.info(f"Created split summary file at {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save split summary: {str(e)}")

def prepare_spes_data(root_dir: str, soz_labels_path: str, test_patient: str = None,
                     target_soz_ratio: float = 0.3) -> None:
    """Process all SPES pickle files and create training/test datasets"""
    # Create output directory with SOZ ratio subdirectory
    output_dir = os.path.join('classification_datasets/SPES', f'soz_{target_soz_ratio}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SOZ labels and normalize format
    soz_df = pd.read_csv(soz_labels_path)
    soz_df['Lead'] = soz_df['Lead'].str.replace(' ', '').str.replace('--', '-')
    soz_df['Pt'] = soz_df['Pt'].str.lower()
    
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
    train_file_metadata = []
    for idx, filepath in enumerate(train_files):
        # Get metadata without processing the file
        patient_id = os.path.basename(os.path.dirname(filepath)).lower()
        parts = os.path.basename(filepath).replace('.pickle', '').split('_')
        stim_pair = parts[1].replace(' ', '').replace('--', '-')
        
        soz_match = soz_df[(soz_df['Pt'] == patient_id) & (soz_df['Lead'] == stim_pair)]
        if len(soz_match) > 0:
            metadata = {
                'patient_id': patient_id,
                'stim_pair': stim_pair,
                'soz': int(soz_match.iloc[0]['SOZ'])
            }
            train_file_metadata.append((idx, metadata))

    # Calculate training resampling counts
    train_resampling_counts = calculate_resampling_counts(train_file_metadata, target_soz_ratio)

    # Process test files similarly
    logger.info("Processing test files...")
    test_file_metadata = []
    for idx, filepath in enumerate(test_files):
        patient_id = os.path.basename(os.path.dirname(filepath)).lower()
        parts = os.path.basename(filepath).replace('.pickle', '').split('_')
        stim_pair = parts[1].replace(' ', '').replace('--', '-')
        
        soz_match = soz_df[(soz_df['Pt'] == patient_id) & (soz_df['Lead'] == stim_pair)]
        if len(soz_match) > 0:
            metadata = {
                'patient_id': patient_id,
                'stim_pair': stim_pair,
                'soz': int(soz_match.iloc[0]['SOZ'])
            }
            test_file_metadata.append((idx, metadata))

    # Calculate test resampling counts
    test_resampling_counts = calculate_resampling_counts(test_file_metadata, target_soz_ratio)

    # Now process files with correct resampling counts
    train_results = []
    train_metadata = []
    global_idx = 0  # Initialize global index counter
    
    for idx, filepath in enumerate(tqdm(train_files, desc="Processing training files")):
        n_resamples = train_resampling_counts.get(idx, 1)
        results = process_single_file((filepath, soz_df, n_resamples, global_idx))
        if results:
            train_results.extend(results)
            for _, _, metadata in results:
                train_metadata.append(metadata)
            global_idx += n_resamples  # Increment global_idx by number of examples created
    
    test_results = []
    test_metadata = []
    # Reset global_idx for test set to keep indices separate
    global_idx = 0
    
    for idx, filepath in enumerate(tqdm(test_files, desc="Processing test files")):
        n_resamples = test_resampling_counts.get(idx, 1)
        results = process_single_file((filepath, soz_df, n_resamples, global_idx))
        if results:
            test_results.extend(results)
            for _, _, metadata in results:
                test_metadata.append(metadata)
            global_idx += n_resamples
    
    # Combine data and save
    train_features_df = pd.concat([feature_df for feature_df, _, _ in train_results])
    train_labels_df = pd.concat([label_df for _, label_df, _ in train_results])
    test_features_df = pd.concat([feature_df for feature_df, _, _ in test_results])
    test_labels_df = pd.concat([label_df for _, label_df, _ in test_results])
    
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
    
    logger.info(f"\nCreated {len(train_results)} training and {len(test_results)} test trials")
    logger.info(f"Training features shape: {train_features_df.shape}")
    logger.info(f"Training labels shape: {train_labels_df.shape}")
    logger.info(f"Test features shape: {test_features_df.shape}")
    logger.info(f"Test labels shape: {test_labels_df.shape}")
    
    # Create split summary with metadata
    create_split_summary(
        train_metadata, test_metadata,
        len(train_results), len(test_results),
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
        # Only show summary, not individual mismatches
        logger.error(f"Found {len(mismatches)} label mismatches!")
        raise ValueError("Label verification failed!")
    else:
        logger.info("✓ All labels verified successfully!")

def calculate_resampling_counts(file_metadata: List[Tuple[int, dict]], target_ratio: float) -> Dict[int, int]:
    """Calculate how many times to resample each file to achieve target SOZ ratio"""
    soz_files = [(idx, meta) for idx, meta in file_metadata if meta['soz'] == 1]
    non_soz_files = [(idx, meta) for idx, meta in file_metadata if meta['soz'] == 0]
    
    n_soz = len(soz_files)
    n_non_soz = len(non_soz_files)
    
    if n_soz == 0:
        return {idx: 1 for idx, _ in non_soz_files}
    
    # Calculate SOZ resampling factor to achieve target ratio
    soz_factor = math.ceil((target_ratio * n_non_soz) / (n_soz * (1 - target_ratio)))
    
    resampling_counts = {}
    # SOZ trials get resampled multiple times
    for idx, _ in soz_files:
        resampling_counts[idx] = soz_factor
    # Non-SOZ trials only get processed once
    for idx, _ in non_soz_files:
        resampling_counts[idx] = 1
        
    return resampling_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare balanced SPES dataset')
    parser.add_argument('--loov', type=str, required=True,
                      help='Patient ID to use for test set (Leave One Out Validation)')
    parser.add_argument('--soz_ratio', type=float, default=0.3,
                      help='Target ratio of SOZ samples (default: 0.3)')
    parser.add_argument('--data_dir', type=str, default='spes_trial_pickles',
                      help='Directory containing SPES pickle files')
    parser.add_argument('--labels_path', type=str, 
                      default='spes_trial_metadata/labels_SOZ.csv',
                      help='Path to SOZ labels CSV file')
    args = parser.parse_args()
    
    # Count SOZ labels in CSV
    soz_count, non_soz_count = count_soz_labels(args.labels_path)
    logger.info("\nSOZ Labels in CSV file:")
    logger.info(f"SOZ stim pairs (1): {soz_count}")
    logger.info(f"Non-SOZ stim pairs (0): {non_soz_count}")
    logger.info(f"Total stim pairs: {soz_count + non_soz_count}")
    logger.info("-" * 50)
    
    # Process data with target ratio
    prepare_spes_data(args.data_dir, args.labels_path, 
                     test_patient=args.loov,
                     target_soz_ratio=args.soz_ratio)
