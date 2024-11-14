import os
import glob
import random
import numpy as np
import pandas as pd
import pickle
import argparse
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Set up logging
def setup_logging(output_dir):
    """Configure logging to both file and console"""
    log_file = os.path.join(output_dir, 'data_preparation.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('SPES_prep')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def count_soz_trials(filepath, soz_df):
    """
    Count if a trial is SOZ or non-SOZ without full processing
    Returns: bool indicating if trial is SOZ
    """
    try:
        patient_id = os.path.basename(os.path.dirname(filepath)).replace('patient_', '')
        parts = os.path.basename(filepath).replace('.pickle', '').split('_')
        stim_pair = parts[1]
        
        soz_match = soz_df[(soz_df['Pt'] == patient_id) & 
                          (soz_df['Lead'] == stim_pair)]
        return bool(int(soz_match.iloc[0]['SOZ'])) if len(soz_match) > 0 else False
    except:
        return False

def process_single_file(args) -> List[pd.DataFrame]:
    """Process a single pickle file and create multiple training examples"""
    filepath, soz_df, n_electrodes, soz_augs, non_soz_augs, start_example_idx = args
    training_examples = []
    
    try:
        # Load and extract data as before
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Extract response matrix and metadata
        response_matrix = data[0] if isinstance(data, list) else data['normalized_seeg']
        available_channels = data[2] if isinstance(data, list) else data['chan']
        if isinstance(available_channels, np.ndarray):
            available_channels = available_channels.tolist()
            
        # Get metadata and SOZ label
        patient_id = os.path.basename(os.path.dirname(filepath)).replace('patient_', '')
        parts = os.path.basename(filepath).replace('.pickle', '').split('_')
        stim_pair = parts[1]
        
        # Get SOZ label
        soz_match = soz_df[(soz_df['Pt'] == patient_id) & 
                          (soz_df['Lead'] == stim_pair)]
        soz_label = int(soz_match.iloc[0]['SOZ']) if len(soz_match) > 0 else 0
        
        # Determine number of training examples to create
        n_examples = soz_augs if soz_label else non_soz_augs
        
        # Create each training example with unique index
        for example_idx in range(start_example_idx, start_example_idx + n_examples):
            # Select channels once for this example
            selected_channels = random.sample(list(available_channels), n_electrodes)
            channel_indices = [list(available_channels).index(ch) for ch in selected_channels]
            
            # Extract data for selected channels
            selected_data = response_matrix[channel_indices, :]  # Shape: (36, 487)
            
            # Create DataFrame with sequential index
            example_df = pd.DataFrame(
                selected_data.T,  # Transpose to get (487, 36)
                columns=[f"channel_{i}" for i in range(n_electrodes)]
            )
            
            # Add the same index value for all 487 timepoints
            example_df.index = [example_idx] * 487
            
            # Create single row for labels DataFrame
            label_df = pd.DataFrame({
                'soz': [soz_label]
            }, index=[example_idx])
            
            training_examples.append((example_df, label_df))
            
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        return []
    
    return training_examples

def process_patient_data(root_dir: str, exclude_patient: str = None, 
                        test_patient: str = None, n_electrodes: int = 36,
                        base_resamples: int = 10, target_ratio: float = 0.3, 
                        test_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process SPES data targeting specific SOZ ratio
    Returns:
        Tuple of (feature_df, labels_df)
    """
    logger = logging.getLogger('SPES_prep')
    
    # Load SOZ labels
    logger.info("Loading SOZ labels...")
    soz_df = pd.read_csv('soz_labels.csv')
    logger.info(f"Loaded SOZ labels for {len(soz_df)} electrodes")
    
    # Scan for pickle files
    logger.info("Scanning for pickle files...")
    pickle_files = []
    with tqdm(os.listdir(root_dir), desc="Scanning patient directories") as pbar:
        for patient_dir in pbar:
            if not patient_dir.startswith('patient_'):
                continue
            if exclude_patient and exclude_patient in patient_dir:
                continue
            if test_patient and test_patient not in patient_dir:
                continue
            
            patient_path = os.path.join(root_dir, patient_dir)
            if os.path.isdir(patient_path):
                pickle_files.extend(glob.glob(os.path.join(patient_path, '*.pickle')))
    
    logger.info(f"Found {len(pickle_files)} pickle files to process")
    
    # Count SOZ and non-SOZ trials
    logger.info("Counting SOZ and non-SOZ trials...")
    with ProcessPoolExecutor() as executor:
        soz_counts = list(tqdm(
            executor.map(count_soz_trials, pickle_files, repeat(soz_df)),
            total=len(pickle_files),
            desc="Counting SOZ trials"
        ))
    
    n_soz_files = sum(soz_counts)
    n_non_soz_files = len(pickle_files) - n_soz_files
    logger.info(f"Found {n_soz_files} SOZ and {n_non_soz_files} non-SOZ files")
    
    # Calculate augmentations to achieve target ratio
    soz_augs = base_resamples
    target_soz = int(n_non_soz_files * target_ratio / (1 - target_ratio))
    non_soz_augs = max(1, min(3, base_resamples * n_soz_files // n_non_soz_files))
    
    logger.info(f"Will create {soz_augs} augmentations per SOZ trial")
    logger.info(f"Will create {non_soz_augs} augmentations per non-SOZ trial")
    
    # Calculate expected number of training examples
    expected_soz_examples = n_soz_files * soz_augs
    expected_non_soz_examples = n_non_soz_files * non_soz_augs
    total_expected_examples = expected_soz_examples + expected_non_soz_examples
    
    logger.info("\nExpected training examples breakdown:")
    logger.info(f"SOZ examples: {n_soz_files} files × {soz_augs} augmentations = {expected_soz_examples}")
    logger.info(f"Non-SOZ examples: {n_non_soz_files} files × {non_soz_augs} augmentations = {expected_non_soz_examples}")
    logger.info(f"Total expected examples: {total_expected_examples}")
    
    # Prepare arguments for parallel processing
    process_args = []
    current_idx = 0
    for filepath, is_soz in zip(pickle_files, soz_counts):
        n_augs = soz_augs if is_soz else non_soz_augs
        process_args.append((filepath, soz_df, n_electrodes, soz_augs, non_soz_augs, current_idx))
        current_idx += n_augs
    
    # Process files in parallel
    logger.info("\nProcessing files in parallel...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        all_examples = list(tqdm(
            executor.map(process_single_file, process_args),
            total=len(pickle_files),
            desc="Processing files"
        ))
    
    # Separate features and labels
    feature_dfs = []
    label_dfs = []
    
    # Flatten the list of examples and separate features and labels
    for example_list in all_examples:
        for feature_df, label_df in example_list:
            feature_dfs.append(feature_df)
            label_dfs.append(label_df)
    
    # Log pre-concatenation stats
    logger.info("\nPre-concatenation statistics:")
    logger.info(f"Number of feature DataFrames: {len(feature_dfs)}")
    logger.info(f"Number of label DataFrames: {len(label_dfs)}")
    
    # Concatenate all features and labels
    final_feature_df = pd.concat(feature_dfs, axis=0)
    final_label_df = pd.concat(label_dfs, axis=0)
    
    # Log index statistics before reindexing
    unique_indices_before = final_feature_df.index.unique()
    logger.info("\nIndex statistics before reindexing:")
    logger.info(f"Number of unique indices: {len(unique_indices_before)}")
    logger.info(f"Index range: {unique_indices_before.min()} to {unique_indices_before.max()}")
    
    # Reset indices to ensure they're sequential
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices_before)}
    
    final_feature_df.index = final_feature_df.index.map(index_map)
    final_label_df.index = final_label_df.index.map(index_map)
    
    # Verify final structure
    unique_indices_after = final_feature_df.index.unique()
    timepoints_per_example = final_feature_df.index.value_counts()
    
    logger.info("\nFinal data structure verification:")
    logger.info(f"Expected number of training examples: {total_expected_examples}")
    logger.info(f"Actual number of unique indices: {len(unique_indices_after)}")
    logger.info(f"Number of examples with exactly 487 timepoints: {sum(timepoints_per_example == 487)}")
    logger.info(f"Examples with wrong number of timepoints: {sum(timepoints_per_example != 487)}")
    logger.info(f"Final feature DataFrame shape: {final_feature_df.shape}")
    logger.info(f"Final label DataFrame shape: {final_label_df.shape}")
    
    # Verify index alignment
    index_match = set(final_feature_df.index.unique()) == set(final_label_df.index)
    logger.info("\nIndex alignment check:")
    logger.info(f"Feature and label indices match: {index_match}")
    
    if len(unique_indices_after) != total_expected_examples:
        logger.warning(f"WARNING: Number of unique indices ({len(unique_indices_after)}) "
                      f"does not match expected number of examples ({total_expected_examples})")
    
    return final_feature_df, final_label_df

def plot_response_curves(df, example_id, output_path, soz_status):
    """Plot channel responses for a single example
    
    Args:
        df: DataFrame containing the data
        example_id: ID of the example to plot
        output_path: Where to save the plot
        soz_status: String indicating if this is 'SOZ' or 'Non-SOZ'
    """
    logger = logging.getLogger('SPES_prep')
    
    try:
        # Get data for this example (first augmentation only)
        example_data = df.loc[example_id].iloc[:487]  # Get first 487 timepoints
        channel_cols = [col for col in df.columns if col.startswith('channel_')]
        
        plt.figure(figsize=(12, 8))
        for channel in channel_cols:
            plt.plot(example_data[channel], alpha=0.5, linewidth=1)
            
        plt.title(f'Channel Responses for {soz_status} Example')
        plt.xlabel('Time Point')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {soz_status} channel plot to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create {soz_status} channel plot: {str(e)}")
        plt.close()  # Ensure figure is closed even if plotting fails

def create_split_summary(train_df, test_df, output_dir):
    """Create summary of data split"""
    logger = logging.getLogger('SPES_prep')
    
    # Create channel visualization plots
    try:
        # Get one SOZ and one non-SOZ example from training set
        soz_example = train_df[train_df['soz'] == 1].index.unique()[0]
        non_soz_example = train_df[train_df['soz'] == 0].index.unique()[0]
        
        # Create only the channel response plots
        plot_response_curves(train_df, soz_example, 
                           os.path.join(output_dir, 'soz_channels.png'), 
                           'SOZ')
        plot_response_curves(train_df, non_soz_example, 
                           os.path.join(output_dir, 'non_soz_channels.png'), 
                           'Non-SOZ')
    except Exception as e:
        logger.error(f"Failed to create channel visualization plots: {str(e)}")
    
    # Rest of the function remains unchanged for text summary
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    
    # Original data statistics
    original_train_groups = train_df.groupby(['patient_id', 'stim_pair', 'trial_number']).first()
    original_test_groups = test_df.groupby(['patient_id', 'stim_pair', 'trial_number']).first()
    
    # Calculate statistics and create summary text
    train_stim_pairs = original_train_groups.groupby('stim_pair').size()
    train_patients = original_train_groups.groupby('patient_id').size()
    train_soz_counts = original_train_groups['soz'].value_counts()
    
    test_stim_pairs = original_test_groups.groupby('stim_pair').size()
    test_patients = original_test_groups.groupby('patient_id').size()
    test_soz_counts = original_test_groups['soz'].value_counts()
    
    # Augmented data statistics
    augmented_train_groups = train_df.groupby(['patient_id', 'stim_pair', 'trial_number', 'augmentation'])
    augmented_test_groups = test_df.groupby(['patient_id', 'stim_pair', 'trial_number', 'augmentation'])
    
    # Calculate resample counts
    train_resample_counts = train_df.groupby(['stim_pair', 'trial_number'])['augmentation'].nunique()
    test_resample_counts = test_df.groupby(['stim_pair', 'trial_number'])['augmentation'].nunique()
    
    # Calculate augmented SOZ ratios
    train_aug_soz_ratio = (train_df['soz'] == 1).mean()
    test_aug_soz_ratio = (test_df['soz'] == 1).mean()
    total_aug_soz_ratio = (pd.concat([train_df, test_df])['soz'] == 1).mean()
    
    # Create summary text with both original and augmented statistics
    summary = [
        "SPES Dataset Split Summary",
        "========================\n",
        "Training Set:",
        f"Total original trials: {len(original_train_groups)}",
        f"Original SOZ trials: {train_soz_counts.get(1, 0)}",
        f"Original non-SOZ trials: {train_soz_counts.get(0, 0)}",
        f"Original SOZ ratio: {(train_soz_counts.get(1, 0)/len(original_train_groups) if len(original_train_groups) > 0 else 0):.2%}",
        f"Original trials per patient: {len(original_train_groups)/len(train_patients):.1f}",
        f"Original trials per stim pair: {len(original_train_groups)/len(train_stim_pairs):.1f}",
        "\nAugmented Data:",
        f"Total augmented trials: {len(augmented_train_groups)}",
        f"Augmented SOZ ratio: {train_aug_soz_ratio:.2%}",
        f"Average times trials were resampled: {train_resample_counts.mean():.1f}",
        f"Number of patients: {len(train_patients)}",
        f"Number of unique stim pairs: {len(train_stim_pairs)}",
        f"Augmented trials per patient: {len(augmented_train_groups)/len(train_patients):.1f}",
        f"Augmented trials per stim pair: {len(augmented_train_groups)/len(train_stim_pairs):.1f}",
        f"Patient IDs: {', '.join(sorted(train_df['patient_id'].unique()))}",
        f"Sequence length: 487",
        f"Number of electrodes: {sum(1 for col in train_df.columns if col.startswith('channel_'))}",
        "\nValidation Set:",
        "Original Data:",
        f"Total original trials: {len(original_test_groups)}",
        f"Original SOZ trials: {test_soz_counts.get(1, 0)}",
        f"Original non-SOZ trials: {test_soz_counts.get(0, 0)}",
        f"Original SOZ ratio: {(test_soz_counts.get(1, 0)/len(original_test_groups) if len(original_test_groups) > 0 else 0):.2%}",
        f"Original trials per patient: {len(original_test_groups)/len(test_patients):.1f}",
        f"Original trials per stim pair: {len(original_test_groups)/len(test_stim_pairs):.1f}",
        "\nAugmented Data:",
        f"Total augmented trials: {len(augmented_test_groups)}",
        f"Augmented SOZ ratio: {test_aug_soz_ratio:.2%}",
        f"Average times trials were resampled: {test_resample_counts.mean():.1f}",
        f"Number of patients: {len(test_patients)}",
        f"Number of unique stim pairs: {len(test_stim_pairs)}",
        f"Augmented trials per patient: {len(augmented_test_groups)/len(test_patients):.1f}",
        f"Augmented trials per stim pair: {len(augmented_test_groups)/len(test_stim_pairs):.1f}",
        f"Patient IDs: {', '.join(sorted(test_df['patient_id'].unique()))}",
        f"Sequence length: 487",
        f"Number of electrodes: {sum(1 for col in test_df.columns if col.startswith('channel_'))}\n",
        "\nTotal Dataset:",
        "Original Data:",
        f"Total original trials: {len(original_train_groups) + len(original_test_groups)}",
        f"Total original SOZ trials: {train_soz_counts.get(1, 0) + test_soz_counts.get(1, 0)}",
        f"Total original non-SOZ trials: {train_soz_counts.get(0, 0) + test_soz_counts.get(0, 0)}",
        f"Overall original SOZ ratio: {((train_soz_counts.get(1, 0) + test_soz_counts.get(1, 0))/(len(original_train_groups) + len(original_test_groups)) if (len(original_train_groups) + len(original_test_groups)) > 0 else 0):.2%}",
        "\nAugmented Data:",
        f"Total augmented trials: {len(augmented_train_groups) + len(augmented_test_groups)}",
        f"Overall augmented SOZ ratio: {total_aug_soz_ratio:.2%}",
        f"Total unique stim pairs: {len(train_stim_pairs) + len(test_stim_pairs)}",
        f"Total unique patients: {len(train_patients) + len(test_patients)}",
        f"Overall average times trials were resampled: {pd.concat([train_resample_counts, test_resample_counts]).mean():.1f}"
    ]
    
    # Write summary to file
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    try:
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary))
        logger.info(f"Created split summary file at {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save split summary: {str(e)}")

def save_spes_data(feature_df: pd.DataFrame, labels_df: pd.DataFrame, output_path: str):
    """Save SPES data in format expected by SPESData class"""
    logger = logging.getLogger('SPES_prep')
    
    try:
        # Verify the structure
        n_examples = len(labels_df)
        expected_shape = (n_examples * 487, 36)  # feature_df shape
        
        logger.info("\nVerifying data structure:")
        logger.info(f"Number of training examples: {n_examples}")
        logger.info(f"Feature DataFrame shape: {feature_df.shape} (expected {expected_shape})")
        logger.info(f"Labels DataFrame shape: {labels_df.shape} (expected ({n_examples}, 1))")
        
        # Verification checks
        assert feature_df.shape == expected_shape, f"Wrong feature shape: {feature_df.shape}"
        assert len(feature_df.index.unique()) == n_examples, "Wrong number of unique indices"
        assert all(feature_df.index.value_counts() == 487), "Not all examples have 487 timepoints"
        assert labels_df.shape == (n_examples, 1), f"Wrong labels shape: {labels_df.shape}"
        assert set(feature_df.index.unique()) == set(labels_df.index), "Index mismatch between features and labels"
        
        # Save to CSV
        feature_df.to_csv(f"{output_path}_features.csv")
        labels_df.to_csv(f"{output_path}_labels.csv")
        logger.info(f"Saved data to {output_path}_features.csv and {output_path}_labels.csv")
        
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise

def load_spes_data(csv_path: str) -> Dict[str, np.ndarray]:
    """Load SPES data and organize into training examples
    
    Returns:
        Dict mapping training_example_id to numpy array of shape (487, 36)
    """
    df = pd.read_csv(csv_path)
    
    # Group by training example and augmentation
    examples = {}
    for (te_id, aug_num), group in df.groupby(['training_example_id', 'augmentation_number']):
        # Sort by sequence_idx to ensure correct order
        group = group.sort_values('sequence_idx')
        
        # Extract channel data into numpy array (487, 36)
        channel_cols = [col for col in group.columns if col.startswith('channel_')]
        data = group[channel_cols].values
        
        example_key = f"{te_id}_aug{aug_num}"
        examples[example_key] = {
            'data': data,
            'soz': group['soz'].iloc[0],  # Same for all rows
            'patient_id': group['patient_id'].iloc[0],
            'stim_pair': group['stim_pair'].iloc[0]
        }
    
    return examples

def main():
    parser = argparse.ArgumentParser(description='Prepare SPES data for training')
    parser.add_argument('--loov', type=str, required=True,
                      help='Patient ID to use for test set (Leave One Out Validation)')
    parser.add_argument('--data_dir', type=str, default='spes_trial_pickles',
                      help='Directory containing SPES pickle files')
    parser.add_argument('--output_dir', type=str, default='classification_datasets/SPES',
                      help='Output directory for processed CSV files')
    parser.add_argument('--test', action='store_true',
                      help='Process only first patient for testing')
    parser.add_argument('--base-resamples', type=int, default=10,
                      help='Base number of resamples for SOZ training examples')
    parser.add_argument('--target-ratio', type=float, default=0.3,
                      help='Target ratio of SOZ to non-SOZ training examples (default: 0.3)')
    args = parser.parse_args()
    
    # Create output directory and setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("Starting SPES data preparation")
    logger.info(f"Arguments: {vars(args)}")
    
    if args.test:
        logger.info("Running in test mode - will only process first patient")
    
    try:
        # Process training data
        logger.info("Processing training data...")
        train_features_df, train_labels_df = process_patient_data(
            args.data_dir, 
            exclude_patient=args.loov,
            base_resamples=args.base_resamples,
            target_ratio=args.target_ratio,
            test_mode=args.test
        )
        train_path = os.path.join(args.output_dir, 'SPES_TRAIN')
        save_spes_data(train_features_df, train_labels_df, train_path)
        logger.info(f"Saved training data to {train_path}")
        
        # Process test data
        logger.info("Processing test data...")
        test_features_df, test_labels_df = process_patient_data(
            args.data_dir, 
            test_patient=args.loov,
            base_resamples=args.base_resamples,
            target_ratio=args.target_ratio,
            test_mode=args.test
        )
        test_path = os.path.join(args.output_dir, 'SPES_TEST')
        save_spes_data(test_features_df, test_labels_df, test_path)
        logger.info(f"Saved test data to {test_path}")
        
        create_split_summary(
            (train_features_df, train_labels_df), 
            (test_features_df, test_labels_df), 
            args.output_dir
        )
        logger.info("Data preparation complete!")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise

if __name__ == '__main__':
    main() 