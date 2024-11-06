# Import necessary libraries
import os                  # For file path operations
import glob               # For file pattern matching
import random            # For random channel selection
import numpy as np       # For numerical operations
import pandas as pd      # For DataFrame handling
import pickle           # For reading pickle files
from typing import Dict, List, Tuple  # For type hints
from src.datasets.data import BaseData


class SPESData(BaseData):
    """
    Dataset class for SPES (Single Pulse Electrical Stimulation) dataset.
    Includes functionality for handling SOZ (Seizure Onset Zone) labels.
    """
    def __init__(self, root_dir: str, n_electrodes: int = 36):
        """
        Initialize the SPES dataset.
        
        Args:
            root_dir (str): Directory containing SPES pickle files
            n_electrodes (int): Number of electrodes to select per trial
        """
        # Set sampling rate and sequence length constants
        self.sampling_rate = 512  # Sampling frequency in Hz
        self.seq_len = 487       # Number of time points per sequence
        self.n_electrodes = n_electrodes  # Number of electrodes to select per trial
        
        # Initialize dictionaries to store metadata
        self.trial_electrodes = {}     # Maps trial_id to list of selected electrodes
        self.trial_stim_channels = {}  # Maps trial_id to stimulation channel
        self.trial_soz_labels = {}     # NEW: Maps trial_id to SOZ label
        
        # Load all data and create main dataframe
        self.all_df = self._load_all_data(root_dir)
        # Extract unique trial IDs from the multi-index
        self.all_IDs = self.all_df.index.get_level_values(0).unique()
        
        # Set feature dataframe (same as all_df in this case)
        self.feature_df = self.all_df
        # Get list of column names (channel names)
        self.feature_names = self.all_df.columns.tolist()

    def _load_all_data(self, root_dir: str) -> pd.DataFrame:
        """
        Load all pickle files and create combined dataframe with SOZ labels.
        
        Args:
            root_dir (str): Directory containing SPES pickle files
            
        Returns:
            pd.DataFrame: Combined DataFrame with trial data and SOZ labels
        """
        # Load SOZ labels from metadata
        try:
            soz_df = pd.read_csv(os.path.join('spes_trial_metadata', 'labels_SOZ.csv'))
            print(f"Loaded SOZ labels for {len(soz_df)} electrodes")
            print("\nFirst few rows of SOZ labels:")
            print(soz_df.head())
            print("\nUnique patients:", sorted(soz_df['Pt'].unique()))
            print("Sample leads:", sorted(soz_df['Lead'].unique())[:5])
        except Exception as e:
            raise ValueError(f"Error loading SOZ labels: {str(e)}")
        
        all_trials = []      # List to store DataFrames for each trial
        trial_counter = 0    # Counter to generate unique trial IDs
        
        # Check if directory exists
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
        
        # Get list of pickle files
        pickle_files = glob.glob(os.path.join(root_dir, "*.pickle"))
        if not pickle_files:
            raise ValueError(f"No pickle files found in {root_dir}")
        
        print(f"Found {len(pickle_files)} pickle files")  # Debug print
        
        for filepath in pickle_files:
            print(f"\nProcessing file: {os.path.basename(filepath)}")  # Debug print
            
            # Load pickle file contents
            try:
                with open(filepath, 'rb') as f:
                    trial_data = pickle.load(f)
                print(f"Type of loaded data: {type(trial_data)}")
                print(f"Length of loaded data: {len(trial_data)}")
                
                if isinstance(trial_data, list):
                    # Print types without showing full arrays
                    type_info = [f"{type(item).__name__}{item.shape if isinstance(item, np.ndarray) else ''}" 
                               for item in trial_data]
                    print(f"Data elements: {type_info}")
                    
                    # Assuming first array is response matrix and third is channel names
                    response_matrix = trial_data[0]
                    available_channels = trial_data[2]
                    
                    # Convert available_channels to list if it's a numpy array
                    if isinstance(available_channels, np.ndarray):
                        available_channels = available_channels.tolist()
                    
                    print(f"Response matrix shape: {response_matrix.shape}")
                    
                else:
                    response_matrix = trial_data['normalized_seeg']
                    available_channels = trial_data['chan']
                    print(f"Response matrix shape: {response_matrix.shape}")
                
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                continue
                
            # Parse filename to extract metadata
            filename = os.path.basename(filepath)
            try:
                # Split on underscores and handle the special case of electrode pairs
                parts = filename.replace('.pickle', '').split('_')
                patient_id = parts[0]  # e.g., 'Epat26'
                stim_pair = parts[1]   # e.g., 'LAC5-LAC6'
                
                # First, handle channel selection
                if len(available_channels) < self.n_electrodes:
                    print(f"Warning: Not enough channels in {filename}. Skipping...")
                    continue
                    
                selected_channels = random.sample(list(available_channels), self.n_electrodes)
                channel_indices = [list(available_channels).index(ch) for ch in selected_channels]
                selected_data = response_matrix[channel_indices, :].T
                
                # Then, look up SOZ label using the full bipolar pair
                soz_match = soz_df[(soz_df['Pt'] == patient_id) & 
                                 (soz_df['Lead'] == stim_pair)]
                
                if len(soz_match) == 0:
                    print(f"Warning: No SOZ label found for patient {patient_id}, electrode pair {stim_pair}")
                    # Debug print to help understand the matching issue
                    matching_patient = soz_df[soz_df['Pt'] == patient_id]
                    if not matching_patient.empty:
                        print(f"Available electrode pairs for {patient_id}:")
                        print(matching_patient['Lead'].unique()[:5], "...")  # Show first 5 pairs
                    soz_label = 0  # Default to non-SOZ if not found
                else:
                    soz_label = int(soz_match.iloc[0]['SOZ'])  # Ensure integer type
                    print(f"Found SOZ label {soz_label} for {patient_id}, {stim_pair}")
                
                # Create trial ID and store metadata
                trial_id = f"trial_{trial_counter}"
                self.trial_soz_labels[trial_id] = soz_label
                self.trial_electrodes[trial_id] = selected_channels
                self.trial_stim_channels[trial_id] = stim_pair
                
                # Create DataFrame with 3-level MultiIndex
                trial_df = pd.DataFrame(
                    selected_data,
                    columns=[f"channel_{i}" for i in range(self.n_electrodes)],
                    index=pd.MultiIndex.from_product(
                        [[trial_id], [soz_label], range(self.seq_len)],
                        names=['trial_id', 'soz', 'timestep']
                    )
                )
                
                all_trials.append(trial_df)
                trial_counter += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        if not all_trials:
            raise ValueError("No valid trials were processed")
            
        return pd.concat(all_trials)

    # NEW: Methods for querying SOZ-related information
    def get_soz_trials(self) -> pd.DataFrame:
        """
        Get all trials where stimulation was in SOZ.
        
        Returns:
            pd.DataFrame: DataFrame containing only SOZ trials
        """
        return self.all_df.xs(1, level='soz')
    
    def get_non_soz_trials(self) -> pd.DataFrame:
        """
        Get all trials where stimulation was not in SOZ.
        
        Returns:
            pd.DataFrame: DataFrame containing only non-SOZ trials
        """
        return self.all_df.xs(0, level='soz')
    
    def get_trial_soz_label(self, trial_id: str) -> int:
        """
        Get SOZ label for a specific trial.
        
        Args:
            trial_id (str): Trial identifier
            
        Returns:
            int: SOZ label (0 or 1)
        """
        if trial_id not in self.trial_soz_labels:
            raise ValueError(f"Trial {trial_id} not found")
        return self.trial_soz_labels[trial_id]
    
    def print_soz_summary(self):
        """
        Print summary of SOZ labels in the dataset.
        """
        soz_counts = pd.Series(self.trial_soz_labels).value_counts()
        print("\nSOZ Label Summary:")
        print(f"SOZ trials (1): {soz_counts.get(1, 0)}")
        print(f"Non-SOZ trials (0): {soz_counts.get(0, 0)}")
        print(f"Total trials: {len(self.trial_soz_labels)}")