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
    
    Attributes:
        all_df: dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains a feature (channel).
        feature_df: same as all_df (all columns are features)
        feature_names: names of columns contained in feature_df
        all_IDs: unique integer indices contained in all_df
        labels_df: dataframe containing SOZ labels for each trial
        metadata_df: dataframe containing additional information (stim_pair, patient_id)
    """
    def __init__(self, root_dir: str, file_list=None, n_electrodes: int = 36,
                 pattern=None, n_proc=1, limit_size=None, config=None):
        """
        Initialize the SPES dataset with training/test split by patient.
        
        Args:
            root_dir (str): Root directory containing patient subfolders
            file_list (list, optional): List containing patient ID to exclude from training (test patient)
            n_electrodes (int): Number of electrodes to select per trial
            pattern (str, optional): File pattern to match
            n_proc (int, optional): Number of processes for parallel processing
            limit_size (int, optional): Limit dataset size
            config (dict, optional): Configuration dictionary
        """
        # Initialize BaseData attributes
        self.config = config
        self.set_num_processes(n_proc)
        
        # Initialize SPES-specific attributes
        self.sampling_rate = 512
        self.seq_len = 487
        self.n_electrodes = n_electrodes
        
        # Initialize metadata dictionaries
        self.trial_electrodes = {}
        self.trial_stim_channels = {}
        self.trial_soz_labels = {}
        
        # Get test patient from file_list
        exclude_patient = file_list[0] if file_list and len(file_list) > 0 else None
        
        # Load training and test data separately
        self.train_df, self.train_labels, train_metadata = self._load_patient_data(
            root_dir, exclude_patient=exclude_patient)
        self.test_df, self.test_labels, test_metadata = self._load_patient_data(
            root_dir, test_patient=exclude_patient)
        
        # Set main dataframes (using training set as default)
        self.all_df = self.train_df
        self.labels_df = self.train_labels
        self.metadata_df = train_metadata
        self.all_IDs = self.train_df.index.unique()
        self.feature_df = self.train_df
        self.feature_names = self.train_df.columns.tolist()
        
        # Store test set information
        self.test_IDs = self.test_df.index.unique()
        
        # Apply size limit if specified
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]
            self.labels_df = self.labels_df.loc[self.all_IDs]

    def load_all(self, root_dir):
        """Load all data and return dataframes in required format for BaseData"""
        return self.all_df, self.labels_df

    def _load_patient_data(self, root_dir: str, exclude_patient: str = None, 
                          test_patient: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data for specified patients.
        
        Args:
            root_dir (str): Root directory containing patient subfolders
            exclude_patient (str): Patient ID to exclude (for training set)
            test_patient (str): Patient ID to include (for test set)
        """
        # Load SOZ labels
        try:
            soz_df = pd.read_csv(os.path.join('spes_trial_metadata', 'labels_SOZ.csv'))
            print(f"Loaded SOZ labels for {len(soz_df)} electrodes")
        except Exception as e:
            raise ValueError(f"Error loading SOZ labels: {str(e)}")
        
        all_trials = []
        all_labels = []
        metadata_dict = {'trial_id': [], 'stim_pair': [], 'patient_id': []}
        trial_counter = 0
        
        # Get patient directories and count total pickle files
        patient_dirs = [d for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
        
        total_pickles = 0
        for patient_dir in patient_dirs:
            patient_id = patient_dir.replace('patient_', '')
            if (exclude_patient and patient_id == exclude_patient) or \
               (test_patient and patient_id != test_patient):
                continue
            pickle_files = glob.glob(os.path.join(root_dir, patient_dir, "*.pickle"))
            total_pickles += len(pickle_files)
        
        # Print dataset split information
        if test_patient:
            print(f"\nProcessing test set for patient: {test_patient}")
        else:
            print(f"\nProcessing training set (excluding patient: {exclude_patient})")
        print(f"Total pickle files to process: {total_pickles}")
        
        processed_pickles = 0
        milestone = total_pickles // 5  # 20% intervals
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.replace('patient_', '')
            
            # Skip or include based on patient ID
            if exclude_patient and patient_id == exclude_patient:
                continue
            if test_patient and patient_id != test_patient:
                continue
                
            patient_path = os.path.join(root_dir, patient_dir)
            pickle_files = glob.glob(os.path.join(patient_path, "*.pickle"))
            
            for filepath in pickle_files:
                try:
                    # Load and process pickle file
                    with open(filepath, 'rb') as f:
                        trial_data = pickle.load(f)
                    
                    # Extract response matrix and channel names
                    if isinstance(trial_data, list):
                        response_matrix = trial_data[0]
                        available_channels = trial_data[2]
                        if isinstance(available_channels, np.ndarray):
                            available_channels = available_channels.tolist()
                    else:
                        response_matrix = trial_data['normalized_seeg']
                        available_channels = trial_data['chan']
                    
                    # Parse filename
                    parts = os.path.basename(filepath).replace('.pickle', '').split('_')
                    stim_pair = parts[1]
                    
                    # Channel selection
                    if len(available_channels) < self.n_electrodes:
                        print(f"Warning: Not enough channels in {filepath}. Skipping...")
                        continue
                    
                    selected_channels = random.sample(list(available_channels), self.n_electrodes)
                    channel_indices = [list(available_channels).index(ch) for ch in selected_channels]
                    selected_data = response_matrix[channel_indices, :].T
                    
                    # SOZ label lookup
                    soz_match = soz_df[(soz_df['Pt'] == patient_id) & 
                                       (soz_df['Lead'] == stim_pair)]
                    
                    soz_label = int(soz_match.iloc[0]['SOZ']) if len(soz_match) > 0 else 0
                    
                    # Store metadata
                    trial_id = f"trial_{trial_counter}"
                    metadata_dict['trial_id'].append(trial_id)
                    metadata_dict['stim_pair'].append(stim_pair)
                    metadata_dict['patient_id'].append(patient_id)
                    
                    # Create DataFrames with integer index
                    trial_df = pd.DataFrame(
                        selected_data,
                        columns=[f"channel_{i}" for i in range(self.n_electrodes)]
                    )
                    trial_df.index = pd.Series([trial_counter] * self.seq_len)
                    
                    # Create labels DataFrame
                    label_df = pd.DataFrame({
                        'soz': [soz_label]
                    }, index=[trial_counter])
                    
                    all_trials.append(trial_df)
                    all_labels.append(label_df)
                    trial_counter += 1
                    
                    processed_pickles += 1
                    # Print progress at 20% intervals
                    if milestone > 0 and processed_pickles % milestone == 0:
                        progress = (processed_pickles / total_pickles) * 100
                        print(f"Progress: {processed_pickles}/{total_pickles} files ({progress:.0f}% complete)")
                    
                except Exception as e:
                    print(f"Error processing file {os.path.basename(filepath)}: {str(e)}")
                    continue
        
        if not all_trials:
            raise ValueError(f"No valid trials processed for {'test' if test_patient else 'training'} set")
        
        print(f"\nFinished processing {processed_pickles} files")
        print(f"Created {len(all_trials)} valid trials")
        
        # Combine all data
        all_df = pd.concat(all_trials)
        labels_df = pd.concat(all_labels)
        metadata_df = pd.DataFrame(metadata_dict)
        metadata_df.set_index('trial_id', inplace=True)
        
        return all_df, labels_df, metadata_df

    @property
    def feature_dim(self):
        """Return feature dimensionality"""
        return self.feature_df.shape[1]

    @property
    def labels(self):
        """Return flattened labels for classification"""
        return self.labels_df['soz'].values
    
    @property
    def class_names(self):
        """Return class names for classification"""
        return [0, 1]  # Binary classification (SOZ vs non-SOZ)

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get test set data and labels"""
        return self.test_df, self.test_labels
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get training set data and labels"""
        return self.train_df, self.train_labels
    
    def get_metadata(self, trial_id: str) -> Dict[str, str]:
        """Get metadata for a specific trial"""
        return self.metadata_df.loc[trial_id].to_dict()