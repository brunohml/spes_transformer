from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
import pickle
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.utils import load_data

from datasets import utils


logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        elif self.norm_type == "preproc_normalized": # data has been normalized in preprocessing, don't normalize during run
            return df

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class WeldData(BaseData):
    """
    Dataset class for welding dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df = self.all_df.sort_values(by=['weld_record_index'])  # datasets is presorted
        # TODO: There is a single ID that causes the model output to become nan - not clear why
        self.all_df = self.all_df[self.all_df['weld_record_index'] != 920397]  # exclude particular ID
        self.all_df = self.all_df.set_index('weld_record_index')
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
        self.max_seq_len = 66
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(WeldData.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(WeldData.load_single(path) for path in input_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = WeldData.read_data(filepath)
        df = WeldData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced by 0".format(num_nan, filepath))
            df = df.fillna(0)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various weld sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):
        """"""
        df = df.rename(columns={"per_energy": "power"})
        # Sometimes 'diff_time' is not measured correctly (is 0), and power ('per_energy') becomes infinite
        is_error = df['power'] > 1e16
        df.loc[is_error, 'power'] = df.loc[is_error, 'true_energy'] / df['diff_time'].median()

        df['weld_record_index'] = df['weld_record_index'].astype(int)
        keep_cols = ['weld_record_index', 'wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        df = df[keep_cols]

        return df


class TSRegressionArchive(BaseData):
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        #self.set_num_processes(n_proc=n_proc)

        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):

        # Every row of the returned df corresponds to a sample;
        # every column is a pd.Series indexed by timestamp and corresponds to a different dimension (feature)
        if self.config['task'] == 'regression':
            df, labels = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        elif self.config['task'] == 'classification':
            df, labels = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
        else:  # e.g. imputation
            try:
                data = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                     replace_missing_vals_with='NaN')
                if isinstance(data, tuple):
                    df, labels = data
                else:
                    df = data
            except:
                df, _ = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                 replace_missing_vals_with='NaN')
            labels_df = None

        lengths = df.applymap(lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning("Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0]*[row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


class PMUData(BaseData):
    """
    Dataset class for Phasor Measurement Unit dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length (optional). Used only if script argument `max_seq_len` is not
            defined.
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)

        if config['data_window_len'] is not None:
            self.max_seq_len = config['data_window_len']
            # construct sample IDs: 0, 0, ..., 0, 1, 1, ..., 1, 2, ..., (num_whole_samples - 1)
            # num_whole_samples = len(self.all_df) // self.max_seq_len  # commented code is for more general IDs
            # IDs = list(chain.from_iterable(map(lambda x: repeat(x, self.max_seq_len), range(num_whole_samples + 1))))
            # IDs = IDs[:len(self.all_df)]  # either last sample is completely superfluous, or it has to be shortened
            IDs = [i // self.max_seq_len for i in range(self.all_df.shape[0])]
            self.all_df.insert(loc=0, column='ExID', value=IDs)
        else:
            # self.all_df = self.all_df.sort_values(by=['ExID'])  # dataset is presorted
            self.max_seq_len = 30

        self.all_df = self.all_df.set_index('ExID')
        # rename columns
        self.all_df.columns = [re.sub(r'\d+', str(i//3), col_name) for i, col_name in enumerate(self.all_df.columns[:])]
        #self.all_df.columns = ["_".join(col_name.split(" ")[:-1]) for col_name in self.all_df.columns[:]]
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = self.all_df.columns  # all columns are used as features
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(PMUData.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(PMUData.load_single(path) for path in input_paths)

        return all_df


class SPESData(BaseData):
    """
    Dataset class for SPES (Single Pulse Electrical Stimulation) dataset.
    
    Attributes:
        all_df: (num_samples * seq_len, num_channels) dataframe indexed by integer indices, 
            with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains a channel feature.
        feature_df: same as all_df (all columns except metadata are features)
        feature_names: names of columns contained in feature_df
        all_IDs: unique integer indices contained in all_df
        labels_df: (num_samples, 1) dataframe containing SOZ labels for each trial
    """
    def __init__(self, data_dir, pattern=None, n_proc=1, limit_size=None, config=None):
        """
        Initialize the SPES dataset from preprocessed CSV files
        
        Args:
            data_dir: Directory containing processed CSV files
            pattern: 'TRAIN' or 'TEST' to specify which split to load
            n_proc: Number of processes (not used when loading preprocessed data)
            limit_size: Optional size limit
            config: Optional configuration
        """
        self.config = config
        
        # Load features and labels
        if pattern == 'TEST':
            df = pd.read_csv(os.path.join(data_dir, 'SPES_TEST.csv'), index_col=0)
            labels_df = pd.read_csv(os.path.join(data_dir, 'SPES_TEST_labels.csv'), index_col=0)
            logger.info(f"Loaded TEST data:")
        else:
            df = pd.read_csv(os.path.join(data_dir, 'SPES_TRAIN.csv'), index_col=0)
            labels_df = pd.read_csv(os.path.join(data_dir, 'SPES_TRAIN_labels.csv'), index_col=0)
            logger.info(f"Loaded TRAIN data:")
        
        # Set up features
        self.feature_names = [col for col in df.columns if col.startswith('channel_')]
        self.all_df = df
        self.feature_df = self.all_df[self.feature_names]
        self.all_IDs = self.all_df.index.unique()
        
        # Take only first label for each sequence
        self.labels_df = pd.DataFrame(index=self.all_IDs)
        self.labels_df['soz'] = [labels_df.iloc[i * 487]['soz'] for i in range(len(self.all_IDs))]
        
        logger.info(f"Features shape: {self.feature_df.shape}")
        logger.info(f"Number of unique trials: {len(self.all_IDs)}")
        logger.info(f"Label distribution: {self.labels_df['soz'].value_counts()}")
        
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]
            self.feature_df = self.feature_df.loc[self.all_IDs]
            self.labels_df = self.labels_df.loc[self.all_IDs]

    @property
    def labels(self):
        """Return labels for classification"""
        return self.labels_df['soz'].values  # Return raw labels, let Dataset handle one-hot encoding

    @property
    def class_names(self):
        """Return class names for classification"""
        return ['non-SOZ', 'SOZ']  # Binary classification

    def sanity_check_sequence(self, sample_id):
        """
        Verify that rows for a given sample ID form a complete sequence
        
        Args:
            sample_id: ID of the sample to check
        
        Returns:
            bool: True if sequence is valid, raises AssertionError otherwise
        """
        # Get all rows for this sample
        sample_rows = self.all_df.loc[sample_id]
        
        # Checks
        try:
            # Check 1: We have exactly 487 time points
            assert len(sample_rows) == 487, f"Sample {sample_id} has {len(sample_rows)} rows instead of 487"
            
            # Check 2: All rows have same metadata
            metadata_cols = ['patient_id', 'stim_pair', 'trial_number', 'training_example_id']
            for col in metadata_cols:
                if col in sample_rows.columns:
                    unique_vals = sample_rows[col].nunique()
                    assert unique_vals == 1, f"Sample {sample_id} has {unique_vals} different {col} values"
            
            # Check 3: Sequence indices are sequential
            if 'sequence_idx' in sample_rows.columns:
                seq_indices = sample_rows['sequence_idx'].values
                assert np.array_equal(seq_indices, np.arange(487)), \
                    f"Sample {sample_id} sequence indices are not sequential"
            
            # Check 4: Channel columns exist and have no NaN values
            channel_cols = [col for col in sample_rows.columns if col.startswith('channel_')]
            assert len(channel_cols) == 36, f"Sample {sample_id} has {len(channel_cols)} channels instead of 36"
            assert not sample_rows[channel_cols].isna().any().any(), \
                f"Sample {sample_id} has NaN values in channel data"
                
            return True
            
        except AssertionError as e:
            print(f"Sanity check failed for sample {sample_id}: {str(e)}")
            return False

class DynamicSPESData(BaseData):
    """
    Dynamic version of SPESData that loads individual examples from pickle files
    instead of loading the entire dataset at once. Randomly subselects 36 channels
    for each training example.
    """
    def __init__(self, data_dir, pattern='train', **kwargs):
        if pattern == 'train':
            self.data_dir = data_dir
        elif pattern == 'test':
            self.data_dir = os.path.join(data_dir, 'test_pickles')
        
        # Initialize logger
        self.logger = logging.getLogger('__main__')
        
        # Get all patient directories
        self.patient_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        logger.info(f"Found {len(self.patient_dirs)} patient directories in {self.data_dir}")

        # Get all pickle files for all patients matching, index files for later referencing
        self.all_IDs = []
        self.file_paths = []

        for patient in self.patient_dirs:
            patient_path = os.path.join(self.data_dir, patient)
            pickle_files = [f for f in os.listdir(patient_path) if f.endswith('.pickle')]
            logger.info(f"Found {len(pickle_files)} pickle files in {patient_path}")
            self.file_paths.extend([os.path.join(patient, f) for f in pickle_files])
        
        self.all_IDs = list(range(len(self.file_paths)))
        logger.info(f"Total pickle files found: {len(self.all_IDs)}")

        # Initialize other attributes that will be set when loading examples
        self.feature_names = [f'channel_{i}' for i in range(36)]
        self.feature_df = pd.DataFrame(np.zeros((487, 36)), columns=self.feature_names) # model factory uses feature_df shape during model initialization
        self.n_channels = 36
        self.train_df = None
        self.val_df = None
        self.test_df = None

        # Load and cache SOZ labels for classification task
        all_labels_df = pd.read_csv('spes_trial_metadata/labels_SOZ.csv')
        
        # Create mapping of patient_stim_pair -> SOZ label
        self.labels_cache = {}
        for _, row in all_labels_df.iterrows():
            # Clean up the Lead string by removing spaces around hyphen
            lead = row['Lead'].replace(' ', '')  # Remove all spaces
            
            # Convert Pt and Lead to match pickle file format (e.g., "Epat26_LA1-LA2")
            key = f"{row['Pt']}_{lead}"
            self.labels_cache[key] = row['SOZ']
            
        logger.info(f"Loaded {len(self.labels_cache)} SOZ labels")

    def load_example(self, idx):
        """Load a single example from pickle file"""
        file_path = os.path.join(self.data_dir, self.file_paths[idx])
        
        with open(file_path, 'rb') as f:
            example_data = pickle.load(f)
            
        # Get response matrices, channel names, and distances
        response_matrices = example_data[0].T  # Transpose to get (487, n_channels)
        channel_names = example_data[4]
        channel_distances = example_data[5]
        
        # self.logger.info(f"Raw response matrix shape for {file_path}: {response_matrices.shape}")
        
        # Set sequence length from data
        self.seq_len = response_matrices.shape[0]
        # self.logger.info(f"Sequence length set to: {self.seq_len}")
        
        # Randomly select 36 channels
        total_channels = len(channel_names)
        selected_indices = np.random.choice(total_channels, size=36, replace=False)
        
        # Get selected data
        selected_channels = [channel_names[i] for i in selected_indices]
        selected_channel_distances = [channel_distances[i] for i in selected_indices]
        
        # Sort channels by distance
        channel_indices = list(range(36))
        sorted_indices = sorted(channel_indices, key=lambda x: selected_channel_distances[x])
        
        # Convert to DataFrame with only channel data
        sequence_data = []
        for t in range(self.seq_len):
            row_data = {}  # No sequence_idx
            # Add channel data for selected channels in sorted order
            for new_idx, old_idx in enumerate(sorted_indices):
                orig_ch_idx = selected_indices[old_idx]
                row_data[f'channel_{new_idx}'] = response_matrices[t, orig_ch_idx]
                
            sequence_data.append(row_data)
            
        # Create DataFrame for this example
        example_id = os.path.splitext(os.path.basename(file_path))[0]
        self.all_df = pd.DataFrame(sequence_data)
        self.all_df.index = [example_id] * self.seq_len

        # Store selected channel information (for reference if needed)
        self.selected_channels = {
            f'channel_{new_idx}': {
                'name': selected_channels[old_idx],
                'distance': channel_distances[selected_indices[old_idx]]
            } for new_idx, old_idx in enumerate(sorted_indices)
        }
        
        # Add logging to verify structure
        channel_cols = [col for col in self.all_df.columns if col.startswith('channel_')]
        # self.logger.info(f"Number of channel columns: {len(channel_cols)}")
        # self.logger.info(f"Final DataFrame shape for {file_path}: {self.all_df.shape}")  # Should be (487, 36) - just channels
        # self.logger.info("=== End Loading ===\n")
        
        return self.all_df
        
    def __len__(self):
        return len(self.all_IDs)
        
    def __getitem__(self, idx):
        """Get a single example by index"""
        df = self.load_example(idx)
        return df
        
    def get_channel_info(self):
        """Return information about currently selected channels"""
        if not hasattr(self, 'selected_channels'):
            raise RuntimeError("No channels selected yet. Load an example first.")
        return self.selected_channels

    def get_label(self, file_path):
        """Extract label for a given pickle file"""
        # Extract patient and stim pair from filename
        # Example: spes_trial_pickles/Epat26_LA1-LA2_3mA_pulse_1.pickle
        filename = os.path.basename(file_path)
        pat_stim = '_'.join(filename.split('_')[:2])  # Gets "Epat26_LA1-LA2"
        
        if pat_stim not in self.labels_cache:
            raise KeyError(f"No SOZ label found for {pat_stim}")
            
        return self.labels_cache[pat_stim]

    @property
    def class_names(self):
        """Return class names for classification"""
        return ['non-SOZ', 'SOZ']  # Binary classification
        
    @property
    def labels(self):
        """Return all labels (used by TimeSeriesDataset)"""
        # Since we're loading dynamically, we'll need to get labels for all files
        return np.array([self.get_label(fp) for fp in self.file_paths])

data_factory = {'weld': WeldData,
                'tsra': TSRegressionArchive,
                'pmu': PMUData,
                'spes': SPESData,
                'dynamic_spes': DynamicSPESData}
