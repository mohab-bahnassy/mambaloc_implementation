"""
UWB Opera Dataset Loader
Loads UWB data from the Opera dataset CSV files for localization tasks.
Handles 50 complex CIR values and ground truth coordinates.
"""

import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union
import glob
from sklearn.preprocessing import StandardScaler
import pickle


class UWBOperaDataset(Dataset):
    """
    PyTorch Dataset for UWB Opera dataset.
    
    Loads CSV files containing UWB measurements with 50 complex CIR values
    and extracts features for localization regression.
    """
    
    def __init__(
        self,
        data_path: str,
        experiments: Optional[List[str]] = None,
        sequence_length: int = 32,
        feature_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
        use_complex_features: bool = True,
        target_tags: List[str] = ['tag4422', 'tag89b3'],
        include_additional_features: bool = True,
        stride: int = 1,
        min_sequence_length: int = 10,
        max_rows_per_file: int = 10000  # Limit rows per file for faster loading
    ):
        """
        Initialize UWB Opera dataset.
        
        Args:
            data_path: Path to directory containing UWB CSV files
            experiments: List of experiment numbers to include (e.g., ['001', '028'])
            sequence_length: Length of sequences to create
            feature_scaler: Scaler for input features (fitted on training data)
            target_scaler: Scaler for target coordinates (fitted on training data)
            use_complex_features: Whether to use complex CIR values (both real and imag)
            target_tags: List of tag IDs to use for ground truth coordinates
            include_additional_features: Whether to include non-CIR features
            stride: Stride for sequence creation
            min_sequence_length: Minimum sequence length to keep
            max_rows_per_file: Maximum number of rows to load from each CSV file
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.use_complex_features = use_complex_features
        self.target_tags = target_tags
        self.include_additional_features = include_additional_features
        self.stride = stride
        self.min_sequence_length = min_sequence_length
        self.max_rows_per_file = max_rows_per_file
        
        # Get all CSV files if no specific experiments specified
        if experiments is None:
            csv_files = glob.glob(os.path.join(data_path, "uwb2_exp*.csv"))
        else:
            csv_files = [os.path.join(data_path, f"uwb2_exp{exp}.csv") for exp in experiments]
        
        # Load and process data
        self.data = self._load_and_process_data(csv_files)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"📊 Loaded {len(self.sequences)} sequences from {len(csv_files)} files")
        
    def _load_and_process_data(self, csv_files: List[str]) -> List[pd.DataFrame]:
        """Load and process CSV files."""
        all_data = []
        
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                print(f"⚠️ Warning: File {csv_file} not found, skipping")
                continue
                
            print(f"📁 Loading {os.path.basename(csv_file)}")
            
            try:
                df = pd.read_csv(csv_file, nrows=self.max_rows_per_file)
                print(f"   📊 Loaded {len(df)} rows from {os.path.basename(csv_file)}")
                
                # Check if this file has ground truth coordinates
                has_ground_truth = any(f"{tag}_x" in df.columns for tag in self.target_tags)
                
                if not has_ground_truth:
                    print(f"⚠️ No ground truth coordinates found in {os.path.basename(csv_file)}, skipping")
                    continue
                
                # Process the data
                print(f"   🔄 Processing {os.path.basename(csv_file)}...")
                processed_df = self._process_dataframe(df)
                
                if len(processed_df) > self.min_sequence_length:
                    all_data.append(processed_df)
                    print(f"   ✅ Added {len(processed_df)} processed rows")
                else:
                    print(f"⚠️ Not enough data in {os.path.basename(csv_file)}, skipping")
                    
            except Exception as e:
                print(f"❌ Error loading {csv_file}: {e}")
                continue
        
        return all_data
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a single dataframe."""
        # Extract CIR features (cir1 to cir50)
        cir_columns = [f'cir{i}' for i in range(1, 51)]
        cir_data = df[cir_columns].copy()
        
        # Convert complex strings to actual complex numbers
        for col in cir_columns:
            cir_data[col] = cir_data[col].apply(self._parse_complex)
        
        # Create feature matrix
        features = []
        
        if self.use_complex_features:
            # Use magnitude and phase (100 features total)
            for col in cir_columns:
                features.append(cir_data[col].apply(lambda x: abs(x)).values)
                features.append(cir_data[col].apply(lambda x: np.angle(x)).values)
        else:
            # Use only magnitude (50 features)
            for col in cir_columns:
                features.append(cir_data[col].apply(lambda x: abs(x)).values)
        
        # Add additional features if requested
        if self.include_additional_features:
            additional_features = [
                'fp_pow_dbm', 'rx_pow_dbm', 'tx_x_coord', 'tx_y_coord', 
                'rx_x_coord', 'rx_y_coord', 'tx_rx_dist_meters',
                'fp_index', 'fp_amp1', 'fp_amp2', 'fp_amp3', 
                'max_growth_cir', 'rx_pream_count'
            ]
            
            for feat in additional_features:
                if feat in df.columns:
                    # Clean the feature values
                    feat_values = df[feat].values
                    feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=1e6, neginf=-1e6)
                    feat_values = np.clip(feat_values, -1e6, 1e6)
                    features.append(feat_values)
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        # Clean the feature matrix - replace inf/nan with zeros and clip extreme values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        feature_matrix = np.clip(feature_matrix, -1e6, 1e6)
        
        # Extract target coordinates
        targets = []
        for tag in self.target_tags:
            x_col = f"{tag}_x"
            y_col = f"{tag}_y"
            
            if x_col in df.columns and y_col in df.columns:
                # Handle NaN values by forward filling - fix deprecation warning
                x_vals = df[x_col].ffill().bfill().values
                y_vals = df[y_col].ffill().bfill().values
                
                # Clean coordinate values
                x_vals = np.nan_to_num(x_vals, nan=0.0, posinf=1000.0, neginf=-1000.0)
                y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=1000.0, neginf=-1000.0)
                x_vals = np.clip(x_vals, -1000.0, 1000.0)  # Reasonable coordinate bounds
                y_vals = np.clip(y_vals, -1000.0, 1000.0)
                
                targets.extend([x_vals, y_vals])
        
        if len(targets) == 0:
            raise ValueError("No valid target coordinates found")
        
        target_matrix = np.column_stack(targets)
        
        # Create result DataFrame
        result_df = pd.DataFrame()
        result_df['features'] = [feature_matrix[i] for i in range(len(feature_matrix))]
        result_df['targets'] = [target_matrix[i] for i in range(len(target_matrix))]
        result_df['timestamp'] = df['timestamp'].values
        result_df['tx_id'] = df['tx_id'].values
        result_df['rx_id'] = df['rx_id'].values
        
        return result_df
    
    def _parse_complex(self, complex_str: str) -> complex:
        """Parse complex number from string format."""
        try:
            # Handle format like "-0.59664+0.042017i"
            complex_str = str(complex_str).strip()
            if complex_str == 'nan' or complex_str == '' or complex_str == 'None':
                return 0+0j
            
            # Replace 'i' with 'j' for Python complex number parsing
            complex_str = complex_str.replace('i', 'j')
            result = complex(complex_str)
            
            # Check for inf or nan in the result
            if not (np.isfinite(result.real) and np.isfinite(result.imag)):
                return 0+0j
            
            # Clip extreme values to prevent overflow
            real_part = np.clip(result.real, -1e6, 1e6)
            imag_part = np.clip(result.imag, -1e6, 1e6)
            
            return complex(real_part, imag_part)
        except:
            return 0+0j
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create sequences from the loaded data."""
        sequences = []
        
        for df in self.data:
            # Group by tx_id and rx_id to maintain link consistency
            grouped = df.groupby(['tx_id', 'rx_id'])
            
            for (tx_id, rx_id), group in grouped:
                if len(group) < self.min_sequence_length:
                    continue
                
                # Extract features and targets
                features = np.array([row for row in group['features'].values])
                targets = np.array([row for row in group['targets'].values])
                
                # Create sequences with stride
                for i in range(0, len(features) - self.sequence_length + 1, self.stride):
                    seq_features = features[i:i + self.sequence_length]
                    seq_targets = targets[i:i + self.sequence_length]
                    
                    sequences.append((seq_features, seq_targets))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence sample."""
        features, targets = self.sequences[idx]
        
        # Clean data before scaling
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features = np.clip(features, -1e6, 1e6)
        targets = np.nan_to_num(targets, nan=0.0, posinf=1000.0, neginf=-1000.0) 
        targets = np.clip(targets, -1000.0, 1000.0)
        
        # Apply scaling if available
        if self.feature_scaler is not None:
            # Reshape for scaling, then reshape back
            orig_shape = features.shape
            features = features.reshape(-1, features.shape[-1])
            features = self.feature_scaler.transform(features)
            features = features.reshape(orig_shape)
            
            # Final cleaning after scaling
            features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            features = np.clip(features, -10.0, 10.0)
        
        if self.target_scaler is not None:
            orig_shape = targets.shape
            targets = targets.reshape(-1, targets.shape[-1])
            targets = self.target_scaler.transform(targets)
            targets = targets.reshape(orig_shape)
            
            # Final cleaning after scaling
            targets = np.nan_to_num(targets, nan=0.0, posinf=10.0, neginf=-10.0)
            targets = np.clip(targets, -10.0, 10.0)
        
        return torch.FloatTensor(features), torch.FloatTensor(targets)


def create_uwb_opera_dataloaders(
    data_path: str,
    train_experiments: List[str],
    val_experiments: List[str],
    batch_size: int = 32,
    sequence_length: int = 32,
    use_complex_features: bool = True,
    target_tags: List[str] = ['tag4422', 'tag89b3'],
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, StandardScaler, StandardScaler]:
    """
    Create train and validation dataloaders for UWB Opera dataset.
    
    Args:
        data_path: Path to UWB dataset directory
        train_experiments: List of experiment numbers for training
        val_experiments: List of experiment numbers for validation
        batch_size: Batch size for dataloaders
        sequence_length: Length of sequences
        use_complex_features: Whether to use complex CIR features
        target_tags: Tags to use for ground truth coordinates
        **kwargs: Additional arguments for dataset
        
    Returns:
        train_loader, val_loader, feature_scaler, target_scaler
    """
    
    # Create training dataset (without scaling)
    train_dataset = UWBOperaDataset(
        data_path=data_path,
        experiments=train_experiments,
        sequence_length=sequence_length,
        use_complex_features=use_complex_features,
        target_tags=target_tags,
        max_rows_per_file=15000,  # More data for larger teacher model
        stride=4,  # Smaller stride for more sequences
        **kwargs
    )
    
    # Fit scalers on training data
    print("📏 Fitting scalers on training data...")
    all_features = []
    all_targets = []
    
    for i in range(min(1000, len(train_dataset))):  # Sample for fitting
        features, targets = train_dataset[i]
        all_features.append(features.numpy().reshape(-1, features.shape[-1]))
        all_targets.append(targets.numpy().reshape(-1, targets.shape[-1]))
    
    # Concatenate and fit scalers
    all_features = np.vstack(all_features)
    all_targets = np.vstack(all_targets)
    
    # Final data cleaning before fitting scalers
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
    all_features = np.clip(all_features, -1e6, 1e6)
    all_targets = np.nan_to_num(all_targets, nan=0.0, posinf=1000.0, neginf=-1000.0)
    all_targets = np.clip(all_targets, -1000.0, 1000.0)
    
    # Verify data is finite
    if not np.all(np.isfinite(all_features)):
        print("⚠️ Warning: Non-finite values detected in features after cleaning")
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if not np.all(np.isfinite(all_targets)):
        print("⚠️ Warning: Non-finite values detected in targets after cleaning")
        all_targets = np.nan_to_num(all_targets, nan=0.0, posinf=0.0, neginf=0.0)
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    feature_scaler.fit(all_features)
    target_scaler.fit(all_targets)
    
    # Create datasets with scalers
    train_dataset = UWBOperaDataset(
        data_path=data_path,
        experiments=train_experiments,
        sequence_length=sequence_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        use_complex_features=use_complex_features,
        target_tags=target_tags,
        max_rows_per_file=15000,  # More data for larger teacher model
        stride=4,  # Smaller stride for more sequences
        **kwargs
    )
    
    val_dataset = UWBOperaDataset(
        data_path=data_path,
        experiments=val_experiments,
        sequence_length=sequence_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        use_complex_features=use_complex_features,
        target_tags=target_tags,
        max_rows_per_file=10000,  # More validation data too
        stride=4,  # Smaller stride for more sequences
        **kwargs
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    print(f"🎯 Training samples: {len(train_dataset)}")
    print(f"🎯 Validation samples: {len(val_dataset)}")
    print(f"📐 Feature dimensions: {train_dataset[0][0].shape}")
    print(f"📐 Target dimensions: {train_dataset[0][1].shape}")
    
    return train_loader, val_loader, feature_scaler, target_scaler


def save_scalers(feature_scaler: StandardScaler, target_scaler: StandardScaler, save_path: str):
    """Save scalers to file."""
    scaler_data = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    with open(save_path, 'wb') as f:
        pickle.dump(scaler_data, f)
    print(f"💾 Scalers saved to {save_path}")


def load_scalers(load_path: str) -> Tuple[StandardScaler, StandardScaler]:
    """Load scalers from file."""
    with open(load_path, 'rb') as f:
        scaler_data = pickle.load(f)
    print(f"📂 Scalers loaded from {load_path}")
    return scaler_data['feature_scaler'], scaler_data['target_scaler'] 