"""
Fixed CSI Dataset Loader for WiFi CSI Data
Handles .mat files with CSI data for localization tasks.
Properly processes all 270 complex CSI values (3x3x30) and ground truth coordinates.
Creates short sequences from CSI packets for model compatibility.
Fixed to avoid multiple data loading and ensure consistent train/val splits.
"""

import torch
import numpy as np
import h5py
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

warnings.filterwarnings('ignore')


class CSIDatasetFixed(Dataset):
    """
    Fixed PyTorch Dataset for WiFi CSI data from .mat files.
    
    Loads HDF5 .mat files containing CSI measurements with 270 complex values
    and creates short sequences for localization regression.
    """
    
    def __init__(
        self,
        mat_file_path: str,
        sequence_length: int = 4,  # Much shorter sequences for CSI
        target_tags: List[str] = ['tag4422'],
        use_magnitude_phase: bool = True,
        max_samples: Optional[int] = None,
        min_coordinate_threshold: float = -50.0,  # More reasonable coordinate bounds
        max_coordinate_threshold: float = 50.0,
        stride: int = 2  # Smaller stride for more sequences
    ):
        """
        Initialize CSI dataset.
        
        Args:
            mat_file_path: Path to the .mat file containing CSI data
            sequence_length: Length of sequences to create (keep small for CSI)
            target_tags: List of tag IDs to use for ground truth coordinates
            use_magnitude_phase: Whether to use magnitude and phase (True) or real/imag (False)
            max_samples: Maximum number of samples to load (None for all)
            min_coordinate_threshold: Minimum valid coordinate value
            max_coordinate_threshold: Maximum valid coordinate value
            stride: Stride for sequence creation
        """
        self.mat_file_path = mat_file_path
        self.sequence_length = sequence_length
        self.target_tags = target_tags
        self.use_magnitude_phase = use_magnitude_phase
        self.max_samples = max_samples
        self.min_coord_threshold = min_coordinate_threshold
        self.max_coord_threshold = max_coordinate_threshold
        self.stride = stride
        
        # Load and process data
        self.raw_features, self.raw_targets = self._load_and_process_data()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"📊 Dataset initialized: {len(self.sequences)} sequences from {len(self.raw_features)} samples")
        
    def _load_and_process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process CSI data from .mat file."""
        print(f"📁 Loading CSI data from {os.path.basename(self.mat_file_path)}")
        
        with h5py.File(self.mat_file_path, 'r') as f:
            # Try direct column access first (MATLAB v7.3 format)
            if 'exp002' in f:
                data_group = f['exp002']
                print("📊 Found exp002 data group")
                
                # Extract CSI columns directly (tx1rx1_sub1 through tx3rx3_sub30)
                csi_features = []
                missing_csi = []
                
                # Generate all expected CSI column names
                expected_csi_cols = []
                for tx in range(1, 4):  # tx1, tx2, tx3
                    for rx in range(1, 4):  # rx1, rx2, rx3
                        for sub in range(1, 31):  # sub1 through sub30
                            expected_csi_cols.append(f'tx{tx}rx{rx}_sub{sub}')
                
                print(f"🔍 Looking for {len(expected_csi_cols)} CSI columns...")
                
                for col_name in expected_csi_cols:
                    if col_name in data_group:
                        csi_array = data_group[col_name]
                        if hasattr(csi_array, 'dtype') and csi_array.dtype.names:
                            # Complex data stored as struct with 'real' and 'imag'
                            real_part = csi_array['real'][...]
                            imag_part = csi_array['imag'][...]
                            complex_data = real_part + 1j * imag_part
                            
                            # Flatten if needed
                            if complex_data.ndim > 1:
                                complex_data = complex_data.flatten()
                            
                            csi_features.append(complex_data)
                        else:
                            # Regular array
                            data = csi_array[...]
                            if data.ndim > 1:
                                data = data.flatten()
                            csi_features.append(data)
                    else:
                        missing_csi.append(col_name)
                
                if missing_csi:
                    print(f"⚠️ Missing {len(missing_csi)} CSI columns: {missing_csi[:5]}...")
                
                print(f"✅ Found {len(csi_features)} CSI columns")
                
                # Use the minimum length among all CSI arrays
                min_length = min(len(arr) for arr in csi_features) if csi_features else 0
                print(f"📏 Using {min_length:,} samples (minimum array length)")
                
                # Apply early sampling if max_samples is specified
                if self.max_samples is not None and min_length > self.max_samples:
                    print(f"💾 Memory management: Sampling {self.max_samples:,} from {min_length:,} available samples")
                    # Use random sampling for reproducibility
                    np.random.seed(42)
                    sample_indices = np.random.choice(min_length, self.max_samples, replace=False)
                    sample_indices = np.sort(sample_indices)  # Keep temporal order
                    min_length = self.max_samples
                else:
                    sample_indices = None
                
                # Truncate all arrays to minimum length (with sampling if needed)
                if sample_indices is not None:
                    csi_features = [arr[sample_indices] for arr in csi_features]
                else:
                    csi_features = [arr[:min_length] for arr in csi_features]
                
                # Stack CSI data: (num_samples, num_csi_features)
                csi_data = np.column_stack(csi_features)
                print(f"📊 CSI data shape: {csi_data.shape}")
                print(f"💾 CSI memory usage: {csi_data.nbytes / 1024**2:.1f} MB")
                
                # Clear individual CSI arrays to free memory
                del csi_features
                
                # Load coordinate data
                coord_data = []
                for tag in self.target_tags:
                    x_col = f'{tag}_x'
                    y_col = f'{tag}_y'
                    
                    if x_col in data_group and y_col in data_group:
                        x_data = data_group[x_col][...]
                        y_data = data_group[y_col][...]
                        
                        # Flatten if needed
                        if x_data.ndim > 1:
                            x_data = x_data.flatten()
                        if y_data.ndim > 1:
                            y_data = y_data.flatten()
                        
                        # Apply same sampling as CSI data
                        if sample_indices is not None:
                            x_data = x_data[sample_indices]
                            y_data = y_data[sample_indices]
                        else:
                            x_data = x_data[:min_length]
                            y_data = y_data[:min_length]
                        
                        coord_data.extend([x_data, y_data])
                        print(f"✅ Loaded {tag} coordinates: {len(x_data):,} samples")
                    else:
                        print(f"⚠️ Coordinates for {tag} not found")
                
                if not coord_data:
                    raise ValueError("No coordinate data found")
                
                coordinate_data = np.column_stack(coord_data)
                print(f"📊 Coordinate data shape: {coordinate_data.shape}")
                print(f"💾 Total memory usage: {(csi_data.nbytes + coordinate_data.nbytes) / 1024**2:.1f} MB")
                
            else:
                # Fallback to refs-based loading (older approach)
                print("📊 Using refs-based loading")
                refs = f['#refs#']
                
                # Get field mappings
                field_names = {}
                for key in refs.keys():
                    data = refs[key]
                    if hasattr(data, 'dtype') and data.dtype == 'uint16':
                        try:
                            ascii_data = data[...].flatten()
                            decoded = ''.join(chr(x) for x in ascii_data if x > 0)
                            if len(decoded) < 50:  # Field names should be short
                                field_names[key] = decoded
                        except:
                            pass
                
                # Find CSI mappings
                csi_mappings = {}
                for key, name in field_names.items():
                    if 'tx' in name and 'rx' in name and 'sub' in name:
                        data_key = key.replace('e', '').replace('f', '').replace('g', '').replace('h', '').replace('i', '')
                        if data_key in refs:
                            csi_mappings[name] = data_key
                
                print(f"📡 Found {len(csi_mappings)} CSI mappings")
                
                # Load CSI data
                csi_data_list = []
                for field_name in sorted(csi_mappings.keys()):
                    data_key = csi_mappings[field_name]
                    csi_array = refs[data_key]
                    
                    if hasattr(csi_array, 'dtype') and csi_array.dtype.names and 'real' in csi_array.dtype.names:
                        # Extract complex data
                        if len(csi_array.shape) == 2:
                            real_part = csi_array['real'][0, :]
                            imag_part = csi_array['imag'][0, :]
                        else:
                            real_part = csi_array['real'][:]
                            imag_part = csi_array['imag'][:]
                        
                        complex_data = real_part + 1j * imag_part
                        csi_data_list.append(complex_data)
                
                min_length = min(len(arr) for arr in csi_data_list) if csi_data_list else 0
                csi_data_list = [arr[:min_length] for arr in csi_data_list]
                csi_data = np.column_stack(csi_data_list)
                
                # Load coordinates using refs
                coord_mappings = {}
                for key, name in field_names.items():
                    for tag in self.target_tags:
                        if tag in name and ('_x' in name or '_y' in name):
                            data_key = key.replace('e', '').replace('f', '').replace('g', '').replace('h', '').replace('i', '')
                            if data_key in refs:
                                coord_mappings[name] = data_key
                
                coord_data = []
                for tag in self.target_tags:
                    x_field = f"{tag}_x"
                    y_field = f"{tag}_y"
                    
                    if x_field in coord_mappings and y_field in coord_mappings:
                        x_array = refs[coord_mappings[x_field]]
                        y_array = refs[coord_mappings[y_field]]
                        
                        # Extract coordinate data
                        if hasattr(x_array, 'dtype') and x_array.dtype.names and 'real' in x_array.dtype.names:
                            # Complex coordinate data - use MAGNITUDE for ground truth mapping
                            # Based on analysis: magnitude gives most realistic movement patterns
                            x_coords = np.abs(x_array['real'][:min_length] + 1j * x_array['imag'][:min_length]).flatten()
                            y_coords = np.abs(y_array['real'][:min_length] + 1j * y_array['imag'][:min_length]).flatten()
                            print(f"   🎯 Using magnitude mapping for complex coordinates (optimal strategy)")
                        else:
                            x_coords = x_array[:min_length].flatten()
                            y_coords = y_array[:min_length].flatten()
                        
                        coord_data.extend([x_coords, y_coords])
                        print(f"✅ Loaded {tag} coordinates: {len(x_coords)} samples")
                
                if coord_data:
                    coordinate_data = np.column_stack(coord_data)
                else:
                    raise ValueError("No coordinate data found")
            
            # Apply sample limit if specified
            if self.max_samples is not None:
                n_samples = min(self.max_samples, len(csi_data))
                csi_data = csi_data[:n_samples]
                coordinate_data = coordinate_data[:n_samples]
            
            print(f"📊 Raw data: {csi_data.shape[0]} samples with {csi_data.shape[1]} CSI features")
            print(f"📍 Coordinates: {coordinate_data.shape}")
            
            # Process features
            features = self._process_csi_features(csi_data)
            
            # Clean coordinates
            targets = self._clean_coordinates(coordinate_data)
            
            # Remove samples with invalid coordinates
            valid_mask = np.all(
                (targets >= self.min_coord_threshold) & 
                (targets <= self.max_coord_threshold), 
                axis=1
            )
            
            features = features[valid_mask]
            targets = targets[valid_mask]
            
            print(f"✅ Final data: {features.shape[0]} valid samples with {features.shape[1]} features")
            
            return features, targets
    
    def _process_csi_features(self, csi_data: np.ndarray) -> np.ndarray:
        """Process CSI data to extract features."""
        if self.use_magnitude_phase:
            # Use magnitude and phase
            magnitude = np.abs(csi_data)
            phase = np.angle(csi_data)
            # Unwrap phase for better continuity
            phase = np.unwrap(phase, axis=1)
            features = np.concatenate([magnitude, phase], axis=1)
        else:
            # Use real and imaginary parts
            real_part = np.real(csi_data)
            imag_part = np.imag(csi_data)
            features = np.concatenate([real_part, imag_part], axis=1)
        
        # Clean features
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features = np.clip(features, -1e6, 1e6)
        
        return features
    
    def _clean_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Clean coordinate data."""
        # Remove NaN and inf values
        coordinates = np.nan_to_num(coordinates, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return coordinates
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create short sequences from CSI data."""
        sequences = []
        
        # Create sequences with sliding window (much shorter than UWB)
        for start_idx in range(0, len(self.raw_features) - self.sequence_length + 1, self.stride):
            end_idx = start_idx + self.sequence_length
            
            # Extract sequence
            sequence_features = self.raw_features[start_idx:end_idx]
            sequence_targets = self.raw_targets[start_idx:end_idx]
            
            # Use the last target as the sequence target (most recent position)
            target = sequence_targets[-1]
            
            sequences.append((sequence_features, target))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_features, target = self.sequences[idx]
        return torch.FloatTensor(sequence_features), torch.FloatTensor(target)


def create_csi_dataloaders_fixed(
    mat_file_path: str,
    train_split: float = 0.8,
    batch_size: int = 32,
    sequence_length: int = 4,  # Short sequences for CSI
    target_tags: List[str] = ['tag4422'],
    use_magnitude_phase: bool = True,
    max_samples: Optional[int] = None,
    temporal_gap: int = 0,  # Gap between train and val data to prevent leakage
    **kwargs
) -> Tuple[DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    Create CSI train and validation dataloaders with FIXED TEMPORAL SPLITTING to prevent data leakage.
    
    CRITICAL FIX: Uses temporal split instead of random split to prevent overlapping sequences
    between train and validation sets.
    
    Args:
        mat_file_path: Path to the .mat file
        train_split: Fraction of data to use for training
        batch_size: Batch size for dataloaders
        sequence_length: Length of sequences to create (keep small for CSI)
        target_tags: Tags to use for coordinates
        use_magnitude_phase: Whether to use magnitude/phase or real/imag
        max_samples: Maximum number of samples to load
        temporal_gap: Number of samples to skip between train and val to prevent leakage
        **kwargs: Additional arguments for CSIDatasetFixed
    
    Returns:
        (train_loader, val_loader, feature_scaler, target_scaler)
    """
    print("🔧 Creating CSI dataloaders with TEMPORAL SPLIT to prevent data leakage...")
    
    # Memory management: Apply early sampling if max_samples is specified
    if max_samples is not None:
        print(f"💾 Memory management: Will limit to {max_samples:,} samples to prevent OOM")
        # Adjust the dataset to load fewer samples
        kwargs['max_samples'] = max_samples
    
    # Step 1: Load RAW data ONCE (before creating sequences)
    print("📁 Loading raw data...")
    temp_dataset = CSIDatasetFixed(
        mat_file_path=mat_file_path,
        sequence_length=1,  # Load as individual samples first
        target_tags=target_tags,
        use_magnitude_phase=use_magnitude_phase,
        stride=1,  # No stride for raw data
        **kwargs
    )
    
    # Extract raw features and targets
    raw_features = temp_dataset.raw_features
    raw_targets = temp_dataset.raw_targets
    
    print(f"📊 Loaded {len(raw_features):,} raw samples")
    print(f"📐 Feature shape: {raw_features.shape}")
    print(f"📍 Target shape: {raw_targets.shape}")
    print(f"💾 Memory usage: {raw_features.nbytes / 1024**2:.1f} MB")
    
    # Clear temporary dataset to free memory
    del temp_dataset
    
    # Step 2: TEMPORAL SPLIT of raw data (no random permutation!)
    n_samples = len(raw_features)
    train_end_idx = int(n_samples * train_split)
    
    # Add temporal gap to prevent leakage at the boundary
    val_start_idx = train_end_idx + temporal_gap
    
    print(f"📅 Temporal split: Train[0:{train_end_idx}], Gap[{train_end_idx}:{val_start_idx}], Val[{val_start_idx}:]")
    
    # Split raw data temporally
    train_raw_features = raw_features[:train_end_idx]
    train_raw_targets = raw_targets[:train_end_idx]
    val_raw_features = raw_features[val_start_idx:]
    val_raw_targets = raw_targets[val_start_idx:]
    
    # Clear original data to free memory
    del raw_features, raw_targets
    
    print(f"🔄 Raw data split: {len(train_raw_features):,} train, {len(val_raw_features):,} val samples")
    
    # Step 3: Create sequences SEPARATELY for train and val (no overlap possible!)
    def create_sequences_from_raw(features, targets, seq_len, stride):
        """Create sequences from raw data with sliding window."""
        sequences = []
        for start_idx in range(0, len(features) - seq_len + 1, stride):
            end_idx = start_idx + seq_len
            seq_features = features[start_idx:end_idx]
            seq_targets = targets[start_idx:end_idx]
            # Use last target as sequence target
            target = seq_targets[-1]
            sequences.append((seq_features, target))
        return sequences
    
    # Create train sequences
    print("🔄 Creating training sequences...")
    train_sequences = create_sequences_from_raw(
        train_raw_features, train_raw_targets, 
        sequence_length, kwargs.get('stride', 2)
    )
    
    # Clear train raw data to free memory
    del train_raw_features, train_raw_targets
    
    # Create val sequences  
    print("🔄 Creating validation sequences...")
    val_sequences = create_sequences_from_raw(
        val_raw_features, val_raw_targets,
        sequence_length, kwargs.get('stride', 2)
    )
    
    # Clear val raw data to free memory
    del val_raw_features, val_raw_targets
    
    print(f"✅ Sequences created: {len(train_sequences):,} train, {len(val_sequences):,} val")
    print(f"📊 NO OVERLAP guaranteed - temporal split prevents data leakage!")
    
    # Step 4: Extract features and targets from sequences
    train_features = np.array([seq[0] for seq in train_sequences])
    train_targets = np.array([seq[1] for seq in train_sequences])
    val_features = np.array([seq[0] for seq in val_sequences])
    val_targets = np.array([seq[1] for seq in val_sequences])
    
    print(f"📏 Final shapes:")
    print(f"  Train features: {train_features.shape}, targets: {train_targets.shape}")
    print(f"  Val features: {val_features.shape}, targets: {val_targets.shape}")
    
    # Step 5: Fit scalers ONLY on training data
    print("🔧 Fitting scalers on training data only...")
    
    # Reshape training features for scaler fitting: (n_train_seqs * seq_len, n_features)
    train_features_flat = train_features.reshape(-1, train_features.shape[-1])
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    feature_scaler.fit(train_features_flat)
    target_scaler.fit(train_targets)
    
    print(f"📐 Feature scaler fitted on {train_features_flat.shape[0]} samples")
    print(f"📍 Target scaler fitted on {train_targets.shape[0]} samples")
    
    # Step 6: Apply scalers to both train and val data
    print("⚖️ Applying scalers to train and validation data...")
    
    # Scale training features
    train_features_scaled = train_features.copy()
    for i in range(len(train_features_scaled)):
        train_features_scaled[i] = feature_scaler.transform(train_features[i])
    
    # Scale validation features
    val_features_scaled = val_features.copy()
    for i in range(len(val_features_scaled)):
        val_features_scaled[i] = feature_scaler.transform(val_features[i])
    
    # Scale targets
    train_targets_scaled = target_scaler.transform(train_targets)
    val_targets_scaled = target_scaler.transform(val_targets)
    
    # Step 7: Create tensor datasets
    train_features_tensor = torch.FloatTensor(train_features_scaled)
    train_targets_tensor = torch.FloatTensor(train_targets_scaled)
    val_features_tensor = torch.FloatTensor(val_features_scaled)
    val_targets_tensor = torch.FloatTensor(val_targets_scaled)
    
    train_dataset = TensorDataset(train_features_tensor, train_targets_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_targets_tensor)
    
    # Step 8: Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ Created dataloaders: Train={len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"                       Val={len(val_loader)} batches ({len(val_dataset)} samples)")
    print("🎉 LEAKAGE-FREE dataloader creation: Temporal split prevents overlapping sequences!")
    
    # Verification: Check that no sequences overlap
    if len(train_sequences) > 0 and len(val_sequences) > 0:
        last_train_end = train_end_idx
        first_val_start = val_start_idx  
        gap_size = first_val_start - last_train_end
        print(f"🔒 Data integrity check: {gap_size} sample gap between train and val prevents any overlap")
    
    return train_loader, val_loader, feature_scaler, target_scaler


def save_csi_scalers_fixed(feature_scaler: StandardScaler, target_scaler: StandardScaler, save_path: str):
    """Save CSI scalers to file."""
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    with open(save_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"💾 Saved CSI scalers to {save_path}")


def load_csi_scalers_fixed(load_path: str) -> Tuple[StandardScaler, StandardScaler]:
    """Load CSI scalers from file."""
    with open(load_path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"📥 Loaded CSI scalers from {load_path}")
    return scalers['feature_scaler'], scalers['target_scaler'] 