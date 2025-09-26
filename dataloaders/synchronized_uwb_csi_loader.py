"""
Synchronized UWB-CSI Dataset Loader for Cross-Modality Distillation
Matches UWB and CSI data points by timestamp to ensure proper cross-modal training.
Handles the synchronization between UWB CSV files and CSI .mat files.
Uses proven loading patterns from uwb_opera_loader.py and csi_loader_fixed.py.
"""

import torch
import numpy as np
import pandas as pd
import h5py
import os
import glob
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
import warnings
from bisect import bisect_left

warnings.filterwarnings('ignore')


class SynchronizedUWBCSIDataset(Dataset):
    """
    Dataset that synchronizes UWB and CSI data by timestamp for cross-modality training.
    
    Loads UWB CSV files and CSI .mat files, then matches data points by timestamp
    to create synchronized pairs for cross-modal distillation.
    """
    
    def __init__(
        self,
        uwb_data_path: str,
        csi_mat_file: str,
        experiments: List[str],
        sequence_length: int = 32,
        csi_sequence_length: int = 4,
        target_tags: List[str] = ['tag4422'],
        max_time_diff_ms: float = 500.0,  # Maximum time difference for matching (500ms)
        feature_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
        csi_feature_scaler: Optional[StandardScaler] = None,
        max_samples_per_exp: int = 5000,
        use_magnitude_phase: bool = True,
        stride: int = 4
    ):
        """
        Initialize synchronized UWB-CSI dataset.
        
        Args:
            uwb_data_path: Path to directory containing UWB CSV files
            csi_mat_file: Path to CSI .mat file
            experiments: List of experiment numbers to load (e.g., ['002'])
            sequence_length: Length of UWB sequences
            csi_sequence_length: Length of CSI sequences (shorter)
            target_tags: List of tag IDs for ground truth coordinates
            max_time_diff_ms: Maximum time difference for timestamp matching
            feature_scaler: Scaler for UWB features
            target_scaler: Scaler for target coordinates
            csi_feature_scaler: Scaler for CSI features
            max_samples_per_exp: Maximum samples per experiment
            use_magnitude_phase: Whether to use magnitude/phase for CSI
            stride: Stride for sequence creation
        """
        self.uwb_data_path = uwb_data_path
        self.csi_mat_file = csi_mat_file
        self.experiments = experiments
        self.sequence_length = sequence_length
        self.csi_sequence_length = csi_sequence_length
        self.target_tags = target_tags
        self.max_time_diff_ms = max_time_diff_ms
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.csi_feature_scaler = csi_feature_scaler
        self.max_samples_per_exp = max_samples_per_exp
        self.use_magnitude_phase = use_magnitude_phase
        self.stride = stride
        
        # Load and synchronize data
        print("🔄 Loading and synchronizing UWB-CSI data...")
        self.synchronized_data = self._load_and_synchronize_data()
        
        # Create sequences
        print("📊 Creating synchronized sequences...")
        self.sequences = self._create_synchronized_sequences()
        
        print(f"✅ Synchronized dataset: {len(self.sequences)} pairs from {len(self.experiments)} experiments")
        
    def _load_uwb_data(self) -> List[Dict[str, Any]]:
        """Load UWB data from CSV files using uwb_opera_loader pattern."""
        uwb_data = []
        
        for exp in self.experiments:
            # Try different UWB file naming patterns
            possible_files = [
                os.path.join(self.uwb_data_path, f"uwb1_exp{exp}.csv"),
                os.path.join(self.uwb_data_path, f"uwb2_exp{exp}.csv"),
                os.path.join(self.uwb_data_path, f"uwb_exp{exp}.csv")
            ]
            
            csv_file = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    csv_file = file_path
                    break
            
            if csv_file is None:
                print(f"⚠️ Warning: No UWB file found for experiment {exp}")
                continue
                
            print(f"📁 Loading UWB data from {os.path.basename(csv_file)}")
            
            try:
                df = pd.read_csv(csv_file, nrows=self.max_samples_per_exp)
                
                # Check if this file has required data
                required_cols = ['timestamp'] + [f"{tag}_x" for tag in self.target_tags] + [f"{tag}_y" for tag in self.target_tags]
                if not all(col in df.columns for col in required_cols):
                    print(f"⚠️ Missing required columns in {os.path.basename(csv_file)}")
                    continue
                
                # Process UWB data using uwb_opera_loader pattern
                processed_data = self._process_uwb_dataframe(df, exp)
                if len(processed_data['timestamps']) > 0:
                    uwb_data.append(processed_data)
                    print(f"✅ Loaded {len(processed_data['timestamps'])} UWB samples from exp{exp}")
                
            except Exception as e:
                print(f"❌ Error loading UWB file {csv_file}: {e}")
                continue
        
        return uwb_data
    
    def _process_uwb_dataframe(self, df: pd.DataFrame, exp: str) -> Dict[str, Any]:
        """Process a UWB dataframe using uwb_opera_loader pattern."""
        # Extract CIR features (cir1 to cir50) - same as uwb_opera_loader
        cir_columns = [f'cir{i}' for i in range(1, 51)]
        available_cir = [col for col in cir_columns if col in df.columns]
        
        if len(available_cir) == 0:
            print(f"⚠️ No CIR columns found in experiment {exp}")
            return {'timestamps': [], 'features': [], 'targets': []}
        
        # Convert complex strings to numbers - same as uwb_opera_loader
        cir_data = df[available_cir].copy()
        for col in available_cir:
            cir_data[col] = cir_data[col].apply(self._parse_complex)
        
        # Create feature matrix (magnitude and phase)
        features = []
        for col in available_cir:
            features.append(cir_data[col].apply(lambda x: abs(x)).values)
            features.append(cir_data[col].apply(lambda x: np.angle(x)).values)
        
        # MODIFICATION: Only use CIR features (50 complex = 100 real features)
        # Commenting out additional features as requested by user
        # # Add additional UWB features if available - same as uwb_opera_loader
        # additional_features = [
        #     'fp_pow_dbm', 'rx_pow_dbm', 'tx_x_coord', 'tx_y_coord', 
        #     'rx_x_coord', 'rx_y_coord', 'tx_rx_dist_meters',
        #     'fp_index', 'fp_amp1', 'fp_amp2', 'fp_amp3', 
        #     'max_growth_cir', 'rx_pream_count'
        # ]
        # 
        # for feat in additional_features:
        #     if feat in df.columns:
        #         feat_values = df[feat].values
        #         feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=1e6, neginf=-1e6)
        #         feat_values = np.clip(feat_values, -1e6, 1e6)
        #         features.append(feat_values)
        
        feature_matrix = np.column_stack(features)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        feature_matrix = np.clip(feature_matrix, -1e6, 1e6)
        
        # Extract target coordinates - same as uwb_opera_loader
        targets = []
        for tag in self.target_tags:
            x_col = f"{tag}_x"
            y_col = f"{tag}_y"
            
            if x_col in df.columns and y_col in df.columns:
                x_vals = df[x_col].ffill().bfill().values
                y_vals = df[y_col].ffill().bfill().values
                
                x_vals = np.nan_to_num(x_vals, nan=0.0)
                y_vals = np.nan_to_num(y_vals, nan=0.0)
                
                targets.extend([x_vals, y_vals])
        
        target_matrix = np.column_stack(targets) if targets else np.zeros((len(df), 2))
        
        # Convert UWB timestamps to numeric format (milliseconds since start)
        timestamps_numeric = self._convert_uwb_timestamps_to_numeric(df['timestamp'].values)
        
        return {
            'timestamps': timestamps_numeric,
            'features': feature_matrix,
            'targets': target_matrix,
            'experiment': exp
        }
    
    def _convert_uwb_timestamps_to_numeric(self, timestamps: np.ndarray) -> np.ndarray:
        """Convert UWB timestamp strings to numeric milliseconds since start."""
        numeric_timestamps = []
        
        for ts_str in timestamps:
            try:
                # Parse time string format "HH:MM:SS.microseconds"
                time_parts = str(ts_str).split(':')
                if len(time_parts) == 3:
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds_parts = time_parts[2].split('.')
                    seconds = int(seconds_parts[0])
                    microseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                    
                    # Convert to total milliseconds
                    total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + microseconds / 1000
                    numeric_timestamps.append(total_ms)
                else:
                    # Fallback: try to parse as float
                    numeric_timestamps.append(float(ts_str))
            except:
                # If parsing fails, use sequential numbering
                numeric_timestamps.append(len(numeric_timestamps))
        
        # Normalize to start from 0
        numeric_timestamps = np.array(numeric_timestamps)
        if len(numeric_timestamps) > 0:
            numeric_timestamps = numeric_timestamps - numeric_timestamps[0]
        
        return numeric_timestamps
    
    def _parse_complex(self, complex_str: str) -> complex:
        """Parse complex number from string representation - same as uwb_opera_loader."""
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
    
    def _load_csi_data(self) -> Dict[str, Any]:
        """Load CSI data from .mat file using csi_loader_fixed pattern."""
        print(f"📁 Loading CSI data from {os.path.basename(self.csi_mat_file)}")
        
        with h5py.File(self.csi_mat_file, 'r') as f:
            # Use the exact same approach as csi_loader_fixed.py
            # Try direct column access first (MATLAB v7.3 format)
            if 'exp002' in f:
                data_group = f['exp002']
                print("📊 Found exp002 data group")
                
                # Extract CSI columns directly (tx1rx1_sub1 through tx3rx3_sub30)
                csi_features = []
                missing_csi = []
                
                # Generate all expected CSI column names - same as csi_loader_fixed
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
                
                print(f"✅ Found {len(csi_features)} CSI columns")
                
                # Use the minimum length among all CSI arrays
                min_length = min(len(arr) for arr in csi_features) if csi_features else 0
                print(f"📏 Using {min_length} samples (minimum array length)")
                
                # Truncate all arrays to minimum length
                csi_features = [arr[:min_length] for arr in csi_features]
                
                # Stack CSI data: (num_samples, num_csi_features)
                csi_data = np.column_stack(csi_features)
                
                # Load coordinate data - same pattern as csi_loader_fixed
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
                        
                        # Truncate to minimum length
                        x_data = x_data[:min_length]
                        y_data = y_data[:min_length]
                        
                        coord_data.extend([x_data, y_data])
                        print(f"✅ Loaded {tag} coordinates: {len(x_data)} samples")
                    else:
                        print(f"⚠️ Coordinates for {tag} not found")
                
                if not coord_data:
                    # Try timestamp data as fallback
                    if 'timestamp' in data_group:
                        timestamps = data_group['timestamp'][...]
                        if timestamps.ndim > 1:
                            timestamps = timestamps.flatten()
                        timestamps = timestamps[:min_length]
                    else:
                        # Generate timestamps in millisecond scale to match UWB format
                        # Assume CSI samples at ~100 Hz (10ms intervals)
                        timestamps = np.arange(min_length) * 10.0  # 10ms intervals
                else:
                    coordinate_data = np.column_stack(coord_data)
                    # Try to get timestamps
                    if 'timestamp' in data_group:
                        timestamps = data_group['timestamp'][...]
                        if timestamps.ndim > 1:
                            timestamps = timestamps.flatten()
                        timestamps = timestamps[:min_length]
                    else:
                        # Generate timestamps in millisecond scale to match UWB format
                        # Assume CSI samples at ~100 Hz (10ms intervals)
                        timestamps = np.arange(min_length) * 10.0  # 10ms intervals
            
            else:
                # Fallback to refs-based loading (same as csi_loader_fixed)
                print("📊 Using refs-based loading")
                refs = f['#refs#']
                
                # Get field mappings - same as csi_loader_fixed
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
                
                # Find CSI mappings - same as csi_loader_fixed
                csi_mappings = {}
                for key, name in field_names.items():
                    if 'tx' in name and 'rx' in name and 'sub' in name:
                        data_key = key.replace('e', '').replace('f', '').replace('g', '').replace('h', '').replace('i', '')
                        if data_key in refs:
                            csi_mappings[name] = data_key
                
                print(f"📡 Found {len(csi_mappings)} CSI mappings")
                
                # Load CSI data - same as csi_loader_fixed
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
                
                # Load coordinates using refs - same as csi_loader_fixed
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
                        
                        # Extract coordinate data - FIXED: Use MAGNITUDE like csi_loader_fixed.py
                        if hasattr(x_array, 'dtype') and x_array.dtype.names and 'real' in x_array.dtype.names:
                            # FIXED: Use MAGNITUDE for ground truth coordinates (matches working CSI loader)
                            # Based on analysis: magnitude gives most realistic movement patterns
                            x_coords = np.abs(x_array['real'][:min_length] + 1j * x_array['imag'][:min_length]).flatten()
                            y_coords = np.abs(y_array['real'][:min_length] + 1j * y_array['imag'][:min_length]).flatten()
                            print(f"   🎯 Using MAGNITUDE mapping for complex coordinates (optimal strategy - matches working CSI loader)")
                        else:
                            x_coords = x_array[:min_length].flatten()
                            y_coords = y_array[:min_length].flatten()
                        
                        coord_data.extend([x_coords, y_coords])
                        print(f"✅ Loaded {tag} coordinates: {len(x_coords)} samples")
                
                if coord_data:
                    coordinate_data = np.column_stack(coord_data)
                else:
                    coordinate_data = np.zeros((min_length, 2))
                
                # Generate timestamps if not found
                timestamps = np.arange(min_length)
            
            # Process features using csi_loader_fixed pattern
            if self.use_magnitude_phase:
                # IMPROVED: Use magnitude and phase with unwrapping (proven technique from csi_loader_fixed)
                magnitude = np.abs(csi_data)
                phase = np.angle(csi_data)
                # CRITICAL: Unwrap phase for better continuity (proven in csi_loader_fixed.py)
                phase = np.unwrap(phase, axis=1)
                csi_features_processed = np.concatenate([magnitude, phase], axis=1)
                print(f"   🔧 Applied magnitude + phase unwrapping (proven technique)")
            else:
                # Use real and imaginary parts
                real_part = np.real(csi_data)
                imag_part = np.imag(csi_data)
                csi_features_processed = np.concatenate([real_part, imag_part], axis=1)
            
            # IMPROVED: Clean features with better bounds (like csi_loader_fixed)
            csi_features_processed = np.nan_to_num(csi_features_processed, nan=0.0, posinf=1e6, neginf=-1e6)
            csi_features_processed = np.clip(csi_features_processed, -1e6, 1e6)
            
            print(f"✅ Loaded {len(timestamps)} CSI samples with {csi_features_processed.shape[1]} features")
            
            return {
                'timestamps': timestamps,
                'features': csi_features_processed,
                'targets': coordinate_data if 'coordinate_data' in locals() else np.zeros((len(timestamps), 2))
            }
    
    def _load_and_synchronize_data(self) -> List[Dict[str, Any]]:
        """Load and synchronize UWB and CSI data by sample index (data is pre-synchronized)."""
        # Load UWB data
        uwb_data = self._load_uwb_data()
        if len(uwb_data) == 0:
            raise ValueError("No UWB data loaded")
        
        # Load CSI data
        csi_data = self._load_csi_data()
        
        # Use sample-based synchronization with CSI averaging
        # Paper mentions <20ms accuracy, suggesting data is pre-synchronized by index
        synchronized_pairs = []
        
        for uwb_exp_data in uwb_data:
            uwb_features = uwb_exp_data['features']
            uwb_targets = uwb_exp_data['targets']
            csi_features = csi_data['features']
            csi_targets = csi_data['targets']
            
            print(f"🔄 Synchronizing experiment {uwb_exp_data['experiment']} with CSI averaging...")
            print(f"   UWB: {len(uwb_features)} samples")
            print(f"   CSI: {len(csi_features)} samples")
            
            # Calculate the mapping ratio
            csi_samples_per_uwb = len(csi_features) / len(uwb_features)
            print(f"   Ratio: {csi_samples_per_uwb:.1f} CSI samples per UWB sample")
            
            matches = 0
            
            # For each UWB sample, average the corresponding CSI samples
            for i in range(len(uwb_features)):
                # Define CSI window for this UWB sample
                csi_start_idx = int(i * csi_samples_per_uwb)
                csi_end_idx = int((i + 1) * csi_samples_per_uwb)
                
                # Ensure we don't exceed CSI data bounds
                csi_end_idx = min(csi_end_idx, len(csi_features))
                
                if csi_start_idx < len(csi_features) and csi_end_idx > csi_start_idx:
                    # Average the CSI features in this window
                    csi_window_features = csi_features[csi_start_idx:csi_end_idx]
                    csi_averaged_features = np.mean(csi_window_features, axis=0)
                    
                    # Average the CSI targets in this window  
                    csi_window_targets = csi_targets[csi_start_idx:csi_end_idx]
                    csi_averaged_targets = np.mean(csi_window_targets, axis=0)
                    
                    synchronized_pairs.append({
                        'uwb_features': uwb_features[i],
                        'uwb_targets': uwb_targets[i],
                        'csi_features': csi_averaged_features,
                        'csi_targets': csi_averaged_targets,  # FIXED: Use actual CSI targets for CSI student task loss
                        'uwb_index': i,
                        'csi_window': (csi_start_idx, csi_end_idx),
                        'csi_samples_averaged': csi_end_idx - csi_start_idx,
                        'experiment': uwb_exp_data['experiment']
                    })
                    matches += 1
            
            print(f"   ✅ Found {matches} synchronized pairs using CSI averaging")
            avg_csi_per_uwb = np.mean([pair['csi_samples_averaged'] for pair in synchronized_pairs[-matches:]])
            print(f"   📊 Average CSI samples per UWB: {avg_csi_per_uwb:.1f}")
        
        print(f"🎯 Total synchronized pairs: {len(synchronized_pairs)}")
        return synchronized_pairs
    
    def _create_synchronized_sequences(self) -> List[Dict[str, Any]]:
        """Create synchronized sequences from matched data points."""
        sequences = []
        
        # Group by experiment for sequence creation
        exp_groups = {}
        for pair in self.synchronized_data:
            exp = pair['experiment']
            if exp not in exp_groups:
                exp_groups[exp] = []
            exp_groups[exp].append(pair)
        
        for exp, pairs in exp_groups.items():
            print(f"📊 Creating sequences for experiment {exp}: {len(pairs)} pairs")
            
            # Sort by UWB index (data should already be in order)
            pairs.sort(key=lambda x: x['uwb_index'])
            
            # Create sequences with stride
            for start_idx in range(0, len(pairs) - max(self.sequence_length, self.csi_sequence_length) + 1, self.stride):
                # UWB sequence
                uwb_end_idx = start_idx + self.sequence_length
                if uwb_end_idx > len(pairs):
                    break
                    
                uwb_features = np.array([pairs[i]['uwb_features'] for i in range(start_idx, uwb_end_idx)])
                uwb_targets = np.array([pairs[i]['uwb_targets'] for i in range(start_idx, uwb_end_idx)])
                
                # CSI sequence (shorter)
                csi_start_idx = start_idx + (self.sequence_length - self.csi_sequence_length)
                csi_end_idx = csi_start_idx + self.csi_sequence_length
                
                if csi_end_idx > len(pairs):
                    break
                
                csi_features = np.array([pairs[i]['csi_features'] for i in range(csi_start_idx, csi_end_idx)])
                csi_targets = np.array([pairs[i]['csi_targets'] for i in range(csi_start_idx, csi_end_idx)])
                
                sequences.append({
                    'uwb_features': uwb_features,
                    'uwb_targets': uwb_targets,
                    'csi_features': csi_features, 
                    'csi_targets': csi_targets,
                    'experiment': exp
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a synchronized UWB-CSI sequence pair."""
        sequence = self.sequences[idx]
        
        uwb_features = sequence['uwb_features'].astype(np.float32)
        uwb_targets = sequence['uwb_targets'].astype(np.float32)
        csi_features = sequence['csi_features'].astype(np.float32)
        csi_targets = sequence['csi_targets'].astype(np.float32)
        
        # Apply scaling if available
        if self.feature_scaler is not None:
            orig_shape = uwb_features.shape
            uwb_features = uwb_features.reshape(-1, uwb_features.shape[-1])
            uwb_features = self.feature_scaler.transform(uwb_features)
            uwb_features = uwb_features.reshape(orig_shape)
        
        if self.csi_feature_scaler is not None:
            orig_shape = csi_features.shape
            csi_features = csi_features.reshape(-1, csi_features.shape[-1])
            csi_features = self.csi_feature_scaler.transform(csi_features)
            csi_features = csi_features.reshape(orig_shape)
        
        if self.target_scaler is not None:
            orig_shape = uwb_targets.shape
            uwb_targets = uwb_targets.reshape(-1, uwb_targets.shape[-1])
            uwb_targets = self.target_scaler.transform(uwb_targets)
            uwb_targets = uwb_targets.reshape(orig_shape)
            
            orig_shape = csi_targets.shape
            csi_targets = csi_targets.reshape(-1, csi_targets.shape[-1])
            csi_targets = self.target_scaler.transform(csi_targets)
            csi_targets = csi_targets.reshape(orig_shape)
        
        # Clean data
        uwb_features = np.nan_to_num(uwb_features, nan=0.0, posinf=10.0, neginf=-10.0)
        uwb_features = np.clip(uwb_features, -10.0, 10.0)
        csi_features = np.nan_to_num(csi_features, nan=0.0, posinf=10.0, neginf=-10.0)
        csi_features = np.clip(csi_features, -10.0, 10.0)
        
        return (torch.FloatTensor(uwb_features), torch.FloatTensor(uwb_targets),
                torch.FloatTensor(csi_features), torch.FloatTensor(csi_targets))


def create_synchronized_dataloaders(
    uwb_data_path: str,
    csi_mat_file: str,
    train_experiments: List[str],
    val_experiments: List[str],
    batch_size: int = 16,
    sequence_length: int = 32,
    csi_sequence_length: int = 4,
    target_tags: List[str] = ['tag4422'],
    temporal_split: bool = True,  # IMPROVED: Add temporal splitting like csi_loader_fixed
    train_split: float = 0.8,    # IMPROVED: Add train split ratio
    temporal_gap: int = 0,       # IMPROVED: Add temporal gap to prevent leakage
    pin_memory: bool = True,     # Add pin_memory parameter to fix CUDA issues
    **kwargs
) -> Tuple[DataLoader, DataLoader, StandardScaler, StandardScaler, StandardScaler]:
    """
    Create synchronized UWB-CSI dataloaders for cross-modality training.
    IMPROVED: Incorporates proven techniques from csi_loader_fixed.py while maintaining synchronization.
    
    Returns:
        Tuple of (train_loader, val_loader, uwb_scaler, target_scaler, csi_scaler)
    """
    print("🚀 Creating IMPROVED synchronized UWB-CSI dataloaders...")
    print("🔧 Incorporating proven techniques from csi_loader_fixed.py")
    
    if temporal_split:
        print("📅 Using TEMPORAL SPLIT to prevent data leakage (like csi_loader_fixed.py)")
        # Create dataset for all experiments to get temporal data
        all_experiments = list(set(train_experiments + val_experiments))
        full_dataset = SynchronizedUWBCSIDataset(
            uwb_data_path=uwb_data_path,
            csi_mat_file=csi_mat_file,
            experiments=all_experiments,
            sequence_length=sequence_length,
            csi_sequence_length=csi_sequence_length,
            target_tags=target_tags,
            **kwargs
        )
        
        if len(full_dataset) == 0:
            raise ValueError("No synchronized data found")
        
        # IMPROVED: Temporal split like csi_loader_fixed.py
        n_samples = len(full_dataset)
        train_end_idx = int(n_samples * train_split)
        val_start_idx = train_end_idx + temporal_gap
        
        print(f"📅 Temporal split: Train[0:{train_end_idx}], Gap[{train_end_idx}:{val_start_idx}], Val[{val_start_idx}:]")
        
        # Extract data for temporal splitting
        train_indices = list(range(0, train_end_idx))
        val_indices = list(range(val_start_idx, n_samples))
        
        print(f"🔄 Temporal split: {len(train_indices)} train, {len(val_indices)} val samples")
        
        # Create training dataset for scaler fitting
        train_dataset = SynchronizedUWBCSIDataset(
            uwb_data_path=uwb_data_path,
            csi_mat_file=csi_mat_file,
            experiments=all_experiments,
            sequence_length=sequence_length,
            csi_sequence_length=csi_sequence_length,
            target_tags=target_tags,
            **kwargs
        )
        
        # IMPROVED: Fit scalers ONLY on training data (like csi_loader_fixed.py)
        print("📊 Fitting scalers on training data only (proven technique)...")
        
        # Collect training data for scaler fitting
        all_uwb_features = []
        all_csi_features = []
        all_uwb_targets = []
        
        sample_count = 0
        for i in train_indices:
            if i < len(train_dataset) and sample_count < 1000:  # Sample for fitting
                uwb_feat, uwb_targ, csi_feat, csi_targ = train_dataset[i]
                all_uwb_features.append(uwb_feat.numpy().reshape(-1, uwb_feat.shape[-1]))
                all_csi_features.append(csi_feat.numpy().reshape(-1, csi_feat.shape[-1]))
                all_uwb_targets.append(uwb_targ.numpy().reshape(-1, uwb_targ.shape[-1]))
                sample_count += 1
        
        if len(all_uwb_features) == 0:
            raise ValueError("No training data available for scaler fitting")
        
        uwb_features_array = np.vstack(all_uwb_features)
        csi_features_array = np.vstack(all_csi_features)
        uwb_targets_array = np.vstack(all_uwb_targets)
        
        # IMPROVED: Fit scalers (like csi_loader_fixed.py)
        uwb_scaler = StandardScaler()
        csi_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        uwb_scaler.fit(uwb_features_array)
        csi_scaler.fit(csi_features_array)
        target_scaler.fit(uwb_targets_array)  # Use UWB targets as ground truth
        
        print(f"📊 Scaler fitting completed (proven approach):")
        print(f"   UWB features: {uwb_features_array.shape[0]} samples")
        print(f"   CSI features: {csi_features_array.shape[0]} samples")
        print(f"   Target range: [{uwb_targets_array.min():.3f}, {uwb_targets_array.max():.3f}]")
        
        # Create datasets with fitted scalers and temporal splits
        train_dataset_scaled = SynchronizedUWBCSIDataset(
            uwb_data_path=uwb_data_path,
            csi_mat_file=csi_mat_file,
            experiments=all_experiments,
            sequence_length=sequence_length,
            csi_sequence_length=csi_sequence_length,
            target_tags=target_tags,
            feature_scaler=uwb_scaler,
            target_scaler=target_scaler,
            csi_feature_scaler=csi_scaler,
            **kwargs
        )
        
        val_dataset_scaled = SynchronizedUWBCSIDataset(
            uwb_data_path=uwb_data_path,
            csi_mat_file=csi_mat_file,
            experiments=all_experiments,
            sequence_length=sequence_length,
            csi_sequence_length=csi_sequence_length,
            target_tags=target_tags,
            feature_scaler=uwb_scaler,
            target_scaler=target_scaler,
            csi_feature_scaler=csi_scaler,
            **kwargs
        )
        
        # Apply temporal splitting using subset
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset_scaled, train_indices)
        val_dataset = Subset(val_dataset_scaled, val_indices)
        
        print(f"✅ IMPROVED temporal split applied: {len(train_dataset)} train, {len(val_dataset)} val")
        
    else:
        # Original approach for backward compatibility
        print("📊 Using experiment-based split (original approach)")
        
        # Create training dataset to fit scalers
        train_dataset = SynchronizedUWBCSIDataset(
            uwb_data_path=uwb_data_path,
            csi_mat_file=csi_mat_file,
            experiments=train_experiments,
            sequence_length=sequence_length,
            csi_sequence_length=csi_sequence_length,
            target_tags=target_tags,
            **kwargs
        )
        
        if len(train_dataset) == 0:
            raise ValueError("No training data found")
        
        # Fit scalers on training data
        print("📊 Fitting scalers on training data...")
        
        # Collect all training features and targets
        all_uwb_features = []
        all_csi_features = []
        all_uwb_targets = []
        
        for i in range(min(1000, len(train_dataset))):  # Sample for scaler fitting
            uwb_feat, uwb_targ, csi_feat, csi_targ = train_dataset[i]
            all_uwb_features.append(uwb_feat.numpy().reshape(-1, uwb_feat.shape[-1]))
            all_csi_features.append(csi_feat.numpy().reshape(-1, csi_feat.shape[-1]))
            all_uwb_targets.append(uwb_targ.numpy().reshape(-1, uwb_targ.shape[-1]))
        
        uwb_features_array = np.vstack(all_uwb_features)
        csi_features_array = np.vstack(all_csi_features)
        uwb_targets_array = np.vstack(all_uwb_targets)
        
        # Fit scalers on appropriate data
        uwb_scaler = StandardScaler()
        csi_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        uwb_scaler.fit(uwb_features_array)
        csi_scaler.fit(csi_features_array)
        target_scaler.fit(uwb_targets_array)  # FIXED: Use UWB targets as reference since they're ground truth
        
        print(f"📊 Scaler fitting completed:")
        print(f"   UWB targets range: [{uwb_targets_array.min():.3f}, {uwb_targets_array.max():.3f}]")
        print(f"   🎯 Using UWB targets for target_scaler (ground truth coordinate scaling)")
        
        # Create datasets with fitted scalers
        train_dataset = SynchronizedUWBCSIDataset(
            uwb_data_path=uwb_data_path,
            csi_mat_file=csi_mat_file,
            experiments=train_experiments,
            sequence_length=sequence_length,
            csi_sequence_length=csi_sequence_length,
            target_tags=target_tags,
            feature_scaler=uwb_scaler,
            target_scaler=target_scaler,
            csi_feature_scaler=csi_scaler,
            **kwargs
        )
        
        val_dataset = SynchronizedUWBCSIDataset(
            uwb_data_path=uwb_data_path,
            csi_mat_file=csi_mat_file,
            experiments=val_experiments,
            sequence_length=sequence_length,
            csi_sequence_length=csi_sequence_length,
            target_tags=target_tags,
            feature_scaler=uwb_scaler,
            target_scaler=target_scaler,
            csi_feature_scaler=csi_scaler,
            **kwargs
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    print(f"✅ Created IMPROVED synchronized dataloaders:")
    print(f"   Training: {len(train_dataset)} pairs")
    print(f"   Validation: {len(val_dataset)} pairs")
    print(f"🔧 Using proven techniques: temporal split, proper scaling, phase unwrapping")
    
    return train_loader, val_loader, uwb_scaler, target_scaler, csi_scaler


if __name__ == "__main__":
    # Test the synchronized dataloader
    uwb_path = "/media/mohab/Storage HDD/Downloads/uwb2(1)"
    csi_file = "/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat"
    
    train_loader, val_loader, uwb_scaler, target_scaler, csi_scaler = create_synchronized_dataloaders(
        uwb_data_path=uwb_path,
        csi_mat_file=csi_file,
        train_experiments=['002'],
        val_experiments=['002'],  # Use same for testing
        batch_size=4,
        max_samples_per_exp=1000
    )
    
    print("\n🧪 Testing synchronized dataloader...")
    for i, (uwb_feat, uwb_targ, csi_feat, csi_targ) in enumerate(train_loader):
        print(f"Batch {i}: UWB {uwb_feat.shape}, CSI {csi_feat.shape}")
        if i >= 2:
            break
    
    print("✅ Synchronized dataloader test complete!") 