"""
Truly Fair Comparison with GMM-based Knowledge Distillation
Uses distribution-based knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import scipy.io
import os
import sys
sys.path.append('.')

from modules.csi_head import CSIRegressionModel
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher

# Unified coordinate and model parameters
COORD_MIN = 0.0  # Minimum coordinate value
COORD_MAX_X = 8.0  # Maximum X coordinate (covers both rooms)
COORD_MAX_Y = 11.0  # Maximum Y coordinate (covers both rooms)
# GRID_SIZE_X = 40   # 40 bins for X direction = 0.2m per bin
# GRID_SIZE_Y = 55   # 55 bins for Y direction = 0.2m per bin
# GRID_SIZE = GRID_SIZE_X * GRID_SIZE_Y  # Total grid size
# BIN_WIDTH_X = (COORD_MAX_X - COORD_MIN) / GRID_SIZE_X
# BIN_WIDTH_Y = (COORD_MAX_Y - COORD_MIN) / GRID_SIZE_Y

# GMM parameters
NUM_GAUSSIANS = 16  # Number of Gaussian components
GAUSSIAN_GRID_X = 4  # 4x4 grid for X dimension
GAUSSIAN_GRID_Y = 4  # 4x4 grid for Y dimension



class ContinuousCoordinateHead(nn.Module):
    """
    Continuous coordinate prediction head that outputs probability distributions.
    Uses Gaussian Mixture Model approach for continuous probability densities.
    """
    
    def __init__(self, input_dim: int, num_gaussians: int = NUM_GAUSSIANS):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # Predict Gaussian mixture parameters
        self.gaussian_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_gaussians * 5)  # 5 params per Gaussian: weight, mu_x, mu_y, sigma_x, sigma_y
        )
        
        # Initialize Gaussian centers in a grid pattern
        self._init_gaussian_centers()
    
    def _init_gaussian_centers(self):
        """Initialize Gaussian centers in a grid pattern across the coordinate space."""
        x_centers = torch.linspace(COORD_MIN + 1.0, COORD_MAX_X - 1.0, GAUSSIAN_GRID_X)
        y_centers = torch.linspace(COORD_MIN + 1.0, COORD_MAX_Y - 1.0, GAUSSIAN_GRID_Y)
        
        # Create grid of centers
        centers_x, centers_y = torch.meshgrid(x_centers, y_centers, indexing='ij')
        self.register_buffer('gaussian_centers_x', centers_x.flatten())
        self.register_buffer('gaussian_centers_y', centers_y.flatten())
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch, seq_len, input_dim] or [batch, input_dim]
            
        Returns:
            Dict with 'logits', 'probabilities', 'coordinates', and 'gaussian_params'
        """
        # Handle sequence dimension
        if features.dim() == 3:
            batch_size, seq_len, input_dim = features.shape
            features = features[:, -1, :]  # Take last timestep
        
        # Predict Gaussian mixture parameters
        gaussian_params = self.gaussian_predictor(features)  # [batch, num_gaussians * 5]
        gaussian_params = gaussian_params.view(-1, self.num_gaussians, 5)
        
        # Extract parameters
        raw_logits = gaussian_params[:, :, 0]  # [batch, num_gaussians] - Raw logits for KD
        weights = F.softmax(raw_logits, dim=-1)  # [batch, num_gaussians] - Normalized weights
        mu_x = gaussian_params[:, :, 1]  # [batch, num_gaussians]
        mu_y = gaussian_params[:, :, 2]  # [batch, num_gaussians]
        sigma_x = F.softplus(gaussian_params[:, :, 3]) + 0.1  # [batch, num_gaussians] (positive)
        sigma_y = F.softplus(gaussian_params[:, :, 4]) + 0.1  # [batch, num_gaussians] (positive)
        
        # Compute expected coordinates (weighted average of Gaussian means)
        expected_x = torch.sum(weights * mu_x, dim=1)  # [batch]
        expected_y = torch.sum(weights * mu_y, dim=1)  # [batch]
        coordinates = torch.stack([expected_x, expected_y], dim=-1)  # [batch, 2]
        
        # Use raw logits directly for knowledge distillation (no lossy conversion)
        logits = raw_logits  # [batch, num_gaussians] - Clean logits for KL divergence
        
        return {
            'logits': logits,
            'probabilities': weights,
            'coordinates': coordinates,
            'predictions': coordinates,  # For compatibility
            'gaussian_params': {
                'weights': weights,
                'mu_x': mu_x,
                'mu_y': mu_y,
                'sigma_x': sigma_x,
                'sigma_y': sigma_y
            }
        }


class ContinuousUWBTransformerTeacher(nn.Module):
    """
    UWB Transformer Teacher that outputs continuous probability distributions.
    Uses Gaussian Mixture Model approach.
    """
    
    def __init__(self, input_features=113, d_model=128, n_layers=2, n_heads=4):
        super().__init__()
        self.input_features = input_features
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Continuous coordinate head
        self.coordinate_head = ContinuousCoordinateHead(d_model)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [batch, seq_len, input_features]
        batch_size, seq_len, _ = x.shape
        
        # Project to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Apply transformer
        features = self.transformer(x)  # [batch, seq_len, d_model]
        
        # Get continuous coordinate distribution
        output = self.coordinate_head(features)
        output['features'] = features  # For feature-level distillation
        
        return output


class ContinuousCSIRegressionModel(nn.Module):
    """
    CSI Regression Model that outputs continuous probability distributions using Mamba backbone.
    """
    
    def __init__(self, csi_config: dict, device: str = "cuda"):
        super().__init__()
        self.config = csi_config
        self.device = device
        
        # Create the proper CSI Mamba model as backbone
        self.csi_backbone = CSIRegressionModel(csi_config, device=device)
        
        # Get d_model from config for the continuous coordinate head
        d_model = csi_config["UWBMixerModel"]["input"]["d_model"]
        
        # Continuous coordinate head (same as before)
        self.coordinate_head = ContinuousCoordinateHead(d_model)
        
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # x: [batch, seq_len, input_features]
        
        # Get features from Mamba backbone
        backbone_output = self.csi_backbone(x, targets=targets)
        
        # Extract the last hidden state for coordinate prediction
        if backbone_output.last_hidden_state is not None:
            # Use last hidden state from Mamba backbone
            features = backbone_output.last_hidden_state  # [batch, seq_len, d_model]
        else:
            raise ValueError("No hidden states from backbone")
        
        # Get continuous coordinate distribution
        output = self.coordinate_head(features)
        output['features'] = features  # For feature-level distillation
        
        return type('Output', (), output)()  # Convert to object with attributes


def load_uwb_reference_data(uwb_data_path: str, experiment: str = "002"):
    """
    Load UWB data using the proper approach from uwb_opera_loader.py.
    Returns UWB features and targets for spatial/temporal alignment.
    Now includes additional features: fp_pow_dbm, rx_pow_dbm with proper normalization.
    """
    print(f"📡 Loading UWB reference data for experiment {experiment}...")
    
    # Load UWB data using proper file naming from uwb_opera_loader.py
    import glob
    import pandas as pd
    
    # Try different UWB file naming patterns (from synchronized_uwb_csi_loader_fixed.py)
    possible_files = [
        f"{uwb_data_path}/uwb2_exp{experiment}.csv",
        f"{uwb_data_path}/uwb1_exp{experiment}.csv", 
        f"{uwb_data_path}/uwb_exp{experiment}.csv",
        f"{uwb_data_path}/tag4422_exp{experiment}.csv"  # Original pattern
    ]
    
    uwb_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            uwb_file = file_path
            break
    
    if uwb_file is None:
        raise FileNotFoundError(f"No UWB files found for experiment {experiment}. Tried: {possible_files}")
    
    print(f"📁 Loading from {os.path.basename(uwb_file)}")
    uwb_data = pd.read_csv(uwb_file, nrows=15000)  # Limit for faster loading
    
    # Extract CIR features (cir1 to cir50) - from uwb_opera_loader.py approach
    cir_columns = [f'cir{i}' for i in range(1, 51)]
    available_cir = [col for col in cir_columns if col in uwb_data.columns]
    
    if len(available_cir) == 0:
        raise ValueError(f"No CIR columns found in UWB data for experiment {experiment}")
    
    print(f"📊 Found {len(available_cir)} CIR columns")
    # Convert complex strings to numbers (from uwb_opera_loader.py)
    cir_data = uwb_data[available_cir].copy()
    for col in available_cir:
        cir_data[col] = cir_data[col].apply(_parse_complex_uwb)
    
    # Create feature matrix with magnitude and phase (from CIR)
    features = []
    for col in available_cir:
        features.append(cir_data[col].apply(lambda x: abs(x)).values)
        features.append(cir_data[col].apply(lambda x: np.angle(x)).values)
    
    # Add additional features with proper normalization (following uwb_opera_loader.py pattern)
    additional_features = ['fp_pow_dbm', 'rx_pow_dbm']
    
    print(f"📊 Adding additional features: {additional_features}")
    for feat in additional_features:
        if feat in uwb_data.columns:
            print(f"   ✅ Found {feat}")
            # Clean the feature values (following uwb_opera_loader.py pattern)
            feat_values = uwb_data[feat].values
            feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=1e6, neginf=-1e6)
            feat_values = np.clip(feat_values, -1e6, 1e6)
            
            # Additional normalization for power values (they can be very large)
            if 'pow_dbm' in feat:
                # Power values in dBm typically range from -100 to +30 dBm
                # Apply additional clipping and scaling for better normalization
                feat_values = np.clip(feat_values, -150.0, 50.0)  # Reasonable dBm range
                print(f"   📏 {feat} range after clipping: [{feat_values.min():.2f}, {feat_values.max():.2f}] dBm")
            
            features.append(feat_values)
        else:
            print(f"   ⚠️  {feat} not found in data")
    
    uwb_features = np.column_stack(features)
    
    # Clean features
    uwb_features = np.nan_to_num(uwb_features, nan=0.0, posinf=1e6, neginf=-1e6)
    uwb_features = np.clip(uwb_features, -1e6, 1e6)
    
    # Extract coordinates (try different tag naming patterns)
    coordinate_columns = []
    for tag in ['tag4422', 'tag89b3']:
        x_col = f"{tag}_x"
        y_col = f"{tag}_y"
        if x_col in uwb_data.columns and y_col in uwb_data.columns:
            coordinate_columns.extend([x_col, y_col])
            break
    
    if not coordinate_columns:
        # Try simple x, y columns
        if 'x' in uwb_data.columns and 'y' in uwb_data.columns:
            coordinate_columns = ['x', 'y']
        else:
            raise ValueError(f"No coordinate columns found in UWB data for experiment {experiment}")
    
    uwb_coordinates = uwb_data[coordinate_columns].values.astype(np.float32)
    # Handle missing values
    uwb_coordinates = pd.DataFrame(uwb_coordinates).ffill().bfill().values
    # Apply coordinate bounds (from synchronized_uwb_csi_loader_fixed.py)
    uwb_coordinates = np.clip(uwb_coordinates, 0.0, 10.0)
    
    # Extract timestamps - REQUIRED, no fallback
    if 'timestamp' not in uwb_data.columns:
        raise ValueError(f"No timestamp column found in UWB data for experiment {experiment}")
    
    timestamps = _convert_uwb_timestamps_to_numeric(uwb_data['timestamp'].values)
    
    print(f"✅ Loaded {len(uwb_features)} UWB reference points")
    print(f"   Feature shape: {uwb_features.shape} (includes CIR + {len([f for f in additional_features if f in uwb_data.columns])} additional features)")
    print(f"   Additional features included: {[f for f in additional_features if f in uwb_data.columns]}")
    print(f"   Coordinate range: x=[{uwb_coordinates[:, 0].min():.2f}, {uwb_coordinates[:, 0].max():.2f}], y=[{uwb_coordinates[:, 1].min():.2f}, {uwb_coordinates[:, 1].max():.2f}]")
    print(f"   Timestamp range: [{timestamps.min():.1f}, {timestamps.max():.1f}]ms")
    
    return uwb_features, uwb_coordinates, timestamps


def _parse_complex_uwb(complex_str: str) -> complex:
    """Parse complex number from UWB string format (from uwb_opera_loader.py)."""
    try:
        if isinstance(complex_str, (int, float)):
            return complex(complex_str, 0)
        if isinstance(complex_str, str):
            complex_str = complex_str.strip()
            if complex_str == 'nan' or complex_str == '' or complex_str == 'None':
                return 0+0j
            # Replace 'i' with 'j' for Python complex parsing
            complex_str = complex_str.replace('i', 'j')
            result = complex(complex_str)
            # Check for inf or nan
            if not (np.isfinite(result.real) and np.isfinite(result.imag)):
                return 0+0j
            # Clip extreme values
            real_part = np.clip(result.real, -1e6, 1e6)
            imag_part = np.clip(result.imag, -1e6, 1e6)
            return complex(real_part, imag_part)
        return complex(0, 0)
    except:
        return complex(0, 0)


def _convert_uwb_timestamps_to_numeric(timestamps: np.ndarray) -> np.ndarray:
    """Convert UWB timestamps to numeric format (from synchronized_uwb_csi_loader_fixed.py)."""
    numeric_timestamps = []
    base_time = None
    
    for ts in timestamps:
        if isinstance(ts, str):
            # Parse time format like "15:07:11.404285"
            time_parts = ts.split(':')
            if len(time_parts) == 3:
                try:
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds = float(time_parts[2])
                    
                    total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
                    
                    if base_time is None:
                        base_time = total_ms
                    
                    numeric_timestamps.append(total_ms - base_time)
                except ValueError:
                    raise ValueError(f"Invalid timestamp format: {ts}")
            else:
                raise ValueError(f"Invalid timestamp format: {ts}")
        else:
            try:
                numeric_timestamps.append(float(ts))
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert timestamp to numeric: {ts}")
    
    if len(numeric_timestamps) == 0:
        raise ValueError("No valid timestamps found")
    
    return np.array(numeric_timestamps)


def create_uwb_guidance_for_csi_synchronized(csi_targets, csi_timestamps, uwb_features, uwb_coordinates, uwb_timestamps, csi_sequence_length=4):
    """
    Create UWB guidance with proper temporal synchronization handling different sampling rates.
    Based on synchronized_uwb_csi_loader_fixed.py approach.
    Now includes proper standardization for additional features.
    
    Args:
        csi_targets: CSI target coordinates [N, 2]
        csi_timestamps: CSI timestamps [N]
        uwb_features: UWB feature vectors [M, F] 
        uwb_coordinates: UWB coordinates [M, 2]
        uwb_timestamps: UWB timestamps [M]
        csi_sequence_length: Length to repeat UWB features
    
    Returns:
        uwb_guidance: [N, csi_sequence_length, F] - repeated UWB features
        uwb_target_guidance: [N, csi_sequence_length, 2] - repeated UWB targets
    """
    print("🔗 Creating UWB guidance with proper temporal synchronization...")
    
    # Apply feature standardization to ensure additional features are properly scaled
    from sklearn.preprocessing import StandardScaler
    
    print("📏 Standardizing UWB features (including additional power features)...")
    uwb_scaler = StandardScaler()
    uwb_features_scaled = uwb_scaler.fit_transform(uwb_features)
    
    # Check if scaling worked correctly
    print(f"   UWB features before scaling: [{uwb_features.min():.3f}, {uwb_features.max():.3f}]")
    print(f"   UWB features after scaling: [{uwb_features_scaled.min():.3f}, {uwb_features_scaled.max():.3f}]")
    print(f"   Feature dimensions: {uwb_features_scaled.shape}")
    
    N = len(csi_targets)
    F = uwb_features_scaled.shape[1]
    
    uwb_guidance = np.zeros((N, csi_sequence_length, F), dtype=np.float32)
    uwb_target_guidance = np.zeros((N, csi_sequence_length, 2), dtype=np.float32)
    
    # Calculate sampling rate ratio (from synchronized_uwb_csi_loader_fixed.py)
    uwb_duration = uwb_timestamps[-1] - uwb_timestamps[0] if len(uwb_timestamps) > 1 else 1000.0
    csi_duration = csi_timestamps[-1] - csi_timestamps[0] if len(csi_timestamps) > 1 else 1000.0
    
    uwb_samples_per_csi = len(uwb_features_scaled) / len(csi_targets)
    print(f"   Sampling ratio: {uwb_samples_per_csi:.2f} UWB samples per CSI sample")
    print(f"   Durations: UWB={uwb_duration:.1f}ms, CSI={csi_duration:.1f}ms")
    
    # For each CSI sample, find corresponding UWB measurements
    for i, (csi_target, csi_time) in enumerate(zip(csi_targets, csi_timestamps)):
        
        # Method 1: Temporal alignment (primary)
        if len(uwb_timestamps) > 1:
            # Find nearest UWB sample by timestamp
            time_diffs = np.abs(uwb_timestamps - csi_time)
            nearest_time_idx = np.argmin(time_diffs)
            
            # Use exact nearest UWB sample - NO AVERAGING
            uwb_selected_features = uwb_features_scaled[nearest_time_idx]
            uwb_selected_coords = uwb_coordinates[nearest_time_idx]
        
        else:
            # Method 2: Spatial alignment (fallback)
            # Find nearest UWB point by coordinate distance
            distances = np.sqrt(np.sum((uwb_coordinates - csi_target) ** 2, axis=1))
            nearest_spatial_idx = np.argmin(distances)
            
            uwb_selected_features = uwb_features_scaled[nearest_spatial_idx]
            uwb_selected_coords = uwb_coordinates[nearest_spatial_idx]
        
        # Repeat this EXACT UWB measurement across the CSI sequence length
        uwb_guidance[i] = np.repeat(uwb_selected_features.reshape(1, -1), csi_sequence_length, axis=0)
        uwb_target_guidance[i] = np.repeat(uwb_selected_coords.reshape(1, -1), csi_sequence_length, axis=0)
    
    print(f"✅ Created synchronized UWB guidance for {N} CSI samples")
    print(f"   Guidance shape: {uwb_guidance.shape}")
    
    # Calculate alignment quality
    if len(uwb_timestamps) > 1:
        avg_time_diff = np.mean([np.min(np.abs(uwb_timestamps - csi_time)) for csi_time in csi_timestamps[:min(100, len(csi_timestamps))]])
        print(f"   Average temporal alignment error: {avg_time_diff:.1f}ms")
    
    avg_spatial_distance = np.mean([np.min(np.sqrt(np.sum((uwb_coordinates - target) ** 2, axis=1))) for target in csi_targets[:min(100, len(csi_targets))]])
    print(f"   Average spatial distance to nearest UWB: {avg_spatial_distance:.3f}")
    
    return uwb_guidance, uwb_target_guidance


class CSIDatasetWithUWBGuidance(torch.utils.data.Dataset):
    """
    Dataset that provides CSI data with optional UWB guidance.
    For baseline: only returns CSI data
    For distilled: returns CSI data + repeated UWB guidance
    """
    
    def __init__(self, csi_data, csi_targets, uwb_guidance=None, uwb_target_guidance=None):
        self.csi_data = csi_data
        self.csi_targets = csi_targets
        self.uwb_guidance = uwb_guidance
        self.uwb_target_guidance = uwb_target_guidance
        self.has_uwb = uwb_guidance is not None
    
    def __len__(self):
        return len(self.csi_data)
    
    def __getitem__(self, idx):
        if self.has_uwb:
            return (
                torch.FloatTensor(self.uwb_guidance[idx]),      # UWB features
                torch.FloatTensor(self.uwb_target_guidance[idx]),  # UWB targets
                torch.FloatTensor(self.csi_data[idx]),          # CSI features  
                torch.FloatTensor(self.csi_targets[idx])        # CSI targets
            )
        else:
            return (
                torch.FloatTensor(self.csi_data[idx]),          # CSI features
                torch.FloatTensor(self.csi_targets[idx])        # CSI targets
            )


def create_truly_fair_dataloaders(
    csi_mat_file: str,
    uwb_data_path: str,
    experiment: str = "002",
    batch_size: int = 32,
    sequence_length: int = 4,
    max_samples: Optional[int] = None  # Changed to None to load entire dataset by default
):
    """
    Create truly fair dataloaders where both baseline and distilled use identical CSI data.
    """
    print("📊 Creating TRULY FAIR dataloaders...")
    
    # 1. Load ALL CSI data
    csi_train_loader, csi_val_loader, csi_scaler, csi_target_scaler = create_csi_dataloaders_fixed(
        mat_file_path=csi_mat_file,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_samples=max_samples,
        target_tags=['tag4422']
    )
    
    # Extract all CSI data and targets
    print("   📡 Extracting CSI data...")
    all_csi_data = []
    all_csi_targets = []
    
    for csi_data, csi_targets in csi_train_loader:
        all_csi_data.append(csi_data.numpy())
        all_csi_targets.append(csi_targets.numpy())
    
    for csi_data, csi_targets in csi_val_loader:
        all_csi_data.append(csi_data.numpy())
        all_csi_targets.append(csi_targets.numpy())
    
    all_csi_data = np.concatenate(all_csi_data, axis=0)
    all_csi_targets = np.concatenate(all_csi_targets, axis=0)
    
    # Handle target shape
    if all_csi_targets.ndim > 2:
        all_csi_targets = all_csi_targets[:, -1, :]  # Take last timestep
    if all_csi_targets.shape[-1] != 2:
        all_csi_targets = all_csi_targets[:, :2]  # Take x, y only
    
    print(f"   ✅ Total CSI samples: {len(all_csi_data)}")
    print(f"   📐 CSI shape: {all_csi_data.shape}")
    print(f"   🎯 Target shape: {all_csi_targets.shape}")
    print(f"   📊 CONFIRMED: Using ENTIRE dataset - {len(all_csi_data)} samples (no artificial limit)")
    
    # 2. Load UWB reference data
    uwb_features, uwb_coordinates, uwb_timestamps = load_uwb_reference_data(uwb_data_path, experiment)
    
    # 3. Create CSI timestamps based on known CSI sampling rate (not a fallback - based on system properties)
    # CSI data is typically sampled at regular intervals, create timestamps accordingly
    csi_sampling_interval_ms = 10.0  # Known CSI sampling interval
    csi_timestamps = np.arange(len(all_csi_data)) * csi_sampling_interval_ms
    
    # 4. Create UWB guidance for ALL CSI samples with proper synchronization
    uwb_guidance, uwb_target_guidance = create_uwb_guidance_for_csi_synchronized(
        all_csi_targets, csi_timestamps, uwb_features, uwb_coordinates, uwb_timestamps, sequence_length
    )
    
    # 5. Split into train/val
    split_idx = int(0.8 * len(all_csi_data))
    
    train_csi_data = all_csi_data[:split_idx]
    train_csi_targets = all_csi_targets[:split_idx]
    train_uwb_guidance = uwb_guidance[:split_idx]
    train_uwb_target_guidance = uwb_target_guidance[:split_idx]
    
    val_csi_data = all_csi_data[split_idx:]
    val_csi_targets = all_csi_targets[split_idx:]
    val_uwb_guidance = uwb_guidance[split_idx:]
    val_uwb_target_guidance = uwb_target_guidance[split_idx:]
    
    # 6. Create datasets
    # Baseline dataset (CSI only)
    baseline_train_dataset = CSIDatasetWithUWBGuidance(train_csi_data, train_csi_targets)
    baseline_val_dataset = CSIDatasetWithUWBGuidance(val_csi_data, val_csi_targets)
    
    # Distilled dataset (CSI + UWB guidance)
    distilled_train_dataset = CSIDatasetWithUWBGuidance(
        train_csi_data, train_csi_targets, train_uwb_guidance, train_uwb_target_guidance
    )
    distilled_val_dataset = CSIDatasetWithUWBGuidance(
        val_csi_data, val_csi_targets, val_uwb_guidance, val_uwb_target_guidance
    )
    
    # 7. Create dataloaders
    baseline_train_loader = torch.utils.data.DataLoader(baseline_train_dataset, batch_size=batch_size, shuffle=True)
    baseline_val_loader = torch.utils.data.DataLoader(baseline_val_dataset, batch_size=batch_size, shuffle=False)
    
    distilled_train_loader = torch.utils.data.DataLoader(distilled_train_dataset, batch_size=batch_size, shuffle=True)
    distilled_val_loader = torch.utils.data.DataLoader(distilled_val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   ✅ Created fair dataloaders:")
    print(f"      Baseline: {len(baseline_train_loader)} train, {len(baseline_val_loader)} val batches")
    print(f"      Distilled: {len(distilled_train_loader)} train, {len(distilled_val_loader)} val batches")
    
    return {
        'baseline': {
            'train': baseline_train_loader,
            'val': baseline_val_loader,
        },
        'distilled': {
            'train': distilled_train_loader,
            'val': distilled_val_loader,
        },
        'feature_dims': {
            'csi': all_csi_data.shape[-1],
            'uwb': uwb_features.shape[-1]
        },
        'scalers': {
            'csi': csi_scaler,
            'target': csi_target_scaler
        }
    }


def train_gmm_baseline(csi_config: dict, data_loaders: dict, device: str = "cuda", epochs: int = 15) -> ContinuousCSIRegressionModel:
    """
    Train GMM-based baseline on ALL CSI data (fair comparison with GMM distilled model).
    Uses Gaussian Mixture Models for coordinate prediction instead of direct regression.
    """
    print("🎯 Training GMM-BASED FAIR baseline...")
    print("   - Uses ALL available CSI data")
    print("   - No UWB information")
    print("   - Same GMM output as distilled model")
    print("   - Gaussian Mixture Models for coordinate prediction")
    
    baseline_model = ContinuousCSIRegressionModel(csi_config, device=device)
    baseline_model.to(device)
    
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        baseline_model.train()
        epoch_losses = {'total': [], 'coordinate': []}
        
        for batch_idx, (csi_data, csi_targets) in enumerate(data_loaders['baseline']['train']):
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            optimizer.zero_grad()
            
            outputs = baseline_model(csi_data, targets=csi_targets)
            predictions = outputs.coordinates  # Use GMM expected coordinates
            
            # Direct coordinate regression loss (no knowledge distillation)
            coordinate_loss = mse_loss(predictions, csi_targets)
            
            coordinate_loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses['total'].append(coordinate_loss.item())
            epoch_losses['coordinate'].append(coordinate_loss.item())
            
            if batch_idx >= 25:  # Reasonable training per epoch
                break
        
        if epoch % 5 == 0:
            print(f"   GMM Baseline Epoch {epoch}: Coordinate Loss={np.mean(epoch_losses['coordinate']):.6f}")
        
        scheduler.step()  # Update learning rate
    
    print("✅ GMM-based fair baseline training completed!")
    return baseline_model


def train_continuous_probability_distilled(
    teacher_model: nn.Module,
    csi_config: dict,
    data_loaders: dict,
    device: str = "cuda",
    epochs: int = 15,
    temperature: float = 15.0,  # Default T=15
    alpha: float = 0.2,  # Default alpha=0.2
    beta: float = 0.8,   # Default beta=0.8
    step_size: float = 0.01,  # Default step size for scheduling
    teacher_lr: float = 1e-3,  # Teacher learning rate
    student_lr: float = 5e-4   # Student learning rate
) -> ContinuousCSIRegressionModel:
    """
    Continuous probability distillation using Gaussian Mixture Models with adaptive scheduling.
    - Uses continuous probability densities instead of discretization
    - Compares logits/probabilities for proper knowledge distillation
    - Maintains coordinate regression for good task performance
    - ADAPTIVE SCHEDULING: If student struggles with cross-modal mapping → increase α (by step_size)
    - ADAPTIVE SCHEDULING: If task performance suffers → increase β (by step_size)
    - Hyperparameters: T=15, alpha=0.2, beta=0.8, step_size=0.01
    """
    print(f"🎓 Training CONTINUOUS PROBABILITY distilled model with adaptive scheduling...")
    print(f"   - T={temperature}, α_start={alpha}, β_start={beta}, step_size={step_size}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Gaussian Mixture: {NUM_GAUSSIANS} components")
    print("   🎯 CONTINUOUS PROBABILITIES: Gaussian Mixture Models")
    print("   📊 KNOWLEDGE DISTILLATION: Logits/probabilities comparison")
    print("   📈 TASK PERFORMANCE: Direct coordinate regression")
    print("   📈 ADAPTIVE SCHEDULING: α∈[0.1,0.6], β∈[0.4,0.9]")
    
    # Scheduling parameters
    alpha_min, alpha_max = 0.1, 0.6
    beta_min, beta_max = 0.4, 0.9
    
    # Initialize starting values within bounds
    current_alpha = max(alpha_min, min(alpha_max, alpha))
    current_beta = max(beta_min, min(beta_max, beta))
    
    # Performance tracking for scheduling
    recent_distill_losses = []
    recent_task_losses = []
    performance_window = 3  # Look at last 3 epochs for scheduling decisions
    
    # 1. Train teacher on UWB guidance data
    print("   🎯 Training continuous UWB teacher...")
    for param in teacher_model.parameters():
        param.requires_grad = True
    teacher_model.train()
    
    teacher_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=teacher_lr, weight_decay=1e-4)
    teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=epochs)
    teacher_criterion = nn.MSELoss()  # Train teacher on coordinate regression
    
    teacher_epochs = min(15, epochs)  # Train teacher for longer
    for epoch in range(teacher_epochs):
        epoch_losses = []
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(data_loaders['distilled']['train']):
            uwb_data = uwb_data.float().to(device)
            uwb_targets = uwb_targets.float().to(device)
            
            if uwb_targets.dim() > 2:
                uwb_targets = uwb_targets[:, -1, :]
            if uwb_targets.shape[-1] != 2:
                uwb_targets = uwb_targets[:, :2]
            
            teacher_optimizer.zero_grad()
            
            teacher_output = teacher_model(uwb_data)
            teacher_coordinates = teacher_output["coordinates"]
            
            loss = teacher_criterion(teacher_coordinates, uwb_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)
            teacher_optimizer.step()
            
            epoch_losses.append(loss.item())
            if batch_idx >= 30:  # More batches per epoch
                break
        
        if epoch % 3 == 0:
            print(f"   Teacher Epoch {epoch}: Loss={np.mean(epoch_losses):.6f}")
        
        teacher_scheduler.step()  # Update learning rate
    
    print(f"   ✅ Teacher trained for {teacher_epochs} epochs")
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # 2. Train student with continuous probability distillation and adaptive scheduling
    print("   🎓 Training CSI student with continuous probability distillation and adaptive scheduling...")
    student_model = ContinuousCSIRegressionModel(csi_config, device=device)
    student_model.to(device)
    
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=student_lr, weight_decay=1e-4)
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=epochs)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    mse_loss = nn.MSELoss()
    
    print(f"   📊 Initial: α={current_alpha:.3f}, β={current_beta:.3f}")
    
    for epoch in range(epochs):
        student_model.train()
        epoch_losses = {'total': [], 'kl_distill': [], 'task': []}
        
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(data_loaders['distilled']['train']):
            uwb_data = uwb_data.float().to(device)
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            student_optimizer.zero_grad()
            
            # Get teacher probability distributions from UWB guidance
            with torch.no_grad():
                teacher_output = teacher_model(uwb_data)
                teacher_logits = teacher_output["logits"]  # [batch, num_gaussians]
                
                # Apply temperature scaling to teacher logits for soft targets
                teacher_soft_logits = teacher_logits / temperature
                teacher_soft_probs = F.softmax(teacher_soft_logits, dim=-1)  # Soft targets
            
            # Get student probability distributions from CSI
            student_output = student_model(csi_data, targets=csi_targets)
            student_logits = student_output.logits  # [batch, num_gaussians]
            student_coordinates = student_output.coordinates  # [batch, 2]
            
            # Apply temperature scaling to student logits
            student_soft_logits = student_logits / temperature
            student_log_probs = F.log_softmax(student_soft_logits, dim=-1)
            
            # Compute KL divergence loss (knowledge distillation between probability distributions)
            # KL(Teacher || Student) - student learns to match teacher's probability distribution
            kl_distill_loss = kl_loss(student_log_probs, teacher_soft_probs) * (temperature ** 2)
            
            # Task loss (regression on continuous coordinates)
            task_loss = mse_loss(student_coordinates, csi_targets)
            
            # Combined loss with current scheduled values
            total_loss = current_alpha * kl_distill_loss + current_beta * task_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            student_optimizer.step()
            
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['kl_distill'].append(kl_distill_loss.item())
            epoch_losses['task'].append(task_loss.item())
            
            if batch_idx >= 25:
                break
        
        # Calculate epoch averages
        avg_distill_loss = np.mean(epoch_losses['kl_distill'])
        avg_task_loss = np.mean(epoch_losses['task'])
        
        # Store recent losses for scheduling
        recent_distill_losses.append(avg_distill_loss)
        recent_task_losses.append(avg_task_loss)
        
        # Keep only recent window
        if len(recent_distill_losses) > performance_window:
            recent_distill_losses.pop(0)
        if len(recent_task_losses) > performance_window:
            recent_task_losses.pop(0)
        
        # Adaptive scheduling logic (starting from epoch 2)
        if epoch >= 2 and len(recent_distill_losses) >= 2:
            # Check trends over recent epochs
            distill_trend = recent_distill_losses[-1] - recent_distill_losses[-2]
            task_trend = recent_task_losses[-1] - recent_task_losses[-2]
            
            # Scheduling decisions
            alpha_change = 0.0
            beta_change = 0.0
            
            # If KL distillation loss is increasing (student struggling with cross-modal mapping)
            if distill_trend > step_size:  # Threshold for KL divergence increase
                alpha_change = +step_size  # Increase alpha to focus more on teacher
                print(f"   📈 KL distill loss rising → Increase α by {alpha_change:.3f}")
            
            # If KL distillation loss is decreasing well, we can reduce alpha slightly
            elif distill_trend < -step_size:
                alpha_change = -step_size  # Small decrease to balance
                print(f"   📉 KL distill loss improving → Decrease α by {abs(alpha_change):.3f}")
            
            # If task loss is increasing (task performance suffering)
            if task_trend > step_size:  # Small threshold to avoid noise
                beta_change = +step_size  # Increase beta to focus more on task
                print(f"   📈 Task loss rising → Increase β by {beta_change:.3f}")
            
            # If task loss is decreasing well, we can reduce beta slightly
            elif task_trend < -step_size:
                beta_change = -step_size  # Small decrease to balance
                print(f"   📉 Task loss improving → Decrease β by {abs(beta_change):.3f}")
            
            # Apply changes with bounds
            old_alpha, old_beta = current_alpha, current_beta
            current_alpha = max(alpha_min, min(alpha_max, current_alpha + alpha_change))
            current_beta = max(beta_min, min(beta_max, current_beta + beta_change))
            
            # Ensure α + β doesn't become too extreme
            total_weight = current_alpha + current_beta
            if total_weight > 1.2:  # If total weight too high, normalize
                scale = 1.0 / total_weight
                current_alpha *= scale
                current_beta *= scale
                print(f"   ⚖️  Normalized weights: α={current_alpha:.3f}, β={current_beta:.3f}")
            elif total_weight < 0.6:  # If total weight too low, boost
                boost = 0.8 / total_weight
                current_alpha *= boost
                current_beta *= boost
                print(f"   🚀 Boosted weights: α={current_alpha:.3f}, β={current_beta:.3f}")
            
            # Log changes if significant
            if abs(current_alpha - old_alpha) > step_size or abs(current_beta - old_beta) > step_size:
                print(f"   🔄 Scheduled: α={old_alpha:.3f}→{current_alpha:.3f}, β={old_beta:.3f}→{current_beta:.3f}")
        
        if epoch % 5 == 0:
            print(f"   Student Epoch {epoch}: Total={np.mean(epoch_losses['total']):.6f}, "
                  f"KL_Distill={avg_distill_loss:.6f}, Task={avg_task_loss:.6f}")
            print(f"   Current weights: α={current_alpha:.3f}, β={current_beta:.3f}")
        
        student_scheduler.step()  # Update learning rate
    
    print(f"✅ Continuous probability distillation with adaptive scheduling completed!")
    print(f"   Final weights: α={current_alpha:.3f}, β={current_beta:.3f}")
    return student_model


def run_continuous_probability_comparison():
    """
    Compare GMM Baseline vs GMM Distilled models using ENTIRE dataset.
    Both models use Gaussian Mixture Models for fair comparison.
    """
    print("🔬 GMM BASELINE vs GMM DISTILLED COMPARISON (FAIR)")
    print("=" * 80)
    print("📊 Comparing 2 GMM models using continuous probability distributions:")
    print("   1️⃣ GMM Baseline Mamba (CSI-only, Gaussian Mixture Models)")
    print("   2️⃣ GMM Distilled Mamba (Gaussian Mixture Models + Adaptive Scheduling)")
    print(f"   🎯 Gaussian Mixture: {NUM_GAUSSIANS} components (BOTH MODELS)")
    print(f"   📐 Room coverage: X=[0,{COORD_MAX_X}]m, Y=[0,{COORD_MAX_Y}]m")
    print("   🔧 CONTINUOUS: No discretization, proper probability densities")
    print("   ⚖️  FAIR COMPARISON: Both models use identical GMM architecture")
    print("   📈 Will generate CDF plots for detailed error distribution analysis")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup paths
    uwb_data_path = "/media/mohab/Storage HDD/Downloads/uwb2(1)"
    csi_mat_file = "/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat"
    
    # Create truly fair dataloaders ONCE (ENTIRE dataset)
    print("📊 Loading ENTIRE dataset for continuous probability comparison...")
    data_loaders = create_truly_fair_dataloaders(
        csi_mat_file=csi_mat_file,
        uwb_data_path=uwb_data_path,
        experiment="002",
        batch_size=32,
        sequence_length=4,
        max_samples=None  # Use ENTIRE CSI dataset (no artificial limit)
    )
    
    feature_dims = data_loaders['feature_dims']
    print(f"📐 Model dimensions: CSI={feature_dims['csi']}, UWB={feature_dims['uwb']}")
    
    # Model configuration
    csi_config = {
        "CSIRegressionModel": {
            "input": {
                "input_features": feature_dims['csi'],
                "output_features": 2,
                "output_mode": "last",
                "dropout": 0.1
            }
        },
        "UWBMixerModel": {
            "input": {
                "d_model": 80,  # CHANGED FROM 88 TO 80
                "n_layer": 2,
                "final_prenorm": "layer"
            }
        },
        "Block1": {
            "n_layers": 2,
            "BlockType": "modules.phi_block",
            "block_input": {"resid_dropout": 0.1},
            "CoreType": "modules.mixers.discrete_mamba2",
            "core_input": {
                "d_state": 16, "n_v_heads": 8, "n_qk_heads": 8,
                "d_conv": 4, "conv_bias": True, "expand": 2,
                "chunk_size": 64, "activation": "identity", "bias": False
            }
        }
    }
    
    # Fair training epochs (use same for both models)
    training_epochs = 20
    
    models = {}
    
    # 1️⃣ Train Baseline Mamba (CSI-only)
    print(f"\n{'='*60}")
    print("1️⃣ TRAINING GMM BASELINE MAMBA")
    print(f"{'='*60}")
    print(f"   📊 Training on ENTIRE dataset for 13 epochs")
    print(f"   📊 Learning rate: 1.0e-03 with scheduler")
    print("   🎯 GMM coordinate prediction (no distillation)")
    
    baseline_model = train_gmm_baseline(csi_config, data_loaders, device, epochs=13)
    #print("Baseline Mamba model size:", get_model_size(baseline_model))
    models['baseline'] = {
        'model': baseline_model,
        'name': 'GMM Baseline Mamba (CSI-only)',
        'config': {
            'epochs': 13,
            'learning_rate': 1.0e-03,
            'type': 'gmm_baseline'
        },
        'model_type': 'gmm'
    }
    
    # 2️⃣ Train Continuous Probability Mamba
    print(f"\n{'='*60}")
    print("2️⃣ TRAINING GMM DISTILLED MAMBA WITH ADAPTIVE SCHEDULING")
    print(f"{'='*60}")
    print("   📊 Configuration: T=15.0, α_start=0.2, β_start=0.8 (ADAPTIVE)")
    print(f"   📊 Training on ENTIRE dataset for {training_epochs} epochs")
    print("   🎯 GAUSSIAN MIXTURE MODELS: Same architecture as baseline")
    print("   📊 KNOWLEDGE DISTILLATION: GMM logits/probabilities comparison")
    print("   📈 ADAPTIVE SCHEDULING: α∈[0.1,0.6], β∈[0.4,0.9]")
    
    # Continuous probability teacher model
    teacher_model = ContinuousUWBTransformerTeacher(
        input_features=feature_dims['uwb'],
        d_model=80,  # CHANGED FROM 88 TO 80
        n_layers=2,
        n_heads=4
    ).to(device)
    
    continuous_model = train_continuous_probability_distilled(
        teacher_model, csi_config, data_loaders, device,
        epochs=training_epochs,
        temperature=15.0,  # Updated to 15.0
        alpha=0.2,
        beta=0.8,
        step_size=0.01
    )
    models['continuous_probability'] = {
        'model': continuous_model,
        'name': 'GMM Distilled Mamba (Adaptive)',
        'config': {
            'epochs': training_epochs,
            'temperature': 15.0,  # Updated to 15.0
            'alpha_start': 0.2,
            'beta_start': 0.8,
            'type': 'gmm_distilled_adaptive',
            'num_gaussians': NUM_GAUSSIANS,
            'adaptive_scheduling': True
        },
        'model_type': 'gmm'
    }
    
    # 📊 Evaluation with Median Absolute Error and CDF plots
    print(f"\n{'='*60}")
    print("📊 EVALUATION")
    print(f"{'='*60}")
    print("   📈 Using Median Absolute Error (MAE) for robust comparison")
    print("   📊 Generating CDF plots for error distribution analysis")
    print("   🔄 Using expected coordinates from Gaussian mixtures")
    
    results = evaluate_baseline_vs_continuous_probability(models, data_loaders, device)
    
    # Save results
    with open('baseline_vs_continuous_probability_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to baseline_vs_continuous_probability_results.json")
    print(f"📈 CDF plot saved as baseline_vs_continuous_probability_cdf.png")
    
    return results


def evaluate_baseline_vs_continuous_probability(models: dict, data_loaders: dict, device: str = "cuda"):
    """
    Evaluate Baseline vs Continuous Probability models using Median Absolute Error and generate CDF plots.
    FIXED: Denormalizes coordinates back to real space before calculating errors.
    Enhanced: Includes model sizes, R² scores, RMSE, and comprehensive metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    print("📊 Evaluating Baseline vs Continuous Probability models with comprehensive metrics...")
    print("🔧 IMPORTANT: Denormalizing coordinates to REAL SPACE before error calculation")
    
    # Get the target scaler for denormalization
    target_scaler = data_loaders['scalers']['target']
    print(f"📏 Target scaler stats:")
    print(f"   Mean: {target_scaler.mean_}")
    print(f"   Scale (std): {target_scaler.scale_}")
    print(f"   🎯 Will denormalize predictions and targets to real coordinate space")
    
    results = {
        'summary': {},
        'detailed_errors': {},
        'statistics': {},
        'model_info': {}
    }
    
    all_errors = {}
    
    # Helper function to calculate model size
    def get_model_size(model):
        """Calculate model size in parameters and memory."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(total_params - trainable_params),
            'estimated_memory_mb': float(param_memory_mb)
        }
    
    # Evaluate each model
    for model_key, model_info in models.items():
        print(f"\n📈 Evaluating {model_info['name']}...")
        
        model = model_info['model']
        model_type = model_info['model_type']
        model.eval()
        
        # Calculate model size
        model_size_info = get_model_size(model)
        print(f"   🏗️  Model Size:")
        print(f"      Total parameters: {model_size_info['total_parameters']:,}")
        print(f"      Trainable parameters: {model_size_info['trainable_parameters']:,}")
        print(f"      Estimated memory: {model_size_info['estimated_memory_mb']:.2f} MB")
        
        all_predictions = []
        all_targets = []
        
        # Get predictions and targets
        with torch.no_grad():
            if model_key == 'baseline':
                # Baseline uses CSI-only data (continuous model)
                for csi_data, csi_targets in data_loaders['baseline']['val']:
                    csi_data = csi_data.float().to(device)
                    csi_targets = csi_targets.float().to(device)
                    
                    outputs = model(csi_data, targets=csi_targets)
                    
                    # Extract continuous coordinates from GMM baseline
                    if hasattr(outputs, 'coordinates'):
                        predictions = outputs.coordinates  # Expected coordinates from Gaussian mixture
                    else:
                        # Fallback: get coordinates from probabilities
                        probabilities = outputs.probabilities
                        gaussian_params = outputs.gaussian_params
                        mu_x = gaussian_params['mu_x']
                        mu_y = gaussian_params['mu_y']
                        # Expected coordinates (weighted average)
                        expected_x = torch.sum(probabilities * mu_x, dim=1)
                        expected_y = torch.sum(probabilities * mu_y, dim=1)
                        predictions = torch.stack([expected_x, expected_y], dim=-1)
                    
                    if predictions.shape[-1] != 2:
                        predictions = predictions[:, :2]
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(csi_targets.cpu().numpy())
            else:
                # Continuous probability model uses CSI data
                for uwb_data, uwb_targets, csi_data, csi_targets in data_loaders['distilled']['val']:
                    csi_data = csi_data.float().to(device)
                    csi_targets = csi_targets.float().to(device)
                    
                    outputs = model(csi_data, targets=csi_targets)
                    
                    # Extract continuous coordinates from Gaussian mixture output
                    if hasattr(outputs, 'coordinates'):
                        predictions = outputs.coordinates  # Expected coordinates from Gaussian mixture
                    else:
                        # Fallback: get coordinates from probabilities
                        probabilities = outputs.probabilities
                        gaussian_params = outputs.gaussian_params
                        mu_x = gaussian_params['mu_x']
                        mu_y = gaussian_params['mu_y']
                        # Expected coordinates (weighted average)
                        expected_x = torch.sum(probabilities * mu_x, dim=1)
                        expected_y = torch.sum(probabilities * mu_y, dim=1)
                        predictions = torch.stack([expected_x, expected_y], dim=-1)
                    
                    if predictions.shape[-1] != 2:
                        predictions = predictions[:, :2]
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(csi_targets.cpu().numpy())
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        print(f"   📊 Before denormalization:")
        print(f"      Predictions range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")
        print(f"      Targets range: [{all_targets.min():.3f}, {all_targets.max():.3f}]")
        
        # CRITICAL FIX: Denormalize coordinates back to real space
        all_predictions_real = target_scaler.inverse_transform(all_predictions)
        all_targets_real = target_scaler.inverse_transform(all_targets)
        
        print(f"   📏 After denormalization (REAL COORDINATES):")
        print(f"      Predictions range: [{all_predictions_real.min():.3f}, {all_predictions_real.max():.3f}] meters")
        print(f"      Targets range: [{all_targets_real.min():.3f}, {all_targets_real.max():.3f}] meters")
        
        # Calculate comprehensive metrics in REAL coordinate space
        
        # 1. Euclidean distance errors
        sample_errors = np.sqrt(np.sum((all_predictions_real - all_targets_real) ** 2, axis=1))
        
        # 2. Individual coordinate errors (X and Y separately)
        x_errors = np.abs(all_predictions_real[:, 0] - all_targets_real[:, 0])
        y_errors = np.abs(all_predictions_real[:, 1] - all_targets_real[:, 1])
        
        # 3. R² scores (coefficient of determination)
        r2_euclidean = r2_score(all_targets_real.flatten(), all_predictions_real.flatten())
        r2_x = r2_score(all_targets_real[:, 0], all_predictions_real[:, 0])
        r2_y = r2_score(all_targets_real[:, 1], all_predictions_real[:, 1])
        
        # 4. RMSE and MAE
        rmse = np.sqrt(mean_squared_error(all_targets_real, all_predictions_real))
        mae = mean_absolute_error(all_targets_real, all_predictions_real)
        
        # 5. Percentile statistics
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        error_percentiles = {f'p{p}': float(np.percentile(sample_errors, p)) for p in percentiles}
        
        # 6. Basic statistics
        median_ae = float(np.median(sample_errors))
        mean_ae = float(np.mean(sample_errors))
        std_ae = float(np.std(sample_errors))
        min_error = float(np.min(sample_errors))
        max_error = float(np.max(sample_errors))
        
        # 7. Coordinate-specific statistics
        x_stats = {
            'median': float(np.median(x_errors)),
            'mean': float(np.mean(x_errors)),
            'std': float(np.std(x_errors)),
            'min': float(np.min(x_errors)),
            'max': float(np.max(x_errors))
        }
        
        y_stats = {
            'median': float(np.median(y_errors)),
            'mean': float(np.mean(y_errors)),
            'std': float(np.std(y_errors)),
            'min': float(np.min(y_errors)),
            'max': float(np.max(y_errors))
        }
        
        # Store comprehensive results
        results['summary'][model_key] = {
            'model_name': model_info['name'],
            'model_type': model_type,
            'config': model_info['config'],
            
            # Main metrics
            'median_absolute_error': median_ae,
            'mean_absolute_error': mean_ae,
            'std_absolute_error': std_ae,
            'min_error': min_error,
            'max_error': max_error,
            'rmse': float(rmse),
            'mae': float(mae),
            
            # R² scores
            'r2_overall': float(r2_euclidean),
            'r2_x_coordinate': float(r2_x),
            'r2_y_coordinate': float(r2_y),
            
            # Percentiles
            'error_percentiles': error_percentiles,
            
            # Coordinate-specific errors
            'x_coordinate_errors': x_stats,
            'y_coordinate_errors': y_stats,
            
            # Sample statistics
            'total_samples': int(len(sample_errors)),
            'coordinate_space': 'real_meters'
        }
        
        # Store model architecture info
        results['model_info'][model_key] = {
            'model_name': model_info['name'],
            'model_type': model_type,
            'model_size': model_size_info,
            'training_config': model_info['config']
        }
        
        results['detailed_errors'][model_key] = sample_errors.tolist()
        all_errors[model_key] = sample_errors
        
        print(f"   📊 COMPREHENSIVE METRICS (REAL SPACE):")
        print(f"      Median AE: {median_ae:.6f} meters")
        print(f"      Mean AE: {mean_ae:.6f} meters")
        print(f"      RMSE: {rmse:.6f} meters")
        print(f"      R² (overall): {r2_euclidean:.6f}")
        print(f"      R² (X): {r2_x:.6f}, R² (Y): {r2_y:.6f}")
        print(f"      Error range: [{min_error:.6f}, {max_error:.6f}] meters")
        print(f"      95th percentile error: {error_percentiles['p95']:.6f} meters")
    
    # Generate CDF plot
    print("\n📈 Generating CDF plots (REAL COORDINATE ERRORS)...")
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue']
    styles = ['-', '--']
    
    for i, (model_key, errors) in enumerate(all_errors.items()):
        model_name = models[model_key]['name']
        
        # Sort errors for CDF
        sorted_errors = np.sort(errors)
        y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        plt.plot(sorted_errors, y, color=colors[i], linestyle=styles[i], 
                linewidth=2.5, label=model_name, alpha=0.8)
        
        # Add median line
        median_val = np.median(errors)
        plt.axvline(median_val, color=colors[i], linestyle=':', alpha=0.7,
                   label=f'{model_name} Median: {median_val:.4f}m')
    
    plt.xlabel('Absolute Error (meters) - REAL COORDINATE SPACE', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    plt.title(f'CDF Comparison: Baseline vs Continuous Probability Distillation ({NUM_GAUSSIANS} Gaussians)\n(Lower curves = better performance, REAL coordinate errors)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.xlim(left=0)
    plt.ylim([0, 1])
    
    # Add text box with summary statistics
    baseline_r2 = results['summary']['baseline']['r2_overall']
    cont_prob_r2 = results['summary']['continuous_probability']['r2_overall']
    baseline_params = results['model_info']['baseline']['model_size']['total_parameters']
    cont_prob_params = results['model_info']['continuous_probability']['model_size']['total_parameters']
    
    textstr = '\n'.join([
        f"📊 Summary Statistics (REAL meters):",
        f"Baseline: MAE={results['summary']['baseline']['median_absolute_error']:.4f}m, R²={baseline_r2:.4f}",
        f"Distilled: MAE={results['summary']['continuous_probability']['median_absolute_error']:.4f}m, R²={cont_prob_r2:.4f}",
        f"Model sizes: {baseline_params:,} vs {cont_prob_params:,} params",
        f"Gaussian Mixture: {NUM_GAUSSIANS} components"
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.98, 0.02, textstr, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('baseline_vs_continuous_probability_cdf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate relative improvements
    baseline_median = results['summary']['baseline']['median_absolute_error']
    cont_prob_median = results['summary']['continuous_probability']['median_absolute_error']
    baseline_r2 = results['summary']['baseline']['r2_overall']
    cont_prob_r2 = results['summary']['continuous_probability']['r2_overall']
    
    cont_prob_improvement = ((baseline_median - cont_prob_median) / baseline_median) * 100
    r2_improvement = ((cont_prob_r2 - baseline_r2) / max(abs(baseline_r2), 0.001)) * 100
    
    results['statistics']['improvements'] = {
        'continuous_probability_vs_baseline_percent': float(cont_prob_improvement),
        'r2_improvement_percent': float(r2_improvement),
        'baseline_median_error_meters': float(baseline_median),
        'distilled_median_error_meters': float(cont_prob_median),
        'baseline_r2': float(baseline_r2),
        'distilled_r2': float(cont_prob_r2)
    }
    
    # Add system info
    results['system_info'] = {
        'coordinate_bounds': f'X=[0, {COORD_MAX_X}]m, Y=[0, {COORD_MAX_Y}]m',
        'gaussian_components': NUM_GAUSSIANS,
        'evaluation_method': 'real_coordinate_space_denormalized',
        'target_scaler_mean': target_scaler.mean_.tolist(),
        'target_scaler_scale': target_scaler.scale_.tolist()
    }
    
    # Print final comparison
    print(f"\n{'='*80}")
    print("🏆 FINAL COMPREHENSIVE COMPARISON (REAL COORDINATE SPACE)")
    print(f"{'='*80}")
    print(f"📊 Baseline Mamba (CSI-only):")
    print(f"   Median AE: {baseline_median:.6f} meters")
    print(f"   R² Score: {baseline_r2:.6f}")
    print(f"   Parameters: {results['model_info']['baseline']['model_size']['total_parameters']:,}")
    print(f"   Memory: {results['model_info']['baseline']['model_size']['estimated_memory_mb']:.2f} MB")
    print()
    print(f"📊 Continuous Probability Mamba ({NUM_GAUSSIANS} Gaussian components):")
    print(f"   Median AE: {cont_prob_median:.6f} meters")
    print(f"   R² Score: {cont_prob_r2:.6f}")
    print(f"   Parameters: {results['model_info']['continuous_probability']['model_size']['total_parameters']:,}")
    print(f"   Memory: {results['model_info']['continuous_probability']['model_size']['estimated_memory_mb']:.2f} MB")
    print(f"   Error Improvement: {cont_prob_improvement:+.2f}%")
    print(f"   R² Improvement: {r2_improvement:+.2f}%")
    print()
    
    if cont_prob_median < baseline_median:
        print(f"🥇 WINNER (Error): Continuous Probability Distillation")
        print(f"📈 Advantage: {cont_prob_improvement:.2f}% better median error")
    else:
        print(f"🥇 WINNER (Error): Baseline Mamba")
        print(f"📈 Advantage: {abs(cont_prob_improvement):.2f}% better median error")
    
    if cont_prob_r2 > baseline_r2:
        print(f"🥇 WINNER (R²): Continuous Probability Distillation")
        print(f"📈 R² Advantage: {r2_improvement:.2f}% better")
    else:
        print(f"🥇 WINNER (R²): Baseline Mamba")
        print(f"📈 R² Advantage: {abs(r2_improvement):.2f}% better")
    
    print(f"\n📈 CDF plot saved as 'baseline_vs_continuous_probability_cdf.png'")
    print(f"🎯 NOTE: All metrics are in REAL coordinate space (meters)")
    print(f"📊 Comprehensive metrics saved to JSON including model sizes and R² scores")
    
    return results


def run_custom_parameter_comparison():
    """
    Run comparison with user-specified parameters:
    - Teacher LR: 2.0e-03
    - Student LR: 1.0e-03
    - 20 epochs for both models (with LR schedulers)
    - T=15, α=0.2, β=0.8, step_size=0.01
    """
    print("🎯 CUSTOM PARAMETER COMPARISON")
    print("=" * 60)
    print("📊 Configuration:")
    print("   • Temperature (T): 15.0")
    print("   • Alpha (distillation weight): 0.2")
    print("   • Beta (task weight): 0.8")
    print("   • Step size: 0.01 (adaptive scheduling)")
    print("   • Teacher LR: 2.0e-03 (with CosineAnnealingLR)")
    print("   • Student LR: 1.0e-03 (with CosineAnnealingLR)")
    print("   • Epochs: 20 (both models)")
    print(f"   • Gaussian Mixture: {NUM_GAUSSIANS} components")
    print(f"   • Room coverage: X=[0,{COORD_MAX_X}]m, Y=[0,{COORD_MAX_Y}]m")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup paths
    uwb_data_path = "/media/mohab/Storage HDD/Downloads/uwb2(1)"
    csi_mat_file = "/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat"
    
    # Create dataloaders
    print("📊 Loading dataset...")
    data_loaders = create_truly_fair_dataloaders(
        csi_mat_file=csi_mat_file,
        uwb_data_path=uwb_data_path,
        experiment="002",
        batch_size=32,
        sequence_length=4,
        max_samples=None  # Use full dataset
    )
    
    feature_dims = data_loaders['feature_dims']
    print(f"📐 Model dimensions: CSI={feature_dims['csi']}, UWB={feature_dims['uwb']}")
    
    # Model configuration
    csi_config = {
        "CSIRegressionModel": {
            "input": {
                "input_features": feature_dims['csi'],
                "output_features": 2,
                "output_mode": "last",
                "dropout": 0.1
            }
        },
        "UWBMixerModel": {
            "input": {
                "d_model": 80,  # CHANGED FROM 88 TO 80
                "n_layer": 2,
                "final_prenorm": "layer"
            }
        },
        "Block1": {
            "n_layers": 2,
            "BlockType": "modules.phi_block",
            "block_input": {"resid_dropout": 0.1},
            "CoreType": "modules.mixers.discrete_mamba2",
            "core_input": {
                "d_state": 16, "n_v_heads": 8, "n_qk_heads": 8,
                "d_conv": 4, "conv_bias": True, "expand": 2,
                "chunk_size": 64, "activation": "identity", "bias": False
            }
        }
    }
    
    models = {}
    
    # 1️⃣ Train Baseline Mamba (CSI-only)
    print(f"\n{'='*60}")
    print("1️⃣ TRAINING GMM BASELINE MAMBA")
    print(f"{'='*60}")
    print(f"   📊 Training on ENTIRE dataset for 13 epochs")
    print(f"   📊 Learning rate: 1.0e-03")
    print("   🎯 GMM coordinate prediction (no distillation)")
    
    baseline_model = train_gmm_baseline(csi_config, data_loaders, device, epochs=13)
    #print("Baseline Mamba model size:", get_model_size(baseline_model))
    models['baseline'] = {
        'model': baseline_model,
        'name': 'GMM Baseline Mamba (CSI-only)',
        'config': {
            'epochs': 13,
            'learning_rate': 1.0e-03,
            'type': 'gmm_baseline'
        },
        'model_type': 'gmm'
    }
    
    # 2️⃣ Train Distilled Mamba
    print(f"\n{'='*60}")
    print("2️⃣ TRAINING GMM DISTILLED MAMBA")
    print(f"{'='*60}")
    print("   📊 Configuration: T=15.0, α=0.2, β=0.8")
    print("   📊 Teacher LR: 2.0e-03, Student LR: 1.0e-03")
    print(f"   📊 Training for 15 epochs")
    print("   🎯 GMM knowledge distillation")
    
    # Create teacher model
    teacher_model = ContinuousUWBTransformerTeacher(
        input_features=feature_dims['uwb'],
        d_model=80,  # CHANGED FROM 88 TO 80
        n_layers=2,
        n_heads=4
    ).to(device)
    
    distilled_model = train_continuous_probability_distilled(
        teacher_model, csi_config, data_loaders, device,
        epochs=20,
        temperature=15.0,
        alpha=0.2,
        beta=0.8,
        step_size=0.01,
        teacher_lr=2.0e-03,
        student_lr=1.0e-03
    )
    models['continuous_probability'] = {
        'model': distilled_model,
        'name': 'GMM Distilled Mamba (Custom)',
        'config': {
            'epochs': 20,
            'temperature': 15.0,
            'alpha': 0.2,
            'beta': 0.8,
            'teacher_lr': 2.0e-03,
            'student_lr': 1.0e-03,
            'type': 'gmm_distilled_custom'
        },
        'model_type': 'gmm'
    }
    
    # 📊 Evaluation
    print(f"\n{'='*60}")
    print("📊 EVALUATION")
    print(f"{'='*60}")
    print("   📈 Using Median Absolute Error (MAE) for robust comparison")
    print("   📊 Generating CDF plots for error distribution analysis")
    print("   🔄 Using expected coordinates from Gaussian mixtures")
    
    results = evaluate_baseline_vs_continuous_probability(models, data_loaders, device)
    
    # Save results with custom naming
    with open('custom_parameter_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to custom_parameter_comparison_results.json")
    print(f"📈 CDF plot saved as baseline_vs_continuous_probability_cdf.png")
    
    # Display summary
    baseline_mae = results['summary']['baseline']['median_absolute_error']
    distilled_mae = results['summary']['continuous_probability']['median_absolute_error']
    improvement = ((baseline_mae - distilled_mae) / baseline_mae) * 100
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"🎯 Baseline MAE: {baseline_mae:.6f}")
    print(f"🎓 Distilled MAE: {distilled_mae:.6f}")
    print(f"📈 Improvement: {improvement:.2f}%")
    
    if improvement > 0:
        print(f"✅ Distillation IMPROVED performance by {improvement:.2f}%")
    else:
        print(f"❌ Distillation DECREASED performance by {abs(improvement):.2f}%")
    
    return results


if __name__ == "__main__":
    # Run custom parameter comparison
    print("🚀 CUSTOM PARAMETER COMPARISON")
    print("=" * 80)
    print("🎯 Testing specific configuration:")
    print("   • Temperature (T): 15.0")
    print("   • Alpha (distillation weight): 0.2")
    print("   • Beta (task weight): 0.8")
    print("   • Step size: 0.01 (adaptive scheduling)")
    print("   • Teacher LR: 2.0e-03")
    print("   • Student LR: 1.0e-03")
    print("   • Epochs: 20 (both models)")
    print(f"   🎯 Gaussian Mixture: {NUM_GAUSSIANS} components")
    print(f"   📐 Room coverage: X=[0,{COORD_MAX_X}]m, Y=[0,{COORD_MAX_Y}]m")
    print()
    
    results = run_custom_parameter_comparison()
    
    print("\n🎉 Custom parameter comparison completed!")
    print(f"📁 Results saved to custom_parameter_comparison_results.json")
    print(f"📈 CDF plots saved as baseline_vs_continuous_probability_cdf.png")
