#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes UWB and CSI datasets to understand their structure, temporal characteristics,
and determine optimal cross-modal alignment strategies.
"""

import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Any
import sys

# Add current directory to path for imports
sys.path.append('.')


def analyze_uwb_dataset():
    """Analyze the UWB dataset structure and temporal characteristics."""
    print("🔍 ANALYZING UWB DATASET")
    print("=" * 50)
    
    uwb_data_path = "/media/mohab/Storage HDD/Downloads/uwb2(1)"
    
    # Sample a few UWB files to understand structure
    sample_files = [
        "uwb2_exp002.csv", 
        "uwb2_exp003.csv", 
        "uwb2_exp004.csv"
    ]
    
    uwb_analysis = {}
    
    for filename in sample_files:
        filepath = os.path.join(uwb_data_path, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filename}")
            continue
            
        print(f"\n📁 Analyzing {filename}")
        print("-" * 30)
        
        # Load sample of file (first 1000 rows for analysis)
        try:
            df = pd.read_csv(filepath, nrows=1000)
            
            # Basic info
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB (sample)")
            
            # Column analysis
            print(f"\n   📊 Column types:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            object_cols = df.select_dtypes(include=[object]).columns
            print(f"      Numeric: {len(numeric_cols)}")
            print(f"      Object/String: {len(object_cols)}")
            
            # Check for CIR columns (core UWB features)
            cir_columns = [col for col in df.columns if 'cir' in col.lower()]
            print(f"      CIR features: {len(cir_columns)}")
            if cir_columns:
                print(f"      CIR range: {cir_columns[0]} to {cir_columns[-1]}")
            
            # Check for coordinate columns
            coord_columns = [col for col in df.columns if any(tag in col for tag in ['tag4422', 'tag']) and ('_x' in col or '_y' in col)]
            print(f"      Coordinate columns: {coord_columns}")
            
            # Timestamp analysis
            if 'timestamp' in df.columns:
                timestamps = df['timestamp'].dropna()
                print(f"\n   ⏰ Temporal characteristics:")
                print(f"      First timestamp: {timestamps.iloc[0]}")
                print(f"      Last timestamp: {timestamps.iloc[-1]}")
                
                # Try to parse timestamps and calculate sampling rate
                try:
                    if timestamps.dtype == 'object':
                        # Convert to numeric (assuming milliseconds)
                        numeric_timestamps = pd.to_numeric(timestamps, errors='coerce')
                        if not numeric_timestamps.isna().all():
                            time_diffs = numeric_timestamps.diff().dropna()
                            avg_interval = time_diffs.median()
                            sampling_rate = 1000.0 / avg_interval if avg_interval > 0 else 0
                            print(f"      Average interval: {avg_interval:.2f} ms")
                            print(f"      Estimated sampling rate: {sampling_rate:.2f} Hz")
                except Exception as e:
                    print(f"      Timestamp parsing error: {e}")
            
            # Data quality check
            print(f"\n   📈 Data quality:")
            null_count = df.isnull().sum().sum()
            print(f"      Total null values: {null_count}")
            print(f"      Null percentage: {null_count / (df.shape[0] * df.shape[1]) * 100:.2f}%")
            
            # Sample coordinate values
            if coord_columns:
                coord_data = df[coord_columns].dropna()
                if len(coord_data) > 0:
                    print(f"\n   🎯 Coordinate statistics:")
                    for col in coord_columns[:4]:  # Show first 4
                        values = coord_data[col]
                        print(f"      {col}: range [{values.min():.3f}, {values.max():.3f}], mean {values.mean():.3f}")
            
            uwb_analysis[filename] = {
                'shape': df.shape,
                'columns': len(df.columns),
                'cir_features': len(cir_columns),
                'coord_columns': coord_columns,
                'null_percentage': null_count / (df.shape[0] * df.shape[1]) * 100
            }
            
        except Exception as e:
            print(f"   ❌ Error analyzing {filename}: {e}")
    
    return uwb_analysis


def analyze_csi_dataset():
    """Analyze the CSI dataset structure and temporal characteristics."""
    print("\n🔍 ANALYZING CSI DATASET")
    print("=" * 50)
    
    csi_file_path = "/home/mohab/Downloads/wificsi1_exp002.mat"
    
    if not os.path.exists(csi_file_path):
        print(f"❌ CSI file not found: {csi_file_path}")
        return {}
    
    print(f"📁 Analyzing {os.path.basename(csi_file_path)}")
    
    csi_analysis = {}
    
    try:
        with h5py.File(csi_file_path, 'r') as f:
            print(f"\n📊 HDF5 structure:")
            print(f"   Top-level keys: {list(f.keys())}")
            
            # Check for experiment groups
            if 'exp002' in f:
                data_group = f['exp002']
                print(f"\n📊 exp002 group structure:")
                print(f"   Keys: {len(list(data_group.keys()))}")
                
                # Categorize keys
                csi_keys = []
                coord_keys = []
                other_keys = []
                
                for key in data_group.keys():
                    if 'tx' in key and 'rx' in key and 'sub' in key:
                        csi_keys.append(key)
                    elif any(tag in key for tag in ['tag4422', 'tag']) and ('_x' in key or '_y' in key):
                        coord_keys.append(key)
                    else:
                        other_keys.append(key)
                
                print(f"   CSI features: {len(csi_keys)}")
                print(f"   Coordinate features: {len(coord_keys)}")
                print(f"   Other features: {len(other_keys)}")
                
                if csi_keys:
                    print(f"   CSI range: {csi_keys[0]} to {csi_keys[-1]}")
                
                if coord_keys:
                    print(f"   Coordinates: {coord_keys}")
                
                # Analyze data dimensions and types
                print(f"\n📏 Data dimensions:")
                sample_sizes = []
                
                # Check a few CSI arrays
                for key in csi_keys[:3]:
                    array = data_group[key]
                    print(f"   {key}: {array.shape}, dtype: {array.dtype}")
                    if hasattr(array, 'dtype') and array.dtype.names:
                        print(f"      Complex struct: {array.dtype.names}")
                        if 'real' in array.dtype.names:
                            real_data = array['real'][...]
                            imag_data = array['imag'][...]
                            print(f"      Real part: {real_data.shape}")
                            print(f"      Imag part: {imag_data.shape}")
                            sample_sizes.append(len(real_data.flatten()))
                    else:
                        sample_sizes.append(len(array[...].flatten()))
                
                # Check coordinate arrays
                for key in coord_keys[:2]:
                    array = data_group[key]
                    print(f"   {key}: {array.shape}, dtype: {array.dtype}")
                    if hasattr(array, 'dtype') and array.dtype.names:
                        if 'real' in array.dtype.names:
                            real_data = array['real'][...]
                            print(f"      Real part range: [{real_data.min():.3f}, {real_data.max():.3f}]")
                    else:
                        data = array[...]
                        print(f"      Range: [{data.min():.3f}, {data.max():.3f}]")
                
                # Determine common sample size
                if sample_sizes:
                    min_samples = min(sample_sizes)
                    max_samples = max(sample_sizes)
                    print(f"\n📊 Sample size analysis:")
                    print(f"   Min samples: {min_samples}")
                    print(f"   Max samples: {max_samples}")
                    print(f"   Consistency: {'✅ Consistent' if min_samples == max_samples else '⚠️ Inconsistent'}")
                
                # Estimate CSI sampling rate
                print(f"\n⏰ Temporal analysis:")
                if 'timestamp' in data_group:
                    timestamp_array = data_group['timestamp']
                    timestamps = timestamp_array[...].flatten()
                    if len(timestamps) > 1:
                        time_diffs = np.diff(timestamps)
                        avg_interval = np.median(time_diffs)
                        sampling_rate = 1.0 / avg_interval if avg_interval > 0 else 0
                        print(f"   Sample count: {len(timestamps)}")
                        print(f"   Average interval: {avg_interval:.6f} units")
                        print(f"   Estimated sampling rate: {sampling_rate:.2f} Hz")
                else:
                    print("   No timestamp data found")
                    # Estimate based on typical WiFi CSI rates
                    if sample_sizes:
                        print(f"   Assuming ~100 Hz CSI sampling rate")
                        estimated_duration = min_samples / 100.0
                        print(f"   Estimated duration: {estimated_duration:.2f} seconds")
                
                csi_analysis = {
                    'csi_features': len(csi_keys),
                    'coord_features': len(coord_keys),
                    'sample_count': min_samples if sample_sizes else 0,
                    'has_timestamps': 'timestamp' in data_group,
                    'data_structure': 'direct_access'
                }
            
            else:
                # Use refs-based approach
                print(f"\n📊 Using refs-based analysis...")
                refs = f['#refs#']
                
                field_names = {}
                for key in refs.keys():
                    data = refs[key]
                    if hasattr(data, 'dtype') and data.dtype == 'uint16':
                        try:
                            ascii_data = data[...].flatten()
                            decoded = ''.join(chr(x) for x in ascii_data if x > 0)
                            if len(decoded) < 50:
                                field_names[key] = decoded
                        except:
                            pass
                
                csi_fields = [name for name in field_names.values() if 'tx' in name and 'rx' in name]
                coord_fields = [name for name in field_names.values() if 'tag' in name and ('_x' in name or '_y' in name)]
                
                print(f"   CSI fields found: {len(csi_fields)}")
                print(f"   Coordinate fields found: {len(coord_fields)}")
                
                csi_analysis = {
                    'csi_features': len(csi_fields),
                    'coord_features': len(coord_fields),
                    'data_structure': 'refs_based'
                }
    
    except Exception as e:
        print(f"❌ Error analyzing CSI file: {e}")
    
    return csi_analysis


def determine_alignment_strategy(uwb_analysis: Dict, csi_analysis: Dict):
    """Determine the best cross-modal alignment strategy."""
    print("\n🎯 CROSS-MODAL ALIGNMENT STRATEGY")
    print("=" * 50)
    
    print("📊 Dataset Comparison:")
    
    # UWB characteristics
    if uwb_analysis:
        avg_uwb_samples = np.mean([info['shape'][0] for info in uwb_analysis.values() if 'shape' in info])
        avg_uwb_features = np.mean([info['cir_features'] for info in uwb_analysis.values() if 'cir_features' in info])
        print(f"   UWB: ~{avg_uwb_samples:.0f} samples/exp, {avg_uwb_features:.0f} CIR features")
    
    # CSI characteristics  
    if csi_analysis:
        csi_samples = csi_analysis.get('sample_count', 0)
        csi_features = csi_analysis.get('csi_features', 0)
        print(f"   CSI: {csi_samples} samples, {csi_features} features")
    
    print(f"\n🔧 Recommended Alignment Strategies:")
    
    # Strategy 1: Index-based alignment (current approach)
    print(f"\n1. INDEX-BASED ALIGNMENT (Current)")
    print(f"   ✅ Pros:")
    print(f"      - Simple implementation")
    print(f"      - Works when data is pre-synchronized")
    print(f"      - Handles different sampling rates via averaging")
    print(f"   ⚠️ Cons:")
    print(f"      - Assumes synchronization")
    print(f"      - May lose temporal precision")
    
    # Strategy 2: Timestamp-based alignment
    has_uwb_timestamps = True  # Assume UWB has timestamps
    has_csi_timestamps = csi_analysis.get('has_timestamps', False)
    
    print(f"\n2. TIMESTAMP-BASED ALIGNMENT")
    print(f"   UWB timestamps: {'✅ Available' if has_uwb_timestamps else '❌ Missing'}")
    print(f"   CSI timestamps: {'✅ Available' if has_csi_timestamps else '❌ Missing'}")
    
    if has_uwb_timestamps and has_csi_timestamps:
        print(f"   ✅ Pros:")
        print(f"      - True temporal synchronization")
        print(f"      - Handles timing drift")
        print(f"      - More accurate alignment")
        print(f"   ⚠️ Implementation needed")
    else:
        print(f"   ❌ Not feasible without timestamps in both modalities")
    
    # Strategy 3: Windowed alignment
    print(f"\n3. WINDOWED ALIGNMENT")
    print(f"   ✅ Pros:")
    print(f"      - Robust to small timing differences")
    print(f"      - Can use overlapping windows")
    print(f"      - Handles different sequence lengths")
    print(f"   ⚠️ Cons:")
    print(f"      - More complex implementation")
    
    # Strategy 4: Feature-based alignment
    print(f"\n4. FEATURE-BASED ALIGNMENT")
    print(f"   ✅ Pros:")
    print(f"      - Uses signal characteristics for alignment")
    print(f"      - Can detect movement patterns")
    print(f"   ❌ Cons:")
    print(f"      - Very complex implementation")
    print(f"      - Computationally expensive")
    
    print(f"\n💡 RECOMMENDED APPROACH:")
    if has_uwb_timestamps and has_csi_timestamps:
        print(f"   🥇 PRIMARY: Timestamp-based alignment with interpolation")
        print(f"   🥈 FALLBACK: Index-based alignment with windowing")
    else:
        print(f"   🥇 PRIMARY: Improved index-based alignment with windowing")
        print(f"   🥈 ENHANCEMENT: Add coordinate-based validation")
    
    return generate_implementation_plan(uwb_analysis, csi_analysis)


def generate_implementation_plan(uwb_analysis: Dict, csi_analysis: Dict):
    """Generate specific implementation recommendations."""
    print(f"\n🛠️ IMPLEMENTATION PLAN")
    print("=" * 50)
    
    plan = {
        'data_loading': [],
        'preprocessing': [],
        'alignment': [],
        'validation': []
    }
    
    # Data loading improvements
    print(f"\n1. DATA LOADING IMPROVEMENTS:")
    plan['data_loading'].extend([
        "✅ Use consistent file naming patterns (uwb2_exp{ID}.csv)",
        "✅ Implement robust error handling for missing files",
        "✅ Add memory-efficient chunked loading for large files",
        "✅ Standardize coordinate column detection across experiments"
    ])
    for item in plan['data_loading']:
        print(f"   {item}")
    
    # Preprocessing improvements
    print(f"\n2. PREPROCESSING IMPROVEMENTS:")
    plan['preprocessing'].extend([
        "🔧 Implement proper target normalization (z-score or min-max)",
        "🔧 Add outlier detection and removal for coordinates",
        "🔧 Standardize complex number handling (magnitude/phase vs real/imag)",
        "🔧 Implement feature scaling per modality",
        "🔧 Add data quality checks (null values, ranges, etc.)"
    ])
    for item in plan['preprocessing']:
        print(f"   {item}")
    
    # Alignment improvements
    print(f"\n3. ALIGNMENT IMPROVEMENTS:")
    has_timestamps = csi_analysis.get('has_timestamps', False)
    
    if has_timestamps:
        plan['alignment'].extend([
            "🎯 Implement timestamp-based matching with tolerance window",
            "🎯 Add interpolation for missing timestamp matches",
            "🎯 Use weighted averaging for multiple CSI samples per UWB sample"
        ])
    else:
        plan['alignment'].extend([
            "🎯 Improve index-based alignment with sequence overlap",
            "🎯 Add coordinate-based validation of alignment quality",
            "🎯 Implement sliding window approach for better temporal coverage"
        ])
    
    plan['alignment'].extend([
        "🎯 Add alignment quality metrics",
        "🎯 Implement alignment visualization tools"
    ])
    for item in plan['alignment']:
        print(f"   {item}")
    
    # Validation improvements
    print(f"\n4. VALIDATION IMPROVEMENTS:")
    plan['validation'].extend([
        "📊 Add cross-validation with temporal splits",
        "📊 Implement alignment quality assessment",
        "📊 Add coordinate prediction accuracy metrics",
        "📊 Create alignment visualization plots",
        "📊 Add statistical tests for alignment quality"
    ])
    for item in plan['validation']:
        print(f"   {item}")
    
    return plan


def create_improved_dataloader():
    """Create an improved dataloader with better alignment strategy."""
    print(f"\n⚡ CREATING IMPROVED DATALOADER")
    print("=" * 50)
    
    improvements = [
        "1. 🎯 Target Normalization: Implement proper coordinate scaling",
        "2. 🔧 Error Handling: Add robust file loading with graceful fallbacks",
        "3. ⏰ Temporal Alignment: Improve synchronization with window-based matching",
        "4. 📊 Data Quality: Add validation and outlier removal",
        "5. 🔄 Memory Efficiency: Implement streaming for large datasets",
        "6. 📈 Monitoring: Add alignment quality metrics and logging"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")
    
    print(f"\n💡 Priority Fixes:")
    print(f"   🚨 HIGH: Target normalization (fixes Stage 3 loss issues)")
    print(f"   🚨 HIGH: Coordinate range validation and clipping")
    print(f"   📊 MEDIUM: Improved temporal alignment with windowing")
    print(f"   🔧 MEDIUM: Better error handling and data quality checks")
    print(f"   ⚡ LOW: Memory optimization and streaming")


def main():
    """Run complete dataset analysis."""
    print("🔍 COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 60)
    print("Analyzing UWB and CSI datasets to optimize cross-modal alignment")
    print("")
    
    try:
        # Analyze UWB dataset
        uwb_analysis = analyze_uwb_dataset()
        
        # Analyze CSI dataset
        csi_analysis = analyze_csi_dataset()
        
        # Determine alignment strategy
        implementation_plan = determine_alignment_strategy(uwb_analysis, csi_analysis)
        
        # Create implementation recommendations
        create_improved_dataloader()
        
        print(f"\n✅ ANALYSIS COMPLETE")
        print("=" * 60)
        print("📋 Next Steps:")
        print("1. 🎯 Implement target normalization in data loaders")
        print("2. 🔧 Fix coordinate range validation")
        print("3. ⏰ Improve temporal alignment strategy")
        print("4. 📊 Add alignment quality metrics")
        print("5. ✅ Test with Stage 3 distillation")
        
        return uwb_analysis, csi_analysis, implementation_plan
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    main() 