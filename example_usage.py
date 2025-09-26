#!/usr/bin/env python3
"""
Example Usage Script for MambaLoc
Demonstrates basic usage with sample configuration.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path

def load_config(config_path: str = "config.json"):
    """Load configuration from file."""
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("   Please run: python setup_data.py --uwb_path /path/to/uwb --csi_path /path/to/csi.mat")
        return None
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

def check_data_availability(config: dict):
    """Check if data files are available."""
    uwb_path = config["data_paths"]["uwb_data_path"]
    csi_path = config["data_paths"]["csi_mat_file"]
    
    print("🔍 Checking data availability...")
    
    if not os.path.exists(uwb_path):
        print(f"❌ UWB data not found: {uwb_path}")
        return False
    
    if not os.path.exists(csi_path):
        print(f"❌ CSI data not found: {csi_path}")
        return False
    
    print("✅ Data files found!")
    return True

def run_example_training(config: dict):
    """Run a simplified example training."""
    print("🚀 Running example training...")
    
    # Import here to avoid issues if data is not available
    try:
        from truly_fair_comparison import (
            create_truly_fair_dataloaders,
            train_gmm_baseline,
            ContinuousUWBTransformerTeacher,
            train_continuous_probability_distilled
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure all dependencies are installed: pip install -r requirements.txt")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Create data loaders with limited samples for quick demo
    print("📊 Creating data loaders (limited samples for demo)...")
    try:
        data_loaders = create_truly_fair_dataloaders(
            csi_mat_file=config["data_paths"]["csi_mat_file"],
            uwb_data_path=config["data_paths"]["uwb_data_path"],
            experiment="002",
            batch_size=16,  # Smaller batch for demo
            sequence_length=4,
            max_samples=500  # Limited samples for quick demo
        )
        print("✅ Data loaders created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print("   Please check your data paths and format.")
        return
    
    # Model configuration (simplified)
    feature_dims = data_loaders['feature_dims']
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
                "d_model": config["model"]["d_model"],
                "n_layer": config["model"]["n_layers"],
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
    
    print("🎯 Training baseline model (quick demo - 3 epochs)...")
    try:
        baseline_model = train_gmm_baseline(csi_config, data_loaders, device, epochs=3)
        print("✅ Baseline model trained!")
        
        # Save model
        output_dir = config["data_paths"]["output_dir"]
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        torch.save(baseline_model.state_dict(), f"{output_dir}/models/baseline_demo.pth")
        print(f"💾 Model saved to: {output_dir}/models/baseline_demo.pth")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    print("\n🎉 Example completed successfully!")
    print("   For full training, run: python truly_fair_comparison.py")

def main():
    print("🚀 MambaLoc Example Usage")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    if config is None:
        return
    
    # Check data availability
    if not check_data_availability(config):
        print("\n💡 To set up data paths, run:")
        print("   python setup_data.py --uwb_path /path/to/uwb --csi_path /path/to/csi.mat")
        return
    
    # Run example
    run_example_training(config)

if __name__ == "__main__":
    main()
