#!/usr/bin/env python3
"""
Data Setup Script for MambaLoc
This script helps users set up the data paths and verify data structure.
"""

import os
import json
import argparse
from pathlib import Path

def create_config_from_template(uwb_path: str, csi_path: str, output_dir: str = "./results"):
    """Create a configuration file from template with user-specified paths."""
    
    # Load template
    with open("config_template.json", "r") as f:
        config = json.load(f)
    
    # Update paths
    config["data_paths"]["uwb_data_path"] = uwb_path
    config["data_paths"]["csi_mat_file"] = csi_path
    config["data_paths"]["output_dir"] = output_dir
    
    # Save as config.json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to config.json")
    print(f"   UWB data path: {uwb_path}")
    print(f"   CSI data path: {csi_path}")
    print(f"   Output directory: {output_dir}")

def verify_data_structure(uwb_path: str, csi_path: str):
    """Verify that the data paths exist and have expected structure."""
    
    print("🔍 Verifying data structure...")
    
    # Check UWB data
    if not os.path.exists(uwb_path):
        print(f"❌ UWB data path does not exist: {uwb_path}")
        return False
    
    # Look for UWB CSV files
    uwb_files = list(Path(uwb_path).glob("*exp*.csv"))
    if not uwb_files:
        print(f"⚠️  No UWB experiment files found in: {uwb_path}")
        print("   Expected files like: uwb2_exp002.csv, tag4422_exp002.csv, etc.")
    else:
        print(f"✅ Found {len(uwb_files)} UWB experiment files")
        for f in uwb_files[:3]:  # Show first 3
            print(f"   - {f.name}")
    
    # Check CSI data
    if not os.path.exists(csi_path):
        print(f"❌ CSI data file does not exist: {csi_path}")
        return False
    
    if not csi_path.endswith('.mat'):
        print(f"⚠️  CSI file should be a .mat file: {csi_path}")
    else:
        print(f"✅ CSI data file found: {os.path.basename(csi_path)}")
    
    return True

def setup_output_directories(output_dir: str):
    """Create necessary output directories."""
    
    dirs_to_create = [
        output_dir,
        f"{output_dir}/models",
        f"{output_dir}/plots",
        f"{output_dir}/logs"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Setup data paths for MambaLoc")
    
    parser.add_argument("--uwb_path", type=str, required=True,
                       help="Path to UWB dataset directory")
    parser.add_argument("--csi_path", type=str, required=True,
                       help="Path to CSI .mat file")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--verify_only", action="store_true",
                       help="Only verify data structure, don't create config")
    
    args = parser.parse_args()
    
    print("🚀 MambaLoc Data Setup")
    print("=" * 50)
    
    # Verify data structure
    if not verify_data_structure(args.uwb_path, args.csi_path):
        print("\n❌ Data verification failed. Please check your paths.")
        return
    
    if not args.verify_only:
        # Create configuration
        create_config_from_template(args.uwb_path, args.csi_path, args.output_dir)
        
        # Setup output directories
        setup_output_directories(args.output_dir)
        
        print("\n✅ Setup complete!")
        print("   Next steps:")
        print("   1. Run: python truly_fair_comparison.py")
        print("   2. Check results in:", args.output_dir)

if __name__ == "__main__":
    main()
