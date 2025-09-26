"""
Debug Script: Investigate Scaling Issues in Cross-Modal Distillation
Analyzes data ranges, encoder outputs, and model predictions to identify performance problems.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataloaders.synchronized_uwb_csi_loader import create_synchronized_dataloaders
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from modules.csi_head import CSIRegressionModel
from encoder_enhanced_cross_modal_distillation import EncoderEnhancedMOHAWKDistiller
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher


def analyze_data_ranges():
    """Analyze the ranges of data in different loaders."""
    print("🔍 INVESTIGATION 1: Data Range Analysis")
    print("=" * 50)
    
    # Setup CSI-only data (baseline)
    print("\n📊 CSI-Only Data (Baseline):")
    csi_train_loader, csi_val_loader, csi_scaler, csi_target_scaler = create_csi_dataloaders_fixed(
        mat_file_path="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
        train_split=0.8,
        batch_size=32,
        sequence_length=4,
        target_tags=['tag4422'],
        use_magnitude_phase=True,
        max_samples=200,
        temporal_gap=0
    )
    
    # Sample CSI data
    for csi_data, csi_targets in csi_train_loader:
        print(f"   CSI Data: {csi_data.shape}, range [{csi_data.min():.3f}, {csi_data.max():.3f}]")
        print(f"   CSI Targets: {csi_targets.shape}, range [{csi_targets.min():.3f}, {csi_targets.max():.3f}]")
        break
    
    # Setup synchronized data (distillation)
    print("\n📊 Synchronized UWB-CSI Data (Distillation):")
    sync_train_loader, sync_val_loader, uwb_scaler, target_scaler, sync_csi_scaler = create_synchronized_dataloaders(
        uwb_data_path="/media/mohab/Storage HDD/Downloads/uwb2(1)",
        csi_mat_file="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
        train_experiments=["002"],
        val_experiments=["002"],
        batch_size=32,
        sequence_length=32,
        csi_sequence_length=4,
        target_tags=['tag4422'],
        max_samples_per_exp=200,
        use_magnitude_phase=True,
        stride=2,
        temporal_split=True,
        train_split=0.8,
        temporal_gap=0
    )
    
    # Sample synchronized data
    for uwb_data, uwb_targets, sync_csi_data, sync_csi_targets in sync_train_loader:
        print(f"   UWB Data: {uwb_data.shape}, range [{uwb_data.min():.3f}, {uwb_data.max():.3f}]")
        print(f"   UWB Targets: {uwb_targets.shape}, range [{uwb_targets.min():.3f}, {uwb_targets.max():.3f}]")
        print(f"   Sync CSI Data: {sync_csi_data.shape}, range [{sync_csi_data.min():.3f}, {sync_csi_data.max():.3f}]")
        print(f"   Sync CSI Targets: {sync_csi_targets.shape}, range [{sync_csi_targets.min():.3f}, {sync_csi_targets.max():.3f}]")
        break
    
    # Check if scalers are different
    print("\n📏 Scaler Comparison:")
    print(f"   CSI-only target scaler mean: {getattr(csi_target_scaler, 'mean_', 'N/A')}")
    print(f"   CSI-only target scaler scale: {getattr(csi_target_scaler, 'scale_', 'N/A')}")
    print(f"   Sync target scaler mean: {getattr(target_scaler, 'mean_', 'N/A')}")
    print(f"   Sync target scaler scale: {getattr(target_scaler, 'scale_', 'N/A')}")
    
    return {
        'csi_train_loader': csi_train_loader,
        'sync_train_loader': sync_train_loader,
        'csi_target_scaler': csi_target_scaler,
        'sync_target_scaler': target_scaler
    }


def analyze_encoder_outputs(sync_train_loader):
    """Analyze encoder outputs to see if they're reasonable."""
    print("\n🔍 INVESTIGATION 2: Encoder Output Analysis")
    print("=" * 50)
    
    # Create encoders
    from modules.cross_modal_encoders import Encoder_UWB, Encoder_CSI
    
    encoder_uwb = Encoder_UWB(input_features=100, latent_dim=128).cuda()
    encoder_csi = Encoder_CSI(input_features=280, latent_dim=128).cuda()
    
    # Sample data and analyze encoder outputs
    for uwb_data, uwb_targets, csi_data, csi_targets in sync_train_loader:
        uwb_data = uwb_data.float().cuda()
        csi_data = csi_data.float().cuda()
        
        # Get encoder outputs
        with torch.no_grad():
            z_uwb = encoder_uwb(uwb_data)
            z_csi = encoder_csi(csi_data)
        
        print(f"🧠 Encoder Analysis:")
        print(f"   UWB latent (z_UWB): {z_uwb.shape}, range [{z_uwb.min():.3f}, {z_uwb.max():.3f}], mean {z_uwb.mean():.3f}")
        print(f"   CSI latent (z_CSI): {z_csi.shape}, range [{z_csi.min():.3f}, {z_csi.max():.3f}], mean {z_csi.mean():.3f}")
        print(f"   UWB targets: range [{uwb_targets.min():.3f}, {uwb_targets.max():.3f}]")
        print(f"   CSI targets: range [{csi_targets.min():.3f}, {csi_targets.max():.3f}]")
        
        # Check latent alignment
        cosine_sim = torch.cosine_similarity(z_uwb, z_csi, dim=1).mean()
        print(f"   Latent cosine similarity: {cosine_sim:.4f}")
        
        break
    
    return encoder_uwb, encoder_csi


def analyze_baseline_vs_distilled_predictions(data_info):
    """Compare predictions from baseline vs distilled models."""
    print("\n🔍 INVESTIGATION 3: Model Prediction Analysis")
    print("=" * 50)
    
    # Create baseline CSI model
    csi_config = {
        "CSIRegressionModel": {
            "input": {
                "input_features": 280,
                "output_features": 2,
                "output_mode": "last",
                "dropout": 0.1
            }
        },
        "UWBMixerModel": {
            "input": {
                "d_model": 64,
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
                "d_conv": 4, "conv_bias": True, "expand": 1,
                "chunk_size": 64, "activation": "identity", "bias": False
            }
        }
    }
    
    baseline_model = CSIRegressionModel(csi_config, device="cuda")
    baseline_model.cuda()
    baseline_model.eval()
    
    # Test baseline model
    print("📊 Baseline CSI Model Analysis:")
    for csi_data, csi_targets in data_info['csi_train_loader']:
        csi_data = csi_data.float().cuda()
        csi_targets = csi_targets.float().cuda()
        
        with torch.no_grad():
            baseline_outputs = baseline_model(csi_data, targets=csi_targets)
            baseline_preds = baseline_outputs.predictions
        
        print(f"   Input CSI: {csi_data.shape}, range [{csi_data.min():.3f}, {csi_data.max():.3f}]")
        print(f"   Baseline Preds: {baseline_preds.shape}, range [{baseline_preds.min():.3f}, {baseline_preds.max():.3f}]")
        print(f"   Targets: {csi_targets.shape}, range [{csi_targets.min():.3f}, {csi_targets.max():.3f}]")
        print(f"   Baseline MSE: {torch.nn.functional.mse_loss(baseline_preds, csi_targets).item():.6f}")
        break
    
    # Test distilled model if available
    try:
        checkpoint = torch.load('./mohawk_cross_modal_results/mohawk_distilled_csi_model.pth', map_location='cuda')
        print("\n📊 Distilled Model Analysis:")
        
        # Create simple test model to analyze saved components
        from modules.cross_modal_encoders import Encoder_CSI
        test_encoder = Encoder_CSI(input_features=280, latent_dim=128).cuda()
        test_encoder.load_state_dict(checkpoint['encoder_csi'])
        test_encoder.eval()
        
        for csi_data, csi_targets in data_info['csi_train_loader']:
            csi_data = csi_data.float().cuda()
            
            with torch.no_grad():
                z_csi = test_encoder(csi_data)
            
            print(f"   Distilled Encoder Output: {z_csi.shape}, range [{z_csi.min():.3f}, {z_csi.max():.3f}]")
            print(f"   🚨 Problem: Encoder outputs latent vectors, not coordinates!")
            print(f"   🚨 The student model needs to learn: latent → coordinates")
            break
            
    except Exception as e:
        print(f"⚠️ Could not load distilled model: {e}")


def investigate_loss_scaling():
    """Investigate why task loss is so much larger than distillation loss."""
    print("\n🔍 INVESTIGATION 4: Loss Scaling Analysis")
    print("=" * 50)
    
    # Simulate the scaling issue observed in training
    
    # Example values from the training log
    teacher_preds = torch.tensor([[2.5, 2.8]], device='cuda')  # Scaled coordinates  
    student_preds = torch.tensor([[150.0, -200.0]], device='cuda')  # Unscaled predictions
    targets = torch.tensor([[2.6, 2.9]], device='cuda')  # Scaled targets
    
    temperature = 4.0
    alpha, beta = 0.7, 0.3
    
    # Compute losses like in the framework
    distill_loss = torch.nn.functional.mse_loss(
        student_preds / temperature,
        teacher_preds / temperature
    )
    
    task_loss = torch.nn.functional.mse_loss(student_preds, targets)
    
    total_loss = alpha * distill_loss + beta * task_loss
    
    print(f"📊 Simulated Loss Analysis:")
    print(f"   Teacher predictions: {teacher_preds}")
    print(f"   Student predictions: {student_preds}")
    print(f"   Targets: {targets}")
    print(f"   Distillation loss: {distill_loss.item():.6f}")
    print(f"   Task loss: {task_loss.item():.6f}")
    print(f"   Total loss: {total_loss.item():.6f}")
    print(f"   🚨 Task loss is {task_loss.item() / distill_loss.item():.1f}x larger!")
    print(f"   🚨 This completely dominates training!")


def analyze_target_coordinate_systems():
    """Analyze different coordinate systems used in baseline vs distillation."""
    print("\n🔍 INVESTIGATION 5: Coordinate System Analysis")
    print("=" * 50)
    
    # Load raw coordinate data to understand the original ranges
    import h5py
    import scipy.io
    
    try:
        csi_file = "/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat"
        
        # Try loading with h5py for v7.3 files
        with h5py.File(csi_file, 'r') as f:
            print("📂 Raw CSI File Analysis:")
            
            # Look for coordinate data
            def find_coordinates(group, prefix=""):
                for key in group.keys():
                    if 'tag4422' in key and ('_x' in key or '_y' in key):
                        data = group[key]
                        if hasattr(data, 'value'):
                            values = data.value
                        else:
                            values = data[:]
                        
                        print(f"   {prefix}{key}: {values.shape}, range [{np.min(values):.3f}, {np.max(values):.3f}]")
            
            # Check different groups
            for group_name in f.keys():
                print(f"   Group: {group_name}")
                if isinstance(f[group_name], h5py.Group):
                    find_coordinates(f[group_name], "     ")
                    
    except Exception as e:
        print(f"⚠️ Could not analyze raw file: {e}")
        
    print("\n💡 Key Insights:")
    print("   1. Check if raw coordinates are in different scales")
    print("   2. Verify if scaling is applied consistently")
    print("   3. Ensure student model learns the right coordinate system")


def main():
    """Run comprehensive investigation."""
    print("🚀 Cross-Modal Distillation Performance Investigation")
    print("=" * 60)
    
    # Run investigations
    data_info = analyze_data_ranges()
    encoder_uwb, encoder_csi = analyze_encoder_outputs(data_info['sync_train_loader'])
    analyze_baseline_vs_distilled_predictions(data_info)
    investigate_loss_scaling()
    analyze_target_coordinate_systems()
    
    print("\n🎯 SUMMARY OF FINDINGS:")
    print("=" * 30)
    print("1. 📊 Data range analysis completed")
    print("2. 🧠 Encoder output analysis completed") 
    print("3. 📈 Model prediction comparison completed")
    print("4. ⚖️ Loss scaling analysis completed")
    print("5. 🗺️ Coordinate system analysis completed")
    
    print("\n🔧 RECOMMENDED FIXES:")
    print("   • Check encoder → coordinate prediction scaling")
    print("   • Verify target scaler consistency")
    print("   • Balance distillation vs task loss properly")
    print("   • Ensure student learns correct coordinate system")


if __name__ == "__main__":
    main() 