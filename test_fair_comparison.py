"""
Fair Comparison Test: Use Same Target Scaler for Baseline and Distilled Models
Fixes the massive coordinate scale difference that was causing poor performance.
"""

import torch
import torch.nn as nn
import numpy as np
from dataloaders.synchronized_uwb_csi_loader import create_synchronized_dataloaders
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from modules.csi_head import CSIRegressionModel
from encoder_enhanced_cross_modal_distillation import EncoderEnhancedMOHAWKDistiller, train_baseline_csi_mamba, evaluate_models_comparison
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher


def create_unified_scalers():
    """Create unified scalers using CSI-only data (the working baseline)."""
    print("🔧 Creating unified scalers based on CSI-only data...")
    
    # Use CSI-only dataloader to create the reference scalers
    csi_train_loader, csi_val_loader, csi_feature_scaler, csi_target_scaler = create_csi_dataloaders_fixed(
        mat_file_path="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
        train_split=0.8,
        batch_size=32,
        sequence_length=4,
        target_tags=['tag4422'],
        use_magnitude_phase=True,
        max_samples=500,
        temporal_gap=0
    )
    
    # Extract scaler parameters
    print(f"📏 CSI Target Scaler (Reference):")
    print(f"   Mean: {csi_target_scaler.mean_}")
    print(f"   Scale: {csi_target_scaler.scale_}")
    
    return csi_feature_scaler, csi_target_scaler, csi_train_loader, csi_val_loader


def create_fixed_synchronized_loader(csi_target_scaler):
    """Create synchronized loader but use the CSI target scaler for consistency."""
    print("🔧 Creating synchronized loader with FIXED target scaling...")
    
    from dataloaders.synchronized_uwb_csi_loader import SynchronizedUWBCSIDataset
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    
    # Create full dataset first
    full_dataset = SynchronizedUWBCSIDataset(
        uwb_data_path="/media/mohab/Storage HDD/Downloads/uwb2(1)",
        csi_mat_file="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
        experiments=["002"],
        sequence_length=32,
        csi_sequence_length=4,
        target_tags=['tag4422'],
        max_samples_per_exp=500,
        use_magnitude_phase=True,
        stride=2
    )
    
    if len(full_dataset) == 0:
        raise ValueError("No synchronized data found")
    
    # Temporal split
    n_samples = len(full_dataset)
    train_end_idx = int(n_samples * 0.8)
    train_indices = list(range(0, train_end_idx))
    val_indices = list(range(train_end_idx, n_samples))
    
    print(f"🔄 Temporal split: {len(train_indices)} train, {len(val_indices)} val samples")
    
    # Collect training data for UWB/CSI feature scalers
    all_uwb_features = []
    all_csi_features = []
    
    for i in train_indices[:min(200, len(train_indices))]:  # Sample for fitting
        uwb_feat, uwb_targ, csi_feat, csi_targ = full_dataset[i]
        all_uwb_features.append(uwb_feat.numpy().reshape(-1, uwb_feat.shape[-1]))
        all_csi_features.append(csi_feat.numpy().reshape(-1, csi_feat.shape[-1]))
    
    uwb_features_array = np.vstack(all_uwb_features)
    csi_features_array = np.vstack(all_csi_features)
    
    # Create feature scalers
    uwb_scaler = StandardScaler()
    csi_scaler = StandardScaler()
    
    uwb_scaler.fit(uwb_features_array)
    csi_scaler.fit(csi_features_array)
    
    print(f"📊 Using FIXED target scaler from CSI-only data (ensures fair comparison)")
    
    # Create datasets with the UNIFIED target scaler
    train_dataset = SynchronizedUWBCSIDataset(
        uwb_data_path="/media/mohab/Storage HDD/Downloads/uwb2(1)",
        csi_mat_file="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
        experiments=["002"],
        sequence_length=32,
        csi_sequence_length=4,
        target_tags=['tag4422'],
        feature_scaler=uwb_scaler,
        target_scaler=csi_target_scaler,  # ✅ Use CSI target scaler
        csi_feature_scaler=csi_scaler,
        max_samples_per_exp=500,
        use_magnitude_phase=True,
        stride=2
    )
    
    val_dataset = SynchronizedUWBCSIDataset(
        uwb_data_path="/media/mohab/Storage HDD/Downloads/uwb2(1)",
        csi_mat_file="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
        experiments=["002"],
        sequence_length=32,
        csi_sequence_length=4,
        target_tags=['tag4422'],
        feature_scaler=uwb_scaler,
        target_scaler=csi_target_scaler,  # ✅ Use CSI target scaler
        csi_feature_scaler=csi_scaler,
        max_samples_per_exp=500,
        use_magnitude_phase=True,
        stride=2
    )
    
    # Apply temporal indices
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✅ Created FIXED synchronized dataloaders:")
    print(f"   Training: {len(train_dataset)} pairs")
    print(f"   Validation: {len(val_dataset)} pairs")
    
    return train_loader, val_loader, uwb_scaler, csi_scaler


def test_scaling_fix():
    """Test the fixed scaling to ensure fair comparison."""
    print("🧪 TESTING SCALING FIX")
    print("=" * 40)
    
    # Step 1: Create unified scalers
    csi_feature_scaler, csi_target_scaler, csi_train_loader, csi_val_loader = create_unified_scalers()
    
    # Step 2: Create synchronized data with FIXED target scaling
    sync_train_loader, sync_val_loader, uwb_scaler, sync_csi_scaler = create_fixed_synchronized_loader(csi_target_scaler)
    
    # Step 3: Compare target ranges
    print("\n📊 AFTER SCALING FIX:")
    
    # Sample CSI-only data
    for csi_data, csi_targets in csi_train_loader:
        print(f"   CSI-only targets: range [{csi_targets.min():.3f}, {csi_targets.max():.3f}]")
        break
    
    # Sample synchronized data
    for uwb_data, uwb_targets, sync_csi_data, sync_csi_targets in sync_train_loader:
        print(f"   Sync CSI targets: range [{sync_csi_targets.min():.3f}, {sync_csi_targets.max():.3f}]")
        print(f"   UWB targets: range [{uwb_targets.min():.3f}, {uwb_targets.max():.3f}]")
        break
    
    print(f"✅ Target ranges should now be similar!")
    
    return {
        'csi_train_loader': csi_train_loader,
        'csi_val_loader': csi_val_loader,
        'sync_train_loader': sync_train_loader,
        'sync_val_loader': sync_val_loader,
        'csi_target_scaler': csi_target_scaler,
        'uwb_scaler': uwb_scaler,
        'sync_csi_scaler': sync_csi_scaler
    }


def train_with_balanced_losses(sync_train_loader, sync_val_loader, uwb_scaler, sync_csi_scaler, csi_target_scaler):
    """Train distillation with properly balanced losses."""
    print("\n🚀 Training with BALANCED LOSSES")
    print("=" * 40)
    
    # Get feature dimensions
    for uwb_data, uwb_targets, csi_data, csi_targets in sync_train_loader:
        uwb_feature_count = uwb_data.shape[-1]
        csi_feature_count = csi_data.shape[-1]
        break
    
    # Create teacher model
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=uwb_feature_count,
        output_features=2,
        d_model=256,
        n_layers=4,
        n_heads=8
    )
    
    # Create CSI config
    csi_config = {
        "CSIRegressionModel": {
            "input": {
                "input_features": csi_feature_count,
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
    
    # Create distiller with BALANCED loss weights
    distiller = EncoderEnhancedMOHAWKDistiller(
        teacher_model=teacher_model,
        csi_student_config=csi_config,
        uwb_input_features=uwb_feature_count,
        csi_input_features=csi_feature_count,
        latent_dim=128,
        device="cuda",
        temperature=4.0,
        alpha=0.5,  # Reduced distillation weight
        beta=0.5,   # Increased task weight for balance
    )
    
    # Quick MOHAWK training with balanced losses
    print("🎯 Running BALANCED MOHAWK distillation...")
    training_history = distiller.full_mohawk_distillation(
        train_loader=sync_train_loader,
        val_loader=sync_val_loader,
        learning_rate=1e-3,
        pretrain_epochs=2,
        stage1_epochs=2,
        stage2_epochs=2,
        stage3_epochs=3,
    )
    
    # Get deployment model
    deployment_model = distiller.get_csi_student_for_deployment()
    
    return deployment_model, training_history


def run_fair_comparison():
    """Run a fair comparison with unified scaling."""
    print("🏁 RUNNING FAIR COMPARISON")
    print("=" * 40)
    
    # Step 1: Fix the scaling
    data_loaders = test_scaling_fix()
    
    # Step 2: Train baseline model
    print("\n📊 Training baseline model...")
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
    
    try:
        baseline_model = train_baseline_csi_mamba(
            csi_config=csi_config,
            train_loader=data_loaders['csi_train_loader'],
            val_loader=data_loaders['csi_val_loader'],
            device="cuda",
            epochs=15,  # More epochs for better baseline
            learning_rate=1e-3
        )
        baseline_success = True
    except Exception as e:
        print(f"❌ Baseline failed (expected): {e}")
        baseline_success = False
    
    # Step 3: Train distilled model with balanced losses
    print("\n📊 Training distilled model with balanced losses...")
    try:
        distilled_model, history = train_with_balanced_losses(
            data_loaders['sync_train_loader'],
            data_loaders['sync_val_loader'],
            data_loaders['uwb_scaler'],
            data_loaders['sync_csi_scaler'],
            data_loaders['csi_target_scaler']
        )
        distillation_success = True
    except Exception as e:
        print(f"❌ Distillation failed: {e}")
        distillation_success = False
    
    # Step 4: Compare if both succeeded
    if baseline_success and distillation_success:
        print("\n📊 Running FAIR comparison...")
        comparison_results = evaluate_models_comparison(
            baseline_model=baseline_model,
            distilled_model=distilled_model,
            eval_loader=data_loaders['csi_val_loader'],
            device="cuda"
        )
        
        print(f"🎉 FAIR comparison completed!")
        return comparison_results
    else:
        print(f"⚠️ Cannot compare: Baseline={baseline_success}, Distillation={distillation_success}")
        return None


if __name__ == "__main__":
    results = run_fair_comparison() 