"""
Simple Fix: Use Identical CSI Data for Both Baseline and Distillation
This ensures 100% identical coordinate systems by using the exact same data processing.
"""

import torch
import torch.nn as nn
import numpy as np
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from modules.csi_head import CSIRegressionModel
from encoder_enhanced_cross_modal_distillation import EncoderEnhancedMOHAWKDistiller, evaluate_models_comparison
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher


def create_identical_datasets():
    """Create datasets with identical CSI processing for fair comparison."""
    print("🔧 Creating IDENTICAL datasets for fair comparison...")
    
    # Use CSI-only dataloader (the working baseline approach)
    csi_train_loader, csi_val_loader, csi_feature_scaler, csi_target_scaler = create_csi_dataloaders_fixed(
        mat_file_path="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
        train_split=0.8,
        batch_size=32,
        sequence_length=4,
        target_tags=['tag4422'],
        use_magnitude_phase=True,
        max_samples=300,  # More data for better results
        temporal_gap=10   # Prevent leakage
    )
    
    # Create synthetic UWB data that's correlated with CSI (for distillation)
    print("🎯 Creating synthetic UWB data correlated with CSI...")
    
    from torch.utils.data import Dataset, DataLoader
    
    class SyntheticUWBCSIDataset(Dataset):
        """Dataset that creates synthetic UWB data correlated with real CSI data."""
        
        def __init__(self, csi_loader, uwb_seq_len=32, uwb_features=100):
            self.data = []
            
            # Extract all CSI data and create correlated UWB data
            for csi_data, csi_targets in csi_loader:
                batch_size = csi_data.shape[0]
                
                for i in range(batch_size):
                    csi_seq = csi_data[i]  # [4, 280]
                    target = csi_targets[i]  # [2]
                    
                    # Create synthetic UWB data correlated with target
                    uwb_data = torch.randn(uwb_seq_len, uwb_features) * 0.5
                    
                    # Add target correlation to first few features
                    uwb_data[:, :2] += target.unsqueeze(0) * 0.2
                    
                    # Add some CSI correlation
                    csi_summary = torch.mean(csi_seq, dim=0)  # [280]
                    uwb_data[:, 2:6] += csi_summary[:4].unsqueeze(0) * 0.1
                    
                    self.data.append((uwb_data, target, csi_seq, target))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Create synthetic datasets
    print(f"📊 Creating synthetic UWB-CSI training data...")
    synthetic_train_dataset = SyntheticUWBCSIDataset(csi_train_loader)
    synthetic_val_dataset = SyntheticUWBCSIDataset(csi_val_loader)
    
    synthetic_train_loader = DataLoader(synthetic_train_dataset, batch_size=32, shuffle=True)
    synthetic_val_loader = DataLoader(synthetic_val_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Created identical datasets:")
    print(f"   CSI-only: {len(csi_train_loader.dataset)} train, {len(csi_val_loader.dataset)} val")
    print(f"   Synthetic UWB-CSI: {len(synthetic_train_dataset)} train, {len(synthetic_val_dataset)} val")
    
    # Verify identical scaling
    for csi_data, csi_targets in csi_train_loader:
        print(f"   CSI-only targets: range [{csi_targets.min():.3f}, {csi_targets.max():.3f}]")
        break
    
    for uwb_data, uwb_targets, sync_csi_data, sync_csi_targets in synthetic_train_loader:
        print(f"   Synthetic CSI targets: range [{sync_csi_targets.min():.3f}, {sync_csi_targets.max():.3f}]")
        print(f"   ✅ IDENTICAL scaling guaranteed!")
        break
    
    return {
        'csi_train_loader': csi_train_loader,
        'csi_val_loader': csi_val_loader,
        'synthetic_train_loader': synthetic_train_loader,
        'synthetic_val_loader': synthetic_val_loader,
        'csi_target_scaler': csi_target_scaler
    }


def train_lightweight_distillation(synthetic_train_loader, synthetic_val_loader):
    """Train a lightweight distillation with balanced losses."""
    print("🚀 Training LIGHTWEIGHT distillation with BALANCED losses...")
    
    # Get feature dimensions
    for uwb_data, uwb_targets, csi_data, csi_targets in synthetic_train_loader:
        uwb_feature_count = uwb_data.shape[-1]
        csi_feature_count = csi_data.shape[-1]
        break
    
    # Create simple teacher model
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=uwb_feature_count,
        output_features=2,
        d_model=128,  # Smaller model
        n_layers=2,   # Fewer layers
        n_heads=4     # Fewer heads
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
                "d_model": 32,  # Much smaller
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
                "d_state": 8, "n_v_heads": 4, "n_qk_heads": 4,  # Much smaller
                "d_conv": 4, "conv_bias": True, "expand": 1,
                "chunk_size": 32, "activation": "identity", "bias": False
            }
        }
    }
    
    # Create distiller with VERY BALANCED losses
    distiller = EncoderEnhancedMOHAWKDistiller(
        teacher_model=teacher_model,
        csi_student_config=csi_config,
        uwb_input_features=uwb_feature_count,
        csi_input_features=csi_feature_count,
        latent_dim=64,  # Smaller latent space
        device="cuda",
        temperature=4.0,
        alpha=0.3,  # Much lower distillation weight
        beta=0.7,   # Much higher task weight for balance
    )
    
    # Very quick MOHAWK training
    print("🎯 Running ULTRA-FAST MOHAWK distillation...")
    training_history = distiller.full_mohawk_distillation(
        train_loader=synthetic_train_loader,
        val_loader=synthetic_val_loader,
        learning_rate=1e-3,
        pretrain_epochs=1,  # Minimal training
        stage1_epochs=1,
        stage2_epochs=1,
        stage3_epochs=2,
    )
    
    # Get deployment model
    deployment_model = distiller.get_csi_student_for_deployment()
    
    return deployment_model, training_history


def train_baseline_on_identical_data(csi_train_loader, csi_val_loader):
    """Train baseline model on the identical CSI data."""
    print("📊 Training baseline on IDENTICAL CSI data...")
    
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
                "d_model": 32,  # Same size as distilled model
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
                "d_state": 8, "n_v_heads": 4, "n_qk_heads": 4,  # Same size
                "d_conv": 4, "conv_bias": True, "expand": 1,
                "chunk_size": 32, "activation": "identity", "bias": False
            }
        }
    }
    
    baseline_model = CSIRegressionModel(csi_config, device="cuda")
    baseline_model.cuda()
    
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # Quick training
    for epoch in range(10):
        baseline_model.train()
        epoch_losses = []
        
        for batch_idx, (csi_data, csi_targets) in enumerate(csi_train_loader):
            csi_data = csi_data.float().cuda()
            csi_targets = csi_targets.float().cuda()
            
            optimizer.zero_grad()
            
            outputs = baseline_model(csi_data, targets=csi_targets)
            loss = criterion(outputs.predictions, csi_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx >= 10:  # Quick training
                break
        
        avg_loss = np.mean(epoch_losses)
        
        # Quick validation
        baseline_model.eval()
        val_losses = []
        with torch.no_grad():
            for csi_data, csi_targets in csi_val_loader:
                csi_data = csi_data.float().cuda()
                csi_targets = csi_targets.float().cuda()
                
                outputs = baseline_model(csi_data, targets=csi_targets)
                loss = criterion(outputs.predictions, csi_targets)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(baseline_model.state_dict(), 'best_baseline_identical.pth')
    
    print(f"✅ Baseline training completed! Best val loss: {best_val_loss:.6f}")
    return baseline_model


def run_identical_comparison():
    """Run comparison with identical data processing."""
    print("🏁 RUNNING IDENTICAL DATA COMPARISON")
    print("=" * 50)
    
    # Step 1: Create identical datasets
    datasets = create_identical_datasets()
    
    # Step 2: Train baseline
    try:
        baseline_model = train_baseline_on_identical_data(
            datasets['csi_train_loader'], 
            datasets['csi_val_loader']
        )
        baseline_success = True
    except Exception as e:
        print(f"❌ Baseline failed: {e}")
        baseline_success = False
    
    # Step 3: Train distillation
    try:
        distilled_model, history = train_lightweight_distillation(
            datasets['synthetic_train_loader'],
            datasets['synthetic_val_loader']
        )
        distillation_success = True
    except Exception as e:
        print(f"❌ Distillation failed: {e}")
        import traceback
        traceback.print_exc()
        distillation_success = False
    
    # Step 4: Compare if both succeeded  
    if baseline_success and distillation_success:
        print("\n📊 Running IDENTICAL comparison...")
        comparison_results = evaluate_models_comparison(
            baseline_model=baseline_model,
            distilled_model=distilled_model,
            eval_loader=datasets['csi_val_loader'],
            device="cuda"
        )
        
        print(f"🎉 IDENTICAL comparison completed!")
        return comparison_results
    else:
        print(f"⚠️ Cannot compare: Baseline={baseline_success}, Distillation={distillation_success}")
        return None


if __name__ == "__main__":
    results = run_identical_comparison() 