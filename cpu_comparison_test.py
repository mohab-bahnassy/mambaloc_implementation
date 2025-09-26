"""
CPU-Based Comparison: Test Distillation vs Simple Baseline (No Mamba)
Avoids CUDA issues by using CPU and simpler models for fair comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from encoder_enhanced_cross_modal_distillation import EncoderEnhancedMOHAWKDistiller


class SimpleCPUBaseline(nn.Module):
    """Simple MLP baseline that works on CPU without Mamba issues."""
    
    def __init__(self, input_features=280, output_features=2, hidden_dim=128):
        super().__init__()
        
        # Simple feedforward network
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_features * 4, hidden_dim),  # 4 timesteps
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_features)
        )
    
    def forward(self, x, targets=None):
        # x: [batch_size, seq_len, features]
        batch_size = x.shape[0]
        x_flat = self.flatten(x)  # [batch_size, seq_len * features]
        predictions = self.network(x_flat)  # [batch_size, 2]
        
        loss = None
        if targets is not None:
            loss = nn.functional.mse_loss(predictions, targets)
        
        return type('Output', (), {
            'predictions': predictions,
            'loss': loss
        })()


def train_simple_baseline_cpu(train_loader, val_loader, device="cpu"):
    """Train simple baseline on CPU."""
    print("📊 Training Simple CPU Baseline (No Mamba)...")
    
    model = SimpleCPUBaseline(input_features=280, output_features=2, hidden_dim=64)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(15):
        model.train()
        epoch_losses = []
        
        for batch_idx, (csi_data, csi_targets) in enumerate(train_loader):
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            optimizer.zero_grad()
            
            outputs = model(csi_data, targets=csi_targets)
            loss = criterion(outputs.predictions, csi_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for csi_data, csi_targets in val_loader:
                csi_data = csi_data.float().to(device)
                csi_targets = csi_targets.float().to(device)
                
                outputs = model(csi_data, targets=csi_targets)
                loss = criterion(outputs.predictions, csi_targets)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_cpu_baseline.pth')
    
    print(f"✅ CPU Baseline completed! Best val loss: {best_val_loss:.6f}")
    return model


def create_cpu_synthetic_data():
    """Create synthetic data for CPU-based comparison."""
    print("🔧 Creating synthetic data for CPU comparison...")
    
    # Use CSI-only dataloader
    csi_train_loader, csi_val_loader, csi_feature_scaler, csi_target_scaler = create_csi_dataloaders_fixed(
        mat_file_path="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
        train_split=0.8,
        batch_size=16,  # Smaller batch for CPU
        sequence_length=4,
        target_tags=['tag4422'],
        use_magnitude_phase=True,
        max_samples=150,  # Smaller dataset for faster CPU training
        temporal_gap=5
    )
    
    # Create synthetic UWB data
    from torch.utils.data import Dataset, DataLoader
    
    class CPUSyntheticDataset(Dataset):
        def __init__(self, csi_loader, uwb_seq_len=16, uwb_features=64):  # Smaller for CPU
            self.data = []
            
            for csi_data, csi_targets in csi_loader:
                batch_size = csi_data.shape[0]
                
                for i in range(batch_size):
                    csi_seq = csi_data[i]
                    target = csi_targets[i]
                    
                    # Create synthetic UWB data
                    uwb_data = torch.randn(uwb_seq_len, uwb_features) * 0.3
                    uwb_data[:, :2] += target.unsqueeze(0) * 0.3
                    
                    self.data.append((uwb_data, target, csi_seq, target))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    synthetic_train_dataset = CPUSyntheticDataset(csi_train_loader)
    synthetic_val_dataset = CPUSyntheticDataset(csi_val_loader)
    
    synthetic_train_loader = DataLoader(synthetic_train_dataset, batch_size=16, shuffle=True)
    synthetic_val_loader = DataLoader(synthetic_val_dataset, batch_size=16, shuffle=False)
    
    print(f"✅ Created synthetic data: {len(synthetic_train_dataset)} train, {len(synthetic_val_dataset)} val")
    
    return {
        'csi_train_loader': csi_train_loader,
        'csi_val_loader': csi_val_loader,
        'synthetic_train_loader': synthetic_train_loader,
        'synthetic_val_loader': synthetic_val_loader
    }


def train_cpu_distillation(synthetic_train_loader, synthetic_val_loader, device="cpu"):
    """Train distillation on CPU with smaller models."""
    print("🚀 Training CPU Distillation...")
    
    from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher
    
    # Get dimensions
    for uwb_data, uwb_targets, csi_data, csi_targets in synthetic_train_loader:
        uwb_feature_count = uwb_data.shape[-1]
        csi_feature_count = csi_data.shape[-1]
        break
    
    # Create small teacher model for CPU
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=uwb_feature_count,
        output_features=2,
        d_model=64,  # Much smaller
        n_layers=2,
        n_heads=4
    )
    
    # Create small CSI config for CPU
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
                "d_model": 16,  # Very small
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
                "d_state": 4, "n_v_heads": 2, "n_qk_heads": 2,  # Very small
                "d_conv": 4, "conv_bias": True, "expand": 1,
                "chunk_size": 16, "activation": "identity", "bias": False
            }
        }
    }
    
    # Create distiller for CPU
    distiller = EncoderEnhancedMOHAWKDistiller(
        teacher_model=teacher_model,
        csi_student_config=csi_config,
        uwb_input_features=uwb_feature_count,
        csi_input_features=csi_feature_count,
        latent_dim=32,  # Small latent space
        device=device,
        temperature=2.0,
        alpha=0.4,
        beta=0.6,
    )
    
    # Move to CPU
    distiller.teacher.to(device)
    distiller.student.to(device)
    distiller.encoder_uwb.to(device)
    distiller.encoder_csi.to(device)
    
    # Quick training
    print("🎯 Running CPU MOHAWK distillation...")
    try:
        training_history = distiller.full_mohawk_distillation(
            train_loader=synthetic_train_loader,
            val_loader=synthetic_val_loader,
            learning_rate=1e-3,
            pretrain_epochs=1,
            stage1_epochs=1,
            stage2_epochs=1,
            stage3_epochs=1,
        )
        
        deployment_model = distiller.get_csi_student_for_deployment()
        deployment_model.to(device)
        return deployment_model, training_history
        
    except Exception as e:
        print(f"❌ CPU Distillation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def evaluate_cpu_comparison(baseline_model, distilled_model, eval_loader, device="cpu"):
    """Compare models on CPU."""
    print("📊 Evaluating CPU comparison...")
    
    baseline_model.eval()
    distilled_model.eval()
    
    baseline_losses = []
    distilled_losses = []
    baseline_maes = []
    distilled_maes = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for csi_data, csi_targets in eval_loader:
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            # Baseline predictions
            baseline_outputs = baseline_model(csi_data, targets=csi_targets)
            baseline_loss = criterion(baseline_outputs.predictions, csi_targets)
            baseline_mae = torch.mean(torch.abs(baseline_outputs.predictions - csi_targets))
            
            baseline_losses.append(baseline_loss.item())
            baseline_maes.append(baseline_mae.item())
            
            # Distilled model predictions
            distilled_preds = distilled_model(csi_data)
            distilled_loss = criterion(distilled_preds, csi_targets)
            distilled_mae = torch.mean(torch.abs(distilled_preds - csi_targets))
            
            distilled_losses.append(distilled_loss.item())
            distilled_maes.append(distilled_mae.item())
    
    results = {
        'baseline': {
            'mse_loss': np.mean(baseline_losses),
            'mae': np.mean(baseline_maes),
        },
        'distilled': {
            'mse_loss': np.mean(distilled_losses),
            'mae': np.mean(distilled_maes),
        }
    }
    
    # Calculate improvement
    mse_improvement = (results['baseline']['mse_loss'] - results['distilled']['mse_loss']) / results['baseline']['mse_loss'] * 100
    mae_improvement = (results['baseline']['mae'] - results['distilled']['mae']) / results['baseline']['mae'] * 100
    
    print("📊 CPU Comparison Results:")
    print("=" * 40)
    print(f"Simple CPU Baseline:")
    print(f"  MSE Loss: {results['baseline']['mse_loss']:.6f}")
    print(f"  MAE: {results['baseline']['mae']:.6f}")
    print()
    print(f"Encoder-Enhanced Distilled Model:")
    print(f"  MSE Loss: {results['distilled']['mse_loss']:.6f}")
    print(f"  MAE: {results['distilled']['mae']:.6f}")
    print()
    print(f"🎯 Improvements:")
    print(f"  MSE Improvement: {mse_improvement:+.2f}%")
    print(f"  MAE Improvement: {mae_improvement:+.2f}%")
    
    return results


def run_cpu_comparison():
    """Run complete CPU-based comparison."""
    print("🖥️ RUNNING CPU-BASED COMPARISON")
    print("=" * 40)
    
    device = "cpu"
    
    # Step 1: Create data
    datasets = create_cpu_synthetic_data()
    
    # Step 2: Train baseline
    baseline_model = train_simple_baseline_cpu(
        datasets['csi_train_loader'], 
        datasets['csi_val_loader'], 
        device=device
    )
    
    # Step 3: Train distillation
    distilled_model, history = train_cpu_distillation(
        datasets['synthetic_train_loader'],
        datasets['synthetic_val_loader'],
        device=device
    )
    
    # Step 4: Compare
    if baseline_model is not None and distilled_model is not None:
        results = evaluate_cpu_comparison(
            baseline_model, distilled_model, 
            datasets['csi_val_loader'], device=device
        )
        return results
    else:
        print("⚠️ Comparison failed")
        return None


if __name__ == "__main__":
    results = run_cpu_comparison() 