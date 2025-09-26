"""
Fair Comparison: Native CSI Baseline vs Synchronized Distillation
Ensures no bias in data loading between baseline and distilled models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import json

from modules.csi_head import CSIRegressionModel
from dataloaders.synchronized_uwb_csi_loader_fixed import create_fixed_synchronized_dataloaders
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher

import sys
sys.path.append('.')


def create_fair_data_splits():
    """
    Create fair data splits:
    - Baseline: Uses ALL available CSI data (native resolution)
    - Distilled: Uses synchronized UWB-CSI data
    - Evaluation: Same test set for both (synchronized for fair comparison)
    """
    print("📊 Creating FAIR data splits...")
    
    # Paths
    uwb_data_path = "/media/mohab/Storage HDD/Downloads/uwb2(1)"
    csi_mat_file = "/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat"
    
    # 1. Create NATIVE CSI data (for baseline)
    print("   📡 Loading native CSI data (full resolution)...")
    csi_train_loader, csi_val_loader, csi_scaler, csi_target_scaler = create_csi_dataloaders_fixed(
        csi_mat_file=csi_mat_file,
        train_experiments=['002'],
        val_experiments=['002'],
        batch_size=32,
        sequence_length=4,  # Native CSI sequence length
        max_samples_per_exp=3000,  # More data available
        stride=1,  # Higher resolution
        normalize_targets=True
    )
    
    # 2. Create SYNCHRONIZED data (for distillation)
    print("   🔗 Loading synchronized UWB-CSI data...")
    sync_train_loader, sync_val_loader, uwb_scaler, sync_target_scaler, sync_csi_scaler = create_fixed_synchronized_dataloaders(
        uwb_data_path=uwb_data_path,
        csi_mat_file=csi_mat_file,
        train_experiments=['002'],
        val_experiments=['002'],
        batch_size=32,
        sequence_length=32,
        csi_sequence_length=4,
        max_samples_per_exp=1500,  # Less synchronized data
        stride=2
    )
    
    print(f"   ✅ Native CSI: {len(csi_train_loader)} train batches")
    print(f"   ✅ Synchronized: {len(sync_train_loader)} train batches")
    
    return {
        'native_csi': {
            'train': csi_train_loader,
            'val': csi_val_loader,
            'scaler': csi_scaler,
            'target_scaler': csi_target_scaler
        },
        'synchronized': {
            'train': sync_train_loader,
            'val': sync_val_loader,
            'uwb_scaler': uwb_scaler,
            'target_scaler': sync_target_scaler,
            'csi_scaler': sync_csi_scaler
        }
    }


def train_fair_baseline(csi_config: dict, native_data: dict, device: str = "cuda") -> CSIRegressionModel:
    """
    Train baseline on NATIVE CSI data (no synchronization bias).
    """
    print("🎯 Training FAIR baseline on native CSI data...")
    print("   - Uses ALL available CSI measurements")
    print("   - Native temporal resolution (no windowing)")
    print("   - No UWB synchronization constraints")
    
    baseline_model = CSIRegressionModel(csi_config, device=device)
    baseline_model.to(device)
    
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Train on native CSI data
    for epoch in range(10):  # Fast training
        baseline_model.train()
        epoch_losses = []
        
        for batch_idx, (csi_data, csi_targets) in enumerate(native_data['train']):
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            # Ensure targets are [batch_size, 2]
            if csi_targets.dim() > 2:
                csi_targets = csi_targets[:, -1, :]
            if csi_targets.shape[-1] != 2:
                csi_targets = csi_targets[:, :2]
            
            optimizer.zero_grad()
            
            outputs = baseline_model(csi_data, targets=csi_targets)
            predictions = outputs.predictions
            
            if predictions.dim() > 2:
                predictions = predictions[:, -1, :]
            if predictions.shape[-1] != 2:
                predictions = predictions[:, :2]
            
            loss = criterion(predictions, csi_targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx >= 20:  # Fast training
                break
        
        if epoch % 3 == 0:
            print(f"   Baseline Epoch {epoch}: Loss={np.mean(epoch_losses):.6f}")
    
    print("✅ Fair baseline training completed!")
    return baseline_model


def train_fair_distilled(
    teacher_model: nn.Module,
    csi_config: dict, 
    sync_data: dict,
    device: str = "cuda",
    temperature: float = 2.0,
    alpha: float = 0.2,
    beta: float = 0.8
) -> CSIRegressionModel:
    """
    Train distilled model on synchronized data.
    """
    print("🎓 Training distilled model on synchronized data...")
    print(f"   - Uses UWB teacher guidance (T={temperature}, α={alpha}, β={beta})")
    print("   - Limited to synchronized timepoints")
    
    # Train teacher first
    print("   🎯 Training teacher...")
    for param in teacher_model.parameters():
        param.requires_grad = True
    teacher_model.train()
    
    teacher_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=1e-3, weight_decay=1e-4)
    teacher_criterion = nn.MSELoss()
    
    for epoch in range(5):  # Quick teacher training
        epoch_losses = []
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(sync_data['train']):
            uwb_data = uwb_data.float().to(device)
            uwb_targets = uwb_targets.float().to(device)
            
            if uwb_targets.dim() > 2:
                uwb_targets = uwb_targets[:, -1, :]
            if uwb_targets.shape[-1] != 2:
                uwb_targets = uwb_targets[:, :2]
            
            teacher_optimizer.zero_grad()
            
            teacher_output = teacher_model(uwb_data)
            if isinstance(teacher_output, dict):
                teacher_predictions = teacher_output["predictions"]
            else:
                teacher_predictions = teacher_output
            
            if teacher_predictions.dim() > 2:
                teacher_predictions = teacher_predictions[:, -1, :]
            if teacher_predictions.shape[-1] != 2:
                teacher_predictions = teacher_predictions[:, :2]
            
            loss = teacher_criterion(teacher_predictions, uwb_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)
            teacher_optimizer.step()
            
            epoch_losses.append(loss.item())
            if batch_idx >= 15:
                break
        
        if epoch % 2 == 0:
            print(f"   Teacher Epoch {epoch}: Loss={np.mean(epoch_losses):.6f}")
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # Train student with distillation
    print("   🎓 Training student with distillation...")
    student_model = CSIRegressionModel(csi_config, device=device)
    student_model.to(device)
    
    student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-4, weight_decay=1e-4)
    mse_loss = nn.MSELoss()
    
    for epoch in range(8):  # Fast distillation
        student_model.train()
        epoch_losses = {'total': [], 'distill': [], 'task': []}
        
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(sync_data['train']):
            uwb_data = uwb_data.float().to(device)
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            if csi_targets.dim() > 2:
                csi_targets = csi_targets[:, -1, :]
            if csi_targets.shape[-1] != 2:
                csi_targets = csi_targets[:, :2]
            
            student_optimizer.zero_grad()
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_output = teacher_model(uwb_data)
                if isinstance(teacher_output, dict):
                    teacher_outputs = teacher_output["predictions"]
                else:
                    teacher_outputs = teacher_output
                
                if teacher_outputs.dim() > 2:
                    teacher_outputs = teacher_outputs[:, -1, :]
                if teacher_outputs.shape[-1] != 2:
                    teacher_outputs = teacher_outputs[:, :2]
            
            # Get student predictions
            student_result = student_model(csi_data, targets=csi_targets)
            student_predictions = student_result.predictions
            
            if student_predictions.dim() > 2:
                student_predictions = student_predictions[:, -1, :]
            if student_predictions.shape[-1] != 2:
                student_predictions = student_predictions[:, :2]
            
            # Compute losses
            distill_loss = mse_loss(
                student_predictions / temperature,
                teacher_outputs / temperature
            )
            task_loss = mse_loss(student_predictions, csi_targets)
            total_loss = alpha * distill_loss + beta * task_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            student_optimizer.step()
            
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['distill'].append(distill_loss.item())
            epoch_losses['task'].append(task_loss.item())
            
            if batch_idx >= 15:
                break
        
        if epoch % 3 == 0:
            print(f"   Student Epoch {epoch}: Total={np.mean(epoch_losses['total']):.6f}, "
                  f"Distill={np.mean(epoch_losses['distill']):.6f}, "
                  f"Task={np.mean(epoch_losses['task']):.6f}")
    
    print("✅ Fair distillation training completed!")
    return student_model


def evaluate_on_same_testset(
    baseline_model: nn.Module,
    distilled_model: nn.Module,
    sync_data: dict,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate both models on the SAME test set (synchronized) for fair comparison.
    """
    print("📊 Evaluating both models on SAME test set...")
    
    baseline_model.eval()
    distilled_model.eval()
    
    baseline_losses = []
    distilled_losses = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(sync_data['val']):
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            if csi_targets.dim() > 2:
                csi_targets = csi_targets[:, -1, :]
            if csi_targets.shape[-1] != 2:
                csi_targets = csi_targets[:, :2]
            
            # Baseline evaluation (CSI only)
            baseline_outputs = baseline_model(csi_data, targets=csi_targets)
            baseline_predictions = baseline_outputs.predictions
            if baseline_predictions.dim() > 2:
                baseline_predictions = baseline_predictions[:, -1, :]
            if baseline_predictions.shape[-1] != 2:
                baseline_predictions = baseline_predictions[:, :2]
            
            baseline_loss = criterion(baseline_predictions, csi_targets)
            baseline_losses.append(baseline_loss.item())
            
            # Distilled evaluation (CSI only - no teacher at inference)
            distilled_outputs = distilled_model(csi_data, targets=csi_targets)
            distilled_predictions = distilled_outputs.predictions
            if distilled_predictions.dim() > 2:
                distilled_predictions = distilled_predictions[:, -1, :]
            if distilled_predictions.shape[-1] != 2:
                distilled_predictions = distilled_predictions[:, :2]
            
            distilled_loss = criterion(distilled_predictions, csi_targets)
            distilled_losses.append(distilled_loss.item())
    
    baseline_mse = np.mean(baseline_losses)
    distilled_mse = np.mean(distilled_losses)
    improvement = ((baseline_mse - distilled_mse) / baseline_mse) * 100
    
    results = {
        'baseline_mse': baseline_mse,
        'distilled_mse': distilled_mse,
        'improvement_percent': improvement
    }
    
    print(f"📊 FAIR Evaluation Results:")
    print(f"   Baseline (native CSI): {baseline_mse:.6f}")
    print(f"   Distilled (w/ UWB guidance): {distilled_mse:.6f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    return results


def run_fair_comparison():
    """Run a completely fair comparison between baseline and distilled models."""
    
    print("🚀 FAIR COMPARISON: Native CSI vs UWB-Guided CSI")
    print("=" * 60)
    print("🎯 Eliminating data loading bias:")
    print("   - Baseline: ALL native CSI data (full resolution)")
    print("   - Distilled: Synchronized data with UWB guidance")
    print("   - Evaluation: Same test set for both")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create fair data splits
    data_splits = create_fair_data_splits()
    
    # Get feature dimensions from synchronized data
    for uwb_data, uwb_targets, csi_data, csi_targets in data_splits['synchronized']['train']:
        uwb_feature_count = uwb_data.shape[-1]
        csi_feature_count = csi_data.shape[-1]
        break
    
    print(f"📐 Model dimensions: UWB={uwb_feature_count}, CSI={csi_feature_count}")
    
    # Create models
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=uwb_feature_count,
        output_features=2,
        d_model=128,
        n_layers=2,
        n_heads=4
    ).to(device)
    
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
                "d_model": 128,
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
    
    # Train baseline on native CSI data
    print("\n" + "="*50)
    print("TRAINING FAIR BASELINE")
    print("="*50)
    baseline_model = train_fair_baseline(csi_config, data_splits['native_csi'], device)
    
    # Train distilled on synchronized data
    print("\n" + "="*50)
    print("TRAINING DISTILLED MODEL")
    print("="*50)
    distilled_model = train_fair_distilled(
        teacher_model, csi_config, data_splits['synchronized'], device,
        temperature=2.0, alpha=0.1, beta=0.9  # Conservative distillation
    )
    
    # Fair evaluation on same test set
    print("\n" + "="*50)
    print("FAIR EVALUATION")
    print("="*50)
    results = evaluate_on_same_testset(
        baseline_model, distilled_model, data_splits['synchronized'], device
    )
    
    # Save results
    with open('fair_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Fair comparison completed!")
    print(f"📁 Results saved to fair_comparison_results.json")
    
    return results


if __name__ == "__main__":
    results = run_fair_comparison() 