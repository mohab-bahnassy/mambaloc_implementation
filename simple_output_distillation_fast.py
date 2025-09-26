"""
Fast Simple Output-Level Distillation with Hyperparameter Tuning
Reduced epochs for faster experimentation with different hyperparameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
import sys
import time

from modules.csi_head import CSIRegressionModel
from dataloaders.synchronized_uwb_csi_loader_fixed import create_fixed_synchronized_dataloaders
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher

sys.path.append('.')


class FastSimpleOutputDistiller:
    """
    Fast simple output-level distillation with tunable hyperparameters.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_config: dict,
        device: str = "cuda",
        temperature: float = 2.0,      # Reduced from 3.5
        alpha: float = 0.2,            # Reduced distillation weight
        beta: float = 0.8,             # Increased task weight
        teacher_epochs: int = 8,       # Reduced from 20
        distill_epochs: int = 15,      # Reduced from 30
        baseline_epochs: int = 15,     # Reduced from 30
    ):
        """
        Initialize fast distiller with tunable hyperparameters.
        """
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.teacher_epochs = teacher_epochs
        self.distill_epochs = distill_epochs
        self.baseline_epochs = baseline_epochs
        
        # Setup teacher (frozen)
        self.teacher = teacher_model.to(device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Setup student (trainable)
        self.student = CSIRegressionModel(student_config, device=device)
        self.student.to(device)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        
        print(f"🎯 Fast Simple Output Distiller initialized:")
        print(f"   Teacher (UWB): {teacher_params:,} parameters (frozen)")
        print(f"   Student (CSI): {student_params:,} parameters (trainable)")
        print(f"   Temperature: {temperature}, α={alpha}, β={beta}")
        print(f"   Epochs: Teacher={teacher_epochs}, Distill={distill_epochs}, Baseline={baseline_epochs}")
        
    def train_teacher(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3
    ) -> Dict[str, List[float]]:
        """
        Train teacher model with reduced epochs.
        """
        print(f"🎓 Training UWB Teacher ({self.teacher_epochs} epochs)...")
        
        # Unfreeze teacher for training
        for param in self.teacher.parameters():
            param.requires_grad = True
        self.teacher.train()
        
        optimizer = torch.optim.AdamW(self.teacher.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.teacher_epochs):
            # Training phase
            self.teacher.train()
            epoch_losses = []
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                uwb_targets = uwb_targets.float().to(self.device)
                
                # Ensure targets are [batch_size, 2] for coordinates
                if uwb_targets.dim() > 2:
                    uwb_targets = uwb_targets[:, -1, :]
                if uwb_targets.shape[-1] != 2:
                    uwb_targets = uwb_targets[:, :2]
                
                optimizer.zero_grad()
                
                # Teacher prediction: UWB → coordinates
                teacher_output = self.teacher(uwb_data)
                
                # Extract predictions from teacher output dict
                if isinstance(teacher_output, dict):
                    teacher_predictions = teacher_output["predictions"]
                else:
                    teacher_predictions = teacher_output
                
                # Ensure predictions match target shape
                if teacher_predictions.dim() > 2:
                    teacher_predictions = teacher_predictions[:, -1, :]
                if teacher_predictions.shape[-1] != 2:
                    teacher_predictions = teacher_predictions[:, :2]
                
                loss = criterion(teacher_predictions, uwb_targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                if batch_idx >= 15:  # Reduced batches per epoch
                    break
            
            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Quick validation
            self.teacher.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                    uwb_data = uwb_data.float().to(self.device)
                    uwb_targets = uwb_targets.float().to(self.device)
                    
                    if uwb_targets.dim() > 2:
                        uwb_targets = uwb_targets[:, -1, :]
                    if uwb_targets.shape[-1] != 2:
                        uwb_targets = uwb_targets[:, :2]
                    
                    teacher_output = self.teacher(uwb_data)
                    
                    # Extract predictions from teacher output dict
                    if isinstance(teacher_output, dict):
                        teacher_predictions = teacher_output["predictions"]
                    else:
                        teacher_predictions = teacher_output
                    
                    if teacher_predictions.dim() > 2:
                        teacher_predictions = teacher_predictions[:, -1, :]
                    if teacher_predictions.shape[-1] != 2:
                        teacher_predictions = teacher_predictions[:, :2]
                    
                    val_loss = criterion(teacher_predictions, uwb_targets)
                    val_losses.append(val_loss.item())
                    
                    if batch_idx >= 5:  # Quick validation
                        break
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Teacher Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        # Freeze teacher after training
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        print(f"✅ Teacher training completed!")
        return history
    
    def train_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 5e-4
    ) -> Dict[str, List[float]]:
        """
        Train student with output-level distillation (reduced epochs).
        """
        print(f"🎯 Training CSI Student with distillation ({self.distill_epochs} epochs)...")
        print(f"   Hyperparameters: T={self.temperature}, α={self.alpha}, β={self.beta}")
        
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        history = {
            'total_loss': [],
            'distill_loss': [],
            'task_loss': [],
            'val_loss': []
        }
        
        for epoch in range(self.distill_epochs):
            # Training phase
            self.student.train()
            epoch_losses = {'total': [], 'distill': [], 'task': []}
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                csi_targets = csi_targets.float().to(self.device)
                
                # Ensure targets are [batch_size, 2]
                if csi_targets.dim() > 2:
                    csi_targets = csi_targets[:, -1, :]
                if csi_targets.shape[-1] != 2:
                    csi_targets = csi_targets[:, :2]
                
                optimizer.zero_grad()
                
                # Get teacher predictions (frozen)
                with torch.no_grad():
                    teacher_output = self.teacher(uwb_data)
                    
                    # Extract predictions from teacher output dict
                    if isinstance(teacher_output, dict):
                        teacher_outputs = teacher_output["predictions"]
                    else:
                        teacher_outputs = teacher_output
                    
                    if teacher_outputs.dim() > 2:
                        teacher_outputs = teacher_outputs[:, -1, :]
                    if teacher_outputs.shape[-1] != 2:
                        teacher_outputs = teacher_outputs[:, :2]
                
                # Get student predictions
                student_result = self.student(csi_data, targets=csi_targets)
                student_predictions = student_result.predictions
                
                if student_predictions.dim() > 2:
                    student_predictions = student_predictions[:, -1, :]
                if student_predictions.shape[-1] != 2:
                    student_predictions = student_predictions[:, :2]
                
                # Compute losses
                # 1. Distillation loss: student learns from teacher
                distill_loss = self.mse_loss(
                    student_predictions / self.temperature,
                    teacher_outputs / self.temperature
                )
                
                # 2. Task loss: student learns CSI → coordinates mapping
                task_loss = self.mse_loss(student_predictions, csi_targets)
                
                # 3. Combined loss
                total_loss = self.alpha * distill_loss + self.beta * task_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Log losses
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['distill'].append(distill_loss.item())
                epoch_losses['task'].append(task_loss.item())
                
                if batch_idx >= 15:  # Reduced batches per epoch
                    break
            
            # Record training losses
            for key in epoch_losses:
                history[key + '_loss'].append(np.mean(epoch_losses[key]))
            
            if epoch % 3 == 0:  # Less frequent logging
                print(f"Distill Epoch {epoch}: "
                      f"Total={history['total_loss'][-1]:.6f}, "
                      f"Distill={history['distill_loss'][-1]:.6f}, "
                      f"Task={history['task_loss'][-1]:.6f}")
        
        print(f"✅ Distillation training completed!")
        return history


def train_fast_baseline_csi_student(
    csi_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 15,
    learning_rate: float = 5e-4
) -> CSIRegressionModel:
    """
    Train baseline CSI student (reduced epochs).
    """
    print(f"📊 Training baseline CSI student ({epochs} epochs)...")
    
    baseline_model = CSIRegressionModel(csi_config, device=device)
    baseline_model.to(device)
    
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training
        baseline_model.train()
        epoch_losses = []
        
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
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
            
            if batch_idx >= 15:  # Reduced batches per epoch
                break
        
        avg_train_loss = np.mean(epoch_losses)
        
        if epoch % 3 == 0:  # Less frequent logging
            print(f"Baseline Epoch {epoch}: Train={avg_train_loss:.6f}")
    
    print(f"✅ Baseline training completed!")
    return baseline_model


def evaluate_models_fast(
    baseline_model: nn.Module,
    distilled_model: nn.Module,
    eval_loader: DataLoader,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Fast evaluation of both models.
    """
    print("📊 Evaluating models...")
    
    baseline_model.eval()
    distilled_model.eval()
    
    baseline_losses = []
    distilled_losses = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(eval_loader):
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            if csi_targets.dim() > 2:
                csi_targets = csi_targets[:, -1, :]
            if csi_targets.shape[-1] != 2:
                csi_targets = csi_targets[:, :2]
            
            # Baseline predictions
            baseline_outputs = baseline_model(csi_data, targets=csi_targets)
            baseline_predictions = baseline_outputs.predictions
            if baseline_predictions.dim() > 2:
                baseline_predictions = baseline_predictions[:, -1, :]
            if baseline_predictions.shape[-1] != 2:
                baseline_predictions = baseline_predictions[:, :2]
            
            baseline_loss = criterion(baseline_predictions, csi_targets)
            baseline_losses.append(baseline_loss.item())
            
            # Distilled predictions  
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
        'baseline': {'mse': baseline_mse},
        'distilled': {'mse': distilled_mse},
        'improvement_percent': improvement
    }
    
    print(f"📊 Evaluation Results:")
    print(f"   Baseline MSE:  {baseline_mse:.6f}")
    print(f"   Distilled MSE: {distilled_mse:.6f}")
    print(f"   Improvement:   {improvement:.2f}%")
    
    return results


def run_hyperparameter_experiment(
    temperature: float = 2.0,
    alpha: float = 0.2,
    beta: float = 0.8,
    teacher_lr: float = 1e-3,
    student_lr: float = 5e-4
):
    """Run a single hyperparameter configuration experiment."""
    
    print(f"\n🧪 EXPERIMENT: T={temperature}, α={alpha}, β={beta}")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create data loaders
    uwb_data_path = "/media/mohab/Storage HDD/Downloads/uwb2(1)"
    csi_mat_file = "/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat"
    
    train_loader, val_loader, uwb_scaler, target_scaler, csi_scaler = create_fixed_synchronized_dataloaders(
        uwb_data_path=uwb_data_path,
        csi_mat_file=csi_mat_file,
        train_experiments=['002'],
        val_experiments=['002'],
        batch_size=32,
        sequence_length=32,
        csi_sequence_length=4,
        max_samples_per_exp=1500,  # Reduced dataset for speed
        stride=2
    )
    
    # Get feature dimensions
    for uwb_data, uwb_targets, csi_data, csi_targets in train_loader:
        uwb_feature_count = uwb_data.shape[-1]
        csi_feature_count = csi_data.shape[-1]
        break
    
    # Create teacher model (smaller for speed)
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=uwb_feature_count,
        output_features=2,
        d_model=128,  # Smaller
        n_layers=2,   # Fewer layers
        n_heads=4     # Fewer heads
    )
    
    # Create student config (smaller for speed)
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
                "d_model": 128,  # Smaller
                "n_layer": 2,    # Fewer layers
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
    
    # Create distiller
    distiller = FastSimpleOutputDistiller(
        teacher_model=teacher_model,
        student_config=csi_config,
        device=device,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        teacher_epochs=5,   # Very fast
        distill_epochs=8,   # Very fast
        baseline_epochs=8   # Very fast
    )
    
    # Train teacher
    print("\n🎓 Training Teacher...")
    distiller.train_teacher(train_loader, val_loader, learning_rate=teacher_lr)
    
    # Train baseline
    print("\n📊 Training Baseline...")
    baseline_model = train_fast_baseline_csi_student(
        csi_config, train_loader, val_loader, device, 
        epochs=distiller.baseline_epochs, learning_rate=student_lr
    )
    
    # Train distilled
    print("\n🎯 Training Distilled...")
    distiller.train_distillation(train_loader, val_loader, learning_rate=student_lr)
    
    # Evaluate
    print("\n📊 Evaluating...")
    results = evaluate_models_fast(baseline_model, distiller.student, val_loader, device)
    
    return results


def main():
    """Run multiple hyperparameter experiments."""
    
    print("🚀 Fast Simple Output Distillation Hyperparameter Search")
    print("=" * 60)
    
    # Experiment configurations
    experiments = [
        # Test different temperatures
        {"temperature": 1.0, "alpha": 0.2, "beta": 0.8, "name": "Low Temperature"},
        {"temperature": 2.0, "alpha": 0.2, "beta": 0.8, "name": "Medium Temperature"},
        {"temperature": 4.0, "alpha": 0.2, "beta": 0.8, "name": "High Temperature"},
        
        # Test different loss weights
        {"temperature": 2.0, "alpha": 0.1, "beta": 0.9, "name": "Low Distillation Weight"},
        {"temperature": 2.0, "alpha": 0.3, "beta": 0.7, "name": "Medium Distillation Weight"},
        {"temperature": 2.0, "alpha": 0.5, "beta": 0.5, "name": "Equal Weights"},
        
        # Test no distillation (pure baseline comparison)
        {"temperature": 2.0, "alpha": 0.0, "beta": 1.0, "name": "No Distillation (Pure Task)"},
    ]
    
    results_summary = []
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}: {exp['name']}")
        print(f"{'='*60}")
        
        try:
            results = run_hyperparameter_experiment(
                temperature=exp["temperature"],
                alpha=exp["alpha"],
                beta=exp["beta"]
            )
            
            results["config"] = exp
            results_summary.append(results)
            
            print(f"\n✅ {exp['name']}: {results['improvement_percent']:.2f}% improvement")
            
        except Exception as e:
            print(f"❌ {exp['name']} failed: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*60}")
    
    # Sort by improvement
    results_summary.sort(key=lambda x: x['improvement_percent'], reverse=True)
    
    for i, result in enumerate(results_summary):
        config = result["config"]
        print(f"{i+1}. {config['name']}")
        print(f"   T={config['temperature']}, α={config['alpha']}, β={config['beta']}")
        print(f"   Improvement: {result['improvement_percent']:.2f}%")
        print(f"   Baseline: {result['baseline']['mse']:.6f}, Distilled: {result['distilled']['mse']:.6f}")
        print()
    
    # Save results
    with open('hyperparameter_search_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("📁 Results saved to hyperparameter_search_results.json")
    return results_summary


if __name__ == "__main__":
    results = main() 