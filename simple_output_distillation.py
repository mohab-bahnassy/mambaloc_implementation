"""
Simple Output-Level Distillation
Direct knowledge transfer from UWB Teacher → CSI Student without encoders.

Architecture:
- Teacher: UWB data → UWB Transformer → coordinates
- Student: CSI data → CSI Mamba → coordinates  
- Distillation: Align student predictions with teacher predictions

This tests whether encoder complexity is necessary or if simple output alignment suffices.
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
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher
from utils.config import Config
sys.path.append('.')


class SimpleOutputDistiller:
    """
    Simple output-level distillation between UWB Teacher and CSI Student.
    No encoders, no latent space - just direct prediction alignment.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_config: dict,
        device: str = "cuda",
        temperature: float = 3.0,
        alpha: float = 0.4,    # Distillation loss weight
        beta: float = 0.6,     # Task loss weight
    ):
        """
        Initialize simple output distiller.
        
        Args:
            teacher_model: Pre-trained UWB transformer teacher
            student_config: Configuration for CSI Mamba student
            device: Device to run on
            temperature: Temperature for distillation softening
            alpha: Weight for distillation loss (teacher → student)
            beta: Weight for task loss (student → targets)
        """
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
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
        
        print(f"🎯 Simple Output Distiller initialized:")
        print(f"   Teacher (UWB): {teacher_params:,} parameters (frozen)")
        print(f"   Student (CSI): {student_params:,} parameters (trainable)")
        print(f"   Temperature: {temperature}, α={alpha}, β={beta}")
        
    def train_teacher(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        learning_rate: float = 1e-3
    ) -> Dict[str, List[float]]:
        """
        Train teacher model from scratch on UWB data.
        """
        print("🎓 Training UWB Teacher from scratch...")
        
        # Unfreeze teacher for training
        for param in self.teacher.parameters():
            param.requires_grad = True
        self.teacher.train()
        
        optimizer = torch.optim.AdamW(self.teacher.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 8
        
        for epoch in range(epochs):
            # Training phase
            self.teacher.train()
            epoch_losses = []
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                uwb_targets = uwb_targets.float().to(self.device)
                
                # Ensure targets are [batch_size, 2] for coordinates
                if uwb_targets.dim() > 2:
                    uwb_targets = uwb_targets[:, -1, :]  # Take last timestep
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
                
                if teacher_predictions.dim() > 2:
                    teacher_predictions = teacher_predictions[:, -1, :]
                if teacher_predictions.shape[-1] != 2:
                    teacher_predictions = teacher_predictions[:, :2]
                
                loss = criterion(teacher_predictions, uwb_targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                if batch_idx % 10 == 0:
                    print(f"  Teacher Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                
                if batch_idx >= 30:  # Reasonable training per epoch
                    break
            
            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.teacher.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(val_loader):
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
                    
                    if batch_idx >= 10:  # Quick validation
                        break
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            scheduler.step(avg_val_loss)
            
            print(f"Teacher Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"💡 Teacher early stopping at epoch {epoch}")
                break
        
        # Freeze teacher after training
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        print(f"✅ Teacher training completed! Best val loss: {best_val_loss:.6f}")
        return history
    
    def train_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        learning_rate: float = 5e-4
    ) -> Dict[str, List[float]]:
        """
        Train student with output-level distillation from teacher.
        """
        print("🎯 Training CSI Student with output-level distillation...")
        
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
        
        history = {
            'total_loss': [],
            'distill_loss': [],
            'task_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
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
                
                if batch_idx % 10 == 0:
                    print(f"  Distill Epoch {epoch}, Batch {batch_idx}: "
                          f"Total={total_loss.item():.6f}, "
                          f"Distill={distill_loss.item():.6f}, "
                          f"Task={task_loss.item():.6f}")
                
                if batch_idx >= 30:  # Reasonable training per epoch
                    break
            
            # Record training losses
            for key in epoch_losses:
                history[key + '_loss'].append(np.mean(epoch_losses[key]))
            
            # Validation phase
            self.student.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(val_loader):
                    csi_data = csi_data.float().to(self.device)
                    csi_targets = csi_targets.float().to(self.device)
                    
                    if csi_targets.dim() > 2:
                        csi_targets = csi_targets[:, -1, :]
                    if csi_targets.shape[-1] != 2:
                        csi_targets = csi_targets[:, :2]
                    
                    student_result = self.student(csi_data, targets=csi_targets)
                    student_predictions = student_result.predictions
                    
                    if student_predictions.dim() > 2:
                        student_predictions = student_predictions[:, -1, :]
                    if student_predictions.shape[-1] != 2:
                        student_predictions = student_predictions[:, :2]
                    
                    val_loss = self.mse_loss(student_predictions, csi_targets)
                    val_losses.append(val_loss.item())
                    
                    if batch_idx >= 10:
                        break
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            scheduler.step(avg_val_loss)
            
            print(f"Distill Epoch {epoch}: "
                  f"Total={history['total_loss'][-1]:.6f}, "
                  f"Val={avg_val_loss:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"💡 Distillation early stopping at epoch {epoch}")
                break
        
        print(f"✅ Distillation training completed! Best val loss: {best_val_loss:.6f}")
        return history


def train_baseline_csi_student(
    csi_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 30,
    learning_rate: float = 5e-4
) -> CSIRegressionModel:
    """
    Train baseline CSI student without any distillation.
    """
    print("📊 Training baseline CSI student (no distillation)...")
    
    baseline_model = CSIRegressionModel(csi_config, device=device)
    baseline_model.to(device)
    
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
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
            
            if batch_idx % 10 == 0:
                print(f"  Baseline Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            if batch_idx >= 30:
                break
        
        avg_train_loss = np.mean(epoch_losses)
        
        # Validation
        baseline_model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(val_loader):
                csi_data = csi_data.float().to(device)
                csi_targets = csi_targets.float().to(device)
                
                if csi_targets.dim() > 2:
                    csi_targets = csi_targets[:, -1, :]
                if csi_targets.shape[-1] != 2:
                    csi_targets = csi_targets[:, :2]
                
                outputs = baseline_model(csi_data, targets=csi_targets)
                predictions = outputs.predictions
                
                if predictions.dim() > 2:
                    predictions = predictions[:, -1, :]
                if predictions.shape[-1] != 2:
                    predictions = predictions[:, :2]
                
                val_loss = criterion(predictions, csi_targets)
                val_losses.append(val_loss.item())
                
                if batch_idx >= 10:
                    break
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        print(f"Baseline Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"💡 Baseline early stopping at epoch {epoch}")
            break
    
    print(f"✅ Baseline training completed! Best val loss: {best_val_loss:.6f}")
    return baseline_model


def evaluate_models(
    baseline_model: nn.Module,
    distilled_model: nn.Module,
    eval_loader: DataLoader,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate both baseline and distilled models on validation data.
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
    
    baseline_mae = np.sqrt(baseline_mse)  # Approximation
    distilled_mae = np.sqrt(distilled_mse)
    
    improvement = ((baseline_mse - distilled_mse) / baseline_mse) * 100
    
    results = {
        'baseline': {
            'mse': baseline_mse,
            'mae': baseline_mae
        },
        'distilled': {
            'mse': distilled_mse,
            'mae': distilled_mae
        },
        'improvement_percent': improvement
    }
    
    print(f"📊 Evaluation Results:")
    print(f"   Baseline MSE:  {baseline_mse:.6f}")
    print(f"   Distilled MSE: {distilled_mse:.6f}")
    print(f"   Improvement:   {improvement:.2f}%")
    
    return results


def main():
    """Main function to run simple output distillation comparison."""
    
    print("🚀 Simple Output-Level Distillation Comparison")
    print("=" * 60)
    print("Testing: UWB Teacher → CSI Student (output-level only)")
    print("No encoders, no latent space - pure prediction alignment")
    print()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Create data loaders
    print("📊 Loading synchronized UWB-CSI data...")
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
        max_samples_per_exp=3000,
        stride=2
    )
    
    # Get feature dimensions
    for uwb_data, uwb_targets, csi_data, csi_targets in train_loader:
        uwb_feature_count = uwb_data.shape[-1]
        csi_feature_count = csi_data.shape[-1]
        break
    
    print(f"📐 Data dimensions: UWB={uwb_feature_count}, CSI={csi_feature_count}")
    
    # Create teacher model
    print("🎓 Creating UWB Teacher model...")
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=uwb_feature_count,
        output_features=2,
        d_model=256,
        n_layers=4,
        n_heads=8
    )
    
    # Create student config
    csi_config = {
        "CSIRegressionModel": {
            "input": {
                "input_features": csi_feature_count,
                "output_features": 2,
                "output_mode": "last",
                "dropout": 0.15
            }
        },
        "UWBMixerModel": {
            "input": {
                "d_model": 256,
                "n_layer": 4,
                "final_prenorm": "layer"
            }
        },
        "Block1": {
            "n_layers": 4,
            "BlockType": "modules.phi_block",
            "block_input": {"resid_dropout": 0.15},
            "CoreType": "modules.mixers.discrete_mamba2",
            "core_input": {
                "d_state": 32, "n_v_heads": 16, "n_qk_heads": 16,
                "d_conv": 4, "conv_bias": True, "expand": 2,
                "chunk_size": 128, "activation": "identity", "bias": False
            }
        }
    }
    
    # Create distiller
    print("🎯 Creating Simple Output Distiller...")
    distiller = SimpleOutputDistiller(
        teacher_model=teacher_model,
        student_config=csi_config,
        device=device,
        temperature=3.5,
        alpha=0.4,  # Balanced distillation
        beta=0.6    # Balanced task loss
    )
    
    # Step 1: Train teacher
    print("\n" + "="*50)
    print("STEP 1: Training UWB Teacher")
    print("="*50)
    
    teacher_history = distiller.train_teacher(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        learning_rate=1e-3
    )
    
    # Step 2: Train baseline student
    print("\n" + "="*50)
    print("STEP 2: Training Baseline CSI Student")
    print("="*50)
    
    baseline_model = train_baseline_csi_student(
        csi_config=csi_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=30,
        learning_rate=5e-4
    )
    
    # Step 3: Train distilled student
    print("\n" + "="*50)
    print("STEP 3: Training Distilled CSI Student")
    print("="*50)
    
    distill_history = distiller.train_distillation(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        learning_rate=5e-4
    )
    
    # Step 4: Compare models
    print("\n" + "="*50)
    print("STEP 4: Model Comparison")
    print("="*50)
    
    results = evaluate_models(
        baseline_model=baseline_model,
        distilled_model=distiller.student,
        eval_loader=val_loader,
        device=device
    )
    
    # Save results
    output_dir = "./simple_output_distillation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    torch.save({
        'teacher_state_dict': distiller.teacher.state_dict(),
        'baseline_state_dict': baseline_model.state_dict(),
        'distilled_state_dict': distiller.student.state_dict(),
        'teacher_history': teacher_history,
        'distill_history': distill_history,
        'evaluation_results': results,
        'config': {
            'teacher_config': {
                'input_features': uwb_feature_count,
                'output_features': 2,
                'd_model': 256,
                'n_layers': 4,
                'n_heads': 8
            },
            'student_config': csi_config,
            'distillation_config': {
                'temperature': 3.5,
                'alpha': 0.4,
                'beta': 0.6
            }
        }
    }, os.path.join(output_dir, 'simple_distillation_results.pth'))
    
    # Save results as JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Simple Output Distillation completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Final Improvement: {results['improvement_percent']:.2f}%")
    
    return results


if __name__ == "__main__":
    results = main() 