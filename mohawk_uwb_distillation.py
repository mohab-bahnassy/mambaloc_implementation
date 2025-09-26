"""
MOHAWK Distillation for UWB Regression Tasks
Adapted from the original phi-mamba MOHAWK implementation for UWB sensor data regression.

Three-stage distillation strategy:
Stage 1: Matrix mixer alignment between teacher and student
Stage 2: Hidden state alignment between teacher and student layers  
Stage 3: Full model output alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple

from modules.uwb_head import UWBRegressionModel
from utils.config import Config


class MOHAWKUWBDistiller:
    def __init__(
        self,
        teacher_model: nn.Module,
        student_config: dict,
        device: str = "cuda",
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        """
        Initialize MOHAWK distiller for UWB regression tasks.
        
        Args:
            teacher_model: Pre-trained teacher model (can be transformer or other architecture)
            student_config: Configuration for UWB student model
            device: Device to run on
            temperature: Temperature for distillation
            alpha: Weight for distillation loss
            beta: Weight for task loss
        """
        self.teacher = teacher_model
        self.device = device
        self.temperature = temperature
        self.alpha = alpha  # distillation loss weight
        self.beta = beta    # task loss weight
        
        # Convert to Config if needed
        if not isinstance(student_config, Config):
            config = Config.from_dict(student_config)
        else:
            config = student_config
        
        # Create student model
        self.student = UWBRegressionModel(student_config, device=device)
        
        # Move teacher to device and freeze
        self.teacher = teacher_model.to(device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.student.to(device)
        
        # Get dimensions
        self.teacher_dim = teacher_model.d_model
        self.student_dim = self.student.backbone.input_projection.out_features  # Get from actual model
        
        # Add projection layers to handle dimension mismatch
        self.teacher_to_student_projections = nn.ModuleList()
        self.attention_to_mixer_projections = nn.ModuleList()
        
        # Create projections for each student layer
        num_student_layers = len(self.student.backbone.layers)
        for i in range(num_student_layers):
            # Project teacher hidden states to student dimension
            self.teacher_to_student_projections.append(
                nn.Linear(self.teacher_dim, self.student_dim, device=device)
            )
            
            # Project teacher attention matrices to student mixer dimension
            # For matrix alignment, we need to handle different matrix sizes
            self.attention_to_mixer_projections.append(
                nn.Linear(self.teacher_dim, self.student_dim, device=device)
            )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        print(f"📐 Dimension setup: Teacher {self.teacher_dim}D → Student {self.student_dim}D")
        print(f"🔧 Created {num_student_layers} projection layers for alignment")
        
        # Ensure all parameters are trainable
        self._configure_student_parameters()
        
    def _configure_student_parameters(self):
        """
        Configure which parameters to train - ensure all parameters are trainable
        """
        # Set all student parameters to require gradients
        self.student.requires_grad_(True)
        
        # Set all projection layer parameters to require gradients
        for projection_layer in self.teacher_to_student_projections:
            projection_layer.requires_grad_(True)
        for projection_layer in self.attention_to_mixer_projections:
            projection_layer.requires_grad_(True)
            
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.student.parameters())
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        projection_params = sum(p.numel() for p in self.teacher_to_student_projections.parameters()) + \
                          sum(p.numel() for p in self.attention_to_mixer_projections.parameters())
        
        print(f"🎯 Parameter Status:")
        print(f"   Student model: {trainable_params:,}/{total_params:,} trainable ({trainable_params/total_params*100:.1f}%)")
        print(f"   Projection layers: {projection_params:,} parameters")
        print(f"   Total trainable: {trainable_params + projection_params:,} parameters")
        
    def verify_all_parameters_trainable(self):
        """
        Verify that all parameters are actually trainable
        """
        all_trainable = True
        non_trainable_params = []
        
        # Check student parameters
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                all_trainable = False
                non_trainable_params.append(f"student.{name}")
        
        # Check projection parameters
        for i, proj in enumerate(self.teacher_to_student_projections):
            for name, param in proj.named_parameters():
                if not param.requires_grad:
                    all_trainable = False
                    non_trainable_params.append(f"teacher_to_student_proj[{i}].{name}")
        
        for i, proj in enumerate(self.attention_to_mixer_projections):
            for name, param in proj.named_parameters():
                if not param.requires_grad:
                    all_trainable = False
                    non_trainable_params.append(f"attention_to_mixer_proj[{i}].{name}")
        
        if all_trainable:
            print("✅ All parameters are trainable")
        else:
            print(f"⚠️ Found {len(non_trainable_params)} non-trainable parameters:")
            for param_name in non_trainable_params[:5]:  # Show first 5
                print(f"   - {param_name}")
            if len(non_trainable_params) > 5:
                print(f"   ... and {len(non_trainable_params) - 5} more")
        
        return all_trainable
    
    def stage1_distill(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        layer_idx: int = 0,
        freeze_mlp: bool = True,
    ) -> List[float]:
        """
        Stage 1: Matrix mixer distillation (MOHAWK Matrix Orientation)
        Compare MIXING MATRICES: Teacher's Softmax(QK^T) vs Student's materialized SSM matrix.
        This aligns the fundamental transformation operations, not the outputs.
        """
        print(f"Starting Stage 1 distillation for layer {layer_idx}")
        self.student.train()
        
        # Verify all parameters are trainable
        self.verify_all_parameters_trainable()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (input_data, targets) in enumerate(train_loader):
                input_data = input_data.float().to(self.device)
                targets = targets.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Get teacher attention weights (mixing matrices)
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(
                        input_data,
                        return_attention=True,
                        return_hidden_states=True
                    )
                
                # Get student input from teacher hidden states
                teacher_hidden = teacher_outputs["all_hidden_states"][layer_idx]
                projected_input = self.teacher_to_student_projections[layer_idx](teacher_hidden)
                
                # Get student layer output with transfer matrix
                student_layer = self.student.backbone.layers[layer_idx]
                student_output = student_layer(
                    hidden_states=projected_input,
                    run_mlp_component=not freeze_mlp,
                    return_mixer_matrix=True
                )
                
                # Stage 1: Compare MIXING MATRICES (as per MOHAWK paper)
                if "all_attention_matrices" in teacher_outputs and "transfer_matrix" in student_output:
                    # Teacher mixing matrix: Softmax(QK^T) - attention weights
                    teacher_attention_weights = teacher_outputs["all_attention_matrices"][layer_idx]
                    # Student mixing matrix: materialized SSM matrix
                    student_transfer_matrix = student_output["transfer_matrix"]
                    
                    # Handle shape differences between attention heads and transfer matrix
                    teacher_shape = teacher_attention_weights.shape
                    student_shape = student_transfer_matrix.shape
                    
                    if len(teacher_shape) == 3 and len(student_shape) == 4:
                        # Teacher: [batch, seq, seq], Student: [batch, heads, seq, seq]
                        batch_size = teacher_shape[0]
                        seq_len = min(teacher_shape[1], student_shape[2])
                        
                        # Ensure same sequence length
                        teacher_weights = teacher_attention_weights[:, :seq_len, :seq_len]  # [batch, seq, seq]
                        student_matrix = student_transfer_matrix[:, :, :seq_len, :seq_len]  # [batch, heads, seq, seq]
                        
                        # Average student heads to match teacher format: [batch, seq, seq]
                        student_matrix_avg = student_matrix.mean(dim=1)  # [batch, seq, seq]
                        
                        # Matrix alignment loss using Frobenius norm (as per MOHAWK paper)
                        matrix_loss = torch.linalg.matrix_norm(
                            student_matrix_avg - teacher_weights, ord="fro"
                        ).mean()
                        
                    elif len(teacher_shape) == 4 and len(student_shape) == 4:
                        # Both are [batch, heads, seq, seq] format
                        batch_size = teacher_shape[0]
                        seq_len = min(teacher_shape[2], student_shape[2])
                        
                        # Ensure same sequence length
                        teacher_weights = teacher_attention_weights[:, :, :seq_len, :seq_len]
                        student_matrix = student_transfer_matrix[:, :, :seq_len, :seq_len]
                        
                        # Handle different number of heads by averaging
                        if teacher_weights.size(1) != student_matrix.size(1):
                            teacher_weights = teacher_weights.mean(dim=1, keepdim=True)
                            student_matrix = student_matrix.mean(dim=1, keepdim=True)
                        
                        # Matrix alignment loss using Frobenius norm (as per paper)
                        matrix_loss = torch.linalg.matrix_norm(
                            student_matrix - teacher_weights, ord="fro"
                        ).mean()
                    
                    else:
                        print(f"⚠️ Unexpected matrix shapes - Teacher: {teacher_shape}, Student: {student_shape}")
                        # Flatten and compare if shapes are too different
                        teacher_flat = teacher_attention_weights.flatten()
                        student_flat = student_transfer_matrix.flatten()
                        min_size = min(teacher_flat.size(0), student_flat.size(0))
                        matrix_loss = F.mse_loss(student_flat[:min_size], teacher_flat[:min_size])
                
                else:
                    print(f"⚠️ Missing matrices for Stage 1 comparison")
                    matrix_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                matrix_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student.parameters() if p.requires_grad] + 
                    [p for p in self.teacher_to_student_projections.parameters() if p.requires_grad] + 
                    [p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_losses.append(matrix_loss.item())
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Matrix Loss: {matrix_loss.item():.6f}")
                
                if batch_idx >= 20:  # Quick training for testing
                    break
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Stage 1 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        return losses
    
    def stage2_distill(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 15,
        layer_idx: int = 0,
        freeze_mlp: bool = True,
    ) -> List[float]:
        """
        Stage 2: Hidden state distillation (MOHAWK Hidden-State Alignment)  
        Compare FULL BLOCK OUTPUTS: Teacher's complete attention block output vs Student's complete Mamba block output.
        This aligns the final representations after complete block processing.
        """
        print(f"Starting Stage 2 distillation for layer {layer_idx}")
        self.student.train()
        
        # Verify all parameters are trainable
        self.verify_all_parameters_trainable()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (input_data, targets) in enumerate(train_loader):
                input_data = input_data.float().to(self.device)
                targets = targets.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Get teacher full block outputs
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(
                        input_data,
                        return_hidden_states=True
                    )
                
                # Get student input from teacher hidden states  
                teacher_hidden = teacher_outputs["all_hidden_states"][layer_idx]
                projected_input = self.teacher_to_student_projections[layer_idx](teacher_hidden)
                
                # Get student full block output
                student_layer = self.student.backbone.layers[layer_idx]
                student_output = student_layer(
                    hidden_states=projected_input,
                    run_mlp_component=not freeze_mlp,
                    return_hidden_states=not freeze_mlp
                )
                
                # Stage 2: Compare FULL BLOCK OUTPUTS (as per MOHAWK paper)
                # Teacher: Full attention block output (after complete block processing)
                teacher_block_output = teacher_outputs["all_hidden_states"][layer_idx + 1]
                
                # Student: Full Mamba block output (after complete block processing)  
                student_block_output = student_output["hidden_states"]
                
                # Project teacher block output to student dimension
                projected_teacher_output = self.teacher_to_student_projections[layer_idx](teacher_block_output)
                
                # Ensure compatible shapes for comparison
                if projected_teacher_output.shape != student_block_output.shape:
                    min_seq = min(projected_teacher_output.size(1), student_block_output.size(1))
                    projected_teacher_output = projected_teacher_output[:, :min_seq, :]
                    student_block_output = student_block_output[:, :min_seq, :]
                
                # Hidden state alignment loss using L2 norm (as per paper)
                hidden_loss = torch.norm(
                    student_block_output - projected_teacher_output, p=2, dim=(-1,)
                ).mean()
                
                hidden_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student.parameters() if p.requires_grad] + 
                    [p for p in self.teacher_to_student_projections.parameters() if p.requires_grad] + 
                    [p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_losses.append(hidden_loss.item())
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Hidden Loss: {hidden_loss.item():.6f}")
                
                if batch_idx >= 20:  # Quick training for testing
                    break
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Stage 2 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        return losses
    
    def stage3_distill(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 20,
    ) -> List[float]:
        """
        Stage 3: Full model distillation
        Align the final outputs between teacher and student models.
        """
        print("Starting Stage 3 full model distillation")
        self.student.train()
        
        # Verify all parameters are trainable
        self.verify_all_parameters_trainable()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (input_data, targets) in enumerate(train_loader):
                input_data = input_data.float().to(self.device)  # Convert to float32
                targets = targets.float().to(self.device)        # Convert to float32
                
                optimizer.zero_grad()
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_predictions(input_data)
                
                # Get student predictions
                student_outputs = self.student(input_data, targets=targets)
                
                # Combined loss: distillation + task loss
                distill_loss = self.mse_loss(
                    student_outputs.predictions / self.temperature,
                    teacher_outputs / self.temperature
                )
                
                task_loss = student_outputs.loss if student_outputs.loss is not None else 0
                
                total_loss = self.alpha * distill_loss + self.beta * task_loss
                
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student.parameters() if p.requires_grad] + 
                    [p for p in self.teacher_to_student_projections.parameters() if p.requires_grad] + 
                    [p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, "
                          f"Total Loss: {total_loss.item():.6f}, "
                          f"Distill: {distill_loss.item():.6f}, "
                          f"Task: {task_loss:.6f}")
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Stage 3 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        return losses
    
    def full_mohawk_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        stage1_epochs: int = 10,
        stage2_epochs: int = 15,
        stage3_epochs: int = 20,
    ) -> Dict[str, List[float]]:
        """
        Run the complete 3-stage MOHAWK distillation process.
        """
        print("Starting complete MOHAWK distillation for UWB regression")
        
        all_losses = {
            "stage1": [],
            "stage2": [],
            "stage3": []
        }
        
        # Stage 1: Matrix mixer distillation for each layer
        for layer_idx in range(len(self.student.backbone.layers)):
            # Ensure all parameters are trainable for Stage 1
            self._configure_student_parameters()
            
            # Include ALL trainable parameters (student + all projections)
            all_trainable_params = []
            all_trainable_params.extend([p for p in self.student.parameters() if p.requires_grad])
            all_trainable_params.extend([p for p in self.teacher_to_student_projections.parameters() if p.requires_grad])
            all_trainable_params.extend([p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad])
            
            optimizer = torch.optim.AdamW(
                all_trainable_params,  # Train ALL parameters
                lr=learning_rate
            )
            
            stage1_losses = self.stage1_distill(
                train_loader, optimizer, stage1_epochs, layer_idx
            )
            all_losses["stage1"].extend(stage1_losses)
        
        # Stage 2: Hidden state distillation for each layer
        for layer_idx in range(len(self.student.backbone.layers)):
            # Ensure all parameters are trainable for Stage 2
            self._configure_student_parameters()
            
            # Include ALL trainable parameters (student + all projections)
            all_trainable_params = []
            all_trainable_params.extend([p for p in self.student.parameters() if p.requires_grad])
            all_trainable_params.extend([p for p in self.teacher_to_student_projections.parameters() if p.requires_grad])
            all_trainable_params.extend([p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad])
            
            optimizer = torch.optim.AdamW(
                all_trainable_params,  # Train ALL parameters
                lr=learning_rate
            )
            
            stage2_losses = self.stage2_distill(
                train_loader, optimizer, stage2_epochs, layer_idx
            )
            all_losses["stage2"].extend(stage2_losses)
        
        # Stage 3: Full model distillation
        # Ensure all parameters are trainable for Stage 3
        self._configure_student_parameters()
        
        # Include ALL trainable parameters (student + all projections)
        all_trainable_params = []
        all_trainable_params.extend([p for p in self.student.parameters() if p.requires_grad])
        all_trainable_params.extend([p for p in self.teacher_to_student_projections.parameters() if p.requires_grad])
        all_trainable_params.extend([p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad])
            
        optimizer = torch.optim.AdamW(
            all_trainable_params,  # Train ALL parameters
            lr=learning_rate
        )
        stage3_losses = self.stage3_distill(train_loader, optimizer, stage3_epochs)
        all_losses["stage3"] = stage3_losses
        
        # Final validation
        self.evaluate(val_loader)
        
        return all_losses
    
    def _get_teacher_outputs(self, input_data, **kwargs):
        """
        Get teacher model outputs. This method should be adapted based on teacher architecture.
        """
        # This is a placeholder - adapt based on your teacher model architecture
        if hasattr(self.teacher, 'forward'):
            return self.teacher(input_data, **kwargs)
        else:
            raise NotImplementedError("Adapt this method for your specific teacher model")
    
    def _get_teacher_predictions(self, input_data):
        """
        Get teacher predictions for distillation.
        """
        teacher_output = self.teacher(input_data)
        
        # Extract predictions tensor from teacher output dict
        if isinstance(teacher_output, dict):
            return teacher_output["predictions"]
        else:
            return teacher_output
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the student model on validation data.
        """
        self.student.eval()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_data, targets in val_loader:
                input_data = input_data.float().to(self.device)  # Convert to float32
                targets = targets.float().to(self.device)        # Convert to float32
                
                outputs = self.student(input_data, targets=targets)
                
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                
                mae = self.l1_loss(outputs.predictions, targets)
                total_mae += mae.item()
                num_batches += 1
        
        metrics = {
            "mse_loss": total_loss / num_batches,
            "mae": total_mae / num_batches
        }
        
        print(f"Validation - MSE: {metrics['mse_loss']:.6f}, MAE: {metrics['mae']:.6f}")
        
        return metrics


# Example usage function
def create_uwb_distillation_setup(teacher_model, train_loader, val_loader, device="cuda"):
    """
    Helper function to set up UWB distillation with reasonable defaults.
    """
    # UWB student configuration
    student_config = Config.from_json("phi-mamba/uwb_config.json")
    
    # Create distiller
    distiller = MOHAWKUWBDistiller(
        teacher_model=teacher_model,
        student_config=student_config.to_dict(),  # Convert Config to dict
        device=device,
        temperature=4.0,
        alpha=0.7,
        beta=0.3
    )
    
    return distiller 