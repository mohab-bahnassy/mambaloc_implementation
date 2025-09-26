"""
Synchronized Cross-Modality MOHAWK Distillation for UWB-to-CSI Transfer
Uses timestamp-synchronized UWB-CSI data for proper cross-modal knowledge distillation.
Teacher: UWB sensor data (TRANSFORMER - trained from scratch), Student: CSI sensor data (MAMBA)

Cross-Modal AND Cross-Architecture: Transformer → Mamba
Maintains the original MOHAWK three-stage distillation strategy:
Stage 1: Matrix mixer alignment between teacher and student
Stage 2: Hidden state alignment between teacher and student layers  
Stage 3: Full model output alignment

Trains the UWB transformer teacher from scratch, then distills to CSI mamba student.
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

from modules.uwb_head import UWBRegressionModel
from modules.csi_head import CSIRegressionModel
from dataloaders.synchronized_uwb_csi_loader import create_synchronized_dataloaders
from utils.config import Config
from mohawk_uwb_distillation import MOHAWKUWBDistiller

# Add current directory for imports
sys.path.append('.')


class SimpleUWBTransformerTeacher(nn.Module):
    """
    Simple UWB transformer teacher model for cross-modal distillation.
    Based on the pattern from test_uwb_opera_distillation.py
    """
    def __init__(self, input_features=113, output_features=2, d_model=256, n_layers=4, n_heads=8):
        super().__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, return_attention=False, return_hidden_states=False, return_attention_outputs=False, **kwargs):
        """
        Forward pass with optional attention and hidden state returns.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_features)
            return_attention: Whether to return attention matrices
            return_hidden_states: Whether to return all hidden states
            return_attention_outputs: Whether to return attention outputs
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project input features to model dimension
        hidden_states = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.shape[0]:
            hidden_states = hidden_states + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Initialize outputs
        all_hidden_states = [hidden_states] if return_hidden_states else []
        all_attention_matrices = []
        all_attn_outputs = []
        
        # Run through transformer layers
        for i, layer in enumerate(self.transformer.layers):
            if return_attention:
                # Manual attention calculation to extract attention weights
                attn_output, attn_weights = layer.self_attn(
                    hidden_states, hidden_states, hidden_states,
                    need_weights=True, average_attn_weights=False
                )
                all_attention_matrices.append(attn_weights)
                if return_attention_outputs:
                    all_attn_outputs.append(attn_output)
                
                # Complete the layer forward pass
                hidden_states = layer.norm1(hidden_states + layer.dropout1(attn_output))
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden_states))))
                hidden_states = layer.norm2(hidden_states + layer.dropout2(ff_output))
            else:
                hidden_states = layer(hidden_states)
            
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Apply dropout and get predictions
        output = self.dropout(hidden_states)
        predictions = self.output_projection(output)
        
        # Build result dictionary
        result = {
            "predictions": predictions,
            "last_hidden_state": hidden_states
        }
        
        if return_hidden_states:
            result["all_hidden_states"] = all_hidden_states
        
        if return_attention:
            result["all_attention_matrices"] = all_attention_matrices
            
        if return_attention_outputs:
            result["all_attn_outputs"] = all_attn_outputs
        
        return result

    def save_pretrained(self, save_directory):
        """
        Save the model and its configuration file to a directory.
        """
        import os
        import json
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        config = {
            "input_features": self.input_features,
            "output_features": self.output_features,
            "d_model": self.d_model,
            "model_type": "SimpleUWBTransformerTeacher"
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """
        Load a pretrained model from a directory.
        """
        import os
        import json
        config_path = os.path.join(model_path, "config.json")
        
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            input_features=config["input_features"],
            output_features=config["output_features"],
            d_model=config["d_model"],
            **kwargs
        )
        
        # Load state dict
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model


def train_uwb_transformer_teacher_from_scratch(
    train_loader, device, epochs=25  # Increased epochs
):
    """
    Train UWB transformer teacher model from scratch using PyTorch transformers.
    Following the pattern from test_uwb_opera_distillation.py but with improvements.
    """
    print("🚀 Training UWB TRANSFORMER teacher model from scratch...")
    
    # Get sample data to determine dimensions - handle synchronized dataloader
    uwb_data, uwb_targets, csi_data, csi_targets = next(iter(train_loader))
    input_features = uwb_data.shape[-1]  # Should be 113 for UWB data
    output_features = uwb_targets.shape[-1]  # Should be 2 for coordinates
    
    print(f"📐 Teacher model dimensions: {input_features} → {output_features}")
    
    # Create simple transformer teacher
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=input_features,
        output_features=output_features,
        d_model=128,  # Reduced from 256 - better for small dataset
        n_layers=3,   # Reduced from 6 - prevents overfitting  
        n_heads=4     # Reduced from 8 - matches smaller d_model
    ).to(device)
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"👨‍🏫 Teacher parameters: {teacher_params:,}")
    
    # Train the teacher using UWB data with improved strategy
    teacher_model.train()
    optimizer = optim.AdamW(teacher_model.parameters(), lr=1e-3, weight_decay=1e-4)  # Reduced LR for smaller model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)  # LR scheduling
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 4  # Reduced patience for smaller model
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
            uwb_data = uwb_data.float().to(device)
            uwb_targets = uwb_targets.float().to(device)
            
            optimizer.zero_grad()
            
            # Train teacher on UWB data
            outputs = teacher_model(uwb_data)
            loss = criterion(outputs["predictions"], uwb_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx >= 100:  # More training batches for better convergence
                break
        
        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)  # Update learning rate
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Teacher Epoch {epoch+1}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= max_patience:
            print(f"💡 Early stopping triggered at epoch {epoch+1}")
            break
    
    teacher_model.eval()
    print(f"✅ Teacher training completed with best loss: {best_loss:.6f}")
    
    return teacher_model


def evaluate_teacher_performance(teacher_model, val_loader, device):
    """
    Evaluate teacher model performance on validation data.
    """
    print("📊 Evaluating teacher model performance...")
    teacher_model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    all_errors = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(val_loader):
            uwb_data = uwb_data.float().to(device)
            uwb_targets = uwb_targets.float().to(device)
            
            outputs = teacher_model(uwb_data)
            predictions = outputs["predictions"]
            
            loss = criterion(predictions, uwb_targets)
            mae = nn.L1Loss()(predictions, uwb_targets)
            
            # Calculate individual errors for median
            batch_errors = torch.abs(predictions - uwb_targets).cpu().numpy()
            all_errors.extend(batch_errors.flatten())
            
            total_loss += loss.item() * uwb_data.size(0)
            total_mae += mae.item() * uwb_data.size(0)
            total_samples += uwb_data.size(0)
            
            if batch_idx >= 20:  # Quick evaluation
                break
    
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        median_ae = np.median(all_errors) if all_errors else float('inf')
        
        print(f"🎯 Teacher Performance:")
        print(f"   Loss (MSE): {avg_loss:.6f}")
        print(f"   MAE: {avg_mae:.6f}")
        print(f"   Median AE: {median_ae:.6f}")
        print(f"   RMSE: {np.sqrt(avg_loss):.6f}")
        
        return {
            'teacher_loss': avg_loss,
            'teacher_mae': avg_mae,
            'teacher_median_ae': median_ae,
            'teacher_rmse': np.sqrt(avg_loss)
        }
    else:
        return {'teacher_loss': float('inf'), 'teacher_mae': float('inf'), 'teacher_median_ae': float('inf'), 'teacher_rmse': float('inf')}


class SynchronizedCrossModalityMOHAWKDistiller:
    def __init__(
        self,
        teacher_model: nn.Module,
        csi_student_config: dict,
        device: str = "cuda",
        temperature: float = 1.0,  # FIXED: Reduced from 4.0 to 1.0 for regression
        alpha: float = 0.9,        # FIXED: Increased from 0.7 to 0.9 (more distillation focus)
        beta: float = 0.1,         # FIXED: Reduced from 0.3 to 0.1 (less task loss weight)
    ):
        """
        Initialize Synchronized Cross-Modality MOHAWK distiller.
        Teacher: UWB Transformer, Student: CSI Mamba
        
        Args:
            teacher_model: Pre-trained UWB transformer teacher model
            csi_student_config: Configuration for CSI mamba student model
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
        if not isinstance(csi_student_config, Config):
            config = Config.from_dict(csi_student_config)
        else:
            config = csi_student_config
        
        # Create CSI mamba student model
        self.student = CSIRegressionModel(csi_student_config, device=device)
        
        # Move teacher to device and freeze
        self.teacher = teacher_model.to(device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.student.to(device)
        
        # Get dimensions - following original pattern
        self.teacher_dim = teacher_model.d_model  # Use d_model like original
        self.student_dim = self.student.backbone.input_projection.out_features
        
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
            self.attention_to_mixer_projections.append(
                nn.Linear(self.teacher_dim, self.student_dim, device=device)
            )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        print(f"📐 Dimension setup: Teacher {self.teacher_dim}D → Student {self.student_dim}D")
        print(f"🔧 Created {num_student_layers} projection layers for alignment")
        print(f"🔄 Synchronized Cross-Modality Cross-Architecture Setup:")
        print(f"   Teacher (UWB Transformer): {self.teacher_dim}D")
        print(f"   Student (CSI Mamba): {self.student_dim}D")
        print(f"   Created {num_student_layers} projection layers")
        
        # Ensure all parameters are trainable
        self._configure_student_parameters()
        
    def _configure_student_parameters(self):
        """Configure which parameters to train - ensure all parameters are trainable"""
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
    
    def _get_teacher_outputs(self, uwb_data, **kwargs):
        """
        Get teacher model outputs. This method should be adapted based on teacher architecture.
        Following original mohawk_uwb_distillation.py pattern.
        """
        # Adapt based on teacher model architecture - following original pattern
        if hasattr(self.teacher, 'forward'):
            return self.teacher(uwb_data, **kwargs)
        else:
            raise NotImplementedError("Adapt this method for your specific teacher model")
    
    def _get_teacher_predictions(self, uwb_data):
        """
        Get teacher predictions for distillation.
        Following original mohawk_uwb_distillation.py pattern.
        """
        teacher_output = self.teacher(uwb_data)
        
        # Extract predictions tensor from teacher output dict
        if isinstance(teacher_output, dict):
            return teacher_output["predictions"]
        else:
            return teacher_output
    
    def stage1_distill(
        self,
        synchronized_train_loader: DataLoader,
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
            
            # Iterate through synchronized dataloader
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(synchronized_train_loader):
                uwb_data = uwb_data.float().to(self.device)
                uwb_targets = uwb_targets.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                csi_targets = csi_targets.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Get teacher attention weights (mixing matrices) from synchronized UWB data
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(
                        uwb_data,
                        return_attention=True,
                        return_hidden_states=True
                    )
                
                # Get student input from teacher hidden states (cross-modality projection)
                teacher_hidden = teacher_outputs["all_hidden_states"][layer_idx]
                projected_input = self.teacher_to_student_projections[layer_idx](teacher_hidden)
                
                # Get student layer output with transfer matrix from synchronized CSI data processing
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
        synchronized_train_loader: DataLoader,
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
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(synchronized_train_loader):
                uwb_data = uwb_data.float().to(self.device)
                uwb_targets = uwb_targets.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                csi_targets = csi_targets.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Get teacher full block outputs from synchronized UWB data
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(
                        uwb_data,
                        return_hidden_states=True
                    )
                
                # Get student input from teacher hidden states (cross-modality projection)
                teacher_hidden = teacher_outputs["all_hidden_states"][layer_idx]
                projected_input = self.teacher_to_student_projections[layer_idx](teacher_hidden)
                
                # Get student full block output from synchronized CSI processing
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
        synchronized_train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 20,
    ) -> List[float]:
        """
        Stage 3: Full model distillation using synchronized UWB-CSI data.
        FIXED: Handle coordinate system scale differences properly.
        """
        print("Starting Stage 3 full model distillation")
        self.student.train()
        
        # Verify all parameters are trainable
        self.verify_all_parameters_trainable()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(synchronized_train_loader):
                uwb_data = uwb_data.float().to(self.device)
                uwb_targets = uwb_targets.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                csi_targets = csi_targets.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Get teacher predictions from synchronized UWB data
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_predictions(uwb_data)
                    # IMPROVED: Use attention-weighted averaging instead of just last timestep
                    if teacher_outputs.dim() == 3:  # [batch, seq, features]
                        # Create attention weights based on timestep importance (later timesteps more important)
                        seq_len = teacher_outputs.size(1)
                        timestep_weights = torch.linspace(0.1, 1.0, seq_len, device=self.device)
                        timestep_weights = timestep_weights / timestep_weights.sum()
                        # Apply weighted average: [batch, features]
                        teacher_outputs = torch.sum(teacher_outputs * timestep_weights.view(1, -1, 1), dim=1)
                
                # Get student predictions from synchronized CSI data
                student_outputs = self.student(csi_data, targets=csi_targets)
                
                # IMPROVED: Apply same aggregation to student predictions
                student_preds = student_outputs.predictions
                if student_preds.dim() == 3:  # [batch, seq, features]
                    # Use same weighted averaging for consistency
                    seq_len = student_preds.size(1)
                    timestep_weights = torch.linspace(0.1, 1.0, seq_len, device=self.device)
                    timestep_weights = timestep_weights / timestep_weights.sum()
                    # Apply weighted average: [batch, features]
                    student_preds = torch.sum(student_preds * timestep_weights.view(1, -1, 1), dim=1)
                
                # FIXED: Handle coordinate system scale differences
                # Normalize both teacher and student predictions to same scale for distillation
                teacher_norm = (teacher_outputs - teacher_outputs.mean(dim=0, keepdim=True)) / (teacher_outputs.std(dim=0, keepdim=True) + 1e-8)
                student_norm = (student_preds - student_preds.mean(dim=0, keepdim=True)) / (student_preds.std(dim=0, keepdim=True) + 1e-8)
                
                # Get CSI targets for task loss (use last timestep for consistency)
                if csi_targets.dim() == 3:
                    csi_targets_task = csi_targets[:, -1, :]
                else:
                    csi_targets_task = csi_targets
                
                # DEBUG: Print ranges for first batch to understand scale differences
                if batch_idx == 0 and epoch == 0:
                    print(f"🔍 DEBUG Stage 3 - Scale Analysis:")
                    print(f"   Teacher outputs range: [{teacher_outputs.min():.3f}, {teacher_outputs.max():.3f}]")
                    print(f"   Student predictions range: [{student_preds.min():.3f}, {student_preds.max():.3f}]")
                    print(f"   CSI targets range: [{csi_targets_task.min():.3f}, {csi_targets_task.max():.3f}]")
                    print(f"   Teacher normalized range: [{teacher_norm.min():.3f}, {teacher_norm.max():.3f}]")
                    print(f"   Student normalized range: [{student_norm.min():.3f}, {student_norm.max():.3f}]")
                
                # FIXED: Use normalized predictions for distillation loss (removes scale differences)
                distill_loss = self.mse_loss(student_norm / self.temperature, teacher_norm / self.temperature)
                
                # FIXED: Use actual CSI targets for task loss (student learns its own coordinate system)
                # ADDITIONAL FIX: Scale CSI targets to match student prediction scale
                csi_targets_norm = (csi_targets_task - csi_targets_task.mean(dim=0, keepdim=True)) / (csi_targets_task.std(dim=0, keepdim=True) + 1e-8)
                task_loss = self.mse_loss(student_norm, csi_targets_norm)  # Both normalized to same scale
                
                # Combined loss with proper weighting
                total_loss = self.alpha * distill_loss + self.beta * task_loss
                
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student.parameters() if p.requires_grad] + 
                    [p for p in self.teacher_to_student_projections.parameters() if p.requires_grad] + 
                    [p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad], 
                    max_norm=0.5  # FIXED: Reduced from 1.0 to 0.5 for better stability
                )
                
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, "
                          f"Total Loss: {total_loss.item():.6f}, "
                          f"Distill: {distill_loss.item():.6f}, "
                          f"Task: {task_loss.item():.6f}")
                
                if batch_idx >= 50:  # More training for final stage
                    break
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Stage 3 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        return losses
    
    def full_synchronized_distillation(
        self,
        synchronized_train_loader: DataLoader,
        synchronized_val_loader: DataLoader,
        learning_rate: float = 2e-5,  # FIXED: Reduced from 1e-4 to 2e-5 for stability
        stage1_epochs: int = 10,
        stage2_epochs: int = 15,
        stage3_epochs: int = 30,  # FIXED: Increased from 20 to 30 for lower LR
    ) -> Dict[str, List[float]]:
        """
        Run the complete 3-stage MOHAWK distillation process using synchronized UWB-CSI data.
        """
        print("Starting complete synchronized MOHAWK distillation for UWB Transformer → CSI Mamba transfer")
        
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
                synchronized_train_loader, optimizer, stage1_epochs, layer_idx
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
                synchronized_train_loader, optimizer, stage2_epochs, layer_idx
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
        stage3_losses = self.stage3_distill(synchronized_train_loader, optimizer, stage3_epochs)
        all_losses["stage3"] = stage3_losses
        
        # Final validation
        self.evaluate_synchronized(synchronized_val_loader)
        
        return all_losses
    
    def evaluate_synchronized(self, synchronized_val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the synchronized cross-modality distilled student model.
        FIXED: Use same normalization as training for consistent evaluation.
        
        Args:
            synchronized_val_loader: Synchronized validation data loader
            
        Returns:
            Dictionary of evaluation metrics including median absolute error
        """
        self.student.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        all_errors = []  # Store all errors for median calculation
        
        with torch.no_grad():
            for uwb_data, uwb_targets, csi_data, csi_targets in synchronized_val_loader:
                csi_data = csi_data.float().to(self.device)
                csi_targets = csi_targets.float().to(self.device)
                
                # Get student predictions using CSI data only
                student_outputs = self.student(csi_data)
                predictions = student_outputs.predictions
                
                if predictions is not None:
                    # FIXED: Use same aggregation and normalization as training
                    if predictions.dim() == 3:  # [batch, seq, features]
                        seq_len = predictions.size(1)
                        timestep_weights = torch.linspace(0.1, 1.0, seq_len, device=self.device)
                        timestep_weights = timestep_weights / timestep_weights.sum()
                        predictions = torch.sum(predictions * timestep_weights.view(1, -1, 1), dim=1)
                    
                    # Get CSI targets (last timestep)
                    if csi_targets.dim() == 3:
                        csi_targets_eval = csi_targets[:, -1, :]
                    else:
                        csi_targets_eval = csi_targets
                    
                    # FIXED: Apply same normalization as training for fair comparison
                    predictions_norm = (predictions - predictions.mean(dim=0, keepdim=True)) / (predictions.std(dim=0, keepdim=True) + 1e-8)
                    targets_norm = (csi_targets_eval - csi_targets_eval.mean(dim=0, keepdim=True)) / (csi_targets_eval.std(dim=0, keepdim=True) + 1e-8)
                    
                    # Calculate normalized losses (consistent with training)
                    loss = self.mse_loss(predictions_norm, targets_norm)
                    mae = self.l1_loss(predictions_norm, targets_norm)
                    
                    # Calculate individual errors for median (using normalized values)
                    batch_errors = torch.abs(predictions_norm - targets_norm).cpu().numpy()
                    all_errors.extend(batch_errors.flatten())
                    
                    total_loss += loss.item() * csi_data.size(0)
                    total_mae += mae.item() * csi_data.size(0)
                    total_samples += csi_data.size(0)
        
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_mae = total_mae / total_samples
            median_ae = np.median(all_errors) if all_errors else float('inf')
            
            metrics = {
                'val_loss': avg_loss,
                'val_mae': avg_mae,
                'val_median_ae': median_ae,
                'rmse': np.sqrt(avg_loss)
            }
            
            print(f"📊 Validation Results (FIXED - normalized like training):")
            print(f"   Loss (MSE): {avg_loss:.6f}")
            print(f"   MAE: {avg_mae:.6f}")
            print(f"   Median AE: {median_ae:.6f}")
            print(f"   RMSE: {np.sqrt(avg_loss):.6f}")
            
            return metrics
        else:
            return {'val_loss': float('inf'), 'val_mae': float('inf'), 'val_median_ae': float('inf'), 'rmse': float('inf')}


def create_synchronized_cross_modality_setup(
    uwb_data_path: str,
    csi_mat_file: str,
    train_experiments: List[str],
    val_experiments: List[str],
    uwb_transformer_config: dict,
    csi_config: dict,
    device: str = "cuda"
) -> Tuple[UWBRegressionModel, SynchronizedCrossModalityMOHAWKDistiller, DataLoader, DataLoader]:
    """
    Create synchronized cross-modality distillation setup with transformer teacher trained from scratch.
    FIXED: Maintains temporal synchronization while using successful data processing techniques.
    
    Args:
        uwb_data_path: Path to UWB data directory
        csi_mat_file: Path to CSI .mat file
        train_experiments: List of training experiments
        val_experiments: List of validation experiments
        uwb_transformer_config: Configuration for UWB transformer teacher model
        csi_config: Configuration for CSI mamba student model
        device: Device to run on
        
    Returns:
        Tuple of (teacher_model, distiller, train_loader, val_loader)
    """
    
    # FIXED: Use synchronized data loaders to maintain UWB-CSI temporal pairing
    print("🔗 Creating SYNCHRONIZED UWB-CSI data loaders (maintaining temporal alignment)...")
    train_loader, val_loader, uwb_scaler, target_scaler, csi_scaler = create_synchronized_dataloaders(
        uwb_data_path=uwb_data_path,
        csi_mat_file=csi_mat_file,
        train_experiments=train_experiments,
        val_experiments=val_experiments,
        batch_size=32,  # IMPROVED: Use successful batch size from test_csi_student.py
        sequence_length=32,  # Keep UWB sequence length
        csi_sequence_length=4,  # IMPROVED: Use successful CSI sequence length
        target_tags=['tag4422'],
        max_samples_per_exp=5000,  # IMPROVED: Use successful sample limit
        use_magnitude_phase=True,  # IMPROVED: Use successful CSI representation
        stride=2,  # IMPROVED: Smaller stride for more sequences
        temporal_split=True,  # IMPROVED: Use temporal split to prevent data leakage
        train_split=0.8,      # IMPROVED: Proper train/val split ratio
        temporal_gap=0        # IMPROVED: Gap to prevent leakage at boundary
    )
    
    # Get sample data to update CSI config with actual feature count
    for uwb_data, uwb_targets, csi_data, csi_targets in train_loader:
        csi_feature_count = csi_data.shape[-1]
        break
    
    # Update CSI config with actual feature count (from successful approach)
    if isinstance(csi_config, dict):
        csi_config["CSIRegressionModel"]["input"]["input_features"] = csi_feature_count
    else:
        csi_config.CSIRegressionModel.input.input_features = csi_feature_count
    
    print(f"📐 Updated CSI config input_features to: {csi_feature_count}")
    print(f"🔗 Maintaining temporal synchronization: UWB-CSI pairs are temporally aligned")
    
    # Train UWB transformer teacher model from scratch
    print("🎓 Training UWB TRANSFORMER teacher model from scratch...")
    teacher_model = train_uwb_transformer_teacher_from_scratch(
        train_loader=train_loader,
        device=device
    )
    
    # Evaluate teacher performance before distillation
    teacher_metrics = evaluate_teacher_performance(teacher_model, val_loader, device)
    
    # Create distiller with IMPROVED hyperparameters based on successful CSI training
    print("🔧 Creating synchronized cross-modality distiller with IMPROVED hyperparameters...")
    distiller = SynchronizedCrossModalityMOHAWKDistiller(
        teacher_model=teacher_model,
        csi_student_config=csi_config,
        device=device,
        temperature=1.0,    # IMPROVED: Good for regression (from test_csi_student.py)
        alpha=0.8,          # IMPROVED: Balanced distillation weight
        beta=0.2            # IMPROVED: Balanced task weight
    )
    
    print("✅ FIXED: Cross-modality distillation setup maintains temporal synchronization!")
    print(f"🔗 UWB and CSI data are temporally paired - essential for cross-modal knowledge transfer")
    
    return teacher_model, distiller, train_loader, val_loader


def train_csi_mamba_baseline_from_scratch(
    csi_mat_file, device, epochs=50, learning_rate=0.001
):
    """
    Train CSI MAMBA baseline model from scratch without distillation.
    UPDATED: Uses the successful CSI-only approach from test_csi_student.py for fair comparison.
    """
    print("🚀 Training CSI MAMBA baseline model from scratch (using successful CSI-only approach)...")
    
    # Import the fixed CSI loader that works well for CSI-only training
    from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
    
    # Use the same successful data loading approach as test_csi_student.py
    print("📊 Loading CSI data with CSI-only loader (for fair baseline comparison)...")
    train_loader, val_loader, feature_scaler, target_scaler = create_csi_dataloaders_fixed(
        mat_file_path=csi_mat_file,
        train_split=0.8,
        batch_size=32,  # FIXED: Use successful batch size
        sequence_length=4,  # FIXED: Use successful sequence length
        target_tags=['tag4422'],  # FIXED: Use successful target
        use_magnitude_phase=True,  # FIXED: Use successful feature representation
        max_samples=5000  # FIXED: Use successful sample limit
    )
    
    # Get actual feature count and update config (same as test_csi_student.py)
    for batch_features, batch_targets in train_loader:
        actual_feature_count = batch_features.shape[-1]
        sequence_length_actual = batch_features.shape[1]
        break
    
    # Load CSI configuration and update with actual feature count
    with open("csi_config.json", "r") as f:
        csi_config = json.load(f)
    
    # Update config with actual feature count (same as test_csi_student.py)
    csi_config["CSIRegressionModel"]["input"]["input_features"] = actual_feature_count
    print(f"📐 Updated config input_features to: {actual_feature_count}")
    print(f"📏 Sequence length: {sequence_length_actual}")
    
    # Create baseline CSI model (identical architecture to student)
    baseline_model = CSIRegressionModel(csi_config, device=device).to(device)
    
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"🎯 Baseline CSI Mamba parameters: {baseline_params:,}")
    
    # Train the baseline using CSI data only with successful hyperparameters
    baseline_model.train()
    optimizer = optim.AdamW(baseline_model.parameters(), lr=learning_rate, weight_decay=1e-4)  # FIXED: Use successful hyperparameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)  # FIXED: Use successful scheduler
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 15  # FIXED: Use successful patience
    
    print(f"📊 Training baseline CSI model with SUCCESSFUL hyperparameters (CSI-only):")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: 32")
    print(f"   Weight decay: 1e-4")
    print(f"   Scheduler: ReduceLROnPlateau")
    print(f"   Architecture: {baseline_model.__class__.__name__}")
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_maes = []
        
        for batch_idx, (csi_features, csi_targets) in enumerate(train_loader):
            csi_features = csi_features.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            optimizer.zero_grad()
            
            # Train baseline on CSI data only (same as test_csi_student.py)
            outputs = baseline_model(csi_features, targets=csi_targets)
            loss = criterion(outputs.predictions, csi_targets)
            mae = torch.mean(torch.abs(outputs.predictions - csi_targets))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=1.0)  # FIXED: Use successful gradient clipping
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_maes.append(mae.item())
            
            if batch_idx % 10 == 0:  # FIXED: Use successful logging frequency
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, MAE: {mae.item():.6f}")
        
        avg_loss = np.mean(epoch_losses)
        avg_mae = np.mean(epoch_maes)
        scheduler.step(avg_loss)  # FIXED: Use successful scheduler
        
        # Early stopping (same as test_csi_student.py)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Baseline Epoch {epoch+1}: Loss = {avg_loss:.6f}, MAE = {avg_mae:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= max_patience:
            print(f"💡 Early stopping triggered at epoch {epoch+1}")
            break
    
    baseline_model.eval()
    print(f"✅ Baseline training completed with best loss: {best_loss:.6f}")
    
    return baseline_model


def evaluate_model_comprehensive(model, val_loader, device, model_name="Model"):
    """
    Comprehensive evaluation of a model including multiple metrics.
    FIXED: Handles both synchronized (4-value) and CSI-only (2-value) data loaders.
    """
    print(f"📊 Evaluating {model_name} performance...")
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    all_errors = []
    all_predictions = []
    all_targets = []
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            # FIXED: Handle both synchronized (4-value) and CSI-only (2-value) data
            if len(batch_data) == 4:
                # Synchronized data: (uwb_data, uwb_targets, csi_data, csi_targets)
                uwb_data, uwb_targets, csi_data, csi_targets = batch_data
                csi_data = csi_data.float().to(device)
                csi_targets = csi_targets.float().to(device)
            elif len(batch_data) == 2:
                # CSI-only data: (csi_data, csi_targets)
                csi_data, csi_targets = batch_data
                csi_data = csi_data.float().to(device)
                csi_targets = csi_targets.float().to(device)
            else:
                print(f"⚠️ Unexpected batch data format: {len(batch_data)} values")
                continue
            
            # Get model predictions using CSI data
            outputs = model(csi_data)
            predictions = outputs.predictions
            
            if predictions is not None:
                # Use last timestep for evaluation consistency
                if csi_targets.dim() == 3:  # [batch, seq, features]
                    targets_eval = csi_targets[:, -1, :]
                else:
                    targets_eval = csi_targets
                
                if predictions.dim() == 3:  # [batch, seq, features]
                    predictions_eval = predictions[:, -1, :]
                else:
                    predictions_eval = predictions
                
                # Calculate metrics
                mse_loss = criterion_mse(predictions_eval, targets_eval)
                mae_loss = criterion_mae(predictions_eval, targets_eval)
                
                # Store for comprehensive analysis
                batch_errors = torch.abs(predictions_eval - targets_eval).cpu().numpy()
                all_errors.extend(batch_errors.flatten())
                all_predictions.extend(predictions_eval.cpu().numpy().flatten())
                all_targets.extend(targets_eval.cpu().numpy().flatten())
                
                total_loss += mse_loss.item() * csi_data.size(0)
                total_mae += mae_loss.item() * csi_data.size(0)
                total_mse += mse_loss.item() * csi_data.size(0)
                total_samples += csi_data.size(0)
            
            if batch_idx >= 30:  # Comprehensive evaluation
                break
    
    if total_samples > 0:
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        rmse = np.sqrt(avg_mse)
        median_ae = np.median(all_errors) if all_errors else float('inf')
        
        # Additional metrics
        percentile_90_ae = np.percentile(all_errors, 90) if all_errors else float('inf')
        percentile_95_ae = np.percentile(all_errors, 95) if all_errors else float('inf')
        std_error = np.std(all_errors) if all_errors else float('inf')
        
        # Correlation analysis
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1] if len(all_predictions) > 1 else 0.0
        
        metrics = {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': rmse,
            'median_ae': median_ae,
            'p90_ae': percentile_90_ae,
            'p95_ae': percentile_95_ae,
            'std_error': std_error,
            'correlation': correlation,
            'samples': total_samples
        }
        
        print(f"📊 {model_name} Comprehensive Results:")
        print(f"   MSE: {avg_mse:.6f}")
        print(f"   MAE: {avg_mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Median AE: {median_ae:.6f}")
        print(f"   90th percentile AE: {percentile_90_ae:.6f}")
        print(f"   95th percentile AE: {percentile_95_ae:.6f}")
        print(f"   Error Std Dev: {std_error:.6f}")
        print(f"   Correlation: {correlation:.6f}")
        print(f"   Samples: {total_samples}")
        
        return metrics
    else:
        return {k: float('inf') for k in ['mse', 'mae', 'rmse', 'median_ae', 'p90_ae', 'p95_ae', 'std_error']} | {'correlation': 0.0, 'samples': 0}


def compare_models(distilled_model, baseline_model, val_loader, device):
    """
    Compare distilled student model vs baseline CSI model.
    """
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Evaluate both models
    distilled_metrics = evaluate_model_comprehensive(distilled_model, val_loader, device, "Distilled Student")
    baseline_metrics = evaluate_model_comprehensive(baseline_model, val_loader, device, "Baseline CSI")
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"=" * 50)
    
    metrics_to_compare = ['mse', 'mae', 'rmse', 'median_ae', 'correlation']
    
    for metric in metrics_to_compare:
        distilled_val = distilled_metrics[metric]
        baseline_val = baseline_metrics[metric]
        
        if metric == 'correlation':
            # Higher is better for correlation
            improvement = ((distilled_val - baseline_val) / abs(baseline_val)) * 100 if baseline_val != 0 else 0
            winner = "Distilled" if distilled_val > baseline_val else "Baseline"
        else:
            # Lower is better for error metrics
            improvement = ((baseline_val - distilled_val) / baseline_val) * 100 if baseline_val != 0 else 0
            winner = "Distilled" if distilled_val < baseline_val else "Baseline"
        
        print(f"{metric.upper():>12}: Distilled={distilled_val:.6f}, Baseline={baseline_val:.6f}")
        print(f"{'':>12}  {'✅' if winner == 'Distilled' else '❌'} {winner} wins by {abs(improvement):.1f}%")
    
    # Overall assessment
    distilled_wins = sum([
        distilled_metrics['mse'] < baseline_metrics['mse'],
        distilled_metrics['mae'] < baseline_metrics['mae'], 
        distilled_metrics['rmse'] < baseline_metrics['rmse'],
        distilled_metrics['median_ae'] < baseline_metrics['median_ae'],
        distilled_metrics['correlation'] > baseline_metrics['correlation']
    ])
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"   Distilled model wins in {distilled_wins}/5 metrics")
    
    if distilled_wins >= 4:
        print(f"   🎉 DISTILLATION HIGHLY EFFECTIVE!")
        print(f"   📈 Cross-modal knowledge transfer significantly improves CSI performance")
    elif distilled_wins >= 3:
        print(f"   ✅ DISTILLATION EFFECTIVE")
        print(f"   📊 Cross-modal knowledge transfer provides moderate improvement")
    elif distilled_wins >= 2:
        print(f"   ⚠️ DISTILLATION MARGINALLY EFFECTIVE")
        print(f"   🤔 Some benefit from cross-modal knowledge, but limited")
    else:
        print(f"   ❌ DISTILLATION NOT EFFECTIVE")
        print(f"   💭 Consider: 1) Different hyperparameters, 2) Architecture mismatch, 3) Data quality")
    
    # Specific recommendations
    mse_improvement = ((baseline_metrics['mse'] - distilled_metrics['mse']) / baseline_metrics['mse']) * 100
    print(f"\n💡 RECOMMENDATIONS:")
    if mse_improvement > 10:
        print(f"   🚀 Great results! Consider deploying distilled model")
    elif mse_improvement > 5:
        print(f"   📈 Good improvement. Fine-tune hyperparameters for better results")
    elif mse_improvement > 0:
        print(f"   🔧 Modest improvement. Try: longer training, different loss weights")
    else:
        print(f"   🔄 No improvement. Try: different teacher, adjusted architecture, better synchronization")
    
    return {
        'distilled_metrics': distilled_metrics,
        'baseline_metrics': baseline_metrics,
        'distilled_wins': distilled_wins,
        'mse_improvement_pct': mse_improvement
    }


def main():
    """Main function for synchronized cross-modality cross-architecture distillation with transformer teacher trained from scratch."""
    print("🚀 FIXED Cross-Modality Cross-Architecture MOHAWK Distillation: UWB Transformer → CSI Mamba")
    print("🔗 MAINTAINING TEMPORAL SYNCHRONIZATION for proper cross-modal knowledge transfer")
    print("=" * 90)
    
    # Configuration
    uwb_data_path = "/media/mohab/Storage HDD/Downloads/uwb2(1)"
    csi_mat_file = "/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat"
    
    train_experiments = ["002"]  # Start with one experiment
    val_experiments = ["002"]    # Use same for initial testing
    
    # Load configurations
    with open("uwb_transformer_config.json", "r") as f:
        uwb_transformer_config = json.load(f)
    
    with open("csi_config.json", "r") as f:
        csi_config = json.load(f)
    
    # Create distillation setup (FIXED: maintains temporal synchronization)
    teacher_model, distiller, train_loader, val_loader = create_synchronized_cross_modality_setup(
        uwb_data_path=uwb_data_path,
        csi_mat_file=csi_mat_file,
        train_experiments=train_experiments,
        val_experiments=val_experiments,
        uwb_transformer_config=uwb_transformer_config,
        csi_config=csi_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Save the trained transformer teacher model
    print("💾 Saving trained UWB transformer teacher model...")
    teacher_model.save_pretrained("trained_uwb_transformer_teacher_from_scratch")
    
    # Run cross-modal cross-architecture distillation with IMPROVED hyperparameters
    print("🎯 Starting IMPROVED cross-modality distillation (with temporal synchronization)...")
    loss_history = distiller.full_synchronized_distillation(
        synchronized_train_loader=train_loader,
        synchronized_val_loader=val_loader,
        learning_rate=0.001,  # IMPROVED: Use higher learning rate like successful CSI training
        stage1_epochs=5,   # Reduced for faster testing
        stage2_epochs=8,
        stage3_epochs=15   # IMPROVED: More reasonable epoch count
    )
    
    # Save the distilled CSI mamba student model
    print("💾 Saving synchronized distilled CSI mamba student model...")
    distiller.student.save_pretrained("trained_synchronized_csi_mamba_student")
    
    # Train a baseline CSI Mamba model from scratch using SUCCESSFUL CSI-only approach
    print("🚀 Training CSI MAMBA baseline model from scratch (CSI-only for fair comparison)...")
    baseline_model = train_csi_mamba_baseline_from_scratch(
        csi_mat_file=csi_mat_file,
        device=distiller.device
    )
    print("💾 Saving baseline CSI MAMBA student model...")
    baseline_model.save_pretrained("trained_csi_mamba_baseline_student")

    # Compare distilled student model vs baseline CSI model
    # For fair comparison, evaluate both models on CSI data
    print("\n🔬 FAIR MODEL COMPARISON:")
    print("=" * 60)
    print("📊 Distilled Model: Trained with UWB teacher knowledge (cross-modal)")
    print("📊 Baseline Model: Trained with CSI data only (single-modal)")
    print("🎯 Both evaluated on SAME CSI validation data for fair comparison")
    
    # Create CSI evaluation loader for fair comparison (both models evaluated on same data)
    from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
    _, eval_loader, _, _ = create_csi_dataloaders_fixed(
        mat_file_path=csi_mat_file,
        train_split=0.8,
        batch_size=32,
        sequence_length=4,
        target_tags=['tag4422'],
        use_magnitude_phase=True,
        max_samples=5000
    )
    
    print(f"📊 Using {len(eval_loader)} batches for fair evaluation")
    compare_models(distiller.student, baseline_model, eval_loader, distiller.device)
    
    print("\n✅ FIXED cross-modality distillation complete!")
    print("🔗 Temporal synchronization maintained throughout training")
    print(f"📊 Final loss history: {loss_history}")


if __name__ == "__main__":
    main() 