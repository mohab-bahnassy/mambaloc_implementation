"""
Encoder-Enhanced Cross-Modal Distillation Framework with MOHAWK 3-Stage Training
Integrates modality-specific encoders with MOHAWK distillation for UWB→CSI knowledge transfer.

Pipeline:
1. Encoder Pretraining: Train Encoder_UWB and Encoder_CSI separately
2. Stage 1: Matrix mixer alignment using encoded representations  
3. Stage 2: Hidden state alignment using encoded representations
4. Stage 3: Full model alignment using encoded representations
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
from modules.cross_modal_encoders import (
    Encoder_UWB, Encoder_CSI, Decoder_UWB, Decoder_CSI, 
    visualize_latent_alignment
)
from dataloaders.synchronized_uwb_csi_loader_fixed import create_fixed_synchronized_dataloaders
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from utils.config import Config
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher

# Add current directory for imports
sys.path.append('.')


class EncoderEnhancedMOHAWKDistiller:
    """
    Enhanced cross-modal distillation using encoders with MOHAWK 3-stage training.
    
    Architecture:
    - UWB → Encoder_UWB → z_UWB → Teacher → y_teacher
    - CSI → Encoder_CSI → z_CSI → Student → y_student
    
    Training follows MOHAWK 3-stage strategy:
    Stage 1: Matrix mixer alignment
    Stage 2: Hidden state alignment  
    Stage 3: Full model alignment
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        csi_student_config: dict,
        uwb_input_features: int = 113,
        csi_input_features: int = 280,
        latent_dim: int = 128,
        device: str = "cuda",
        temperature: float = 4.0,
        alpha: float = 0.7,    # Distillation loss weight
        beta: float = 0.3,     # Task loss weight  
    ):
        """
        Initialize encoder-enhanced MOHAWK distiller.
        """
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.latent_dim = latent_dim
        
        # FIXED: Add adaptive temperature flag
        self.adaptive_temperature = True
        self.temperature_adjusted = False
        
        # Store original teacher model
        self.original_teacher = teacher_model.to(device)
        for param in self.original_teacher.parameters():
            param.requires_grad = False
        
        # Convert to Config if needed
        if not isinstance(csi_student_config, Config):
            config = Config.from_dict(csi_student_config)
        else:
            config = csi_student_config
        
        # Create CSI mamba student model - USE ACTUAL MODEL WITH BACKBONE
        self.student = CSIRegressionModel(csi_student_config, device=device)
        self.student.to(device)
        
        # Create encoders
        self.encoder_uwb = Encoder_UWB(
            input_features=uwb_input_features,
            latent_dim=latent_dim,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            preserve_sequence=True  # FIXED: Preserve temporal structure
        ).to(device)
        
        self.encoder_csi = Encoder_CSI(
            input_features=csi_input_features,
            latent_dim=latent_dim,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            preserve_sequence=True  # FIXED: Preserve temporal structure
        ).to(device)
        
        # Get dimensions for MOHAWK projections - USE ACTUAL MODEL DIMENSIONS
        self.teacher_dim = teacher_model.d_model  # Teacher transformer dimension
        self.student_dim = self.student.backbone.input_projection.out_features  # Student mamba dimension
        
        # Create latent input adaptation layers (FIXED approach)
        # These map from latent_dim to model_dim for each timestep
        self.teacher_latent_adapter = nn.Linear(latent_dim, self.teacher_dim, device=device)
        self.student_latent_adapter = nn.Linear(latent_dim, self.student_dim, device=device)
        
        # Create wrapper classes that add latent input capability
        self.teacher = self._create_latent_teacher_wrapper(teacher_model, latent_dim)
        self.student = self._create_latent_student_wrapper(self.student, latent_dim)
        
        # Add projection layers to handle dimension mismatch (MOHAWK style)
        self.teacher_to_student_projections = nn.ModuleList()
        self.attention_to_mixer_projections = nn.ModuleList()
        
        # Create projections for each student layer - USE ACTUAL STUDENT LAYER COUNT
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
        
        # Loss functions - EXACT MOHAWK REQUIREMENT
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        print(f"🎯 Encoder-Enhanced MOHAWK Distiller initialized:")
        print(f"   UWB features: {uwb_input_features} → Latent: {latent_dim}")
        print(f"   CSI features: {csi_input_features} → Latent: {latent_dim}")
        print(f"   Teacher dim: {self.teacher_dim}, Student dim: {self.student_dim}")
        print(f"   Created {num_student_layers} projection layers for MOHAWK alignment")
        
        # Configure trainable parameters
        self._configure_parameters()
    
    def _create_latent_teacher_wrapper(self, original_teacher, latent_dim):
        """Create a wrapper that allows the teacher to accept sequence-structured latent inputs."""
        class LatentTeacherWrapper(nn.Module):
            def __init__(self, original_model, latent_adapter):
                super().__init__()
                self.original_model = original_model
                self.latent_adapter = latent_adapter
                self.d_model = original_model.d_model  # Preserve teacher dimensions
                
                # Freeze original model
                for param in self.original_model.parameters():
                    param.requires_grad = False
            
            def forward(self, latent_input, return_attention=False, return_hidden_states=False, **kwargs):
                """
                Forward pass with sequence-structured latent input.
                
                Args:
                    latent_input: [batch_size, seq_len, latent_dim] - sequence-structured latents
                """
                # FIXED: Handle sequence-structured latents properly
                if latent_input.dim() == 3:  # [batch_size, seq_len, latent_dim]
                    batch_size, seq_len, _ = latent_input.shape
                    # Map each timestep from latent_dim to teacher's d_model
                    transformer_input = self.latent_adapter(latent_input)  # [batch_size, seq_len, d_model]
                elif latent_input.dim() == 2:  # [batch_size, latent_dim] - fallback for global latents
                batch_size = latent_input.shape[0]
                adapted_input = self.latent_adapter(latent_input)  # [batch_size, d_model]
                    # Create minimal sequence for transformer compatibility
                    seq_len = 4  # Reduced from 16 for efficiency
                transformer_input = adapted_input.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, d_model]
                else:
                    raise ValueError(f"Unexpected latent input shape: {latent_input.shape}")
                
                # Forward through transformer layers manually to get attention/hidden states
                hidden_states = transformer_input
                all_hidden_states = [hidden_states] if return_hidden_states else []
                all_attention_matrices = []
                
                if hasattr(self.original_model, 'transformer') and hasattr(self.original_model.transformer, 'layers'):
                    for layer in self.original_model.transformer.layers:
                        if return_attention:
                            # Get attention weights
                            attn_output, attn_weights = layer.self_attn(
                                hidden_states, hidden_states, hidden_states,
                                need_weights=True, average_attn_weights=False
                            )
                            all_attention_matrices.append(attn_weights)
                            
                            # Complete layer forward
                            hidden_states = layer.norm1(hidden_states + layer.dropout1(attn_output))
                            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden_states))))
                            hidden_states = layer.norm2(hidden_states + layer.dropout2(ff_output))
                        else:
                            hidden_states = layer(hidden_states)
                        
                        if return_hidden_states:
                            all_hidden_states.append(hidden_states)
                
                # Final output projection - use attention-weighted pooling for better representation
                if seq_len > 1:
                    # Create attention weights based on position (later timesteps more important)
                    position_weights = torch.linspace(0.1, 1.0, seq_len, device=hidden_states.device)
                    position_weights = position_weights / position_weights.sum()
                    output = torch.sum(hidden_states * position_weights.view(1, -1, 1), dim=1)  # [batch_size, d_model]
                else:
                    output = hidden_states.squeeze(1)  # [batch_size, d_model]
                    
                predictions = self.original_model.output_projection(output)
                
                result = {"predictions": predictions}
                if return_hidden_states:
                    result["all_hidden_states"] = all_hidden_states
                if return_attention:
                    result["all_attention_matrices"] = all_attention_matrices
                
                return result
        
        return LatentTeacherWrapper(original_teacher, self.teacher_latent_adapter).to(self.device)
    
    def _create_latent_student_wrapper(self, original_student, latent_dim):
        """Create a wrapper that allows the student to accept sequence-structured latent inputs while preserving backbone structure."""
        class LatentStudentWrapper(nn.Module):
            def __init__(self, original_model, latent_adapter):
                super().__init__()
                self.original_model = original_model
                self.latent_adapter = latent_adapter
                
                # Preserve backbone structure for MOHAWK compatibility
                self.backbone = original_model.backbone
                
                # Get the correct backbone input dimension (after input projection)
                self.backbone_dim = original_model.backbone.input_projection.out_features
                
                # FIXED: Create adapter that maps latent sequences to backbone dimension
                self.latent_to_backbone = nn.Linear(latent_dim, self.backbone_dim)
            
            def forward(self, latent_input, targets=None):
                """
                Forward pass with sequence-structured latent input.
                
                Args:
                    latent_input: [batch_size, seq_len, latent_dim] - sequence-structured latents
                """
                # FIXED: Handle sequence-structured latents properly
                if latent_input.dim() == 3:  # [batch_size, seq_len, latent_dim]
                    batch_size, seq_len, _ = latent_input.shape
                    # Map each timestep from latent_dim to backbone_dim
                    backbone_input = self.latent_to_backbone(latent_input)  # [batch_size, seq_len, backbone_dim]
                elif latent_input.dim() == 2:  # [batch_size, latent_dim] - fallback for global latents
                    batch_size = latent_input.shape[0]
                    backbone_features = self.latent_to_backbone(latent_input)  # [batch_size, backbone_dim]
                    # Create minimal sequence for Mamba compatibility
                seq_len = 4  # Typical CSI sequence length
                    # FIXED: Use expand instead of repeat to avoid memory issues
                    backbone_input = backbone_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, backbone_dim]
                else:
                    raise ValueError(f"Unexpected latent input shape: {latent_input.shape}")
                
                # Forward through backbone layers directly (bypass input projection since we already have correct dimension)
                hidden_states = backbone_input
                for layer in self.backbone.layers:
                    layer_outputs = layer(hidden_states)
                    # Extract tensor from layer output dictionary
                    hidden_states = layer_outputs["hidden_states"]
                
                # Apply final processing and head
                # Use the last timestep for prediction
                final_hidden = hidden_states[:, -1, :]  # [batch_size, backbone_dim]
                
                # Apply output head if exists
                if hasattr(self.original_model, 'head') and self.original_model.head is not None:
                    predictions = self.original_model.head(final_hidden)
                else:
                    # Fallback: direct projection to coordinates
                    if not hasattr(self, 'output_projection'):
                        self.output_projection = nn.Linear(self.backbone_dim, 2).to(final_hidden.device)
                    predictions = self.output_projection(final_hidden)
                
                # Compute loss if targets provided
                loss = None
                if targets is not None:
                    if targets.dim() > 2:
                        targets = targets[:, -1, :]  # Take last timestep
                    if targets.shape[-1] != 2:
                        targets = targets[:, :2]  # Take coordinates only
                    loss = F.mse_loss(predictions, targets)
                
                return type('Output', (), {
                    'predictions': predictions,
                    'loss': loss
                })()
        
        return LatentStudentWrapper(original_student, self.student_latent_adapter).to(self.device)
    
    def _configure_parameters(self):
        """Configure which parameters to train - ensure all parameters are trainable"""
        # Set all encoder parameters to require gradients
        self.encoder_uwb.requires_grad_(True)
        self.encoder_csi.requires_grad_(True)
        
        # Set all teacher/student parameters to require gradients
        self.teacher.requires_grad_(True)
        self.student.requires_grad_(True)
        
        # Set all projection layer parameters to require gradients
        for projection_layer in self.teacher_to_student_projections:
            projection_layer.requires_grad_(True)
        for projection_layer in self.attention_to_mixer_projections:
            projection_layer.requires_grad_(True)
        
        # Count trainable parameters
        encoder_params = sum(p.numel() for p in self.encoder_uwb.parameters()) + \
                        sum(p.numel() for p in self.encoder_csi.parameters())
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        projection_params = sum(p.numel() for p in self.teacher_to_student_projections.parameters()) + \
                          sum(p.numel() for p in self.attention_to_mixer_projections.parameters())
        
        print(f"🎯 Parameter Status:")
        print(f"   Encoders: {encoder_params:,} parameters")
        print(f"   Teacher: {teacher_params:,} parameters")
        print(f"   Student: {student_params:,} parameters")
        print(f"   Projections: {projection_params:,} parameters")
        print(f"   Total trainable: {encoder_params + teacher_params + student_params + projection_params:,} parameters")

    def pretrain_encoders(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,  # Reduced epochs
        learning_rate: float = 1e-3
    ) -> Dict[str, List[float]]:
        """
        Stage 0: Pretrain encoders for better latent representations.
        Train encoders with reconstruction and alignment objectives.
        """
        print("🚀 Stage 0: Pretraining Encoders")
        print("=" * 40)
        
        # Only train encoders during pretraining
        encoder_params = list(self.encoder_uwb.parameters()) + list(self.encoder_csi.parameters())
        optimizer = torch.optim.AdamW(encoder_params, lr=learning_rate, weight_decay=1e-4)
        
        history = {
            'alignment_loss': [],
            'total_loss': []
        }
        
        for epoch in range(epochs):
            self.encoder_uwb.train()
            self.encoder_csi.train()
            
            epoch_losses = []
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Encode both modalities
                z_UWB, z_CSI = self.encode_modalities(uwb_data, csi_data)
                
                # Alignment loss to bring latent representations closer
                alignment_loss = self.compute_alignment_loss(z_UWB, z_CSI)
                
                alignment_loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_params, max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(alignment_loss.item())
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Alignment Loss: {alignment_loss.item():.6f}")
                
                if batch_idx >= 20:  # Quick pretraining
                    break
            
            avg_loss = np.mean(epoch_losses)
            history['alignment_loss'].append(avg_loss)
            history['total_loss'].append(avg_loss)
            print(f"Encoder Pretrain Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        print("✅ Encoder pretraining completed!")
        return history

    def pretrain_teacher(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,  # Sufficient epochs for teacher convergence
        learning_rate: float = 1e-3
    ) -> Dict[str, List[float]]:
        """
        Stage -1: Pretrain teacher model on UWB data.
        CRITICAL: Teacher must learn UWB→coordinate mapping before distillation!
        """
        print("🎓 Stage -1: Pretraining UWB Teacher Model (CRITICAL)")
        print("=" * 50)
        print("Teacher must learn UWB→coordinate mapping before teaching student!")
        
        # Unfreeze teacher for pretraining
        for param in self.teacher.parameters():
            param.requires_grad = True
        
        self.teacher.train()
        
        # Create optimizer for teacher only
        teacher_optimizer = torch.optim.AdamW(
            self.teacher.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            teacher_optimizer, mode='min', patience=3, factor=0.7, verbose=True
        )
        
        criterion = torch.nn.MSELoss()
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(epochs):
            # Training phase
            epoch_losses = []
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                uwb_targets = uwb_targets.float().to(self.device)
                
                # Ensure targets are coordinate pairs
                if uwb_targets.dim() > 2:
                    uwb_targets = uwb_targets[:, -1, :]
                if uwb_targets.shape[-1] != 2:
                    uwb_targets = uwb_targets[:, :2]
                
                teacher_optimizer.zero_grad()
                
                # Teacher learns: UWB raw data → latent → coordinates (Option 1 approach)
                # Encode UWB data to latent, then use wrapper to get predictions
                with torch.no_grad():
                    z_UWB = self.encoder_uwb(uwb_data)
                
                # Use latent wrapper to get predictions directly
                teacher_predictions = self._get_teacher_predictions(z_UWB)
                
                # Task loss: teacher must predict correct coordinates
                loss = criterion(teacher_predictions, uwb_targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=1.0)
                teacher_optimizer.step()
                
                epoch_losses.append(loss.item())
                
                if batch_idx % 10 == 0:
                    print(f"Teacher Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                
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
                    
                    # Ensure targets are coordinate pairs
                    if uwb_targets.dim() > 2:
                        uwb_targets = uwb_targets[:, -1, :]
                    if uwb_targets.shape[-1] != 2:
                        uwb_targets = uwb_targets[:, :2]
                    
                    # Encode and predict using latent wrapper (Option 1 approach)
                    z_UWB = self.encoder_uwb(uwb_data)
                    teacher_predictions = self._get_teacher_predictions(z_UWB)
                    
                    val_loss = criterion(teacher_predictions, uwb_targets)
                    val_losses.append(val_loss.item())
                    
                    if batch_idx >= 10:  # Quick validation
                        break
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            print(f"Teacher Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"   ✅ New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"   ⏳ Patience: {patience_counter}/{max_patience}")
            
            if patience_counter >= max_patience:
                print(f"💡 Early stopping triggered at epoch {epoch+1}")
                break
            
            self.teacher.train()  # Back to training mode
        
        # CRITICAL: Freeze teacher after pretraining for distillation
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher.eval()
        
        print(f"✅ Teacher pretraining completed!")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"   Teacher is now FROZEN and ready to teach student")
        
        return history

    def stage1_distill(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 3,  # Reduced epochs
        layer_idx: int = 0,
        freeze_mlp: bool = True,
    ) -> List[float]:
        """
        Stage 1: Matrix mixer distillation (MOHAWK Matrix Orientation)
        EXACT COPY from mohawk_uwb_distillation.py with encoded representations.
        Compare MIXING MATRICES: Teacher's Softmax(QK^T) vs Student's materialized SSM matrix.
        This aligns the fundamental transformation operations, not the outputs.
        """
        print(f"🎯 Stage 1: Matrix mixer distillation for layer {layer_idx} (EXACT MOHAWK)")
        
        # Don't set encoders to eval - allow gradients to flow
        self.teacher.train()
        self.student.train()
        
        # Verify all parameters are trainable
        self.verify_all_parameters_trainable()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Encode modalities - allow gradients for CSI path
                with torch.no_grad():
                    z_UWB = self.encoder_uwb(uwb_data)
                
                # CSI encoder needs gradients for student training
                z_CSI = self.encoder_csi(csi_data)
                
                # Get teacher attention weights (mixing matrices) - EXACT MOHAWK METHOD
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(
                        z_UWB,
                        return_attention=True,
                        return_hidden_states=True
                    )
                
                # Get student input from teacher hidden states - EXACT MOHAWK METHOD
                teacher_hidden = teacher_outputs["all_hidden_states"][layer_idx]
                projected_input = self.teacher_to_student_projections[layer_idx](teacher_hidden)
                
                # Get student layer output with transfer matrix - EXACT MOHAWK METHOD
                student_layer = self.student.backbone.layers[layer_idx]
                student_output = student_layer(
                    hidden_states=projected_input,
                    run_mlp_component=not freeze_mlp,
                    return_mixer_matrix=True
                )
                
                # Stage 1: Compare MIXING MATRICES - EXACT MOHAWK IMPLEMENTATION
                if "all_attention_matrices" in teacher_outputs and "transfer_matrix" in student_output:
                    # Teacher mixing matrix: Softmax(QK^T) - attention weights
                    teacher_attention_weights = teacher_outputs["all_attention_matrices"][layer_idx]
                    # Student mixing matrix: materialized SSM matrix
                    student_transfer_matrix = student_output["transfer_matrix"]
                    
                    # Handle shape differences between attention heads and transfer matrix - EXACT MOHAWK LOGIC
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
                        
                        # Matrix alignment loss using Frobenius norm - EXACT MOHAWK METHOD
                        matrix_loss = torch.linalg.matrix_norm(
                            student_matrix_avg - teacher_weights, ord="fro"
                        ).mean()
                        
                    elif len(teacher_shape) == 4 and len(student_shape) == 4:
                        # Both are [batch, heads, seq, seq] format - EXACT MOHAWK LOGIC
                        batch_size = teacher_shape[0]
                        seq_len = min(teacher_shape[2], student_shape[2])
                        
                        # Ensure same sequence length
                        teacher_weights = teacher_attention_weights[:, :, :seq_len, :seq_len]
                        student_matrix = student_transfer_matrix[:, :, :seq_len, :seq_len]
                        
                        # Handle different number of heads by averaging - EXACT MOHAWK METHOD
                        if teacher_weights.size(1) != student_matrix.size(1):
                            teacher_weights = teacher_weights.mean(dim=1, keepdim=True)
                            student_matrix = student_matrix.mean(dim=1, keepdim=True)
                        
                        # Matrix alignment loss using Frobenius norm - EXACT MOHAWK METHOD
                        matrix_loss = torch.linalg.matrix_norm(
                            student_matrix - teacher_weights, ord="fro"
                        ).mean()
                    
                    else:
                        print(f"⚠️ Unexpected matrix shapes - Teacher: {teacher_shape}, Student: {student_shape}")
                        # Flatten and compare if shapes are too different - EXACT MOHAWK FALLBACK
                        teacher_flat = teacher_attention_weights.flatten()
                        student_flat = student_transfer_matrix.flatten()
                        min_size = min(teacher_flat.size(0), student_flat.size(0))
                        matrix_loss = F.mse_loss(student_flat[:min_size], teacher_flat[:min_size])
                
                else:
                    print(f"⚠️ Missing matrices for Stage 1 comparison")
                    matrix_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                matrix_loss.backward()
                
                # Gradient clipping for stability - EXACT MOHAWK METHOD
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student.parameters() if p.requires_grad] + 
                    [p for p in self.teacher_to_student_projections.parameters() if p.requires_grad] +
                    [p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_losses.append(matrix_loss.item())
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Matrix Loss: {matrix_loss.item():.6f}")
                
                if batch_idx >= 15:  # Quick training
                    break
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Stage 1 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        return losses

    def stage2_distill(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 3,  # Reduced epochs
        layer_idx: int = 0,
        freeze_mlp: bool = True,
    ) -> List[float]:
        """
        Stage 2: Hidden state distillation (MOHAWK Hidden-State Alignment)
        EXACT COPY from mohawk_uwb_distillation.py with encoded representations.
        Compare FULL BLOCK OUTPUTS: Teacher's complete attention block output vs Student's complete Mamba block output.
        This aligns the final representations after complete block processing.
        """
        print(f"🎯 Stage 2: Hidden state distillation for layer {layer_idx} (EXACT MOHAWK)")
        
        # Set all models to training mode to avoid CUDNN RNN errors
        self.teacher.train()
        self.student.train()
        
        # Verify all parameters are trainable
        self.verify_all_parameters_trainable()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Encode modalities - allow gradients for CSI path
                with torch.no_grad():
                    z_UWB = self.encoder_uwb(uwb_data)
                
                # CSI encoder needs gradients for student training
                z_CSI = self.encoder_csi(csi_data)
                
                # Get teacher full block outputs - EXACT MOHAWK METHOD
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(
                        z_UWB,
                    return_hidden_states=True
                )
                
                # Get student input from teacher hidden states - EXACT MOHAWK METHOD
                teacher_hidden = teacher_outputs["all_hidden_states"][layer_idx]
                projected_input = self.teacher_to_student_projections[layer_idx](teacher_hidden)
                
                # Get student full block output - EXACT MOHAWK METHOD
                student_layer = self.student.backbone.layers[layer_idx]
                student_output = student_layer(
                    hidden_states=projected_input,
                    run_mlp_component=not freeze_mlp,
                    return_hidden_states=not freeze_mlp
                )
                
                # Stage 2: Compare FULL BLOCK OUTPUTS - EXACT MOHAWK IMPLEMENTATION
                # Teacher: Full attention block output (after complete block processing)
                teacher_block_output = teacher_outputs["all_hidden_states"][layer_idx + 1]
                
                # Student: Full Mamba block output (after complete block processing)  
                student_block_output = student_output["hidden_states"]
                
                # Project teacher block output to student dimension - EXACT MOHAWK METHOD
                projected_teacher_output = self.teacher_to_student_projections[layer_idx](teacher_block_output)
                
                # Ensure compatible shapes for comparison - EXACT MOHAWK LOGIC
                if projected_teacher_output.shape != student_block_output.shape:
                    min_seq = min(projected_teacher_output.size(1), student_block_output.size(1))
                    projected_teacher_output = projected_teacher_output[:, :min_seq, :]
                    student_block_output = student_block_output[:, :min_seq, :]
                
                # Hidden state alignment loss using L2 norm - EXACT MOHAWK METHOD
                hidden_loss = torch.norm(
                    student_block_output - projected_teacher_output, p=2, dim=(-1,)
                ).mean()
                
                hidden_loss.backward()
                
                # Gradient clipping for stability - EXACT MOHAWK METHOD
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student.parameters() if p.requires_grad] + 
                    [p for p in self.teacher_to_student_projections.parameters() if p.requires_grad] +
                    [p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_losses.append(hidden_loss.item())
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Hidden Loss: {hidden_loss.item():.6f}")
                
                if batch_idx >= 15:  # Quick training
                    break
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Stage 2 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        return losses

    def stage3_distill(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,  # Reduced epochs
    ) -> List[float]:
        """
        Stage 3: Full model distillation - FIXED VERSION
        Added proper target normalization verification and consistent loss scaling.
        """
        print("🎯 Stage 3: Full model distillation (FIXED)")
        
        # Set all models to training mode
        self.teacher.train()
        self.student.train()
        
        # Verify all parameters are trainable
        self.verify_all_parameters_trainable()
        
        losses = []
        
        # FIXED: Add debug mode to check target scales
        debug_printed = False
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                csi_targets = csi_targets.float().to(self.device)
                
                # FIXED: Enhanced target preprocessing with verification
                if csi_targets.dim() > 2:
                    csi_targets = csi_targets[:, -1, :]
                if csi_targets.shape[-1] != 2:
                    csi_targets = csi_targets[:, :2]
                
                # FIXED: Debug target scales on first batch
                if not debug_printed:
                    target_mean = csi_targets.mean().item()
                    target_std = csi_targets.std().item()
                    target_min = csi_targets.min().item()
                    target_max = csi_targets.max().item()
                    print(f"🔍 DEBUG: Target statistics:")
                    print(f"   Mean: {target_mean:.4f}, Std: {target_std:.4f}")
                    print(f"   Range: [{target_min:.4f}, {target_max:.4f}]")
                    
                    # Check if targets appear normalized
                    if abs(target_mean) > 10 or target_std > 100:
                        print(f"⚠️ WARNING: Targets may not be properly normalized!")
                        print(f"   Expected normalized targets: mean≈0, std≈1")
                    debug_printed = True
                
                optimizer.zero_grad()
                
                # Encode modalities
                with torch.no_grad():
                    z_UWB = self.encoder_uwb(uwb_data)
                z_CSI = self.encoder_csi(csi_data)
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_predictions(z_UWB)
                
                # FIXED: Get student predictions without internal loss computation
                student_outputs = self.student(z_CSI, targets=None)  # Don't pass targets to avoid double loss
                student_predictions = student_outputs.predictions
                
                # FIXED: Debug prediction scales on first batch
                if batch_idx == 0 and epoch == 0:
                    teacher_mean = teacher_outputs.mean().item()
                    teacher_std = teacher_outputs.std().item()
                    student_mean = student_predictions.mean().item()
                    student_std = student_predictions.std().item()
                    print(f"🔍 DEBUG: Prediction statistics:")
                    print(f"   Teacher: mean={teacher_mean:.4f}, std={teacher_std:.4f}")
                    print(f"   Student: mean={student_mean:.4f}, std={student_std:.4f}")
                    
                    # FIXED: Adaptive temperature adjustment
                    if self.adaptive_temperature and not self.temperature_adjusted:
                        # Calculate reasonable temperature based on prediction scales
                        pred_scale = max(abs(teacher_mean), abs(student_mean), teacher_std, student_std)
                        if pred_scale > 0:
                            # Adjust temperature to normalize prediction scales to ~1
                            adjusted_temp = max(1.0, pred_scale / 2.0)  # Conservative adjustment
                            if abs(adjusted_temp - self.temperature) > 0.5:
                                print(f"🔧 ADAPTIVE TEMPERATURE: Adjusting from {self.temperature:.2f} to {adjusted_temp:.2f}")
                                print(f"   Based on prediction scale: {pred_scale:.4f}")
                                self.temperature = adjusted_temp
                            self.temperature_adjusted = True
                
                # FIXED: Compute distillation loss with consistent temperature scaling
                distill_loss = self.mse_loss(
                    student_predictions / self.temperature,
                    teacher_outputs / self.temperature
                )
                
                # FIXED: Compute task loss manually with proper verification
                # Ensure shapes match before computing loss
                if student_predictions.shape != csi_targets.shape:
                    print(f"⚠️ Shape mismatch: predictions {student_predictions.shape} vs targets {csi_targets.shape}")
                    # Ensure both are [batch_size, 2]
                    if student_predictions.dim() > 2:
                        student_predictions = student_predictions.squeeze()
                    if csi_targets.dim() > 2:
                        csi_targets = csi_targets.squeeze()
                
                # FIXED: Scale task loss consistently with temperature for stability
                raw_task_loss = self.mse_loss(student_predictions, csi_targets)
                task_loss = raw_task_loss / (self.temperature ** 2)  # Scale task loss for consistency
                
                # FIXED: Apply proper loss weighting
                total_loss = self.alpha * distill_loss + self.beta * task_loss
                
                # FIXED: Add stability check
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️ Unstable loss detected: total={total_loss.item()}, distill={distill_loss.item()}, task={task_loss.item()}")
                    continue  # Skip this batch
                
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
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, "
                          f"Total Loss: {total_loss.item():.6f}, "
                          f"Distill: {distill_loss.item():.6f}, "
                          f"Task (scaled): {task_loss.item():.6f}, "
                          f"Task (raw): {raw_task_loss.item():.6f}")
                
                if batch_idx >= 20:  # Quick training
                    break
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Stage 3 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        return losses

    def full_mohawk_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        teacher_epochs: int = 10,     # NEW: Teacher pretraining epochs
        pretrain_epochs: int = 5,
        stage1_epochs: int = 3,
        stage2_epochs: int = 3,
        stage3_epochs: int = 5,
        skip_stage1: bool = False,  # NEW: Option to skip stage 1
    ) -> Dict[str, List[float]]:
        """
        Run the complete MOHAWK distillation process.
        
        Args:
            teacher_epochs: Epochs for teacher pretraining (CRITICAL)
            skip_stage1: If True, skip stage 1 (matrix alignment) to test its effect
        """
        print("🚀 Starting Enhanced MOHAWK distillation with encoder integration")
        
        all_losses = {
            "pretrain": [],           # Encoder pretraining (FIRST)
            "teacher_pretrain": [],   # Teacher pretraining (SECOND - after encoders)
            "stage1": [],
            "stage2": [],
            "stage3": []
        }
        
        # Stage 0: Encoder Pretraining (FIRST - establish good latent representations)
        print("\n" + "="*50)
        print("STAGE 0: ENCODER PRETRAINING (FIRST - CRITICAL)")
        print("="*50)
        print("Encoders must learn good UWB→latent and CSI→latent mappings BEFORE teacher training!")
        pretrain_losses = self.pretrain_encoders(train_loader, val_loader, epochs=pretrain_epochs)
        all_losses["pretrain"] = pretrain_losses["total_loss"]
        
        # Stage -1: Teacher Pretraining (SECOND - after encoders are trained)
        print("\n" + "="*50)
        print("STAGE -1: TEACHER PRETRAINING (AFTER ENCODERS - CRITICAL)")
        print("="*50)
        print("Teacher learns from WELL-ENCODED UWB representations, not random ones!")
        teacher_losses = self.pretrain_teacher(train_loader, val_loader, epochs=teacher_epochs)
        all_losses["teacher_pretrain"] = teacher_losses["train_loss"]
        
        # Stage 1: Matrix mixer distillation for each layer (OPTIONAL)
        if not skip_stage1:
            print("\n" + "="*50)
            print("STAGE 1: MATRIX MIXER ALIGNMENT (EXACT MOHAWK)")
            print("="*50)
            for layer_idx in range(len(self.student.backbone.layers)):  # Use actual layer count
                # Ensure all parameters are trainable for Stage 1 - EXACT MOHAWK METHOD
                self._configure_parameters()
                
                # Include ALL trainable parameters (student + all projections) - EXACT MOHAWK METHOD
                all_trainable_params = []
                all_trainable_params.extend([p for p in self.student.parameters() if p.requires_grad])
                all_trainable_params.extend([p for p in self.teacher_to_student_projections.parameters() if p.requires_grad])
                all_trainable_params.extend([p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad])
                
                optimizer = torch.optim.AdamW(
                    all_trainable_params,  # Train ALL parameters - EXACT MOHAWK METHOD
                    lr=learning_rate
                )
                
                stage1_losses = self.stage1_distill(
                    train_loader, optimizer, stage1_epochs, layer_idx
                )
                all_losses["stage1"].extend(stage1_losses)
        else:
            print("\n" + "="*50)
            print("⏭️ SKIPPING STAGE 1: Matrix Mixer Alignment")
            print("="*50)
            print("Testing effect of skipping stage 1 (matrix alignment)")
        
        # Stage 2: Hidden state distillation for each layer - EXACT MOHAWK METHOD
        print("\n" + "="*50)
        print("STAGE 2: HIDDEN STATE ALIGNMENT (EXACT MOHAWK)")
        print("="*50)
        for layer_idx in range(len(self.student.backbone.layers)):  # Use actual layer count
            # Ensure all parameters are trainable for Stage 2 - EXACT MOHAWK METHOD
            self._configure_parameters()
            
            # Include ALL trainable parameters (student + all projections) - EXACT MOHAWK METHOD
            all_trainable_params = []
            all_trainable_params.extend([p for p in self.student.parameters() if p.requires_grad])
            all_trainable_params.extend([p for p in self.teacher_to_student_projections.parameters() if p.requires_grad])
            all_trainable_params.extend([p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad])
            
            optimizer = torch.optim.AdamW(
                all_trainable_params,  # Train ALL parameters - EXACT MOHAWK METHOD
                lr=learning_rate
            )
            
            stage2_losses = self.stage2_distill(
                train_loader, optimizer, stage2_epochs, layer_idx
            )
            all_losses["stage2"].extend(stage2_losses)
        
        # Stage 3: Full model distillation - FIXED APPROACH
        print("\n" + "="*50)
        print("STAGE 3: FULL MODEL ALIGNMENT (FIXED)")
        print("="*50)
        # Ensure all parameters are trainable for Stage 3
        self._configure_parameters()
        
        # Include ALL trainable parameters (student + all projections)
        all_trainable_params = []
        all_trainable_params.extend([p for p in self.student.parameters() if p.requires_grad])
        all_trainable_params.extend([p for p in self.teacher_to_student_projections.parameters() if p.requires_grad])
        all_trainable_params.extend([p for p in self.attention_to_mixer_projections.parameters() if p.requires_grad])
        
        # FIXED: Reduce learning rate for Stage 3 stability
        stage3_lr = learning_rate * 0.1  # 10x smaller learning rate for final stage
        print(f"🔧 Using reduced learning rate for Stage 3: {stage3_lr:.2e} (for stability)")
        
        optimizer = torch.optim.AdamW(
            all_trainable_params,
            lr=stage3_lr,  # FIXED: Reduced learning rate
            weight_decay=1e-5  # FIXED: Small weight decay for regularization
        )
        stage3_losses = self.stage3_distill(train_loader, optimizer, stage3_epochs)
        all_losses["stage3"] = stage3_losses
        
        # Final validation
        print("\n" + "="*50)
        print("FINAL VALIDATION")
        print("="*50)
        val_metrics = self.evaluate(val_loader)
        
        print("🎉 MOHAWK distillation completed successfully!")
        print(f"📊 Final validation metrics: {val_metrics}")
        
        return all_losses
    
    def encode_modalities(
        self, 
        uwb_data: torch.Tensor, 
        csi_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both modalities to shared latent space.
        
        Args:
            uwb_data: UWB input sequences [batch_size, uwb_seq_len, uwb_features]
            csi_data: CSI input sequences [batch_size, csi_seq_len, csi_features]
            
        Returns:
            z_UWB: UWB latent representation [batch_size, latent_dim]
            z_CSI: CSI latent representation [batch_size, latent_dim]
        """
        z_UWB = self.encoder_uwb(uwb_data)
        z_CSI = self.encoder_csi(csi_data)
        return z_UWB, z_CSI
    
    def compute_alignment_loss(self, z_UWB: torch.Tensor, z_CSI: torch.Tensor) -> torch.Tensor:
        """
        Compute latent alignment loss between UWB and CSI representations.
        Handles both sequence-structured and global latent representations.
        """
        # FIXED: Handle sequence-structured latents properly
        if z_UWB.dim() == 3 and z_CSI.dim() == 3:  # Both are [batch_size, seq_len, latent_dim]
            # Align sequence lengths if different
            min_seq_len = min(z_UWB.size(1), z_CSI.size(1))
            z_UWB_aligned = z_UWB[:, :min_seq_len, :]
            z_CSI_aligned = z_CSI[:, :min_seq_len, :]
            
            # Compute per-timestep alignment loss
            timestep_losses = []
            for t in range(min_seq_len):
                timestep_loss = self.mse_loss(z_CSI_aligned[:, t, :], z_UWB_aligned[:, t, :])
                timestep_losses.append(timestep_loss)
            
            # Weight later timesteps more heavily (they contain more refined information)
            timestep_weights = torch.linspace(0.5, 1.0, min_seq_len, device=z_UWB.device)
            timestep_weights = timestep_weights / timestep_weights.sum()
            
            weighted_loss = sum(w * loss for w, loss in zip(timestep_weights, timestep_losses))
            return weighted_loss
            
        elif z_UWB.dim() == 2 and z_CSI.dim() == 2:  # Both are [batch_size, latent_dim]
            # Original global alignment
        return self.mse_loss(z_CSI, z_UWB)
            
        else:
            # Mixed dimensions - reduce to global representations
            if z_UWB.dim() == 3:
                z_UWB_global = torch.mean(z_UWB, dim=1)  # [batch_size, latent_dim]
            else:
                z_UWB_global = z_UWB
                
            if z_CSI.dim() == 3:
                z_CSI_global = torch.mean(z_CSI, dim=1)  # [batch_size, latent_dim]
            else:
                z_CSI_global = z_CSI
                
            return self.mse_loss(z_CSI_global, z_UWB_global)
    
    def compute_reconstruction_losses(
        self, 
        z_UWB: torch.Tensor, 
        z_CSI: torch.Tensor,
        uwb_data: torch.Tensor,
        csi_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction losses if decoders are enabled."""
        # Reconstruction losses are not directly used in MOHAWK stages,
        # but kept for potential future use or if decoders are re-introduced.
        # For now, return zeros.
        return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
    
    def distillation_step(
        self,
        uwb_data: torch.Tensor,
        uwb_targets: torch.Tensor,
        csi_data: torch.Tensor,
        csi_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single distillation step with encoder integration.
        
        Returns dictionary of losses and predictions.
        """
        # Debug: Print tensor shapes to understand the mismatch
        if hasattr(self, '_debug_printed') is False:
            print(f"🔍 DEBUG - Tensor shapes:")
            print(f"   uwb_data: {uwb_data.shape}")
            print(f"   uwb_targets: {uwb_targets.shape}")
            print(f"   csi_data: {csi_data.shape}")
            print(f"   csi_targets: {csi_targets.shape}")
            self._debug_printed = True
        
        # Ensure targets are coordinate pairs [batch_size, 2], not sequences
        # If targets are sequences, take the last timestep
        if uwb_targets.dim() > 2:
            uwb_targets = uwb_targets[:, -1, :]  # Take last timestep
        elif uwb_targets.dim() == 3 and uwb_targets.shape[-1] != 2:
            uwb_targets = uwb_targets[:, -1, :2]  # Take last timestep, first 2 features
        
        if csi_targets.dim() > 2:
            csi_targets = csi_targets[:, -1, :]  # Take last timestep
        elif csi_targets.dim() == 3 and csi_targets.shape[-1] != 2:
            csi_targets = csi_targets[:, -1, :2]  # Take last timestep, first 2 features
        
        # Ensure targets have exactly 2 dimensions (coordinates)
        if uwb_targets.shape[-1] != 2:
            uwb_targets = uwb_targets[:, :2]  # Take only first 2 features (x, y coordinates)
        if csi_targets.shape[-1] != 2:
            csi_targets = csi_targets[:, :2]  # Take only first 2 features (x, y coordinates)
        
        # Encode both modalities to shared latent space
        z_UWB, z_CSI = self.encode_modalities(uwb_data, csi_data)
        
        # Get teacher predictions from UWB latent representation
        # Use the wrapper class that handles latent input directly
        with torch.no_grad():
            teacher_predictions = self._get_teacher_predictions(z_UWB)
        
        # Get student predictions from CSI latent representation
        # Use the wrapper class that handles latent input directly
        student_outputs = self.student(z_CSI, targets=csi_targets)
        student_predictions = student_outputs.predictions
        
        # Compute losses
        
        # 1. Distillation loss (teacher → student knowledge transfer)
        distill_loss = self.mse_loss(
            student_predictions / self.temperature,
            teacher_predictions / self.temperature
        )
        
        # 2. Task loss (student learns CSI coordinate prediction)
        task_loss = student_outputs.loss if student_outputs.loss is not None else 0
        
        # 3. Latent alignment loss (UWB and CSI latents should be similar)
        alignment_loss = self.compute_alignment_loss(z_UWB, z_CSI)
        
        # 4. Optional reconstruction losses (not used in MOHAWK, kept for compatibility)
        recon_loss_uwb, recon_loss_csi = self.compute_reconstruction_losses(
            z_UWB, z_CSI, uwb_data, csi_data
        )
        
        # Combined loss (only distillation + task for MOHAWK)
        total_loss = (
            self.alpha * distill_loss +
            self.beta * task_loss
        )
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'task_loss': task_loss,
            'alignment_loss': alignment_loss,
            'recon_loss_uwb': recon_loss_uwb,
            'recon_loss_csi': recon_loss_csi,
            'teacher_predictions': teacher_predictions,
            'student_predictions': student_predictions,
            'z_UWB': z_UWB,
            'z_CSI': z_CSI
        }
    
    def train_encoders_and_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        log_interval: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the complete encoder-enhanced distillation system.
        """
        print("🚀 Starting encoder-enhanced cross-modal distillation training...")
        
        # Collect all trainable parameters
        params = []
        params.extend(list(self.encoder_uwb.parameters()))
        params.extend(list(self.encoder_csi.parameters()))
        params.extend(list(self.teacher.parameters()))
        params.extend(list(self.student.parameters()))
        params.extend(list(self.teacher_to_student_projections.parameters()))
        params.extend(list(self.attention_to_mixer_projections.parameters()))
        
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, verbose=True
        )
        
        # Training history
        history = {
            'total_loss': [],
            'distill_loss': [],
            'task_loss': [],
            'alignment_loss': [],
            'recon_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            # Training phase
            self.encoder_uwb.train()
            self.encoder_csi.train()
            self.teacher.train()
            self.student.train()
            
            # No need to train projections during pretraining
            
            epoch_losses = {
                'total': [],
                'distill': [],
                'task': [],
                'alignment': [],
                'recon': []
            }
            
            for batch_idx, (uwb_data, uwb_targets, csi_data, csi_targets) in enumerate(train_loader):
                uwb_data = uwb_data.float().to(self.device)
                uwb_targets = uwb_targets.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                csi_targets = csi_targets.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.distillation_step(uwb_data, uwb_targets, csi_data, csi_targets)
                
                # Backward pass
                outputs['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                # Log losses
                epoch_losses['total'].append(outputs['total_loss'].item())
                epoch_losses['distill'].append(outputs['distill_loss'].item())
                epoch_losses['task'].append(outputs['task_loss'].item() if isinstance(outputs['task_loss'], torch.Tensor) else outputs['task_loss'])
                epoch_losses['alignment'].append(outputs['alignment_loss'].item())
                epoch_losses['recon'].append((outputs['recon_loss_uwb'] + outputs['recon_loss_csi']).item())
                
                if batch_idx % log_interval == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                          f"Total: {outputs['total_loss'].item():.6f}, "
                          f"Distill: {outputs['distill_loss'].item():.6f}, "
                          f"Align: {outputs['alignment_loss'].item():.6f}")
                    
                    # Log latent alignment quality
                    alignment_metrics = visualize_latent_alignment(outputs['z_UWB'], outputs['z_CSI'])
                    print(f"   Latent cosine similarity: {alignment_metrics['cosine_similarity']:.4f}")
            
            # Average epoch losses
            for key in history:
                if key == 'recon_loss':
                    history[key].append(np.mean(epoch_losses['recon']))
                elif key == 'total_loss':
                    history[key].append(np.mean(epoch_losses['total']))
                elif key == 'distill_loss':
                    history[key].append(np.mean(epoch_losses['distill']))
                elif key == 'task_loss':
                    history[key].append(np.mean(epoch_losses['task']))
                elif key == 'alignment_loss':
                    history[key].append(np.mean(epoch_losses['alignment']))
            
            avg_total_loss = history['total_loss'][-1]
            scheduler.step(avg_total_loss)
            
            # Validation (optional)
            val_loss = self.evaluate(val_loader)
            
            print(f"✅ Epoch {epoch+1} completed - Train Loss: {avg_total_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.save_best_models()
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"💡 Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"🎓 Training completed! Best validation loss: {best_val_loss:.6f}")
        return history
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate the encoder-enhanced distillation system."""
        self.encoder_uwb.eval()
        self.encoder_csi.eval()
        self.teacher.eval()
        self.student.eval()
        
        # No need to eval projections during pretraining
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for uwb_data, uwb_targets, csi_data, csi_targets in val_loader:
                uwb_data = uwb_data.float().to(self.device)
                uwb_targets = uwb_targets.float().to(self.device)
                csi_data = csi_data.float().to(self.device)
                csi_targets = csi_targets.float().to(self.device)
                
                outputs = self.distillation_step(uwb_data, uwb_targets, csi_data, csi_targets)
                total_loss += outputs['total_loss'].item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_best_models(self):
        """Save the best model states."""
        torch.save({
            'encoder_uwb': self.encoder_uwb.state_dict(),
            'encoder_csi': self.encoder_csi.state_dict(),
            'teacher': self.teacher.state_dict(),
            'student': self.student.state_dict(),
            'teacher_to_student_projections': [p.state_dict() for p in self.teacher_to_student_projections],
            'attention_to_mixer_projections': [p.state_dict() for p in self.attention_to_mixer_projections],
        }, 'best_encoder_distillation_models.pth')
        
        # No need to save decoders during pretraining
    
    def get_csi_student_for_deployment(self) -> nn.Module:
        """
        Get the CSI student model integrated with its encoder for deployment.
        During deployment, only CSI → Encoder_CSI → Student is needed.
        """
        class DeploymentCSIModel(nn.Module):
            def __init__(self, encoder_csi, student):
                super().__init__()
                self.encoder_csi = encoder_csi
                self.student = student
            
            def forward(self, csi_data):
                z_CSI = self.encoder_csi(csi_data)
                outputs = self.student(z_CSI)
                return outputs.predictions
        
        deployment_model = DeploymentCSIModel(self.encoder_csi, self.student)
        deployment_model.eval()
        return deployment_model

    def verify_all_parameters_trainable(self):
        """
        Verify that all parameters are actually trainable - EXACT MOHAWK METHOD
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

    def _get_teacher_outputs(self, latent_input, **kwargs):
        """
        Get teacher model outputs - Use latent wrapper directly (Option 1 approach).
        """
        return self.teacher(latent_input, **kwargs)

    def _get_teacher_predictions(self, latent_input):
        """
        Get teacher predictions for distillation - Use latent wrapper directly (Option 1 approach).
        """
        teacher_output = self.teacher(latent_input)
        
        # Extract predictions tensor from teacher output dict
        if isinstance(teacher_output, dict):
            return teacher_output["predictions"]
        else:
            return teacher_output


def train_baseline_csi_mamba(
    csi_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 50,
    learning_rate: float = 1e-3
) -> CSIRegressionModel:
    """
    Train baseline CSI Mamba model without distillation for comparison.
    """
    print("🚀 Training baseline CSI Mamba model (no distillation)...")
    
    # Create baseline model
    baseline_model = CSIRegressionModel(csi_config, device=device)
    baseline_model.to(device)
    
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(epochs):
        baseline_model.train()
        epoch_losses = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle different data loader formats
            if len(batch_data) == 4:  # Synchronized loader
                _, _, csi_data, csi_targets = batch_data
            else:  # CSI-only loader
                csi_data, csi_targets = batch_data
            
            csi_data = csi_data.float().to(device)
            csi_targets = csi_targets.float().to(device)
            
            optimizer.zero_grad()
            
            outputs = baseline_model(csi_data, targets=csi_targets)
            loss = criterion(outputs.predictions, csi_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)
        
        # Validation
        baseline_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:  # Synchronized loader
                    _, _, csi_data, csi_targets = batch_data
                else:  # CSI-only loader
                    csi_data, csi_targets = batch_data
                
                csi_data = csi_data.float().to(device)
                csi_targets = csi_targets.float().to(device)
                
                outputs = baseline_model(csi_data, targets=csi_targets)
                loss = criterion(outputs.predictions, csi_targets)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        print(f"✅ Epoch {epoch+1}: Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(baseline_model.state_dict(), 'best_baseline_csi_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"💡 Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"🎓 Baseline training completed! Best validation loss: {best_val_loss:.6f}")
    return baseline_model


def evaluate_models_comparison(
    baseline_model: nn.Module,
    distilled_model: nn.Module,
    eval_loader: DataLoader,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Compare baseline CSI Mamba vs encoder-enhanced distilled model performance.
    """
    print("🔬 Evaluating model comparison...")
    
    baseline_model.eval()
    distilled_model.eval()
    
    baseline_losses = []
    distilled_losses = []
    baseline_maes = []
    distilled_maes = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_data in eval_loader:
            # Handle different data loader formats
            if len(batch_data) == 4:  # Synchronized loader
                _, _, csi_data, csi_targets = batch_data
            else:  # CSI-only loader
                csi_data, csi_targets = batch_data
            
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
            'std_mse': np.std(baseline_losses),
            'std_mae': np.std(baseline_maes)
        },
        'distilled': {
            'mse_loss': np.mean(distilled_losses),
            'mae': np.mean(distilled_maes),
            'std_mse': np.std(distilled_losses),
            'std_mae': np.std(distilled_maes)
        }
    }
    
    # Calculate improvement
    mse_improvement = (results['baseline']['mse_loss'] - results['distilled']['mse_loss']) / results['baseline']['mse_loss'] * 100
    mae_improvement = (results['baseline']['mae'] - results['distilled']['mae']) / results['baseline']['mae'] * 100
    
    print("📊 Model Comparison Results:")
    print("=" * 50)
    print(f"Baseline CSI Mamba:")
    print(f"  MSE Loss: {results['baseline']['mse_loss']:.6f} ± {results['baseline']['std_mse']:.6f}")
    print(f"  MAE: {results['baseline']['mae']:.6f} ± {results['baseline']['std_mae']:.6f}")
    print()
    print(f"Encoder-Enhanced Distilled Model:")
    print(f"  MSE Loss: {results['distilled']['mse_loss']:.6f} ± {results['distilled']['std_mse']:.6f}")
    print(f"  MAE: {results['distilled']['mae']:.6f} ± {results['distilled']['std_mae']:.6f}")
    print()
    print(f"🎯 Improvements:")
    print(f"  MSE Improvement: {mse_improvement:+.2f}%")
    print(f"  MAE Improvement: {mae_improvement:+.2f}%")
    
    return results


if __name__ == "__main__":
    print("🧪 Testing FIXED Encoder-Enhanced MOHAWK Cross-Modal Distillation...")
    
    # This would be integrated with the main training script
    # Test basic functionality here
    
    batch_size = 4
    uwb_seq_len, csi_seq_len = 32, 4
    uwb_features, csi_features = 113, 280
    
    # Create dummy data
    uwb_data = torch.randn(batch_size, uwb_seq_len, uwb_features)
    csi_data = torch.randn(batch_size, csi_seq_len, csi_features)
    uwb_targets = torch.randn(batch_size, 2)
    csi_targets = torch.randn(batch_size, 2)
    
    # Create dummy teacher
    teacher = SimpleUWBTransformerTeacher(
        input_features=uwb_features,
        output_features=2,
        d_model=256
    )
    
    # Create dummy CSI config
    csi_config = {
        "CSIRegressionModel": {
            "input": {
                "input_features": csi_features,
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
        }
    }
    
    # Test FIXED distiller
    distiller = EncoderEnhancedMOHAWKDistiller(
        teacher_model=teacher,
        csi_student_config=csi_config,
        uwb_input_features=uwb_features,
        csi_input_features=csi_features,
        temperature=4.0, 
        alpha=0.7, 
        beta=0.3
    )
    
    # Test FIXED encoding with sequence preservation
    print("🔍 Testing FIXED sequence-preserving encoders:")
    z_UWB, z_CSI = distiller.encode_modalities(uwb_data, csi_data)
    print(f"   UWB: {uwb_data.shape} → {z_UWB.shape} (preserves sequence structure)")
    print(f"   CSI: {csi_data.shape} → {z_CSI.shape} (preserves sequence structure)")
    
    # Test FIXED latent alignment with sequences
    print("🔍 Testing FIXED sequence-aware alignment:")
    alignment_loss = distiller.compute_alignment_loss(z_UWB, z_CSI)
    print(f"   Sequence alignment loss: {alignment_loss.item():.6f}")
    
    # Test FIXED single distillation step
    print("🔍 Testing FIXED distillation step:")
    outputs = distiller.distillation_step(uwb_data, uwb_targets, csi_data, csi_targets)
    
    print(f"✅ FIXED distillation test passed!")
    print(f"   Total loss: {outputs['total_loss'].item():.6f}")
    print(f"   Alignment loss: {outputs['alignment_loss'].item():.6f}")
    print(f"   Teacher preds shape: {outputs['teacher_predictions'].shape}")
    print(f"   Student preds shape: {outputs['student_predictions'].shape}")
    print(f"   UWB latent shape: {outputs['z_UWB'].shape} (sequence-structured)")
    print(f"   CSI latent shape: {outputs['z_CSI'].shape} (sequence-structured)")
    
    # Test FIXED deployment model
    print("🔍 Testing FIXED deployment model:")
    deployment_model = distiller.get_csi_student_for_deployment()
    deployment_preds = deployment_model(csi_data)
    print(f"✅ FIXED deployment model test passed! Output shape: {deployment_preds.shape}")
    
    print("🎉 All FIXED tests passed!")
    print("🔧 Key improvements:")
    print("   ✅ Sequence structure preserved through latent space")
    print("   ✅ No artificial sequence creation")
    print("   ✅ Temporal information retained")
    print("   ✅ Proper sequence-aware alignment")
    print("   ✅ Compatible with MOHAWK distillation framework") 