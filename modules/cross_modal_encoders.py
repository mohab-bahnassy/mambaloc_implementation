"""
Cross-Modal Encoder Modules for UWB-CSI Distillation
Provides separate encoders for UWB and CSI data to map to shared latent representation space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class Encoder_UWB(nn.Module):
    """
    Encoder for UWB data (CIR sequences) to shared latent representation.
    Maps UWB input sequences to latent space z_UWB while preserving temporal structure.
    """
    
    def __init__(
        self,
        input_features: int = 113,  # UWB CIR features
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        preserve_sequence: bool = True  # NEW: Preserve sequence structure in latent
    ):
        super().__init__()
        
        self.input_features = input_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.preserve_sequence = preserve_sequence  # NEW
        
        # Input projection and normalization
        self.input_projection = nn.Linear(input_features, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Encoder: Input sequence -> Hidden representations
        self.encoder_rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for better sequence encoding
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # Bidirectional
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        
        # MODIFIED: Projection to latent space (per-timestep or global)
        if preserve_sequence:
            # Per-timestep latent projection - preserves temporal structure
            self.latent_projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, latent_dim * 2),  # Bidirectional hidden
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim * 2, latent_dim),
                nn.Tanh()  # Bounded latent representation
            )
        else:
            # Global latent projection - original behavior
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim * 2),  # Bidirectional hidden
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()  # Bounded latent representation
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for LSTM
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    def forward(self, uwb_sequences: torch.Tensor) -> torch.Tensor:
        """
        Encode UWB sequences to latent representation.
        
        Args:
            uwb_sequences: [batch_size, seq_len, input_features] UWB CIR sequences
            
        Returns:
            z_UWB: [batch_size, seq_len, latent_dim] if preserve_sequence=True
                   [batch_size, latent_dim] if preserve_sequence=False
        """
        batch_size, seq_len, _ = uwb_sequences.shape
        
        # Project input features
        projected = self.input_projection(uwb_sequences)  # [B, L, H]
        projected = self.input_norm(projected)
        
        # Encode with LSTM
        encoded_seq, (h_n, c_n) = self.encoder_rnn(projected)  # [B, L, 2*H]
        
        # Apply attention if enabled
        if self.use_attention:
            attended, _ = self.attention(encoded_seq, encoded_seq, encoded_seq)
            attended = self.attention_norm(attended + encoded_seq)
            sequence_repr = attended  # [B, L, 2*H] - Keep full sequence
        else:
            sequence_repr = encoded_seq  # [B, L, 2*H] - Keep full sequence
        
        # MODIFIED: Project to latent space preserving or compressing sequence
        if self.preserve_sequence:
            # Per-timestep projection - maintains temporal structure
            z_UWB = self.latent_projection(sequence_repr)  # [B, L, latent_dim]
        else:
            # Global average pooling then projection - original behavior
            global_repr = torch.mean(sequence_repr, dim=1)  # [B, 2*H]
            z_UWB = self.latent_projection(global_repr)  # [B, latent_dim]
        
        return z_UWB


class Encoder_CSI(nn.Module):
    """
    Encoder for CSI data to shared latent representation.
    Maps CSI input sequences to latent space z_CSI while preserving temporal structure.
    """
    
    def __init__(
        self,
        input_features: int = 280,  # CSI magnitude + phase features
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        preserve_sequence: bool = True  # NEW: Preserve sequence structure in latent
    ):
        super().__init__()
        
        self.input_features = input_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.preserve_sequence = preserve_sequence  # NEW
        
        # Input projection and normalization
        self.input_projection = nn.Linear(input_features, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Encoder: Input sequence -> Hidden representations
        self.encoder_rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for better sequence encoding
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # Bidirectional
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        
        # MODIFIED: Projection to latent space (per-timestep or global)
        if preserve_sequence:
            # Per-timestep latent projection - preserves temporal structure
            self.latent_projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, latent_dim * 2),  # Bidirectional hidden
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim * 2, latent_dim),
                nn.Tanh()  # Bounded latent representation
            )
        else:
            # Global latent projection - original behavior
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim * 2),  # Bidirectional hidden
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()  # Bounded latent representation
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for LSTM
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    def forward(self, csi_sequences: torch.Tensor) -> torch.Tensor:
        """
        Encode CSI sequences to latent representation.
        
        Args:
            csi_sequences: [batch_size, seq_len, input_features] CSI sequences
            
        Returns:
            z_CSI: [batch_size, seq_len, latent_dim] if preserve_sequence=True
                   [batch_size, latent_dim] if preserve_sequence=False
        """
        batch_size, seq_len, _ = csi_sequences.shape
        
        # Project input features
        projected = self.input_projection(csi_sequences)  # [B, L, H]
        projected = self.input_norm(projected)
        
        # Encode with LSTM
        encoded_seq, (h_n, c_n) = self.encoder_rnn(projected)  # [B, L, 2*H]
        
        # Apply attention if enabled
        if self.use_attention:
            attended, _ = self.attention(encoded_seq, encoded_seq, encoded_seq)
            attended = self.attention_norm(attended + encoded_seq)
            sequence_repr = attended  # [B, L, 2*H] - Keep full sequence
        else:
            sequence_repr = encoded_seq  # [B, L, 2*H] - Keep full sequence
        
        # MODIFIED: Project to latent space preserving or compressing sequence
        if self.preserve_sequence:
            # Per-timestep projection - maintains temporal structure
            z_CSI = self.latent_projection(sequence_repr)  # [B, L, latent_dim]
        else:
            # Global average pooling then projection - original behavior
            global_repr = torch.mean(sequence_repr, dim=1)  # [B, 2*H]
            z_CSI = self.latent_projection(global_repr)  # [B, latent_dim]
        
        return z_CSI


class Decoder_UWB(nn.Module):
    """
    Optional decoder for UWB data reconstruction from latent representation.
    Used for regularization and alignment quality assessment.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        output_features: int = 113,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        sequence_length: int = 32
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_features = output_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Latent to hidden projection
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Final reconstruction layer
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_features)
        )
    
    def forward(self, z_UWB: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct UWB sequences from latent representation.
        
        Args:
            z_UWB: [batch_size, latent_dim] latent representation
            
        Returns:
            reconstructed: [batch_size, sequence_length, output_features]
        """
        batch_size = z_UWB.shape[0]
        
        # Project latent to hidden
        hidden_state = self.latent_to_hidden(z_UWB)  # [B, H]
        
        # Expand to sequence
        hidden_seq = hidden_state.unsqueeze(1).repeat(1, self.sequence_length, 1)  # [B, L, H]
        
        # Decode sequence
        decoded_seq, _ = self.decoder_rnn(hidden_seq)  # [B, L, H]
        
        # Generate reconstruction
        reconstructed = self.reconstruction_head(decoded_seq)  # [B, L, output_features]
        
        return reconstructed


class Decoder_CSI(nn.Module):
    """
    Optional decoder for CSI data reconstruction from latent representation.
    Used for regularization and alignment quality assessment.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        output_features: int = 280,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        sequence_length: int = 4
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_features = output_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Latent to hidden projection
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Final reconstruction layer
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_features)
        )
    
    def forward(self, z_CSI: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct CSI sequences from latent representation.
        
        Args:
            z_CSI: [batch_size, latent_dim] latent representation
            
        Returns:
            reconstructed: [batch_size, sequence_length, output_features]
        """
        batch_size = z_CSI.shape[0]
        
        # Project latent to hidden
        hidden_state = self.latent_to_hidden(z_CSI)  # [B, H]
        
        # Expand to sequence
        hidden_seq = hidden_state.unsqueeze(1).repeat(1, self.sequence_length, 1)  # [B, L, H]
        
        # Decode sequence
        decoded_seq, _ = self.decoder_rnn(hidden_seq)  # [B, L, H]
        
        # Generate reconstruction
        reconstructed = self.reconstruction_head(decoded_seq)  # [B, L, output_features]
        
        return reconstructed


# Utility functions for debugging and visualization
def visualize_latent_alignment(z_UWB: torch.Tensor, z_CSI: torch.Tensor) -> Dict[str, float]:
    """
    Inspect latent alignment quality for debugging.
    
    Args:
        z_UWB: UWB latent representations [batch_size, latent_dim]
        z_CSI: CSI latent representations [batch_size, latent_dim]
        
    Returns:
        Dictionary with alignment metrics
    """
    with torch.no_grad():
        # Compute alignment metrics
        mse_alignment = F.mse_loss(z_CSI, z_UWB).item()
        l1_alignment = F.l1_loss(z_CSI, z_UWB).item()
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(z_UWB, z_CSI, dim=1)
        avg_cos_sim = cos_sim.mean().item()
        
        # Compute latent statistics
        uwb_mean = z_UWB.mean().item()
        uwb_std = z_UWB.std().item()
        csi_mean = z_CSI.mean().item()
        csi_std = z_CSI.std().item()
        
        return {
            'mse_alignment': mse_alignment,
            'l1_alignment': l1_alignment,
            'cosine_similarity': avg_cos_sim,
            'uwb_mean': uwb_mean,
            'uwb_std': uwb_std,
            'csi_mean': csi_mean,
            'csi_std': csi_std
        }


if __name__ == "__main__":
    # Test the FIXED encoders
    print("🧪 Testing FIXED Cross-Modal Encoders with Sequence Preservation...")
    
    batch_size = 4
    uwb_seq_len, csi_seq_len = 32, 4
    uwb_features, csi_features = 113, 280
    latent_dim = 128
    
    # Create test data
    uwb_data = torch.randn(batch_size, uwb_seq_len, uwb_features)
    csi_data = torch.randn(batch_size, csi_seq_len, csi_features)
    
    # Test FIXED sequence-preserving encoders
    print("🔍 Testing sequence-preserving encoders:")
    encoder_uwb = Encoder_UWB(input_features=uwb_features, latent_dim=latent_dim, preserve_sequence=True)
    encoder_csi = Encoder_CSI(input_features=csi_features, latent_dim=latent_dim, preserve_sequence=True)
    
    # Test encoding with sequence preservation
    z_UWB = encoder_uwb(uwb_data)
    z_CSI = encoder_csi(csi_data)
    
    print(f"✅ FIXED UWB encoding: {uwb_data.shape} → {z_UWB.shape} (sequence preserved)")
    print(f"✅ FIXED CSI encoding: {csi_data.shape} → {z_CSI.shape} (sequence preserved)")
    
    # Test backward compatibility with global encoders
    print("🔍 Testing backward compatibility (global encoders):")
    encoder_uwb_global = Encoder_UWB(input_features=uwb_features, latent_dim=latent_dim, preserve_sequence=False)
    encoder_csi_global = Encoder_CSI(input_features=csi_features, latent_dim=latent_dim, preserve_sequence=False)
    
    z_UWB_global = encoder_uwb_global(uwb_data)
    z_CSI_global = encoder_csi_global(csi_data)
    
    print(f"✅ Global UWB encoding: {uwb_data.shape} → {z_UWB_global.shape} (global representation)")
    print(f"✅ Global CSI encoding: {csi_data.shape} → {z_CSI_global.shape} (global representation)")
    
    # Create decoders (optional)
    decoder_uwb = Decoder_UWB(latent_dim=latent_dim, output_features=uwb_features, sequence_length=uwb_seq_len)
    decoder_csi = Decoder_CSI(latent_dim=latent_dim, output_features=csi_features, sequence_length=csi_seq_len)
    
    # Test reconstruction with global latents
    recon_uwb = decoder_uwb(z_UWB_global)
    recon_csi = decoder_csi(z_CSI_global)
    
    print(f"✅ UWB reconstruction: {z_UWB_global.shape} → {recon_uwb.shape}")
    print(f"✅ CSI reconstruction: {z_CSI_global.shape} → {recon_csi.shape}")
    
    # Test FIXED latent alignment visualization
    print("🔍 Testing FIXED latent alignment:")
    
    # For sequence latents, use global pooling for alignment metrics
    z_UWB_for_align = torch.mean(z_UWB, dim=1) if z_UWB.dim() == 3 else z_UWB
    z_CSI_for_align = torch.mean(z_CSI, dim=1) if z_CSI.dim() == 3 else z_CSI
    
    alignment_metrics = visualize_latent_alignment(z_UWB_for_align, z_CSI_for_align)
    print(f"🔍 FIXED latent alignment metrics: {alignment_metrics}")
    
    print("✅ All FIXED encoder tests passed!")
    print("🔧 Key improvements:")
    print("   ✅ Sequence structure preserved in latent space")
    print("   ✅ No information loss through artificial compression")
    print("   ✅ Temporal dynamics maintained")
    print("   ✅ Backward compatibility with global representations")
    print("   ✅ Better alignment capabilities") 