import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm import Mamba


class Mixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        n_qk_heads=8,
        n_v_heads=8,
        d_conv=4,
        expand=2,
        activation="silu",
        bias=False,
        conv_bias=True,
        chunk_size=64,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        """
        Official Mamba mixer using mamba-ssm library.
        
        Args:
            d_model: Model dimension
            d_state: State dimension for SSM
            n_qk_heads: Number of QK heads (not used in Mamba, kept for compatibility)
            n_v_heads: Number of value heads (not used in Mamba, kept for compatibility)
            d_conv: Convolution dimension
            expand: Expansion factor
            activation: Activation function
            bias: Whether to use bias in linear layers
            conv_bias: Whether to use bias in convolution
            chunk_size: Chunk size for processing
            layer_idx: Layer index
            device: Device to place model on
            dtype: Data type
        """
        super().__init__()
        
        # Store parameters for compatibility
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.n_qk_heads = n_qk_heads
        self.n_v_heads = n_v_heads
        self.headdim = self.d_inner // self.n_v_heads
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.bias = bias
        self.kwargs = kwargs
        
        # Create official Mamba model
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=bias,
            conv_bias=conv_bias,
            device=device,
            dtype=dtype,
        )
        
        # D parameter for skip connection (as in original implementation)
        self.D = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.D._optim = {"weight_decay": 0.0}

    @property
    def d_output(self):
        return self.d_model

    def forward(self, u, return_mixer_matrix=False, inference_params=None, **kwargs):
        """
        Forward pass using official Mamba.
        
        Args:
            u: Input tensor (B, L, D)
            return_mixer_matrix: Whether to return transfer matrix (not supported in official Mamba)
            inference_params: Inference parameters
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with hidden_states and optional transfer_matrix
        """
        batch, seqlen, dim = u.shape
        
        # Apply official Mamba
        y = self.mamba(u)
        
        # Add skip connection (D parameter)
        y = y + self.D[None, None, :] * u
        
        outputs = {"hidden_states": y}
        
        # Note: Official Mamba doesn't support transfer matrix computation
        # This is a limitation compared to the custom implementation
        if return_mixer_matrix:
            print("Warning: Transfer matrix computation not supported with official Mamba")
            outputs["transfer_matrix"] = None
            
        return outputs

    def step(self, u, state, **kwargs):
        """
        Step function for inference (not fully implemented with official Mamba).
        
        Args:
            u: Input tensor (B, D)
            state: State dictionary
            **kwargs: Additional arguments
            
        Returns:
            Output and updated state
        """
        # Note: Official Mamba doesn't have a direct step function
        # This would need to be implemented differently for inference
        raise NotImplementedError("Step function not implemented with official Mamba")

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate inference cache (not needed for official Mamba).
        
        Args:
            batch_size: Batch size
            max_seqlen: Maximum sequence length
            dtype: Data type
            **kwargs: Additional arguments
            
        Returns:
            Empty dictionary (no cache needed)
        """
        # Official Mamba doesn't need explicit cache allocation
        return {}


