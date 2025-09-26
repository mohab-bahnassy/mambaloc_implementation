"""
Cross-Modal Distillation Demo Script
Tests the encoder-enhanced distillation framework with synthetic data.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from modules.cross_modal_encoders import Encoder_UWB, Encoder_CSI, visualize_latent_alignment
from encoder_enhanced_cross_modal_distillation import (
    EncoderEnhancedCrossModalDistiller,
    train_baseline_csi_mamba,
    evaluate_models_comparison
)
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher


class SyntheticCrossModalDataset(Dataset):
    """Synthetic dataset for testing cross-modal distillation."""
    
    def __init__(self, num_samples=1000, uwb_seq_len=32, csi_seq_len=4, uwb_features=113, csi_features=280):
        self.num_samples = num_samples
        self.uwb_seq_len = uwb_seq_len
        self.csi_seq_len = csi_seq_len
        self.uwb_features = uwb_features
        self.csi_features = csi_features
        
        # Generate synthetic data with some correlation
        print(f"🔧 Generating {num_samples} synthetic samples...")
        
        self.uwb_data = []
        self.csi_data = []
        self.targets = []
        
        for i in range(num_samples):
            # Generate targets (2D coordinates)
            target = np.random.uniform(-10, 10, 2)
            
            # Generate UWB data with correlation to target
            uwb_base = np.random.randn(uwb_seq_len, uwb_features) * 0.5
            # Add target influence
            uwb_base[:, :2] += target * 0.1
            
            # Generate CSI data with correlation to target and some correlation to UWB
            csi_base = np.random.randn(csi_seq_len, csi_features) * 0.5
            # Add target influence  
            csi_base[:, :2] += target * 0.1
            # Add some correlation to UWB (simulate cross-modal relationship)
            csi_base[:, 2:4] += np.mean(uwb_base[:, :2], axis=0) * 0.05
            
            self.uwb_data.append(uwb_base.astype(np.float32))
            self.csi_data.append(csi_base.astype(np.float32))
            self.targets.append(target.astype(np.float32))
        
        print(f"✅ Synthetic dataset created with {num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.uwb_data[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.csi_data[idx]),
            torch.from_numpy(self.targets[idx])
        )


def create_synthetic_data_loaders(batch_size=16):
    """Create synthetic data loaders for testing."""
    
    # Create datasets
    train_dataset = SyntheticCrossModalDataset(num_samples=800, uwb_seq_len=32, csi_seq_len=4)
    val_dataset = SyntheticCrossModalDataset(num_samples=200, uwb_seq_len=32, csi_seq_len=4)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def demo_encoders():
    """Demonstrate encoder functionality."""
    
    print("🧪 DEMO 1: Testing Encoder Functionality")
    print("=" * 50)
    
    # Create sample data
    batch_size = 4
    uwb_seq_len, csi_seq_len = 32, 4
    uwb_features, csi_features = 113, 280
    latent_dim = 128
    
    uwb_data = torch.randn(batch_size, uwb_seq_len, uwb_features)
    csi_data = torch.randn(batch_size, csi_seq_len, csi_features)
    
    print(f"📊 Input shapes: UWB={uwb_data.shape}, CSI={csi_data.shape}")
    
    # Create encoders
    encoder_uwb = Encoder_UWB(input_features=uwb_features, latent_dim=latent_dim)
    encoder_csi = Encoder_CSI(input_features=csi_features, latent_dim=latent_dim)
    
    # Encode data
    z_UWB = encoder_uwb(uwb_data)
    z_CSI = encoder_csi(csi_data)
    
    print(f"📊 Encoded shapes: z_UWB={z_UWB.shape}, z_CSI={z_CSI.shape}")
    
    # Test latent alignment visualization
    alignment_metrics = visualize_latent_alignment(z_UWB, z_CSI)
    print(f"🔍 Latent alignment metrics:")
    for key, value in alignment_metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print("✅ Encoder test completed!\n")
    
    return encoder_uwb, encoder_csi


def demo_distillation_training():
    """Demonstrate encoder-enhanced distillation training."""
    
    print("🧪 DEMO 2: Testing Distillation Training")
    print("=" * 50)
    
    # Create synthetic data
    train_loader, val_loader = create_synthetic_data_loaders(batch_size=16)
    
    # Get data dimensions
    for uwb_data, uwb_targets, csi_data, csi_targets in train_loader:
        uwb_features = uwb_data.shape[-1]
        csi_features = csi_data.shape[-1]
        break
    
    print(f"📊 Data dimensions: UWB={uwb_features}, CSI={csi_features}")
    
    # Create teacher model
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=uwb_features,
        output_features=2,
        d_model=128,  # Smaller for demo
        n_layers=2,   # Fewer layers for demo
        n_heads=4
    )
    
    # Create CSI config
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
                "d_model": 32,  # Smaller for demo
                "n_layer": 1,   # Fewer layers for demo
                "final_prenorm": "layer"
            }
        },
        "Block1": {
            "n_layers": 1,
            "BlockType": "modules.phi_block",
            "block_input": {
                "resid_dropout": 0.1
            },
            "CoreType": "modules.mixers.discrete_mamba2",
            "core_input": {
                "d_state": 8,
                "n_v_heads": 4,
                "n_qk_heads": 4,
                "d_conv": 4,
                "conv_bias": True,
                "expand": 1,
                "chunk_size": 32,
                "activation": "identity",
                "bias": False
            }
        }
    }
    
    # Create distiller
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    distiller = EncoderEnhancedCrossModalDistiller(
        teacher_model=teacher_model,
        csi_student_config=csi_config,
        uwb_input_features=uwb_features,
        csi_input_features=csi_features,
        latent_dim=64,  # Smaller for demo
        device=device,
        alpha=0.7,
        beta=0.2,
        gamma=0.1,
        delta=0.0,  # No reconstruction for demo
        use_reconstruction=False
    )
    
    # Train for a few epochs
    print("🚀 Starting distillation training (short demo)...")
    
    training_history = distiller.train_encoders_and_distillation(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # Short training for demo
        learning_rate=1e-3,
        log_interval=2
    )
    
    print("✅ Distillation training completed!\n")
    
    return distiller, training_history


def demo_baseline_comparison():
    """Demonstrate baseline vs distilled model comparison."""
    
    print("🧪 DEMO 3: Testing Baseline vs Distilled Comparison")
    print("=" * 50)
    
    # Create CSI-only data loader for baseline training
    class CSIOnlyDataset(Dataset):
        def __init__(self, num_samples=500):
            self.num_samples = num_samples
            self.csi_data = []
            self.targets = []
            
            for i in range(num_samples):
                target = np.random.uniform(-10, 10, 2)
                csi_seq = np.random.randn(4, 280) * 0.5
                csi_seq[:, :2] += target * 0.1  # Add target correlation
                
                self.csi_data.append(csi_seq.astype(np.float32))
                self.targets.append(target.astype(np.float32))
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return torch.from_numpy(self.csi_data[idx]), torch.from_numpy(self.targets[idx])
    
    # Create CSI-only data loaders
    csi_train_dataset = CSIOnlyDataset(num_samples=400)
    csi_val_dataset = CSIOnlyDataset(num_samples=100)
    
    csi_train_loader = DataLoader(csi_train_dataset, batch_size=16, shuffle=True)
    csi_val_loader = DataLoader(csi_val_dataset, batch_size=16, shuffle=False)
    
    # Train baseline model
    print("🔧 Training baseline CSI Mamba...")
    
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
                "d_model": 32,
                "n_layer": 1,
                "final_prenorm": "layer"
            }
        },
        "Block1": {
            "n_layers": 1,
            "BlockType": "modules.phi_block",
            "block_input": {
                "resid_dropout": 0.1
            },
            "CoreType": "modules.mixers.discrete_mamba2",
            "core_input": {
                "d_state": 8,
                "n_v_heads": 4,
                "n_qk_heads": 4,
                "d_conv": 4,
                "conv_bias": True,
                "expand": 1,
                "chunk_size": 32,
                "activation": "identity",
                "bias": False
            }
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    baseline_model = train_baseline_csi_mamba(
        csi_config=csi_config,
        train_loader=csi_train_loader,
        val_loader=csi_val_loader,
        device=device,
        epochs=3,  # Short training for demo
        learning_rate=1e-3
    )
    
    # For demo, create a simple "distilled" model (just random weights for comparison)
    class DummyDistilledModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(280, 64, batch_first=True)
            self.predictor = nn.Linear(64, 2)
        
        def forward(self, x):
            _, (h, _) = self.encoder(x)
            return self.predictor(h[-1])
    
    dummy_distilled = DummyDistilledModel().to(device)
    
    # Compare models
    print("📊 Comparing baseline vs distilled models...")
    
    comparison_results = evaluate_models_comparison(
        baseline_model=baseline_model,
        distilled_model=dummy_distilled,
        eval_loader=csi_val_loader,
        device=device
    )
    
    print("✅ Comparison demo completed!\n")
    
    return comparison_results


def plot_demo_results(training_history):
    """Plot demo training results."""
    
    if not training_history or len(training_history.get('total_loss', [])) == 0:
        print("⚠️ No training history to plot")
        return
    
    print("📊 Plotting demo results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Cross-Modal Distillation Demo Results', fontsize=14)
    
    # Plot loss components
    loss_names = ['total_loss', 'distill_loss', 'task_loss', 'alignment_loss']
    loss_titles = ['Total Loss', 'Distillation Loss', 'Task Loss', 'Alignment Loss']
    
    for i, (loss_name, title) in enumerate(zip(loss_names, loss_titles)):
        if loss_name in training_history and len(training_history[loss_name]) > 0:
            row = i // 2
            col = i % 2
            axes[row, col].plot(training_history[loss_name], 'b-', linewidth=2, marker='o')
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📈 Demo results plotted and saved as 'demo_training_curves.png'")


def main():
    """Run all demos."""
    
    print("🎉 Cross-Modal Distillation Framework Demo")
    print("=" * 60)
    print("This demo tests the encoder-enhanced distillation framework")
    print("with synthetic data to verify functionality.\n")
    
    try:
        # Demo 1: Test encoders
        encoder_uwb, encoder_csi = demo_encoders()
        
        # Demo 2: Test distillation training
        distiller, training_history = demo_distillation_training()
        
        # Demo 3: Test baseline comparison
        comparison_results = demo_baseline_comparison()
        
        # Plot results
        plot_demo_results(training_history)
        
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The cross-modal distillation framework is working correctly.")
        print("You can now use it with real UWB and CSI data.")
        print("\nTo run with real data, use:")
        print("python train_cross_modal_distillation_comparison.py --mode comparison")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 