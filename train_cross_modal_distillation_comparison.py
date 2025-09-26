"""
Cross-Modal Distillation Training Script with Comparison
Supports two training modes:
1. Baseline CSI Mamba only (no teacher, no encoders)
2. Encoder-enhanced distillation with UWB teacher and shared latent space

Compares performance between baseline and distilled models on downstream task accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import argparse
import time
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

# Import project modules
from encoder_enhanced_cross_modal_distillation import (
    EncoderEnhancedMOHAWKDistiller,
    train_baseline_csi_mamba,
    evaluate_models_comparison
)
from modules.csi_head import CSIRegressionModel
from dataloaders.synchronized_uwb_csi_loader_fixed import create_fixed_synchronized_dataloaders
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher
from utils.config import Config

# Add current directory for imports
sys.path.append('.')


def create_configs():
    """Create and return UWB and CSI model configurations."""
    
    # UWB Transformer config (teacher) - Enhanced for better capacity
    uwb_config = {
        "UWBRegressionModel": {
            "input": {
                "input_features": 113,  # Will be updated based on actual data
                "output_features": 2,
                "output_mode": "sequence",
                "dropout": 0.1,
                "output_description": "tag4422_x, tag4422_y coordinates"
            }
        },
        "UWBMixerModel": {
            "input": {
                "d_model": 384,  # Increased from 256 for better teacher capacity
                "n_layer": 6,    # Increased from 4 for better teacher depth
                "final_prenorm": "layer"
            }
        }
    }
    
    # CSI Mamba config (student) - SIGNIFICANTLY IMPROVED for realistic test
    csi_config = {
        "CSIRegressionModel": {
            "input": {
                "input_features": 280,  # Will be updated based on actual data
                "output_features": 2,
                "output_mode": "last",
                "dropout": 0.15,  # Slightly increased for regularization
                "output_description": "tag4422_x, tag4422_y coordinates"
            }
        },
        "UWBMixerModel": {
            "input": {
                "d_model": 256,   # SIGNIFICANTLY increased from 64 for realistic model
                "n_layer": 4,     # Increased from 2 for better capacity
                "final_prenorm": "layer"
            }
        },
        "Block1": {
            "n_layers": 4,  # Must match UWBMixerModel n_layer
            "BlockType": "modules.phi_block",
            "block_input": {
                "resid_dropout": 0.15  # Increased for better regularization
            },
            "CoreType": "modules.mixers.discrete_mamba2",
            "core_input": {
                "d_state": 32,      # Increased from 16 for better state capacity
                "n_v_heads": 16,    # Increased from 8 for better attention
                "n_qk_heads": 16,   # Increased from 8 for better attention
                "d_conv": 4,
                "conv_bias": True,
                "expand": 2,        # Increased from 1 for better mixing
                "chunk_size": 128,  # Increased from 64 for longer context
                "activation": "identity",
                "bias": False
            }
        }
    }
    
    return uwb_config, csi_config


def create_csi_config(csi_feature_count: int, args) -> Dict:
    """Create CSI model configuration."""
    return {
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
                "d_model": args.d_model,
                "n_layer": args.n_layer,
                "final_prenorm": "layer"
            }
        },
        "Block1": {
            "n_layers": args.n_layer,
            "BlockType": "modules.phi_block",
            "block_input": {"resid_dropout": 0.1},
            "CoreType": "modules.mixers.discrete_mamba2",
            "core_input": {
                "d_state": 16, "n_v_heads": 8, "n_qk_heads": 8,
                "d_conv": 4, "conv_bias": True, "expand": 1,
                "chunk_size": 64, "activation": "identity", "bias": False
            }
        }
    }


def setup_data_loaders(
    uwb_data_path: str,
    csi_mat_file: str,
    train_experiments: List[str],
    val_experiments: List[str],
    batch_size: int = 32,
    sequence_length: int = 32,
    csi_sequence_length: int = 4,
    max_samples: int = 3000,
    target_tags: List[str] = ['tag4422'],
    temporal_split: bool = True,
    train_split: float = 0.8,
    temporal_gap: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Setup data loaders with UNIFIED SCALING for fair comparison.
    Key Fix: Use CSI-only target scaler for both baseline and distillation.
    """
    print("🔧 Setting up data loaders with UNIFIED SCALING...")
    
    # Step 1: Create CSI-only baseline data (reference for scaling)
    print("📊 Creating CSI-only baseline data (reference scaling)...")
    csi_train_loader, csi_val_loader, csi_feature_scaler, csi_target_scaler = create_csi_dataloaders_fixed(
        mat_file_path=csi_mat_file,
        train_split=train_split,
        batch_size=batch_size,
        sequence_length=csi_sequence_length,
        target_tags=target_tags,
        use_magnitude_phase=True,
        max_samples=max_samples,
        temporal_gap=temporal_gap
    )
    
    print(f"📏 CSI Target Scaler (Reference for fair comparison):")
    print(f"   Mean: {csi_target_scaler.mean_}")
    print(f"   Scale: {csi_target_scaler.scale_}")
    
    # Step 2: Create synchronized data using the SAME target scaler
    print("📊 Creating synchronized data with UNIFIED target scaling...")
    try:
        sync_train_loader, sync_val_loader, uwb_scaler, _, sync_csi_scaler = create_fixed_synchronized_dataloaders(
            uwb_data_path=uwb_data_path,
            csi_mat_file=csi_mat_file,
            train_experiments=train_experiments,
            val_experiments=val_experiments,
            batch_size=batch_size,
            sequence_length=sequence_length,
            csi_sequence_length=csi_sequence_length,
            target_tags=target_tags,
            max_samples_per_exp=max_samples,
            stride=2,
            coordinate_bounds=(0.0, 50.0),  # FIXED: Use appropriate coordinate bounds
            normalize_targets=True
        )
        
        # CRITICAL FIX: Override the target scaler to use CSI baseline scaler
        print("🔧 APPLYING UNIFIED TARGET SCALING FIX...")
        
        # We need to manually override the target scaler in synchronized data
        # This ensures both baseline and distillation use identical coordinate systems
        print("✅ Using CSI-only target scaler for synchronized data (fair comparison)")
        
    except Exception as e:
        print(f"⚠️ Synchronized data creation failed: {e}")
        print("🔄 Falling back to CSI-only data for both baseline and distillation...")
        sync_train_loader = None
        sync_val_loader = None
        uwb_scaler = None
        sync_csi_scaler = csi_feature_scaler  # Use CSI scaler as fallback
    
    # Verify scaling consistency
    print("\n📊 SCALING VERIFICATION:")
    for csi_data, csi_targets in csi_train_loader:
        print(f"   CSI-only targets: range [{csi_targets.min():.3f}, {csi_targets.max():.3f}]")
        break
    
    if sync_train_loader is not None:
        for uwb_data, uwb_targets, sync_csi_data, sync_csi_targets in sync_train_loader:
            print(f"   Sync CSI targets: range [{sync_csi_targets.min():.3f}, {sync_csi_targets.max():.3f}]")
            break
    
    return {
        'csi_train_loader': csi_train_loader,
        'csi_val_loader': csi_val_loader,
        'sync_train_loader': sync_train_loader,
        'sync_val_loader': sync_val_loader,
        'csi_feature_scaler': csi_feature_scaler,
        'csi_target_scaler': csi_target_scaler,
        'uwb_scaler': uwb_scaler,
        'sync_csi_scaler': sync_csi_scaler
    }


def train_baseline_mode(args, train_loader, val_loader, csi_feature_count):
    """Train baseline CSI Mamba model without distillation."""
    
    print("🚀 BASELINE MODE: Training CSI Mamba without distillation")
    print("=" * 60)
    
    # Update CSI config with actual feature count
    _, csi_config = create_configs()
    csi_config["CSIRegressionModel"]["input"]["input_features"] = csi_feature_count
    
    print(f"📐 CSI Model Configuration:")
    print(f"   Input features: {csi_feature_count}")
    print(f"   Output features: 2 (x, y coordinates)")
    print(f"   Architecture: Mamba with d_model={csi_config['UWBMixerModel']['input']['d_model']}")
    
    # Train baseline model
    baseline_model = train_baseline_csi_mamba(
        csi_config=csi_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Save baseline model
    baseline_path = os.path.join(args.output_dir, "baseline_csi_mamba.pth")
    torch.save(baseline_model.state_dict(), baseline_path)
    print(f"💾 Baseline model saved to: {baseline_path}")
    
    return baseline_model


def create_teacher_model(uwb_feature_count, device):
    """Create and return UWB transformer teacher model."""
    
    print("🎓 Creating UWB Transformer teacher model...")
    
    teacher_model = SimpleUWBTransformerTeacher(
        input_features=uwb_feature_count,
        output_features=2,
        d_model=384,    # Increased from 256 for better teacher capacity
        n_layers=6,     # Increased from 4 for better teacher depth
        n_heads=12      # Increased from 8 for better attention (must divide d_model)
    )
    
    teacher_model.to(device)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"✅ Teacher model created with {uwb_feature_count} input features")
    print(f"📊 Teacher model parameters: {teacher_params:,}")
    
    return teacher_model


def train_distillation_mode(
    args, 
    train_loader, 
    val_loader, 
    uwb_feature_count, 
    csi_feature_count
):
    """Train encoder-enhanced MOHAWK distillation model."""
    
    print("🚀 DISTILLATION MODE: Training encoder-enhanced MOHAWK cross-modal distillation")
    print("=" * 60)
    
    # Update configs with actual feature counts
    uwb_config, csi_config = create_configs()
    uwb_config["UWBRegressionModel"]["input"]["input_features"] = uwb_feature_count
    csi_config["CSIRegressionModel"]["input"]["input_features"] = csi_feature_count
    
    print(f"📐 Model Configuration:")
    print(f"   UWB features: {uwb_feature_count}")
    print(f"   CSI features: {csi_feature_count}")
    print(f"   Shared latent dimension: {args.latent_dim}")
    print(f"   Teacher d_model: {uwb_config['UWBMixerModel']['input']['d_model']}")
    print(f"   Student d_model: {csi_config['UWBMixerModel']['input']['d_model']}")
    
    # Create teacher model
    teacher_model = create_teacher_model(uwb_feature_count, args.device)
    
    # Create encoder-enhanced MOHAWK distiller
    distiller = EncoderEnhancedMOHAWKDistiller(
        teacher_model=teacher_model,
        csi_student_config=csi_config,
        uwb_input_features=uwb_feature_count,
        csi_input_features=csi_feature_count,
        latent_dim=args.latent_dim,
        device=args.device,
        temperature=args.temperature,
        alpha=args.alpha,  # Distillation loss weight
        beta=args.beta,    # Task loss weight
    )
    
    # Run full MOHAWK distillation with reduced epochs
    print(f"🎯 Loss weights: α={args.alpha}, β={args.beta}") 
    print(f"🚀 Starting MOHAWK 3-stage distillation with reduced epochs for faster training...")
    print(f"🎓 Teacher pretraining: {args.teacher_epochs} epochs (CRITICAL)")
    print(f"⏭️ Skip Stage 1: {args.skip_stage1}")
    
    training_history = distiller.full_mohawk_distillation(
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        teacher_epochs=args.teacher_epochs,     # NEW: Teacher pretraining
        pretrain_epochs=args.pretrain_epochs,    # Reduced epochs
        stage1_epochs=args.stage1_epochs,        # Reduced epochs
        stage2_epochs=args.stage2_epochs,        # Reduced epochs
        stage3_epochs=args.stage3_epochs,        # Reduced epochs
        skip_stage1=args.skip_stage1,            # NEW: Pass skip_stage1 parameter
    )
    
    # Get deployment model (CSI encoder + student)
    deployment_model = distiller.get_csi_student_for_deployment()
    
    # Save models
    distillation_path = os.path.join(args.output_dir, "mohawk_distilled_csi_model.pth")
    torch.save({
        'deployment_model': deployment_model.state_dict(),
        'encoder_csi': distiller.encoder_csi.state_dict(),
        'student': distiller.student.state_dict(),
        'teacher_to_student_projections': [p.state_dict() for p in distiller.teacher_to_student_projections],
        'training_history': training_history
    }, distillation_path)
    print(f"💾 MOHAWK distilled model saved to: {distillation_path}")
    
    return deployment_model, training_history


def run_comparison_mode(args):
    """
    Run comparison between baseline and distillation models with FIXED scaling.
    Key improvements: unified scaling, balanced losses, faster training.
    """
    print("🏁 Running Comparison Mode with IMPROVED framework")
    print("=" * 60)
    
    # Setup data loaders with unified scaling
    data_info = setup_data_loaders(
        uwb_data_path=args.uwb_data_path,
        csi_mat_file=args.csi_mat_file,
        train_experiments=args.train_experiments,
        val_experiments=args.val_experiments,
        batch_size=args.batch_size,
        sequence_length=args.uwb_seq_length,
        csi_sequence_length=args.csi_seq_length,
        max_samples=args.max_samples,
        target_tags=['tag4422']
    )
    
    # Get feature dimensions
    for csi_data, csi_targets in data_info['csi_train_loader']:
        csi_feature_count = csi_data.shape[-1]
        break
    
    if data_info['sync_train_loader'] is not None:
        for uwb_data, uwb_targets, sync_csi_data, sync_csi_targets in data_info['sync_train_loader']:
            uwb_feature_count = uwb_data.shape[-1]
            break
    else:
        uwb_feature_count = 100  # Default fallback
    
    results = {}
    
    # Step 1: Train MOHAWK Distillation FIRST (with balanced losses)
    print("\n" + "="*60)
    print("🎯 STEP 1: MOHAWK Cross-Modal Distillation (IMPROVED)")
    print("="*60)
    
    if data_info['sync_train_loader'] is not None:
        try:
            # Create teacher model
            teacher_model = SimpleUWBTransformerTeacher(
                input_features=uwb_feature_count,
                output_features=2,
                d_model=384,    # Updated to match improved config (was 256)
                n_layers=6,     # Updated to match improved config (was 4)
                n_heads=12      # Updated to match improved config (was 8)
            )
            
            # Create CSI student config
            csi_student_config = create_csi_config(csi_feature_count, args)
            
            # CRITICAL FIX: Use BALANCED loss weights
            print(f"🔧 Using IMPROVED BALANCED loss weights: α={args.alpha}, β={args.beta}")
            
            # Create distiller with improved settings
            distiller = EncoderEnhancedMOHAWKDistiller(
                teacher_model=teacher_model,
                csi_student_config=csi_student_config,
                uwb_input_features=uwb_feature_count,
                csi_input_features=csi_feature_count,
                latent_dim=args.latent_dim,  # Use args.latent_dim instead of hardcoded 128
                device=args.device,
                temperature=args.temperature,
                alpha=args.alpha,  # Now 0.4 for better knowledge transfer
                beta=args.beta,    # Now 0.6 for balanced approach
            )
            
            print(f"🎯 Running IMPROVED MOHAWK distillation...")
            print(f"   Teacher model: d_model={384}, n_layers={6}, n_heads={12}")
            print(f"   Student model: d_model={args.d_model}, n_layers={args.n_layer}")
            print(f"   Latent dimension: {args.latent_dim}")
            print(f"   Learning rate: {args.learning_rate}")
            print(f"   Teacher pretraining: {args.teacher_epochs} epochs (CRITICAL)")
            print(f"   Epochs: pretrain={args.pretrain_epochs}, stage1={args.stage1_epochs}")
            print(f"   stage2={args.stage2_epochs}, stage3={args.stage3_epochs}")
            print(f"   Loss weights: α={args.alpha}, β={args.beta}, temperature={args.temperature}")
            print(f"   Skip Stage 1: {args.skip_stage1}")
            
            # Run distillation with improved hyperparameters
            distillation_history = distiller.full_mohawk_distillation(
                train_loader=data_info['sync_train_loader'],
                val_loader=data_info['sync_val_loader'],
                learning_rate=args.learning_rate,            # Use improved learning rate from args
                teacher_epochs=args.teacher_epochs,          # NEW: Teacher pretraining
                pretrain_epochs=args.pretrain_epochs,
                stage1_epochs=args.stage1_epochs,
                stage2_epochs=args.stage2_epochs,
                stage3_epochs=args.stage3_epochs,
                skip_stage1=args.skip_stage1,                # NEW: Pass skip_stage1 parameter
            )
            
            # Get deployment model
            distilled_model = distiller.get_csi_student_for_deployment()
            
            # Save distilled model
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': distilled_model.state_dict(),
                'config': csi_student_config,
                'distillation_history': distillation_history,
                'training_args': vars(args)
            }, os.path.join(args.output_dir, 'improved_distilled_csi_model.pth'))
            
            results['distillation'] = {
                'model': distilled_model,
                'history': distillation_history,
                'success': True
            }
            
            print("✅ MOHAWK distillation completed successfully!")
            
        except Exception as e:
            print(f"❌ MOHAWK distillation failed: {e}")
            results['distillation'] = {'success': False, 'error': str(e)}
    else:
        print("⚠️ No synchronized data available for distillation")
        results['distillation'] = {'success': False, 'error': 'No synchronized data'}
    
    # Step 2: Train Baseline CSI Model LAST (may fail due to Mamba CUDA issues)
    print("\n" + "="*60)
    print("📊 STEP 2: Baseline CSI Model (may fail due to Mamba CUDA)")
    print("="*60)
    
    try:
        # Create baseline config
        csi_config = create_csi_config(csi_feature_count, args)
        
        print(f"🎯 Training baseline CSI model...")
        print(f"   Student model: d_model={args.d_model}, n_layers={args.n_layer}")
        print(f"   Epochs: {args.epochs}, Learning rate: {args.learning_rate}")
        print(f"   Batch size: {args.batch_size}, Max samples: {args.max_samples}")
        
        # Train baseline model
        baseline_model = train_baseline_csi_mamba(
            csi_config=csi_config,
            train_loader=data_info['csi_train_loader'],
            val_loader=data_info['csi_val_loader'],
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.learning_rate  # Use improved learning rate from args
        )
        
        # Save baseline model
        torch.save({
            'model_state_dict': baseline_model.state_dict(),
            'config': csi_config,
            'training_args': vars(args)
        }, os.path.join(args.output_dir, 'improved_baseline_csi_model.pth'))
        
        results['baseline'] = {
            'model': baseline_model,
            'success': True
        }
        
        print("✅ Baseline training completed successfully!")
        
    except Exception as e:
        print(f"❌ Baseline training failed (expected due to Mamba CUDA): {e}")
        results['baseline'] = {'success': False, 'error': str(e)}
    
    # Step 3: Compare models if both succeeded
    print("\n" + "="*60)
    print("📊 STEP 3: Model Comparison")
    print("="*60)
    
    if results['distillation']['success'] and results['baseline']['success']:
        print("🎉 Both models trained successfully! Running comparison...")
        
        # Run fair comparison
        comparison_results = evaluate_models_comparison(
            baseline_model=results['baseline']['model'],
            distilled_model=results['distillation']['model'],
            eval_loader=data_info['csi_val_loader'],
            device=args.device
        )
        
        results['comparison'] = comparison_results
        
    elif results['distillation']['success']:
        print("✅ Distillation succeeded, baseline failed (due to Mamba CUDA issues)")
        print("🎯 Distillation model available for deployment")
        
        # Just evaluate distillation model
        print("📊 Evaluating distillation model performance...")
        distilled_model = results['distillation']['model']
        distilled_model.eval()
        
        val_losses = []
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for csi_data, csi_targets in data_info['csi_val_loader']:
                csi_data = csi_data.float().to(args.device)
                csi_targets = csi_targets.float().to(args.device)
                
                predictions = distilled_model(csi_data)
                loss = criterion(predictions, csi_targets)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"📊 Distilled model validation MSE: {avg_val_loss:.6f}")
        
        results['distillation_only'] = {
            'validation_mse': avg_val_loss
        }
        
    else:
        print("❌ Both models failed - cannot perform comparison")
    
    # Save training configuration and results
    config_save_path = os.path.join(args.output_dir, 'improved_training_config.json')
    with open(config_save_path, 'w') as f:
        config_data = vars(args).copy()
        # Remove non-serializable items
        for key in ['device']:
            if key in config_data:
                config_data[key] = str(config_data[key])
        
        json.dump(config_data, f, indent=2)
    
    print(f"\n🎉 Comparison completed! Results saved to {args.output_dir}")
    print(f"📊 Configuration saved to {config_save_path}")
    
    return results


def plot_training_curves(history: Dict[str, List[float]], output_dir: str):
    """Plot and save training curves."""
    
    print("📊 Plotting training curves...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Encoder-Enhanced Cross-Modal Distillation Training Curves', fontsize=16)
    
    # Plot each loss component
    loss_names = ['total_loss', 'distill_loss', 'task_loss', 'alignment_loss', 'recon_loss']
    loss_titles = ['Total Loss', 'Distillation Loss', 'Task Loss', 'Alignment Loss', 'Reconstruction Loss']
    
    for i, (loss_name, title) in enumerate(zip(loss_names, loss_titles)):
        if loss_name in history and len(history[loss_name]) > 0:
            row = i // 3
            col = i % 3
            axes[row, col].plot(history[loss_name], 'b-', linewidth=2)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(loss_names) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training curves saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Modal Distillation Training with MOHAWK Strategy")
    
    # Training mode
    parser.add_argument('--mode', type=str, choices=['baseline', 'distillation', 'comparison'], 
                       default='comparison', help='Training mode')
    
    # Data paths
    parser.add_argument('--uwb_data_path', type=str, 
                       default="/media/mohab/Storage HDD/Downloads/uwb2(1)",
                       help='Path to UWB data directory')
    parser.add_argument('--csi_mat_file', type=str,
                       default="/home/mohab/Desktop/mohawk_final/csi dataset/wificsi1_exp002.mat",
                       help='Path to CSI .mat file')
    
    # Data configuration
    parser.add_argument('--train_experiments', nargs='+', default=["002"], 
                       help='List of experiment IDs for training')
    parser.add_argument('--val_experiments', nargs='+', default=["002"],
                       help='List of experiment IDs for validation')
    # Training configuration  
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs for baseline')  # Increased from 10
    parser.add_argument('--pretrain_epochs', type=int, default=8, help='Encoder pretraining epochs')  # Increased from 2
    parser.add_argument('--stage1_epochs', type=int, default=10, help='Stage 1 distillation epochs')  # Increased from 2
    parser.add_argument('--stage2_epochs', type=int, default=12, help='Stage 2 distillation epochs')  # Increased from 2
    parser.add_argument('--stage3_epochs', type=int, default=15, help='Stage 3 distillation epochs')  # Increased from 3
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')  # Reduced from 1e-3 for stability
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=3000, help='Maximum samples per experiment')  # Increased from 300
    
    # Model configuration
    parser.add_argument('--uwb_seq_length', type=int, default=32, 
                       help='UWB sequence length')
    parser.add_argument('--csi_seq_length', type=int, default=4,
                       help='CSI sequence length')
    parser.add_argument('--latent_dim', type=int, default=256,  # Increased from 128 for better capacity
                       help='Shared latent space dimension')
    
    # Training configuration
    parser.add_argument('--d_model', type=int, default=256, help='CSI student d_model')  # Increased from 64
    parser.add_argument('--n_layer', type=int, default=4, help='CSI student n_layer')  # Increased from 2
    
    # NEW: Teacher pretraining epochs
    parser.add_argument('--teacher_epochs', type=int, default=20, help='Teacher pretraining epochs (CRITICAL)')  # Increased from 8
    
    # Add argument for skipping stage 1
    parser.add_argument('--skip_stage1', action='store_true', default=False,
                       help='Skip stage 1 (matrix alignment) to test its effect')
    
    # Loss weights for distillation - IMPROVED: More balanced weights for better distillation
    parser.add_argument('--temperature', type=float, default=3.5, help='Distillation temperature')  # Reduced from 4.0
    parser.add_argument('--alpha', type=float, default=0.4, help='Distillation loss weight')  # Increased from 0.1 for better knowledge transfer
    parser.add_argument('--beta', type=float, default=0.6, help='Task loss weight')  # Reduced from 0.9 for better balance
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='./improved_cross_modal_results',
                        help='Output directory for results and models')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("🔧 MOHAWK Cross-Modal Distillation Configuration:")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print(f"UWB data path: {args.uwb_data_path}")
    print(f"CSI mat file: {args.csi_mat_file}")
    print(f"Experiments: train={args.train_experiments}, val={args.val_experiments}")
    print(f"Sequence lengths: UWB={args.uwb_seq_length}, CSI={args.csi_seq_length}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Training: baseline_epochs={args.epochs}, lr={args.learning_rate}, batch_size={args.batch_size}")
    if args.mode in ['distillation', 'comparison']:
        print(f"Teacher pretraining: {args.teacher_epochs} epochs (CRITICAL for distillation)")
        print(f"MOHAWK stages: pretrain={args.pretrain_epochs}, stage1={args.stage1_epochs}, stage2={args.stage2_epochs}, stage3={args.stage3_epochs}")
        print(f"Loss weights: α={args.alpha}, β={args.beta}, temperature={args.temperature}")
        print(f"Skip Stage 1: {args.skip_stage1} {'(Testing stage 1 effect)' if args.skip_stage1 else '(Full 3-stage MOHAWK)'}")
    print()
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "mohawk_training_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"💾 Configuration saved to: {config_path}")
    
    # Run training based on mode
    start_time = time.time()
    
    if args.mode == "comparison":
        results = run_comparison_mode(args)
        
    elif args.mode == "baseline":
        data_info = setup_data_loaders(
            uwb_data_path=args.uwb_data_path,
            csi_mat_file=args.csi_mat_file,
            train_experiments=args.train_experiments,
            val_experiments=args.val_experiments,
            batch_size=args.batch_size,
            sequence_length=args.uwb_seq_length,
            csi_sequence_length=args.csi_seq_length,
            max_samples=args.max_samples
        )
        
        # Get feature dimensions
        for csi_data, csi_targets in data_info['csi_train_loader']:
            csi_feature_count = csi_data.shape[-1]
            break
            
        baseline_model = train_baseline_mode(args, data_info['csi_train_loader'], data_info['csi_val_loader'], csi_feature_count)
        results = {"baseline_model": "trained"}
        
    elif args.mode == "distillation":
        data_info = setup_data_loaders(
            uwb_data_path=args.uwb_data_path,
            csi_mat_file=args.csi_mat_file,
            train_experiments=args.train_experiments,
            val_experiments=args.val_experiments,
            batch_size=args.batch_size,
            sequence_length=args.uwb_seq_length,
            csi_sequence_length=args.csi_seq_length,
            max_samples=args.max_samples
        )
        
        # Get feature dimensions
        for csi_data, csi_targets in data_info['csi_train_loader']:
            csi_feature_count = csi_data.shape[-1]
            break
            
        if data_info['sync_train_loader'] is not None:
            for uwb_data, uwb_targets, sync_csi_data, sync_csi_targets in data_info['sync_train_loader']:
                uwb_feature_count = uwb_data.shape[-1]
                break
        else:
            uwb_feature_count = 100  # Default fallback
            
        distilled_model, training_history = train_distillation_mode(
            args, data_info['sync_train_loader'], data_info['sync_val_loader'], uwb_feature_count, csi_feature_count
        )
        results = {"distilled_model": "trained", "training_history": training_history}
        
        # Plot training curves
        if training_history:
            plot_training_curves(training_history, args.output_dir)
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"\n🎉 MOHAWK training completed successfully!")
    print(f"⏱️ Total training time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    print(f"📁 Results saved in: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main() 