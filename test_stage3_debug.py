#!/usr/bin/env python3
"""
Stage 3 Debugging Test Script
Tests the encoder-enhanced cross-modal distillation Stage 3 specifically
to identify and fix the high task loss and diverging distillation loss issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

# Add current directory to path for imports
sys.path.append('.')

from encoder_enhanced_cross_modal_distillation import EncoderEnhancedMOHAWKDistiller
from synchronized_cross_modality_distillation import SimpleUWBTransformerTeacher
from torch.utils.data import DataLoader, TensorDataset


class MockDataLoader:
    """Mock data loader that generates realistic data for testing."""
    
    def __init__(self, batch_size=4, num_batches=5, target_scale="normalized"):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.target_scale = target_scale
        self.current_batch = 0
        
    def __iter__(self):
        self.current_batch = 0
        return self
        
    def __len__(self):
        return self.num_batches
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
            
        # Generate realistic mock data
        uwb_data = torch.randn(self.batch_size, 32, 113) * 0.5  # UWB CIR data
        csi_data = torch.randn(self.batch_size, 4, 280) * 0.3   # CSI data
        
        # Generate targets with different scales to test normalization
        if self.target_scale == "normalized":
            # Properly normalized targets (mean≈0, std≈1)
            uwb_targets = torch.randn(self.batch_size, 2) * 1.0
            csi_targets = torch.randn(self.batch_size, 2) * 1.0
        elif self.target_scale == "large":
            # Large scale targets (simulating unnormalized coordinates)
            uwb_targets = torch.randn(self.batch_size, 2) * 100 + 50  # Range: [-50, 150]
            csi_targets = torch.randn(self.batch_size, 2) * 100 + 50
        elif self.target_scale == "pixel":
            # Pixel-scale targets (common in computer vision)
            uwb_targets = torch.randn(self.batch_size, 2) * 200 + 400  # Range: [200, 600]
            csi_targets = torch.randn(self.batch_size, 2) * 200 + 400
        else:  # "mixed"
            # Mixed scales to test robustness
            uwb_targets = torch.randn(self.batch_size, 2) * torch.randint(1, 100, (1,)).float()
            csi_targets = torch.randn(self.batch_size, 2) * torch.randint(1, 100, (1,)).float()
        
        self.current_batch += 1
        return uwb_data, uwb_targets, csi_data, csi_targets


def create_test_models():
    """Create test teacher and student models."""
    print("🔧 Creating test models...")
    
    # Create teacher model - FIXED: Use correct parameter names
    teacher = SimpleUWBTransformerTeacher(
        input_features=113,
        output_features=2,
        d_model=256,
        n_layers=2,  # FIXED: Use n_layers instead of num_layers
        n_heads=8    # FIXED: Use n_heads instead of num_heads
    )
    
    # Create CSI student config
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
                "d_model": 64,
                "n_layer": 2,
                "final_prenorm": "layer"
            }
        }
    }
    
    return teacher, csi_config


def test_target_scales():
    """Test different target scales to identify normalization issues."""
    print("\n" + "="*60)
    print("🧪 TESTING DIFFERENT TARGET SCALES")
    print("="*60)
    
    teacher, csi_config = create_test_models()
    
    scales_to_test = ["normalized", "large", "pixel"]
    results = {}
    
    for scale in scales_to_test:
        print(f"\n📊 Testing target scale: {scale}")
        print("-" * 30)
        
        # Create distiller
        distiller = EncoderEnhancedMOHAWKDistiller(
            teacher_model=teacher,
            csi_student_config=csi_config,
            uwb_input_features=113,
            csi_input_features=280,
            temperature=4.0,
            alpha=0.7,
            beta=0.3,
            device="cpu"  # Use CPU for testing
        )
        
        # Create mock data loader
        mock_loader = MockDataLoader(batch_size=4, num_batches=3, target_scale=scale)
        
        # Test single distillation step
        batch = next(iter(mock_loader))
        uwb_data, uwb_targets, csi_data, csi_targets = batch
        
        print(f"   Target statistics:")
        print(f"     Mean: {csi_targets.mean().item():.4f}")
        print(f"     Std: {csi_targets.std().item():.4f}")
        print(f"     Range: [{csi_targets.min().item():.4f}, {csi_targets.max().item():.4f}]")
        
        # Run distillation step
        try:
            outputs = distiller.distillation_step(uwb_data, uwb_targets, csi_data, csi_targets)
            
            print(f"   Loss results:")
            print(f"     Total loss: {outputs['total_loss'].item():.6f}")
            print(f"     Distillation loss: {outputs['distill_loss'].item():.6f}")
            print(f"     Task loss: {outputs['task_loss'].item():.6f}")
            print(f"     Alignment loss: {outputs['alignment_loss'].item():.6f}")
            
            results[scale] = {
                'target_mean': csi_targets.mean().item(),
                'target_std': csi_targets.std().item(),
                'total_loss': outputs['total_loss'].item(),
                'task_loss': outputs['task_loss'].item(),
                'distill_loss': outputs['distill_loss'].item()
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[scale] = {'error': str(e)}
    
    # Analyze results
    print(f"\n📈 SCALE ANALYSIS RESULTS:")
    print("-" * 40)
    for scale, result in results.items():
        if 'error' not in result:
            task_loss = result['task_loss']
            target_scale_val = result['target_std']
            print(f"{scale:>10}: Task Loss = {task_loss:>10.2f}, Target Scale = {target_scale_val:>8.2f}")
        else:
            print(f"{scale:>10}: ERROR - {result['error']}")
    
    return results


def test_stage3_detailed():
    """Detailed test of Stage 3 with our fixes."""
    print("\n" + "="*60)
    print("🔍 DETAILED STAGE 3 DEBUGGING")
    print("="*60)
    
    teacher, csi_config = create_test_models()
    
    # Create distiller with our fixes
    distiller = EncoderEnhancedMOHAWKDistiller(
        teacher_model=teacher,
        csi_student_config=csi_config,
        uwb_input_features=113,
        csi_input_features=280,
        temperature=4.0,
        alpha=0.7,
        beta=0.3,
        device="cpu"
    )
    
    # Test with large-scale targets (common issue)
    mock_loader = MockDataLoader(batch_size=4, num_batches=2, target_scale="large")
    
    # Create optimizer
    all_trainable_params = []
    all_trainable_params.extend([p for p in distiller.student.parameters() if p.requires_grad])
    all_trainable_params.extend([p for p in distiller.teacher_to_student_projections.parameters() if p.requires_grad])
    all_trainable_params.extend([p for p in distiller.attention_to_mixer_projections.parameters() if p.requires_grad])
    
    optimizer = torch.optim.AdamW(all_trainable_params, lr=1e-4)  # Small LR for testing
    
    print("\n🏃 Running Stage 3 with detailed logging...")
    
    # Run our fixed stage3_distill
    losses = distiller.stage3_distill(
        train_loader=mock_loader,
        optimizer=optimizer,
        epochs=2  # Short test
    )
    
    print(f"\n📊 Stage 3 Results:")
    print(f"   Final average loss: {losses[-1]:.6f}")
    print(f"   Loss trend: {losses}")
    
    return losses


def test_temperature_adaptation():
    """Test the adaptive temperature adjustment feature."""
    print("\n" + "="*60)
    print("🌡️ TESTING ADAPTIVE TEMPERATURE")
    print("="*60)
    
    teacher, csi_config = create_test_models()
    
    # Test different scenarios
    scenarios = [
        ("small_predictions", 0.1),   # Small prediction scales
        ("large_predictions", 10.0),  # Large prediction scales
        ("normal_predictions", 1.0)   # Normal prediction scales
    ]
    
    for scenario_name, prediction_scale in scenarios:
        print(f"\n🧪 Testing scenario: {scenario_name} (scale: {prediction_scale})")
        print("-" * 40)
        
        # Create distiller
        distiller = EncoderEnhancedMOHAWKDistiller(
            teacher_model=teacher,
            csi_student_config=csi_config,
            uwb_input_features=113,
            csi_input_features=280,
            temperature=4.0,  # Initial temperature
            alpha=0.7,
            beta=0.3,
            device="cpu"
        )
        
        # Mock prediction scales by modifying model outputs temporarily
        original_temp = distiller.temperature
        print(f"   Initial temperature: {original_temp}")
        
        # Create data
        mock_loader = MockDataLoader(batch_size=4, num_batches=1, target_scale="normalized")
        batch = next(iter(mock_loader))
        uwb_data, uwb_targets, csi_data, csi_targets = batch
        
        # Manually set prediction scales for testing
        with torch.no_grad():
            # Modify targets to simulate different prediction scales
            if scenario_name == "large_predictions":
                csi_targets = csi_targets * 10.0  # Large scale
            elif scenario_name == "small_predictions":
                csi_targets = csi_targets * 0.1   # Small scale
        
        # Run distillation step (will trigger adaptive temperature)
        outputs = distiller.distillation_step(uwb_data, uwb_targets, csi_data, csi_targets)
        
        print(f"   Adjusted temperature: {distiller.temperature}")
        print(f"   Temperature change: {distiller.temperature - original_temp:+.2f}")
        print(f"   Task loss: {outputs['task_loss'].item():.6f}")


def test_loss_scaling_consistency():
    """Test that our loss scaling fixes work correctly."""
    print("\n" + "="*60)
    print("⚖️ TESTING LOSS SCALING CONSISTENCY")
    print("="*60)
    
    teacher, csi_config = create_test_models()
    
    # Test different temperatures
    temperatures = [1.0, 4.0, 10.0]
    
    for temp in temperatures:
        print(f"\n🌡️ Testing temperature: {temp}")
        print("-" * 30)
        
        distiller = EncoderEnhancedMOHAWKDistiller(
            teacher_model=teacher,
            csi_student_config=csi_config,
            temperature=temp,
            device="cpu"
        )
        
        # Create test data
        mock_loader = MockDataLoader(batch_size=4, num_batches=1, target_scale="large")
        batch = next(iter(mock_loader))
        uwb_data, uwb_targets, csi_data, csi_targets = batch
        
        outputs = distiller.distillation_step(uwb_data, uwb_targets, csi_data, csi_targets)
        
        # Check loss scaling
        distill_loss = outputs['distill_loss'].item()
        task_loss = outputs['task_loss'].item()
        
        print(f"   Distillation loss: {distill_loss:.6f}")
        print(f"   Task loss (scaled): {task_loss:.6f}")
        print(f"   Loss ratio: {distill_loss/task_loss:.2f}")


def visualize_results(scale_results):
    """Create visualizations of the test results."""
    print("\n" + "="*60)
    print("📊 VISUALIZING RESULTS")
    print("="*60)
    
    # Extract data for plotting
    scales = []
    task_losses = []
    target_scales = []
    
    for scale, result in scale_results.items():
        if 'error' not in result:
            scales.append(scale)
            task_losses.append(result['task_loss'])
            target_scales.append(result['target_std'])
    
    if len(scales) > 0:
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Task Loss vs Target Scale
        ax1.bar(scales, task_losses, color=['green', 'orange', 'red'])
        ax1.set_ylabel('Task Loss')
        ax1.set_title('Task Loss by Target Scale')
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Plot 2: Target Standard Deviation
        ax2.bar(scales, target_scales, color=['green', 'orange', 'red'])
        ax2.set_ylabel('Target Standard Deviation')
        ax2.set_title('Target Scale by Test Type')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('stage3_debug_results.png', dpi=150, bbox_inches='tight')
        print("📈 Results saved to: stage3_debug_results.png")
    else:
        print("❌ No valid results to plot")


def main():
    """Run all debugging tests."""
    print("🚀 STAGE 3 DEBUGGING TEST SUITE")
    print("="*60)
    print("This script will test the Stage 3 fixes and identify issues.")
    print("")
    
    try:
        # Test 1: Different target scales
        scale_results = test_target_scales()
        
        # Test 2: Detailed Stage 3 debugging
        stage3_losses = test_stage3_detailed()
        
        # Test 3: Temperature adaptation
        test_temperature_adaptation()
        
        # Test 4: Loss scaling consistency
        test_loss_scaling_consistency()
        
        # Test 5: Visualize results
        try:
            visualize_results(scale_results)
        except ImportError:
            print("📊 Matplotlib not available, skipping visualization")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
        # Summary
        print("\n📋 SUMMARY:")
        print("-" * 20)
        
        # Analyze scale results
        if scale_results:
            print("🎯 Target Scale Analysis:")
            for scale, result in scale_results.items():
                if 'error' not in result:
                    task_loss = result['task_loss']
                    if task_loss > 1000:
                        status = "❌ PROBLEMATIC"
                    elif task_loss > 100:
                        status = "⚠️ HIGH"
                    else:
                        status = "✅ GOOD"
                    print(f"   {scale}: {status} (Task Loss: {task_loss:.2f})")
        
        # Analyze Stage 3 trend
        if stage3_losses:
            if len(stage3_losses) > 1:
                if stage3_losses[-1] < stage3_losses[0]:
                    print("📉 Stage 3: ✅ Loss is decreasing (GOOD)")
                else:
                    print("📈 Stage 3: ⚠️ Loss is increasing (NEEDS ATTENTION)")
            print(f"   Final loss: {stage3_losses[-1]:.6f}")
        
        print("\n🎉 Debug test suite completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 