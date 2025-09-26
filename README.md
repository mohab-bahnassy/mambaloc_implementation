# MambaLoc: Lightweight Indoor Localization For Unmanned Vehicles Using Cross-Modal Knowledge Distillation

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.2.1-red.svg)

## 📄 Paper Information

This repository contains the implementation of the paper:

**"MambaLoc: Lightweight Indoor Localization For Unmanned Vehicles Using Cross-Modal Knowledge Distillation"**

- **Authors**: Mohab Bahnassy, Omar Saqr, Hamada Rizk, Moustafa Youssef
- **Conference**: ACM SIGSPATIAL 2025 Student Research Competition
- **Status**: Accepted
- **Conference Link**: [https://sigspatial2025.sigspatial.org/src-accepted/](https://sigspatial2025.sigspatial.org/src-accepted/)

## 🚧 Project Status

**Note**: This implementation is currently under improvement, with additional experiments being added continuously. The codebase represents ongoing research and development.

## 🏗️ Architecture Overview

MambaLoc leverages cross-modal knowledge distillation to train lightweight Mamba-based models for indoor localization using CSI (Channel State Information) data, guided by UWB (Ultra-Wideband) teacher models.

### Key Components:
- **Mamba Backbone**: Efficient state-space models for sequence processing
- **Cross-Modal Distillation**: Knowledge transfer from UWB teacher to CSI student
- **Gaussian Mixture Models**: Continuous probability distributions for coordinate prediction
- **Adaptive Scheduling**: Dynamic weight adjustment during training

## 📊 Dataset

This implementation utilizes the **OPERAnet** dataset, a comprehensive multimodal activity recognition dataset acquired from radio frequency and vision-based sensors. The dataset provides synchronized RF data including CSI, UWB signals, and vision-based measurements, making it ideal for cross-modal localization research.

**Dataset Reference:**
- **Paper**: "OPERAnet, a multimodal activity recognition dataset acquired from radio frequency and vision-based sensors"
- **Authors**: Bocus, M.J., Li, W., Vishwakarma, S. et al.
- **Journal**: Scientific Data 9, 474 (2022)
- **Link**: [https://www.nature.com/articles/s41597-022-01573-2](https://www.nature.com/articles/s41597-022-01573-2)

The dataset features:
- **~8 hours** of annotated measurements
- **Multi-modal data**: WiFi CSI, UWB, Passive WiFi Radar, and Kinect sensors
- **6 participants** performing 6 daily activities across 2 rooms
- **Synchronized measurements** enabling cross-modal learning approaches

## 📁 Project Structure

```
mambaloc_implementation/
├── truly_fair_comparison.py      # 🎯 MAIN ENTRY POINT - Fair comparison framework
├── modules/                      # Core model components
│   ├── csi_head.py              # CSI regression head
│   ├── uwb_backbone.py          # UWB mixer backbone
│   ├── cross_modal_encoders.py  # Cross-modal encoding layers
│   └── mixers/                  # Mamba and attention mixers
│       ├── discrete_mamba2.py   # Main Mamba implementation
│       ├── official_mamba.py    # Reference Mamba implementation
│       └── phi_attention.py     # Phi attention mechanism
├── dataloaders/                 # Data loading and preprocessing
│   ├── csi_loader_fixed.py     # CSI data loader
│   ├── uwb_opera_loader.py     # UWB data loader
│   └── synchronized_uwb_csi_loader_fixed.py  # Synchronized data loading
├── config files/                # Model and training configurations
│   ├── config.json
│   ├── csi_config.json
│   └── training_config.json
└── experiments/                 # Additional experimental scripts
    ├── demo_cross_modal_distillation.py
    ├── model_benchmarking.py
    ├── profile_csi_mamba_flops_inference.py
    └── various comparison scripts
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch 2.2.1+
- mamba-ssm 2.2.4
- numpy, pandas, scikit-learn
- matplotlib, seaborn (for visualization)

### Data Setup

1. **Download the OPERAnet dataset** from the [Nature Scientific Data paper](https://www.nature.com/articles/s41597-022-01573-2)
2. **Extract the dataset** to your preferred location
3. **Set up data paths** using the setup script:

```bash
python setup_data.py --uwb_path /path/to/uwb/data --csi_path /path/to/csi/data.mat
```

### Running the Experiments

**Quick Example (recommended for first run):**
```bash
python example_usage.py
```

**Full Comparison:**
```bash
python truly_fair_comparison.py
```

The main script will:
1. Load and preprocess CSI and UWB datasets
2. Train baseline Mamba model (CSI-only)
3. Train distilled Mamba model with UWB teacher guidance
4. Generate comprehensive evaluation metrics and CDF plots

### Configuration

**Step 1: Set up data paths**
```bash
python setup_data.py --uwb_path /path/to/your/uwb/dataset --csi_path /path/to/your/csi/dataset.mat
```

This will create a `config.json` file with your data paths and verify the data structure.

**Step 2: Run example (optional)**
```bash
python example_usage.py
```

**Step 3: Run full comparison**
```bash
python truly_fair_comparison.py
```

## 📊 Evaluation Metrics

The framework provides comprehensive evaluation including:
- **Median Absolute Error (MAE)**: Robust error measurement
- **R² Score**: Coefficient of determination
- **RMSE**: Root mean square error
- **CDF Plots**: Error distribution visualization
- **Model Size Analysis**: Parameter count and memory usage

## 🔬 Experimental Scripts

### Supporting Files (Helpers)

- `demo_cross_modal_distillation.py`: Basic cross-modal distillation demo
- `model_benchmarking.py`: Model performance benchmarking
- `profile_csi_mamba_flops_inference.py`: FLOPS profiling for inference
- `cpu_comparison_test.py`: CPU vs GPU performance comparison
- `fair_comparison_test.py`: Alternative fair comparison implementation
- `debug_scaling_investigation.py`: Scaling behavior analysis
- `cdf_comparison_csi_transformer_vs_mamba.py`: CDF comparison utilities
- Various ablation and configuration test scripts

### Data Analysis

- `analyze_datasets.py`: Dataset statistics and visualization
- `test_fixed_dataloader.py`: Data loader validation
- `simple_fix_test.py`: Quick functionality tests

## 🎯 Key Features

1. **Fair Comparison Framework**: Ensures identical data usage between baseline and distilled models
2. **Gaussian Mixture Models**: Continuous probability distributions for coordinate prediction
3. **Adaptive Scheduling**: Dynamic adjustment of distillation vs task loss weights
4. **Comprehensive Metrics**: Multiple evaluation metrics with statistical significance
5. **Real Coordinate Space**: Denormalized evaluation in actual meters

## 📚 Acknowledgments

This implementation incorporates and adapts components from:

> **"Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models"**  
> Authors: Aviv Bick, Kevin Y. Li, Eric P. Xing, J. Zico Kolter, Albert Gu  
> *Used for Mamba backbone implementation*

## 🛠️ Technical Details

### Model Architecture
- **Backbone**: Mamba-based state-space models (discrete_mamba2)
- **Input Features**: CSI magnitude and phase components
- **Output**: 2D coordinates (x, y) with Gaussian mixture distributions
- **Knowledge Distillation**: KL divergence on probability distributions

### Training Configuration
- **Temperature Scaling**: T=15.0 for knowledge distillation
- **Loss Weighting**: α=0.2 (distillation), β=0.8 (task)
- **Adaptive Scheduling**: Dynamic weight adjustment based on loss trends
- **Optimization**: AdamW with CosineAnnealingLR

## 📈 Results

The framework generates:
- Detailed JSON results with all metrics
- CDF plots comparing error distributions
- Model size and complexity analysis
- Statistical significance tests

## 🤝 Contributing

This is an active research project. Contributions, suggestions, and discussions are welcome!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

- **Mohab Bahnassy**: mohabbahnassy@aucegypt.edu

For questions about the implementation or research, please open an issue or contact the authors.

---

**Note**: This implementation represents ongoing research. Some experimental features may be under development or require additional testing.
