import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataloaders.csi_loader_fixed import create_csi_dataloaders_fixed

# GMM Head from truly_fair_comparison.py
COORD_MIN = 0.0
COORD_MAX_X = 8.0
COORD_MAX_Y = 11.0
NUM_GAUSSIANS = 16
GAUSSIAN_GRID_X = 4
GAUSSIAN_GRID_Y = 4

class ContinuousCoordinateHead(nn.Module):
    def __init__(self, input_dim: int, num_gaussians: int = NUM_GAUSSIANS):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.gaussian_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_gaussians * 5)
        )
        self._init_gaussian_centers()
    def _init_gaussian_centers(self):
        x_centers = torch.linspace(COORD_MIN + 1.0, COORD_MAX_X - 1.0, GAUSSIAN_GRID_X)
        y_centers = torch.linspace(COORD_MIN + 1.0, COORD_MAX_Y - 1.0, GAUSSIAN_GRID_Y)
        centers_x, centers_y = torch.meshgrid(x_centers, y_centers, indexing='ij')
        self.register_buffer('gaussian_centers_x', centers_x.flatten())
        self.register_buffer('gaussian_centers_y', centers_y.flatten())
    def forward(self, features: torch.Tensor):
        if features.dim() == 3:
            features = features[:, -1, :]
        gaussian_params = self.gaussian_predictor(features)
        gaussian_params = gaussian_params.view(-1, self.num_gaussians, 5)
        weights = F.softmax(gaussian_params[:, :, 0], dim=-1)
        mu_x = gaussian_params[:, :, 1]
        mu_y = gaussian_params[:, :, 2]
        sigma_x = F.softplus(gaussian_params[:, :, 3]) + 0.1
        sigma_y = F.softplus(gaussian_params[:, :, 4]) + 0.1
        expected_x = torch.sum(weights * mu_x, dim=1)
        expected_y = torch.sum(weights * mu_y, dim=1)
        coordinates = torch.stack([expected_x, expected_y], dim=-1)
        logits = torch.log(weights + 1e-8)
        return {
            'logits': logits,
            'probabilities': weights,
            'coordinates': coordinates,
            'predictions': coordinates,
            'gaussian_params': {
                'weights': weights,
                'mu_x': mu_x,
                'mu_y': mu_y,
                'sigma_x': sigma_x,
                'sigma_y': sigma_y
            }
        }

# Transformer backbone for CSI
class CSILocalizationTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_layers=3, n_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.gmm_head = ContinuousCoordinateHead(d_model)
    def forward(self, x):
        x = self.input_proj(x)
        features = self.transformer(x)
        return self.gmm_head(features)

def main():
    # Hyperparameters
    batch_size = 32
    sequence_length = 4
    epochs = 20
    d_model = 128
    n_layers = 3
    n_heads = 4
    dropout = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data paths (update as needed)
    csi_mat_file = "csi dataset/wificsi1_exp002.mat"

    # Load data
    train_loader, val_loader, feature_scaler, target_scaler = create_csi_dataloaders_fixed(
        mat_file_path=csi_mat_file,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_samples=None
    )
    input_dim = next(iter(train_loader))[0].shape[-1]

    # Model
    model = CSILocalizationTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        losses = []
        for csi_data, csi_targets in train_loader:
            csi_data = csi_data.to(device)
            csi_targets = csi_targets.to(device)
            optimizer.zero_grad()
            outputs = model(csi_data)
            predictions = outputs['coordinates']
            if csi_targets.dim() > 2:
                csi_targets = csi_targets[:, -1, :]
            loss = criterion(predictions, csi_targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {np.mean(losses):.6f}")
    print("Training complete.")

    # Evaluation and metrics
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for csi_data, csi_targets in val_loader:
            csi_data = csi_data.to(device)
            csi_targets = csi_targets.to(device)
            outputs = model(csi_data)
            predictions = outputs['coordinates']
            if csi_targets.dim() > 2:
                csi_targets = csi_targets[:, -1, :]
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(csi_targets.cpu().numpy())
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize
    all_predictions_real = target_scaler.inverse_transform(all_predictions)
    all_targets_real = target_scaler.inverse_transform(all_targets)

    # Metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    sample_errors = np.sqrt(np.sum((all_predictions_real - all_targets_real) ** 2, axis=1))
    x_errors = np.abs(all_predictions_real[:, 0] - all_targets_real[:, 0])
    y_errors = np.abs(all_predictions_real[:, 1] - all_targets_real[:, 1])
    r2_euclidean = r2_score(all_targets_real.flatten(), all_predictions_real.flatten())
    r2_x = r2_score(all_targets_real[:, 0], all_predictions_real[:, 0])
    r2_y = r2_score(all_targets_real[:, 1], all_predictions_real[:, 1])
    rmse = float(np.sqrt(mean_squared_error(all_targets_real, all_predictions_real)))
    mae = float(mean_absolute_error(all_targets_real, all_predictions_real))
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    error_percentiles = {f'p{p}': float(np.percentile(sample_errors, p)) for p in percentiles}
    median_ae = float(np.median(sample_errors))
    mean_ae = float(np.mean(sample_errors))
    std_ae = float(np.std(sample_errors))
    min_error = float(np.min(sample_errors))
    max_error = float(np.max(sample_errors))
    x_stats = {
        'median': float(np.median(x_errors)),
        'mean': float(np.mean(x_errors)),
        'std': float(np.std(x_errors)),
        'min': float(np.min(x_errors)),
        'max': float(np.max(x_errors))
    }
    y_stats = {
        'median': float(np.median(y_errors)),
        'mean': float(np.mean(y_errors)),
        'std': float(np.std(y_errors)),
        'min': float(np.min(y_errors)),
        'max': float(np.max(y_errors))
    }
    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory_mb = total_params * 4 / (1024 * 1024)
    model_size_info = {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'non_trainable_parameters': int(total_params - trainable_params),
        'estimated_memory_mb': float(param_memory_mb)
    }
    # Save results
    import json
    results = {
        'summary': {
            'median_absolute_error': median_ae,
            'mean_absolute_error': mean_ae,
            'std_absolute_error': std_ae,
            'min_error': min_error,
            'max_error': max_error,
            'rmse': rmse,
            'mae': mae,
            'r2_overall': float(r2_euclidean),
            'r2_x_coordinate': float(r2_x),
            'r2_y_coordinate': float(r2_y),
            'error_percentiles': error_percentiles,
            'x_coordinate_errors': x_stats,
            'y_coordinate_errors': y_stats,
            'total_samples': int(len(sample_errors)),
            'coordinate_space': 'real_meters'
        },
        'model_info': model_size_info
    }
    with open('csi_transformer_gmm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to csi_transformer_gmm_results.json")
    
if __name__ == "__main__":
    main()
