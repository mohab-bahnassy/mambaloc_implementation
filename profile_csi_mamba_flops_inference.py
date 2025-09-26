import json
import torch
import time
from ptflops import get_model_complexity_info
from truly_fair_comparison import ContinuousCSIRegressionModel

# Load config
with open('csi_config.json') as f:
    csi_config = json.load(f)

input_dim = csi_config['CSIRegressionModel']['input']['input_features']
seq_len = 4  # typical sequence length
batch_size = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate model
model = ContinuousCSIRegressionModel(csi_config, device=device).to(device)
model.eval()

# Dummy input
dummy_input = torch.randn(batch_size, seq_len, input_dim).to(device)

# ptflops expects a tuple for input size (excluding batch)
def flops_forward_hook(input_res, model):
    return model(dummy_input)

with torch.no_grad():
    macs, params = get_model_complexity_info(
        model,
        (seq_len, input_dim),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )

# Inference time
n_runs = 100
with torch.no_grad():
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    start = time.time()
    for _ in range(n_runs):
        _ = model(dummy_input)
    end = time.time()
    avg_time_ms = (end - start) / n_runs * 1000

print(f"\n--- CSI Mamba Model Profiling ---")
print(f"Input shape: (batch={batch_size}, seq_len={seq_len}, input_dim={input_dim})")
print(f"Parameters: {params:,}")
print(f"FLOPs (per forward): {macs * 2:,}")
print(f"Average inference time (ms): {avg_time_ms:.3f}") 