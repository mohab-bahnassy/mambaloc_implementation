import torch
import torch.nn as nn
from mamba_ssm import Mamba
import time
from torch.profiler import profile, ProfilerActivity

# Constants
BATCH_SIZE = 1
SEQ_LEN = 128
INPUT_DIM = 280  # From csi_config.json
NUM_WARMUP = 10
NUM_RUNS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # Can be changed to torch.float16 for FP16 or torch.bfloat16 for BF16

class MambaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(
                d_model=INPUT_DIM,
                d_state=16,  # From config
                d_conv=4,    # From config
                expand=2     # From config
            ) for _ in range(3)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=INPUT_DIM,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, x):
        return self.transformer(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model, input_tensor):
    model.eval()
    
    # Warmup runs
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Timed runs
    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    return avg_time

def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name()}")
    print(f"Precision: {DTYPE}")
    print("-" * 50)
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Input Dimension: {INPUT_DIM}")
    print("-" * 50)
    
    # Create input tensor with specified precision
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, dtype=DTYPE).to(DEVICE)
    
    # Initialize models with specified precision
    mamba_model = MambaModel().to(DEVICE).to(dtype=DTYPE)
    transformer_model = TransformerModel().to(DEVICE).to(dtype=DTYPE)
    
    models = {
        "Mamba": mamba_model,
        "Transformer": transformer_model
    }
    
    for name, model in models.items():
        print(f"\nBenchmarking {name} Model:")
        print(f"Parameters: {count_parameters(model):,}")
        
        # Profile FLOPs
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            profile_memory=True,
        ) as prof:
            with torch.no_grad():
                _ = model(input_tensor)
        
        flops = sum(e.flops for e in prof.key_averages())
        print(f"FLOPs per forward pass: {flops:,}")
        
        # Measure inference time
        avg_time = benchmark_model(model, input_tensor)
        print(f"Average inference time: {avg_time*1000:.2f} ms")

if __name__ == "__main__":
    main()
