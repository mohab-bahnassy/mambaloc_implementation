import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load results
with open('csi_transformer_gmm_results.json') as f:
    csi_transformer = json.load(f)
with open('custom_parameter_comparison_results.json') as f:
    mamba = json.load(f)

# Helper to extract percentiles and build CDF points
percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
def get_cdf_points(percentiles_dict):
    x = [percentiles_dict[f'p{p}'] for p in percentiles]
    y = [p/100 for p in percentiles]
    # Add (0,0) and (max,1) for full CDF
    x = [0] + x + [max(x)*1.2]
    y = [0] + y + [1]
    return np.array(x), np.array(y)

# Extract CDF points
csi_x, csi_y = get_cdf_points(csi_transformer['summary']['error_percentiles'])
baseline_x, baseline_y = get_cdf_points(mamba['summary']['baseline']['error_percentiles'])
distilled_x, distilled_y = get_cdf_points(mamba['summary']['continuous_probability']['error_percentiles'])

# Interpolate for smooth curves (try cubic, fallback to quadratic, then linear)
def smooth_interp(x, y):
    try:
        return interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(0,1))
    except Exception:
        try:
            return interp1d(x, y, kind='quadratic', bounds_error=False, fill_value=(0,1))
        except Exception:
            return interp1d(x, y, kind='linear', bounds_error=False, fill_value=(0,1))

x_dense = np.linspace(0, max(csi_x[-1], baseline_x[-1], distilled_x[-1]), 1000)
csi_interp = smooth_interp(csi_x, csi_y)
baseline_interp = smooth_interp(baseline_x, baseline_y)
distilled_interp = smooth_interp(distilled_x, distilled_y)

# Plot
plt.figure(figsize=(10, 7))
plt.plot(x_dense, distilled_interp(x_dense), label='MambaLoc', color='red', linewidth=2)
plt.plot(x_dense, csi_interp(x_dense), label='TIPS (Transformer)', color='blue', linewidth=2)
plt.plot(x_dense, baseline_interp(x_dense), label='Baseline Mamba (no distillation)', color='green', linewidth=2)

plt.xlabel('Absolute Error (meters)', fontsize=20, fontweight='bold')
plt.ylabel('Cumulative Probability', fontsize=20, fontweight='bold')
plt.title('CDF of Absolute Error: TIPS (Transformer) vs Mamba Models', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=18, loc='lower right')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()

# Add median errors in a box just above the legend (bottom left, left-aligned)
csi_median = csi_transformer['summary']['error_percentiles']['p50']
baseline_median = mamba['summary']['baseline']['error_percentiles']['p50']
distilled_median = mamba['summary']['continuous_probability']['error_percentiles']['p50']

# Use bold for the title
textstr = (
    f"$\\bf{{Median\\ Absolute\\ Errors\\ (meters):}}$\n"
    f"MambaLoc: {distilled_median:.3f}\n"
    f"TIPS (Transformer): {csi_median:.3f}\n"
    f"Baseline Mamba (no distillation): {baseline_median:.3f}"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.gca().text(0.42, 0.26, textstr, fontsize=18, verticalalignment='bottom', horizontalalignment='right',
               transform=plt.gca().transAxes, bbox=props, linespacing=1.3, ha='left')

plt.savefig('cdf_comparison_csi_transformer_vs_mamba.png', dpi=300)
plt.show() 