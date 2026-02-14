#!/usr/bin/env python3
import subprocess
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# Convolution parameters: (n, c, h, w, k, r, s, u, v, p, q, description)
CONV_CONFIGS = [
    (64, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, "64x256x14x14/256"),
    (256, 192, 14, 14, 192, 3, 3, 1, 1, 1, 1, "256x192x14x14/192"),
    (16, 256, 26, 26, 512, 3, 3, 1, 1, 1, 1, "16x256x26x26/512"),
    (32, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, "32x256x14x14/256"),
    (2, 1280, 16, 16, 1280, 3, 3, 1, 1, 1, 1, "2x1280x16x16/1280"),
    (2, 960, 64, 64, 32, 3, 3, 1, 1, 1, 1, "2x960x64x64/32"),
]

# Kernels: (kernel_name, display_name, marker_style)
KERNELS = [
    ("native",    "Native",       'o-'),
    ("implgemm",  "Implicit GEMM",'s-'),
    ("winograd",  "Winograd",     '^-'),
]

# Paths (adjust if your executable is elsewhere)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir) 
EXECUTABLE = os.path.join(root_dir, "bin", "test_conv2d") 
SAVE_DIR = os.path.join(root_dir, "performance", "test_conv2d")
os.makedirs(SAVE_DIR, exist_ok=True)


def run_test(config, kernel_name):
    """Run a single convolution test and return GFLOPS."""
    n, c, h, w, k, r, s, u, v, p, q, _ = config
    cmd = [EXECUTABLE, str(n), str(c), str(h), str(w), str(k),
           str(r), str(s), str(u), str(v), str(p), str(q),
           f"--launch_func={kernel_name}", "--iter_num=10"]   # use 10 iterations as default

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "Kernel execution speed:" in line:
                    gflops = float(line.split(':')[1].strip().replace('GFLOPS', ''))
                    return gflops
        else:
            print(f"    Error (return code {result.returncode})")
            if result.stderr:
                print("    stderr:", result.stderr.strip())
    except subprocess.TimeoutExpired:
        print("    Timeout after 300s")
    except Exception as e:
        print(f"    Exception: {e}")
    return None

def run_benchmark():
    """Run all configurations and kernels."""
    print("=" * 70)
    print("Convolution Performance Benchmark (3x3, stride=1, padding=1)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Results dict: RESULTS[kernel_name][config_index] = GFLOPS
    RESULTS = {k[0]: [] for k in KERNELS}

    for idx, config in enumerate(CONV_CONFIGS):
        desc = config[-1]
        print(f"\n📊 Config {idx+1}: {desc}")
        print("-" * 50)

        for kernel_name, display_name, _ in KERNELS:
            print(f"  {display_name:15}... ", end="", flush=True)
            gflops = run_test(config, kernel_name)
            if gflops:
                RESULTS[kernel_name].append(gflops)
                print(f"{gflops:8.2f} GFLOPS")
            else:
                RESULTS[kernel_name].append(None)
                print("Failed")

    return RESULTS

def save_results(RESULTS):
    """Save results to CSV."""
    csv_path = os.path.join(SAVE_DIR, "result.csv")
    with open(csv_path, 'w') as f:
        # Header
        header = "Config,"
        header += ",".join([name for _, name, _ in KERNELS])
        f.write(header + "\n")

        # Data rows
        for i, config in enumerate(CONV_CONFIGS):
            desc = config[-1]
            row = [desc]
            for kernel_name, _, _ in KERNELS:
                val = RESULTS[kernel_name][i]
                row.append(f"{val:.2f}" if val else "N/A")
            f.write(",".join(row) + "\n")

    print(f"\n✅ Results saved to {csv_path}")

def plot_results(RESULTS):
    """Create grouped bar chart of GFLOPS per config for each kernel."""
    n_configs = len(CONV_CONFIGS)
    x = np.arange(n_configs)               # config indices
    width = 0.25                            # bar width
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 6))

    for kernel_name, display_name, _ in KERNELS:
        values = RESULTS[kernel_name]
        # Convert None to 0 for plotting (or skip)
        plot_vals = [v if v is not None else 0 for v in values]
        offset = width * multiplier
        bars = ax.bar(x + offset, plot_vals, width, label=display_name)
        # Add value labels on bars (only if >0)
        for bar, val in zip(bars, plot_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        multiplier += 1

    # Labels and titles
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_title('Convolution Kernel Performance Comparison\n(Higher is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([c[-1] for c in CONV_CONFIGS], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    png_path = os.path.join(SAVE_DIR, "result.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✅ Chart saved to {png_path}")
    plt.show()

def print_summary(RESULTS):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("Performance Summary (GFLOPS)")
    print("=" * 70)

    # Build header
    header = f"{'Config':<20} |"
    for _, display_name, _ in KERNELS:
        header += f" {display_name:>12} |"
    print(header)
    print("-" * len(header))

    # Rows
    for i, config in enumerate(CONV_CONFIGS):
        desc = config[-1][:18] + ".." if len(config[-1]) > 18 else config[-1]
        row = f"{desc:<20} |"
        for kernel_name, _, _ in KERNELS:
            val = RESULTS[kernel_name][i]
            if val:
                row += f" {val:12.2f} |"
            else:
                row += f" {'N/A':>12} |"
        print(row)

    print("=" * 70)


if __name__ == "__main__":
    if not os.path.exists(EXECUTABLE):
        print(f"❌ Error: Executable '{EXECUTABLE}' not found.")
        print("Please compile the convolution test program and ensure it is placed at:")
        print(f"   {EXECUTABLE}")
        print("You can adjust the EXECUTABLE variable in the script if needed.")
        sys.exit(1)

    RESULTS = run_benchmark()
    save_results(RESULTS)
    plot_results(RESULTS)
    print_summary(RESULTS)