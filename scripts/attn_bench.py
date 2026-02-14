#!/usr/bin/env python3
import subprocess
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Test configurations - (batch, len_q, len_kv, dim)
# Keep len_q == len_kv for simplicity unless you need asymmetry
ATTENTION_SIZES = [
    (4,   64,   64, 64),
    (4,  128,  128, 64),
    (4,  256,  256, 64),
    (4,  512,  512, 64),
    (4, 1024, 1024, 64),
    (4, 2048, 2048, 64),
]

# Define kernels: (kernel_name, display_name, marker_style)
KERNELS = [
    ("native", "Naive", 'o-'),
    ("cublas", "cuBLAS", 'v-'),
    ("flash_v1", "Flash v1", 's-'),
    ("flash_v2", "Flash v2", '*-'),
]

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

EXECUTABLE = os.path.join(root_dir, "bin", "test_sdpa_attn")
SAVE = os.path.join(root_dir, "performance", "test_sdpa_attn")
RESULTS = {}  # kernel_name -> list of GFLOPS
os.makedirs(SAVE, exist_ok=True)

def run_test(b, l_q, l_kv, d, kernel):
    """Run a single attention test"""
    cmd = [EXECUTABLE, str(b), str(l_q), str(l_kv), str(d), f"--launch_func={kernel}"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "Kernel execution speed:" in line:
                    gflops = float(line.split(':')[1].strip().replace('GFLOPS', ''))
                    return gflops
    except Exception as e:
        print(f"Error running {kernel} on ({b},{l_q},{l_kv},{d}): {e}")
    
    return None

def run_benchmark():
    print("=" * 70)
    print("Attention Kernel Performance Benchmark")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    # Initialize results dict
    for kernel_name, _, _ in KERNELS:
        RESULTS[kernel_name] = []

    for config in ATTENTION_SIZES:
        b, l_q, l_kv, d = config
        print(f"\n📊 Testing: batch={b}, len_q={l_q}, len_kv={l_kv}, dim={d}")
        print("-" * 50)

        for kernel_name, display_name, marker in KERNELS:
            print(f"  {display_name:12}... ", end="", flush=True)
            gflops = run_test(b, l_q, l_kv, d, kernel_name)
            if gflops is not None:
                RESULTS[kernel_name].append(gflops)
                print(f"{gflops:8.2f} GFLOPS")
            else:
                RESULTS[kernel_name].append(None)
                print("Failed")

    save_results()
    plot_results()

def save_results():
    with open(os.path.join(SAVE, "result.csv"), 'w') as f:
        header = "Batch,LenQ,LenKV,Dim"
        for _, display_name, _ in KERNELS:
            header += f",{display_name}"
        f.write(header + "\n")

        for i, (b, l_q, l_kv, d) in enumerate(ATTENTION_SIZES):
            row = [str(b), str(l_q), str(l_kv), str(d)]
            for kernel_name, _, _ in KERNELS:
                val = RESULTS[kernel_name][i]
                row.append(f"{val:.2f}" if val else "N/A")
            f.write(",".join(row) + "\n")

    print(f"\n✅ Results saved to attn_bench_results.csv")

def plot_results():
    plt.figure(figsize=(12, 8))

    # Use sequence length as x-axis (assuming len_q == len_kv)
    x_vals = [cfg[1] for cfg in ATTENTION_SIZES]  # len_q

    for kernel_name, display_name, marker in KERNELS:
        y_vals = []
        x_plot = []
        for i, gflops in enumerate(RESULTS[kernel_name]):
            if gflops is not None:
                y_vals.append(gflops)
                x_plot.append(x_vals[i])
        if y_vals:
            plt.plot(x_plot, y_vals, marker, linewidth=2, markersize=6,
                     label=display_name, markevery=1)

    plt.xlabel('Sequence Length (len_q = len_kv)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.title('Attention Kernel Performance Comparison\n(Higher is Better)', 
              fontsize=14, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(x_vals, [str(x) for x in x_vals])
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left', fontsize=10)

    # Annotate large points
    for kernel_name, _, _ in KERNELS:
        for i, (b, l_q, l_kv, d) in enumerate(ATTENTION_SIZES):
            if RESULTS[kernel_name][i] is not None and l_q >= 512:
                plt.annotate(f'{RESULTS[kernel_name][i]:.0f}',
                             (l_q, RESULTS[kernel_name][i]),
                             textcoords="offset points",
                             xytext=(0,10),
                             ha='center',
                             fontsize=8,
                             alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE, "result.png"), dpi=150, bbox_inches='tight')
    print("✅ Performance chart saved to attn_benchmark_gflops.png")
    plt.show()

def print_summary():
    print("\n" + "=" * 80)
    print("Attention Performance Summary (GFLOPS)")
    print("=" * 80)
    
    header = "{:<6} {:<6} {:<6} {:<6} |".format("B", "Lq", "Lkv", "D")
    for _, name, _ in KERNELS:
        header += f" {name:>10} |"
    print(header)
    print("-" * len(header))

    for i, (b, l_q, l_kv, d) in enumerate(ATTENTION_SIZES):
        row = f"{b:<6} {l_q:<6} {l_kv:<6} {d:<6} |"
        for kernel_name, _, _ in KERNELS:
            val = RESULTS[kernel_name][i]
            if val:
                row += f" {val:10.2f} |"
            else:
                row += f" {'N/A':>10} |"
        print(row)
    
    print("=" * 80)

if __name__ == "__main__":
    if not os.path.exists(EXECUTABLE):
        print(f"❌ Error: Executable '{EXECUTABLE}' not found")
        print("Please compile the program first (e.g., make test_attention)")
        sys.exit(1)

    run_benchmark()
    print_summary()