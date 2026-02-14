#!/usr/bin/env python3
import subprocess
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Test configuration - exponentially growing matrix sizes
MATRIX_SIZES = [128, 256, 512, 1024, 2048, 4096]

# Define kernels: (kernel_name, display_name, marker_style)
KERNELS = [
    ("native", "Naive", 'o-'),
    ("tiled_v1", "Tiled v1", 's-'),
    ("tiled_v2", "Tiled v2", '^-'),
    ("tiled_v3", "Tiled v3", 'd-'),
    ("tiled_dbuf", "Tiled + DBuf", '*-'),
    ("mma", "Tensor Core", 'x-'),
    ("cublas", "cuBLAS", 'v-')
]

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

EXECUTABLE = os.path.join(root_dir, "bin", "test_matmul")
SAVE = os.path.join(root_dir, "performance", "test_matmul")
RESULTS = {}  # Using kernel_name as key
os.makedirs(SAVE, exist_ok=True)

def run_test(size, kernel):
    """Run a single test"""
    cmd = [EXECUTABLE, str(size), str(size), str(size), f"--launch_func={kernel}"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Parse GFLOPS
            for line in result.stdout.split('\n'):
                if "Kernel execution speed:" in line:
                    gflops = float(line.split(':')[1].strip().replace('GFLOPS', ''))
                    return gflops
    except Exception as e:
        print(f"Error: {e}")
    
    return None

def run_benchmark():
    """Run all tests"""
    print("=" * 60)
    print("Matrix Multiplication Performance (m=n=k)")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Initialize results dictionary - using kernel_name as key
    for kernel_name, display_name, _ in KERNELS:
        RESULTS[kernel_name] = []
    
    # Run tests
    for size in MATRIX_SIZES:
        print(f"\n📊 Testing size: {size} x {size} x {size}")
        print("-" * 40)
        
        for kernel_name, display_name, marker in KERNELS:
            print(f"  {display_name:12}... ", end="", flush=True)
            
            gflops = run_test(size, kernel_name)
            
            if gflops:
                RESULTS[kernel_name].append(gflops)
                print(f"{gflops:8.2f} GFLOPS")
            else:
                RESULTS[kernel_name].append(None)
                print("Failed")
    
    # Save results to file
    save_results()
    
    # Plot results
    plot_results()

def save_results():
    """Save results to CSV"""
    with open(os.path.join(SAVE, "result.csv"), 'w') as f:
        # Write header - using display names
        header = "Size"
        for _, display_name, _ in KERNELS:
            header += f",{display_name}"
        f.write(header + "\n")
        
        # Write data rows
        for i, size in enumerate(MATRIX_SIZES):
            row = [str(size)]
            for kernel_name, display_name, _ in KERNELS:
                val = RESULTS[kernel_name][i]  # Using kernel_name as key
                row.append(f"{val:.2f}" if val else "N/A")
            f.write(",".join(row) + "\n")
    
    print(f"\n✅ Results saved to bench_results.csv")

def plot_results():
    """Plot GFLOPS vs Size curves"""
    plt.figure(figsize=(12, 8))
    
    # Plot each curve
    for kernel_name, display_name, marker in KERNELS:
        # Filter out None values
        valid_sizes = []
        valid_gflops = []
        
        for i, size in enumerate(MATRIX_SIZES):
            if RESULTS[kernel_name][i] is not None:
                valid_sizes.append(size)
                valid_gflops.append(RESULTS[kernel_name][i])
        
        if valid_gflops:
            plt.plot(valid_sizes, valid_gflops, marker, linewidth=2, 
                    markersize=6, label=display_name, markevery=1)
    
    # Set plot properties
    plt.xlabel('Matrix Size (m = n = k)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.title('CUDA Matrix Multiplication Performance Comparison\n(Higher is Better)', 
              fontsize=14, fontweight='bold')
    
    # Set x-axis to log scale
    plt.xscale('log', base=2)
    
    # Set x-axis ticks
    plt.xticks(MATRIX_SIZES, [str(s) for s in MATRIX_SIZES])
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    plt.legend(loc='upper left', fontsize=10)
    
    # Add value labels for large sizes
    for kernel_name, display_name, marker in KERNELS:
        for i, size in enumerate(MATRIX_SIZES):
            if RESULTS[kernel_name][i] is not None and size >= 1024:
                plt.annotate(f'{RESULTS[kernel_name][i]:.0f}', 
                           (size, RESULTS[kernel_name][i]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=8,
                           alpha=0.7)
    
    plt.tight_layout()
    
    # Save image
    plt.savefig(os.path.join(SAVE, "result.png"), dpi=150, bbox_inches='tight')
    print("✅ Performance chart saved to benchmark_gflops.png")
    
    # Display image
    plt.show()

def print_summary():
    """Print performance summary"""
    print("\n" + "=" * 60)
    print("Performance Summary (GFLOPS)")
    print("=" * 60)
    
    # Print header
    header = "Size    |"
    for _, display_name, _ in KERNELS:
        header += f" {display_name:>10} |"
    print(header)
    print("-" * len(header))
    
    # Print each row
    for i, size in enumerate(MATRIX_SIZES):
        row = f"{size:6d} |"
        for kernel_name, display_name, _ in KERNELS:
            val = RESULTS[kernel_name][i]
            if val:
                row += f" {val:10.2f} |"
            else:
                row += f" {'N/A':>10} |"
        print(row)
    
    print("=" * 60)

if __name__ == "__main__":
    # Check if executable exists
    if not os.path.exists(EXECUTABLE):
        print(f"❌ Error: Executable '{EXECUTABLE}' not found")
        print("Please compile the program first: make")
        sys.exit(1)
    
    # Run benchmark
    run_benchmark()
    
    # Print summary
    print_summary()