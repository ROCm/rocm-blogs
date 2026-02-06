"""
Benchmark the performance of PyTorch graph layout algorithms.
These include:
- torch CPU version
- torch GPU version

The benchmarks are run for different graph sizes: 2^n nodes, n * 2^n edges, for different n.

To run in Docker with ROCm and PyTorch:
    1. Copy graph_layout_benchmark.py and graph_layout.py to your local directory
    2. Run: docker run --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined -v $(pwd):/workspace -w /workspace rocm/pytorch:latest python graph_layout_benchmark.py
"""

import torch
import time
import numpy as np
from graph_layout import update_all_positions_torch


def run_benchmarks():
    """Benchmark PyTorch CPU vs GPU implementations across different graph sizes."""
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print("Will benchmark both CPU and GPU")
    else:
        print("GPU not available, will benchmark CPU only")
    print("-" * 50)

    rng = np.random.default_rng(42)
    n_iterations = 20

    # Print table header
    header = ["Nodes", "Edges", "PyTorch_CPU_ms", "PyTorch_GPU_ms", "GPU_vs_CPU"]
    print("\t".join(header))

    # Benchmark across different sizes: 2^n nodes for different n
    for n in range(12):
        N_NODES = 2 ** n
        N_EDGES = n * N_NODES

        # Generate random graph data
        angles = rng.uniform(0, 2 * np.pi, N_NODES)
        radii = rng.uniform(0, 1, N_NODES) ** 0.5  # Uniform distribution in disc
        positions_np = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        sizes_np = rng.uniform(0.01, 0.05, N_NODES)

        edge_pairs = rng.choice(N_NODES, size=(N_EDGES, 2), replace=True)
        edge_pairs = edge_pairs[edge_pairs[:, 0] != edge_pairs[:, 1]]  # Remove self-loops
        edge_weights_np = rng.uniform(0.5, 2.0, len(edge_pairs))

        # Convert to torch format
        positions_torch = torch.from_numpy(positions_np).float()
        sizes_torch = torch.from_numpy(sizes_np).float()
        edge_indices_torch = torch.from_numpy(edge_pairs).long()
        edge_weights_torch = torch.from_numpy(edge_weights_np).float()

        # Warmup runs
        for _ in range(3):
            _ = update_all_positions_torch(positions_torch.clone(), sizes_torch,
                                          edge_indices_torch, edge_weights_torch, device=torch.device('cpu'))
            if gpu_available:
                _ = update_all_positions_torch(positions_torch.clone(), sizes_torch,
                                              edge_indices_torch, edge_weights_torch, device=torch.device('cuda'))

        # Benchmark torch CPU version
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = update_all_positions_torch(positions_torch.clone(), sizes_torch,
                                          edge_indices_torch, edge_weights_torch, device=torch.device('cpu'))
        torch_cpu_time = (time.perf_counter() - start) / n_iterations

        # Benchmark torch GPU version (if available)
        torch_gpu_time = None
        if gpu_available:
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = update_all_positions_torch(positions_torch.clone(), sizes_torch,
                                              edge_indices_torch, edge_weights_torch, device=torch.device('cuda'))
            torch_gpu_time = (time.perf_counter() - start) / n_iterations

        # Format results as tab-separated row
        gpu_vs_cpu = torch_cpu_time / torch_gpu_time if torch_gpu_time is not None else None

        row = [
            str(N_NODES),
            str(N_EDGES),
            f"{torch_cpu_time*1000:.3f}",
            f"{torch_gpu_time*1000:.3f}" if torch_gpu_time is not None else "N/A",
            f"{gpu_vs_cpu:.2f}" if gpu_vs_cpu is not None else "N/A"
        ]
        print("\t".join(row))


if __name__ == "__main__":
    run_benchmarks()

