import torch


def time_matmul(M, N, K):
    n_iter = 10000  # Number of iterations to time
    n_warmup = 10  # Number of warmup iterations

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    # Construct input matrices
    A = torch.rand(M, K, device="cuda")
    B = torch.rand(K, N, device="cuda")

    # Benchmark the GEMM
    for i in range(n_iter + n_warmup):
        if i == n_warmup:
            t0.record()  # Don't start recording until warmup is finished
        C = A @ B

    # Compute elapsed time
    t1.record()
    torch.cuda.synchronize()
    dt = t0.elapsed_time(t1) / 1000

    print(f"{n_iter/dt:0.2f} iter/s ({dt:0.4g}s)")


time_matmul(512, 1024, 2048)
