import math
import torch # type: ignore
import triton # type: ignore
import triton.language as tl # type: ignore

@triton.jit()
def matmul_kernel(x_ptr, y_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                  BLOCK_K: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    row_start = pid_r * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)

    col_start = pid_c * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)

    out = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    for k in tl.range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        row = row_offsets[:, None] * K + k_offsets[None, :]
        mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
        x = tl.load(x_ptr + row, mask = mask)

        col = col_offsets[None, :] + k_offsets[:, None] * N
        mask = (col_offsets[None, :] < N) & (k_offsets[:, None] < K)
        y = tl.load(y_ptr + col, mask = mask)

        out = tl.dot(x, y, out)

    out_offsets = row_offsets[:, None] * N + col_offsets[None, :]
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    tl.store(out_ptr + out_offsets, out, mask = mask)

def matmul(x, y, BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64):
    M, K = x.size()
    N = y.size(1)
    assert K == y.size(0)
    out = torch.empty(M, N, device = 'cuda', dtype = torch.float32)
    grid = (math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N))
    matmul_kernel[grid](x, y, out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    return out

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"],
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider",
        line_vals = ["triton", "torch"],
        line_names = ["Triton", "Torch"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS",
        plot_name = "matmul-performance",
        args = {},
    ))
def benchmark(M, N, K, provider):
    x = torch.randn(M, K, device = 'cuda', dtype = torch.float32)
    y = torch.randn(K, N, device = 'cuda', dtype = torch.float32)

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.matmul(x, y))
    else:
        ms = triton.testing.do_bench(lambda: matmul(x, y))
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)

benchmark.run(print_data = True, save_path = "plots")