import math
import torch # type: ignore
import triton # type: ignore
import triton.language as tl # type: ignore

@triton.jit()
def vector_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask = mask)

def vector_add(x, y, BLOCK_SIZE = 128):
    out = torch.empty_like(x)
    N = x.numel()
    grid = int(math.ceil(N / BLOCK_SIZE))
    vector_add_kernel[(grid, )](x, y, out, N, BLOCK_SIZE)
    return out

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['size'],
        x_vals = [2**i for i in range(12, 28)],
        x_log = True,
        line_arg = 'BLOCK_SIZE',
        line_vals = [2**i for i in range(7, 12)],
        line_names = [str(2**i) for i in range(7, 12)],
        styles = [('red', '-'), ('blue', '-'), ('green', '-'), ('orange', '-'), ('magenta', '-'), ('cyan', '-')],
        ylabel = 's',
        plot_name = 'vector-add-performance',
        args = {},
    ))
def benchmark(size, BLOCK_SIZE):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    ms = triton.testing.do_bench(lambda: vector_add(x, y, BLOCK_SIZE))
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

benchmark.run(print_data = True, save_path = "plots")