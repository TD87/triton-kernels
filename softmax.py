import torch # type: ignore
import triton # type: ignore
import triton.language as tl # type: ignore
from triton.runtime import driver # type: ignore

NUM_WARPS = 8
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

@triton.jit()
def softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    n_programs = tl.num_programs(axis = 0)

    num_rows = tl.cdiv(n_rows, n_programs)
    row_start = pid * num_rows
    row_end = min((pid + 1) * num_rows, n_rows)

    for row_idx in tl.range(row_start, row_end):
        row_offset = row_idx * n_cols
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row = tl.load(x_ptr + row_offset + offsets, mask = mask, other = -float('inf'))
        row_max = tl.max(row, axis = 0)
        row = row - row_max
        row_exp = tl.exp(row)
        row_sum = tl.sum(row_exp, axis = 0)
        row_softmax = row_exp / row_sum
        tl.store(out_ptr + row_offset + offsets, row_softmax, mask = mask)

def calculate_num_programs(*args, **kwargs):
    kernel = softmax_kernel.warmup(*args, **kwargs, num_warps = NUM_WARPS, grid = (1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared

    occupancy = NUM_REGS // (n_regs * WARP_SIZE * NUM_WARPS)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, args[2])
    return num_programs

def softmax(x):
    out = torch.empty_like(x)
    n_rows, n_cols = x.size()
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    args = (x, out, n_rows, n_cols, BLOCK_SIZE)
    n_programs = calculate_num_programs(*args)
    softmax_kernel[(n_programs, )](*args)
    return out

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['N'],
        x_vals = [128 * i for i in range(2, 80)],
        line_arg = 'provider',
        line_vals = ['triton', 'torch'],
        line_names = ["Triton", "Torch"],
        styles = [('blue', '-'), ('green', '-')],
        ylabel = "GB/s", 
        plot_name = "softmax-performance",
        args = {'M': 4096},
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda')
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

benchmark.run(print_data = True, save_path = "plots")