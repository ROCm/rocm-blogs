# Reproducers

Hand-written gfx942 assembly for the kernels used in the experiments for this blog.

The driver launches **608** workgroups of **256** threads (4 waves), which is two workgroups per compute unit on MI300X. Workgroup executions are serialized as all kernels have occupancy 1.

## Files

| File | Blog figure | What it is |
|---|---|---|
| `figure02_clustered_loads_with_barriers.s` | Figure 2 | Sixteen clustered loads, then VALU and MFMA under `s_waitcnt vmcnt(N)` for N = 14, 12, 10, 8, 6, 4, 2, 0. `s_barrier` at both ends of the inner loop. |
| `figure03_clustered_loads_no_barriers.s` | Figure 3 | Same sixteen loads and compute, with the in-loop barriers removed and one `s_waitcnt vmcnt(0)` before the compute. |
| `figure04_2_consecutive_loads.s` | Figure 4 | Two loads between the barriers. |
| `figure05_12_consecutive_loads.s` | Figure 5, 12-load row | Twelve loads between the barriers. |
| `figure05_13_consecutive_loads.s` | Figure 5, 13-load row | Thirteen loads between the barriers. |
| `figure05_14_consecutive_loads.s` | Figure 5, 14-load row | Fourteen loads between the barriers. |
| `figure05_16_consecutive_loads.s` | Figure 5, 16-load row | Sixteen loads between the barriers. |
| `figure07_coalesced_loads.s` | Figure 7 | Same 16-load schedule as the pure-load kernel, with lane-linear addresses. |
| `figure08_swp_pairs.s` | Figure 8 | Software pipeline. Each sub-block is two VALU, two MFMA, two loads, under `s_waitcnt vmcnt(14)`. |
| `figure09_swp_interleaved.s` | Figure 9 | Same pipeline, interleaved one VALU, one MFMA, one load. |
| `figure10_eight_valu.s` | Figure 10 | Eight `v_mov` before each load. `s_waitcnt vmcnt(14)` stays in each sub-block. |
| `figure11_sixteen_valu.s` | Figure 11 | Sixteen `v_mov` before each load. `s_waitcnt vmcnt(14)` stays in each sub-block. |
| `figure12_extra_mfmas.s` | Figure 12 | Two VALU, then three back-to-back `v_mfma_f32_32x32x8_f16`, then each load. |
| `figure13_eight_valu_wait_early.s` | Figure 13 | Eight `v_mov` before each load, with a single `s_waitcnt vmcnt(0)` at loop entry. |
| `figure14_gemm_clustered.s` | Figure 14 | Motivating tiled GEMM. The eight `buffer_load_dwordx4` of an iteration are issued as a group. |
| `figure15_gemm_interleaved.s` | Figure 15 | Same GEMM with those loads mixed into the MFMA stream. |
| `driver.cpp` | | Loads an assembled code object and launches `matmul_kernel`. |
| `matmul.json` | | `rocprofv3` ATT input used for the traces in the post. |


## Compile and Run

Assemble one kernel to a code object. Repeat for each `.s` file; the
host binary is built once.

```bash
/opt/rocm/llvm/bin/clang -cc1as -triple amdgcn-amd-amdhsa -filetype obj -main-file-name 4x12.ll -target-cpu gfx942 -fdebug-compilation-dir=<ROOT> -dwarf-version=5 -mrelocation-model pic -o <ASM>.o <path_to_ASM>.s
/opt/rocm/llvm/bin/ld.lld -shared -o <ASM>.hsaco <ASM>.o
/opt/rocm/llvm/bin/clang++ -std=c++17 -Wall -Wextra -O2 -g  -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__= -x hip -c -o driver.o driver.cpp
/opt/rocm/llvm/bin/clang++ -L/opt/rocm/lib -lamdhip64 -Wl,-rpath=/opt/rocm/lib -o driver driver.o
./driver <ASM>.hsaco
```

## ATT

```bash
export LD_LIBRARY_PATH=<path to rocprof trace decoder>:$LD_LIBRARY_PATH
/opt/rocm/bin/rocprofv3 --att -i matmul.json -d <OUTDIR> -- ./driver <ASM>.hsaco
```
To get started with Advanced Thread Traces, please see here: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html

Use the same command for the other kernels. Change the `.hsaco` path and
the `-d` directory. 