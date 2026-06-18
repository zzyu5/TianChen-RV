# E2E Baseline — Clean Mainline llama.cpp on `ssh rvv` (Q4_0 Llama-2-7B)

Honest default ggml prefill/decode throughput baseline. This is the e2e denominator
that anchors our kernel-level (q4_0 GEMM) speedups and resolves whether ggml repacks
q4_0 weights by default on this RISC-V board.

**OUR OWN clean copy** — cloned fresh into `~/tcrv-llamacpp` on rvv. No contact with the
other project's dirs (`~/llama_integ*`, `~/workspace/workspace3/llama.cpp`,
`*_repackON/_canary/_override`). Default config: no repack forced on or off.

---

## Host facts (`ssh rvv`)

| field | value |
|---|---|
| uname | `Linux ubuntu 6.12.23 #1 SMP riscv64 GNU/Linux` |
| ISA | `rv64imafdcv_zicbom_zicboz_..._zba_zbb_zbc_zbs_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_...` |
| Vector | RVV 1.0 present (`v`, `zve64*`, `zvfh`); **VLEN = 128 bit** (`vlenb` CSR = 16 bytes, confirmed at runtime) |
| Physical harts | **4** (`/proc/cpuinfo` processors 0–3). `nproc` reports 64 — a misreport; only 4 real harts. Per task discipline, builds use `-j4` and benches use `-t {1,4,8}`, never `-t 64`. |
| RAM | 121 GiB total, ~119 GiB available |
| cmake | 3.28.3 |
| Default C/C++ compiler used by build | GNU 14.2.0 (`/usr/bin/cc`, `/usr/bin/c++`) |
| (also available) | clang/clang-18 18.1.3, gcc-14 14.2.0 |

## llama.cpp version (OUR baseline copy)

| field | value |
|---|---|
| repo | `https://github.com/ggml-org/llama.cpp` (clone into `~/tcrv-llamacpp`) |
| commit | `f3e182816421c648188b5eab269853bf1531d950` |
| tag | `b9692` (mainline `master`, dated 2026-06-17) |
| ggml version / commit | 0.15.1 / `f3e1828` |

## cmake feature summary (DEFAULT config — `cmake -B build -DCMAKE_BUILD_TYPE=Release`)

Auto-detected by ggml (full log: `cmake_configure.log`):

```
-- CMAKE_SYSTEM_PROCESSOR: riscv64
-- GGML_SYSTEM_ARCH: riscv64
-- Including CPU backend
-- riscv64 detected
-- Adding CPU backend variant ggml-cpu: -march=rv64gcv_zfh_zvfh_zicbop_zihintpause;-mabi=lp64d
-- ggml version: 0.15.1
-- ggml commit:  f3e1828
-- Found OpenMP: TRUE (found version "4.5")
```

- ggml auto-enabled **RVV** (the `v` in `-march=rv64gcv...`, plus `zfh`/`zvfh`). This is `GGML_RVV`
  on by default for this riscv host.
- **No** `aarch64`, `x86`, `avx`, `neon`, or separate "repack"/"extra buffers" backend-variant
  line appears anywhere in the configure log — only ONE `ggml-cpu` variant (riscv). The repack
  code (`GGML_CPU_REPACK`, formerly `GGML_CPU_AARCH64`) is **compiled in** by default, but that
  alone does NOT mean it activates (see verdict below).
- Build: `cmake --build build -j4 --target llama-bench llama-cli` (low parallelism per host
  sensitivity). Exit 0. Full log: `build.log`.

## Model

`/home/ubuntu/llama-2-7b-chat.Q4_0.gguf` (3.56 GiB on disk, 4.54 BPW). From `llama-bench`/`print_info`:

| field | value |
|---|---|
| quant (file type) | **Q4_0** (confirmed loaded as Q4_0) |
| arch | llama (LLaMA v2) |
| params | 6.74 B (7B) |
| n_embd | **4096** |
| n_ff | **11008** |
| n_layer | 32 |
| n_head / n_head_kv | 32 / 32 (no GQA) |
| n_ctx_train | 4096 |

n_embd=4096 / n_ff=11008 match our micro-probe n for the q4_0 GEMM.

---

## REPACK VERDICT (critical, baseline fairness)

### Does default ggml repack q4_0 on this rv64gcv (VLEN=128) board? **NO.**

Two independent signals agree:

**(1) Runtime — weight tensors fall back to the plain CPU buffer.**
From the model load (`bench-verbose-evidence.log`):

```
done_getting_tensors: tensor 'token_embd.weight' (q4_0) (and 290 others) cannot be used
                      with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:   CPU_Mapped model buffer size =  3647.87 MiB
```

All 291 q4_0 weight tensors are explicitly **rejected by the CPU_REPACK buffer and loaded into
the plain CPU (`CPU_Mapped`) buffer instead**. There is a single CPU_Mapped buffer and NO
separate CPU_REPACK buffer. All 32 layers `assigned to device CPU`. => the q4_0 mul_mat takes
the standard per-block `ggml_vec_dot_q4_0_q8_0` path, not a repacked block-as-lane GEMM.

**(2) Source — the riscv q4_0 repack GEMM is unimplemented at VLEN=128.**
`ggml/src/ggml-cpu/repack.cpp`, `ggml_repack_get_optimal_repack_type()`, the `GGML_TYPE_Q4_0`
branch for a riscv-V host (the only branch our host can take — no avx2/sve/neon):

```c
if (ggml_cpu_has_riscv_v()) {
    #if defined __riscv_zvfh
    switch (__riscv_vlenb() * 8) {
        case 128:  { break; } // TODO        <-- OUR BOARD: no repack type selected
        case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; }
        case 512:  { break; } // TODO
        case 1024: { break; } // TODO
        default:   { return nullptr; }
    }
    #endif
}
...
return nullptr;   // function returns nullptr for VLEN=128
```

The riscv q4_0 repack path is implemented **only for VLEN=256**. Our board is VLEN=128
(`vlenb`=16 → 128 bit, confirmed at runtime), which hits `case 128: { break; } // TODO` and the
function returns `nullptr`. A nullptr makes `extra_buffer_type::supports_op` decline the tensor,
so the CPU_REPACK buffer never claims the q4_0 weights — exactly what signal (1) shows at runtime.

**Conclusion:** On this rv64gcv VLEN=128 host, default mainline ggml does **NOT** repack q4_0.
The default `llama-bench pp512` tok/s below is therefore the honest per-`ggml_vec_dot_q4_0_q8_0`
denominator — our q4_0 GEMM micro-win is measured against the real default path, not a soft one.
(Note: the cmake flag `GGML_CPU_REPACK`/`AARCH64` being ON is NOT evidence either way — it only
compiles the code; activation requires a matching host kernel, which does not exist at VLEN=128.)

---

## llama-bench: pp512 / tg128, thread sweep -t {1,4,8}

Default repetitions (5) + stddev as llama-bench reports them. Run as
`llama-bench -m <model> -p 512 -n 128 -t {1,4,8}` (run per-thread fastest-first so each row banks
independently). Full output: `llama-bench-sweep.log`.

<!-- PENDING: table is being measured on rvv (the -t 1 single-threaded 7B phase is slow,
     ~15-30 min). Numbers will be filled in below once the detached sweep completes. -->

| threads | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|
| 1 | _pending_ | _pending_ |
| 4 | _pending_ | _pending_ |
| 8 | _pending_ | _pending_ |

(Expected shape: pp512 rises 1→4 threads; -t 8 may regress vs -t 4 due to oversubscription of
the 4 physical harts; tg128 is memory-bound and may plateau early. Reported honestly, not tuned.)

---

## Artifacts in this dir

- `RESULTS.md` — this file
- `cmake_configure.log` — full default cmake configure output (feature summary)
- `build.log` — full `-j4` build log (llama-bench + llama-cli, exit 0)
- `bench-verbose-evidence.log` — `print_info` dims + `load_tensors` buffer evidence (repack verdict source)
- `llama-bench-sweep.log` — the pp512/tg128 thread-sweep table (pulled when complete)
