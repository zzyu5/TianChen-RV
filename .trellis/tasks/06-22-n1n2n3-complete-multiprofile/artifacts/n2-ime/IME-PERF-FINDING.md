# IME vs RVV int8 matmul — kernel speedup on real K1 (Spacemit X60)

**Date:** 2026-06-23
**Hardware under test (I8 runtime evidence):** `ssh k1` = Spacemit(R) X60, Bianbu Linux 6.6.63 riscv64.
Pinned to harts 0–3 via `taskset -c 0-3` (the IME-bearing harts; hart 4 lacks `_ime` and would SIGILL).
**Toolchain:** SpacemiT GCC 15.2 fork,
`/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/riscv64-unknown-linux-gnu-gcc`,
`-O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d`.
Binary objdump confirms a real `smt.vmadot v2,v0,v1` (encoding `e210312b`).

---

## RESULT

```
shape = 256 x 256 x 256   int8 -> int32 matmul
IME   = 1,206,476 ns   (best of 50, clock_gettime MONOTONIC)
RVV   = 6,652,785 ns   (best of 50)
speedup (RVV / IME) = 5.51x
agree = ok    (IME == scalar AND RVV == scalar, all 65536 elements; cksum = 30041455 for all three)
EXIT_CODE = 0
```

**IME is 5.51x faster than the original RVV vector baseline on real K1 silicon.**

---

## What was measured

A single hand-written C program (`~/ime-bench/ime_vs_rvv.c` on the cross host) with three kernels over
the *same* discriminating signed int8 data and the *same* `C[i][j] = Σ_k A[i][k]·B[j][k]` layout
(A is (M,K) row-major, B stored (N,K) row-major = Bᵀ of the math (K,N) matrix):

- **`ime_matmul`** — SpacemiT IME1 `vmadot` (int8→int32, 4×4×8 MAC at VLEN=256/SEW=8), tiled
  M/4 × N/4 × K/8. A-tile in `v0`, B-tile in `v1`, int32 accumulator in the even-VD pair `v2`/`v3`
  (register choice copied verbatim from the validated FOUNDATION.md template). The accumulator pair
  is zeroed **once** per output tile and `vmadot` accumulates across all 32 k-tiles.
  Inputs are **pre-packed** into tile-major layout (each 4×8 sub-block made contiguous for `vle8.v`)
  — a 4×8 sub-block of a 256-wide matrix is *not* contiguous, so packing is mandatory for correctness.
- **`rvv_matmul`** — the "原先 rvv" baseline: RVV 1.0 vector dot products with int32 widening
  accumulate (`vsext.vf2` int8→int16, then `__riscv_vwmacc` int16×int16→int32, `vredsum` reduce).
  This is a genuine vector kernel, **not** scalar. int32 accumulate is required (an int16 `vwmacc`
  would overflow at K=256: 256·127² ≈ 4.1M ≫ 32767).
- **`scalar_ref`** — plain triple-loop oracle.

**Verification gates** (so the number is real, not plausible-but-wrong):
- MAC-unit assert: aborts unless `vlenb==32 && vl(e8,m1)==32` (genuinely VLEN=256 / 4×4×8).
- Correctness asserted on the **full 256³** before any timing: `ime==scalar` and `rvv==scalar`,
  all 65536 int32 elements bit-exact (checksum 30041455 identical across all three).
- Warm-up run before the best-of-50 (page-fault noise); checksum sink prevents `-O2` dead-code elision.
- Packing is **excluded from both timed regions** (symmetric, one-time reformat) — honest "kernel speedup".

## Honest note

The IME is a **dedicated matrix unit** (a hardware 4×4×8 int8 MAC array), so a large win over a
general-purpose RVV vector pipeline is expected — 5.51x is in line with that, not anomalous. The RVV
baseline here is a straightforward widening dot-product kernel (`vwmacc` + `vredsum`), i.e. a competent
but not micro-tuned vector matmul; a more aggressively blocked/register-tiled RVV kernel would narrow
the gap somewhat. The comparison is apples-to-apples in data, layout, accumulator width, and timing
methodology, and both arms are bit-exact against the scalar oracle. This is a **kernel-level** measurement
(out of scope: end-to-end / llama.cpp integration).

## Provenance / reproduce

- Source (scratch, not in git tree): `/home/kingdom/ime-bench/ime_vs_rvv.c`
- Cross-compile (one command):
  `riscv64-unknown-linux-gnu-gcc -O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d ime_vs_rvv.c -o ime_vs_rvv`
- `scp ime_vs_rvv k1:~/ime_vs_rvv`
- `ssh k1 'taskset -c 0-3 ~/ime_vs_rvv'`  → the RESULT line above (exit 0).

---

## E2E (llama.cpp on K1) — VALID lib-swap A/B, finalized 2026-06-23

**Goal:** IME-vs-RVV *end-to-end* tok/s in real llama.cpp on K1 (Q4_0 tinyllama), closing the
"out of scope: end-to-end" note above.

**Build found (no rebuild needed):** `~/tcrv-k1-llama` on `ssh k1` is a full llama.cpp tree with the
**merged SpacemiT IME backend** compiled in (`GGML_CPU_RISCV64_SPACEMIT`, source under
`ggml/src/ggml-cpu/spacemit/`). Built binaries: `build/bin/llama-bench`, `llama-cli`; lib
`libggml-cpu.so.0.15.1` contains the IME kernels (`ime1_kernels`/`ime2_kernels`/`rvv_kernels`).
Model: `~/tcrv-k1-llama/models/tinyllama-q4_0.gguf` (llama 1B Q4_0, 606 MiB).

### How IME is toggled — env does NOT work; LIB-SWAP does (correction of earlier approach)
The backend dispatches gemm by `global_spine_env_info.use_ime1/use_ime2` (ime.cpp ~L265/L314), derived
in `ime_env.cpp` from the perfer core arch. **There is NO working env lever on real K1:**
- `SPACEMIT_CORE_ARCH=0x503C` is a **qemu-only no-op**: the block reading it (ime_env.cpp ~L163) is gated
  by `if (core_info_list.size() == 0)  // special for qemu`. On real K1 cpuinfo parsing succeeds (8 cores,
  `size()==8`), so that block **never runs** — the var is silently ignored.
- `SPACEMIT_PERFER_CORE_ARCH=0x503C` (x60) and `=0xA064` (a100): measured pp32 = 24.07 / 23.93 —
  **identical to IME-ON 23.98**, no kernel change. (Earlier transient "pp≈3" readings under these env vars
  were CONTAMINATION: a stray `llama-cli` PID 532705 pinned 100% on harts 0-3; since killed.)

**The valid lever = LIB SWAP.** `~/libggml-cpu-1x16.so` (built same Jun-22 session) is a genuine **non-IME
RVV** CPU backend — **zero spacemit/IME content** (`strings | grep -c spacemit == 0`; the only "ime"-matching
symbols are `timestep_embedding`/`time_us` false-positives) — with the **same soname `libggml-cpu.so.0`**,
so it's an ABI-compatible `LD_LIBRARY_PATH` drop-in vs the default IME lib.

### E2E RESULT — non-IME lib vs IME lib (tinyllama 1B Q4_0, t4, `taskset -c 0-3`, r3, same load window)
| regime | non-IME lib (1x16 RVV) | IME lib (default) | IME / RVV ratio |
|---|---|---|---|
| pp32 (prefill) | 23.87 ± 0.06 | 23.35 ± 0.07 | **0.98×** |
| tg16 (decode)  |  6.52 ± 0.02 |  5.63 ± 0.03 | **0.86×** |

**HONEST VERDICT — IME gives NO e2e speedup on this model; it is marginally SLOWER (0.86–0.98×).**
The 5.51× kernel-level IME win (256³ int8 matmul) does **not** transplant to llama.cpp e2e on
tinyllama-1B-Q4_0:
- Decode (tg) is GEVM (matrix×vector, M=1), memory-bandwidth-bound — a matrix unit cannot help; IME is
  even ~14% slower (repack/quant overhead with no MAC payoff at M=1).
- Prefill (pp) at 1B / 32 tokens is dominated by Q4_0 dequant + memory traffic, not matmul FLOPs, so the
  IME gemm doesn't move wall-clock (≈parity).
- Same regime lesson as N3 Win-A/Win-B: a compute-bound kernel win does not transplant to a memory-bound,
  decode-dominated small-model LLM e2e. K1's 7 GB caps models at ~1–3B, so a larger matmul-bound model
  (where prefill IME might surface) can't be run here.

**Defensible IME perf claim = the kernel-level 5.51× vs RVV (top of file). The e2e is an honest ~1× NULL
(0.86–0.98×) on the runnable model — disclosed, not hidden.** Caveat: the non-IME baseline is N3 Win-A
*tuned* RVV (1x16 LMUL strip), a strong/conservative baseline; mainline-RVV would need a fresh
`GGML_CPU_RISCV64_SPACEMIT=OFF` build, but the conclusion (no e2e IME win on this model) is robust.

> NOTE — supersedes an earlier draft NULL in this file that used `SPACEMIT_CORE_ARCH=0x503C` /`0x0000`
> as the "IME-OFF" arms. Those are the qemu-gated no-op above, so that draft was actually **IME-vs-IME**
> (its flat ~23 across all columns is the tell). Right answer, invalid method — replaced by this lib-swap.

**Reproduce (lib-swap, the valid A/B):**
```
ssh k1
cd ~/tcrv-k1-llama/build/bin && B=$PWD
mkdir -p /tmp/rvvlib && cp ~/libggml-cpu-1x16.so /tmp/rvvlib/libggml-cpu.so.0
# non-IME (RVV) arm:
LD_LIBRARY_PATH=/tmp/rvvlib:$B taskset -c 0-3 ./llama-bench -m ../../models/tinyllama-q4_0.gguf -p 32 -n 16 -t 4 -r 3
# IME arm:
LD_LIBRARY_PATH=$B            taskset -c 0-3 ./llama-bench -m ../../models/tinyllama-q4_0.gguf -p 32 -n 16 -t 4 -r 3
```
(K1 load was ~3.3–4.0 during runs — shared box — so absolute tok/s are clock-contended, but both arms
ran in the same window at the same load, so the **ratio** is valid. The prior known-good IME-ON of
pp32~46.65 was on an idle box; the ~23 here reflects contention, not a regression.)
