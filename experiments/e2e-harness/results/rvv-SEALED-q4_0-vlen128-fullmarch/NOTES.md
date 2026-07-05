# q4_0@128 e2e — SEALED (full-capability march) — rvv (openEuler VLEN128)

**Status = `SEALED` (VLEN128 prefill claim).** Promotes the 2026-07-06 bring-up
(`rvv-bringup-q4_0-vlen128/`, advisory) to a sealed cell: gate-1 is now
NON-advisory, both trees rebuilt with the board's full-capability march, and the
4.99× prefill ratio HOLDS. Our repack GEMM/GEMV vs upstream ggml block-dot.

## Board identity (instance-hash inputs)
- Board = `ssh rvv` — openEuler 24.03, `localhost.localdomain`, kernel 6.12.66
  riscv64, **VLEN128**, 64c, governor=performance, **2.6 GHz locked (DVFS span
  0.00% measured across the entire run)**, glibc 2.38.
- Board ISA (`/proc/cpuinfo`) = `rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_
  zifencei_zihintntl_zihintpause_zihpm_zawrs_zfa_zfh_zfhmin_zca_zcb_zcd_zba_zbb_
  zbc_zbs_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_sscofpmf_sstc_svinval_
  svnapot_svpbmt`.
- Toolchain = gcc-15.2.0 (`/opt/tcrv-toolchains/gcc-15.2.0`; needs
  `. /opt/tcrv-toolchains/env.sh` → `LIBRARY_PATH` for `-lgcc_s` at link time).
- Model = `tinyllama-q4_0.gguf` (sha256 `da3087fb14aede55…`, 606.53 MiB, 1.10B).

## Full-capability march (the seal precondition)
```
-march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause -mabi=lp64d -O3
```
Derived from the board ISA (all user-space codegen exts; privileged
sscofpmf/sstc/sv* dropped — not codegen). Applied **identically to A and B**.

### Mechanism finding (why a naive re-flag would have been a no-op)
The bring-up's `CMAKE_C_FLAGS="-march=rv64gcv"` was NOT what compiled the hot
kernels: ggml's own `ggml-cpu/CMakeLists.txt` APPENDS a *second, later*
`-march=${MARCH_STR}` to the RISC-V TUs (quants.c, repack.cpp) and gcc honours
the LAST `-march`. So the bring-up hot path actually compiled with
`rv64gcv_zfh_zvfh_zicbop_zihintpause` (this is why its fp16-libcall scan was
already clean). But ggml's MARCH_STR builder has **no knob** for
zbb/zbc/zbs/zfhmin/zicbom/…, so changing CMAKE_C_FLAGS alone would leave the hot
kernels unchanged — a relabel, not a real full-march test. The rebuild therefore
**overrides MARCH_STR** (one surgical `set(MARCH_STR …)` line, backed up to
`ggml-cpu/CMakeLists.txt.bak-l1seal`, applied identically to both trees) so the
full march reaches BOTH the hot path and the general TUs. Verified per-TU:
`repack.cpp` and `src/llama.cpp` both compile with the full march on both trees.
This is a **recompile of already-emitted C** (`tcrv_emitted_repack_*.inc`
untouched) — not a re-emit.

## (1) Strict preflight — PASS 4/4 (gate-1 NON-advisory)
- gate-3 same-compiler + same-flags: OK (both trees byte-identical flags).
- **gate-1 march-complete: OK** (full march covers all board critical exts) —
  the bring-up advisory is CLOSED.
- gate-2 fp16 softfloat libcall-free: OK (A & B, no `__extendhfsf2`-class).
- gate-4 fingerprint↔VLEN: OK (board VLEN=128 == target).

## (2) Prefill / decode phase-split (paired A/B, same session, 2 passes × r4)
| phase | ours t/s | stock t/s | ratio | 95% CI | DVFS | verdict |
|---|---:|---:|---:|---|---|---|
| **prefill pp128 (GEMM)** | 19.82 | 3.90 | **5.077×** | [5.039, 5.111] | 0.00% | **DIFFERENCE** |
| decode tg32 (GEVM) | 3.08 | 2.00 | 1.537× | [1.517, 1.557] | 0.00% | DIFFERENCE |

**Prefill 4.99× (bring-up) → 5.08× (full march): HOLDS** (marginally higher).
Decisive: **stock did NOT speed up** with the fuller ISA (3.90 t/s vs bring-up
3.94, within noise) → the gap is a real repack-locality win, **not** a
crippled-opponent (march-asymmetry) artifact. Both sides symmetric full march.

## (3) Decode board-pressure snapshots (memory-bound → honest RANGE)
Paired A/B, decode-only, at 3 injected memory-pressure levels (STREAM-triad hogs
on cores 12-63, measurement pinned 8-11):
| level | inject | ours | stock | ratio | 95% CI | verdict |
|---|---|---:|---:|---:|---|---|
| light  | 0 hogs (natural loadavg ~8.6, free 41.9GB) | 3.13 | 2.00 | **1.565×** | [1.548, 1.572] | DIFFERENCE |
| medium | 12 hogs (free 32.5GB)                      | 3.10 | 2.00 | 1.551× | [1.541, 1.561] | DIFFERENCE |
| heavy  | 40 hogs (loadavg 22→47, free 11.5GB)        | 2.98 | 1.99 | **1.500×** | [1.484, 1.505] | DIFFERENCE |

**Decode ratio range = 1.50× – 1.56×** (all DIFFERENCE, DVFS 0.00%). The paired
ratio is robust to injected pressure (compresses modestly toward 1.5× under heavy
memory contention). NOTE the board is multi-tenant: even "light" is not truly
idle (natural loadavg ~8.6), so absolute decode throughput and the ratio are
board-pressure sensitive — reported as a range, not a point. (The bring-up's
one-off "2.0×" light smoke was a genuinely idler board regime.)

## (4) objdump mechanism seal + symmetry
- **A (ours)** repack kernels present: `ggml_gemm_q4_0_8x8_q8_0` (262-insn body,
  4 unit-stride `vle`, **0 strided, 0 gather**, 20 widening MACs, 8×8 register
  tile), `ggml_gemm_q4_0_16x1`, `ggml_gemv_q4_0_8x8`, `tcrv_emitc_ggml_repack_
  gemm/gemv_q4_0_q8_0` entry. Mechanism = **packed-weight contiguous streaming +
  register tiling → weight reuse = memory locality**.
- **B (stock)** `ggml_vec_dot_q4_0_q8_0` (quants.c.o): 40-insn body, 3 `vle`,
  3 MACs — per-(row,col) block dequant+dot, no output tiling → poor weight reuse.
- Both use unit-stride loads only (no `vlse`/`vlux`) → clean symmetric
  vectorization. Repack advantage is MAC-density-per-load, not gather tricks.
- **Symmetry**: fp16 softfloat libcall scan — A scanned=15 objs hit=0; B
  scanned=15 hit=0. Confound §1第9条 guards against is empirically ABSENT on BOTH
  under full march (third confirmation).

## (5) Correctness gate — GREEN
- E2E greedy-token consistency: **3/3 prompts A==B** (temp0, top-k1, seed1). Real
  generation verified (A ours: "The capital of France is **Paris.**").
- logits sanity: no NaN/Inf. **Never** claims bit-exact vs ggml (fp accum order
  differs, both IEEE-legal; token identity under argmax is the correct gate).

## Verdict
**q4_0@128 prefill = SEALED** under full-capability march: 5.08× [5.04,5.11],
DVFS 0.00%, symmetric, correctness GREEN, mechanism = repack locality, no fp16
confound. Decode = DIFFERENCE across the full board-pressure range (1.50–1.56×).

## Remaining [PERF-1] 8-gate steps (for q4_0@128 prefill)
Passed here: strict-march build, symmetric A/B, DVFS/T-N noise floor,
median+IQR+bootstrap-CI, DIFFERENCE verdict, correctness GREEN, mechanism seal.
Still open (out of this touch-set): (a) **dual-board** — replicate on k1 / a
VLEN256 board (board-labeled, non-comparable across boards); (b) **micro↔e2e
Amdahl transduction accounting** — pair the archived repack micro-win with this
e2e prefill share to close `[PERF-1]` gate ④; (c) sweep prefill GEMM **M-shape**
class + add q8_0 for the T6 first-batch cells.

## Reproduce
Board scripts in `experiments/e2e-harness/board/`: `fullmarch_rebuild.sh`
(symmetric full-march recompile), `run_seal_measure.sh` (preflight→phase-split→
3× decode snapshots→objdump→correctness), `decode_snapshots.sh`, `objdump_seal.sh`.
Raw + aggregates in this dir (`evidence.json`, `phase_split_raw.txt`,
`decode_{light,medium,heavy}.txt`, `objdump_mechanism_fixed.txt`, `correctness.txt`,
`preflight.txt`).

---

## ★ VERIFY MECHANISM CORRECTION (adversarial verify, authoritative)

The mechanism above must read **capability-keyed dispatch/routing + memory locality**, NOT
"our GEMM beats their GEMM". Adversarial-verify objdump finding:

- The tiled `ggml_gemm_q4_0_8x8_q8_0` prefill kernel is **BYTE-IDENTICAL in both A and B trees**
  (upstream ggml repack code, compiled in both) → the 5.08× is NOT our-vs-their codegen superiority.
- The win is **runtime DISPATCH/ROUTING**: A engages repack for q4_0 at VLEN128 (A prints
  `TCRV EMITTED GEMV(q4_0_16x1 VLEN128 compiler-emitted) ENGAGED`), while stock B leaves q4_0 on
  the hand-tuned **block-dot** path at VLEN128 (ggml repack is VLEN-gated/TODO there). Repack's
  packed-weight contiguous streaming + register tiling → memory locality → 5.08×.
- The **decode GEVM (q4_0_16x1) IS our compiler-emitted**; the prefill tiled GEMM is ggml's, routed
  by our capability-keyed path selection. So the honest claim = "our compiler's path-selection
  engages repack for 5× e2e prefill", consistent with option-2 path-selection novelty; NOT "faster
  repack kernel codegen".
- Attribution basis: A's ENGAGED runtime diagnostic + the 5× gap + static objdump. **perf PMU gave
  0 samples (RISC-V PMU paranoid=2)** — a SAMPLED runtime hot-symbol profile is a remaining upgrade.
