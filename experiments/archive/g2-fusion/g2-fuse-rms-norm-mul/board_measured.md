# G2 [FUSE] rms_norm->mul — BOARD PAIRED MEASURE (rvv/VLEN128, 裁决二.5, 2026-07-07)

> The 贯通 tracer (NOTES.md, STOP-at-贯通) proved the mechanism + **emit-level** byte accounting
> (y[] round trip = 8·n bytes/row eliminated) but left the real DRAM benefit **pending-hardware**.
> This is that hardware measure: paired fused-vs-unfused on the L3 memory axis, with **measured
> wall speedup + measured DRAM bytes** (LSU/LLC-miss counters). Raw: `board_raw.txt`.

## 0. Pre-registration (stated before the numbers, per 裁决二.5)
- **铺量触发 (rollout-trigger) rule**: a statistically-significant POSITIVE isolated speedup of FUSED
  over UNFUSED, passing the T-N effect rule (delta > 2× the noise floor AND economically > 2% AND
  non-overlapping IQR / drift-separated), **AND** the measured DRAM-byte elimination matching the
  emit-level prediction (y[] round trip = 8·n·rows). If met, the fusion mechanism is validated on
  silicon and rollout to other epilogue pairs (silu/scale/bias, [FMT-PROP]) is justified.
- **Protocol**: preflight(0) toolchain symmetry / cache-cold working set > 3×L3 / paired interleaved /
  N≥10 median+IQR / T-N floor / restore. board fingerprint pinned below.

## 1. Setup (faithful to the emit census)
Driver `tools/e2e-harness/board/g2_fuse_driver.c` reconstructs the emit census exactly
(`emit_census.txt`, from `tcrv-opt --tcrv-rvv-lower-to-emitc` of the fused epilogue lit @HEAD 0ad9cf6d):
- **FUSED** (one pass/row): scalar-double Σx² reduce → `scale=1/sqrtf(mean+eps)`; then the m8 loop
  `vx=vle32(x) ; vy=vfmul_vf(vx,scale) ; vw=vle32(w) ; vz=vfmul_vv(vy,vw) ; vse32(z)` — y[] never stored.
- **UNFUSED** (two passes over the whole tensor): pass1 rms_norm `…; vse32(y)` **materializes y[]**;
  pass2 mul `vy=vle32(y) ; vw=vle32(w) ; vz=vfmul_vv ; vse32(z)` **reloads y[]**.
- The ONLY difference is the intermediate row-tensor y[] round trip. Each tensor (X,Y,Z) = 128 MiB,
  y[] (128 MiB) > L3 (64 MiB) so it genuinely evicts between pass1-write and pass2-read → real DRAM
  round trip, isolating exactly the 8·n·rows bytes the fusion elides.
- **Toolchain symmetry (preflight 0)** is trivial and total: both kernels are ONE source file, ONE
  `clang-17.0.6 -O3 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d`, no fast-math (reduce stays
  serial = byte-exact posture). No compiler-asymmetry defect is possible here.
- **board_fp** = rvv-VLEN128-clang17.0.6-march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs-core8(taskset)-gov=perf-2.6GHz.
  main tree + build/ UNTOUCHED, nothing committed, no git stash.

## 2. Correctness on silicon (bonus beyond emit-level byte-exact)
`verify`: fused-Z vs unfused-Z **ndiff=0 / max_ulp=0**; paired folds identical
(0xb6533448dbf94a01). The register-kept `vy` equals the store-then-reload `y[]` **bit-for-bit on
hardware** — the fusion is value-preserving on silicon, not only at the emit level.

## 3. Wall speedup (cold, N=12, interleaved) ★
| kernel | median (ms) | IQR (ms) | IQR% | min (ms) |
|---|---:|---:|---:|---:|
| FUSED       | 165.180 | 0.063 | 0.038% | 165.081 |
| UNFUSED     | 215.993 | 0.127 | 0.059% | 215.932 |
| sentinel (fused repeat) | 167.867 | 0.085 | — | 167.815 |

- **speedup (unfused/fused) = 1.3076×**  (fused takes **23.5% less wall time**; −50.81 ms/row-tensor-pass).
- **Noise floor**: between-round IQR = 0.038–0.059%; cross-position drift sentinel (fused@pos1 vs
  fused@pos3) = 1.0163 (**1.63%**, the conservative floor — it absorbs the cache state unfused leaves).
- **T-N verdict**: speedup 30.76% is **≈19× the conservative 1.63% floor** and the fused/unfused IQR
  bands are **completely non-overlapping** (a ~50 ms gap vs ~0.1 ms IQR). Bootstrap-CI of the ratio
  excludes 1.0 by a wide margin. → **statistically-significant positive gain: T-N PASS.**

## 4. Measured DRAM bytes eliminated = the y[] round trip (dual evidence) ★
`perf stat` LLC load/store-miss deltas (unfused − fused; init/memset is byte-identical between the two
perf runs so it cancels, leaving the pure kernel signal). ×64 B/line:

| counter | per-iter delta | bytes | maps to |
|---|---:|---:|---|
| LLC-load-miss delta (5-iter run)  | 2,101,447 lines | 128.25 MiB | y[] **reload** |
| LLC-store-miss delta (5-iter run) | 2,097,426 lines | 127.97 MiB | y[] **store**  |
| **total** | **4,198,873 lines** | **256.22 MiB** | y[] **round trip** |

- **Predicted** (byte_accounting.csv, emit-level): eliminated = 8·n·rows = 8·4096·8192 = 268,435,456 B
  = **256.00 MiB**. **Measured 256.22 MiB = 100.1%** (3-iter run: 255.93 MiB = 99.97%). The hardware
  DRAM-byte count confirms the emit-level 8·n accounting to within 0.1%.
- **Cycle cross-check**: cycles delta = 130.58 Mcyc/iter (5-iter) / 2.6 GHz = **50.22 ms/iter**, matching
  the wall delta **50.81 ms/iter** (1.2%). And 256 MiB at the board's ~5.2 GB/s aggregate read wall
  (roofline cell) ≈ 51 ms — so the ~51 ms speedup is **almost entirely explained by the 256 MiB of
  DRAM traffic removed at the memory bandwidth.** Memory-axis mechanism → silicon, closed.

## 5. Verdict — 铺量触发 SATISFIED
Both pre-registered conditions are met: (a) significant positive isolated speedup (1.308×, ~19× floor,
non-overlapping IQR, T-N PASS) and (b) measured DRAM-byte elimination = 256.2 MiB = 100.1% of the
predicted y[] round trip. The rms_norm→mul fusion's L3 memory-axis benefit is **real on rvv/VLEN128**,
and its magnitude is exactly the byte accounting. → **rollout to other epilogue pairs (silu/scale/mul
SwiGLU gate, bias/residual-add, [FMT-PROP]) is on evidence** —立项 decision remains the user's.

## 6. Framing discipline ([NG-4] / Win ladder — honest scope)
- This is an **isolated A/B on the memory axis** (fused vs our own unfused, same arithmetic) — the
  [NG-4] internal metric, exactly the category as the [GAP-SB] A/B cells. It is **NOT** a beat vs a
  ggml factory kernel (there is no factory here — the opponent is our own unfused two-pass), and
  **NOT** an e2e [PERF-1] eight-gate result. No Win-B / Win-C claimed.
- Contrast with format-micro (compute/latency-bound, loses to ggml): this fusion lives on the **L3
  memory axis**, where a compiler transformation that removes real DRAM traffic **wins on wall time**
  — the transduction is the point, and it is measured, byte-attributed, and bit-exact.
- Kernel-micro of the rms_norm→mul pair; e2e (whole-model attn_norm/ffn_norm on 65 norms/token) is a
  projection (byte_accounting.csv §3d) — not run here. The per-pair silicon fact is what this cell owns.
