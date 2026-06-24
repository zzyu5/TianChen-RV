# q4_K x q8_K 16x1-REPACKED GEVM — numeric oracle (mf2/VLEN128 re-derivation) — FINDING

**Date:** 2026-06-24
**Scope:** the GATING correctness step for the q4_K repack GEVM emitter
`tcrv_rvv.repack_gemv_q4_K_q8_K` (committed on main `710ad067`, structurally complete but
numerically UNVERIFIED). Does our compiler-emitted q4_K GEVM kernel compute the CORRECT
q4_K dequant-matmul result on the real SG2044?
Measured on `ssh rvv` (Sophgo SG2044, RVV1.0, **VLEN128**, clang18 with zvfh). `taskset -c 0`.
Methodology per `N3-METHODOLOGY.md`: this is a CORRECTNESS check (norm), **NOT a speedup** —
a scalar oracle is not a methodology-valid speed baseline.

## VERDICT — PASS (correct). The q4_K emitter is now NUMERICALLY VALIDATED.

The compiler-emitted `tcrv_emitc_..._ggml_repack_gemv_q4_K_q8_K` matches an INDEPENDENT scalar
q4_K dequant-matmul reference within fp32 rounding across **all 8 shapes**. **WORST_NORM = 7.07e-7**,
bar = norm < 1e-4. This upgrades the q4_K GEVM from "structurally complete, numerically UNVERIFIED"
to "**correct**." No kernel code modified; the kernel is the frozen on-main emit.

| shape (nc × n) | max_abs_err | rms(ref) | **norm** | rel | verdict |
|---|---|---|---|---|---|
| 16 × 256    | 1.526e-05 | 9.758e+01 | **1.564e-07** | 1.17e-07 | PASS |
| 16 × 4096   | 6.104e-05 | 2.875e+02 | **2.123e-07** | 1.15e-06 | PASS |
| 16 × 11008  | 2.441e-04 | 6.323e+02 | **3.861e-07** | 4.76e-07 | PASS |
| 32 × 256    | 7.629e-06 | 5.930e+01 | **1.287e-07** | 1.24e-07 | PASS |
| 32 × 4096   | 6.104e-05 | 2.511e+02 | **2.430e-07** | 1.07e-06 | PASS |
| 256 × 256   | 1.526e-05 | 6.674e+01 | **2.286e-07** | 1.76e-07 | PASS |
| 256 × 4096  | 2.441e-04 | 8.369e+02 | **2.917e-07** | 2.47e-07 | PASS |
| 256 × 11008 | 3.662e-04 | 5.183e+02 | **7.065e-07** | 1.47e-04 | PASS |

Norm sits at ~1e-7 (slightly TIGHTER than q4_1's ~6e-6 and q8_0's ~1e-6). That is expected, not
a red flag: q4_K's integer dot is exact, and the only fp delta is (a) the fp16 d/dmin scale strips
and (b) the kernel's f32 block-fold over nb≤43 blocks vs the reference's double accumulation. The
`rel` column up to 1.47e-4 (256×11008) is a near-zero-denominator artifact on near-cancellation
output rows (same treatment as q4_1's `rel`); `norm = max_abs_err / rms(ref)` is the correct
aggregate and it is clean (≤7.07e-7).

## Why this is a TRUSTWORTHY gate, not just a small number (the two false-PASS risks closed)

q4_K's mf2/VLEN128 path is a re-derivation (ggml's `ggml_gemv_q4_K_16x1_q8_K` is VLEN≥256, one
16-lane `m2` strip — it cannot even run correctly on this VLEN128 part), so the oracle had to be
robust against the two ways a q4_K test gives a FALSE PASS. Both were closed with QUANTIFIED margin
(negative controls on shape 256×4096; the controls perturb the REFERENCE — a spike proves the
correct test would catch the corresponding KERNEL bug):

- **Min-term / bsums path (risk: zero-mean activation → bsums≈0 → the `dmin·m·bsums` correction is
  multiplied by ~0, hiding a min-term sign/index bug).** Mitigated by injecting a per-16-group
  DISTINCT nonzero DC offset on the activation. Confirmed live: **mean|bsum| = 186.7, max|bsum| = 716**.
  **NOMIN control** (zero the min term in the ref) shifts the reference by **norm 1.156e-1** (~11.6% of
  output) → the min path is exercised at **~396,000× margin** over the true-ref norm. A min-term sign
  flip or wrong bsums index would spike at ~1e-1, far above the 1e-4 bar.
- **Sub-block scale pairing (risk: iid weights → the 8 sub-block 6-bit scales are nearly equal → a
  sub-block permutation/offset bug — the "every 8th element wrong" class q4_0/q4_1 never had to test
  — cancels).** Mitigated by giving each of the 8 sub-blocks a DISTINCT 6-bit scale (and min) code.
  **PERM control** (rotate the per-sub-block scale assignment by +1 in the ref) shifts the reference by
  **norm 6.07e-1** (~61% of output) → the distinct-sub-block-scale pairing is exercised at **~2,080,000×
  margin**. A sub-block-offset bug would spike at ~6e-1.

So the 1e-7 PASS is not "small because the test is weak" — the two highest-risk paths of the mf2
re-derivation are demonstrably driven HARD, and the kernel still matches to fp-rounding.

## Oracle framing (HONEST — independent reference, stronger than the task's literal wording)

The reference is an INDEPENDENT scalar q4_K dequant-matmul computed from the **ORIGINAL pre-repack
per-block `block_q4_K`** weights (the task allowed consuming the repacked `block_q4_Kx16`; computing
from the originals is strictly MORE robust — a repack-inversion bug in the kernel cannot agree with a
reference that never inverts the repack). The reference:
- dequantizes each original q4_K block via ggml's exact `dequantize_row_q4_K` semantics
  (`get_scale_min_k4` 6-bit unpack + `d·sc·nibble − dmin·m` per sub-block, verbatim from
  ggml-quants.c:822 / :1471), so the **min/bsums correction is tested IMPLICITLY and EXACTLY** —
  `−dmin·m` is baked into each dequantized weight float, never hand-derived;
- dots those weight floats with the q8_K int8 quants scaled by the activation fp32 `d`;
- feeds the kernel the SAME 16 original blocks per column-group repacked via `make_block_q4_Kx16`
  (verbatim from repack.cpp:2913, `blck_size_interleave=1` — the only non-asserting branch:
  `out.qs[i]=in[i%16].qs[i/16]`, RVV scales/mins repack into the 192-byte region).

The kernel and reference therefore consume the SAME quantized data in DIFFERENT layouts; agreement is
real, not two copies of one inversion bug agreeing. fp16 fidelity: d/dmin are stored as native
`_Float16` in BOTH the constructed blocks and the kernel's `vle16_v_f16m1`+`vfwcvt` load, so the fp16
round-trip is bit-identical to the hardware path (no ggml fp16 table needed).
`out` is zero-init per trial → any output element the kernel failed to write reads 0 against a ref
~O(10–800) and would spike the norm; norm ~1e-7 confirms the FULL nc output is correct, not
spot-checked. Per-column-DISTINCT weight data (every column/block built distinct) also exercises the
lane→column mapping of the repack. The ggml VLEN≥256 `ggml_gemv_q4_K_16x1_q8_K` SIMD cross-check was
correctly ruled OUT — it requires one 16-lane strip (VLMAX≥16, VLEN≥256) and would mis-compute on the
VLEN128 SG2044; the scalar oracle is the primary and sufficient gate.

## Block layout confirmed (what the kernel reads, matched by the oracle)

- weight `block_q4_Kx16`, stride **2304**: `d[16]` fp16 @0..31, `dmin[16]` fp16 @32..63,
  `scales[192]` @64..255 (RVV 6-bit scale/min repack), `qs[2048]` @256.. (nibbles, interleave=1).
  The kernel reads the low scale bytes at +64/+72/+80/+88/+96/+104/+112 and high bytes at
  +192/+208/+224/+240 — the two-8-lane-half (mf2) split of ggml's 64-byte j-group region.
- activation `block_q8_K`, stride **292**: d FLOAT @0, qs[256] int8 @4, bsums[16] int16 @260.
- per q4_K math: `out[col] = Σ_l d_y · ( d_x · Σ_sb scale_sb·Σ_i(nibble·q8) − dmin_x · Σ_sb min_sb·bsum_sb )`
  — the integer core is EXACT; only the fp scale-fold rounds → norm ~1e-7, not 0.

## How the kernel was produced (compiler-emitted, host tool — NO hand edits)

Local host (x86, LLVM/MLIR 20):
`build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-K-q8-K.mlir
--tcrv-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp` → `k_gemv_q4K.cpp`
(single `tcrv_rvv.repack_gemv_q4_K_q8_K` op; RVV1.0 leaves `integer_core_lmul` unset → the mf2
fractional chain, `half_lanes=8`, i.e. the 2×8-lane-strip path that is exactly what runs on the
SG2044). Emitted `.cpp` scp'd to rvv, compiled with `clang++-18 -O3 -march=rv64gcv_zvfh
-mabi=lp64d -ffp-contract=fast` (zvfh for the fp16 d/dmin scale strips).

## Artifacts
- `q4_K-repack-oracle.log` — canonical rvv log: 8-shape norm table + bsums diagnostic + NOMIN/PERM
  control magnitudes + WORST_NORM/VERDICT.
- `q4_K-emit/oracle_gemv_q4K.cpp` — the harness (independent scalar dequant-matmul reference,
  `make_block_q4_Kx16` repack verbatim, adversarial input builder, `--controls` negative-control mode).
- `q4_K-emit/k_gemv_q4K.cpp` — the compiler-emitted mf2/VLEN128 kernel under test (provenance above).
- On rvv under `~/q4k-oracle-agent/`: same sources + binary `oracle_q4k` (`clang++-18 -O3
  -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=fast`).
- Caveat (carried from q4_1/q8_0): the SG2044 binutils `objdump` does not decode RVV vector mnemonics
  (emits `.word`), so no binary vsetvli histogram. Correctness is established by the independent-ref
  norm + the two quantified negative controls, which is the gate that matters here.
