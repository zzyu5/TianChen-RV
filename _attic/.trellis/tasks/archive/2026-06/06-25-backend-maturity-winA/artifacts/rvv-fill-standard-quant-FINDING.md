# rvv fill: standard-quant block-dot vs-ggml·rvv micro (q8_0/q4_0/q4_1/q5_0/q5_1) — FINDING

**Date:** 2026-06-25 (rvv board, serial, rvv-only — k1 untouched per discipline)
**Board:** `ssh rvv` (Sophgo SG2044, RVV1.0, **VLEN128** confirmed `vlenb=16`, 64 cores, zvfh).
`taskset -c 0`, best-of-reps **min** latency (noise only ADDS time → lowest = closest to true).
**Tooling:** existing `build/bin/tcrv-opt` (built 06-25 21:52) — **NO rebuild, no lib/ change**.
Emit host-side: `tcrv-opt <test>.mlir --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`.
Compiled on the board with `clang 18.1.3 -O3 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=fast`.
**byte-exact gate (max_rel_norm vs ggml real `_vl128`) passed before any perf reported.**

Fills Table-1 `vs-ggml·rvv` column for the 5 standard quants (were all empty). Method per
archive coverage findings + the q41-q10 / forward-pass sibling findings (same harness family).

---

## Results — all byte-exact (norm 0.000e+00), VLEN128, min-of-reps

| quant | OURS vle8/wmul LMUL | byte-exact gate | RATIO ggml/ours | verdict |
|---|---|---|---|---|
| **q8_0** | i8**m2** / i16**m4** (full 32-lane strip) | 0.000e+00 | **1.17–1.19×** | **WIN** |
| **q4_0** | i8**mf4** / i16**mf2** (¼-LMUL 8-lane) | 0.000e+00 | **0.21×** | LOSS (~4.8×) |
| **q4_1** | i8**mf4** / i16**mf2** | 0.000e+00 | **0.26×** | LOSS (~3.9×) |
| **q5_0** | i8**mf4** / i16**mf2** | 0.000e+00 | **0.26×** | LOSS (~3.8×) |
| **q5_1** | i8**mf4** / i16**mf2** | 0.000e+00 | **0.29×** | LOSS (~3.4×) |

Stability: q8_0 = 1.167 / 1.190 / 1.190 / 1.193 across 4 reps (stable WIN). The 4 LOSS
ratios are flat across reps (0.208–0.218 / 0.258 / 0.260 / 0.294). All 8 seeds × 6 sizes
agree byte-exact for every quant; `nonzero=1 nonfinite=0` everywhere.

**q8_0 WIN discriminators (mechanism-isolation, per reviewer):**
| variant | RATIO ggml/ours | verdict |
|---|---|---|
| baseline | 1.167–1.193× | WIN |
| ggml `v_zero` HOISTED out of loop | 1.167 / 1.192× | WIN survives → not the zero-mv loop-invariant |
| call order REVERSED (ours first) | 1.200 / 1.201× | WIN survives → not an ordering/cache artifact |

---

## Root cause (single mechanism, grep-confirmed in the emitted .cpp)

The split is **purely the LMUL/strip shape our emitter picks**, not algorithm:

- **q8_0 WIN (~1.17–1.20×, byte-exact; mechanism NOT isolated)** — q8_0 block-dot has **NO
  nibble decode**: plain i8×i8 widen+reduce. Our emit takes the **m2 anchor** = one
  `vle8_v_i8m2(32)` strip + one `vwredsum` per block, the SAME shape as ggml's
  `ggml_vec_dot_q8_0_q8_0` (`vle8m2`/`vwmul_i16m4`/`vwredsum`). At VLEN128 e8m2 VLMAX=32 the
  inner strip loop runs **exactly once** (one m2 strip), so the two are the same shape — yet
  ours is stably faster. **Two discriminators rule out the cheap explanations:** (1) **hoisting
  ggml's `vmv_v_x_i32m1(0,vl)` out of its block loop** (loop-invariant the compiler might not
  lift for ggml's single-expression form) → still **1.167–1.192×**, so it is NOT the zero-mv
  loop-invariant; (2) **reversing the rep-loop call order** (ours timed first) → still
  **1.200–1.201×**, so it is NOT an ordering/cache artifact. The ~1.19× is therefore a REAL
  same-shape edge, but its **mechanism is not isolated** (most likely finer load scheduling /
  explicit-temporary materialization in our emit vs ggml's nested-expression form) — reported
  as WIN with mechanism unclaimed. This is the lone wide-strip standard block-dot.

- **q4_0/q4_1/q5_0/q5_1 LOSS** — all four emit the nibble-decode core at **`vle8_v_i8mf4`
  (¼-LMUL, 8-lane) + `vwmul/vwmacc_i16mf2`**, i.e. **sub-VLMAX 8-lane strips** with per-strip
  `vsetvl` overhead. ggml's shipped kernels use **m1 (16-lane)** halves (q4_0/q4_1:
  `vand/vsrl_u8m1`+`vwmul_i16m2`; q5_0/q5_1: m1→m2 `vcreate`/`vwmul_i16m4`, one `vwredsum`
  over the whole 32-elem block). So ours does ~2× more strips of half the width → 3.4–4.8×
  slower. This is **the recorded N3 "narrow-strip, not LMUL/shape-tuned" emitter-maturity gap**
  (same family as q1_0's 8-lane-group 0.033× and the IQ block-dots), NOT a correctness or
  algorithm-choice issue. Named emitter target: **wide-LMUL (m1/m2) block-dot emit for the
  nibble-decode quants** (lift mf4→m1, halve the strip count).

## Honest framing (standard quants: algorithm-family is REPACK, block-dot is the construct)
For q4_0/q4_1/q5_0/q5_1 the kernel that actually ships in llama decode is the **repack GEVM**
(Table 2), where q4_0 is a real 2.6× e2e WIN (it changes memory movement). These block-dot
cells are the **fallback construct**, measured here only to fill the table honestly — a
block-dot LOSS here does not contradict the repack WIN (different axis). q8_0's block-dot is
itself the lean kernel (no nibble work), so its 1.19× is a genuine wide-strip parity-plus-edge.

## Provenance / equality discipline
- q5_0 / q5_1 fresh emits are **code-identical** (grep-diff, comments aside) to the archived
  `blockdot-bench/q5_{0,1}/ours.cpp` that were already correctness-validated → prior evidence
  carries; the new deliverable is just the vs-ggml ratio.
- ggml refs lifted **verbatim** from `llama.cpp ggml/src/ggml-cpu/arch/riscv/quants.c`
  (q4_0 :222, q4_1 :277, q5_0 :328, q5_1 :382, q8_0 :435), byte-offset addressed; q5_0/q5_1
  take the `vlenb==16` (VLEN128) branch. q5_{0,1} refs reused verbatim from the validated
  archive blockdot-bench.
- ABI note: q4_0 emit uses the **8-arg ggml ABI** (n,s,bs,vx,bx,vy,by,nrc) → wrapped in a
  4-arg adapter (`adapter_q4_0.cpp`, bs=bx=by=0, nrc=1, all unused by the body). The other 4
  emits use the 4-arg ABI directly.

## Not measured / why
- e2e for any of these block-dots: out of scope (task = micro vs-ggml·rvv only). For the
  nibble quants the e2e-relevant kernel is the repack GEVM (already in Table 2).
- LMUL Win-A knob for these: q4_0/q4_1 have N1-flip cells already; q5_0/q5_1 are m1-only
  (nothing to sweep) — unchanged, not this task.

## Artifacts (this dir: `rvv-fill-standard-quant/`)
- `results.log` — consolidated byte-exact + ratio raw log (VLEN probe + all 5 quants).
- `harness.cpp` — parameterized harness (`-DQK -DHWB -DHYB -DXMIN_OFF -DYSUM_OFF -DOURS`).
- `ggml_q{4_0,4_1,5_0,5_1,8_0}.cpp` — ggml `_vl128` refs (verbatim, byte-offset).
- `q{8-0,4-0,4-1,5-0,5-1}-*_ours.cpp` — our emitted kernels (current tcrv-opt).
- `adapter_q4_0.cpp` — 8-arg→4-arg ABI shim for q4_0.
- On rvv: `~/fill-std-quant/` (binaries `bench_q{8_0,4_0,4_1,5_0,5_1}`).
