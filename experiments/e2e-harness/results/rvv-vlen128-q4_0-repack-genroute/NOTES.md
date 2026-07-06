# q4_0 repack — generation-vs-routing phase split (rvv VLEN128, 2026-07-06)

Step-2 of the on-board B+step-2 session. Question: for the q4_0 repack e2e story,
**which phase runs a kernel our compiler GENERATED (front-door constructed) vs a
kernel that is merely compiler-EMITTED from a hand-authored op or vendor-routed?**

## Compiler-side facts (repo @539d4eca, refactor/full-refactor-m1)

| repack route | dialect op | front-door CONSTRUCTED? | full-pipeline export | silicon bit-exact (batch-1) |
|---|---|---|---|---|
| **GEVM** (decode, M=1) | `tcrv_rvv.typed_repack_gemv_loop_body` | **YES** — `--tcrv-rvv-lower-quant-contraction` (capability path-selection) realizes it via `lowerToRepackGemv` | `q4-0-q8-0-repack-gemv-full-pipeline-export-e2e.mlir` | YES (FMA-fold, <1e-3 rel, ≥ggml-accurate) |
| **GEMM** (prefill, M>1) | `tcrv_rvv.repack_gemm_q4_0_q8_0` | **NO** — no path-selection realizes GEMM; `lower-quant-contraction` is GEVM-only. The GEMM op is **hand-authored** in `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-0-q8-0.mlir` (hand-written `tcrv.exec.kernel` + `runtime_abi_value`s) and only lowered `--tcrv-rvv-lower-to-emitc`. | (none) | (not in batch-1) |

So on the **generation grid**, only the **GEVM (decode)** is front-door CONSTRUCTED.
The **GEMM (prefill)** is compiler-EMITTED from a hand-placed op — "dispatch-wired,
not constructed".

## Deployment on the board (A tree = `tcrv-llamacpp/build-gcc15-rv64gcv`)

`ggml/src/ggml-cpu/arch/riscv/repack.cpp` `#include`s two emitted kernels and
ENGAGES them at VLEN128 (proven by runtime banners, this session):
- prefill `ggml_gemm_q4_0_16x1_q8_0` -> `TCRV EMITTED GEMM(...) ENGAGED nr=.. nc=.. nb=..`
- decode  `ggml_gemv_q4_0_16x1_q8_0` -> `TCRV EMITTED GEMV(...) ENGAGED nc=.. nb=..`

Stock B tree (`llama.cpp-upstream-native`): VLEN128 refuses the CPU_REPACK buffer
-> falls back to `vec_dot_q4_0_q8_0` **block-dot** (no TCRV banner). So the opponent
at VLEN128 is genuinely the stock **block-dot** factory path (a legitimate B1-class
comparison — this board cannot repack in stock).

### Provenance nuance on the DEPLOYED decode kernel
The A-tree `tcrv_emitted_repack_gemv.inc` symbol is
`tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_...` with ABI `(n,s,vx,vy,ncols)`, emitted
from the **monolithic** `tcrv_rvv.repack_gemv_q4_0_q8_0` op — NOT the current
front-door-CONSTRUCTED `typed_repack_gemv_loop_body` (ABI `(n,s,ncols,vx,vy)`, the
one silicon-validated in batch-1). They are the same 16x1/288B q4_0 GEVM and
numerically equivalent, but the deployed one carries pre-construction (monolith)
provenance.

## Honest phase-relation summary
- **decode phase**: runs OUR compiler-emitted GEVM (16x1/288B), numerically the
  front-door-constructed kernel — GENERATION side (with the monolith-provenance
  caveat above).
- **prefill phase**: runs OUR compiler-emitted GEMM, but that GEMM is **not
  front-door constructed** (hand-authored op, no path-selection). It is emission,
  not generation-grid construction.
- A clean "our GENERATED repack delivers ~5x prefill" claim therefore requires the
  **remaining surgical step: construct the repack GEMM through the front-door
  generation grid** (a `lowerToRepackGemm` path-selection + full-pipeline export),
  NOT dressing up the GEVM as the prefill kernel. GEVM (decode) and GEMM (prefill)
  must not be conflated.

## Sealed e2e A/B (constitutional protocol) — BASELINE (A tree as-shipped: monolith GEVM decode + hand-authored GEMM prefill)
Raw `phase_split_raw.txt`; aggregate `aggregate_baseline.txt` / `evidence_baseline.json`.
Freq locked 2.6 GHz (DVFS span 0.00%). tokens/s, higher=faster; ratio=median(ours)/median(stock).

| phase   | ours median (t/s) | stock median (t/s) | ratio (ours/stock) | 95% CI        | verdict |
|---------|-------------------|--------------------|--------------------|---------------|---------|
| prefill | 19.69 (IQR 0.26%) | 3.90 (IQR 0.01%)   | **5.045×**         | [5.041,5.056] | DIFFERENCE |
| decode  | 3.13 (IQR 1.05%)  | 2.00 (IQR 0.27%)   | **1.564×**         | [1.554,1.575] | DIFFERENCE |

Opponent = stock **block-dot** (VLEN128 stock cannot repack). Both verdicts pass 2×-floor
& CI-excludes-1 by huge margins (Δ 404%/56% vs floors 0.5%/2.1%).
Completeness caveat: the ssh session had a 10-min cap that cut stock's pass-2, so the
baseline is ours×2-pass (n=10) / stock×1-pass (n=5). Verdict robust to this (stock IQR
0.01-0.27%; margins >> any plausible between-pass floor).

## Framing (locked)
**Kernel-level parity vs ggml block-dot is the pass bar** (blank-cell: the win is
path-EXISTENCE at VLEN128, not per-instruction supremacy). The prefill 5.045× is our
compiler-EMITTED GEMM vs stock block-dot — NOT a clean "GENERATED repack delivers 5×"
claim, because that GEMM is not front-door constructed (see above). Decode 1.564× is our
GEVM (front-door constructed) vs stock block-dot.

## Constructed-GEVM redeploy (does decode change?)
Rebuilt the A tree with the front-door-CONSTRUCTED `typed_repack_gemv_loop_body`
(silicon-batch-1 kernel) swapped into `tcrv_emitted_repack_gemv.inc` via a thin ABI shim
(reversible; `.orig` kept). Constructed GEVM ENGAGES at decode (banner confirmed).
Correctness gate + decode re-measure: `redeploy_correctness.txt` / `redeploy_aggregate.txt`
/ `evidence_redeploy.json` / `redeploy_phase_split_raw.txt`.

**RESULT 1 — correctness (clean, load-independent): GREEN.** The front-door-constructed
GEVM, deployed in-situ, is greedy-token-consistent with stock ggml on 3/3 prompts, no
NaN/Inf. So the CONSTRUCTED decode kernel is correct end-to-end in the real llama.cpp
pipeline (a stronger statement than the isolated silicon bit-exact of batch-1).

**RESULT 2 — decode throughput "did it change?": NOT cleanly attributable (shared-board
load confound), and unchanged by construction.**
| phase   | baseline (monolith GEVM) ratio | redeploy (constructed GEVM) ratio | stock t/s baseline→redeploy |
|---------|-------------------------------|-----------------------------------|-----------------------------|
| prefill | 5.045×                        | 5.530×                            | 3.90 → 5.24 |
| decode  | 1.564×                        | 1.950×                            | 2.00 → 4.73 |

The two sessions are NOT comparable: the **prefill kernel (GEMM) was byte-IDENTICAL in
both builds** (only the decode GEVM `.inc` changed), yet the prefill **ratio drifted
5.045×→5.530×** and stock's untouched block-dot decode moved 2.00→4.73 t/s. That proves
the shared board's memory-bandwidth availability differed between sessions and moves BOTH
absolute t/s AND the ours/stock ratio. So the decode ratio change (1.564×→1.950×) CANNOT
be attributed to the GEVM swap — the confound dominates.

Since the constructed GEVM is byte-equivalent to the monolith it replaced (same 16x1/288B
q4_0 GEVM math; correctness GREEN), decode is **unchanged by construction**; no
kernel-attributable delta is expected or measurable here. Cleanly measuring a
kernel-attributable decode delta would require a SAME-SESSION paired A/B of the two GEVM
variants (monolith-build vs constructed-build interleaved) — not run this session.

Methodological note for the harness: on this shared board, even the constitutional
WITHIN-session ours/stock ratio is not stable ACROSS sessions (the invariant prefill-GEMM
ratio drifted). Cross-session ratio comparison is therefore not admissible; kernel-variant
comparisons must be same-session paired.

Board left as found: constructed `.inc` reverted to the original monolith emit + rebuilt
(engagement re-confirmed).
