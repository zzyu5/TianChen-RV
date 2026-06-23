# N1 breadth: q4_1 repack GEVM (matrix-path coverage +1) — FINDING

**Date:** 2026-06-23
**Deliverable:** extend the RVV matrix (repack GEMM/GEMV) path beyond q4_0-ONLY to a
second quant type. **Done = q4_1 repack GEVM (decode), GEMM deferred.**

## ⚠️ Lead-merge note (read first)

This worktree was created from a STALE base `c70febc6` that predated ALL the q4_0
repack / `RVVToEmitCBlockQuantLinear.cpp` machinery the task builds on (the base was
**1169 commits behind** the local `main` ref). The q4_0 template literally did not
exist here. Recovery: I **fast-forwarded the worktree branch ref**
`worktree-agent-aa5c5c424bd1a4ef9` from `c70febc6` to the local `main` tip
`22c844a2` (a clean fast-forward — the worktree had NO commits of its own; verified
`git merge-base --is-ancestor HEAD main`). My q4_1 changes are uncommitted
working-tree edits on top of `22c844a2`.

**Discrepancy to flag:** the session's initial `gitStatus` listed main's tip as
`3abc9dc5`, but the live `main` ref in this checkout resolves to `22c844a2` (both are
recent N1/N3 commits, different SHAs). The lead should confirm which base to merge my
q4_1 diff onto. My diff is 6 files (below), all q4_1-repack-GEVM-specific.

## What was added (file:line, all worktree-absolute under repo root)

| Piece | File | Location |
|---|---|---|
| ODS op `GgmlRepackGemvQ41Q81Op` (`tcrv_rvv.repack_gemv_q4_1_q8_1`) | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | def at **4377** |
| Fail-closed verifier `GgmlRepackGemvQ41Q81Op::verify()` (I7) | `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` | **1750–1938** |
| EmitC emitter `emitRepackGemvQ4_1Q8_1` (structured, raw()==0, I5) | `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | **3167–3681** |
| Emitter + recognizer decls | `lib/Conversion/RVV/RVVToEmitCInternal.h` | emitter decl ~1068, recognizer decl ~134 |
| Recognizer `isRepackGemvQ4_1Q8_1Body` + dispatch-table entry | `lib/Conversion/RVV/RVVToEmitC.cpp` | dispatch **349–350**, recognizer **1095** |
| Gearbox strip-width branch (VLEN→half_lanes / RVV0.7→m1) | `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp` | q4_1 branch at **150** |

### What the kernel does (Family B = scale + MIN, asymmetric)
Mirrors the q4_0 repack GEVM block-as-lane structure (16 weight rows as 16 vector
lanes, dot accumulates LANE-WISE via `vwmacc`, NO cross-lane vredsum — the per-block
reduction wall is gone) but exercises the emitter's generality with TWO q4_1-specific
differences, validated against the in-tree `GgmlBlockDotQ41Q81Op` fold:

1. **UNSIGNED nibble decode** `[0,15]`: `vand 0x0F` (lo) / `vsrl 0x04` (hi) ->
   `vreinterpret` to i8, NO offset-binary `^0x88` bias (q4_1 stores RAW nibbles; the
   bias lives in the per-block MIN scale). Unlike q4_0's `vsll/vsra` sign-extend.
2. **Per-block MIN correction, folded LANE-WISE**: `block_q4_1x16` = `{fp16 d[16];
   fp16 m[16]; uint8 qs[256]}` (stride **320**, d at +0, m at **+32**, qs at **+64**);
   `block_q8_1` = `{fp16 d; fp16 s; int8 qs[32]}` (stride **36**, s at **+2**, qs at
   **+4**). ggml's fold `sumf += (d_x*d_y)*sumi + m_x*s_y` becomes, in the lane-wise
   form, `vfmacc` of the scale term (d_x vector strip × d_y scalar × sumi) then
   `vfadd` of the MIN term `vfwmul(m_x_strip, s_y_scalar)` — d_x/m_x are VECTOR strips
   (one lane per row), d_y/s_y are the single activation column's SCALARS.

## Status

- **GEVM: DONE. GEMM: deferred** (decode GEVM is the requested target; the +min
  term + lane-wise fold proved tractable in one pass, no fallback to q8_0 needed).
- **Build: GREEN.** Fresh CMake (LLVM/MLIR 20, Ninja) + `tcrv-opt` + full default
  targets + `tcrv-translate`, all exit 0. (Forced clean configure in the worktree's
  own `build/` — does NOT touch the shared checkout build.)
- **Lit: GREEN.** 3 new tests PASS; all 375 RVV lit tests PASS; project-wide 688/691.
  - `test/Dialect/RVV/repack-gemv-q4-1-q8-1-dataflow.mlir` — op round-trip + 8
    fail-closed verifier negatives (wrong kind / weight stride 320 / act stride 36 /
    MIN offset 32 / sum offset 2 / scale model / off-grid half_lanes=12 / m1+half8).
  - `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-1-q8-1.mlir` — VLEN=128 two-strip
    materialization (unsigned decode + lane-wise MIN fold pinned, NOWALL: no redsum).
  - `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-1-q8-1-vlen256.mlir` — the
    strip-width gearbox pass (`--tcrv-rvv-materialize-repack-strip-width=
    march=rv64gcv_zvl256b`) re-stamps half_lanes 8→16, ONE 16-lane strip.
  - **The 3 project-wide failures are PRE-EXISTING base-state failures**, NOT
    regressions: all are `scripts/rvv_generated_bundle_abi_e2e.py --self-test`
    (Python tooling, "product-reduction dequant metadata" assertion) — a codepath my
    6-file diff does not touch (no `scripts/`/`.py` in the diff). Reproduced
    identically by running the script directly on the clean base.
- **I5 (structured EmitC, raw()==0):** confirmed — `grep -c raw(` on the emitted
  emitc MLIR = **0**. The kernel uses `__riscv_*` intrinsics, so raw()==0 is
  achievable (unlike the IME family).

## Numeric verdict — REAL K1 silicon (RVV1.0, VLEN=256, Spacemit X60), clang18

Validated on `ssh k1` (the rvv board was busy with another agent; K1 is the
preferred N1 cross-profile target anyway). Built with
`clang++ -O3 -march=rv64gcv_zfh_zvfh -mabi=lp64d -ffp-contract=fast -fno-integrated-as
-fuse-ld=bfd` (K1's gcc13 does NOT expose `vfloat16m1_t`; clang18 does — same toolchain
constraint the q4_0 GEVM validation hit). taskset -c 0.

### Oracle framing (HONEST — this is norm-based, NOT byte-exact ggml parity)
There is **no upstream ggml q4_1 *repack* kernel** to diff against — `block_q4_1x16`
is an extrapolation from ggml's `block_q4_0x16` + the q4_1 min field. So validation is
norm-based against TWO independent oracles, **not** byte-exact parity with a shipped
ggml kernel:

1. **Hand-derived scalar q4_1 reference** (`verify_emitted_gemv_q4_1.cpp`): the exact
   unsigned-nibble MAC + dual-scale fold, dotting the QUANTIZED activation. The
   reference uses ggml's grouping `sumf + (scale·sumi + min)` while the emitter folds
   `(sumf + scale·sumi) + min` — equal in exact arithmetic, ~fp-rounding apart (which
   is why norm sits at ~6e-6, not 0). All 4 column widths PASS (bar = norm < 1e-4):

   | NC | groups | norm | rel | |
   |---|---|---|---|---|
   | 16 | 1 | 5.71e-6 | 7.4e-4 | PASS |
   | 32 | 2 | 6.08e-6 | 2.0e-3 | PASS |
   | 64 | 4 | 6.36e-6 | 1.2e-3 | PASS |
   | 336 | 21 | 9.52e-6 | 1.6e-3 | PASS |

   Both strip arms (half_lanes=8 two-strip AND half_lanes=16 one-16-lane-strip)
   produce byte-identical results — confirms capability-driven strip SELECTION
   preserves correctness for the new kernel.

2. **Cross-check vs the SEPARATELY-AUTHORED in-tree `GgmlBlockDotQ41Q81Op`**
   (`crosscheck_repack_vs_blockdot_q4_1.cpp`): dots each of the 16 weight rows with
   the pre-existing, separately-validated q4_1 block-dot kernel and matches the 16
   lane-wise repack-GEVM outputs. This is the circularity-killer — it diffs the new
   kernel against DIFFERENT compiler-emitted code, not my hand-written reference.
   **norm = 5.05e-6, PASS.** The q4_1 arithmetic (unsigned decode, low/high pairing,
   `(d_x·d_y)·sumi + m_x·s_y` with `s_y = d_y·Σq8`) is thereby corroborated by code I
   read but did not author.

## Artifacts
Under `artifacts/kernel-coverage/q4_1-emit/`: `input-repack-gemv-q4_1.mlir`,
`emitted.emitc.mlir`, `emitted-repack-gemv-q4_1.cpp` (+ `-h16`),
`emitted-blockdot-q4_1.cpp`, the adapter / verify / crosscheck harnesses.
K1 build dir: `k1:~/q41-verify-agent-aa5c/`.

## Honest caveat
This is a **kernel-coverage increase** (N1 breadth: q4_1 joins q4_0 on the matrix
repack path) + the new kernel participating in the SAME N1 capability-divergence
strip-selection gearbox — NOT a new perf claim. GEMM (prefill) was deferred (GEVM
was the requested target and the deliverable). The numeric agreement is norm-based
fp-rounding agreement against two independent oracles, not byte-exact parity with a
shipped ggml q4_1 repack kernel (none exists upstream).
