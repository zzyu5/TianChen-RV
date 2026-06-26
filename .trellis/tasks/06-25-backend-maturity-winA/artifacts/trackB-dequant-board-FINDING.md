# Track B DEQUANT rung — two-board byte-exact FINDING

**Date:** 2026-06-26
**Subject:** Verify the auto-constructed widening int8 dot-reduce **+ runtime-f32-scale dequant**
body (`RVVDequantDotSourceFrontDoor`, merged to main `8d8c4aa0`, impl commit `1a701d6e`,
pass `--tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door`) is silicon
byte-exact on two boards, with the capability-driven integer-core LMUL anchor flip.
**Discipline:** measure-only, **no `lib/` change, used main's `build/bin/tcrv-opt` (no rebuild)**.

> Distinct from the prior `trackB-board-FINDING.md` (the BARE dot-reduce MVP, front-door
> `RVVReductionSourceFrontDoor`). THIS finding is the **dequant rung above it**: the same
> generic vector source plus the `sitofp` + `mulf %scale` + f32-store epilogue, auto-built
> into `load → widening_product → standalone_reduce → DEQUANTIZE → store`.

## What was verified
The SAME generic `vector.multi_reduction + arith.sitofp + arith.mulf %scale + f32 store`
source auto-emits a **byte-DIFFERENT** RVV body by the `deriveMinimumVLEN` capability fact,
and both bodies are numerically byte-exact vs one shared scalar oracle:

    out[0] = (float)( acc[0] + Σ_{i=0}^{31} (int32)lhs[i]*(int32)rhs[i] ) * scale

i.e. signed i8×i8 widen→i16→i32 EXACT integer accumulation, then ONE single-precision
multiply (cast-then-multiply, matching the source's `sitofp` then `mulf` order).

ABI (EmitC-reordered from the source func sig): `(const int8_t* lhs, const int8_t* rhs,
const int32_t* acc, float scale, float* out, size_t n)` — `acc` is a pointer, kernel reads `[0]`.

## Why byte-exact is a HARD bar here (no reassoc escape hatch)
- Integer core is exact: at K=32, |Σ| ≤ 32·127·127 ≈ 5.2e5 < 2³¹ → no i32 overflow; and
  `vwmul` i8×i8→i16 never overflows (−128·−128=16384 < 32767). The acc-seed round-trip is i32.
- The dequant epilogue is **exactly one f32 op** (`(float)i32_sum * scale`). A single multiply
  has NO floating-point reassociation freedom. ⇒ any bit mismatch would be a real codegen bug
  (or an oracle-writing error), never reassociation. So the bar is bit-pattern equality.
- Oracle written to mirror exactly: single-precision, cast-then-multiply, NO double anywhere;
  `scale` passed as a `float`, all scale literals f-suffixed; comparison is raw 4-byte
  (`memcpy`→`uint32_t`) bit equality, NOT `%f`/epsilon.

## Pipeline (measure path, same as prior bare-dot finding)
- Body emit: `tcrv-opt <front-door MLIR> --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=<M> --tcrv-rvv-lower-to-emitc`
- EmitC→C: `mlir-translate (LLVM 21.1.8) --mlir-to-cpp`.
- The two boards' C differs **ONLY** in the LMUL anchor (e8m2/i16m4 vs e8m1/i16m2); the dequant
  epilogue `(float)v22 * v4 → vfmv → vse32` is byte-IDENTICAL across boards (verified by `diff`).
- Cases: 8 scales × (6 integer edge cases + 12 LCG-random rounds) = **144 cases per board**.
  Scales deliberately include non-power-of-2 + negative (`0.0123456f`, `3.1415927f`, `-0.0078125f`,
  `-7.7777f`, tiny `0.000031f`, large `123456.75f`) to exercise mantissa rounding of the f32 multiply;
  pow-of-2 (`1.0f`, `0.5f`) included as trivial controls. Integer edges: all +127, −128×127 (most-neg
  product), all −128, +127 with acc=100000, −1×1 acc=−50, zero-dot acc=7. K=32 (single VL chunk) = task mandate.

## Board A — rvv (native riscv64, VLEN=128, rv64gcv, g++ 14.2.0)
- objdump of the kernel confirmed the **e8m2** anchor + dequant epilogue:
  `vsetvli e8,m2` → `vle8.v` ×2 → `vwmul.vv` (`e16,m4`) → `vwredsum.vs` (i16m4→i32m1) →
  `vmv.x.s` (extract i32) → `vfmv.s.f` + `vse32.v` (f32 store epilogue).
  (gcc lowered the EmitC `__riscv_vfmv_v_f_f32m1` to `vfmv.s.f` here — semantically equal for the
  VL=1 single-f32 store; the float cast+multiply fold into scalar FPU ops before vfmv. The stray
  `e32,m1`/`vsetivli` are the VL=1 dequant config, not a different LMUL.)
- Result: **144/144 BYTE-EXACT** (bit-pattern compare).

## Board B — k1 (riscv64, VLEN=256, rv64gcv [+ime], g++ 13.2.0, taskset -c 0-3, /data)
- objdump confirmed the **e8m1 FLIP** + dequant epilogue:
  `vsetvli e8,m1` → `vle8.v` ×2 → `vwmul.vv` (`e16,m2`) → `vwredsum.vs` (i16m2→i32m1) →
  `vmv.x.s` → `vfmv.v.f` (k1's gcc kept the exact EmitC `vfmv.v.f` form) + `vse32.v`.
- Result: **144/144 BYTE-EXACT** (bit-pattern compare). Each board is compared against its own copy of
  the shared deterministic oracle; cross-board f32 bit patterns are therefore expected-identical (same
  correctly-rounded IEEE-754 single `(float)sum*scale`, no FMA-contractible add) — an inference, not a direct cross-board diff.

## Verdict
- rvv VLEN128 (e8m2 dequant body): byte-exact, **PASS 144/144**, dequant epilogue present, no bug.
- k1  VLEN256 (e8m1 dequant body, capability flip): byte-exact, **PASS 144/144**, dequant epilogue present, no bug.
- Both boards produce identical f32 bit patterns and both match the shared scalar oracle byte-for-byte.
- The auto-generated (no per-kernel emitter) dot+dequant chain — `vwmul`→`vwredsum_i16mX_i32m1`,
  acc-seed via `acc[0]` round-trip, then `(float)i32 * scale` → `vfmv`/`vse32` — is **silicon-correct
  on both VLEN tiers**, and the capability-fact LMUL anchor flip (e8m2@128 ↔ e8m1@256) is a real
  byte-different-but-numerically-equal kernel. **No bug found.**

## Scope honesty
One bounded dequant contraction: signed-i8 dot + ONE runtime-f32 scale (q8_0-style integer core
`scale*(acc+Σi8·i8)`). This is NOT the per-block fp16-scale ggml `q8_0_q8_0` block-dot KERNEL
(no `nb = n/QK` block loop, no per-block `d_x·d_y` fp16 reads) — that is a later rung, as the lit
header's HONEST SCOPE states. Proves *correctness* of the auto-lowered dequant body + the capability
LMUL flip on real silicon; it is NOT a perf claim and NOT a general auto-tuner.

## Note (inherited limitation, unchanged from bare-dot finding)
Verification used `--tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp` (the correct body→C
route; intrinsics match the lit assertions). The production header-artifact export path
`tcrv-translate --tcrv-rvv-emitc-to-cpp` still rejects this m2 body (its typed-config resolver
defaults to block-quant-linear `mf4`) — the open emitter-maturity item is teaching that resolver
the front-door's gearbox-selected LMUL. Not exercised here (out of scope for board byte-exactness).
