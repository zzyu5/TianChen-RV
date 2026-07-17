# Track B auto-lowering — two-board byte-exact FINDING

**Date:** 2026-06-26
**Subject:** Verify the auto-constructed widening int8 dot-reduce body (commit `0ed17da4`,
front-door `RVVReductionSourceFrontDoor` / `--tcrv-rvv-materialize-widening-dot-reduce-source-front-door`)
is silicon byte-exact on two boards with the capability-driven LMUL anchor flip.
**Discipline:** measure-only, no `lib/` change, used existing `build/bin/tcrv-opt` (no rebuild).

## What was verified
The SAME generic `vector.multi_reduction` source (int8→int32 dot-reduce, K=32, ABI `lhs,rhs,acc,out,n`)
auto-emits a **byte-DIFFERENT** RVV body by the `deriveMinimumVLEN` capability fact, and both bodies are
numerically byte-exact vs one shared scalar oracle:
`out[0] = acc[0] + Σ_{i=0}^{31} (int32)lhs[i]*(int32)rhs[i]` (signed i8×i8 widen→i16→i32).

## Toolchain / pipeline (measure path)
- Body emit: `tcrv-opt <front-door MLIR> --tcrv-rvv-materialize-widening-dot-reduce-source-front-door=march=<M> --tcrv-rvv-lower-to-emitc`
- EmitC→C: `mlir-translate(LLVM 21.1.8) --mlir-to-cpp` (the `tcrv-rvv-emitc-to-cpp` export path requires the
  header-artifact emission-plan + its own block-quant `mf4` typed-config resolver — wrong consumer for the bare
  dot-reduce; `lower-to-emitc` + generic `mlir-to-cpp` is the correct body→C route and matches the lit-asserted intrinsics).
- Compiled `g++ -march=rv64gcv -O2` (kernel is `extern "C"` emitc/C++); 16 cases incl. ±127, **-128** (full i8 range),
  -128×127 (most-negative product), nonzero/negative `acc` seeds, 10 deterministic-LCG random rounds.
- Note: at K=32 max |sum| < 2^20 — no i32 overflow, so byte-exact MUST hold; any mismatch = real codegen bug.

## Board A — rvv (native riscv64, VLEN=128, rv64gcv, gcc/g++ 14.2.0)
- objdump/asm confirmed the **e8m2** anchor: `vsetvli ...,e8,m2`, widen `e16,m4`, `vwmul.vv`, `vwredsum.vs`.
  (extra `e8,mf4`/`e32,m1` vsetvli's are gcc auto-vectorizing the scalar oracle, not the kernel.)
- Result (K=32, single chunk): **16/16 BYTE-EXACT**.
- Result (n=64, **2 chunks** — exercises the cross-iteration `out[0]`-round-trip accumulation loop): **8/8 BYTE-EXACT**.

## Board B — k1 (riscv64, VLEN=256, rv64gcv, gcc/g++ 13.2.0, taskset -c 0-3, /data)
- objdump/asm confirmed the **e8m1** anchor (the FLIP): `vsetvli ...,e8,m1`, widen `e16,m2`, `vwmul.vv`, `vwredsum.vs`.
- Result (K=32, single chunk): **16/16 BYTE-EXACT**.
- Result (n=64, **2 chunks** — accumulation loop): **8/8 BYTE-EXACT** (identical i32 to Board A's n=64 across both gcc versions).

## Verdict
- rvv VLEN128 (e8m2 body): byte-exact, no numerical bug. PASS 16/16.
- k1 VLEN256 (e8m1 body): byte-exact, no numerical bug. PASS 16/16.
- Both boards produce identical i32 results, both match the shared scalar oracle byte-for-byte.
- The auto-generated (not hand-written) widening-reduce chain (`vwmul`→`vwredsum_i16mX_i32m1`, acc-seed via
  `out[0]` round-trip) is **silicon-correct on both VLEN tiers**, and the capability-fact LMUL anchor flip
  (e8m2@128 ↔ e8m1@256) is a real byte-different-but-numerically-equal kernel. No bug found.

## Scope honesty
One bounded contraction (signed int8 dot-reduce). K=32 is the task mandate (single VL chunk/board); the extra n=64
run additionally validates the multi-chunk accumulation loop. Proves *correctness* of the auto-lowered body + the
capability LMUL flip on real silicon; it is NOT a perf claim and NOT a general dot-reduce auto-tuner.

## Future-work note (genuine limitation, not just a detour)
The production header-artifact export path `tcrv-translate --tcrv-rvv-emitc-to-cpp` (after
`--tcrv-materialize-emission-plans`) currently **rejects** this `m2` body: its typed-config resolver
(`lib/Plugin/RVV/EmitC/RVVEmitCRouteConfigBinding.cpp`) defaults to the block-quant-linear `mf4` integer-core LMUL
and fails with "requires product-reduction lhs source vector LMUL 'm2' to match selected config LMUL 'mf4'".
Body→C verification here used `--tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp` (correct for the bare
dot-reduce, intrinsics match the lit assertions). Anyone wanting full artifact-export for this auto-lowered kernel
must teach that resolver the front-door's gearbox-selected LMUL — open emitter-maturity item.
