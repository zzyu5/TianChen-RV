# G6-B · [GAP-EMIT-UNROLL] · rolled-loop emission mode (q2_K repack GEMM) — Phase 1

- **campaign**: G6-B emit-quality lever — positive answer to the q2_K loss cause ([K-7] 5th first-emission lever)
- **board**: rvv / openEuler / riscv64 / VLEN128 / 64c · compiler: native clang-17.0.6
- **date**: 2026-07-13 · **status**: Phase-1 COMPLETE · board byte-exact GREEN · NOT committed (main session commits)
- **scope**: build the rolled-loop EMISSION mode (capability-keyed) + byte-exact hard gate + kernel-axis FORM seal. Perf verdict = Phase 2 (deferred).

## X-0 前置 — bottleneck re-confirmed on disassembly (性能宪章规则 1)
q2_K@k1 e2e prefill **0.873× LOSS** (evidence `experiments/active/g5-wiring/M2-q2_K-k1-e2e/`) roots in an
**emit-quality gap** (correctness GREEN, not gcc-death, not env): OUR shipped S6-tiled q2_K GEMM
**FULL-STATIC-unrolls** the per-16-weight-group main term → **2304 static vwmacc, ~27KB register-resident
text**, losing to the SpacemiT hand-brick's **compact rolled loop (16 vwmacc)**. Same root as q6_K's
2995-vsetivli giant. This is [T8·LAW-FIRST-EMISSION] (逐片段发射仪式开销 = 全展开 code volume).
**Lever = a rolled-loop emission MODE**: materialize the dominant inner loop as a runtime `emitc.for`
instead of unrolling.

## Implementation (emitter domain + ODS/verifier — 4 tracked files, reversible)
1. **ODS knob** `include/Weft/Dialect/RVV/IR/RVVOps.td`: `OptionalAttr<StrAttr> emit_loop_schedule`
   ("unrolled"|"rolled") on `TypedRepackGemmLoopBodyOp` — the schedule *how*, never the *what*
   (P2c `fold_structure` precedent). ABSENT ⇒ capability-derived default.
2. **verifier** `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (`TypedRepackGemmLoopBodyOp::verify`):
   bounds the attr to {"unrolled","rolled"} fail-closed (I7). Verified: `"bogus"` → op error.
3. **capability-keyed selection** `resolveRepackMainTermRolled` (`RVVToEmitCBlockQuantLinear.cpp`):
   key hierarchy = (1) explicit `emit_loop_schedule` stamp [A/B forcing override + capability policy pin]
   → (2) else CODE-VOLUME-vs-Icache-budget derived default. The DISCRIMINANT capability fact is **code
   volume**, NOT register pressure (the S6 body is already ≤32 vreg). **Phase-1 default FROZEN to
   unrolled** (no shipped-behavior change) pending Phase-2 e2e budget calibration; the derivation
   function is fully wired (computes the volume).
4. **rolled emit path** `emitRepackKQuantGemmBodyQ2K` (same file, `bool rolledMainTerm`): the per-16-
   weight-group main-term loop → ONE runtime `emitc.for` with the per-column-per-shift i16 partials
   staged to a stack panel (load-accumulate-store), instead of full static unroll. mh/k/j folds
   unchanged. **Byte-exact by construction**: identical integer vwmacc accumulation order (mm asc, j, c);
   end-of-block f32 fold untouched.

Structural proof (weft-opt lowering): default main term = 3 outer `for` + full unroll (16213 emitted ops,
25673 MLIR lines); ROLLED = **4 nested `for`** (the +1 inner mm loop) + **5717 ops / 8889 lines** (2.84×
fewer ops). lit `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q2-K-q8-K-rolled.mlir` GREEN; default fixture
+ NOWALL still GREEN (no regression).

## Byte-exact HARD GATE — GREEN (board, clang-17, seed 0xC0FFEE) · raw: `raw/board_seal_clang17.txt`
Two independent proofs (ZERO-MODEL discipline):
- **INDEPENDENT scalar oracle** (`kquant_repack_verify_q2K`, decodes original pre-repack blocks, isum−summs
  fold): ROLLED **INT_mismatch_total=0** all 8 shapes (GEVM nr=1 + GEMM nr∈{4,8,16}), NORM worst_ulp=3402
  worst_norm=6.618e-07 — **identical to UNROLLED to the digit**.
- **direct A/B identity** (`g6b_q2k_identity_ab.c`, both schedules on same random input, memcmp full fp32
  output): **ROLLED == UNROLLED byte-for-byte, 43008/43008 bytes, 0 mismatch**, both INT and NORM (full
  fp16 scale/min fold) paths.

## Kernel-axis FORM seal (compiler-SYMMETRIC clang-17 A/B — both sides OUR emit, no opponent)
| form | vsetvli | spill | reload | maxVreg | vwmacc | textB |
|---|---:|---:|---:|---:|---:|---:|
| UNROLLED (== shipped S6 form, l1-t3 provenance) | 86 | 7 | 12 | v30 | 2304 | 26968 |
| **ROLLED** | 82 | 63 | 73 | v31 | **384** | **12522** |

**包袱-收益**: textB **−53.6%** (26968→12522, >halved) · vwmacc **−83.3%** (2304→384) = the code-volume win
the lever targets. spill 7→63 · reload 12→73 · maxVreg v30→v31 = register residence **deliberately
TRADED** for compactness (the stack-panel i16 partials spill). This IS the capability-keyed schedule
tradeoff made explicit (the "re-roll trap" the S6 comments warn of is now the chosen point on the
code-volume × register-budget axis).

**双账本 note**: this A/B is compiler-SYMMETRIC (both clang-17, no opponent) → the FORM delta is valid.
The gcc-15 deployment-compiler seal is DEFERRED to Phase 2 (board gcc=12.3.1 lacks riscv_vector.h; shipped
ggml = gcc-15, [CASE-COMPILER-ASYMMETRY]). [CASE-KQUANT-GCC-CODEGEN] a-priori: gcc explodes on the mixed-SEW
full-unroll core (820 vsetvli/742 spill, q6_K report) → rolling BOUNDS the regalloc code, HYPOTHESIZED to
help MORE under gcc than clang (not measured here).

## [PAT-1] registration — FLAGGED (schema/ off-limits for this line)
The pattern registry is `schema/pattern-registry.v1.json`, OWNED by the docs 收尾线 (schema-X3). G6-B
(emitter line) did NOT touch schema/. The POPULATED proposed row is `PROPOSED-PAT-1-entry.json`
(pattern_id `PAT-EMIT-ROLLED-LOOP-repack-gemm-main-term-GAP-EMIT-UNROLL`, status `mechanized`, metrics_hook
= the lit + this board seal). **Main session: append it after coordinating with the schema line.**

## VERDICT — Phase 1 COMPLETE (mechanism name)
rolled-loop emission MODE **built** ∧ **byte-exact GREEN** (independent oracle + A/B identity) ∧ [PAT-1]
row populated+flagged ∧ kernel-axis FORM seal (text −53.6%, vwmacc −83.3%; residence traded for
compactness). **PERF NOT decided** (Phase 2 = q2_K@k1 e2e re-measure vs hand-brick + gcc-15 seal). Even if
Phase 2 does not flip positive, the rolled-loop mode = an emitter-maturity milestone ([K-7] 5th lever built).
**[NG-4]: FORM seal is kernel-axis, NOT a beat, NOT e2e.**

## Phase 2 readiness
Ready. Board build+objects live at `/tmp/g6b-emit-unroll` (rvv). Phase 2 = deploy ROLLED q2_K GEMM into the
G5 `M2-q2_K-k1-e2e` harness (k1, VLEN256, gcc-15/clang-18 deployment), re-measure prefill/decode vs stock
hand-brick, and seal the gcc-15 form. Selection: pass `emit_loop_schedule="rolled"` on the front-door op
(or calibrate `resolveRepackMainTermRolled`'s Phase-1 sentinel to the measured I-cache budget).
