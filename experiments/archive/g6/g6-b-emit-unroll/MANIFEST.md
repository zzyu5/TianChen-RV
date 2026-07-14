# cell MANIFEST — g6-b-emit-unroll

- **campaign**: G6-B emit-quality lever · [GAP-EMIT-UNROLL] rolled-loop emission mode · [K-7] 5th
  first-emission lever candidate ([T8·LAW-FIRST-EMISSION]). Positive answer to the q2_K@k1 e2e loss
  cause (full-static-unroll 27KB register-resident core loses to opponent's compact rolled loop).
- **status**: Phase-1 COMPLETE (2026-07-13, rvv/VLEN128/clang-17). rolled-loop emission MODE built +
  byte-exact HARD GATE GREEN (independent scalar oracle INT_mismatch=0 + direct A/B output
  byte-identical 43008/43008) + kernel-axis FORM seal (textB 26968→12522 −53.6%, vwmacc 2304→384
  −83.3%; spill 7→63 / v30→v31 = register residence traded for compactness). Capability-keyed
  ([PAT-1]-shaped) via OptionalAttr emit_loop_schedule; Phase-1 default FROZEN to unrolled (no
  shipped-behavior change). PERF verdict = Phase 2 (deferred). NOT committed (main session commits).
- **role**: emitter-maturity milestone evidence. Isolates the rolled-vs-unrolled emission increment
  (byte-exact) on the q2_K repack GEMM main term, compiler-SYMMETRIC clang-17 A/B. Data/evidence only;
  emitted .c deterministically regenerable (REGEN_RECIPE.md); board build live at /tmp/g6b-emit-unroll.

## touched files (production compiler path — 4 tracked, reversible; NO git action)
- include/Weft/Dialect/RVV/IR/RVVOps.td            — OptionalAttr emit_loop_schedule on TypedRepackGemmLoopBodyOp
- lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp      — verifier bounds it {"unrolled","rolled"} fail-closed
- lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp — resolveRepackMainTermRolled (cap-key) + rolled emit path + dispatch
- lib/Conversion/RVV/RVVToEmitCInternal.h           — emitRepackKQuantGemmBodyQ2K signature (+bool rolledMainTerm)
- test/Conversion/RVV/rvv-to-emitc-repack-gemm-q2-K-q8-K-rolled.mlir (NEW) — rolled STRUCTURE lit (green)

## durable files (this cell)
- evidence.md               — full writeup: X-0 bottleneck, impl (4 files), byte-exact gate, FORM seal, [PAT-1] flag, Phase-2 readiness
- raw/board_seal_clang17.txt — rvv/VLEN128/clang-17 objdump seal + independent oracle verify + direct A/B identity (raw)
- PROPOSED-PAT-1-entry.json  — POPULATED [PAT-1] row for schema/pattern-registry.v1.json — FLAGGED (schema/ owned by docs 收尾线; main session applies after schema-line coordination)
- REGEN_RECIPE.md            — deterministic emit + board build recipe (emitted .c md5s recorded, not stored)
- board_src/g6b_q2k_identity_ab.c — hand-written self-contained A/B identity driver (no ggml)

## disposition
Phase-1 lever built + byte-exact + FORM improvement. Default UNCHANGED (frozen unrolled) — zero regression
risk. Perf is Phase 2 (q2_K@k1 e2e re-measure vs hand-brick + gcc-15 deployment seal), main-session序后单独派
(avoid k1 board contention). Mechanism-name立项: even a null Phase-2 = emitter-maturity milestone ([K-7]).
[NG-4]: FORM seal kernel-axis, NOT a beat, NOT e2e.
