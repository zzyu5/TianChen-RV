# Win-A LMUL-width ablation — now ALL-COMPILER (2026-06-22, rvv SG2044 VLEN=128)

Closes the "2–4× 没讲清楚 / is it a clean ablation?" concern. After Thread A (commit
3d2a2b3f) made the narrow-deferred body compiler-emitted, BOTH arms of the ablation are
verbatim tcrv-opt output from the SAME kernel via the SAME deferred-accumulate algorithm —
only the LMUL width differs. Previously the narrow arm was HAND-WRITTEN, so the 2–4× was a
yardstick (compiler-vs-hand), not a pass ON/OFF ablation. Now it is a true ablation.

## Result (best-of-N, taskset 1 core, clang18, HEAD 3d2a2b3f)
| n | wide ns (m4→m8) | narrow ns (mf2→m1) | wide÷narrow |
|---|---|---|---|
| 256 | 121.25 | 363.75 | 3.00× |
| 1024 | 377.50 | 1323.75 | 3.51× |
| 4096 | 1363.75 | 5165.00 | 3.79× |
| 16384 | 6073.75 | 20652.50 | 3.40× |
| 65536 | 40558.75 | 91876.25 | 2.27× |

**2.27–3.79×**, stable across a 2nd run. Both arms numerically EXACT vs the genuine-scalar
oracle on every n (incl. odd n=257/1000 exercising the tail remainder). Structurally both =
widening_product + deferred_accumulate + exactly ONE trailing vredsum, ZERO vwredsum, ZERO
per-iteration reduce — only LMUL widths differ.

## Honest provenance + framing
- WIDE: budget 32 via the default `--tcrv-rvv-materialize-gearbox-schedules` path → i16m4→i32m8.
- NARROW: budget **sed-injected to 9** (constrains the resource model; skips gearbox-schedules,
  runs selector + emitc lowering) → i16mf2→i32m1. The narrow *body* is fully compiler-emitted;
  the budget VALUE is an injected modeled vreg profile (RVV always has 32 architecturally), but
  the *rung selection given that budget* is automatic. So: "what a constrained-vreg profile would
  emit" — the resource-aware selection, both arms the compiler's.
- **This is NOT a new number** — it confirms the existing 2–4× (wide_vs_narrow) with all-compiler
  provenance. Distinct from: the cross-VLEN tune ratios (K1), the repack-vs-naive Win-B (5–6×),
  and the VLEN-strip Win-A-in-llama (1.48× on K1).
- Symbol-collision blocker (both arms now share the emitted symbol — the old hand-written narrow
  never hit this) fixed by sed-renaming `_wide`/`_narrow` + thin noinline wrappers; bodies verbatim
  (`nm` confirms distinct T symbols). Artifacts: `all-compiler-wide-vs-narrow-rvv.log`,
  `wide_body.cpp`, `narrow_body.cpp`, `ablation_driver.c`.
