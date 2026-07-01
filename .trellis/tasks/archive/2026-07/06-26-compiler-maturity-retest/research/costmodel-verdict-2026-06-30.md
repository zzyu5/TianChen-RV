# ③ cost-model 真资源 fact — verdict (2026-06-30, current-code-grounded)

**VERDICT: no implementable cost-model win exists. The cost model is capability-blind BY DESIGN, not by omission; the one derivable-but-unused fact (RVV0.7 LMUL-allowlist prune) is defensive-only — flips no selection, not reachable on real hardware. ③ closes as investigated/characterized, NOT as a build.** This is the third "都做" category; it is materially addressed this session (current-code reconnaissance), not left resting on a stale prior finding.

## What the cost model actually is (code-grounded)
- **Scalar formula** = `computeBlockDotShapeCostCore(reductionsPerBlock, multiBlockFactor, stripElision, coreLatencyDepth)` at `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h:2472`. Its header (`:2303`) is explicit: **"CAPABILITY-BLIND structural cost … depends ONLY on the structural facts …, never on the target capability."** It carries **no VLEN/ELEN/vreg/mask/tail parameter** — that is the deliberate anti-lookup design, not a gap.
- **VLEN already flips what it can**, but in the *legality/enumeration* layer, not the formula: `enumerateBlockDotShapeCandidates` (`:2633`) prunes by real VLEN fact via `getRVVStripVLMAXElements(coreLMUL, stripSEW, minimumVLEN)` (`:2544`) + reduction-rank (`rankVLEN`, `:2654`). These are the flips already **sealed** in q8_0 / FP4-codebook (the Win-A bricks). The formula only sees the *derived* reduction count, never VLEN directly.
- **Live path = offline memoization.** `selectGenericSchedule` (`:3563`) tries `lookupGenericTuningRecord` first (`:3572-3590`) and only falls back to the capability-blind argmin (`selectGenericMinCostCandidate`, `:2259`) on cold-start. So on the production/live path **the cost formula never decides a selection** — the offline table does. Enriching the formula changes nothing observable.

## The only derivable-but-unused fact, and why it's defensive-only
- **RVV0.7 LMUL allowlist prune** (prior Track A design `.trellis/tasks/archive/2026-06/06-25-backend-maturity-winA/artifacts/trackA-costmodel-resourcefact-DESIGN.md:63-105, 206-211`): RVV0.7 (C920) supports `{m1,m2,m4,m8}` — no fractional; RVV1.0 adds `{mf8,mf4,mf2}`. Injecting the allowlist would prune fractional anchors.
- **But it flips NO selection** — mf4 always costs 4× m1 (reduction 4 vs 1) so m1 is already chosen; codebook mf2 is already gather-VLMAX-pruned at VLEN128. And it is **not triggerable on real hardware**: C920 is the only RVV0.7 silicon and it's VLEN128, where fractional anchors never win and are already gated. The prune only blocks a hypothetical impossible config (RVV0.7 + VLEN≥256). Prior verdict labels it exactly: **"category (b) legality-hardening (defensive code), produces NO new perf numbers, flips NO existing selections."**
- Implementing it = **gilding** (no N1/N2/N3 value, no perf, no selection flip, no reachable correctness exposure). Declined for this maturation push; if ever wanted, it's small pure-wiring defensive hardening, separable from the maturation thesis.

## Why this verdict is USEFUL, not just a null
The reconnaissance bottom line — **"no bounded cost-model improvement will flip a real lowering selection; the Win-A upside is emitter-gated, not cost-ranking-gated — change the emitter, not the cost model"** — **validates this session's emitter-focused work as the correct lever**:
- **①** wa nibble wide-LMUL = flipping the *emitter's* default anchor (mf4→m1). Emitter-gated. ✓
- **②** Track B q4_K = *emitter* auto-construction of the integer core, incl. the mf2↔m2 wide path (brick 3). Emitter-gated. ✓

So ③'s finding is not "nothing to do" in a vacuum — it's the evidence that the maturation effort's energy belongs exactly where ① and ② put it. The cost model is **done w.r.t. what can flip selections**; the residual maturity gap lives in the emitter (Track B's remaining fp/structural bricks 4–8), not the cost model.

## "都做" status after this session
- ① wa nibble wide-LMUL — **DONE** (implemented, byte-exact, trellis-check PASS).
- ② Track B q4_K 真自动构造 — **integer-core milestone (bricks 1–3/8 done, byte-exact; brick 3 proves the mf2↔m2 flip)**; fp/structural wrapper (bricks 4–8, fp32 fold = the genuine hard seam) is multi-day, deferred to fresh context. See `06-26-track-b-generic-lowering/research/g3-q4k-feasibility.md`.
- ③ cost-model 真资源 fact — **INVESTIGATED & CHARACTERIZED this session (current-code): no implementable win; capability-blind by design; residual is defensive-only/non-hardware-reachable; Win-A lever is the emitter, which ①/② pursue.** Closed as a verdict, not a build.
