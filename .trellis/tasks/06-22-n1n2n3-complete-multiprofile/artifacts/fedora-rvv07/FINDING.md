# Fedora RVV0.7.1 (Sophgo SG2042 / T-Head C920) — proven targetable (2026-06-22)

## Decisive proof: the C920 runs RVV0.7.1, faults RVV1.0
- **Toolchain**: RuyiSDK `gnu-plct-xthead` 3.1.0 (XuanTie sources, PLCT build) — GCC 14.1.1, binutils
  2.42, native riscv64 host (runs on the C920), ships `riscv_vector.h`. Via proxy from the ISCAS
  RuyiSDK mirror, sha256-verified, installed `~/xuantie` (2.6G) on Fedora.
- **Hand RVV0.7.1 kernel** (i32 vadd + i16 widening dot-reduce, `__riscv_*` intrinsics, `-march=
  rv64gc_xtheadvector`): objdump shows genuine 0.7.1 `th.v*` encodings (`th.vsetvli`, `th.vadd.vv`,
  `th.vwmul.vv`, `th.vredsum.vs`); **runs green** → `dot=1258 ref=1258 PASS, exit 0`.
- **Control**: an RVV1.0 (ratified-V) binary (`vmv1r.v`, `-march=rv64gcv`) → **Illegal instruction,
  exit 132**. Clean dichotomy: C920 executes 0.7.1, faults 1.0. The exact dot arithmetic confirms a
  real run, not emulated/no-op.
- **march-spelling trap (settled)**: `-march=rv64gcv0p7` is TOOLCHAIN-dependent — stock gcc13 binutils
  assembles it to standard-V (`010072d7`, SIGILLs); XuanTie binutils assembles it to `th.*` (runs).
  Portable spelling = **`rv64gc_xtheadvector`**.

## Emitter/parser gap (bounded, ~1 eng-week to first green tcrv-opt RVV0.7 e2e)
- (A) march hardcoded `rv64gcv` at `lib/Target/RVV/RVVTargetSupportBundle.cpp:1683-1684` + probe
  candidates `scripts/rvv_remote_probe.py:440-448` → need xtheadvector march + XuanTie gcc.
- (B) ~40 `__riscv_*` 1.0 intrinsics in `lib/Plugin/RVV/EmitC/` are **empirically source-compatible**
  with XuanTie gcc (the test kernel used the emitter's vocabulary + ran) → the work is a **0.7-policy
  variant** (no ta/ma policy field; explicit EMUL re-vsetvl on widening) + a legality gate banning
  1.0-only ops (whole-register moves, differing segment/indexed forms).
- (C) **N1 GAP — the tractable entry point**: the march-parser in `lib/Plugin/RVV/RVVCapabilityProfile.cpp`
  does NOT recognize RVV0.7 as distinct — the substring tokenizer matches "gcv" inside "rv64gcv0p7" and
  **silently folds it into full RVV1.0** (SEW/LMUL/Zvl128b); `rv64gc_xtheadvector` matches no axis token
  → vector-present with ZERO divergence axes. **A real RVV0.7 capability (version property + narrowed
  0.7.1 allow-lists) must be added** so the SAME kernel gets DIFFERENT legality/selection on RVV0.7 vs
  RVV1.0. This is an N1 ISA-generation capability divergence — the deepest N1 axis (vs the VLEN axis K1
  gives). Full findings: `rvv07-targetability.log`.

## Emission feasibility — MEASURED on the C920 (replaces the ~1-week estimate)
Ran ACTUAL tcrv-opt-emitted kernels (not hand kernels) on the C920 via XuanTie. Two-valued by class:
- **Contraction (i16 dot-reduce, the Win-A class) → RUNS CORRECTLY TODAY.** Verbatim tcrv-opt e2e output
  compiled CLEAN `-march=rv64gc_xtheadvector` (0 intrinsics rejected, incl. m8-LMUL), ran EXACT vs scalar
  oracle across all 10 sizes incl. the partial-final-strip discriminators 33/257/1000/65537 (the only 0.7
  tail-policy hazard cases — did not materialize). objdump = 23 genuine `th.v*` ops, 0 standard-V leak, no
  whole-register `vmvNr.v` (m8 SSA copies coalesced), widening `e32,m8` re-vsetvl auto-inserted by GCC →
  **empirically refutes the "emitter needs explicit EMUL re-vsetvl policy" claim.** RVV0.7 emission for this
  class = toolchain swap + the march plumbing (RVVTargetSupportBundle.cpp:1683 / rvv_remote_probe.py:440).
- **Repack/block-dot → PARTIAL, ONE bounded gap.** Fails to compile under xtheadvector on the
  **fractional-LMUL** symbols: `vint8mf2_t`, `__riscv_vle8_v_i8mf2`, `vsll/vsra_vx_i8mf2`, `vwmacc_vx_i16m1`.
  Root cause (measured vs the XuanTie header): **RVV0.7.1 declares ZERO fractional-LMUL types (no mf2/mf4)**;
  the emitter selected `i8mf2` sources. The full fp16/fp32 surface (vfloat16m1/f32m2, vfwmul, vfmacc, vfcvt,
  vse32) IS accepted → not a hardware/fp limit, an emitter SELECTION-policy gap. Fix: a 0.7 LMUL-floor
  (clamp fractional LMUL → m1). Log: `emitted-kernel-on-c920.log`.

**→ This is a SECOND N1 RVV0.7 capability divergence (beyond the ta/ma policy gate): RVV0.7's LMUL
allow-list EXCLUDES mf2/mf4 → the gearbox LMUL selection floors at m1 on RVV0.7 vs RVV1.0. Fixing it both
unblocks repack/block-dot RVV0.7 emission AND is a same-kernel selection divergence. Next increment.**

## RVV0.7 SELECTION divergence — structural analysis (where it lives)
The capability-FACT divergences (version, ta/ma policy, LMUL allow-list) are committed
(a71aa201, 5d49eda2). For a same-kernel SELECTION divergence ("RVV0.7 floors to m1"):
- **NOT in the contraction LMUL-rung selectors** (`enumerateRVVLowPrecision/DotReduceDeferredWide
  LMULRungs` + their max-widest-legal selectors): empirically proven structurally unreachable —
  the only budget where RVV1.0 picks an mf2-source rung is budget 9, where the m1-source rung is
  already illegal (footprint 10 > 9); pruning mf2 there yields nullopt→narrow fallthrough, NOT an
  m1 promotion. The "wider is monotone-better" rule + the 1-vreg footprint gap make the mf2↔m1
  flip impossible. (Landing the enumerator allow-list param with default=all would be untested
  dead code — correctly NOT landed.)
- **YES in the Q4_0 block-dot min-cost argmin** (`selectGenericMinCostCandidate`, integer_core_lmul
  ∈ {mf4,m1}): it picks by MIN COST not max width, so capability-pruning the mf4 candidate on RVV0.7
  makes m1 the new argmin winner — a genuine same-budget flip, distinguishable by reduction count
  (`getRVVQ40ReductionsPerHalfBlock`: mf4=4 vs m1=1). This is legitimate capability-LEGALITY
  enforcement (RVV0.7 genuinely lacks fractional LMUL), NOT the perf-axis "block-dot LMUL ablation"
  trap (design-boundaries.md) — that trap was adding a tuning knob; this enforces a hardware fact.
  This is the correct home for the RVV0.7 m1-floor selection divergence + it is the q4_0 hot path.

## RVV0.7 LMUL-SELECTION-flip is a DEAD END (proven twice) — honest conclusion
Both candidate selectors were probed; an RVV0.7-vs-RVV1.0 *selection-output* flip via
fractional-LMUL pruning is **structurally impossible**:
- Contraction max-width rungs: the 1-vreg footprint gap means mf2 only wins at budget 9 where
  m1 is already illegal → pruning mf2 falls through to narrow, never promotes to m1.
- Q4_0 block-dot min-cost argmin: **m1 already wins on RVV1.0** (cost 1005 vs mf4's 3110 — the
  cost model rewards the widest whole-multiplier LMUL; an existing test `runCostModelIsCapability
  BlindTest` pins mf4≫m1, and the existing divergence lit already stamps `integer_core_lmul="m1"`
  on rv64gcv). Pruning mf4 on RVV0.7 only drops `legal_candidate_count`; the winner stays m1.
**Root law**: every block-dot winner is a WHOLE multiplier (q4_0→m1, q8_0→m2, q4_1→m1); fractional
LMULs are always cost-losers. So removing RVV0.7's (absent) fractional LMUL can never flip a winner.
A true selection-output flip would need a selector where RVV0.7's restriction lands on the WINNING
candidate (a 1.0-only policy/encoding or an RVV0.7-absent emission shape) — a different design, not
the LMUL axis. **Both agents correctly STOPPED rather than fabricate it (no cost-model hack, no
manufactured budget) — honest-ledger discipline held.**

## Honest RVV0.7 N1 evidence (what stands)
1. **Capability FACTS diverge** (committed): `rvv_version` (1.0/0.7), `supported_lmul` ({mf8..m8} vs
   {m1..m8}, no fractional on 0.7) — queryable, I1/I3, same-kernel profile differs.
2. **ta/ma policy LEGALITY divergence** (committed a71aa201): a ratified-agnostic-policy body is
   fail-closed-REJECTED on RVV0.7 (1.0 emits, 0.7 rejects) — a real same-kernel legality outcome flip.
3. **Hardware**: C920 runs RVV0.7 (`th.v*`), faults RVV1.0. Contraction-class tcrv-opt output runs
   on it today.
A same-kernel *emit-A-vs-emit-B* output divergence (vs the reject-vs-emit legality one) needs the
deferred RVV0.7 EMITTER variant (no-policy form + repack m1 floor). That is the remaining RVV0.7 work.

## Status
- RVV0.7 hardware: **proven real + targetable** (no blocker).
- Next increment (this campaign): teach the capability model to recognize RVV0.7 (xtheadvector) as a
  distinct version → same kernel diverges on RVV0.7 vs RVV1.0 (N1 at the model level, bounded, local).
- Full RVV0.7 emission + on-C920 e2e run: ~1 eng-week (toolchain + emitter 0.7-policy variant + legality
  gate). Deferred as the follow-on; the hardware/toolchain are ready on Fedora.
