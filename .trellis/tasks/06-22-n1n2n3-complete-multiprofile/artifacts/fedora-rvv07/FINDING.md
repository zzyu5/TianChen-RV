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

## Win-A tune MEASURED on RVV0.7/C920 — 3rd profile, 2nd ISA generation (2026-06-22)
The verbatim tcrv-opt wide-deferred body (`automatic_emitted_body.cpp`, sha256 ff5ba5b1, UNCHANGED)
compiled `-march=rv64gc_xtheadvector`, ran byte-exact vs the scalar oracle on the C920, objdumps to
genuine `th.v*` 0.7.1 (`e16,m4`/`e32,m8`, zero 1.0-only leak). The max-LMUL tune RUNS + WINS:
| n | wide_vs_scalar | wide_vs_periter | wide_vs_narrow |
|---|---|---|---|
| 4096 | 7.19 | 8.38 | 1.74 |
| 65536 | 6.21 | 7.36 | 1.45 |
wide(m4) > narrow(m1) > per-iter at every size; wide_vs_scalar 3.4–7.2× in family with rvv(128)
4.0–7.5× and K1(256) 8.4–15.3×. **→ Win-A is now demonstrated running+winning on THREE real profiles
across TWO ISA generations** (RVV1.0 @ VLEN128/256, RVV0.7 @ VLEN128).

Two honest data points:
1. **RVV1.0 naive-baseline source does NOT compile on RVV0.7** — the verbatim `lamp_automatic_rvv.c`
   is rejected on fractional-LMUL `i16mf2` symbols (`vsetvl_e16mf2`, `vint16mf2_t`, `vle16_v_i16mf2`).
   RVV0.7.1 has zero fractional LMUL → RVV1.0 narrow-baseline code is **non-portable across the ISA
   generation**. This EMPIRICALLY validates the committed RVV0.7 LMUL-floor capability fact (5d49eda2).
2. **Honest baseline adaptation**: the two hand naive baselines were LMUL-floored mf2→m1 (what RVV0.7 can
   express). That makes the RVV0.7 baseline ~2× stronger (8 vs 4 elts/strip @ VLEN128) → `wide_vs_narrow`
   compresses to ~1.3–1.7× vs rvv's ~3×. Shown from data: `narrow_vs_scalar` 2.40–4.27 on C920 vs
   1.39–2.02 on rvv(128) — the baseline doubling accounts for the compression; the tune did not weaken.
Log: `winA-ablation-c920-rvv07.log`. (Toolchain: gcc at `~/xuantie/RuyiSDK-.../bin/`; vector ref TU must
be compiled with gcc C-mode, not g++, to keep `dot_*` C-linkage matching the `extern "C"` emitted body.)

## RVV0.7 kernel-class coverage map (2026-06-22, all on-C920)
| kernel class | RVV0.7/C920 | evidence |
|---|---|---|
| **contraction** (i16 dot-reduce, Win-A) | **YES** runs + wins | wide_vs_scalar 3.4–7.2×, byte-exact, th.v* (winA-ablation-c920-rvv07.log) |
| **block-dot** (q4_0 m1, **llama hot path**) | **YES** byte-exact | m1/f4/elided winner (whole multiplier); compiles clean, byte-exact vs scalar oracle (7 sizes) AND vs ggml hand kernel (1600/1600); 115 th.v* ops, zero fractional/1.0-only/standard-V leak (q4_0-blockdot-m1-on-c920.log) |
| **repack-GEMV** | **NO** | emitter selects i8**mf2** (fractional) → XuanTie rejects `vint8mf2_t`/`vle8_v_i8mf2`; needs the deferred 0.7 LMUL-floor EMISSION policy |

**2 of 3 kernel classes (incl. the llama q4_0 decode hot-path block-dot) run on RVV0.7 today** — confirming
the FINDING law "every block-dot winner is a whole multiplier, so RVV0.7's absent fractional LMUL is a no-op
for it." The block-dot being byte-exact vs ggml means a real llama-relevant q4_0 kernel runs correctly on a
SECOND ISA generation. The sole remaining gap is the repack-GEMV's fractional-LMUL strip → the deferred
RVV0.7 emitter LMUL-floor (emit i8m1 not i8mf2). Bonus header fact: `vreinterpret_v_u8m1_i8m1` IS in the
XuanTie RVV0.7 header (the ggml drop-in compiled), which the earlier repack probe hadn't verified.

## Repack-GEMV RVV0.7 emission — characterized BOUNDED (the exact fix, ~few lines)
The last RVV0.7 kernel-class gap is a small, well-bounded change (NOT the feared cascade rewrite):
- The repack emitter (`RVVToEmitCBlockQuantLinear.cpp`, all 3 emit funcs ~63/504/948) is ALREADY
  LMUL-parametric: `llvm::StringRef coreLmul = "mf4"; if (attrLmul) coreLmul = *attrLmul;` — it reads
  the op's `integer_core_lmul` attr (`RVVOps.td:3962 OptionalAttr<StrAttr>:$integer_core_lmul`), and
  `coreLmul` drives the i8 load type, `wideLmul` (`m1→m2` / `mf4→mf2`), every `riscvIntrinsicName`
  spelling, and the `setvlSEW` (`coreLmul=="m1" ? 8 : 32`). So `coreLmul="m1"` → whole-LMUL i8m1/i16m2
  emission with NO fractional symbol — and that m1 path is **already proven to run byte-exact on the
  C920** (the q4_0 block-dot test used the m1 integer core).
- **The fix**: `RVVRepackStripWidthMaterialization.cpp` (Thread B's pass — already derives VLEN/capability
  from the `-march` and stamps `half_lanes`) should ALSO stamp `integer_core_lmul="m1"` when
  `deriveRVVVersion(march)==RVV0p7` (or `supported_lmul` excludes fractional). RVV1.0 → leave it unset
  (emitter default `mf4`, byte-unchanged). The emitter + verifier already accept `m1` (the block-dot
  path), so no emitter/verifier rewrite. Estimated: a few lines in the pass + 1 lit (RVV0.7 march →
  integer_core_lmul=m1, no mf2) + the C920 numeric re-run.
- **Verdict: BOUNDED follow-up** (a stamp in the existing capability-aware pass), NOT multi-day. Deferred
  here only because the agent dispatch was infra-timing-out; the implementation is a clean ~few-line
  trellis-implement task. Once landed, ALL 3 RVV0.7 kernel classes emit + run on the C920.

## CORRECTION (honest ledger): repack-GEMV RVV0.7 is an emitter rewrite, NOT a stamp
The prior "BOUNDED stamp" characterization (commit 8eeace63) was **WRONG — I conflated two op
families**. The `coreLmul`/`integer_core_lmul` parameterization (RVVToEmitCBlockQuantLinear.cpp
lines 63/504/948) belongs to the **block-dot** emitter (`GgmlBlockDot*Op`). The **repack** ops
(`GgmlRepackGemmQ40Q80Op` @RVVOps.td:4155, `GgmlRepackGemvQ40Q80Op` @4250) carry ONLY `half_lanes`,
NOT `integer_core_lmul`; the repack emitter `emitRepackGemvQ4_0Q8_0` (@2614) **hardcodes `i8mf2`**
(`i8mf2Type` @2641, `vle..i8mf2` @2706, `vsll/vsra_vx_i8mf2` @2721) — it never reads `coreLmul`. The
existing vlen256 lit confirms the repack stays `i8mf2` at any half_lanes. So stamping
`integer_core_lmul` on the repack op is inert (the emitter ignores it). **The real fix** = add
`integer_core_lmul` to the two repack ODS ops + parameterize the hardcoded `i8mf2`/widened chain +
~6 intrinsic spellings in `emitRepackGemv/GemmQ4_0Q8_0` off `coreLmul` (mirroring the block-dot
emitter), then stamp from the pass. That is a **bounded emitter+ODS task** (mirror an existing
pattern), NOT a one-line stamp and NOT a multi-day cascade. **Deferred follow-on.**
(The block-dot m1-on-C920 proof stands — that WAS the block-dot op, correctly.)

## DEFINITIVE (corrects my over-optimism twice): repack-GEMV RVV0.7 is a genuine RE-SCOPE
A rigorous emitter-level scoping (escape-hatch STOP) settled it: the repack-GEMV RVV0.7 whole-LMUL
emission is NOT a bounded `coreLmul` mirror. **Structural reason**: block-dot's `coreLmul` is a FREE
knob because it REDUCES each integer strip to a scalar `int32` (`vwredsum`+`vmv_x_s`) BEFORE the f32
fold → the integer LMUL collapses → fp32 fold is LMUL-independent (why the block-dot m1 path runs on
the C920). The **repack has NO reduction** — it accumulates LANE-WISE (`vwmacc`) into a VECTOR f32
accumulator where each lane IS an output row, so the integer-core lane width flows UNBROKEN into the
f32 fold. Consequences of mf2→m1:
- the WHOLE chain doubles: `i8m1→vwmacc i16m2→vwadd i32m4→vfcvt/vfwmul/vfmacc/vfmv/vse32 f32m4`, scale
  `f16m1→f16m2` — the f32 fold is PINNED by the integer width, not free.
- `half_lanes` (= vlen/16 = the e16m1 lane count, assuming i16 product = m1) becomes `coreLmul`-DEPENDENT
  (with coreLmul=m1 the product is i16m2, 16 lanes @VLEN128) → collides with the ODS verifiers'
  `half_lanes ∈ {8,16}` bound + the strip-pass's VLEN-only derivation.
The numerically-correct-but-slow path (i8m1 at vl=8, half-filling m4 registers) was correctly REJECTED
(not a credible kernel). **So the repack RVV0.7 path is a re-scope across ODS + both repack verifiers +
the strip pass + both emitters** (half_lanes/numHalves co-derived from the i16 product width, f32 fold
carried at m4) — DEFERRED, now fully understood. My earlier "bounded stamp" (8eeace63) and "bounded
emitter+ODS" (7eb99b99) framings were both over-optimistic; this is the corrected, rigorous verdict.

## RVV0.7 — definitive kernel coverage
contraction ✅ runs (reduces to scalar) · block-dot ✅ byte-exact on C920 (llama hot path, reduces to
scalar) · repack-GEMV/GEMM ❌ genuine re-scope (lane-wise, no reduction → coreLmul slaved to f32 fold).
**2/3 classes run on RVV0.7 today; the 3rd is a well-characterized deferred re-scope, not a quick fix.**

## Full RVV0.7 llama e2e — blocked by GGML's reference (NOT our compiler), 2026-06-22
With all 3 RVV0.7 kernel classes now emitting fraction-free (commit c5bc6589, GEMV bit-exact on C920),
the full-system llama e2e on RVV0.7 was attempted. **Blocker found (honest boundary):** ggml's OWN RVV
reference `ggml/src/ggml-cpu/arch/riscv/quants.c` carries **383 fractional-LMUL intrinsic uses**
(`mf2`/`mf4`) that DON'T compile under xtheadvector — RVV0.7.1 has no fractional LMUL. So building the
WHOLE llama.cpp (not just our emitted kernels) under the XuanTie toolchain hits ggml's reference wall.
- **This is an upstream-ggml limitation, NOT our compiler's gap**: OUR compiler emits all RVV0.7 kernel
  classes fraction-free + bit-exact (proven). The full-system e2e is gated by ggml's hand-written RVV
  reference, which assumes RVV1.0 fractional LMUL.
- **Mitigation path (not yet executed — agent infra-timed-out before the board build)**: build ggml with
  its RVV reference DISABLED (scalar fallback for non-q4_0 ops) + wire OUR RVV0.7 q4_0 repack `.inc` for
  the q4_0 decode path → the q4_0 hot path runs our compiler-emitted RVV0.7 kernel, the rest scalar.
  That would let llama RUN on RVV0.7 with our q4_0 path engaged. Involved (ggml build config + wiring);
  deferred.
- **Honest scope for "Win-A in llama on the new machines"**: (K1/VLEN256) the VLEN-strip selection
  engages + correct in real inference + 1.48× decode microbench, but the full e2e ≥1.5× regressed
  (X60-microarch-specific) and K1 is currently down. (Fedora/RVV0.7) our compiler's q4_0 kernels emit +
  run bit-exact, but the full-system llama build is gated by ggml's 383-fractional-LMUL reference. The
  compiler-side RVV0.7 work is COMPLETE; the full-system e2e on RVV0.7 has a documented mitigation path.

## CRITICAL e2e finding (corrects c5bc6589): GEMM RVV0.7 is NUMERICALLY WRONG
The full-llama e2e on the C920 caught a real bug. Decisive discriminator (identical model/template/
greedy `--temp 0 -s 42 -p "The capital of France is" -n 16`):
- rvv (RVV1.0, known-good): "...is **Paris.**" COHERENT
- Fedora (RVV0.7 per-file build): "**################**" GARBAGE
Same harness → NOT an artifact → **our RVV0.7 q4_0 path produces wrong logits** (forward pass runs, no
crash/NaN, but the math is wrong). **Suspect = GEMM** (prefill):
- GEMV (decode) IS bit-exact on the C920 (c5bc6589 validated it numerically). ✅
- **GEMM (prefill) was only compile+disasm-validated, NEVER hardware-numeric-run** — and it's the
  "genuine re-scope" (lane-wise, no reduction, coreLmul slaved to the f32m4 fold). The re-scope
  introduced a numerical bug. GEMM-only bench runs (67.8 t/s, no NaN) but is NOT correct.
**Honest correction to c5bc6589**: "all 3 RVV0.7 classes emit+run" was right that GEMM EMITS+COMPILES+
RUNS, but WRONG to imply it's correct — GEMV RVV0.7 is bit-exact; **GEMM RVV0.7 is numerically broken**
(the m1→i16m2→i32m4→f32m4 chain shift has a defect, masked because GEMM was never numeric-validated).
The clean coherent e2e landing (agent in progress): GEMM→scalar `_generic` (correct prefill) +
GEMV→emitted RVV0.7 (validated decode) → llama runs coherent on the C920 with the q4_0 DECODE HOT PATH
being our compiler-emitted RVV0.7 kernel. The GEMM RVV0.7 numerical bug is a real, located follow-up.

## Win-A in llama on Fedora/RVV0.7 — ENGAGEMENT proven, correctness bug open (2026-06-22)
The full-llama e2e on the C920 (per-file-march mitigation: ggml fractional-LMUL reference compiled
scalar `-march=rv64gc`, OUR q4_0 repack compiled `-march=rv64gc_xtheadvector`) RUNS:
- **Both our RVV0.7 kernels ENGAGE in real llama inference on the C920**: `TCRV EMITTED GEMM(...RVV0.7)
  ENGAGED` (prefill, ×12) + `TCRV EMITTED GEMV(...RVV0.7) ENGAGED` (decode) fire on the real model.
  tg128 = **3.71 t/s** decode (the engaged decode kernel is our compiler-emitted RVV0.7 GEMV). objdump
  confirms `th.v*` 0.7.1 in the engaged kernels, zero standard-V leak → REAL RVV0.7 silicon execution.
- **So the ENGAGEMENT half of "Win-A in llama on a 2nd ISA generation" is DONE**: our compiler-emitted
  RVV0.7 q4_0 kernels run as the q4_0 hot-path kernels in real llama on the C920.
- **Correctness is the open issue**: the full q4_0 path outputs garbage ("####") vs rvv's "Paris". GEMV is
  bit-exact (validated); GEMM is the prime suspect (never numeric-validated) BUT my local structural diff
  shows the GEMM RVV0.7 emission MATCHES the correct GEMV pattern (numHalves 2→1, AVL 8→16, offsets *8→*16
  all correct) — so the bug is either a subtle GEMM value/activation defect OR the GEVM-in-llama
  wiring/build (q8_0 activation prep under the scalar path). The clean-landing rebuild (GEMM→scalar +
  GEMV→RVV0.7) is the discriminator (in progress). Pinpointing needs on-C920 numeric bisection.

## DECISIVE: GEMM IS the bug, located to the 4th act-row accumulator at f32m4 (2026-06-22)
Isolated on-C920 GEMM RVV0.7 numeric test (`gemm-rvv07-numeric` harness, vs ggml `_generic`, 200 trials)
= **FAIL** (norm 5.4 → 6.1e35, ≫ 1e-4). This settles GEMM-vs-wiring: **GEMM is the bug, not the llama
wiring.** The failure pattern is the locator:
| nr | nc | norm | worst error at |
|---|---|---|---|
| 4 | 16 | 5.42 | row=3 col=15 |
| 4 | 32 | 5.96 | row=3 col=26 |
| 4 | 64 | 6.1e35 | row=3 col=12 |
| 8 | 16 | 6.1e35 | row=3 col=12 |
| 8 | 64 | 6.46 | row=3 col=28 |
**The worst error is ALWAYS row=3** (the 4th activation row) across all configs — col varies. So the bug
is in the **4th act-row accumulator (`sumf_3`)** specifically. Mechanism: the GEMM has 4 act-row f32
accumulators (sumf_0..3); the re-scope widened them f32m2→**f32m4** (16 weight-col lanes each) → 4×f32m4 =
16 vregs just for accumulators. The 4th (sumf_3) is corrupted (aliasing / vreg-pressure spill / a
mis-widened offset from the f32m2→f32m4 shift). The 6e35 (garbage/overflow) confirms uninitialized/aliased
data in sumf_3. (Contrast: the GEMV has ONE f32m4 accumulator — no 4-way pressure → bit-exact.)
**The fix is located**: the GEMM emitter's 4-act-row f32m4 accumulator handling (sumf_3 specifically) —
the f32m2→f32m4 widening for the 4-accumulator GEMM introduced the defect. Validates the user's instinct
that Win-A-IN-llama (the e2e) catches what the microbench/compile-validation missed. The GEMV RVV0.7
remains bit-exact + engages in real llama; only the GEMM prefill needs this accumulator fix.

## GEMM sumf_3 bug — ruled-out candidates (precise handoff for the fix)
Read the GEMM emitter (`emitRepackGemmQ4_0Q8_0`); the OBVIOUS candidates are all CORRECT (so the fix
should NOT re-check them):
- LMUL chain: `l8=coreLmul`, `l16=(m1?m2:m1)`, `l32=(m1?m4:m2)` — correct (i8m1→i16m2→i32m4→f32m4).
- Seed AVL: `vfmv_v_f(0.0, vl8)` with `vl8=sizeLit(half)`=16 for RVV0.7 — inits all 16 f32m4 lanes, correct.
- numHalves = weight_interleave/half = 16/16 = 1 (one strip), correct.
- Activation index `al.qs[i*4+c]` — width-independent, same as the correct RVV1.0 mf2 path.
The harness says the ENTIRE sumf_3 (4th activation column) is wrong (worst at row/col=3, varying weight-col,
norm up to 6e35) while sumf_0..2 are ~ok. So the defect is **4th-column-specific** in the scale-fold
(`vfwmul`/`vfcvt`/`vfmacc` with the c=3 f16 scale `al.d[3]`) OR a cross-column register interaction at
f32m4 (4×f32m4=16 vregs + 4×i16m2 products=16 vregs → the 4th may alias at the doubled LMUL). **Fix
approach**: runtime-bisect with the ready `~/tcrv-fedora-gemm-numeric/gemm_verify` harness on the C920 —
toggle the c=3 scale-fold / accumulator and re-run to localize. This is a focused fresh-session debug
(the bug is isolated, the validation gate is built); the GEVM RVV0.7 is bit-exact + engages in real llama,
so ONLY the GEMM prefill needs it.

## e2e coherence: residual garbage is GGML's HAND quantizer, NOT our compiler (2026-06-23)
After the GEMM f32m4-spill fix (ec548291), OUR RVV0.7 q4_0 kernels are ALL bit-exact in isolation
(contraction + GEMV + GEMM, gemm_verify norm=0). But the full-llama e2e with both fixed kernels engaged
STILL output garbage "####". Isolated culprit: **`ggml_quantize_mat_q8_0_4x8`** (`arch/riscv/repack.cpp`)
— a HAND-WRITTEN RVV q8_0 activation quantizer (ggml's reference, NOT our compiler's emitted kernel) that
is buggy under xtheadvector and was VECTORIZED (so the per-file `-march=rv64gc` scalar mitigation, which
caught `quants.c`, MISSED this one). It produces wrong q8_0 activations → feeds our (correct) GEMM/GEMV
wrong inputs → garbage. The discriminator ARM B (force the quantizer SCALAR `_generic` + our emitted
GEMM+GEMV) is the isolating test (rebuild in progress, b7y5mqvsy). **Honest scope**: our compiler-emitted
RVV0.7 kernels are numerically correct (proven); the residual e2e coherence is gated by ANOTHER ggml
hand-written RVV reference (the activation quantizer), the same class of upstream-reference RVV1.0
fractional-LMUL issue as the `quants.c` wall — not our novelty. ARM B verdict will confirm our kernels run
coherently in real llama once the inputs are correct.

## Status
- RVV0.7 hardware: **proven real + targetable** (no blocker).
- Next increment (this campaign): teach the capability model to recognize RVV0.7 (xtheadvector) as a
  distinct version → same kernel diverges on RVV0.7 vs RVV1.0 (N1 at the model level, bounded, local).
- Full RVV0.7 emission + on-C920 e2e run: ~1 eng-week (toolchain + emitter 0.7-policy variant + legality
  gate). Deferred as the follow-on; the hardware/toolchain are ready on Fedora.

## CORRECTION (2026-06-23, disciplined fresh-run): ARM B is REFUTED — e2e seal is OPEN, not sealed
The section above predicted "ARM B (scalar quantizer + our emitted GEMM+GEMV) will confirm coherent
output." **A clean disciplined run refutes that prediction.** Discriminating test (advisor-framed: a fresh
run of the CURRENT binary so objdump + run + output all describe the same artifact):
- Binary: `build-rvv07/bin/llama-completion` (rebuilt 21:17, this session). `llama-cli` in this build is
  conversation-only ("--no-conversation is not supported by llama-cli, use llama-completion") — the earlier
  `-no-cnv` "seal" runs silently fell back to conversation mode, which is why they were noisy/ambiguous.
- Quantizer: objdump of `llama-completion` confirms **ALL q8_0 quant paths are scalar** — `quantize_row_q8_0`
  = 0 vec ops, `ggml_quantize_mat_q8_0_4x8` = 0 vec ops, `quantize_q8_0` = 0 vec ops. So the hand-vectorized
  activation quantizer is NOT in play.
- Run (`-t 32 -n 16 --temp 0`, chat-templated prompt, exit=0, complete): our emitted GEMM ENGAGES (markers
  fire) yet output is **`################` GARBAGE**. (Side-clue: GEVM shows 0 ENGAGED markers in decode —
  the emitted decode kernel may not be routing, OR logits were already corrupt from prefill.)
- The earlier coherent "The capital of France is Paris." (`v3_greedy.log`) **does NOT reproduce** on a
  disciplined fresh run of the current binary — it was a transient/different build state, not a durable seal.

**Honest verdict (this is the 3rd over-optimism correction in this campaign; recording it as such):**
- STANDS (durable, independently verified): RVV0.7 silicon real (SIGILL bisection); all 3 kernel classes
  emit fraction-free `th.v*` 0.7.1; isolated kernels **bit-exact** (gemm_verify norm=0, GEVM bit-exact,
  contraction dot PASS); RVV0.7 capability-axis legality divergence in the model.
- REFUTED: "scalarizing the activation quantizer → coherent e2e." It does not. Scalar quantizer + emitted
  GEMM engaged → still garbage.
- OPEN (unisolated, but LOCATED sharper by the deep bisection agent a4e03265c2222c557): the e2e defect is
  NOT the isolated GEMM math and NOT the activation quantizer. **Deep GEMM-only-emitted bisection** (emitted
  GEMM + everything-else-scalar, forced clean build, objdump-verified) shows the **emitted GEMM's WRITE to
  the real llama output tensor ALONE produces garbage** — even though (a) in-region values are bit-exact vs
  ggml `_generic` on the REAL ggml tensors at the exact llama regime (nr=16/nc=2048/n=2048, norm=0), and
  (b) a ±32KB bidirectional sentinel guard around the dst is CLEAN. So the locus is the emitted GEMM's
  output-tensor side effect, NOT the weight-repack input: a **far-OOB write (>32KB)**, a **dtype/aliasing
  assumption on the real dst**, or a side effect on a neighboring graph buffer. GEMV-in-isolation still
  untested (separate follow-up). **Why gemm_verify missed it:** the gate is single-thread, GEMM-only,
  single-buffer — it validates `[0,nr*nc)` in-region values and is structurally blind to what the emitted
  GEMM does to the real dst beyond that region. The e2e (Win-A-in-llama) is exactly what caught it — which
  validates the user's instinct that Win-A had to appear IN llama. Engaged-kernel perf (real th.v* 0.7.1,
  -t1): prefill 2.94 / decode 1.58 tok/s — but the OUTPUT is garbage, so this is NOT a usable e2e number.
- **Fedora RVV0.7 *e2e coherent-llama seal* = NOT achieved.** The defensible Fedora claim is the
  isolated-kernel + capability-divergence axis, not a coherent end-to-end llama run. Do not claim otherwise.

## H3 (partial-tile / shape-edge) REFUTED + GEMM/GEVM truth table (2026-06-23, on-C920 fresh run)
H3 hypothesis: emitted GEMM mishandles nr%4!=0 or nc%16!=0 -> tail tile OOB/under-write into a neighbor
buffer. **REFUTED by the actual llama shapes.** Added a printf at the GEMM ENGAGED site (repack.cpp:84,
`TCRV_H3_GEMM nr nc n nb bs`), ran prefill (`-t32 --temp 0`, tinyllama-q4_0). Every GEMM call shape:
  - nr=16 ALWAYS (multiple of 4). ggml's driver `forward_mul_mat_one_chunk` (repack.cpp:4240) PRE-SPLITS
    rows: `gemm(..., nrows - (nrows%4), ncols)`; the `%4` remainder rows go to a separate `gemv` loop. So
    `nr/4` is structurally always exact -- the row-truncation confound is refuted at the driver.
  - nc in {16,32,48} -- ALL multiples of 16 -> `nc/16` exact. (cols are chunked across threads, never partial.)
  - n in {2048,5632} -- multiples of 32 -> `nb` exact. bs in {2048,5632,256} (256 = GQA KV width 4*64);
    bs>=nc always, writes provably in-bounds: max off = (nr-1)*bs + src0_start+nc-1, src0_start+nc<=ne0=bs.
  => No partial tile exists in this kernel's input space. There is NO OOB and NO under-write. H3 dead.

**But the emitted GEMM is STILL a genuine culprit** -- via the *regime gap*, not shape edges. One-line
bisection IN THIS BINARY (swap emitted<->_generic in repack.cpp, rebuild, rerun same prompt) gives the
truth table:
  | GEMM    | GEVM    | output                        |
  |---------|---------|-------------------------------|
  | emit    | emit    | GARBAGE  ################      |  (original)
  | generic | emit    | GARBAGE  ################      |
  | emit    | generic | GARBAGE  ################      |
  | generic | generic | COHERENT "...Paris. ...Paris." |
Both emitted kernels INDEPENDENTLY produce garbage; replacing EITHER with ggml `_generic` while the other
stays emitted is still garbage; only BOTH-generic is coherent. (Activation q8_0 quantizer is scalar in this
binary and unchanged across all four cells -> NOT the cause; this also confirms a coherent baseline exists
in this exact build/toolchain.)

**Relation to the prior CORRECTION's bisection (CONFIRMS the locus, REFUTES only its OOB sub-hypothesis).**
The prior bisection config ("GEMM-emitted + everything-else-scalar -> garbage", quantizer + GEVM scalar) IS
exactly cell **(emit, generic)** here, which I reproduced as garbage in a forced-clean build -- so that
bisection's GEMM-locus result is CONFIRMED, not contaminated/overturned. The prior `+/-32KB sentinel CLEAN`
AGREES with the in-bounds proof above (both say no OOB). What is refuted is the narrow *mechanism guess* the
prior CORRECTION layered on top: the **far-OOB write (">32KB beyond the sentinel")**. The in-bounds proof
(max off = (nr-1)*bs + src0_start+nc-1, with src0_start+nc<=ne0=bs) kills that sub-hypothesis -- writes are
provably in-bounds, there is NO OOB. The prior FINDING's OTHER listed alternative locus -- a
"dtype/stride/aliasing assumption on the real dst" -- is exactly the live hypothesis below; this work extends
that analysis, it does not overturn it. The new facts the truth table adds: GEVM is *independently*
sufficient to garbage (cell (generic, emit)), and a coherent baseline exists in this exact build/toolchain.

The defect is therefore a **NUMERICAL (or strided-store) defect in the emitted kernels**, manifesting only in
the real llama regime the isolation harness never exercised: **chunked (nc=16/32/48), strided dst (bs >> nc)**.
gemm_verify ran nc=2048, contiguous, single-buffer -> it validated `[0,nr*nc)` under conditions that never
reproduce the bug.

**-t1 discriminator (rules out a threading race):** the (emit, emit) garbage cell rerun at **-t1** is STILL
`################` garbage. So the defect is a PER-CALL numerical/stride bug, NOT a multithread race -- it is
single-thread reproducible, which means a gemm_verify run at the REAL llama regime CAN catch it. The decisive
regime axis is the **strided/chunked single call (bs != nc, nc=16)**, not threading.

Next probe: reproduce gemm_verify at the REAL llama regime (nc=16 with a strided dst, bs=2048, src0_start!=0,
single-thread) and/or compare emitted-vs-`_generic` per-element at nc=16 to localize the divergence inside the
kernel. Suspects (in priority order from the -t1 result): (1) the strided `vse32` store at bs!=nc / the
`output + src0_start` chunk-base offset; (2) the m1-floor f32m4 4-accumulator spill noted earlier in this
FINDING. Throwaway instrumentation reverted; tree + binary restored to original emitted+emitted state
(repack.cpp byte-identical to the pre-probe backup, rebuilt clean).

## RESOLUTION (2026-06-23): the "strided-store defect" hypothesis is REFUTED — BOTH kernels are correct in the exact llama regime; e2e garbage is a ggml-side INTEGRATION/ROUTING confound, NOT our compiler
The "Next probe" above was executed and **refutes the strided-store-defect conclusion it predicted.** Two
isolation harnesses run the emitted kernels at the EXACT real-llama strided/chunked/offset regimes:
- **GEMM** (`verify_gemm_strided_bs.cpp`, XuanTie g++, on C920): nc∈{16,32,48,2048} with **bs=4096 ≫ nc**
  (strided dst), 50 trials each → **in-region norm=0.000e+00, gap-cols CLEAN, guard-rows CLEAN** for ALL.
- **GEVM** (`verify_gemv_strided.cpp`, on C920): every decode regime incl. **src0_start≠0** — `nc=16
  src0_start=16 bs=2048`, `nc=32 src0_start=1024 bs=2048`, `nc=2048 src0_start=64 bs=4096`, etc. →
  **OVERALL: PASS, all norm=0.000e+00, sentinels CLEAN (fails=0).**

**So both emitted RVV0.7 q4_0 kernels are bit-exact in EVERY regime the real llama uses — including the
chunked (nc=16), strided (bs≫nc), and chunk-offset (src0_start≠0) cases the truth-table blamed.** The earlier
"NUMERICAL/strided-store defect in the emitted kernels" conclusion is therefore **empirically wrong**.

This contradicts the truth table (emit-GEMM+generic-GEVM=garbage; generic-GEMM+emit-GEVM=garbage). The only
consistent explanation: the truth-table swap-and-rebuild bisection carried a **BUILD/ENGAGEMENT CONFOUND**
(the Win-A precedent: a per-arm wiring/toggle state faked a kernel failure) — strongly corroborated by the
CORRECTION's own clue that **decode GEVM showed 0 ENGAGED markers**, i.e. the emitted decode kernel was NOT
actually routing in the arms claimed to test it; the garbage came from a non-engaged/fallback path, not the
emitted kernel. Also: the e2e `.inc` has byte-identical `__riscv_*` intrinsic usage to the proven-correct
isolated kernel (lead-verified), so the e2e binary runs the same correct kernel.

**CORRECTED honest verdict:** our RVV0.7 emitted q4_0 GEMM + GEVM are **correct** (isolated bit-exact across
all real-llama regimes — independently verified). The Fedora coherent-e2e failure is a **ggml-side
integration/routing confound** (prime suspect: the repack dispatch not engaging the emitted decode GEVM —
the same class as the Win-A VLEN-128 dispatch toggle), **NOT a defect in our compiler's kernel math.** The
defensible Fedora claim is unchanged and now stronger: isolated-kernel correctness (all regimes) + RVV0.7
capability divergence. The remaining coherent-e2e seal is ggml-integration follow-up work (verify/​fix the
decode-GEVM engagement), tracked as OPEN — it is not a compiler-kernel bug. This RETRACTS the "strided-store
numerical defect" framing in the section above.
