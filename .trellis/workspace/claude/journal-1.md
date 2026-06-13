# Journal - claude (Part 1)

> AI development session journal
> Started: 2026-06-12

---

## 2026-06-12 — Stage 1 去伪, session 2

Task 06-12-mlir-audit-refactor, Stage 1 (strangler-fig "去伪"). Prior session
committed d0934210 (excised the 17.5k-line golden-string validation module +
its golden cpp tests, −81k lines) and left 10 uncommitted test-only edits
mid-flight.

**Phase A — finish the test-sync loose end (DONE, committed f418cdf9).**
The 10 edits sync stale lit CHECK-lines/counts to *current* real
verifier/emitter behavior (not regressions). trellis-check verified all green:
8 lit edits pass against the current binary; ConstructionProtocolCommonTest.cpp
22→24 confirmed by incremental rebuild (production op-count is authority);
TensorExtLite negative test was incomplete — prior session added an
`emission_kind` fixture field that only shifted which missing-attr is reported,
so reverted the fixture and pointed the CHECK at the stable prefix
`unsupported selected emission-plan requires non-empty string attribute`.
Full lit: 539 pass / 3 fail, all 3 pre-existing baseline (gate4 evidence JSON
moved). Zero new failures.

**Phase B — Stage 1 part (b): zombie front-door removal (IN PROGRESS).**
Target: lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp (1,777 lines, fail-closed
pass `tcrv-rvv-fail-closed-legacy-vector-source-front-door`) + direct-C scar
(`containsForbiddenDirectCMarker` dup'd in RVVTargetSupportBundle.cpp +
TargetArtifactExport.cpp). More entangled than PRD implied: referenced from
Passes.h, tcrv-opt.cpp, ExecutionPlanningPipeline.cpp, RVVOps.td, the plugin;
and Toy/TensorExtLite have a parallel SourceFrontDoor pattern that may be LIVE.
Launched investigation Workflow w28cusx46 (6 parallel readers → synthesis) to
produce a verified removal plan at research/frontdoor-removal-plan.md before
dispatching trellis-implement. Decision rule: only remove RVV-specific dead
code; preserve any shared infra live families need; keep I7-enforcing guards.

**Investigation finding (key):** the front-door file is NOT pure dead weight —
RVVVectorSourceFrontDoor.cpp co-hosts THREE LIVE materialize family passes
(binary / compare-select / runtime-scalar-cmp-select) with passing tests. So a
file-level delete would have amputated live functionality. Verified scope was
narrow & surgical.

**Front-door removal DONE + committed 8d022042** (−225 lines net). Removed only
the FailClosedRVVLegacyVectorSourceFrontDoorPass class + factory + grep-gated
dead helpers (hasForeignLoweringSeedAttr, requireLegacySourceOnlyModule,
failLegacyMaterializer, kLoweringSeedAttrSuffix) + 1 registration block + header
decl; deleted 4 dead i32m1 fixtures; retargeted 3 tests to the same contracts.
trellis-implement + trellis-check both green: build clean, full lit 538
discovered / 3 known baseline reds, I1–I9 clean, I7 guards intact. NOTE: real
current baseline is 3 reds (Scripts widening-dot-reduce ×2 + e2e-self-test);
baseline-lit-failures.txt's "15 failures/544 tests" is a stale older snapshot —
HEAD discovers 542, working tree 538 (−4 = the deleted fixtures).

**Advisor:** tried advisor() twice (incl. after user set Fable 5) — returns
"unavailable" both times. Not usable this session; proceeding on own judgment +
empirical build/lit grounding via sub-agents.

**Bold dead-weight sweep DONE (Workflow w6ziod5qt → research/orphan-sweep-plan.md).**
Honest finding: d0934210 already took the biggest chunk; what it ORPHANED was the
route-validation / metadata-mirror CONTRACT cluster the golden validator was the
sole consumer of (NOT the RVVTargetSupportBundle string tables — those turned out
LIVE on the export/header-emission path; sweep dropped that candidate). Preservation
boundary held: export path + I7 guards + live get*RouteFacts/add*MetadataMirrors
emitters + CapabilityModel.

Executed in risk order, each its own commit, all build-green + lit 535/3 baseline:
- **4740b7c2** Batch C+B: zero-caller orphan tails (route-ID wrappers + 13 dead
  construction-protocol string-mirror getters + 5 role-op wrappers + helpers). −139.
- **43d44446** Batch A (the prize): the route-validation/metadata-mirror contract
  cluster. Estimated ~4,800; actual −6,919 across 7 EmitC files (full closure incl.
  transitive private helpers + member structs + getRVVContractionArtifactContractCore).
  Removed by-symbol, closed under -Wunused-function via from-scratch full re-link →
  0 symbols restored (closed-graph held exactly). trellis-check verdict: I4/I5/I7/I8
  clean (deleted structs self-describe as "consistency mirrors, never authority" = the
  I4 病灶 the thesis names). Preservation boundary verified intact.
- Batch D (prune ConstructionProtocolCommonTest golden-style gtest, ~95 lines):
  DEFERRED — it covers a LIVE I7 construction-protocol gate (verifyRVVConstruction-
  ProtocolReady) and was just op-count-realigned in f418cdf9; pruning trades real
  coverage of a live invariant, not clean dead weight. Documented, not done.

**Session total: ~−7,283 lines of dead weight, 0 behavior change, 0 new lit reds.**
Stage 1 去伪 pure-deletion now substantially exhausted.

**Next:** direct-C guard dedupe (DEDUPE-INTO-SHARED-HELPER, a "重构"/aesthetics win,
IF the two containsForbiddenDirectCMarker copies are truly identical — they guard a
live I7 path, must not weaken). Then scope Stage 2 (N1: typify CapabilityDescriptor
std::string/map → real MLIR Attr with provides/implies/conflicts in ODS) — the
highest-novelty next refactor, but build-work not deletion.

**Advisor:** unavailable across 3 calls (incl. after /advisor Fable 5). Not usable
this session; every step instead gated on independent trellis-check + from-scratch
build + full lit + tree-wide grep.

## 2026-06-12 — Stage 2 N1 typification + the measurement-mirage finding

**Stage 2 (N1) — capability relations now a typed MLIR object, string repr deleted:**
- `ad013a9a` typed `TCRVExec_CapabilityRelationsAttr` (Exec dialect's first AttrDef;
  StringAttr ids NOT SymbolRef — by-id namespace ≠ requires=[@sym]); round-trip test.
- `e0a37f64` N1 first evidence: descriptor build prefers the typed attr (string
  fallback), so CheckCapabilityRequires conflict-legality is driven by the typed
  attr. trellis-check proved load-bearing by neutralization.
- `431d43f3` Phase-A: migrated 14 fixtures to typed relations + DELETED the string
  relation representation (verifyCapabilityIDRelationAttr + collectCapabilityID-
  Relation + the k*AttrName constants + the fallback). Relations now flow ONLY via
  the typed attr. (Phase-A's first agent hit the session limit mid-deletion leaving
  a broken build — `collectCapabilityIDRelation` deleted but still called; a second
  agent finished it.)

**⚠️ MEASUREMENT MIRAGE (critical):** the Phase-A agent's clean rebuild exposed that
the real lit baseline was NOT "3 reds" — every earlier agent this session reported 3,
but a from-scratch rebuild showed 32, and at session-start commit d0934210 a clean
rebuild = 46. Cause: incremental `ninja` left stale tcrv-opt/tcrv-translate after
lib/Target/RVV edits → false-green lit. Investigated by rebuilding at d0934210
(detached HEAD): the 29 `Target/RVV/*-selected-body-artifact-*` reds were PRE-EXISTING
(present before my first commit), stale STALE-* checks asserting old route-specific
validator wording that d0934210 removed when it deleted the golden validator. **My
session introduced ZERO new failures — it reduced 46 → 32 → 3.** Fixed:
- `38e8c2d0` synced 122 STALE-*/MISSING-* CHECK lines in the 29 tests to current
  generic wording, preserving key+expected+actual (verified load-bearing by
  neutralization). Suite now 539/536 — only the 3 environmental Scripts e2e reds
  (need real RVV hw, I8).
- `c83ae744` recorded the clean-rebuild verification discipline in the dead-mirror
  guide; baseline-lit-failures.txt rewritten to the honest 3 + the mirage warning.

**Lesson:** never trust a prior agent's lit count; re-measure with a clean build,
especially after emitter/validator edits.

## 2026-06-12 — Stage 1 wrap + Stage 2 kickoff

Stage 1 去伪 wrapped: spec lesson captured (de948920 dead-mirror-removal guide),
audit-task DoD checked (audit reports + plans in task dir, subtasks established,
spec updated). Audit task PRD updated with the Stage-1 commit ledger + status.

Subtasks established (linked to 06-12-mlir-audit-refactor):
- 06-12-stage2-typify-capability-attr (N1 立真) — PRD seeded.
- 06-12-stage3-replace-string-machine (N3 换心) — stub.

**Stage 2 (N1) scoping IN PROGRESS** (Workflow wpla0gg0w, 4 readers → synthesis →
research/capability-typification-plan.md): design the capability MLIR Attribute
(provides/implies/conflicts), map the ~CapabilityModel consumer surface, pin the
minimal ADDITIVE first step (define the AttrDef parallel to the string model,
breaks nothing) + the N1 first-evidence decision point. CapabilityModel today =
std::string/map (lib/Support/CapabilityModel.{cpp,h}, 668 lines) — the I1 反例.
Approach = strangler-fig additive-first: add typed Attr → migrate consumers →
delete string model (per dead-mirror guide). Next: execute the first ODS step
when the scoping plan lands, then trellis-implement → check → commit.

Note: .trellis task/workspace state left uncommitted on disk (durable), per prior
session pattern; /finish-work will archive. All code+spec deliverables committed.



## Session 1: Stage 1 去伪 complete + Stage 2 N1 capability typification (relations)

**Date**: 2026-06-12
**Task**: Stage 1 去伪 complete + Stage 2 N1 capability typification (relations)
**Branch**: `main`

### Summary

Stage 1 去伪: removed ~7.3k lines dead weight (front-door fail-closed pass, golden-validator orphan tails, 6.9k-line route-validation/metadata-mirror contract cluster, direct-C dedupe). Stage 2 N1: typed TCRVExec_CapabilityRelationsAttr drives conflict-legality; Phase-A migrated fixtures + deleted the string relation IR representation. Found+fixed a measurement mirage (incremental ninja stale binaries hid the true baseline; real session-start baseline was 46 reds not 3); synced 29 stale Target/RVV fail-closed tests -> suite at honest green (3 environmental reds). Added dead-mirror-removal guide + clean-rebuild verification discipline.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `f418cdf9` | (see git log) |
| `8d022042` | (see git log) |
| `4740b7c2` | (see git log) |
| `43d44446` | (see git log) |
| `21dc35a9` | (see git log) |
| `de948920` | (see git log) |
| `ad013a9a` | (see git log) |
| `e0a37f64` | (see git log) |
| `431d43f3` | (see git log) |
| `38e8c2d0` | (see git log) |
| `c83ae744` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete

## 2026-06-13 — Stage 3 换心: first family converted on real hardware

Goal (user): 完成真正的去除字符串/换心 + Stage 2 along the way + ssh rvv 灯证据 +
clean/aesthetic/real-research. ssh rvv CONFIRMED live (rv64imafdcv...zve64d, clang).

Fable architecture decision (Workflow wecebi3aa → research/heart-replacement-plan.md):
**HYBRID** = real MLIR DialectConversion (RewritePatternSet+OpConversionPattern+
TypeConverter+ConversionTarget — the repo's FIRST, fixes the 0-ConversionPattern red
flag) over the EXISTING typed tcrv_rvv ops; pattern bodies build emitc directly on
typed SSA Values (no string plan, no re-parser); emitc.call_opaque keeps the intrinsic
NAME (idiomatic) but TYPED operands; strangler-fig per family. Beachhead: unit-stride
elementwise add (i32,m1).

Progress (commits, each build-green + lit 542/3 environmental-only):
- Baseline I8 lamp: existing add path compiles+runs+numeric-correct on real ssh rvv
  (research/baseline-add-hardware-lamp/, golden-add-emitc.cpp = equivalence oracle).
- **205460d8** conversion scaffold: TypeConverter+ConversionTarget+applyPartialConversion
  no-op pass --tcrv-rvv-lower-to-emitc. First real conversion infra in the repo.
- **bc4b0251** add-family OpConversionPatterns: typed body → structured emitc; renders
  BYTE-IDENTICAL to the hardware-validated golden (mlir-translate --mlir-to-cpp). riscv-
  IntrinsicName mangler over types; with_vl→emitc.for, runtime_abi_value→func params,
  n-i→emitc.sub, base+i→emitc.add. No leftover tcrv ops / casts.
- **769739b0** export swap: try-convert-else-fallback at TargetArtifactExport.cpp seam
  (speculative conv on a clone; full-legalize → use, else legacy). No family-name branch
  (N2/I3-clean). convertRVVModuleToEmitC shared by pass+seam. convertVectorTypeToEmitC
  guard fixed an m2 half-conversion regression. **ssh rvv lamp via the CONVERTED export:
  PASS op=add** (real riscv64, clang 18.1.3). add/sub/mul all convert; **mul lamp PASS** too.

**State:** 换心 PROVEN + operational on the LIVE export path for elementwise add/sub/mul,
hardware-validated, byte-identical output, real MLIR mechanism. HONEST caveat: the string
machine is NOT yet shrunk — conversion is so far ADDITIVE (parallel path); deleting the
elementwise string plan is gated on converting its fuller scope (masked/scalar-broadcast/
strided variants still fall back) so the fallback isn't broken. lib/Plugin/RVV/EmitC still
~82k lines until coverage grows enough to delete owners.

**Next (family ladder, repeat the proven recipe):** convert the rest of elementwise
(scalar-broadcast, masked, strided) → then DELETE the elementwise *PlanOwners string plan
(first real 去除字符串) → then reduction/macc/segment2/widening-dot → retire RoutePlanning
+ the re-parser. Each family: patterns → structural lit → ssh rvv lamp → delete its string
plan → re-green. Stage 2 Phase-B (retire string CapabilityModel, advisor #1) interleave.


## Session 2: Stage3 换心: decouple export seam from string route (STEP1); elementwise owner deletion blocked by 2nd live consumer

**Date**: 2026-06-13
**Task**: Stage3 换心: decouple export seam from string route (STEP1); elementwise owner deletion blocked by 2nd live consumer
**Branch**: `main`

### Summary

STEP1 done: materializeSelectedEmitCArtifactModule now attempts conversion FIRST and validates converted elementwise families WITHOUT the string route (new requireConvertedSelectedEmitCMaterializedHandoff checks well-formed emitc + exact exported fn name); route built only on fallback. getSelectedEmitCArtifactFunctionName no longer builds the route. Build green, full lit 3 reds==baseline, all 7 elementwise op-kinds PASS on ssh rvv. STEP2 (delete owner) BLOCKED+reverted: --tcrv-materialize-emission-plans pass is a 2nd live consumer of the elementwise statement-plan owner (61 PLAN fixtures + HEADER export chain off it); deleting broke 57 lit tests with diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner. Per dead-mirror guide rule4 / task guardrail: do NOT force. Evidence in research/elementwise-postdeletion-hardware-lamps/.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `6f3ba3ad` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 3: Stage3 换心: extend RVV->emitc conversion to i64 + m2 elementwise rungs (unblock owner deletion)

**Date**: 2026-06-13
**Task**: 06-12-stage3-replace-string-machine — UNBLOCK elementwise-owner deletion by covering the
SEW=64 (i64) and LMUL=m2 elementwise rungs the retirement attempt correctly STOPPED on.
**Branch**: `main` (HEAD 7454e64b). NOT committed (per task constraint).

### Summary

The 4 blocked fixtures (i64-add, lmul-m2-add, masked-i64-add, masked-lmul-m2-sub) fell back to the
legacy string owner because the TypeConverter only mapped i32/m1. Root cause was PURELY the type
mapping — the intrinsic mangler riscvIntrinsicName and maskWidthForConfig (all 4 {sew,lmul} mask
combos) were already type-parameterized. Extended 3 spots in lib/Conversion/RVV/RVVToEmitC.cpp:
vectorDType (+i64), the VectorType conversion lambda (generalized to {i32,i64} x {m1,m2} ->
vint<sew>m<lmul>_t), and the MaskType conversion lambda (-> vbool<maskWidthForConfig>_t). No pattern
body changed.

### Evidence

- ALL 7 in-scope elementwise rungs CONVERT (rc=0, zero leftover tcrv_rvv/casts). Full elementwise-
  arithmetic set: 30/32 CONVERT; the 2 FALLBACK (explicit broadcast-mul/sub) use legacy Stage1
  tcrv_rvv.i32_load, fail-closed at materialization BEFORE conversion (route to scalar fallback,
  never reach the elementwise emitc owner) — unrelated to this change.
- BYTE-IDENTICAL converted-vs-legacy-oracle C for all 4 harness rungs (diff -q clean).
- ssh rvv (rv64imafdcv...zve64d, clang): PASS op=i64_add, lmul_m2_add, masked_i64_add,
  masked_lmul_m2_sub (mask lanes verified).
- Build full ninja green; lit 548 discovered / 545 passed / 3 failed = pre-existing environmental
  reds (proven by stash-to-HEAD run, none touched by this change); i32/m1 still converts (no regress).

### Main Changes

- lib/Conversion/RVV/RVVToEmitC.cpp — i64 + m2 type/mask mappings.
- test/Conversion/RVV/rvv-to-emitc-i64-add.mlir (new), rvv-to-emitc-masked-lmul-m2-sub.mlir (new).

### Status

[OK] Complete — elementwise owner FULLY covered for its in-scope set; unblocked for deletion (the
deletion is the separate next step, not done here). NOT committed.

### Next Steps

- DELETE the elementwise statement-plan owner (gated separately on the 2nd live consumer
  --tcrv-materialize-emission-plans per Session 2's blocked STEP2).

---

## 2026-06-13 — Stage 3 换心: elementwise owner RETIRED (the first 去除字符串)

Task 06-12-stage3-replace-string-machine. Deleted the 736-line
`RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp` after decoupling ALL FIVE
consumer paths that reach it. HEAD baseline 4de23a4e. NOT committed.

**The owner was reached through 5 paths; decoupled with ONE shared gate.** Added
`rvvSelectedBodyFullyConvertsToEmitC(request)` to the route-provider lib
(RVVEmitCRouteProvider.cpp) — clones the enclosing module and runs
`convertRVVModuleToEmitC` under a quiet diagnostic handler, returning true iff the
selected body fully legalizes (a converted family). Zero family-name branch.
Added `TianChenRVConversionRVV` dep to the route-provider CMake (no cycle).

1. export-materialize (TargetArtifactExport.cpp:2137) — ALREADY decoupled (7454e64b). Unchanged.
2. C1 emission-plan (RVVExtensionPlugin.cpp:617) — converted → describe(nullptr), route discarded.
3. C2 candidate-validation (RVVTargetSupportBundle.cpp:423) — converted → description-only branch.
   **CORRECTION to task instruction:** the route-id mirror compares against the candidate's
   `emitc_lowerable_route` metadata which carries `emitCRouteID` (RVVConstructionProtocol.cpp:988),
   NOT `targetArtifactRouteID`. The route's getRouteID() == description.emitCRouteID, so the
   decouple redirects to `description.emitCRouteID` (using targetArtifactRouteID would always FAIL).
   Provenance + ABI route-checks are subsumed (with_vl guaranteed by full conversion; ABI re-checked
   at :581 candidate-vs-description). Refactored validateRVVRouteMetadataMirrorsSelectedBody to take
   a StringRef routeID.
4. PATH B boundary can-build gate (RVVExtensionPlugin.cpp:381) — converted → return success (skip
   string can-build); non-converted keeps firing.
5. PATH R materialize-emitc-lowerable-routes pass (EmitCLowerableMaterialization.cpp:305) —
   ARCHITECTURE DECISION resolved (heart-replacement-plan.md:287): the pass try-converts a clone;
   fully-converted → replaceModuleBodyWithMaterializedEmitC with the converted module; else fall
   back to the string route. **EMPIRICAL byte-identity:** conversion output == string-plan output for
   in-scope bodies EXCEPT a harmless appended `specifiers = ["extern","C"]` on emitc.func. All 6
   in-scope PATH R fixtures' substring CHECK lines (func name + `tcrv_emitc.source_op=… callee=…`
   verbatims) match the converted output byte-identically → NO fixture migration needed. cmp-select
   (in the mixed rvv-generic-stage2-materialization.mlir) stays on the owner via the gate.

**Deleted (820 gross):** owner file 736 + RVVEmitCStatementPlanOwners.cpp 42 (table entry + predicate
+ 3 classifiers) + header decls 24 + struct 17 + CMake 1. Added ~135 (helper + 4 gates). Net ~685.
Grep-gate: zero references to any deleted symbol.

**Validation:** build GREEN (tcrv-opt+tcrv-translate relinked). lit 545/548, 3 environmental reds
(widening-dot dequant dry-run + self-test — same as baseline, zero new). ALL 10 op-kinds PASS on
REAL ssh rvv (riscv64, no --dry-run): add/sub/mul/scalar_broadcast_add/masked_add/strided_add
(explicit mode); i64_add/lmul_m2_add/masked_i64_add/masked_lmul_m2_sub (--pre-realized-selected-body
mode). evidence.json status=success ssh_evidence=true riscv64 for each →
research/elementwise-postdeletion-hardware-lamps/*-evidence.json.

## 2026-06-13 — FIRST real 去除字符串: elementwise owner retired (hardware-validated)

After proving the conversion + decoupling the export seam, drove to an ACTUAL string
deletion. Two prior retirement attempts correctly STOPPED at blockers (i64/m2 didn't
convert; then PATH R discovered) — fixed each:
- **4de23a4e** extend conversion to i64+m2 elementwise ({i32,i64}x{m1,m2} + vbool
  masks) → all in-scope rungs convert byte-identical + ssh rvv PASS.
- **7454e64b** decouple the export-materialize seam from the string route (self-
  validate the converted module).
- **57255ba9** RETIRE the 736-line elementwise owner (−668 net). The owner had FIVE
  consumer paths (the string plan is the shared substrate of every materialization/
  validation/diagnostic entry point): export [done], C1 emission-plan diag (route
  discarded→nullptr), C2 candidate validation (route→description.emitCRouteID +
  source ABI ops; provenance/ABI checks subsumed), PATH B boundary can-build gate
  (converted→success), PATH R --tcrv-materialize-emitc-lowerable-routes pass
  (try-convert; converted emitc byte-identical to string-plan emitc → 6 fixtures
  unchanged). One shared family-agnostic gate rvvSelectedBodyFullyConvertsToEmitC.
  trellis-implement caught a primary-source correction (C2 must use emitCRouteID not
  targetArtifactRouteID) + disproved the prior PATH-R "differs" claim.

I4-honest: PLAN/HEADER/STALE-ELEM-* golden mirrors are description-derived (different
file, kept) → byte-identical, NO fixture weakened/migrated. Build green; full lit
548/3 environmental (all families pass, non-elementwise unaffected); all 10
elementwise op-kinds re-PASS on real ssh rvv with the owner GONE.

**The retirement PATTERN is now proven + repeatable**: convert all of a family's
rungs (byte-identical + ssh rvv) → decouple every consumer path behind the shared
gate → delete the owner → re-validate on hardware. Remaining: ~5 more *PlanOwners
(compare-select, base-memory, MAcc, contraction/widening-dot, segment2) + the 40k
RoutePlanning, each via the same pattern. lib/Plugin/RVV/EmitC still ~87k → the bulk
remains, but the first owner is GONE and the mechanism is hardware-proven.

**Advisor:** advisor() tool unavailable throughout; used Agent model=fable for the
architecture + retirement plans (excellent ROI — Fable's plans were precise; sub-
agents caught Fable's over-claims empirically, e.g. i64/m2 coverage + PATH R).

---

## 2026-06-13 — Check Agent adversarial verification (regression fix + cmp-select retirement)

Independent adversarial check of the uncommitted fix (RVVToEmitC.cpp guards +
producedFunc gate) layered on the cmp-select owner retirement (staged delete of
RVVEmitCCompareSelectStatementPlanOwners.cpp + 4 new positive tests + 8 re-targeted
negatives). VERDICT: **correct, honest, safe to commit. No self-fix needed.**

- **Clean-rebuild lit = 3** (only the environmental widening-dot-reduce dry-run x2 +
  e2e-self-test; all reference reduction/dequant families, none touch the changed
  conversion). Confirmed tcrv-opt (04:10) newer than the static libs (04:08).
- **producedFunc gate sound**: convertRVVModuleToEmitC returns false unless an
  emitc.func materialized from an RVV body; non-RVV families fall through.
- **Mislowering guards SOUND+COMPLETE**: adversarially probed i64-load-from-int32-buf,
  undisturbed-tail-policy, f32-from-int32-buf, i64-strided (byte-stride=8 not 4) →
  all either lower byte-correct or REFUSE (zero intrinsics, variant stays illegal).
  bufferPointeeMatchesVectorElement `.contains` substring: int32_t⊄int64_t (the real
  width-miscompile danger is caught); only uint32_t/int32_t collide (same 4B width,
  harmless).
- **8 negatives honest**: 3 i64 removed sections (sub/m1, add/m2, single-load x+x) now
  genuinely materialize (probed: correct vsub_vv_i64m1 / vadd_vv_i64m2 / %5,%5 reuse);
  retained sections all fire at the OP-VERIFIER level (c_type, output-buffer role, sew
  mismatch, vl-ownership). Removed = legacy owner/route conventions (role labels,
  construction order, structural counts); retained = op-verifier type/semantic errors.
  4 positive tests assert real intrinsics (vmseq/vmerge/vmflt), not tautologies.
- **cmp-select retirement clean**: zero dangling refs to deleted symbols; unused
  Family::CompareSelect enumerator harmless (build green).
- **Hardware (ssh rvv, no --dry-run, pre-realized mode)**: add / masked_add /
  strided_add / i64_add / cmp_select / f32_clamp_select → all 6 PASS on riscv64.

## 2026-06-13 — 2nd owner (compare-select) + a SHIPPED regression caught & fixed

**Integrity scare (important):** commit a1292e05 (elementwise owner deletion + the
PATH-R conversion gate) actually shipped 18 lit regressions — I reported "548/3"
from an incremental build that left tcrv-opt STALE (the depfile bug: ninja rebuilds
the lib but doesn't relink the tools; `touch+ninja` is NOT enough — must `rm` the
binaries). A clean rebuild revealed the true 21 reds. Two root causes:
1. convertRVVModuleToEmitC returned true VACUOUSLY for non-RVV bodies (no emitc.func)
   → the gate used the unconverted body as "materialized" → broke toy/template/
   tensorext materialization.
2. The conversion MISLOWERED some bodies to type-broken C (undisturbed tail/mask
   policy; ABI buffer c_type ≠ vector element, e.g. const int32_t* feeding vle64).
   → my earlier "byte-identical + hw-validated" was only true for well-formed bodies;
   the conversion was unsafe for malformed ones.
Fixed (commit 8c3449c1→amended): producedFunc gate + 2 mislowering guards (policy +
bufferPointeeMatchesVectorElement) → malformed bodies fall back to the legacy
validator which errors correctly; re-targeted 8 obsolete legacy-scope negative tests
(kept all genuine op-verifier negatives). trellis-check verified ADVERSARIALLY
(probed 4 type-broken bodies → all rejected; each removed section confirmed obsolete);
clean rebuild (forced fresh link) → exactly 3 environmental reds; 6/6 op-kinds PASS
on real ssh rvv. Hardened the dead-mirror guide with the rm-binaries recipe (commit).

**2nd owner RETIRED**: RVVEmitCCompareSelectStatementPlanOwners.cpp (1,335 lines).
The shared gate auto-decoupled it once the compare-select family converted (vmseq/
vmslt/vmsle + f32 vmf*/vfmv_v_f + vmerge select + dequant-clamp epilogue). −887 net.

**Lesson reinforced:** NEVER trust a lit count without rm-ing the tool binaries first.
The conversion needs malformed-body guards (it's permissive by default — applyPartial-
Conversion lowers what it can; the legacy validators that caught bad bodies were in
the owner/route path, so each owner retirement must preserve or re-guard that
validation). Going forward: rm-binaries + full lit + adversarial guard probe per owner.

---
## 2026-06-13 — Stage 3 换心: reduction owner (3rd owner) — CONVERTED reduce_add, owner BLOCKED (not deletable)

Target: RVVEmitCReductionAccumulationStatementPlanOwners.cpp (1,950 lines).
SCOPE FINDING (load-bearing): the file is "Reduction**Accumulation**" — it owns FOUR
dispatch builders, not one: Reduction(reduce_add), StandaloneReduction(add/min/max +
computed-mask + runtime-scalar-cmp-masked + WIDENING-standalone), PlainMAcc(macc +
scalar-broadcast), ComputedMaskAccumulation(masked macc). Deleting the file requires
ALL FOUR to convert. Only reduce_add is cleanly tractable by the elementwise-style
pattern. So the owner is BLOCKED for deletion (convert-what's-tractable + STOP, per prd).

Blockers (verified file:line):
- StandaloneReduce = scalar-carry-through-memory (pre-loop init out[0] + in-loop RMW of
  out[0]); min/max/m2/masked layers; the WIDENING-standalone rung uses
  `__riscv_vwredsum_vs_i16mf2_i32m1` (mf2 fractional-LMUL the conversion can't do —
  RVVEmitCRoutePlanning.cpp:5108) = the same env-red family.
- PlainMAcc / ComputedMaskAccumulation = multiply-accumulate (tcrv_rvv.macc) — separate effort.
- widening-DOT-reduce (the 3 baseline env reds) is correctly in the SEPARATE contraction
  owner (RVVEmitCContractionRouteFamilyPlanOwners.cpp); 0 widening_dot builders in this file.

DONE: added emitReduce (tcrv_rvv.reduce -> __riscv_vredsum_vs_i32m1_i32m1(input,acc,vl),
lane-0 result stored with VL=literal "1") + reductionMnemonic + riscvReductionIntrinsicName.
Byte-identical to legacy export (full e2e export C diff = 0). Structural lit
test/Conversion/RVV/rvv-to-emitc-reduce-add.mlir. Migrated the materialization fixture +
the negative fixture stays red-via-fallback.

GUARD LESSON (caught a real mislowering): the negative fixture rvv-generic-stage2-
reduction-negative.mlir (broadcast_load accumulator) was being CONVERTED (mislowered) —
the legacy rejects it (RoutePlanning.cpp:20663 `isReduction && hasRHSBroadcastLike`).
Added the guard: reduce input AND accumulator must be explicit tcrv_rvv.load (not
broadcast/splat) → falls back → legacy errors. Plus result_layout guard (chunk-base only,
not scalar-standalone). Adversarial probe: undisturbed policy / wrong layout / ABI
mismatch / broadcast seed → ALL refused.

Fresh-link (rm'd both binaries) full lit = EXACTLY 3 env reds (553 tests, +1 my structural
test). ssh rvv (riscv64): PASS op=reduce_add counts=0,1,16,17,257, ssh_evidence=true.
40 standalone-reduce/macc fixtures unaffected (strangler-fig isolation holds).
Owner NOT deleted — reported blocked. -0 net lines (additive conversion only).

## 2026-06-13 — MAcc family + Stage 2 Phase-B.1 (capability resolution typed)

Two threads (fable-scoped):
- **Stage 3 MAcc** (commit 8c450bb3): converted plain/scalar-broadcast/computed-mask
  macc (vmacc_vv + vmerge), byte-identical, ssh rvv PASS, guards caught 5 mislowering
  cases. Removes 2 of the reduction owner's 4 families (reduce_add + MAcc convert;
  StandaloneReduction's mf2-fractional widening + the rest still block deletion).
- **Stage 2 Phase-B.1** (commit 8d4c02b6, Fable plan capability-model-retirement-plan.md):
  CapabilityDescriptor now STORES the typed CapabilityRelationsAttr (not 3 string
  vectors); providesID/impliesID/conflictsWithID scan it directly. Capability conflict
  resolution (the primitive 4 passes share) is now genuinely TYPED at the resolution
  layer — advisor's #1, I1 no longer cosmetic. Deleted the restringify (collectTyped-/
  sourceCapabilityIDRelation/containsID). Chokepoint retype → all ~13 consumers + 4
  passes migrate with ZERO pass edits. Kept TargetCapabilitySet (IR+ssh-rvv-probe union,
  permanent N1/N3 infra) + properties/kind/status (Phase-B.2/C).

**2nd depfile trap (worse):** header/struct-layout change → other targets' .o don't
recompile against the changed header → ABI mismatch → runtime corruption, build green.
`rm tcrv-opt` insufficient; need `ninja -t clean && ninja && ninja bin/tcrv-opt
bin/tcrv-translate`. (A "494 reds" full-clean scare was just plain `ninja` not building
the tools → exit 127.) Verified Phase-B.1 clean → 3 env reds. Guide hardened (commit).

**Session tally:** Stage 1 去伪 archived; Stage 2 N1 — relations typed in IR (Phase-A) +
resolution typed (Phase-B.1), N1 real; Stage 3 — conversion infra (first ConversionPattern)
+ elementwise/compare-select/f32-clamp/reduce_add/MAcc converted + 2 owners deleted
(elementwise, compare-select). Remaining去除: standalone-reduce(mf2), memory/segment/
indexed/widening/contraction owners + 40k RoutePlanning; Stage 2 Phase-B.2 status enum.


## Session 3: Stage 3 换心: RVV body-emission string machine fully retired (all families real MLIR, ssh rvv 灯)

**Date**: 2026-06-13
**Task**: Stage 3 换心: RVV body-emission string machine fully retired (all families real MLIR, ssh rvv 灯)
**Branch**: `main`

### Summary

Completed the RVV codegen 去除字符串 (the audit's 病灶). Flipped the last per-family owner's Gearbox dequant to typed IR: built the conversion + 2 ODS ops (packed_i4_nibble_unpack_product, with_vl.unroll_factor); flipped packed-i4 (single-scope, emitc.variable accumulator) and grouped (single-slice + unroll-expand, single-slot invariant intact) — both ssh rvv PASS tolerance=1e-05. Converted standalone dequantize-i32-to-f32 (gearbox.unroll realization-structural attr). A poison-test of the legacy fallback proved ZERO valid RVV bodies remain on the string path, then DELETED the RVV body-emission string machine: -19k lines (2 statement-plan owner files + the route-CONSTRUCTION half of RoutePlanning 41k->33k + provider-plan/registry layers), replaced with an I7 fail-closed guard; live family-classification predicates decoupled byte-faithfully; cleared dead header residue. KEPT (separate ADRs per advisor): the analyze/describe PLAN/HEADER description engine, the generic TCRVEmitCLowerableMaterializer re-parser (still used by the non-RVV skeleton/demonstration families Toy/Template/TensorExtLite — the N2 generalization layer, no hardware). All verified at full-clean-rebuild + fresh-link: 589 tests, exactly 3 environmental reds. EmitC dir ~88k->63k. OPEN SCOPE QUESTION for the user: is '去除字符串' the RVV 病灶 (done) or all string machines incl. the N2 re-parser?

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `7bfd6012` | (see git log) |
| `5dc65ec7` | (see git log) |
| `b270dcb3` | (see git log) |
| `e7bca68b` | (see git log) |
| `311af0b1` | (see git log) |
| `b012995e` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 4: Modular backend-emission base + re-parser deleted (no string machine in production)

**Date**: 2026-06-14
**Task**: Modular backend-emission base + re-parser deleted (no string machine in production)
**Branch**: `main`

### Summary

Directives 1+3: built a TypedBackendEmissionDriver base + table-driven BackendEmissionRegistry (mirrors the plugin registry → a future RVM backend is one-line, zero core edits), removed the 2 hardcoded convertRVVModuleToEmitC call sites. Migrated the non-RVV skeleton families (Toy/Template/TensorExtLite) off the generic string re-parser onto typed emitc emission via the base (byte-identical), then DELETED the re-parser TCRVEmitCLowerableMaterializer (the 自写C表达式 re-parser 病灶, 1269+35 lines) + its unit test + 5 test-consumer excisions; both production fallbacks now FAIL-CLOSED (every family must have a typed driver — no string-route escape hatch). 28 negatives re-targeted (fail-closed preserved, variant-specific, no-synthesis guard). All behavior-preserving / byte-identical, full-clean-rebuild + fresh-link, 589 tests / 3 environmental reds. DEFERRED (separate description-engine ADR): the TCRVEmitCLowerableRoute struct + its std::string expression 病灶 fields remain, coupled to the live ~57k RVV route-planning/cost/header-evidence cascade (PLAN/HEADER asserted by ~167 fixtures). FOLLOW-UPS for directive 2: surface the typed drivers' notifyMatchFailure reasons (the 28 negatives' specific diagnostics coarsened); delete ProviderPlanVerifier dead fields. Directive 4 pending: ~290 redundant/smoke test fixtures + metadata-mirror consolidation per the test inventory.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `128af6ee` | (see git log) |
| `850902a6` | (see git log) |
| `767c76ac` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
