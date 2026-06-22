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


## Session 5: Description-engine retirement: std::string struct deleted (MOVE 1); resource metadata kept as N3 evidence (MOVE 2A reverted)

**Date**: 2026-06-14
**Task**: Description-engine retirement: std::string struct deleted (MOVE 1); resource metadata kept as N3 evidence (MOVE 2A reverted)
**Branch**: `main`

### Summary

Engine-first (user: 两个都做引擎优先). MOVE 1 (committed fc9aa69f): deleted the TCRVEmitCLowerableRoute std::string carrier + 10 sub-structs + 4 route-builders + the ExtensionPlugin::buildVariantEmitCLowerableRoute virtual, swapping the build-and-discard validation gates to conversion-success validation — the LAST std::string-expression 病灶 gone, -1431 lines, artifacts byte-identical, behavior-preserving. MOVE 2A (attempted, REVERTED): tried to delete ~15k lines of low_precision_resource.* synthesis + shrink the shipped .h (user-approved under a 'vestigial' premise), but careful staged execution disproved the premise — the metadata is N3 gearbox-tune-measurement evidence: the e2e harness gate4 (the '实测胜出' tune-measurement gate) consumes ~27 candidate/measurement keys with NO typed survivor, and the packed-i4 numerical oracle gated on selected_candidate (re-sourceable from the typed nibble-op, but gate4 not). Reverted to protect N3 (user owns the tune). Saved the finding to project memory (low-precision-resource-is-n3-evidence) so no future session re-deletes it as vestigial. Lesson: Target/RVV lit 178/178 is INSUFFICIENT verification — it doesn't exercise the harness oracle/measurement gating. Net description-engine outcome: the genuinely-vestigial std::string struct removed; the I4-looking-but-actually-N3 resource metadata kept. Remaining: directive 4 (test cleanup) + bounded directive-2 (ProviderPlanVerifier dead fields, diagnostics-coarsening, construction-manifest base).

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `fc9aa69f` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 6: Directive 2/3/4 finish: dead-code swept, modular base done, test suite + description engine validated

**Date**: 2026-06-14
**Task**: Directive 2/3/4 finish: dead-code swept, modular base done, test suite + description engine validated
**Branch**: `main`

### Summary

Closed out the post-去除字符串 directives. Directive 3 (modular): the TypedBackendEmissionDriver base + table-driven registry is the RVM-ready seam; the construction-protocol already has a shared base (ConstructionProtocol.cpp) + Template intentionally mirrors Toy as the worked example, so the construction-manifest 'base' was correctly LEFT (don't factor intentional duplication). Directive 4 (test cleanup): conservative curation found 0 safe-to-archive — no byte-identical dups, no superseded intermediates; the per-op-kind×per-path×per-width parallelism is load-bearing coverage (115 fixtures are by-path inputs to the e2e harness), so 'complex/redundant' is actually comprehensive coverage; a reduction needs per-cluster policy calls (user's). Directive 2 (删除失效) dead-code-only sweeps: ProviderPlanVerifier typedef+field (18 sites), the -Wunused residue (printField/printQuoted + the dead artifact-bridge test scaffolding + 11 orphaned constants + ScopedTempPath), the empty TCRVEmitCLowerableInterface.cpp TU, and the 9 fully-dead route-family Owner struct+getter pairs (-84) — all grep+full-link certified, lit unchanged at exactly 3 environmental reds throughout. Remaining minor/noted follow-ups: surface the typed drivers' notifyMatchFailure reasons (the 26 fail-closed negatives' diagnostics coarsened to generic — a maturity refinement, not a regression that breaks anything); per-cluster test-reduction is a user policy decision.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `5e67adbc` | (see git log) |
| `fe33faca` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 7: File-modularization: the 3 giant monoliths (33k/14k/11k) split into per-concern files — no 10k+ file remains

**Date**: 2026-06-14
**Task**: File-modularization: the 3 giant monoliths (33k/14k/11k) split into per-concern files — no 10k+ file remains
**Branch**: `main`

### Summary

Behavior-preserving file-modularization (审美/maturity) of the survivor monoliths, all relocation-only (byte-identical bodies, lit 589/3 + zero fixtures moved throughout, each verified by full-clean-rebuild). RVVEmitCRoutePlanning.cpp 33,148 -> 6 files (largest RouteAnalysis 9,006, floored by the single 4,771-line collectRVVSelectedBodyRouteSlice). RVVDialect.cpp 14,426 -> 9 files (ODR-critical .inc/initialize/registration stays in the 3,682-line core; op verifiers split per category, largest 3,751). RVVEmitCContractionRouteFamilyPlanOwners.cpp 10,906 -> 5 files (largest the N3 low-precision-resource cluster 4,729, relocated untouched). Each split uses a co-located implementation-private internal-API header (NOT under include/) for the cross-TU decls + promoted-from-static helpers; public headers unchanged. Result: NO source file over ~9k (was 33k); the 5-9k survivors are reasonable for a compiler. The one genuine remaining maturity smell is the 4,771-line collectRVVSelectedBodyRouteSlice single function — decomposing it is a behavior-CHANGING logic refactor (riskier than relocation), noted for a possible follow-up.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `5728f39d` | (see git log) |
| `62bb1e09` | (see git log) |
| `26b5f995` | (see git log) |
| `cb71a4ee` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 8: Maturity: both giant functions decomposed (4.7k + 3.2k) — no monolith file or function remains

**Date**: 2026-06-14
**Task**: Maturity: both giant functions decomposed (4.7k + 3.2k) — no monolith file or function remains
**Branch**: `main`

### Summary

Decomposed the two genuine giant functions via EXTRACT-METHOD (verbatim statement relocation into phase helpers taking shared state by ref; NO logic rewrite/reorder; lit-green + zero-fixtures-moved verified after EACH of the ~56 total extractions, not batched). collectRVVSelectedBodyRouteSlice 4,771 -> 796 orchestrator + 8 phase helpers (gearbox-resolve / generic-op-collect / shape-guards / op-count / memory-ABI-bind / out-ABI-resolve / epilogue-ABI / shape-dispatch). buildRVVSelectedBodyExecutableRoleSteps 3,159 -> 1,104 orchestrator + 46 append*RoleSteps helpers (one per operation family). Both left an irreducible ~54-bool predicate spine inline (extracting it would force an isX->flags struct rewrite = a logic change, forbidden); two cohesive if/else-if chains (validateRVVSelectedBodyShapeDispatch 1531, shape-guards 1175) left whole because splitting an else-if into a fresh if changes mutual-exclusion semantics. Verified: no awk-flagged other giants are real (1830/2352 were span-to-next-anchored artifacts; actual fns 125/157). Net maturity state: NO source file over ~9k (was 33k), NO function over ~1.6k (was 4.7k), all behavior-preserving. The 1.5-1.6k survivors (shape-dispatch, analyzeRVVSelectedBodyRoute) are normal compiler sizes.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `2456955a` | (see git log) |
| `0eab0a2c` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 9: Trellis close-out: both tasks finished + archived; clean workspace

**Date**: 2026-06-14
**Task**: Trellis close-out: both tasks finished + archived; clean workspace
**Branch**: `main`

### Summary

Per the user's request, ran the trellis finish flow to clean the workspace for next actions. trellis-check final quality gate PASSED: full clean rebuild green + fresh-link + zero new warnings; 589/586/3 (the 3 reds proven red-at-baseline + independent of the session, even reproduced on b270dcb3~1 — they are local dry-run/self-consistency asserts, NOT regressions); I5/I7/I8 spot-checks pass; the audit's headline '0 ConversionPattern / not real DialectConversion' red flag is CLOSED (TypedBackendEmissionDriver runs a genuine mlir::applyPartialConversion). Spec needed NO update (it is abstract invariants, already accurate — the code now COMPLIES with I5; status lives in journal not spec, per spec discipline). Archived both tasks to archive/2026-06/: 06-12-stage3-replace-string-machine (criteria 1-4 met; criterion 5 partial-by-design = the intentional strangler-fig STOPs documented in its prd) and 06-12-stage2-typify-capability-attr (capability relations/resolution/status typed; Phase-B/C deferred per prd). Current task pointer cleared; 0 active tasks; git tree fully clean.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `71af66cd` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 10: N3 性能灯 ON (2 families e2e + deployable) + N1 divergence LIVE — goal novelty core achieved

**Date**: 2026-06-15
**Task**: N3 性能灯 ON (2 families e2e + deployable) + N1 divergence LIVE — goal novelty core achieved
**Branch**: `main`

### Summary

Achieved the goal's two evidence-bearing novelty claims on real ssh rvv. N3 性能灯 ON: built a REAL resource-aware autotuner (selectRVVLowPrecisionMaxLegalAccumulatorLMULRung — enumerates the LMUL ladder, prunes by the vreg budget [the prune BINDS], per-family cost models: byte two-widening i8->i16->i32, i16 dot-reduce one-widening i16->i32) that produces the deferred-wide winner AUTOMATICALLY from a kernel (new tcrv_rvv.widening_accumulate + deferred_accumulate ops, I5-structural) and WINS on 2 families e2e vs genuine-scalar AND competent-naive-RVV: byte int8 contraction 4.1-10.8x scalar/3.3-5.4x naive, i16 dot-reduce 4.3-7.5x/2.1-3.8x. BOTH deployable (.o/.h bundles, ssh-rvv abs_err=0). Method: P-A fair 3-way measure (the prior '0.76x regression' was an autovectorized-baseline artifact) -> P-B1 sweep found the win is resource-aware max-LMUL selection -> P-B2 selector -> P-B3/4 emission+灯 -> P-B5 selector-driven e2e -> P-B6 deployable bundle -> P-B7/8/9 2nd family (i16) e2e+deployable. N1 覆盖增量: the capability authority derives the divergence axes from real ISA facts; the same SEW=64 kernel diverges purely by --march (rv64gcv accept/zve32x fail-closed), no fixture attrs (the research's 'zero writers' gap closed); coverage += f64 + wide-LMUL m2/m4/m8. Honest residuals (incremental/out-of-scope): N1 hardware-probe ingestion (march-selection-live, not ssh-rvv-probe-live, I6); N1 materialize pass opt-in; N3 bounded-binary realization; more RVV features (fp16/segment3-4/vrgather) asymptotic 覆盖完全. 13 commits, each ssh-rvv-validated, full-clean-rebuild-green, 601 tests/3 documented environmental reds, all narrow/existing paths byte-identical. The project moved from '通但慢' to a measured 2-family hardware win.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `7ea69566` | (see git log) |
| `b07dd5cb` | (see git log) |
| `b141cad3` | (see git log) |
| `03223f5e` | (see git log) |
| `2af0663e` | (see git log) |
| `97e96fe6` | (see git log) |
| `07f844d5` | (see git log) |
| `a525d630` | (see git log) |
| `ec50b227` | (see git log) |
| `087d7aee` | (see git log) |
| `a5e0b4fe` | (see git log) |
| `28be2aad` | (see git log) |
| `ee455b67` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 11: N1 coverage: compiler-emitted kernel really replaces ggml_vec_dot_q4_0_q8_0 in LIVE llama-2-7b inference

**Date**: 2026-06-15
**Task**: N1 coverage: compiler-emitted kernel really replaces ggml_vec_dot_q4_0_q8_0 in LIVE llama-2-7b inference
**Branch**: `main`

### Summary

Pushed N1 coverage breadth to a REAL llama.cpp kernel. Research (3 primary-source files) mapped ggml_vec_dot_q4_0_q8_0 (the hot Q4_0 weight-matmul; board runs llama-2-7b Q4_0): asymmetric i4xi8, offset-binary nibble decode, per-block dual fp16 scale, AoS QK=32 block loop. INC-1 (de5d5db3): new op packed_i4_offset_binary_x_i8_product — the integer core; key trick (nibble-8)==twos_complement(nibble XOR 8) so a single vxor 0x88 reuses our sign-extend path; byte-exact vs ggml _generic over 4005 blocks on ssh rvv. INC-2a (f6f1a73a): new op q4_0_q8_0_block_dot lowers the COMPLETE kernel as STRUCTURED emitc IR (emitc.for block loop + AoS addressing via add/mul + fp16 read via call_opaque + INC-1 core + emitc.expression fp32 accumulate). CRITICAL PROCESS: the first INC-2a attempt emitted the kernel as raw() C-string blobs (the string-machine I5 rejects) — I caught it, reverted, and rebuilt structurally (advisor-confirmed); zero raw(), every value an emitc node. Byte-exact vs ggml's REAL RVV kernel AND _generic over ~5900 adversarial cases under ALL -ffp-contract flags (the emitc.expression wrap fixed a 1-ULP =on FMA-fusion gap). INC-3 (6a3b384f): dropped the compiler-emitted kernel into llama.cpp's ggml_vec_dot_q4_0_q8_0 dispatch on the board; LIVE llama-2-7b greedy inference token-for-token IDENTICAL to stock; a canary (wrong kernel -> garbage tokens, 113M delegations counted) proves it's the live hot path, not bypassed by repack. Verified via adversarial 3-dim workflow + trellis-check (all invariants I4/I5/I7/I8 PASS). HONEST: this is N1 coverage BREADTH (a real external kernel genuinely compiled byte-exact), NOT the N1 divergence novelty bar; and our kernel is correct-but-~1.7x-slower (i8mf4 vs ggml's i8m1 LMUL anchor) -> N3 perf is the open competitive step. First time our compiler stands in for an actual llama.cpp kernel byte-for-byte in a real model.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `de5d5db3` | (see git log) |
| `f6f1a73a` | (see git log) |
| `6a3b384f` | (see git log) |
| `7f539a4b` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 12: Compiler-driven capability-aware autotuner on real llama.cpp kernels (Q4_0 beats ggml; q8_0/q4_1 parity); cost-model limits surfaced honestly

**Date**: 2026-06-16
**Task**: Compiler-driven capability-aware autotuner on real llama.cpp kernels (Q4_0 beats ggml; q8_0/q4_1 parity); cost-model limits surfaced honestly
**Branch**: `main`

### Summary

Built the mature-compiler N3 core the user mandated ('编译的高性能, not 写死内核'): the COMPILER autotunes real llama.cpp quant-dot kernels — enumerate {lmul}×{multi_block 1/2/4}×{strip_elision robust/elided} → prune by capability (deriveHasZvl128b: rv64gcv⇒V⇒Zvl128b⇒VLEN≥128, so strip-elision is correct codegen / dead strip loop for full-V; zve32x⇒robust) + vreg budget → rank by a capability-BLIND structural cost (computeBlockDotShapeCostCore in RVVGearboxSchedule.h) → select → stamp (per-kernel materialize passes). DERIVED not lookup (cost takes no capability arg; capability enters only the legality prune). N1 DIVERGENCE on real kernels: same attr-less kernel, rv64gcv→elided vs zve32x→robust strip-loop. N3: Q4_0 the compiler-emitted mb4-elided BEATS ggml's hand-written kernel ~13% (and live llama-2-7b token-identical, INC-3). INC-8: latency-aware cost — the unroll factor EMERGES from a derived structural coreLatencyDepth (base-2 + decode-prefix; q4_0=7→factor4, q8_0=2→factor2), not per-kernel constants. BREADTH: q8_0 (Family A, INC-7) + q4_1 (Family B scale+min, INC-9) both byte-exact vs ggml real+_generic, autotuner-inherited. HONEST LIMITS (the credible result, kept verbatim, not sanded off): the q4_0 beat does NOT generalize — q8_0 lands at parity (ggml's q8_0 is already optimal m2-elided, the compiler derives it) and q4_1's static pick is its WORST shape (1.58x slower; 4-scalar scale+min fold spills 49 slots under ×4 unroll = register pressure the static model can't see; measured optimum mb1-robust=parity). q4_1 is the 2nd kernel to expose a missing static-cost dimension (after q8_0's latency-saturation) — a curve-fit treadmill. The honest end-state per the N3 '实测胜出' thesis is MEASUREMENT-BACKED selection (tune-once-per-kernel,target → cache a tuning record → compile reads it; cost model demoted to prune+offline-fallback) — a scoped paper-level next phase, not yet built. All structured emitc (raw()=0), full-clean-rebuild green, 3 documented environmental reds, q4_0/q8_0/q4_1/deferred-wide byte-identical/additive.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `8a7c5e36` | (see git log) |
| `a6cdeca6` | (see git log) |
| `8518b819` | (see git log) |
| `3e639be1` | (see git log) |
| `3a32d40b` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 13: Measurement-backed autotuner (实测胜出) + super-block K-quant q6_K byte-exact; 4 real ggml kernels covered

**Date**: 2026-06-16
**Task**: Measurement-backed autotuner (实测胜出) + super-block K-quant q6_K byte-exact; 4 real ggml kernels covered
**Branch**: `main`

### Summary

INC-10: replaced the static cost model's GUESS with on-board MEASUREMENT (the genuine N3 实测胜出). Architecture = tune-once-per-(kernel,march) -> cache a human-readable tuning record (winning shape + measured_ns + audit ladder, produced by tune_block_dot.py emitting each legal candidate via the COMPILER + benchmarking on ssh rvv + byte-exact-gating) -> the materialize passes READ the record and stamp the measured-best shape (RVVBlockDotScheduleTuning.h), fail-closed-revalidated, falling back to the static cost model (now pruner+offline-fallback) when absent. This FIXED q4_1 (static picked its slowest shape m1/4/elided=0.842x; measurement crowns m1/1/elided=1.012x) and overturned the static 'more-unroll-is-better' premise (measured elided optimum is factor 1>2>4 for q8_0/q4_1, factor 4 only for q4_0 — the per-kernel truth no static model captures without a curve-fit treadmill). Honest measured results (rv64gcv -zfh, authoritative): q4_0 beats ggml ~4.5% (the no-zfh ~13% was inflated), q8_0 parity, q4_1 beats ~1.2% — all 3 at parity-or-win by the genuine measured optimum. INC-11+12 (q6_K K1+K2): first SUPER-BLOCK K-quant — a full STRUCTURED byte-exact deployable drop-in for ggml_vec_dot_q6_K_q8_K (256-elem super-block, 16 int8-scaled sub-blocks, 6-bit ql+qh unpack, per-sub-block scale in i32 -> aux32[8], then the deferred two-level fp32 fold: d=fp16(x.d)*fp32(y.d), per-lane sums[l]+=d*aux32[l] across 8 fp32 lanes, SEQUENTIAL horizontal sum into *s, no reassociation/no fma/no vector-reduce). Byte-exact *s vs ggml _generic: 0/2011 cases at -ffp-contract=off; proven via a real extern C ggml_vec_dot_q6_K_q8_K symbol. Coverage now = 4 real ggml kernels across 3 families (A: q4_0/q8_0; B scale+min: q4_1; super-block K-quant: q6_K). raw()=0 throughout; clean rebuilds green; 3 documented environmental reds; all additive. Honest residuals: q6_K VLEN>=128-pinned (no <128 re-strip); q6_K perf not yet measured (K3 = tuner inherits q6_K); more K-quants (q4_K etc.) + the broader kernel set remain (asymptote).

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `4c2999b9` | (see git log) |
| `ae479367` | (see git log) |
| `f7f90eac` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 14: All 3 goal fronts advanced: GEMM perf win + forward-pass op family (all 3 structural classes byte-exact)

**Date**: 2026-06-16
**Task**: All 3 goal fronts advanced: GEMM perf win + forward-pass op family (all 3 structural classes byte-exact)
**Branch**: `main`

### Summary

Advanced all three fronts of the locked goal (coverage + high-perf + complete forward pass, trellis flow). PERF: INC-14 GEMM G1 — q4_0 weight-decode-reuse tile (decode each weight block ONCE, reuse across M activation columns) measures 1.27-1.33x over per-row vec_dot at K=4096, byte-exact vs Mx ggml_vec_dot. The research honestly falsified the gemv premise (decode is COMPUTE-bound, activation-reuse buys nothing; the win is GEMM weight-decode-amortization) and found ggml's gemm/repack path is DISABLED at VLEN=128 (case 128->nullptr) so ggml itself falls back to per-(row,col) vec_dot with redundant re-decode — exactly what G1 eliminates. Honest: ~1.10x blended / 1.3x tile, VLEN=128-capped (bigger needs VLEN>=256, no such board). FORWARD PASS (was 0%): opened the f32 non-dot op family + proved ALL THREE structural classes byte-exact on ssh rvv — F1 ggml_vec_scale_f32 (elementwise, 3849/3849), F3 ggml_rms_norm_f32 (the REDUCTION; matched ggml's scalar-DOUBLE sum-of-squares fold exactly — the f32-mul->f64-cast->f64-add chain is both the byte-exact key and an FMA barrier, fp-contract-invariant, 4805/4805), F5 ggml_vec_silu_f32 (the TRANSCENDENTAL; replicated ggml_v_expf_m2's degree-5 minimax polynomial node-for-node incl. the exact hex-float constants + the unconditional-vmerge resolution of ggml's vcpop short-circuit, 2641/2641 vs the vectorized silu; libm-expf NC discriminates). All STRUCTURED emitc (raw()=0), each a new emitc lowering. The compiler now demonstrably handles every structural kernel CLASS of llama.cpp: quant-dot (3 families) + GEMM + f32 elementwise/reduction/transcendental. Clean rebuilds green, 3 documented reds, all additive. The remaining work is breadth-EXTENSION of proven classes: F4 quantizers (f32->quant bridge), F5b soft_max (reuses F5 exp + F3 double-reduction), F6 rope; more dot kernels (q5_0/q5_1/q4_K/q5_K/q2_K/q3_K/iq4_nl); G2/G3 (full GEMM ABI + autotuner tunes M) — multi-session.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `d48533bb` | (see git log) |
| `70d35660` | (see git log) |
| `9e5ba826` | (see git log) |
| `edaeb886` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 15: Forward-pass PRIMITIVE SET complete (scale/rms_norm/silu/soft_max/quantize/rope, byte-exact) + GEMM perf win

**Date**: 2026-06-16
**Task**: Forward-pass PRIMITIVE SET complete (scale/rms_norm/silu/soft_max/quantize/rope, byte-exact) + GEMM perf win
**Branch**: `main`

### Summary

Completed the forward-pass primitive set — the compiler now expresses every structural class AND every forward-pass primitive of llama.cpp inference, all STRUCTURED byte-exact (raw()=0) on ssh rvv: F1 scale, F3 rms_norm (scalar-double reduction, fp-contract-invariant), F5 silu (ggml_v_expf_m2 minimax polynomial node-for-node), F5b soft_max (reuses F5 exp via a shared helper + ggml's exact f64 widening-reduce; byte-exact y[] AND returned sum incl. -inf masked rows), F4 quantize_row_q8_0 (the f32->quant BRIDGE; full block_q8_0 d+qs byte-exact + CLOSE-THE-LOOP: our quantize->our q4_0_q8_0 dot vs ggml quantize->ggml dot = 1920/1920 *s bit-exact across two composed compiler-emitted kernels), F6 rope (NORMAL variant = llama-2's; iterative theta + scalar-libm cos/sin via call_opaque + FMA-grouped rotation; 66/66 byte-exact under all 4 fp-contract modes; self-includes math.h for a standalone drop-in). Each is a new structured emitc lowering reusing the established discipline (the FMA-grouping/emitc.expression byte-exactness fix, the loop-carried-accumulator reduction, the shared exp helper). All additive (every sibling re-renders byte-identical), clean rebuilds green, 3 documented environmental reds. The compiler's demonstrated coverage now spans: quant-dot (q4_0/q8_0/q4_1/q6_K, 3 families, capability-divergent, measurement-tuned, q4_0 beats ggml), GEMM (q4_0 weight-reuse tile 1.27-1.33x, perf front), and the complete f32 forward-pass primitive set. Remaining = breadth-extension of proven patterns (trivial elementwise add/mul/mad, q8_1 quantizer, more dot kernels q5_0/q4_K/etc., full GEMM ABI G2/G3) + end-to-end forward-pass assembly.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `c0f06843` | (see git log) |
| `41ee3bb2` | (see git log) |
| `a053c41e` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 16: Coverage: 12 dot kernels (every structural class) + GEMM autotuner + complete forward pass — all 3 goal fronts

**Date**: 2026-06-17
**Task**: Coverage: 12 dot kernels (every structural class) + GEMM autotuner + complete forward pass — all 3 goal fronts
**Branch**: `main`

### Summary

Drove all three goal conditions (覆盖+高性能+完整forward, 都做) to substantial completion. COVERAGE: 12 ggml dot kernels, byte-exact + STRUCTURED (raw()=0) on ssh rvv, spanning EVERY structural class — block-quant linear (q4_0/q8_0/q4_1/q5_0/q5_1), K-quant super-block (q2_K/q3_K/q4_K/q5_K/q6_K), codebook (iq4_nl: DenseI8ArrayAttr + vrgather), FP4 (mxfp4: e2m1 codebook + the E8M0 2^(e-128) exact-bit scale) — each reusing the prior machinery (the bit-dances, the super-block fold, the codebook gather), plus the q4_0 GEMM (weight-reuse) and the f32->q8 quantizer bridge (close-the-loop proven). The K-quants reuse a shared super-block core (q5_K = q4_K + qh plane byte-identical; q3_K composes q2_K 2-bit + subtractive hmask + signed scale + q6_K no-min fold). PERF: the measurement-backed autotuner selects the optimal GEMM M (measured M6 ~1.19x at K=4096, overturning G2's sequential-timing estimate — the N3 实测胜出) and the dot-kernel shapes; q4_0 vec_dot beats ggml ~4.5%; honest VLEN=128-capped ceiling stated throughout. FORWARD PASS: complete primitive set (scale/rms_norm/silu/soft_max/quantize/rope, all byte-exact, the transcendentals matching ggml's minimax exp + libm cos/sin node-for-node). All additive (every sibling byte-identical), clean rebuilds green, 652 tests / 3 documented environmental reds. Remaining = the niche IQ ternary super-block tail (iq1/iq2/iq3 + iq4_xs) — a distinct uncommon sub-class, the asymptotic remainder.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `947ccbd3` | (see git log) |
| `d237d37e` | (see git log) |
| `6de78efd` | (see git log) |
| `6ea547d1` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 17: GOAL COMPLETE — 24/24 ggml dot kernels + GEMM autotuner + complete forward pass, all byte-exact + trellis-checked

**Date**: 2026-06-17
**Task**: GOAL COMPLETE — 24/24 ggml dot kernels + GEMM autotuner + complete forward pass, all byte-exact + trellis-checked
**Branch**: `main`

### Summary

All three goal conditions met (覆盖+高性能+完整forward) + trellis flow closed (implement via sub-agents -> holistic trellis-check -> update-spec -> finish). COVERAGE literal 100%: all 24 ggml_vec_dot_* kernels compiler-emitted + byte-exact on ssh rvv (INC-1..INC-43), every structural class -- block-quant linear (q4_0/q8_0/q4_1/q5_0/q5_1), K-quant super-block (q2_K/q3_K/q4_K/q5_K/q6_K, shared super-block core), codebook (iq4_nl/iq4_xs), FP4 (mxfp4 E8M0 / nvfp4 UE4M3), grid-codebook (iq2_xxs/xs/s, iq3_xxs/s), ternary (iq1_s/m, tq1_0 base-3, tq2_0 2-bit), binary (q1_0) -- each reusing prior machinery (bit-dances, super-block fold, codebook gather, ternary grid), 24 distinct recognizers 1:1 with the _generic set. HIGH PERFORMANCE: measurement-backed autotuner (tune-once->cache tuning record->compile reads it; static cost model = pruner + fallback) selects the GEMM M (measured M6 ~1.19x) + dot shapes; q4_0 vec_dot beats ggml ~4.5%; N1 capability divergence (rv64gcv elided / zve32x robust); honest VLEN=128-capped ceiling stated throughout. COMPLETE FORWARD PASS: scale/rms_norm(double-accum)/silu(minimax exp)/soft_max/quantize/rope(libm), byte-exact, close-the-loop (f32->quantize->matmul) proven. Holistic trellis-check PASS: raw()=0 across all (I5 structured, no string-machine), I7 fail-closed verifiers, additive (q4_0 byte-identical to its inc2a artifact across the entire ~40-commit sweep), clean rebuild green, 676 tests/673 pass/3 documented environmental reds. update-spec: the measured>static autotune authority contract. Plus live token-identical llama-2-7b integration (earlier). Honest scope: IQ kernels are coverage-rung 4-arg byte-exact kernels (not full-8-arg-ABI deployed); perf modest+VLEN-capped (bigger needs VLEN>=256, unavailable).

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `0fc67d67` | (see git log) |
| `4a28013d` | (see git log) |
| `a6f43069` | (see git log) |
| `d26fcb75` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 18: Maturity goal: scalar-hardening + tune-reuse interface + roofline/measured-win + core split (all 4 WS)

**Date**: 2026-06-18
**Task**: Maturity goal: scalar-hardening + tune-reuse interface + roofline/measured-win + core split (all 4 WS)
**Branch**: `main`

### Summary

Completed the 4-workstream compiler-maturity goal. WS-A: scalar variant now flows via registry collectVariantProposals by abstract ConservativeFallback role (front-door scalar strings 5->0, core family-grep empty). WS-C: TunableScheduleOpInterface (Tier-1) — one tcrv-rvv-materialize-schedule pass auto-discovers all tunable ops via dyn_cast (0 hardcoded op types); descriptor data provably identical -> byte-exact stamps; family-neutral. WS-B: ssh-rvv roofline showed kernels are latency/overhead-bound (4-13% of compute ceiling), so the lever is multi_block_factor overlap — and the measured win already exists (inc10: q4_1 static m1/4 loss->measured m1/1 win, factor 1>2>4 overturns the cost model), flowing through the new Tier-1 pass byte-identically. WS-D: 28-branch block-dot dispatch->first-match table; 33 support helpers extracted; VariantToEmitCFunc de-anonymized + split across 7 family TUs (RVVToEmitC.cpp 23939->5237). All byte-exact (clean-build fingerprint f810ce6b), raw()=0, lit 674/3. Deferred (documented): WS-D verifier de-dup (90% bespoke I7 diagnostics, pure churn). Build-hygiene lesson: this tree's incremental builds are unreliable (always-dirty RVVOps.cpp.inc) — fingerprint gates must use forced/clean rebuilds.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `ef7179e7` | (see git log) |
| `bd7c8b53` | (see git log) |
| `bac3acd5` | (see git log) |
| `a73ab62d` | (see git log) |
| `81b57ba5` | (see git log) |
| `ec3949d2` | (see git log) |
| `eb2e09d8` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 19: closeout: full-set byte-exact proof + roofline honesty

**Date**: 2026-06-18
**Task**: closeout: full-set byte-exact proof + roofline honesty
**Branch**: `main`

### Summary

Closed the WS-D byte-exact coverage gap (full test/Conversion/RVV set: pre-split 2918bb22 == post-split HEAD == cb04b219, so the ForwardElementwise+DeferredDequant emitters are byte-proven too, not just block-dot). Softened wsB-roofline.md: measurement is the N3 claim; roofline is context (factor=1 winning is in tension with 'more overlap').

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `f9dc05ac` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 20: N1/N2/N3 RVV maturity: compiler emits q4_0 GEMV (whole path) + ①②⑤ sealed

**Date**: 2026-06-22
**Task**: N1/N2/N3 RVV maturity: compiler emits q4_0 GEMV (whole path) + ①②⑤ sealed
**Branch**: `main`

### Summary

Item ① same-build A/B 5.98x (baseline pinned naive-RVV); ⑤ compiler emits repack GEMV (raw=0, numeric PASS NC≤336, decode ENGAGED tg128); ② budget dormant-on-RVV honest finding (EMUL cap is the live lever); Win-A ablation re-measured fresh on rvv (2-4x vs naive, adversarial-corrected: realization-selection not pure-LMUL-knob, not in llama yet).

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `868de602` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
