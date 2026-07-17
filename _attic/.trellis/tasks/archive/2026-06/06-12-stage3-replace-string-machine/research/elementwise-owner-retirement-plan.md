# Retire the elementwise string-plan owner — the first real 去除字符串 of Stage 3 换心

Author: Fable (senior MLIR architect). Date: 2026-06-13.
Task: `.trellis/tasks/06-12-stage3-replace-string-machine/`. Parent: 06-12-mlir-audit-refactor (ADR Stage 3).
Target of retirement: `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp` (736 lines).

This plan retires the **elementwise statement-plan owner** — the smallest live string-plan owner —
now that the export-materialize seam is already decoupled (commit `7454e64b`,
`TargetArtifactExport.cpp:2123-2155`). It is the beachhead deletion of the master 换心 plan
(`research/heart-replacement-plan.md`).

---

## 1. Verdict — **deletable-after-decoupling**

The owner is retirable, but ONLY after **two remaining live consumers** are decoupled first. The
naive "delete the file + table entry" already failed once (57 lit reds) because both consumers still
reach the owner through `describeRVVSelectedBodyEmitCRoute(request, &route)` and, with the entry
gone, fall into `diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner`
(`RVVEmitCStatementPlanOwners.cpp:2776`).

The decisive thesis facts that make this `deletable-after-decoupling` rather than `blocked`:

- **The owner produces ZERO diagnostic/header content.** Every PLAN field, every HEADER
  `tianchenrv.rvv.*` evidence comment, and every STALE-ELEM-* mirror negative is derived from the
  route **DESCRIPTION** (`analyzeRVVSelectedBodyRoute` + `getRVVSelectedBodyConfigArtifactMetadata`
  / `getRVVSelectedBodyConstructionMetadataFacts`), which is computed **independently of the route
  build** (`RVVEmitCRouteProvider.cpp:416-431`: the analysis runs unconditionally; the owner runs
  only when `verifiedRoute` is non-null). The owner only emits EmitC route steps that are either
  **discarded** (emission-plan path) or used as a **redundant consistency gate** (validation path).
- **The diagnostic is a pure I4 MIRROR.** Export re-derives every field from the IR and rejects any
  mismatch ("must mirror", `RVVTargetSupportBundle.cpp:108-288, 491-544`), so the in-IR diagnostic
  carries no downstream authority — it can be sourced from the description with zero loss.
- **The conversion already covers every elementwise family.** `convertRVVModuleToEmitC`
  (`RVVToEmitC.cpp:882`) + `populateRVVElementwiseToEmitCPatterns` fully legalizes add/sub/mul,
  scalar-broadcast, masked, strided — exactly the families this owner serves. The export seam
  already takes the converted path for all of them (`TargetArtifactExport.cpp:2108-2155`).
- **The deletion surface is clean and isolated** (Section 6): no sibling owner shares the elementwise
  file's symbols, the consumer predicate's operation-classifier helpers are used only by that
  predicate, and the statement-plan struct is owner-only.

Not `partially`: once both consumers are gated, the whole owner file and its scaffolding go in one
deletion. Not `blocked`: no bigger refactor is required — the decouple reuses the *exact* try-convert
pattern already proven at the export seam.

---

## 2. Retirement strategy (decouple-then-delete, hardware-gated)

Strangler-fig, in strict order. **Decouple the two live consumers, hardware/lit-verify, THEN delete.**

The two live owner consumers reached via a non-null `&route`:

| # | Consumer | Site | Reached by | Role of the route it builds |
|---|----------|------|-----------|------------------------------|
| C1 | Emission-plan pass (`--tcrv-materialize-emission-plans`) | `RVVExtensionPlugin::buildVariantEmissionPlan` `RVVExtensionPlugin.cpp:594` | `EmissionReadiness.cpp:180` → registry → plugin | **Discarded.** Only `routeDescription` populates the plan (lines 598-623); `route` (declared 589) is never read after 594. |
| C2 | Candidate validation (object **and** header export) | `validateRVVSelectedVariantRouteAgreesWithCandidate` `RVVTargetSupportBundle.cpp:411` | header: `TargetArtifactExport.cpp:1756`; object: `RVVTargetSupportBundle.cpp:577` via `config.candidateValidationFn` (wired 1745) | **Redundant consistency gate.** Used only by three route-consuming validators (428/432/434), all I4 mirrors (Section 4). |

A third historical seam — the export-materialize route build (`buildSelectedEmitCArtifactRoute`) — is
ALREADY decoupled: `TargetArtifactExport.cpp:2123-2155` try-converts first and `return`s the
converted module at 2152 before ever calling `buildSelectedEmitCArtifactRoute` (2161). Elementwise
always fully converts, so that seam never reaches the owner.

Decouple C1 and C2 → the elementwise owner is unreachable for every path → delete.

---

## 3. emission_plan_decoupling — how `--tcrv-materialize-emission-plans` stops needing the owner

**Primary mechanism: the same try-convert gate the export seam uses (N2-clean, zero family branch).**

In `RVVExtensionPlugin::buildVariantEmissionPlan` (`RVVExtensionPlugin.cpp:559-628`), before line 589:

1. Isolate + clone the selected typed body into a standalone `ModuleOp` (the unit
   `convertRVVModuleToEmitC` expects — same shape the export path feeds it).
2. Run `conversion::rvv::convertRVVModuleToEmitC(*clone)` under a `mlir::ScopedDiagnosticHandler`
   that swallows the speculative "failed to legalize" diagnostics — byte-for-byte the pattern at
   `TargetArtifactExport.cpp:2132-2138`.
3. If `fullyConverted` → call `describeRVVSelectedBodyEmitCRoute(routeRequest)` with **nullptr route**
   (`RVVEmitCRouteProvider.cpp:425` skips `buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis`, i.e.
   the owner dispatch). The description still returns from `analysis->description` (430).
4. If NOT converted → keep `describeRVVSelectedBodyEmitCRoute(routeRequest, &route)` so non-converted
   families' owners (compare/select, reduction, contraction, …) still build their route.

Because the route is **discarded** in this path, the converted branch loses nothing: the diagnostic is
built entirely from `routeDescription` (lines 598-623) and is **byte-identical**. This is the
"minimal converted-family diagnostic" — produced *without* the string statement-plan, even though the
emitted attributes are unchanged. The gate is "did the patterns legalize it," **not** a family-name
branch — the N2 seam.

**The I4-aligned simplification (preferred end-state, gate on full lit):** since (a) the route is
discarded here and (b) the selected boundary is already validated independently at
`RVVExtensionPlugin.cpp:585` (`validateSelectedRVVSelectedBodyBoundary`), the route-build consistency
gate during *emission-plan materialization* is plausibly redundant for **all** families. If, during
execution, passing nullptr **unconditionally** (no clone, no convert probe) keeps full lit green, prefer
it — it is less code and still zero-branch. Decide empirically: try-convert is the safe primary; the
unconditional-nullptr simplification ships only if full lit + ssh rvv confirm no family relied on the
emission-plan-time route build. Do NOT pre-commit to the simplification without that evidence.

What must NOT change: `routeDescription`, `getRVVSelectedBodyConstructionMetadataFacts`,
`getRVVSelectedBodyConfigArtifactMetadata`, and `analyzeRVVSelectedBodyRoute`
(`RVVEmitCRoutePlanning.cpp`, incl. `elementwiseArithmeticRouteFamilyPlan` at 35084). These live in
`RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` (CMake line 8) and the planner — **a DIFFERENT file
from the deletion target** and the actual source of the diagnostic metadata. Keep them.

---

## 4. header_decoupling — source the header from the conversion signature, not the string plan

The I8 HEADER export (`tcrv-translate --tcrv-export-target-header-artifact`) reaches the owner only
through `config.candidateValidationFn` →
`validateRVVSelectedBodyTargetArtifactCandidate` (`RVVTargetSupportBundle.cpp:547`) →
`validateRVVSelectedVariantRouteAgreesWithCandidate` (382) → `&route` (411).

**The header's load-bearing output needs no string plan.** The only functional output is the C
prototype `void <fn>(<runtime ABI params>);` (`TargetArtifactExport.cpp:1657-1722`, 1705-1716), fully
determined by:
- the exported **function name** — derived from kernel+variant, string-route-independent
  (`TargetArtifactExport.cpp:2099-2103`), arity-checked against the conversion `emitc.func`
  (`TargetArtifactExport.cpp:1893`); and
- the ordered `(c_name, c_type, role, ownership)` runtime-ABI tuples — read off the **source**
  `tcrv_rvv.runtime_abi_value` ops by the slice analysis (`RVVEmitCRoutePlanning.cpp:13540`), not the
  owner's text. Every `tianchenrv.rvv.*` line is an *optional + dynamic* I4 evidence comment
  (`RVVTargetSupportBundle.cpp:1490-1629`), never pinned.

**Decouple step:** in `validateRVVSelectedVariantRouteAgreesWithCandidate`, stop passing `&route` —
call `describeRVVSelectedBodyEmitCRoute(request)` (nullptr, owner-free; still returns the full
description). Then:

| Current route-consuming validator | What it actually checks | Decoupled replacement |
|-----------------------------------|--------------------------|------------------------|
| `route.getRouteID()` mismatch (`RVVTargetSupportBundle.cpp:121`) inside `validateRVVRouteMetadataMirrorsSelectedBody` (108) | route id vs candidate metadata | swap to `description.targetArtifactRouteID` — identical value, owner-free. **All other 175 lines of this validator already read `description.*`** (operation, comparePredicateKind, routeOperandBindingPlanID/Summary, providerSupportedMirror, capability mirrors). Keeps every STALE-ELEM-* firing. |
| `validateRVVRouteSourceProvenance(route)` (339) | route carries exactly one `tcrv_rvv.with_vl` scope provenance | assert directly off the selected `with_vl` boundary op (already resolved + validated at `RVVExtensionPlugin.cpp:585`); or drop as redundant with that boundary validation. |
| `validateRVVRouteABIMappings(candidate, route)` (359) | route ABI mappings == `candidate.runtimeABIParameters` | compare `candidate.runtimeABIParameters` against the source `tcrv_rvv.runtime_abi_value` ops in declared order — the same source `collectRuntimeABIParameters` already reads. A re-derivation of identical source data. |

`validateRVVConfigArtifactMetadataMirrorsSelectedBody` (491) is **already description-only** and is the
source of the STALE-ELEM-PROVIDER / PLAN / MEMORY / TYPE / HEADER / runtime_abi_order rejections — it is
untouched and keeps firing.

Net: the prototype text is **byte-identical**, every STALE-ELEM-* negative still rejects, and the
candidateValidationFn no longer builds the string route for **any** family (an unconditional decouple —
the description carries all the metadata for every family; only the two route-only checks redirect to
IR/source-ops). This is also the cleanest answer to "source the header from the conversion `emitc.func`
signature": name + arity from the converted `emitc.func`, semantic tuples from the source ABI ops.

---

## 5. fixture_migration — what stays, what's added (≈55-61 elementwise fixtures)

**Scope (consumer predicate `RVVEmitCStatementPlanOwners.cpp:195-210`):** plain elementwise add/sub/mul
(incl. broadcast-load, i64, lmul-m2), scalar-broadcast add/sub/mul, masked add/sub/mul (incl. i64,
lmul-m2), and strided-add. **EXCLUDE** (different owners/files, out of scope): cmp-select, widening
conversion, dequantize, reduction/standalone-reduce, MAcc, segment2, indexed gather/scatter,
base/computed-mask memory movement, runtime-scalar-splat-store.

**Key finding that simplifies migration:** because the decouple keeps the **description** path intact,
the PLAN diagnostic, the HEADER evidence comments, the HEADER C prototype, and the STALE-ELEM-* mirror
negatives are all **byte-identical** after retirement. So **no golden-mirror assertion is deleted or
rewritten.** (Grep confirms there are **no** `diagnoseMissing` / "owner required" / STALE-MISSING
negative fixtures to rebuild — the m2 "~9 STALE MISSING" concern does not correspond to any existing
fixture.)

Migration table (per elementwise fixture, e.g. `pre-realized-selected-body-artifact-add.mlir`):

| Assertion block | Lines (add fixture) | Source | Action |
|-----------------|---------------------|--------|--------|
| `REALIZED` structural (typed body) | 36-47 | realization pass, not owner | **STAY** unchanged |
| `PLAN` emission_plan diagnostic (golden mirror) | 49-69 | route **description** | **STAY** byte-identical (description-derived) |
| `HEADER` `tianchenrv.rvv.*` evidence comments | 71-82 | description, optional+dynamic | **STAY** byte-identical |
| `HEADER` `void <fn>(...);` C prototype | 83 | name-derivation + source ABI ops | **STAY** byte-identical (now sourced per Section 4) |
| `STALE-ELEM-*` mirror rejections | 85-118 | description mirror checks | **STAY** — still fire (description-derived) |
| **NEW** converted-path EMITC structural RUN | — | `convertRVVModuleToEmitC` patterns | **ADD** — the behavioral backstop |

The single migration *change* is **additive**: add one converted-path RUN line per op-kind asserting the
**real** structured emitc (`emitc.for`, `emitc.call_opaque "__riscv_vadd_vv_i32m1"`,
`emitc.add`/`emitc.sub` addressing) — the dual-RUN template already used by 51/54 EmitC fixtures
(e.g. `rvv-generic-stage2-widening-product-unsigned-u8.mlir:2`). This replaces *trusting the string
owner's plan* with *proving the converted IR* — the I4-honest move: the golden mirror stays as a mirror,
the converted IR becomes the authority. The git working tree already shows several
`rvv-first-slice-materialization-*.mlir` fixtures mid-migration to this template.

Rationale for keeping (not deleting) the golden mirrors: they are load-bearing for the **I8 header
export** (the header export *requires* the emission-plan diagnostic to exist and mirror — Section 4),
and they remain valid I4 mirrors of the description. Deleting them would remove header-export coverage,
not string-owner coverage.

---

## 6. Deletion surface (clean, isolated — verified no sibling sharing)

| # | What | Location | Lines |
|---|------|----------|------:|
| 1 | Owner source file (whole) | `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp` | 736 |
| 2 | CMake source entry | `lib/Plugin/RVV/EmitC/CMakeLists.txt:12` | 1 |
| 3 | Dispatch-table entry | `RVVEmitCStatementPlanOwners.cpp:412-415` | 4 |
| 4 | Consumer predicate `isRVVSelectedBodyElementwiseArithmeticStatementPlanConsumer` | `RVVEmitCStatementPlanOwners.cpp:195-210` | 16 |
| 5 | Operation classifiers (only used by #4): `isRVVSelectedBodyPlainElementwiseStatementPlanOperation` / `…Masked…` / `…ScalarBroadcast…` | `RVVEmitCStatementPlanOwners.cpp:65-85` | ~21 |
| 6 | Public header decls (predicate, struct getter, migrated-route builder) | `include/.../RVVEmitCStatementPlanOwners.h:72, 97-98, 230` | ~8 |
| 7 | Statement-plan struct (owner-only) `RVVSelectedBodyElementwiseArithmeticRouteStatementPlan` | `include/.../RVVEmitCRoutePlanning.h:1337-1352` | ~16 |

Verified isolation (grep, excluding the owner file):
- Operation classifiers #5 are referenced **only** by predicate #4 (same file).
- Struct #7 is referenced **only** by the owner's public API (#6) and the owner file.
- No sibling `*PlanOwners.cpp` (compare/select, widening, reduction, contraction, …) imports any of
  these symbols.

**Do NOT touch** `RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` (CMake line 8) — it produces the
analysis family-plan/metadata the diagnostic still needs. Do NOT touch
`diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner` (2776) — it must keep firing for any genuinely
ownerless non-converted family.

---

## 7. Ordered steps

1. **Pin baseline.** Record `build/bin` green, full `lit` baseline (note the ≤3 environmental reds),
   and the converted-path equivalence: confirm `convertRVVModuleToEmitC` fully legalizes all 7 op-kinds
   (add, sub, mul, scalar-broadcast add/sub/mul). Capture current PLAN/HEADER golden for the
   elementwise fixtures (oracle for byte-identity).
2. **Decouple C1 (emission-plan).** Add the try-convert gate in
   `RVVExtensionPlugin::buildVariantEmissionPlan` (594): converted → nullptr route; else → `&route`.
   Build green. Full lit: the ~61 elementwise PLAN/HEADER/STALE fixtures must be **byte-identical
   green**; non-elementwise unaffected.
3. **Decouple C2 (header + object candidate validation).** In
   `validateRVVSelectedVariantRouteAgreesWithCandidate` (382) drop `&route`; redirect the three
   route-consuming validators per Section 4 (routeID→description; provenance→`with_vl` op; ABI→source
   ABI ops). Build green. Full lit: STALE-ELEM-* still reject; header prototype byte-identical.
4. **Migrate fixtures (additive).** Add the converted-path EMITC structural RUN line to each in-scope
   elementwise fixture (dual-RUN template). Keep all golden-mirror + STALE-ELEM-* assertions. Exclude
   cmp-select and all non-elementwise.
5. **Grep-gate (prove dead).** Confirm zero remaining `&route` callers of
   `describeRVVSelectedBodyEmitCRoute` for elementwise; confirm
   `buildRVVSelectedBodyElementwiseArithmeticMigratedRouteStatementPlan` and the classifiers/struct have
   no references outside the deletion surface (Section 6). The owner is now unreachable for every path.
6. **Delete the owner.** Remove items #1-#7 of Section 6 in one commit. Keep
   `RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` and `diagnoseMissing…`.
7. **Build green + full lit honest-green** (≤3 environmental reds). Any "must mirror" / diagnoseMissing
   regression = a decouple bug, not a fixture rewrite — fix the decouple, do not weaken the test.
8. **ssh rvv re-validate all 7 op-kinds** (Section 8). Only after a real `ssh_evidence=true` lamp for
   each op-kind is the deletion considered landed.

Sequencing is load-bearing: steps 2-3 (decouple) MUST land and be lit/hardware-verified BEFORE step 6
(delete). Deleting first reproduces the earlier 57-red failure.

---

## 8. ssh rvv re-validation gate (I8) — all 7 op-kinds

`ssh rvv` is live (riscv64, `/usr/bin/clang`). For EACH op-kind in
`{add, sub, mul, scalar-broadcast-add, scalar-broadcast-sub, scalar-broadcast-mul}` (and the masked /
strided rungs if their patterns are in the converted set), run the existing bundle-ABI e2e harness
WITHOUT `--dry-run`, reusing `research/elementwise-postdeletion-hardware-lamps`:

```
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/elem-retire.artifacts --run-id <kind>-hw --overwrite --ssh-target rvv \
  --op-kind <kind> \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Pass condition per kind: remote `uname -m = riscv64`, clang compile/link OK, `PASS op=<kind>` +
`tcrv_rvv_generated_bundle_abi_<kind>_ok`, and `evidence.json` with `status=success`,
`ssh_evidence=true`. Expect the baseline 3 environmental reds, **zero new**. A dry-run is NOT an I8
lamp — never assert hardware correctness from `--dry-run`. The conversion keeps feeding the same
`--tcrv-rvv-emitc-to-cpp` / `--tcrv-export-target-artifact-bundle` exports, so the harness validates the
rewritten path with zero harness changes.

---

## 9. lines_removable

~**800 lines gross** removed: 736 (owner file) + ~65 (table entry 4, consumer predicate 16, operation
classifiers ~21, header decls ~8, struct ~16) + 1 (CMake). The decouple ADDS ~30-50 lines (try-convert
gate in C1, redirected validators in C2) and the fixtures gain one RUN line each. **Net ≈ 750 lines
removed.** Headline: the 736-line owner file plus its dispatch/predicate/struct scaffolding.

---

## 10. Risks

- **C1 module isolation.** `convertRVVModuleToEmitC` wants a standalone `ModuleOp`; the emission-plan
  pass runs per-variant inside the kernel module. Cloning/isolating the selected body into a unit module
  inside `buildVariantEmissionPlan` is the main implementation cost. Mitigation: reuse the export path's
  isolation helper (the same one feeding `convertRVVModuleToEmitC` at the export seam); or, if full lit
  confirms the discarded route is redundant, take the unconditional-nullptr simplification (no clone).
- **Byte-identity drift.** Any spacing/ordering change in the converted-path metadata would break the
  ~61 golden fixtures. Mitigation: the decouple keeps the *description* path that produces the metadata —
  no metadata builder changes; diff PLAN/HEADER against the step-1 oracle and treat any delta as a
  decouple bug.
- **Over-broad C2 decouple.** Dropping `&route` in `validateRVVSelectedVariantRouteAgreesWithCandidate`
  affects ALL families, not just elementwise. Mitigation: the replacement validators read description +
  source ABI ops, which exist for every family; full lit across all artifact fixtures (not just
  elementwise) is the gate.
- **Hidden owner side-effect.** If the owner's route build catches a malformed-body error the analysis
  + boundary validation do not, removing it for elementwise could mask a real defect. Mitigation: the
  converted path IS the new authority and is ssh-rvv-validated; a body the patterns cannot legalize
  falls back (not converted) and keeps the owner path until its own retirement.
- **ssh rvv flakiness.** ProxyJump can be down. Mitigation: gate deletion on a real
  `ssh_evidence=true` per op-kind; never on dry-run.
- **Scope creep into siblings.** cmp-select is the very next table entry (416-419) and tempting.
  Mitigation: strict scope — this task deletes ONE owner; siblings are separate retirements.

## 11. Open questions

- Is the emission-plan-time route build (C1) truly redundant given the separate boundary validation at
  `RVVExtensionPlugin.cpp:585`, such that unconditional-nullptr keeps full lit green? (Decides try-convert
  vs the simpler no-clone path. Resolve empirically in step 2.)
- What is the minimal, reusable body-isolation helper to feed `convertRVVModuleToEmitC` from inside the
  plugin (vs the export path that already holds an isolated module)?
- Are the masked and strided-add rungs in the *current* `populateRVVElementwiseToEmitCPatterns`
  converted set, or only plain + scalar-broadcast? (Decides whether masked/strided fixtures migrate now
  or stay on the owner until their patterns land — confirm before step 4 so the grep-gate in step 5 is
  honest.)
- After C2 drops `&route`, can `validateRVVRouteSourceProvenance` be deleted outright (fully redundant
  with the boundary validation) or must it be reimplemented against the `with_vl` op?
