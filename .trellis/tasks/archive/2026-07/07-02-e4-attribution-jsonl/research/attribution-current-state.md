# Research: E4 — CURRENT attribution state (three-level per [D-4])

- **Query**: Map the current compile-time selection attribution + loading-time resolution, exact file:line, HEAD-verified.
- **Scope**: internal
- **Date**: 2026-07-02
- **Branch**: `refactor/full-refactor-m1`

All line numbers below are re-grepped at HEAD (prior audit anchors like `:326-358`
still hold; a couple drifted — noted inline).

---

## ① Compile-time selection attribution (richest, non-compliant shape)

### `addPreferenceMetadata` — `lib/Transforms/VariantSelection.cpp:326-358`

Attaches, onto an `mlir::OperationState` (i.e. baked into the op the pass
creates — `DispatchCaseOp`, `FallbackOp`, or the selected/​missing-fallback
`DiagnosticOp`), these attributes. Attr names are the `constexpr StringLiteral`s
at `VariantSelection.cpp:28-52`:

| Attr name (constant) | Literal | Type | Source | Condition |
|---|---|---|---|---|
| `kOriginAttrName` | `origin` | string | `cost.getOriginPlugin()` | always |
| `kPreferenceAvailableAttrName` | `preference_available` | bool | `cost.hasExplicitPreference()` | always |
| `kPreferenceScoreAttrName` | `preference_score` | f64 | `cost.getScore()` | always |
| `kPreferenceRankAttrName` | `preference_rank` | i64 | `rankValue` (index in `plan.rankedVariants`, else `originalIndex`) | always |
| `kPreferencePolicyAttrName` | `preference_policy` | string | `cost.getPolicy()` | if `cost.hasPolicy()` |
| `kPreferenceExplanationAttrName` | `preference_explanation` | string | `cost.getExplanation()` | if `cost.hasExplanation()` |
| `kPreferenceTieBreakAttrName` | `preference_tie_break` | string | `buildPreferenceTieBreakReason(plan, rankValue)` (`:292-324`) | always |
| `kFallbackRoleAttrName` | `fallback_role` | string (`plugin::kConservativeFallbackRoleValue`) | — | if `isConservativeFallbackCandidate(...)` |

`addPreferenceMetadata` is invoked from **three** materialization sites:
- `createDispatchCase` — `VariantSelection.cpp:379`
- `createFallback` — `VariantSelection.cpp:391`
- `materializeSelectedVariantMarker` — `VariantSelection.cpp:919`
- (also `materializeMissingFallbackCoverageDiagnostic` — `VariantSelection.cpp:530`)

### Selected-path marker — `materializeSelectedVariantMarker` `VariantSelection.cpp:906-919`

Builds a `DiagnosticOp` with:
- `reason` (`kReasonAttrName`) = **`"variant-selected"`** (`kSelectedReasonValue`, `:36`)
- `message` = `getSelectedMarkerMessage(plan.kind)` (`:409-420`)
- `severity` = `"note"`, `status` = `"selected"`
- `target` = `FlatSymbolRefAttr(selectedVariant)`
- `selection_kind` (`kSelectionKindAttrName`) = `stringifySelectedMarkerKind(kind)` (`:395-407`):
  `"static-variant"` | `"fallback-only"` | `"runtime-dispatch"` | `"no-viable-variant"`
- `+ addPreferenceMetadata`

### Missing-fallback marker — `materializeMissingFallbackCoverageDiagnostic` `VariantSelection.cpp:516-531`

Builds a separate `DiagnosticOp` with:
- `reason` = **`"fallback-coverage-missing"`** (`kMissingFallbackReasonValue`, `:37-38`)
- `message`, `severity` = `"warning"`, `status` = `"missing"`, `target`
- `selection_kind` = `"missing-conservative-fallback"` (`kMissingFallbackSelectionKindValue`, `:39-40`)
- `+ addPreferenceMetadata`

> **Key semantic:** `reason` here is a **diagnostic-status marker** ("this is the
> selected path" / "fallback coverage is missing"), NOT the [D-4] "why-chosen"
> axis `{only_feasible, prior, measured}`. The two are **orthogonal**. See
> `jsonl-export-design.md` for the derivation.

### RVV-path schedule extras (NOT part of ① — belong to the *scheduling* stage)

`selection_reason / candidate_count / legal_candidate_count / selected_cost /
measured_ns` are stamped by `stampRVVSchedule` in
`include/TianChenRV/Plugin/RVV/RVVScheduleMaterialization.h:195-216`
(prefixed `tcrv_rvv.<kernel>_schedule.*`), plus mirrors in
`RVVDequantDotSourceFrontDoor.cpp:771/779/838`,
`RVVEmitCRouteMetadata.cpp:121-129`, etc.
These attribute **why this LMUL/schedule** was picked — that is the [D-4]
**scheduling-stage** attribution, which the execution doc marks **M2 增量**
(`docs/TianChen-RV_执行总纲v2.md:202`). **Out of E4 scope** (E4 = selection
stage ① only). See `e4-scope-summary.md`.

---

## Where `candidates[]` and `keys_evaluated{}` live at materialize time

**Both are still in scope at the sink point — nothing is discarded.** This is the
task's main "does E4 balloon?" question; the answer is **no** on this axis.

- **`candidates[]` = `plan.rankedVariants`** — a live member of
  `VariantSelectionPlan` (`include/TianChenRV/Transforms/VariantSelection.h:43`).
  Each entry is a `VariantSelectionCase` (`VariantSelection.h:26-34`) carrying:
  `variant`, `cost` (`plugin::VariantCostEstimate`), `originalIndex`,
  `genericallyAvailable`, `conflictFree`, `hasGenericDecisionMetadata`,
  `requiresRuntimeCapabilityGuard`. The whole ranked set is alive after
  `planKernelVariantSelection` returns. Fully serializable **as-is**.

- **`keys_evaluated{}` inputs are both live**, but the *evaluated key list* is not
  currently retained:
  - The `TargetCapabilitySet` (`capabilities`) is passed into `runSelection`
    (`VariantSelection.cpp:563`) and to `planKernelVariantSelection`.
  - Each variant's `requires` `ArrayAttr` (capability symbol refs) is on the IR
    (read in `analyzeRequirementLegality`, `VariantSelection.cpp:159-207`).
  - `analyzeRequirementLegality` iterates `requires` × `TargetCapabilitySet` but
    collapses the result to booleans in `RequirementLegality`
    (`VariantSelection.cpp:67-71`: `available/conflictFree/requiresRuntimeCapabilityGuard`)
    — it does **not** keep *which keys* were looked up or their per-key verdict.
  - **Consequence:** `keys_evaluated{}` must be **re-derived at the sink** from
    `variant.requires` + `TargetCapabilitySet.lookupBySymbolName(...)`. All inputs
    are still in scope, so this is a small additive read-loop, **not** re-plumbing
    of the legality analysis. (Recommended: re-derive; do NOT extend
    `RequirementLegality` — keeps E4 purely additive.)

## VariantCostEstimate — accessors available for `candidates[]`

`include/TianChenRV/Plugin/ExtensionPlugin.h:338-390`:
`hasScore()/getScore()`, `hasExplicitPreference()`, `getOriginPlugin()`,
`getVariantSymbol()`, `hasExplanation()/getExplanation()`,
`hasPolicy()/getPolicy()`, `getFallbackRole()/hasFallbackRole()`.

## Ranking source (relevant to the `reason` derivation)

`ExtensionPluginRegistry::rankKernelVariantsByCost`
(`lib/Plugin/ExtensionPlugin.cpp:1362-1376`, decl `ExtensionPlugin.h:762-765`)
produces the ranked cost entries. There is **no capability-prior ranking layer**
and **no memoized-measurement consumption** at this (exec selection) layer — the
order is static-cost + `hasExplicitPreference`. Execution doc §5 item ⑦
(`docs/TianChen-RV_执行总纲v2.md:218`) states the selector is today
*"capability-blind 结构成本+烘焙测量, 先验层不存在"*. This is the crux of the
`prior` honesty escalation (see `jsonl-export-design.md` and `e4-scope-summary.md`).

---

## ② Loading-time resolution ([D-2a]) — CURRENT state

`lib/Transforms/DispatchRuntimeGuard.cpp` (whole file, HEAD-verified):
- `ensureDispatchAvailabilityGuardParam` (`:210-223`) materializes a **single**
  `dispatch_available` runtime ABI param via
  `support::getDispatchAvailabilityGuardParamSpec("dispatch_available")` →
  `RuntimeParamOp`. The caller provides this switch **once**; guarded
  `DispatchCaseOp`s get a `runtime_guard` symbol ref to it
  (`attachRuntimeGuardLink`, `:225-248`). Hot path reads the caller-provided
  switch — **zero per-dispatch capability checks** ([NG-3] respected).
- **Missing:** no `declared-instance-hash` (grep = 0 in `lib/`/`include/`), no
  per-process resolution record dropped anywhere. Confirmed:
  `docs/TianChen-RV_执行总纲v2.md:61`.

## ③ Runtime attribution — absent

No hwprobe / instance-hash keyed dispatch table (`docs/...执行总纲v2.md:179`).
This is [D-2b]/[D-3], **M2/M3**, not E4.

---

## declared_instance_hash producer — what exists / what must be added

- **"Explicit schema fact instances" available at compile time** =
  `TargetCapabilitySet::getCapabilities()` →
  `llvm::ArrayRef<CapabilityDescriptor>` (`CapabilityModel.h:97-99`). Each
  `CapabilityDescriptor` (`CapabilityModel.h:26-79`) exposes: `getSymbolName()`,
  `getID()`, `getKind()`, `getStatus()`, `getAvailability()`, `getProperties()`
  (`std::map<std::string,std::string>`), and relation lists
  `getProvidedIDs()/getImpliedIDs()/getConflictingIDs()`.
- `TargetCapabilitySet::buildFromKernel` (`CapabilityModel.h:90-93`) **already
  expands** profiles into individual descriptors, so hashing `getCapabilities()`
  hits the **expanded normalized fact set** required by canon 7d781994
  ("profile ≡ explicit list ⇒ same hash").
- **No C++-side canonical serializer or hash exists** — `declared_instance_hash`
  / `declaredInstanceHash` / `InstanceHash` grep = **0** in `lib/`+`include/`
  (verified). E4 **must add** one. LLVM `llvm/Support/SHA256.h` is in-tree.
  Detailed design in `d2a-resolution-record-design.md`.

## Caveats / Not found

- The only in-tree SHA256 usage is content/binary digests
  (`sourceSHA256`/`generatedArtifact*SHA256`), NOT a shape/instance hash — cannot
  be reused as-is, only the `hashlib.sha256` *idiom* (E1's canonical-JSON→SHA256)
  is a template.
- `only_feasible` / `prior` / `measured` string tokens: grep = 0 anywhere in code
  (only in docs/spec). Confirmed net-new.
