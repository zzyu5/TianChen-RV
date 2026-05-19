# TianChen-RV Trellis Spec Audit And Agent Update Prompt

## Audit Baseline

Use the 2026-05-18/05-19 RVV grill consensus as the source of truth:

```text
TianChen-RV RVV-first is not an i32m1 add/sub/mul demo,
not a Linalg frontend phase,
not an intrinsic-wrapper dialect,
and not an EmitC scheduling trick.

The durable path is:

tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv vector-level body
  -> RVV plugin-owned legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV intrinsic C/C++ or equivalent backend representation
  -> target artifact
  -> runtime / ssh-rvv evidence when correctness/runtime/performance is claimed
```

Authority placement:

```text
tcrv.exec:
  owns execution envelope, ABI/runtime role declarations, capability scope,
  variant organization, dispatch/fallback, diagnostics.
  It does not own compute, dtype authority, RVV schedule, or intrinsic spelling.

tcrv_rvv body:
  owns typed vector-level RVV compute/config/control/dataflow.
  Dtype enters through typed values/config, not route ids, C ABI strings,
  parameter names, artifact names, or old i32m1 helper names.

RVV plugin:
  owns legality, selected-body realization, route support, intrinsic mapping,
  C/RVV type mapping, ABI mapping, fail-closed diagnostics.

Common EmitC/export:
  owns neutral materialization and packaging only.
  It must not invent RVV compute, dtype, schedule, or intrinsic choices.
```

Stage order:

```text
Stage 1:
  RVV route-authority reset.
  Replace or fail-close i32m1-as-architecture-authority.
  End condition: no active compiler path treats bounded i32m1 arithmetic,
  RVVI32M1 route specs/slices, finite i32_* ops, route ids, artifact names,
  source-front-door patterns, descriptors, or exact __riscv_*_i32m1 spellings
  as RVV route authority.

Stage 2:
  Route-supported RVV coverage expansion on the corrected typed tcrv_rvv surface.
  Coverage calibrated by Linalg-like structured computation classes, not by
  current Linalg frontend work.
  Includes RVV plugin-local selected-body realization for performance.

Stage 3:
  Extension-family generalization / second-family integration after RVV maturity.
  Prove a real non-RVV family uses the same common TCRV path without leaking
  family-specific branches into core/common.
```

Performance layer:

```text
selected pre-realized tcrv_rvv body
  + target capability
  + runtime SSA / ABI values
  + hints / policy / profile
    -> RVV plugin-local one-linear selected-body realization
    -> realized tcrv_rvv selected body
    -> faithful EmitC / intrinsic lowering
```

No readiness state machine, no dashboard, no artifact ledger, no status metadata as authority. Hints/config must be consumed into real body structure if they affect generated code.

---

## Main Audit Conclusion

The spec tree is partially updated but internally mixed. Several files already contain the correct 5/18 grill direction, but other sections still preserve older Trellis assumptions:

1. Legacy `i32m1` route-table support is sometimes described as Stage 1 debt, but elsewhere still appears as a supported materialized EmitC/object route.
2. Emission-plan diagnostics and target artifact metadata are sometimes treated as route authority/status objects instead of non-authoritative mirrors.
3. Source-front-door / source-artifact-bundle flows are sometimes specified as current positive workflows, which conflicts with RVV-first Stage 1/2.
4. Toy / TensorExtLite / Template construction examples are too authoritative and can pull Codex into Stage3-like work before RVV maturity.
5. `tcrv.exec` and core/common layers sometimes still own “selected route,” “route selection,” or artifact route identities, which should be plugin-owned route provider output.
6. Several docs still use readiness/status/state language that should be recast as diagnostics/result codes only, not IR or architecture state.

---

## File-by-file Audit

### `index.md`

Severity: P1.

Problems:

- It lacks a root-level RVV-first source-of-truth section.
- “Current main lowering route is extension family ops -> EmitC ops -> intrinsic/vendor builtin/runtime C/C++” is too compressed. It omits `tcrv.exec` selected variant, typed `tcrv_rvv` body, RVV plugin selected-body realization, and `TCRVEmitCLowerableRoute`.
- It does not clearly say Stage1/Stage2 RVV work comes before high-level frontend / IME / Offload / Stage3 work.

Required update:

- Add a global “RVV-first current authority” section with the full chain.
- Add stage gates.
- State that source-front-door/high-level frontend work is future unless explicitly selected after RVV maturity.

### `architecture/index.md`

Severity: P1.

Problems:

- Navigation/checklist does not expose the updated stage gates or typed-body authority.

Required update:

- Add checklist items for: dtype in typed body, parameters explicit-bound in body, selected-body realization, no legacy i32 route, no status/artifact authority.

### `architecture/design-boundaries.md`

Severity: P1.

Problems:

- Good non-goals, but missing the most important current non-goals: no Stage2 before Stage1 reset; no high-level Linalg frontend as current authority; no source-front-door artifact path; no status/dashboard/report as progress.

Required update:

- Add explicit current-stage non-goals:
  - no retained executable i32m1 compatibility route;
  - no per-Linalg-op frontend current work;
  - no source-artifact front-door positive RVV path;
  - no emission-plan/readiness/dashboard/status authority.

### `architecture/system-positioning.md`

Severity: P1.

Problems:

- Long-term dataflow starts from high-level MLIR. That is fine as future positioning, but it can be misread as current Stage1/Stage2 work.
- Tuning examples say RVV LMUL/SEW/VL policy/unroll/thread partition, but do not say the output must be realized `tcrv_rvv` body, not metadata.

Required update:

- Add “current RVV-first path” separate from long-term frontend path.
- State that Linalg/Vector calibrates RVV coverage, but does not become current source authority.
- State tuning output is realized body structure.

### `architecture/unified-riscv-mlir.md`

Severity: P1/P0.

Problems:

- Core responsibilities include `route selection`. That is dangerous; core selects variants/dispatch, while route construction is plugin-owned.
- It says the shared system uses “manifests”; manifests must not become source/route authority.
- Example RVV op names look like `tcrv.rvv_add`; examples should not encourage dtype/intrinsic wrapper style.
- Current route omits selected typed body, realization, and provider-built route.

Required update:

- Replace core-owned `route selection` with `variant selection / dispatch orchestration`.
- Clarify plugin owns route provider and extension-specific route mapping.
- Reword examples as vector-level ops, for example `tcrv_rvv.load`, `tcrv_rvv.binary {kind = ...}`, `tcrv_rvv.store`, `tcrv_rvv.setvl`, `tcrv_rvv.mask`, not `i32_*` or intrinsic wrappers.
- Clarify manifests are optional scaffolding/provenance, never route authority.

### `capability-model/index.md`

Severity: P1.

Problems:

- Checklist does not guard against capability facts becoming route config or dtype authority.

Required update:

- Add checklist: capability facts constrain legality/realization; executable RVV dtype/config must be in typed body or consumed into realized body before route construction.

### `capability-model/capability-contract.md`

Severity: P1.

Problems:

- The parameter layering rule is mostly correct, but “compile-time variant config belongs in plugin-proposed variant metadata, selected config, tuning, or lowering-boundary metadata” is still too metadata-friendly.
- `CapabilityDescriptor` terminology can be confused with deleted descriptor-driven compute.
- Microarchitecture facts such as preferred LMUL / dtype throughput can become route config if not explicitly bounded.

Required update:

- State that metadata/hints may propose config, but route-supported RVV config must be structural in `tcrv_rvv` body or consumed into the realized body before route construction.
- Add note: `CapabilityDescriptor` is only a capability query container, not a compute descriptor and not route authority.
- Hardware facts and preferred LMUL are constraints/hints consumed by RVV plugin realization, not direct route choices.

### `capability-model/profiles.md`

Severity: P2.

Problems:

- Mostly aligned.
- Could use stronger stage language: RVV probe/profile facts are evidence and capability inputs, not Stage2 coverage units or route authority.

Required update:

- Add one sentence: profile facts never create `tcrv_rvv` bodies, dtype, route ids, or intrinsic choices.

### `core-dialect/index.md`

Severity: P1.

Problems:

- Checklist should include ABI role vs body binding separation.

Required update:

- Add: `mem_window`/`runtime_param` declare roles; selected typed body imports/consumes ABI values explicitly; C type strings are export spelling only.

### `core-dialect/tcrv-exec-contract.md`

Severity: P0/P1.

Problems:

- `tcrv.exec` ownership list includes “selected route.” This contradicts the route-provider model.
- `mem_window` and `runtime_param` examples include C types, but do not strongly say C ABI strings are not dtype authority.
- Emission-plan diagnostics require `status` and supported fields, encouraging a status/readiness/metadata authority layer.
- Verifier rules say core checks offload/IME/RVV capability directly, which can imply hard-coded family semantics.
- Relation to high-level MLIR allows “selected-path metadata” among current inputs, too broad.

Required update:

- Remove “selected route”; replace with “selected variant/path reference” or “selected-path diagnostic mirror.”
- Add ABI rule: `c_type`, `c_name`, role strings, and artifact metadata do not define dtype or compute.
- Recast emission-plan diagnostics as optional/non-authoritative mirrors; they may exist only after plugin route/realized body decisions, and must not be required as route authority.
- Core verifier should check generic structure and delegate extension legality to plugin hooks; no hard-coded RVV/IME/offload checks.
- “selected-path metadata” may be diagnostic/control metadata only, not current route/compute input.

### `extension-plugins/index.md`

Severity: P1.

Problems:

- Needs stage gate language: RVV first, scalar fallback no active executable, IME/offload/future later.

Required update:

- Add Stage1/Stage2/Stage3 rules.

### `extension-plugins/rvv-plugin.md`

Severity: P0.

Good:

- Already contains many correct rules: route-authority replacement, dtype/body authority, Stage2 coverage, selected-body realization.

Problems:

- The deprecated `i32m1` inventory is too detailed and sometimes acts like a current route spec.
- “Base: already realized explicit selected-body add/sub/mul fixtures remain valid” and “Existing fully realized explicit selected-body add/sub/mul target fixtures must continue to pass” can preserve legacy executable positive routes.
- The legacy route section still lists provider builders, route ids, artifact kinds, and target route APIs as if still supported.
- `with_vl` selected-boundary attrs carry `status`, route mapping, and conformance fields that risk becoming body/route authority.
- “bounded legacy i32m1 route-table path may materialize an MLIR EmitC module” conflicts with Stage1 if it remains a positive artifact path.

Required update:

- Keep only a short “legacy inventory to delete/fail-close” appendix.
- Remove or rewrite all positive support for legacy i32m1 object/header/bundle routes.
- Replace positive fixture language with: “legacy explicit bodies may remain parse/verify/fail-closed fixtures only, unless rewritten as ordinary instances of corrected generic typed vector surface.”
- Make `with_vl` boundary attrs legacy diagnostic mirrors only; mature RVV route must be recognized from typed body structure and plugin legality, not `status`/route mapping attrs.
- Ensure Stage1 desired example is generic typed structure:

```text
!tcrv_rvv.vector<elem = i32, lmul = m1>
tcrv_rvv.load
tcrv_rvv.binary {kind = add}
tcrv_rvv.store
route derives intrinsic from op kind + element type + SEW + LMUL + policy + operand/memory form
```

### `extension-plugins/scalar-fallback-plugin.md`

Severity: P1.

Problems:

- Uses “readiness status: unsupported.”
- Could be interpreted as current Stage3 proof if not explicitly gated.

Required update:

- Replace readiness/status language with unsupported diagnostic/result language.
- State scalar fallback has no active executable scalar body unless later rebuilt; it must not pull Stage3 forward before RVV maturity.

### `extension-plugins/ime-plugin.md`

Severity: P1.

Problems:

- “Same high-level op gains IME execution variant” is future only, but reads like expected current behavior.
- Variant generation scope lists matmul/attention/MLP; can steer agent into high-level frontend before RVV maturity.

Required update:

- Gate all IME executable integration to Stage3/later after RVV maturity and hardware/toolchain evidence.
- Say listed op classes are future evidence scenarios, not current source-front-door work.

### `extension-plugins/offload-runtime-plugin.md`

Severity: P1.

Problems:

- First-slice boundary is mostly fail-closed, but op examples and “can this high-level op be offloaded?” can be misread as current source/offload implementation work.

Required update:

- Gate Offload executable integration to Stage3/later after RVV maturity unless explicitly selected.
- State current offload boundary op is stale/no-active-route validation only; no artifact/export route.

### `extension-plugins/future-plugins.md`

Severity: P1.

Problems:

- “Supported high-level op classes” should be evidence/coverage planning only, not current frontend or route authority.

Required update:

- Add Stage3/later gate and source-authority rule.

### `guides/index.md`

Severity: P1.

Problems:

- Missing a guide/checklist for RVV route-authority and realization.

Required update:

- Add a quick route for RVV changes: read RVV plugin spec, variant pipeline, EmitC route, testing contract.

### `guides/capability-first-design-guide.md`

Severity: P1.

Problems:

- Does not ask whether capability facts are being misused as RVV dtype/config/route authority.

Required update:

- Add checklist: hardware facts constrain plugin realization; dtype/config/operation kind must be in typed body.

### `guides/compute-boundary-review-guide.md`

Severity: P1.

Problems:

- Correct transform shape says high-level MLIR -> plugin registry proposes variants -> exec envelope; this should be marked future, not current Stage1/2 requirement.
- Does not check dtype/parameter/performance authority.

Required update:

- Add current path from hand-authored TCRV MLIR to selected typed `tcrv_rvv` body.
- Add “body imports ABI values explicitly” and “performance config consumed into realized body.”

### `guides/plugin-locality-review-guide.md`

Severity: P1.

Problems:

- Does not mention selected-body realization and common orchestration vs plugin-owned semantics.

Required update:

- Add locality checks for `realizeSelectedVariantBody`, route provider ownership, and no common RVV scheduling/intrinsic mapping.

### `implementation-stack/index.md`

Severity: P1.

Problems:

- Needs stage-gate reminder for the implementation stack.

Required update:

- Add RVV-first progression and no Python compiler-core reminder.

### `implementation-stack/compiler-stack-contract.md`

Severity: P2.

Problems:

- Mostly aligned.

Required update:

- Add explicit no-Python for selected-body realization, plugin route authority, capability mapping, and typed body generation except tooling/probes.

### `implementation-stack/supervision-loop.md`

Severity: P1.

Good:

- Already contains strong Stage1/Stage2 rules and anti-dashboard/test-only guidance.

Problems:

- Mature path says “explicit extension-family ops -> selected-body realization -> materialized common EmitC module,” which omits `tcrv.exec envelope`, selected variant, typed `tcrv_rvv` body, and provider-built `TCRVEmitCLowerableRoute`.
- It should explicitly require Hermes to reject positive source-front-door RVV artifact work and positive legacy i32m1 object/header/bundle support.
- It should mention config/hints consumed into body, not status.

Required update:

- Replace mature path with the full RVV-first chain.
- Add Hermes redirect rules for emission-plan/status/artifact authority and source-front-door default-positive drift.
- Keep “no test/dashboard/report-only progress.”

### `lowering-runtime/index.md`

Severity: P1.

Problems:

- Needs clear one-line invariant: emission plan/artifact metadata is non-authoritative.

Required update:

- Add route authority chain and no status/artifact route authority.

### `lowering-runtime/emission-runtime-contract.md`

Severity: P0.

Good:

- Correctly deletes descriptor-driven C string export and defines selected-body realization boundary.

Problems:

- The “explicit typed RVV i32m1 legacy route-table case” is still allowed to return a supported materialized EmitC object plan with `rvv-i32m1-arithmetic-emitc-route-family` and `riscv-elf-relocatable-object`. This is a direct contradiction with Stage1 route-authority reset.
- Emission plan is treated as a central decision/status object. It risks becoming a readiness/status layer.
- Target artifact export consumes selected-path/lowering-boundary/emission-plan metadata as front-door authority.

Required update:

- Remove supported legacy i32m1 plan. All legacy i32m1 route-table cases must be unsupported/fail-closed or rewritten onto corrected typed surface before support.
- Recast emission plan as optional diagnostic/handoff mirror. Provider-built route + realized body are authority.
- Target artifact export must consume materialized EmitC route produced from `TCRVEmitCLowerableRoute`, with metadata only as exact mirrors.

### `lowering-runtime/emitc-route.md`

Severity: P0.

Good:

- Correctly says EmitC cannot invent compute/schedule.

Problems:

- Common template lists “C type mapping helper” and “intrinsic name resolver” as common. For RVV those are plugin-owned decisions; common may hold precomputed strings/payloads only.
- Selected emission-plan to target artifact handoff requires emission-plan diagnostics/status too strongly.
- “Common Source-Artifact Bundle Front Door” is specified as current positive workflow.
- There is a direct contradiction: it says bounded RVV vector-source front door should fail closed, but tests require positive RVV vector-source source-artifact-bundle coverage.
- Good cases mention RVV source-front-door selected variants add/sub/mul as if positive.

Required update:

- Make common helpers neutral; plugin resolves RVV C type/intrinsic/header choices.
- Demote emission-plan diagnostics to optional mirrors; artifacts require provider-built route/materialized EmitC.
- Move source-artifact front-door to future/Stage3 or explicit-only disabled-by-default; no positive RVV source-front-door artifact coverage in Stage1.
- Remove/rewrite positive RVV add/sub/mul source-front-door artifact cases unless they are corrected generic typed body routes after Stage1 reset.

### `plugin-protocol/index.md`

Severity: P1/P0.

Problems:

- “Extension Manifest is the machine-readable entry point” overstates manifest authority.
- Current plugin work can start from selected-path metadata; too broad.

Required update:

- Manifest may be scaffolding/provenance only, not source/route authority.
- Current valid inputs require explicit typed extension-family body or selected boundary that plugin can legally consume; metadata alone is diagnostic/control only.

### `plugin-protocol/interfaces-and-registry.md`

Severity: P0/P1.

Good:

- Selected-body realization provider is close to desired.

Problems:

- `DefaultArtifactFrontDoorPolicy::Eligible` is described as safe default. This is unsafe for RVV Stage1/2.
- Toy/TensorExtLite source front-door scenarios are positive and detailed, which can derail RVV-first.
- `VariantSelectedBodyRealizationResult::isRealized` can be mistaken for an IR state marker.
- Emission-plan slice validates/supports status fields too strongly.
- `TCRVComputeOpInterface` may be read as common compute semantic interpretation.

Required update:

- Make `ExplicitOnly` or disabled default the safe default for source front doors. Eligible must be opt-in after mature typed body route exists.
- Mark Toy/TensorExtLite source front-door scenarios as future/Stage3 examples only.
- Clarify realization result booleans are transient C++ result codes, not persisted IR readiness state.
- Recast emission-plan diagnostics as optional mirrors, not route authority.
- Clarify common compute interface exposes generic role/provenance only; common cannot interpret computation semantics.

### `plugin-protocol/locality-contract.md`

Severity: P1.

Problems:

- Plugin-owned “emission plans” are listed but selected-body realization and route provider authority should be clearer.

Required update:

- Add plugin owns selected-body realization and route provider; core can only orchestrate.
- State emission-plan metadata is not executable route authority.

### `plugin-protocol/extension-plugin-integration.md`

Severity: P1/P0.

Problems:

- Durable construction sequence relies on semantic role graph and Extension Manifest too much.
- Lowering template says plugin-owned config attrs/runtime ABI metadata -> emission plan route -> common lower-to-EmitC. It omits selected-body realization and typed body authority.
- Core may know generic route ids/artifact kind; needs “mirror only” limitation.

Required update:

- Recast manifest/semantic role graph as optional scaffolding/provenance.
- Standard flow must be: selected variant -> typed extension body/boundary -> plugin legality -> optional plugin realization -> provider-built route -> common EmitC.
- Core sees route ids/artifact kinds only as validated mirrors, not authority.

### `plugin-protocol/extension-family-plugin-template.md`

Severity: P0/P1.

Problems:

- The template over-formalizes manifests, semantic role graphs, construction-template artifacts, object/header bundle bridges, and readiness/emission planning. This can become a RAG/status/metadata system.
- Toy/Template/TensorExt examples are too positive/current.
- RVV exemplar says `rvv-finite-binary`, which preserves the wrong mental model.
- Common pass list includes `tcrv-select-route`; should be variant selection or route materialization through plugin provider.

Required update:

- Demote manifest/role graph to optional planning artifacts, never executable authority.
- Mark Toy/Template/TensorExt object/header/bundle examples as Stage3+/example-only.
- Remove readiness language.
- Replace RVV exemplar with corrected typed vector-level RVV surface.
- Replace `tcrv-select-route` with `tcrv-select-variants` and `tcrv-materialize-emitc-lowerable-routes` through origin plugins.

### `testing/index.md`

Severity: P1.

Problems:

- Checklist does not include the updated RVV anti-drift test rules.

Required update:

- Add: no positive legacy i32m1 route; no source-front-door positive route; tests are attached to production-path changes; runtime claims require ssh rvv.

### `testing/mlir-testing-contract.md`

Severity: P0/P1.

Good:

- Strongly says legacy RVV i32m1 route tests must be deleted/fail-closed.

Problems:

- Later sections still allow source-level positive coverage through plugin source artifact bundle front door if default eligible.
- “legal RVV explicit microkernel” and “selected RVV capacity metadata” are stale concepts.
- “current bounded RVV explicit-op EmitC materialization route” can be read as preserving legacy positive tests.
- End-to-end helper may export deterministic RVV C source; this must be evidence tooling only and not route authority.

Required update:

- Positive RVV generated artifact tests are allowed only for corrected generic typed `tcrv_rvv` route after Stage1 reset, not legacy i32m1.
- Default source-front-door RVV positive tests must be removed/fail-closed during Stage1.
- Replace “explicit microkernel” with “explicit typed vector-level `tcrv_rvv` body.”
- Remove “selected RVV capacity metadata” as authority.

### `validation/index.md`

Severity: P1.

Problems:

- Needs stage-gate phrasing.

Required update:

- Add: validation scenarios do not authorize frontend/offload/IME work before RVV maturity.

### `validation/experiment-reference.md`

Severity: P1.

Problems:

- Q1 lists matmul/softmax/layernorm/etc. These are valid future validation objects, but can be mistaken for current high-level frontend coverage.
- Q2/Q4/Q5 mention offload/IME dispatch before RVV maturity.

Required update:

- Mark structured kernels as validation scenarios for Stage2 coverage and future frontend proof, not current source route authority.
- Mark offload/IME experiments Stage3/later.
- Keep MLIR Linalg/Vector default lowering as comparison/reference only.

### `variant-pipeline/index.md`

Severity: P1.

Problems:

- Needs updated stage gates and no metadata authority rule.

Required update:

- Add Stage1/Stage2/Stage3 gates and typed-body authority checklist.

### `variant-pipeline/generation-selection-tuning.md`

Severity: P1.

Good:

- Mostly aligned on Stage1/Stage2 and selected-body realization.

Problems:

- Current inputs include “bounded selected-path metadata,” too broad.
- Variant required fields include shape/dtype/layout preconditions, cost/tuning metadata, emission path. This can imply metadata defines dtype/route.
- Emission-plan diagnostics are still presented as a regular post-realization step; this is okay only if non-authoritative.

Required update:

- Restrict selected-path metadata to diagnostic/control mirrors only.
- Say dtype/config/operation facts for executable RVV route must be structural in typed body or consumed into realized body.
- Make emission plans optional diagnostics/handoff mirrors, not route authority or progress status.

---

## Final Agent Prompt

Use the following prompt for Hermes/Codex or a spec-update agent.

```text
You are updating the TianChen-RV `.trellis/spec/` tree. Do not modify compiler source in this round unless explicitly requested. Your task is to make the Trellis specs internally consistent with the 2026-05-18/05-19 RVV grill consensus and remove contradictions that would mislead future Hermes+Codex rounds.

Source of truth
===============
Use this architecture as the canonical RVV-first path:

tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv vector-level body
  -> RVV plugin-owned legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV intrinsic C/C++ or equivalent backend representation
  -> target artifact
  -> ssh rvv evidence when runtime/correctness/performance is claimed

Authority placement
===================
1. `tcrv.exec` owns execution envelope and organization only:
   kernel, capability scope, variants, requires, dispatch/fallback, diagnostics,
   mem_window/runtime_param/ABI role declarations.
   It does not own compute semantics, dtype authority, RVV schedule, intrinsic spelling,
   or selected route authority.

2. `tcrv_rvv` owns typed vector-level RVV execution structure:
   typed vector values, dtype, SEW/LMUL/vtype policy constraints, VL/AVL/setvl,
   load/store/memory forms, arithmetic/compare/select/FMA, reduction/accumulator,
   mask/tail, movement/layout/conversion, runtime ABI value use, and low-level
   vector control.

3. Dtype enters through typed `tcrv_rvv` body/config, for example a generic typed value
   such as `!tcrv_rvv.vector<elem = i32, lmul = m1>` or equivalent structural type/config.
   Route derives intrinsic from op kind + element type + SEW + LMUL + policy + operand form + memory form.
   Do not infer dtype/config/operation from C ABI strings, parameter names, route ids,
   artifact names, test names, exact `__riscv_*_i32m1` spellings, or old `!tcrv_rvv.i32m1` helpers.

4. Parameters flow as follows:
   `tcrv.exec.mem_window` / `tcrv.exec.runtime_param` declare ABI/runtime roles
   such as lhs/rhs/out/n, buffer/scalar, input/output, runtime count, C ABI spelling.
   The selected `tcrv_rvv` body must explicitly bind/import those values and consume them
   through typed RVV control/dataflow ops. `tcrv.exec` must not infer compute from roles.

5. RVV plugin owns legality, selected-body realization, route support, intrinsic mapping,
   C/RVV type mapping, ABI mapping, and fail-closed diagnostics.
   Common EmitC/export only materializes and packages provider-built routes.
   Common code must not choose RVV intrinsics, infer dtype, create RVV schedules,
   or branch on RVV semantics.

Stages
======
Stage 1 is RVV route-authority reset. Replace or fail-close active paths that still treat
bounded `i32m1` arithmetic, `RVVI32M1*` route specs/slices, finite `tcrv_rvv.i32_*` ops,
route ids, artifact names, source-front-door patterns, descriptor residue, or exact
`__riscv_*_i32m1` spellings as RVV route authority. Stage 1 ends only when no active compiler
path treats those as the RVV family architecture. Do not preserve an executable legacy i32
compatibility route. Do not add new dtype-prefixed helper families.

Stage 2 is route-supported RVV coverage expansion on the corrected typed low-level `tcrv_rvv`
surface. Coverage is calibrated by Linalg-like structured computation classes, including
elementwise, broadcast, reduction, contraction-like accumulation, memory movement, dtype
conversion, mask/tail, runtime shape/control, and layout/movement. This is not permission to
build a current Linalg frontend, high-level kernel ops, one-intrinsic wrappers, dtype/LMUL clone
batches, global autotuning databases, dashboards, or readiness state machines.

Stage 2 also contains the RVV performance layer: RVV plugin-local selected-body realization.
It is one linear step:

selected pre-realized tcrv_rvv body
  + target capability
  + runtime SSA / ABI values
  + optional hints / policy / profile
    -> realized tcrv_rvv selected body
    -> faithful EmitC/intrinsic lowering

Hints/config are not final products. If they affect generated code, they must be consumed into
real `tcrv_rvv` structure: dynamic VL/setvl placement, legal SEW/LMUL/policy, memory forms,
mask/tail materialization, register-pressure-safe unroll, prefetch/software pipeline structure,
and accumulator/reduction layout. No readiness state machine, no status dashboard, no artifact
ledger, no progress/report surface.

Stage 3 is extension-family generalization / second-family integration after RVV maturity.
IME, Offload, TensorExtLite, Template/Toy source-front-door examples, and future plugin examples
must be marked Stage3/later or optional examples unless an explicit task says otherwise.
They must not displace Stage1/Stage2 RVV work.

Files to update
===============
Update all Markdown files under `.trellis/spec/` for consistency, especially:

- `index.md`: add root RVV-first source of truth, stage gates, typed-body authority.
- `architecture/*`: add current RVV-first path vs future frontend path; remove core route-selection authority; demote manifests to optional scaffolding/provenance.
- `capability-model/*`: clarify capability facts constrain legality/realization but do not create dtype/config/route; config hints must be consumed into typed/realized body.
- `core-dialect/tcrv-exec-contract.md`: remove `selected route` from `tcrv.exec` ownership; make diagnostics non-authoritative; clarify C ABI spelling is not dtype authority; core verifier delegates extension legality to plugins.
- `extension-plugins/rvv-plugin.md`: keep correct route-authority/body/realization sections, but remove or rewrite all positive legacy i32m1 route-table support. Legacy explicit bodies may be parse/verify/fail-closed fixtures only unless rewritten as ordinary instances of corrected generic typed `tcrv_rvv` surface. Do not leave a supported `rvv-i32m1-*` object/header/bundle plan.
- `extension-plugins/scalar-fallback-plugin.md`: replace readiness/status wording with unsupported diagnostics; no active scalar executable path.
- `extension-plugins/ime-plugin.md`, `offload-runtime-plugin.md`, `future-plugins.md`: gate to Stage3/later and evidence scenarios; no current source/high-level frontend authority.
- `plugin-protocol/interfaces-and-registry.md`: common selected-body realization slot is good; clarify result booleans are transient C++ result codes, not persisted IR states. Make source front-door default safe policy explicit-only/disabled-by-default, not broadly Eligible. Move Toy/TensorExtLite source-front-door positive flows to future/Stage3 examples. Demote emission-plan diagnostics to optional mirrors.
- `plugin-protocol/extension-plugin-integration.md` and `extension-family-plugin-template.md`: demote Extension Manifest / semantic role graph / Template object-header bundle examples to planning/provenance/future examples. They cannot be source or route authority. Add selected-body realization in the standard flow. Replace `tcrv-select-route` wording with variant selection and plugin route materialization.
- `lowering-runtime/emission-runtime-contract.md`: remove the supported explicit typed RVV i32m1 legacy route-table object plan. All legacy i32m1 route-table cases must be unsupported/fail-closed unless already rewritten onto corrected typed generic surface. Emission plans are optional diagnostics/handoff mirrors, not executable route/status authority. Target artifact export must consume provider-built `TCRVEmitCLowerableRoute` and materialized EmitC, with metadata only as exact mirrors.
- `lowering-runtime/emitc-route.md`: common EmitC helpers are neutral. Plugin resolves RVV C vector type, header, intrinsic name, ABI mapping, and route payload. Remove positive RVV source-artifact front-door coverage. Source-artifact bundle front doors are future/Stage3 or explicit-only disabled-by-default unless a mature corrected typed route exists.
- `variant-pipeline/generation-selection-tuning.md`: keep Stage1/Stage2 rules, but restrict selected-path metadata to diagnostic/control mirrors; dtype/config/route facts must be structural in typed or realized body.
- `testing/mlir-testing-contract.md`: remove all positive legacy i32m1 route expectations. Positive generated artifacts are allowed only for corrected generic typed `tcrv_rvv` routes after Stage1 reset, not old `RVVI32M1*` paths. Default source-front-door RVV positive tests must fail closed during Stage1. Replace “explicit microkernel” with “explicit typed vector-level `tcrv_rvv` body.”
- `implementation-stack/supervision-loop.md`: keep anti-stall and stage gates, but replace mature path with the full chain. Add Hermes redirect rules for legacy i32 positive artifact paths, source-front-door positive RVV paths, emission-plan/status authority, and test/dashboard/report-only progress.
- `guides/*`, `validation/*`, and all `index.md` files: update checklists to include typed body authority, explicit ABI value binding, selected-body realization, no legacy i32 positive route, no source-front-door current route, and no status/artifact metadata authority.

Forbidden outcomes
==================
After the update, no spec may require or praise these as current positive paths:

- supported `rvv-i32m1-*` materialized EmitC object/header/bundle plans;
- `RVVI32M1*` route table as executable compatibility route;
- finite `tcrv_rvv.i32_*` namespace as dtype propagation;
- exact `__riscv_*_i32m1` spelling as maturity target;
- source-front-door / source-artifact-bundle positive RVV artifact generation in Stage1;
- `tcrv.exec` owning a selected route or compute semantics;
- emission-plan `status` / readiness / dashboard / artifact metadata as route authority;
- common EmitC/export choosing RVV intrinsic, dtype, SEW/LMUL, schedule, or body shape;
- manifest / semantic role graph / construction template as executable source authority;
- IME/Offload/TensorExtLite/Template/Toy current positive workflows before RVV maturity.

Required grep/sweep
===================
Run a consistency sweep over `.trellis/spec/` for these strings and fix context:

```text
RVVI32M1
rvv-i32m1
tcrv_rvv.i32_
!tcrv_rvv.i32m1
__riscv_*_i32m1
with_vl
source-artifact
source-front-door
DefaultArtifactFrontDoorPolicy::Eligible
status = "supported"
readiness
emission_plan
artifact kind: riscv-elf-relocatable-object
selected route
route selection
descriptor
manifest
semantic role graph
Template
TensorExtLite
Toy
explicit microkernel
selected RVV capacity metadata
```

Do not delete every occurrence blindly. Legacy terms may remain only in clearly labeled deprecated/fail-closed inventory or historical residue sections. No occurrence may define a current supported route, artifact path, positive test, or future architecture template unless it is the corrected generic typed `tcrv_rvv` surface.

Acceptance criteria
===================
The updated spec tree is acceptable only if:

1. The root docs clearly state the full RVV-first authority chain and stage order.
2. No file says legacy `i32m1` route-table paths can return supported object/header/bundle plans.
3. No file treats emission-plan status, readiness, route id, artifact name, manifest, or source-front-door metadata as route/compute/dtype authority.
4. `tcrv.exec` is consistently compute-free and route-free; it owns envelope and selected variant organization only.
5. Dtype/config/operation facts for RVV executable lowering are consistently structural in typed `tcrv_rvv` body or consumed into realized body.
6. Parameter roles are consistently declared in `tcrv.exec` and explicitly imported/consumed by `tcrv_rvv` body.
7. Selected-body realization is consistently described as common orchestration plus plugin-owned semantics, before emission planning, with no state machine.
8. Common EmitC/export consistently materializes provider-built routes and never invents RVV semantics.
9. Source-front-door, Toy, TensorExtLite, Template, IME, Offload, and future plugin positives are future/Stage3 examples unless explicitly tied to corrected typed-body route maturity.
10. Testing and supervision specs direct Hermes/Codex toward production-path compiler changes and away from test-only, dashboard-only, source-front-door, and legacy-route compatibility work.

Deliverables
============
Commit only spec/documentation changes. In the final report, list:

- changed files;
- which contradictions were removed;
- which legacy references remain and why they are now safe fail-closed inventory;
- grep/sweep results for the required strings;
- any unresolved ambiguity that needs human decision.
```
