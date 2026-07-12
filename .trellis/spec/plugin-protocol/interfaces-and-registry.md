# Interfaces And Registry

## Scope

This spec defines the generic plugin interfaces used by core orchestration.
Interfaces let core call extension plugins without hard-coding RVV, IME,
Offload, TensorExtLite, Template/Toy, scalar fallback, or future-family
semantics.

The current RVV authority chain remains:

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv vector-level body
  -> RVV plugin-owned legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
```

## [P-1] Interface Freeze (Quasi-ABI)

`ExtensionPlugin` is a **quasi-ABI**, not an internal convenience surface. Its
serializable signature form is a declared schema object that enters schema.def ⑤
(a declared-schema artifact governed by 总纲 [S-5]⑤ — reference by ID; the
schema.def content and its owning layer are declared elsewhere, do not restate
them here). Because it is a
quasi-ABI, any change to the interface signature goes through **RFC**, and the
change is recorded as `extension not modification` when it is purely additive
([F-2′] operation gate).

Through this frozen interface a plugin may contribute:

```text
fact + relation declarations
legality predicates
emission patterns (registered to [PAT-1], not open-coded into core)
lowering hooks
tests
```

A plugin **must not edit any file in the core directory** to integrate. Needing a
core edit means either the interface is genuinely missing an execution semantic
(justify via RFC + core-invariants I3 review, 见 [locality-contract.md](./locality-contract.md))
or the work is misplaced.

## Registry Rules

The registry owns lookup and orchestration only:

- plugin registration;
- capability/provider lookup;
- dialect and interface registration;
- variant builder lookup;
- legality hook lookup;
- selected-body realization hook lookup;
- route provider lookup;
- optional diagnostic mirror collection.

The registry must not derive computation from plugin names, family names, route
ids, manifests, semantic role graphs, source-front-door markers, artifact
metadata, or descriptors.

## Required Plugin Interfaces

### Hook Signatures — where to find the compilable form [GAP-P4-HOOK-SIG]

Per [P-1] the exact serializable signatures are declared once (schema.def ⑤) and
the C++ interface is not re-copied prose-by-prose. But an author still needs to
know **which header, which class, and which virtuals to override**. Authoritative
source: `include/TianChenRV/Plugin/ExtensionPlugin.h`, class `ExtensionPlugin`
(verify exact types there; the list below is a discovery index only, 2026-07-12).
The overridable `virtual llvm::Error …` hooks a family plugin implements:

```text
getName / getVersion / getCapabilities / registerDialects   identity, fact instances, dialect registration
supportsOperation / proposeVariants / collectVariantProposals  variant proposal
verifyVariantLegality                                        ② legality predicate
estimateVariantCost(VariantCostRequest&, VariantCostEstimate&)   selector cost hook ([SEL-1] cold-start prior)
checkVariantEmissionReadiness / buildVariantEmissionPlan     emission planning
materializeSelectedLoweringBoundary / validateSelectedLoweringBoundary  selected-body realization boundary
configureTargetSupportExtensionBundle / registerTargetSupportTranslateRoutes  target-support wiring
```

The registration free function `register<Fam>ExtensionPlugin(ExtensionPluginRegistry&)`
(an `ExtensionPluginRegistrationFn`, declared in the family header) is separate
from these virtuals — it is what the built-in catalog calls (见 [GAP-P4-REGISTER]
in [extension-plugin-integration.md](./extension-plugin-integration.md)).

### Capability And Proposal

Plugins may expose capability matching and variant proposal hooks. Proposed
variants must carry structured capability requirements and origin plugin
identity. Capability facts constrain legality/realization; they do not create
RVV dtype/config/body/route identities.

### Legality

Plugin legality owns extension-specific validation. Core checks generic
structure, symbol resolution, capability presence, dispatch/fallback shape, and
fallback coherence. Core does not hard-code RVV/IME/offload legality.

### Selected-Body Realization

`realizeSelectedVariantBody` or an equivalent hook is a transient C++ compiler
operation:

```text
selected pre-realized extension body
  + target capability
  + runtime SSA / ABI values
  + optional hints / policy / profile
    -> realized selected extension body
```

For RVV, any hint/config/profile fact that affects generated code must be
consumed into concrete `tcrv_rvv` body structure before route construction.

`VariantSelectedBodyRealizationResult::isRealized` and similar booleans are
transient C++ result codes. They are not persisted IR readiness states,
dashboards, artifact ledgers, or progress markers.

### Route Provider

The origin plugin owns route construction:

```text
selected typed/realized body
  -> plugin route provider
  -> TCRVEmitCLowerableRoute
```

For RVV, the provider maps op kind, dtype, SEW, LMUL, policy, operand form,
memory form, and ABI binding to RVV intrinsic/backend representation. Common
EmitC never performs that mapping.

### EmitC Lowerable Interface

The common EmitC lowerable interface exposes only generic materialization
shape. It may carry provider-owned payloads, but it must not interpret
extension computation.

### Compute Role Interface

`TCRVComputeOpInterface` or similar common interfaces expose generic
role/provenance only. They must not let common code interpret computation
semantics, infer dtype, choose intrinsics, or branch on concrete families.

## Source Front-Door Policy

The safe default policy for source front doors is explicit-only or disabled:

```text
DefaultArtifactFrontDoorPolicy::ExplicitOnly
```

`DefaultArtifactFrontDoorPolicy::Eligible` is not a safe default for RVV work.
A broadly eligible source-front-door policy may be used only by an explicit
future task once mature typed-body routes exist and the target family proves
plugin-owned source interpretation.

Standing rules (source-front-door routes fail closed, 见 core-invariants I7):

- source-only RVV markers fail closed by default;
- source-artifact bundle pipelines do not generate positive RVV artifacts;
- source-front-door metadata is not route, dtype, or compute authority;
- tests for source-front-door RVV behavior are negative/fail-closed unless
  explicitly tied to a corrected typed-body route.

Toy and TensorExtLite source-front-door scenarios are future examples. They
must not become current default workflows or templates that displace RVV work.

## Optional Manifests And Templates

Extension manifests, semantic role graphs, construction templates, and example
source-front-door materials are optional planning/provenance artifacts. They
can document how a plugin is organized, but they cannot be executable source
authority, route authority, dtype authority, artifact authority, or proof of
progress.

## Emission-Plan Mirrors

Plugin interfaces may return optional diagnostic mirrors after route
construction. These mirrors may be serialized as
`tcrv.exec.diagnostic {reason = "emission_plan"}`. They are not route inputs.

Allowed mirror fields include:

```text
origin plugin
selected variant reference
lowering boundary reference
runtime ABI name
artifact kind/name mirror
required capability mirror
diagnostic result/reason
```

Fields must mirror already-made plugin decisions. They must not decide or
recover route support.

## Current RVV Fail-Closed Inventory

The registry and route provider must fail closed if the only authority is:

```text
RVVI32M1* route specs/slices
rvv-i32m1-* route ids
tcrv_rvv.i32_* helper namespace
!tcrv_rvv.i32m1 helper type
exact __riscv_*_i32m1 spelling
source-front-door marker
source-artifact bundle marker
descriptor residue
emission_plan status
artifact name/kind
manifest / semantic role graph / construction template
```

These names may remain only in negative tests, deprecated inventory, or
historical migration notes.

## Good / Bad Cases

Good:

```text
core registry calls RVV legality
-> RVV realizes selected body
-> RVV route provider returns TCRVEmitCLowerableRoute
-> common EmitC materializes provider payload
```

Good:

```text
source-front-door RVV input
-> fail-closed diagnostic
-> no target artifact
```

Bad:

```text
DefaultArtifactFrontDoorPolicy::Eligible
-> positive RVV source-artifact route
```

Bad:

```text
Toy source marker or TensorExtLite construction template
-> current default route authority
```

Bad:

```text
VariantSelectedBodyRealizationResult::isRealized
-> persisted IR readiness state
```

## Tests Required

Interface or registry changes require focused tests:

- C++ tests for plugin lookup, selected-body realization result handling, and
  route provider dispatch;
- lit/FileCheck for generic diagnostics and fail-closed behavior visible in
  textual MLIR;
- negative RVV tests for legacy `RVVI32M1*`, `rvv-i32m1`,
  source-front-door, and metadata-only route inputs;
- positive tests only when they use corrected typed extension bodies and
  provider-built routes.
