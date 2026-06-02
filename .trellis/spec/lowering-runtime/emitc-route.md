# Unified EmitC Route

## Scope

The unified EmitC route is the common materialization path after an origin
plugin has selected, legalized, optionally realized, and provided a lowerable
route for an extension-family body.

Current RVV route:

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv vector-level body
  -> RVV plugin-owned legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV intrinsic C/C++ or equivalent backend representation
  -> target artifact
```

## Common Responsibility

Common EmitC owns neutral materialization:

- invoke the generic lowerable-route interface;
- emit MLIR EmitC operations from provider payloads;
- preserve deterministic symbol names and packaging structure when supplied;
- hand materialized EmitC to C/C++ emission and target export;
- report diagnostics when the provider route is absent or malformed.

Common EmitC must not:

- choose RVV intrinsics;
- infer dtype, SEW, LMUL, mask/tail policy, or operation kind;
- create RVV schedules or body shape;
- synthesize runtime ABI from parameter names;
- branch on RVV/IME/Offload/Toy/TensorExtLite semantics;
- treat route ids, emission-plan status, manifests, or artifact metadata as
  route authority.

For RVV, C vector type strings, intrinsic names, headers, ABI mapping, route
payload, and legality remain RVV-plugin-owned. Common code may carry
precomputed strings/payload fragments supplied by the plugin, but it does not
derive them.

## `TCRVEmitCLowerableRoute`

`TCRVEmitCLowerableRoute` is the common contract object passed from origin
plugin/provider to common materialization. It should expose only the generic
information needed for materialization:

```text
origin plugin / family
selected variant reference
typed/realized body or selected boundary reference
provider-owned route payload
required headers / includes as provider output
runtime ABI bindings as provider output
EmitC operation construction hooks or payload
diagnostic mirrors
```

The route object is provider-built. It is not reconstructed from
emission-plan diagnostics, route ids, artifact names, source-front-door
markers, semantic role graphs, manifests, or descriptors.

## Provider Operand Binding Summary Contract

### 1. Scope / Trigger

When an extension provider emits an operand-binding summary that is consumed by
Common EmitC and later mirrored into a target artifact bundle, every runtime
ABI parameter that appears in the generated C/C++ prototype or target header
must be represented as a provider-derived binding entry and must explicitly
mark header/prototype participation.

### 2. Signatures

The durable payload shape is a provider-owned route field:

```text
route_operand_binding_plan = <provider-plan-id>
route_operand_binding_operands =
  <provider-plan-id>;<logical>=<role>:<c-name>:abi|...|hdr;...
```

The exact tokens remain route-family-owned, but the `hdr` marker is required
for runtime ABI parameters exported through the generated header/prototype.

### 3. Contracts

- The provider builds the summary from the same typed body/config/runtime facts
  used to construct the route.
- Common EmitC carries the summary unchanged as a provider payload; it must not
  add, remove, or infer route-family tokens.
- Target artifact validation compares the provider summary and candidate
  mirror exactly before accepting the bundle.
- Combined routes must keep all families of ABI facts in one summary. For
  example, a computed-mask plus strided-input contraction route must bind
  compare operands, dot source operands, accumulator/result operands, runtime
  count, and stride operands in the same provider summary.
- The summary is mirrored as target artifact metadata and must fit the target
  export bounded single-line metadata contract. When a route family needs many
  structural use tokens, shorten only the provider plan label (for example, an
  abbreviated family label) rather than dropping logical operands, provider
  `abi` markers, exported `hdr` markers, or required use tokens.

### 4. Validation & Error Matrix

- Missing binding entry for a generated header/prototype parameter -> fail
  before target artifact acceptance.
- Binding entry lacks the header/prototype marker for an exported runtime ABI
  parameter -> fail before target artifact acceptance.
- Binding role, C name, order, or route-family token disagrees with the
  provider route description -> fail before target artifact acceptance.
- Candidate artifact metadata carries a stale binding summary that does not
  exactly match provider facts -> fail before target artifact acceptance.
- Binding summary exceeds the target artifact bounded metadata limit -> fail
  during export; repair by shortening the provider plan label while keeping
  every provider-derived operand fact intact.

### 5. Good/Base/Bad Cases

- Good: provider summary binds all exported compare/source/stride/result ABI
  parameters and marks them as header/prototype participants.
- Good: for `segment2_deinterleave_unit_store`, the provider summary binds
  `src`, `out0`, `out1`, and `n` in that runtime ABI order, and every exported
  entry carries both `abi` and `hdr`. Target artifact validation must reject
  stale `runtime-abi-mirror` / `header` summaries before bundle acceptance.
- Good: for `segment2_interleave_unit_load`, the provider summary binds
  `src0`, `src1`, `dst`, and `n` in that runtime ABI order, and every exported
  entry carries both `abi` and `hdr`. Target artifact validation must reject
  stale `runtime-abi-mirror` / `header` summaries before bundle acceptance.
- Base: non-exported internal temporaries do not receive runtime ABI header
  markers.
- Bad: a combined route omits header/prototype markers on compare, dot source,
  or stride runtime ABI entries while still emitting those parameters in the C
  prototype.
- Bad: a route deletes field-role, arithmetic, tuple, memory, `abi`, or `hdr`
  use tokens just to fit artifact metadata length; the structural contract was
  weakened even if the string became shorter.

### 6. Tests Required

- FileCheck or script dry-run tests must check the exact provider summary for
  the route family.
- C++ target artifact tests must mutate route descriptions or candidate
  mirrors and prove stale/missing operand-binding summaries fail closed.
- When a provider plan label is abbreviated to satisfy bounded metadata, tests
  must check the abbreviated plan id and the full operand summary, proving the
  abbreviation did not remove logical operands or `abi`/`hdr` participation.
- Runtime evidence must not be used to define the binding contract; it only
  proves executable behavior after the provider and target validators accept
  the route.

### 7. Wrong vs Correct

Wrong:

```text
route summary omits hdr on an exported stride parameter
  -> target header still emits size_t lhs_stride
```

Correct:

```text
typed body/runtime ABI fact
  -> provider operand binding includes lhs_stride=...:abi|str|addr|hdr
  -> Common EmitC carries it unchanged
  -> target artifact mirror matches exactly
```

## RVV Rules

An RVV lowerable route is valid only when:

1. The selected path is a selected RVV variant in `tcrv.exec`.
2. A typed or realized `tcrv_rvv` body is present.
3. Dtype/config/operation facts are structural in the body.
4. ABI values are imported/consumed by the body.
5. RVV plugin legality accepts the body.
6. RVV selected-body realization has consumed code-affecting hints/config.
7. RVV provider maps the body to a `TCRVEmitCLowerableRoute`.

Invalid RVV route inputs:

```text
RVVI32M1* route table as executable compatibility route
rvv-i32m1-* route id
tcrv_rvv.i32_* helper namespace as dtype propagation
!tcrv_rvv.i32m1 as mature body type
exact __riscv_*_i32m1 spelling as maturity target
source-front-door marker
source-artifact bundle metadata
emission_plan status
artifact kind/name
manifest / semantic role graph / construction template
descriptor residue
```

These terms may appear only in negative tests, deprecated/fail-closed
inventory, or historical migration notes.

## Source-Artifact Front Door

The common source-artifact bundle front door is disabled by default for current
RVV Stage 1. It may exist only as a future/Stage3 or explicit opt-in mechanism
after a mature corrected typed route exists.

During Stage 1:

- no positive RVV source-front-door artifact coverage;
- no source-only RVV marker -> object/header/bundle path;
- no source pattern -> route id -> intrinsic route;
- source-front-door tests must be negative/fail-closed unless they already
  consume corrected typed body routes through an explicitly selected mature
  path.

Toy, TensorExtLite, Template, IME, Offload, and future plugin source-front-door
examples are Stage3/later examples, not current default workflows.

## Emission-Plan Mirrors

Selected emission-plan diagnostics may be serialized as optional mirrors after
provider route construction and materialized EmitC. They must not be required
for materialization and must not define route support.

Correct:

```text
provider-built route -> common EmitC -> optional emission-plan mirror
```

Wrong:

```text
emission_plan status/route id -> common EmitC route
```

## Target Artifact Handoff

Target artifact export consumes materialized EmitC plus provider route mirrors.
It may validate that mirrors match the selected variant and route, but it must
not rebuild or infer route details from metadata.

Export fails closed when only these are present:

```text
emission-plan metadata
selected-path metadata
route id
artifact kind/name
source-front-door marker
manifest
semantic role graph
descriptor
legacy i32m1 route table
```

## Good / Bad Cases

Good:

```text
typed tcrv_rvv binary body
  -> RVV provider derives intrinsic/type/header/ABI payload
  -> common materializes EmitC
```

Good:

```text
legacy rvv-i32m1 route-table input
  -> fail-closed diagnostic
  -> no target artifact
```

Bad:

```text
Default source-artifact pipeline recognizes RVV add/sub/mul source marker
  -> emits object/header/bundle during Stage 1
```

Bad:

```text
common intrinsic name resolver chooses __riscv_*_i32m1
```

Bad:

```text
artifact metadata says riscv-elf-relocatable-object
  -> export succeeds without provider-built route and materialized EmitC
```

## Review Checklist

- [ ] Does the route provider, not common EmitC, own extension-specific mapping?
- [ ] Does common materialization consume `TCRVEmitCLowerableRoute`?
- [ ] Are RVV dtype/config/schedule/intrinsic choices plugin-owned?
- [ ] Are source-front-door paths disabled/fail-closed for RVV Stage 1?
- [ ] Are emission-plan diagnostics optional mirrors only?
- [ ] Are legacy i32m1 route-table paths unsupported/fail-closed?
- [ ] Do target artifacts come from materialized EmitC, not metadata?
