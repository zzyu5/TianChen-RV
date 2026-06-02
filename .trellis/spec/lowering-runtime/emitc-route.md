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

## Provider Header And Type Summary Contract

### 1. Scope / Trigger

When an extension provider emits `required_header_declarations` or
`c_type_mapping` summaries that are later mirrored into a target artifact,
target artifact validation must check those summaries against the
operation-specific provider contract before accepting the bundle.

### 2. Signatures

For RVV selected-body routes, the provider-owned route description carries
bounded summary fields such as:

```text
required_header_declarations = stddef.h,stdint.h,riscv_vector.h
c_type_mapping = vl:size_t,compare/source:signed-e32m1,index:u32m1,mask:b32,dst:masked-indexed-store
```

### 3. Contracts

- The provider derives these summaries from the same typed
  `tcrv_rvv` body/config/runtime facts used to build the route.
- Common EmitC carries route headers, type mappings, and metadata mirrors
  without inventing RVV header/type semantics.
- Target validation must verify both:
  - the rebuilt route actually has the required headers and type mappings;
  - the provider summary fields exactly match the expected route-family facts.
- Candidate artifact metadata must mirror the provider summary exactly after
  provider route construction.

### 4. Validation & Error Matrix

- Missing provider header summary -> fail before artifact export.
- Provider header summary differs from the expected route-family header set ->
  fail before artifact export.
- Provider C type summary differs from the expected route-family C type facts
  -> fail before artifact export.
- Rebuilt route lacks a header/type mapping named by the provider summary ->
  fail before artifact export.
- Candidate artifact metadata carries stale header/type summary mirrors ->
  fail before bundle acceptance.

### 5. Good/Base/Bad Cases

- Good: computed-mask indexed scatter-store records
  `stddef.h,stdint.h,riscv_vector.h` and
  `vl:size_t,compare/source:signed-e32m1,index:u32m1,mask:b32,dst:masked-indexed-store`,
  and the target validator compares those exact strings with provider facts and
  artifact mirrors.
- Base: non-RVV routes use their own provider-family summary contract or omit
  RVV-specific keys.
- Bad: a route has `riscv_vector.h` in the rebuilt route object but mirrors
  `required_header_declarations = stddef.h,stdint.h`; this is stale metadata
  and must not be accepted.
- Bad: indexed routes fold the index vector into a signed payload group in
  `c_type_mapping`; the index must remain explicit as `index:u32m1`.

### 6. Tests Required

- Target artifact C++ tests must mutate provider route descriptions and
  candidate metadata mirrors to prove stale header/type summaries fail closed.
- Generated-bundle dry-run FileCheck tests must assert representative
  `required_header_declarations` and `c_type_mapping` values for routes that
  claim generated artifact support.

### 7. Wrong vs Correct

Wrong:

```text
route object contains riscv_vector.h
  -> provider summary says stddef.h,stdint.h
  -> artifact accepted because route headers were usable
```

Correct:

```text
typed tcrv_rvv body/config/runtime facts
  -> provider emits operation-specific headers and C type summary
  -> target validator checks the provider summary and route payload
  -> artifact metadata mirrors the provider summary exactly
```

## RVV Route-Family Fact Surface Contract

### 1. Scope / Trigger

When an RVV route family is consumed by both the RVV provider and target
artifact validation, operation-specific constants must live behind a
provider-owned fact surface rather than being independently reconstructed in
target code. This applies to combined routes such as computed-mask MAcc where
ABI order, compare operands, mask source, accumulator passthrough, headers,
types, and binding summaries must stay identical across provider planning,
route construction, target validation, and generated-bundle evidence.

### 2. Signatures

The concrete C++ shape is a small provider-owned facts struct plus accessor:

```c++
struct RVV<RouteFamily>RouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  // Route-family fields: typed op, predicate, mask, layout, intrinsic,
  // accumulator/result contracts, stride/index/segment facts, etc.
};

std::optional<RVV<RouteFamily>RouteFacts>
getRVV<RouteFamily>RouteFacts(RVVSelectedBodyOperationKind operation);
```

For route families whose concrete facts vary by typed RVV configuration, expose
the selected configuration explicitly:

```c++
std::optional<RVV<RouteFamily>RouteFacts>
getRVV<RouteFamily>RouteFacts(RVVSelectedBodyOperationKind operation,
                              std::int64_t sew,
                              llvm::StringRef lmul);
```

### 3. Contracts

- The accessor is implemented in the RVV plugin/provider layer and is derived
  from the same typed body/config/runtime authority used to build the route.
- Provider plan validation must fail closed when a supported operation cannot
  obtain its canonical facts from the accessor.
- Target artifact validation may consume the accessor, but must pass the
  operation, SEW, and LMUL from the provider-built route description whenever
  those fields affect runtime ABI types, RVV intrinsic spellings, type mapping,
  or binding summaries.
- Target artifact validation must not define its own duplicate source of
  route-family truth for the same fields.
- Candidate artifact metadata mirrors the provider facts only after provider
  route construction; mirror fields remain non-authoritative.
- A runtime-scalar variant and a vector/vector variant of the same route class
  should each have explicit facts when their ABI roles, predicates, or mask
  producer sources differ.

### 4. Validation & Error Matrix

- Missing provider fact accessor result for a supported operation -> provider
  and target validators fail before artifact export.
- Target-local constant disagrees with provider fact -> remove the target-local
  constant and validate against the provider fact.
- Candidate mirror carries stale runtime-scalar facts for a vector/vector
  route, or stale vector facts for a runtime-scalar route -> fail before
  bundle acceptance.
- Binding summary, ABI order, predicate, mask source, layout, header, or type
  mapping differs from the provider facts -> fail before target artifact
  acceptance.
- For `widening_macc_add`, stale source/result SEW/LMUL, accumulator/result
  layout, route operand order, contraction family plan, header/type facts, or
  provider mirror -> fail before target artifact acceptance with diagnostics
  that name the missing `i16mf2` source and `i32m1` accumulator/result facts.

### 5. Good/Base/Bad Cases

- Good: `computed_masked_macc_add` gets canonical facts for
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, `slt`, vector compare mask source,
  accumulator passthrough, binding summary, headers, and C type mapping; both
  provider and target validation consume those facts.
- Good: `widening_macc_add` gets canonical facts for `lhs,rhs,acc,out,n`,
  `tcrv_rvv.widening_macc`, `i16mf2` lhs/rhs sources, `i32m1` accumulator and
  result vectors, `signed-i16mf2xi16mf2-plus-i32m1-to-i32m1`,
  `separate-i32-vector-accumulator-input`,
  `store-widening-multiply-accumulate-result-to-output-buffer`, contraction
  route-family plan, route operand binding summary, header declarations, C
  type mapping, and explicit `provider_supported_mirror`; provider and target
  validation consume the same accessor surface.
- Base: a route family with only one consumer may keep local constants until
  those facts are shared across provider, target, scripts, or evidence.
- Bad: target validation defines its own `computed_masked_macc_add` ABI order
  and binding summary while the provider owns another copy.
- Bad: target validation accepts `widening_macc_add` because the route id,
  intrinsic spelling, generated artifact name, or C string looks like a
  widening route while provider facts are missing or stale.

### 6. Tests Required

- MLIR/FileCheck or direct pipeline tests must mutate candidate mirrors for
  stale provider facts, binding summaries, ABI order, predicate, mask source,
  accumulator/result layout, headers, and types.
- C++ target tests should cover non-textual validator behavior when practical.
- Generated-bundle dry-run must expose representative route facts and mirror
  fields in evidence JSON.
- Widening MAcc generated-bundle evidence must include provider-derived
  widening boundary facts and runtime counts that prove `n`/AVL is honored.
- RVV runtime claims still require real `ssh rvv` execution after provider and
  target validation accept the route.

### 7. Wrong vs Correct

Wrong:

```text
provider constants say cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n
target constants independently say cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n
```

Correct:

```text
typed tcrv_rvv body/config/runtime facts
  -> RVV provider fact accessor
  -> provider plan and TCRVEmitCLowerableRoute
  -> target validator consumes the same fact surface
  -> artifact/evidence mirrors match exactly
```

### Widening Conversion Fact Surface

For selected-body widening conversion routes, provider/target shared constants
must use the provider-owned surface:

```c++
struct RVVWideningConversionRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef routeFamilyPlanID;
  llvm::StringRef typedComputeOpName;
  std::int64_t sourceSEW;
  llvm::StringRef sourceLMUL;
  std::int64_t resultSEW;
  llvm::StringRef resultLMUL;
  llvm::StringRef conversionRelation;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef conversionIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vlCType;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  llvm::StringRef resultName;
  std::string routeOperandBindingSummary;
};

std::optional<RVVWideningConversionRouteFacts>
getRVVWideningConversionRouteFacts(RVVSelectedBodyOperationKind operation);
```

The accessor is the canonical provider-owned fact surface for
`widen_i32_to_i64` and `widen_i16_to_i32`. `widen_i32_to_i64` facts must include
runtime ABI order `lhs,out,n`, source `i32/m1`, result `i64/m2`, conversion
relation `signed-i32m1-to-i64m2`, typed compute op
`tcrv_rvv.widening_convert`, unit-stride conversion memory form, route family
plan, route operand binding plan/summary, required headers, C type mapping,
target leaf profile, and explicit `provider_supported_mirror`.

Provider route-family plan derivation may use typed body/config/runtime facts
to select the operation, but every shared constant above must be copied from
`getRVVWideningConversionRouteFacts(...)` or validated against it. Target
artifact validation must consume this same accessor and reject stale local
copies of source/result SEW-LMUL, conversion relation, header/type mapping,
target profile, provider mirror, route-family plan, or operand binding summary
before accepting a bundle.

Good:

```text
typed widen_i32_to_i64 body/config/runtime facts
  -> getRVVWideningConversionRouteFacts(WidenI32ToI64)
  -> widening conversion route-family plan
  -> provider-built TCRVEmitCLowerableRoute
  -> target validator consumes the same accessor
```

Bad:

```text
target validator local table says source=i32/m1 result=i64/m2
  -> accepts artifact while provider fact accessor is missing or stale
```

Required tests:

- C++ target/provider tests must fail closed for stale provider mirror, target
  leaf/profile, source/result SEW-LMUL, conversion relation, route-family plan,
  route operand binding plan/summary, and stale non-conversion route-family
  mirrors.
- Generated-bundle dry-run must expose provider-derived conversion facts and
  mirror fields.
- Runtime RVV correctness claims still require real `ssh rvv` execution after
  provider and target validators accept the route.

### Base Memory Movement Fact Surface

For selected-body base memory movement routes, provider and target shared facts
must use the provider-owned surface:

```c++
struct RVVBaseMemoryMovementRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef routeFamilyPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef sourceStrideSource;
  llvm::StringRef destinationStrideSource;
  std::int64_t indexEEW;
  llvm::StringRef offsetUnit;
  llvm::StringRef indexSource;
  llvm::StringRef indexUniqueness;
  llvm::StringRef indexedDataMemoryForm;
  llvm::StringRef indexedDestinationMemoryForm;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

std::optional<RVVBaseMemoryMovementRouteFacts>
getRVVBaseMemoryMovementRouteFacts(RVVSelectedBodyOperationKind operation);
```

Trigger: use this surface when base memory routes such as strided load/unit
store, unit load/strided store, indexed gather/unit store, indexed
scatter/unit load, masked unit load/store, or masked unit store share facts
across provider planning, target artifact validation, and generated-bundle
evidence.

Contracts:

- The accessor is implemented in the RVV provider layer and is the canonical
  source for base memory operation, memory form, runtime ABI order, target leaf
  profile, provider-supported mirror, header/type summaries, route operand
  binding plan/summary, and route-family plan.
- Indexed gather/unit store facts must include ABI order `data,index,out,n`,
  `index_eew = 32`, `offset_unit = element`, `index_source =
  runtime_abi:index`, indexed layout
  `element-indexed-data-index-unit-stride-output-runtime-abi`,
  indexed data memory form `indexed-load`, destination memory form
  `unit-stride-store`, and typed compute op `tcrv_rvv.move`.
- Indexed scatter/unit load facts must include ABI order `src,index,dst,n`,
  `index_eew = 32`, `offset_unit = element`, `index_source =
  runtime_abi:index`, `index_uniqueness = unique`, indexed layout
  `unit-stride-source-indexed-destination-index-runtime-abi`, source memory
  form `unit-stride-load`, indexed destination memory form `indexed-store`,
  destination memory form `indexed-store`, and typed compute op
  `tcrv_rvv.move`.
- Target artifact validation consumes the accessor and rejects missing or
  stale provider facts before accepting object/header/bundle artifacts.
  Candidate metadata may only mirror accessor facts after the provider route is
  rebuilt.
- Common EmitC carries the provider-built route and metadata mirrors; it must
  not infer base memory kind, index role, offset unit, header set, C type
  mapping, or ABI order from artifact names, route ids, tests, scripts, C
  strings, or mirror metadata.

Validation and errors:

- Missing accessor result for a supported base memory operation -> provider and
  target validators fail before artifact export.
- ABI order, memory form, indexed/strided/masked layout, index EEW, offset
  unit, index source, route-family plan, target leaf profile, provider mirror,
  header summary, C type mapping, or binding summary differs from accessor
  facts -> fail before artifact acceptance.
- Candidate artifact metadata carries stale strided, unit-load, indexed,
  masked, or non-base route-family facts for the selected base memory route ->
  fail before bundle acceptance.

Good:

```text
typed indexed_gather_unit_store body/config/runtime facts
  -> getRVVBaseMemoryMovementRouteFacts(IndexedGatherUnitStore)
  -> base memory route-family plan and operand-binding summary
  -> provider-built TCRVEmitCLowerableRoute
  -> target validator consumes the same facts
```

Bad:

```text
target validator local table says index_eew=32 and offset_unit=element
  -> accepts artifact while provider accessor is missing or route metadata is stale
```

Required tests:

- C++ target artifact tests must mutate provider descriptions and candidate
  metadata mirrors for stale ABI order, index EEW, offset unit, index source,
  indexed layout, route-family plan, target profile, provider mirror, header
  facts, type facts, binding summary, and accidental strided/unit-load
  fallback.
- Generated-bundle dry-run tests must assert representative accessor facts,
  base memory boundary evidence, and provider-derived operand/header/type
  summaries.
- Runtime RVV correctness claims still require real `ssh rvv` execution after
  provider and target validators accept the route.

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
