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

For computed-mask MAcc, this applies to both the vector-compare and
runtime-scalar compare variants because `runtime_scalar_cmp_masked_macc_add`
has supported LMUL-specific artifacts such as LMUL m2:

```c++
std::optional<RVVComputedMaskMAccRouteFacts>
getRVVComputedMaskMAccRouteFacts(RVVSelectedBodyOperationKind operation,
                                 std::int64_t sew, llvm::StringRef lmul);

std::optional<RVVRuntimeScalarComputedMaskMAccRouteFacts>
getRVVRuntimeScalarComputedMaskMAccRouteFacts(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul);
```

The no-argument accessor may remain as the default SEW32/LMUL m1 convenience
surface, but provider and target validators that have a rebuilt route
description must call the parameterized accessor with the actual selected
`sew` and `lmul`.

For active widening dot-reduce contraction routes, the provider-owned surface is
the shared authority for plain, strided-input, computed-mask, and
computed-mask-strided variants:

```c++
struct RVVWideningDotReduceRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef accumulatorElementTypeName;
  llvm::StringRef resultElementTypeName;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  llvm::StringRef typedComputeOpName;
  std::int64_t sourceSEW;
  llvm::StringRef sourceLMUL;
  std::int64_t accumulatorSEW;
  llvm::StringRef accumulatorLMUL;
  std::int64_t resultSEW;
  llvm::StringRef resultLMUL;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef lhsStrideSource;
  llvm::StringRef rhsStrideSource;
  llvm::StringRef wideningDotProductRelation;
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef maskedWideningProductIntrinsic;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef reductionIntrinsic;
  llvm::StringRef reductionStoreVL;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneZeroingRequirement;
  llvm::SmallVector<RuntimeABIParameter, 9> runtimeABIParameters;
};

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(RVVSelectedBodyOperationKind operation);
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
- Computed-mask MAcc facts include selected SEW/LMUL/policy/runtime-control,
  arithmetic kind, compare and runtime-scalar threshold roles, accumulator and
  output roles, mask producer/source/form, runtime ABI parameters, header/type
  summaries, target leaf profile, and `provider_supported_mirror`.
- Runtime-scalar computed-mask MAcc validation must reject an unsupported
  selected LMUL by missing provider facts rather than accepting the route via
  default LMUL m1 facts, route ids, candidate metadata, or intrinsic strings.
- Widening dot-reduce facts include narrow source type, widened
  accumulator/result type, source/result SEW-LMUL, runtime ABI order and
  parameters, route operand binding plan/summary, seed/reduction/store facts,
  unit-stride versus strided-input memory facts, header/type summary, target
  leaf profile, and `provider_supported_mirror`.
- Computed-mask widening dot-reduce variants add mask producer/source/form,
  predicate kind, inactive-lane zeroing, mask type/C type, masked product, and
  merge facts. Non-computed-mask variants must carry empty mask facts and fail
  closed when stale mask facts are present.
- Standalone reduction facts include the plain, computed-mask, and
  runtime-scalar computed-mask standalone reduce add/min/max operations.
  Provider facts carry typed compute op, memory form, selected SEW-derived
  scalar ABI parameter types, runtime ABI order and parameters, route operand
  binding plan/summary, accumulator/result layout, scalar-result runtime
  boundary, reduction store VL, header/type summary, target leaf profile, and
  `provider_supported_mirror`.
- Computed-mask standalone reduction facts add compare predicate,
  mask producer/source/form, inactive-lane zeroing versus neutral-lane
  requirement, operation-specific inactive neutral literal by SEW, accumulation
  carry contracts, and vector-compare or runtime-scalar mask producer source.
  Plain standalone reduction facts must carry empty mask/accumulation fields
  and fail closed when stale masked facts are present.

### 4. Validation & Error Matrix

- Missing provider fact accessor result for a supported operation -> provider
  and target validators fail before artifact export.
- Target-local constant disagrees with provider fact -> remove the target-local
  constant and validate against the provider fact.
- Candidate mirror carries stale runtime-scalar facts for a vector/vector
  route, or stale vector facts for a runtime-scalar route -> fail before
  bundle acceptance.
- Provider or target validation uses a no-argument/default facts accessor while
  the rebuilt route description carries LMUL m2 or another non-default selected
  configuration -> fail or fix the validator to use the parameterized accessor.
- Binding summary, ABI order, predicate, mask source, layout, header, or type
  mapping differs from the provider facts -> fail before target artifact
  acceptance.
- For `widening_macc_add`, stale source/result SEW/LMUL, accumulator/result
  layout, route operand order, contraction family plan, header/type facts, or
  provider mirror -> fail before target artifact acceptance with diagnostics
  that name the missing `i16mf2` source and `i32m1` accumulator/result facts.
- For widening dot-reduce routes, stale widening MAcc facts, stale strided facts
  on unit-stride routes, stale unit-stride facts on strided routes, missing
  reduction/result facts, stale source/result type mapping, stale mask facts,
  stale route operand binding, stale header/type mapping, stale target profile,
  stale provider mirror, or stale candidate metadata mirrors -> fail before
  target artifact acceptance.
- For standalone reduction routes, stale dot-reduce or MAcc facts, stale plain
  facts on masked routes, stale masked facts on plain routes, cross-operation
  add/min/max binding or inactive-lane facts, missing scalar-result boundary
  facts, stale neutral literal facts, stale route operand binding, stale
  header/type mapping, stale target profile, stale provider mirror, or stale
  candidate metadata mirrors -> fail before target artifact acceptance.

### 5. Good/Base/Bad Cases

- Good: `computed_masked_macc_add` gets canonical facts for
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, `slt`, vector compare mask source,
  accumulator passthrough, binding summary, headers, and C type mapping; both
  provider and target validation consume those facts.
- Good: `runtime_scalar_cmp_masked_macc_add` with selected SEW32/LMUL m2 gets
  runtime-scalar facts through the parameterized accessor, including
  `rhs_scalar` as `rhs-scalar-value`, `sle`, runtime-scalar splat compare
  source, accumulator passthrough, binding summary, headers, type mapping, and
  LMUL m2 typed config mirrors.
- Good: `widening_macc_add` gets canonical facts for `lhs,rhs,acc,out,n`,
  `tcrv_rvv.widening_macc`, `i16mf2` lhs/rhs sources, `i32m1` accumulator and
  result vectors, `signed-i16mf2xi16mf2-plus-i32m1-to-i32m1`,
  `separate-i32-vector-accumulator-input`,
  `store-widening-multiply-accumulate-result-to-output-buffer`, contraction
  route-family plan, route operand binding summary, header declarations, C
  type mapping, and explicit `provider_supported_mirror`; provider and target
  validation consume the same accessor surface.
- Good: standalone reduce add/min/max routes get canonical facts for
  `lhs,acc,out,n`, `tcrv_rvv.standalone_reduce`, scalar seed/result layout,
  `scalar-result-out0` runtime boundary, route operand binding summary, header
  declarations, C type mapping, and explicit `provider_supported_mirror`.
  Computed-mask variants get additional canonical facts for
  `cmp_lhs,cmp_rhs,src,acc,out,n`, `sle`, compare-produced mask source,
  inactive-lane zeroing or neutral-lane requirement, and accumulation carry
  contracts. Runtime-scalar computed-mask variants use `rhs_scalar` and the
  runtime-scalar mask producer facts rather than vector RHS facts.
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

### Computed-Mask Indexed Memory Fact Surface

For `computed_masked_indexed_gather_load_unit_store` and
`computed_masked_indexed_scatter_store_unit_load`, provider/target shared
constants must use the provider-owned surface:

```c++
struct RVVComputedMaskIndexedMemoryRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef computedMaskMemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryMaskProducerSource;
  llvm::StringRef maskTailPolicyRouteFamilyPlanID;
  llvm::StringRef maskTailPolicyOwner;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  std::int64_t indexEEW;
  llvm::StringRef offsetUnit;
  llvm::StringRef indexSource;
  llvm::StringRef indexUniqueness;
  llvm::StringRef indexedDataMemoryForm;
  llvm::StringRef indexedDestinationMemoryForm;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
};

std::optional<RVVComputedMaskIndexedMemoryRouteFacts>
getRVVComputedMaskIndexedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);
```

Contracts:

- Gather and scatter both use runtime ABI order
  `cmp_lhs,cmp_rhs,src,index,dst,n` and exported header/prototype binding for
  every ABI parameter.
- Gather uses typed compute op `tcrv_rvv.masked_indexed_load`, memory form
  `computed-mask-indexed-gather-load-unit-store`, indexed data memory form
  `masked-indexed-load`, and no scatter uniqueness fact.
- Scatter uses typed compute op `tcrv_rvv.masked_indexed_store`, memory form
  `computed-mask-unit-load-indexed-scatter-store`, indexed destination memory
  form `masked-indexed-store`, and index uniqueness `unique`.
- Both routes require SEW/LMUL/policy `32/m1/agnostic/agnostic`, compare
  predicate `slt`, compare-produced mask facts, mask/tail route-family plan,
  inactive-lane contract, passthrough/no-write layout, indexed memory layout,
  index source `runtime_abi:index`, index EEW `32`, offset unit `element`,
  provider-supported mirror, target leaf profile, header declarations, C type
  summary, operand binding plan, and exact operand binding summary from the
  accessor.
- Common EmitC/export may carry these provider-built strings and metadata
  mirrors, but must not infer any mask/index/load/store fact from the route id,
  artifact name, descriptor, C ABI spelling, or candidate metadata.

Validation and error behavior:

- Missing accessor result for either computed-mask indexed operation -> fail
  before target artifact export.
- Gather facts applied to scatter, or scatter facts applied to gather -> fail
  on memory form, typed compute op, indexed data/destination form, uniqueness,
  target profile, provider mirror, or binding summary before bundle acceptance.
- Missing or stale mask facts, index facts, inactive-lane contract, route-family
  plan, header/type summary, target profile, provider mirror, runtime ABI order,
  or binding summary -> fail before target artifact acceptance.
- Computed-mask indexed routes must reject stale plain base-memory route-family
  facts before target artifact acceptance. A provider description carrying
  `baseMemoryMovementRouteFamilyPlanID` or candidate metadata carrying
  `tcrv_rvv.base_memory_movement_route_family_plan` is an unmasked indexed
  memory fallback residue, not support evidence.
- Candidate metadata mirrors must match the provider-built route description
  exactly. Stale candidate mirrors for typed compute op, index source/EEW,
  offset unit, uniqueness, header/type summary, provider mirror, target
  profile, or operand binding summary fail closed.
- Accidental fallback to unmasked indexed memory, unit-stride computed-mask
  memory, strided computed-mask memory, descriptor/direct-C/source-export, or
  legacy `i32m1` route authority is invalid.

Tests required:

- C++ target artifact tests must mutate provider route descriptions for stale
  gather/scatter typed compute, mask/index facts, binding facts, header/type
  facts, target profile, provider mirror, and gather/scatter cross-contamination.
- C++ target artifact tests must mutate candidate metadata mirrors for the same
  fields and prove stale metadata cannot be accepted.
- C++ target artifact tests must prove stale plain base-memory provider facts
  and `tcrv_rvv.base_memory_movement_route_family_plan` candidate mirrors fail
  closed on computed-mask indexed routes.
- Generated-bundle dry-run FileCheck tests must keep explicit and pre-realized
  gather/scatter coverage and expose representative `typed_compute_op`,
  memory form, binding summary, mask/index facts, provider mirror, and target
  profile in evidence JSON.
- Runtime `ssh rvv` evidence is required only when the task claims new runtime,
  correctness, or performance behavior. Pure validation tightening may reuse
  prior runtime evidence and must state that no generated runtime semantics
  changed.

### Plain Segment2 Memory Fact Surface

#### 1. Scope / Trigger

For `segment2_interleave_unit_load` and
`segment2_deinterleave_unit_store`, provider/target shared constants must use
a provider-owned fact surface. Target artifact validation consumes that surface
after rebuilding the provider route; it must not reconstruct segment2 lane,
direction, ABI, header/type, target-profile, provider-support, or route-family
facts from route ids, artifact names, fixture names, candidate metadata
mirrors, descriptors, common EmitC, scripts, or exact RVV intrinsic spellings.

#### 2. Signatures

The durable provider-owned surface is:

```c++
struct RVVPlainSegment2MemoryRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef segment2MemoryRouteFamilyPlanID;
  llvm::StringRef segment2Direction;
  bool usesDeinterleaveLoad;
  bool usesInterleaveStore;
  llvm::StringRef segmentMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  std::int64_t segmentCount;
  llvm::StringRef segmentTupleCType;
  llvm::StringRef segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic;
  llvm::StringRef field0Role;
  llvm::StringRef field1Role;
  llvm::StringRef field0Name;
  llvm::StringRef field1Name;
  llvm::StringRef field0SourceMemoryForm;
  llvm::StringRef field1SourceMemoryForm;
  llvm::StringRef field0DestinationMemoryForm;
  llvm::StringRef field1DestinationMemoryForm;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
};

std::optional<RVVPlainSegment2MemoryRouteFacts>
getRVVPlainSegment2MemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);
```

#### 3. Contracts

- The accessor is implemented in the RVV plugin/provider layer and derives
  facts from the same typed body/config/runtime authority used to construct
  the route.
- Deinterleave uses runtime ABI order `src,out0,out1,n`, typed compute op
  `tcrv_rvv.move`, memory form `segment2-load-unit-store`, interleaved
  segment2 source memory, unit-stride field destinations, segment-load,
  field-extract, field-store, and no tuple-create/segment-store facts.
- Interleave uses runtime ABI order `src0,src1,dst,n`, typed compute op
  `tcrv_rvv.segment2_store`, memory form `unit-load-segment2-store`,
  unit-stride field source loads, an interleaved segment2 destination,
  tuple-create, segment-store, and no segment-load/field-extract facts.
- Both routes require SEW/LMUL/policy `32/m1/agnostic/agnostic`, segment count
  `2`, segment tuple C type, field roles/names, source/destination memory
  forms, target leaf profile, provider-supported mirror, required headers, C
  type summary, runtime ABI parameters, route operand binding plan, and exact
  route operand binding summary.
- Common EmitC/export may carry the provider-built payload and metadata mirrors
  unchanged. It must not infer segment2 direction, lane roles, ABI order,
  headers, C types, or route support.

#### 4. Validation & Error Matrix

- Missing accessor result for either supported plain segment2 operation -> fail
  before target artifact export.
- Deinterleave facts applied to interleave, or interleave facts applied to
  deinterleave -> fail on memory form, typed compute op, direction booleans,
  runtime ABI order, field roles, source/destination memory forms,
  segment-load/store/tuple/extract facts, target profile, provider mirror, or
  binding summary.
- Missing or stale segment lane facts, typed compute facts, route-family plan,
  runtime control plan, header/type summary, target profile, provider mirror,
  runtime ABI order/parameters, or binding summary -> fail before target
  artifact acceptance.
- Candidate metadata mirrors must match the provider-owned facts exactly.
  Stale mirrors for lane, field, segment, ABI, header/type, provider, target
  profile, runtime control, route-family, or binding facts fail closed.
- Plain segment2 routes must reject computed-mask memory route-family plan,
  computed-mask producer, mask role/source/form, compare predicate, masked
  layout, scalar/unit/strided/indexed fallback, descriptor/direct-C/source-
  export/source-front-door residue, and legacy `i32m1` route authority.

#### 5. Good/Base/Bad Cases

- Good: typed plain segment2 interleave body -> plain segment2 family facts ->
  provider-built `TCRVEmitCLowerableRoute` -> target validator consumes the
  same fact surface -> generated-bundle evidence mirrors those facts.
- Good: deinterleave accepts one segment2 load, two field extracts, and two
  unit field stores, and rejects tuple-create/segment-store residue.
- Good: interleave accepts two unit field loads, tuple create, and one
  segment2 store, and rejects segment-load/field-extract residue.
- Base: computed-mask segment2 load/store/update consume their own
  computed-mask segment2 fact surface and must not consume plain facts.
- Bad: target validation accepts an artifact because the route id or artifact
  filename says `segment2_interleave_unit_load`.
- Bad: target validation duplicates ABI/order/header/type/field constants that
  disagree with provider facts but still accepts candidate metadata.

#### 6. Tests Required

- C++ target artifact tests must mutate provider route descriptions for stale
  interleave/deinterleave cross-contamination, segment lane facts, typed
  compute facts, source/destination facts, route-family plan, binding summary,
  runtime ABI parameters, header/type facts, target profile, provider mirror,
  and computed-mask residue.
- C++ target artifact tests must mutate candidate metadata mirrors for the same
  fields and prove mirror-only metadata cannot become route authority.
- Generated-bundle dry-run FileCheck tests must keep explicit and pre-realized
  interleave/deinterleave coverage and expose representative operation, memory
  form, binding summary, segment facts, provider mirror, target profile,
  header/type facts, and no descriptor/direct-C/source-export/source-front-door
  residue.
- Runtime `ssh rvv` evidence is required only when the task changes generated
  runtime semantics or claims new runtime/correctness/performance behavior.
  Pure validation tightening may reuse prior runtime evidence and must state
  that no runtime semantics changed.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  switch operation name or route id
  -> rebuild ABI/header/type/segment constants locally
  -> accept candidate mirrors when the strings look plausible
```

Correct:

```text
typed tcrv_rvv body/config/runtime facts
  -> RVV provider fact accessor
  -> provider plan and TCRVEmitCLowerableRoute
  -> target validator consumes the same fact surface
  -> artifact/evidence mirrors match exactly
```

### Computed-Mask Segment2 Memory Fact Surface

#### 1. Scope / Trigger

For `computed_masked_segment2_load_unit_store`,
`computed_masked_segment2_store_unit_load`, and
`computed_masked_segment2_update_unit_load`, provider/target shared constants
must use a provider-owned fact surface. Target artifact validation consumes
that surface after rebuilding the provider route; it must not reconstruct
segment2, compare/mask, field, update, header/type, ABI, target-profile, or
provider-support facts from route ids, artifact names, fixture names,
descriptors, scripts, common EmitC, candidate metadata mirrors, or exact RVV
intrinsic spellings.

#### 2. Signatures

The durable provider-owned surface is:

```c++
struct RVVComputedMaskSegment2MemoryRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef routeOperandBindingPlanID;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef computedMaskMemoryRouteFamilyPlanID;
  llvm::StringRef computedMaskMemoryMaskProducerSource;
  llvm::StringRef maskTailPolicyRouteFamilyPlanID;
  llvm::StringRef maskTailPolicyOwner;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef segmentMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  std::int64_t segmentCount;
  llvm::StringRef segmentTupleCType;
  llvm::StringRef segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic;
  llvm::StringRef segment2UpdateArithmeticKind;
  llvm::StringRef segment2UpdateArithmeticIntrinsic;
  llvm::StringRef field0Role;
  llvm::StringRef field1Role;
  llvm::StringRef field0Name;
  llvm::StringRef field1Name;
  llvm::StringRef field0SourceMemoryForm;
  llvm::StringRef field1SourceMemoryForm;
  llvm::StringRef field0DestinationMemoryForm;
  llvm::StringRef field1DestinationMemoryForm;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
};

std::optional<RVVComputedMaskSegment2MemoryRouteFacts>
getRVVComputedMaskSegment2MemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);
```

#### 3. Contracts

- The accessor is implemented in the RVV plugin/provider layer and derives
  facts from the same typed body/config/runtime authority used to construct
  the route.
- Load uses runtime ABI order `cmp_lhs,cmp_rhs,src,out0,out1,n`, typed op
  `tcrv_rvv.masked_segment2_load`, interleaved segment2 source memory, unit
  destination field stores, a masked segment-load callee, tuple-create
  passthrough, and field-extract facts.
- Store and update use runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`.
  Store uses typed op `tcrv_rvv.masked_segment2_store`; update uses typed op
  `tcrv_rvv.binary` plus update arithmetic kind `add` and its provider-derived
  arithmetic callee.
- Store/update carry masked segment-store and tuple-create facts and must not
  carry a segment-load or field-extract fact. Load carries segment-load,
  tuple-create passthrough, and field-extract facts and must not carry a
  segment-store mirror.
- All three routes require SEW/LMUL/policy `32/m1/agnostic/agnostic`,
  compare predicate `slt`, compare-produced mask role/source/form, computed
  mask memory route-family plan, mask/tail policy plan and owner, inactive
  lane contract, segment count `2`, segment tuple C type, field roles/names,
  source/destination memory forms, target leaf profile, provider-supported
  mirror, required headers, C type summary, runtime ABI parameters, route
  operand binding plan, and exact route operand binding summary.
- Common EmitC/export may carry the provider-built payload and metadata mirrors
  unchanged. It must not infer segment2 or computed-mask semantics.

#### 4. Validation & Error Matrix

- Missing accessor result for any supported computed-mask segment2 operation
  -> fail before target artifact export.
- Load facts applied to store/update, or store/update facts applied to load ->
  fail on memory form, typed compute op, runtime ABI order, field roles,
  segment load/store/tuple/extract facts, target profile, provider mirror, or
  binding summary.
- Missing or stale mask facts, compare predicate, inactive-lane contract,
  passthrough/no-write layout, segment memory layout, source/destination memory
  form, field role/name, field source/destination form, segment count, route
  family plan, runtime control plan, header/type summary, target profile,
  provider mirror, runtime ABI order/parameters, or binding summary -> fail
  before target artifact acceptance.
- Update routes with stale or missing arithmetic kind/callee, stale field-load
  source, stale arithmetic operands/result, or stale masked segment-store
  statement -> fail before target artifact acceptance.
- Candidate metadata mirrors must match the provider-built route description
  exactly. Stale mirrors for mask, field, segment, update arithmetic, ABI,
  header/type, provider, target profile, runtime control, route-family, or
  binding facts fail closed.
- Accidental fallback to scalar, unit, strided, indexed, plain segment2,
  descriptor/direct-C/source-export/source-front-door, or legacy `i32m1`
  route authority is invalid.

#### 5. Good/Base/Bad Cases

- Good: typed computed-mask segment2 update body -> computed-mask memory
  family facts -> segment2 planning owner -> provider fact accessor ->
  provider-built `TCRVEmitCLowerableRoute` -> target validator consumes the
  same fact surface -> generated-bundle evidence mirrors those facts.
- Good: computed-mask segment2 load accepts a provider-derived masked
  segment-load with old-field passthrough tuple and field stores, and rejects
  any stale store/update-only fact.
- Good: computed-mask segment2 store accepts field payload loads, tuple create,
  and masked segment-store, and rejects segment-load or field-extract residue.
- Base: plain segment2 deinterleave/interleave continue to use the plain
  segment2 target export contract and must not consume computed-mask facts.
- Bad: target validation accepts an artifact because the route id or artifact
  filename says `computed_masked_segment2_update_unit_load`.
- Bad: target validation duplicates ABI/order/header/type/field/mask constants
  that disagree with provider facts but still accepts candidate metadata.

#### 6. Tests Required

- C++ target artifact tests must mutate provider route descriptions for stale
  load/store/update cross-contamination, mask facts, segment lane facts, typed
  compute facts, inactive-lane/passthrough facts, source/destination facts,
  route-family plan, binding summary, runtime ABI parameters, header/type
  facts, target profile, provider mirror, and update arithmetic facts.
- C++ target artifact tests must mutate rebuilt route statements for stale
  setvl, compare/mask, field payload/passthrough loads, tuple create/extract,
  masked segment load/store, update arithmetic, field stores, and selected
  typed RVV source provenance.
- C++ target artifact tests must mutate candidate metadata mirrors for the
  same fields and prove mirror-only metadata cannot become route authority.
- Generated-bundle dry-run FileCheck tests must keep explicit and pre-realized
  load/store/update coverage and expose representative operation, memory form,
  binding summary, mask/segment/update facts, provider mirror, target profile,
  header/type facts, and no descriptor/direct-C/source-export/source-front-door
  residue.
- Runtime `ssh rvv` evidence is required only when the task changes generated
  runtime semantics or claims new runtime/correctness/performance behavior.
  Pure validation tightening may reuse prior runtime evidence and must state
  that no runtime semantics changed.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  switch operation name or route id
  -> rebuild ABI/header/type/mask/segment constants locally
  -> accept candidate mirrors when the strings look plausible
```

Correct:

```text
typed tcrv_rvv body/config/runtime facts
  -> RVV provider fact accessor
  -> provider plan and TCRVEmitCLowerableRoute
  -> target validator compares provider facts, rebuilt route, and mirrors
  -> artifact fails closed when any mask/segment/update/ABI/header/type fact is stale
```

### Widening Conversion Fact Surface

For selected-body widening conversion routes, provider/target shared constants
must use the provider-owned surface:

```c++
struct RVVWideningConversionRouteFacts {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  llvm::StringRef sourceElementTypeName;
  llvm::StringRef resultElementTypeName;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
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
  llvm::StringRef conversionKind;
  llvm::StringRef conversionRelation;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
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
  llvm::SmallVector<RuntimeABIParameter, 3> runtimeABIParameters;
};

std::optional<RVVWideningConversionRouteFacts>
getRVVWideningConversionRouteFacts(RVVSelectedBodyOperationKind operation);
```

The accessor is the canonical provider-owned fact surface for
`widen_i32_to_i64` and `widen_i16_to_i32`. `widen_i32_to_i64` facts must include
runtime ABI order `lhs,out,n`, source `i32/m1`, result `i64/m2`,
source/result element type names, tail/mask policy, source/destination memory
forms, conversion kind `widen_i32_to_i64`, conversion relation
`signed-i32m1-to-i64m2`, typed compute op `tcrv_rvv.widening_convert`, route
family plan, route operand binding plan/summary, required headers, C type
mapping, target leaf profile, runtime ABI parameter facts, and explicit
`provider_supported_mirror`. `widen_i16_to_i32` facts must carry the analogous
source `i16/mf2`, result `i32/m1`, conversion kind `sign_extend_widen_vf2`,
and relation `signed-i16mf2-to-i32m1` facts.

Provider route-family plan derivation may use typed body/config/runtime facts
to select the operation, but every shared constant above must be copied from
`getRVVWideningConversionRouteFacts(...)` or validated against it. Target
artifact validation must consume this same accessor and reject stale local
copies of source/result element type, source/result SEW-LMUL, tail/mask
policy, conversion kind/relation, source/destination memory form, runtime ABI
parameter facts, header/type mapping, target profile, provider mirror,
route-family plan, or operand binding summary before accepting a bundle.

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
  leaf/profile, source/result element type, source/result SEW-LMUL,
  tail/mask policy, conversion kind/relation, source/destination memory form,
  route-family plan, route operand binding plan/summary, runtime ABI parameter
  facts, and stale non-conversion route-family mirrors.
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
