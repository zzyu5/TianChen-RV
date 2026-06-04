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

## RVV Runtime AVL/VL Selected-Boundary Contract

### 1. Scope / Trigger

When an RVV route-family validation contract is consumed by target artifact
validation, runtime `n` / AVL / VL facts must be represented as a named
provider-owned selected-boundary contract before the target accepts route
payload, artifact, or metadata mirrors. This applies first to segment2 memory
routes and should be extended to other existing route families when they are
promoted.

### 2. Signatures

The concrete provider-facing shape is:

```c++
struct RVVRuntimeAVLVLSelectedBoundaryContract {
  llvm::StringRef consumerLabel;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeVLContractID;
  std::string runtimeAVLABIParameterName;
  std::string runtimeAVLASource;
  std::string runtimeABIOrder;
  std::string selectedBoundaryOpName;
  std::string selectedBodyProvenance;
  std::string vlDefOpName;
  std::string vlScopeOpName;
  std::string vlUses;
  std::string setVLIntrinsic;
  std::string vlCType;
  std::string emitCLoopKind;
  std::string emitCLoopInductionName;
  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string remainingAVLMetadata;
  std::string pointerAdvanceMetadata;
  std::string boundedSlice;
  std::string multiVL;
  RuntimeABIParameter runtimeAVLParameter;
};

std::optional<RVVRuntimeAVLVLSelectedBoundaryContract>
getRVVRuntimeAVLVLSelectedBoundaryContract(
    std::int64_t sew, llvm::StringRef lmul, llvm::StringRef tailPolicy,
    llvm::StringRef maskPolicy, llvm::StringRef configContractID,
    llvm::StringRef setVLIntrinsic, llvm::StringRef vlCType,
    llvm::StringRef runtimeABIOrder,
    llvm::ArrayRef<RuntimeABIParameter> runtimeABIParameters,
    llvm::StringRef consumerLabel);
```

Route-family validation contracts that consume runtime AVL/VL facts should
embed this contract instead of duplicating the same expected fields in target
code.

### 3. Contracts

- The provider derives the contract from selected typed `tcrv_rvv` body/config
  facts, provider route-family facts, the selected runtime ABI parameter list,
  and the RVV config/VL contract.
- The contract carries exactly one `runtime-element-count` ABI parameter and
  names it as the runtime AVL source used by `tcrv_rvv.setvl`.
- The contract covers SEW, LMUL, tail/mask policy, config contract,
  runtime-VL contract, runtime AVL source, runtime ABI order, selected
  `with_vl` boundary, selected-body provenance, `setvl` intrinsic, VL C type,
  full-chunk VL, loop remaining-AVL VL, loop induction, remaining-AVL
  metadata, pointer advancement metadata, bounded slice, and multi-VL support.
- Target artifact validation consumes the contract before route payload,
  header/type, ABI mapping, statement-plan, or candidate metadata mirror
  acceptance.
- Promoted memory-family validation contracts may retain route-local
  runtime/control copies only as target-side consistency mirrors. Target
  validation must first accept the embedded `runtimeAVLVLContract`, then compare
  any retained local runtime control plan, runtime ABI order, `setvl` callee,
  VL C type, and EmitC AVL/VL statement-name copies against that embedded
  contract. Stale or missing route-local mirrors fail closed and must not become
  acceptance authority.
- Route-family contract builders must pass provider-derived canonical runtime
  facts into the runtime AVL/VL contract. For standalone reduction and similar
  families, `configContractID`, `setVLIntrinsic`, and `vlCType` must come from
  the selected typed RVV config / provider fact surface for the selected
  SEW/LMUL, not be copied from mutable route-description mirrors.
- Common EmitC/export may carry route statements and metadata mirrors supplied
  by the provider, but must not infer runtime AVL/VL facts from route ids,
  artifact names, descriptors, C snippets, manifests, or tests.

### 4. Validation & Error Matrix

- Missing runtime AVL/VL selected-boundary contract for an accepted route ->
  fail before artifact export.
- More than one runtime-element-count ABI parameter, or none -> fail before
  artifact export.
- Runtime AVL ABI parameter name, type, role, or ownership differs from the
  provider contract -> fail before artifact export.
- Description SEW, LMUL, policy, config contract, runtime-VL contract,
  runtime AVL source, runtime ABI order, `setvl` callee, VL type, `setvl` /
  `with_vl` op names, loop induction, full-chunk VL, loop VL, remaining-AVL
  metadata, pointer advancement metadata, bounded slice, or multi-VL support
  differs from the contract -> fail before artifact export.
- Route statements use stale pre-loop AVL, stale loop remaining AVL, stale
  loop VL, stale induction-based pointer advancement, or stale selected-body
  provenance -> fail before artifact export.

### 5. Good/Base/Bad Cases

- Good: `computed_masked_segment2_update_unit_load` route facts build a
  segment2 validation contract that embeds
  `RVVRuntimeAVLVLSelectedBoundaryContract`; target validation checks runtime
  AVL source `runtime_abi:n`, runtime VL contract, `setvl` / `with_vl`, loop
  induction `offset`, full-chunk VL `full_chunk_vl`, loop VL `vl`, pointer
  advancement `offset`, and runtime ABI parameter `n` before accepting the
  route.
- Good: `standalone_reduce_add` route facts build a standalone reduction
  validation contract that embeds the runtime AVL/VL contract using canonical
  selected SEW/LMUL config facts for `configContractID`, `__riscv_vsetvl_*`,
  and `size_t` VL type. Target validation rejects stale runtime AVL source,
  runtime VL contract, `with_vl` scope, setvl callee, full-chunk VL, loop VL,
  loop induction, runtime `n` ABI role, remaining AVL, or pointer advancement
  before accepting pre-loop setvl, loop setvl, scalar seed, reduction, or
  scalar-result store statements.
- Good: computed-mask indexed memory route facts build
  `RVVComputedMaskIndexedMemoryRouteValidationContract` with an embedded
  `RVVRuntimeAVLVLSelectedBoundaryContract`; target validation checks runtime
  AVL source `runtime_abi:n`, runtime VL contract, selected `with_vl` scope,
  setvl callee/type, full-chunk VL, loop VL, loop induction, runtime `n` ABI
  role/order, remaining AVL, and pointer advancement before accepting
  computed-mask producer facts, index load/scale facts, indexed gather/scatter
  statement-plan facts, header/type facts, ABI mappings, or mirrors.
- Good: computed-mask strided memory route facts build
  `RVVComputedMaskStridedMemoryRouteValidationContract` with an embedded
  `RVVRuntimeAVLVLSelectedBoundaryContract`; target validation checks runtime
  AVL source `runtime_abi:n`, runtime VL contract, selected `with_vl` scope,
  setvl callee/type, full-chunk VL, loop VL, loop induction, runtime `n` ABI
  role/order, remaining AVL, and pointer advancement before accepting
  computed-mask producer facts, strided load/store facts, byte-stride source
  and unit facts, statement-plan facts, header/type facts, ABI mappings, or
  mirrors.
- Base: a route family not yet promoted may keep its existing runtime checks
  until a bounded task embeds the shared runtime AVL/VL contract.
- Bad: target validation accepts a route because generated statements happen
  to compile while runtime AVL source, runtime-VL contract, or pointer
  advancement metadata is stale.
- Bad: a route-family contract builder passes `description.setVLIntrinsic` or
  `description.vlCType` into the embedded runtime contract. A stale route
  description can then bypass the shared runtime contract and fail only in a
  later statement-plan check, weakening the selected-boundary consumer order.

### 6. Tests Required

- C++ target artifact tests must include at least one positive consumer proving
  the route-family validation contract embeds the runtime AVL/VL contract and
  matches the rebuilt route description.
- C++ negative tests must mutate runtime AVL source, runtime-VL contract,
  selected `with_vl` scope, full-chunk VL, loop remaining-AVL VL, pointer
  advancement metadata, or runtime ABI parameter facts and assert fail-closed
  diagnostics before route acceptance.
- For standalone reduction consumers, C++ negative tests must also mutate the
  setvl callee, VL C type, loop induction, remaining AVL metadata, and runtime
  `n` ABI role so the shared runtime contract, not a later route statement
  fallback, owns the first rejection.
- Promoted computed-mask indexed memory consumers must have positive target
  coverage proving the embedded runtime contract is populated for both gather
  and scatter variants, and negative target coverage for stale runtime AVL
  source, runtime VL contract, selected `with_vl` scope, setvl callee, loop VL
  or loop induction, and runtime `n` ABI role before indexed memory-form or
  mirror checks can accept the artifact.
- Promoted computed-mask strided memory consumers must have positive target
  coverage proving the embedded runtime contract is populated for both strided
  store and strided-load/unit-store variants, and negative target coverage for
  stale runtime AVL source, runtime VL contract, selected `with_vl` scope,
  setvl callee, loop VL or loop induction, runtime `n` ABI role, and pointer
  advancement metadata before strided memory-form, byte-stride, statement-plan,
  or mirror checks can accept the artifact.
- Focused lit/generated-bundle dry-run tests for the promoted route family
  must remain green.
- Real `ssh rvv` is required only when emitted C/C++, runtime ABI order,
  runtime counts, pointer advancement, or correctness behavior changes.

### 7. Wrong vs Correct

Wrong:

```text
target validator:
  route id looks like computed_masked_segment2_update_unit_load
  -> reconstructs runtime n/VL from route metadata and C snippets
  -> accepts stale runtime AVL source or pointer advancement
```

Correct:

```text
typed selected tcrv_rvv body/config/runtime facts
  -> RVV provider route-family facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> route-family validation contract
  -> target validator consumes runtime contract before artifact acceptance
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

### MAcc Metadata Mirror Contract

#### 1. Scope / Trigger

When a selected RVV MAcc route emits target artifact metadata, candidate
metadata validation must consume a provider-owned normalized MAcc mirror
contract. This applies to the existing MAcc route families:

```text
macc_add
scalar_broadcast_macc_add
computed_masked_macc_add
runtime_scalar_cmp_masked_macc_add
widening_macc_add
```

The contract is used after the route description has been rebuilt from the
provider-built `TCRVEmitCLowerableRoute`. It is not a route-construction input
and is not artifact metadata authority.

#### 2. Signatures

The provider-owned API shape is:

```c++
struct RVVMAccRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVMAccRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVMAccRouteMetadataMirrorContract, 40> mirrors;
  llvm::SmallVector<llvm::StringRef, 24> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVMAccRouteMetadataMirrorContractSet>
getRVVMAccRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

Computed-mask and runtime-scalar computed-mask MAcc contracts must use the
actual selected `sew` and `lmul` from the rebuilt route description when
obtaining provider facts. A runtime-scalar LMUL m2 route must not be validated
through default LMUL m1 facts.

#### 3. Contracts

- The RVV provider builds the mirror contract from the same MAcc route facts
  and rebuilt route description fields used by provider route validation.
- Target artifact validation may iterate contract entries and stale keys, but
  must not rebuild MAcc mirror truth from route names, artifact metadata,
  fixture names, descriptors, C strings, scripts, or intrinsic spellings.
- The contract covers route operand binding plan/summary, provider support
  mirror, target leaf profile, typed compute op, config/element/SEW/LMUL,
  memory form, runtime-control plan, runtime ABI order, header/type summary,
  MAcc arithmetic kind, accumulator/result layout, and family-specific
  plain/scalar-broadcast/computed-mask/widening facts.
- Stale mirror keys are provider-owned rejection lists. They are used to fail
  closed on cross-family facts such as plain vs scalar-broadcast MAcc,
  computed-mask vs non-mask MAcc, runtime-scalar vs vector computed-mask MAcc,
  and widening vs non-widening MAcc.
- Rebuilt route payload validation remains separate: headers/types, runtime
  ABI mappings, statement plans, accumulator/passthrough behavior, and
  intrinsic/type payloads are still checked against provider facts before
  artifact acceptance.

#### 4. Validation & Error Matrix

- Missing MAcc contract for an accepted MAcc route -> fail before candidate
  artifact acceptance.
- Candidate mirror value differs from provider contract entry -> fail before
  artifact acceptance.
- Candidate carries a key in the provider-owned stale-key list -> fail before
  artifact acceptance.
- Runtime-scalar computed-mask MAcc selected LMUL is unsupported by provider
  facts -> fail rather than falling back to default LMUL m1 facts.
- Target-local constants disagree with provider MAcc fact accessors -> remove
  the target-local constants and validate against the provider contract.

#### 5. Good/Base/Bad Cases

- Good: `runtime_scalar_cmp_masked_macc_add` with LMUL m2 builds a mirror
  contract from parameterized runtime-scalar MAcc provider facts, including
  LMUL m2 config/type/intrinsic mirrors and the runtime-scalar binding summary.
- Good: `widening_macc_add` validates source/result SEW/LMUL, widening MAcc
  relation, accumulator/result layouts, route operand binding, target profile,
  and stale non-widening MAcc mirrors through the provider contract.
- Base: MAcc provider payload and statement-plan validators remain separate
  checks because candidate metadata mirrors do not prove executable behavior.
- Bad: target validation accepts `computed_masked_macc_add` because candidate
  metadata says `provider_supported_mirror` while the provider contract is
  missing or mismatched.
- Bad: target validation accepts LMUL m2 runtime-scalar MAcc candidate metadata
  by comparing against default LMUL m1 computed-mask facts.

#### 6. Tests Required

- C++ target artifact tests must cover positive provider-contract consumption
  for plain, scalar-broadcast, computed-mask, runtime-scalar computed-mask,
  LMUL-specific runtime-scalar computed-mask, and widening MAcc candidates.
- C++ target artifact tests must mutate candidate mirrors for runtime ABI
  order, typed config, typed compute op, predicate/mask/source/passthrough
  facts, widening source/result facts, header/type mapping, provider support,
  route operand binding, and stale cross-family keys.
- Focused lit/FileCheck or generated-bundle dry-run tests must continue to
  expose representative MAcc metadata mirrors.
- Runtime `ssh rvv` is required only when route emission, generated C/C++,
  runtime ABI, accumulator/passthrough behavior, correctness, or performance
  claims change.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  route id looks like runtime_scalar_cmp_masked_macc_add
  -> compare candidate mirrors against default m1 constants
```

Correct:

```text
provider-built route description carries operation + sew + lmul
  -> getRVVMAccRouteMetadataMirrorContract(description)
  -> parameterized provider facts build mirror/stale-key contract
  -> target validator iterates the contract
```

### MAcc Route Validation Contract

#### 1. Scope / Trigger

When a selected RVV MAcc route is rebuilt as a
`TCRVEmitCLowerableRoute`, target artifact validation must consume a
provider-owned route validation contract for executable route payload facts.
This contract is separate from candidate metadata mirrors and applies to the
existing MAcc route families:

```text
macc_add
scalar_broadcast_macc_add
computed_masked_macc_add
runtime_scalar_cmp_masked_macc_add
widening_macc_add
```

The trigger is any target validator that checks MAcc route payload,
header/type/intrinsic/profile, runtime ABI mapping, accumulator/passthrough,
mask/tail, or statement-plan shape. Those checks must not reconstruct MAcc
truth from target-local constants, route names, artifact metadata, fixture
names, descriptors, C strings, scripts, or intrinsic spellings.

#### 2. Signatures

The provider-owned API shape is:

```c++
enum class RVVMAccRouteValidationKind {
  Plain,
  ScalarBroadcast,
  ComputedMask,
  RuntimeScalarComputedMask,
  Widening,
};

struct RVVMAccRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVMAccRouteValidationContract {
  RVVMAccRouteValidationKind kind;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string arithmeticKind;

  // Family-specific plain/scalar/computed-mask/widening fields.
  std::string plainMAccRouteFamilyPlanID;
  std::string scalarBroadcastMAccRouteFamilyPlanID;
  std::string accumulationRouteFamilyPlanID;
  std::string contractionRouteFamilyPlanID;
  std::string maccAccumulatorLayout;
  std::string maccResultLayout;
  std::string comparePredicateKind;
  std::string accumulationMaskProducerSource;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::int64_t sourceSEW;
  std::string sourceLMUL;
  std::int64_t resultSEW;
  std::string resultLMUL;
  std::string wideningMAccRelation;

  // Rebuilt route payload expectations.
  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string sourceVectorLoadIntrinsic;
  std::string rhsBroadcastIntrinsic;
  std::string compareIntrinsic;
  std::string maskedMergeIntrinsic;
  std::string intrinsic;
  std::string storeIntrinsic;

  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVMAccRouteTypeMappingContract, 4> typeMappings;
};

std::optional<RVVMAccRouteValidationContract>
getRVVMAccRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

Computed-mask and runtime-scalar computed-mask contracts must use the selected
`sew` and `lmul` from the rebuilt provider description when obtaining facts.
The route token must be derived through the RVV provider route helper for the
selected operation, not by treating target artifact metadata as authority.

#### 3. Contracts

- The RVV provider builds the validation contract from provider-owned MAcc
  route facts plus the rebuilt provider route description.
- The target validator consumes the contract to compare the rebuilt route,
  rebuilt provider description, and statement-plan shape. It must not keep
  duplicate MAcc expected-fact accessors or local fallback constants for fields
  already represented in the contract.
- The contract must include common route facts: route token, memory form,
  SEW/LMUL, tail/mask policy, config contract, runtime-control plan,
  runtime ABI order,
  target leaf profile, provider-supported mirror, required headers, C type
  summary, operand-binding plan/summary, typed compute op, arithmetic kind,
  accumulator/result layout, and runtime ABI parameters.
- The contract must embed `RVVRuntimeAVLVLSelectedBoundaryContract`. Target
  provider-fact validation must consume that embedded contract before accepting
  MAcc route payloads, headers, type mappings, ABI mappings, statement plans,
  accumulator/result layout, scalar-broadcast facts, computed-mask /
  runtime-scalar facts, widening facts, or metadata mirrors.
- The embedded runtime contract is the authority for runtime AVL source,
  runtime-VL contract, selected `with_vl` boundary/scope, selected-body
  provenance, setvl callee, VL C type, full-chunk VL, loop VL, loop induction,
  remaining AVL metadata, pointer advancement metadata, bounded-slice,
  multi-VL, and runtime `n` ABI parameter facts. MAcc statement-plan
  validation must read those runtime/control names and the runtime AVL
  parameter from the embedded contract instead of reconstructing them from
  MAcc-local fields.
- Non-widening MAcc runtime contract construction must use canonical
  provider/typed-config facts for setvl and VL type, such as the MAcc
  provider's SEW/LMUL setvl derivation and `size_t` VL type. It must not copy
  mutable route-description mirrors for setvl or VL type into the embedded
  runtime contract.
- The contract must include family-specific facts for plain, scalar-broadcast,
  computed-mask, runtime-scalar computed-mask, and widening MAcc. Empty fields
  are explicit stale-residue rejection points for non-consumer families.
- The contract must include route payload requirements for rebuilt headers,
  type mappings, ABI mappings, source provenance, and statement-plan step
  counts. Candidate metadata mirrors are not enough to prove those payloads.
- Common EmitC/export may carry the provider-built route and metadata mirrors,
  but must not choose MAcc semantics, header/type facts, ABI roles, mask/tail
  behavior, accumulator passthrough, or statement shape.

#### 4. Validation & Error Matrix

- Missing validation contract for an accepted MAcc route -> fail before target
  artifact provider-fact validation.
- Rebuilt route token differs from the provider contract -> fail before target
  artifact acceptance.
- Description memory form, SEW/LMUL, tail/mask policy, runtime-control plan,
  config contract, runtime ABI order, target leaf profile,
  provider-supported mirror,
  header/type summary, operand-binding plan/summary, typed compute op,
  arithmetic kind, or layout differs from the contract -> fail before target
  artifact acceptance.
- Missing embedded runtime AVL/VL selected-boundary contract, more than one or
  zero runtime-element-count ABI parameter, stale runtime AVL source, stale
  runtime-VL contract id, stale selected `with_vl` boundary/scope, stale
  setvl callee, stale VL C type, stale full-chunk VL, stale loop VL, stale loop
  induction, stale runtime `n` ABI role/order/ownership, stale remaining AVL
  metadata, or stale pointer advancement metadata -> fail before target
  artifact validation accepts MAcc route payloads, statement plans, or mirrors.
- Runtime ABI parameter count, order, role, C name, or C type differs from the
  contract -> fail before target artifact acceptance with a diagnostic naming
  the parameter index and provider-owned parameter.
- Route headers or type mappings are missing from the rebuilt route -> fail
  before target artifact acceptance.
- Route ABI mappings do not mirror the provider-owned runtime ABI parameters
  and value names -> fail before target artifact acceptance.
- Plain MAcc carries scalar-broadcast, computed-mask, runtime-scalar, or
  widening residue fields -> fail as stale cross-family facts.
- Scalar-broadcast MAcc carries plain, computed-mask, runtime-scalar, or
  widening residue fields -> fail as stale cross-family facts.
- Computed-mask and runtime-scalar computed-mask MAcc carry stale standalone
  reduction, scalar-carry, plain, scalar-broadcast, or widening residue fields
  -> fail as stale cross-family facts.
- Widening MAcc carries non-widening MAcc route-family facts, stale
  source/result SEW-LMUL, stale widening relation, stale source/destination
  memory form, or stale header/type facts -> fail before target artifact
  acceptance.
- Statement-plan pre-loop or loop-body step count differs from the contract,
  or required setvl/load/splat/compare/MAcc/merge/store steps are missing or
  miswired -> fail before target artifact acceptance.

#### 5. Good/Base/Bad Cases

- Good: `macc_add` consumes the provider contract for plain route family plan,
  vector RHS memory form, `lhs,rhs,acc,out,n` ABI order, headers, type mapping,
  accumulator layout, embedded runtime AVL/VL contract, and six loop-body
  statement steps.
- Good: `scalar_broadcast_macc_add` consumes the scalar-broadcast contract for
  RHS scalar ABI role, splat leaf, scalar-broadcast memory form, operand
  binding summary, and stale plain route-family rejection.
- Good: `computed_masked_macc_add` consumes the computed-mask contract for
  compare predicate, mask producer/source/form, inactive-lane passthrough,
  mask type mapping, active MAcc/merge/store statement plan, and stale
  runtime-scalar or standalone-reduction residue rejection.
- Good: `runtime_scalar_cmp_masked_macc_add` with LMUL m2 obtains a
  parameterized runtime-scalar contract and rejects unsupported LMULs by
  missing provider facts rather than falling back to LMUL m1.
- Good: `widening_macc_add` consumes the widening contract for `i16mf2`
  sources, `i32m1` accumulator/result, widening relation, contraction plan,
  source/result type mappings, and exact source/result statement plan.
- Base: candidate metadata mirror validation remains a second consumer of the
  provider metadata mirror contract; it does not replace route payload
  validation.
- Bad: target validation accepts a route because candidate metadata mirrors
  `provider_supported_mirror` while rebuilt route headers, type mappings, ABI
  mappings, or statement plan disagree with the provider contract.
- Bad: target validation uses route token, artifact name, test name,
  descriptor residue, C string, or intrinsic spelling to decide which MAcc
  facts should apply.

#### 6. Tests Required

- C++ target artifact tests must cover positive contract consumption for
  plain, scalar-broadcast, computed-mask, runtime-scalar computed-mask,
  LMUL-specific runtime-scalar computed-mask, and widening MAcc candidates.
- C++ target artifact tests must assert the embedded runtime AVL/VL
  selected-boundary contract mirrors rebuilt MAcc route facts for every
  production-active MAcc validation kind.
- C++ target artifact tests must mutate runtime AVL source, runtime-VL
  contract, selected `with_vl` scope, setvl callee, VL C type, full-chunk VL,
  loop VL, loop induction, runtime `n` ABI facts, remaining AVL metadata, and
  pointer advancement metadata to prove fail-closed diagnostics before MAcc
  route acceptance.
- C++ target artifact tests must mutate provider descriptions for stale route
  payload facts: memory form, route-family plan, runtime ABI order/parameters,
  operand-binding plan/summary, target leaf profile, provider-supported mirror,
  header/type summary, typed compute op, arithmetic kind, accumulator/result
  layout, mask/passthrough fields, widening source/result facts, and
  cross-family residue fields.
- C++ target artifact tests must mutate rebuilt route payloads where practical:
  route token, headers, type mappings, ABI mappings, statement counts, and
  statement operand wiring.
- Focused lit/FileCheck or generated-bundle dry-run tests must continue to
  expose representative MAcc route families.
- Runtime `ssh rvv` is required only when route emission, generated C/C++,
  runtime ABI order, accumulator/passthrough behavior, mask/tail behavior,
  correctness, or performance claims change.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  operation name looks like runtime_scalar_cmp_masked_macc_add
  -> local target constants choose ABI order, mask producer, and statements
  -> candidate metadata mirror says supported
  -> artifact accepted
```

Correct:

```text
selected typed tcrv_rvv MAcc body
  -> RVV provider facts and route description
  -> getRVVMAccRouteValidationContract(description)
  -> target validator consumes contract for rebuilt route payload
  -> candidate metadata mirrors are checked separately as mirrors only
```

### Widening Dot-Reduce Route Validation Contract

#### 1. Scope / Trigger

When a selected RVV widening dot-reduce route is rebuilt as a
`TCRVEmitCLowerableRoute`, target artifact validation must consume a
provider-owned route validation contract for executable route payload facts.
This contract is separate from candidate metadata mirrors and applies to the
existing widening dot-reduce route families:

```text
widening_dot_reduce_add
strided_input_widening_dot_reduce_add
computed_masked_widening_dot_reduce_add
computed_masked_strided_input_widening_dot_reduce_add
```

The trigger is any target validator that checks widening dot-reduce route
payload, header/type/intrinsic/profile, runtime ABI mapping,
source/accumulator/result layout, unit-stride versus strided-input memory
facts, computed-mask facts, mask/tail policy, or statement-plan shape. Those
checks must not reconstruct widening dot-reduce truth from target-local
constants, route names, artifact metadata, fixture names, descriptors, C
strings, scripts, or intrinsic spellings.

#### 2. Signatures

The provider-owned API shape is:

```c++
enum class RVVWideningDotReduceRouteValidationKind {
  Plain,
  StridedInput,
  ComputedMask,
  ComputedMaskStridedInput,
};

struct RVVWideningDotReduceRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVWideningDotReduceRouteValidationContract {
  RVVWideningDotReduceRouteValidationKind kind;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sourceSEW;
  std::string sourceLMUL;
  std::int64_t accumulatorSEW;
  std::string accumulatorLMUL;
  std::int64_t resultSEW;
  std::string resultLMUL;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string contractionRouteFamilyPlanID;
  std::string typedComputeOpName;

  // Widening dot-reduce family facts.
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string stridedMemoryLayout;
  std::string lhsStrideSource;
  std::string rhsStrideSource;
  std::string wideningDotProductAccumulatorLayout;
  std::string wideningDotProductResultLayout;
  std::string wideningDotProductRelation;
  std::string wideningProductIntrinsic;
  std::string maskedWideningProductIntrinsic;
  std::string scalarSeedSplatIntrinsic;
  std::string stridedLoadIntrinsic;
  std::string sourceVectorLoadIntrinsic;
  std::string compareVectorLoadIntrinsic;
  std::string reductionIntrinsic;
  std::string storeIntrinsic;
  std::string setVLIntrinsic;
  std::string compareIntrinsic;
  std::string maskedMergeIntrinsic;
  std::string reductionStoreVL;
  std::string inactiveLaneZeroingRequirement;

  // Rebuilt route payload expectations.
  std::string vlCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string resultVectorTypeName;
  std::string resultVectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  std::string resultName;
  std::string maskName;

  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  llvm::SmallVector<RuntimeABIParameter, 9> runtimeABIParameters;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVWideningDotReduceRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVWideningDotReduceRouteValidationContract>
getRVVWideningDotReduceRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

The contract must be built in the RVV provider layer from
`RVVWideningDotReduceRouteFacts` plus the rebuilt provider route description.
The route token must be derived through the RVV provider route helper for the
selected operation, not by treating target artifact metadata as authority.

#### 3. Contracts

- The RVV provider builds the validation contract from provider-owned widening
  dot-reduce route facts plus dynamic route-description payload names and
  selected runtime ABI parameters.
- The target validator consumes the contract to compare the rebuilt route,
  rebuilt provider description, and statement-plan shape. It must not keep
  duplicate widening dot-reduce expected-fact accessors or local fallback
  constants for fields already represented in the contract.
- The contract must include common route facts: route token, memory form,
  source/accumulator/result SEW-LMUL, tail/mask policy, config contract,
  runtime-control plan, runtime ABI order, target leaf profile,
  provider-supported mirror, required headers, C type summary, operand-binding
  plan/summary, typed compute op, contraction route-family plan, and runtime
  ABI parameters.
- The contract must embed `RVVRuntimeAVLVLSelectedBoundaryContract`. Target
  provider-fact validation must consume that embedded contract before
  accepting widening dot-reduce runtime ABI facts, route payloads, source/
  accumulator/result type facts, memory-form facts, contraction facts, mask
  facts, statement plans, headers/type mappings, or metadata mirrors.
- The embedded runtime contract is the authority for runtime AVL source,
  runtime-VL contract, selected `with_vl` boundary/scope, selected-body
  provenance, setvl callee, VL C type, full-chunk VL, loop VL, loop induction,
  remaining AVL metadata, pointer advancement metadata, bounded-slice,
  multi-VL, and runtime `n` ABI parameter facts. Widening dot-reduce
  statement-plan validation must read those runtime/control names from the
  embedded contract instead of reconstructing them from widening-dot-local
  fields.
- The contract must include widening dot-reduce facts: narrow source type,
  widened accumulator/result type, accumulator/result layout, widening
  relation, product/reduction/store/setvl intrinsics, reduction store VL,
  scalar seed splat, source/result vector types, and statement-plan counts.
- The contract must include family-specific fields for strided-input and
  computed-mask routes. Empty strided fields are explicit stale-residue
  rejection points for unit-stride routes, and empty mask fields are explicit
  stale-residue rejection points for non-computed-mask routes.
- Candidate metadata mirrors remain a separate consume-only check. They may
  mirror provider-derived contract fields after route construction, but they
  cannot prove rebuilt route payload, ABI mappings, or statement wiring.
- Common EmitC/export may carry the provider-built route and metadata mirrors,
  but must not choose widening dot-reduce semantics, header/type facts, ABI
  roles, mask/tail behavior, stride behavior, source/result layout, or
  statement shape.

#### 4. Validation & Error Matrix

- Missing validation contract for an accepted widening dot-reduce route -> fail
  before target artifact provider-fact validation.
- Rebuilt route token differs from the provider contract -> fail before target
  artifact acceptance.
- Description memory form, source/accumulator/result SEW-LMUL, tail/mask
  policy, config contract, runtime-control plan, runtime ABI order, target leaf
  profile, provider-supported mirror, header/type summary, operand-binding
  plan/summary, contraction family plan, typed compute op, layout, or widening
  relation differs from the contract -> fail before target artifact acceptance.
- Embedded runtime AVL/VL contract is missing, stale, or mismatched for runtime
  AVL source, runtime VL contract, selected `with_vl` scope, setvl callee, VL C
  type, full-chunk VL, loop VL, loop induction, runtime `n` ABI role/order,
  remaining AVL metadata, or pointer advancement metadata -> fail before target
  artifact acceptance.
- Runtime ABI parameter count, order, role, C name, or C type differs from the
  contract -> fail before target artifact acceptance with a diagnostic naming
  the parameter index and provider-owned parameter.
- Route headers or type mappings are missing from the rebuilt route -> fail
  before target artifact acceptance.
- Route ABI mappings do not mirror the provider-owned runtime ABI parameters
  and value names -> fail before target artifact acceptance.
- Plain/unit-stride widening dot-reduce carries strided-input residue fields
  such as strided memory layout, stride sources, source/destination memory form,
  or strided load intrinsic -> fail as stale strided facts.
- Strided-input widening dot-reduce is missing stride ABI roles, stride
  sources, source/destination memory form, strided load intrinsic, or strided
  statement wiring -> fail before target artifact acceptance.
- Non-computed-mask widening dot-reduce carries computed-mask residue fields
  such as mask role/source/form, compare predicate, inactive-lane zeroing,
  masked widening product, mask type, or masked merge -> fail as stale mask
  facts.
- Computed-mask widening dot-reduce is missing compare loads, compare
  predicate, mask result, inactive-lane zeroing, masked product, merge, mask
  type mapping, or computed-mask statement wiring -> fail before target
  artifact acceptance.
- Statement-plan pre-loop or loop-body step count differs from the contract,
  or required setvl/load/strided-load/splat/product/compare/merge/reduction/
  store steps are missing or miswired -> fail before target artifact
  acceptance.

#### 5. Good/Base/Bad Cases

- Good: `widening_dot_reduce_add` consumes the plain contract for unit-stride
  `lhs,rhs,acc,out,n` ABI order, `i16` source vector loads, `i32`
  accumulator/result vector type, product/reduction/store intrinsics, and seven
  loop-body statements.
- Good: `strided_input_widening_dot_reduce_add` consumes the strided-input
  contract for `lhs_stride`/`rhs_stride` ABI roles, strided source memory form,
  byte-stride calculation, strided load intrinsic, and stale unit-stride
  rejection.
- Good: `computed_masked_widening_dot_reduce_add` consumes the computed-mask
  contract for compare operands, mask role/source/form, compare predicate,
  inactive-lane zeroing, masked product, merge, mask type mapping, and twelve
  loop-body statements.
- Good: `computed_masked_strided_input_widening_dot_reduce_add` consumes both
  computed-mask and strided-input contract fields without target-local
  recomputation of ABI order or statement shape.
- Base: candidate metadata mirror validation remains a second consumer of the
  provider metadata mirror contract; it does not replace route payload
  validation.
- Bad: target validation accepts a route because candidate metadata mirrors
  `provider_supported_mirror` while rebuilt route headers, type mappings, ABI
  mappings, or statement plan disagree with the provider contract.
- Bad: target validation uses route token, artifact name, test name,
  descriptor residue, C string, or intrinsic spelling to decide which widening
  dot-reduce facts should apply.

#### 6. Tests Required

- C++ target artifact tests must cover positive contract consumption for plain,
  strided-input, computed-mask, and computed-mask-strided widening dot-reduce
  candidates.
- C++ target artifact tests must mutate provider descriptions for stale route
  payload facts: memory form, source/accumulator/result SEW-LMUL, route-family
  plan, runtime AVL source, runtime VL contract, selected `with_vl` scope,
  setvl callee, VL C type, full-chunk VL, loop VL, loop induction, runtime ABI
  order/parameters, remaining AVL metadata, pointer advancement metadata,
  operand-binding plan/summary, target leaf profile, provider-supported mirror,
  header/type summary, typed compute op, source/destination memory form, stride
  facts, mask facts, intrinsic facts, reduction store VL, and cross-family
  residue fields.
- C++ target artifact tests must mutate rebuilt route payloads where practical:
  route token, headers, type mappings, ABI mappings, statement counts, and
  statement operand wiring.
- Focused lit/FileCheck or generated-bundle dry-run tests must continue to
  expose representative widening dot-reduce route families.
- Runtime `ssh rvv` is required only when route emission, generated C/C++,
  runtime ABI order, source/accumulator/result layout, mask/tail behavior,
  stride behavior, correctness, or performance claims change.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  operation name looks like computed_masked_strided_input_widening_dot_reduce_add
  -> local target constants choose ABI order, mask facts, stride facts, and statements
  -> candidate metadata mirror says supported
  -> artifact accepted
```

Correct:

```text
selected typed tcrv_rvv widening dot-reduce body
  -> RVV provider facts and route description
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> getRVVWideningDotReduceRouteValidationContract(description)
  -> target validator consumes contract for rebuilt route payload
  -> candidate metadata mirrors are checked separately as mirrors only
```

### Standalone Reduction Route Validation Contract

#### 1. Scope / Trigger

When a selected RVV standalone reduction route is rebuilt as a
`TCRVEmitCLowerableRoute`, target artifact validation must consume a
provider-owned route validation contract for executable route payload facts.
This contract is separate from candidate metadata mirrors and applies to the
existing standalone reduction route families:

```text
standalone_reduce_{add,min,max}
computed_mask_standalone_reduce_{add,min,max}
runtime_scalar_computed_mask_standalone_reduce_{add,min,max}
```

The trigger is any target validator that checks standalone reduction route
payload, header/type/intrinsic/profile, runtime ABI mapping, source/vector/
scalar-result layout, reduction kind, mask/tail policy, inactive-lane neutral
policy, passthrough or initial scalar-result policy, computed-mask producer
facts, runtime-scalar RHS facts, or statement-plan shape. Those checks must not
reconstruct standalone reduction truth from target-local constants, route
names, artifact metadata, fixture names, descriptors, C strings, scripts, or
intrinsic spellings.

#### 2. Signatures

The provider-owned API shape is:

```c++
enum class RVVStandaloneReductionRouteValidationKind {
  Plain,
  ComputedMask,
  RuntimeScalarComputedMask,
};

struct RVVStandaloneReductionRouteTypeMappingContract {
  std::string sourceType;
  std::string cType;
  llvm::StringRef label;
};

struct RVVStandaloneReductionRouteValidationContract {
  RVVStandaloneReductionRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;

  std::string standaloneReductionRouteFamilyPlanID;
  std::string standaloneReductionSourceVectorTypeName;
  std::string standaloneReductionSourceVectorCType;
  std::string standaloneReductionScalarCType;
  std::string standaloneReductionScalarResultVectorTypeName;
  std::string standaloneReductionScalarResultVectorCType;
  std::string standaloneReductionScalarResultRuntimeBoundary;
  std::string reductionAccumulatorLayout;
  std::string reductionResultLayout;
  std::string reductionStoreVL;

  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string accumulationRouteFamilyPlanID;
  std::string accumulationComputeSuffix;
  std::string accumulationMaskProducerSource;
  std::string accumulationAccumulatorContract;
  std::string accumulationResultContract;
  std::string accumulationScalarCarryContract;
  std::string inactiveLaneUse;
  std::string inactiveLaneRequirement;
  std::string inactiveLaneZeroingRequirement;
  std::string inactiveNeutralLiteral;

  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string sourceSplatIntrinsic;
  std::string rhsBroadcastIntrinsic;
  std::string scalarSeedSplatIntrinsic;
  std::string reductionIntrinsic;
  std::string intrinsic;
  std::string storeIntrinsic;
  std::string compareIntrinsic;
  std::string maskedMergeIntrinsic;
  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  std::string resultName;
  std::string maskName;

  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  llvm::SmallVector<RuntimeABIParameter, 6> runtimeABIParameters;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVStandaloneReductionRouteTypeMappingContract, 5>
      typeMappings;
};

std::optional<RVVStandaloneReductionRouteValidationContract>
getRVVStandaloneReductionRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

struct RVVStandaloneReductionRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVStandaloneReductionRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVStandaloneReductionRouteMetadataMirrorContract, 40>
      mirrors;
  llvm::SmallVector<llvm::StringRef, 24> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVStandaloneReductionRouteMetadataMirrorContractSet>
getRVVStandaloneReductionRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

The validation contract must be built in the RVV provider layer from
`RVVStandaloneReductionRouteFacts` plus the rebuilt provider route
description. The metadata mirror contract must be built from the same provider
facts and route description, but it remains a mirror-only candidate validation
surface.

#### 3. Contracts

- The RVV provider builds the route validation contract from provider-owned
  standalone reduction facts plus dynamic route-description payload names,
  selected operation, selected SEW/LMUL/policy, and runtime ABI parameters.
- The target validator consumes the contract to compare the rebuilt route,
  rebuilt provider description, ABI mappings, type mappings, and statement-plan
  shape. It must not keep duplicate standalone reduction expected-fact accessors
  or local fallback constants for fields represented in the contract.
- The contract must include common route facts: route token, memory form,
  element type, SEW, LMUL, tail/mask policy, runtime-control plan, runtime ABI
  order, target leaf profile, provider-supported mirror, required headers, C
  type summary, operand-binding plan/summary, typed compute op, runtime ABI
  parameters, and EmitC setvl/loop names.
- The contract must embed `RVVRuntimeAVLVLSelectedBoundaryContract`. Target
  validation consumes that selected-boundary contract before accepting
  standalone reduction route payload, runtime ABI parameters, statement-plan
  shape, or candidate metadata mirrors. Retained runtime-control plan, runtime
  ABI order, setvl/VL type, full-chunk VL, loop VL, and loop induction fields
  are route-local runtime AVL/VL mirrors only.
- The contract must include standalone reduction facts: route-family plan,
  source vector type/C type, scalar type, scalar-result vector type/C type,
  scalar-result runtime boundary, accumulator/result layout, reduction store
  VL, load/splat/reduction/store intrinsics, result name, and pre-loop/loop
  statement counts.
- Computed-mask contracts must include compare predicate, mask role/source/
  memory form, mask type/C type, compare and merge intrinsics, inactive-lane
  requirement, inactive neutral literal, accumulation route-family plan,
  accumulation compute suffix, mask producer source, accumulator/result/scalar
  carry contracts, and ten loop-body statements.
- Runtime-scalar computed-mask contracts must include the same computed-mask
  facts plus provider-derived RHS scalar broadcast intrinsic and runtime ABI
  role/order for `cmp_lhs,rhs_scalar,src,acc,out,n`.
- Plain contracts must reject computed-mask and runtime-scalar residue fields
  as stale facts; computed-mask contracts must reject cross-family MAcc,
  widening dot, memory, compare/select, segment2, conversion, and contraction
  residue fields.
- Candidate metadata mirrors must use
  `getRVVStandaloneReductionRouteMetadataMirrorContract(description)`. They may
  mirror provider-derived contract fields, but they cannot prove rebuilt route
  payload, ABI mappings, type mappings, or statement wiring.
- Candidate metadata labels for `tcrv_rvv.runtime_control_plan` and
  `tcrv_rvv.runtime_abi_order` must say `route-local runtime AVL/VL control
  plan mirror` and `route-local runtime AVL/VL ABI order mirror`.
- Common EmitC/export may carry the provider-built route and metadata mirrors,
  but must not choose standalone reduction semantics, reduction kind, ABI roles,
  mask/tail behavior, inactive-lane policy, scalar-result layout, or statement
  shape.

#### 4. Validation & Error Matrix

- Missing validation contract for an accepted standalone reduction route ->
  fail before target artifact provider-fact validation.
- Missing or stale embedded runtime AVL/VL selected-boundary contract -> fail
  before standalone reduction route payload, statement-plan, or mirror
  acceptance.
- Missing or stale route-local runtime AVL/VL mirrors -> fail after
  selected-boundary validation with diagnostics that name
  `route-local runtime AVL/VL mirror`.
- Rebuilt route token differs from the provider contract -> fail before target
  artifact acceptance.
- Description memory form, typed compute op, SEW/LMUL, tail/mask policy,
  target leaf profile, provider-supported mirror, header/type summary,
  operand-binding plan/summary,
  route-family plan, scalar-result boundary, layout, reduction store VL, or
  intrinsic field differs from the contract -> fail before target artifact
  acceptance.
- Runtime ABI parameter count, order, role, C name, or C type differs from the
  contract -> fail before target artifact acceptance with a diagnostic naming
  the parameter index and provider-owned parameter.
- Route headers or type mappings are missing from the rebuilt route -> fail
  before target artifact acceptance.
- Route ABI mappings do not mirror the provider-owned runtime ABI parameters
  and value names -> fail before target artifact acceptance.
- Plain standalone reduction carries computed-mask fields, mask type mapping,
  compare/merge intrinsics, RHS broadcast, accumulation facts, or inactive-lane
  facts -> fail as stale computed-mask residue.
- Computed-mask standalone reduction is missing mask role/source/form, compare
  predicate, compare/merge intrinsics, source splat, inactive neutral literal,
  accumulation route-family plan, mask producer source, or accumulation
  accumulator/result/scalar-carry contract -> fail before target artifact
  acceptance.
- Runtime-scalar computed-mask standalone reduction is missing RHS scalar
  broadcast intrinsic, runtime-scalar ABI role/order, or runtime-scalar mask
  producer facts -> fail before target artifact acceptance.
- Candidate metadata is missing a provider mirror, carries a stale mirror from
  another route family, or mirrors a stale standalone reduction contract field
  -> fail before artifact acceptance.
- Statement-plan pre-loop or loop-body step count differs from the contract,
  or required setvl/load/splat/compare/merge/reduction/store steps are missing
  or miswired -> fail before target artifact acceptance.

#### 5. Good/Base/Bad Cases

- Good: `standalone_reduce_add` consumes the plain contract for `lhs,acc,out,n`
  ABI order, unit-stride standalone reduction memory form, scalar seed splat,
  scalar-result store, reduction intrinsic, and five loop-body statements.
- Good: `standalone_reduce_min` and `standalone_reduce_max` use provider facts
  for reduction kind and inactive/initial scalar-result policy instead of target
  validators branching on route names or intrinsic strings.
- Good: `computed_mask_standalone_reduce_add` consumes the computed-mask
  contract for `cmp_lhs,cmp_rhs,src,acc,out,n`, mask role/source/form, compare
  predicate, inactive neutral literal, merge, reduction, scalar-result store,
  and ten loop-body statements.
- Good: `runtime_scalar_computed_mask_standalone_reduce_max` consumes the
  runtime-scalar computed-mask contract for `rhs_scalar` ABI binding and RHS
  scalar splat without target-local runtime-scalar special cases.
- Base: candidate metadata mirror validation remains a second consumer of the
  provider metadata mirror contract; it does not replace route payload
  validation.
- Bad: target validation accepts a route because candidate metadata mirrors
  `provider_supported_mirror` while rebuilt route headers, type mappings, ABI
  mappings, or statement plan disagree with the provider validation contract.
- Bad: target validation uses route token, artifact name, test name,
  descriptor residue, C string, or intrinsic spelling to decide which
  standalone reduction facts should apply.

#### 6. Tests Required

- C++ provider/interface tests must cover representative positive validation
  contracts for plain, computed-mask, and runtime-scalar computed-mask
  standalone reductions, including ABI count, type mappings, required headers,
  metadata mirror count, and statement-plan counts.
- C++ target artifact tests must mutate provider descriptions for stale route
  payload facts: memory form, typed compute op, reduction kind, runtime
  selected-boundary facts, runtime ABI parameters, operand-binding plan/summary,
  target leaf profile,
  provider-supported mirror, header/type summary, scalar-result vector type,
  accumulator/result layout, mask producer source, inactive-lane policy, RHS
  scalar broadcast, and cross-family residue fields.
- C++ target artifact tests must mutate candidate runtime-control and runtime
  ABI metadata mirrors and require route-local runtime AVL/VL mirror
  diagnostics.
- C++ target artifact tests must mutate rebuilt route payloads where practical:
  route token, headers, type mappings, ABI mappings, statement counts, and
  statement operand wiring.
- Focused lit/FileCheck or generated-bundle dry-run tests must continue to
  expose representative standalone reduction route families.
- Runtime `ssh rvv` is required only when route emission, generated C/C++,
  runtime ABI order, standalone reduction computation, scalar-result layout,
  mask/tail behavior, inactive-lane behavior, correctness, or performance
  claims change.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  operation name looks like runtime_scalar_computed_mask_standalone_reduce_min
  -> local target constants choose ABI order, mask producer, neutral literal,
     scalar-result layout, and statements
  -> candidate metadata mirror says supported
  -> artifact accepted
```

Correct:

```text
selected typed tcrv_rvv standalone reduction body
  -> RVV provider facts and route description
  -> RVVRuntimeAVLVLSelectedBoundaryContract embedded in the route contract
  -> getRVVStandaloneReductionRouteValidationContract(description)
  -> target validator consumes contract for rebuilt route payload
  -> getRVVStandaloneReductionRouteMetadataMirrorContract(description)
  -> candidate metadata mirrors are checked separately as mirrors only
```

### Vector Reduction Runtime AVL/VL Route Boundary

#### 1. Scope / Trigger

When the existing vector RHS-load `reduce_add` route is accepted as a
target artifact, runtime `n` / AVL / VL facts must be accepted only through the
provider-built `RVVRuntimeAVLVLSelectedBoundaryContract` embedded in
`RVVVectorReductionRouteValidationContract`.

#### 2. Signatures

```c++
struct RVVVectorReductionRouteValidationContract {
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  std::string reductionAccumulatorLayout;
  std::string reductionResultLayout;
  std::string reductionStoreVL;
  std::string setVLIntrinsic;
  std::string vlCType;
  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  llvm::SmallVector<RuntimeABIParameter, 4> runtimeABIParameters;
};
```

#### 3. Contracts

- The provider builds `runtimeAVLVLContract` from selected typed
  SEW/LMUL/policy/config facts, selected runtime ABI parameters, setvl/VL type
  facts, and the vector-reduction consumer label.
- Target validation consumes `runtimeAVLVLContract` before accepting route id,
  route operand binding, runtime ABI parameters, headers, type mappings,
  vector load/reduction/store statements, or candidate metadata mirrors.
- `runtimeControlPlanID`, `runtimeABIOrder`, setvl/VL type, full-chunk VL,
  loop VL, and loop induction fields are retained only as route-local runtime
  AVL/VL mirrors checked against `runtimeAVLVLContract`.
- Candidate metadata for `tcrv_rvv.runtime_control_plan` and
  `tcrv_rvv.runtime_abi_order` is mirror-only and must use the same
  route-local runtime AVL/VL mirror labels as other promoted route families.

#### 4. Validation & Error Matrix

- Missing embedded runtime AVL/VL selected-boundary contract -> fail before
  target artifact provider-fact validation.
- Stale runtime AVL source, runtime VL contract, selected `with_vl` scope,
  setvl callee, VL C type, full-chunk VL, loop VL, loop induction, runtime
  ABI order, or runtime `n` ABI parameter -> fail before vector reduction route
  payload acceptance.
- Stale route-local runtime mirror fields -> fail after selected-boundary
  validation with `route-local runtime AVL/VL mirror` diagnostics.
- Candidate metadata carries stale runtime-control plan or runtime ABI order
  mirrors -> fail before artifact acceptance with route-local mirror labels.

#### 5. Good/Base/Bad Cases

- Good: selected typed vector `reduce_add` body builds the vector reduction
  contract, embeds the runtime selected-boundary contract, and target
  validation checks route-local runtime mirrors only after that contract.
- Base: candidate metadata continues to mirror provider facts after route
  validation; it never establishes runtime `n` / AVL / VL authority.
- Bad: target validation accepts a vector reduction route because
  `tcrv_rvv.runtime_abi_order` metadata matches while the embedded runtime
  selected-boundary contract is missing or stale.

#### 6. Tests Required

- C++ target artifact tests must assert the vector reduction contract embeds a
  runtime selected-boundary contract matching the rebuilt route description.
- C++ target artifact tests must mutate runtime selected-boundary facts,
  runtime ABI parameters, route-local runtime mirrors, and candidate runtime
  metadata mirrors.
- Focused lit/FileCheck tests should continue to expose the generated vector
  reduction artifact surface when route emission changes.

#### 7. Wrong vs Correct

Wrong:

```text
candidate metadata tcrv_rvv.runtime_abi_order matches
  -> vector reduction artifact accepted
```

Correct:

```text
selected typed vector reduction body
  -> provider route validation contract with runtimeAVLVLContract
  -> target validation accepts runtime AVL/VL selected-boundary first
  -> route-local runtime/control fields and metadata are mirrors only
```

### Computed-Mask Strided Memory Fact Surface

For `computed_masked_strided_store` and
`computed_masked_strided_load_unit_store`, provider/target shared constants
must use the provider-owned surface:

```c++
struct RVVComputedMaskStridedMemoryRouteFacts {
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
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef maskedLoadIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
  llvm::StringRef compareIntrinsic;
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
  llvm::StringRef maskedMemoryLayout;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef sourceStrideSource;
  llvm::StringRef destinationStrideSource;
  llvm::StringRef sourceStrideCType;
  llvm::StringRef destinationStrideCType;
  llvm::StringRef sourceStrideUnit;
  llvm::StringRef destinationStrideUnit;
  std::string routeOperandBindingSummary;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
};

std::optional<RVVComputedMaskStridedMemoryRouteFacts>
getRVVComputedMaskStridedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation);
```

Contracts:

- Strided load/unit store uses runtime ABI order `cmp_lhs,cmp_rhs,src,dst,n,
  src_stride_bytes`, typed compute op `tcrv_rvv.masked_load`, memory form
  `computed-mask-strided-load-unit-store`, source memory form
  `masked-strided-load`, destination memory form `unit-stride-store`, source
  stride source `runtime_abi:src_stride_bytes`, source stride C type matching
  the runtime ABI parameter, and source stride unit `byte`. It carries
  `maskedLoadIntrinsic` and ordinary `storeIntrinsic`, and must carry empty
  `stridedStoreIntrinsic` and destination-stride facts.
- Strided store uses runtime ABI order `cmp_lhs,cmp_rhs,src,dst,n,
  dst_stride_bytes`, typed compute op `tcrv_rvv.masked_store`, memory form
  `computed-mask-unit-load-strided-store`, source memory form
  `unit-stride-load`, destination memory form `masked-strided-store`,
  destination stride source `runtime_abi:dst_stride_bytes`, destination stride
  C type matching the runtime ABI parameter, and destination stride unit
  `byte`. It carries `stridedStoreIntrinsic`, and must carry empty
  `maskedLoadIntrinsic`, ordinary `storeIntrinsic`, and source-stride facts.
- Both routes require SEW/LMUL/policy `32/m1/agnostic/agnostic`, runtime
  control, compare predicate, compare-produced mask facts, mask/tail
  route-family plan, inactive-lane contract, masked layout, strided layout,
  provider-supported mirror, target leaf profile, header declarations, C type
  summary, operand binding plan, and exact operand binding summary from the
  accessor.
- Operand binding summaries must expose every runtime ABI parameter as an ABI
  binding and must mark the active stride operand as a byte stride. Target
  artifact validation consumes these provider facts; candidate metadata may
  only mirror them after the provider route is rebuilt.
- Common EmitC/export may carry these provider-built strings and metadata
  mirrors, but must not infer mask, stride, load/store direction, header/type,
  or support facts from route ids, artifact names, descriptor metadata, C ABI
  spelling, scripts, or candidate metadata.

Validation and error behavior:

- Missing accessor result for either computed-mask strided operation -> fail
  before target artifact export.
- Load facts applied to store, or store facts applied to load -> fail on
  memory form, typed compute op, source/destination stride source, active
  intrinsic leaf, target profile, provider mirror, or binding summary before
  bundle acceptance.
- Missing or stale mask facts, strided facts, inactive-lane contract,
  route-family plan, header/type summary, target profile, provider mirror,
  runtime ABI order, or binding summary -> fail before target artifact
  acceptance.
- Missing or stale provider-derived VL/vector/mask C type facts, setvl, vector
  load, compare leaf, masked load leaf, ordinary unit-store leaf, or strided
  store leaf -> fail before target artifact acceptance.
- A load route carrying destination-stride or strided-store facts, or a store
  route carrying source-stride or masked-load/unit-store facts, is stale
  cross-route residue and must fail before target artifact acceptance.
- A stride source whose runtime ABI role, C type, source string, operand
  binding use, or unit differs from the accessor facts must fail before target
  artifact acceptance. Current strided memory routes use byte strides, not
  element strides.
- Computed-mask strided routes must reject stale indexed, segment2, unit-only,
  scalar-splat, arithmetic, descriptor/direct-C/source-export, or legacy
  `i32m1` route authority before accepting a target artifact.

Tests required:

- C++ target artifact tests must mutate provider route descriptions for stale
  load/store typed compute, stride source/unit/C type, mask facts, binding
  facts, header/type facts, target profile, provider mirror, masked-load leaf,
  unit-store leaf, strided-store leaf, and load/store cross-contamination.
- C++ target artifact tests must mutate candidate metadata mirrors for the same
  fields and prove stale metadata cannot be accepted.
- Generated-bundle dry-run FileCheck tests must keep explicit and
  pre-realized coverage for computed-mask strided load/unit-store and
  pre-realized coverage for computed-mask strided store, exposing
  `typed_compute_op`, memory form, binding summary, mask/stride facts,
  provider mirror, and target profile in evidence JSON.
- Runtime `ssh rvv` evidence is required only when the task claims new runtime,
  correctness, ABI order, stride-unit, inactive-lane, passthrough, destination
  preservation, or performance behavior. Pure validation tightening may reuse
  prior runtime evidence and must state that no generated runtime semantics
  changed.

### Computed-Mask Strided Memory Route Validation Contract

#### 1. Scope / Trigger

When target artifact validation accepts rebuilt provider payloads for
`computed_masked_strided_store` or
`computed_masked_strided_load_unit_store`, it must consume a provider-owned
route validation contract after rebuilding the RVV provider route. This
contract sits above candidate metadata mirrors: mirrors are checked only after
the rebuilt route description, route payload, ABI mappings, headers, types,
and statement-plan shape match the provider contract.

#### 2. Signatures

The durable provider-owned surface is:

```c++
enum class RVVComputedMaskStridedMemoryRouteValidationKind {
  StridedStore,
  StridedLoadUnitStore,
};

struct RVVComputedMaskStridedMemoryRouteValidationContract {
  RVVComputedMaskStridedMemoryRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string computedMaskMemoryRouteFamilyPlanID;
  std::string computedMaskMemoryMaskProducerSource;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::string maskedMemoryLayout;
  std::string stridedMemoryLayout;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string sourceStrideSource;
  std::string destinationStrideSource;
  std::string sourceStrideCType;
  std::string destinationStrideCType;
  std::string sourceStrideUnit;
  std::string destinationStrideUnit;
  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string maskedLoadIntrinsic;
  std::string storeIntrinsic;
  std::string stridedStoreIntrinsic;
  std::string compareIntrinsic;
  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
  llvm::SmallVector<RuntimeABIParameterRole, 8> runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVComputedMaskStridedMemoryRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVComputedMaskStridedMemoryRouteValidationContract>
getRVVComputedMaskStridedMemoryRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVComputedMaskStridedMemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

#### 3. Contracts

- The RVV provider builds the validation contract from
  `RVVComputedMaskStridedMemoryRouteFacts` plus rebuilt route description
  payload details such as result/mask names, EmitC loop/VL names, config
  contract id, and route id.
- Target artifact validation must require this contract before accepting
  operation, memory form, dtype/config, mask/tail policy, compare-produced mask
  facts, stride source/unit/C type, runtime ABI order and parameters, route
  operand binding summary, headers, C type mappings, intrinsic leaves, target
  profile, provider support mirror, and statement-plan counts.
- Store contracts must carry destination byte-stride facts and reject
  source-stride or masked-load residue. Load/unit-store contracts must carry
  source byte-stride facts and reject destination-stride or strided-store
  residue.
- Candidate metadata mirror validation remains separate and must use
  `getRVVComputedMaskStridedMemoryRouteMetadataMirrorContract(...)` after route
  payload validation succeeds.

#### 4. Validation & Error Matrix

- Missing validation contract for either computed-mask strided operation ->
  fail before target artifact acceptance.
- Operation, memory form, route family plan, mask/tail facts, target profile,
  provider mirror, header/type summary, binding plan, or binding summary
  differs from the contract -> fail before candidate mirrors are considered.
- Runtime ABI count, order, C name, C type, role, or active stride byte role
  differs from the contract -> fail before artifact acceptance.
- Rebuilt route headers, type mappings, ABI mappings, or statement-plan counts
  disagree with the contract -> fail before artifact acceptance.
- Candidate metadata matches while provider payload mismatches -> fail;
  mirrors cannot override provider route validation.
- Stale indexed, segment2, base-memory, unit-only, scalar-splat, arithmetic,
  descriptor/direct-C/source-export, or legacy i32 route-authority residue on a
  computed-mask strided route -> fail closed.

#### 5. Good/Base/Bad Cases

- Good: `computed_masked_strided_store` facts -> route validation contract ->
  target checks destination byte stride, masked store leaf, provider binding
  summary, route headers/types, and statement-plan shape before mirrors.
- Good: `computed_masked_strided_load_unit_store` facts -> route validation
  contract -> target checks source byte stride, masked load leaf, passthrough
  load/store layout, route headers/types, and statement-plan shape before
  mirrors.
- Base: computed-mask indexed, segment2, base-memory, and compare/select
  routes consume their own validation contracts or fact surfaces.
- Bad: target validation accepts a route because the route id, artifact name,
  candidate metadata, or C intrinsic spelling looks like a computed-mask
  strided route while the provider contract is missing or stale.

#### 6. Tests Required

- C++ target artifact tests must cover positive contract access for both
  computed-mask strided store and load/unit-store.
- C++ target artifact tests must mutate provider route descriptions for stale
  provider mirror, target profile, header/type facts, runtime ABI roles, mask
  facts, active stride source/unit/C type, binding summary, intrinsic leaves,
  statement counts, and load/store cross-contamination.
- C++ target artifact tests must mutate candidate metadata mirrors for the
  same fields and prove stale mirrors cannot be accepted.
- Focused lit/generated-bundle dry-run tests must keep representative
  computed-mask strided store/load fixtures exposing provider-derived
  `typed_compute_op`, memory form, binding summary, mask/stride facts,
  provider mirror, and target profile.
- Runtime `ssh rvv` evidence is required only when emitted C, runtime ABI,
  mask/stride behavior, inactive-lane/passthrough behavior, correctness, or
  performance behavior changes.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  switch operation kind
  -> fetch computed-mask strided facts directly
  -> trust candidate metadata mirrors for provider_supported_mirror
```

Correct:

```text
typed computed-mask strided body/config/runtime facts
  -> RVVComputedMaskStridedMemoryRouteFacts
  -> RVVComputedMaskStridedMemoryRouteValidationContract
  -> target validator consumes contract for route payload and statements
  -> metadata mirror contract validates candidate mirrors only after that
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
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef maskedIndexedLoadIntrinsic;
  llvm::StringRef maskedIndexedStoreIntrinsic;
  llvm::StringRef maskedStoreIntrinsic;
  llvm::StringRef compareIntrinsic;
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
- Gather carries `maskedIndexedLoadIntrinsic` as the masked indexed load leaf
  and `maskedStoreIntrinsic` as the ordinary unit-store leaf used to write the
  gathered result to the destination buffer. It must carry an empty
  `maskedIndexedStoreIntrinsic`.
- Scatter carries `maskedIndexedStoreIntrinsic` as the masked indexed store
  leaf through the route description's indexed-store field. It must carry empty
  `maskedIndexedLoadIntrinsic` and empty ordinary `maskedStoreIntrinsic`; the
  destination update is not a unit-stride store leaf.
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
- Missing or stale provider-derived VL/vector/index/mask C type facts, setvl,
  vector load, index load, index scale, compare leaf, masked indexed load leaf,
  masked indexed store leaf, or ordinary store leaf -> fail before target
  artifact acceptance.
- Scatter carrying a non-empty ordinary `storeIntrinsic` /
  `maskedStoreIntrinsic` is stale unit-store residue and must fail before target
  artifact acceptance. The masked indexed store leaf belongs in
  `indexedStoreIntrinsic` / `maskedIndexedStoreIntrinsic`.
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
  facts, target profile, provider mirror, masked indexed load/store leaves,
  ordinary store residue, and gather/scatter cross-contamination.
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

### Computed-Mask Indexed Memory Route Validation Contract

#### 1. Scope / Trigger

When target artifact validation accepts rebuilt provider payloads for
`computed_masked_indexed_gather_load_unit_store` or
`computed_masked_indexed_scatter_store_unit_load`, it must consume a
provider-owned route validation contract after rebuilding the RVV provider
route. This contract sits above candidate metadata mirrors: mirrors are checked
only after the rebuilt route description, route payload, ABI mappings, headers,
types, and statement-plan shape match the provider contract.

#### 2. Signatures

The durable provider-owned surface is:

```c++
enum class RVVComputedMaskIndexedMemoryRouteValidationKind {
  IndexedGatherLoadUnitStore,
  IndexedScatterStoreUnitLoad,
};

struct RVVComputedMaskIndexedMemoryRouteValidationContract {
  RVVComputedMaskIndexedMemoryRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string computedMaskMemoryRouteFamilyPlanID;
  std::string computedMaskMemoryMaskProducerSource;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::string indexedMemoryLayout;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::int64_t indexEEW;
  std::string offsetUnit;
  std::string indexSource;
  std::string indexUniqueness;
  std::string indexedDataMemoryForm;
  std::string indexedDestinationMemoryForm;
  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string indexVectorTypeName;
  std::string indexVectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string indexLoadIntrinsic;
  std::string indexScaleIntrinsic;
  std::string maskedIndexedLoadIntrinsic;
  std::string maskedIndexedStoreIntrinsic;
  std::string maskedStoreIntrinsic;
  std::string compareIntrinsic;
  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
  llvm::SmallVector<RuntimeABIParameterRole, 8> runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVComputedMaskIndexedMemoryRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVComputedMaskIndexedMemoryRouteValidationContract>
getRVVComputedMaskIndexedMemoryRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVComputedMaskIndexedMemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

#### 3. Contracts

- The RVV provider builds the validation contract from
  `RVVComputedMaskIndexedMemoryRouteFacts` plus rebuilt route description
  payload details such as result/mask names, EmitC loop/VL names, config
  contract id, and route id.
- Target artifact validation must require this contract before accepting
  operation, memory form, dtype/config, mask/tail policy, compare-produced
  mask facts, index source/EEW/offset/uniqueness, runtime ABI order and
  parameters, route operand binding summary, headers, C type mappings,
  intrinsic leaves, target profile, provider support mirror, and statement-plan
  counts.
- Gather contracts must carry masked indexed load facts, indexed data memory
  form, passthrough load plus unit-store facts, and empty masked indexed store
  and scatter uniqueness residue.
- Scatter contracts must carry masked indexed store facts, indexed destination
  memory form, unique-index facts, and empty masked indexed load plus ordinary
  unit-store residue.
- Candidate metadata mirror validation remains separate and must use
  `getRVVComputedMaskIndexedMemoryRouteMetadataMirrorContract(...)` after route
  payload validation succeeds.

#### 4. Validation & Error Matrix

- Missing validation contract for either computed-mask indexed operation ->
  fail before target artifact acceptance.
- Operation, memory form, route family plan, mask/tail facts, target profile,
  provider mirror, header/type summary, binding plan, binding summary, or
  runtime ABI differs from the contract -> fail before candidate mirrors are
  considered.
- Rebuilt route headers, type mappings, ABI mappings, or statement-plan counts
  disagree with the contract -> fail before artifact acceptance.
- Gather carrying scatter typed compute, masked indexed store, uniqueness, or
  missing unit-store facts -> fail closed. Scatter carrying gather typed
  compute, masked indexed load, ordinary unit-store, or missing uniqueness
  facts -> fail closed.
- Candidate metadata matches while provider payload mismatches -> fail;
  mirrors cannot override provider route validation.
- Stale plain base-memory, strided, segment2, unit-only, scalar-splat,
  arithmetic, descriptor/direct-C/source-export, or legacy i32 route-authority
  residue on a computed-mask indexed route -> fail closed.

#### 5. Good/Base/Bad Cases

- Good: `computed_masked_indexed_gather_load_unit_store` facts -> route
  validation contract -> target checks compare-produced mask, index load/scale,
  masked indexed load leaf, provider binding summary, route headers/types, and
  statement-plan shape before mirrors.
- Good: `computed_masked_indexed_scatter_store_unit_load` facts -> route
  validation contract -> target checks compare-produced mask, index load/scale,
  masked indexed store leaf, unique-index fact, provider binding summary, route
  headers/types, and statement-plan shape before mirrors.
- Base: plain indexed, computed-mask strided, segment2, base-memory, and
  compare/select routes consume their own validation contracts or fact surfaces.
- Bad: target validation accepts a route because the route id, artifact name,
  candidate metadata, or C intrinsic spelling looks like a computed-mask
  indexed route while the provider contract is missing or stale.

#### 6. Tests Required

- C++ target artifact tests must cover positive contract access for both
  computed-mask indexed gather and scatter.
- C++ target artifact tests must mutate provider route descriptions for stale
  provider mirror, target profile, header/type facts, runtime ABI roles, mask
  facts, index source/EEW/offset/uniqueness facts, binding summary, intrinsic
  leaves, statement counts, and gather/scatter cross-contamination.
- C++ target artifact tests must mutate candidate metadata mirrors for the
  same fields and prove stale mirrors cannot be accepted.
- Focused lit/generated-bundle dry-run tests must keep representative
  computed-mask indexed gather/scatter fixtures exposing provider-derived
  `typed_compute_op`, memory form, binding summary, mask/index facts,
  provider mirror, and target profile.
- Runtime `ssh rvv` evidence is required only when emitted C, runtime ABI,
  mask/index behavior, inactive-lane/passthrough behavior, correctness, or
  performance behavior changes.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  switch operation kind
  -> fetch computed-mask indexed facts directly
  -> trust candidate metadata mirrors for provider_supported_mirror
```

Correct:

```text
typed computed-mask indexed body/config/runtime facts
  -> RVVComputedMaskIndexedMemoryRouteFacts
  -> RVVComputedMaskIndexedMemoryRouteValidationContract
  -> target validator consumes contract for route payload and statements
  -> metadata mirror contract validates candidate mirrors only after that
```

### Unit-Stride Masked Memory Route Validation Contract

#### 1. Scope / Trigger

When target artifact validation accepts rebuilt provider payloads for the
existing unit-stride masked memory family, it must consume a provider-owned
route validation contract after rebuilding the RVV provider route. The primary
static routes are `masked_unit_load_store` and `masked_unit_store`; the same
contract surface also covers the already-existing unit-stride computed-mask and
runtime-scalar computed-mask routes that share
`RVVUnitStrideMaskedMemoryRouteFacts`, so target validation does not fall back
to direct fact reconstruction for any unit-stride masked-memory consumer.

This contract sits above candidate metadata mirrors: mirrors are checked only
after the rebuilt route description, route payload, ABI mappings, headers,
types, mask policy, and statement-plan shape match the provider contract.

#### 2. Signatures

The durable provider-owned surface is:

```c++
enum class RVVUnitStrideMaskedMemoryRouteValidationKind {
  MaskedUnitLoadStore,
  MaskedUnitStore,
  ComputedMaskUnitLoadStore,
  RuntimeScalarComputedMaskStore,
  RuntimeScalarComputedMaskLoadStore,
};

struct RVVUnitStrideMaskedMemoryRouteValidationContract {
  RVVUnitStrideMaskedMemoryRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string baseMemoryMovementRouteFamilyPlanID;
  std::string computedMaskMemoryRouteFamilyPlanID;
  std::string computedMaskMemoryMaskProducerSource;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  std::string maskedMemoryLayout;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string maskTypeName;
  std::string maskCType;
  std::string scalarCType;
  std::string setVLIntrinsic;
  std::string vectorLoadIntrinsic;
  std::string maskedLoadIntrinsic;
  std::string storeIntrinsic;
  std::string compareIntrinsic;
  std::string rhsScalarSplatIntrinsic;
  std::string resultName;
  std::string maskName;
  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  llvm::SmallVector<std::string, 8> logicalOperands;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
  llvm::SmallVector<RuntimeABIParameterRole, 8> runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVUnitStrideMaskedMemoryRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVUnitStrideMaskedMemoryRouteValidationContract>
getRVVUnitStrideMaskedMemoryRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVUnitStrideMaskedMemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

#### 3. Contracts

- The RVV provider builds the validation contract from
  `RVVUnitStrideMaskedMemoryRouteFacts` plus rebuilt route description payload
  details such as result/mask names, EmitC loop/VL names, config contract id,
  route id, headers, type mappings, ABI mappings, and statement counts.
- The validation contract embeds
  `RVVRuntimeAVLVLSelectedBoundaryContract` built from provider-owned SEW,
  LMUL, tail/mask policy, config contract, setvl callee, VL C type, runtime
  ABI order, and runtime ABI parameters. Target validation must consume this
  embedded runtime contract before accepting route-local runtime control,
  payload, statement-plan, or mirror facts.
- Target artifact validation must require this contract before accepting
  operation, memory form, dtype/config, mask source, mask/tail policy, runtime
  ABI order and parameters, route operand binding summary, headers, C type
  mappings, intrinsic leaves, target profile, provider support mirror, and
  statement-plan counts.
- Static masked unit-load/store contracts must carry the base-memory route
  family plan and empty computed-mask producer facts. Computed-mask and
  runtime-scalar computed-mask unit-stride contracts must carry computed-mask
  route-family facts and reject stale base-memory residue.
- Unit-load/store contracts must carry a masked load leaf and an ordinary
  unit-store leaf; unit-store-only contracts must carry a unit load plus
  masked store leaf and reject stale masked-load residue. Runtime-scalar
  contracts must additionally carry the scalar C type, RHS scalar ABI role, and
  RHS scalar splat intrinsic.
- Candidate metadata mirror validation remains separate and must use
  `getRVVUnitStrideMaskedMemoryRouteMetadataMirrorContract(...)` after route
  payload validation succeeds.

#### 4. Validation & Error Matrix

- Missing validation contract for any accepted unit-stride masked-memory
  operation -> fail before target artifact acceptance.
- Missing or incomplete embedded runtime AVL/VL selected-boundary contract,
  stale runtime AVL source, stale runtime VL contract id, stale selected
  `with_vl` boundary/scope, stale setvl callee, stale VL type, stale
  full-chunk VL, stale loop VL, stale loop induction, stale runtime n ABI
  role/order, stale remaining-AVL metadata, or stale pointer advancement
  metadata -> fail before target artifact acceptance.
- Operation, memory form, route family plan, mask/tail facts, target profile,
  provider mirror, header/type summary, binding plan, binding summary, or
  runtime ABI differs from the contract -> fail before candidate mirrors are
  considered.
- Rebuilt route headers, type mappings, ABI mappings, EmitC loop/VL names,
  result/mask names, or statement-plan counts disagree with the contract ->
  fail before artifact acceptance.
- Static masked routes carrying computed-mask producer facts, or computed-mask
  routes carrying stale base-memory route-family facts -> fail closed.
- Candidate metadata matches while provider payload mismatches -> fail;
  mirrors cannot override provider route validation.
- Stale strided, indexed, segment2, compare/select, scalar-splat-only,
  arithmetic, descriptor/direct-C/source-export, or legacy i32 route-authority
  residue on a unit-stride masked-memory route -> fail closed.

#### 5. Good/Base/Bad Cases

- Good: `masked_unit_load_store` facts -> route validation contract -> target
  checks selected typed masked load, passthrough/source/destination forms,
  unit-store leaf, provider binding summary, route headers/types, and
  statement-plan shape before mirrors.
- Good: `masked_unit_store` facts -> route validation contract -> target
  checks selected typed masked store, unit-load source form, masked destination
  form, provider binding summary, route headers/types, and statement-plan shape
  before mirrors.
- Base: computed-mask strided, computed-mask indexed, segment2, compare/select,
  and scalar-splat-store routes consume their own validation contracts.
- Bad: target validation accepts a route because the route id, artifact name,
  candidate metadata, test fixture name, or C intrinsic spelling looks like a
  masked unit route while the provider contract is missing or stale.

#### 6. Tests Required

- C++ target artifact tests must cover positive contract access for
  `masked_unit_load_store` and `masked_unit_store`, and any existing
  unit-stride computed-mask/runtime-scalar consumers covered by the same
  accessor.
- C++ target artifact tests must prove the embedded runtime AVL/VL
  selected-boundary contract mirrors rebuilt unit-stride masked memory route
  facts and rejects stale runtime AVL source, runtime VL contract, selected
  `with_vl` scope, setvl callee, VL C type, full-chunk VL, loop VL, loop
  induction, runtime n ABI role, and pointer advancement metadata.
- C++ target artifact tests must mutate provider route descriptions for stale
  provider mirror, target profile, header/type facts, runtime ABI roles, mask
  facts, binding summary, intrinsic leaves, statement counts, and load/store
  cross-contamination.
- C++ target artifact tests must mutate candidate metadata mirrors for the
  same fields and prove stale mirrors cannot be accepted.
- Focused lit/generated-bundle dry-run tests must keep representative
  unit-stride masked-memory fixtures exposing provider-derived
  `typed_compute_op`, memory form, binding summary, mask facts, provider
  mirror, and target profile.
- Runtime `ssh rvv` evidence is required only when emitted C, runtime ABI,
  mask behavior, inactive-lane/passthrough behavior, correctness, or
  performance behavior changes.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  switch operation kind
  -> fetch RVVUnitStrideMaskedMemoryRouteFacts directly
  -> trust candidate metadata mirrors for provider_supported_mirror
```

Correct:

```text
typed unit-stride masked body/config/runtime facts
  -> RVVUnitStrideMaskedMemoryRouteFacts
  -> RVVUnitStrideMaskedMemoryRouteValidationContract
  -> target validator consumes contract for route payload and statements
  -> metadata mirror contract validates candidate mirrors only after that
```

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
- Route-description/provider validation must classify these three operations as
  segment2 consumers before applying generic computed-mask memory validation.
  Generic computed-mask memory checks may verify the shared computed-mask family
  plan fields, but must not own target leaf profile, header/type, segment
  layout, field, passthrough, update, or binding truth for computed-mask
  segment2 routes.
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
- A computed-mask segment2 route that falls through to the non-segment
  computed-mask memory expectation set, or receives an empty segment2 canonical
  fact set while carrying computed-mask segment2 route metadata -> fail closed
  before provider materialization or target artifact acceptance.
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

### Segment2 Memory Route Validation Contract

#### 1. Scope / Trigger

When target artifact validation accepts any existing segment2 memory route, it
must consume a provider-owned route validation contract after rebuilding the
provider route. This applies to plain segment2 deinterleave/interleave and
computed-mask segment2 load/store/update route families. The contract sits
above metadata mirror validation: mirrors are checked after the rebuilt route
and provider description have matched the provider-owned contract.

#### 2. Signatures

The durable provider-owned surface is:

```c++
enum class RVVSegment2MemoryRouteValidationKind {
  PlainDeinterleaveUnitStore,
  PlainInterleaveUnitLoad,
  ComputedMaskLoadUnitStore,
  ComputedMaskStoreUnitLoad,
  ComputedMaskUpdateUnitLoad,
};

struct RVVSegment2MemoryRouteValidationContract {
  RVVSegment2MemoryRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string segment2MemoryRouteFamilyPlanID;
  std::string computedMaskMemoryRouteFamilyPlanID;
  std::string computedMaskMemoryMaskProducerSource;
  std::string maskTailPolicyRouteFamilyPlanID;
  std::string maskTailPolicyOwner;
  std::string comparePredicateKind;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string inactiveLaneContract;
  std::string maskedPassthroughLayout;
  bool usesPlainSegment2;
  bool usesComputedMaskSegment2;
  bool usesDeinterleaveLoad;
  bool usesInterleaveStore;
  bool usesComputedMaskLoad;
  bool usesComputedMaskStore;
  bool usesComputedMaskUpdate;
  std::string segmentMemoryLayout;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::int64_t segmentCount;
  std::string segmentTupleCType;
  std::string segmentLoadIntrinsic;
  std::string segmentStoreIntrinsic;
  std::string segmentFieldExtractIntrinsic;
  std::string segmentTupleCreateIntrinsic;
  std::string segment2UpdateArithmeticKind;
  std::string segment2UpdateArithmeticIntrinsic;
  llvm::SmallVector<RuntimeABIParameter, 8> runtimeABIParameters;
  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  // Plus provider-owned field names, field roles, type mappings, headers,
  // vector/mask C types, setvl/load/store/compare callees, and EmitC names.
};

std::optional<RVVSegment2MemoryRouteValidationContract>
getRVVSegment2MemoryRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

#### 3. Contracts

- The RVV provider builds the contract from the same plain or computed-mask
  segment2 fact accessor used for route planning plus rebuilt route
  description fields such as route id, config contract, field/mask names, and
  EmitC AVL/VL statement names.
- Target artifact validation must require the contract before accepting route
  payload, headers, type mappings, ABI mappings, segment/field facts,
  mask/tail facts, update facts, runtime ABI order, and statement-plan counts.
- Computed-mask segment2 rebuilt route descriptions must carry the
  provider-owned mask/tail route-family plan and owner. Target validation must
  reject missing or stale values rather than falling back to metadata mirrors.
- Candidate metadata mirror validation remains separate and continues to use
  `getRVVSegment2MemoryRouteMetadataMirrorContract(...)` after provider-fact
  validation succeeds.

#### 4. Validation & Error Matrix

- Missing validation contract for a segment2 operation -> fail before target
  artifact acceptance.
- Cross-family operation, route id, memory form, route-family plan, binding
  summary, runtime ABI order, header/type summary, field layout, segment
  intrinsic, mask/tail fact, or update arithmetic fact differs from the
  contract -> fail before artifact acceptance.
- Plain segment2 route carries computed-mask contract fields, or computed-mask
  segment2 route carries plain route-family facts -> fail closed.
- Rebuilt route statement counts or pre-loop/loop AVL/VL statement names differ
  from the contract -> fail before accepting the generated artifact.
- Candidate metadata matches while provider payload mismatches -> fail; mirrors
  cannot override provider route validation.

#### 5. Good/Base/Bad Cases

- Good: computed-mask segment2 update route facts -> route validation contract
  -> target checks mask/tail plan, update arithmetic, field roles, ABI order,
  statement plan, and then candidate mirrors.
- Good: plain segment2 interleave route facts -> route validation contract ->
  target checks two field loads, tuple create, segment store, field roles, and
  absence of computed-mask fields.
- Base: non-segment2 memory routes consume their own validation contracts and
  must not call the segment2 contract accessor.
- Bad: target validation accepts segment2 because route id, artifact name,
  candidate metadata, or C intrinsic spelling looks plausible while the
  provider contract is missing or stale.

#### 6. Tests Required

- C++ target artifact tests must cover positive contract access for plain
  deinterleave/interleave and computed-mask load/store/update routes.
- C++ target artifact tests must keep fail-closed mutations for stale route id,
  binding summary, runtime ABI order/roles, target profile, provider mirror,
  header/type summary, segment layout, field roles, mask facts, update
  arithmetic, and statement steps.
- Focused generated-bundle or lit tests must continue to expose segment2
  mirrors and prove existing explicit/pre-realized segment2 artifacts still
  pass.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  switch operation kind
  -> fetch plain/computed-mask segment2 facts directly
  -> rebuild local statement counts and cross-family stale rules
```

Correct:

```text
typed segment2 body/config/runtime facts
  -> RVV provider fact accessor
  -> RVVSegment2MemoryRouteValidationContract
  -> target validator consumes contract for route payload and statements
  -> metadata mirror contract validates candidate mirrors only after that
```

### Runtime Scalar Splat-Store Route Validation Contract

#### 1. Scope / Trigger

When target artifact validation accepts the existing
`runtime_scalar_splat_store` route family, it must consume a provider-owned
route validation contract after rebuilding the provider route. This route is
the runtime-scalar ABI boundary for scalar value -> vector splat -> unit-stride
store. Target validation must not reconstruct scalar binding, splat vector,
destination store layout, dtype/config, ABI order, statement counts, or
intrinsic leaves from route ids, artifact names, C strings, test names, or
metadata mirrors.

#### 2. Signatures

The durable provider-owned surface is:

```c++
struct RVVRuntimeScalarSplatStoreRouteValidationContract {
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  std::string runtimeScalarSplatStoreRouteFamilyPlanID;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string scalarCType;
  std::string vlCType;
  std::string vectorTypeName;
  std::string vectorCType;
  std::string setVLIntrinsic;
  std::string rhsScalarSplatIntrinsic;
  std::string storeIntrinsic;
  std::string resultName;
  std::string emitCFullChunkVLName;
  std::string emitCLoopVLName;
  std::string emitCLoopInductionName;
  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  llvm::SmallVector<std::string, 4> logicalOperands;
  llvm::SmallVector<RuntimeABIParameter, 4> runtimeABIParameters;
  llvm::SmallVector<RuntimeABIParameterRole, 4> runtimeABIParameterRoles;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  llvm::SmallVector<RVVRuntimeScalarSplatStoreRouteTypeMappingContract, 4>
      typeMappings;
};

std::optional<RVVRuntimeScalarSplatStoreRouteValidationContract>
getRVVRuntimeScalarSplatStoreRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

#### 3. Contracts

- The accessor returns a contract only for
  `RVVSelectedBodyOperationKind::RuntimeScalarSplatStore`.
- The contract is built from provider-owned route/control/statement/binding
  facts for the current typed SEW32 LMUL m1 runtime-scalar splat-store surface:
  ABI order `rhs_scalar,out,n`, typed compute op `tcrv_rvv.splat`, memory form
  `runtime-scalar-splat-store`, and the canonical route-family plan id.
- The contract must embed `RVVRuntimeAVLVLSelectedBoundaryContract`. Target
  validation consumes that embedded contract before accepting local runtime
  ABI, setvl/pre-loop/loop, scalar ABI, vector/type, splat/store intrinsic,
  header/type, route operand binding, statement-plan, or metadata mirror facts.
- Target artifact validation is a consume-only client. It compares rebuilt
  route id, provider-supported mirror, route-family plan id, runtime control
  plan, ABI order, runtime ABI parameters/roles, operand-binding summary,
  dtype/config, header/type summary, splat/store leaf facts, EmitC AVL/VL
  names, and exact pre-loop/loop statement counts against the contract.
- Candidate metadata mirrors are checked after provider payload validation and
  must mirror the contract values for binding plan/summary, memory form,
  typed compute op, provider support, route-family plan, runtime control plan,
  ABI order, header declarations, and C type mapping.

#### 4. Validation & Error Matrix

- Missing validation contract -> fail before target artifact acceptance.
- Missing or incomplete embedded runtime AVL/VL selected-boundary contract ->
  fail before route payload, statement-plan, or mirror acceptance.
- Route id, operation, memory form, typed compute op, provider mirror,
  route-family plan, runtime control plan, runtime ABI order, dtype/config,
  header/type summary, splat/store/result leaf, EmitC AVL/VL name, or
  statement-plan count differs from the contract -> fail before artifact export.
- Runtime AVL source, runtime VL contract id, selected `with_vl` scope, setvl
  callee, VL C type, full-chunk VL, loop VL, loop induction, runtime `n` ABI
  role/order/ownership, remaining AVL metadata, or pointer advancement
  metadata differs from the embedded runtime contract -> fail before artifact
  export.
- Description runtime ABI parameters or route ABI mappings do not match
  `rhs_scalar`, `out`, and `n` in provider order -> fail before statement
  validation.
- Route operand-binding summary does not exactly match the provider summary ->
  fail before artifact export.
- Candidate mirror missing/stale, or stale non-splat-store route-family mirror
  present -> fail before accepting candidate metadata.

#### 5. Good/Base/Bad Cases

- Good: selected typed runtime-scalar splat-store body -> provider route
  validation contract -> target validates runtime scalar binding, splat vector,
  `out + offset` destination store layout, ABI order, and then candidate
  mirrors.
- Base: runtime-scalar compare/mask, MAcc, reduction, memory, segment2,
  conversion, and widening-dot routes consume their own contracts and must not
  call the splat-store accessor.
- Bad: target validation accepts the artifact because the route id says
  `runtime_scalar_splat_store`, the generated C contains plausible RVV calls,
  or metadata mirrors say supported while the provider contract is missing or
  stale.

#### 6. Tests Required

- RVV plugin C++ tests must prove the accessor returns canonical contract
  facts for runtime-scalar splat-store and `std::nullopt` for non-splat routes.
- Target artifact C++ tests must mutate provider payload fields and candidate
  mirrors for stale route id, provider mirror, route-family plan, memory form,
  ABI order, binding summary, splat/store leaf facts, header/type summary, and
  stale non-family mirrors.
- Focused lit or generated-bundle dry-run tests for existing explicit and
  pre-realized runtime-scalar splat-store fixtures must continue to pass.

#### 7. Wrong vs Correct

Wrong:

```text
target validator:
  route id says runtime_scalar_splat_store
  -> inspect C string / metadata mirror for splat and store
  -> accept artifact
```

Correct:

```text
typed runtime-scalar splat-store body/config/runtime facts
  -> RVV provider builds RVVRuntimeScalarSplatStoreRouteValidationContract
     with embedded RVVRuntimeAVLVLSelectedBoundaryContract
  -> target validator consumes contract for route payload and statements
  -> candidate metadata mirrors are checked only as mirrors
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

### Widening Conversion Route Validation Contract

Scope / trigger: selected-body `widen_i32_to_i64` and `widen_i16_to_i32`
routes must expose a provider-owned validation contract before target artifact
validation accepts rebuilt route payloads or candidate metadata mirrors.

Signatures:

```c++
struct RVVConversionDtypePolicyRouteValidationContract {
  RVVConversionDtypePolicyRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string sourceElementTypeName;
  std::string resultElementTypeName;
  std::int64_t sourceSEW;
  std::string sourceLMUL;
  std::int64_t resultSEW;
  std::string resultLMUL;
  std::string conversionKind;
  std::string conversionRelation;
  std::string sourceMemoryForm;
  std::string destinationMemoryForm;
  std::string runtimeABIOrder;
  llvm::SmallVector<RuntimeABIParameter, 3> runtimeABIParameters;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  llvm::SmallVector<RVVConversionDtypePolicyRouteTypeMappingContract, 3>
      typeMappings;
  llvm::SmallVector<std::string, 4> requiredHeaders;
  std::size_t expectedPreLoopStepCount;
  std::size_t expectedLoopBodyStepCount;
  /* plus provider support/header/type/intrinsic/result/statement names */
};

std::optional<RVVConversionDtypePolicyRouteValidationContract>
getRVVConversionDtypePolicyRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<RVVConversionDtypePolicyRouteMetadataMirrorContractSet>
getRVVConversionDtypePolicyRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

Contracts:

- The validation contract must be built from
  `getRVVWideningConversionRouteFacts(description.operation)` and selected
  route statement names. Unsupported/non-conversion operations return
  `std::nullopt`.
- Target artifact route-family validation is a consume-only client: it compares
  the embedded `RVVRuntimeAVLVLSelectedBoundaryContract` before accepting
  rebuilt route id, headers, type mappings, ABI mappings, source/result dtype
  policy, SEW/LMUL relation, conversion relation, memory forms, tail/mask
  policy, intrinsic leaves, and statement-plan counts against the contract.
- The embedded runtime contract is the authority for runtime AVL source,
  runtime-VL contract id, selected `with_vl` boundary/scope,
  selected-body provenance, setvl callee, VL C type, full-chunk VL, loop VL,
  loop induction, remaining AVL metadata, pointer advancement metadata,
  bounded-slice, multi-VL, and runtime `n` ABI parameter facts. Conversion
  statement-plan validation must read those runtime/control names and the
  runtime AVL parameter from the embedded contract instead of reconstructing
  them from conversion-local fields.
- Candidate metadata validation must consume the metadata mirror contract and
  must reject stale non-conversion route-family mirrors from that contract.

Validation & error matrix:

- Missing validation contract -> fail before route payload validation.
- Missing runtime AVL/VL selected-boundary contract, stale runtime AVL source,
  stale runtime VL contract id, stale selected `with_vl` scope, stale setvl
  callee, stale VL C type, stale full-chunk VL, stale loop VL, stale loop
  induction, stale runtime `n` ABI role/order, stale remaining AVL metadata,
  or stale pointer advancement metadata -> fail before route payload,
  statement-plan, or metadata mirror acceptance.
- Route id, required header, type mapping, ABI parameter, source/result dtype,
  source/result SEW/LMUL, conversion kind/relation, memory form, intrinsic, or
  statement-plan mismatch -> fail before artifact export.
- Candidate mirror missing/stale or stale non-conversion mirror present -> fail
  before accepting candidate metadata.

Good/base/bad:

- Good: typed widening conversion body -> provider facts -> conversion
  validation contract -> target validator consumes contract -> metadata mirrors
  checked separately as mirrors.
- Base: `widen_i32_to_i64` and `widen_i16_to_i32` expose contracts derived
  from their existing canonical widening conversion facts.
- Bad: target artifact validation reconstructs conversion semantics from route
  ids, artifact metadata, C strings, test names, or local target tables.

Tests required:

- C++ target artifact tests must prove positive contract access for both
  existing widening conversion variants, including embedded runtime AVL/VL
  selected-boundary contract fields, and no contract for non-conversion
  operations.
- C++ target artifact tests must mutate runtime AVL source, runtime VL
  contract id, selected `with_vl` scope, setvl callee, VL C type,
  full-chunk VL, loop VL, loop induction, runtime `n` ABI role, remaining AVL
  metadata, and pointer advancement metadata so the embedded runtime contract
  owns the first rejection.
- C++ target artifact tests must mutate route payload fields and metadata
  mirrors to prove fail-closed behavior.
- Focused conversion lit/dry-run fixtures must continue to pass for existing
  `widen_i32_to_i64` and `widen_i16_to_i32` artifact flows.

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
artifact validation must consume
`getRVVConversionDtypePolicyRouteValidationContract(...)`, then consume the
embedded runtime AVL/VL contract before rejecting stale local copies of
source/result element type, source/result SEW-LMUL, tail/mask policy,
conversion kind/relation, source/destination memory form, runtime ABI parameter
facts, header/type mapping, target profile, provider mirror, route-family plan,
or operand binding summary before accepting a bundle.

Good:

```text
typed widen_i32_to_i64 body/config/runtime facts
  -> getRVVWideningConversionRouteFacts(WidenI32ToI64)
  -> widening conversion route-family plan
  -> provider-built TCRVEmitCLowerableRoute
  -> target validator consumes the provider validation contract
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
  llvm::StringRef sourceStrideCType;
  llvm::StringRef destinationStrideCType;
  llvm::StringRef sourceStrideUnit;
  llvm::StringRef destinationStrideUnit;
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

The target-facing validation contract for this route family embeds the shared
runtime AVL/VL selected-boundary contract:

```c++
struct RVVBaseMemoryMovementRouteValidationContract {
  RVVBaseMemoryMovementRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string baseMemoryMovementRouteFamilyPlanID;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  ...
};

std::optional<RVVBaseMemoryMovementRouteValidationContract>
getRVVBaseMemoryMovementRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
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
- `getRVVBaseMemoryMovementRouteValidationContract(...)` builds
  `runtimeAVLVLContract` through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after the provider-owned
  base-memory runtime ABI parameter list is available. The runtime contract is
  the selected-boundary authority for runtime AVL source, runtime-VL contract,
  selected `with_vl` scope, `setvl` callee, VL C type, full-chunk VL, loop VL,
  loop induction, runtime `n` ABI parameter, remaining-AVL metadata, pointer
  advancement metadata, bounded-slice, and multi-VL facts.
- Base-memory target artifact validation must consume
  `runtimeAVLVLContract` before accepting route ids, route payload, header
  declarations, C type mappings, ABI mappings, statement plans, stride/index/
  mask facts, pointer advancement, or candidate metadata mirrors. The older
  family-local runtime fields remain mirrors checked against the same
  provider-built route description; they are not an independent runtime
  authority.
- Base-memory statement-plan validation uses the embedded runtime contract for
  the pre-loop `setvl`, loop bounds, loop `setvl`, loop VL operand, loop
  induction name, runtime `n` ABI parameter, and pointer-advance induction.
  It may use route-family fields for memory-form-specific callees, vector
  types, result names, strides, indices, masks, and passthrough facts only
  after the runtime contract has already matched the rebuilt route description.
- Common EmitC carries the provider-built route and metadata mirrors; it must
  not infer base memory kind, index role, offset unit, header set, C type
  mapping, or ABI order from artifact names, route ids, tests, scripts, C
  strings, or mirror metadata.

Validation and errors:

- Missing accessor result for a supported base memory operation -> provider and
  target validators fail before artifact export.
- Missing embedded runtime AVL/VL selected-boundary contract, more than one or
  zero runtime-element-count ABI parameter, stale runtime AVL source, stale
  runtime-VL contract id, stale selected `with_vl` boundary/scope, stale
  `setvl` callee, stale VL C type, stale full-chunk VL, stale loop VL, stale
  loop induction, stale runtime `n` ABI role/order/ownership, stale
  remaining-AVL metadata, or stale pointer advancement metadata -> target
  validation fails before accepting base-memory route payload, headers,
  ABI/type mappings, statement plans, or artifact mirrors.
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
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> base memory route-family plan and operand-binding summary
  -> provider-built TCRVEmitCLowerableRoute
  -> target validator consumes runtime contract before the same route-family
     facts
```

Base:

```text
typed strided_load_unit_store / unit_load_strided_store /
indexed_gather_unit_store / indexed_scatter_unit_load /
masked_unit_load_store / masked_unit_store
  -> RVVBaseMemoryMovementRouteValidationContract embeds
     RVVRuntimeAVLVLSelectedBoundaryContract
  -> target validation accepts only matching provider facts
```

Bad:

```text
target validator local table says index_eew=32 and offset_unit=element
  -> accepts artifact while provider accessor is missing or route metadata is stale
```

Bad:

```text
base-memory statement validation:
  reads loop induction, runtime n, or VL names from family-local mirrors
  -> accepts a stale runtime AVL/VL selected-boundary contract
```

Required tests:

- C++ target artifact tests must mutate provider descriptions and candidate
  metadata mirrors for stale ABI order, index EEW, offset unit, index source,
  indexed layout, route-family plan, target profile, provider mirror, header
  facts, type facts, binding summary, and accidental strided/unit-load
  fallback.
- C++ target artifact tests must assert
  `RVVBaseMemoryMovementRouteValidationContract::runtimeAVLVLContract` matches
  the rebuilt route description for every production-active base-memory
  movement operation and must mutate runtime AVL source, runtime-VL contract,
  selected `with_vl` boundary/scope, `setvl` callee, VL C type, full-chunk VL,
  loop VL, loop induction, runtime `n` ABI facts, and pointer advancement
  metadata to prove fail-closed diagnostics before route acceptance.
- Generated-bundle dry-run tests must assert representative accessor facts,
  base memory boundary evidence, and provider-derived operand/header/type
  summaries.
- Runtime RVV correctness claims still require real `ssh rvv` execution after
  provider and target validators accept the route.

Wrong:

```text
target base-memory validator:
  description route id names indexed_gather_unit_store
  -> reconstructs runtime n/VL from ABI strings and loop C snippets
  -> accepts stale runtime AVL/VL or pointer advancement facts
```

Correct:

```text
typed selected tcrv_rvv base-memory body/config/runtime facts
  -> RVV provider base-memory facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> RVVBaseMemoryMovementRouteValidationContract
  -> target validator consumes runtime contract before route-family payloads
```

### Compare/Select Route Validation Contract

Use this contract when target artifact validation consumes production-active
selected-body compare/select routes: plain compare-select, computed-mask
select, runtime-scalar computed-mask select, and runtime-scalar dual
compare-mask-and-select.

The concrete provider-facing target validation surface is:

```c++
struct RVVCompareSelectRouteValidationContract {
  RVVCompareSelectRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  ...
};

std::optional<RVVCompareSelectRouteValidationContract>
getRVVCompareSelectRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

Contracts:

- `getRVVCompareSelectRouteValidationContract(...)` is implemented in the RVV
  provider layer and is the canonical source for compare/select operation,
  memory form, config, runtime ABI order, target leaf profile,
  provider-supported mirror, header/type summaries, operand binding summary,
  route-family plan ids, predicate facts, select layout, runtime-scalar
  single/dual markers, mask/tail provider facts, and statement-plan counts.
- The contract builder must populate `configContractID` from the rebuilt
  selected-body route description and must build `runtimeAVLVLContract` through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after the provider-owned
  compare/select runtime ABI parameter list is available.
- The embedded runtime contract is the selected-boundary authority for runtime
  AVL source, runtime-VL contract, selected `with_vl` boundary/scope, `setvl`
  callee, VL C type, full-chunk VL, loop VL, loop induction, runtime `n` ABI
  parameter, remaining-AVL metadata, pointer advancement metadata,
  bounded-slice, and multi-VL facts.
- Compare/select target artifact validation must consume
  `runtimeAVLVLContract` before accepting route payloads, header declarations,
  C type mappings, runtime ABI mappings, compare predicate facts, select
  layout, dual-compare mask-composition facts, statement plans, or candidate
  metadata mirrors. Family-local runtime fields remain mirrors checked against
  the same provider-built route description; they are not an independent
  runtime authority.
- Compare/select statement-plan validation must use the embedded runtime
  contract for pre-loop `setvl`, loop bounds, loop `setvl`, loop VL operand,
  loop induction name, runtime `n` ABI parameter, and remaining-AVL expression
  checks when the compare/select validation contract is present. It may use
  route-family fields for compare/select callees, vector/mask types, result
  names, select layout, scalar splat, secondary compare, and mask-and facts
  only after the runtime contract has already matched the rebuilt route
  description.
- Common EmitC carries the provider-built route and metadata mirrors; it must
  not infer compare/select operation kind, predicate, select layout, runtime
  control, ABI order, headers, type mappings, or route support from artifact
  names, route ids, tests, scripts, C strings, or mirror metadata.

Validation and errors:

- Missing accessor result for a supported compare/select operation -> provider
  and target validators fail before artifact export.
- Missing embedded runtime AVL/VL selected-boundary contract, more than one or
  zero runtime-element-count ABI parameter, stale runtime AVL source, stale
  runtime-VL contract id, stale selected `with_vl` boundary/scope, stale
  `setvl` callee, stale VL C type, stale full-chunk VL, stale loop VL, stale
  loop induction, stale runtime `n` ABI role/order/ownership, stale
  remaining-AVL metadata, or stale pointer advancement metadata -> target
  validation fails before accepting compare/select route payload, headers,
  ABI/type mappings, statement plans, predicate/select facts, or artifact
  mirrors.
- ABI order, operation, memory form, predicate, secondary predicate,
  route-family plan id, target leaf profile, provider mirror, header summary,
  C type mapping, binding summary, mask role/source/form, select layout,
  inactive-lane contract, or masked passthrough layout differs from accessor
  facts -> fail before artifact acceptance.
- Candidate artifact metadata carries stale computed-mask memory, base-memory,
  segment2, or unrelated route-family mirrors for the selected compare/select
  route -> fail before bundle acceptance.

Good:

```text
typed runtime_scalar_dual_compare_mask_and_select body/config/runtime facts
  -> compare/select route-family plan and operand-binding facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> RVVCompareSelectRouteValidationContract
  -> target validator consumes runtime contract before predicates/select layout
```

Base:

```text
typed cmp_select / computed_mask_select / runtime_scalar_cmp_select /
runtime_scalar_dual_cmp_mask_and_select
  -> RVVCompareSelectRouteValidationContract embeds
     RVVRuntimeAVLVLSelectedBoundaryContract
  -> target validation accepts only matching provider facts
```

Bad:

```text
compare/select statement validation:
  reads runtime n, full-chunk VL, loop VL, or induction only from description strings
  -> accepts a stale runtime AVL/VL selected-boundary contract
```

Required tests:

- C++ target artifact tests must assert
  `RVVCompareSelectRouteValidationContract::runtimeAVLVLContract` matches the
  rebuilt route description for plain compare-select, computed-mask select,
  runtime-scalar compare/select, and runtime-scalar dual compare/select
  contract consumers.
- C++ target artifact tests must mutate runtime AVL source, runtime-VL
  contract, selected `with_vl` scope, `setvl` callee, VL C type,
  full-chunk VL, loop VL, loop induction, runtime `n` ABI facts, and pointer
  advancement metadata to prove fail-closed diagnostics before route
  acceptance.
- Generated-bundle dry-run tests must assert representative compare/select
  selected-body artifacts still pass after the target consumer requires the
  embedded runtime contract.
- Runtime RVV correctness claims still require real `ssh rvv` execution after
  provider and target validators accept the route.

Correct:

```text
typed selected tcrv_rvv compare/select body/config/runtime facts
  -> RVV provider compare/select facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> RVVCompareSelectRouteValidationContract
  -> target validator consumes runtime contract before route-family payloads
```

### Elementwise Arithmetic Route Validation Contract

Use this contract when target artifact validation consumes production-active
selected-body elementwise arithmetic routes: ordinary vector/vector
`Add`/`Sub`/`Mul`, masked vector/vector `Add`/`Sub`/`Mul`,
scalar-broadcast `Add`/`Sub`/`Mul`, and already connected strided elementwise
arithmetic routes.

The concrete provider-facing target validation surface is:

```c++
struct RVVElementwiseArithmeticRouteValidationContract {
  RVVElementwiseArithmeticRouteValidationKind kind;
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef consumerLabel;

  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
  std::string elementTypeName;
  std::int64_t sew;
  std::string lmul;
  std::string tailPolicy;
  std::string maskPolicy;
  std::string configContractID;
  std::string runtimeControlPlanID;
  std::string runtimeABIOrder;
  std::string targetLeafProfile;
  std::string providerSupportedMirror;
  std::string requiredHeaderDeclarations;
  std::string cTypeMappingSummary;
  std::string routeOperandBindingPlanID;
  std::string routeOperandBindingSummary;
  std::string typedComputeOpName;
  RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract;
  ...
};

std::optional<RVVElementwiseArithmeticRouteValidationContract>
getRVVElementwiseArithmeticRouteValidationContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

Contracts:

- `getRVVElementwiseArithmeticRouteValidationContract(...)` is implemented in
  the RVV provider layer and is the canonical source for elementwise operation,
  memory/source form, scalar-broadcast form, masked form, strided form,
  runtime ABI order, target leaf profile, provider-supported mirror,
  header/type summaries, operand binding plan/summary, route-family plan ids,
  typed compute op, intrinsic leaves, result/mask names, and statement-plan
  counts.
- The contract builder must build `runtimeAVLVLContract` through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after the provider-owned
  elementwise runtime ABI parameter list is available. The embedded runtime
  contract must use provider-derived typed config facts for `setvl` callee and
  VL C type, not mutable route-description mirrors.
- The embedded runtime contract is the selected-boundary authority for runtime
  AVL source, runtime-VL contract, selected `with_vl` boundary/scope, `setvl`
  callee, VL C type, full-chunk VL, loop VL, loop induction, runtime `n` ABI
  parameter, remaining-AVL metadata, pointer advancement metadata,
  bounded-slice, and multi-VL facts.
- Elementwise target artifact validation must consume `runtimeAVLVLContract`
  before accepting route payloads, route ids, header declarations, C type
  mappings, ABI mappings, operation/source/memory-form facts, scalar-broadcast
  facts, masked facts, strided facts, statement plans, or candidate metadata
  mirrors. Family-local runtime fields remain mirrors checked against the same
  provider-built route description; they are not an independent runtime
  authority.
- Elementwise statement-plan validation uses the embedded runtime contract for
  the pre-loop `setvl`, loop bounds, loop `setvl`, full-chunk VL, loop VL,
  loop induction name, runtime `n` ABI parameter, remaining-AVL expression,
  and pointer-advance induction. It may use family fields for vector loads,
  scalar splat, arithmetic, compare/merge, strided loads/stores, result names,
  mask names, and store leaves only after the runtime contract has matched the
  rebuilt route description.
- Common EmitC carries the provider-built route and metadata mirrors; it must
  not infer elementwise operation kind, scalar-broadcast form, mask/tail facts,
  strided layout, runtime control, ABI order, headers, type mappings, or route
  support from artifact names, route ids, tests, scripts, C strings, or mirror
  metadata.

Validation and errors:

- Missing accessor result for a supported elementwise operation -> provider and
  target validators fail before artifact export.
- Missing embedded runtime AVL/VL selected-boundary contract, more than one or
  zero runtime-element-count ABI parameter, stale runtime AVL source, stale
  runtime-VL contract id, stale selected `with_vl` boundary/scope, stale
  `setvl` callee, stale VL C type, stale full-chunk VL, stale loop VL, stale
  loop induction, stale runtime `n` ABI role/order/ownership, stale remaining
  AVL metadata, or stale pointer advancement metadata -> target validation
  fails before accepting elementwise route payload, headers, ABI/type mappings,
  statement plans, source/memory/mask facts, or artifact mirrors.
- ABI order, operation, memory form, scalar-broadcast plan, elementwise plan,
  target leaf profile, provider mirror, header summary, C type mapping,
  binding summary, mask role/source/form, inactive-lane contract,
  passthrough layout, strided layout, stride sources, intrinsic leaves, result
  name, or mask name differs from accessor facts -> fail before artifact
  acceptance.
- Candidate artifact metadata carries stale conversion, compare/select,
  computed-mask memory, MAcc, segment2, standalone reduction, contraction, base
  memory, or unrelated route-family mirrors for the selected elementwise route
  -> fail before bundle acceptance.

Required tests:

- C++ target artifact tests must assert
  `RVVElementwiseArithmeticRouteValidationContract::runtimeAVLVLContract`
  matches the rebuilt route description for representative plain, masked, and
  scalar-broadcast elementwise routes.
- C++ target artifact tests must mutate runtime AVL source, runtime-VL
  contract, selected `with_vl` scope, `setvl` callee, VL C type,
  full-chunk VL, loop VL, loop induction, runtime `n` ABI facts, and pointer
  advancement metadata to prove fail-closed diagnostics before route
  acceptance.
- Generated-bundle dry-run tests must keep representative plain, masked, and
  scalar-broadcast elementwise artifacts passing after the target consumer
  requires the embedded runtime contract.
- Runtime RVV correctness claims still require real `ssh rvv` execution after
  provider and target validators accept the route.

Correct:

```text
typed selected tcrv_rvv elementwise body/config/runtime facts
  -> RVV provider elementwise route-family facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> RVVElementwiseArithmeticRouteValidationContract
  -> target validator consumes runtime contract before route-family payloads
```

### Provider-Owned Memory Metadata Mirror Contract

#### 1. Scope / Trigger

Use this contract when target artifact validation needs a normalized
`key/expected/label` metadata mirror table for existing RVV memory route
families. This includes base/unit-stride, masked unit-stride, strided,
indexed, plain segment2, and computed-mask segment2 memory routes. The trigger
is any target artifact consumer that would otherwise assemble memory mirror
keys, expected values, stale-route-family keys, or labels locally.

#### 2. Signatures

The durable provider-owned C++ surface is:

```c++
struct RVVMemoryRouteMetadataMirrorContract {
  llvm::StringRef key;
  std::string expected;
  llvm::StringRef label;
};

struct RVVMemoryRouteMetadataMirrorContractSet {
  llvm::SmallVector<RVVMemoryRouteMetadataMirrorContract, 32> mirrors;
  llvm::SmallVector<llvm::StringRef, 16> staleMirrorKeys;
  llvm::StringRef staleMirrorLabel;
};

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVBaseMemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);

std::optional<RVVMemoryRouteMetadataMirrorContractSet>
getRVVSegment2MemoryRouteMetadataMirrorContract(
    const RVVSelectedBodyEmitCRouteDescription &description);
```

`expected` is owning storage so provider accessors may safely return computed
string values such as `index_eew` or `segment_count`.

#### 3. Contracts

- Provider contract accessors build mirror entries from the same provider fact
  surfaces and rebuilt provider route description used for route validation.
- Target artifact validation consumes the returned contract set after the
  provider route is rebuilt. Target code may look up candidate metadata and
  format target bridge diagnostics, but must not locally define memory-family
  mirror tables or stale-route-family key sets.
- Candidate artifact metadata may only mirror entries in the provider-owned
  contract. A non-empty `expected` requires an exact candidate mirror; an empty
  `expected` rejects stale candidate metadata for that key.
- `staleMirrorKeys` are provider-owned rejection lists for unrelated
  route-family mirrors that must be absent on the selected memory route.
- The contract does not replace family-specific provider fact validation,
  route payload validation, runtime ABI validation, header/type validation, or
  segment2 statement-plan validation.

#### 4. Validation & Error Matrix

- Missing contract for a route family selected by the target validator -> fail
  before artifact acceptance.
- Missing candidate mirror for a non-empty provider contract entry -> fail
  before artifact acceptance.
- Candidate mirror value differs from `expected` -> fail before artifact
  acceptance.
- Candidate carries a key whose provider contract entry has empty `expected`
  -> fail as stale mirror residue.
- Candidate carries any provider-owned `staleMirrorKeys` entry -> fail as
  cross-family or cross-route residue.
- Provider fact validation disagrees with the provider route description ->
  fail before the target consumes candidate mirrors.

#### 5. Good/Base/Bad Cases

- Good: base indexed memory facts -> provider-owned contract entries for
  binding summary, provider mirror, memory form, index facts, header/type
  facts, and stale non-base route-family keys -> target consumes the contract.
- Good: computed-mask segment2 route facts plus rebuilt route description ->
  provider-owned contract entries for binding, mask, segment, field, update,
  header/type, and stale non-segment2 route-family keys.
- Base: route families without memory artifact mirrors use their own provider
  fact and target validation contracts.
- Bad: target artifact validation reconstructs expected `index_eew`,
  `segment_count`, mask role/source/form, header/type summaries, or stale
  memory route-family keys in a target-local table.

#### 6. Tests Required

- C++ target artifact tests must continue to mutate provider descriptions and
  candidate metadata mirrors for representative stale memory fields and prove
  fail-closed behavior.
- Provider or plugin C++ builds must cover the public contract accessors when
  the provider header or implementation changes.
- Focused lit/generated-bundle dry-run tests must keep representative memory
  fixtures exposing provider-derived metadata mirrors.
- Runtime `ssh rvv` evidence is required only when generated runtime ABI,
  load/store behavior, mask/tail behavior, passthrough/destination
  preservation, correctness, or performance behavior changes.

#### 7. Wrong vs Correct

Wrong:

```text
target validator local table:
  tcrv_rvv.index_eew = 32
  tcrv_rvv.segment_count = 2
  stale keys = local target list
```

Correct:

```text
typed tcrv_rvv body/config/runtime facts
  -> RVV provider fact accessor and route description
  -> provider-owned memory metadata mirror contract set
  -> target validator consumes the contract and checks candidate mirrors
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
