# RVV Plugin

## Role

RVV is the current primary real-hardware plugin for TianChen-RV. It is the
first family that must become route-supported, typed, plugin-owned, and backed
by `ssh rvv` evidence whenever runtime, correctness, or performance is claimed.

RVV-first does not mean an `i32m1` add/sub/mul demo, a current Linalg frontend
phase, one op per intrinsic, or an EmitC scheduling trick. The durable path is:

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv vector-level body
  -> RVV plugin-owned legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV intrinsic C/C++ or equivalent backend representation
  -> target artifact
  -> ssh rvv evidence when runtime/correctness/performance is claimed
```

## Authority Placement

### `tcrv.exec`

`tcrv.exec` owns execution envelope and organization only:

```text
kernel
capability scope
selected variants
requires
dispatch / fallback
diagnostics
mem_window / runtime_param ABI role declarations
```

It does not own RVV compute semantics, dtype authority, RVV schedule,
intrinsic spelling, or selected route authority. A route id, artifact name,
C ABI string, test name, source-front-door marker, descriptor, or diagnostic
status must not be used to infer RVV computation.

### `tcrv_rvv` Body

The selected `tcrv_rvv` body owns typed vector-level RVV execution structure:

```text
typed vector values
element dtype
SEW / LMUL / vtype policy constraints
VL / AVL / setvl
load/store and memory forms
arithmetic / compare / select / FMA
reduction primitive / accumulator
mask / tail behavior
movement / layout / conversion
runtime ABI value use
low-level vector control
```

The desired generic typed shape is:

```text
!tcrv_rvv.vector<elem = i32, lmul = m1>
%vl = tcrv_rvv.setvl %remaining {sew = 32, lmul = m1, policy = ...}
%a  = tcrv_rvv.load %lhs[%i], %vl
%b  = tcrv_rvv.load %rhs[%i], %vl
%c  = tcrv_rvv.binary {kind = add} %a, %b, %vl
tcrv_rvv.store %out[%i], %c, %vl
```

Dtype/config/operation facts enter through typed values, config, and body
structure. Do not infer them from route ids, C ABI strings, parameter names,
artifact names, test names, exact `__riscv_*_i32m1` spellings, or old
`!tcrv_rvv.i32m1` helper names.

The RVV route provider derives concrete intrinsic/backend spelling from:

```text
operation kind
element type
SEW
LMUL
tail/mask policy
operand form
memory form
runtime ABI binding
target capability
```

### RVV Plugin

The RVV plugin owns:

- legality for RVV body/config/control/dataflow;
- selected-body realization when hints/config/profile affect generated code;
- route support and route provider output;
- intrinsic mapping;
- C/RVV vector type mapping;
- ABI mapping;
- fail-closed diagnostics.

Common/core orchestration may call plugin interfaces and validate generic
structure. It must not branch on RVV semantics, choose RVV intrinsics, infer
dtype, build schedules, or synthesize body shape.

### Common EmitC / Export

Common EmitC/export owns neutral materialization and packaging only. It
materializes provider-built `TCRVEmitCLowerableRoute` payloads. It must not
invent RVV compute, dtype, SEW/LMUL, schedule, intrinsic choices, or ABI role
semantics.

## Parameter Flow

`tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime roles:

```text
lhs / rhs / out / n
buffer / scalar
input / output
runtime count
C ABI spelling / provenance
```

The selected `tcrv_rvv` body must explicitly bind/import those values and
consume them in typed control/dataflow:

```text
%lhs = tcrv_rvv.runtime_abi_value @lhs
%rhs = tcrv_rvv.runtime_abi_value @rhs
%out = tcrv_rvv.runtime_abi_value @out
%n   = tcrv_rvv.runtime_abi_value @n
```

Then the body uses them through `setvl`, loads/stores, compute ops, masks,
reductions, or movement ops. `tcrv.exec` must not infer add/mul/reduce/dtype
from ABI role names, C type strings, or artifact metadata.

## Stage Gates

### Stage 1: RVV Route-Authority Reset

Stage 1 replaces or fail-closes active paths that still treat any of these as
RVV architecture authority:

```text
bounded i32m1 arithmetic
RVVI32M1 route specs/slices
finite tcrv_rvv.i32_* ops
!tcrv_rvv.i32m1 helper types
rvv-i32m1 route ids
artifact names
source-front-door / source-artifact markers
descriptor residue
exact __riscv_*_i32m1 spellings
```

Stage 1 ends only when no active compiler path uses those as the RVV family
architecture. Do not preserve a supported legacy `rvv-i32m1-*` object/header
or bundle compatibility route. Do not add new dtype-prefixed helper families
such as reduction, accumulator, multiply-accumulate, conversion, memory-form,
or LMUL clones.

Legacy explicit bodies may remain only as parse/verify/fail-closed fixtures,
or be rewritten as ordinary instances of the corrected generic typed vector
surface. They must not be positive generated-artifact tests unless rewritten
onto the corrected typed route.

### Stage 2: Corrected Typed RVV Coverage

Stage 2 expands route-supported RVV coverage on the corrected typed
`tcrv_rvv` surface. Coverage is calibrated by Linalg-like structured
computation classes:

```text
elementwise
broadcast
reduction / accumulation
contraction-like accumulation
memory movement
dtype conversion
mask / tail
runtime shape/control
layout / movement
```

This is not permission to build a current Linalg frontend, high-level kernel
ops, one-intrinsic wrappers, dtype/LMUL clone batches, global autotuning
databases, dashboards, or readiness state machines.

## Low-Precision Widening-Product Primitive Facts

### 1. Scope / Trigger

Use this contract when a selected typed RVV body claims standalone
low-precision widening-product route support, including the bounded signed and
unsigned forms:

```text
i8mf4 lhs/rhs  -> signed i8*i8 widening product to i16mf2
u8mf4 lhs/rhs  -> unsigned u8*u8 widening product to u16mf2
```

This is a Stage 2 product-fact contract. It is not q8/q4 route authority, not
an artifact-name route, not a Common EmitC semantic branch, and not a promise
of widening reduction or accumulation support.

### 2. Required Facts

The RVV provider must derive standalone widening-product facts from the typed
body/config/runtime facts before route construction. The structural fact set
must include:

```text
source element type and signedness
source SEW/LMUL and vector/C type
source byte-load fact
extension/sign policy
lhs/rhs multiplicand roles
runtime ABI order and exported lhs/rhs/out/n parameter facts
product/result element type, SEW/LMUL, vector/C type
widening-product relation and intrinsic
target leaf profile, headers, C type mapping, and operand binding plan
tail/mask policy and runtime AVL/VL control plan
```

Current provider-owned mirror keys include:

```text
tcrv_rvv.widening_product_multiplicand_roles
tcrv_rvv.widening_product_extension_policy
tcrv_rvv.low_precision_primitive.source_signedness
tcrv_rvv.low_precision_primitive.source_load
tcrv_rvv.low_precision_primitive.source_extension
tcrv_rvv.low_precision_primitive.product_dtype
tcrv_rvv.widening_product_relation
tcrv_rvv.widening_product_intrinsic
```

The multiplicand-role summary must name both ABI-visible operands as product
multiplicands, for example `lhs=lhs-input-buffer:wprod-lhs:src-i8mf4` and
`rhs=rhs-input-buffer:wprod-rhs:src-i8mf4` for the signed form, or the
corresponding `src-u8mf4` facts for the unsigned form. The extension policy
must pair source signedness with the widening extension rule and product shape,
for example
`source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2` or
`source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2`.

### 3. Contracts

- Dialect verifier legality only proves typed vector/control structure.
- Route support starts only after the RVV provider derives the product facts
  above from typed body/config/runtime/capability facts.
- The route-family plan, route description, route operand binding summary,
  target validation contract, and target artifact mirrors must compare the
  same provider-owned product facts exactly.
- Common EmitC may carry provider payloads only; it must not infer
  multiplicand roles, signedness, extension policy, product dtype/SEW/LMUL, or
  intrinsic spelling.
- Target artifact metadata may mirror these fields only. Stale multiplicand
  roles, extension policy, product dtype, relation, intrinsic, ABI order, or
  C type mapping must fail before artifact acceptance.

### 4. Validation & Error Matrix

- Missing or stale lhs/rhs multiplicand-role facts -> provider/target
  validation error before artifact acceptance.
- Stale signedness, source-load, source-extension, product dtype, relation, or
  intrinsic facts -> provider/target validation error before artifact
  acceptance.
- Verifier-legal typed body with missing provider-owned product facts -> fail
  before `TCRVEmitCLowerableRoute` construction.
- Artifact metadata, route id, ABI string, test name, or Common EmitC helper
  chooses product support -> invalid architecture; move derivation back to the
  RVV provider.

### 5. Good/Base/Bad Cases

- Good: typed signed i8 body -> RVV provider derives signed product facts,
  multiplicand roles, sign-extension policy, product dtype, relation, and
  intrinsic -> target mirrors match exactly.
- Good: typed unsigned u8 body -> RVV provider derives unsigned product facts,
  multiplicand roles, zero-extension policy, product dtype, relation, and
  intrinsic -> target mirrors match exactly.
- Base: standalone widening-product support stops at product facts and does
  not claim widening reduction, accumulation, Gearbox realization, or measured
  performance.
- Bad: route operand binding text or artifact metadata says `wprod-lhs`, so
  target accepts the artifact even though the provider description lacks the
  multiplicand-role summary.
- Bad: Common EmitC selects signed or unsigned product support from ABI C
  pointer spelling instead of consuming an RVV provider-built route.

### 6. Tests Required

- Provider/C++ tests must assert accepted signed and unsigned product facts,
  runtime ABI operands, multiplicand-role summary, extension policy, product
  dtype/SEW/LMUL, relation, and intrinsic.
- Provider/C++ tests must mutate multiplicand roles and extension policy and
  observe provider-owned fail-closed diagnostics before route support.
- Target artifact tests must assert accepted signed and unsigned metadata
  mirrors and stale multiplicand-role, extension-policy, signedness,
  source-load, source-extension, product dtype, relation, intrinsic, ABI, and
  C type mapping rejection.
- Lit/FileCheck fixtures should cover user-visible target export diagnostics
  for signed i8 and unsigned u8 standalone widening-product routes.

### 7. Wrong vs Correct

Wrong:

```text
artifact metadata has wprod-lhs/wprod-rhs
  -> target accepts standalone widening-product support
```

Correct:

```text
typed i8/u8 tcrv_rvv body
  -> RVV provider derives multiplicand-role and extension-policy facts
  -> route-family plan and route description validate those facts
  -> target metadata mirrors those facts exactly or fails closed
```

## Unsigned u8 Low-Precision Widening-Product Boundary

### 1. Scope / Trigger

Use this contract when a selected typed RVV body contains:

```text
tcrv_rvv.widening_product
  kind = "unsigned_widening_product"
  product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"
```

This is a Stage 2 low-precision primitive boundary. It is not q8 route
authority, not an artifact-name route, and not a Common EmitC semantic branch.

### 2. Signatures

The verifier-legal typed body surface is:

```text
lhs, rhs: !tcrv_rvv.vector<ui8, "mf4">
vl:       !tcrv_rvv.vl
result:   !tcrv_rvv.vector<ui16, "mf2">
with_vl:  sew = 16, lmul = "mf2", policy = agnostic/agnostic
ABI:      lhs/rhs use const uint8_t *, out uses uint16_t *, n uses size_t
```

Accepted route support additionally requires provider-owned unsigned facts:

```text
source dtype/sign: u8 / unsigned
product/result dtype/sign: u16
source load fact: unit-stride-byte-load
source extension fact: zero-extend-u8-to-u16-product
vector C types: vuint8mf4_t, vuint16mf2_t
load/store leaves: unsigned u8/u16 RVV leaves
widening product leaf: vwmulu.vv / __riscv_vwmulu_vv_u16mf2
target mirrors: exact unsigned primitive source signedness, type, header, and
load/extension/intrinsic facts
```

### 3. Contracts

- Dialect verifier legality only proves the typed u8 body/config surface.
- Route support starts only after the RVV provider derives the unsigned
  primitive facts above from typed body/config/runtime/capability facts.
- Until those facts exist, the RVV provider must fail closed before
  `TCRVEmitCLowerableRoute` construction and name the missing unsigned
  intrinsic/target facts.
- Common EmitC may carry provider payloads only; it must not infer unsigned
  dtype, vector C types, load/store leaves, source extension, or intrinsic
  spelling.
- Target artifact validation must compare provider-owned unsigned mirrors
  exactly before accepting an artifact.

### 4. Validation & Error Matrix

- Wrong `kind`/`product_relation` pair -> dialect verifier error.
- Wrong ui8/ui16 vector type, SEW, LMUL, policy, or VL use -> dialect verifier
  error.
- Verifier-legal u8 body but no unsigned provider intrinsic/type/target facts
  -> RVV provider error before route construction.
- Stale unsigned primitive source signedness, type, byte-load, zero-extension,
  header, or intrinsic mirror -> target artifact validation error.
- Common EmitC-derived unsigned route payload -> invalid architecture; move the
  derivation back to the RVV provider.

### 5. Good/Base/Bad Cases

- Good: typed u8 body -> RVV provider derives unsigned primitive facts ->
  provider-built route -> Common EmitC carries payload -> target mirrors match.
- Base: typed u8 body parses/verifies, but provider lacks
  `__riscv_vwmulu_vv_u16mf2` and matching target mirror support -> fail closed
  at the provider boundary.
- Bad: q8 artifact name, route id, ABI string, or Common EmitC branch decides
  unsigned widening-product support.

### 6. Tests Required

- Dialect lit: verifier-legal unsigned u8 widening-product typed surface.
- Provider/EmitC lit or C++: unsupported u8 fails at the RVV provider boundary
  with a diagnostic naming missing unsigned intrinsic/target facts.
- When accepted support is added: positive provider/target tests for unsigned
  vector C types, byte load, zero-extension, load/store leaves, `vwmulu.vv`,
  and exact target mirrors.
- Negative target tests: stale unsigned primitive source signedness,
  source load, source extension, source/product/result, intrinsic, type, or
  header mirror fails before artifact acceptance.

### 7. Wrong vs Correct

Wrong:

```text
artifact says q8 route
  -> Common EmitC chooses __riscv_vwmulu_vv_u16mf2
```

Correct:

```text
typed ui8/u16 tcrv_rvv body
  -> RVV provider derives unsigned primitive facts or fails closed
  -> provider-built TCRVEmitCLowerableRoute
  -> Common EmitC carries provider payload unchanged
```

## Low-Precision Widening-Reduction Primitive Facts

### 1. Scope / Trigger

Use this contract when a selected typed RVV body or provider-owned product
reduction chain claims low-precision widening accumulation/reduction facts,
including the bounded signed and unsigned chains:

```text
i8mf4 lhs/rhs
  -> signed i8*i8 widening product to i16mf2
  -> signed widening reduction / vwredsum-style accumulation to i32m1

u8mf4 lhs/rhs
  -> unsigned u8*u8 widening product to u16mf2
  -> unsigned widening reduction / vwredsumu-style accumulation to u32m1
```

This is a Stage 2 primitive-fact contract. It is not a q8/q4 route, not a
llama.cpp-specific path, not an artifact-name authority, and not a Common EmitC
semantic branch.

### 2. Signatures

The provider-owned primitive facts must have a structural home equivalent to:

```c++
struct RVVLowPrecisionWideningReductionPrimitiveFacts {
  bool hasFacts;
  StringRef contractID;
  StringRef lowPrecisionPrimitiveContractID;
  StringRef lowPrecisionPrimitiveKind;
  StringRef kind;
  StringRef sourceElementTypeName;
  StringRef sourceSignedness;
  StringRef sourceLoadKind;
  StringRef sourceExtensionKind;
  StringRef productElementTypeName;
  StringRef accumulatorElementTypeName;
  StringRef reductionResultElementTypeName;
  StringRef finalResultElementTypeName;
  int sourceSEW;
  StringRef sourceLMUL;
  int productSEW;
  StringRef productLMUL;
  int accumulatorSEW;
  StringRef accumulatorLMUL;
  int reductionResultSEW;
  StringRef reductionResultLMUL;
  StringRef wideningProductRelation;
  StringRef productReductionChainRelation;
  StringRef wideningProductIntrinsic;
  StringRef reductionIntrinsic;
  StringRef scalarSeedSplatIntrinsic;
  StringRef accumulatorLayout;
  StringRef resultLayout;
  StringRef reductionStoreVL;
};
```

Exact C++ names may differ, but those facts must be derived by the RVV provider
from the selected typed body/config/runtime facts before route construction.

### 3. Contracts

- Source and product facts must mirror the typed low-precision product surface.
  The bounded signed chain uses source `i8/mf4`, source signedness `signed`,
  source load `unit-stride-byte-load`, source extension
  `sign-extend-i8-to-i16-product`, product `i16/mf2`, and relation
  `signed-i8mf4xi8mf4-to-i16mf2`. The bounded unsigned chain uses source
  `u8/mf4`, source signedness `unsigned`, source load
  `unit-stride-byte-load`, source extension
  `zero-extend-u8-to-u16-product`, product `u16/mf2`, and relation
  `unsigned-u8mf4xu8mf4-to-u16mf2`.
- Accumulator and reduction-result facts must be explicit and distinct from
  final epilogue result facts: the signed primitive reduction result is
  `i32/m1`, the unsigned primitive reduction result is `u32/m1`, and
  dequantized epilogues may have a later `f32` final result.
- The chain relation, widening product intrinsic, widening reduction intrinsic,
  scalar seed splat, accumulator layout, result layout, and reduction store VL
  are provider-owned facts. They must not be reconstructed from route ids,
  artifact names, test names, ABI strings, or Common EmitC helpers.
- For the bounded signed chain, the provider-owned route facts must choose the
  signed product/reduction chain relation, signed widening product intrinsic,
  signed widening reduction intrinsic, signed scalar seed splat, and signed
  accumulator/result vector C types. For the bounded unsigned chain, they must
  choose the unsigned product/reduction chain relation, unsigned widening
  product intrinsic, unsigned widening reduction intrinsic, unsigned scalar seed
  splat, and unsigned accumulator/result vector C types.
- Route-planning and target-validation consumers must choose the expected
  signed or unsigned primitive fact family from provider-derived typed
  plan/description facts before validating a resource selection. The resource
  selection being validated, target metadata, candidate IDs, route IDs, or
  artifact names must not choose their own primitive fact family.
- Target artifact metadata may carry these fields only as exact mirrors. A
  stale mirror must fail target validation before candidate acceptance.
- Common EmitC may materialize only the provider-built route payload. It must
  not choose dtype/sign/SEW/LMUL, `vwmul`, `vwredsum`, seed splat, layout, or
  store-VL semantics itself.

### 4. Validation & Error Matrix

- Product-reduction chain without primitive facts -> RVV provider error before
  `TCRVEmitCLowerableRoute` construction.
- Primitive facts disagree with the selected typed body source signedness,
  source load, source extension, source/product dtype, SEW, LMUL, product
  relation, or signed-vs-unsigned chain relation -> provider fail-closed
  diagnostic.
- Unsigned product-reduction body carries signed accumulator/result dtype,
  signed vector C type, signed `vwredsum` intrinsic, signed scalar seed splat,
  or signed C type mapping -> provider or target fail-closed diagnostic before
  candidate acceptance.
- Primitive facts use stale accumulator/result dtype, SEW, LMUL, layout, seed
  splat, reduction intrinsic, or store VL -> provider fail-closed diagnostic.
- Resource selection source signedness disagrees with the provider-derived
  primitive signedness used by the route plan/description -> provider or target
  validation error before the selection can choose a different signed/unsigned
  primitive fact family.
- Target candidate mirror disagrees with any primitive fact -> target
  validation error before artifact acceptance.
- Common EmitC or target validation locally reconstructs primitive facts from
  metadata or intrinsic spellings -> invalid architecture; move the derivation
  back to the RVV provider contract.

### 5. Good/Base/Bad Cases

- Good: typed `i8mf4` product-reduction body -> RVV provider derives signed
  product and widening-reduction primitive facts -> route validation consumes
  them -> target mirrors compare exactly.
- Good: typed `u8mf4` product-reduction body -> RVV provider derives unsigned
  product and widening-reduction primitive facts, including unsigned
  accumulator/result dtype and unsigned intrinsics -> route validation consumes
  them -> target mirrors compare exactly.
- Base: standalone signed or unsigned widening-product routes keep their own
  product facts without claiming a widening-reduction primitive chain.
- Bad: artifact metadata says `vwredsum`, so the target accepts the candidate
  even though the provider contract lacks accumulator/result facts.
- Bad: the final dequantized `f32` result is treated as the primitive reduction
  result and replaces the required `i32/m1` accumulator/reduction boundary.

### 6. Tests Required

- Provider/C++ coverage for positive product-reduction primitive facts:
  source signedness, source load, source extension,
  source/product/accumulator/result dtype, SEW/LMUL, relation, intrinsics,
  seed splat, layouts, and store VL for both the bounded signed i8 and unsigned
  u8 product-reduction chains.
- Provider negative coverage for stale or unsupported primitive facts before
  route construction.
- Provider/target negative coverage for a stale resource selection signedness
  that would otherwise select the wrong signed/unsigned primitive fact family.
- Target artifact coverage proving exact mirror acceptance and stale source
  signedness, source load, source extension,
  source/product/accumulator/result dtype, SEW/LMUL, intrinsic, seed, layout,
  C type mapping, or store-VL rejection.
- Focused lit coverage for the selected product-reduction artifact path when
  metadata mirror diagnostics are user-visible.
- Runtime `ssh rvv` evidence is required only when the task claims executable
  correctness or performance.

### 7. Wrong vs Correct

Wrong:

```text
candidate metadata has tcrv_rvv.widening_reduction_intrinsic
  -> target accepts the product-reduction route
```

Correct:

```text
typed product-reduction tcrv_rvv body
  -> RVV provider derives signed or unsigned widening-reduction primitive facts
  -> route/provider validation consumes those facts
  -> target metadata mirrors those facts exactly or fails closed
```

### Stage 2 Performance Layer

RVV plugin-local selected-body realization is one linear compiler step:

```text
selected explicit or pre-realized tcrv_rvv body
  + target capability
  + runtime SSA / ABI values
  + optional hints / policy / profile
    -> realized tcrv_rvv selected body
    -> faithful EmitC / intrinsic lowering
```

Hints/config/profile are not final products. If they affect generated code,
the RVV plugin must consume them into real `tcrv_rvv` structure:

```text
dynamic VL/setvl placement
legal SEW/LMUL/policy
memory forms
mask/tail materialization
register-pressure-safe unroll
prefetch/software-pipeline structure
accumulator/reduction layout
```

## Low-Precision No-Win Dispatch Preference Boundary

### 1. Scope / Trigger

Use this contract when a packed-i4 low-precision product-reduction route has
route-supported and correctness-executable RVV evidence, but the source-backed
same-target measurement classifies the current schedule as no-win or
regression. This is a production dispatch-preference policy boundary, not an
artifact-reporting convention.

### 2. Signatures

The RVV performance policy API consumes:

```c++
evaluateRVVLowPrecisionPerformancePolicy(selection, record, dispatchBoundary,
                                         context)
evaluateRVVLowPrecisionPerformancePolicy(selection, input, dispatchBoundary,
                                         context)
verifyRVVLowPrecisionPerformancePolicy(selection, record, dispatchBoundary,
                                       context)
```

where `dispatchBoundary` is an
`RVVLowPrecisionSelectedDispatchPolicyBoundary` carrying the selected RVV case,
fallback case, case/fallback policies, and selected-dispatch mirrors.

### 3. Contracts

- Route support and correctness execution may remain allowed for the accepted
  packed-i4 no-win/regression measurement.
- Performance selection, performance-win claims, and performance-preferred
  dispatch must remain denied unless a newer same-target measured-win record
  and matching provider maturity facts promote the policy.
- For the no-win/regression path, the selected-dispatch case policy,
  selected-dispatch case mirror, fallback policy, and fallback mirror must not
  contain a `performance-preferred` marker. Such a marker is treated as a
  route-support-as-performance-preference authority attempt.
- The policy decision must select the `correctness-fallback` dispatch path and
  carry `same-target-measurement-no-win-or-regression` as the denial reason.

### 4. Validation & Error Matrix

- Missing selected dispatch case or fallback facts -> policy boundary error.
- Selected dispatch case policy or case mirror contains
  `performance-preferred` while the measurement is no-win/regression -> policy
  boundary error before dispatch-preference acceptance.
- Fallback policy or fallback mirror contains `performance-preferred` while the
  measurement is no-win/regression -> policy boundary error.
- Route-supported packed-i4 selection with no accepted source-backed
  measurement -> correctness fallback may be resolved, but performance
  preference remains denied.
- Measured-win claim without matching provider maturity, remediation, and
  dispatch-preference facts -> policy boundary error.

### 5. Good/Base/Bad Cases

- Good: route-supported packed-i4 selected body plus source-backed no-win
  evidence -> `correctness-fallback`, route support preserved, performance
  preference denied.
- Good: source-backed measured-win evidence plus provider maturity fields
  updated to performance-mature -> `performance-preferred`.
- Base: stale measurement identity or stale sibling-route measurement ->
  conservative correctness fallback, no performance preference.
- Bad: selected RVV dispatch case mirror says `performance-preferred` only
  because the route is supported, while the accepted measurement remains
  no-win/regression.

### 6. Tests Required

- C++ policy tests must assert the accepted no-win path preserves route support
  and correctness execution while selecting `correctness-fallback`.
- C++ policy tests must mutate selected-dispatch case policy and case mirror
  with `performance-preferred` and assert fail-closed diagnostics.
- Target artifact/provider tests that carry packed-i4 low-precision resource
  facts must continue to consume the same policy API before accepting dispatch
  preference mirrors.

### 7. Wrong vs Correct

Wrong:

```text
route-supported packed-i4 selected body
  -> selected dispatch case mirror says performance-preferred
  -> dispatch treats RVV as performance-preferred despite no-win measurement
```

Correct:

```text
route-supported packed-i4 selected body
  -> source-backed same-target no-win/regression evidence
  -> RVV policy selects correctness-fallback
  -> selected-dispatch case/mirror cannot carry performance-preferred markers
```

## Runtime-Scalar Indexed Gather-MAcc-Scatter Route Contract

### 1. Scope / Trigger

Use this contract when a selected RVV body structurally represents the bounded
runtime-scalar compare + computed mask + masked indexed gather + masked MAcc +
masked indexed scatter path. This is a plugin-owned Stage 2 route-supported
contract. It is not a source-front-door route, not a route-id shortcut, and not
a Common EmitC semantic branch.

### 2. Body Shape

The selected explicit body must contain one `tcrv_rvv.with_vl` under one
selected RVV variant with:

```text
runtime ABI values:
  cmp_lhs, rhs_scalar, gather_src, payload, acc, index, dst, n

body:
  setvl/with_vl using n
  load cmp_lhs
  splat rhs_scalar
  load payload
  load acc
  load old dst passthrough
  index_load index
  compare cmp_lhs <= rhs_scalar
  masked_indexed_load gather_src, index, mask, old dst
  masked_macc mask, gathered, payload, acc
  masked_indexed_store dst, index, mask, macc result
```

The provider-owned operation kind is
`RuntimeScalarComputedMaskIndexedGatherMAccScatter`; the memory form is
`RuntimeScalarComputedMaskIndexedGatherMAccScatter`; the typed compute chain is
`tcrv_rvv.masked_indexed_load+tcrv_rvv.masked_macc+tcrv_rvv.masked_indexed_store`.

### 3. Contracts

- Runtime ABI order is exactly
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`.
- `cmp_lhs` binds `lhs-input-buffer`; `rhs_scalar` binds
  `rhs-scalar-value`; `gather_src` binds `source-input-buffer`; `payload`
  binds `dot-rhs-input-buffer`; `acc` binds `accumulator-input-buffer`;
  `index` binds `index-input-buffer`; `dst` binds `output-buffer`; `n` binds
  `runtime-element-count`.
- The compare predicate is runtime-scalar `sle`, and gather, MAcc, and scatter
  must consume the same compare-produced mask in the same VL scope.
- Indexed gather and indexed scatter both use the same index vector, element
  offsets, i32 indices, and the provider-derived index source mirror.
- Gather false lanes use the old destination vector as passthrough; scatter
  false lanes preserve the output buffer. The masked passthrough layout is
  `old-destination-vector-preserves-inactive-lanes`.
- The MAcc lhs is the indexed-gather result, rhs is the payload load, and
  accumulator is the accumulator load. The masked indexed store value must be
  the MAcc result.
- Provider route facts must carry the computed-mask memory family plan, the
  composite operand-binding plan, target leaf profile, provider support mirror,
  C type mapping, header declarations, masked indexed gather leaf, MAcc leaf,
  masked indexed scatter leaf, and ABI/header participation markers.

### 4. Validation & Error Matrix

- Missing or duplicate splat, compare, index load, masked indexed load, masked
  MAcc, masked indexed store, or the four required loads -> fail before route
  construction.
- Stale runtime ABI role, C name, C type, ordering, or ownership -> fail before
  construction protocol acceptance.
- Gather, MAcc, or scatter consuming a stale mask, index, VL token, operand, or
  destination -> fail in the RVV provider before Common EmitC.
- Missing inactive-lane, passthrough, index uniqueness, memory form,
  accumulator layout, result layout, or typed config fact -> fail closed with a
  targeted RVV provider diagnostic.
- A pre-realized multi-family composite is supported only when the
  plugin-local composite realization owner rewrites the bounded gather, MAcc,
  and scatter family bodies into the explicit realized body shape above before
  route construction. Missing, duplicate, incomplete, or stale pre-realized
  family bodies must fail closed with a named owner-boundary diagnostic before
  Common EmitC or target artifact export can claim executability.
- The composite owner candidate gate must claim any selected variant containing
  more than one pre-realized body from the bounded gather/MAcc/scatter family
  set, including incomplete or duplicate clusters, so those clusters fail at
  the named composite owner boundary. A single pre-realized gather, MAcc, or
  scatter family body remains a standalone family-owner candidate and must not
  be stolen by the composite owner.

### 5. Good/Base/Bad Cases

- Good: explicit selected composite body -> RVV provider composite facts ->
  one provider-built `TCRVEmitCLowerableRoute`.
- Good: pre-realized gather/MAcc/scatter family bodies -> RVV plugin-local
  composite realization owner -> explicit realized composite body -> provider
  composite facts -> one provider-built `TCRVEmitCLowerableRoute`.
- Base: future or incomplete pre-realized composite family combinations fail
  closed at the composite realization-owner boundary until a bounded owner
  proves the exact realized body shape and provider facts.
- Base: gather+MAcc without scatter, gather+scatter without MAcc, MAcc+scatter
  without gather, or duplicate gather/MAcc/scatter bodies are multi-body
  composite clusters; they fail at the composite owner boundary with
  gather/MAcc/scatter counts instead of falling through to a generic
  selected-body registry error.
- Bad: route id, artifact metadata, helper name, ABI string, or Common EmitC
  code infers gather, MAcc, scatter, dtype, mask, or intrinsic facts.

### 6. Tests Required

- Positive C++ coverage must assert operation kind, memory form, typed compute
  chain, ABI order, runtime ABI parameters, operand-binding summary, masked
  gather leaf, MAcc leaf, masked scatter leaf, inactive-lane contract, and
  passthrough layout.
- Negative C++ coverage must cover at least one stale structural fact, such as
  scatter not consuming the MAcc result.
- Pre-realized positive coverage must check that the named composite
  realization owner consumes the pre-realized gather, MAcc, and scatter family
  bodies, emits the explicit realized body shape, preserves ABI/runtime facts,
  and carries the composite plan id, typed compute chain, resource selection,
  header metadata, and target validation facts into the generated bundle.
- Pre-realized negative coverage must keep fail-closed owner-boundary checks
  for missing, duplicate, incomplete, stale, or unsupported composite family
  facts.
- Pre-realized negative C++ coverage must include at least one incomplete
  multi-body cluster and one duplicate-family cluster, and assert that the
  diagnostic names the composite owner boundary and reports the
  gather/MAcc/scatter family counts.

### 7. Wrong vs Correct

Wrong:

```text
route id or metadata says gather-macc-scatter
  -> Common EmitC or target export infers the composite route
```

Correct:

```text
selected typed tcrv_rvv body
  -> RVV-owned composite structure/fact validation
  -> provider-owned route contract
  -> neutral Common EmitC materialization
```

## Elementwise/Broadcast Route Provider Fact Contract

### 1. Scope / Trigger

Use this contract when the RVV provider constructs a
`TCRVEmitCLowerableRoute` for plain elementwise, masked elementwise, strided
elementwise, or scalar-broadcast elementwise routes. Route construction must
fail closed before Common EmitC sees the route if provider facts no longer
match the selected typed `tcrv_rvv` body and plugin-owned family plans.

### 2. Inputs

The provider-facts verifier must consume the same-analysis objects that feed
route construction:

```text
RVVSelectedBodyRouteAnalysis
RVVSelectedBodyRouteMaterializationFacts
RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
RVVSelectedBodyResidualRouteOperandBindingFacts
RVVSelectedBodyRouteStatementPlanOwnerSelection
```

The verifier may call RVV-owned route-control and family-plan validators. It
must not reconstruct arithmetic, broadcast, mask, dtype, VL, or ABI semantics
from route ids, artifact names, helper names, ABI strings, test names,
descriptors, source-front-door markers, or Common EmitC metadata.

### 3. Contracts

- Elementwise and scalar-broadcast consumers must carry the matching
  route-family plan from the same selected route analysis.
- Typed config facts must mirror the selected body facts for element type,
  SEW, LMUL, tail/mask policy, config contract, vector type/C type, setvl,
  load, store, and any route-specific mask or scalar-splat facts.
- Operand-binding facts must bind the required runtime ABI roles for vector
  inputs, scalar RHS, output, runtime element count, and stride operands when
  used.
- Masked elementwise routes must carry provider-derived compare, merge, mask
  type/C type, mask role/source, inactive-lane, and passthrough layout facts.
- Scalar-broadcast elementwise routes must carry the provider-derived scalar
  splat source and `rhs_scalar` ABI role.
- The migrated elementwise statement-plan owner must be selected, and required
  setvl/load/broadcast/compute/store leaves must appear in its statements
  before route construction.

### 4. Validation & Error Matrix

- Missing or stale route-family plan -> provider route construction error.
- Wrong SEW/LMUL/policy or stale typed materialization facts -> provider route
  construction error.
- Missing ordinary, masked, scalar, runtime-count, or stride ABI binding ->
  provider route construction error.
- Missing scalar broadcast source or stale scalar splat leaf -> provider route
  construction error.
- Missing mask provenance, compare, merge, inactive-lane, or passthrough fact
  for masked elementwise -> provider route construction error.
- Wrong statement owner or missing statement leaf -> provider route
  construction error.
- Non-elementwise route carrying stale elementwise/broadcast provider facts ->
  provider route construction error.

### 5. Good/Base/Bad Cases

- Good: selected typed elementwise body -> RVV family plan and materialization
  facts -> operand-binding facts -> migrated elementwise statement owner ->
  verifier -> provider-built `TCRVEmitCLowerableRoute`.
- Base: explicit and pre-realized fixtures may exercise the same verifier as
  long as selected typed-body facts remain the authority.
- Bad: route id, test name, helper name, artifact metadata, or Common EmitC
  mirror chooses the arithmetic kind, scalar source, mask behavior, dtype, or
  intrinsic spelling.

### 6. Tests Required

- Positive C++ coverage must include at least one plain elementwise, one masked
  elementwise, and one scalar-broadcast elementwise route through the verifier.
- Negative C++ coverage must fail closed for stale typed config facts, missing
  scalar broadcast source, stale mask provenance, missing operand binding, or
  wrong migrated statement owner.
- If emitted route statements, ABI order, headers, artifact export, or harness
  behavior changes, add a focused generated-bundle dry-run or real `ssh rvv`
  evidence for the affected route before claiming runtime correctness.

### 7. Wrong vs Correct

Wrong:

```text
route id or metadata says add
  -> Common EmitC or route construction infers add_i32m1 facts
  -> generated route
```

Correct:

```text
selected typed tcrv_rvv body
  -> RVV-owned family/materialization/operand/statement facts
  -> elementwise/broadcast provider-facts verifier
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral Common EmitC materialization
```

## Explicit Widening Product Reduce Dequant-Clamp Realization Boundary

### 1. Scope / Trigger

Use this contract when a selected RVV variant contains the bounded explicit
compound body:

```text
tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_body
```

The body represents the same fused Stage 2 chain as the already supported
pre-realized fixture:

```text
i8mf4 lhs/rhs loads
  -> widening_product
  -> standalone signed widening reduction with i32 scalar seed/carry
  -> f32 dequantize using runtime scale
  -> lower/upper f32 clamp
  -> f32 output store
```

This is an RVV selected-body realization boundary, not a direct route-entry
shortcut and not a Common EmitC semantic branch.

### 2. Signatures

The explicit body signature is:

```text
tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_body
  %lhs, %rhs, %acc, %scale, %lower_bound, %upper_bound, %out, %n
  {
    op_kind = "widening_product_reduce_dequant_clamp_f32",
    memory_form = "unit-stride-widening-product-reduce-dequant-clamp-f32",
    source_sew = 8, source_lmul = "mf4",
    product_sew = 16, product_lmul = "mf2",
    accumulator_sew = 32, accumulator_lmul = "m1",
    result_sew = 32, result_lmul = "m1",
    product_relation = "signed-i8mf4xi8mf4-to-i16mf2",
    product_reduction_chain_relation =
      "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32",
    dequant_relation = "signed-i32m1-to-f32m1-scale-f32",
    scale_role = "dequant-scale-value",
    lower_predicate_kind = "slt",
    upper_predicate_kind = "slt",
    bound_order = "lower-bound-before-upper-bound",
    select_layout = "clamp-lower-then-upper",
    policy = tail agnostic, mask agnostic
  }
```

The runtime ABI order remains:

```text
lhs, rhs, acc, scale, lower_bound, upper_bound, out, n
```

### 3. Contracts

- The explicit op carries typed/config/runtime facts only. Route ids, artifact
  names, ABI names, metadata mirrors, exact intrinsic spellings, descriptors,
  or test names must not authorize the route.
- The RVV contraction selected-body realization owner must consume the explicit
  op and create the realized vector-level structure before route-family facts,
  route-control facts, statement plans, or `TCRVEmitCLowerableRoute`
  construction.
- The realized structure must contain the legal RVV body sequence:
  `setvl`, `with_vl`, lhs/rhs loads, `widening_product`,
  `standalone_reduce`, `dequantize`, lower/upper splats, lower/upper compares,
  two selects, and store.
- The existing provider-derived fused route contract remains authority after
  realization. Common EmitC and target export consume provider output and
  metadata mirrors only.

### 4. Validation & Error Matrix

- Unsupported `op_kind` or `memory_form` -> fail before realization.
- Stale `route_id`, descriptor, source-front-door, artifact-name, or other
  authority metadata on the explicit body -> fail before realization.
- Source/product/accumulator/result SEW or LMUL mismatch -> fail before
  realization.
- Missing or stale accumulator role/layout, result layout, product relation,
  product-reduction relation, dequant relation, scale role, clamp predicates,
  bound order, select layout, or policy -> fail before provider facts.
- Runtime ABI role or C type mismatch for lhs, rhs, accumulator, scale,
  lower/upper bounds, output, or runtime `n` -> fail before provider facts.
- Target artifact mirrors for provider support, ABI order, operand binding,
  header/type mapping, product/reduction/dequant/clamp facts, or stale
  cross-family route facts disagree with the rebuilt provider route -> fail
  before bundle/header acceptance.

### 5. Good/Base/Bad Cases

- Good: explicit compound selected body -> RVV contraction realization owner ->
  realized `tcrv_rvv` vector-level body -> provider-derived fused route ->
  Common EmitC -> target artifact mirrors.
- Base: pre-realized fused fixtures keep using the same owner and realized
  structure; they are regression evidence, not the explicit-body achievement.
- Bad: route construction accepts the explicit op directly without materialized
  `setvl/load/widening_product/reduce/dequant/clamp/store` structure.
- Bad: Common EmitC or target export infers fused semantics from route id,
  artifact name, ABI string, test name, descriptor, or script option.

### 6. Tests Required

- A positive lit/FileCheck fixture must show the explicit body is removed and
  replaced by the realized vector-level structure before emission-plan and
  target header checks.
- The same fixture must check provider-derived route facts such as operation
  kind, typed compute chain, runtime ABI order, operand binding, route-family
  plan, provider mirror, type mapping, product-reduction relation, dequant
  relation, and clamp roles.
- Negative tests must cover stale or missing operation, dtype/config,
  accumulator/reduction, dequant scale, clamp bound, runtime ABI, policy, and
  route-authority facts.
- Existing pre-realized fused artifact/generated-bundle tests must remain
  passing when shared realization/provider/target code changes.

### 7. Wrong vs Correct

Wrong:

```text
typed_widening_product_reduce_dequant_clamp_f32_body
  -> provider route constructed from op name or route metadata
```

Correct:

```text
typed_widening_product_reduce_dequant_clamp_f32_body
  -> RVV contraction selected-body realization owner
  -> realized setvl/load/widening_product/reduce/dequant/clamp/store body
  -> provider-derived TCRVEmitCLowerableRoute
```

## Selected-Body Realization Owner Registry

### 1. Scope / Trigger

When a selected RVV variant contains a pre-realized `tcrv_rvv` body, the
production realization entrypoint must select one RVV plugin-local realization
owner before creating `setvl`, `with_vl`, loads, stores, compute, mask,
accumulator, reduction, memory, or segment operations. The owner registry sits
upstream of route-family analysis, route-control provider plans, and
`TCRVEmitCLowerableRoute` construction.

### 2. Signatures

The durable C++ surface is:

```c++
struct RVVSelectedBodyRealizationOwner {
  llvm::StringLiteral familyName;
  bool (*isConsumer)(mlir::Operation *);
  llvm::Expected<tcrv::rvv::WithVLOp> (*realize)(
      const VariantLoweringBoundaryRequest &, mlir::Operation *);
};

llvm::ArrayRef<RVVSelectedBodyRealizationOwner>
getRVVSelectedBodyRealizationOwners();

llvm::Expected<const RVVSelectedBodyRealizationOwner *>
getRVVSelectedBodyRealizationOwnerForBody(mlir::Operation *bodyOp,
                                          llvm::StringRef context);
```

`realizePreRealizedRVVSelectedBody(...)` must dispatch through this owner
registry. Direct route-entry realization is retired for active production route
construction. The production owner API must not expose an
`isRouteEntryConsumer` field, route-entry owner registry, route-entry variant
query, or `WithVLOp`-returning direct route-entry helper. A retained direct
route-entry API may only be an explicit retired diagnostic:

```c++
llvm::Error diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(
    const VariantLoweringBoundaryRequest &request);
```

This diagnostic function is negative-test inventory only and must not provide
executable route support.

### 3. Contracts

The registry owns only selected-body realization family classification and
realization dispatch. Each owner must have an explicit family name, a
structural consumer predicate, and a realization hook. Current active owner
entries must not be route-entry capable; pre-realized selected bodies are
consumed by the public selected lowering-boundary materialization producer
before route facts, route-control provider plans, statement plans,
`TCRVEmitCLowerableRoute`, common EmitC, or target artifact export.

Owner-family public predicates, realization hooks, owner-local result types,
and owner-local diagnostics belong in dedicated owner interfaces such as
`RVV<Element>SelectedBodyRealizationOwner.h`. The central selected-body
registry includes these owner interfaces directly. EmitC route-family planning
headers must not be used as carriers for selected-body owner predicates,
realization hooks, or owner-local result types; they remain responsible for
route-plan derivation, application, validation, mirror checks, and operand
binding APIs after realization.

Currently supported realization-owner families are:

- elementwise/compare-select;
- runtime scalar splat-store;
- runtime scalar computed-mask store;
- runtime scalar computed-mask load-store;
- reduction;
- standalone reduction;
- MAcc;
- computed-mask MAcc;
- contraction;
- widening conversion;
- base memory movement;
- computed-mask memory;
- segment2 memory.

Owner predicates may inspect typed pre-realized op classes and RVV-owned attrs
such as `op_kind`, `memory_form`, mask facts, layout facts, accumulator/result
layout, SEW, LMUL, policy, and runtime-control roles. The predicates must not
derive realization or route support from route ids, artifact names, test names,
ABI strings, descriptors, scripts, common EmitC, source-front-door markers, or
legacy i32 helper names.

All current realization-owner families are selected-boundary-only for
production route construction. This includes plain elementwise, compare/select,
computed-mask select, runtime scalar splat-store, runtime scalar computed-mask
store/load-store, reduction, standalone reduction, MAcc, computed-mask MAcc,
contraction, widening conversion, base memory movement, computed-mask memory,
and segment2 memory. A pre-realized body may belong to an owner family, but it
must be consumed through the public selected lowering-boundary producer before
provider facts are collected. A later task may reintroduce direct route-entry
support only by explicitly adding a new owner API with matching provider facts,
diagnostics, generated-bundle evidence, and real RVV evidence for executable
claims.

Segment2 production planning is selected-body route-family provider planning,
not segment2 route-entry family ownership. A stale `route_id`, artifact name,
script option, or mirror metadata cannot repair a mismatched typed `op_kind`,
memory form, segment count, mask facts, field roles, SEW/LMUL, policy, update
arithmetic, or runtime binding facts.

Tests for retired route-entry inventory must assert selected-boundary
realization, direct route-entry fail-closed behavior, and absence of production
route-entry API surfaces by bounded scan or compile-time API use. Positive
executable tests must prove the public selected lowering-boundary producer
consumes the selected pre-realized body before provider facts are collected and
that the realized typed `tcrv_rvv` body feeds the verified route-family plan and
statement-plan boundary.

Owner-local extraction must preserve the canonical selected-body construction
role sequence consumed by provider conformance. Moving a family out of the
central realization file may change code ownership, but it must not reorder the
realized runtime ABI roles, `setvl`, `with_vl`, loads, compares, mask
operations, passthrough loads, index/stride loads, stores, or family compute
ops relative to the route's construction route. For example, non-segment
computed-mask memory realization must keep compare input loads before the
passthrough/source/index materialization required by that route, then create
the compare mask before the masked load/store/strided/indexed operation. A
provider conformance error about a role operation carrying a different
construction order is a realization-owner bug, not permission to weaken the
provider route check.

### 4. Validation & Error Matrix

- Null body operation -> fail closed before owner selection.
- No matching owner for a pre-realized body -> fail closed before realization.
- More than one owner matches a body -> fail closed before realization.
- Owner has no realization hook -> fail closed before creating realized ops.
- Any direct route-entry request for a pre-realized selected RVV body -> fail
  closed with a retired route-entry diagnostic before provider route
  construction.
- Owner-specific validator rejects runtime AVL/VL source, ABI role/order,
  mem_window/imported value role, typed config, SEW/LMUL, policy,
  operation kind, memory form, mask/passthrough, accumulator/result layout, or
  selected capability facts -> fail closed before creating provider facts.
- Provider route analysis sees any pre-realized body after realization should
  have run -> fail closed with a selected-body realization diagnostic.
- Provider construction-role conformance reports a realized role operation in
  the wrong canonical order -> fix the owner-local materialization order; do
  not accept route ids, metadata, artifact names, ABI strings, or common EmitC
  as an override.

### 5. Good/Base/Bad Cases

- Good: selected RVV variant -> registry selects one realization owner ->
  owner validates typed/config/runtime/capability facts -> realized
  `tcrv_rvv` body -> route-family analysis -> route-control provider plan ->
  provider-built route.
- Base: every supported selected-body realization family realizes through the
  owner registry when the explicit selected lowering-boundary path is used.
- Bad: production route construction auto-realizes a pre-realized selected body
  through a direct route-entry fallback.
- Bad: route-entry realization keeps an active central allowlist or active
  owner predicate that can bypass the selected lowering-boundary producer.
- Bad: common EmitC, target artifacts, scripts, descriptors, route ids, or ABI
  strings infer realization family, dtype, SEW/LMUL, policy, or route support.

### 6. Tests Required

- C++ registry tests must assert owner names, owner count, and non-null
  consumer and realization hooks.
- Bounded production scans must assert no `isRouteEntryConsumer`,
  segment2 route-entry registry/query, or route-entry variant query remains in
  production headers or implementations.
- C++ or lit route-path tests must prove pre-realized bodies fail closed when
  they reach route construction without the public selected lowering-boundary
  producer.
- Negative tests must prove direct route-entry requests fail closed before
  provider route construction.
- Representative lit/FileCheck or generated-bundle dry-runs must prove
  pre-realized selected-boundary artifacts still consume the pre-realized body
  and explicit selected-body artifacts remain unaffected.
- Owner-extraction tests for a moved family must cover the realized typed op
  sequence that the provider consumes, not only owner registry membership.
- Changed-line scans must show no new name-, route-id-, metadata-,
  descriptor-, ABI-string-, script-, artifact-, common-EmitC-,
  source-front-door-, or legacy-i32-derived realization authority.

### 7. Wrong vs Correct

Wrong:

```text
direct route-entry path
  -> central string allowlist says op_kind looks supported
  -> route provider consumes pre-realized body or metadata
```

Correct:

```text
direct route-entry path
  -> selected-body realization owner registry selects exactly one owner
  -> owner validates typed/config/runtime/capability facts
  -> owner materializes realized tcrv_rvv body
  -> route-control registry and route provider consume realized structure
```

## Elementwise/Compare-Select Selected-Body Realization Boundary

### 1. Scope / Trigger

For statement-plan-backed elementwise arithmetic and compare/select
pre-realized selected bodies, RVV selected-body realization must be an
explicit RVV plugin-owned compiler boundary before route planning/provider
construction. The boundary applies when a selected `tcrv.exec` RVV variant
contains a pre-realized body for:

- plain, scalar-broadcast, strided, or masked elementwise arithmetic;
- plain compare-select;
- computed-mask select;
- runtime-scalar compare-select;
- runtime-scalar dual compare-mask-and-select.

Unrelated pre-realized families, such as reduction, memory, conversion, or
contraction, must receive an empty/not-applicable result from this boundary and
continue through their own realization path.

### 2. Signatures

The durable RVV plugin-local API is:

```c++
// include/TianChenRV/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.h
struct RVVElementwiseCompareSelectRealizationResult {
  tcrv::rvv::WithVLOp boundary;
  bool applies() const;
};

bool isPreRealizedRVVElementwiseCompareSelectClusterOp(mlir::Operation *op);

bool variantContainsPreRealizedRVVElementwiseCompareSelectSelectedBody(
    tcrv::exec::VariantOp variant);

llvm::Expected<RVVElementwiseCompareSelectRealizationResult>
realizePreRealizedRVVElementwiseCompareSelectCluster(
    const VariantLoweringBoundaryRequest &request,
    mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVElementwiseCompareSelectSelectedBody(
    const VariantLoweringBoundaryRequest &request);
```

The production `realizePreRealizedRVVSelectedBody(...)` path must call this
boundary before unrelated selected-body realization fallbacks.

### 3. Contracts

The boundary consumes only RVV-owned compiler facts:

- the selected `tcrv.exec.variant` and enclosing kernel from
  `VariantLoweringBoundaryRequest`;
- typed pre-realized `tcrv_rvv` body structure;
- runtime ABI SSA imports;
- SEW, LMUL, policy, memory form, predicate, mask/source/layout attrs carried
  by the pre-realized body;
- selected variant `requires` metadata;
- RVV runtime AVL/VL control helpers where runtime-scalar routes need them.

The boundary emits realized typed `tcrv_rvv` structure such as `setvl`,
`with_vl`, `load`, `splat`, `strided_load`, `compare`, `mask_and`,
`select`, `binary`, `masked_binary`, `store`, or `strided_store`, then erases
the consumed pre-realized body. It does not build
`TCRVEmitCLowerableRoute`; route construction remains provider-owned after
route analysis, materialization facts, operand-binding facts, and statement
plans are validated.

Elementwise/compare-select owner declarations must remain in the dedicated
owner header. The elementwise EmitC route-family planning header may expose
route operation predicates, route-family consumer predicates, route-plan
derivation/application/validation, route description mirror validation, and
operand-binding helpers only. If an implementation unit needs both
selected-body realization and route-planning helpers, it must include both
interfaces explicitly instead of making the route-family planning header
implicitly export owner APIs.

### 4. Validation & Error Matrix

- Null body operation -> fail closed before realization.
- Missing kernel or selected variant -> fail closed before realization.
- Body is outside the elementwise/compare-select cluster -> return
  not-applicable without mutation.
- Missing or wrong runtime ABI role -> fail closed with the logical operand
  name and expected role.
- Unsupported op kind, predicate, policy, SEW/LMUL, memory form, mask role,
  mask source, mask memory form, select layout, or stride operand shape ->
  fail closed before creating route/provider facts.
- Pre-realized cluster body is mixed with an already realized `setvl`,
  `with_vl`, or other realized `tcrv_rvv` route body op -> fail closed before
  route construction.
- Runtime-scalar route cannot derive an AVL/VL control plan -> fail closed
  before route construction.
- Route analysis/provider sees a pre-realized elementwise/compare-select body
  before facts are collected -> fail closed with a selected-body realization
  diagnostic. Route facts, operand-binding facts, statement plans, and
  provider-built routes consume realized `setvl` / `with_vl` structure only.

### 5. Good/Base/Bad Cases

- Good: selected RVV variant -> typed pre-realized elementwise/compare-select
  body -> `realizePreRealizedRVVElementwiseCompareSelectCluster` -> realized
  `tcrv_rvv` body -> RVV route analysis/materialization/operand-binding/
  statement-plan facts -> provider-built route.
- Base: selected RVV variant -> typed pre-realized reduction/memory/math body
  -> empty/not-applicable cluster result -> owning family realization path.
- Bad: provider/common EmitC sees a pre-realized body and invents missing
  loads, compares, masks, select layout, arithmetic, dtype, policy, schedule,
  or body shape.
- Bad: the cluster boundary treats route ids, artifact names, status fields,
  C ABI names, or intrinsic spellings as realization authority.

### 6. Tests Required

- C++ positive tests for at least one elementwise arithmetic pre-realized body
  and one compare/select pre-realized body showing the boundary creates
  realized `setvl`/`with_vl`/typed dataflow ops and the realized body still
  reaches the RVV provider route path.
- C++ not-applicable coverage for an unrelated pre-realized family.
- C++ fail-closed diagnostics for at least one missing, stale, or unsupported
  realization dependency before route construction.
- Representative lit/FileCheck coverage proving pre-realized and explicit
  selected-body elementwise/compare-select artifacts still pass.
- Bounded include/API scans must assert that the elementwise route-family
  planning header does not declare selected-body owner predicates,
  realization hooks, owner-local result types, or selected-body realization
  helpers.
- Bounded provider/common scan proving no semantic realization logic moved
  into route planning/provider construction or common EmitC.

### 7. Wrong vs Correct

Wrong:

```text
provider/common EmitC:
  sees typed_*_pre_realized_body or route metadata
  -> synthesizes setvl/load/compare/select/binary/store sequence
```

Correct:

```text
selected pre-realized elementwise/compare-select tcrv_rvv body
  -> RVV plugin-owned realization boundary
  -> realized typed tcrv_rvv body
  -> route analysis / materialization facts / operand-binding facts
  -> RVV-owned statement plan
  -> provider-built TCRVEmitCLowerableRoute
```

Wrong:

```text
RVVEmitCElementwiseRouteFamilyPlanOwners.h
  -> exports selected-body owner predicate / owner hook / owner-local result
```

Correct:

```text
RVVElementwiseSelectedBodyRealizationOwner.h
  -> exports selected-body owner predicate / owner hook / owner-local result
RVVEmitCElementwiseRouteFamilyPlanOwners.h
  -> exports route-family plan and mirror/operand-binding APIs only
```

## Elementwise/Compare-Select Route-Provider Facts Preflight

### 1. Scope / Trigger

When the RVV provider is about to build a `TCRVEmitCLowerableRoute` for
plain compare-select, computed-mask select, runtime-scalar compare-select, or
runtime-scalar dual compare-mask-and-select, it must prove that route
construction is consuming realized typed `tcrv_rvv` facts from the selected
body and RVV-owned provider plans. This preflight exists because parseable
selected-body IR, route-family mirror metadata, and generated artifact names
are not route authority.

### 2. Signatures

The durable provider-side contract is declared by
`RVVEmitCStatementPlanOwners.h` and implemented by the RVV compare/select
statement-plan owner path:

```c++
llvm::Error verifyRVVSelectedBodyCompareSelectRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyCompareSelectRouteStatementPlan
        &compareSelectStatementPlan,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this preflight after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(...)`, after
`getRVVSelectedBodyRouteMaterializationFacts(...)`, and after
`getRVVSelectedBodyElementwiseSelectRouteOperandBindingFacts(...)`, after the
RVV-owned compare/select statement plan has been built for compare/select
consumers, but before constructing the `TCRVEmitCLowerableRoute`.

### 3. Contracts

For compare/select consumers, the preflight must require:

- same-analysis route-family provider plans;
- typed RVV config facts for element type, SEW, LMUL, policy, VL/vector/mask
  C types, `setvl`, vector load, and store leaves;
- materialization facts that mirror the verified compare/select family plan for
  VL type, vector type, mask type, `setvl`, vector load, compare, select, and
  store leaves;
- same-analysis operand-binding facts for the exact selected sub-family;
- runtime AVL/VL and ABI role facts already checked by route-control and
  operand-binding providers.
- the RVV-owned compare/select statement plan for the exact selected
  sub-family, including route-control and mask/tail provider facts where
  the statement-plan boundary requires them.

The preflight must return success without changing behavior for unrelated RVV
families. It must not build statements, choose intrinsics, infer dtype/config,
read artifact metadata, consult route ids, or call selected-body owner hooks.

### 4. Validation & Error Matrix

- Unrelated operation -> return success and leave the route provider unchanged.
- Compare/select route lacks typed config facts -> fail closed before creating
  `TCRVEmitCLowerableRoute`.
- Plain compare-select route lacks the verified plain compare-select family
  plan, carries a computed-mask plan, or has stale materialization facts ->
  fail closed before route construction.
- Computed-mask/runtime-scalar select route lacks the verified computed-mask
  select family plan, carries a plain compare-select plan, or has stale
  materialization facts -> fail closed before route construction.
- Runtime-scalar compare-select route lacks the matching single or dual
  compare/select statement plan -> fail closed before route construction.
- Family-plan type/config facts disagree with selected typed RVV body/config
  facts -> fail closed before route construction.
- Operand-binding facts come from another analysis or do not match the selected
  sub-family, including single/dual runtime-scalar mismatch -> fail closed
  before route construction.
- RHS scalar splat leaf is missing or stale for runtime-scalar computed-mask
  select -> fail closed before route construction.
- Mask/tail policy provider facts in the statement plan are missing, stale, or
  not from the same computed-mask select family plan, typed config, selected
  target capability, and operand-binding plan -> fail closed before route
  construction.

### 5. Good/Base/Bad Cases

- Good: pre-realized `cmp_select` -> owner-local realization -> realized
  load/compare/select/store body -> plain compare-select family plan ->
  materialization facts -> operand-binding facts -> provider preflight ->
  `TCRVEmitCLowerableRoute`.
- Good: pre-realized `computed_mask_select` -> owner-local realization ->
  realized compare-mask/value-select body -> computed-mask select family plan
  -> materialization facts -> operand-binding facts -> provider preflight ->
  `TCRVEmitCLowerableRoute`.
- Good: pre-realized `runtime_scalar_cmp_select` or
  `runtime_scalar_dual_cmp_mask_and_select` -> owner-local realization ->
  realized runtime-scalar splat/compare/select or dual mask-and body ->
  computed-mask select family plan -> materialization facts ->
  operand-binding facts -> compare/select statement plan -> provider
  preflight -> `TCRVEmitCLowerableRoute`.
- Base: elementwise arithmetic, memory, reduction, contraction, conversion,
  segment2, and residual routes do not consume this preflight.
- Bad: route construction trusts `provider_supported_mirror`, route ids,
  artifact names, ABI strings, exact intrinsic spellings, or direct-route-entry
  claims instead of the realized typed facts and verified provider plans.

### 6. Tests Required

- C++ positive tests must call the preflight for plain compare-select and
  computed-mask select analyses, including the runtime-scalar computed-mask
  select subcases, before route construction.
- C++ negative tests must mutate typed config facts, materialization leaves,
  operand-binding family markers, runtime-scalar single/dual statement-plan
  markers, and mask/tail statement-plan provider facts and assert fail-closed
  diagnostics before `TCRVEmitCLowerableRoute` construction.
- Production route tests must still prove pre-realized `cmp_select` and
  `computed_mask_select` flow through selected lowering-boundary realization,
  provider route facts, statement plans, and target artifact generation.
- Generated-bundle or target artifact dry-run coverage must include the
  selected-boundary `cmp_select` and `computed_mask_select` paths.
- Bounded scans must show selected-body owner declarations remain out of
  `RVVEmitCElementwiseRouteFamilyPlanOwners.h` and that the provider preflight
  does not introduce source-front-door, descriptor, route-id, artifact-name,
  common-EmitC, exact-intrinsic, or legacy-i32 authority.

### 7. Wrong vs Correct

Wrong:

```text
RVVEmitCRouteProvider
  -> sees route mirrors / artifact metadata / intrinsic names
  -> builds compare/select TCRVEmitCLowerableRoute
```

Correct:

```text
realized typed compare/select tcrv_rvv body
  -> verified compare/select family plan
  -> route materialization facts + operand-binding facts
  -> verifyRVVSelectedBodyCompareSelectRouteProviderFacts
  -> provider-built TCRVEmitCLowerableRoute
```

## Retired Direct Route-Entry Diagnostic Inventory

### 1. Scope / Trigger

RVV production route/emission entries must require an already materialized
selected lowering boundary before provider route facts are collected. If a
selected variant still contains a pre-realized `tcrv_rvv` selected body at
emission-plan or EmitC route construction time, production must fail closed with
a targeted diagnostic. Production must not auto-realize the body through a
direct route-entry fallback.

The public selected lowering-boundary materialization path remains the only
active path that consumes selected pre-realized RVV bodies:

```text
selected pre-realized tcrv_rvv body
  -> public selected lowering-boundary materialization
  -> RVV owner registry realization
  -> realized typed tcrv_rvv body
  -> provider route facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
```

### 2. Signatures

The route-entry inventory diagnostic may remain for negative tests and future
explicit owner work:

```c++
llvm::Error diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(
    const VariantLoweringBoundaryRequest &request);
```

In active production, the retired diagnostic function must not create `setvl`,
`with_vl`, loads, stores, compute, masks, statement plans, or provider routes.
Production/public APIs must not expose a route-entry variant query or a
`WithVLOp`-returning direct route-entry realization helper.

`RVVExtensionPlugin::buildVariantEmissionPlan(...)` and
`RVVExtensionPlugin::buildVariantEmitCLowerableRoute(...)` must call a boundary
requirement helper that either returns the existing unique realized
`tcrv_rvv.with_vl` boundary or fails closed if a pre-realized body still needs
selected lowering-boundary materialization. They must not call
`diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(...)`.

### 3. Contracts

The retired route-entry inventory is not route support. It must not infer RVV
family, dtype, SEW, LMUL, policy, operation kind, memory form, mask/source
facts, statement shape, intrinsic spelling, route id, target artifact support,
or executable status from:

- typed pre-realized op names;
- route ids or artifact names;
- emission-plan or artifact metadata;
- descriptors or source-front-door markers;
- ABI strings, test names, scripts, or exact intrinsic spellings;
- legacy i32 helper names.

Any future reintroduction of direct route-entry support requires a new explicit
owner task that updates this spec, adds owner-scoped predicates, proves provider
facts and statement plans, adds generated-bundle evidence, and supplies real
`ssh rvv` evidence for executable claims.

### 4. Validation & Error Matrix

- Existing realized `setvl`/`with_vl` boundary and no pre-realized body ->
  production route construction may continue.
- Existing realized `setvl`/`with_vl` boundary mixed with any pre-realized body
  -> fail closed before route construction.
- No realized boundary and no pre-realized body -> preserve the structural
  missing-boundary diagnostic.
- No realized boundary and any pre-realized body -> fail closed with a
  diagnostic requiring public selected lowering-boundary materialization before
  provider route construction.
- Retired direct route-entry diagnostic invoked directly -> fail closed with a
  retired route-entry diagnostic.
- Provider route analysis sees any pre-realized body -> fail closed with a
  selected-body realization diagnostic; provider/common code must not invent
  missing typed structure.

### 5. Good/Base/Bad Cases

- Good: selected pre-realized RVV variant -> public selected
  lowering-boundary materialization -> owner validates and realizes typed body
  -> route-family facts -> statement plan -> provider-built route -> common
  EmitC.
- Good: selected RVV variant already contains exactly one valid realized
  `setvl/with_vl` boundary and no pre-realized body -> provider route facts are
  collected.
- Base: retained route-entry diagnostic exists only so negative tests can assert
  the retired diagnostic.
- Bad: production route construction calls route-entry realization to create a
  selected boundary.
- Bad: route provider sees `typed_*_pre_realized_body` and synthesizes
  setvl/load/store/compare/select/memory structure itself.
- Bad: common EmitC infers RVV dtype, operation kind, memory form, intrinsic
  spelling, or route support from route ids, status fields, artifact metadata,
  ABI names, or test names.

### 6. Tests Required

- C++ registry tests must assert owner names, owner count, and non-null
  consumer and realization hooks.
- C++ or lit tests must show production emission-plan and EmitC route
  construction fail closed when pre-realized bodies arrive without selected
  lowering-boundary materialization.
- C++ or lit tests must show the retained direct route-entry diagnostic is
  diagnostic-only.
- Regression coverage must show selected-boundary materialization and already
  realized selected-body artifacts still pass.
- Generated-bundle direct pre-realized CLI tests must remain fail-closed.
- Bounded scans must show common EmitC/export and provider/common code did not
  become semantic realization owners.
- Bounded scans must show production headers and implementations no longer
  expose route-entry owner predicates, segment2 route-entry registry/query
  surfaces, route-entry variant queries, or `WithVLOp`-returning direct
  route-entry helper APIs.

### 7. Wrong vs Correct

Wrong:

```text
production route construction
  -> sees pre-realized body
  -> invokes direct route-entry realization
  -> provider route facts
```

Correct:

```text
selected pre-realized tcrv_rvv body
  -> public selected lowering-boundary materialization
  -> realized typed tcrv_rvv body
  -> existing RVV facts / operand bindings / statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC materialization
```

### Stage 3: Other Families After RVV Maturity

IME, Offload, TensorExtLite, Template/Toy source-front-door examples, and
future plugin workflows are Stage3/later unless an explicit task says
otherwise. They must not displace Stage1/Stage2 RVV work.

## `with_vl` Boundary

`tcrv_rvv.with_vl` may remain a structural control boundary for VL-scoped RVV
body regions. Its selected-boundary attrs are legacy diagnostic mirrors only.
A mature RVV route must be recognized from typed body structure and
plugin-owned legality/realization, not from `status`, route mapping attrs,
artifact ids, or conformance labels on `with_vl`.

Good:

```text
typed body structure + plugin legality -> provider-built route
```

Bad:

```text
with_vl status/route attr -> route authority
```

## Source Front Doors

Source-front-door and source-artifact bundle pipelines are disabled by default
for current RVV Stage 1. A source-only RVV marker must fail closed unless an
explicit future task enables an opt-in path after mature typed body routes
exist.

No Stage1 spec or test may require positive RVV artifact generation from
source-front-door/source-artifact metadata. Positive RVV generated artifacts
must come from corrected typed `tcrv_rvv` bodies and plugin-built routes.

### Bounded Vector Binary Source Front Door

This is an explicit-only RVV source-front-door materializer. It remains only as
manual typed-body fixture/import scaffolding because it immediately converts a
bounded Vector-like source pattern into a selected `tcrv.exec` RVV variant
containing a typed generic `tcrv_rvv` body, then hands that body to the existing
RVV provider route path. It is not a default frontend, not a source-marker
route, and not eligible for the default source-artifact or generated-bundle
artifact path.

#### 1. Scope / Trigger

Use this contract only for the RVV plugin-owned bounded Vector binary
source-front-door materializer. The supported source shape is one source-only
module with:

```text
tcrv_rvv.source_front_door = "bounded_vector_source"
func(lhs: memref<?xi32>, rhs: memref<?xi32>, out: memref<?xi32>, n: index)
two unmasked unit-stride vector.transfer_read ops from lhs/rhs
one vector arith.addi, arith.subi, or arith.muli
one unmasked unit-stride vector.transfer_write to out
```

The materializer may derive only the binary operation kind from source IR. All
route support, ABI/header facts, C type mapping, and intrinsic selection must
come later from the typed `tcrv_rvv` body and RVV provider.

#### 2. Signatures

The public pass signature is:

```text
tcrv-rvv-materialize-vector-binary-source-front-door
```

The pass factory is:

```c++
createMaterializeRVVVectorBinarySourceFrontDoorPass()
```

The positive selected-body skeleton is:

```text
tcrv.exec.variant @rvv_vector_{add,sub,mul}
  %lhs = tcrv_rvv.runtime_abi_value role = "lhs-input-buffer"
  %rhs = tcrv_rvv.runtime_abi_value role = "rhs-input-buffer"
  %out = tcrv_rvv.runtime_abi_value role = "output-buffer"
  %n   = tcrv_rvv.runtime_abi_value role = "runtime-element-count"
  %vl  = tcrv_rvv.setvl %n {sew = 32, lmul = "m1", policy = ...}
  tcrv_rvv.with_vl %vl {
    %a = tcrv_rvv.load %lhs, %vl
    %b = tcrv_rvv.load %rhs, %vl
    %c = tcrv_rvv.binary {kind = "add"|"sub"|"mul"} %a, %b, %vl
    tcrv_rvv.store %out, %c, %vl
  }
```

#### 3. Contracts

- The source marker is only an explicit opt-in materialization boundary.
- The source function name is not route authority.
- The derived kind must come from the single supported source arith op, not
  from marker text, route ids, artifact names, test names, or ABI strings.
- The selected variant symbol may mirror the derived kind
  (`rvv_vector_add`, `rvv_vector_sub`, `rvv_vector_mul`), but the provider must
  still validate the typed body before route/export support.
- Dispatch policy may mirror the boundary as
  `rvv-vector-binary-source-front-door-case`; it must not be consumed as route
  support.
- Common EmitC/export must remain neutral and consume only provider-built route
  payloads.

#### 4. Validation & Error Matrix

- Missing or stale marker -> no materialization or fail closed.
- Non-source-only module with existing `tcrv.exec`, `tcrv_rvv`,
  `tcrv_toy`, or `tcrv_tensorext_lite` residue -> fail before body creation.
- Wrong ABI signature, dtype, rank, missing runtime `n`, masked transfers, or
  non-unit-stride transfers -> fail before body creation.
- Unsupported vector arith op such as `arith.andi` -> fail before body
  creation.
- Multiple candidate funcs, reads, writes, or binary ops -> fail before body
  creation.
- Any provider route failure after materialization -> fail in the existing RVV
  provider/EmitC/target validation path, not by source marker fallback.

#### 5. Good/Base/Bad Cases

- Good: explicit materializer invocation on an `arith.subi` source pattern ->
  selected `rvv_vector_sub` variant -> typed
  `tcrv_rvv.binary {kind = "sub"}` -> RVV provider route checks. Artifact
  evidence must use explicit typed selected-body fixtures, not this source
  marker as the front door.
- Base: `arith.addi` remains an ordinary instance of the generic binary
  materializer, not an add-only source-front-door owner.
- Bad: source marker -> route id -> target artifact without typed
  `tcrv_rvv.binary` body/provider validation.
- Bad: adding compare/select/reduction/MAcc/source-shape cases to this
  materializer instead of a separate typed-body/realization owner.

#### 6. Tests Required

- Positive `tcrv-opt` FileCheck must cover add/sub/mul materialization with
  derived `kind`, runtime ABI values, `setvl`, `load`, `binary`, and `store`.
- Emission-plan/header checks may prove the existing RVV provider consumes the
  typed body facts after explicit materializer invocation. They must not turn
  the source marker into a default artifact front door.
- Negative tests must cover stale selected-boundary residue and unsupported
  source arithmetic.
- Default source-artifact pipeline and generated-bundle evidence tests for this
  source marker must fail closed before target artifact export.
- Runtime correctness claims for non-add source paths require real `ssh rvv`
  evidence.

#### 7. Wrong vs Correct

Wrong:

```text
source marker / source op name
  -> route id
  -> Common EmitC picks intrinsic
  -> target artifact accepted
```

Correct:

```text
bounded Vector source IR
  -> RVV plugin materializes selected typed tcrv_rvv binary body
  -> RVV provider validates body/config/runtime facts
  -> provider-built TCRVEmitCLowerableRoute
  -> Common EmitC materializes neutral payload
  -> target artifact mirrors provider facts
```

### Bounded Vector Compare/Select Source Front Door

This is a second explicit-only RVV source-front-door materializer. It follows
the binary source-front-door pattern: the source marker may only invoke a
manual plugin-owned materializer, and the materializer must immediately create
a selected `tcrv.exec` RVV variant containing explicit realized typed
`tcrv_rvv` compare/select body facts. It is not eligible for the default
source-artifact or generated-bundle artifact path.

#### 1. Scope / Trigger

Use this contract only for the RVV plugin-owned bounded Vector compare/select
source-front-door materializer. The supported source shape is one source-only
module with:

```text
tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"
func(lhs: memref<?xi32>, rhs: memref<?xi32>, out: memref<?xi32>, n: index)
two unmasked unit-stride vector.transfer_read ops from lhs/rhs
one vector arith.cmpi predicate in {eq, slt, sle}
one arith.select using the compare mask, lhs when true, rhs when false
one unmasked unit-stride vector.transfer_write to out
```

The materializer may derive only the compare predicate and select layout from
source IR. Route support, ABI/header facts, C type mapping, compare/select
intrinsic spelling, mask type, and VL policy must come later from the typed
`tcrv_rvv` body and RVV provider.

#### 2. Signatures

The public pass signature is:

```text
tcrv-rvv-materialize-vector-compare-select-source-front-door
```

The pass factory is:

```c++
createMaterializeRVVVectorCompareSelectSourceFrontDoorPass()
```

The positive selected-body skeleton is:

```text
tcrv.exec.variant @rvv_vector_cmp_select_{predicate}
  %lhs = tcrv_rvv.runtime_abi_value role = "lhs-input-buffer"
  %rhs = tcrv_rvv.runtime_abi_value role = "rhs-input-buffer"
  %out = tcrv_rvv.runtime_abi_value role = "output-buffer"
  %n   = tcrv_rvv.runtime_abi_value role = "runtime-element-count"
  %vl  = tcrv_rvv.setvl %n {sew = 32, lmul = "m1", policy = ...}
  tcrv_rvv.with_vl %vl {
    %a = tcrv_rvv.load %lhs, %vl
    %b = tcrv_rvv.load %rhs, %vl
    %mask = tcrv_rvv.compare {kind = "eq"|"slt"|"sle"} %a, %b, %vl
    %selected = tcrv_rvv.select %mask, %a, %b, %vl
    tcrv_rvv.store %out, %selected, %vl
  }
```

#### 3. Contracts

- The source marker is only an explicit opt-in materialization boundary.
- The source function name is not route authority.
- The predicate must come from the single supported `arith.cmpi`, not from
  marker text, route ids, artifact names, test names, ABI strings, or Common
  EmitC metadata.
- The selected value layout must come from the `arith.select` operands and is
  bounded to `select-lhs-when-mask-else-rhs` for this contract.
- The selected variant symbol may mirror the derived predicate
  (`rvv_vector_cmp_select_eq`, `rvv_vector_cmp_select_slt`,
  `rvv_vector_cmp_select_sle`), but the provider must still validate the
  typed body before route/export support.
- Dispatch policy may mirror the boundary as
  `rvv-vector-compare-select-source-front-door-case`; it must not be consumed
  as route support.
- Common EmitC/export must remain neutral and consume only provider-built route
  payloads.
- A materializer must no-op for a known sibling marker so explicit pass
  sequencing can reach the matching pass. Unknown or stale RVV
  source-front-door markers must still fail closed.

#### 4. Validation & Error Matrix

- Missing marker -> no materialization.
- Known sibling marker -> no-op in this pass; the matching sibling pass owns
  the marker.
- Unknown marker or stale lowering seed marker -> fail closed before body
  creation.
- Non-source-only module with existing `tcrv.exec`, `tcrv_rvv`, `tcrv_toy`, or
  `tcrv_tensorext_lite` residue -> fail before body creation.
- Wrong ABI signature, dtype, rank, missing runtime `n`, masked transfers, or
  non-unit-stride transfers -> fail before body creation.
- Unsupported predicate such as `ne`, missing compare, or extra compare/select
  ops -> fail before body creation.
- Unsupported select layout, including rhs-when-true/lhs-when-false -> fail
  before body creation.
- Any provider route failure after materialization -> fail in the existing RVV
  provider/EmitC/target validation path, not by source marker fallback.

#### 5. Good/Base/Bad Cases

- Good: `arith.cmpi slt` + `arith.select %mask, %lhs, %rhs` source pattern ->
  selected `rvv_vector_cmp_select_slt` variant -> typed
  `tcrv_rvv.compare {kind = "slt"}` + `tcrv_rvv.select` -> RVV provider route
  -> bundle -> optional `ssh rvv` correctness evidence.
- Base: `eq` remains an ordinary instance of the compare/select materializer,
  not a route id, artifact name, or source marker authority.
- Bad: source marker -> route id -> target artifact without typed
  `tcrv_rvv.compare/select` body/provider validation.
- Bad: extending the binary materializer with compare/select cases or adding
  dtype/LMUL clone source front doors instead of a bounded typed-body owner.

#### 6. Tests Required

- Positive `tcrv-opt` FileCheck must cover `eq`, `slt`, and `sle`
  materialization with derived predicate, runtime ABI values, `setvl`, `load`,
  `compare`, `select`, and `store`.
- Source-artifact-front-door pipeline coverage must prove binary and
  compare/select sibling markers do not false-fail each other.
- Emission-plan/header/bundle checks must prove the existing RVV provider
  consumes the typed body facts and records marker/artifact metadata as
  mirror-only evidence.
- Negative tests must cover unsupported predicate, unsupported select layout,
  missing runtime `n`, stale TCRV residue, and stale lowering seed metadata.
- Runtime correctness claims require real `ssh rvv` evidence for at least one
  representative compare/select artifact.

#### 7. Wrong vs Correct

Wrong:

```text
source marker / source function name
  -> compare predicate or select layout inferred by route id
  -> Common EmitC picks compare/select intrinsics
  -> target artifact accepted
```

Correct:

```text
bounded Vector compare/select source IR
  -> RVV plugin materializes selected typed tcrv_rvv compare/select body
  -> RVV provider validates body/config/runtime/mask facts
  -> provider-built TCRVEmitCLowerableRoute
  -> Common EmitC materializes neutral payload
  -> target artifact mirrors provider facts
```

### RVV Vector Source Front-Door Family Registry

#### 1. Scope / Trigger

Use this contract when two or more RVV bounded Vector source-front-door
families are active. The registry is a plugin-local dispatcher over the active
opt-in source families. It owns marker classification and pass registration
for those families, but each family still owns its source-shape parser and
typed `tcrv_rvv` body materializer.

This registry currently covers exactly:

```text
bounded-vector-binary-source-front-door
bounded-vector-compare-select-source-front-door
bounded-vector-runtime-scalar-cmp-select-source-front-door
```

The legacy `rvv-i32m1` source-front-door pass remains an explicit-only
fail-closed guardrail, not an active family registry entry.

#### 2. Signatures

The public plugin-local registration surface is:

```c++
llvm::Error registerRVVVectorSourceFrontDoorFamilyPasses(
    llvm::StringRef ownerPlugin,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out);
```

Each active family registry entry must carry:

```text
family name
source marker value
pass argument
pass description
source function candidate description
selected variant prefix
runtime ABI purpose prefix
dispatch policy mirror
default source-artifact-front-door policy (currently explicit-only /
non-eligible for RVV)
family-local diagnostic hook
```

The active family materializer pass should also be consumed through the same
family-owner boundary. A family descriptor or equivalent owner object must
drive the pass argument, pass description, default artifact-front-door policy,
factory construction, and common source-only/fail-closed dispatch entry. The
owning family still keeps its parser and typed-body materializer local.

#### 3. Contracts

- RVV plugin registration must add active bounded Vector source-front-door
  family passes through `registerRVVVectorSourceFrontDoorFamilyPasses(...)`,
  not by manually duplicating sibling pass registrations in
  `RVVExtensionPlugin`.
- Active family pass factories must create materializer passes whose public
  pass argument and description come from the same family-owner entry used for
  registration. A new active family must not add another sibling pass class
  that hand-copies argument, description, dialect registration, marker dispatch,
  and source-only cleanup boilerplate when the common family-owned pass entry
  can consume the descriptor.
- The marker dispatcher must return no-op for missing markers.
- The marker dispatcher must return no-op for a known sibling marker so the
  source-artifact-front-door pipeline can reach the matching pass.
- Unknown RVV vector source-front-door markers must fail closed through the
  registry before source-shape parsing or body creation.
- Stale `tcrv_rvv.lowering_seed` metadata on a matched active family must fail
  closed through the registry before source-shape parsing or body creation.
- Selected variant symbols, scalar fallback symbols, runtime ABI purpose
  strings, and dispatch policy mirrors may be generated from family descriptor
  fields. They remain mirrors or construction names only; provider support
  still comes from the typed `tcrv_rvv` body and RVV provider route.
- Family-specific source-shape parsing, operation-kind derivation, compare
  predicate derivation, selected layout validation, and typed body
  materialization must stay in the owning family, not in the registry.

#### 4. Validation & Error Matrix

- Missing `tcrv_rvv.source_front_door` -> registry no-op.
- Known sibling marker -> registry no-op in this pass; matching family owns it.
- Unknown marker -> registry diagnostic naming the unknown marker and the
  registered RVV vector source-front-door markers.
- Matched active marker plus stale `tcrv_rvv.lowering_seed` -> registry
  diagnostic naming the active family and rejecting source-route authority.
- Matched marker plus stale TCRV residue, malformed ABI, malformed source
  shape, unsupported op, or unsupported select layout -> family-local
  diagnostic before body creation.
- Active family registry entry with a null or wrong pass factory -> C++ plugin
  registration test failure before source-artifact pipeline claims support.
- Registry-created pass factory returns a pass whose public argument or
  description disagrees with the family registration entry -> C++ plugin
  registration test failure before source-artifact pipeline claims support.

#### 5. Good/Base/Bad Cases

- Good: binary marker -> registry selects the binary family pass -> binary
  family parses `arith.addi/subi/muli` -> selected typed
  `tcrv_rvv.binary` body -> provider route.
- Good: compare/select marker -> binary pass no-ops, compare/select pass
  selects the compare/select family -> selected typed
  `tcrv_rvv.compare/select` body -> provider route.
- Base: adding a third bounded Vector source-front-door family means adding one
  descriptor entry plus one family-local parser/materializer; it must not add
  marker-specific branches to `RVVExtensionPlugin`.
- Base: an active family may keep a public compatibility factory such as
  `createMaterializeRVVVector...SourceFrontDoorPass()`, but that factory should
  instantiate the same family-owned materializer pass used by the registry.
- Bad: registry infers binary kind, compare predicate, dtype, ABI order,
  intrinsic spelling, or route support from the marker or family name.
- Bad: every new active family adds another pass class that manually duplicates
  pass argument, pass description, dialect population, no-op behavior, matched
  marker cleanup, and failure handling already owned by the registry boundary.
- Bad: Common EmitC or target export owns the source-front-door family
  registry.

#### 6. Tests Required

- C++ plugin test must prove the active family registry exposes exactly the
  registered active family pass arguments, explicit-only default artifact
  policy, ownership, and non-null factories consumed by the RVV plugin, and
  that registry-created materializer passes expose the same family-owned pass
  argument.
- Positive lit must continue to prove binary and compare/select source
  materialization and runtime-scalar compare/select materialization under
  explicit materializer invocation only.
- Negative lit must cover unknown marker, stale matched marker, malformed
  source shape, stale selected-boundary/TCRV residue, and legacy i32m1
  source-front-door fail-closed behavior.
- Source-artifact pipeline and generated-bundle dry-run tests for RVV
  source-front-door markers must fail closed before positive emission-plan or
  target artifact export.

#### 7. Wrong vs Correct

Wrong:

```text
RVVExtensionPlugin manually pushes each active source-front-door pass
  -> each pass reimplements sibling/unknown marker validation
  -> each pass class manually copies owner fields and run/cleanup boilerplate
  -> source marker text drifts into route or artifact authority
```

Correct:

```text
RVV plugin active source-front-door family registry
  -> pass registration, pass factory construction, and marker classification
     from family-owner entries
  -> owning family parses source shape and materializes typed tcrv_rvv body
  -> RVV provider validates body/config/runtime facts
  -> provider-built TCRVEmitCLowerableRoute
```

## Route-Family Owner Boundaries

When several active selected-body routes have been closed as one RVV
route-family cluster, the family ownership boundary should be explicit in
RVV planning/provider code rather than recreated as a manual list in the
central provider path.

For a family cluster, the planning-owned owner surface should expose:

```text
family name
consumer predicate over RVVSelectedBodyOperationKind
provider-plan verifier over RVVSelectedBodyRouteAnalysis
```

The production provider should consume an aggregate owner verifier for that
cluster. The aggregate verifier dispatches to each registered owner and must
fail closed when:

- a route-family consumer is missing its validated family plan;
- a non-consumer carries a stale family plan;
- route description mirrors, runtime ABI parameters, intrinsic/type/header
  mirrors, or `RouteOperandBindingPlan` closure no longer match the validated
  plan.

When multiple cluster or standalone family verifier boundaries are active, the
production selected-body RVV provider should consume one top-level owner
registry rather than manually sequencing verifier calls in the provider body.
That top-level registry is still dispatch structure only. Its durable API
shape is:

```text
getRVVSelectedBodyRouteFamilyProviderOwners()
isRVVSelectedBodyRouteFamilyProviderConsumer(RVVSelectedBodyOperationKind)
verifyRVVSelectedBodyRouteFamilyProviderPlans(RVVSelectedBodyRouteAnalysis, context)
```

Each top-level owner entry exposes the same contract as a cluster owner:
family name, consumer predicate, and provider-plan verifier. Entries may point
to aggregate cluster verifiers, such as memory or elementwise/select, or to a
standalone mature family verifier, such as runtime scalar splat-store or
widening conversion. The top-level verifier must preserve the underlying
missing-plan, stale-plan, mirror, runtime ABI, and binding-closure diagnostics;
it must not replace sub-family verification or merge family semantics.

The registry is dispatch/locality structure only. It must not merge family
semantics: mask producer/source facts, segment field roles, stride/index facts,
inactive-lane contracts, dtype/config facts, runtime ABI order, and intrinsic
mapping remain in the owning RVV family plan and verifier. Common EmitC and
target export may consume mirrors after route construction, but they must not
own the route-family registry or infer RVV semantics from it.

### Standalone Reduction Scalar Channel Boundary

#### 1. Scope / Trigger

Standalone reduction routes have two typed data channels when the RVV reduction
source LMUL differs from the scalar reduction result LMUL. The source/work
channel owns vector loads, masks, runtime-scalar compares, inactive-lane
neutralization, and the reduction source. The scalar accumulator/result channel
owns the seed, reduction accumulator/result vector, lane-0 scalar result
layout, and scalar output store.

#### 2. Signatures

The standalone-reduction family plan and route materialization facts must expose
distinct source and scalar-result fields, for example:

```c++
sourceVectorTypeName
sourceVectorCType
scalarCType
scalarResultVectorTypeName
scalarResultVectorCType
sourceSplatIntrinsic
```

The RVV provider also owns the operation-specific inactive neutral literal used
before masked standalone horizontal reduction:

```c++
llvm::StringRef getRVVSelectedBodyStandaloneReductionInactiveNeutralLiteral(
    RVVSelectedBodyOperationKind operation, std::int64_t sew);
```

For runtime-scalar computed-mask standalone reductions, the provider-owned
canonical route facts must also expose the per-SEW inactive neutral literals
that target artifact validation consumes when checking provider-built loop
statements:

```c++
llvm::StringRef inactiveNeutralLiteralSEW32;
llvm::StringRef inactiveNeutralLiteralSEW64;
```

#### 3. Contracts

The RVV selected-body realization must preserve the source SEW/LMUL on the
vector input/work channel and must materialize a scalar accumulator/result
channel with the same element dtype/SEW and the RVV scalar reduction result
LMUL required by the operation. For SEW32 LMUL m2 standalone reduce-add, the
source/work channel is `!tcrv_rvv.vector<i32, "m2">` and the scalar
accumulator/result channel is `!tcrv_rvv.vector<i32, "m1">`.

Provider construction must derive the reduction intrinsic, seed splat, result
store, source-channel splat, scalar C type, C type mapping, route operand
binding closure, and target artifact mirrors from these typed family-plan
facts. Common EmitC may only materialize the provider-built route.
Header/object exporters may mirror the source scalar C type and the source /
scalar-result C/vector types after rebuilding and validating the provider
route.

For computed-mask standalone reductions, inactive-lane neutralization belongs
to the source/work channel, not the scalar accumulator/result channel. The
inactive neutral splat must therefore use the source/work vector type and C
type, while scalar seed splats and scalar-result stores must use the
provider-derived `sourceSplatIntrinsic`; scalar seed splats and scalar-result
stores must use the scalar-result channel and lane-0 store VL. This remains
true when the source LMUL is m2 and the scalar-result LMUL is m1.

Target artifact validation must consume the RVV provider's inactive-neutral
literal helper when checking the provider-built inactive neutral splat. It must
not keep a target-local reduction-kind table that independently chooses `0`,
`INT32_MAX`, `INT32_MIN`, or their SEW64 equivalents. The helper result is
provider-owned route-family behavior; target validation only verifies that the
rebuilt route statements match it.

For the runtime-scalar computed-mask standalone reduction path, target
statement validation consumes the canonical route fact fields
`inactiveNeutralLiteralSEW32` / `inactiveNeutralLiteralSEW64` rather than a
target-local table. This makes stale add/min/max neutral literals fail closed
at the same provider-fact boundary as ABI order, mask source, accumulator
layout, and scalar-result boundary facts.

#### 4. Validation & Error Matrix

- Missing scalar accumulator role/layout -> fail closed before provider route
  construction.
- Missing scalar result role/layout -> fail closed before provider route
  construction.
- Source/work vector type equals the scalar result type only when RVV reduction
  semantics require that relation; otherwise mismatched or absent relation ->
  fail closed.
- Scalar result dtype/SEW differs from the source reduction dtype/SEW without an
  explicit widening/narrowing family plan -> fail closed.
- Runtime scalar mask binding, AVL/VL binding, or accumulator/output ABI order
  does not match the realized typed body -> fail closed.
- Computed-mask inactive neutral splat uses the scalar-result vector channel,
  a stale neutral literal, or a stale mask/merge operand instead of the
  validated source/work channel and mask facts -> fail closed.
- Target artifact validation derives the inactive neutral literal from a local
  route id, operation string, artifact name, intrinsic spelling, or target-only
  reduction-kind table instead of the RVV provider helper -> fail closed /
  refactor before accepting the artifact.
- Header/artifact metadata claims source or scalar-result types not present in
  the validated family plan -> fail closed as stale mirror metadata.
- Header/artifact metadata omits or changes the scalar C type or source splat
  leaf carried by the provider description -> fail closed as stale or missing
  mirror metadata.

#### 5. Good/Base/Bad Cases

Good: typed selected body carries source m2 and scalar-result m1 facts -> RVV
realization preserves the split -> provider builds an m2-to-m1 reduction route
-> artifact metadata mirrors the split after route validation.

Base: source LMUL m1 standalone reductions may have source and scalar-result
channels with the same LMUL, but that equality is still a typed family-plan
fact, not route-name authority.

Bad: route id, artifact name, ABI parameter string, exact intrinsic spelling,
test name, descriptor residue, or common EmitC code decides the reduction
source/result relation.

#### 6. Tests Required

- lit/FileCheck must cover selected-body realization, emission-plan/provider
  facts, header artifact mirrors, and direct route-entry fail-closed behavior.
- Generated-bundle evidence must prove `route_entry_realization: false`,
  `pre_realized_body_consumed: true`, source vector type/C type, scalar
  accumulator/result vector type/C type, runtime AVL/VL, mask/runtime-scalar
  binding, and scalar output ABI order.
- Runtime correctness claims require `ssh rvv` compile/run evidence over zero,
  one, exact-VL, tail, and stress counts with signed data and multiple runtime
  scalar thresholds.

#### 7. Wrong vs Correct

Wrong:

```text
standalone_reduce_add route name -> choose i32m1 reduction/result shape
```

Correct:

```text
typed source/result channels -> RVV family plan -> provider-built route
```

### Standalone Reduction Route-Provider Facts Preflight

#### 1. Scope / Trigger

When the RVV provider is about to build a `TCRVEmitCLowerableRoute` for a
plain, computed-mask, or runtime-scalar computed-mask standalone reduction, it
must prove that route construction is consuming the selected typed RVV body,
the validated standalone reduction family plan, provider materialization facts,
math operand-binding facts, route-control facts, and the RVV-owned standalone
reduction statement plan. This preflight is the provider-side closure for the
existing supported selected-body standalone reduction cases; it must not add new
reduction kinds, dtype/LMUL coverage, frontend coverage, or Common EmitC
semantics.

#### 2. Signatures

The durable provider-side contract is declared by
`RVVEmitCStatementPlanOwners.h` and implemented by the RVV residual
statement-plan owner path:

```c++
llvm::Error verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyStandaloneReductionRouteStatementPlan &statementPlan,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this preflight after the top-level
route-family verifier, after `getRVVSelectedBodyRouteMaterializationFacts(...)`,
after `getRVVSelectedBodyMathRouteOperandBindingFacts(...)`, and after the
RVV-owned standalone reduction statement plan has been built, but before
constructing the `TCRVEmitCLowerableRoute`. The standalone reduction verifier
must not be exposed as a generic `RVVEmitCRoutePlanning.h` public-surface hook.

#### 3. Contracts

For standalone reduction consumers, the preflight must require:

- the same-analysis `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`;
- the selected standalone reduction operation kind and reduction relation from
  the typed body/family plan, not from route ids or helper names;
- typed source/work channel facts for source SEW/LMUL, policy, VL C type,
  source vector type/C type, setvl leaf, source load leaf, memory form, and
  source splat/inactive-neutral leaves when required;
- scalar accumulator/result channel facts for seed, scalar result vector
  type/C type, scalar C type, lane-0 scalar result store, and scalar result ABI
  role;
- computed-mask facts for mask provenance, compare predicate, mask type/form,
  mask materialization, accumulation plan, inactive-lane neutralization, and
  zeroing/merge policy when the route is masked;
- runtime-scalar computed-mask facts for RHS scalar ABI role/type/name, scalar
  splat leaf, compare input binding, mask source, and runtime ABI order;
- materialization facts that mirror the family plan for required headers, VL
  type, source/scalar-result vector types, setvl, source load, reduction,
  scalar seed/splat, inactive neutralization, mask/compare leaves, and store;
- math operand-binding facts from the same selected analysis for the required
  source, accumulator, output, runtime scalar, mask, and runtime `n` roles;
- the RVV-owned route-control provider plan for the same typed config, selected
  target capability, runtime AVL/VL control plan, tail/mask policy, and runtime
  ABI order;
- the RVV-owned standalone reduction statement plan for the exact plain,
  computed-mask, or runtime-scalar computed-mask route family.

The preflight must return success without changing behavior for unrelated RVV
families only when they carry no stale standalone reduction facts. It must not
build statements, choose intrinsics, infer dtype/config, read artifact metadata,
consult route ids, or call selected-body owner hooks.

#### 4. Validation & Error Matrix

- Standalone reduction consumer lacks the family plan -> fail closed before
  `TCRVEmitCLowerableRoute` construction.
- Materialization facts do not point at the same selected-analysis family plan
  -> fail closed.
- Operation kind, reduction kind, neutral literal, intrinsic, or memory form is
  missing, stale, or derived from route id, helper name, metadata, artifact
  name, exact intrinsic spelling, or Common EmitC -> fail closed.
- Source SEW/LMUL, source vector type/C type, VL type, setvl, source load,
  source splat, memory form, or policy disagrees with the family plan -> fail
  closed.
- Scalar accumulator/result vector type/C type, scalar C type, seed leaf,
  scalar-result store, lane-0 result layout, or scalar result ABI role is absent
  or stale -> fail closed.
- Plain standalone reduction carries computed-mask, runtime-scalar,
  inactive-lane, or accumulation residue -> fail closed.
- Computed-mask standalone reduction is missing mask provenance, compare
  predicate, mask type/form, inactive-lane neutral/zeroing policy, or shared
  accumulation facts -> fail closed.
- Runtime-scalar computed-mask standalone reduction is missing RHS scalar ABI,
  scalar splat, compare binding, mask provenance, inactive-lane policy, or the
  runtime ABI order required by the family plan -> fail closed.
- Route-control provider plan is absent, from another analysis, or carries
  stale target capability, runtime AVL/VL, tail/mask policy, or runtime ABI
  mirrors -> fail closed.
- Math operand-binding facts are absent, from another analysis, missing required
  source/accumulator/output/runtime-scalar/runtime-`n` roles, or carry the wrong
  ABI role/order -> fail closed.
- Statement plan is absent, targets the wrong standalone reduction sub-family,
  points at another family plan, or lacks required setvl/load/mask/compare/
  inactive-neutral/reduce/store leaves -> fail closed.
- Non-standalone routes carry standalone reduction family-plan,
  materialization, binding, statement-plan, accumulation, mask, scalar-result,
  or route-family mirrors -> fail closed as stale provider facts.

#### 5. Good/Base/Bad Cases

Good: typed `standalone_reduce_add`, computed-mask standalone reduction, or
runtime-scalar computed-mask standalone reduction body -> standalone reduction
family plan -> materialization facts -> math operand bindings ->
route-control provider plan -> standalone reduction statement plan -> provider
preflight -> provider-built `TCRVEmitCLowerableRoute`.

Base: elementwise, memory, segment2, widening conversion, dequantization, MAcc,
and direct contraction routes do not consume this standalone reduction preflight
and must not carry standalone reduction facts.

Bad: provider construction trusts route ids, generated artifact names, ABI
strings, helper names, exact intrinsic spellings, status fields, source-front
door markers, descriptors, or Common EmitC choices instead of the selected
typed body and verified standalone reduction provider facts.

#### 6. Tests Required

- C++ positive tests or production provider tests for plain standalone
  reduction, computed-mask standalone reduction, and runtime-scalar
  computed-mask standalone reduction preflight success before route
  construction.
- C++ fail-closed tests for missing family plan, stale materialization facts,
  wrong source/scalar-result dtype or SEW/LMUL relation, wrong reduction kind,
  wrong scalar channel, wrong inactive-lane policy, stale mask provenance,
  missing RHS scalar splat, wrong operand binding or ABI role, wrong runtime ABI
  order, stale route description mirrors, stale statement leaves, and
  non-standalone routes carrying standalone reduction facts.
- Provider-route tests proving selected-body standalone reduction routes attach
  only provider-built statement plans to `TCRVEmitCLowerableRoute`.
- Focused FileCheck or `tcrv-opt` coverage for existing explicit and
  pre-realized selected-body standalone reduction fixtures.
- Generated-bundle or `ssh rvv` evidence only when emitted statements, runtime
  ABI, headers, bundle content, or runtime harness behavior changes.
- Bounded scans over touched planning/provider/test/target/script files for
  name-, route-id-, metadata-, descriptor-, ABI-string-, script-,
  artifact-name-, common-EmitC-, source-front-door-, exact-intrinsic-, or
  legacy-i32-derived route authority.

#### 7. Wrong vs Correct

Wrong:

```text
provider:
  sees standalone_reduce_add route id / artifact metadata / helper name
  -> builds TCRVEmitCLowerableRoute
```

Correct:

```text
typed standalone reduction tcrv_rvv body
  -> verified standalone reduction family plan
  -> materialization facts + math operand-binding facts
  -> route-control provider plan
  -> RVV-owned standalone reduction statement plan
  -> verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts
  -> provider-built TCRVEmitCLowerableRoute
```

The conversion dtype-policy cluster is a route-family owner boundary over:

- widening conversion routes, where the RVV widening conversion plan owns
  source/result dtype, source/result SEW/LMUL, conversion relation, source
  load, conversion intrinsic, store, runtime ABI, and route operand bindings;
- adjacent scalar-broadcast elementwise routes, where the scalar-broadcast
  elementwise plan owns the runtime scalar ABI value, RHS splat/broadcast,
  result dtype/config/policy, elementwise intrinsic, store, runtime ABI, and
  route operand bindings.

The conversion dtype-policy owner may share aggregate verification and target
artifact consumer checks across those two consumers, but it must not collapse
their semantics. A scalar-broadcast elementwise route must fail closed if it
carries stale widening source/destination/conversion facts. A widening
conversion route must fail closed if source/result dtype policy or conversion
relation facts are absent, stale, or derived from route ids, artifact names,
ABI strings, scripts, or exact intrinsic spellings rather than the validated
typed `tcrv_rvv` body and RVV family plan.

Target artifact/header/object consumers for this cluster must rebuild the
provider route from the selected variant and validate provider-derived headers,
type mappings, ABI mappings, statement leaves, route operand binding mirrors,
family-plan mirrors, source/result dtype policy facts, and conversion or
scalar-broadcast facts before export. Artifact metadata, route ids, generated
file names, or status fields are mirror-only and must not authorize export.

Required tests for a new or changed owner registry:

- C++ tests for registry membership, owner names, consumer classification,
  missing-plan diagnostics, stale-plan diagnostics, and aggregate verifier
  dispatch;
- for a top-level provider owner registry, C++ tests must also prove that
  production-owned entries include every active aggregate or standalone
  provider verifier boundary and that production construction calls the
  top-level aggregate verifier;
- representative lit/FileCheck coverage showing existing explicit or
  pre-realized selected-body artifacts still flow from typed `tcrv_rvv` bodies;
- at least one fail-closed typed-body mismatch diagnostic before
  materialization;
- runtime `ssh rvv` evidence only when emitted target sequence, ABI, or
  materialized operands changed.

### Widening Conversion Route-Provider Facts Preflight

#### 1. Scope / Trigger

When the RVV provider is about to build a `TCRVEmitCLowerableRoute` for a
widening conversion route, it must prove that route construction is consuming
the selected typed RVV body, the validated widening conversion family plan,
provider materialization facts, math operand-binding facts, route-control facts,
and the RVV-owned widening conversion statement plan. This preflight is the
provider-side closure for the existing supported selected-body widening
conversion cases; it must not add new dtype, LMUL, frontend, or conversion
coverage.

#### 2. Signatures

The durable provider-side contract is declared by
`RVVEmitCStatementPlanOwners.h` and implemented by the RVV residual
statement-plan owner path:

```c++
llvm::Error verifyRVVSelectedBodyWideningConversionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyWideningConversionRouteStatementPlan &statementPlan,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this preflight after the top-level
route-family verifier, after `getRVVSelectedBodyRouteMaterializationFacts(...)`,
after `getRVVSelectedBodyMathRouteOperandBindingFacts(...)`, and after the
RVV-owned widening conversion statement plan has been built, but before
constructing the `TCRVEmitCLowerableRoute`.

#### 3. Contracts

For widening conversion consumers, the preflight must require:

- the same-analysis `RVVSelectedBodyWideningConversionRouteFamilyPlan`;
- typed result config facts for result SEW/LMUL, policy, VL C type, result
  vector type/C type, setvl leaf, and store leaf;
- source dtype/channel facts from the widening conversion family plan for
  source SEW/LMUL, source vector type/C type, and source load leaf;
- conversion relation and conversion intrinsic from the family plan;
- materialization facts that mirror the family plan for required headers,
  VL type, source/result vector types, setvl, source load, conversion, and
  store;
- math operand-binding facts from the same selected analysis for `lhs`, `out`,
  and runtime `n`;
- the RVV-owned route-control provider plan for the same typed config, selected
  target capability, runtime AVL/VL control plan, and runtime ABI order;
- the RVV-owned widening conversion statement plan for the exact selected
  conversion form.

The preflight must return success without changing behavior for unrelated RVV
families only when they carry no stale widening conversion facts. It must not
build statements, choose intrinsics, infer dtype/config, read artifact metadata,
consult route ids, or call selected-body owner hooks.

#### 4. Validation & Error Matrix

- Widening conversion consumer lacks the family plan -> fail closed before
  `TCRVEmitCLowerableRoute` construction.
- Materialization facts do not point at the same selected-analysis family plan
  -> fail closed.
- Non-conversion route carries a widening conversion family plan,
  materialization flag, math binding flag, statement plan, family-plan mirror,
  or conversion relation -> fail closed as stale provider facts.
- Result typed config disagrees with result SEW/LMUL, policy, VL type, result
  vector type/C type, setvl, or store facts -> fail closed.
- Source SEW/LMUL, source vector type/C type, source load, conversion
  intrinsic, or conversion relation disagrees with the family plan -> fail
  closed.
- Route-control provider plan is absent, from another analysis, or carries
  stale runtime ABI, target capability, tail/mask policy, or runtime AVL/VL
  mirrors -> fail closed.
- Math operand-binding facts are absent, from another analysis, missing
  `lhs`/`out`/`n`, or carry the wrong ABI role/order -> fail closed.
- Statement plan is absent, targets the wrong widening form, points at another
  family plan, or lacks setvl/source-load/convert/store leaves -> fail closed.
- Route description mirrors or runtime ABI parameter mirrors disagree with the
  family plan -> fail closed.
- Stale mask, scalar-broadcast, standalone-reduction, accumulation,
  contraction, artifact-name, route-id, exact-intrinsic-as-authority, common
  EmitC, source-front-door, descriptor, or legacy-i32 residue attempts to
  authorize the route -> fail closed.

#### 5. Good/Base/Bad Cases

Good: typed `widen_i32_to_i64` or `widen_i16_to_i32` selected body ->
widening conversion family plan -> materialization facts -> math operand
bindings -> route-control provider plan -> widening conversion statement plan
-> provider preflight -> provider-built `TCRVEmitCLowerableRoute`.

Base: scalar-broadcast elementwise, reduction, memory, contraction, segment2,
and unrelated routes do not consume this preflight and must not carry widening
conversion facts.

Bad: provider construction trusts route ids, generated artifact names, ABI
strings, exact intrinsic spellings, script options, status fields, or common
EmitC choices instead of the selected typed body and verified widening
conversion provider facts.

#### 6. Tests Required

- C++ positive tests for `widen_i32_to_i64` and `widen_i16_to_i32` preflight
  success before route construction.
- C++ fail-closed tests for missing family plan, stale materialization facts,
  wrong source/result dtype or SEW/LMUL relation, wrong conversion relation,
  wrong operand binding or ABI role, wrong runtime ABI order, stale route
  description mirrors, stale statement leaves, and non-conversion routes
  carrying widening conversion facts.
- Provider-route tests proving the selected-body widening conversion route
  attaches only provider-built statement plans to `TCRVEmitCLowerableRoute`.
- Generated-bundle dry-run coverage for existing selected-boundary widening
  conversion fixtures.
- Runtime correctness evidence via `ssh rvv` when executable widening
  conversion correctness is claimed.
- Bounded scans over touched planning/provider/test/target/script files for
  name-, route-id-, metadata-, descriptor-, ABI-string-, script-,
  artifact-name-, common-EmitC-, source-front-door-, exact-intrinsic-, or
  legacy-i32-derived route authority.

#### 7. Wrong vs Correct

Wrong:

```text
provider:
  sees widen_i32_to_i64 route id / artifact name / intrinsic string
  -> builds TCRVEmitCLowerableRoute
```

Correct:

```text
typed widening conversion tcrv_rvv body
  -> verified widening conversion family plan
  -> materialization facts + math operand-binding facts
  -> route-control provider plan
  -> RVV-owned widening conversion statement plan
  -> verifyRVVSelectedBodyWideningConversionRouteProviderFacts
  -> provider-built TCRVEmitCLowerableRoute
```

### Dequantization Route-Provider Facts Preflight

#### 1. Scope / Trigger

When the RVV provider is about to build a `TCRVEmitCLowerableRoute` for the
standalone unit-stride `dequantize_i32_to_f32` route, it must prove that route
construction is consuming the selected typed RVV body, the validated
dequantization family plan, provider materialization facts, math
operand-binding facts, route-control facts, and the RVV-owned dequantization
statement plan. Product-reduction dequantization and dequant-clamp epilogues
are separate families and must not be accepted through this standalone
dequantization preflight.

#### 2. Signatures

The durable provider-side contract is declared by
`RVVEmitCStatementPlanOwners.h` and implemented by the RVV residual
statement-plan owner path:

```c++
llvm::Error verifyRVVSelectedBodyDequantizationRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyDequantizationRouteStatementPlan &statementPlan,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this preflight after the top-level
route-family verifier, after `getRVVSelectedBodyRouteMaterializationFacts(...)`,
after `getRVVSelectedBodyMathRouteOperandBindingFacts(...)`, and after the
RVV-owned dequantization statement plan has been built, but before constructing
the `TCRVEmitCLowerableRoute`.

#### 3. Contracts

For standalone dequantization consumers, the preflight must require:

- the same-analysis `RVVSelectedBodyDequantizationRouteFamilyPlan`;
- typed source config facts for source SEW/LMUL, policy, VL C type, source
  vector type/C type, setvl leaf, and source load leaf;
- result dtype facts for result element type, result SEW/LMUL, result vector
  type/C type, store leaf, and result name;
- dequantization kind/relation, convert intrinsic, scale intrinsic, and scale
  ABI role/type/name from the family plan;
- materialization facts that mirror the family plan for required headers, VL
  type, source/result vector types, setvl, source load, convert, runtime scale,
  and store;
- math operand-binding facts from the same selected analysis for `lhs`,
  `scale`, `out`, and runtime `n`;
- the RVV-owned route-control provider plan for the same typed config, selected
  target capability, runtime AVL/VL control plan, runtime ABI order
  `lhs,scale,out,n`, and provider/legality mirrors;
- the RVV-owned dequantization statement plan for the selected Gearbox schedule
  and statement leaves.

The preflight must return success without changing behavior for unrelated RVV
families only when they carry no stale standalone dequantization facts. It must
not build statements, choose intrinsics, infer dtype/config, read artifact
metadata, consult route ids, or call selected-body owner hooks.

#### 4. Validation & Error Matrix

- Standalone dequantization consumer lacks the family plan -> fail closed before
  `TCRVEmitCLowerableRoute` construction.
- Materialization facts do not point at the same selected-analysis family plan
  -> fail closed.
- Product-reduction dequantization carries standalone dequantization plan,
  materialization, binding, or statement facts -> fail closed because those
  facts belong to the contraction route-family plan.
- Dequant-clamp epilogue carries standalone dequantization plan,
  materialization, binding, or statement facts -> fail closed because those
  leaves belong to computed-mask select.
- Non-dequantization route carries a standalone dequantization family plan,
  materialization flag, math binding flag, statement plan, family-plan mirror,
  dequantization relation, convert/scale intrinsic, or scale ABI mirror -> fail
  closed as stale provider facts.
- Source typed config disagrees with source SEW/LMUL, policy, VL type, source
  vector type/C type, setvl, or source load facts -> fail closed.
- Result element type, result SEW/LMUL, result vector type/C type,
  dequantization kind/relation, convert intrinsic, scale intrinsic, scale
  role/type/name, store intrinsic, or result name disagrees with the family plan
  -> fail closed.
- Route-control provider plan is absent, from another analysis, or carries
  stale runtime ABI, target capability, tail/mask policy, runtime AVL/VL,
  provider, or legality mirrors -> fail closed.
- Math operand-binding facts are absent, from another analysis, missing
  `lhs`/`scale`/`out`/`n`, or carry the wrong ABI role/order -> fail closed.
- Statement plan is absent, targets the wrong dequantization form, points at
  another family plan, lacks setvl/source-load/convert/scale/store leaves, or
  carries a stale Gearbox unroll/schedule -> fail closed.
- Route description mirrors or runtime ABI parameter mirrors disagree with the
  family plan -> fail closed.
- Stale widening conversion, scalar-broadcast, mask, standalone-reduction,
  accumulation, contraction, artifact-name, route-id,
  exact-intrinsic-as-authority, common EmitC, source-front-door, descriptor, or
  legacy-i32 residue attempts to authorize the route -> fail closed.

#### 5. Good/Base/Bad Cases

Good: typed `dequantize_i32_to_f32` selected body -> dequantization family plan
-> materialization facts -> math operand bindings -> route-control provider
plan -> dequantization statement plan -> provider preflight -> provider-built
`TCRVEmitCLowerableRoute`.

Base: widening conversion, elementwise, reduction, memory, contraction,
segment2, product-reduction dequantization, and dequant-clamp epilogue routes
do not consume this standalone preflight and must not carry standalone
dequantization facts.

Bad: provider construction trusts route ids, generated artifact names, ABI
strings, exact intrinsic spellings, script options, status fields, or common
EmitC choices instead of the selected typed body and verified dequantization
provider facts.

#### 6. Tests Required

- C++ positive tests or production provider tests for
  `dequantize_i32_to_f32` preflight success before route construction.
- C++ fail-closed tests for missing family plan, stale materialization facts,
  wrong source/result dtype or SEW/LMUL relation, wrong dequantization relation,
  wrong scale role/type/name, wrong operand binding or ABI role, wrong runtime
  ABI order, stale route description mirrors, stale Gearbox schedule facts,
  stale statement leaves, and non-dequantization routes carrying standalone
  dequantization facts.
- Provider-route tests proving the selected-body standalone dequantization
  route attaches only provider-built statement plans to
  `TCRVEmitCLowerableRoute`.
- Generated-bundle dry-run or focused FileCheck coverage for existing
  selected-boundary dequantization fixtures.
- Runtime correctness evidence via `ssh rvv` when executable dequantization
  correctness is claimed.
- Bounded scans over touched planning/provider/test/target/script files for
  name-, route-id-, metadata-, descriptor-, ABI-string-, script-,
  artifact-name-, common-EmitC-, source-front-door-, exact-intrinsic-, or
  legacy-i32-derived route authority.

#### 7. Wrong vs Correct

Wrong:

```text
provider:
  sees dequantize_i32_to_f32 route id / artifact name / scale ABI name
  -> builds TCRVEmitCLowerableRoute
```

Correct:

```text
typed dequantization tcrv_rvv body
  -> verified dequantization family plan
  -> materialization facts + math operand-binding facts
  -> route-control provider plan
  -> RVV-owned dequantization statement plan
  -> verifyRVVSelectedBodyDequantizationRouteProviderFacts
  -> provider-built TCRVEmitCLowerableRoute
```

### Runtime Scalar Splat-Store Route-Provider Facts Preflight

#### 1. Scope / Trigger

When the RVV provider is about to build a `TCRVEmitCLowerableRoute` for a
runtime scalar splat-store route, it must prove that route construction is
consuming the selected typed RVV body, the validated runtime scalar
splat-store family plan, provider materialization facts, residual
operand-binding facts, route-control facts, and the RVV-owned runtime scalar
splat-store statement plan. This preflight is the provider-side closure for
the existing `runtime_scalar_splat_store` selected-body route; it must not add
new dtype, LMUL, frontend, runtime-scalar, or store-family coverage.

#### 2. Signatures

The durable provider-side contract is:

```c++
llvm::Error verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    const RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan
        &statementPlan,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this preflight after the top-level
route-family verifier, after `getRVVSelectedBodyRouteMaterializationFacts(...)`,
after `getRVVSelectedBodyResidualRouteOperandBindingFacts(...)`, and after the
RVV-owned runtime scalar splat-store statement plan has been built, but before
constructing the `TCRVEmitCLowerableRoute`.

#### 3. Contracts

For runtime scalar splat-store consumers, the preflight must require:

- the same-analysis `RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan`;
- typed config facts for SEW/LMUL, tail/mask policy, scalar C type, VL C type,
  result vector type/C type, setvl leaf, store leaf, and result vector name;
- scalar splat and store facts from the runtime scalar splat-store family plan;
- materialization facts that mirror the family plan for required headers,
  scalar type, VL type, result vector type/C type, setvl, runtime scalar
  splat, and store;
- residual operand-binding facts from the same selected analysis for
  `rhs_scalar`, `out`, and runtime `n`;
- the RVV-owned route-control provider plan for the same typed config, selected
  target capability, runtime AVL/VL control plan, policy, provider/legality
  mirrors, and runtime ABI order;
- the RVV-owned runtime scalar splat-store statement plan for the exact
  selected splat/store form.

The preflight must return success without changing behavior for unrelated RVV
families only when they carry no stale runtime scalar splat-store facts. It
must not build statements, choose intrinsics, infer dtype/config, read artifact
metadata, consult route ids, or call selected-body owner hooks.

#### 4. Validation & Error Matrix

- Runtime scalar splat-store consumer lacks the family plan -> fail closed
  before `TCRVEmitCLowerableRoute` construction.
- Materialization facts do not point at the same selected-analysis family plan
  -> fail closed.
- Non-runtime-splat-store route carries a runtime scalar splat-store family
  plan, materialization facts, residual binding flag, statement plan, or
  family-plan mirror -> fail closed as stale provider facts.
- Typed config disagrees with SEW/LMUL, policy, scalar type, VL type, result
  vector type/C type, setvl, scalar splat, or store facts -> fail closed.
- Current support may remain bounded to the typed SEW32/LMUL m1 instance, but
  that support must be accepted as an ordinary `runtime_scalar_splat_store`
  typed config instance. The old i32 operation mnemonic, route id, artifact
  name, helper namespace, or exact intrinsic spelling must not be the route
  authority.
- Materialization facts lack the scalar splat or store leaf, carry the wrong
  result vector type, or carry stale source, mask, conversion, reduction,
  accumulation, contraction, or segment facts as authority -> fail closed.
- Route-control provider plan is absent, from another analysis, or carries
  stale runtime ABI, target capability, tail/mask policy, or runtime AVL/VL
  mirrors -> fail closed.
- Residual operand-binding facts are absent, from another analysis, missing
  `rhs_scalar`/`out`/`n`, or carry the wrong ABI role/order -> fail closed.
- Statement plan is absent, targets the wrong splat-store form, points at
  another family plan, or lacks setvl/scalar-splat/store leaves -> fail closed.
- Route description mirrors or runtime ABI parameter mirrors disagree with the
  family plan -> fail closed.
- Stale scalar-broadcast, compare/select, widening conversion,
  standalone-reduction, accumulation, contraction, segment, artifact-name,
  route-id, exact-intrinsic-as-authority, common EmitC, source-front-door,
  descriptor, or legacy-i32 residue attempts to authorize the route -> fail
  closed.

#### 5. Good/Base/Bad Cases

Good: typed `runtime_scalar_splat_store` selected body -> runtime scalar
splat-store family plan -> materialization facts -> residual operand bindings
for `rhs_scalar`, `out`, and `n` -> route-control provider plan -> runtime
scalar splat-store statement plan -> provider preflight -> provider-built
`TCRVEmitCLowerableRoute`.

Good: current i32m1 splat-store support is derived from typed SEW32/LMUL m1
body/config facts and the common typed config profile, then mirrored through a
typed target leaf profile such as
`rvv-v1-typed-runtime-scalar-splat-store-leaf-profile.v1`.

Base: elementwise arithmetic, compare/select, widening conversion, reduction,
memory, contraction, segment2, and unrelated routes do not consume this
preflight and must not carry runtime scalar splat-store facts.

Bad: provider construction trusts route ids, generated artifact names, ABI
strings, exact intrinsic spellings, script options, status fields, common
EmitC choices, or source-front-door fixtures instead of the selected typed
body and verified runtime scalar splat-store provider facts.

#### 6. Tests Required

- C++ positive tests for `runtime_scalar_splat_store` preflight success before
  route construction.
- C++ fail-closed tests for missing family plan, stale materialization facts,
  wrong result vector type, missing scalar splat/store leaves, wrong residual
  binding or ABI role, wrong runtime ABI order, stale route-control provider
  facts, stale route description mirrors, stale statement leaves, and
  non-consumer routes carrying runtime scalar splat-store facts.
- Provider-route tests proving the selected-body runtime scalar splat-store
  route attaches only provider-built statement plans to
  `TCRVEmitCLowerableRoute`.
- Generated-bundle dry-run coverage for the existing selected-boundary
  runtime scalar splat-store fixture.
- Runtime correctness evidence via `ssh rvv` when executable runtime scalar
  splat-store correctness is claimed.
- Bounded scans over touched planning/provider/test/target/script files for
  name-, route-id-, metadata-, descriptor-, ABI-string-, script-,
  artifact-name-, common-EmitC-, source-front-door-, exact-intrinsic-, or
  legacy-i32-derived route authority.

#### 7. Wrong vs Correct

Wrong:

```text
provider:
  sees runtime_scalar_splat_store route id / artifact name / ABI string
  -> builds TCRVEmitCLowerableRoute
```

Correct:

```text
typed runtime_scalar_splat_store tcrv_rvv body
  -> verified runtime scalar splat-store family plan
  -> materialization facts + residual operand-binding facts
  -> route-control provider plan
  -> RVV-owned runtime scalar splat-store statement plan
  -> verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts
  -> provider-built TCRVEmitCLowerableRoute
```

## Route Materialization Facts Boundary

### 1. Scope / Trigger

After the top-level route-family provider verifier accepts a selected-body
analysis, RVV provider construction must obtain route materialization facts
from one RVV-owned boundary instead of recreating family-specific type,
header, intrinsic, mask/VL, and route-shape choices in the central provider
prelude.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyRouteMaterializationFacts>
getRVVSelectedBodyRouteMaterializationFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)` before
consuming these facts.

### 3. Contracts

`RVVSelectedBodyRouteMaterializationFacts` is RVV-local provider input. It may
carry:

- pointers to the validated family plans present in
  `RVVSelectedBodyRouteAnalysis`;
- route-shape booleans derived from those plans, such as widening conversion,
  contraction dot reduction, computed-mask accumulation, and standalone
  reduction;
- provider-owned required headers;
- VL/result/source/mask type names and C type strings;
- selected `setvl`, load, store, compute, compare, merge, splat, widening, and
  source-load intrinsic leaves.

These facts are consumed to build `TCRVEmitCLowerableRoute`. They are not
common EmitC facts, not artifact metadata authority, and not acceptance state.
They must remain derived from verified typed body/config/runtime facts and
their owning family plans.

### 4. Validation & Error Matrix

- Top-level verifier rejects missing plan for a route-family consumer ->
  materialization facts must not be consumed.
- Top-level verifier rejects stale plan on a non-consumer -> materialization
  facts must not be consumed.
- A route requiring computed-mask accumulation has no shared accumulation plan
  -> `getRVVSelectedBodyRouteMaterializationFacts` fails closed before
  provider materialization.
- Mirrors, route ids, artifact names, or status fields disagree with the
  verified family plan -> provider verification fails before common EmitC.

### 5. Good/Base/Bad Cases

- Good: typed `tcrv_rvv` body -> family plan verifier -> materialization facts
  -> provider-built `TCRVEmitCLowerableRoute`.
- Base: ordinary elementwise or memory route uses description/default facts
  only when no more-specific family plan owns that fact.
- Bad: central provider code manually chooses RVV C types or intrinsics from
  operation names, route ids, artifacts, status fields, or common EmitC logic.

### 6. Tests Required

- C++ tests for representative materialization facts across memory,
  elementwise/select, math, runtime scalar splat-store, and widening conversion
  families.
- C++ fail-closed diagnostic for a computed-mask accumulation route without
  the required shared accumulation plan.
- Representative lit/FileCheck coverage proving existing selected-body
  artifact materialization still passes.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider prelude:
  if operation is memory/select/math/... choose headers/types/intrinsics here
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyRouteMaterializationFacts
  -> provider-built TCRVEmitCLowerableRoute
```

## Route-Control Provider-Plan Boundary

### 1. Scope / Trigger

When a route-family provider consumes runtime AVL/VL, SEW/LMUL, tail policy,
mask policy, runtime ABI order, or selected capability/profile facts, those
facts must pass through one RVV-owned route-control provider plan before route
statement construction. This boundary sits after route-family provider-plan
verification and materialization facts, and before family statement plans are
attached to `TCRVEmitCLowerableRoute`.

The required consumers are mature ordinary elementwise arithmetic, masked
elementwise arithmetic, scalar-broadcast elementwise arithmetic, plain
compare/select, computed-mask select, widening conversion, non-segment
computed-mask memory, segment2 memory, base memory movement, standalone
reduction, scalar-broadcast MAcc, runtime scalar splat-store, computed-mask
accumulation MAcc, and contraction families. Contraction may remain a
direct-provider route family only after consuming this boundary through the
direct contraction route-provider owner. It must not keep central ad hoc
statement construction as active route authority. Other migrated families may
continue to use their existing family-local checks until they are explicitly
moved onto this boundary.

### 2. Signatures

The durable planning/provider API is:

```c++
struct RVVSelectedBodyRouteControlProviderOwner {
  family name;
  consumer predicate over RVVSelectedBodyEmitCRouteDescription;
  provider-plan builder over RVVSelectedBodyRouteAnalysis,
      RVVSelectedBodyRouteMaterializationFacts,
      RVVSelectedBodyRouteControlProviderPlan;
};

getRVVSelectedBodyRouteControlProviderOwners()
isRVVSelectedBodyRouteControlProviderConsumer(
    RVVSelectedBodyEmitCRouteDescription)

struct RVVSelectedBodyRouteAnalysis {
  RVVSelectedBodyTypedConfigFacts typedConfigFacts;
  RVVSelectedTargetCapabilityFacts selectedTargetCapabilityFacts;
  ...
};

llvm::Expected<RVVSelectedBodyRouteControlProviderPlan>
getRVVSelectedBodyRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    llvm::StringRef context);
```

Family provider code must call this boundary with materialization facts from
the same selected route analysis. The plan may be empty for unrelated route
families.

Route-control eligibility must be registry-owned. Every currently adopted
route-control family must appear exactly once in the RVV plugin-local owner
registry, and `getRVVSelectedBodyRouteControlProviderPlan(...)` must select the
owner through that registry before running family-specific plan/materialization
same-analysis checks. The registry may dispatch to family-specific builders; it
must not merge family semantics, infer route support from names/metadata, or
move route-control authority into common EmitC, artifacts, scripts, descriptors,
ABI strings, route ids, or legacy i32 spellings. If no owner matches, unrelated
routes receive an empty route-control plan. If multiple owners match, provider
construction fails closed before statement construction.

### 3. Contracts

`RVVSelectedBodyRouteControlProviderPlan` is RVV-local provider input. It may
carry:

- a pointer to the same-analysis typed config facts;
- a pointer to the same-analysis selected target capability facts;
- a pointer to the owning family `RVVRuntimeAVLVLControlPlan`;
- consumer flags for ordinary elementwise arithmetic, scalar-broadcast
  elementwise arithmetic, masked elementwise arithmetic, plain compare/select,
  computed-mask select, widening conversion, non-segment computed-mask memory,
  segment2 memory, base memory movement, standalone reduction,
  scalar-broadcast MAcc, runtime scalar splat-store, computed-mask
  accumulation MAcc, contraction, or future adopted families;
- mirror labels for control plan id, config contract, runtime VL contract,
  runtime AVL source, runtime ABI order, tail policy, mask policy, selected
  capability provider, and selected legality.

The plan validates provider-owned facts. It is not a common EmitC fact, not
artifact metadata, not a route-support state, and not a substitute for
route-family legality. Target artifacts and generated headers may mirror these
fields only after provider route construction.

### 4. Validation & Error Matrix

- Missing typed config facts for a control-plan consumer -> fail closed before
  provider route construction.
- Materialization facts from a different selected route analysis -> fail
  closed before route statement construction.
- Missing selected target capability facts -> fail closed before provider
  route construction.
- Selected target capability facts reject typed SEW, LMUL, tail policy, or
  mask policy -> fail closed before provider route construction.
- Runtime AVL/VL control plan is missing, has the wrong plan id, lacks runtime
  `n`, uses an unsupported SEW/LMUL/policy, or does not route through
  `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` -> fail closed.
- Route description mirrors disagree with the route-control plan for
  SEW/LMUL, tail/mask policy, runtime ABI order, config contract, runtime VL
  contract, VL op names, loop names, remaining AVL metadata, pointer advance
  metadata, bounded slice, or multi-VL support -> fail closed.
- Capability/provider mirror fields disagree with collected selected target
  capability facts -> fail closed; target metadata must not repair them.

### 5. Good/Base/Bad Cases

- Good: typed body/config/runtime facts + selected capability facts -> family
  runtime-control plan -> `RVVSelectedBodyRouteControlProviderPlan` ->
  family statement plan -> provider-built route.
- Good: typed plain compare/select `tcrv_rvv` body -> plain compare/select
  family plan verifier -> materialization facts -> elementwise/select
  operand-binding facts -> route-control provider plan -> compare/select
  statement plan -> provider-built route.
- Good: typed masked elementwise arithmetic `tcrv_rvv` body -> elementwise
  arithmetic family plan verifier with compare-produced mask and passthrough
  facts -> materialization facts -> residual operand-binding facts ->
  route-control provider plan -> elementwise arithmetic statement plan ->
  provider-built route.
- Good: typed computed-mask select `tcrv_rvv` body -> computed-mask select
  family plan verifier with vector/runtime-scalar producer-source facts ->
  materialization facts -> elementwise/select operand-binding facts ->
  route-control provider plan -> compare/select statement plan ->
  provider-built route.
- Good: typed widening conversion `tcrv_rvv` body -> widening conversion
  family plan verifier with source/result type policy and conversion-form facts
  -> materialization facts -> math operand-binding facts -> route-control
  provider plan -> widening conversion statement plan -> provider-built route.
- Good: typed non-segment computed-mask memory `tcrv_rvv` body ->
  computed-mask memory family plan verifier with mask-producer and memory-form
  facts -> materialization facts -> memory operand-binding facts ->
  route-control provider plan -> computed-mask memory statement plan ->
  provider-built route.
- Good: typed segment2 memory `tcrv_rvv` body -> plain segment2 or
  computed-mask memory family plan verifier with segment direction, memory-form,
  field-role, mask-producer where applicable, and runtime-control facts ->
  materialization facts -> memory operand-binding facts -> route-control
  provider plan -> segment2 memory statement plan -> provider-built route.
- Good: typed runtime-scalar computed-mask segment2 load `tcrv_rvv` body ->
  runtime scalar ABI binding for `lhs` and `rhs_scalar` -> RVV-local scalar
  splat/compare producer facts -> segment2 masked-load facts with interleaved
  `src`, old `out0`/`out1` passthrough loads, `out0`/`out1` stores, runtime
  `n`, and field ordering -> provider-built route. Missing or stale
  runtime-scalar producer source, ABI order, mask provenance, inactive-lane
  passthrough, or field role facts must fail closed before common EmitC or
  target artifact export.
- Good: typed runtime scalar splat-store `tcrv_rvv` body -> runtime scalar
  splat-store family plan verifier with scalar input, vector result, memory-form,
  splat/store leaves, and runtime-control facts -> materialization facts ->
  residual operand-binding facts -> route-control provider plan ->
  provider-built route.
- Good: typed computed-mask MAcc `tcrv_rvv` body -> computed-mask accumulation
  family plan verifier with vector/runtime-scalar mask-producer facts,
  accumulator/MAcc classification, inactive-lane contracts, and runtime-control
  facts -> materialization facts -> math operand-binding facts ->
  route-control provider plan -> computed-mask accumulation statement plan ->
  provider-built route.
- Good: typed contraction `tcrv_rvv` body -> contraction family plan verifier
  with widening MAcc or widening dot-reduction classification, accumulator and
  result layout, optional strided-input facts, optional computed-mask producer
  facts, and runtime-control facts -> materialization facts -> math
  operand-binding facts -> route-control provider plan -> direct contraction
  provider statement construction -> provider-built route.
- Base: migrated families not yet adopted by the route-control plan retain
  their family-local verifier checks and receive an empty route-control plan.
- Bad: a family statement plan reads tail policy, mask policy, runtime `n`,
  SEW/LMUL, or capability legality from route ids, artifact names, metadata,
  ABI strings, scripts, common EmitC, or target export.

### 6. Tests Required

- C++ positive coverage for every family that adopts the route-control plan,
  including masked elementwise arithmetic, asserting that typed config facts,
  selected target capability facts, and the family runtime-control plan are
  joined before statement planning.
- For direct-provider adopters such as contraction, C++ positive coverage must
  assert the route-control provider plan is consumed before direct statement
  construction, without requiring a wrapper-only statement-plan layer.
- C++ fail-closed diagnostics for stale materialization facts, missing or
  invalid runtime AVL/VL control facts, policy mismatches, and stale selected
  target capability facts.
- Provider-route tests proving the adopted family still attaches only
  provider-built statement plans to `TCRVEmitCLowerableRoute`.
- Representative lit/FileCheck or generated-header checks proving emitted
  metadata remains explicit mirror labels after provider route construction.
- Active-authority scan over touched planning/provider/test/target/script
  files for name-, route-id-, metadata-, descriptor-, ABI-string-, script-,
  artifact-, common-EmitC-, source-front-door-, or legacy-i32-derived
  AVL/VL/policy authority.

### 7. Wrong vs Correct

Wrong:

```text
base-memory provider:
  read tcrv_rvv.tail_policy metadata or route id
  -> choose setvl/policy/ABI loop facts
```

Correct:

```text
typed tcrv_rvv body/config/runtime facts
  + selected target capability facts
  + verified family runtime-control plan
  -> RVVSelectedBodyRouteControlProviderPlan
  -> family statement plan
  -> provider-built TCRVEmitCLowerableRoute
```

## Elementwise/Select Operand-Binding Facts Boundary

### 1. Scope / Trigger

For mature elementwise/select selected-body routes, `RVVEmitCRouteProvider`
must not locally recreate logical operand to materialized-use binding rules in
the central provider prelude. After provider-plan verification, the RVV
planning layer must expose one RVV-owned operand-binding facts boundary for
ordinary elementwise arithmetic, scalar-broadcast elementwise, plain
compare-select, computed-mask select, and runtime-scalar computed-mask select.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts>
getRVVSelectedBodyElementwiseSelectRouteOperandBindingFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);
```

`RVVEmitCRouteProvider` must consume these facts after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)` and after
obtaining route materialization facts for the same analysis.

### 3. Contracts

`RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts` is RVV-local
provider input. It may carry:

- the verified `RouteOperandBindingPlan` pointer used for binding closure;
- cluster and sub-family booleans for ordinary elementwise arithmetic,
  scalar-broadcast elementwise, plain compare-select, computed-mask select,
  runtime-scalar computed-mask select, and the single versus dual runtime
  scalar shape;
- bound runtime ABI parameters for lhs/rhs, optional secondary compare lhs and
  scalar rhs, true/false value inputs, output, and runtime element count.

These facts are consumed only to assign provider-local `RuntimeABIParameter`
pointers before building `TCRVEmitCLowerableRoute` statements. They are not
common EmitC facts, not artifact metadata, and not route support state.

### 4. Validation & Error Matrix

- A non elementwise/select route requests the boundary -> return default empty
  facts without changing unrelated family binding.
- An elementwise/select consumer has no matching family plan -> fail closed
  before statement construction.
- Runtime-scalar computed-mask select has a stale single/dual marker -> fail
  closed before statement construction.
- A required logical operand lacks the expected materialized use -> fail closed
  with the logical operand, materialized use, and route-family binding context.
- Header mirror, loop-control, setvl AVL, load, compute, select, or store uses
  are missing from the binding plan -> fail closed before common EmitC.

### 5. Good/Base/Bad Cases

- Good: typed `tcrv_rvv` body -> family plan verifier -> materialization facts
  -> elementwise/select operand-binding facts -> provider-built route.
- Base: non elementwise/select route families keep their own binding surface
  and receive empty elementwise/select binding facts.
- Bad: central provider branches manually enumerate the mature
  elementwise/select logical operand and materialized-use table.

### 6. Tests Required

- C++ tests for ordinary elementwise arithmetic, scalar-broadcast
  elementwise, plain compare-select, computed-mask select, and runtime-scalar
  computed-mask select binding facts.
- C++ fail-closed diagnostics for at least one missing or stale materialized
  use in the elementwise/select cluster.
- Representative lit/FileCheck coverage proving existing explicit or
  pre-realized elementwise/select selected-body artifacts still pass.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider prelude:
  if operation is cmp_select or scalar_broadcast_add, bind lhs/rhs/out here
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
  -> provider-built TCRVEmitCLowerableRoute
```

## Memory Operand-Binding Facts Boundary

### 1. Scope / Trigger

For mature selected-body memory movement routes, `RVVEmitCRouteProvider` must
not locally recreate logical operand to materialized-use binding rules in the
central provider prelude. After provider-plan verification, the RVV planning
layer must expose one RVV-owned operand-binding facts boundary for base
unit/strided/indexed/static-mask memory movement, computed-mask memory
movement, runtime-scalar computed-mask store/load-store, and segment2 memory
movement.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyMemoryRouteOperandBindingFacts>
getRVVSelectedBodyMemoryRouteOperandBindingFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);
```

`RVVEmitCRouteProvider` must consume these facts after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)` and after
obtaining route materialization facts for the same analysis.

### 3. Contracts

`RVVSelectedBodyMemoryRouteOperandBindingFacts` is RVV-local provider input. It
may carry:

- the verified `RouteOperandBindingPlan` pointer used for binding closure;
- memory cluster booleans for base memory movement, computed-mask memory,
  runtime-scalar computed-mask memory, plain segment2 memory, and segment2
  memory;
- bound runtime ABI parameters for compare lhs/rhs, runtime scalar threshold,
  source/destination windows, old-destination passthrough, index, mask, segment
  fields, runtime element count, source stride, and destination stride.

These facts are consumed only to assign provider-local `RuntimeABIParameter`
pointers before building `TCRVEmitCLowerableRoute` statements. They are not
common EmitC facts, not artifact metadata, and not route support state.

### 4. Validation & Error Matrix

- A non-memory route requests the boundary -> return default empty facts
  without changing unrelated family binding.
- A memory consumer has no matching base/computed-mask/segment2 family plan ->
  fail closed before statement construction.
- A memory consumer has a stale route-shape marker, such as wrong
  strided/indexed/static-mask/load-merge/store-only/segment2 direction marker ->
  fail closed before statement construction.
- A required logical operand lacks the expected materialized use -> fail closed
  with the logical operand, materialized use, and memory route-family binding
  context.
- Header mirror, loop-control, setvl AVL, source/destination, stride, index,
  mask, passthrough, or segment field uses are missing from the binding plan ->
  fail closed before common EmitC.

### 5. Good/Base/Bad Cases

- Good: typed memory `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> memory operand-binding facts -> provider-built route.
- Base: non-memory route families keep their own binding surface and receive
  empty memory binding facts.
- Bad: central provider branches manually enumerate the mature memory logical
  operand and materialized-use table.

### 6. Tests Required

- C++ tests for representative strided memory, indexed memory, static-mask/base
  memory, computed-mask memory, runtime-scalar computed-mask memory, and
  segment2 memory binding facts.
- C++ fail-closed diagnostics for at least one missing or stale materialized
  use in the memory cluster.
- Representative lit/FileCheck coverage proving existing explicit or
  pre-realized memory selected-body artifacts still pass.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider prelude:
  if operation is indexed_gather or computed_masked_strided_load, bind memory
  logical operands here
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyMemoryRouteOperandBindingFacts
  -> provider-built TCRVEmitCLowerableRoute
```

## Math Operand-Binding Facts Boundary

### 1. Scope / Trigger

For mature selected-body reduction, accumulation, contraction, and widening
routes, `RVVEmitCRouteProvider` must not locally recreate logical operand to
materialized-use binding rules in the central provider prelude. After
provider-plan verification, the RVV planning layer must expose one RVV-owned
operand-binding facts boundary for route families such as `reduce_add`,
standalone reductions, plain and computed-mask MAcc, widening MAcc, widening
conversion, widening dot-reduction, and computed-mask widening dot-reduction.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyMathRouteOperandBindingFacts>
getRVVSelectedBodyMathRouteOperandBindingFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);
```

`RVVEmitCRouteProvider` must consume these facts after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)` and after
obtaining route materialization facts for the same analysis. It may obtain the
elementwise/select and memory operand-binding facts in the same provider
prelude, but math route branches must consume math facts for their own
logical operands and materialized uses.

### 3. Contracts

`RVVSelectedBodyMathRouteOperandBindingFacts` is RVV-local provider input. It
may carry:

- the verified `RouteOperandBindingPlan` pointer used for binding closure;
- math cluster booleans for reduction, plain MAcc, computed-mask MAcc,
  standalone reduction, computed-mask standalone reduction, runtime-scalar
  computed-mask standalone reduction, widening MAcc, widening conversion, and
  widening dot-reduction shapes;
- bound runtime ABI parameters for source lhs/rhs, compare lhs/rhs or runtime
  scalar producers, payload dot lhs/rhs, source payload, accumulator or scalar
  seed, result/output, runtime element count, and strided dot input strides.

These facts are consumed only to assign provider-local `RuntimeABIParameter`
pointers before building `TCRVEmitCLowerableRoute` statements. They are not
common EmitC facts, not artifact metadata, and not route support state.

### 4. Validation & Error Matrix

- A non-math route requests the boundary -> return default empty facts without
  changing unrelated family binding.
- A math consumer has no matching family plan, such as contraction,
  standalone reduction, widening conversion, or shared computed-mask
  accumulation where required -> fail closed before statement construction.
- A math consumer has a stale route-shape marker, such as wrong
  computed-mask, strided-input, runtime-scalar producer, vector MAcc suffix, or
  scalar horizontal reduction suffix marker -> fail closed before statement
  construction.
- A required logical operand lacks the expected materialized use -> fail closed
  with the logical operand, materialized use, and math route-family binding
  context.
- Header mirror, loop-control, setvl AVL, source/payload, compare producer,
  accumulator/seed, width/config mirror, conversion relation, stride address,
  result, or store uses are missing from the binding plan -> fail closed before
  common EmitC.

### 5. Good/Base/Bad Cases

- Good: typed math `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> math operand-binding facts -> provider-built route.
- Base: non-math route families keep their own binding surface and receive
  empty math binding facts.
- Bad: central provider branches manually enumerate the mature reduction,
  MAcc, widening conversion, or widening dot logical operand and
  materialized-use table.

### 6. Tests Required

- C++ tests for representative `reduce_add`, MAcc, computed-mask MAcc or
  computed-mask accumulation, widening MAcc, widening conversion, and widening
  dot-reduction binding facts.
- C++ fail-closed diagnostics for at least one missing or stale materialized
  use in the math cluster.
- Representative lit/FileCheck coverage proving existing explicit or
  pre-realized math and widening selected-body artifacts still pass.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider prelude:
  if operation is macc_add or widening_dot_reduce, bind math logical operands here
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyMathRouteOperandBindingFacts
  -> provider-built TCRVEmitCLowerableRoute
```

## Residual Operand-Binding Facts Boundary

### 1. Scope / Trigger

For the remaining mature selected-body routes that are not owned by the
elementwise/select, memory, or math operand-binding facts boundaries,
`RVVEmitCRouteProvider` must not locally recreate logical operand to
materialized-use binding rules in the central provider prelude. After
provider-plan verification, the RVV planning layer must expose one RVV-owned
residual operand-binding facts boundary for masked elementwise arithmetic,
strided elementwise add, and runtime scalar splat-store.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyResidualRouteOperandBindingFacts>
getRVVSelectedBodyResidualRouteOperandBindingFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);
```

`RVVEmitCRouteProvider` must consume these facts after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the elementwise/select,
memory, and math operand-binding facts for the same analysis.

### 3. Contracts

`RVVSelectedBodyResidualRouteOperandBindingFacts` is RVV-local provider input.
It may carry:

- the verified `RouteOperandBindingPlan` pointer used for binding closure;
- residual route booleans for masked elementwise arithmetic, strided
  elementwise add, and runtime scalar splat-store;
- bound runtime ABI parameters for lhs/rhs/output, runtime scalar RHS,
  runtime element count, and lhs/rhs/output stride roles.

These facts are consumed only to assign provider-local `RuntimeABIParameter`
pointers before building `TCRVEmitCLowerableRoute` statements. They are not
common EmitC facts, not artifact metadata, and not route support state.

### 4. Validation & Error Matrix

- A non-residual route requests the boundary -> return default empty facts
  without changing unrelated family binding.
- A residual route has no matching elementwise arithmetic or runtime scalar
  splat-store family plan -> fail closed before statement construction.
- A residual route has a stale route-shape marker, such as masked versus
  strided elementwise classification or runtime scalar splat-store memory form
  mismatch -> fail closed before statement construction.
- A required logical operand lacks the expected materialized use -> fail closed
  with the logical operand, materialized use, and residual binding context.
- Header mirror, loop-control, setvl AVL, compare producer, masked
  passthrough, stride address, scalar splat, store, or runtime count uses are
  missing from the binding plan -> fail closed before common EmitC.

### 5. Good/Base/Bad Cases

- Good: typed residual `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> residual operand-binding facts -> provider-built
  route.
- Base: elementwise/select, memory, and math route families keep their own
  binding surfaces and receive empty residual binding facts.
- Bad: central provider branches manually enumerate the residual
  masked/strided/splat logical operand and materialized-use tables.

### 6. Tests Required

- C++ tests for masked elementwise arithmetic, strided elementwise add, and
  runtime scalar splat-store binding facts.
- C++ fail-closed diagnostics for at least one missing or stale materialized
  use in the residual cluster.
- Representative lit/FileCheck coverage proving existing explicit or
  pre-realized residual selected-body artifacts still pass.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider prelude:
  if operation is masked_add, strided_add, or runtime_scalar_splat_store,
  bind residual logical operands here
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyResidualRouteOperandBindingFacts
  -> provider-built TCRVEmitCLowerableRoute
```

## Elementwise Arithmetic Statement-Plan Boundary

### 1. Scope / Trigger

For mature selected-body elementwise arithmetic routes,
`RVVEmitCRouteProvider` must not locally recreate the route statement sequence
from operation names, ABI strings, or memory-form branches after RVV-owned
family plans, materialization facts, and operand-binding facts have been
validated. The RVV planning layer must expose one RVV-owned statement-plan
boundary for ordinary `Add`/`Sub`/`Mul`, scalar-broadcast `Add`/`Sub`/`Mul`,
masked `Add`/`Sub`/`Mul`, and strided `Add` where those routes are
production-active. Ordinary `Add`/`Sub`/`Mul`, masked `Add`/`Sub`/`Mul`, and
scalar-broadcast `Add`/`Sub`/`Mul` must consume the shared route-control
provider plan before this statement plan builds the setvl/load/compare/
broadcast-or-load/compute/merge/store sequence.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. It must not duplicate
the included elementwise arithmetic loop/setvl/load/compute/merge/store
sequence in the central provider path.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyElementwiseArithmeticRouteStatementPlan>
getRVVSelectedBodyElementwiseArithmeticRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the
elementwise/select and residual operand-binding facts for the same analysis.
Non-consumer route families receive an empty/default statement plan.

### 3. Contracts

`RVVSelectedBodyElementwiseArithmeticRouteStatementPlan` is RVV-local provider
input. It may carry:

- pointers to the verified elementwise arithmetic and scalar-broadcast family
  plans that justify the statement sequence;
- sub-family booleans for ordinary elementwise arithmetic,
  scalar-broadcast elementwise arithmetic, masked elementwise arithmetic, and
  strided elementwise add;
- provider-ready `TCRVEmitCCallOpaqueStep` entries for full-chunk `setvl`;
- one provider-ready `TCRVEmitCForLoop` with loop `setvl`, load/broadcast or
  strided-load steps, compute/compare/merge steps where needed, and the store
  step.

The plan must be derived only from verified typed body/config/runtime facts,
route materialization facts, and RVV-owned operand-binding facts. It is not a
common EmitC fact, not artifact metadata, not an acceptance/status mirror, and
not a route-support declaration by itself.

For elementwise route-family leaf derivation, `RVVSelectedBodyTypedConfigFacts`
must carry the typed `setvl`, unit load, unit store, strided load/store, and
scalar-splat leaves needed by already supported elementwise sub-families.
Scalar-broadcast elementwise must consume the typed config scalar-splat leaf
instead of choosing `vmv` spelling from an owner-local `i32m1` template.
Strided elementwise must consume typed config strided load/store leaves.
Arithmetic, compare, and masked-merge leaves may be composed by the RVV owner
from operation kind plus typed element/SEW/LMUL facts, but exact intrinsic
spellings are derived outputs only and must fail closed when the typed
combination is unsupported.

### 4. Validation & Error Matrix

- A non elementwise-arithmetic route requests the boundary -> return an empty
  plan without changing unrelated route-family behavior.
- An included elementwise arithmetic route has no matching verified family
  plan -> fail closed before route statement construction.
- An included ordinary, masked, or scalar-broadcast elementwise arithmetic
  route lacks the shared route-control provider plan, carries stale
  materialization facts from another selected route analysis, has wrong runtime
  AVL/VL control facts, or has SEW/LMUL/policy/capability mirrors that
  disagree with the typed body/config and selected target facts -> fail closed
  before route statement construction.
- An included route lacks the required operand-binding facts from the
  elementwise/select or residual facts boundary -> fail closed before route
  statement construction.
- Required ABI roles such as `lhs`, `rhs`, `rhs_scalar`, `out`, runtime count,
  or stride roles are absent -> fail closed with the logical operand name and
  operation/memory-form context.
- Required materialization leaves such as `setvl`, load, scalar broadcast,
  compute, compare, masked merge, strided load, or strided store are absent ->
  fail closed before common EmitC.
- Scalar-broadcast, strided, or masked elementwise plan leaves disagree with
  `RVVSelectedBodyTypedConfigFacts` for scalar splat, strided load/store,
  vector type/C type, mask type/C type, setvl, unit load, or store -> fail
  closed before provider route construction.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before common
  EmitC.

### 5. Good/Base/Bad Cases

- Good: typed ordinary elementwise arithmetic `tcrv_rvv` body -> family plan
  verifier -> materialization facts -> elementwise/select operand-binding
  facts -> route-control provider plan -> RVV-owned statement plan ->
  provider-built route.
- Good: typed scalar-broadcast elementwise arithmetic `tcrv_rvv` body ->
  scalar-broadcast family plan verifier -> materialization facts ->
  elementwise/select operand-binding facts -> route-control provider plan ->
  RVV-owned statement plan with scalar splat leaf -> provider-built route.
- Good: typed masked elementwise arithmetic `tcrv_rvv` body -> family plan
  verifier with compare-produced mask and passthrough facts ->
  materialization facts -> residual operand-binding facts -> route-control
  provider plan -> RVV-owned statement plan -> provider-built route.
- Good: typed strided elementwise arithmetic `tcrv_rvv` body -> family plan
  verifier -> materialization facts -> residual operand-binding facts ->
  RVV-owned statement plan -> provider-built route.
- Base: compare/select, memory, math, runtime scalar splat-store, and future
  families keep their own statement construction surfaces and receive an empty
  elementwise arithmetic statement plan.
- Bad: central provider code branches on `Add`, `MaskedAdd`,
  `RHSScalarBroadcast`, or `StridedLoadStore` to rebuild the included
  elementwise arithmetic setvl/load/compute/merge/store sequence after the
  statement-plan boundary exists.

### 6. Tests Required

- C++ tests for positive statement-plan construction and provider consumption
  for ordinary `Add`/`Sub`/`Mul`, scalar-broadcast `Add`/`Sub`/`Mul`, masked
  `Add`/`Sub`/`Mul`, and strided `Add`, including ordinary and
  scalar-broadcast elementwise route-control provider-plan consumption.
- C++ fail-closed diagnostics for at least one missing or stale statement-plan
  dependency before route statement construction, plus stale ordinary and
  masked elementwise route-control analysis/runtime AVL/policy/capability
  facts.
- C++ default/empty-plan coverage for unrelated route families.
- Representative lit/FileCheck coverage proving existing explicit,
  pre-realized, generic, and selected-boundary elementwise arithmetic artifacts
  still pass.
- Bounded provider scan showing included elementwise arithmetic statement
  sequence construction is reached through the RVV-owned plan before the older
  generic provider-local statement assembly path.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider body:
  if operation is add/sub/mul/scalar_broadcast_sub/masked_add/strided_add,
  locally assemble setvl/load/splat/compute/merge/store statements from
  operation names, memory forms, and ABI strings
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyRouteMaterializationFacts
  -> RVV-owned operand-binding facts
  -> RVVSelectedBodyElementwiseArithmeticRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
```

## Compare/Select Statement-Plan Boundary

### 1. Scope / Trigger

For mature selected-body compare/select routes, `RVVEmitCRouteProvider` must
not locally recreate the route statement sequence from operation names, ABI
strings, or memory-form branches after RVV-owned family plans,
materialization facts, and operand-binding facts have been validated. Plain
compare-select and computed-mask select must additionally consume the shared
route-control provider plan before statement construction. The RVV planning
layer must expose one
RVV-owned statement-plan boundary for plain compare-select, computed-mask
select, runtime-scalar computed-mask select, and runtime-scalar dual
compare-mask-and-select where those routes are production-active.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. It must not duplicate
the included compare/select loop/setvl/load/splat/compare/mask-and/select/store
sequence in the central provider path.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyCompareSelectRouteStatementPlan>
getRVVSelectedBodyCompareSelectRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyCompareSelectRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyCompareSelectRouteStatementPlan
        &compareSelectStatementPlan,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the
elementwise/select operand-binding facts for the same analysis. Non-consumer
route families receive an empty/default statement plan.

For active compare/select consumers, `RVVEmitCRouteProvider` must then call
`verifyRVVSelectedBodyCompareSelectRouteProviderFacts(...)` through the
compare/select statement-plan owner API before constructing
`TCRVEmitCLowerableRoute`. The verifier is the provider-facing
compare/select owner contract; it must not be declared or treated as a generic
route-planning authority surface.

### 3. Contracts

`RVVSelectedBodyCompareSelectRouteStatementPlan` is RVV-local provider input.
It may carry:

- pointers to the verified plain compare-select and computed-mask select
  family plans that justify the statement sequence;
- sub-family booleans for plain compare-select, computed-mask select,
  runtime-scalar computed-mask select, and runtime-scalar dual
  compare-mask-and-select;
- provider-ready `TCRVEmitCCallOpaqueStep` entries for full-chunk `setvl`;
- one provider-ready `TCRVEmitCForLoop` with loop `setvl`, load/splat steps,
  compare and optional secondary-compare/mask-and steps, select/merge step,
  and store step.

The plan must be derived only from verified typed body/config/runtime facts,
route materialization facts, RVV-owned elementwise/select operand-binding
facts, and for plain compare/select and computed-mask select the RVV-owned
route-control provider plan. It is not a common EmitC fact, not artifact
metadata, not an acceptance/status mirror, and not a route-support declaration
by itself.

`verifyRVVSelectedBodyCompareSelectRouteProviderFacts(...)` must consume the
same selected route analysis, materialization facts, elementwise/select
operand-binding facts, and compare/select statement-plan owner output that feed
route construction. It validates that predicate, scalar operand, true/false
value roles, selected result role, mask producer/source/form, mask/tail
provider facts, typed SEW/LMUL/policy/config, runtime ABI order, header/type/
intrinsic facts, and statement-plan leaves still mirror the selected typed RVV
body and owner-built family plans before the common route value is created.

### 4. Validation & Error Matrix

- A non compare/select route requests the boundary -> return an empty plan
  without changing unrelated route-family behavior.
- An included compare/select route has no matching verified plain
  compare-select or computed-mask select family plan -> fail closed before
  route statement construction.
- An included plain compare/select route lacks the shared route-control
  provider plan, carries stale materialization facts from another selected
  route analysis, has wrong runtime AVL/VL control facts, or has SEW/LMUL/
  policy/capability mirrors that disagree with the typed body/config and
  selected target facts -> fail closed before route statement construction.
- An included computed-mask select route lacks the shared route-control
  provider plan, carries stale materialization facts from another selected
  route analysis, has wrong runtime AVL/VL control facts, stale vector or
  runtime-scalar mask-producer facts, stale operand-binding facts, or has
  SEW/LMUL/policy/capability mirrors that disagree with the typed body/config
  and selected target facts -> fail closed before route statement construction.
- An included route lacks the required elementwise/select operand-binding
  facts -> fail closed before route statement construction.
- Required ABI roles such as `lhs`, `rhs`, `cmp_lhs`, `cmp_rhs`,
  `rhs_scalar`, `cmp_lhs_b`, `rhs_scalar_b`, `true_value`, `false_value`,
  `out`, or runtime count are absent -> fail closed with the logical operand
  name and operation/memory-form context.
- Provider-fact verification sees a stale family plan, typed config,
  materialization leaf, scalar operand binding, true/false value role, selected
  result role, mask provenance, mask/tail provider mirror, runtime ABI order,
  header/type mapping, intrinsic, or statement-plan owner leaf -> fail closed
  before creating `TCRVEmitCLowerableRoute`.
- Required materialization leaves such as `setvl`, vector load, runtime scalar
  splat, compare, secondary compare, mask-and, select, or store are absent ->
  fail closed before common EmitC.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before common
  EmitC.

### 5. Good/Base/Bad Cases

- Good: typed plain compare/select `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> elementwise/select operand-binding facts ->
  route-control provider plan -> RVV-owned statement plan -> provider-built
  route.
- Good: typed computed-mask select `tcrv_rvv` body -> family plan verifier
  with producer-source facts -> materialization facts -> elementwise/select
  operand-binding facts -> route-control provider plan -> RVV-owned statement
  plan -> provider-built route.
- Base: memory, math, residual runtime scalar splat-store, and future
  families keep their own statement construction surfaces and receive an empty
  compare/select statement plan.
- Bad: central provider code branches on `CmpSelect`,
  `ComputedMaskSelect`, `RuntimeScalarCompareSelect`, or
  `RuntimeScalarDualCompareMaskAndSelect` to rebuild the included
  compare/select setvl/load/splat/compare/mask-and/select/store sequence after
  the statement-plan boundary exists.

### 6. Tests Required

- C++ tests for positive statement-plan construction and provider consumption
  for plain compare-select, computed-mask select, runtime-scalar computed-mask
  select, and runtime-scalar dual compare-mask-and-select.
- C++ fail-closed diagnostics for at least one missing or stale
  statement-plan dependency before route statement construction.
- C++ default/empty-plan coverage for unrelated route families.
- Representative lit/FileCheck coverage proving existing explicit,
  pre-realized, and selected-boundary compare/select artifacts still pass.
- Bounded provider scan showing included compare/select statement sequence
  construction is reached through the RVV-owned plan before the older generic
  provider-local statement assembly path.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider body:
  if operation is cmp_select/computed_mask_select/runtime_scalar_cmp_select,
  locally assemble setvl/load/splat/compare/mask-and/select/store statements
  from operation names, memory forms, and ABI strings
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyRouteMaterializationFacts
  -> RVV-owned elementwise/select operand-binding facts
  -> RVVSelectedBodyRouteControlProviderPlan for adopted sub-families
  -> RVVSelectedBodyCompareSelectRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
```

## Base Memory Movement Statement-Plan Boundary

### 1. Scope / Trigger

For mature selected-body base memory movement routes,
`RVVEmitCRouteProvider` must not locally recreate the route statement sequence
from operation names, ABI strings, memory-form branches, or intrinsic mirrors
after RVV-owned family plans, materialization facts, and memory operand-binding
facts have been validated. The RVV planning layer must expose one RVV-owned
statement-plan boundary for strided load/unit store, unit load/strided store,
indexed gather/unit store, indexed scatter/unit load, static-mask unit
load/store, and static-mask unit store where those routes are
production-active.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. It must not duplicate
the included base memory movement loop/setvl/load/index/scale/masked-load/
store sequence in the central provider path.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteStatementPlan>
getRVVSelectedBodyBaseMemoryMovementRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the memory
operand-binding facts for the same analysis. Non-consumer route families
receive an empty/default statement plan.

### 3. Contracts

`RVVSelectedBodyBaseMemoryMovementRouteStatementPlan` is RVV-local provider
input. It may carry:

- a pointer to the verified base memory movement family plan that justifies the
  statement sequence;
- sub-family booleans for strided load/unit store, unit load/strided store,
  indexed gather/unit store, indexed scatter/unit load, static-mask unit
  load/store, and static-mask unit store;
- provider-ready `TCRVEmitCCallOpaqueStep` entries for full-chunk `setvl`;
- one provider-ready `TCRVEmitCForLoop` with loop `setvl`, load, strided
  load/store, index load/scale, indexed gather/scatter, static-mask load and
  passthrough handling where needed, and store steps.

The plan must be derived only from verified typed body/config/runtime facts,
route materialization facts, RVV-owned memory operand-binding facts, and the
RVV-owned route-control provider plan. It is not a common EmitC fact, not
artifact metadata, not an acceptance/status mirror, and not a route-support
declaration by itself.

### 4. Validation & Error Matrix

- A non base-memory movement route requests the boundary -> return an empty
  plan without changing unrelated route-family behavior.
- An included base memory movement route has no verified base memory movement
  family plan -> fail closed before route statement construction.
- An included route lacks required memory operand-binding facts -> fail closed
  before route statement construction.
- Required ABI roles such as `src`, `data`, `dst`, `out`, `index`, `mask`,
  passthrough `dst`, runtime count, source stride, or destination stride are
  absent -> fail closed with the logical operand name and operation/memory-form
  context.
- Required materialization leaves such as `setvl`, unit load/store, strided
  load/store, index load/scale, indexed load/store, static-mask compare, or
  masked load/store are absent -> fail closed before common EmitC.
- Required source operation provenance for configure/load/store steps is absent
  or reports the wrong EmitC source role -> fail closed before common EmitC.
- Generated-bundle evidence for a claimed executable base memory route must
  expose a `base_memory_movement_boundary` summary whose authority is typed
  `tcrv_rvv` body/config/runtime facts, whose `statement_plan` names the base
  memory family and ordered pre-loop/loop callees, and whose route metadata is
  explicitly mirror-only after provider route construction.

### 5. Good/Base/Bad Cases

- Good: typed base memory `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> memory operand-binding facts -> RVV-owned statement
  plan -> provider-built route.
- Base: computed-mask memory, segment2 memory, elementwise/select, math,
  residual runtime scalar splat-store, and future families keep their own
  statement construction surfaces and receive an empty base memory movement
  statement plan.
- Bad: central provider code branches on `StridedLoadUnitStore`,
  `IndexedGatherUnitStore`, `MaskedUnitLoadStore`, or related memory forms to
  rebuild the included base memory movement setvl/load/index/mask/store
  sequence after the statement-plan boundary exists.

### 6. Tests Required

- C++ tests for positive statement-plan construction and provider consumption
  for strided load/unit store, unit load/strided store, indexed gather/unit
  store, indexed scatter/unit load, static-mask unit load/store, and
  static-mask unit store.
- C++ fail-closed diagnostics for at least one missing or stale statement-plan
  dependency before route statement construction.
- C++ default/empty-plan coverage for unrelated route families.
- Representative lit/FileCheck coverage proving existing explicit or
  pre-realized base memory selected-body artifacts still pass.
- Generated-bundle dry-run and `ssh rvv` evidence for at least one executable
  base memory route must check `base_memory_movement_boundary`, ordered
  statement-plan callees, route-family mirror metadata, and runtime counts as
  execution cases rather than memory route authority.
- Executable base memory route harnesses must compare against a host/reference
  over multiple runtime counts and positive stride values, verify
  destination tail or skipped-slot sentinel preservation, and verify
  read-only source preservation where the memory form does not write the
  source. These checks are external ABI evidence only; they must not infer
  route support, memory form, stride side, dtype, policy, or intrinsic
  spelling.
- Bounded provider scan showing included base memory movement statement
  sequence construction is reached through the RVV-owned plan before the older
  generic provider-local statement assembly path.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider body:
  if operation is strided_load_unit_store/indexed_gather/masked_unit_store,
  locally assemble setvl/load/index/mask/store statements from operation names,
  memory forms, ABI strings, and intrinsic mirrors
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyRouteMaterializationFacts
  -> RVV-owned memory operand-binding facts
  -> RVVSelectedBodyBaseMemoryMovementRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
```

## Computed-Mask Memory Statement-Plan Boundary

### 1. Scope / Trigger

For production-active non-segment computed-mask memory movement routes,
`RVVEmitCRouteProvider` must not locally recreate the route statement sequence
from operation names, ABI strings, memory-form branches, mask-producer
mirrors, or intrinsic mirrors after RVV-owned family plans, materialization
facts, memory operand-binding facts, and the shared route-control provider
plan have been validated. The RVV planning layer must expose one RVV-owned
statement-plan boundary for runtime-scalar computed-mask store/load-store,
computed-mask unit load/store, computed-mask strided store, computed-mask
strided load/unit-store, computed-mask indexed gather/unit-store, and
computed-mask indexed scatter/unit-load where those routes are
production-active.

Computed-mask segment2 memory remains outside this boundary until a dedicated
segment2 statement-plan owner takes it. Segment2 computed-mask memory routes
must receive an empty/default computed-mask memory statement plan from this
boundary and continue through the segment2 memory path.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. It must not duplicate
the included computed-mask memory loop/setvl/load/splat/compare/masked-load/
masked-store/strided/indexed sequence in the central provider path.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyComputedMaskMemoryRouteStatementPlan>
getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);
```

The durable provider-fact entry point is:

```c++
llvm::Error verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyComputedMaskMemoryRouteStatementPlan
        &computedMaskMemoryStatementPlan,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the memory
operand-binding facts for the same analysis. Included non-segment
computed-mask memory routes must consume
`RVVSelectedBodyRouteControlProviderPlan` before statement construction.
Non-consumer route families and excluded computed-mask segment2 routes receive
an empty/default statement plan.

`RVVEmitCRouteProvider` must call
`verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts` after obtaining the
computed-mask memory statement plan and before constructing
`TCRVEmitCLowerableRoute`. Runtime-scalar and regular non-segment sub-family
checks may stay as owner-internal helpers, but the central provider must not
dispatch directly to those split helpers as its production fact boundary.

### 3. Contracts

`RVVSelectedBodyComputedMaskMemoryRouteStatementPlan` is RVV-local provider
input. It may carry:

- a pointer to the verified computed-mask memory family plan that justifies the
  statement sequence;
- sub-family booleans for runtime-scalar computed-mask store/load-store,
  computed-mask unit load/store, strided store, strided load/unit-store,
  indexed gather/unit-store, and indexed scatter/unit-load;
- provider-ready `TCRVEmitCCallOpaqueStep` entries for full-chunk `setvl`;
- one provider-ready `TCRVEmitCForLoop` with loop `setvl`, compare LHS load,
  runtime-scalar splat or compare RHS load, compare mask construction,
  payload or passthrough load, masked load/store, strided address/stride
  handling, index load/scale, indexed gather/scatter, and store steps where
  required by the selected route.

The plan must be derived only from verified typed body/config/runtime facts,
route materialization facts, RVV-owned memory operand-binding facts, and the
RVV-owned route-control provider plan. It is not a common EmitC fact, not
artifact metadata, not an acceptance/status mirror, and not a route-support
declaration by itself.

### 4. Validation & Error Matrix

- A non computed-mask memory route requests the boundary -> return an empty
  plan without changing unrelated route-family behavior.
- A computed-mask segment2 memory route requests the boundary -> return an
  empty plan and leave segment2 statement construction to its own owner.
- An included computed-mask memory route has no verified computed-mask memory
  family plan -> fail closed before route statement construction.
- An included computed-mask memory route lacks the shared route-control
  provider plan, carries stale materialization facts from another selected
  route analysis, has wrong runtime AVL/VL control facts, stale mask-producer
  or memory-form facts, or has SEW/LMUL/policy/capability mirrors that
  disagree with the typed body/config and selected target facts -> fail closed
  before route statement construction.
- An included route lacks required memory operand-binding facts -> fail closed
  before route statement construction.
- An included route carries memory operand-binding facts from another selected
  route analysis -> fail closed before route statement construction.
- Required ABI roles such as `cmp_lhs`, `cmp_rhs`, `rhs_scalar`, `src`,
  `dst`/`out`, `index`, runtime count, source stride, or destination stride
  are absent -> fail closed with the logical operand name and operation/
  memory-form context.
- Required materialization leaves such as `setvl`, vector load,
  runtime-scalar splat, compare, masked load, masked store, strided store,
  index load/scale, or indexed store are absent -> fail closed before common
  EmitC.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before
  common EmitC.

### 5. Good/Base/Bad Cases

- Good: typed computed-mask memory `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> memory operand-binding facts -> route-control
  provider plan -> RVV-owned statement plan -> provider-built route.
- Base: computed-mask segment2 memory, base memory, compare/select, math,
  residual runtime scalar splat-store, and future families keep their own
  statement construction surfaces and receive an empty computed-mask memory
  statement plan.
- Bad: central provider code branches on `RuntimeScalarComputedMaskStore`,
  `ComputedMaskStridedLoadUnitStore`, `ComputedMaskIndexedGatherLoadUnitStore`,
  `ComputedMaskIndexedScatterStoreUnitLoad`, or related memory forms to rebuild
  the included computed-mask memory setvl/load/splat/compare/mask/store
  sequence after the statement-plan boundary exists.

### 6. Tests Required

- C++ tests for positive statement-plan construction and provider consumption
  for runtime-scalar computed-mask store/load-store, computed-mask unit
  load/store, strided store, strided load/unit-store, indexed gather/unit-store,
  and indexed scatter/unit-load.
- C++ fail-closed diagnostics for at least one missing or stale statement-plan
  dependency before route statement construction, including stale
  route-control analysis, runtime AVL/VL control, policy/capability,
  mask-producer, memory-form, and memory operand-binding facts.
- C++ default/empty-plan coverage for unrelated route families and excluded
  computed-mask segment2 memory routes.
- Representative lit/FileCheck coverage proving existing explicit or
  pre-realized computed-mask memory selected-body artifacts still pass.
- Bounded provider scan showing included computed-mask memory statement
  sequence construction is reached through the RVV-owned plan before the older
  generic provider-local statement assembly path.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider body:
  if operation is runtime_scalar_cmp_masked_store/computed_masked_strided_load/
  computed_masked_indexed_scatter,
  locally assemble setvl/load/splat/compare/masked-load/masked-store/index
  statements from operation names, memory forms, ABI strings, and intrinsic
  mirrors
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyRouteMaterializationFacts
  -> RVV-owned memory operand-binding facts
  -> RVVSelectedBodyRouteControlProviderPlan
  -> RVVSelectedBodyComputedMaskMemoryRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
```

## Segment2 Route-Family Planning Owner Boundary

### 1. Scope / Trigger

For production-active segment2 selected-body route families, the RVV
planning layer must not keep sub-family selection as a local boolean cluster
inside `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)`. Segment2
route-family planning is a plugin-local provider boundary that classifies the
route description, validates selected-body route-family facts, memory
operand-binding facts, materialization facts, and route-control facts, and then
returns one owner-built provider plan for the segment2 statement-plan boundary.

The active entries are `computed-mask segment2 load`, `runtime-scalar
computed-mask segment2 load`, `computed-mask segment2 store`,
`runtime-scalar computed-mask segment2 store`, `computed-mask segment2
update`, `plain segment2 deinterleave`, and `plain segment2 interleave`. A
route may match at most one planning owner.

### 2. Signatures

The durable planning-owner API is:

```c++
struct RVVSelectedBodySegment2RouteFamilyProviderPlan {
  const RVVSelectedBodySegment2MemoryRouteFamilyPlan *segment2MemoryPlan;
  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan
      *computedMaskMemoryPlan;
  const RVVRouteOperandBindingPlan *bindingPlan;
  const RVVRuntimeAVLVLControlPlan *runtimeControlPlan;
  llvm::StringRef selectedBodyFamilyName;
  bool plansSegment2MemoryRoute;
  bool plansPlainSegment2DeinterleaveUnitStore;
  bool plansPlainSegment2InterleaveUnitLoad;
  bool plansComputedMaskSegment2LoadUnitStore;
  bool plansComputedMaskSegment2StoreUnitLoad;
  bool plansComputedMaskSegment2UpdateUnitLoad;
};

struct RVVSelectedBodySegment2RouteFamilyPlanningOwner {
  llvm::StringRef familyName;
  bool (*isConsumer)(const RVVSelectedBodyEmitCRouteDescription &);
  llvm::Error (*buildProviderPlan)(
      RVVSelectedBodyRouteAnalysis &,
      const RVVSelectedBodyRouteMaterializationFacts &,
      const RVVSelectedBodyMemoryRouteOperandBindingFacts &,
      RVVSelectedBodySegment2RouteFamilyProviderPlan &, llvm::StringRef);
};

llvm::ArrayRef<RVVSelectedBodySegment2RouteFamilyPlanningOwner>
getRVVSelectedBodySegment2RouteFamilyPlanningOwners();

bool isRVVSelectedBodySegment2RouteFamilyPlanningConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::Expected<RVVSelectedBodySegment2RouteFamilyProviderPlan>
getRVVSelectedBodySegment2RouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);
```

Additional provider-plan fields such as ABI pointers, intrinsic spelling
mirrors, C type mirrors, and required header mirrors are provider-ready facts
derived by the selected owner. They mirror validated typed body/config/runtime
facts and must not become support authority.

### 3. Contracts

`getRVVSelectedBodySegment2RouteFamilyProviderPlan(...)` returns an empty plan
for non-consumer route descriptions. For a consumer route, it must select
exactly one planning owner, build the provider plan through that owner, and
verify that the selected-body family mirror and operation-specific booleans
match the same owner-selected family.

The planning owner may consume verified plain segment2 family plans,
computed-mask memory family plans, memory operand-binding facts, route-control
facts, materialization leaves, selected target capability facts, typed
SEW/LMUL/policy/config facts, runtime `n`/AVL facts, mask/passthrough facts,
and field-role facts. It must not infer support from route ids, artifact names,
test names, ABI strings alone, descriptors, scripts, common EmitC, source-front-
door markers, metadata mirrors, exact intrinsic spelling, or legacy i32 helper
names.

`getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` must consume the
owner-built provider plan. It must not rediscover segment2 sub-family dispatch
by running its own central predicate cluster after the planning-owner boundary.

### 4. Validation & Error Matrix

- Non-segment2 route description -> return an empty/default provider plan.
- No matching segment2 planning owner -> fail closed before statement-plan
  construction.
- More than one planning owner matches -> fail closed before statement-plan
  construction.
- Matching owner has no builder -> fail closed before statement-plan
  construction.
- Owner-selected family mirror disagrees with the verified selected-body
  route-family facts -> fail closed before provider route construction.
- Required plain segment2 or computed-mask memory family plan is missing,
  stale, or mismatched to the operation kind, segment count, direction, memory
  form, field roles, mask producer/source, passthrough policy, arithmetic kind,
  SEW/LMUL, tail/mask policy, runtime ABI order, or selected capability facts
  -> fail closed.
- Materialization facts, operand-binding facts, route-control facts, runtime
  AVL/VL facts, ABI roles, or required leaves come from another analysis or are
  absent -> fail closed with the owner family and logical operand context.

### 5. Good/Base/Bad Cases

- Good: typed computed-mask segment2 update body -> computed-mask memory
  family verifier -> materialization facts -> memory operand-binding facts ->
  route-control provider plan -> `computed-mask segment2 update` planning
  owner -> segment2 statement plan -> provider-built route.
- Good: typed plain segment2 deinterleave/interleave body -> plain segment2
  family verifier -> materialization facts -> memory operand-binding facts ->
  route-control provider plan -> exact planning owner -> segment2 statement
  plan -> provider-built route.
- Base: non-segment2 memory, elementwise/select, reduction, conversion, MAcc,
  contraction, and scalar-broadcast routes receive an empty segment2
  route-family provider plan and continue through their own planning surfaces.
- Bad: segment2 statement-plan construction branches locally on
  `operationKind`, route ids, ABI strings, artifact names, or intrinsic spelling
  to decide whether the route is update, store, load, deinterleave, or
  interleave after the planning-owner boundary exists.

### 6. Tests Required

- C++ tests for registry membership, owner order/names, and non-null predicate
  and builder hooks.
- C++ exact-one classification tests for computed-mask segment2 load,
  runtime-scalar computed-mask segment2 load, computed-mask segment2 store,
  computed-mask segment2 update, plain deinterleave, and plain interleave.
- C++ empty-plan coverage for unrelated route descriptions.
- C++ fail-closed coverage for missing or stale route-family plans,
  materialization facts, route-control facts, operand-binding facts, operation
  kind, segment count, memory form, mask source, arithmetic kind, ABI order, and
  typed config/policy/capability facts.
- Generated-bundle dry-run and representative `ssh rvv` evidence when the
  owner change affects executable segment2 route-family behavior.
- Bounded authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
segment2 statement-plan body:
  if route description says computed_masked_segment2_update_unit_load,
  set local update/store/load/deinterleave/interleave booleans and then
  rebuild provider facts from operation names, ABI strings, and intrinsic
  mirrors
```

Correct:

```text
verified route-family plans
  -> route materialization facts
  -> memory operand-binding facts
  -> route-control provider plan
  -> exact segment2 route-family planning owner
  -> RVVSelectedBodySegment2RouteFamilyProviderPlan
  -> RVVSelectedBodySegment2MemoryRouteStatementPlan
  -> provider-built route
```

## Runtime-Scalar Computed-Mask Segment2 Store Pre-Realized Boundary

### 1. Scope / Trigger

Use this contract when a selected RVV variant contains:

```text
tcrv_rvv.typed_runtime_scalar_computed_mask_segment2_store_pre_realized_body
```

The op represents one bounded pre-realized selected body for
`runtime_scalar_cmp_masked_segment2_store_unit_load`. It is a selected-body
realization boundary, not a route-entry shortcut and not a Common EmitC
semantic branch.

### 2. Signatures

The pre-realized body signature is:

```text
tcrv_rvv.typed_runtime_scalar_computed_mask_segment2_store_pre_realized_body
  %lhs, %rhs_scalar, %src0, %src1, %dst, %n
  {
    op_kind = "runtime_scalar_cmp_masked_segment2_store_unit_load",
    predicate_kind = "sle",
    memory_form = "computed-mask-unit-load-segment2-store",
    segment_count = 2,
    field0_role = "segment-field0-input-buffer",
    field1_role = "segment-field1-input-buffer",
    source0_memory_form = "unit-stride-load",
    source1_memory_form = "unit-stride-load",
    destination_memory_form = "segment2-interleaved-unit-stride-store",
    mask_role = "predicate-mask-produced-by-compare",
    mask_source = "compare-produced-mask-same-vl-scope",
    mask_memory_form = "compare-produced-mask",
    inactive_lane_policy = "preserve-output-on-false-lanes",
    sew = 32, lmul = "m1",
    policy = tail agnostic, mask agnostic
  }
```

The runtime ABI order is:

```text
lhs, rhs_scalar, src0, src1, dst, n
```

`rhs_scalar` must be an `i32` scalar runtime ABI value with C type `int32_t`
and role `rhs-scalar-value`.

### 3. Contracts

- The segment2 memory selected-body realization owner must consume the
  pre-realized op through the public selected lowering-boundary materializer.
- Realization must materialize the typed body sequence:
  `setvl`, `with_vl`, lhs `load`, `rhs_scalar` `splat`, field0/field1
  `load`, `compare`, and `masked_segment2_store`.
- The realized compare must use the splatted runtime scalar RHS and preserve
  `runtime-scalar-splat-compare-rhs` as the provider-derived mask producer
  source.
- The RVV provider, segment2 route-family planning owner, statement-plan
  owner, target export validator, and generated bundle ABI must consume the
  same realized typed body facts. Common EmitC remains a neutral materializer.

### 4. Validation & Error Matrix

- Unsupported `op_kind`, `predicate_kind`, memory form, field role, segment
  count, source/destination form, mask role/source/form, inactive-lane policy,
  SEW/LMUL, or policy -> fail before provider facts.
- Missing or stale runtime ABI roles for `lhs`, `rhs_scalar`, `src0`, `src1`,
  `dst`, or `n` -> fail before realization or provider construction.
- `rhs_scalar` is not `i32`, lacks role `rhs-scalar-value`, or has stale C type
  -> fail before splat realization.
- Realized route facts report vector-compare RHS load instead of
  `runtime-scalar-splat-compare-rhs` -> target artifact export must fail.
- Target header/prototype ABI order, route operand binding, C type mapping, or
  provider-supported mirror disagrees with the rebuilt provider route -> fail
  before executable bundle acceptance.

### 5. Good/Base/Bad Cases

- Good: pre-realized runtime-scalar segment2 store body -> segment2 memory
  owner realization -> realized splat/compare/masked segment2 store body ->
  provider-built route -> Common EmitC -> target artifact -> generated bundle.
- Base: the explicit selected-body
  `runtime_scalar_cmp_masked_segment2_store_unit_load` route remains a
  regression baseline with the same ABI and provider route facts.
- Bad: the old `typed_computed_mask_segment2_store_pre_realized_body` accepts a
  scalar RHS by role/name mutation.
- Bad: target export accepts stale vector-compare mask producer metadata because
  the route id or artifact name says runtime-scalar.

### 6. Tests Required

- A positive lit fixture must prove the pre-realized op is removed and replaced
  with `load/splat/load/load/compare/masked_segment2_store` before emission
  plan and target header checks.
- Generated-bundle dry-run evidence must check pre-realized body consumption,
  `rhs_scalar` ABI/header participation, `runtime-scalar-splat-compare-rhs`,
  field0/field1 roles, inactive-lane preservation, segment count, and runtime
  AVL/VL.
- Runtime correctness claims require non-dry-run `ssh rvv` evidence for active
  and inactive lanes, field distinction, source preservation, and tail
  preservation.
- Negative evidence must reject at least one stale runtime-scalar boundary fact,
  such as stale mask producer source or stale `rhs_scalar` ABI/type binding.

### 7. Wrong vs Correct

Wrong:

```text
pre-realized body or artifact metadata says runtime_scalar segment2 store
  -> target export accepts vector compare RHS load facts
```

Correct:

```text
typed_runtime_scalar_computed_mask_segment2_store_pre_realized_body
  -> RVV segment2 selected-body realization owner
  -> realized lhs load + rhs_scalar splat + compare + masked_segment2_store
  -> provider-derived runtime-scalar segment2 route facts
  -> target export mirror validation
```

## Segment2 Memory Statement-Plan Boundary

### 1. Scope / Trigger

For production-active segment2 memory movement routes,
`RVVEmitCRouteProvider` must not locally recreate the route statement sequence
from operation names, ABI strings, memory-form branches, mask-producer
mirrors, field-role mirrors, or intrinsic mirrors after RVV-owned family
plans, materialization facts, and memory operand-binding facts have been
validated. The RVV planning layer must expose one RVV-owned statement-plan
boundary for plain segment2 deinterleave/unit-store, plain segment2
interleave/unit-load, computed-mask segment2 load/unit-store,
runtime-scalar computed-mask segment2 load/unit-store, and computed-mask
segment2 store/unit-load where those routes are production-active.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. It must not duplicate
the included segment2 loop/setvl/field-load/field-store/compare/tuple/
segment-load/segment-store sequence in the central provider path.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodySegment2MemoryRouteStatementPlan>
getRVVSelectedBodySegment2MemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedBodySegment2MemoryRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodySegment2RouteFamilyProviderPlan
        &segment2ProviderPlan,
    const RVVSelectedBodyRouteStatementPlanOwnerSelection
        &statementPlanOwnerSelection,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the memory
operand-binding facts for the same analysis. It must then call
`verifyRVVSelectedBodySegment2MemoryRouteProviderFacts(...)` with the
owner-built segment2 route-family provider plan and migrated segment2
statement-plan owner selection before constructing `TCRVEmitCLowerableRoute`.
Non-consumer route families receive an empty/default statement plan.

### 3. Contracts

`RVVSelectedBodySegment2MemoryRouteStatementPlan` is RVV-local provider input.
It may carry:

- pointers copied from the owner-built
  `RVVSelectedBodySegment2RouteFamilyProviderPlan`, including the verified
  plain segment2 family plan or computed-mask memory family plan that justifies
  the selected segment2 statement sequence;
- owner-built sub-family booleans for plain deinterleave, plain interleave,
  computed-mask segment2 load, runtime-scalar computed-mask segment2 load,
  computed-mask segment2 store, runtime-scalar computed-mask segment2 store,
  and computed-mask segment2 update;
- provider-ready `TCRVEmitCCallOpaqueStep` entries for full-chunk `setvl`;
- one provider-ready `TCRVEmitCForLoop` with loop `setvl`, compare-mask
  producer steps for computed-mask segment2 routes, field payload or
  passthrough loads, tuple create/extract calls, segment load/store or masked
  segment load/store calls, field stores, and address expressions.

The plan must be derived only from verified typed body/config/runtime facts,
route materialization facts, and RVV-owned memory operand-binding facts. It is
not a common EmitC fact, not artifact metadata, not an acceptance/status
mirror, and not a route-support declaration by itself.

### 4. Validation & Error Matrix

- A non segment2 memory route requests the boundary -> return an empty plan
  without changing unrelated route-family behavior.
- An included plain segment2 route has no verified plain segment2 family plan
  -> fail closed before route statement construction.
- An included computed-mask segment2 route has no verified computed-mask
  memory family plan -> fail closed before route statement construction.
- An included route lacks required memory operand-binding facts -> fail closed
  before route statement construction.
- An included route lacks the shared route-control provider plan, carries stale
  materialization facts from another selected route analysis, has wrong runtime
  AVL/VL control facts, stale segment direction or memory-form facts, stale
  computed-mask segment facts, or has SEW/LMUL/policy/capability mirrors that
  disagree with the typed body/config and selected target facts -> fail closed
  before route statement construction.
- An included route reaches the statement-plan boundary without a matching
  segment2 route-family planning owner, with more than one matching owner, or
  with an owner-built provider plan whose selected-body family mirror disagrees
  with the verified route-family facts -> fail closed before route statement
  construction.
- Required ABI roles such as `src`, `dst`, `cmp_lhs`, `cmp_rhs`, `field0`,
  `field1`, and runtime count are absent -> fail closed with the logical
  operand name and operation/memory-form context.
- Required materialization leaves such as `setvl`, vector load, compare,
  segment load/store, masked segment load/store, tuple create, field extract,
  or field store are absent -> fail closed before common EmitC.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before
  common EmitC.
- The central provider constructs `TCRVEmitCLowerableRoute` for a segment2
  consumer without first passing
  `verifyRVVSelectedBodySegment2MemoryRouteProviderFacts(...)` -> fail closed
  as an owner/provider contract violation.

### 5. Good/Base/Bad Cases

- Good: typed segment2 `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> memory operand-binding facts -> route-control
  provider plan -> segment2 route-family planning owner -> RVV-owned statement
  plan -> provider-built route.
- Base: base memory, non-segment computed-mask memory, compare/select, math,
  residual runtime scalar splat-store, and future families keep their own
  statement construction surfaces and receive an empty segment2 memory
  statement plan.
- Bad: central provider code branches on `Segment2DeinterleaveUnitStore`,
  `Segment2InterleaveUnitLoad`, `ComputedMaskSegment2LoadUnitStore`, or
  `ComputedMaskSegment2StoreUnitLoad` to rebuild the included segment2
  setvl/load/compare/tuple/segment/store sequence after the statement-plan
  boundary exists.

### 6. Tests Required

- C++ tests for positive statement-plan construction and provider consumption
  for plain segment2 deinterleave/unit-store, plain segment2
  interleave/unit-load, computed-mask segment2 load/unit-store,
  runtime-scalar computed-mask segment2 load/unit-store, and computed-mask
  segment2 store/unit-load.
- C++ fail-closed diagnostics for at least one missing or stale
  statement-plan dependency before route statement construction, including
  missing/stale route-control, same-analysis materialization, runtime AVL/VL,
  policy/capability, segment direction, memory-form, and operand-binding facts.
- C++ default/empty-plan coverage for unrelated route families.
- Representative lit/FileCheck coverage proving existing explicit or
  pre-realized segment2 selected-body artifacts still pass.
- Bounded provider scan showing included segment2 memory statement sequence
  construction is reached through the RVV-owned plan before the older generic
  provider-local statement assembly path.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider body:
  if operation is segment2_deinterleave/computed_masked_segment2_store,
  locally assemble setvl/load/compare/tuple/segment/store statements from
  operation names, memory forms, ABI strings, field roles, and intrinsic mirrors
```

Correct:

```text
verified route-family plans
  -> RVVSelectedBodyRouteMaterializationFacts
  -> RVV-owned memory operand-binding facts
  -> RVVSelectedBodySegment2RouteFamilyProviderPlan
  -> RVVSelectedBodySegment2MemoryRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
```

## Computed-Mask Accumulation Statement-Plan Boundary

### 1. Scope / Trigger

For production-active computed-mask MAcc selected-body routes,
`RVVEmitCRouteProvider` must not locally recreate the route statement sequence
from operation names, ABI strings, intrinsic mirrors, mask-producer mirrors, or
accumulator-role mirrors after RVV-owned family plans, materialization facts,
and math operand-binding facts have been validated. The RVV planning layer must
expose one RVV-owned statement-plan boundary for `ComputedMaskedMAccAdd` and
`RuntimeScalarComputedMaskedMAccAdd` where those routes are production-active.

This boundary is narrower than the broader computed-mask accumulation
route-family owner. Computed-mask standalone reductions may still use their own
statement construction surface until a later owner moves them behind a
dedicated reduction statement plan.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. It must not duplicate
the included computed-mask MAcc setvl/compare-load/payload-load/accumulator-load/
mask/active-MAcc/merge/store sequence in the central provider path.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan>
getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, after obtaining math operand-binding
facts for the same analysis, and after the computed-mask accumulation MAcc path
has consumed `RVVSelectedBodyRouteControlProviderPlan`. Non-consumer route
families receive an empty/default statement plan.

### 3. Contracts

`RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan` is RVV-local
provider input. It may carry:

- a pointer to the verified computed-mask accumulation family plan that
  justifies the selected statement sequence;
- sub-family booleans for vector-compare computed-mask MAcc and runtime-scalar
  computed-mask MAcc;
- provider-ready `TCRVEmitCCallOpaqueStep` entries for full-chunk `setvl`;
- one provider-ready `TCRVEmitCForLoop` with loop `setvl`, compare producer
  load or splat steps, payload loads, accumulator load, compare-mask creation,
  active MAcc compute, masked merge/passthrough, store, operands, and results.

The plan must be derived only from verified typed body/config/runtime facts,
route materialization facts, RVV-owned math operand-binding facts, and the
RVV-owned route-control provider plan. It is not a common EmitC fact, not
artifact metadata, not an acceptance/status mirror, and not a route-support
declaration by itself.

### 4. Validation & Error Matrix

- A non computed-mask MAcc route requests the boundary -> return an empty plan
  without changing unrelated route-family behavior.
- An included computed-mask MAcc route has no verified computed-mask
  accumulation family plan -> fail closed before route statement construction.
- An included route lacks required math operand-binding facts -> fail closed
  before route statement construction.
- The verified family plan has stale route-shape markers such as wrong
  runtime-scalar producer, vector-compare producer, vector-MAcc suffix, or
  scalar-horizontal suffix -> fail closed before common EmitC.
- An included computed-mask MAcc route lacks the shared route-control provider
  plan, has stale same-analysis materialization/control ownership, or carries
  policy/runtime ABI/capability mirrors that disagree with route-control facts
  -> fail closed before route statement construction.
- Required ABI roles such as `cmp_lhs`, `cmp_rhs` or `rhs_scalar`, payload
  `lhs`/`rhs`, `acc`, `out`, and runtime count are absent -> fail closed with
  the logical operand name and operation/memory-form context.
- Required materialization leaves such as `setvl`, vector load, scalar splat,
  compare, active MAcc, masked merge, or store are absent -> fail closed before
  common EmitC.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before common
  EmitC.

### 5. Good/Base/Bad Cases

- Good: typed computed-mask MAcc `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> math operand-binding facts -> route-control provider
  plan -> RVV-owned statement plan -> provider-built route.
- Base: computed-mask standalone reductions, widening dot reductions, plain
  MAcc, memory, compare/select, residual runtime scalar splat-store, and future
  families keep their own statement construction surfaces and receive an empty
  computed-mask accumulation statement plan.
- Bad: central provider code branches on `ComputedMaskedMAccAdd` or
  `RuntimeScalarComputedMaskedMAccAdd` to rebuild the included
  setvl/load/splat/compare/MAcc/merge/store sequence after the statement-plan
  boundary exists.

### 6. Tests Required

- C++ tests for positive statement-plan construction and provider consumption
  for computed-mask MAcc and runtime-scalar computed-mask MAcc, including
  positive route-control provider-plan consumption before statement planning.
- C++ fail-closed diagnostics for at least one missing or stale
  statement-plan dependency before route statement construction, including
  stale route-control, runtime AVL/VL, policy, selected capability,
  mask-producer/classification, and math operand-binding facts.
- C++ default/empty-plan coverage for unrelated route families.
- Representative lit/FileCheck coverage proving existing explicit or
  pre-realized computed-mask MAcc selected-body artifacts still pass.
- Bounded provider scan showing included computed-mask accumulation statement
  sequence construction is reached through the RVV-owned plan before the older
  generic provider-local statement assembly path.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider body:
  if operation is computed_masked_macc_add/runtime_scalar_cmp_masked_macc_add,
  locally assemble setvl/load/splat/compare/MAcc/merge/store statements from
  operation names, ABI strings, mask-source mirrors, and accumulator mirrors
```

Correct:

```text
verified computed-mask accumulation family plan
  -> RVVSelectedBodyRouteMaterializationFacts
  -> RVV-owned math operand-binding facts
  -> RVVSelectedBodyRouteControlProviderPlan
  -> RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
```

## MAcc Route-Family Provider-Plan Owner Boundary

### 1. Scope / Trigger

For production-active plain `macc_add`, scalar-broadcast
`scalar_broadcast_macc_add`, `computed_masked_macc_add`, and
`runtime_scalar_cmp_masked_macc_add` selected-body routes, central
`RVVEmitCRoutePlanning` must not own the MAcc route-family owner registry or
MAcc-specific provider-plan verification bodies. Those decisions belong to an
explicit RVV-local MAcc route-family owner boundary.

Central route planning may keep shared route-analysis structs, route-family
plan structs, route descriptions, materialization facts, operand-binding facts,
and top-level aggregate orchestration. If route-family plan derivation owns
structural plan validators, central may expose thin structural-validation
wrappers for the MAcc owner, but MAcc consumer selection, exact-owner checks,
and provider-plan verification must stay owner-local.

### 2. Signatures

The durable owner API is:

```c++
struct RVVSelectedBodyMAccRouteFamilyOwner {
  StringRef familyName;
  bool (*isConsumer)(RVVSelectedBodyOperationKind);
  Error (*verifyProviderPlan)(const RVVSelectedBodyRouteAnalysis &,
                              StringRef context);
};

ArrayRef<RVVSelectedBodyMAccRouteFamilyOwner>
getRVVSelectedBodyMAccRouteFamilyOwners();

bool isRVVSelectedBodyMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

Error verifyRVVSelectedBodyMAccRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, StringRef context);
```

The same MAcc owner boundary also owns MAcc route-operand binding authority:

```c++
std::optional<StringRef>
getExpectedRVVSelectedBodyMAccRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation);

std::optional<RuntimeABIParameterRole>
getExpectedRVVSelectedBodyMAccRouteOperandBindingRole(
    StringRef planID, StringRef logicalOperand);

Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyMAccRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis);
```

Central route planning may keep the shared `RVVRouteOperandBindingPlan`
container, generic closure comparison, and aggregate dispatch, but MAcc plan
ids, MAcc logical operands, and MAcc logical-operand-to-runtime-ABI-role
mapping must be supplied by the owner boundary.

The active owner registry has exactly three MAcc-family owners:

- plain MAcc;
- scalar-broadcast MAcc;
- computed-mask MAcc.

The computed-mask MAcc owner may reuse the shared computed-mask accumulation
route-family plan verifier because computed-mask standalone reductions share
that plan surface. The MAcc owner must classify only the MAcc sub-family as
MAcc consumers; computed-mask standalone reductions remain outside the MAcc
owner and are reached through the standalone reduction/accumulation aggregate.

### 3. Contracts

- Plain MAcc ownership is selected only for `macc_add`.
- Scalar-broadcast MAcc ownership is selected only for
  `scalar_broadcast_macc_add`.
- Computed-mask MAcc ownership is selected only for
  `computed_masked_macc_add` and `runtime_scalar_cmp_masked_macc_add`.
- The owner verifier consumes the same selected `RVVSelectedBodyRouteAnalysis`
  as the central aggregate and must validate the selected route-family plan,
  runtime/control mirrors, type/intrinsic/header mirrors, route operand-binding
  plan, runtime ABI order, and MAcc/accumulation sub-family facts before route
  materialization is accepted.
- Plain, scalar-broadcast, and computed-mask MAcc family plans must carry a
  typed config snapshot from `RVVSelectedBodyTypedConfigFacts`: facts id,
  element type, element bit width, SEW, LMUL, tail policy, mask policy, and
  config contract id. Their vector type/C type, VL C type, setvl, load, store,
  scalar-splat where applicable, mask type/C type where applicable, MAcc leaf,
  compare leaf, and masked-merge leaf must mirror typed body/config facts or
  be derived from typed element/SEW/LMUL plus operation/predicate facts.
- The owner verifier may call shared typed/body/plan validators, but it must
  not infer support from route ids, artifact names, ABI strings, exact
  intrinsic spellings, descriptor residue, or emission-plan/status metadata.
- Central math-cluster orchestration may call the MAcc owner verifier as one
  aggregate entry, but it must not locally duplicate plain/scalar/computed-mask
  MAcc consumer predicates or provider-plan verification bodies.
- Central route-operand binding validation may call the MAcc owner as a neutral
  subroutine, but it must not locally define MAcc binding plan ids, assemble
  MAcc binding plans from operation names, or map `lhs`, `rhs`, `rhs_scalar`,
  `cmp_lhs`, `cmp_rhs`, `acc`, `out`, or `n` to runtime ABI roles for the
  MAcc sub-families.

### 4. Validation & Error Matrix

- A MAcc route lacks its required plain, scalar-broadcast, or computed-mask
  route-family plan -> fail closed before provider materialization.
- A non-MAcc route carries a stale plain/scalar/computed-mask MAcc family plan
  -> fail closed before provider materialization.
- More than one MAcc owner matches the selected operation -> fail closed and
  report the matching owner names.
- An owner registry entry has a null consumer or verifier hook -> fail closed.
- Route description mirrors disagree with the validated family plan for
  runtime control, ABI order, type mapping, headers, intrinsic leaf, result
  name, MAcc layout, mask producer/source, inactive-lane contract, or
  computed-mask accumulation suffix -> fail closed.
- A MAcc family plan has no typed config snapshot, carries a stale typed
  vector/load/store/scalar-splat/mask leaf, or carries a MAcc/compare/merge
  leaf that cannot be derived from typed element/SEW/LMUL and operation facts
  -> fail closed before provider materialization.
- The route operand-binding plan is absent, stale, or does not match the
  selected operation -> fail closed before materialization.
- A MAcc operand-binding plan binds a logical operand to the wrong runtime ABI
  role, omits the accumulator/result/runtime count binding, mismatches scalar
  broadcast or runtime-scalar compare binding, or carries a non-owner plan id
  -> fail closed through the MAcc owner-owned operand-binding API before
  provider materialization.
- Computed-mask standalone reductions request the MAcc owner boundary ->
  return non-consumer behavior; their shared accumulation checks are reached
  through the standalone reduction/accumulation owner.

### 5. Good/Base/Bad Cases

- Good: typed plain `tcrv_rvv.macc` body -> typed config facts snapshot ->
  route-family plan derivation -> MAcc owner registry selects plain MAcc
  exactly once -> owner verifies typed leaf mirrors and operand bindings ->
  materialization facts -> MAcc statement-plan owner.
- Good: typed scalar-broadcast MAcc body -> scalar-broadcast MAcc owner
  verifies RHS scalar broadcast leaf from typed config facts and runtime ABI ->
  route-control provider plan -> MAcc statement-plan owner.
- Good: typed computed-mask MAcc body -> computed-mask MAcc owner verifies the
  shared accumulation family plan as a vector-MAcc suffix with derived
  compare/MAcc/merge leaves, the correct mask producer, and inactive-lane
  contracts -> computed-mask accumulation statement owner.
- Base: standalone reductions, direct contractions, memory, segment2, compare/
  select, residual runtime scalar splat-store, and conversion routes use their
  own owner boundaries and receive non-consumer behavior from the MAcc owner.
- Bad: central `RVVEmitCRoutePlanning` keeps a local MAcc owner registry or
  branches on `MAccAdd`, `ScalarBroadcastMAccAdd`,
  `ComputedMaskedMAccAdd`, or `RuntimeScalarComputedMaskedMAccAdd` to verify
  route-family provider-plan mirrors after this boundary exists.

### 6. Tests Required

- Focused C++ tests for MAcc owner registry membership, owner order/names,
  non-null consumer/verifier hooks, exact classification for all active MAcc
  routes, and exclusion of standalone accumulation and elementwise routes.
- Focused C++ fail-closed tests for missing family plans, stale non-consumer
  MAcc plans, mismatched mirrors, stale operand-binding plans, and
  computed-mask MAcc suffix/provenance mismatches.
- Focused C++ positive/negative tests for typed config-derived MAcc leaves:
  plan snapshot mirrors selected typed facts; stale vector type/load/store,
  scalar splat, mask type, MAcc, compare, or masked-merge leaves fail closed
  before provider route construction.
- Aggregate math-cluster owner tests proving central orchestration reaches the
  MAcc owner through the owner API rather than local MAcc predicates.
- Representative generated-bundle dry-runs for plain MAcc, scalar-broadcast
  MAcc, and computed-mask MAcc selected-body routes.
- Bounded symbol scan showing moved MAcc owner symbols are concentrated in the
  MAcc owner module and central planning retains only neutral calls/wrappers.
- Authority scan over touched RVV planning/provider/test files for legacy
  i32/source-front-door/descriptor/direct-C/source-export, exact-intrinsic, or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
central RoutePlanning:
  owns MAcc owner registry
  verifies plain/scalar/computed-mask MAcc provider-plan mirrors locally
  lets aggregate math-cluster verification branch on MAcc operation names
```

Correct:

```text
selected typed MAcc body
  -> route-family plan derivation
  -> RVV-owned MAcc route-family owner registry
  -> exact-one MAcc provider-plan verification
  -> shared materialization/operand-binding facts
  -> RVV-owned MAcc or computed-mask accumulation statement plan
```

## Plain And Scalar-Broadcast MAcc Statement-Plan Boundary

### 1. Scope / Trigger

For production-active plain `macc_add` and scalar-broadcast
`scalar_broadcast_macc_add` selected-body routes, `RVVEmitCRouteProvider` must
not locally recreate the setvl/load/splat/load-accumulator/MAcc/store
statement sequence from operation names, ABI strings, intrinsic mirrors,
accumulator-layout mirrors, route ids, or artifact metadata after route
materialization facts and math operand-binding facts have been validated. The
scalar-broadcast sub-family must additionally consume the shared
route-control provider plan before statement construction. The RVV planning
layer must expose one RVV-owned statement-plan boundary for the bounded typed
MAcc route family and its scalar-broadcast sub-family.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`,
adds neutral headers, type mappings, ABI mappings, selected-boundary source
provenance, and attaches the RVV-owned statement plan. It must not duplicate
the included plain MAcc sequence in the central provider path.

### 2. Signatures

The durable planning/provider API is:

```c++
llvm::Expected<RVVSelectedBodyPlainMAccRouteStatementPlan>
getRVVSelectedBodyPlainMAccRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);
```

`RVVEmitCRouteProvider` must call this boundary through
`getRVVSelectedBodyMigratedRouteStatementPlan(...)` after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining math operand-binding
facts for the same analysis. Non-consumer route families receive an
empty/default plain MAcc statement plan.

### 3. Contracts

`RVVSelectedBodyPlainMAccRouteStatementPlan` is RVV-local provider input. It
may carry:

- the verified math operand-binding plan pointer;
- pointers to the verified plain or scalar-broadcast MAcc family plan when the
  selected operation requires one;
- sub-family booleans for plain MAcc, `macc_add`, and
  `scalar_broadcast_macc_add`;
- provider-ready full-chunk `setvl` pre-loop steps;
- one provider-ready `TCRVEmitCForLoop` with loop `setvl`, lhs load, rhs load
  or RHS scalar splat, accumulator load, active MAcc compute, and store steps.

For the bounded i32/SEW32/LMUL m1 unit-stride route, the emitted MAcc step must
use the RVV-owned compute leaf as `accumulator_vector, lhs_vector, rhs_vector,
loop_vl`; for scalar-broadcast MAcc the RHS vector must come from the
family-owned scalar splat leaf and the `rhs_scalar` ABI role. The store
destination must advance by the loop induction (`out + induction`). The plan
must be derived only from verified typed body/config/runtime facts, route
  materialization facts, RVV-owned math operand-binding facts, and for
  scalar-broadcast MAcc the RVV-owned route-control provider plan. It is not a
  common EmitC fact, not artifact metadata, not an acceptance/status mirror,
  and not a route-support declaration by itself.

For scalar-broadcast MAcc, `RVVSelectedBodyRouteAnalysis` must carry a
validated `RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan`, and
`RVVSelectedBodyEmitCRouteDescription::scalarBroadcastMAccRouteFamilyPlanID`
must mirror that plan only after validation. Target artifacts may mirror the
plan through `tcrv_rvv.scalar_broadcast_macc_route_family_plan`; they must not
infer support, dtype, ABI order, or intrinsic spelling from the mirror.

### 4. Validation & Error Matrix

- A non plain-MAcc route requests the boundary -> return an empty plan without
  changing unrelated route-family behavior, unless it is an included
  scalar-broadcast MAcc route.
- An included plain MAcc route lacks `bindsPlainMAcc` math operand-binding
  facts -> fail closed before route statement construction.
- An included scalar-broadcast MAcc route lacks its validated
  scalar-broadcast MAcc route-family plan, or the route description mirror does
  not match that plan -> fail closed before route statement construction.
- Required ABI roles `lhs`, `rhs`, `acc`, `out`, or runtime count `n` are
  absent -> fail closed with the logical operand name and operation/memory-form
  context.
- Required ABI roles `lhs`, `rhs_scalar`, `acc`, `out`, or runtime count `n`
  are absent for scalar-broadcast MAcc -> fail closed with the logical operand
  name and operation/memory-form context.
- Scalar-broadcast MAcc lacks the shared route-control provider plan, carries
  stale materialization facts from another selected route analysis, has wrong
  runtime AVL/VL control facts, or has SEW/LMUL/policy/capability mirrors that
  disagree with the typed body/config and selected target facts -> fail closed
  before route statement construction.
- Required materialization leaves `setvl`, vector load, scalar splat where
  needed, MAcc compute, or store are absent -> fail closed before common EmitC.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before common
  EmitC.
- Accumulator/result layout, dtype, SEW, LMUL, policy, or runtime AVL facts
  must already be validated from the typed body/config/runtime structure; the
  statement plan must not repair those facts from route ids, artifact names,
  ABI names, helper names, or harness constants.

### 5. Good/Base/Bad Cases

- Good: typed `tcrv_rvv.macc` body -> route-family provider verifier ->
  materialization facts -> math operand-binding facts -> RVV-owned plain MAcc
  statement plan -> provider-built route.
- Good: typed scalar-broadcast `tcrv_rvv.macc` body with `rhs_scalar` ABI ->
  scalar-broadcast MAcc family plan verifier -> materialization facts -> math
  operand-binding facts -> route-control provider plan -> RVV-owned statement
  plan with scalar splat leaf -> provider-built route.
- Base: computed-mask MAcc, widening MAcc, dot-reduction, memory, compare/
  select, residual runtime scalar splat-store, and future families keep their
  own statement construction surfaces and receive an empty plain MAcc plan.
- Bad: central provider code branches on `MAccAdd` to rebuild
  setvl/lhs-load/rhs-load/acc-load/MAcc/store after the statement-plan boundary
  exists.
- Bad: the store statement uses bare `out` instead of `out + induction` for a
  vector result path.

### 6. Tests Required

- Focused C++ tests for positive plain MAcc statement-plan construction and
  provider consumption, plus scalar-broadcast MAcc positive statement-plan
  construction, route-control provider-plan consumption, and provider
  consumption.
- Focused C++ fail-closed diagnostics for missing math facts, MAcc compute
  leaf, vector load leaf, scalar splat leaf, missing/stale scalar-broadcast
  family plan mirror, stale route-control analysis, invalid runtime AVL/VL
  control, policy/config/capability mismatch, or source-role provenance before
  route statement construction.
- C++ default/empty-plan coverage for unrelated route families.
- Representative lit/FileCheck coverage proving existing explicit and
  pre-realized MAcc selected-body artifacts still pass.
- Generated-bundle ABI evidence showing typed `tcrv_rvv.macc` body/config,
  math operand-binding facts, RVV-owned plain MAcc statement plan, emitted
  `vmacc` operands, scalar-broadcast family-plan mirror where applicable, and
  mirror-only artifact metadata.
- Bounded authority scan over touched RVV dialect/realization/planning/
  provider/target/script/fixture files for legacy i32/source-front-door/
  descriptor/direct-C/source-export or mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
provider body:
  if operation is macc_add or scalar_broadcast_macc_add,
  locally assemble setvl/load/splat/load-acc/MAcc/store from operation names,
  ABI strings, route ids, intrinsic mirrors, accumulator mirrors, or artifact
  metadata
```

Correct:

```text
verified route-family provider plans
  -> RVVSelectedBodyRouteMaterializationFacts
  -> RVV-owned math operand-binding facts
  -> RVVSelectedBodyPlainMAccRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
```

## Direct Contraction Route-Provider Owner Boundary

### 1. Scope / Trigger

For active direct-provider contraction routes, `RVVEmitCRouteProvider` must not
locally recreate widening MAcc or widening dot-reduction statement sequences
from operation names, memory forms, ABI strings, intrinsic mirrors, route ids,
or artifact metadata after RVV-owned family plans, materialization facts,
math operand-binding facts, and route-control provider-plan facts have been
validated.

The active direct-provider contraction routes are `widening_macc_add`,
`widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
`computed_masked_widening_dot_reduce_add`, and
`computed_masked_strided_input_widening_dot_reduce_add`. They must be selected
exactly once by an RVV plugin-owned direct contraction route-provider owner.

`RVVEmitCRouteProvider` remains the owner that instantiates
`TCRVEmitCLowerableRoute`, records neutral headers/type mappings/ABI mappings,
preserves selected-boundary source provenance, and attaches returned
provider-ready statements. It must not obtain or validate the direct
contraction provider plan in the central provider body. After route-family
provider-plan verification, route materialization facts, and math
operand-binding facts are available, the central provider passes those
same-analysis facts to the statement-plan owner selection boundary. That
boundary derives the direct contraction provider plan, builds the direct
contraction statement plan from the same provider plan, validates the provider
facts against the owner selection, and only then returns provider-ready
statements for `TCRVEmitCLowerableRoute` attachment. Contraction routes must
not fall through to central ad hoc contraction statement construction when the
owner boundary exists.

### 2. Signatures

The durable planning/provider API is:

```c++
struct RVVSelectedBodyDirectContractionRouteProviderPlan {
  contraction route-family plan pointer;
  route-control provider plan;
  direct contraction / widening MAcc / dot-reduction / computed-mask /
      strided-input booleans;
  bound runtime ABI parameters for compare lhs/rhs, dot lhs/rhs, lhs/rhs,
      accumulator, output, runtime n, and optional lhs/rhs strides;
  provider-owned VL/result/source/mask C type facts;
  provider-owned setvl, load, strided-load, compare, widening product,
      masked widening product, merge, seed splat, contraction compute,
      accumulator load, and store leaves;
};

struct RVVSelectedBodyDirectContractionRouteStatementPlan {
  contraction route-family plan pointer;
  direct contraction / widening MAcc / dot-reduction / computed-mask /
      strided-input booleans;
  provider-ready pre-loop TCRVEmitCCallOpaqueStep entries;
  one provider-ready TCRVEmitCForLoop;
};

struct RVVSelectedBodyDirectContractionRouteProviderOwner {
  family name;
  consumer predicate over RVVSelectedBodyEmitCRouteDescription;
  statement-plan builder over RVVSelectedBodyRouteAnalysis,
      direct contraction provider plan,
      output direct contraction statement plan;
};

getRVVSelectedBodyDirectContractionRouteProviderOwners()
isRVVSelectedBodyDirectContractionRouteProviderConsumer(
    RVVSelectedBodyEmitCRouteDescription)

llvm::Expected<RVVSelectedBodyDirectContractionRouteProviderPlan>
getRVVSelectedBodyDirectContractionRouteProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
getRVVSelectedBodyDirectContractionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyRouteStatementPlanOwnerSelection>
getRVVSelectedBodyRouteStatementPlanOwnerSelection(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context);
```

The owner registry must have exactly one active owner for the existing
direct-provider contraction family. The provider-plan getter returns an
empty/default provider plan for non-consumer routes and otherwise joins the
same-analysis family plan, route-control plan, materialization facts, math
operand-binding facts, typed config, selected capability, and ABI bindings.
The public direct-contraction statement-plan getter derives this provider plan
inside the RVV statement-owner module before invoking the provider-plan-backed
owner builder. The aggregate statement-plan owner selection boundary must
derive one direct-contraction provider plan inside the owner path, select
ownership through the registry, return an empty/default statement plan for
non-consumer routes, and fail closed when an owner entry is incomplete, more
than one owner matches, the direct-contraction provider plan cannot be derived
from the same-analysis facts, or the selected owner fails to produce a direct
contraction statement plan.

### 3. Contracts

`RVVSelectedBodyDirectContractionRouteProviderPlan` is the RVV-local provider
input that proves direct contraction facts before `TCRVEmitCLowerableRoute`
construction. It may carry:

- a pointer to the verified same-analysis contraction route-family plan;
- booleans for widening MAcc, dot-reduction, computed-mask, and strided-input
  classification;
- the same-analysis route-control provider plan;
- bound runtime ABI parameters for compare lhs/rhs, dot lhs/rhs, lhs/rhs,
  accumulator, output, runtime element count, and strided-input strides where
  active;
- provider-owned VL/result/source/mask type facts;
- provider-owned setvl, load, strided-load, compare, widening product, masked
  widening product, merge, seed splat, contraction compute, accumulator load,
  and store leaves.

`RVVSelectedBodyDirectContractionRouteStatementPlan` is RVV-local statement
input. It may carry:

- a pointer to the verified same-analysis contraction route-family plan;
- booleans for widening MAcc, dot-reduction, computed-mask, and strided-input
  classification;
- provider-ready full-chunk `setvl` pre-loop steps;
- dot-reduction seed/store pre-loop steps when required;
- one provider-ready loop with loop `setvl`, source loads, optional compare and
  masked-product/merge steps, accumulator load where required, contraction
  compute, and store steps.

The provider-plan getter must consume verified typed body/config/runtime facts,
same-analysis route materialization facts, the RVV-owned route-control provider
plan, and RVV-owned math operand-binding facts before route construction. The
statement owner selection boundary must derive and consume this provider plan
before `TCRVEmitCLowerableRoute` construction rather than asking the central
provider to pass a prebuilt direct-contraction provider plan. The
provider-plan-backed statement builder must consume the provider plan rather
than rediscovering those facts. Neither plan is a common EmitC fact, artifact
metadata, an acceptance/status mirror, or a route-support declaration by
itself.

For product-reduction dequantization and dequant-clamp direct contraction
routes, the selected typed body legitimately carries both an i32 accumulator
stage and an f32 dequantized result stage. The direct contraction family plan
must expose the provider route result element as f32, while its typed config
mirror may accept the same-analysis selected typed config element as either the
i32 accumulator element or the f32 result element. SEW, LMUL, tail/mask policy,
config contract, VL C type, setvl leaf, route-control plan, materialization
facts, and math operand-binding facts must still mirror the same selected route
analysis exactly. Accepting f32 here is not metadata authority; rejecting stale
family-plan element mirrors remains fail-closed before `TCRVEmitCLowerableRoute`
construction.

The product-reduction dequantization epilogue is not the standalone
`dequantize_i32_to_f32` vector route. For
`widening_product_reduce_dequantize_f32` and
`widening_product_reduce_dequant_clamp_f32`, the RVV direct-contraction owner
keeps the post-loop scalar expression `dot_acc_scalar * scale` as the
provider-derived dequantization fact and then materializes the final VL=1 f32
vector through the provider-owned `rhs_broadcast_intrinsic` scalar splat. The
product-reduction route description and target metadata may mirror
`dequantization_relation`, `dequant_scale_role`, `dequant_scale_c_type`,
`dequant_scale_name`, and `rhs_broadcast_intrinsic`. They must not mirror
standalone vector-dequant leaves such as `tcrv_rvv.dequantize_convert_intrinsic`
or `tcrv_rvv.dequantize_scale_intrinsic`; those keys remain valid only for the
standalone dequant/dequant-clamp epilogue route families.

### 4. Validation & Error Matrix

- A non-contraction route requests the boundary -> return an empty direct
  contraction provider/statement plan without changing unrelated provider
  behavior.
- A contraction route has no verified contraction route-family materialization
  facts -> fail closed before route construction.
- A contraction route lacks the RVV-owned route-control provider plan, carries
  stale materialization/control facts, has an invalid runtime AVL/VL source, or
  has SEW/LMUL/policy/capability mirrors that disagree with the typed body/config
  and selected target facts -> fail closed before route construction.
- A product-reduction dequantization or dequant-clamp route carries a family-plan
  result element mirror other than f32, or typed config facts that are neither
  the same-analysis i32 accumulator nor f32 result config -> fail closed before
  route construction.
- A product-reduction dequantization or dequant-clamp route lacks the
  provider-derived post-loop scalar dequant splat leaf, scale role/C type/name,
  or dequantization relation -> fail closed before route construction or target
  artifact acceptance.
- A product-reduction dequantization or dequant-clamp candidate carries
  `tcrv_rvv.dequantize_convert_intrinsic` or
  `tcrv_rvv.dequantize_scale_intrinsic` metadata -> fail closed before target
  artifact acceptance as stale standalone vector-dequant mirror authority.
- A contraction route lacks same-analysis math operand-binding facts for its
  specific sub-family -> fail closed before route construction.
- The owner selection boundary cannot derive the direct contraction provider
  plan from the same-analysis route materialization facts and math
  operand-binding facts -> fail closed before statement construction and before
  `TCRVEmitCLowerableRoute` construction.
- Required ABI roles such as compare lhs/rhs, dot lhs/rhs, lhs/rhs,
  accumulator, output, runtime element count, or strided-input strides are absent
  -> fail closed through the provider plan with the logical operand name and
  operation/memory-form context.
- Required materialized leaves such as `setvl`, source load, compare vector
  load, strided source load, widening product, masked widening product,
  masked merge, scalar seed splat, accumulator load, contraction compute, or
  store are absent -> fail closed through the provider plan before common EmitC.
- Required source operation provenance for configure/load/compute/store steps is
  absent or reports the wrong EmitC source role -> fail closed before common
  EmitC.

### 5. Good/Base/Bad Cases

- Good: typed widening MAcc `tcrv_rvv` body -> contraction family plan verifier
  -> materialization facts -> math operand-binding facts -> route-control
  provider plan -> statement-plan owner selection derives the direct
  contraction provider plan -> direct contraction owner statement plan ->
  provider creates `TCRVEmitCLowerableRoute` and attaches statements into the
  route.
- Good: typed computed-mask strided widening dot-reduction body -> contraction
  family plan verifier with compare producer, strided-input, accumulator/result,
  product/seed, and runtime-control facts -> materialization facts -> math
  operand-binding facts -> route-control provider plan -> statement-plan owner
  selection derives the direct contraction provider plan -> direct contraction
  owner statement plan -> provider-built route.
- Base: migrated elementwise, compare/select, widening conversion, standalone
  reduction, plain MAcc, memory, segment2, and computed-mask accumulation routes
  remain outside this owner and are handled by the migrated statement-plan
  boundary.
- Bad: provider body branches on direct contraction operation names, ABI strings,
  memory forms, field-role mirrors, intrinsic mirrors, route ids, or artifact
  metadata to rebuild the contraction statement sequence after this owner exists.

### 6. Tests Required

- Focused C++ tests for direct contraction owner registry membership, hook
  presence, exact-once classification for every active direct-provider
  contraction route, and empty-plan behavior for unrelated routes.
- Focused C++ tests for positive direct contraction provider-plan construction
  and owner statement-plan construction, including route-control provider-plan
  consumption and math operand-binding facts.
- Focused C++ fail-closed diagnostics for missing contraction materialization
  facts, stale route-control/family facts, missing math operand-binding facts,
  same-analysis provider-plan derivation failure inside owner selection, and
  missing required materialized leaves.
- Provider-route tests proving `RVVEmitCRouteProvider` attaches the returned
  direct contraction owner statements into `TCRVEmitCLowerableRoute` without
  centrally obtaining or validating the direct contraction provider plan.
- Representative generated-bundle or `tcrv-translate` dry-run coverage for one
  direct-provider contraction route and one migrated statement-plan route.
- Product-reduction dequantization/dequant-clamp coverage must assert
  `rhs_broadcast_intrinsic` and the `dequant-splat` C type mapping summary, and
  must prove inserted stale `dequantize_convert_intrinsic` or
  `dequantize_scale_intrinsic` metadata is rejected by target validation.
- Bounded provider scan showing direct contraction statement construction is
  reached through the RVV-owned direct contraction owner and that the provider
  no longer carries central direct-contraction provider-plan construction,
  provider-fact validation, or statement branches.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

### 7. Wrong vs Correct

Wrong:

```text
RVVEmitCRouteProvider:
  create TCRVEmitCLowerableRoute
  build or validate direct contraction provider plan centrally
  call direct contraction statement owner with a central provider-plan aggregate
```

Wrong:

```text
widening_product_reduce_dequantize_f32 metadata
  -> tcrv_rvv.dequantize_convert_intrinsic = __riscv_vfcvt_f_x_v_f32m1
  -> tcrv_rvv.dequantize_scale_intrinsic = __riscv_vfmul_vf_f32m1
  -> target accepts product-reduction route through standalone dequant mirrors
```

Correct:

```text
verified contraction family/materialization/math facts
  -> statement-plan owner selection derives one direct contraction provider plan
  -> direct contraction statement owner consumes provider plan
  -> direct contraction provider facts validated against owner selection
  -> TCRVEmitCLowerableRoute construction
  -> provider attaches returned statements
```

Correct:

```text
widening_product_reduce_dequantize_f32 selected body
  -> RVV direct-contraction owner derives dot_acc_scalar * scale
  -> provider mirrors rhs_broadcast_intrinsic for the final f32 VL=1 splat
  -> target rejects standalone vector-dequant convert/scale metadata keys
```

## MAcc And Direct-Contraction Artifact Contract Core

### 1. Scope / Trigger

Use this contract when the RVV provider builds target-artifact validation
contracts for production-active MAcc and direct-contraction families. It covers
facts shared by `RVVMAccRouteValidationContract` and
`RVVWideningDotReduceRouteValidationContract` before target artifact
acceptance: route token, memory form, config contract, runtime-control and ABI
mirrors, headers, C type mapping summary, route operand binding plan/summary,
target leaf profile, provider-supported mirror, typed compute op, VL C type,
source/result/mask C type names, and runtime ABI parameter order.

### 2. Signatures

The durable provider contract surface is:

```c++
struct RVVContractionArtifactContractCore {
  std::string emitCRouteID;
  RVVSelectedBodyMemoryForm memoryForm;
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
  std::string vlCType;
  std::string resultVectorTypeName;
  std::string resultVectorCType;
  std::string sourceVectorTypeName;
  std::string sourceVectorCType;
  std::string maskTypeName;
  std::string maskCType;
  llvm::SmallVector<support::RuntimeABIParameter, 9> runtimeABIParameters;
};

RVVContractionArtifactContractCore getRVVContractionArtifactContractCore(
    const RVVSelectedBodyEmitCRouteDescription &description,
    RVVSelectedBodyMemoryForm memoryForm,
    llvm::StringRef runtimeControlPlanID,
    llvm::StringRef runtimeABIOrder,
    llvm::StringRef targetLeafProfile,
    llvm::StringRef providerSupportedMirror,
    llvm::StringRef requiredHeaderDeclarations,
    llvm::StringRef cTypeMappingSummary,
    llvm::StringRef routeOperandBindingPlanID,
    llvm::StringRef routeOperandBindingSummary,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef vlCType,
    llvm::StringRef resultVectorTypeName,
    llvm::StringRef resultVectorCType,
    llvm::StringRef sourceVectorTypeName,
    llvm::StringRef sourceVectorCType,
    llvm::StringRef maskTypeName,
    llvm::StringRef maskCType,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters);
```

### 3. Contracts

- MAcc and widening dot-reduce validation-contract builders must populate and
  retain this core before target artifact validation.
- Target artifact validators for these families must consume the core for
  shared provider-owned facts instead of duplicating local target-side copies.
- Family-specific facts stay in their owner contracts: MAcc accumulator/result,
  arithmetic, mask/passthrough, and statement facts remain MAcc-owned;
  dot-reduction product/reduction/dequant/clamp, strided-input, mask,
  inactive-lane, scalar seed/result, and statement facts remain
  contraction-owned.
- The core is provider output after selected typed-body analysis. It is not a
  route selector, artifact metadata authority, Common EmitC semantic branch, or
  replacement for family-specific validation.

### 4. Validation & Error Matrix

- Missing core route token, header declarations, type mapping summary, runtime
  ABI order, binding summary, or provider mirror -> target validation fails
  before artifact acceptance.
- Core runtime ABI parameters differ from the provider route description or
  rebuilt `TCRVEmitCLowerableRoute` ABI mappings -> target validation fails.
- Core runtime-control mirrors disagree with the selected-boundary runtime
  AVL/VL contract -> target validation fails.
- MAcc-specific or dot-specific facts are absent even when the core is valid ->
  the relevant owner validator still fails closed.
- Candidate metadata matching the core is insufficient if the rebuilt provider
  route or family-specific facts disagree.

### 5. Good/Base/Bad Cases

- Good: typed MAcc or direct-contraction body -> route-family facts -> shared
  artifact contract core -> family-specific validation contract -> target
  artifact validator consumes core plus family fields.
- Base: existing MAcc and widening dot-reduce generated-bundle fixtures keep
  their ABI/header behavior while target validation reads the common fields
  through the core.
- Bad: target validation accepts a MAcc or dot-reduce artifact because metadata,
  artifact name, test name, route id, or exact intrinsic spelling looks
  plausible while the provider core is absent or stale.

### 6. Tests Required

- C++ target artifact tests must keep failing closed for stale provider mirrors,
  header/type summaries, route operand bindings, runtime ABI order, and ABI
  mappings on both a MAcc representative and a direct-contraction
  representative.
- Focused generated-bundle or target artifact dry-runs must cover at least one
  MAcc route and one widening dot/direct-contraction route after the core is
  changed.
- Runtime `ssh rvv` evidence is required only when emitted code, ABI/header
  shape, generated harness behavior, runtime correctness, or performance is
  newly claimed.

### 7. Wrong vs Correct

Wrong:

```text
target artifact validator
  -> reads route metadata / artifact name / route id
  -> accepts MAcc or dot-reduce artifact
```

Correct:

```text
provider route description + family route facts
  -> RVVContractionArtifactContractCore
  -> MAcc or widening dot validation contract
  -> target validator consumes shared core plus family-specific facts
```

## Low-Precision Direct-Contraction Resource-Aware Closure

### 1. Scope / Trigger

Use this contract when RVV Stage 2 work claims that low-precision contraction,
Gearbox scheduling, q8/q4-like examples, or llama.cpp parity/performance is the
current bottleneck. This contract does not make q8, q4, llama.cpp, route ids,
artifact names, test names, or handwritten intrinsic spellings authoritative.
They are pressure tests for whether typed low-precision `tcrv_rvv` bodies,
RVV-local realization/provider facts, and real runtime evidence exist.

The bounded Gearbox schedule materialization pass is only an MVP unless it
enumerates legal candidates, estimates resource pressure, rejects impossible or
unsafe candidates, and consumes the selected schedule into realized `tcrv_rvv`
structure or provider-owned plans before route construction.

Current implementation calibration:

- `--tcrv-rvv-materialize-gearbox-schedules` is the currently registered RVV
  Gearbox MVP pass. It materializes bounded static schedule facts for selected
  typed `dequantize_i32_to_f32` bodies and must not be described as the completed
  resource-aware autotuner.
- The low-precision direct-contraction resource candidate contract introduced
  after the product-dequant route repairs is now shared by the bounded Gearbox
  pass, selected-body realization owner, and provider pre-route validator for
  product-reduction-dequantize and product-reduction-dequant-clamp selected
  bodies. It is still a bounded static model, not a completed runtime autotuner.
- Product-dequant and product-dequant-clamp executable ABI evidence proves route
  and artifact behavior for those representatives. It is not a performance or
  llama.cpp parity claim.

The target RVV plugin-local pass pipeline is:

```text
selected typed tcrv_rvv body
  -> build resource candidates
  -> prune by legality/resource model
  -> select resource candidate
  -> realize selected candidate into tcrv_rvv structure
     or transitional provider-owned plan
  -> provider-built TCRVEmitCLowerableRoute
```

During the transitional period, provider-owned plans are acceptable only when
they are validated and consumed before route construction. They must not become
metadata authority.

### 2. Signatures

Resource-aware low-precision contraction work must introduce or preserve a
provider-local contract equivalent to:

```c++
struct RVVLowPrecisionContractionResourceCandidate {
  planning contract id;
  element/source/result dtype facts;
  product and accumulator dtype facts;
  widening-product resource-candidate fact;
  widening-reduction resource-candidate fact;
  SEW, LMUL, product EMUL, accumulator EMUL;
  operand form, source signedness, storage/effective element width;
  packing layout and unpack intent when storage differs from effective width;
  memory form, stride facts, mask/tail policy;
  unroll factor, accumulator count, reduction layout;
  vsetvl placement / region count;
  estimated peak live vector groups;
  estimated load/store/mask/accumulator live ranges;
  legality decision and rejection reason when not legal;
};

struct RVVLowPrecisionContractionResourceSelection {
  planning contract id;
  selected candidate id;
  selected resource candidate;
  selected widening-product and widening-reduction candidate facts;
  selected-body realization producer and decision;
  realized unroll, vsetvl region count, and peak live-vector groups;
  realized product/dequant region indexes and product/dequant phases;
  selected-body realization or provider-consumed owner plan;
  runtime AVL/VL and ABI parameter mapping;
  target capability/profile facts used by the selector;
};
```

Exact C++ names may differ, but these fields must have a structural home before
the route provider claims resource-aware tuning.

The current production resource-planning boundary uses:

```text
tcrv_rvv.low_precision_resource.planning_contract =
  "rvv-low-precision-production-resource-planning-contract.v1"

tcrv_rvv.vsetvl_region_marker %vl {
  phase,
  planning_contract =
    "rvv-low-precision-production-resource-planning-contract.v1",
  region_index,
  region_count,
  resource_decision
}
```

`RVVLowPrecisionContractionResourceCandidate` and
`RVVLowPrecisionContractionResourceSelection` must carry this id when a bounded
low-precision product-reduction resource candidate is selected. Gearbox schedule
materialization, selected-body realization, provider route-family validation,
and target artifact validation consume the same id as a fail-closed marker and
handoff check. It is not route authority and is not derived from artifact
metadata, route ids, q8/q4 names, helper names, or Common EmitC.

### 3. Contracts

- Candidate facts must be derived from typed selected-body/config/runtime facts
  and target capability facts. They must not be inferred from q8/q4 names,
  benchmark names, route ids, generated artifact names, or C ABI strings.
- The first competitive low-precision target should be one narrow,
  apples-to-apples direct-contraction kernel such as q8_0_q8_0-equivalent int8
  widening dot/reduce. If the typed surface cannot express it, the task must
  fail closed with the exact missing primitive surface.
- Resource pruning must consider at least SEW/LMUL/EMUL legality, widening
  pressure, accumulator count, mask/v0 usage, peak live vector groups, memory
  form/stride, and vsetvl region count.
- Resource-aware pruning happens before optional profile/runtime feedback. It is
  not pure black-box benchmarking after code generation.
- A selected candidate must be consumed by selected-body realization, provider
  planning, or target artifact validation before route construction. Mirroring a
  candidate in artifact metadata is insufficient.
- The low-precision resource-planning contract id must be attached by the
  RVV-owned Gearbox/resource candidate path and copied into realized
  product-reduction/dequant or product-reduction/dequant-clamp `with_vl` facts.
  Provider route-family validation must require the same id before
  `TCRVEmitCLowerableRoute` construction. Target artifact validation may compare
  metadata mirrors against provider-owned facts, but metadata does not invent or
  repair the planning contract.
- When the selected low-precision Gearbox candidate affects `vsetvl` placement
  or region count, selected-body realization must materialize provider-verifiable
  realized body structure for that placement. The provider must compare the
  realized structure with the RVV-owned resource facts and fail closed if marker
  count, ordering, phase, resource decision, or bound `!tcrv_rvv.vl` token is
  stale or inconsistent. Such structure is RVV plugin-local scheduling evidence;
  it is not a common EmitC role op, route id, artifact mirror, or intrinsic
  authority.
- For generated-bundle evidence on pre-realized
  `widening_product_reduce_dequantize_f32`, the Gearbox resource-fact pass must
  run before selected-body realization so `tcrv_rvv.low_precision_resource.*`
  facts are attached to the pre-realized body and copied into the realized
  `with_vl` operation before provider planning. Running selected-body
  realization first is a fail-closed missing-resource-fact path, not a valid
  executable pipeline. Plain `dequantize_i32_to_f32` Gearbox evidence may still
  run the Gearbox pass after selected-boundary materialization because that MVP
  schedule pass consumes the realized `with_vl`/`dequantize` structure.
- For low-precision product-reduction selected bodies, selected-body realization
  must consume both provider-owned widening-reduction primitive facts and
  pass-produced `tcrv_rvv.low_precision_resource.*` facts before materializing
  realized product/reduction/dequantize structure. The realization owner must
  compare typed source/product/accumulator/result dtype and SEW/LMUL, product
  and product-reduction relations, accumulator/result layout, scalar seed and
  reduction-store facts, selected widening-product and widening-reduction
  resource-candidate facts, policy, memory form, runtime AVL/ABI ordering,
  `vsetvl` region count, live-vector pressure, and selected resource decision
  with the RVV-owned facts. The candidate facts summarize the provider-selected
  widening product and widening reduction resources after typed primitive facts
  are chosen; they are not route ids, artifact names, helper-op names, or
  Common EmitC semantics. Missing or stale primitive/resource combinations fail
  closed before route construction; Common EmitC and artifact metadata remain
  mirror consumers only.
- For bounded low-precision product-reduction Gearbox candidates, candidate
  build/prune/select semantics must be shared by the Gearbox pass,
  selected-body realization owner, and provider pre-route validator. Selection
  inputs include memory form, policy, source/product/accumulator/result
  SEW/LMUL, runtime AVL source, and vector-register budget. A candidate whose
  peak-live vector-group estimate exceeds the vector-register budget, whose
  policy is unsupported, or whose typed shape is unsupported must be pruned with
  a targeted diagnostic before realization or route construction.
- Once a candidate is selected, that candidate owns the realization decision:
  realized `with_vl` attrs, producer/consumer `tcrv_rvv.vsetvl_region_marker`
  ops, and the Gearbox cross-region handoff must carry marker count, ordering,
  phase, scope, runtime AVL source, planning contract, and resource decision
  derived from the selected candidate. The provider must derive the expected
  planning contract and resource decision from the selected candidate and reject
  stale marker/handoff/resource facts before constructing
  `TCRVEmitCLowerableRoute`.
- For Gate 3 low-precision product-reduction realization consumption, producer
  and consumer realized `tcrv_rvv.with_vl` ops must carry the provider-verifiable
  realization facts `realization_producer`, `realization_decision`,
  `realized_unroll_factor`, `realized_vsetvl_region_count`,
  `realized_peak_live_vector_groups`, `product_region_index`,
  `dequant_region_index`, `product_phase`, and `dequant_phase`. The route
  planner must read these fields from the realized body before route
  construction and compare them with the selected candidate/resource decision.
  It must not rebuild accepted product/dequant region indexes or phases solely
  from route ids, artifact metadata, helper names, intrinsic strings, or Common
  EmitC inference.
- For Gate 3B dequant-clamp realization consumption, a selected
  `product-reduction-dequant-clamp-f32` candidate must additionally carry
  `clamp_region_index`, `clamp_phase`,
  `clamp_compare_select_phase`, and `clamp_select_layout` on the realized
  producer and consumer `tcrv_rvv.with_vl` scopes, on the Gearbox handoff, in
  route-plan metadata, and in target artifact/header mirrors. The RVV provider
  derives these four fields from the selected resource candidate and realized
  compare/select structure; non-clamp candidates must reject these fields, and
  stale dequant-clamp values must fail before
  `TCRVEmitCLowerableRoute` construction or target artifact acceptance.
- For low-precision product-reduction Gearbox markers,
  `tcrv_rvv.vsetvl_region_marker` must carry `planning_contract =
  "rvv-low-precision-production-resource-planning-contract.v1"` when
  `resource_decision` is a supported low-precision realization decision. The
  marker verifier, handoff verifier, selected-body route planner, and
  contraction route-family validation must reject missing or stale marker
  planning contracts before Common EmitC route materialization. A marker-only
  resource decision without this contract is still marker-level metadata, not a
  valid selected resource-plan consumer.
- For product-reduction Gearbox selected-body realization, the cross-region
  handoff must also carry the selected primitive-chain resource facts:
  `primitive_chain_contract`, `primitive_chain_kind`,
  `primitive_source_signedness`,
  `primitive_widening_product_relation`,
  `primitive_product_reduction_chain_relation`,
  `primitive_widening_product_intrinsic`, `primitive_reduction_intrinsic`,
  `primitive_scalar_seed_splat_intrinsic`, `primitive_accumulator_layout`,
  `primitive_result_layout`, `primitive_reduction_store_vl`,
  `widening_product_candidate_fact`, and `reduction_candidate_fact`. These
  fields are derived from the selected RVV resource candidate and the typed
  product/reduction body. The handoff verifier, selected-body owner validation,
  and route planning must reject missing or stale values before Common EmitC or
  target artifact export can accept the route. Good: realized handoff copies the
  selected `vwredsum` primitive fact and the selected reduction candidate fact.
  Bad: a handoff keeps marker/resource decision facts but carries a stale
  reduction intrinsic string or stale reduction candidate string.
- For product-reduction dequant/dequant-clamp direct-contraction routes, the
  statement-plan owner must re-consume the provider-owned low-precision resource
  selection before constructing statement steps. It must compare the provider
  plan and route-family plan selection mirrors, then require either the accepted
  byte operand facts (`unpacked-byte-elements`, signed source, 8-bit
  storage/effective element width, `one-element-per-byte`, and
  `none-direct-widening-product`) or the accepted packed-i4 facts below. Stale
  packed/sub-byte claims that do not match the selected resource family fail at
  this statement consumer boundary before Common EmitC or target artifact export
  can treat the path as executable.
- A positive packed sub-byte product-reduction candidate may be introduced only
  as explicit typed/resource facts, not as q4/q8/llama.cpp, route-id,
  artifact-name, descriptor, or benchmark authority. The first bounded positive
  packed family is `packed-i4-nibbles`: signed source, 8-bit storage element,
  4-bit effective element, layout
  `two-signed-i4-elements-per-byte-low-high-nibbles`, and unpack intent
  `sign-extend-i4-nibbles-before-widening-product`. Selected-body realization
  and provider planning may accept this candidate as a coherent structural
  resource selection only if these facts, dtype/SEW/LMUL/product/accumulator/
  result/runtime facts, resource decision, and mirrors all agree.
- The bounded packed-i4 statement boundary is RVV-provider-owned. For a selected
  signed packed-i4 product-reduction candidate, direct-contraction statement
  planning loads packed i8 vectors for both operands, sign-extends the low
  nibble by shift-left 4, computes the shifted low widening product, arithmetic
  rescales that i16 product right by 8, sign-extends the high nibble by
  arithmetic shift-right 4, accumulates the high-nibble signed widening product
  into the rescaled low product with `__riscv_vwmacc_vv_i16mf2`, and performs
  exactly one `__riscv_vwredsum_vs_i16mf2_i32m1` from that pair-sum vector into
  the i32 accumulator. The current provider leaves are
  `__riscv_vsll_vx_i8mf4`, `__riscv_vsra_vx_i8mf4`,
  `__riscv_vwmul_vv_i16mf2`, `__riscv_vsra_vx_i16mf2`,
  `__riscv_vwmacc_vv_i16mf2`, and
  `__riscv_vwredsum_vs_i16mf2_i32m1`; their use is derived from the selected
  packed-i4 resource facts, not from route ids or artifacts. This boundary
  proves statement-plan and `TCRVEmitCLowerableRoute` eligibility only.
- The target artifact/export boundary for the accepted signed packed-i4
  representative must consume the provider-owned resource facts and rebuilt
  route statement payload. It accepts artifact export only when the route
  carries packed source loads, low-nibble shift-left statements for both
  operands, the shifted low widening product, the low-product arithmetic
  rescale, high-nibble arithmetic shift-right statements for both operands,
  the high-nibble widening product-accumulate statement, one widening reduction
  from that pair-sum, and the final carry
  assignment derived from the selected packed-i4 resource facts. Stale resource
  mirrors, stale unpack-intent metadata, missing nibble payloads, mismatched
  low-product or high-vwmacc operands, stale single
  reduction input/result, or a stale carry assignment fail closed in the target
  bridge. Artifact support is still not generated-bundle support, runtime
  correctness, timing, or parity evidence.
- Do not apply the product-reduction byte operand-form contract to every
  low-precision resource representative. For example, strided or computed-mask
  strided widening-dot resource representatives may use a different operand
  form such as `unpacked-source-elements`; those families need their own
  explicit resource contract and tests. A product-reduction packed-fact guard
  that rejects sibling low-precision representatives is over-broad.
- Runtime/performance parity claims require generated TianChen-RV output and the
  baseline RVV implementation to run on the same named `ssh rvv` environment with
  correctness checked before timing.
- Executable generated-bundle correctness for the accepted packed-i4
  product-reduction representative must select its external scalar/reference
  oracle from validated provider-owned low-precision resource metadata, not from
  fixture names, route ids, artifact names, or op-kind strings alone. When the
  object/header metadata validates `packed-i4-nibbles`, the same-target harness
  treats runtime `n` as packed input bytes, sign-extends both low and high
  signed i4 nibbles from each byte, accumulates both products per byte into the
  scalar seed, then applies the runtime f32 scale. The default unpacked-byte
  product-dequant path must keep the existing byte-wise `i8*i8` oracle.
- Generated-bundle evidence for the accepted packed-i4 product-reduction
  representative must also verify object/header mirrors for the provider-owned
  low-precision realization schedule: realization producer/decision, realized
  unroll, realized `vsetvl` region count, realized peak live-vector groups,
  product/dequant region indices, product/dequant phases, runtime ABI order, and
  target capability mirrors. The evidence JSON may expose these facts through a
  mirror-only summary such as `generated_artifact_resource_schedule_evidence`,
  but the summary is valid only when object/header metadata and expected
  provider facts agree exactly.
- Same-target timing for the accepted packed-i4 product-reduction representative
  must use generated TianChen-RV output and a named scalar C baseline on the
  same `ssh rvv` target after correctness guards pass. The packed baseline
  identity is `scalar-c-reference/product-reduction-dequant-packed-i4-v1`; it is
  a comparator/oracle identity only. The measurement harness may select that
  baseline only after the generated object/header bundle metadata validates the
  provider-owned `packed-i4-nibbles` resource facts. A default unpacked-byte
  product-dequant measurement must keep
  `scalar-c-reference/product-reduction-dequant-v1` and must not emit
  `packed_i4_reference_oracle`.
- Full runtime autotuning caches are not required for the first closure. A
  bounded static resource model is acceptable if it is explicit, tested, and
  consumed by provider/target contracts.

### 4. Validation & Error Matrix

- A low-precision contraction route lacks typed source/product/accumulator/result
  dtype facts -> fail closed before provider route construction.
- Widening product or accumulator EMUL exceeds legal RVV limits for the selected
  LMUL -> reject the candidate before materialization.
- Estimated peak live vector groups exceed the target vector register budget
  after reserved mask/v0 usage -> reject or choose a smaller unroll/LMUL.
- Candidate selection exists only as artifact metadata, test name, route token,
  benchmark name, or emitted C spelling -> fail closed as non-authoritative.
- The selected resource candidate or realized product-reduction body carries a
  missing or stale `tcrv_rvv.low_precision_resource.planning_contract` value ->
  fail closed in RVV Gearbox/selected-body/provider validation before Common
  EmitC materializes a route.
- A realized `tcrv_rvv.vsetvl_region_marker` carries a supported
  low-precision resource decision but has missing or stale `planning_contract`
  -> fail closed in the RVV marker verifier or provider marker-structure
  validation before route construction.
- A realized product-reduction `with_vl` reaches route planning with missing or
  stale realization producer/decision, realized unroll/region/peak-live facts,
  product/dequant region indexes, or product/dequant phases -> fail closed
  before `TCRVEmitCLowerableRoute` construction; do not accept a planner-local
  reconstruction as the source of truth.
- Target artifact export sees a stale planning-contract metadata mirror ->
  fail closed by comparing the mirror to provider-owned resource facts; do not
  use the mirror to repair the provider selection.
- Target artifact export sees a stale widening-product or widening-reduction
  candidate metadata mirror -> fail closed by comparing the mirror to
  provider-owned resource selection facts before accepting the artifact.
- A performance comparison omits baseline identity, target profile, compile
  flags, input sizes, correctness check, timing method, or raw `ssh rvv` evidence
  -> it is not performance evidence.
- A q8/q4-named path bypasses typed `tcrv_rvv` body authority or RVV provider
  validation -> fail closed.
- A selected packed-i4 product-reduction candidate carries storage width 8 but
  effective width not 4, an unknown layout, missing signedness, stale unpack
  intent, or byte operand-form mirrors -> fail closed in route-family/provider
  validation before statement planning.
- A selected packed-i4 product-reduction candidate reaches statement planning
  without the exact selected packed-i4 resource facts, runtime AVL/VL facts,
  source/product/accumulator/result facts, or matching provider/family mirrors
  -> fail closed at the RVV statement consumer boundary.
- A selected packed-i4 statement plan reuses the unpacked byte `lhs_vec` /
  `rhs_vec` product path, omits low/high nibble sign-extension statements, uses
  a logical shift-right for signed high-nibble unpack, reduces low/high products
  separately, omits the high-nibble vwmacc accumulate, or reduces anything other
  than the product-pair sum -> fail closed before `TCRVEmitCLowerableRoute`
  construction.
- A provider-built packed-i4 statement route reaches target artifact export
  with stale resource mirrors, stale unpack-intent metadata, missing low/high
  nibble payload, mismatched low-product rescale operands, mismatched
  high-nibble vwmacc operands, a single reduction input not derived from the
  product-pair sum, a stale single reduction result, or a final carry assignment
  not derived from that single reduction -> fail closed in the target artifact
  bridge.
- A packed-i4 artifact/header/source result is treated as generated-bundle,
  runtime correctness, performance, or llama.cpp parity evidence without the
  matching executable generated-bundle and `ssh rvv` checks -> fail closed as an
  evidence-boundary violation.
- A packed-i4 generated-bundle executable harness compares generated output
  against the default byte-wise `i8*i8` scalar oracle, or switches to packed
  low/high signed-i4 semantics because of a fixture name or script option rather
  than validated provider resource metadata -> fail closed as a same-target
  reference-oracle violation.
- A packed-i4 generated-bundle evidence path accepts missing or stale
  low-precision realization schedule mirrors, or lets artifact/header metadata
  disagree with provider-owned expected fields -> fail closed before evidence is
  accepted.
- A packed-i4 same-target measurement omits the packed scalar baseline identity,
  target profile, compile flags, input sizes, correctness guard, timing method,
  raw `ssh rvv` timing records, or parsed summaries -> it is not packed-i4
  timing evidence.
- A default product-dequant timing dry-run or runtime measurement emits
  `packed_i4_reference_oracle`, or a packed-i4 timing measurement selects its
  baseline before object/header metadata validates provider-owned packed facts
  -> fail closed as an evidence-boundary violation.

### 5. Good/Base/Bad Cases

- Good: typed int8/u8 low-precision selected body -> widening product and
  reduction/accumulator facts -> bounded resource candidate set -> legal selected
  candidate -> realized `tcrv_rvv` body or provider-consumed owner plan ->
  provider route -> generated RVV C/C++ -> same-target correctness/timing
  comparison against a named llama.cpp RVV baseline.
- Good: typed signed packed-i4-in-i8 product-reduction-dequant candidate with
  explicit `packed-i4-nibbles`, storage width 8, effective width 4, low/high
  nibble layout, and sign-extension unpack intent -> selected-body realization
  and provider mirrors consume the resource facts -> direct-contraction
  statement planning emits RVV-owned low-nibble shift-left statements, a shifted
  low widening product, low-product i16 rescale, high-nibble sign-extension
  statements, a high-nibble vwmacc accumulate into the rescaled low product, and
  one widening reduction from the pair-sum -> provider-built
  `TCRVEmitCLowerableRoute` is eligible ->
  target artifact export accepts only the exact rebuilt provider payload and
  mirrors -> generated-bundle/runtime evidence remains a separate gate.
- Good: a product-reduction/dequant selected body reaches provider validation
  with `rvv-low-precision-production-resource-planning-contract.v1` copied from
  the selected RVV resource candidate into the realized body and route
  selection; provider and target validators compare the same provider-owned
  contract before route or artifact acceptance.
- Good: selected-body realization consumes the grouped or packed-i4
  low-precision candidate, writes product/dequant region indexes and phases to
  both realized producer/consumer `with_vl` scopes, and route planning rejects a
  stale `product_phase` or `dequant_region_index` before route construction.
- Base: existing MAcc, widening dot-reduce, dequant, and Gearbox MVP routes keep
  their current route-support contracts without claiming performance parity.
- Bad: q8_0_q8_0 appears in a test or artifact name -> route provider emits a
  handwritten intrinsic sequence and claims llama.cpp parity without typed body
  facts, resource estimates, correctness checks, or timing evidence.
- Bad: packed-i4 selected candidate -> provider accepts mirrors -> statement
  planner silently reuses `unpacked-byte-elements` widening-product statements
  or target artifact export claims executability before Gate 4 validation.
- Bad: artifact metadata names
  `rvv-low-precision-production-resource-planning-contract.v1`, but the selected
  RVV resource selection is missing or carries a different planning contract;
  provider/target acceptance must fail instead of treating metadata as a source
  of truth.

### 6. Tests Required

- lit/FileCheck coverage for selected-body realization or provider-consumed owner
  schedule facts when resource choices affect generated code.
- C++ tests for resource candidate legality and rejection diagnostics covering
  EMUL overflow, live-vector-group pressure, missing low-precision dtype facts,
  stale candidate mirrors, and invalid ABI/runtime AVL facts.
- Provider/target artifact validation proving selected candidate facts are
  consumed before artifact acceptance and stale metadata-only candidates fail.
- Focused C++ or lit tests proving the planning contract is produced by the
  Gearbox/selected-body resource path, copied into realized
  `tcrv_rvv.vsetvl_region_marker` and Gearbox handoff structure, consumed by
  provider validation, and rejected when stale or missing before route
  construction or artifact export.
- Focused C++ statement-plan coverage for the positive packed-i4 low/high nibble
  unpack sequence, low-product rescale, high-nibble vwmacc accumulate, single
  widening reduction, and provider-built lowerable route eligibility without
  target artifact/runtime claims.
- Focused target/export coverage proving selected packed-i4 statement routes
  remain fail-closed at the Gate 4 artifact boundary until provider-payload
  validation and runtime evidence exist.
- Focused selected-body/provider tests for resource-fact and realized-structure
  mismatch, including stale `vsetvl` region placement structure when `vsetvl`
  placement or region count is part of the selected low-precision Gearbox
  candidate.
- Focused selected-body/provider tests proving route planning consumes
  realization-produced `with_vl` facts for product/dequant region indexes and
  phases, including stale `product_phase` and stale `dequant_region_index`
  negative cases before route construction.
- Focused selected-body/provider tests proving realized Gearbox handoff carries
  the selected primitive-chain contract/kind/relation/intrinsic/layout/store-VL
  fields and selected widening-product/reduction candidate facts, plus stale
  handoff primitive/candidate fact negative cases before route or artifact
  acceptance.
- A focused generated-bundle or benchmark harness for the first comparable
  low-precision direct-contraction kernel.
- Packed-i4 executable harness coverage must assert the validated metadata
  switch, low/high signed-i4 sign-extension, both low and high product
  accumulation per packed byte, runtime scale application, source/accumulator
  preservation, output sentinel preservation, and no change to the default
  unpacked-byte product-dequant oracle.
- Packed-i4 same-target timing coverage must assert the validated metadata
  switch, the named packed scalar baseline identity, `CLOCK_MONOTONIC_RAW`
  timing records, correctness guards before every measured case, target profile
  capture, raw repeat measurements, parsed summaries, and a default
  product-dequant regression proving packed timing/oracle text does not leak to
  the unpacked-byte baseline.
- Real `ssh rvv` correctness evidence for executable claims and real same-target
  timing evidence for performance or llama.cpp-parity claims.

### 7. Wrong vs Correct

Wrong:

```text
benchmark name says q8_0_q8_0
  -> emit fixed RVV intrinsic C
  -> artifact metadata says selected=u2
  -> claim Gearbox/performance maturity
```

Correct:

```text
typed low-precision tcrv_rvv body
  -> RVV resource candidate set
  -> legality/resource pruning
  -> selected-body realization or provider-consumed owner plan
  -> provider-built route
  -> generated RVV artifact
  -> same-target correctness and timing against named baseline
```

## Low-Precision Realization Admission Schedule Handoff

### 1. Scope / Trigger

Use this contract when a source-backed production pressure profile is used to
admit RVV low-precision selected-body realization and the selected packed-i4
Gearbox/resource schedule must be consumed by the contraction realization owner.
This is a Gate 2 selected-body realization boundary. It is not a route id,
artifact metadata, q4/q8 label, Common EmitC decision, or performance claim.

### 2. Signatures

The admission result must carry the schedule decision fields after pressure
profile validation:

```c++
struct RVVLowPrecisionSelectedBodyRealizationAdmission {
  std::string admissionContract;
  std::string admissionOwner;
  RVVLowPrecisionRealizationAdmissionDecision decision;
  std::string selectedCandidateID;
  std::string pressureProfileContract;
  std::string measurementEvidenceID;
  std::string dispatchPolicyPath;
  std::string scheduleDecisionContract;
  std::string scheduleDecision;
  std::string scheduleDecisionReason;
  std::string diagnostic;
};
```

The realized low-precision `with_vl` and Gearbox handoff may expose those facts
only as explicit realization-admission mirrors:

```text
tcrv_rvv.low_precision_resource.realization_admission_schedule_decision_contract
tcrv_rvv.low_precision_resource.realization_admission_schedule_decision
tcrv_rvv.low_precision_resource.realization_admission_schedule_decision_reason
```

### 3. Contracts

- The schedule decision fields come from the accepted
  `RVVLowPrecisionProductionPressureProfile`, after that profile has been
  matched against the selected packed-i4 resource candidate, source-backed
  measurement record, selected-dispatch boundary, primitive facts, runtime ABI
  order, and target capability mirrors.
- The contraction selected-body realization owner may materialize admitted
  schedule mirrors only when `admission.decision = Realize`.
- The admitted schedule mirrors must match the provider-owned packed-i4
  schedule contract, decision, and reason already selected by the RVV
  Gearbox/resource candidate path.
- Realization admission mirrors may be copied to realized `with_vl` operations
  and Gearbox handoff structure. They do not define compute semantics, dtype,
  runtime AVL/VL, dispatch/fallback behavior, route support, or intrinsic
  spelling.
- Common EmitC/export may carry provider-built route payloads and mirrors only;
  it must not infer or repair these schedule facts.

### 4. Validation & Error Matrix

- Missing source-backed pressure profile -> selected-body realization admission
  fails before schedule consumption.
- Missing packed-i4 schedule decision contract, decision, or reason on the
  selected resource candidate -> admission fails before realization mirrors are
  accepted.
- Pressure-profile schedule fields differ from the selected resource candidate
  -> admission fails with a schedule-decision diagnostic.
- Metadata-only, label-only, q4/q8, sibling-route, or non-source-backed
  schedule/profile facts -> admission fails before realized body facts are
  trusted.
- Realized handoff carries admitted schedule mirrors that differ from the
  selected candidate -> provider/handoff validation must fail before route
  construction.

### 5. Good/Base/Bad Cases

- Good: packed-i4 resource candidate -> source-backed pressure profile admits
  realization -> admission carries the same schedule contract/decision/reason
  -> realized `with_vl` and Gearbox handoff expose admission schedule mirrors.
- Base: low-precision realization without a production pressure profile can
  still use the older non-admission owner entry, but it must not claim Gate 2
  source-backed schedule admission.
- Bad: schedule decision appears only in artifact metadata, a route id, or a
  q4/q8 benchmark label, then realization treats it as accepted.
- Bad: a metadata-only schedule marker is copied to `with_vl` without the
  pressure-profile admission boundary.

### 6. Tests Required

- C++ admission tests must assert successful packed-i4 admission carries
  schedule decision contract, decision, and reason.
- C++ selected-body realization tests must assert the admitted schedule fields
  appear on realized `with_vl` and Gearbox handoff structure.
- Negative C++ tests must cover missing schedule contract, stale schedule
  decision, missing pressure profile, metadata-only schedule/profile facts, and
  stale selected-dispatch or source-backed measurement tie-backs.
- Bounded checks must prove Common EmitC/export did not gain RVV schedule
  inference logic.

### 7. Wrong vs Correct

Wrong:

```text
artifact metadata says packed-i4 schedule selected
  -> realization copies schedule attrs
  -> route claims resource-aware schedule consumption
```

Correct:

```text
selected packed-i4 resource candidate
  -> source-backed pressure profile validates schedule tie-back
  -> admission decision carries schedule contract/decision/reason
  -> selected-body realization mirrors admitted schedule on realized structure
  -> provider validates before route construction
```

## Low-Precision Gate 3 Realization Admission Proof Consumption

### 1. Scope / Trigger

Use this contract when the realized packed-i4 low-precision contraction body
has Gate 2 realization-admission schedule mirrors and the RVV route/provider,
target artifact, and same-target measurement policy path must prove that those
mirrors still match provider-owned resource facts and source-backed same-target
records. This is a Gate 3 proof-consumption boundary. It is not artifact
metadata authority, route-id authority, helper-name authority, q4/q8 label
authority, or Common EmitC schedule inference.

### 2. Signatures

The route/provider resource selection must carry the admission proof as
provider-owned facts before target artifact or measurement policy consumers can
accept a packed-i4 product-reduction route:

```c++
struct RVVLowPrecisionContractionResourceSelection {
  std::string realizationAdmissionContract;
  std::string realizationAdmissionDecision;
  std::string realizationAdmissionEvidence;
  std::string realizationAdmissionDispatchPolicy;
  std::string realizationAdmissionScheduleDecisionContract;
  std::string realizationAdmissionScheduleDecision;
  std::string realizationAdmissionScheduleDecisionReason;
};
```

Same-target record, policy input, measurement outcome, and production pressure
profile payloads must carry exact provider-prefixed mirrors:

```text
provider_realization_admission_contract
provider_realization_admission_decision
provider_realization_admission_evidence
provider_realization_admission_dispatch_policy
provider_realization_admission_schedule_decision_contract
provider_realization_admission_schedule_decision
provider_realization_admission_schedule_decision_reason
```

The route path may use:

```c++
llvm::Error populateRVVLowPrecisionSelectedBodyRealizationAdmissionProof(
    RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);
```

### 3. Contracts

- The proof is accepted only when
  `realizationAdmissionContract =
  rvv-low-precision-selected-body-realization-admission.v1` and
  `realizationAdmissionDecision = realize`.
- `realizationAdmissionEvidence` must equal the selected same-target
  `measurementEvidenceID` or the selection's remediation measurement evidence
  when a measured-win/remediation branch rewrites the evidence identity before
  the record is rebuilt.
- Admission schedule contract, decision, and reason must exactly equal the
  provider-owned packed-i4 schedule contract, decision, and reason already on
  the selected resource candidate.
- Target artifact metadata and generated headers may mirror these fields only
  after route/provider validation. They cannot create, repair, or select a
  schedule proof.
- Common EmitC/export remains neutral. It may transport a validated
  `TCRVEmitCLowerableRoute` payload and mirrored metadata, but it must not
  infer admission, schedule, measurement identity, dtype, ABI roles, or
  dispatch policy.

### 4. Validation & Error Matrix

- Missing admission contract/decision/evidence on a packed-i4 selection ->
  provider or target artifact validation fails before artifact acceptance.
- Admission decision other than `realize` -> measurement policy and target
  artifact validation fail before performance or artifact proof is accepted.
- Admission evidence differs from the source-backed same-target record identity
  -> record/policy validation fails as stale or sibling-route proof.
- Admission dispatch policy differs from selected-dispatch policy facts when a
  dispatch boundary is present -> policy validation fails before realization
  admission is trusted.
- Admission schedule contract, decision, or reason differs from the selected
  resource schedule facts -> provider, measurement policy, or target artifact
  validation fails with an admission schedule diagnostic.
- Artifact metadata contains a matching label but the provider selection lacks
  the proof -> target artifact validation fails as metadata-only proof.

### 5. Good/Base/Bad Cases

- Good: realized `with_vl`/Gearbox handoff carries Gate 2 admitted schedule
  mirrors -> route selection imports them as provider-owned proof -> same-target
  record/policy input and target artifact validation prove the same evidence
  and schedule before accepting the artifact path.
- Base: older non-dispatch fixtures may populate the proof from provider-owned
  selection facts without claiming selected-dispatch policy preference.
- Bad: measurement JSON copies the schedule decision from a sibling packed-i4
  route while `measurementEvidenceID` names another operation.
- Bad: artifact metadata has
  `tcrv_rvv.low_precision_resource.realization_admission_schedule_decision`,
  but the provider selection lacks matching admission fields.
- Bad: Common EmitC or a generated-bundle script accepts a packed-i4 artifact
  because the route id, artifact name, or q4/q8 label looks right.

### 6. Tests Required

- C++ RVV plugin tests must assert successful policy input/materialized
  pressure-profile proof carries admission contract, decision, evidence,
  dispatch policy, and schedule decision fields.
- C++ RVV plugin tests must reject missing admission record fields, stale
  admission schedule decisions, and sibling admission evidence.
- Target artifact tests must assert provider-built packed-i4 artifacts require
  admission proof and reject stale provider or metadata-only admission schedule
  mirrors before header/artifact acceptance.
- Focused lit/export coverage must show the admission mirrors appear in
  emission-plan metadata and generated headers only after route validation, and
  stale artifact admission schedule mirrors fail closed.
- Bounded diff scans must show no new Common EmitC semantic inference and no
  legacy i32 route-authority path is added.

### 7. Wrong vs Correct

Wrong:

```text
artifact metadata says realization_admission_schedule_decision = packed-i4
  -> target artifact exporter accepts the route
  -> measurement report claims schedule proof
```

Correct:

```text
realized packed-i4 body
  -> route/provider imports admitted schedule proof
  -> same-target record ties proof to measurement evidence
  -> target artifact validator compares provider facts and metadata mirrors
  -> Common EmitC/export only transports the validated route payload
```

## Packed-I4 Same-Target Measurement Classification Evidence

### 1. Scope / Trigger

Use this contract when a same-target measurement tool records packed-i4
product-reduction-dequant timing evidence for the generated RVV artifact against
`scalar-c-reference/product-reduction-dequant-packed-i4-v1`. The classification
is evidence interpretation only. It is not route authority, not a q4/q8 route
label, not a benchmark-name owner, and not permission for Common EmitC to infer
RVV semantics.

### 2. Signatures

Per-op measurement evidence must contain a structured result object equivalent
to:

```json
{
  "result_classification": {
    "classification": "win | no-win | regression | not-measured",
    "outcome_family": "win | no-win | not-measured",
    "best_speedup_min": 0.0,
    "best_speedup_max": 0.0,
    "best_speedup_range": "0.000000..0.000000",
    "summary_record_count": 12,
    "measurement_record_count": 60,
    "correctness_record_count": 12,
    "case_summaries": [
      {
        "n": "257",
        "pattern": "0",
        "scale": "-0.125",
        "baseline_best_per_iter_ns": 0.0,
        "generated_best_per_iter_ns": 0.0,
        "best_speedup": 0.0
      }
    ],
    "timing_method": "clock_gettime(CLOCK_MONOTONIC_RAW)",
    "correctness_before_timing": true
  }
}
```

Packed-i4 measurement evidence must also contain a provider feedback tie-back
object equivalent to:

```json
{
  "provider_feedback_tie_back": {
    "packed_i4_resource_metadata_selected": true,
    "fields": {
      "performance_feedback": "same-target-packed-i4-no-win.v1",
      "performance_baseline": "scalar-c-reference/product-reduction-dequant-packed-i4-v1",
      "performance_best_speedup_range": "0.896848..1.020953",
      "performance_action": "no-win-repair-required-before-performance-claim",
      "operand_form": "packed-i4-nibbles",
      "packing_layout": "two-signed-i4-elements-per-byte-low-high-nibbles",
      "unpack_intent": "sign-extend-i4-nibbles-before-widening-product"
    },
    "result_alignment": "consistent-with-current-no-win-feedback",
    "performance_win_claim_allowed": false,
    "next_repair_owner_if_no_win": "RVV plugin-local Gearbox/resource/statement planning for the selected packed-i4 product-reduction candidate"
  }
}
```

### 3. Contracts

- `best_speedup` is the scalar-baseline best per-iteration time divided by the
  generated-artifact best per-iteration time for one parsed `SUMMARY` case.
- Classify as `win` only when every parsed `SUMMARY best_speedup` is greater
  than `1.0`.
- Classify as `regression` when every parsed `SUMMARY best_speedup` is less
  than `1.0`; its `outcome_family` is still `no-win`.
- Classify as `no-win` for mixed or tie cases that are not all above or all
  below `1.0`.
- Dry-run evidence must classify as `not-measured`; it may validate bundle and
  harness structure but must not present timing as runtime/performance evidence.
- The packed-i4 scalar baseline/oracle may be selected only after validated
  generated object/header metadata proves provider-owned `packed-i4-nibbles`
  resource facts.
- Provider feedback tie-back fields must be copied from validated provider route
  or generated object/header metadata mirrors, not from fixture names, artifact
  paths, route ids, or benchmark labels.

### 4. Validation & Error Matrix

- Missing parsed `SUMMARY` records in a real run -> fail the measurement
  evidence.
- Missing or non-numeric `best_speedup`, `baseline_best_per_iter_ns`, or
  `generated_best_per_iter_ns` -> fail the measurement evidence.
- Packed-i4 measurement evidence lacks the named packed scalar baseline,
  correctness guard count, target profile, raw timing stdout, parsed summaries,
  or compile flags -> it is not packed-i4 performance evidence.
- Packed-i4 provider feedback tie-back omits or changes any expected
  performance/resource field -> fail closed before accepting the measurement
  summary.
- A report claims a packed-i4 performance win while classification is
  `no-win`, `regression`, or `not-measured`, or while
  `performance_win_claim_allowed = false` -> invalid evidence boundary.

### 5. Good/Base/Bad Cases

- Good: generated packed-i4 RVV bundle validates provider-owned resource facts
  -> harness selects the packed scalar baseline -> correctness guards pass ->
  same-target `SUMMARY` records classify as `regression` -> provider feedback
  tie-back remains `consistent-with-current-no-win-feedback`.
- Base: dry-run bundle/harness evidence records `classification =
  not-measured` and cannot support runtime or performance claims.
- Bad: a fixture name or artifact path selects the packed scalar baseline before
  object/header metadata validates `packed-i4-nibbles`.
- Bad: a raw stdout table is pasted into a PRD without machine-readable
  classification and provider feedback tie-back.

### 6. Tests Required

- Python self-test or equivalent unit coverage for `win`, `no-win`,
  `regression`, and `not-measured` classification behavior.
- Dry-run script/lit coverage asserting `not-measured`, packed-i4 baseline/oracle
  selection only after validated metadata, and no packed-i4 oracle leakage into
  default product-dequant/dequant-clamp paths.
- Real `ssh rvv` evidence for any runtime/performance claim, including raw
  `MEASURE` lines, parsed `SUMMARY` records, correctness guards, target profile,
  compile flags, and structured classification.
- Focused assertions that provider feedback tie-back preserves
  `performance_feedback`, `performance_baseline`, `performance_best_speedup_range`,
  `performance_action`, operand form, packing layout, and unpack intent.

### 7. Wrong vs Correct

Wrong:

```text
artifact path contains packed-i4
  -> harness chooses packed scalar baseline
  -> PRD says performance improved
```

Correct:

```text
validated provider-owned packed-i4 resource metadata
  -> harness chooses packed scalar baseline
  -> same-target raw timing and correctness guards
  -> structured win/no-win/regression classification
  -> provider feedback tie-back controls whether a win claim is allowed
```

## Packed-I4 No-Win Performance Feedback Facts

### 1. Scope / Trigger

Use this contract after same-target packed-i4 product-reduction-dequant timing
records an honest no-win/regression against the packed scalar baseline. The
feedback is a production compiler guard for the selected packed-i4 resource
surface. It is not a performance win claim, route id, benchmark label, q4/q8
authority, or artifact-name authority.

### 2. Signatures

The selected packed-i4 resource selection must carry these provider-owned
fields when `operand_form = "packed-i4-nibbles"`:

```c++
struct RVVLowPrecisionContractionResourceSelection {
  std::string performanceFeedback;
  std::string performanceBaseline;
  std::string performanceBestSpeedupRange;
  std::string performanceAction;
  std::string remediationPlanContract;
  std::string remediationPlan;
  std::string remediationStatementStrategy;
  std::string remediationVectorBudget;
};
```

The accepted packed-i4 values calibrated from the latest same-target timing
evidence and the first production repair boundary are:

```text
tcrv_rvv.low_precision_resource.performance_feedback =
  "same-target-packed-i4-no-win.v1"
tcrv_rvv.low_precision_resource.performance_baseline =
  "scalar-c-reference/product-reduction-dequant-packed-i4-v1"
tcrv_rvv.low_precision_resource.performance_best_speedup_range =
  "0.896848..1.020953"
tcrv_rvv.low_precision_resource.performance_action =
  "no-win-repair-required-before-performance-claim"
tcrv_rvv.low_precision_resource.remediation_plan_contract =
  "rvv-low-precision-packed-i4-resource-remediation-plan.v1"
tcrv_rvv.low_precision_resource.remediation_plan =
  "close-packed-i4-local-statement-repair-frontier-before-performance-claim.v1"
tcrv_rvv.low_precision_resource.remediation_statement_strategy =
  "low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum"
tcrv_rvv.low_precision_resource.remediation_vector_budget =
  "packed-i4-remediation-budget-5of32-vector-groups"
```

### 3. Contracts

- The Gearbox resource pass must attach the four performance feedback attrs to
  the selected packed-i4 pre-realized body; selected-body realization must copy
  them into the realized `with_vl` structure before provider planning.
- The selected packed-i4 candidate must also carry the remediation plan attrs
  above as provider-owned resource-planning facts. These facts state the
  concrete production repair boundary: packed signed i4-in-i8 low/high nibble
  unpack, low-product rescale, high-nibble vwmacc pair-sum, single `vwredsum`,
  and a 5-of-32 vector-group
  budget. They are not artifact metadata, benchmark labels, or route ids.
- The route-family provider must parse, verify, and retain the fields as part
  of `RVVLowPrecisionContractionResourceSelection` before route construction.
- The statement-plan owner must compare provider and family-plan copies before
  constructing the packed-i4 statement payload and must require the exact
  remediation statement strategy before emitting packed-i4 load/unpack/product/
  high-vwmacc/reduce statements.
- Route metadata, target support bundles, target artifact validation, and
  generated-bundle dry-run indexes may expose these values only as exact
  mirrors of provider-owned facts.
- The `performance_action` value means the compiler path requires a repair or
  an explicit fail-closed decision before any future packed-i4 performance win
  claim. It must not be rewritten into success/readiness/status language.
- `performance_best_speedup_range` is the provider resource feedback mirror
  attached to the selected candidate. The strict dispatch policy consumes the
  fresh same-target measurement record's `measurement_best_speedup_range`;
  these values are compared at different layers and must not be substituted for
  each other.

### 4. Validation & Error Matrix

- Packed-i4 selected candidate missing any of the feedback or remediation plan
  fields -> fail
  closed in Gearbox/provider validation before route construction.
- Realized body feedback differs from the selected resource facts -> fail
  closed in selected-body/provider validation.
- Statement provider and route-family copies disagree -> fail closed at the
  statement-plan owner boundary.
- Artifact/header metadata omits or changes any mirrored feedback/remediation
  plan field -> fail closed in target artifact validation before accepting
  generated-bundle evidence.
- Target support bundle export omits a remediation plan mirror -> the header is
  incomplete and the target artifact surface must be treated as stale.
- Same-target measurement or a report claims packed-i4 performance improvement
  while the action remains `no-win-repair-required-before-performance-claim` and
  no new same-target timing exists -> invalid evidence boundary.

### 5. Good/Base/Bad Cases

- Good: measured packed-i4 no-win evidence -> provider-owned feedback facts ->
  provider-owned remediation plan facts -> realized body copies them ->
  statement planning requires the exact packed-i4 remediation strategy -> target
  export mirrors them exactly -> generated artifact remains correctness/evidence-
  capable but cannot be reported as a performance win.
- Base: unpacked-byte product-dequant and sibling low-precision representatives
  keep their existing resource contracts and do not inherit the packed-i4
  feedback fields unless their selected candidate defines an explicit feedback
  contract.
- Bad: a fixture name, benchmark name, or artifact path implies no-win or
  performance repair status while provider-owned packed-i4 selection lacks the
  feedback or remediation plan fields.
- Bad: artifact metadata is edited to
  `same-target-packed-i4-performance-win.v1` without a provider-owned schedule
  change and new same-target timing; target validation must reject it.
- Bad: route metadata says `metadata-only-packed-i4-unpack-plan` while the
  provider-owned selection requires
  `low-shifted-i4-product-rescale-high-nibble-vwmacc-single-vwredsum`; target
  validation must reject it before header/artifact acceptance.

### 6. Tests Required

- Selected-body realization lit/FileCheck coverage showing the feedback and
  remediation plan fields on the realized packed-i4 body and on emitted
  route/header metadata.
- C++ provider tests for accepted packed-i4 feedback/remediation fields and
  stale feedback/remediation rejection.
- C++ target artifact validation tests proving stale candidate metadata mirrors
  fail before artifact acceptance.
- Target support bundle/header coverage proving remediation plan mirrors are
  exported and not silently dropped.
- Generated-bundle dry-run coverage proving the index/evidence metadata carries
  the four fields only after provider/header validation.
- Same-target timing is required only after a real production compiler change
  claims a new packed-i4 performance result.

### 7. Wrong vs Correct

Wrong:

```text
same-target stdout says no-win
  -> write a report
  -> artifact metadata later claims performance win
```

Correct:

```text
same-target no-win evidence
  -> provider-owned packed-i4 performance feedback and remediation plan facts
  -> selected-body realization/provider/statement/target validators compare them
  -> future performance claim requires a production repair plus new timing
```

## Packed-I4 Performance-Maturity Selection Mirrors

### 1. Scope / Trigger

Use this contract when a packed-i4 low-precision RVV route is executable for
correctness but same-target timing evidence still classifies it as no-win or
regression. This contract separates route-supported/executable correctness from
performance-ready selection and dispatch preference. It is not a new route id,
not a q4/q8 benchmark authority, and not a Common EmitC semantic decision.

### 2. Signatures

`RVVLowPrecisionContractionResourceSelection` must carry the feedback fields
above plus these provider-owned maturity fields:

```c++
struct RVVLowPrecisionContractionResourceSelection {
  std::string performanceMaturity;
  std::string performanceMaturityEvidence;
  std::string performanceMaturityOutcome;
  std::string performanceSelectionEligible;
  std::string dispatchPreference;
};
```

The accepted packed-i4 regression values are:

```text
tcrv_rvv.low_precision_resource.performance_maturity =
  "executable-not-performance-mature"
tcrv_rvv.low_precision_resource.performance_maturity_evidence =
  "same-target-packed-i4-local-repair-frontier-no-win-gate4.v1"
tcrv_rvv.low_precision_resource.performance_maturity_outcome =
  "regression"
tcrv_rvv.low_precision_resource.performance_selection_eligible =
  "false"
tcrv_rvv.low_precision_resource.dispatch_preference =
  "not-performance-preferred"
```

### 3. Contracts

- Gearbox/resource selection owns the five fields and attaches them to the
  selected packed-i4 body with the existing low-precision resource facts.
- Selected-body realization, route-family planning, statement planning, route
  metadata, target support bundles, artifact validation, and generated-bundle
  scripts may carry these fields only as exact mirrors of provider-owned facts.
- `performance_maturity = "executable-not-performance-mature"` preserves
  executable correctness support; it does not disable legality, route support,
  artifact generation, or correctness evidence.
- `performance_selection_eligible = "false"` and
  `dispatch_preference = "not-performance-preferred"` block any
  performance-ready selected-route, manifest, target artifact, dispatch, or
  report claim until a later provider-owned schedule/resource repair plus new
  same-target timing changes the contract.
- Same-target measurement output is evidence input for the provider-owned
  contract. Measurement scripts must not become route authority or silently
  rewrite provider maturity fields.

### 4. Validation & Error Matrix

- Packed-i4 selected resource missing any maturity field -> fail closed before
  statement planning or target artifact acceptance.
- Provider facts say `performance_selection_eligible = "false"` but route,
  candidate metadata, manifest, or header mirror says `"true"` -> fail closed at
  provider/target validation.
- Provider facts say `dispatch_preference = "not-performance-preferred"` but an
  artifact mirror claims dispatch-preferred/performance-selected status -> fail
  closed before artifact acceptance.
- Same-target classification is `regression` or no-win and no newer timing
  exists, but a report or script allows a performance win claim -> invalid
  evidence boundary.
- Common EmitC derives or modifies any of the five fields -> boundary violation;
  Common EmitC may only carry a provider-built `TCRVEmitCLowerableRoute`.

### 5. Good/Base/Bad Cases

- Good: packed-i4 route is legal and generated artifact passes correctness
  checks, while provider/target mirrors state
  `executable-not-performance-mature`, `regression`, `false`, and
  `not-performance-preferred`.
- Base: sibling unpacked-byte low-precision routes keep their own resource
  contracts and do not inherit packed-i4 maturity fields unless the RVV provider
  defines them.
- Bad: target artifact metadata is edited to
  `performance_selection_eligible = "true"` while provider-owned packed-i4
  facts still record Gate 6 regression.
- Bad: a script or fixture path claims dispatch preference because a filename
  contains packed-i4, without matching provider/header metadata.

### 6. Tests Required

- C++ provider tests asserting the five maturity fields are populated, carried
  through route-family planning, and stale performance-selection eligibility
  fails closed.
- C++ target artifact tests asserting stale candidate/header maturity mirrors
  fail closed while executable route support remains accepted.
- MLIR/FileCheck coverage showing selected-body, statement-plan, route/header,
  and artifact mirrors contain the exact maturity values.
- Generated-bundle and same-target measurement self-tests asserting a
  performance win claim is not allowed unless provider maturity and selection
  eligibility explicitly permit it.

### 7. Wrong vs Correct

Wrong:

```text
packed-i4 executable artifact exists
  -> dispatch treats it as performance preferred
  -> report claims packed low-precision performance maturity
```

Correct:

```text
packed-i4 executable artifact exists
  -> provider-owned maturity mirrors record regression and selection=false
  -> dispatch/performance-ready claims fail closed until new timing repairs it
```

## Packed-I4 Same-Target Maturity Evidence Input

### 1. Scope / Trigger

Use this contract when `rvv_generated_bundle_same_target_measure.py` reports
packed-i4 same-target measurement output after validating generated
object/header metadata against provider-owned low-precision resource facts. The
payload is measurement evidence input to the provider-owned maturity contract.
It is not route support, RVV compute semantics, dispatch authority, or Common
EmitC logic.

The per-op payload may also carry `same_target_measurement_record`, which is the
script-output object consumed by the C++
`RVVLowPrecisionSameTargetMeasurementRecord` boundary. This record is a carrier
for parsed measurement and provider tie-back fields only; it is not a policy
decision and does not replace provider-owned resource facts.

### 2. Signatures

Packed-i4 per-op evidence and root summaries may carry:

```json
{
  "performance_maturity_contract_evidence_input": {
    "contract": "packed-i4-same-target-performance-maturity-evidence-input.v1",
    "measurement_evidence_id": "run/op/same_target_measurement_evidence.json",
    "measurement_classification": "win | no-win | regression | not-measured",
    "measurement_outcome_family": "win | no-win | not-measured",
    "measurement_best_speedup_range": "0.895307..1.027027",
    "measurement_summary_record_count": 12,
    "measurement_record_count": 60,
    "provider_maturity": "executable-not-performance-mature",
    "provider_maturity_evidence": "same-target-packed-i4-campaign-no-further-repair-no-win-gate4.v1",
    "provider_maturity_outcome": "no-win",
    "provider_performance_selection_eligible": "false",
    "provider_dispatch_preference": "not-performance-preferred",
    "provider_performance_action": "no-win-repair-required-before-performance-claim",
    "provider_schedule_decision_contract": "rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1",
    "provider_schedule_decision": "select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1",
    "provider_schedule_decision_reason": "accepted-beyond-local-scalar-epilogue-high-nibble-vwmacc-single-vwredsum-budget-5of32",
    "source_record_contract": "rvv-low-precision-source-backed-artifact-measurement-record.v1",
    "source_selected_variant": "rvv_lp_pack_i4_widening_product_reduce_dequantize_f32",
    "source_selected_input": "test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir",
    "source_generated_function": "tcrv_rvv_lp_i4_widening_product_reduce_dequantize_f32",
    "generated_artifact_identity_contract": "generated-object-header-sha256-after-target-artifact-validation.v1",
    "generated_artifact_object_path": "generated_bundle/object.o",
    "generated_artifact_object_sha256": "<sha256>",
    "generated_artifact_header_path": "generated_bundle/header.h",
    "generated_artifact_header_sha256": "<sha256>",
    "measurement_target": "ssh rvv",
    "measurement_target_provenance": "same-target-measurement-workflow-ssh-target.v1",
    "measurement_runtime_count_set": "257,4096,65536",
    "measurement_runtime_count_provenance": "same-target-measurement-config-input-sizes.v1",
    "pressure_profile_label": "low-precision-quantized-contraction-production-pressure",
    "pressure_profile_label_provenance": "non-authoritative-pressure-label-derived-from-selected-typed-rvv-provider-facts-and-source-backed-measurement-record",
    "contract_alignment": "matches-provider-maturity-outcome",
    "performance_win_claim_allowed": false,
    "performance_preference_denied": true,
    "performance_preference_denial_reason": "same-target-measurement-no-win-or-regression",
    "correctness_execution_allowed": true,
    "route_support_effect": "preserve-executable-route-support; measurement evidence only gates performance preference and claims",
    "provider_contract_update_required": false
  }
}
```

Root evidence may also aggregate per-op objects under
`performance_maturity_contract_inputs` and per-op C++ record objects under
`same_target_measurement_records`.

The C++ consumption surface is:

```c++
llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord>
buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceInput(
    const llvm::json::Object &evidenceInput, llvm::StringRef context);

llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput>
buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceInput, llvm::StringRef context);
```

`same_target_measurement_record` must contain the fields of
`RVVLowPrecisionSameTargetMeasurementRecord` using the script's snake_case field
names, including `measurement_evidence_id`,
`measurement_classification`, `measurement_best_speedup_range`,
`measurement_summary_record_count`, `measurement_record_count`,
`correctness_record_count`, `same_target_measurement`, `ssh_evidence`,
`target_profile`, `provider_resource_planning_contract`,
`provider_resource_operand_form`, `provider_resource_source_signedness`,
`provider_resource_storage_element_width`,
`provider_resource_effective_element_width`,
`provider_resource_packing_layout`, `provider_resource_unpack_intent`,
`provider_resource_vsetvl_region_count`, `provider_runtime_avl_source`,
`provider_runtime_abi_order`, `provider_primitive_chain_kind`,
`provider_schedule_decision`, `source_record_contract`,
`source_selected_variant`, `source_selected_input`,
`source_generated_function`, `generated_artifact_identity_contract`,
`generated_artifact_object_path`, `generated_artifact_object_sha256`,
`generated_artifact_header_path`, `generated_artifact_header_sha256`,
`measurement_target`, `measurement_target_provenance`,
`measurement_runtime_count_set`, `measurement_runtime_count_provenance`,
`pressure_profile_label`, `pressure_profile_label_provenance`, and the other
provider/resource/remediation/target tie-backs. It must not include
reporting-only fields such as `contract_alignment` or remediation-plan detail
fields that are not part of `RVVLowPrecisionSameTargetMeasurementRecord`.

### 3. Contracts

- `measurement_*` fields come from the parsed same-target result
  classification and the current evidence path.
- `provider_*` fields must be copied from provider-owned route/resource facts or
  generated object/header metadata mirrors after target artifact validation.
- Gate 3 measurement records must tie the parsed measurement back to the Gate
  1/2 resource boundary by carrying the exact planning contract,
  packed-resource operand form, signedness, storage/effective element widths,
  packing layout, unpack intent, vsetvl region count, runtime AVL source, and
  runtime ABI order from validated provider/resource facts. These fields are
  evidence tie-backs only; they do not make artifact metadata, fixture names, or
  raw stdout route authority.
- `same_target_measurement_record` is an exact record-shaped subset of the
  validated evidence input. It feeds the C++ record boundary before policy input
  is built; it must not carry reporting-only alignment fields as policy facts.
- Source-backed record fields are mandatory for Gate 2 production pressure
  profile construction:
  `source_record_contract =
  rvv-low-precision-source-backed-artifact-measurement-record.v1`,
  selected variant/input/generated function copied from the selected RVV
  generated-bundle workflow, generated object/header path and SHA256 copied
  after target artifact validation, `measurement_target` copied from the actual
  measurement workflow target, runtime count set copied from the measurement
  configuration, and pressure label provenance explicitly marking the label as
  non-authoritative.
- `pressure_profile_label` may classify the production pressure family, but the
  label and q4/q8-style naming must never satisfy selected-boundary,
  primitive/resource, artifact identity, target, or runtime-count provenance.
- C++ record consumption must fail closed for missing or type-mismatched record
  fields before policy input is materialized.
- Gate 4 policy input must carry the Gate 2b provider-owned schedule-decision
  fields: `provider_schedule_decision_contract`,
  `provider_schedule_decision`, and `provider_schedule_decision_reason`.
- The script may report alignment or conflicts, but must not rewrite
  `RVVLowPrecisionContractionResourceSelection` maturity fields.
- `performance_win_claim_allowed` is true only when measurement classification
  is `win`, provider selection eligibility is `"true"`, dispatch preference is
  `"performance-preferred"`, and the provider performance action no longer
  records the no-win repair guard.
- `regression`, `no-win`, and `not-measured` deny performance preference and win
  claims while preserving executable correctness support.
- A measurement `win` that conflicts with the provider no-win/regression
  contract sets `provider_contract_update_required = true`; it is not itself a
  maturity-field update.

### 4. Validation & Error Matrix

- Packed-i4 measurement evidence lacks the evidence-input object -> reporting
  bridge is incomplete.
- Packed-i4 measurement evidence lacks `same_target_measurement_record`, or the
  record lacks `measurement_evidence_id` or another required field -> C++ record
  parsing fails before policy input.
- Provider maturity, eligibility, dispatch, or action fields are missing from
  validated metadata -> fail the provider feedback tie-back before accepting
  the measurement summary.
- Record target profile differs from `ssh rvv`, `ssh_evidence` is false for a
  measured record, planning contract differs from the selected resource plan,
  packed resource form/signedness/width/layout/unpack facts differ from provider
  facts, runtime AVL source or runtime ABI order differs from provider facts,
  primitive chain tie-back differs, or schedule-decision tie-back differs -> C++
  policy input construction fails before dispatch/performance policy
  consumption.
- Source record contract is missing or stale, selected variant/generated
  function does not match the packed-i4 selected boundary, selected input is
  missing, object/header path or SHA256 is missing or marker-only, measurement
  target is not `ssh rvv`, measurement target/runtime-count provenance is stale,
  or pressure label provenance is label-only -> C++ record and production
  pressure profile construction fail before policy consumption.
- Measurement classification is `regression`, `no-win`, or `not-measured` while
  `performance_win_claim_allowed = true` -> invalid evidence boundary.
- Provider selection eligibility is `"false"` but reporting claims dispatch or
  performance preference -> fail closed in reporting/tests.
- Reporting disables route support or correctness execution because performance
  preference is denied -> boundary violation.
- Script computes provider maturity fields from fixture names, route ids,
  artifact paths, q4/q8 labels, or raw stdout alone -> invalid authority path.

### 5. Good/Base/Bad Cases

- Good: validated packed-i4 generated bundle -> parsed same-target
  `regression` -> evidence-input records provider outcome `regression`,
  selection `false`, dispatch `not-performance-preferred`,
  claim allowed `false`, and correctness execution allowed `true`.
- Good: the same payload emits `same_target_measurement_record`; C++ parses it
  into `RVVLowPrecisionSameTargetMeasurementRecord`, then builds
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` through provider
  tie-back validation.
- Base: dry-run measurement records `not-measured`, denies performance claims,
  sets `same_target_measurement = false` and `ssh_evidence = false`, and keeps
  provider maturity mirrors visible as non-authoritative reporting input.
- Bad: a same-target dry-run object or raw stdout table edits provider maturity
  fields or marks performance selection eligible.
- Bad: a script record changes `provider_runtime_abi_order`,
  `provider_primitive_chain_kind`, or `target_profile` while keeping the
  evidence id intact; C++ policy input construction must reject it as stale.
- Bad: a script record carries `q8` or `q4` as the pressure truth, or uses
  `metadata-only` object/header identity, while omitting selected-boundary and
  generated artifact source fields; C++ source-backed record verification must
  reject it before production pressure profile materialization.
- Bad: a measurement win directly enables dispatch preference without a
  provider/resource contract update and new target artifact validation.

### 6. Tests Required

- Script self-test must cover `regression`, mixed `no-win`, and hypothetical
  `win` classifications against the current provider contract.
- Script self-test must assert `same_target_measurement_record` is exactly the
  C++ record-shaped field subset and does not leak reporting-only fields.
- C++ provider/policy tests must assert evidence-object parsing into
  `RVVLowPrecisionSameTargetMeasurementRecord`, policy-input construction from
  that object, and fail-closed missing measurement id, missing/stale planning
  contract, stale packed resource form, stale target, stale runtime AVL/ABI, and
  stale primitive-chain cases.
- C++ provider/policy tests must also assert the Gate 2 source-backed fields
  reach `RVVLowPrecisionProductionPressureProfile`, and must reject missing
  source record contract, stale selected boundary, marker-only generated
  artifact identity, wrong measurement target, stale runtime-count provenance,
  and q4/q8 label-only pressure.
- Dry-run FileCheck coverage must assert the evidence-input object, provider
  maturity mirrors, `same_target_measurement_record`,
  selected-boundary/artifact/measurement source-backed fields,
  planning/resource/runtime tie-backs, claim allowance, denial reason,
  route-support effect, and correctness execution allowance.
- Same-target real-run evidence is required only when claiming runtime,
  correctness, or performance results beyond the dry-run/reporting bridge.

### 7. Wrong vs Correct

Wrong:

```text
same-target stdout says win
  -> script flips performance_selection_eligible to true
  -> dispatch reports packed-i4 performance preferred
```

Correct:

```text
same-target parsed classification
  -> evidence-input records provider maturity mirror and measured outcome
  -> source-backed record ties selected boundary, generated object/header
     identity, measurement target, runtime-count set, and non-authoritative
     pressure label provenance to provider facts
  -> no-win/regression/not-measured deny performance preference
  -> win still requires provider contract update plus new validation before
     performance dispatch can be claimed
```

## Packed-I4 Dispatch/Performance Policy Consumption

### 1. Scope / Trigger

Use this contract when RVV low-precision packed-i4 same-target measurement
evidence is consumed by production dispatch/performance policy. This is the
Gate 4 policy surface after provider/resource facts, target artifact mirrors,
and Gate 3 evidence input have already been structured. It is not a route id,
artifact-name policy, report status, script rewrite, q4/q8 benchmark label, or
Common EmitC decision.

### 2. Signatures

The production API surface is:

```c++
struct RVVLowPrecisionPerformancePolicyDecision {
  std::string policyContract;
  RVVLowPrecisionPerformancePolicyHandoff handoff;
  bool routeSupportAllowed;
  bool correctnessExecutionAllowed;
  bool performanceSelectionAllowed;
  bool performanceWinClaimAllowed;
  bool performancePreferredPathSelected;
  bool correctnessFallbackPathSelected;
  std::string dispatchPolicyPath; // correctness-fallback | performance-preferred
  std::string dispatchPreference;
  std::string performancePreferenceDenialReason;
  std::string fallbackReason;
};

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord>
buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceRoot(
    const llvm::json::Object &evidenceRoot, llvm::StringRef context);

llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput>
buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceRoot(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceRoot, llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceRoot,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error populateRVVLowPrecisionSelectedDispatchPolicyOutput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);
```

Accepted dispatch policy paths are:

```text
correctness-fallback
performance-preferred
```

### 3. Contracts

- `evaluateRVVLowPrecisionPerformancePolicy` is the strict accepted-policy
  entry. It succeeds only when the selected packed-i4 provider resource facts
  and the measurement outcome form a complete accepted policy handoff.
- The Gate 3 selected-dispatch policy seam must prefer the
  `RVVLowPrecisionSameTargetMeasurementRecord` overload when source-backed
  measurement evidence is available. The record overload first builds and
  validates the same-target policy input from the record, then produces the
  explicit `RVVLowPrecisionPerformancePolicyDecision`.
- `resolveRVVLowPrecisionDispatchPerformancePolicy` is the safe dispatch
  resolver. For stale or missing performance evidence, it must deny
  performance preference, preserve correctness fallback when the selected route
  remains legal, and carry the fail-closed reason in `fallbackReason`.
- The selected-dispatch boundary overloads must validate explicit
  `tcrv.exec.dispatch` case/fallback facts after the record/resource/measurement
  handoff is accepted. If the record is stale, resolver variants preserve the
  original stale measurement diagnostic instead of replacing it with a generic
  dispatch-boundary error.
- Selected-dispatch case/fallback mirrors are provider-produced mirrors of the
  structured boundary facts, not independent dispatch authority. The policy
  boundary and target artifact consumer must reject a mirror unless it ties back
  to the same selected case variant, case role, runtime guard flag/value, case
  origin, case policy, fallback variant, fallback path role, fallback role,
  fallback origin, and fallback policy already collected from the real
  `tcrv.exec.dispatch` envelope.
- Selected-dispatch policy output is also provider-owned boundary state, not
  artifact/header authority. For a packed-i4 selected-dispatch pressure path,
  the provider must populate `hasSelectedDispatchPolicyOutput` and mirror the
  accepted `RVVLowPrecisionPerformancePolicyDecision` into the selected
  dispatch boundary before artifact export:

  ```text
  selected_dispatch_policy_contract
  dispatch_policy_path
  selected_dispatch_preference
  performance_preference_denial_reason
  fallback_reason
  route_support_allowed
  correctness_execution_allowed
  performance_selection_allowed
  performance_win_claim_allowed
  correctness_fallback_path_selected
  performance_preferred_path_selected
  ```

  Target artifact metadata and generated header comments may expose these only
  through `tcrv_rvv.low_precision_resource.*` mirrors of that provider-owned
  policy-output boundary. `dispatch_preference` remains the existing
  low-precision resource mirror; it must agree with the selected-dispatch
  policy decision, but it is not a standalone selected-dispatch policy-output
  key.
- When source-backed same-target measurement evidence is available, the
  selected-dispatch policy-output population path must use the
  `RVVLowPrecisionSameTargetMeasurementRecord` overload. The overload first
  validates the record against provider resource/admission/maturity, target
  capability, runtime ABI, source/generated artifact identity, and
  selected-dispatch case/fallback facts, then writes the policy-output mirrors
  from the accepted `RVVLowPrecisionPerformancePolicyDecision`. The older
  selection-only helper may construct the current accepted no-win record for
  the default packed-i4 path, but it is not the measured-record admission seam.
- The current accepted Gate 4 packed-i4 regression/no-win outcome must set:

  ```text
  measurementEvidenceID =
    gate4-packed-i4-scalar-epilogue-dequant-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json
  measurementClassification = no-win
  measurementOutcomeFamily = no-win
  measurementBestSpeedupRange = 0.895307..1.027027
  measurementSummaryRecordCount = 12
  measurementRecordCount = 60
  correctnessRecordCount = 12
  routeSupportAllowed = true
  correctnessExecutionAllowed = true
  performanceSelectionAllowed = false
  performanceWinClaimAllowed = false
  performancePreferredPathSelected = false
  correctnessFallbackPathSelected = true
  dispatchPolicyPath = correctness-fallback
  dispatchPreference = not-performance-preferred
  performancePreferenceDenialReason =
    same-target-measurement-no-win-or-regression
  providerScheduleDecisionContract =
    rvv-low-precision-packed-i4-resource-aware-schedule-decision.v1
  providerScheduleDecision =
    select-packed-i4-high-nibble-vwmacc-scalar-epilogue-single-reduce-u1-two-region-budget-5of32.v1
  providerScheduleDecisionReason =
    accepted-beyond-local-scalar-epilogue-high-nibble-vwmacc-single-vwredsum-budget-5of32
  providerPerformanceAdmissionClosure =
    no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1
  providerPerformanceAdmissionReopenRequirement =
    new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1
  ```

- The strict Gate 4 no-win/regression verifier must derive the accepted
  `measurementEvidenceID`, `measurementBestSpeedupRange`,
  `measurementSummaryRecordCount`, `measurementRecordCount`, and
  `correctnessRecordCount` from the selected provider resource candidate. A
  dequant packed-i4 record and a dequant-clamp packed-i4 record are sibling
  measurement records, not interchangeable evidence. The current accepted
  dequant-clamp record uses:

  ```text
  measurementEvidenceID =
    gate4-packed-i4-scalar-epilogue-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json
  measurementClassification = no-win
  measurementOutcomeFamily = no-win
  measurementBestSpeedupRange = 0.874735..1.061579
  measurementSummaryRecordCount = 24
  measurementRecordCount = 120
  correctnessRecordCount = 24
  dispatchPolicyPath = correctness-fallback
  performancePreferenceDenialReason =
    same-target-measurement-no-win-or-regression
  providerPerformanceAdmissionClosure =
    no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1
  providerPerformanceAdmissionReopenRequirement =
    new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1
  ```

- A measured win may select `performance-preferred` only when all of these
  structured facts agree: same-target `ssh rvv` measurement, `classification =
  win`, provider maturity outcome `win`, provider performance selection
  eligibility `true`, provider dispatch preference `performance-preferred`,
  provider performance action no longer carrying the no-win repair guard,
  provider resource-cost admission decision
  `admit-performance-preferred-with-resource-cost-measured-win`, remediation
  diagnosis `performance-preferred-measured-win`, target profile, measurement
  counts, provider schedule-decision tie-back fields, provider
  resource/primitive/remediation tie-back fields, provider performance
  admission closure `performance-preferred-measured-win-admission-open.v1`,
  provider reopen requirement `none`, and win-claim allowance.
- Current no-win/regression evidence may be accepted only when the provider
  resource selection, route metadata, target artifact validation, same-target
  evidence input, and policy record all carry the same
  `no-further-repair-packed-i4-campaign-loop-11-budget-5of32.v1`
  closure and the
  `new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1`
  reopen requirement. These fields are provider-owned admission facts, not
  artifact metadata, script status, route ids, or report text.
- After the scalar-epilogue beyond-local repair has also measured no-win,
  current no-win/regression evidence must carry the provider-owned
  campaign-level no-further-repair admission tuple:
  `rvv-low-precision-packed-i4-campaign-no-further-repair-admission.v1`,
  `deny-performance-preferred-campaign-no-further-provider-repair`,
  `packed-i4-campaign-no-further-provider-repair-after-scalar-epilogue-no-win`,
  and
  `new-typed-provider-campaign-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1`.
  Selected-body realization, route metadata, statement planning, target
  artifact mirrors, same-target evidence roots, generated-bundle pressure
  profiles, and `RVVLowPrecisionPerformancePolicy` must consume the same tuple.
  It records a production-consumed no-further-repair admission blocker, not a
  documentation/report-only status and not a performance-preferred route.
- The current closure is the packed-i4 campaign no-further-repair boundary
  after the low-shifted-product rescale, high-nibble vwmacc, beyond-local
  admission blocker, and scalar-epilogue repair have all been consumed and
  measured as no-win on the same target. Another performance-preferred claim
  must update typed/provider schedule/resource facts, target mirrors,
  source-backed measurement evidence, resource-cost admission, campaign
  closure/reopen facts, and policy facts as one contract.
- Measurement scripts may report evidence inputs and alignment; they must not
  edit provider maturity fields or directly authorize dispatch.
- Common EmitC may carry provider-built route payloads only; it must not choose
  `dispatchPolicyPath` or infer packed-i4 performance preference.
- When the policy surface consumes a complete same-target evidence JSON object,
  the evidence-root overload must first validate the root object, not only the
  nested `same_target_measurement_record`. The root must carry `status =
  success`, `ssh_evidence = true`, `ssh_target = rvv`, `dry_run = false`,
  `timing_method = clock_gettime(CLOCK_MONOTONIC_RAW)`, the expected packed-i4
  `op_kind`, the candidate-specific packed scalar baseline identity, validated
  packed-i4 resource metadata selection, `result_classification` values that
  exactly match the nested record, and measurement harness/schedule evidence
  fields that exactly match the provider-owned schedule and realization
  admission schedule mirrors.
- The evidence-root overload is a policy ingestion boundary only. It verifies
  that source-backed evidence, parsed measurement record, packed oracle
  selection, schedule-decision mirrors, and provider feedback tie-backs agree
  before dispatch-policy evaluation. It must not let the JSON root select a
  route, edit provider maturity fields, or promote `performance-preferred`
  without the provider-owned measured-win contract.

### 4. Validation & Error Matrix

- Missing selected packed-i4 resource facts -> strict policy evaluation fails
  before performance dispatch.
- Stale measurement identity, stale measured best-speedup range, missing
  `ssh rvv` evidence, stale target profile, stale provider tie-back, or stale
  primitive/remediation/schedule-decision fact -> strict policy evaluation
  fails; resolver returns correctness fallback if the route remains legal.
- A complete evidence root has a stale root-level `result_classification`,
  `measurement_harness`, packed oracle, schedule-decision evidence, maturity
  evidence input, or provider feedback tie-back that disagrees with the nested
  record/provider schedule facts -> strict policy evaluation fails before
  selected dispatch can consume the record.
- A measurement record belongs to the dequant packed-i4 sibling while the
  selected provider resource candidate is dequant-clamp packed-i4, or the
  reverse -> strict policy evaluation fails as stale sibling-route measurement;
  resolver may keep correctness fallback only if the selected route remains
  legal.
- Current regression/no-win evidence with provider
  `performance_selection_eligible = true` or `dispatch_preference =
  performance-preferred` -> fail closed before performance preference.
- Current or future measured-win evidence with provider
  `performance_admission_decision =
  deny-performance-preferred-with-resource-cost-no-win-blocker` -> fail closed
  before performance preference; the resource-cost admission decision must be
  updated by the RVV provider, not by measurement JSON or artifact metadata.
- Current no-win/regression evidence missing the provider local-repair-frontier
  closure, carrying a sibling-route closure, or omitting the provider reopen
  requirement -> fail closed before target validation or policy evaluation can
  accept the measurement record.
- Current no-win/regression evidence missing the beyond-local repair admission
  tuple, or carrying a metadata/report-derived beyond-local blocker that does
  not match the provider-selected packed-i4 resource facts -> fail closed
  before target validation or policy evaluation can accept the measurement
  record.
- Future measured-win evidence with stale local-repair-frontier closure or stale
  reopen requirement -> fail closed before performance preference; the provider
  must first update schedule/resource facts beyond the local statement
  frontier, admission decision, closure, reopen requirement, beyond-local
  admission decision/blocker tuple, target mirrors, and same-target
  source-backed evidence as one contract.
- Measurement classification `win` without a provider maturity/selection update
  -> fail closed as measurement-only win promotion.
- Provider says `performance-preferred` but target artifact mirrors still carry
  stale no-win or selection-ineligible facts -> target validation fails before
  artifact acceptance.
- Route selection is illegal or rejected -> correctness fallback cannot claim
  route support; policy must keep performance preference denied.
- A selected-dispatch case or fallback mirror names a sibling variant, stale
  policy, stale runtime guard, stale origin, or stale fallback role while the
  structured dispatch boundary carries different facts -> policy evaluation and
  target artifact validation fail before the mirror can be accepted.
- A selected-dispatch policy-output mirror is present in target artifact
  metadata while `hasSelectedDispatchPolicyOutput` is false -> target artifact
  validation rejects it as a metadata-only selected-dispatch policy-output
  mirror before export.
- A selected-dispatch policy-output mirror disagrees with the accepted policy
  decision, such as `dispatch_policy_path = performance-preferred` or
  `performance_win_claim_allowed = true` while provider facts select
  correctness fallback/no-win -> provider or target validation fails before the
  artifact/header can claim performance-preferred dispatch.

### 5. Good/Base/Bad Cases

- Good: accepted Gate 4 regression/no-win measurement -> provider no-win
  maturity mirrors match -> policy selects `correctness-fallback` while keeping
  correctness execution allowed.
- Good: future provider/resource repair plus new same-target measured win ->
  provider maturity and remediation facts are updated and target mirrors match
  -> policy selects `performance-preferred`.
- Base: stale or not-yet-run measurement -> resolver preserves safe correctness
  fallback for a legal route, while strict verification refuses to authorize
  performance preference.
- Bad: same-target stdout says win -> script flips selection eligibility ->
  dispatch becomes performance-preferred without provider contract update.
- Bad: artifact metadata or route id says packed-i4 performance-selected ->
  policy accepts performance preference without provider-owned maturity facts.

### 6. Tests Required

- C++ provider/policy tests must assert current accepted regression/no-win
  selects `correctness-fallback` and denies performance preference.
- C++ provider/policy tests must assert the same accepted regression/no-win
  decision is produced through the direct
  `RVVLowPrecisionSameTargetMeasurementRecord` + selected-dispatch boundary
  overload, not only through a prebuilt policy input.
- C++ provider/policy tests must assert explicit no-win and measured-win
  records can populate selected-dispatch policy-output facts through
  `populateRVVLowPrecisionSelectedDispatchPolicyOutput(selection, record,
  boundary, context)`, and stale target/runtime/provider facts fail before any
  target/header mirror is accepted.
- Gate 4 final-audit coverage must also feed a record parsed from the generated
  evidence JSON object through that same selected-dispatch record overload; a
  helper-built representative record alone is not enough to prove the
  source-backed record ingestion boundary.
- Gate 4 root-ingestion coverage must feed both current source-backed dequant
  and dequant-clamp same-target evidence JSON roots through the evidence-root
  overload and assert correctness fallback, performance-preference denial, and
  stale root-level result/schedule rejection.
- Focused provider/target/script coverage must assert stale, missing, or
  metadata-derived beyond-local repair admission fields are rejected at the
  selected-body, route/target mirror, evidence-root, and policy boundaries.
- Candidate-sensitive Gate 4 coverage must feed the current dequant-clamp
  source-backed `same_target_measurement_record` through the record overload
  and assert correctness fallback, performance-preference denial, stale
  schedule-decision rejection, and correctness-disabled rejection.
- C++ provider/policy tests must assert a measured-win fixture selects
  `performance-preferred` only after provider maturity, eligibility, dispatch,
  remediation, and measurement tie-back facts all agree.
- C++ provider/policy tests must assert stale measurement identity, stale
  measured best-speedup range, missing `ssh rvv` evidence, stale target profile,
  stale provider tie-back, stale primitive facts, stale schedule-decision facts,
  correctness-disabled records, and measurement-only win promotion fail strict
  verification.
- Target artifact tests must assert stale performance-selection and dispatch
  mirrors fail closed before artifact acceptance.
- Target artifact tests must assert stale provider selected-dispatch mirrors
  fail closed even when candidate metadata mirrors the stale provider value
  exactly.
- Target artifact tests must assert selected-dispatch policy-output mirrors are
  exported only after provider-owned policy-output facts exist, and that stale
  `dispatch_policy_path`, `performance_win_claim_allowed`, route-support,
  correctness-execution, performance-selection, and path-selection mirrors fail
  closed.
- Script self-tests and dry-run lit coverage must keep evidence-input reporting
  mirror-only and must not allow no-win/regression/not-measured evidence to
  authorize performance dispatch.

### 7. Wrong vs Correct

Wrong:

```text
same-target measurement says win
  -> script or artifact metadata sets dispatch_preference=performance-preferred
  -> dispatch policy reports performance-preferred
```

Correct:

```text
same-target measurement says win
  -> provider-owned resource/maturity/remediation contract is updated
  -> target artifact mirrors validate the updated provider facts
  -> policy consumes matching measurement and provider facts
  -> dispatch policy selects performance-preferred
```

Wrong:

```text
artifact metadata sets performance_win_claim_allowed=true
  -> header exporter reports a packed-i4 performance win
```

Correct:

```text
selected tcrv.exec.dispatch case/fallback facts
  -> RVV provider evaluates the packed-i4 policy decision
  -> selected-dispatch boundary carries provider-owned policy-output fields
  -> artifact/header mirrors compare those fields exactly or fail closed
```

## Gearbox Product-Reduce-Dequant/Clamp Cross-Region Handoff

### 1. Scope / Trigger

Use this contract when `widening_product_reduce_dequantize_f32` or
`widening_product_reduce_dequant_clamp_f32` claims that Gearbox `vsetvl`
placement has progressed beyond marker-only evidence. The handoff is bounded
to one selected producer/consumer `with_vl` pair in the selected RVV body: the
product/reduction phase is a producer `with_vl`, and the dequant/store phase is
a nested consumer `with_vl` after the handoff. Product and reduction values,
dequant input, optional clamp inputs/results, runtime AVL, active VL, phase
ordering, region count, and resource decision must be represented as
`tcrv_rvv` structure before route support.

### 2. Signatures

The bounded structural op is:

```mlir
%handoff = tcrv_rvv.gearbox_cross_region_handoff
  %reduced, %vl, %n
  {
    contract = "gearbox-product-reduce-to-dequant-cross-region-handoff.v1",
    from_phase = "load-product-reduce",
    to_phase = "dequant-store",
    region_count = 2 : i64,
    runtime_avl_source = "runtime_abi:n",
    planning_contract =
      "rvv-low-precision-production-resource-planning-contract.v1",
    resource_decision =
      "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"
  }
  : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl, index
    -> !tcrv_rvv.vector<i32, "m1">
```

The paired region markers are part of the same structural contract:

```mlir
tcrv_rvv.vsetvl_region_marker %vl {
  phase = "load-product-reduce" | "tail-product-reduce" | "dequant-store",
  planning_contract =
    "rvv-low-precision-production-resource-planning-contract.v1",
  region_index = <one-based region index>,
  region_count = <selected resource region count>,
  resource_decision = <selected low-precision resource realization decision>
} : !tcrv_rvv.vl
```

For a selected `product-reduction-dequant-clamp-f32` candidate, the realized
producer/consumer scopes, the handoff, the route-plan metadata, and the target
artifact/header mirrors must additionally expose the provider-derived clamp
realization facts:

```mlir
clamp_region_index = <dequant region index> : i64
clamp_phase = "dequant-clamp-store"
clamp_compare_select_phase = "lower-then-upper-compare-select"
clamp_select_layout = "clamp-lower-then-upper"
```

For non-clamp product-reduction-dequant candidates, these four fields are
invalid and must be rejected rather than ignored.

The realized body shape is:

```text
setvl %n -> with_vl %vl
  vsetvl_region_marker phase=load-product-reduce index=1 count=2
  load lhs/rhs
  widening_product
  standalone_reduce -> !tcrv_rvv.vector<i32, "m1">
  gearbox_cross_region_handoff %reduced, %vl, %n
  with_vl %vl
    vsetvl_region_marker phase=dequant-store index=2 count=2
    dequantize %handoff, %scale, %vl
    optional clamp: splat lower/upper -> compare/select lower -> compare/select upper
    store
```

The construction-protocol typed compute chain for the non-clamp route is:

```text
tcrv_rvv.widening_product
+tcrv_rvv.standalone_reduce
+tcrv_rvv.gearbox_cross_region_handoff
+tcrv_rvv.dequantize
```

The dequant-clamp route extends the same chain with compare/select:

```text
tcrv_rvv.widening_product
+tcrv_rvv.standalone_reduce
+tcrv_rvv.gearbox_cross_region_handoff
+tcrv_rvv.dequantize
+tcrv_rvv.compare
+tcrv_rvv.select
```

### 3. Contracts

- The handoff op is a structural boundary, not a math op, intrinsic wrapper,
  route id, descriptor, artifact mirror, or executable proof.
- `dequantize` must consume the handoff result for
  `widening_product_reduce_dequantize_f32` and
  `widening_product_reduce_dequant_clamp_f32`. Direct
  `standalone_reduce -> dequantize` is fail-closed for both routes.
- The producer `with_vl` must contain the product/reduction chain and the
  direct handoff. The consumer `with_vl` must be nested under the producer,
  structurally after the handoff, use the same `!tcrv_rvv.vl`, and contain the
  dequant-store marker, handoff-consuming `dequantize`, optional lower/upper
  clamp compare/select epilogue, and final store.
- For the dequant-clamp route, lower/upper splats must use the runtime
  `lower_bound`/`upper_bound` ABI values, compare/select order must be
  lower-then-upper, and the store must consume the final clamped f32 value
  inside the consumer `with_vl`.
- The handoff input must be the selected `standalone_reduce` i32 result; the
  reduction input must be the selected `widening_product` result.
- The handoff must consume the same `!tcrv_rvv.vl` token as the producer
  `with_vl`, consumer `with_vl`, product, reduction, and dequantize.
- The handoff runtime AVL operand must be the same SSA value consumed by
  `setvl`, and `runtime_avl_source` must be `runtime_abi:n`.
- The handoff `contract`, `from_phase`, `to_phase`, `region_count`, and
  `planning_contract`, and `resource_decision` must match the RVV-owned
  low-precision Gearbox resource facts.
- For `widening_product_reduce_dequant_clamp_f32`, the RVV-owned selected
  resource candidate must also derive and consume
  `clamp_region_index`, `clamp_phase`, `clamp_compare_select_phase`, and
  `clamp_select_layout`. `clamp_region_index` must equal the dequant consumer
  region index, `clamp_phase` must be `dequant-clamp-store`,
  `clamp_compare_select_phase` must be
  `lower-then-upper-compare-select`, and `clamp_select_layout` must be
  `clamp-lower-then-upper`.
- The selected-body realization owner writes those clamp facts onto both
  realized `with_vl` scopes and the Gearbox handoff. The route planner must
  read them from the realized body/handoff before constructing a
  `TCRVEmitCLowerableRoute`; target artifact validation must compare the
  route-plan/header mirrors against the same provider-owned selection.
- The RVV schedule pass, selected-body realizer, construction protocol, route
  planner, provider family plan, and target artifact/header validation must all
  agree on the typed compute chain, producer/consumer scope facts, resource
  selected candidate, and runtime ABI order. Common EmitC must not infer or
  invent this handoff, scope pairing, region order, resource decision, or clamp
  semantics.

### 4. Validation & Error Matrix

- Missing handoff before dequantize -> fail closed before route support.
- `dequantize` consumes the reduction result while a handoff exists -> provider
  fails closed before route support.
- Handoff uses stale contract, phase, region count, runtime AVL source,
  planning contract, or resource decision -> dialect verifier or provider fails
  closed.
- Handoff consumes a different VL token or runtime AVL SSA value than `setvl` /
  `with_vl` -> verifier/provider fails closed.
- Missing, sibling, unordered, or wrong-VL consumer `with_vl` for the
  dequant-store phase -> selected-boundary collection or provider planning
  fails closed before route support.
- Marker count/order/phase/planning contract/resource decision diverges from
  the handoff/resource facts -> fail closed; markers remain transitional
  evidence, not route authority.
- Marker has a supported low-precision resource decision but a missing or stale
  `planning_contract` -> fail closed in RVV marker verifier or provider
  marker-structure validation before route support.
- Dequant-clamp route missing lower/upper bound roles, lower-then-upper
  compare/select dataflow, dequant-store marker, clamp result store, or
  `product-reduction-dequant-clamp-f32` resource candidate -> fail closed
  before target artifact export.
- Dequant-clamp realized scope lacks `clamp_phase`, carries a stale
  `clamp_phase`, has `clamp_region_index != dequant_region_index`, or carries
  a stale compare/select phase/layout -> route planner fails before route
  construction.
- Dequant-clamp handoff lacks or stales `clamp_region_index`, `clamp_phase`,
  `clamp_compare_select_phase`, or `clamp_select_layout` -> handoff verifier
  or route planner fails before Common EmitC materialization.
- Target route metadata/header mirrors a stale clamp region/phase/layout fact
  -> target artifact validation fails before accepting the artifact.
- Non-clamp product-reduction-dequant route carries any dequant-clamp clamp
  realization fact -> selected-body/provider/target validation fails closed.

### 5. Good/Base/Bad Cases

- Good: pre-realized product-reduce-dequant body -> Gearbox resource pass ->
  selected-body realization emits producer/consumer `with_vl` scopes and
  handoff -> provider validates handoff, scope order, markers, and resource
  facts -> route plan/header export.
- Good: pre-realized or explicit product-reduce-dequant-clamp body -> Gearbox
  resource pass -> selected-body realization emits producer/consumer `with_vl`
  scopes and handoff -> consumer dequantizes the handoff result, performs
  lower/upper compare/select clamp, and stores the clamped f32 value -> provider
  validates handoff, scope, resource, clamp, and ABI facts -> route plan/header
  export.
- Bad: only `vsetvl_region_marker` or artifact metadata says two regions while
  `dequantize` directly consumes `standalone_reduce`; route support must fail.
- Bad: dequant-clamp direct epilogue preserves
  `standalone_reduce -> dequantize -> compare/select -> store` without a
  handoff-consuming nested consumer `with_vl`; route support must fail.

### 6. Tests Required

- Dialect/verifier evidence for stale handoff attrs, VL/runtime AVL mismatch,
  and invalid source chain.
- Selected-body realization FileCheck showing
  `standalone_reduce -> gearbox_cross_region_handoff -> nested consumer with_vl
  -> dequantize -> store` with `runtime_abi:n`, matching marker phases, and
  marker `planning_contract =
  "rvv-low-precision-production-resource-planning-contract.v1"`.
- Dequant-clamp selected-body realization FileCheck must additionally show
  lower/upper splats, lower compare/select, upper compare/select, and final
  clamped store inside the nested consumer `with_vl`.
- Explicit already-realized fixture containing the same handoff as the authority
  for route planning.
- Provider fail-closed evidence for missing handoff, stale dequantize consumer,
  stale scope facts, stale or missing marker/handoff `planning_contract`, and
  stale realized region/resource facts after schedule/resource facts exist.
- Gate 3B dequant-clamp evidence showing full clamp realization facts on
  producer and consumer `with_vl` scopes, short clamp handoff attrs on
  `tcrv_rvv.gearbox_cross_region_handoff`, route-plan metadata mirrors, and
  target header mirrors.
- Gate 3B negative evidence for stale realized-body clamp phase, stale handoff
  clamp select layout, and stale target clamp select layout before route or
  artifact acceptance.
- Gate 3C family completion/audit evidence must prove the same provider-owned
  realization-decision mapping drives Gearbox schedule validation,
  selected-body realization, handoff verification, route planning, and target
  validation. It must also prove non-clamp product-dequant siblings reject
  dequant-clamp realization facts before Common EmitC materialization, while
  grouped/unpacked-byte and packed-i4 siblings remain passing.
- Header/artifact export evidence showing the four-op non-clamp typed compute
  chain, or six-op dequant-clamp typed compute chain, is accepted by
  construction and target validation.

### 7. Wrong vs Correct

Wrong:

```text
standalone_reduce -> dequantize
plus marker phase metadata
plus route artifact says product-reduce-dequant
```

Correct:

```text
standalone_reduce
  -> gearbox_cross_region_handoff(%reduced, %vl, %n)
  -> nested consumer with_vl(%vl)
       -> dequantize(%handoff, %scale, %vl)
       -> optional clamp compare/select using lower_bound/upper_bound
       -> store
  -> provider validates handoff + scope + resource facts before route support
```

## Migrated Statement-Plan Provider Consumption Boundary

### 1. Scope / Trigger

After elementwise arithmetic, compare/select, widening conversion, ordinary
vector reduction, standalone reduction, plain MAcc, base memory,
computed-mask memory, segment2 memory, and computed-mask accumulation have
their own RVV-owned statement plans, the
selected-body RVV provider must consume those migrated families through one
shared provider-neutral boundary.

`RVVEmitCRouteProvider` remains the owner that instantiates
`TCRVEmitCLowerableRoute`, records neutral headers/type mappings/ABI mappings,
preserves selected-boundary source provenance, and attaches returned
provider-ready statements. It must not manually sequence each migrated family
statement-plan getter in the central provider body and must not locally rebuild
migrated family statements from operation names, ABI strings, route ids,
intrinsic mirrors, mask/address/accumulator mirrors, or artifact metadata.

### 2. Signatures

The durable planning/provider API is:

```c++
struct RVVSelectedBodyMigratedRouteStatementPlanOwner {
  family name;
  migrated statement-plan family enum;
  consumer predicate over RVVSelectedBodyEmitCRouteDescription;
  statement-plan builder over RVVSelectedBodyRouteAnalysis,
      route materialization facts,
      elementwise/select operand-binding facts,
      memory operand-binding facts,
      math operand-binding facts,
      residual operand-binding facts,
      output migrated statement plan;
};

getRVVSelectedBodyMigratedRouteStatementPlanOwners()
isRVVSelectedBodyMigratedRouteStatementPlanConsumer(
    RVVSelectedBodyEmitCRouteDescription)

llvm::Expected<RVVSelectedBodyMigratedRouteStatementPlan>
getRVVSelectedBodyMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context);
```

The provider must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the
elementwise/select, memory, math, and residual operand-binding facts for the
same analysis.

The aggregate boundary must select statement-plan ownership through
`getRVVSelectedBodyMigratedRouteStatementPlanOwners()`. Every migrated family
must appear exactly once in that registry. The aggregate boundary may dispatch
to family-specific builders, but it must not manually call every family getter
from a central sequence and then infer ownership from whichever plan happens
to return non-empty.

### 3. Contracts

`RVVSelectedBodyMigratedRouteStatementPlan` is RVV-local provider input. It
may carry:

- a migrated family tag for exactly one matched family or `None` for unrelated
  routes;
- provider-ready full-chunk `setvl` pre-loop steps;
- one provider-ready `TCRVEmitCForLoop` that was produced by the owning family
  statement-plan builder;
- source operation provenance and ABI/VL/mask/address/accumulator/source facts
  preserved by the family-specific plan.

This aggregate boundary is not a common EmitC fact, not artifact metadata, not
an acceptance/status mirror, and not a route-support declaration by itself.
The owner registry is dispatch/locality structure only. It must not merge
family semantics, choose intrinsics, infer dtype/config, or weaken the owning
family statement-plan validators.

### 4. Validation & Error Matrix

- An owner registry entry lacks a consumer or builder hook -> fail closed
  before statement construction.
- More than one migrated statement-plan owner matches one selected route ->
  fail closed with all matching owner names before provider route construction.
- Exactly one migrated owner matches but its builder returns no migrated plan
  or the wrong family tag -> fail closed before provider route construction.
- A migrated-family route lacks its valid family statement plan -> fail closed
  before generic provider-local statement assembly and before common EmitC.
- More than one migrated family claims the same selected route -> fail closed
  before route statement construction.
- A non-migrated or unrelated route requests the boundary -> return an
  empty/default migrated statement plan and leave the older route surface
  unchanged.
- Required family-specific plan dependencies remain checked by the owning
  family statement-plan builder; the aggregate boundary must not weaken those
  diagnostics.

### 5. Good/Base/Bad Cases

- Good: verified route-family plans -> materialization facts -> RVV-owned
  operand-binding facts -> owner registry selects exactly one migrated
  statement-plan owner -> `RVVSelectedBodyMigratedRouteStatementPlan` ->
  provider attaches returned statements into `TCRVEmitCLowerableRoute`.
- Base: ordinary vector `reduce_add` with `VectorRHSLoad` is a migrated owner
  family. Its statement plan must be produced from same-analysis math
  operand-binding facts and materialization facts before the provider attaches
  it; the provider must not reconstruct its setvl/load/reduce/store sequence
  from operation names or intrinsic mirrors.
- Base: widening MAcc, dot-reduction routes, computed-mask standalone reduction
  variants, residual runtime scalar splat-store, and future families remain
  outside this migrated aggregate until their statement plans become migrated
  owners.
- Bad: provider body manually calls each migrated family statement-plan getter
  and carries family-specific fallback diagnostics.
- Bad: provider body branches on migrated operation names to rebuild
  setvl/load/splat/compare/mask/store/accumulator statements after this shared
  boundary exists.

### 6. Tests Required

- Focused C++ tests for migrated statement-plan owner registry membership,
  owner names, family tags, hook presence, exact-once classification for each
  migrated family, and empty-plan behavior for unrelated routes.
- Focused C++ tests for positive aggregate-boundary construction and provider
  consumption across representative migrated families.
- Focused C++ fail-closed diagnostics for at least one missing or stale
  migrated statement-plan dependency through the aggregate boundary.
- C++ default/empty-plan coverage for unrelated route families.
- Bounded provider scan showing migrated-family statement-plan consumption is
  reached through the aggregate boundary and that the provider no longer
  manually sequences the six family-specific statement-plan getters.
- Active-authority scan over touched RVV planning/provider/test files for
  legacy i32/source-front-door/descriptor/direct-C/source-export or
  mirror-only authority drift.

## Emission Diagnostics And Artifacts

Emission-plan diagnostics, route ids, artifact metadata, manifests, and
generated filenames are exact mirrors after route construction. They are not
authority for route support, dtype, compute, schedule, body shape, correctness,
runtime, performance, or progress.

Target artifact export is valid only after:

1. Selected RVV variant exists.
2. Typed/realized `tcrv_rvv` body exists.
3. RVV plugin legality accepts the body.
4. RVV route provider returns a `TCRVEmitCLowerableRoute`.
5. Common EmitC materializes that route faithfully.
6. Target export packages the materialized output with metadata as mirrors.

Runtime, correctness, or performance claims additionally require real
`ssh rvv` evidence.

### Target Artifact Route-Family Validator Boundary

RVV target artifact export has two target-owned validation layers:

- the bridge layer, which rebuilds the selected-body route, checks generic
  candidate shape, runtime ABI consistency, source-op provenance, forbidden
  descriptor/direct-C/source-export residue, and artifact packaging mechanics;
- route-family validators, which own family-specific artifact acceptance for a
  selected `RVVSelectedBodyEmitCRouteDescription` and the rebuilt
  `TCRVEmitCLowerableRoute`.

The route-family validator registry must dispatch from the rebuilt provider
description, not from artifact metadata, route ids, artifact names, ABI
strings, tests, scripts, direct route entries, or exact intrinsic spellings.
Each validator receives the target candidate, rebuilt lowerable route, and
provider description. It may require metadata keys to mirror provider facts,
but metadata remains evidence after route construction, never support
authority.

Required validator behavior:

- Validate provider facts that are semantic to that family, such as family
  plan ids, provider support labels, type/config relations, route operand
  bindings, headers, type mappings, ABI mappings, and provider-built statement
  plans.
- Validate candidate metadata only as mirrors of those provider facts.
- Fail closed when a required provider fact is absent, when a candidate mirror
  is stale, or when stale facts for a different route subfamily appear on the
  selected body.
- Return success for unrelated route families only through registry dispatch;
  family-specific validators should not make unrelated routes look accepted.

Good: selected typed RVV body -> provider description and rebuilt
`TCRVEmitCLowerableRoute` -> target bridge dispatches to the matching family
validator -> artifact metadata mirrors the validated facts.

Bad: central target-bundle code grows per-family semantic branches, or a family
validator accepts an artifact because a route id, metadata key, ABI string,
artifact filename, script expectation, or intrinsic spelling looks plausible.

### Segment2 Target Export Consumer Contract

For segment2 route families, target artifact export must rebuild the
provider route from the selected typed RVV body and consume the rebuilt
`TCRVEmitCLowerableRoute` plus provider description as authority before
accepting generated artifact/header claims. Emission-plan metadata remains a
mirror after route construction.

Required target-side validations:

- `rvv_emitc_lowerable_route` must equal the rebuilt route id, but the
  metadata route id must not choose the route.
- `tcrv_rvv.provider_supported_mirror`, runtime ABI order, route operand
  binding plan/operands, required header declarations, C type mapping,
  segment memory layout, source/destination memory form, and segment count
  must exactly mirror the provider description.
- Plain segment2 families (`segment2_deinterleave_unit_store`,
  `segment2_interleave_unit_load`) require the plain
  `segment2MemoryRouteFamilyPlanID` /
  `tcrv_rvv.segment2_memory_route_family_plan` mirror and must not accept a
  computed-mask route-family mirror as support authority.
- Computed-mask segment2 families
  (`computed_masked_segment2_load_unit_store`,
  `runtime_scalar_cmp_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`,
  `runtime_scalar_cmp_masked_segment2_store_unit_load`,
  `computed_masked_segment2_update_unit_load`) require
  `computedMaskMemoryRouteFamilyPlanID` /
  `tcrv_rvv.computed_mask_memory_route_family_plan` plus mask producer/role/
  source/memory-form mirrors and must not require the plain segment2 family
  plan mirror.
- The rebuilt route must carry provider-owned ABI mappings in order, selected
  typed RVV source provenance for pre-loop and loop statements, and a runtime
  AVL/VL loop whose upper bound is the runtime `n`/AVL ABI parameter.

Statement-plan checks are family-specific:

- Plain segment2 deinterleave is a segment-load path. It validates the
  provider-built `segment2_load`, field extraction/move, and field-store
  statements; it must not be rejected for lacking ordinary vector loads.
- Plain segment2 interleave validates ordinary field loads plus tuple creation
  and segment-store statements.
- Runtime-scalar computed-mask segment2 load validates lhs load, runtime
  scalar splat, compare/mask construction, old field passthrough loads,
  masked segment-load, field extraction, and field-store statements. It must
  reject stale vector RHS load, segment-store, or field-payload facts before
  artifact export.
- Computed-mask segment2 store validates compare/mask construction, field
  payload loads, tuple creation, and masked segment-store statements. It must
  reject stale segment-load or field-extract facts before artifact export.
- Runtime-scalar computed-mask segment2 store validates lhs load, runtime
  scalar splat, compare/mask construction, field payload loads, tuple
  creation, and masked segment-store statements. It must reject stale vector
  RHS load, segment-load, or field-extract facts before artifact export.
- Computed-mask segment2 update validates compare/mask construction, field
  payload loads, update arithmetic, tuple creation, and masked segment-store
  statements. It must reject stale segment-load or field-extract facts before
  artifact export.

Wrong vs correct:

- Wrong: target export accepts a stale route id, ABI order, C type map,
  header list, provider mirror, operand binding, or artifact name because the
  emission plan says `supported`.
- Correct: target export rebuilds the provider route, compares every mirror to
  the rebuilt route/provider description, and fails closed before emitting an
  executable artifact when any mirror, ABI binding, header/type mapping, mask
  fact, or segment2 statement fact is stale or missing.

## Fail-Closed Legacy Inventory

The following names may appear in tests/specs only as deprecated/fail-closed
inventory or historical residue:

```text
RVVI32M1ArithmeticRouteSpec
RVVI32M1ArithmeticSlice
collectRVVI32M1ArithmeticSlice
rvv-i32m1-arithmetic-emitc-route-family
rvv-i32m1-{add,sub,mul,cmp-select}-callable-c-abi.v1
tcrv_rvv.i32_load
tcrv_rvv.i32_broadcast_load
tcrv_rvv.i32_add
tcrv_rvv.i32_sub
tcrv_rvv.i32_mul
tcrv_rvv.i32_cmp_eq
tcrv_rvv.i32_select
tcrv_rvv.i32_store
!tcrv_rvv.i32m1
__riscv_*_i32m1
```

Safe use:

- negative tests proving these paths fail closed;
- parser/verifier fixtures that do not generate artifacts;
- migration notes pointing to deletion/rewrite;
- historical comments explaining why they are not architecture authority.

Unsafe use:

- supported object/header/bundle plans;
- positive generated-artifact tests;
- compatibility routes;
- route-provider inputs;
- dtype propagation;
- route ids or artifact names used as compute authority.

## Review Checklist

- [ ] Does the selected RVV path start from `tcrv.exec` envelope plus selected RVV variant?
- [ ] Does the variant contain or reference a typed low-level `tcrv_rvv` body?
- [ ] Are dtype/config/operation facts structural in the typed or realized body?
- [ ] Are ABI roles declared in `tcrv.exec` and explicitly consumed in the body?
- [ ] Does RVV plugin legality run before route construction?
- [ ] Does selected-body realization consume any code-affecting hints/config into body structure?
- [ ] Does the route provider build `TCRVEmitCLowerableRoute`?
- [ ] Does common EmitC only materialize provider output?
- [ ] Are source-front-door and legacy i32m1 paths fail-closed by default?
- [ ] Are emission diagnostics and artifacts mirrors only?
- [ ] Is `ssh rvv` evidence present for any runtime/correctness/performance claim?
