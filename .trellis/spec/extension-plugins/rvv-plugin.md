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

### Stage 2 Performance Layer

RVV plugin-local selected-body realization is one linear compiler step:

```text
selected pre-realized tcrv_rvv body
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
struct RVVElementwiseCompareSelectRealizationResult {
  tcrv::rvv::WithVLOp boundary;
  bool applies() const;
};

llvm::Expected<RVVElementwiseCompareSelectRealizationResult>
realizePreRealizedRVVElementwiseCompareSelectCluster(
    const VariantLoweringBoundaryRequest &request,
    mlir::Operation *bodyOp);
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

## Selected-Body Route-Entry Realization Bridge

### 1. Scope / Trigger

RVV production route/emission entries must use an RVV plugin-owned
selected-body route-entry realization bridge when a selected variant reaches
route construction with a supported pre-realized selected body. This bridge is
for statement-plan-backed selected-body families whose realization already
exists in RVV plugin code. It must not turn provider/common EmitC into a
semantic realization fallback.

The route-entry bridge may support bounded family groups such as:

- elementwise/compare-select pre-realized bodies owned by
  `realizePreRealizedRVVElementwiseCompareSelectCluster(...)`;
- base memory movement pre-realized bodies whose realized structure already
  feeds RVV-owned base-memory materialization facts, memory operand-binding
  facts, and statement plans.
- standalone reduction pre-realized bodies whose realized structure already
  feeds RVV-owned standalone-reduction family plans, materialization facts,
  math operand-binding facts, and migrated statement plans.

Unlisted pre-realized families must fail closed at the route-entry bridge
unless their owning route-entry support is explicitly added with matching
facts, statement-plan, and tests.

### 2. Signatures

The durable route-entry predicates/helpers are:

```c++
bool variantContainsPreRealizedRVVRouteEntrySelectedBody(
    tcrv::exec::VariantOp variant);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRouteEntrySelectedBody(
    const VariantLoweringBoundaryRequest &request);
```

`RVVExtensionPlugin::buildVariantEmissionPlan(...)` and
`RVVExtensionPlugin::buildVariantEmitCLowerableRoute(...)` must call a
route-entry helper that either returns the existing unique realized
`tcrv_rvv.with_vl` boundary or invokes
`realizePreRealizedRVVRouteEntrySelectedBody(...)` before route facts are
collected.

### 3. Contracts

The route-entry bridge consumes only RVV-owned inputs:

- selected `tcrv.exec.kernel` and `tcrv.exec.variant`;
- typed pre-realized `tcrv_rvv` body structure;
- runtime ABI SSA imports and selected variant `requires` metadata;
- attrs carried by the pre-realized body, such as SEW, LMUL, policy, operation
  kind, memory form, predicate/mask/layout, stride/index facts, and runtime
  `n`/AVL values.

It produces a realized `tcrv_rvv.with_vl` boundary by calling the owning RVV
realization path. It does not build `TCRVEmitCLowerableRoute`; route
construction still occurs after route analysis, materialization facts,
operand-binding facts, and family statement plans validate realized typed body
structure.

### 4. Validation & Error Matrix

- Missing kernel or variant -> fail closed before realization.
- Existing realized `setvl`/`with_vl` boundary and no pre-realized body ->
  return the realized boundary.
- Existing realized `setvl`/`with_vl` boundary mixed with any pre-realized
  body -> fail closed before route construction.
- No realized boundary and no pre-realized body -> preserve the structural
  missing-boundary diagnostic.
- No realized boundary and an unsupported pre-realized route-entry family ->
  fail closed with a route-entry realization diagnostic.
- Supported route-entry family with malformed runtime ABI roles, attrs, or
  incomplete realization facts -> fail closed in the owning RVV realization
  path before provider/common route construction.
- Provider route analysis sees any pre-realized body -> fail closed with a
  selected-body realization diagnostic; provider/common code must not invent
  missing typed structure.

### 5. Good/Base/Bad Cases

- Good: selected RVV variant -> pre-realized compare/select, base-memory, or
  standalone-reduction body -> route-entry realization bridge -> realized
  `tcrv_rvv` body -> RVV-owned facts/statement plan -> provider-built route ->
  common EmitC.
- Base: explicit already-realized selected body -> route-entry helper returns
  the unique `with_vl` boundary and preserves existing route behavior.
- Bad: route provider sees `typed_*_pre_realized_body` and synthesizes
  setvl/load/store/compare/select/memory structure itself.
- Bad: common EmitC infers RVV dtype, operation kind, memory form, intrinsic
  spelling, or route support from route ids, status fields, artifact metadata,
  ABI names, or test names.

### 6. Tests Required

- C++ tests showing production emission/provider route entries realize at
  least one compare/select pre-realized body, one base-memory pre-realized
  body, and one standalone-reduction pre-realized body before route facts are
  collected.
- C++ fail-closed coverage for an unsupported route-entry family or incomplete
  realization dependency.
- Representative lit/FileCheck coverage proving direct pre-realized route
  materialization works without first running
  `--tcrv-materialize-selected-lowering-boundaries`.
- Regression coverage showing explicit already-realized selected-body
  artifacts and explicit selected-boundary materialization still pass.
- Bounded scans showing common EmitC/export and provider/common code did not
  become semantic realization owners.

### 7. Wrong vs Correct

Wrong:

```text
provider/common EmitC:
  sees pre-realized body or route metadata
  -> chooses RVV family, memory form, dtype, and statement sequence
```

Correct:

```text
selected pre-realized tcrv_rvv body
  -> RVV route-entry realization bridge
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
  if operation is masked_add, strided_add, or runtime_i32_splat_store,
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
production-active.

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

### 4. Validation & Error Matrix

- A non elementwise-arithmetic route requests the boundary -> return an empty
  plan without changing unrelated route-family behavior.
- An included elementwise arithmetic route has no matching verified family
  plan -> fail closed before route statement construction.
- An included route lacks the required operand-binding facts from the
  elementwise/select or residual facts boundary -> fail closed before route
  statement construction.
- Required ABI roles such as `lhs`, `rhs`, `rhs_scalar`, `out`, runtime count,
  or stride roles are absent -> fail closed with the logical operand name and
  operation/memory-form context.
- Required materialization leaves such as `setvl`, load, scalar broadcast,
  compute, compare, masked merge, strided load, or strided store are absent ->
  fail closed before common EmitC.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before common
  EmitC.

### 5. Good/Base/Bad Cases

- Good: typed elementwise arithmetic `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> elementwise/select or residual operand-binding facts
  -> RVV-owned statement plan -> provider-built route.
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
  `Add`/`Sub`/`Mul`, and strided `Add`.
- C++ fail-closed diagnostics for at least one missing or stale statement-plan
  dependency before route statement construction.
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
materialization facts, and operand-binding facts have been validated. The RVV
planning layer must expose one RVV-owned statement-plan boundary for plain
compare-select, computed-mask select, runtime-scalar computed-mask select, and
runtime-scalar dual compare-mask-and-select where those routes are
production-active.

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
```

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the
elementwise/select operand-binding facts for the same analysis. Non-consumer
route families receive an empty/default statement plan.

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
route materialization facts, and RVV-owned elementwise/select operand-binding
facts. It is not a common EmitC fact, not artifact metadata, not an
acceptance/status mirror, and not a route-support declaration by itself.

### 4. Validation & Error Matrix

- A non compare/select route requests the boundary -> return an empty plan
  without changing unrelated route-family behavior.
- An included compare/select route has no matching verified plain
  compare-select or computed-mask select family plan -> fail closed before
  route statement construction.
- An included route lacks the required elementwise/select operand-binding
  facts -> fail closed before route statement construction.
- Required ABI roles such as `lhs`, `rhs`, `cmp_lhs`, `cmp_rhs`,
  `rhs_scalar`, `cmp_lhs_b`, `rhs_scalar_b`, `true_value`, `false_value`,
  `out`, or runtime count are absent -> fail closed with the logical operand
  name and operation/memory-form context.
- Required materialization leaves such as `setvl`, vector load, runtime scalar
  splat, compare, secondary compare, mask-and, select, or store are absent ->
  fail closed before common EmitC.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before common
  EmitC.

### 5. Good/Base/Bad Cases

- Good: typed compare/select `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> elementwise/select operand-binding facts ->
  RVV-owned statement plan -> provider-built route.
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
route materialization facts, and RVV-owned memory operand-binding facts. It is
not a common EmitC fact, not artifact metadata, not an acceptance/status
mirror, and not a route-support declaration by itself.

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
facts, and memory operand-binding facts have been validated. The RVV planning
layer must expose one RVV-owned statement-plan boundary for runtime-scalar
computed-mask store/load-store, computed-mask unit load/store, computed-mask
strided store, computed-mask strided load/unit-store, computed-mask indexed
gather/unit-store, and computed-mask indexed scatter/unit-load where those
routes are production-active.

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

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the memory
operand-binding facts for the same analysis. Non-consumer route families and
excluded computed-mask segment2 routes receive an empty/default statement
plan.

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
route materialization facts, and RVV-owned memory operand-binding facts. It is
not a common EmitC fact, not artifact metadata, not an acceptance/status
mirror, and not a route-support declaration by itself.

### 4. Validation & Error Matrix

- A non computed-mask memory route requests the boundary -> return an empty
  plan without changing unrelated route-family behavior.
- A computed-mask segment2 memory route requests the boundary -> return an
  empty plan and leave segment2 statement construction to its own owner.
- An included computed-mask memory route has no verified computed-mask memory
  family plan -> fail closed before route statement construction.
- An included route lacks required memory operand-binding facts -> fail closed
  before route statement construction.
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
  materialization facts -> memory operand-binding facts -> RVV-owned statement
  plan -> provider-built route.
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
  dependency before route statement construction.
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
  -> RVVSelectedBodyComputedMaskMemoryRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
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
interleave/unit-load, computed-mask segment2 load/unit-store, and
computed-mask segment2 store/unit-load where those routes are
production-active.

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
```

`RVVEmitCRouteProvider` must call this boundary after
`verifyRVVSelectedBodyRouteFamilyProviderPlans(analysis, context)`, after
obtaining route materialization facts, and after obtaining the memory
operand-binding facts for the same analysis. Non-consumer route families
receive an empty/default statement plan.

### 3. Contracts

`RVVSelectedBodySegment2MemoryRouteStatementPlan` is RVV-local provider input.
It may carry:

- a pointer to the verified plain segment2 family plan or computed-mask memory
  family plan that justifies the selected segment2 statement sequence;
- sub-family booleans for plain deinterleave, plain interleave,
  computed-mask segment2 load, and computed-mask segment2 store;
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
- Required ABI roles such as `src`, `dst`, `cmp_lhs`, `cmp_rhs`, `field0`,
  `field1`, and runtime count are absent -> fail closed with the logical
  operand name and operation/memory-form context.
- Required materialization leaves such as `setvl`, vector load, compare,
  segment load/store, masked segment load/store, tuple create, field extract,
  or field store are absent -> fail closed before common EmitC.
- Required source operation provenance for configure/load/compute/store steps
  is absent or reports the wrong EmitC source role -> fail closed before
  common EmitC.

### 5. Good/Base/Bad Cases

- Good: typed segment2 `tcrv_rvv` body -> family plan verifier ->
  materialization facts -> memory operand-binding facts -> RVV-owned statement
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
  interleave/unit-load, computed-mask segment2 load/unit-store, and
  computed-mask segment2 store/unit-load.
- C++ fail-closed diagnostics for at least one missing or stale
  statement-plan dependency before route statement construction.
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
obtaining route materialization facts, and after obtaining math operand-binding
facts for the same analysis. Non-consumer route families receive an
empty/default statement plan.

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
route materialization facts, and RVV-owned math operand-binding facts. It is
not a common EmitC fact, not artifact metadata, not an acceptance/status mirror,
and not a route-support declaration by itself.

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
  materialization facts -> math operand-binding facts -> RVV-owned statement
  plan -> provider-built route.
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
  for computed-mask MAcc and runtime-scalar computed-mask MAcc.
- C++ fail-closed diagnostics for at least one missing or stale
  statement-plan dependency before route statement construction.
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
  -> RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan
  -> provider attaches plan into TCRVEmitCLowerableRoute
```

## Plain And Scalar-Broadcast MAcc Statement-Plan Boundary

### 1. Scope / Trigger

For production-active plain `macc_add` and scalar-broadcast
`scalar_broadcast_macc_add` selected-body routes, `RVVEmitCRouteProvider` must
not locally recreate the setvl/load/splat/load-accumulator/MAcc/store
statement sequence from operation names, ABI strings, intrinsic mirrors,
accumulator-layout mirrors, route ids, or artifact metadata after route
materialization facts and math operand-binding facts have been validated. The
RVV planning layer must expose one RVV-owned statement-plan boundary for the
bounded typed MAcc route family and its scalar-broadcast sub-family.

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
materialization facts, and RVV-owned math operand-binding facts. It is not a
common EmitC fact, not artifact metadata, not an acceptance/status mirror, and
not a route-support declaration by itself.

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
  operand-binding facts -> RVV-owned statement plan with scalar splat leaf ->
  provider-built route.
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
  construction and provider consumption.
- Focused C++ fail-closed diagnostics for missing math facts, MAcc compute
  leaf, vector load leaf, scalar splat leaf, missing/stale scalar-broadcast
  family plan mirror, or source-role provenance before route statement
  construction.
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

## Migrated Statement-Plan Provider Consumption Boundary

### 1. Scope / Trigger

After elementwise arithmetic, compare/select, widening conversion, standalone
reduction, plain MAcc, base memory, computed-mask memory, segment2 memory, and
computed-mask accumulation have their own RVV-owned statement plans, the
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

### 4. Validation & Error Matrix

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
  operand-binding facts -> `RVVSelectedBodyMigratedRouteStatementPlan` ->
  provider attaches returned statements into `TCRVEmitCLowerableRoute`.
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
