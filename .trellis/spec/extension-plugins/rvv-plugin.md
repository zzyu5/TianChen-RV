# RVV Plugin

## Role

RVV plugin is the current primary real hardware extension family for
TianChen-RV MLIR. It is part of the unified TCRV system, not an independent
backend dialect.

Environment:

```text
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: primary development, primary performance, primary correctness
```

## Responsibilities

RVV plugin provides:

- RVV capability registration;
- `tcrv.rvv` dialect registration;
- RVV variant generation;
- RVV legality verification;
- RVV plugin-local selected-body realization and tuning;
- RVV cost model;
- RVV emission paths;
- RVV runtime/threading integration.

It does not define high-level matmul semantics, generic tensor/tile IR, IME internals, offload internals, custom ISA internals, an intrinsic-wrapper dialect, or a global autotuning/status system.

## RVV Route-Authority Replacement

The durable RVV direction is:

```text
tcrv.exec envelope
  -> selected RVV variant
  -> vector-level typed `tcrv_rvv` body
  -> RVV plugin-local legality and selected-body realization
  -> RVV-owned `TCRVEmitCLowerableRoute`
  -> common EmitC materialization and target export
```

Stage 1 is RVV route-authority replacement. It must replace or fail-close
active compiler paths that treat bounded `i32m1` arithmetic, route ids,
artifact names, source-front-door patterns, descriptor residue, or intrinsic
spellings as RVV route authority. The replacement authority is the selected
vector-level `tcrv_rvv` body plus RVV-owned legality, realization, and route
construction. Unsupported combinations must fail closed.
During Stage 1, realization work is limited to the fail-closed plugin boundary,
hook, or faithful selected-body consumption needed to remove old route
authority. Performance-sensitive selected-body realization and tuning are
Stage 2 RVV completion work.

The route builder must faithfully emit from the selected and, when needed,
realized `tcrv_rvv` body. It may map body structure to RVV vector types,
headers, intrinsic callees, ABI bindings, and `TCRVEmitCLowerableRoute`
payloads, but it must not invent missing RVV computation, schedule, dtype,
SEW/LMUL policy, or body shape from route strings, artifact metadata, test
names, or old i32m1 helpers.

An RVV route is not a decorator on an old operation, route id, descriptor, or
artifact. It is the provider-built lowering payload after the selected
vector-level `tcrv_rvv` body already carries the operation, dtype, config,
memory form, runtime value use, and policy facts needed for legality. The RVV
provider builds `TCRVEmitCLowerableRoute`; the common materializer lowers that
route to MLIR EmitC. Neither route construction nor common EmitC/export may
retrofit semantics onto a body that only encodes legacy `i32_*` names or
metadata.

RVV target artifact export is part of the same authority chain. Object,
declaration-header, and bundle export may consume route ids, artifact names,
runtime ABI parameter lists, arithmetic metadata, source-op provenance, and
bundle component metadata only as mirrors of the selected typed body after the
RVV-owned route builder has rebuilt and validated the `TCRVEmitCLowerableRoute`.
If the selected variant has no explicit typed `tcrv_rvv` body, the selected
boundary does not match the selected variant, the runtime ABI roles are missing
or reordered, or the candidate metadata disagrees with the rebuilt body, target
export must fail before emitting object/header/bundle bytes.

### Dtype / Config / Body Authority

For current Stage 1 and Stage 2 work, "start from `tcrv.exec`" means start from
a complete TianChen-RV execution surface:

```text
tcrv.exec.kernel
  -> selected tcrv.exec.variant {origin = "rvv-plugin", requires = [...]}
  -> explicit typed vector-level tcrv_rvv body inside that variant
```

It does not mean a bare `tcrv.exec.kernel` may infer RVV compute, dtype, shape,
or schedule from capability names, route ids, ABI strings, parameter names,
artifact metadata, or test filenames.

Authority is layered:

```text
tcrv.exec envelope:
  declares kernel/capability/variant/dispatch/fallback and binds ABI/runtime
  roles through mem_window/runtime_param.

tcrv_rvv body:
  imports those ABI/runtime values and carries the vector-level RVV dataflow,
  dtype semantics, memory forms, operation kinds, VL/AVL use, mask/tail policy,
  and config constraints needed by RVV lowering.

RVV plugin:
  validates the selected body against target capability, consumes runtime SSA
  values and optional hints, realizes legal RVV structure when needed, and
  derives concrete vector type, intrinsic, ABI, and EmitC route payload choices.
```

Dtype comes from source semantics in future frontend flows, or from the explicit
typed `tcrv_rvv` body in current hand-authored / fixture-generated Stage 1/2
flows. SEW, LMUL, vtype policy, VL placement, memory form, accumulator layout,
and unroll/prefetch choices are RVV config/realization decisions constrained by
that dtype/body, target capability, runtime SSA values, and optional hints.
They must be present in the typed `tcrv_rvv` body or consumed into a realized
`tcrv_rvv` body before route construction. They must not be recovered from
`!tcrv_rvv.i32m1` helper names, route strings, intrinsic spellings, ABI strings,
artifact names, descriptor residue, or common EmitC/export code.

Wrong vs correct:

```text
Wrong:
  route id says rvv-i32m1-add -> therefore dtype/config/body is i32/m1/add.

Correct:
  selected tcrv_rvv body contains typed vector dataflow and config constraints;
  RVV plugin validates/realizes it and then derives the route/intrinsic names.
```

### Scenario: Pre-Realized I32 Binary Selected-Body Realization

#### 1. Scope / Trigger

Use this contract when a selected `origin = "rvv-plugin"` variant contains a
bounded pre-realized typed RVV body that is not yet in the realized
`setvl -> with_vl -> load -> compute -> store` form consumed by the current
provider route.

#### 2. Signatures

- Pre-realized op:
  `tcrv_rvv.i32_binary_pre_realized_body`.
- Required operands:
  `lhs`, `rhs`, `out`, and `n`; each must be an explicit
  `tcrv_rvv.runtime_abi_value` result, with `n` using index type.
- Required attributes:
  `op_kind`, `memory_form`, `sew`, `lmul`, and `policy`.
- Supported arithmetic specializations:
  `op_kind = "add"`, `"sub"`, or `"mul"`, with
  `memory_form = "vector-rhs-load"`, `sew = 32`, `lmul = "m1"`, and
  tail/mask agnostic policy.
- Realization entry point:
  selected lowering-boundary materialization routed through the selected
  variant's origin plugin.
- Realized output:
  `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, two `tcrv_rvv.i32_load` ops, the
  matching `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, or `tcrv_rvv.i32_mul`
  compute op, and `tcrv_rvv.i32_store` in the same selected variant.

#### 3. Contracts

- The pre-realized op is RVV plugin-owned selected-body IR. It is not a route
  id, descriptor, source-front-door marker, direct-C exporter, intrinsic
  wrapper, artifact authority, runtime proof, correctness proof, or performance
  proof.
- Realization must happen before provider route construction. The provider and
  common EmitC/export paths continue to consume only the realized selected body.
- Realization must preserve computation semantics, dtype semantics, memory
  roles, selected variant origin, required capabilities, dispatch/fallback
  structure, and the exact runtime `n`/AVL SSA value.
- Selected-boundary attributes such as source kernel, selected variant, origin,
  role, status, required capabilities, and construction protocol are attached
  to the realized `tcrv_rvv.with_vl` boundary by the RVV plugin. They must not
  be accepted as authority metadata on the pre-realized op.
- Unsupported operation kinds, memory forms, dtype/config combinations,
  policies, missing runtime ABI roles, or mixed pre-realized plus already
  realized bodies must fail closed before route or artifact construction.

#### 4. Validation & Error Matrix

- Missing selected `tcrv.exec.variant` or enclosing `tcrv.exec.kernel` ->
  fail before realization.
- Variant origin is not `rvv-plugin` -> fail before realization.
- No realized body and not exactly one
  `tcrv_rvv.i32_binary_pre_realized_body` -> fail before route construction.
- `op_kind` is not supported by the realization hook -> fail before route
  construction.
- `memory_form` is unsupported -> fail before route construction.
- `sew`/`lmul`/policy does not match the supported specialization -> fail
  before route construction.
- `lhs`/`rhs`/`out`/`n` is missing, has the wrong role, or `n` is not the
  runtime element-count ABI value -> fail before route construction.
- The pre-realized op carries stale source, route, artifact, selected-boundary,
  descriptor, or capability-summary authority metadata -> verifier rejects it.
- A pre-realized op is mixed with existing `setvl`, `with_vl`, or realized
  dataflow ops -> fail before route construction.

#### 5. Good/Base/Bad Cases

- Good: selected variant contains explicit runtime ABI values and one
  `tcrv_rvv.i32_binary_pre_realized_body` for add, sub, or mul; the RVV plugin
  realizes it into the matching existing arithmetic body, then the provider
  derives the EmitC route from that realized body.
- Base: already realized explicit selected-body add/sub/mul fixtures remain
  valid and do not require this rewrite.
- Bad: common EmitC/export, route ids, artifact names, stale
  `rvv_emitc_route_mapping`, source-front-door labels, or descriptors choose
  the RVV operation or config.

#### 6. Tests Required

- Positive lit coverage for pre-realized add, sub, and mul reaching
  selected-boundary realization, supported emission-plan metadata, and target
  artifact/header or object routing.
- Negative lit coverage for unsupported operation, unsupported config/policy,
  wrong runtime ABI role, missing runtime `n`/AVL authority, and stale route or
  source metadata on the pre-realized op.
- Existing fully realized explicit selected-body add/sub/mul target fixtures
  must continue to pass.
- Focused scans over touched RVV and common EmitC code must show no common-code
  RVV semantic branching, descriptor/direct-C/source-export restoration, or
  route-id/artifact-name authority.

#### 7. Wrong vs Correct

Wrong:

```text
pre-realized metadata or route id -> provider/common export chooses add/i32m1
```

Correct:

```text
pre-realized typed RVV op with explicit ABI/config facts
  -> RVV plugin selected-body realization
  -> realized setvl/with_vl/load/add/store body
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC/target mechanics
```

### Scenario: Selected-Body EmitC Route Description API

#### 1. Scope / Trigger

Use this contract when RVV plugin code decides whether a selected RVV variant
is route-supported or builds a `TCRVEmitCLowerableRoute` for common EmitC
materialization. The trigger is a selected `tcrv.exec.variant` owned by
`rvv-plugin` with an explicit typed `tcrv_rvv` body.

#### 2. Signatures

- Operation kind:
  `RVVSelectedBodyOperationKind`.
- Memory form:
  `RVVSelectedBodyMemoryForm`.
- Route description:
  `RVVSelectedBodyEmitCRouteDescription`.
- Required route-description fields include:
  `emitCRouteID`, `targetArtifactRouteID`, `targetArtifactKind`,
  `runtimeABIName`, and structured runtime ABI parameters.
- Description builder:
  `describeRVVSelectedBodyEmitCRoute(const VariantEmitCLowerableRequest &request, TCRVEmitCLowerableRoute *verifiedRoute = nullptr)`.
- Description verifier:
  `verifyRVVSelectedBodyEmitCRouteDescription(const RVVSelectedBodyEmitCRouteDescription &description, StringRef context)`.
- Route builder:
  `buildRVVSelectedBodyEmitCLowerableRoute(const VariantEmitCLowerableRequest &request, TCRVEmitCLowerableRoute &out)`.

#### 3. Contracts

- `describeRVVSelectedBodyEmitCRoute` is the RVV provider authority boundary.
  It must derive operation kind, memory form, SEW, LMUL, runtime ABI
  parameters, runtime ABI name, EmitC route id, target artifact route id,
  target artifact kind, and intrinsic mapping from the selected typed
  `tcrv_rvv` body plus RVV config/construction contracts.
- A route id, runtime ABI name, artifact metadata entry, source-front-door
  name, helper name, or intrinsic spelling is an output label after validation.
  It must not be parsed as the input authority for operation semantics.
- The optional `verifiedRoute` output may be filled only from the same
  validated selected-body description; it must not bypass body/config/runtime
  checks.
- Before a provider-built description is returned or consumed to build an
  EmitC route, `verifyRVVSelectedBodyEmitCRouteDescription` must prove that
  target artifact route/kind, EmitC route ID, runtime ABI name/contract,
  boundary op, vector/mask C types, exact intrinsic spellings, and ordered
  runtime ABI parameters mirror the selected-body specialization derived from
  validated operation, memory form, SEW, LMUL, tail policy, and mask policy.
- RVV emission readiness, emission-plan materialization, and target artifact
  candidate validation may consume `targetArtifactRouteID` and
  `targetArtifactKind` only after `describeRVVSelectedBodyEmitCRoute` has
  validated the selected typed body. They must not require a stale
  `rvv_emitc_route_mapping` attribute on `tcrv_rvv.with_vl` before this
  provider analysis.
- Common EmitC/export code may consume the provider-built route and generic
  provenance fields, but it must not choose RVV operation, dtype, SEW/LMUL,
  policy, memory form, or intrinsic mapping.
- Retained i32m1 arithmetic support is an ordinary specialization of this
  selected-body description API. It is not the public RVV route-provider
  protocol.

#### 4. Validation & Error Matrix

- Missing materialized `tcrv.exec.variant` or enclosing `tcrv.exec.kernel` ->
  fail before route description.
- Variant origin is not `rvv-plugin` -> fail before route description.
- Selected variant lacks explicit typed `tcrv_rvv` body -> fail before route
  description.
- Missing or inconsistent `setvl`/`with_vl` SEW, LMUL, policy, or visible VL
  relationship -> fail before route description.
- Runtime AVL, load, store, or buffer operands are not defined by explicit
  `tcrv_rvv.runtime_abi_value` ops with expected roles -> fail before route
  description.
- Typed compute/dataflow does not match the bounded selected-body route shape
  -> fail before route id or intrinsic labels are emitted.
- Provider description has a stale target artifact route, runtime ABI label,
  typed compute op, vector type, or intrinsic spelling relative to the
  selected-body specialization -> fail before construction metadata, EmitC
  route materialization, or target artifact export consumes it.
- Artifact metadata disagrees with the rebuilt selected-body route description
  -> target export must fail before object/header/bundle output.
- `tcrv_rvv.with_vl` carries stale or missing `rvv_emitc_route_mapping`
  metadata -> provider analysis must ignore it for route support; unsupported
  or incomplete typed bodies still fail closed.

#### 5. Good/Base/Bad Cases

- Good: selected RVV variant contains runtime ABI values, `setvl`, `with_vl`,
  typed load/compute/store body, and compatible config; the provider derives a
  selected-body description and then emits a matching `TCRVEmitCLowerableRoute`.
- Base: the current bounded i32m1 add/sub/mul/compare-select slice remains
  route-supported only after typed body/config/runtime ABI validation succeeds.
- Bad: an emission plan or target artifact candidate parses
  `rvv-i32m1-add-emitc-route`, `rvv_arithmetic_op`, artifact names, ABI names,
  or intrinsic spellings to decide compute semantics without rebuilding the
  selected typed body route description.

#### 6. Tests Required

- Positive provider or lit coverage for at least one selected typed
  `tcrv_rvv` body route materializing through common EmitC.
- Negative provider/lit coverage for missing runtime ABI or config/VL facts.
- Negative provider coverage proving stale description identity facts such as
  target artifact route IDs or intrinsic spellings fail
  `verifyRVVSelectedBodyEmitCRouteDescription`.
- Target/export coverage proving stale route-id or arithmetic metadata that
  disagrees with the rebuilt selected-body description fails closed.
- Source-front-door or plugin-boundary coverage proving a valid typed body does
  not require `rvv_emitc_route_mapping` on `tcrv_rvv.with_vl`, and stale
  route metadata cannot authorize an empty or unsupported body.
- Residue scan of the RVV route provider must not show public
  route-id-to-operation symbolizers or per-operation route-builder entry
  points as production authority.

#### 7. Wrong vs Correct

Wrong:

```text
candidate route id -> parse add/sub/mul -> choose RVV intrinsic -> emit route
```

Correct:

```text
selected typed tcrv_rvv body/config/runtime ABI
  -> RVVSelectedBodyEmitCRouteDescription
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> route id and artifact metadata consumed only as mirrors
```

### Scenario: Bounded RHS Broadcast Memory-Form Selected-Body Route

#### 1. Scope / Trigger

Use this contract when a selected `origin = "rvv-plugin"` variant contains a
bounded explicit i32m1 arithmetic body where the RHS value is produced by
`tcrv_rvv.i32_broadcast_load` and consumed by `tcrv_rvv.i32_add`,
`tcrv_rvv.i32_sub`, or `tcrv_rvv.i32_mul`.

This is a selected-body memory form on the corrected RVV route surface. It is
not a route-id shortcut, not a descriptor source-export path, not a high-level
broadcast lowering, not broadcast compare/select support, and not a new
dtype/LMUL family.

#### 2. Signatures

- Runtime ABI values:
  `lhs`, `rhs`, `out`, and `n` in ordered callable ABI role order.
- Required control/config:
  one `tcrv_rvv.setvl` and one `tcrv_rvv.with_vl` with SEW 32, LMUL `m1`,
  and tail/mask agnostic policy.
- Required dataflow:
  exactly one `tcrv_rvv.i32_load` for lhs;
  exactly one `tcrv_rvv.i32_broadcast_load` for rhs;
  one typed arithmetic op among `tcrv_rvv.i32_add`,
  `tcrv_rvv.i32_sub`, or `tcrv_rvv.i32_mul`;
  one `tcrv_rvv.i32_store`.
- RHS broadcast source:
  the `tcrv_rvv.i32_broadcast_load` buffer operand must be the explicit
  `rhs-input-buffer` runtime ABI value.
- Provider memory form:
  `RVVSelectedBodyMemoryForm::RHSBroadcastLoad`, surfaced as a provider-derived
  metadata mirror such as `tcrv_rvv.memory_form = "rhs-broadcast-load"`.
- Current bounded intrinsic mapping:
  RHS broadcast may lower to the selected RVV vector-scalar/broadcast intrinsic
  only after typed body/config/runtime ABI validation succeeds.

#### 3. Contracts

- Broadcast semantics come from the typed `tcrv_rvv.i32_broadcast_load` result
  and its typed arithmetic consumer. The provider must not infer broadcast from
  route IDs, artifact names, ABI names, test names, descriptor residue,
  intrinsic spellings, or common EmitC/export code.
- The RHS broadcast load must feed the arithmetic RHS operand. The lhs operand
  must be the explicit lhs vector load result. Store must consume the
  arithmetic result.
- The selected-body route description must record the RHS broadcast memory
  form before construction metadata, emission-plan metadata, target artifact
  export, or generated bundle evidence consumes it.
- Common EmitC materialization may provide generic expression mechanics such
  as safe pointer subscript materialization, but it must not branch on RVV,
  `tcrv_rvv`, `i32_broadcast_load`, route IDs, intrinsic spellings, or
  operation names to choose broadcast behavior.
- Retained i32m1 route labels and exact `__riscv_*_i32m1` spellings are output
  labels for this bounded specialization only. They must not become a new
  broadcast route table or a template for dtype/LMUL/source-shape expansion.

#### 4. Validation & Error Matrix

- Missing lhs vector load, missing RHS broadcast load, multiple RHS broadcast
  loads, or mixed two RHS vector loads plus an extra broadcast load -> fail
  before route construction.
- RHS broadcast buffer is not the explicit `rhs-input-buffer` runtime ABI
  value -> verifier/provider failure before artifact construction.
- Arithmetic lhs/rhs operands do not consume the explicit lhs load and RHS
  broadcast results -> fail before route construction.
- Store does not consume the arithmetic result -> fail before artifact output.
- Compare/select mixed with RHS broadcast load -> fail in the current bounded
  route.
- Provider description lacks or misreports RHS broadcast memory form,
  runtime ABI order, config, or intrinsic mapping -> fail before emission-plan
  metadata or target artifact export is consumed.

#### 5. Good/Base/Bad Cases

- Good: selected RVV variant contains
  `runtime_abi_value(lhs/rhs/out/n) -> setvl -> with_vl -> i32_load(lhs) ->
  i32_broadcast_load(rhs) -> i32_add|i32_sub|i32_mul -> i32_store`, and the
  provider maps that body to a `TCRVEmitCLowerableRoute` whose artifact
  metadata mirrors `rhs-broadcast-load`.
- Base: vector-RHS add/sub/mul fixtures remain valid with
  `memory_form = "vector-rhs-load"`; compare/select remains vector-RHS-only in
  the current bounded slice.
- Bad: target/export code sees an add ABI name or artifact route and chooses
  `rhs[0]` broadcast behavior without the typed `tcrv_rvv.i32_broadcast_load`
  body.

#### 6. Tests Required

- Positive lit coverage for explicit RHS broadcast add, sub, and mul selected
  bodies reaching supported emission-plan metadata and target header export.
- Generated bundle ABI evidence proving selected variants, typed compute ops,
  `rhs-broadcast-load` metadata, ordered `lhs,rhs,out,n` ABI parameters,
  runtime AVL metadata, and object/header coherence.
- Real `ssh rvv` correctness evidence is required before making executable
  correctness claims.
- Negative lit/provider coverage proving compare/select broadcast and wrong or
  missing RHS broadcast binding fail before artifact construction.
- Focused scans must show no descriptor/direct-C/source-export restoration, no
  source-front-door default authority, no old `RVVI32M1*` route-table
  authority, and no common EmitC RVV semantic branch.

#### 7. Wrong vs Correct

Wrong:

```text
route id / artifact name / ABI name -> infer RHS broadcast -> emit intrinsic
```

Correct:

```text
selected typed tcrv_rvv body with explicit i32_broadcast_load(rhs)
  -> RVV provider validates RHS binding, memory form, config, and ABI roles
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization and target export
  -> generated bundle / ssh rvv evidence consumes metadata as mirrors
```

### Scenario: Bounded I32 Compare/Select Selected-Body Route

#### 1. Scope / Trigger

Use this contract when a selected `origin = "rvv-plugin"` variant contains a
bounded explicit `tcrv_rvv.i32_cmp_eq` plus `tcrv_rvv.i32_select` body that is
route-supported by the current i32m1 selected-body EmitC provider.

This is a typed selected-body route, not a high-level predicate lowering, not a
broadcast compare route, and not a new dtype/LMUL family.

#### 2. Signatures

- Runtime ABI values:
  `lhs`, `rhs`, `out`, and `n` in ordered callable ABI role order.
- Required control/config:
  one `tcrv_rvv.setvl` and one `tcrv_rvv.with_vl` with SEW 32, LMUL `m1`,
  and tail/mask agnostic policy.
- Required dataflow:
  exactly two `tcrv_rvv.i32_load` ops for explicit lhs/rhs vector values;
  one `tcrv_rvv.i32_cmp_eq`;
  one `tcrv_rvv.i32_select`;
  one `tcrv_rvv.i32_store`.
- Compare operands:
  each operand must be one of the explicit lhs/rhs vector load results.
- Select operands:
  mask must be the result of the same-body `tcrv_rvv.i32_cmp_eq`; true and
  false data operands must each be one of the explicit lhs/rhs vector load
  results; store must consume the select result.
- Current route labels:
  operation mnemonic `cmp_select`, typed compute op `tcrv_rvv.i32_select`,
  EmitC route `rvv-i32m1-cmp-select-emitc-route`, runtime ABI
  `rvv-i32m1-cmp-select-callable-c-abi.v1`.

#### 3. Contracts

- The RVV provider must map compare operands and select true/false operands
  from the selected typed body SSA values. It must not hard-code a canonical
  `compare(lhs, rhs)` plus `select(lhs, rhs)` body when the explicit typed
  body chose another supported lhs/rhs-loaded operand combination.
- Route IDs, ABI names, artifact names, intrinsic spellings, test names, and
  common EmitC/export code are output labels only. They must not select
  compare/select behavior.
- The current bounded route may lower the selected body to RVV compare and
  merge intrinsics only after the body has already proved mask/dataflow shape,
  config, memory form, and runtime ABI role order.
- Common EmitC materialization consumes the provider-built route payload only;
  it must not branch on `cmp_select`, `tcrv_rvv.i32_select`, mask type names,
  or RVV intrinsic names to recover semantics.

#### 4. Validation & Error Matrix

- Missing `tcrv_rvv.i32_cmp_eq` or missing `tcrv_rvv.i32_select` -> fail
  before route construction.
- Compare/select mixed with RHS broadcast load -> fail in the current bounded
  route.
- Compare operands are not explicit lhs/rhs vector load results -> fail before
  route construction.
- Select mask is not produced by the same-body `tcrv_rvv.i32_cmp_eq` -> fail
  before route construction.
- Select true/false operands are not explicit lhs/rhs vector load results ->
  fail before route construction.
- Store does not consume the select result -> fail before artifact output.
- Stale route ID, runtime ABI name, target artifact route, or intrinsic
  spelling that disagrees with the provider-derived description -> fail before
  emission-plan metadata or target artifact export is consumed.

#### 5. Good/Base/Bad Cases

- Good: compare/select body compares two explicit lhs/rhs-loaded vector values,
  selects between explicit lhs/rhs-loaded vector values with that typed mask,
  stores the select result, and the provider maps the actual SSA operands into
  the compare and merge call operands.
- Base: an all-true mask fixture such as compare `lhs` with `lhs` may be used
  to make the selected true branch observable in external ABI evidence, while
  still deriving route semantics from typed RVV body operands.
- Bad: provider/common code assumes the route label `cmp_select` means
  compare `lhs,rhs` and select `lhs,rhs` without reading the typed
  `tcrv_rvv.i32_cmp_eq` / `tcrv_rvv.i32_select` operands.

#### 6. Tests Required

- Positive target fixture coverage for explicit selected-body compare/select
  reaching supported emission-plan metadata and declaration header export.
- Generated bundle ABI evidence proving selected variant, ordered
  `lhs,rhs,out,n` ABI parameters, runtime AVL metadata, typed compute metadata
  `tcrv_rvv.i32_select`, and route metadata `cmp_select`.
- Real `ssh rvv` compile/link/run evidence for runtime correctness when an
  executable compare/select claim is made.
- Negative dialect/provider coverage proving malformed mask/dataflow fails
  before target artifact construction.
- Focused residue scans showing no descriptor/direct-C/source-export
  restoration and no common EmitC RVV semantic branch.

#### 7. Wrong vs Correct

Wrong:

```text
route label cmp_select -> provider hard-codes compare(lhs,rhs), select(lhs,rhs)
```

Correct:

```text
selected typed i32_cmp_eq / i32_select SSA operands
  -> RVV provider validates bounded lhs/rhs-loaded mask/dataflow shape
  -> provider maps actual compare/select operands to RVV compare/merge payload
  -> common EmitC materializes the validated route neutrally
```

Route support has three distinct levels:

```text
parseable / verifier-legal:
  The dialect accepts the IR. This is not a lowering, artifact, runtime, or
  performance promise.

route-supported:
  The RVV plugin declares legality and a lowering route for the selected op,
  body pattern, or region. Valid instances lower through that route; invalid or
  unsupported combinations fail closed with targeted diagnostics.

executable:
  The route-supported body is inside a selected tcrv.exec envelope with complete
  ABI/runtime binding, materialization, target artifact/export support, and
  runtime evidence when runtime/correctness/performance is claimed.
```

Stage 2 is RVV completion work on this corrected vector-level surface. It
includes both route-supported RVV coverage expansion and RVV plugin-local
selected-body realization for performance-sensitive vector-level bodies.
Implementation order may follow dependencies, but early executable subsets are
plumbing proofs only and must not be treated as RVV maturity.

Stage 2 must not start by adding more cases to the legacy `i32m1` route
architecture. Stage 1 is not complete while the active route provider is still
organized around finite `RVVI32M1*` route specs, slices, exact `i32_*` op
families, route ids, or hard-coded `__riscv_*_i32m1` intrinsic spellings as the
family architecture. If those structures remain, the next RVV owner is route
surface correction: convert the legacy i32 slice into an ordinary instance of a
typed RVV value/config/body route surface, or fail-close/delete it when it does
not fit. Adding broadcast, compare/select, reduction, dtype, or LMUL coverage
to the old table is compatibility debt, not Stage 2 progress.

Stage 2 is complete only when the route-supported `tcrv_rvv` surface can cover
the math and data-movement classes represented by structured kernels such as
Linalg, without making Linalg the current input contract. The completeness
target is not "all arbitrary high-level `linalg.generic` regions" and not a
new high-level frontend. It is the low-level RVV execution coverage needed for
future semantic-preserving frontend lowering:

```text
elementwise and broadcast/vector-scalar maps
masked and tail-safe contiguous/strided/indexed memory movement when supported
compare/select and FMA/update-style arithmetic
reduction, accumulator, and contraction-like multiply-accumulate bodies
movement/layout forms such as slide/gather/scatter/compress when supported
conversion, widening/narrowing, dtype, SEW, LMUL, and policy legality
dynamic AVL/VL, mask/tail behavior, runtime ABI value use, and boundary control
RVV plugin-local realization of VL/setvl, legal vtype, unroll, software
pipeline, prefetch, and accumulator layout when those structures are modeled
or supported
```

The abstraction level is intentionally Vector-like: higher than a list of
`riscv_vector.h` intrinsic calls and lower than Linalg/tensor/kernel semantics.
Stage 2 must not be implemented as per-Linalg-op lowerers, high-level kernel
ops, one-op-per-intrinsic wrappers, dtype/LMUL clone batches, a global
autotuning database, a dashboard, or a readiness state machine.

## Deprecated Legacy RVV i32 Broadcast-Load Route Note

The current repository contains `tcrv_rvv.i32_broadcast_load` and legacy
route-table support for RHS broadcast under the bounded `!tcrv_rvv.i32m1`
path. That is current implementation debt, not a Stage 2 owner template.

Do not add new broadcast, dtype, LMUL, source-shape, intrinsic-wrapper, target
artifact, or test-matrix work to the old `RVVI32M1*` route table. Valid work
touching this area is limited to:

- deleting the legacy broadcast branch;
- fail-closing it when selected typed route authority is missing;
- replacing it as an ordinary memory/broadcast form on the corrected
  vector-level `tcrv_rvv` value/config/body surface;
- updating tests so they stop protecting old route-table expansion.

This note intentionally does not provide a 7-part scenario template. Treating
the old broadcast path as a reusable scenario was the misleading part: it made
legacy compatibility look like forward RVV coverage.

### Wrong vs Correct

Wrong:

```text
route id or ABI name says broadcast -> common export chooses broadcast code
```

Correct:

```text
selected vector-level tcrv_rvv body expresses a memory/broadcast form with
  dtype, config, policy, runtime ABI value use, and operation semantics carried
  structurally by the body/config surface
  -> RVV provider validates the RHS binding, memory form, and route legality
  -> legacy `tcrv_rvv.i32_broadcast_load` is deleted/fail-closed, or retained
     only after it has become an ordinary specialization of the corrected
     vector-level RVV route surface
  -> common EmitC/target export consumes validated route payload only
```

## Legacy Narrow C++ Slice

The existing bounded C++ RVV slice is a legacy narrow path from earlier
implementation work. It proves plugin identity, capability participation, typed
RVV dialect registration, explicit typed-variant legality routing, and
selection preference through `ExtensionPluginRegistry`, but it is not the RVV
maturity architecture and must not be cloned into dtype/LMUL/source batches.

The deleted metadata-only proposal route is not active compiler authority:
bare high-level/no-body RVV capability evidence must not produce an RVV
proposal, selected variant, lowering boundary, emission plan, runtime ABI,
artifact route, or metadata-only artifact authority. A selected RVV variant is legal only when it
already contains explicit typed `tcrv_rvv` extension-family IR. That typed IR
becomes executable only through the bounded selected EmitC artifact bridge
described below; typed IR alone is not runtime or performance evidence.

Stable current names:

```text
plugin name: rvv-plugin
plugin version: 0.1.0
plugin capability id: rvv
plugin capability kind: isa-vector
preferred kernel capability symbol: @rvv
variant origin: rvv-plugin
registered plugin capability ids:
  rvv
explicit typed variant form: a hand-authored or future materialized RVV variant
  must contain real `tcrv_rvv` ops in its body before RVV plugin legality or
  selection preference can accept it
typed policy attr name: tcrv_rvv.policy
typed policy attr value: #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
optional vlenb attr name: tcrv_rvv.vlenb_bytes
```

If the bounded explicit RVV i32m1 route table is retained during migration, it
is only a legacy narrow route and may export a relocatable object only through
the selected emission-plan target artifact handoff. Current code evidence still
centers this route on `RVVI32M1ArithmeticRouteSpec`,
`RVVI32M1ArithmeticSlice`, optional `i32_broadcast_load`, and a
`cmp_select` case, so it remains Stage 1 route-authority debt:

```text
selected RVV path
  -> finite `tcrv_rvv.i32_*` legacy route-table body
  -> validated tcrv_rvv.with_vl selected lowering boundary
  -> RVV-owned EmitC lowerable route
  -> MLIR EmitC C/C++ emitter
  -> clang RISC-V relocatable object
```

This bounded route does not authorize descriptor-driven computation, deleted
microkernel wrappers, new dtype/LMUL families, a generic RVV source printer, or
`i32m1` arithmetic as the route architecture.
It also does not let target/export code select an RVV artifact by assuming a
module contains exactly one direct variant; the selected emission-plan
candidate is the handoff authority. The rebuilt header route is declaration
only and derives its function name and ordered callable ABI parameters from the
same selected materialized EmitC candidate as the object route. The rebuilt
bundle route packages the object and header as one selected-variant component
group. Historical header and bundle route ids remain deleted and must not be
used as compatibility aliases.

The bounded RVV i32m1 source materializer is a legacy source-front-door
compatibility path, not RVV maturity work. If retained, it may only construct a
selected `tcrv.exec` envelope with an explicit typed `tcrv_rvv` body from the
exact source pattern owned by the RVV plugin. It must not be widened into
broadcast, compare/select, dtype, LMUL, or source-shape families. The produced
`tcrv_rvv` body is the only downstream compiler authority. Stale
`tcrv_rvv.lowering_seed` metadata, route ids, descriptors, artifact names,
deleted finite-family records, and the source pattern itself must not authorize
emission planning or target export after Stage 1. Unsupported source shapes and
stale pre-existing `tcrv.exec`/`tcrv_rvv` residue must fail before emission
planning or target artifact export.

If this legacy source path is retained, the accepted runtime-count source
pattern must make tail behavior explicit. For the bounded i32m1 arithmetic
slice, the source loop body computes
`remaining = n - iv`, creates `vector.create_mask remaining : vector<4xi1>`,
uses that same mask for both `vector.transfer_read` inputs and the
`vector.transfer_write` output, and keeps the loop upper bound equal to the
runtime ABI `n` operand. The old fixed-width `vector.load` / `vector.store`
step-4 shape is not valid authority for arbitrary runtime `n` unless a future
source contract explicitly proves a multiple-of-width precondition. The current
production source-artifact path must reject that old shape.

RVV probe facts remain bounded hardware/toolchain evidence inputs. They may be
validated into raw `TargetCapabilitySet` evidence facts, but they must not
manufacture finite SEW/LMUL/tail/mask config capabilities or authorize a
compiler route unless an explicit typed RVV variant body already exists.
Proposal collection for a no-body RVV-capable kernel records a recoverable
decline and produces no RVV proposal. Materialized RVV variant legality is
strict: the variant must be owned by `origin = "rvv-plugin"` and contain real
`tcrv_rvv` operations in its body. Metadata-only RVV attributes, finite
capability facts, deleted selected-shape facts, or profile evidence alone are
not sufficient.

The RVV plugin must not create any binary-family proposal from a no-body
`tcrv.exec.kernel`, from deleted frontend metadata, from finite-family registry
metadata, or from hand-authored microkernel names alone. Kernel-based
executable planning requires a future explicit extension-family op contract
and a materialized EmitC route before selecting family, dtype, artifact route,
callable ABI, artifact kind, or emitted body.

## Scenario: Legacy Tail-Safe RVV Vector Source Front Door

### 1. Scope / Trigger

This applies only while the RVV plugin keeps the legacy source front door as an
explicit, non-default materialization seed that recognizes source MLIR and
materializes the bounded i32m1 add/sub/mul selected boundary for runtime counts
provided by the callable ABI parameter `n`. This front door is not a maturity
unit, not a template for future source recognizers, and not route authority
after the selected typed `tcrv_rvv` body exists. Default source-artifact
front-door pipelines must not run this RVV source recognizer as an artifact
route authority.

### 2. Signatures

- Source function ABI: `(%lhs: memref<?xi32>, %rhs: memref<?xi32>,
  %out: memref<?xi32>, %n: index) -> ()`.
- Source loop: `scf.for %i = %c0 to %n step %c4`.
- Tail source ops in order:
  `arith.subi %n, %i : index`, `vector.create_mask`, two masked
  `vector.transfer_read` ops, one `arith.addi|arith.subi|arith.muli` on
  `vector<4xi32>`, and one masked `vector.transfer_write`.

### 3. Contracts

- `%n` is the runtime element-count ABI parameter and must be the loop upper
  bound.
- The tail mask must be derived from `n - iv` and have type `vector<4xi1>`.
- Both source input transfers and the output transfer must consume the same
  tail mask, use minor-identity transfer maps, avoid `in_bounds = true`, and
  operate on `vector<4xi32>`.
- The source materializer may then construct the existing selected boundary:
  `runtime_abi_value(lhs,rhs,out,n) -> setvl(n) -> with_vl ->
  i32_load/i32_load -> i32_add|i32_sub|i32_mul -> i32_store`.
- The source front door must remain RVV plugin-owned. Common source-artifact
  orchestration must only run registered plugin passes and target exporters.

### 4. Validation & Error Matrix

- Missing or reordered ABI operands -> reject before materialization.
- Loop upper bound is not `%n`, lower bound is not zero, or step is not four ->
  reject before materialization.
- Loop body lacks `remaining = n - iv`, lacks `vector.create_mask`, omits a
  transfer mask, uses a different mask, or uses `vector.load`/`vector.store` ->
  reject before selected boundary materialization.
- Stale `tcrv_rvv.lowering_seed`, pre-existing `tcrv.exec` residue, or
  pre-existing `tcrv_rvv` residue -> reject before emission planning.

### 5. Good/Base/Bad Cases

- Good: explicit invocation of the RVV source materializer on masked
  `vector.transfer_read`/`vector.transfer_write` source with `n - iv` tail mask
  materializes the existing selected typed `tcrv_rvv` boundary. The resulting
  already materialized selected-body IR may then flow through selected-body
  route construction and target artifact export.
- Good: default source-artifact front doors reject source-only RVV input before
  object/header/bundle export.
- Base: materialized hand-authored `tcrv_rvv` selected-boundary fixtures remain
  valid for target/export tests when they already contain explicit typed RVV IR.
- Bad: fixed step-4 `vector.load`/`vector.store` source is accepted and then
  used to claim correctness for runtime counts such as 7 or 23.
- Bad: a default source-artifact pipeline uses source pattern matching,
  `source-pattern-selected-rvv-case`, lowering-seed metadata, route ids, or
  artifact names to produce RVV target artifacts from source-only input.

### 6. Tests Required

- Positive lit coverage for add/sub/mul tail-safe source fixtures through
  explicit RVV source materializer invocation and selected-body-derived
  emission-plan metadata.
- Positive target artifact coverage for object/header/bundle export from
  already materialized selected typed `tcrv_rvv` IR.
- Negative lit coverage proving default source-artifact front-door pipelines
  reject source-only RVV inputs before target artifact export.
- Negative lit coverage for the old fixed `vector.load`/`vector.store` source
  shape and for stale metadata/residue, ABI order mismatch, loop bound
  mismatch, missing mask, and unsupported arithmetic.
- Runtime claims from the accepted source route require real `ssh rvv` evidence
  with at least one non-multiple-of-four count and one multiple-of-four count.

### 7. Wrong vs Correct

Wrong:

```mlir
scf.for %i = %c0 to %n step %c4 {
  %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
  %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
  %sum = arith.addi %a, %b : vector<4xi32>
  vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
}
```

Correct:

```mlir
scf.for %i = %c0 to %n step %c4 {
  %remaining = arith.subi %n, %i : index
  %mask = vector.create_mask %remaining : vector<4xi1>
  %a = vector.transfer_read %lhs[%i], %pad, %mask
      : memref<?xi32>, vector<4xi32>
  %b = vector.transfer_read %rhs[%i], %pad, %mask
      : memref<?xi32>, vector<4xi32>
  %sum = arith.addi %a, %b : vector<4xi32>
  vector.transfer_write %sum, %out[%i], %mask
      : vector<4xi32>, memref<?xi32>
}
```

## Scenario: Legacy Bounded RVV i32m1 Config Policy Slice

### 1. Scope / Trigger

This scenario applies only while the RVV plugin retains the bounded i32m1
compatibility route table. The current table includes add/sub/mul plus
route-table extensions such as broadcast-load and compare/select; that wider
table is still a legacy narrow cross-layer contract, not the RVV maturity unit:
target capability profile facts, variant
`requires`, selected lowering-boundary validation, and emission-plan
validation must agree on the same SEW/LMUL/tail/mask policy ids. Remote probe
output does not create those compiler config facts.

### 2. Signatures

- C++ capability ids: `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic`.
- Preferred MLIR symbols from explicit target/profile fixtures:
  `@rvv_i32_m1_sew32`, `@rvv_i32_m1_lmul_m1`,
  `@rvv_i32_m1_tail_agnostic`, and `@rvv_i32_m1_mask_agnostic`.
- Variant metadata: `requires` must satisfy `rvv` plus all four legacy
  compatibility config/policy ids; `tcrv_rvv.policy` must be
  `#tcrv_rvv.policy<tail = agnostic, mask = agnostic>`.

### 3. Contracts

- `rvv.i32_m1.sew32` requires `sew_bits = 32 : i64`.
- `rvv.i32_m1.lmul_m1` requires `lmul = "m1"`.
- `rvv.i32_m1.tail_policy.agnostic` requires
  `tail_policy = "agnostic"`.
- `rvv.i32_m1.mask_policy.agnostic` requires
  `mask_policy = "agnostic"`.
- Runtime `n`, AVL, VL, and dispatch guard values remain runtime SSA/control
  or ABI values. They must not be encoded as these compile-time config
  capability facts. Deleted local RVV element-count metadata must not be
  reintroduced as a runtime trip count or artifact descriptor.
- If retained, the selected i32m1 arithmetic EmitC route must consume one
  RVV-owned config/VL contract shared with RVV dialect verification. That
  contract validates `tcrv_rvv.setvl` and `tcrv_rvv.with_vl` as the same
  compile-time SEW32, LMUL m1, tail agnostic, mask agnostic config; verifies
  that `with_vl` consumes the visible `setvl` VL result; and keeps AVL/VL as
  runtime SSA values. Dialect-level bounded dataflow may parse non-executable
  sibling config such as i32m2 when explicitly modeled, but the selected
  i32m1 arithmetic artifact route must fail closed before route payload
  construction if the selected body is not the exact i32m1 config.
- Target artifact export may consume the selected emission-plan route and
  runtime ABI metadata, but it must not re-derive RVV SEW/LMUL/policy/VL
  semantics in common target code.

### 4. Validation & Error Matrix

- Missing `rvv` provider -> plugin decline during proposal collection.
- Bare RVV capability evidence without an explicit typed RVV variant body ->
  no RVV proposal and no selected RVV route.
- Explicit selected RVV variant without `origin = "rvv-plugin"` -> fatal
  legality error.
- Explicit selected RVV variant without `tcrv_rvv` ops in its body -> fatal
  legality error.
- Selected i32m1 arithmetic route whose `setvl` / `with_vl` compile-time
  config does not match SEW32, LMUL m1, tail agnostic, mask agnostic -> fail
  before EmitC call payload construction.
- Selected i32m1 arithmetic route whose `with_vl` does not consume the visible
  `setvl` VL SSA result -> fail before EmitC call payload construction.
- Selected route id whose expected legacy op does not match the typed operation
  sequence in the selected body -> fail before target artifact export.
- Capability/profile facts may still be validated for replay and typed-body
  checks, but they must not create proposal, boundary, emission-plan, runtime
  ABI, artifact, or metadata-only route authority by themselves.

### 5. Good/Base/Bad Cases

- Good: a hand-authored or future materialized variant contains real
  `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` control-plane IR in the variant body
  and is owned by `origin = "rvv-plugin"`.
- Base: probe-derived capability facts are replayed as bounded capabilities
  but produce no RVV variant unless explicit typed RVV IR is present.
- Bad: `@rvv` is available but the kernel has no explicit typed RVV body; this
  must remain no-proposal/no-boundary/no-emission.

### 6. Tests Required

- RVV plugin unit/lit tests must assert no-body RVV capability input produces
  no RVV proposal.
- Legality tests must reject stale metadata-only RVV variants that do not
  contain explicit typed `tcrv_rvv` ops.
- Dialect and EmitC route tests must cover the i32m1 config/VL contract:
  missing or mismatched `with_vl` config, non-agnostic policy, unsupported
  LMUL for i32m1 artifact export, `with_vl` consuming a non-`setvl` VL token,
  and stale selected route/op combinations.
- Lowering-boundary, emission-plan, manifest, target export, and probe
  replay tests must not require RVV boundary, runtime ABI, artifact, or route
  metadata from bare capability evidence.
- Probe and probe-to-MLIR self-tests may assert replayed capability facts are
  preserved, but legality remains in C++ and no compiler route is synthesized.

### 7. Wrong vs Correct

Wrong:

```mlir
tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector",
                           status = "available"}
tcrv.exec.variant @rvv_stale_metadata attributes {
  origin = "rvv-plugin",
  requires = [@rvv],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {}
```

Correct for a relation-provider fixture:

```mlir
tcrv.exec.capability @rvv {
  id = "rvv",
  kind = "isa-vector",
  provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1",
              "rvv.i32_m1.tail_policy.agnostic",
              "rvv.i32_m1.mask_policy.agnostic"],
  sew_bits = 32 : i64,
  lmul = "m1",
  tail_policy = "agnostic",
  mask_policy = "agnostic",
  status = "available"
}
tcrv.exec.variant @rvv_explicit_body attributes {
  origin = "rvv-plugin",
  requires = [@rvv],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {
  %n = "builtin.unrealized_conversion_cast"() : () -> index
  %vl = tcrv_rvv.setvl %n {
    lmul = "m1",
    policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
    sew = 32 : i64
  } : index -> !tcrv_rvv.vl
  tcrv_rvv.with_vl %vl attributes {
    lmul = "m1",
    policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
    sew = 32 : i64
  } {
  } : !tcrv_rvv.vl
}
```

Correct for a replay/profile fixture:

```mlir
tcrv.exec.variant @rvv_explicit_profile_body attributes {
  origin = "rvv-plugin",
  requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1,
              @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {
  %n = "builtin.unrealized_conversion_cast"() : () -> index
  %vl = tcrv_rvv.setvl %n {
    lmul = "m1",
    policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
    sew = 32 : i64
  } : index -> !tcrv_rvv.vl
  tcrv_rvv.with_vl %vl attributes {
    lmul = "m1",
    policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
    sew = 32 : i64
  } {
  } : !tcrv_rvv.vl
}
```

The former plugin-local standalone smoke-probe source route is deleted.
Availability of plugin-local smoke-probe capability facts must not select a
compiler generated-source path, materialize an executable RVV microkernel,
create runtime callable ABI parameters, claim kernel lowering, or produce
correctness/performance evidence. RVV hardware/toolchain probe evidence belongs
in separate probe tooling and recorded `ssh rvv` artifacts.

When a relation-provider capability satisfies id `rvv`, generic variant
materialization records the provider symbol in `requires`, for example
`requires = [@rvv_profile]`. RVV legality must treat that provider as satisfying
the RVV id through the C++ `CapabilityDescriptor::satisfiesID("rvv")` relation
API; it must not require an exact `id = "rvv"` capability when a structured
provider relation is present.

The former finite descriptor selected microkernel materialization route is
deleted. A selected `rvv-plugin` path must not materialize plugin-owned
selected-boundary metadata or synthesize a plugin-local direct kernel-child
`tcrv_rvv.*_microkernel`, `setvl`/`with_vl`,
load/arithmetic/store body, or callable runtime ABI boundary from finite family
records, legacy route helpers, lowering tokens, route ids, or descriptor
mirrors. Until the rebuild supplies explicit extension-family IR plus a
materialized MLIR EmitC module route, deleted RVV wrapper attachments and
descriptor-derived callable ABI data are fail-closed historical inputs, not
active emission authority.

The former microkernel direct C slice and finite RVV target object/header route
slice are deleted as production authority. The only active bounded RVV target
artifact route is the materialized EmitC i32m1 arithmetic route family: object
export compiles the MLIR EmitC C/C++ emitter output, header export emits only a
callable declaration for that materialized function boundary, and bundle export
packages the object and header under one selected-variant ABI component group.
Target/export code must not synthesize RVV compute C bodies from selected
metadata, family records, route records, or deleted wrapper records. RVV routes
must fail closed instead of producing `riscv_vector.h` intrinsic source,
relocatable objects, headers, bundles, or self-check harnesses from the old
direct printer path.

## Remote Evidence Probe Contract

The repo-owned RVV evidence probe is `scripts/rvv_remote_probe.py`. It is
Python runner/evidence tooling and must not be used as the implementation of
capabilities, plugin registry behavior, legality, lowering, emission, runtime
ABI, or compiler-generated executable paths.

Probe artifacts live under:

```text
artifacts/tmp/rvv_probe/<run-id>/
  rvv_probe_evidence.json
  logs/*.log
```

The JSON artifact records a bounded schema:

- `schema_version`, `probe_name`, `run_id`, timestamp, ssh target, artifact dir,
  status, and success boolean;
- hardware/toolchain facts for `uname`, architecture, hart count, clang,
  cmake, bounded RISC-V/vector hints from `/proc/cpuinfo`, and non-interactive
  sudo availability as a boolean capability fact;
- a sanitized `capability_facts` section containing only bounded evidence
  facts: architecture, hart count, optional raw VLENB bytes,
  ISA/vector hint string, clang and CMake
  availability/version facts, minimal RVV compile/run success, selected
  march/mabi, and optional source/binary digests;
- a minimal RVV intrinsic compile/run probe result with command references,
  exit status, diagnostics, compiler path/version, source digest, selected
  compiler flags, and binary digest when a binary was produced;
- sanitized command logs that avoid secrets, credentials, private keys,
  environment tokens, and unrelated raw environment dumps.

Interpretation rules:

- Successful `ssh rvv` probe output is hardware/toolchain/probe-program
  evidence only.
- The transition from probe evidence to compiler capabilities is
  `sanitized capability_facts -> plugin-local C++ RVV capability profile ->
  TargetCapabilitySet`. The Python probe may emit artifacts, but it must not
  implement capability relations, legality, selection, lowering, or emission.
- The RVV C++ capability profile must validate facts before producing any
  `TargetCapabilitySet`. Required gates include `riscv64`, positive hart count,
  RVV ISA/vector hints, clang and CMake availability, and minimal RVV
  compile/run success. Negative cases must return structured diagnostics rather
  than partial target capabilities.
- Profile-derived capability identities are stable and plugin-local raw
  evidence identities. Current probe-derived profile IDs include `rvv`,
  `rvv.hart_count`, `rvv.vlenb_bytes`,
  `rvv.toolchain.clang`, `rvv.toolchain.cmake`, `rvv.probe.compile_run`,
  `rvv.toolchain.march`, and `rvv.toolchain.mabi`. SEW/LMUL/tail/mask config
  IDs such as `rvv.i32_m1.sew32` belong to explicit RVV config/profile fixtures
  or plugin-selected variant requirements, not to
  probe-derived profile construction. These identities must not include
  ssh/provider names, raw command logs, secrets, benchmark names, or
  performance measurements.
- The probe does not prove that TianChen-RV generated RVV IR, lowered a
  `tcrv.exec` variant, emitted an object, linked runtime glue, proved compiler
  correctness, or measured performance.
- If clang, RVV headers, candidate flags, or remote execution are unavailable,
  the artifact must record failure with exact non-secret command diagnostics
  rather than synthesizing success.
- Future RVV emission beyond the finite explicit legacy i32m1 route-table cases
  requires plugin-local lowering/runtime implementation and successful named
  `ssh rvv` evidence; probe artifacts alone do not create broader supported RVV
  emission.

## Deleted Remote Evidence Replay Route

The Python RVV probe-to-MLIR replay route is deleted as active compiler
authority. Repo-owned Python probe tooling may collect, sanitize, validate, and
record bounded RVV hardware/toolchain evidence, but it must not translate probe
JSON into `tcrv.exec` MLIR capability, target, kernel, variant, selected route,
or scalar fallback modeling.

Compiler-visible RVV capability/profile behavior must be implemented by the
C++/MLIR RVV extension-family/plugin path. The transition from a probe artifact
to compiler capabilities is:

```text
sanitized evidence artifact
  -> plugin-local C++ RVV capability profile validation
  -> TargetCapabilitySet
```

There is no supported Python fallback route, hidden compatibility mode, or
schema adapter that preserves the old replay helper. Tests must not pipe
Python-generated RVV replay MLIR into `tcrv-opt` as compiler-path evidence.
First-slice SEW/LMUL/tail/mask policy facts are plugin-selected compiler config
facts, not remote-probe output fields.

`registerDialects` now registers the minimal RVV dialect skeleton through the
RVV plugin path. The default `registerAllDialects` path remains core-only; RVV
dialect availability is proven by populating an `ExtensionPluginRegistry` with
the RVV plugin and calling `registerPluginDialects`.

The current RVV dialect slice is metadata/control-plane plus bounded explicit
dataflow only; the former executable microkernel wrapper ops are deleted. It
introduces the vector-length token type `!tcrv_rvv.vl`, the finite policy
attribute
`#tcrv_rvv.policy<tail = agnostic|undisturbed, mask =
agnostic|undisturbed>`, the bounded `!tcrv_rvv.runtime_abi_value` token and
`tcrv_rvv.runtime_abi_value` ABI binding op, the bounded runtime AVL-to-VL
control-plane operation `tcrv_rvv.setvl`, the bounded VL scope region operation
`tcrv_rvv.with_vl`, and finite legacy i32m1 dataflow ops such as
`tcrv_rvv.i32_load`, `tcrv_rvv.i32_broadcast_load`, `tcrv_rvv.i32_add`,
`tcrv_rvv.i32_sub`, `tcrv_rvv.i32_mul`, `tcrv_rvv.i32_cmp_eq`,
`tcrv_rvv.i32_select`, and `tcrv_rvv.i32_store` nested under that scope. The
previous plugin-local selected lowering-boundary operation is deleted as active
compiler authority. The `runtime_abi_value` op binds one callable C ABI value by
role, C name, C type, and ownership, and produces an SSA value consumed by the
legacy bounded RVV IR. The setvl op consumes a runtime AVL SSA value, produces a
`!tcrv_rvv.vl` token, and carries only bounded compatibility-slice compile-time
config metadata: SEW 32, LMUL m1 or m2, and the finite policy attribute. The
with_vl op consumes one `!tcrv_rvv.vl` value and owns one single-block region
for bounded RVV control/body work. Optional duplicated SEW/LMUL/policy metadata
is limited to the same bounded compatibility config and must agree with the
visible defining setvl when present. The bounded legacy route-table body
consumes explicit `lhs`, `rhs`, `out`, and predicate/dataflow SSA values from
typed RVV ops; concrete C parameter names are plugin-owned route inputs from the
defining runtime ABI value ops, not descriptor fields or target-side inference.
It is not a generic vector memory or compute model. These surfaces are not
generic RVV arithmetic, generic memory operations, LLVM/RISC-V lowering, full
runtime ABI glue, hardware execution, correctness evidence, or performance
evidence. `tcrv_rvv`
is the concrete MLIR dialect namespace because MLIR dialect namespaces cannot
contain `.` characters; the architectural extension family remains `tcrv.rvv`.

For the bounded i32m1 arithmetic route, runtime `n` may drive repeated RVV
chunks only through the RVV extension-family route and the common materialized
EmitC path. A supported multi-VL route must materialize a real `emitc.for`
boundary with an induction value, compute remaining AVL from `n - offset`,
derive the per-iteration VL from `tcrv_rvv.setvl` ownership, advance lhs/rhs/out
pointers or indices by the induction value, and place load/compute/store calls
inside that structured loop. Artifact metadata such as `multi_vl`, loop
induction, step, remaining-AVL, or pointer-advance keys is evidence and
preflight input only; it must match the materialized EmitC loop and must not
synthesize RVV compute bodies, intrinsic calls, loop source text, or route
semantics in target/export code. Headers, objects, and bundles must fail closed
when they claim multi-VL support without that materialized loop.

## Capability Fields

RVV plugin should register the base plugin capability and may query explicit
profile or typed-variant config facts without advertising them from a target
helper catalog:

```text
rvv
rvv.version
rvv.i32_m1.sew32
rvv.i32_m1.lmul_m1
rvv.i32_m1.tail_policy.agnostic
rvv.i32_m1.mask_policy.agnostic
rvv.i32_m2.sew32
rvv.i32_m2.lmul_m2
rvv.i32_m2.tail_policy.agnostic
rvv.i32_m2.mask_policy.agnostic
vlen
elen
supported SEW
supported LMUL
mask support
tail policy support
Zvfh / Zvfbfmin / other vector dtype extensions
LLVM scalable vector support
RVV intrinsic support
inline asm policy
thread runtime availability
```

Reference attribute:

```mlir
#tcrv.ext<"rvv",
          kind = "isa-vector",
          version = "1.0",
          vlen = 128,
          supports_mask = true,
          supports_tail_policy = true>
```

## Current Dialect Skeleton

Architectural family:

```text
tcrv.rvv
```

Concrete MLIR namespace:

```text
tcrv_rvv
```

Legacy bounded compatibility type:

```text
!tcrv_rvv.vl
!tcrv_rvv.i32m1
!tcrv_rvv.i32m2
```

These legacy bounded compatibility types are not the mature RVV route surface.
They may remain only as migration scaffolding or ordinary specializations after
the corrected typed value/config/body route architecture exists. They must not
be the center of new Stage 2 coverage, dtype/LMUL expansion, performance
realization, or frontend work.

Legacy bounded compatibility policy attribute:

```text
#tcrv_rvv.policy<tail = agnostic, mask = agnostic>
```

Legacy bounded runtime VL control-plane op:

```mlir
%vl = tcrv_rvv.setvl %avl {
  lmul = "m1",  // or "m2" for the finite i32m2 slice
  policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
  sew = 32 : i64
} : index -> !tcrv_rvv.vl
```

Legacy bounded VL scope control-plane op:

```mlir
tcrv_rvv.with_vl %vl attributes {
  lmul = "m1",  // or "m2" for the finite i32m2 slice
  policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
  sew = 32 : i64,
  source_kernel = "example_kernel",
  selected_variant = @example_rvv_variant,
  origin = "rvv-plugin",
  selected_path_role = "direct variant",
  status = "selected-lowering-boundary",
  required_capabilities = [@rvv],
  rvv_construction_protocol = "extension-family-construction-protocol.v1",
  rvv_emitc_route_mapping = "rvv-i32m1-arithmetic-emitc-route-family"
} {
} : !tcrv_rvv.vl
```

The type is a non-compute vector-length token used by the bounded setvl and
with_vl surfaces and parser/printer ownership tests. The policy attribute is
finite non-compute metadata for proposal preservation, setvl/with_vl
configuration, and RVV plugin-local legality. The setvl op is a bounded
runtime VL control-plane surface: AVL is a runtime SSA operand, vl is a
`!tcrv_rvv.vl` result, SEW/LMUL/policy are compile-time config metadata for
this legacy compatibility slice, and VLEN/vlenb, `element_count`, `required_march`, and
`required_capabilities` are explicitly not accepted on the op. The with_vl op
is the bounded structural companion to setvl: it consumes one runtime VL SSA
token, owns one single-block region with no region arguments, and may repeat
the same bounded SEW/LMUL/policy config. When the bounded i32m1 materialized
EmitC artifact path selects this op as the construction-template boundary, the
same op must also carry selected-boundary conformance facts:
`source_kernel`, `selected_variant`, `origin`, `selected_path_role`, `status`,
`required_capabilities`, `rvv_construction_protocol`, and
`rvv_emitc_route_mapping`. These selected-boundary attrs are validation facts
only; they are not descriptors, source printers, runtime ABI synthesis,
hardware facts, or compute semantics. The with_vl op continues to reject
VLEN/vlenb, `element_count`, `required_march`,
`tcrv_rvv.required_capabilities`, and raw capability facts. The former
selected-boundary op and typed RVV microkernel wrappers are deleted as
structural authority; future executable emission must be rebuilt through
explicit extension-family IR plus a materialized MLIR EmitC module route.
These surfaces are not vector registers, masks, memory operations, RVV
intrinsics, LLVM/RISC-V lowering, runtime ABI, executable emission, correctness
evidence, or performance evidence.

## RVV Parameter Boundary

RVV work must keep these parameter layers distinct:

- VLEN and vlenb are hardware facts / target capability evidence. They may
  constrain legality and selection only after provenance is validated, but
  active profiles must keep VLENB as raw byte evidence and must not derive
  finite i32/M1 lane-capacity capability facts or selected-path metadata from
  it. Lane behavior belongs to explicit RVV config IR plus runtime AVL/VL/ABI
  surfaces.
- SEW, LMUL, tail policy, and mask policy are compile-time variant config
  selected or proposed by the RVV plugin and checked against target
  capabilities. The current non-executable bounded RVV dataflow slice admits only SEW 32 with
  LMUL m1 or m2, tail agnostic, and mask agnostic. The m1 shape is backed by
  `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic`; the m2 shape is backed by
  `rvv.i32_m2.sew32`, `rvv.i32_m2.lmul_m2`,
  `rvv.i32_m2.tail_policy.agnostic`, and
  `rvv.i32_m2.mask_policy.agnostic`. A selected variant must carry explicit
  typed RVV IR whose SEW/LMUL/policy attributes agree with its requirements,
  and future EmitC/export routes must reject capability/body LMUL mismatches
  before artifact bytes are emitted. These config ids are not
  sufficient as standalone hardware facts without the surrounding
  RVV/profile/toolchain evidence.
- Target helper catalogs must not serialize the selected RVV compile-time
  choice. Materialized explicit RVV variants and any future generated artifacts
  must derive SEW, LMUL, tail/mask policy, and intrinsic spellings from real
  `tcrv_rvv` IR plus the materialized EmitC/runtime route, not from selected
  metadata descriptors, suffix tables, or target-owned shape catalogs.
- AVL and vl are runtime SSA values / runtime control values. The current
  bounded `tcrv_rvv.setvl` surface models AVL as a real runtime SSA operand and
  vl as a real `!tcrv_rvv.vl` result. The bounded `tcrv_rvv.with_vl` surface
  models vl only when it consumes a real `!tcrv_rvv.vl` SSA operand. It must not
  imply that AVL or vl is IR-modeled unless a real op attribute, SSA value,
  region argument, or ABI parameter exists.
- The deleted legacy RVV local element-count marker is not active selected-path
  metadata. Active RVV dialect ops must reject local `element_count`
  attributes; the value does not describe a valid descriptor, emitted source
  slice, production input, high-level MLIR tensor shape, global problem size,
  AVL, or vl.
- `tcrv_rvv.required_march` string matching is a bounded plugin-owned
  compatibility bridge for the legacy compatibility slice. Do not expand dependence on
  `required_march` string comparisons when structured capabilities or
  properties are available or should be added.

Emission plans and manifests for RVV paths must not claim VLEN, vlenb, SEW,
LMUL, AVL, vl, `setvl`, `with_vl`, `element_count`, or `required_march` are
IR-modeled unless the real IR has the corresponding attribute, type, SSA value,
region argument, or generated ABI parameter. The current `tcrv_rvv.setvl` and
`tcrv_rvv.with_vl` ops model only runtime AVL/VL control-plane IR; they do not
make VLEN/vlenb or deleted local RVV element-count residue runtime values.
Selected-boundary attrs on `tcrv_rvv.with_vl` are artifact-handoff
conformance facts for the selected materialized EmitC route; they do not make
capability facts, route ids, ABI records, or compute bodies executable by
themselves.
RVV emission plans must not use selected-shape metadata descriptors as
lowering, runtime ABI, or artifact authority. Bounded diagnostics may mention
validated raw hardware/profile facts such as `tcrv_rvv.vlenb_bytes`, but
executable artifacts require explicit extension-family IR plus a materialized
EmitC/runtime route.

## Selected `with_vl` Lowering Boundary Route

The canonical public `tcrv-opt` pass
`--tcrv-materialize-selected-lowering-boundaries` still routes through the
generic `ExtensionPluginRegistry` interface. For the bounded legacy i32m1
route-table path, RVV recognizes the existing `tcrv_rvv.with_vl` operation in
the selected variant body as the selected lowering boundary. There is no
RVV-specific public wrapper pass for this route.

Rules:

- RVV-specific interpretation stays in the RVV plugin/dialect implementation.
- The generic pass routes selected direct variants and dispatch cases through
  the plugin interface; the RVV plugin may report the existing
  `tcrv_rvv.with_vl` op as the materialized boundary when the selected variant
  is a legal legacy i32m1 route-table body.
- Dispatch fallback references continue through their generic fallback
  envelope and do not create an RVV boundary.
- Kernels without a dispatch or direct selected-path diagnostic are diagnosed
  before any plugin lowering-boundary hook is invoked.
- The old boundary op must not be materialized as an unsupported placeholder.
- `tcrv_rvv.with_vl` boundary validation must use the RVV-owned config/VL
  contract and fail closed for missing, duplicate, mismatched, or unsupported
  selected boundary shapes.
- The selected `tcrv_rvv.with_vl` boundary must carry IR-owned conformance
  facts before emission planning or artifact export: source kernel, selected
  variant, origin plugin, selected-path role, status, required capabilities,
  construction protocol id, route mapping id, and bounded RVV config attrs.
  RVV plugin and target/export code must validate these attrs; they must not
  synthesize missing selected-boundary truth at planning or export time.
- RVV target exporters must not consume selected-boundary route identity fields
  as artifact authority by themselves. Executable emission requires the
  extension-family op -> MLIR EmitC module route.
- No RVV boundary result may be reported as hardware execution, correctness, or
  performance evidence.

## Vector-Level RVV Execution Surface

`tcrv_rvv` is RVV-owned vector-level execution IR inside selected RVV variants.
Its abstraction level is close to MLIR Vector and below Linalg/tensor/kernel
semantics. It is not a high-level kernel IR and not a one-op-per-intrinsic
wrapper dialect. It should express vector-level execution categories that the
RVV plugin can legalize, realize, and route.

This section is the Stage 2 target surface. The legacy `!tcrv_rvv.i32m1`
route becomes valid Stage 2 work only after it is represented as an ordinary
typed specialization of this surface. A provider still centered on
`RVVI32M1ArithmeticRouteSpec`, `RVVI32M1ArithmeticSlice`,
`collectRVVI32M1ArithmeticSlice`, finite `i32_*` route cases, or exact
`__riscv_*_i32m1` route spellings has not exited Stage 1.

Durable RVV execution dialect work may add richer types:

```text
!tcrv_rvv.vreg<dtype, lmul>
!tcrv_rvv.mask<lmul>
!tcrv_rvv.vl
!tcrv_rvv.policy<tail, mask>
```

Durable RVV execution dialect work may add vector-level ops and body patterns:

```text
tcrv_rvv.load / store / masked_load / masked_store
tcrv_rvv.broadcast / splat / movement forms
tcrv_rvv.add / mul / fma / max / min / compare / select
tcrv_rvv.reduce and accumulator primitives
tcrv_rvv.slide / gather / scatter / compress when supported
tcrv_rvv.convert / widen / narrow
low-level control around dynamic AVL/VL and masks
```

These are RVV execution units and body-shape contracts, not high-level tensor
ops, not route-id-as-compute ops, and not direct wrappers for every
`riscv_vector.h` intrinsic spelling.

## Variant Generation

Structured-kernel names are coverage references and evidence scenarios, not
current route authority. RVV maturity is calibrated by the math and
data-movement classes needed by structured AI kernels:

```text
elementwise maps and broadcasts
reductions and accumulator update patterns
contraction-like multiply-accumulate bodies
movement, layout, mask, tail, dtype, and runtime-shape handling
softmax/layernorm/rope/attention fragments as evidence scenarios
```

Future frontend/source construction must lower already-expressed source
semantics into a `tcrv.exec` envelope plus selected variant containing explicit
vector-level `tcrv_rvv` bodies. It must not add `tcrv.exec` matmul/reduction
compute ops, `tcrv_rvv.matmul`, `tcrv_rvv.softmax`, per-kernel route ids, or
per-intrinsic wrapper ops as route authority.

## Legality Rules

RVV plugin checks:

- target supports RVV;
- required dtype has corresponding extension or fallback path;
- SEW/LMUL combination is legal;
- VL policy is expressible;
- mask/tail policy is complete;
- load/store pattern is RVV-expressible;
- reduction handles tail/mask correctly;
- selected toolchain supports the emission path.

## Selected-Body Realization And Tuning

RVV performance tuning is plugin-local selected-body realization. It consumes
capability facts, the selected vector-level `tcrv_rvv` body, runtime SSA/ABI
values, and optional hints/policy/profile inputs, then rewrites the selected
body into realized vector-level RVV structure before emission planning. It is
a one-time linear lowering step in normal compilation, not a repeated
optimization loop, dashboard, readiness state machine, or global autotuning
database.

RVV plugin-local realization may choose and materialize:

```text
SEW
LMUL
mask/tail/VL policy
dynamic setvl / AVL placement
unroll factor
low-level movement / memory form
accumulator and reduction layout
software pipelining or prefetch structure when explicitly modeled or supported
boundary handling strategy
```

Reference metadata:

```mlir
#tcrv.tuning<
  lmul = 4,
  sew = 16,
  k_block = 64,
  unroll = 2,
  thread_partition = "row_block"
>
```

Hints or tuning metadata are not route authority. A hint affects generated code
only after RVV realization consumes it into operative `tcrv_rvv` vector-level
structure. Common passes, EmitC route builders, target artifacts, test names,
and route ids must not invent performance configuration.

Realization changes RVV execution structure, not computation semantics. It may
materialize legal VL/setvl placement, SEW/LMUL/policy choices, low-level vector
control, memory form selection, accumulator/reduction layout, and unroll or
pipeline organization. It must not change the kernel computation, dtype
semantics, parameter roles, variant origin, required capabilities,
dispatch/fallback semantics, runtime `n`/AVL values, or the body into direct
EmitC/C. The normal product is:

```text
selected pre-realized tcrv_rvv body
  -> RVV plugin-local realization
  -> realized tcrv_rvv body
```

## Selection Preference

The first RVV slice returns explicit plugin-owned selection preference metadata
for legal materialized RVV variants. Its score is a heuristic ordering input
used by the target-neutral selector; it is not a runtime, correctness, or
performance claim. Raw VLENB may be considered only as validated hardware
evidence; the plugin must not preserve derived finite i32/M1 lane-capacity
facts as selection preference authority. RVV-specific interpretation of
preserved capability facts stays inside the RVV plugin before the generic
preference record is returned.

## Emission Paths

Durable RVV emission path:

```text
selected RVV variant
  -> vector-level typed tcrv_rvv body
  -> RVV plugin-local realization when needed
  -> RVV-owned TCRVEmitCLowerableRoute
  -> parseable MLIR EmitC module
  -> C/C++ emitter / target artifact route when explicitly supported
bare RVV capability/no-body input -> no RVV proposal and no RVV route
```

Legacy bounded i32m1 materialized routes are evidence that explicit RVV
extension-family ops can construct a common `TCRVEmitCLowerableRoute` and
materialize `emitc.include`, `emitc.func`, and `emitc.call_opaque` operations
with interface-backed provenance. They do not define the mature route surface.
Any unsupported target artifact or runtime claim must fail closed until the
selected body, runtime ABI, C/C++ emitter handoff, and target artifact route
are all explicitly supported.

Public `tcrv-opt` registers the built-in RVV plugin at the tool boundary, so
materialized variants with `origin = "rvv-plugin"` can route through
`RVVExtensionPlugin` for plugin-local preflight diagnostics and through the
common EmitC lowerable materialization pass. Unknown origins must still fail
through the generic unregistered-origin registry diagnostic.
Tests that need the historical empty-registry parser surface should pass
`--tcrv-disable-builtin-plugins`.

The RVV emission-plan hook must not return generic unsupported runtime
metadata for a deleted selected-boundary route. Absence of supported runtime
ABI/glue remains a diagnostic, not code generation, hardware execution,
correctness evidence, or performance evidence.

### Deleted Standalone Smoke-Probe Target Export

The former standalone smoke-probe target export surface is deleted.
`RVVExtensionPlugin` must not turn plugin-local metadata or route records into
supported target-artifact authority, a supported emission plan, or a generic
target generated-source route for that standalone harness. Built-in target
artifact exporter registration must not publish the former smoke-probe route
identity or any standalone direct C output artifact kind as a supported RVV
output route.

Historical standalone smoke-probe metadata must not remain as active
code/spec/test fixtures. The compiler front door must not print
`riscv_vector.h`, `__riscv_` intrinsics, probe functions, or a probe `main`.
RVV hardware/toolchain smoke evidence belongs in explicit probe tooling and
separate `ssh rvv` artifacts.

### Deleted RVV Boundary And Bounded EmitC Route

The historical metadata/direct-wrapper RVV selected-boundary path remains
deleted and fail-closed for target emission. It must not materialize
selected-boundary metadata, RVV microkernel bodies, direct source/header/object
artifacts, self-check helpers, or intrinsic C/C++ output. The bounded legacy
i32m1 route-table path may materialize an MLIR EmitC module from explicit RVV
ops through the common lowerable route; a selected target artifact may package
that module only after the selected emission-plan candidate and materialized
EmitC handoff are verified.

The intended rebuild route is:

```text
explicit TCRV RVV extension-family ops
  -> EmitC
  -> RVV intrinsic C/C++
  -> clang default, gcc compatible
```

MLIR vector, LLVM scalable vector, LLVM RVV intrinsic IR, inline asm, and
backend patches are optional future routes. They are not the current RVV system
definition and should not be described as the mainline until promoted by a
separate spec and implementation evidence.

### Scenario: Legacy Bounded RVV Extension-Family EmitC Intrinsic Route

#### 1. Scope / Trigger

This scenario applies only while the bounded RVV i32m1 route-table
materialization route remains as a legacy narrow path. Current code evidence
includes `RVVI32M1ArithmeticOp::{Add,Sub,Mul,CmpSelect}`, optional
`tcrv_rvv.i32_broadcast_load`, and exact `__riscv_*_i32m1` spellings; that is
why this route is Stage 1 debt, not Stage 2 coverage. The route consumes
verified RVV family ops and lowers them through the shared EmitC lowerable route
into an MLIR EmitC module. Printing C/C++, compiling, target artifact
packaging, and `ssh rvv` runtime evidence are separate stages. A target artifact
bridge may invoke them only after this provider-owned route has been built and
verified. This scenario exists to constrain validation/fail-closed behavior
while replacing the old route table; it must not be used as the architecture for
adding dtype/LMUL batches, intrinsic wrappers, or per-source recognizers.

#### 2. Signatures

- Source family body:
  four explicit `tcrv_rvv.runtime_abi_value` bindings for `lhs`, `rhs`,
  `out`, and `n`, then
  `tcrv_rvv.setvl -> tcrv_rvv.with_vl -> tcrv_rvv.i32_load ->
  (tcrv_rvv.i32_load | tcrv_rvv.i32_broadcast_load) ->
  (tcrv_rvv.i32_add | tcrv_rvv.i32_sub | tcrv_rvv.i32_mul |
  tcrv_rvv.i32_cmp_eq -> tcrv_rvv.i32_select) -> tcrv_rvv.i32_store`
  for the legacy SEW32 LMUL m1 agnostic-policy route table.
- Selected lowering boundary: the existing `tcrv_rvv.with_vl` operation in the
  selected variant body, not a synthesized wrapper. The boundary op must carry
  `source_kernel`, `selected_variant`, `origin`, `selected_path_role`,
  `status`, `required_capabilities`, `rvv_construction_protocol`, and
  `rvv_emitc_route_mapping` before the route is planned or exported.
- Route plan: an explicit EmitC intrinsic route object with standard headers,
  source op names, `emitc.call_opaque` callee names, and one setvl callee.
- Materialized EmitC provenance comments must include typed source op names,
  source roles, `TCRVEmitCLowerableOpInterface`, and `emitc.call_opaque`
  callee evidence.
- Provider header: `TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`.
- Provider route builders:
  `plugin::rvv::buildRVVI32M1ArithmeticEmitCLowerableRoute`,
  `plugin::rvv::buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation`,
  `plugin::rvv::buildRVVI32M1AddEmitCLowerableRoute`,
  `plugin::rvv::buildRVVI32M1SubEmitCLowerableRoute`, and
  `plugin::rvv::buildRVVI32M1MulEmitCLowerableRoute`. The generic
  operation-indexed builder takes `RVVI32M1ArithmeticOp` plus
  `const plugin::VariantEmitCLowerableRequest &` and populates
  `conversion::emitc::TCRVEmitCLowerableRoute &`; the add/sub/mul wrappers are
  convenience entry points. The compare/select case is selected through the
  generic operation-indexed builder, not a separate public
  `buildRVVI32M1CompareSelect...` wrapper.
- Provider metadata accessors own the bounded route id, emission kind,
  lowering-boundary op name, runtime ABI kind/name/glue role, and ordered
  runtime ABI parameters for this legacy RVV i32m1 route-table slice.
- Selected target artifact route id:
  `rvv-i32m1-arithmetic-emitc-route-family`.
- Supported artifact kind: `riscv-elf-relocatable-object`.
- Target handoff kind:
  `materialized-emitc-cpp-rvv-intrinsic-object`.
- Target route API:
  `target::rvv::getRVVMaterializedEmitCTargetArtifactRouteID()`.

#### 3. Contracts

- The computation op is the RVV family op in the verified body, not metadata
  tokens.
- Explicit extension-family ops carry the bounded operation sequence, dtype,
  shape, predicate/dataflow, runtime ABI value use, and policy facts. The RVV
  provider derives the intrinsic spelling from that verified body. Metadata-only
  variants fail closed before source output.
- `emitc.call_opaque` callees map from verified family ops:
  setvl maps to the selected vsetvl intrinsic; load/broadcast, arithmetic,
  compare/select, and store map to the selected RVV intrinsics.
- Runtime `n` remains the IR-backed runtime-element-count ABI parameter.
- Runtime `lhs`, `rhs`, `out`, and `n` must be explicit SSA values produced by
  `tcrv_rvv.runtime_abi_value` in the selected RVV body before this bounded
  route may construct EmitC. `setvl` consumes the `n` SSA value; load/store
  ops consume the buffer SSA values. Synthetic
  `builtin.unrealized_conversion_cast` placeholders and buffer-role-only
  metadata are not artifact handoff authority for this route.
- Materialized EmitC comments are compiler route evidence only; they are not
  runtime correctness, hardware execution, throughput, latency, or performance
  evidence.
- RVV intrinsic/header names, typed-body shape validation, source-op
  provenance checks, and ABI value mapping are provider-owned. Target artifact
  support may consume the provider and package source/header/object artifacts,
  but it must not duplicate or re-own these RVV route semantics.
- Common selected EmitC artifact front doors call a route-builder callback and
  remain generic; they must not contain RVV intrinsic names, RVV header names,
  or typed RVV body-shape rules.
- Target/RVV artifact support registers one family-level materialized EmitC
  exporter for the selected legacy RVV i32m1 route-table object route. It must
  not register per-op target route tables, descriptor adapters, old
  object/header/bundle route ids, or compatibility wrappers.
- Target/RVV artifact support may compile the MLIR-emitted C/C++ source to a
  RISC-V relocatable object with clang. It must not synthesize the C/C++ source
  from metadata, selected-path records, route ids, family registries, or
  descriptors.
- Target/RVV artifact support must also not synthesize missing selected
  `with_vl` boundary conformance attrs. It may validate the IR-owned
  `with_vl` attrs against the selected variant, required capabilities,
  selected path role, construction protocol, route mapping, and bounded RVV
  config, then fail closed when any fact is absent or stale.

#### 4. Validation & Error Matrix

- Missing or duplicate family-op body step -> fail before source output.
- Body arithmetic op disagrees with selected bounded family -> fail before
  source output.
- Missing `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` control surface -> fail before
  source output.
- Missing, duplicate, mismatched, or unsupported selected `tcrv_rvv.with_vl`
  boundary shape -> fail before route payload construction or source output.
- Missing or stale selected `tcrv_rvv.with_vl` conformance attr
  (`source_kernel`, `selected_variant`, `origin`, `selected_path_role`,
  `status`, `required_capabilities`, `rvv_construction_protocol`,
  `rvv_emitc_route_mapping`, or bounded config attrs required by the route) ->
  fail before route payload construction, generated C++ source, object,
  header, or bundle output.
- Missing, duplicate, malformed, or unsupported explicit runtime ABI value
  binding for `lhs`, `rhs`, `out`, or `n` -> fail before source output.
- Missing route callee for any body step -> fail before source or object
  output.
- Stale runtime ABI role/name/type/ownership mirror -> fail before source
  output through the target artifact preflight.
- Target artifact code contains RVV intrinsic/header names or reconstructs the
  typed body shape instead of calling the provider -> ownership violation.
- Missing `rvv_emitc_lowerable_route` artifact metadata, stale selected
  runtime ABI identity, non-materialized EmitC handoff, missing route/call
  provenance, or clang object-packaging failure -> fail before claiming a
  target artifact.

#### 5. Good/Base/Bad Cases

- Good: while this legacy route is still retained, the selected body contains
  the exact typed op sequence for the legacy route case, the route records the
  matching `emitc.call_opaque` intrinsic sequence, generated C calls those
  provider-selected intrinsics, and the target bridge packages the MLIR-emitted
  source as a RISC-V relocatable object.
- Base: a hand-authored bounded RVV explicit dataflow body with the same
  verified legacy route-table sequence can use the same route after
  selected-path and ABI preflight pass.
- Bad: stale descriptor mirror metadata says i32 add but the body contains
  `tcrv_rvv.i32_sub`; export must fail instead of printing vadd C from the
  metadata.
- Bad: Target/RVV artifact packaging directly scans `tcrv_rvv` ops, chooses
  `riscv_vector.h`, or spells `__riscv_vadd_vv_i32m1` instead of delegating to
  the RVV provider.

#### 6. Tests Required

- lit/FileCheck or C++ tests must show selected-boundary materialization does
  not auto-create an RVV family body from descriptor/family records.
- Tests must show RVV selected emission planning does not build callable ABI
  parameters or supported source/header/object routes from selected metadata.
- Tests must show explicit typed RVV source/extension-op paths for retained
  legacy cases produce a supported family-level object artifact plan and that
  the generic target artifact front door can emit a RISC-V relocatable object
  through the common selected EmitC artifact bridge. New tests must not expand
  the old route table except as part of deleting, fail-closing, or replacing it.
- Negative coverage must keep stale body, stale descriptor, missing boundary,
  and malformed ABI cases fail-closed before source/header/object output.
- Target artifact tests must prove RVV target artifact exporters use the
  materialized EmitC family route, keep deleted direct route ids absent, and
  reject missing route provenance or stale runtime ABI metadata.

#### 7. Wrong vs Correct

Wrong:

```text
stale descriptor metadata -> generated microkernel body -> direct C/intrinsic source
```

Correct:

```text
explicit extension-family ops
  -> selected family-level EmitC artifact plan
  -> provider-owned TCRVEmitCLowerableRoute
  -> materialized MLIR EmitC module
  -> MLIR EmitC C/C++ emitter
  -> clang RISC-V relocatable object
```

## Hart Parallelism

RVV plugin does not define a thread/block model. Multi-core execution is organized by `tcrv.exec.hart_parallel`.

RVV plugin provides per-hart RVV execution behavior and lowering preferences for OpenMP, pthread, runtime thread pool, or single-thread paths.

## Diagnostics

Diagnostics must report:

- illegal LMUL/SEW;
- unsupported dtype;
- missing mask/tail policy;
- unavailable emission path;
- unsuitable memory pattern;
- unsatisfied capability.
