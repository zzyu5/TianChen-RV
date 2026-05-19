# `tcrv.exec` Contract

## Scope

`tcrv.exec` is the stable core execution-envelope family. It organizes
capability-scoped RISC-V extension execution variants. It does not express
generic computation, hardware operation bodies, or extension-family execution
semantics.

Textual MLIR operation names should remain in the `tcrv.exec.*` family. Because MLIR resolves the dialect namespace from the segment before the first dot, an implementation may register the concrete MLIR dialect namespace as `tcrv` and define ODS operation mnemonics such as `exec.kernel`, `exec.variant`, and `exec.dispatch`. This is an implementation compatibility detail only: the architectural contract and review boundary remain the `tcrv.exec` core execution dialect family.

It owns only execution envelope and organization:

```text
kernel
target
capability
variant
requires
region
hart_parallel
mem_window
runtime_param
dispatch
case
fallback
diagnostics
selected variant/path diagnostic mirror
ABI envelope
```

It does not own:

```text
generic matmul
generic softmax
generic reduce
generic tensor tile
algorithm-level compute semantics
extension-specific registers, fragments, buffers, or ops
RVV vector add/load/store
IME mma
TensorExt tensor op
offload runtime call semantics
selected route authority
RVV dtype/config/schedule/intrinsic spelling
```

Concrete computation belongs to TCRV extension families such as RVV, IME,
TensorExt, Offload, or future vendor/custom families. The implementation may
place those families in separate ODS/C++ files or concrete MLIR namespaces such
as `tcrv_rvv`, but architecturally they are part of one unified TCRV dialect
suite rather than independent backend dialects.

## Implementation Stack

`tcrv.exec` must be implemented as MLIR compiler code:

- operation definitions in TableGen / ODS;
- C++ verifiers, parsers/printers when custom behavior is needed, interfaces, and passes;
- CMake build integration;
- lit/FileCheck tests for syntax, parsing, verification, diagnostics, and pass-visible behavior.

Do not implement `tcrv.exec` as Python classes, Python dictionaries, JSON-only schema, or a Python pseudo-IR. Python may only run tools or parse artifacts around the compiler.

## Execution Hierarchy

```text
tcrv.exec.kernel
  -> tcrv.exec.variant
       -> tcrv.exec.hart_parallel / tcrv.exec.region
            -> TCRV extension family ops
```

## Core Ops

### `tcrv.exec.kernel`

Responsibilities:

- represent a callable execution plan envelope, not a mathematical kernel,
  high-level operator container, compute IR, or hardware IR body;
- define callable kernel inputs and outputs;
- bind target capability, either through kernel-local capability providers or
  an explicit module-level `tcrv.exec.target` reference;
- contain one or more execution variants;
- define fallback strategy;
- declare whether runtime dispatch is allowed.

Reference shape:

```mlir
tcrv.exec.target @rvv_profile attributes {
  id = "rvv.profile.rv64gcv",
  kind = "profile",
  provides = ["rvv"]
}

tcrv.exec.kernel @matmul attributes { target = @rvv_profile } {
  ... variants ...
}
```

The operation name is historical and practical. Reviews should read it as
`execution envelope`. Future lowering work that needs finer granularity should
prefer `tcrv.exec.region`, `tcrv.exec.variant`, `tcrv.exec.case`, and selected
extension boundaries as the main units rather than making `kernel` a high-level
compute center.

### `tcrv.exec.target` / target attachment

Represents the target capability object attached to a kernel or module.

Rules:

- target data must be a structured MLIR-level compiler object or attribute;
- passes and plugins must query it through compiler APIs;
- it must not be plain text metadata that is ignored by the compiler.
- a kernel-local `tcrv.exec.target` may remain a parse-only profile anchor when
  it carries no capability identity, but when it carries both non-empty `id`
  and `kind` attributes it is a structured capability provider;
- capability-provider target profiles participate in `TargetCapabilitySet`
  construction with the same generic `status` / `availability`, `provides`,
  `implies`, `conflicts`, and property preservation rules as
  `tcrv.exec.capability`;
- capability-provider target profile `id` values share the enclosing
  `tcrv.exec.kernel` uniqueness domain with direct `tcrv.exec.capability`
  ids, because plugin proposal, legality, and requires mapping use the same
  capability lookup object.
- a capability-provider target profile may declare additional module-level
  provider refs through `capability_providers = [@provider, ...]`. Each ref
  must resolve to a module-level `tcrv.exec.capability` or capability-provider
  `tcrv.exec.target` with non-empty `id` and `kind`; provider symbols and ids
  must be unique across the composed target scope. Missing refs, non-provider
  refs, malformed refs, self references, duplicate symbols/ids, and obvious
  nested target cycles are invalid. This is a target/capability composition
  contract only, not a compute or extension-route surface.
- a kernel may reference exactly one module-level capability-provider
  `tcrv.exec.target` using `target = @profile`. That referenced profile enters
  the kernel's `TargetCapabilitySet` before its composed providers and before
  direct kernel-local capability providers; all of these may satisfy variant
  `requires` through exact id, `provides`, or `implies` lookup. The reference
  must resolve to a direct module-level `tcrv.exec.target` with non-empty `id`
  and `kind`; parse-only targets, missing targets, non-target symbols,
  malformed provider composition, and direct kernel symbols that shadow the
  referenced profile are invalid.
- module-level targets are not collected implicitly. A kernel receives only its
  explicit `target = @profile` attachment, that target profile's explicit
  `capability_providers` composition, and its direct kernel-local capability
  providers, so unrelated module targets cannot silently affect plugin
  availability or selection.

### `tcrv.exec.capability`

Represents one compiler-visible target, toolchain, runtime/offload, policy, or
microarchitecture capability available to a kernel.

Rules:

- `id` and `kind` are required non-empty structured MLIR attributes;
- direct capability `id` values must be unique within the enclosing
  `tcrv.exec.kernel`, because generic capability queries use exact id lookup
  before relation-provider lookup;
- `provides`, `implies`, and `conflicts` are relation lists over capability ids
  and do not relax the owning capability id uniqueness rule;
- extension-specific interpretation of capability properties remains
  plugin-local.

### `tcrv.exec.variant`

Responsibilities:

- declare required capabilities;
- contain or reference TCRV extension family ops / selected extension
  boundaries;
- carry generic decision metadata plus cost, tuning, and dispatch metadata;
- record origin plugin;
- be checked by capability and plugin verifiers.

Reference shape:

```mlir
tcrv.exec.variant @rvv
  requires = #tcrv.requires<["rvv", "zvfh"]>
  origin = "rvv-plugin" {
  ... tcrv.rvv ops ...
}
```

### `tcrv.exec.requires`

Structured requirement attribute or op for variant prerequisites:

```mlir
requires = #tcrv.requires<["rvv", "zvl128b", "zvfh"]>
```

The first verifier implementation may spell the same structured requirement
with builtin MLIR attributes:

```mlir
tcrv.exec.variant @rvv
  attributes {
    origin = "rvv-plugin",
    requires = [@rvv],
    condition = "runtime_probe_available",
    guard = "shape_guard_passed",
    policy = "prefer_accelerated"
  } { ... }
```

For this compatibility form, `requires` must be an `ArrayAttr` of
`FlatSymbolRefAttr` capability references resolved in the kernel capability
scope. That scope contains direct `tcrv.exec.capability` providers,
kernel-local capability-provider `tcrv.exec.target` anchors, the explicitly
referenced module-level `tcrv.exec.target` profile when the kernel has
`target = @profile`, and that profile's explicit `capability_providers`
composition. This is a compiler-visible structured requirement field, not an
arbitrary string list.

Optional `condition`, `guard`, and `policy` attributes on
`tcrv.exec.variant` are generic non-empty strings when present. They preserve
plugin-proposed decision metadata for later dispatch synthesis. The core
dialect records these strings but does not interpret target family, dtype,
shape, layout, runtime, vendor, extension, cost-model, or tuning semantics.

Optional `fallback_role = "conservative"` marks a plugin-proposed variant as a
generic conservative fallback candidate. The marker is target-neutral: core
selection may use it to decide whether a `tcrv.exec.fallback` can be
materialized, but scalar/RVV/IME/offload semantics remain plugin-local. Core
passes must not invent a fallback from an arbitrary available variant that lacks
this generic role.

Selection may also materialize target-neutral preference metadata on
`tcrv.exec.diagnostic`, `tcrv.exec.case`, or `tcrv.exec.fallback`, including
`origin`, `preference_available`, `preference_score`, `preference_rank`,
`preference_policy`, `preference_explanation`, `preference_tie_break`, and
`fallback_role`. These attributes explain deterministic compiler ordering only.
They are not legality, lowering, runtime ABI, executable generation,
correctness, performance, or hardware evidence.

### `tcrv.exec.hart_parallel`

Represents coarse-grained RISC-V hart/core parallelism.

Rules:

- May lower to OpenMP, pthread, runtime thread pool, or single-thread loops.
- Must not model GPU-style thread/block execution.
- Must not assume all RISC-V targets have the same thread runtime.

The initial compiler verifier slice may spell hart organization with builtin
MLIR attributes:

```mlir
tcrv.exec.hart_parallel attributes {harts = 64 : i64, policy = "static"} {
  ... extension-owned work ...
}
```

For this compatibility form, `harts` is optional but must be a positive integer
when present. `policy` is optional but must be non-empty when present.
`tcrv.exec.hart_parallel` must be nested in a `tcrv.exec.variant`. Core
verification checks structure only; concrete runtime threading decisions remain
plugin/local lowering responsibility.

The compiler planning check `--tcrv-check-hart-parallel-capabilities` gives
`harts` its target-capability meaning. When `harts = N` is present, the pass
requires an available provider for generic capability id `target.hart_count`
with positive integer property `count`, and rejects `N > count`. Extension
profiles may expose plugin-local facts such as `rvv.hart_count` through
`provides = ["target.hart_count"]`; the core check must not branch on RVV,
IME, offload, scalar fallback, vendor names, or thread runtime implementations.
A `hart_parallel` without explicit `harts` remains structural and does not
request a concrete capacity.

### `tcrv.exec.region`

Optional structured region for extension-resource use:

```mlir
tcrv.exec.region kind = "rvv" {
  ... tcrv.rvv ops ...
}
```

Use it for verification and analysis when a variant contains a recognizable extension resource region.

The initial compiler verifier slice may use builtin string attributes:

```mlir
tcrv.exec.region attributes {
  kind = "extension-resource",
  name = "rvv-resource",
  purpose = "extension-owned-body"
} {
  ... extension-owned work ...
}
```

For this compatibility form, `kind` is required and non-empty. `name` and
`purpose` are optional but must be non-empty when present. The region must be
nested in a `tcrv.exec.variant`. Core verification must not interpret the kind
as RVV, IME, offload, or future-plugin legality.

### `tcrv.exec.mem_window`

Represents memory windows, shape specialization, stride/view data, and offload buffer binding context.

It supports:

- legal memory slices for plugins;
- contiguous/strided RVV access checks;
- offload buffer binding;
- shape guards for dispatch.

The initial compiler verifier slice may model memory windows as named symbol
ops. For runtime ABI buffer boundaries, the same op may also carry a bounded
ABI role, access direction, target/runtime ownership, and already-known C ABI
type spelling:

```mlir
tcrv.exec.mem_window @input_window {
  purpose = "variant-dispatch-guard",
  binding = "args",
  memory_space = "host"
}

tcrv.exec.mem_window @abi_lhs_input_buffer {
  purpose = "runtime-abi-buffer",
  binding = "kernel-argument",
  memory_space = "host",
  abi_role = "lhs-input-buffer",
  access = "read",
  ownership = "target-export-abi-owned",
  c_type = "const int32_t *"
}
```

For this compatibility form, `purpose` is required and non-empty. `binding` and
`memory_space` are optional but must be non-empty when present. `abi_role`,
`access`, `ownership`, and `c_type` are optional but, when present, must be
non-empty stable single-line strings. Direct kernel-child mem_window ops must
not duplicate the same `abi_role`, because duplicate runtime ABI roles would
make a target export boundary ambiguous. The op must be nested in a
`tcrv.exec.kernel` or `tcrv.exec.variant`. It describes memory organization
context only, not tensor computation, tensor shape, vector math, or
extension-owned buffer ops.

`abi_role`, `access`, `ownership`, `c_type`, symbol names, and artifact-facing
metadata are ABI/export declarations only. They do not define element dtype,
operation kind, memory form, RVV config, or compute semantics. A selected
typed extension-family body must explicitly import and consume these ABI values
before any plugin route provider can build executable lowering.

### `tcrv.exec.runtime_param`

Represents a named runtime scalar ABI/control parameter for execution
organization. It is the scalar counterpart to `tcrv.exec.mem_window` for
bounded runtime ABI values such as a runtime element count or explicit dispatch
availability guard.

The first compatibility form is a named symbol op:

```mlir
tcrv.exec.runtime_param @abi_runtime_element_count {
  purpose = "runtime-abi-scalar",
  abi_role = "runtime-element-count",
  c_name = "n",
  c_type = "size_t",
  ownership = "target-export-abi-owned"
}

tcrv.exec.runtime_param @abi_dispatch_availability_guard {
  purpose = "runtime-abi-scalar",
  abi_role = "dispatch-availability-guard",
  c_name = "rvv_available",
  c_type = "int",
  ownership = "target-export-abi-owned"
}
```

`purpose`, `abi_role`, `c_name`, `c_type`, and `ownership` are required
non-empty stable single-line strings. `c_name` must be a simple C identifier.
Direct kernel-child runtime_param ops must not duplicate the same `abi_role`,
because duplicate runtime scalar ABI roles would make a target export boundary
ambiguous. The op must be nested in a `tcrv.exec.kernel` or
`tcrv.exec.variant`. It describes runtime ABI/control organization only, not
tensor computation, tensor shape, vector math, hardware probing, or
target-family legality.

`c_name`, `c_type`, `abi_role`, and parameter names are export spellings and
runtime roles only. They must not be used by core code, common EmitC/export, or
tests to infer RVV dtype, operation kind, vector length policy, or route
identity.

### `tcrv.exec.dispatch`

Represents runtime variant selection.

Reference shape:

```mlir
tcrv.exec.dispatch {
  tcrv.exec.case @offload {
    condition = "runtime_available",
    guard = "large_shape",
    policy = "prefer_accelerated"
  }
  tcrv.exec.case @rvv {condition = "vector_capability_available"}
  tcrv.exec.fallback @scalar_or_default
}
```

Dispatch conditions may use capability availability, runtime probe, shape/dtype guard, cost threshold, and user policy.
The structured first-slice form uses `tcrv.exec.case` operations directly nested
inside `tcrv.exec.dispatch`. Each case references a sibling
`tcrv.exec.variant` symbol in the enclosing `tcrv.exec.kernel`. Optional
`condition`, `guard`, and `policy` attributes are non-empty generic strings; the
core dialect records them but does not interpret RVV, IME, offload, Sophgo, AME,
or future-plugin logic. A case may also carry optional typed
`runtime_guard_required = true` metadata. This BoolAttr is the compiler-owned
generic contract that the dispatch case requires an explicit runtime
dispatch-availability guard; printable `condition`, `guard`, and `policy`
strings are annotations and are not the semantic trigger for runtime ABI guard
creation. A guarded case may also carry optional `runtime_guard = @symbol`
executable-control linkage only when it also carries
`runtime_guard_required = true`. When present, the symbol must resolve to a
direct same-kernel `tcrv.exec.runtime_param` with generic ABI role
`dispatch-availability-guard`. After runtime-guard materialization, selected
cases with `runtime_guard_required = true` must have that same-kernel
`runtime_guard` link before target artifact export can claim a coherent plan.
The core verifier/checks only this generic typed marker, symbol, op-kind, and
ABI-role contract; plugin-specific meaning for why that guard is true remains
plugin-local or target-owned. A dispatch must be directly
nested in a kernel, contain at least one case, and contain exactly one
`tcrv.exec.fallback`. The compiler-owned runtime-guard materialization pass may
attach `runtime_guard` links to guarded cases, but fallback operations do not
carry case `runtime_guard_required` or `runtime_guard` metadata. When no
plugin-provided conservative fallback candidate is present, selection must
record a structured diagnostic instead of creating a fallback-less dispatch or
relabeling the selected variant as an implicit fallback.

### `tcrv.exec.fallback`

Conservative correctness path for missing capability, unsupported shape, unavailable runtime, or dispatch failure.

Fallback may lower through scalar/scf, default MLIR lowering, portable C/C++, conservative RVV, or another declared path.
In the structured first-slice form, `tcrv.exec.fallback` is directly nested in
`tcrv.exec.dispatch` and references a sibling `tcrv.exec.variant` symbol in the
enclosing kernel. It is not valid as arbitrary metadata inside a variant body.
The target variant must be selected through generic fallback eligibility such as
`fallback_role = "conservative"` or an equivalent plugin-owned cost/proposal
field carried into generic selection. The core dialect must not infer fallback
semantics from plugin names, target families, or capability IDs.
Target-owned dispatch exporters that consume a selected `tcrv.exec.dispatch`
must treat this fallback op as the scalar/fallback branch source of truth: a
detached callable route or emission-plan candidate cannot substitute for the
actual `tcrv.exec.fallback` target, and the target must resolve to a direct
same-kernel `tcrv.exec.variant` before export.

### `tcrv.exec.diagnostic` / diagnostic metadata

Represents structured reasons for variant rejection, dispatch choice, missing toolchain, missing runtime, unsupported dtype/layout, or fallback selection.

Diagnostics must be compiler-visible and testable with lit/FileCheck when they originate from dialect verification or passes.

The initial compiler verifier slice may model diagnostics as metadata ops:

```mlir
tcrv.exec.diagnostic {
  reason = "variant-selected",
  message = "rvv variant selected by capability guard",
  severity = "note",
  status = "accepted",
  target = @rvv_variant,
  selection_kind = "static-variant"
}
```

For this compatibility form, `reason` and `message` are required and non-empty.
`severity`, `status`, and `selection_kind` are optional but must be non-empty
when present. `target` is optional selected-path metadata and, when present,
must resolve to a variant symbol in the enclosing `tcrv.exec.kernel`. The op
must be nested in a `tcrv.exec.kernel` or `tcrv.exec.variant`.
When present on a selection diagnostic, preference metadata is plugin-provided
heuristic ranking context and must not be interpreted as performance truth or
soft legality.

Emission-plan diagnostics are an optional, non-authoritative specialization of
`tcrv.exec.diagnostic`, not a new core op and not route authority. They may be
materialized only after plugin-owned legality, optional selected-body
realization, and route-provider decisions have already produced the selected
typed body / provider-built route:

```mlir
tcrv.exec.diagnostic {
  reason = "emission_plan",
  message = "plugin-owned lowering/runtime route for selected path",
  severity = "info",
  // mirror only; not route authority or acceptance state
  result = "provider_supported_mirror",
  target = @selected_variant,
  origin = "example-plugin",
  role = "direct variant",
  lowering_boundary = "example.lowering_boundary",
  plan_kind = "plugin-emission-plan",
  emission_kind = "metadata-intent",
  lowering_pipeline = "example.lowering.pipeline.v1",
  runtime_abi = "example.runtime.abi.v1",
  runtime_abi_kind = "host-runtime-c-abi",
  runtime_abi_name = "example.runtime.abi.v1",
  runtime_glue_role = "plugin-owned-runtime-glue",
  required_capabilities = [@example_capability],
  artifact_kind = "compiler-emission-plan"
}
```

For `reason = "emission_plan"`, fields such as `target`, `origin`, `role`,
`result`, `runtime_abi_kind`, `runtime_abi_name`, `runtime_glue_role`,
`required_capabilities`, `emission_kind`, `lowering_pipeline`, `runtime_abi`,
`artifact_kind`, and `lowering_boundary` are exact mirrors when present. They
must not be required as the source of route construction, and they must not
define dtype, compute, schedule, selected route identity, target artifact
authority, correctness, performance, or progress. Unsupported results require
diagnostic text through `message`. Prefer mirror-explicit result labels such as
`provider_supported_mirror` over bare `supported` so agents do not interpret
the field as acceptance state. Duplicate mirrors for the same selected path are
invalid only to keep diagnostics deterministic.

## Core Types And Attributes

Core types/attributes should remain lightweight:

```text
#tcrv.target<...>
#tcrv.requires<...>
#tcrv.cap<...>
#tcrv.variant_info<...>
#tcrv.dispatch_cond<...>
#tcrv.cost<...>
#tcrv.tuning<...>
#tcrv.diagnostic<...>
```

Extension-specific types belong outside core:

```text
!tcrv.rvv.vreg<...>
!tcrv.rvv.mask<...>
!tcrv.ime.frag<...>
!tcrv.offload.buffer<...>
```

## Verifier Rules

The `tcrv.exec` verifier must check:

- each kernel has fallback or explicit external fallback declaration;
- each variant declares `requires`;
- generic variant/body structure is well formed;
- variant origin is registered plugin or explicitly marked external;
- dispatch covers capability-unavailable conditions;
- no high-level generic compute op appears in core dialect;
- extension legality, including RVV/IME/offload capability compatibility, is
  delegated to plugin hooks;
- diagnostics are emitted for missing capabilities, invalid extension ops, missing emission path, or incomplete fallback/dispatch.

## Relation To High-Level MLIR

High-level MLIR lowering is a future frontend integration phase, not a
precondition for current `tcrv.exec` and extension plugin development. Current
tests and bounded slices may start from hand-written TianChen-RV MLIR,
materialized `tcrv.exec.variant`, selected-boundary IR, `mem_window`,
`runtime_param`, or typed extension-family bodies. Selected-path metadata is
allowed only as diagnostic/control mirror and cannot be the compute or route
input. When
frontend lowering becomes the selected owner, it may start from hand-written or
test `linalg` inputs and lower them into complete TianChen-RV execution
surfaces: `tcrv.exec` envelope plus plugin-owned typed bodies or selected
boundaries. It must not stop at a bare `tcrv.exec` shell and infer extension
compute later from route ids, ABI names, artifact metadata, or test fixtures.

Correct:

```text
linalg.matmul
  -> semantic-preserving construction of tcrv.exec envelope
     + plugin-owned typed extension-family bodies
  -> plugins legalize / select rvv/ime/offload/fallback variants
```

Wrong:

```text
linalg.matmul -> tcrv.matmul -> rvv/ime/offload
```
