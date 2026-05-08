# `tcrv.exec` Contract

## Scope

`tcrv.exec` is the stable core dialect. It organizes RISC-V AI kernel execution variants. It does not express generic computation.

Textual MLIR operation names should remain in the `tcrv.exec.*` family. Because MLIR resolves the dialect namespace from the segment before the first dot, an implementation may register the concrete MLIR dialect namespace as `tcrv` and define ODS operation mnemonics such as `exec.kernel`, `exec.variant`, and `exec.dispatch`. This is an implementation compatibility detail only: the architectural contract and review boundary remain the `tcrv.exec` core execution dialect family.

It owns only execution organization:

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
```

It does not own:

```text
generic matmul
generic softmax
generic reduce
generic tensor tile
algorithm-level compute semantics
extension-specific registers, fragments, buffers, or ops
```

Concrete computation belongs to extension dialects or extension op families, for example `tcrv.rvv`, `tcrv.ime`, `tcrv.offload`, or future plugin dialects.

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
            -> extension dialect ops
```

## Core Ops

### `tcrv.exec.kernel`

Responsibilities:

- define callable kernel inputs and outputs;
- bind target capability;
- contain one or more execution variants;
- define fallback strategy;
- declare whether runtime dispatch is allowed.

Reference shape:

```mlir
tcrv.exec.kernel @matmul(%A, %B, %C)
  attributes { target = #tcrv.target<...> } {
  ... variants ...
}
```

### `tcrv.exec.target` / target attachment

Represents the target capability object attached to a kernel or module.

Rules:

- target data must be a structured MLIR-level compiler object or attribute;
- passes and plugins must query it through compiler APIs;
- it must not be plain text metadata that is ignored by the compiler.

### `tcrv.exec.variant`

Responsibilities:

- declare required capabilities;
- contain extension dialect ops;
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
`FlatSymbolRefAttr` capability references resolved inside the enclosing
`tcrv.exec.kernel`. This is a compiler-visible structured requirement field,
not an arbitrary string list.

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
or future-plugin logic. A dispatch must be directly nested in a kernel, contain
at least one case, and contain exactly one `tcrv.exec.fallback`. When no
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

Emission-plan diagnostics are a structured specialization of
`tcrv.exec.diagnostic`, not a new core op. They use
`reason = "emission_plan"` and must remain generic compiler-visible metadata:

```mlir
tcrv.exec.diagnostic {
  reason = "emission_plan",
  message = "plugin-owned lowering/runtime route for selected path",
  severity = "info",
  status = "supported",
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

For `reason = "emission_plan"`, `target`, `origin`, `role`, `status`,
`runtime_abi_kind`, `runtime_abi_name`, `runtime_glue_role`, and
`required_capabilities` are required and non-empty. `status` must be
`supported`, `metadata-only`, or `unsupported`. `required_capabilities` must
contain capability symbol references that are a safe subset of the selected
target variant `requires` metadata. Supported and metadata-only diagnostics
additionally require non-empty `emission_kind`, `lowering_pipeline`,
`runtime_abi`, and `artifact_kind`. Unsupported diagnostics require non-empty
diagnostic text through `message`; they may still carry plugin-owned runtime
ABI ownership metadata to explain the unsupported boundary. The target must
resolve to a direct sibling `tcrv.exec.variant` in the same kernel.
Duplicate emission-plan diagnostics for the same target in one kernel are
invalid. When the selected path has a materialized plugin lowering boundary, a
diagnostic may also carry non-empty `lowering_boundary` metadata naming the
boundary operation used for the plan. This is a generic diagnostic link only;
it is not lowering, execution, correctness, or performance evidence. These
diagnostics do not prove that executable code was generated, linked, run,
correct, or performant.

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
- variant body extension ops are compatible with variant requirements;
- variant origin is registered plugin or explicitly marked external;
- dispatch covers capability-unavailable conditions;
- no high-level generic compute op appears in core dialect;
- offload variant declares runtime ABI and synchronization boundary;
- IME variant declares IME capability;
- RVV variant declares RVV capability.
- diagnostics are emitted for missing capabilities, invalid extension ops, missing emission path, or incomplete fallback/dispatch.

## Relation To High-Level MLIR

Correct:

```text
linalg.matmul
  -> plugins propose variants
  -> tcrv.exec.kernel with rvv/ime/offload/fallback variants
```

Wrong:

```text
linalg.matmul -> tcrv.matmul -> rvv/ime/offload
```
