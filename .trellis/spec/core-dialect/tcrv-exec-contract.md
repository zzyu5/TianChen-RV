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
ops:

```mlir
tcrv.exec.mem_window @input_window {
  purpose = "variant-dispatch-guard",
  binding = "args",
  memory_space = "host"
}
```

For this compatibility form, `purpose` is required and non-empty. `binding` and
`memory_space` are optional but must be non-empty when present. The op must be
nested in a `tcrv.exec.kernel` or `tcrv.exec.variant`. It describes memory
organization context only, not tensor computation or extension-owned buffer ops.

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
at least one case, and contain exactly one `tcrv.exec.fallback`.

### `tcrv.exec.fallback`

Conservative correctness path for missing capability, unsupported shape, unavailable runtime, or dispatch failure.

Fallback may lower through scalar/scf, default MLIR lowering, portable C/C++, conservative RVV, or another declared path.
In the structured first-slice form, `tcrv.exec.fallback` is directly nested in
`tcrv.exec.dispatch` and references a sibling `tcrv.exec.variant` symbol in the
enclosing kernel. It is not valid as arbitrary metadata inside a variant body.

### `tcrv.exec.diagnostic` / diagnostic metadata

Represents structured reasons for variant rejection, dispatch choice, missing toolchain, missing runtime, unsupported dtype/layout, or fallback selection.

Diagnostics must be compiler-visible and testable with lit/FileCheck when they originate from dialect verification or passes.

The initial compiler verifier slice may model diagnostics as metadata ops:

```mlir
tcrv.exec.diagnostic {
  reason = "variant-selected",
  message = "rvv variant selected by capability guard",
  severity = "note",
  status = "accepted"
}
```

For this compatibility form, `reason` and `message` are required and non-empty.
`severity` and `status` are optional but must be non-empty when present. The op
must be nested in a `tcrv.exec.kernel` or `tcrv.exec.variant`.

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
