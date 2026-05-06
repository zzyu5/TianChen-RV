# `tcrv.exec` Contract

## Scope

`tcrv.exec` is the stable core dialect. It organizes RISC-V AI kernel execution variants. It does not express generic computation.

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
- carry cost, tuning, and dispatch metadata;
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

### `tcrv.exec.hart_parallel`

Represents coarse-grained RISC-V hart/core parallelism.

Rules:

- May lower to OpenMP, pthread, runtime thread pool, or single-thread loops.
- Must not model GPU-style thread/block execution.
- Must not assume all RISC-V targets have the same thread runtime.

### `tcrv.exec.region`

Optional structured region for extension-resource use:

```mlir
tcrv.exec.region kind = "rvv" {
  ... tcrv.rvv ops ...
}
```

Use it for verification and analysis when a variant contains a recognizable extension resource region.

### `tcrv.exec.mem_window`

Represents memory windows, shape specialization, stride/view data, and offload buffer binding context.

It supports:

- legal memory slices for plugins;
- contiguous/strided RVV access checks;
- offload buffer binding;
- shape guards for dispatch.

### `tcrv.exec.dispatch`

Represents runtime variant selection.

Reference shape:

```mlir
tcrv.exec.dispatch {
  case @offload if #tcrv.cond<"sophgo_available && M*N*K > threshold">
  case @rvv     if #tcrv.cond<"rvv_available">
  fallback @scalar_or_default
}
```

Dispatch conditions may use capability availability, runtime probe, shape/dtype guard, cost threshold, and user policy.

### `tcrv.exec.fallback`

Conservative correctness path for missing capability, unsupported shape, unavailable runtime, or dispatch failure.

Fallback may lower through scalar/scf, default MLIR lowering, portable C/C++, conservative RVV, or another declared path.

### `tcrv.exec.diagnostic` / diagnostic metadata

Represents structured reasons for variant rejection, dispatch choice, missing toolchain, missing runtime, unsupported dtype/layout, or fallback selection.

Diagnostics must be compiler-visible and testable with lit/FileCheck when they originate from dialect verification or passes.

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
