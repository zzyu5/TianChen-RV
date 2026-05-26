# RVV Slice Contribution Guide

This document describes how to add one bounded RVV compiler slice to
TianChen-RV. It is not a tutorial and it is not a high-level frontend plan.
Each contributor should own a narrow RVV feature from typed IR through route
materialization and tests.

## Contribution Shape

A slice should follow this path:

```text
typed tcrv_rvv body
  -> RVV verifier / legality
  -> optional selected-body realization
  -> RVV provider route planning
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> RVV intrinsic C/C++
  -> optional QEMU proof
```

The route must derive from typed facts:

```text
operation kind
element type
SEW
LMUL
tail/mask policy
operand form
memory form
runtime ABI binding
target capability facts
```

Do not infer computation from route ids, artifact names, parameter names,
`c_type` strings, test names, source-front-door markers, or exact intrinsic
spellings.

## Where A Slice Usually Lands

### 1. RVV IR Surface

Primary files:

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
lib/Dialect/RVV/IR/RVVConfigContract.cpp
```

Use the generic typed surface:

```text
!tcrv_rvv.vector<element_type, "lmul">
!tcrv_rvv.mask<element_type, "lmul">
!tcrv_rvv.index_vector<element_type, "lmul">
tcrv_rvv.setvl
tcrv_rvv.with_vl
tcrv_rvv.load / store / binary / compare / select / ...
```

Add a new typed op only when an existing generic op cannot express the slice.
Do not add dtype-prefixed helper namespaces such as `tcrv_rvv.i8_add`.

### 2. Pre-Realized Body, If Needed

Primary files:

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h
lib/Plugin/RVV/RVVSelectedBodyRealization.cpp
```

Some slices may start from compact selected-body ops such as
`tcrv_rvv.typed_*_pre_realized_body`. The RVV plugin must realize those ops
into explicit `setvl` / `with_vl` / typed vector-level body structure before
route construction. A pre-realized op is not a route id, direct C exporter, or
metadata shortcut.

Existing owner families are a good local pattern:

```text
elementwise/compare-select
runtime scalar splat-store
runtime scalar computed-mask store
reduction
standalone reduction
MAcc
computed-mask MAcc
contraction
widening conversion
base memory movement
computed-mask memory
segment2 memory
```

### 3. Route Kind And Memory Form

Primary files:

```text
include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp
```

Most slices need a new or extended:

```text
RVVSelectedBodyOperationKind
RVVSelectedBodyMemoryForm
route family plan
route operand binding plan
```

The RVV provider owns RVV C vector type mapping, mask type mapping, intrinsic
selection, required headers, and route payload construction.

### 4. Runtime ABI Roles

Primary files:

```text
include/TianChenRV/Support/RuntimeABI.h
include/TianChenRV/Support/RuntimeABIContract.h
include/TianChenRV/Support/RuntimeABIMemWindow.h
include/TianChenRV/Support/RuntimeABIParam.h
```

Add a runtime ABI role only when the slice has a real new ABI concept, for
example a third segment field, an index buffer, a shift scalar, or an external
mask buffer. `tcrv.exec` and `tcrv_rvv.runtime_abi_value` declare/bind values;
they do not define computation.

### 5. EmitC / Target Output

Primary files:

```text
lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp
lib/Target/TargetArtifactExport.cpp
```

Most RVV slices should not need common EmitC semantic changes. If common code
must change, keep it neutral: it may materialize provider-built route payloads,
but it must not decide RVV dtype, SEW/LMUL, schedule, intrinsic spelling, or
body shape.

### 6. Tests

Expected test locations:

```text
test/Dialect/RVV/
test/Conversion/EmitC/
test/Target/RVV/
test/Scripts/               optional evidence helper coverage
```

Minimum useful coverage:

```text
dialect positive parse/print
dialect or route negative verifier case
EmitC materialization FileCheck
target/header/C++ output FileCheck
optional QEMU command/output for correctness
```

If the slice has a pre-realized form, also add:

```text
test/Target/RVV/pre-realized-selected-body-artifact-<feature>.mlir
```

## Forbidden Paths

Do not add:

- positive `tcrv_rvv.i32_*` or `!tcrv_rvv.i32m*` route authority;
- `RVVI32M1*` or `rvv-i32m1` compatibility route ids;
- source-front-door or source-artifact positive RVV routes;
- common EmitC RVV semantic branches;
- Toy / Template / TensorExtLite work as part of an RVV slice;
- internal automation or supervisor workflow files;
- frontend/Linalg/offload/runtime projects hidden inside a slice.

## PR Checklist

Before opening a PR, include:

- the slice name and bounded scope;
- touched operation kind(s), memory form(s), and dtype/SEW/LMUL/policy;
- tests run;
- generated RVV C/C++ evidence or FileCheck;
- QEMU command/output if runtime correctness is claimed;
- a note that no legacy helper/source-front-door path was added.
