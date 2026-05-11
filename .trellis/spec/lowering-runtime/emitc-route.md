# Unified EmitC Route

## Current Main Route

The current TianChen-RV lowering route is:

```text
TCRV extension family ops
  -> EmitC ops
  -> C/C++ emitter
  -> intrinsic / vendor builtin / runtime C ABI
  -> native compiler
```

The default native compiler is clang/LLVM. GCC is a compatibility path.
Vendor compilers are extension-specific compatibility paths.

This route applies across extension families:

```text
TCRV RVV family ops
  -> EmitC
  -> RVV intrinsic C/C++
  -> clang default, gcc compatible

TCRV IME family ops
  -> EmitC
  -> IME/vendor intrinsic C/C++

TCRV TensorExt family ops
  -> EmitC
  -> vendor intrinsic C/C++

TCRV Offload family ops
  -> EmitC
  -> runtime C ABI
```

MLIR vector, LLVM scalable vector, LLVM RVV intrinsic IR, inline assembly, and
backend patches are future optional routes. They are not the current default
system definition.

## Common EmitC Lowering Template

All extension families should reuse common lowering infrastructure:

```text
header/include builder
function boundary builder
emitc.call_opaque builder
C type mapping helper
ABI argument mapping helper
pointer/buffer mapping helper
intrinsic name resolver
generated C compile test harness
```

Each extension family contributes only the mapping:

```text
extension op
  -> intrinsic / runtime call name
  -> header
  -> operand mapping
  -> result mapping
```

The common lowering pass consumes `TCRVEmitCLowerableInterface` and emits
EmitC ops. This is the key difference from one hardware target owning one
separate lowering pass.

## Descriptor Boundary

Descriptor-driven computation is invalid as long-term architecture.

A descriptor must not:

- define computation semantics;
- decide which microkernel or extension operation represents the computation;
- serve as the way to add a new op;
- become the RAG template for generating a new extension family;
- replace extension family ops;
- justify direct C string emission as the primary route.

Existing fields and paths such as `lowering_descriptor`,
`i32-vadd-microkernel.v1`, descriptor-based microkernel dispatch, and
descriptor-to-C exporters are bounded implementation debt. They may be used
only to preserve existing transition slices until migrated to:

```text
extension family ops
  -> EmitC lowering
```

Direct handwritten C string emission is not the architecture. Generated C
should come from EmitC operations or a clearly marked bounded legacy helper
that is not used as the template for new extension work.
