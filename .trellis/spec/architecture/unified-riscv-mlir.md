# Unified TCRV RISC-V MLIR

## System Definition

TianChen-RV MLIR is a unified RISC-V extension IR. It is one TCRV system with
a shared core envelope and multiple extension families:

```text
Unified TCRV RISC-V MLIR
  -> core execution / capability / ABI / dispatch envelope
  -> RVV extension family
  -> IME extension family
  -> TensorExt extension family
  -> Offload extension family
  -> future vendor/custom extension families
```

Implementation may split ODS files, C++ files, plugin bundles, directories,
and tests for maintainability. Conceptually, RVV, IME, TensorExt, Offload, and
future vendor/custom targets are not unrelated independent backend dialects.
They are extension families inside the same TCRV dialect suite and share the
same capability model, extension interfaces, common orchestration passes,
EmitC route framework, manifests, and test expectations.

## Common Abstraction

The common abstraction is not a mathematical kernel, tile, tensor compute op,
or one dialect per hardware target. The common abstraction is:

```text
capability-scoped extension execution
```

That means TCRV models:

```text
target capability
extension family
extension configuration state
extension resource
extension operation
ABI boundary
dispatch / fallback
EmitC lowering route
```

## Core Responsibilities

The TCRV core owns only the shared execution and decision envelope:

```text
capability
variant
requires
dispatch
fallback
ABI boundary
route selection
plugin registry
extension family registration
diagnostics
```

Core orchestration must work through TCRV interfaces and plugin registry
contracts. It must not branch on RVV, IME, TensorExt, Offload, scalar fallback,
vendor names, intrinsic names, fragment layouts, or extension-specific compute
semantics.

## Extension Family Responsibilities

Each extension family contributes hardware- or runtime-specific compiler
surfaces:

```text
hardware-specific ops
types
attributes
verifiers
local canonicalization / legalization hooks
EmitC lowering mapping
tests
```

Example architectural operation families may look like:

```text
tcrv.rvv_setvl
tcrv.rvv_with_vl
tcrv.rvv_load
tcrv.rvv_add
tcrv.rvv_store

tcrv.ime_config
tcrv.ime_load_frag
tcrv.ime_mma
tcrv.ime_store_frag

tcrv.tensorext_config
tcrv.tensorext_load_frag
tcrv.tensorext_mma
tcrv.tensorext_store_frag

tcrv.offload_bind
tcrv.offload_call
tcrv.offload_wait
```

Concrete MLIR namespaces may use implementation-compatible spellings such as
`tcrv_rvv.*` or `tcrv_scalar.*`. Those spellings are implementation details of
the TCRV dialect suite. They do not make the family an independent backend
dialect.

## Non-Architecture Surfaces

Descriptor-driven computation is not part of the TianChen-RV architecture.
A descriptor must not:

- define computation semantics;
- decide the long-term lowering target;
- stand in for a new extension op;
- become a RAG template for adding new extensions;
- replace extension family ops;
- justify direct C string emission as the main compiler route.

Existing finite descriptors, microkernel descriptors, and direct exporter
helpers are historical residue, deletion targets, or fail-closed
implementation debt. They must not be treated as transition architecture,
semantic source, compatibility aid, production input, or evidence authority.
Executable rebuild work must start from:

```text
extension family ops
  -> common EmitC lowering
  -> intrinsic / vendor builtin / runtime C ABI
```

## Current Main Lowering Route

The current main lowering route is:

```text
TCRV extension family ops
  -> EmitC ops
  -> C/C++ emitter
  -> intrinsic / vendor builtin / runtime C ABI
  -> native compiler
```

The default native compiler is clang/LLVM. GCC is a compatibility path.
Vendor compilers are extension-specific compatibility paths.

MLIR vector, LLVM scalable vector, LLVM RVV intrinsic IR, inline assembly, or
backend patches may be future optional routes for particular families. They
are not the current system definition and must not be described as the default
mainline until a separate spec promotes them with implementation evidence.
