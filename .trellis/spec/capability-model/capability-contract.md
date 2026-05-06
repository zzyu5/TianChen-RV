# Capability Contract

## Scope

Capability objects are MLIR-level compiler objects. They describe what the target can execute, what the toolchain can emit, and what runtime services can be used.

They must be queryable by C++ MLIR passes and extension plugins. They are not prose annotations, plain strings, JSON-only records, or Python dictionaries.

They must influence:

- enabled plugins;
- variant proposal;
- variant legality;
- legality verification;
- tuning space;
- cost model inputs;
- variant selection;
- runtime dispatch;
- emission path selection;
- lowering diagnostics;
- fallback requirements.

## Capability Sources

### ISA capabilities

Examples:

```text
rv64
rvv
zvl*
zvfh
zvfbf*
I / M / A / F / D / C
V
Zvl128b
Zvfh
Zvfbfmin
Zvfbfwma
ime
vendor custom opcode
SpacemiT IME extension
future matrix extension
future custom ISA
```

### Microarchitecture capabilities

Examples:

```text
core count
VLEN
cache size
memory bandwidth
preferred LMUL
supported dtype throughput
dtype support
NUMA / memory topology
OpenMP or thread runtime availability
toolchain availability cross-links
```

### Runtime/offload capabilities

Examples:

```text
Sophgo accelerator present
TPU runtime available
runtime name
ABI
PCIe mode
SoC mode
supported offload operator set
supported offload operations
supported model format
host-device transfer cost
async execution support
```

### Toolchain capabilities

Examples:

```text
LLVM RVV scalable vector support
RVV intrinsic support
compiler builtin support
inline asm allowed
vendor header available
patched compiler available
runtime library linkable
```

## Logical Shape

Target capability should be represented as a structured target-level or module-level MLIR attribute.

Reference shape:

```mlir
#tcrv.target<
  arch = "riscv64",
  isa = ["i", "m", "a", "f", "d", "c", "v", "zvl128b", "zvfh"],
  uarch = {
    cores = 64,
    vlen = 128,
    has_openmp = true,
    cache_model = "target_specific"
  },
  extensions = [
    #tcrv.ext<"rvv", kind = "isa-vector", status = "available">
  ],
  accelerators = [
    #tcrv.accel<"sophgo.bm1684x", kind = "runtime-offload",
                mode = "pcie", runtime = "vendor-c-abi">
  ],
  toolchain = {
    llvm_rvv = true,
    rvv_intrinsic = true,
    inline_asm = true,
    vendor_runtime_link = true
  }
>
```

IME may add:

```mlir
#tcrv.ext<"spacemit.ime",
          kind = "isa-matrix-vector-backed",
          status = "available",
          register_model = "rvv-vector-register-backed",
          dtype = ["int8", "fp16", "bf16"]>
```

## Open Capability Kinds

Capability kind is an open set. Initial kinds include:

```text
isa-scalar
isa-vector
isa-matrix-vector-backed
isa-matrix-separate-register
isa-custom-instruction
runtime-offload
toolchain
uarch
memory
thread-runtime
```

Future kinds may include:

```text
isa-sparse
isa-dma
isa-cluster
isa-crypto-ai
runtime-remote-device
runtime-shared-memory-accelerator
```

Core passes must not exhaustively switch over all kind values. They must use plugin-registered interfaces and capability queries.

## Relations

### require

Variant declares its required capabilities:

```mlir
tcrv.exec.variant @rvv_fp16
  requires = #tcrv.requires<["rvv", "zvfh", "zvl128b"]> { ... }
```

The initial compiler verifier slice may use MLIR builtin structured
attributes until custom `#tcrv.requires<...>` parsing is introduced:

```mlir
tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
tcrv.exec.variant @rvv_fp16
  attributes {origin = "rvv-plugin", requires = [@rvv]} { ... }
```

In that slice, `requires` is an `ArrayAttr` whose entries must be
`FlatSymbolRefAttr` references to `tcrv.exec.capability` ops in the enclosing
`tcrv.exec.kernel`. Capability `id` and `kind` are non-empty `StringAttr`
fields. Core verification checks structure and symbol resolution only; concrete
extension legality stays plugin-owned.

### provide

Plugin declares provided capabilities:

```text
RVV plugin: rvv, vector-stripe, scalable-vl, rvv-load-store, rvv-reduce
IME plugin: ime, vector-register-backed-matrix, ime-frag-mma
Offload plugin: sophgo-runtime, async-offload, runtime-buffer
```

### imply

Examples:

```text
rv64gcv implies rvv
zvfh implies fp16 vector arithmetic, subject to toolchain support
spacemit.ime implies vector-register-backed matrix capability, subject to vendor toolchain support
```

### conflict

Examples:

```text
variant requires vendor runtime but target lacks runtime library
variant requires inline asm but build policy forbids inline asm
offload variant requires fixed shape but input shape is dynamic or unsupported
```

### dispatch condition

Runtime or shape-dependent capability conditions become dispatch predicates:

```text
if runtime_available && large_shape -> offload variant
else if rvv_available -> RVV variant
else -> fallback
```

## Verifier Expectations

Capability verifier must check:

- variant `requires` is satisfied by target capability or guarded by dispatch;
- extension ops appear only in compatible variants;
- `tcrv.rvv` ops appear only in RVV-capable variants;
- `tcrv.ime` ops appear only in IME-capable variants;
- `tcrv.offload` ops appear only in runtime-offload-capable variants;
- selected emission path is supported by toolchain capability;
- runtime ABI declarations are complete;
- dispatch/fallback covers unavailable conditions.

The verifier does not prove numerical correctness. It prevents illegal target-feature usage and missing execution prerequisites.

## Implementation Requirements

Capability model implementation belongs in the primary compiler stack:

- C++ data structures and MLIR attributes/types/interfaces;
- TableGen / ODS definitions when represented in IR;
- CMake build targets;
- lit/FileCheck tests for IR parsing, printing, verification, and diagnostic behavior;
- C++ tests for registry/query helper semantics where textual MLIR tests are insufficient.

Python may probe a target, parse remote output, or generate a report. Python must not be the source-of-truth implementation of capability relations, plugin availability, legality, selection, dispatch, or lowering diagnostics.
