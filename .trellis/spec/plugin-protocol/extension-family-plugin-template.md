# Extension Family Plugin Template

This is the durable template for adding a new RISC-V extension family such as
IME, TensorExt, or a vendor/custom extension. A plugin is not an independent
backend. It contributes one extension family to the unified TCRV RISC-V MLIR
system.

## Required Contributions

A new extension family plugin must provide:

1. Extension Manifest.
2. Capability declarations.
3. Extension family ops, types, and attributes.
4. TCRV interface implementations.
5. Local legality and verifier hooks.
6. Local canonicalization or legalization hooks when needed.
7. EmitC lowering mapping.
8. Tests.

The plugin must not modify these core orchestration surfaces for a
family-specific special case:

- core capability verification pass;
- core variant selection pass;
- core dispatch/fallback pass;
- core ABI envelope logic;
- core EmitC lowering framework.

Core changes are allowed only when they extend a public interface or generic
orchestration surface for all families.

## TCRV Common Interfaces

Common passes should operate through interfaces rather than family-name
branches. The long-term interface set includes:

```text
TCRVExtensionOpInterface
TCRVCapabilityOpInterface
TCRVConfigOpInterface
TCRVResourceOpInterface
TCRVMemoryOpInterface
TCRVComputeOpInterface
TCRVEmitCLowerableInterface
```

Interface responsibilities:

- expose required capability ids and capability predicates;
- expose config scope and selected extension state;
- expose resource kind, memory role, and compute primitive kind;
- expose runtime ABI and buffer/pointer mapping needs;
- expose EmitC lowering metadata without exposing family-specific logic to core
  passes.

## Common Passes

The TCRV pass framework should favor common passes that consume these
interfaces:

```text
tcrv-verify-capability
tcrv-verify-extension-config
tcrv-verify-resource-legality
tcrv-canonicalize-extension-state
tcrv-select-route
tcrv-lower-extension-to-emitc
```

These passes must not implement:

```text
if RVV
if IME
if TensorExt
if Offload
```

They query:

```text
required capability
config scope
resource kind
memory role
compute primitive kind
EmitC lowering mapping
```

Extension families may provide local verification, canonicalization,
legalization, and mapping hooks, but those hooks are attached through the
shared interfaces and registry.

## Extension Manifest

Each new family should define a machine-readable or semi-machine-readable
manifest. The manifest is the source for future RAG-assisted skeleton
generation; descriptors are not.

Example:

```yaml
extension: tensorext
kind: custom-riscv-extension

capabilities:
  - tensorext
  - tensorext.fp16
  - tensorext.int8

family:
  name: tensorext
  namespace: tcrv_tensorext

types:
  - name: frag
    role: tensor-fragment
  - name: accfrag
    role: accumulator-fragment

attributes:
  - name: layout
    role: fragment-layout
  - name: shape
    role: intrinsic-tile-shape

ops:
  - name: config
    role: configure-extension
  - name: load_frag
    role: memory-to-fragment
  - name: mma
    role: tensor-compute
  - name: store_frag
    role: fragment-to-memory

interfaces:
  - TCRVExtensionOpInterface
  - TCRVConfigOpInterface
  - TCRVResourceOpInterface
  - TCRVComputeOpInterface
  - TCRVEmitCLowerableInterface

emitc_route:
  headers:
    - tensorext_intrinsics.h
  intrinsic_map:
    config: __tensorext_config
    load_frag: __tensorext_load_frag
    mma: __tensorext_mma
    store_frag: __tensorext_store_frag

local_passes:
  - canonicalize-config
  - verify-fragment-layout
  - legalize-intrinsic-shape

tests:
  - parse_verify
  - capability
  - selected_boundary
  - interface_verification
  - emitc_lowering
  - generated_c_compile
```

RAG-assisted extension addition should use this manifest to generate:

```text
ODS skeleton
C++ family registration
plugin class
capability provider
interface implementations
verifier skeleton
local pass skeleton
EmitC lowering skeleton
tests
```

## Recommended Directory Layout

```text
include/TianChenRV/Plugin/<Ext>/
lib/Plugin/<Ext>/

include/TianChenRV/Dialect/<Ext>/IR/
lib/Dialect/<Ext>/IR/

include/TianChenRV/Target/<Ext>/        optional
lib/Target/<Ext>/                       optional

test/Plugin/<Ext>/
test/Dialect/<Ext>/
test/Target/<Ext>/
```

These directories are implementation organization for one unified TCRV system.
They do not imply independent backends.

## Test Template

New extension family work should include focused tests for:

- capability recognized and rejected when unavailable;
- plugin can register the family ops and capabilities;
- variant or selected boundary appears only with matching capability;
- illegal config fails closed;
- common interfaces are implemented and queried by common passes;
- selected boundary materialization is family-owned;
- EmitC lowering route is family-owned mapping through the common pass;
- generated C compiles with clang/LLVM by default when the route claims it;
- GCC is accepted only as a compatibility path where supported;
- no extension-specific semantic branch is added to core passes.
