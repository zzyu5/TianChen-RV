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

## Scenario: Generated EmitC Lowerable Op Interface Boundary

### 1. Scope / Trigger

Use this contract when a typed extension-family operation participates in the
common EmitC route. The operation must carry a generated MLIR op interface, and
the target/export path must query that generated interface before constructing
the C++ route payload.

### 2. Signatures

- Generated ODS interface:
  `TCRVEmitCLowerableOpInterface`.
- Generated interface methods:
  `llvm::StringRef getTCRVEmitCLowerableSourceOpName()`;
  `llvm::StringRef getTCRVEmitCLowerableSourceRole()`.
- C++ route payload:
  `TCRVEmitCLowerableRoute`, built through the hand-written
  `TCRVEmitCLowerableInterface` adapter API.
- Route provenance field:
  `TCRVEmitCSourceOpProvenance::opInterface`.

### 3. Contracts

- `TCRVEmitCLowerableOpInterface` is the IR-modeled source-op boundary.
- `TCRVEmitCLowerableInterface` remains the C++ adapter for building and
  verifying a `TCRVEmitCLowerableRoute`.
- The generated op interface must expose bounded provenance only: source op
  name and source role. It must not expose descriptor-selected computation,
  target hardware facts, runtime results, correctness evidence, performance
  evidence, or family-specific intrinsic spelling to common code.
- Target/plugin-owned code may map the interface-backed provenance to
  family-owned intrinsic or runtime names, then store the interface class name
  in route provenance for source/export evidence.
- Descriptor metadata remains selected-config, ABI identity, legacy id, and
  mismatch cross-check data only.

### 4. Validation & Error Matrix

- Typed family op lacks `TCRVEmitCLowerableOpInterface` before route
  construction -> fail before artifact export.
- Interface source op name disagrees with the selected family operation ->
  fail before artifact export.
- Interface source role is unsupported for the bounded route -> fail before
  artifact export.
- Descriptor family disagrees with the typed body op -> fail before artifact
  export.
- Route provenance text is empty, unbounded, multiline, or unsafe -> route
  verification fails before source emission.

### 5. Good/Base/Bad Cases

- Good: `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul`
  implement `TCRVEmitCLowerableOpInterface`; RVV target code queries the
  interface, then builds `TCRVEmitCLowerableRoute` with
  `opInterface = "TCRVEmitCLowerableOpInterface"`.
- Base: a target-specific bounded route still owns intrinsic names, vector
  suffixes, C type spellings, and ABI details locally.
- Bad: descriptor strings alone choose the arithmetic intrinsic, or common code
  adds an `if RVV` branch to recover family semantics.

### 6. Tests Required

- TableGen/CMake build coverage for generated op-interface declarations and
  definitions.
- C++ coverage proving the bounded typed ops implement the generated interface
  and that route provenance records the interface-backed source op.
- lit/FileCheck coverage proving exported source contains the interface-backed
  route evidence for each affected family op.
- Negative coverage proving descriptor/body mismatch still fails before source
  export.

### 7. Wrong vs Correct

Wrong:

```text
descriptor id -> choose arithmetic intrinsic -> emit C source
```

Correct:

```text
typed extension op implements generated op interface
  -> target/plugin queries interface-backed source provenance
  -> target/plugin maps to family-owned intrinsic/runtime call
  -> common TCRVEmitCLowerableRoute carries bounded route payload
  -> EmitC/C source evidence records generated-interface provenance
```

## Descriptor Boundary

Descriptor-driven computation is invalid as long-term architecture.

A descriptor must not:

- define computation semantics;
- decide which microkernel or extension operation represents the computation;
- serve as the way to add a new op;
- become the RAG template for generating a new extension family;
- replace extension family ops;
- justify direct C string emission as the primary route.

Existing fields and paths such as descriptor-based microkernel dispatch and
descriptor-to-C exporters are historical residue, deletion targets, or
fail-closed implementation debt. They must not be treated as transition
architecture, compatibility aids, semantic sources, or production inputs.
Executable rebuild work must use:

```text
extension family ops
  -> EmitC lowering
```

Direct handwritten C string emission is not the architecture. Generated C
should come from EmitC operations or a clearly marked bounded legacy helper
that is not used as the template for new extension work.
