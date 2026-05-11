# IME Plugin

## Role

IME plugin is the later K3/IME extension family path inside the unified TCRV
system. Its primary value is to validate plugin-local matrix-extension
integration without turning IME into an independent backend.

Expected system behavior when IME is available:

```text
existing system supports RVV
add IME plugin
same high-level op gains IME execution variant
core pass does not hard-code IME
core pass modifications remain small or zero when existing interfaces suffice
```

## Positioning

IME is not ordinary runtime offload. It is closer to:

```text
ISA-level matrix-like extension
vector-register-backed matrix execution
RVV-resource-related matrix/dot execution
```

IME plugin must understand:

- IME matrix/dot instructions;
- RVV vector register resources;
- VLEN impact on fragment shape;
- dtype and accumulator policy;
- EmitC mapping to IME/vendor intrinsic C/C++ and the required toolchain path.

## Capability Fields

IME plugin should register:

```text
spacemit.ime or target-specific IME capability
ime.version
ime.supported_dtype
ime.vector_register_backed
ime.vlen_range
ime.fragment_shapes
ime.accumulator_types
ime.toolchain_path
```

Reference attribute:

```mlir
#tcrv.ext<"spacemit.ime",
          kind = "isa-matrix-vector-backed",
          status = "available",
          register_model = "rvv-vector-register-backed",
          dtype = ["int8", "fp16", "bf16"],
          vlen_dependent = true>
```

## Extension Family

Architectural family name:

```text
tcrv.ime
```

Types:

```text
!tcrv.ime.frag<dtype, shape, layout>
!tcrv.ime.accfrag<dtype, shape>
!tcrv.ime.config<...>
```

The type design must express relation to RVV resources:

```text
register_model = rvv-vector-register-backed
vlen_dependent = true
```

Ops:

```text
tcrv.ime.config
tcrv.ime.load_frag
tcrv.ime.store_frag
tcrv.ime.pack
tcrv.ime.unpack
tcrv.ime.mma
tcrv.ime.dot
tcrv.ime.accumulate
tcrv.ime.convert
```

These are IME execution ops, not high-level matmul ops.

## Variant Generation Scope

IME plugin prioritizes:

```text
matmul
batched matmul
attention qk/av matrix block
MLP dense block
int8/fp16/bf16 dot-like kernels
```

It does not need to cover every operator. It should cover operators that show matrix-extension value.

## Legality Rules

IME plugin checks:

- target declares IME capability;
- RVV/vector-register-backed prerequisites are satisfied;
- dtype is supported by IME;
- fragment shape matches VLEN and instruction limits;
- accumulator type is supported;
- layout and packing requirements are satisfied;
- toolchain supports selected IME emission path;
- IME/vendor intrinsic C/C++ route is available when required. Inline asm or
  backend adapter routes are optional future compatibility paths, not the
  current default.

## Tuning Space

IME tuning includes:

```text
fragment shape
K blocking
accumulator residency
packing format
register pressure
thread partition
IME instruction selection
fallback to RVV for unsupported shapes or dtype
```

Tuning metadata must stay variant-local and must not pollute high-level semantics.

## Emission Paths

IME emission currently follows:

```text
TCRV IME family ops
  -> EmitC
  -> IME/vendor intrinsic C/C++
```

All IME-specific emission logic belongs inside the IME plugin. Compiler
builtin, inline asm, external assembly stub, or patched backend adapter routes
are future optional compatibility paths.

## Plugin Evaluation

When IME becomes available, record:

- core pass modified LOC;
- new capabilities;
- new ops/types;
- generated high-level op coverage;
- whether same high-level op gains IME variant;
- whether IME and RVV variants coexist;
- whether selection/dispatch reuse core logic;
- whether emission stays plugin-local.
