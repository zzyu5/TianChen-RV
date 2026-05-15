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
generic RISC-V selected march/mabi facts
LLVM RVV scalable vector support
RVV intrinsic support
compiler builtin support
inline asm allowed
vendor header available
patched compiler available
runtime library linkable
```

## Parameter Layering Rule

RVV, variant, lowering, runtime, and artifact work must keep parameter meaning
layered. A parameter may cross layers only through an explicit compiler object
or ABI surface that states the new meaning.

1. Hardware facts / target capabilities belong in capability, profile, or probe
   objects and constrain legality and selection. Examples include VLEN,
   vlenb-derived vector capacity, ISA/profile facts, hart or core count,
   toolchain availability, remote probe evidence, and capability provenance.
   A plugin-local hart-count fact may expose the generic relation id
   `target.hart_count` with positive integer property `count` so
   target-neutral core checks can consume capacity without branching on a
   concrete extension. RVV `rvv.i32_m1_lane_count` is a base i32 M1 capacity
   fact derived from vlenb/profile evidence; when serialized into plugin
   metadata it must use a name such as `base_i32_m1_lanes` so an i32m2 selected
   path cannot confuse the hardware capacity fact with selected m1 vector
   config. Finite RVV i64m1 profile facts such as `rvv.i64_m1.sew64`,
   `rvv.i64_m1.lmul_m1`, `rvv.i64_m1.tail_policy.agnostic`, and
   `rvv.i64_m1.mask_policy.agnostic` are compile-time capability/profile facts
   that constrain i64 family legality; they are not runtime `n`, AVL/VL, or
   descriptor-local element-count claims.
2. Compile-time variant config belongs in plugin-proposed variant metadata,
   selected config, tuning, or lowering-boundary metadata and must be checked
   against target capability. Examples include SEW, LMUL, tail policy, mask
   policy, unroll, selected vector type/suffix, setvl suffix, and selected
   lowering strategy. RVV selected vector-shape config is plugin/target-owned
   metadata and must not become `tcrv.exec` compute semantics.
3. Runtime SSA values / runtime control values belong in real IR or ABI
   surfaces: SSA values, region or block arguments, op attributes that
   explicitly mean ABI/control values, or generated C ABI parameters. Examples
   include AVL, vl, pointer arguments, length `n`, `rvv_available`, and
   dispatch guard parameters. These are not target capabilities or
   compile-time variant constants.
4. Legacy bounded fixture parameters may describe selected-path metadata for a
   historical or fail-closed slice only. They must not masquerade as high-level
   tensor shape, global problem size, AVL, vl, source authority, production
   input, or target artifact authority unless real IR carries that meaning.

Emission plans, manifests, diagnostics, and generated artifacts must not claim
that a parameter is IR-modeled unless the real IR has the corresponding
attribute, type, SSA value, region argument, or ABI parameter.

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
isa-vector-config
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
`FlatSymbolRefAttr` references to symbols in the kernel capability scope.
Capability providers are direct `tcrv.exec.capability` ops, kernel-local
`tcrv.exec.target` profile anchors that carry both non-empty `id` and `kind`
attributes, one explicit module-level `tcrv.exec.target` profile referenced by
`tcrv.exec.kernel target = @profile`, or module-level providers explicitly
named by that profile's generic `capability_providers = [@provider, ...]`
composition. A `tcrv.exec.target` with no capability identity remains a
parse-only profile anchor and must not satisfy `requires`.

Module-level target profiles are opt-in per kernel. A kernel receives only the
single profile named by `target = @profile` plus only the providers explicitly
named by that profile's `capability_providers` composition; unrelated
module-level targets do not enter `TargetCapabilitySet` and cannot affect
plugin availability, proposal, legality, selection, dispatch, lowering, or
artifact routing. The referenced target profile must be a direct module-level
`tcrv.exec.target` with non-empty `id` and `kind`, and it must not be shadowed
by a direct symbol with the same name inside the kernel. Provider composition
refs must resolve to module-level `tcrv.exec.capability` or
capability-provider `tcrv.exec.target` symbols, carry non-empty capability
identity, reject duplicate symbols/ids, reject self references, and reject
obvious nested target cycles.

Provider `id` and `kind` are non-empty `StringAttr` fields. Provider `id`
values are unique within one kernel capability scope, including the referenced
module-level target profile, its composed providers, and direct kernel-local
providers. Duplicate ids are invalid because C++ capability queries use id
lookup for plugin proposal, legality, selection, and conflict decisions. Core
verification checks structure, identity uniqueness, provider composition, and
symbol resolution; concrete extension legality stays plugin-owned.

Capability query passes may also consume a generic string `status` attribute
or, equivalently, a generic string `availability` attribute on
`tcrv.exec.capability` or a capability-provider `tcrv.exec.target`. Missing
status means present/available. The generic strings `unavailable`, `disabled`,
and `missing` mean unavailable for core requires checks. Core code must not
interpret concrete target-family status values; extension-specific
availability semantics remain plugin-owned.

Non-core attributes on `tcrv.exec.capability` and capability-provider
`tcrv.exec.target` ops are structured capability properties.
`TargetCapabilitySet::buildFromKernel` must preserve them in the C++
`CapabilityDescriptor` property map so microarchitecture, dtype, toolchain,
runtime/offload mode, ABI, and cost/probe facts remain available to compiler
decisions. Core identity/status fields (`sym_name`, `id`, `kind`, `status`,
`availability`) stay first-class descriptor fields and must not be duplicated
as generic properties. Property values are deterministic textual renderings of
MLIR attributes for diagnostics and plugin decisions; they are derived from
structured IR attributes, not from comments, JSON blobs, or Python-only
records.

## Generic Capability Query Contract

### 1. Scope / Trigger

Use this contract when a core C++ pass needs target-independent capability
availability from `tcrv.exec.kernel` without invoking plugin-specific legality.

### 2. Signatures

- Query API: `TargetCapabilitySet::buildFromKernelChecked(tcrv::exec::KernelOp)`
  for compiler consumers that need diagnostic-bearing construction failure;
  `buildFromKernel` remains a convenience wrapper only for already-verified
  contexts.
- Pass command: `--tcrv-check-capability-requires`.

### 3. Contracts

- Each query descriptor records capability symbol name, `id`, `kind`, generic
  status, and available/unavailable state.
- Direct `tcrv.exec.capability` ids, kernel-local capability-provider
  `tcrv.exec.target` ids, the explicit module-level profile named by
  `tcrv.exec.kernel target = @profile`, and that profile's composed providers
  are unique within one kernel capability scope before a `TargetCapabilitySet`
  is treated as a compiler decision object.
- `TargetCapabilitySet` construction itself is fail-closed for duplicate
  owning capability ids and duplicate capability symbol names. Parsed IR should
  be rejected first by the `tcrv.exec.kernel` verifier; synthetic C++
  construction and kernel extraction paths must return a diagnostic-bearing
  compiler error instead of silently keeping the first descriptor, overwriting
  an existing descriptor, or letting ambiguous by-id/by-symbol lookup reach
  plugins or variant planning.
- Each query descriptor preserves additional non-core MLIR attributes as
  structured property entries queryable by property name.
- Each query descriptor preserves `provides`, `implies`, and `conflicts` as
  first-class relation lists rather than generic property-map entries.
- Lookup must be available by capability symbol name and by capability `id`.
- Relation-aware provider lookup must be available by exact id, provided id,
  and implied id. Exact id lookup is authoritative when present; relation
  providers are used only when there is no direct capability with that id.
- The generic id `target.hart_count` is reserved for target-neutral hart/core
  count decisions. A provider may keep a plugin-local owning id such as
  `rvv.hart_count` and list `provides = ["target.hart_count"]`; compiler
  consumers must read the positive integer `count` property and must not infer
  runtime thread behavior from the capability alone.
- Collection/query by `kind` must treat kind values as an open string set.
- Missing generic status means available/present.
- `status` takes precedence over `availability` when both are present.

### 4. Validation & Error Matrix

- `requires = [@cap]` and `@cap` has missing status -> pass succeeds.
- `requires = [@cap]` and `@cap` has `status = "available"` -> pass succeeds.
- `requires = [@cap]` and `@cap` has `status = "unavailable"` -> pass fails.
- `requires = [@cap]` and `@cap` has `status = "disabled"` -> pass fails.
- `requires = [@cap]` and `@cap` has `status = "missing"` -> pass fails.
- An unavailable requirement on a static variant not referenced by a structured
  `tcrv.exec.case` -> pass fails.
- An unavailable requirement on a `tcrv.exec.case` target may pass only when
  that case carries typed generic `runtime_guard_required = true`; the core
  pass records that guard requirement without parsing printable
  `condition`/`guard`/`policy` strings.
- An unavailable requirement on a `tcrv.exec.fallback` target -> pass fails,
  because fallback must remain generically executable under the same
  `TargetCapabilitySet`.
- A static variant requirement whose available capability conflicts with
  another available capability in the same kernel -> pass fails. Conflict ids
  are capability ids and may be satisfied by exact id, `provides`, or
  `implies` relations.
- A `tcrv.exec.case` target with a conflicting required capability may pass
  only when that case carries typed generic `runtime_guard_required = true`;
  the core pass records that guard requirement but does not solve or parse the
  conflict.
- A `tcrv.exec.fallback` target with a conflicting required capability -> pass
  fails, because fallback must remain generically executable without relying on
  a preferred-path guard.
- malformed `requires` or unknown `@cap` -> existing `tcrv.exec` verifier owns
  the diagnostic, not the capability query pass.
- `tcrv.exec.hart_parallel {harts = N}` with no available provider for
  `target.hart_count` -> `--tcrv-check-hart-parallel-capabilities` fails.
- `tcrv.exec.hart_parallel {harts = N}` with a provider whose `count < N` ->
  `--tcrv-check-hart-parallel-capabilities` fails.
- `tcrv.exec.hart_parallel` without explicit `harts` -> the hart-count
  capability check does not create a concrete capacity request.

### 5. Good/Base/Bad Cases

- Good: a runtime-offload or toolchain capability is declared generically and
  the pass rejects a requiring variant when it is disabled.
- Good: a runtime-dispatch case can reference a generically unavailable
  variant when the case has typed `runtime_guard_required = true` and the
  fallback variant remains available.
- Good: a dispatch case can reference a capability that conflicts with an
  available build/runtime policy only when typed `runtime_guard_required`
  records the dispatch protection surface.
- Base: a capability has no status field and is treated as present.
- Bad: core code branches on target-family names such as RVV, IME, or Sophgo to
  decide generic requires availability.

### 6. Tests Required

- lit/FileCheck positive test for available requirements.
- lit/FileCheck negative test for declared-but-unavailable requirements.
- C++ smoke coverage for symbol-name lookup, id lookup, kind collection, and
  generic availability status handling when textual IR cannot directly assert
  helper API behavior.
- C++ smoke coverage for `TargetCapabilitySet` checked construction, including
  duplicate owning id rejection, duplicate symbol-name rejection when symbol
  lookup is represented, and atomic failure that preserves existing lookup
  results.
- C++ smoke coverage for textual MLIR capability properties, including scalar
  facts such as hart/VLEN counts and runtime-offload mode/ABI metadata.
- C++ smoke coverage for relation lists and relation-aware provider lookup.
- C++ smoke coverage for bounded conflict queries, including direct conflict
  ids and relation-satisfied conflict providers.
- lit/FileCheck coverage that at least one plugin proposal/materialization path
  is driven by a structured capability provider relation rather than only an
  exact capability id.
- lit/FileCheck and/or C++ smoke coverage that a module-level
  `tcrv.exec.target` referenced by `tcrv.exec.kernel target = @profile` enters
  `TargetCapabilitySet` and can satisfy plugin proposal/materialization through
  relation-aware provider lookup.
- lit/FileCheck coverage that a module-level target profile's
  `capability_providers` composition makes additional providers visible to
  planning without a frontend-specific provider list, plus focused negative
  coverage for malformed composition such as unresolved provider refs or
  duplicate provider identity.
- lit/FileCheck coverage for static conflict rejection, guarded dispatch-case
  conflict allowance, unguarded dispatch-case conflict rejection, and fallback
  conflict rejection.

### 7. Wrong vs Correct

Wrong:

```cpp
if (target.hasRVV()) { ... }
```

Correct:

```cpp
TargetCapabilitySet capabilities =
    TargetCapabilitySet::buildFromKernel(kernel);
if (!capabilities.isCapabilityAvailableBySymbolName(requiredSymbol))
  variant.emitError() << "requires unavailable capability";
```

### provide

Plugin declares provided capabilities:

```text
RVV plugin: rvv, vector-stripe, scalable-vl, rvv-load-store, rvv-reduce
IME plugin: ime, vector-register-backed-matrix, ime-frag-mma
Offload plugin: sophgo-runtime, async-offload, runtime-buffer
```

The current C++/MLIR relation slice represents provided capability ids with a
structured `provides = ["..."]` array on `tcrv.exec.capability`. These entries
are capability ids, not symbol names and not prose strings. They are parsed into
first-class `CapabilityDescriptor` relation fields by
`TargetCapabilitySet::buildFromKernel`.

`TargetCapabilitySet::lookupProviderByID(id)` resolves an exact capability id
first. If no exact id is declared, it may resolve an available capability whose
`provides` or `implies` relation satisfies that id. This lets a profile
capability such as `id = "rvv.profile.rv64gcv", provides = ["rvv"]` satisfy a
plugin proposal requiring capability id `rvv`, while preserving deterministic
direct-id override behavior when an exact `id = "rvv"` capability is present.

### imply

Examples:

```text
rv64gcv implies rvv
zvfh implies fp16 vector arithmetic, subject to toolchain support
spacemit.ime implies vector-register-backed matrix capability, subject to vendor toolchain support
```

The current relation slice represents implied capability ids with
`implies = ["..."]` and exposes them through the same relation-aware provider
lookup and availability query APIs as `provides`. `implies` is bounded to
compiler decision routing in this slice; it does not implement a full
capability lattice, target-family inference engine, or performance model.

### conflict

Examples:

```text
variant requires vendor runtime but target lacks runtime library
variant requires inline asm but build policy forbids inline asm
offload variant requires fixed shape but input shape is dynamic or unsupported
```

The current relation slice preserves `conflicts = ["..."]` as first-class
descriptor relation metadata and verifies that it is a structured array of
non-empty capability id strings. `TargetCapabilitySet` exposes a bounded
conflict query for an available required capability: each declared conflict id
is matched against available exact/provided/implied providers in the same
kernel, and the reverse direction is also checked when another available
capability declares a conflict id satisfied by the required capability.

`--tcrv-check-capability-requires` uses that bounded query as a legality gate.
Static variants and dispatch fallbacks fail closed when a required capability
conflicts with another available capability. Dispatch cases may reference a
conflicting requirement only when the case has typed generic
`runtime_guard_required = true`, which records the runtime-dispatch protection
surface without parsing extension-specific printable annotations.

This is not a full conflict solver, profile lattice, provider ranking model,
or automatic conflict-resolution strategy. It only prevents unprotected
conflicting requirements from reaching later selection, lowering, and artifact
export stages.

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
