# RVV Plugin

## Role

RVV plugin is the current primary real hardware extension family for
TianChen-RV MLIR. It is part of the unified TCRV system, not an independent
backend dialect.

Environment:

```text
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: primary development, primary performance, primary correctness
```

## Responsibilities

RVV plugin provides:

- RVV capability registration;
- `tcrv.rvv` dialect registration;
- RVV variant generation;
- RVV legality verification;
- RVV tuning space;
- RVV cost model;
- RVV emission paths;
- RVV runtime/threading integration.

It does not define high-level matmul semantics, generic tensor/tile IR, IME internals, offload internals, or custom ISA internals.

## Current C++ Plugin Slice

The current concrete C++ RVV slice is intentionally narrower than the full RVV
plugin above. It proves plugin identity, capability participation, typed RVV
dialect registration, explicit typed-variant legality routing, and selection
preference through `ExtensionPluginRegistry`.

The deleted metadata-only proposal route is not active compiler authority:
bare high-level/no-body RVV capability evidence must not produce an RVV
proposal, selected variant, lowering boundary, emission plan, runtime ABI,
artifact route, or route metadata. A selected RVV variant is legal only when it
already contains explicit typed `tcrv_rvv` extension-family IR. That typed IR
is still non-executable until a future EmitC/runtime route is implemented.

Stable current names:

```text
plugin name: rvv-plugin
plugin version: 0.1.0
plugin capability id: rvv
plugin capability kind: isa-vector
preferred kernel capability symbol: @rvv
variant origin: rvv-plugin
finite capability ids:
  rvv
  rvv.i32_m1.sew32
  rvv.i32_m1.lmul_m1
  rvv.i32_m1.tail_policy.agnostic
  rvv.i32_m1.mask_policy.agnostic
  rvv.i32_m2.sew32
  rvv.i32_m2.lmul_m2
  rvv.i32_m2.tail_policy.agnostic
  rvv.i32_m2.mask_policy.agnostic
  rvv.i64_m1.sew64
  rvv.i64_m1.lmul_m1
  rvv.i64_m1.tail_policy.agnostic
  rvv.i64_m1.mask_policy.agnostic
optional finite i32 binary selected vector-shape selector capability id:
  rvv.i32_binary.selected_vector_shape
selector property:
  shape = "i32m1" | "i32m2"
preferred finite config symbols:
  @rvv_i32_m1_sew32
  @rvv_i32_m1_lmul_m1
  @rvv_i32_m1_tail_agnostic
  @rvv_i32_m1_mask_agnostic
  @rvv_i32_m2_sew32
  @rvv_i32_m2_lmul_m2
  @rvv_i32_m2_tail_agnostic
  @rvv_i32_m2_mask_agnostic
explicit typed variant form: a hand-authored or future materialized RVV variant
  must contain real `tcrv_rvv` ops in its body before RVV plugin legality or
  selection preference can accept it
typed policy attr name: tcrv_rvv.policy
typed policy attr value: #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
deleted legacy element-count attr name: tcrv_rvv.element_count
optional vlenb attr name: tcrv_rvv.vlenb_bytes
optional base i32 m1 lane attr name: tcrv_rvv.base_i32_m1_lanes
selected vector-shape attr names:
  tcrv_rvv.selected_vector_shape
  tcrv_rvv.selected_vector_sew
  tcrv_rvv.selected_vector_lmul
  tcrv_rvv.selected_tail_policy
  tcrv_rvv.selected_mask_policy
  tcrv_rvv.selected_vector_type
  tcrv_rvv.selected_vector_suffix
  tcrv_rvv.selected_setvl_suffix
```

RVV probe facts remain bounded hardware/toolchain evidence inputs. They may be
validated into raw `TargetCapabilitySet` evidence facts, but they must not
manufacture finite SEW/LMUL/tail/mask config capabilities or authorize a
compiler route unless an explicit typed RVV variant body already exists.
Proposal collection for a no-body RVV-capable kernel records a recoverable
decline and produces no RVV proposal. Materialized RVV variant legality is
strict: the variant must be owned by `origin = "rvv-plugin"` and contain real
`tcrv_rvv` operations in its body. Metadata-only RVV attributes, finite
capability facts, selected vector shape facts, or profile evidence alone are
not sufficient.

The RVV plugin must not create any binary-family proposal from a no-body
`tcrv.exec.kernel`, from deleted frontend metadata, from finite-family registry
metadata, or from hand-authored microkernel names alone. Kernel-based
executable planning requires a future explicit extension-family op contract
and a materialized EmitC route before selecting family, dtype, route metadata,
callable ABI, artifact kind, or emitted body.

## Scenario: Capability-Backed RVV i32m1 Config Policy Slice

### 1. Scope / Trigger

This scenario applies when the RVV plugin proposes, materializes, verifies, or
exports the bounded i32 add/sub/mul first slice. It is a cross-layer contract:
target capability profile facts, variant `requires`, selected
lowering-boundary validation, and emission readiness must agree on the same
SEW/LMUL/tail/mask policy ids. Remote probe output does not create those
compiler config facts.

### 2. Signatures

- C++ capability ids: `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic`.
- Preferred MLIR symbols from explicit target/profile fixtures:
  `@rvv_i32_m1_sew32`, `@rvv_i32_m1_lmul_m1`,
  `@rvv_i32_m1_tail_agnostic`, and `@rvv_i32_m1_mask_agnostic`.
- Variant metadata: `requires` must satisfy `rvv` plus all four first-slice
  config/policy ids; `tcrv_rvv.policy` must be
  `#tcrv_rvv.policy<tail = agnostic, mask = agnostic>`.

### 3. Contracts

- `rvv.i32_m1.sew32` requires `sew_bits = 32 : i64`.
- `rvv.i32_m1.lmul_m1` requires `lmul = "m1"`.
- `rvv.i32_m1.tail_policy.agnostic` requires
  `tail_policy = "agnostic"`.
- `rvv.i32_m1.mask_policy.agnostic` requires
  `mask_policy = "agnostic"`.
- Runtime `n`, AVL, VL, and dispatch guard values remain runtime SSA/control
  or ABI values. They must not be encoded as these compile-time config
  capability facts. Deleted descriptor-local `tcrv_rvv.element_count` metadata
  must not be reintroduced as a runtime trip count or artifact descriptor.

### 4. Validation & Error Matrix

- Missing `rvv` provider -> plugin decline during proposal collection.
- Bare RVV capability evidence without an explicit typed RVV variant body ->
  no RVV proposal and no selected RVV route.
- Explicit selected RVV variant without `origin = "rvv-plugin"` -> fatal
  legality error.
- Explicit selected RVV variant without `tcrv_rvv` ops in its body -> fatal
  legality error.
- Capability/profile facts may still be validated for replay and typed-body
  checks, but they must not create proposal, boundary, emission-plan, runtime
  ABI, artifact, or route metadata authority by themselves.

### 5. Good/Base/Bad Cases

- Good: a hand-authored or future materialized variant contains real
  `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` control-plane IR in the variant body
  and is owned by `origin = "rvv-plugin"`.
- Base: probe-derived capability facts are replayed as bounded capabilities
  but produce no RVV variant unless explicit typed RVV IR is present.
- Bad: `@rvv` is available but the kernel has no explicit typed RVV body; this
  must remain no-proposal/no-boundary/no-emission.

### 6. Tests Required

- RVV plugin unit/lit tests must assert no-body RVV capability input produces
  no RVV proposal.
- Legality tests must reject stale metadata-only RVV variants that do not
  contain explicit typed `tcrv_rvv` ops.
- Lowering-boundary, emission-readiness, manifest, target export, and probe
  replay tests must not require RVV boundary, runtime ABI, artifact, or route
  metadata from bare capability evidence.
- Probe and probe-to-MLIR self-tests may assert replayed capability facts are
  preserved, but legality remains in C++ and no compiler route is synthesized.

### 7. Wrong vs Correct

Wrong:

```mlir
tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector",
                           status = "available"}
tcrv.exec.variant @rvv_stale_metadata attributes {
  origin = "rvv-plugin",
  requires = [@rvv],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {}
```

Correct for a relation-provider fixture:

```mlir
tcrv.exec.capability @rvv {
  id = "rvv",
  kind = "isa-vector",
  provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1",
              "rvv.i32_m1.tail_policy.agnostic",
              "rvv.i32_m1.mask_policy.agnostic"],
  sew_bits = 32 : i64,
  lmul = "m1",
  tail_policy = "agnostic",
  mask_policy = "agnostic",
  status = "available"
}
tcrv.exec.variant @rvv_explicit_body attributes {
  origin = "rvv-plugin",
  requires = [@rvv],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {
  %n = "builtin.unrealized_conversion_cast"() : () -> index
  %vl = tcrv_rvv.setvl %n {
    lmul = "m1",
    policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
    sew = 32 : i64
  } : index -> !tcrv_rvv.vl
  tcrv_rvv.with_vl %vl attributes {
    lmul = "m1",
    policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
    sew = 32 : i64
  } {
  } : !tcrv_rvv.vl
}
```

Correct for a replay/profile fixture:

```mlir
tcrv.exec.variant @rvv_explicit_profile_body attributes {
  origin = "rvv-plugin",
  requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1,
              @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {
  %n = "builtin.unrealized_conversion_cast"() : () -> index
  %vl = tcrv_rvv.setvl %n {
    lmul = "m1",
    policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
    sew = 32 : i64
  } : index -> !tcrv_rvv.vl
  tcrv_rvv.with_vl %vl attributes {
    lmul = "m1",
    policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
    sew = 32 : i64
  } {
  } : !tcrv_rvv.vl
}
```

The former plugin-local standalone smoke-probe source route is deleted.
Availability of plugin-local smoke-probe capability facts must not select a
compiler source artifact path, materialize an executable RVV microkernel,
create runtime callable ABI parameters, claim kernel lowering, or produce
correctness/performance evidence. RVV hardware/toolchain probe evidence belongs
in separate probe tooling and recorded `ssh rvv` artifacts.

When a relation-provider capability satisfies id `rvv`, generic variant
materialization records the provider symbol in `requires`, for example
`requires = [@rvv_profile]`. RVV legality must treat that provider as satisfying
the RVV id through the C++ `CapabilityDescriptor::satisfiesID("rvv")` relation
API; it must not require an exact `id = "rvv"` capability when a structured
provider relation is present.

The former finite descriptor selected microkernel materialization route is
deleted. A selected `rvv-plugin` path must not materialize plugin-owned
selected-boundary metadata or synthesize a plugin-local direct kernel-child
`tcrv_rvv.*_microkernel`, `setvl`/`with_vl`,
load/arithmetic/store body, or callable runtime ABI boundary from finite family
records, legacy route helpers, lowering tokens, route ids, or descriptor
mirrors. Until the rebuild supplies explicit extension-family IR plus a
materialized MLIR EmitC module route, deleted RVV wrapper attachments and
descriptor-derived callable ABI data are fail-closed historical inputs, not
active emission authority.

The former microkernel direct C slice is deleted as production authority. If
the selected `rvv-plugin` path reaches emission planning, `RVVExtensionPlugin`
must report an unsupported emission plan for runtime-callable source until the
rebuild provides a materialized MLIR EmitC module route. Target/export code
must not synthesize RVV compute C bodies from selected metadata, family
records, route records, or deleted wrapper records. RVV source/object/header
routes must fail closed instead of producing
`riscv_vector.h` intrinsic source, relocatable objects, or self-check harnesses
from the old direct printer path.

## Remote Evidence Probe Contract

The repo-owned RVV evidence probe is `scripts/rvv_remote_probe.py`. It is
Python runner/evidence tooling and must not be used as the implementation of
capabilities, plugin registry behavior, legality, lowering, emission, runtime
ABI, or compiler-generated executable paths.

Probe artifacts live under:

```text
artifacts/tmp/rvv_probe/<run-id>/
  rvv_probe_evidence.json
  logs/*.log
```

The JSON artifact records a bounded schema:

- `schema_version`, `probe_name`, `run_id`, timestamp, ssh target, artifact dir,
  status, and success boolean;
- hardware/toolchain facts for `uname`, architecture, hart count, clang,
  cmake, bounded RISC-V/vector hints from `/proc/cpuinfo`, and non-interactive
  sudo availability as a boolean capability fact;
- a sanitized `capability_facts` section containing only bounded evidence
  facts: architecture, hart count, optional vlenb bytes and
  derived i32 m1 lane count, ISA/vector hint string, clang and CMake
  availability/version facts, minimal RVV compile/run success, selected
  march/mabi, and optional source/binary digests;
- a minimal RVV intrinsic compile/run probe result with command references,
  exit status, diagnostics, compiler path/version, source digest, selected
  compiler flags, and binary digest when a binary was produced;
- sanitized command logs that avoid secrets, credentials, private keys,
  environment tokens, and unrelated raw environment dumps.

Interpretation rules:

- Successful `ssh rvv` probe output is hardware/toolchain/probe-program
  evidence only.
- The transition from probe evidence to compiler capabilities is
  `sanitized capability_facts -> plugin-local C++ RVV capability profile ->
  TargetCapabilitySet`. The Python probe may emit artifacts, but it must not
  implement capability relations, legality, selection, lowering, or emission.
- The RVV C++ capability profile must validate facts before producing any
  `TargetCapabilitySet`. Required gates include `riscv64`, positive hart count,
  RVV ISA/vector hints, clang and CMake availability, and minimal RVV
  compile/run success. Negative cases must return structured diagnostics rather
  than partial target capabilities.
- Profile-derived capability identities are stable and plugin-local raw
  evidence identities. Current probe-derived profile IDs include `rvv`,
  `rvv.hart_count`, `rvv.vlenb_bytes`, `rvv.i32_m1_lane_count`,
  `rvv.toolchain.clang`, `rvv.toolchain.cmake`, `rvv.probe.compile_run`,
  `rvv.toolchain.march`, and `rvv.toolchain.mabi`. Finite SEW/LMUL/tail/mask
  config IDs such as `rvv.i32_m1.sew32` belong to explicit RVV vector-shape
  config/profile fixtures or plugin-selected variant requirements, not to
  probe-derived profile construction. These identities must not include
  ssh/provider names, raw command logs, secrets, benchmark names, or
  performance measurements.
- The probe does not prove that TianChen-RV generated RVV IR, lowered a
  `tcrv.exec` variant, emitted an object, linked runtime glue, proved compiler
  correctness, or measured performance.
- If clang, RVV headers, candidate flags, or remote execution are unavailable,
  the artifact must record failure with exact non-secret command diagnostics
  rather than synthesizing success.
- Future RVV emission beyond the finite explicit i32 add/sub/mul microkernel
  routes requires plugin-local lowering/runtime implementation and successful
  named `ssh rvv` evidence; probe artifacts alone do not create broader
  supported RVV emission.

## Deleted Remote Evidence Replay Route

The Python RVV probe-to-MLIR replay route is deleted as active compiler
authority. Repo-owned Python probe tooling may collect, sanitize, validate, and
record bounded RVV hardware/toolchain evidence, but it must not translate probe
JSON into `tcrv.exec` MLIR capability, target, kernel, variant, selected route,
or scalar fallback modeling.

Compiler-visible RVV capability/profile behavior must be implemented by the
C++/MLIR RVV extension-family/plugin path. The transition from a probe artifact
to compiler capabilities is:

```text
sanitized evidence artifact
  -> plugin-local C++ RVV capability profile validation
  -> TargetCapabilitySet
```

There is no supported Python fallback route, hidden compatibility mode, or
schema adapter that preserves the old replay helper. Tests must not pipe
Python-generated RVV replay MLIR into `tcrv-opt` as compiler-path evidence.
First-slice SEW/LMUL/tail/mask policy facts are plugin-selected compiler config
facts, not remote-probe output fields.

`registerDialects` now registers the minimal RVV dialect skeleton through the
RVV plugin path. The default `registerAllDialects` path remains core-only; RVV
dialect availability is proven by populating an `ExtensionPluginRegistry` with
the RVV plugin and calling `registerPluginDialects`.

The current RVV dialect slice is metadata/control-plane plus bounded explicit
dataflow only; the former executable microkernel wrapper ops are deleted. It
introduces the vector-length token type `!tcrv_rvv.vl`, the finite policy
attribute
`#tcrv_rvv.policy<tail = agnostic|undisturbed, mask =
agnostic|undisturbed>`, the bounded runtime AVL-to-VL control-plane operation
`tcrv_rvv.setvl`, the bounded VL scope region operation `tcrv_rvv.with_vl`,
the finite `tcrv_rvv.i32_load`, `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`,
`tcrv_rvv.i32_mul`, and `tcrv_rvv.i32_store` ops nested under that scope for
non-executable i32 dataflow modeling, and the corresponding bounded i64
dataflow ops. The previous plugin-local selected lowering-boundary operation
is deleted as active compiler authority. The setvl op
consumes a runtime AVL SSA value, produces a `!tcrv_rvv.vl` token, and carries
only bounded first-slice compile-time config metadata: SEW 32, LMUL m1 or m2,
and the finite policy attribute. The with_vl op consumes one `!tcrv_rvv.vl` value and
owns one single-block region for bounded RVV control/body work. Optional
duplicated SEW/LMUL/policy metadata is limited to the same bounded first-slice
config and must agree with the visible defining setvl when present. The bounded
i32 add/sub/mul dataflow body carries finite runtime ABI role references on
the explicit lhs load, rhs load, and output store operations; concrete C
parameter names are not resolved into executable artifacts until a future
EmitC route rebuild provides the required ABI boundary. It is not a generic
vector memory model. These surfaces are not generic RVV arithmetic, generic
memory operations, LLVM/RISC-V lowering, full runtime ABI glue, hardware
execution, correctness evidence, or performance evidence. `tcrv_rvv`
is the concrete MLIR dialect namespace because MLIR dialect namespaces cannot
contain `.` characters; the architectural extension family remains `tcrv.rvv`.

## Capability Fields

RVV plugin should register and query:

```text
rvv
rvv.version
rvv.i32_m1.sew32
rvv.i32_m1.lmul_m1
rvv.i32_m1.tail_policy.agnostic
rvv.i32_m1.mask_policy.agnostic
rvv.i32_m2.sew32
rvv.i32_m2.lmul_m2
rvv.i32_m2.tail_policy.agnostic
rvv.i32_m2.mask_policy.agnostic
rvv.i32_binary.selected_vector_shape
vlen
elen
supported SEW
supported LMUL
mask support
tail policy support
Zvfh / Zvfbfmin / other vector dtype extensions
LLVM scalable vector support
RVV intrinsic support
inline asm policy
thread runtime availability
```

Reference attribute:

```mlir
#tcrv.ext<"rvv",
          kind = "isa-vector",
          version = "1.0",
          vlen = 128,
          supports_mask = true,
          supports_tail_policy = true>
```

## Current Dialect Skeleton

Architectural family:

```text
tcrv.rvv
```

Concrete MLIR namespace:

```text
tcrv_rvv
```

Current first-slice type:

```text
!tcrv_rvv.vl
!tcrv_rvv.i32m1
!tcrv_rvv.i32m2
```

Current first-slice policy attribute:

```text
#tcrv_rvv.policy<tail = agnostic, mask = agnostic>
```

Current first-slice runtime VL control-plane op:

```mlir
%vl = tcrv_rvv.setvl %avl {
  lmul = "m1",  // or "m2" for the finite i32m2 slice
  policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
  sew = 32 : i64
} : index -> !tcrv_rvv.vl
```

Current first-slice VL scope control-plane op:

```mlir
tcrv_rvv.with_vl %vl attributes {
  lmul = "m1",  // or "m2" for the finite i32m2 slice
  policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
  sew = 32 : i64
} {
} : !tcrv_rvv.vl
```

The type is a non-compute vector-length token used by the bounded setvl and
with_vl surfaces and parser/printer ownership tests. The policy attribute is
finite non-compute metadata for proposal preservation, setvl/with_vl
configuration, and RVV plugin-local legality. The setvl op is the first bounded
runtime VL control-plane surface: AVL is a runtime SSA operand, vl is a
`!tcrv_rvv.vl` result, SEW/LMUL/policy are compile-time config metadata for
this first slice, and VLEN/vlenb, `element_count`, `required_march`, and
`required_capabilities` are explicitly not accepted on the op. The with_vl op
is the bounded structural companion to setvl: it consumes one runtime VL SSA
token, owns one single-block region with no region arguments, may repeat only
the same bounded SEW/LMUL/policy config, and rejects VLEN/vlenb,
`element_count`, `required_march`, `required_capabilities`, and raw capability
facts. The former selected-boundary op and typed RVV microkernel wrappers are
deleted as structural authority; future executable emission must be rebuilt
through explicit extension-family IR plus a materialized MLIR EmitC module
route.
These surfaces are not vector registers, masks, memory operations, RVV
intrinsics, LLVM/RISC-V lowering, runtime ABI, executable emission, correctness
evidence, or performance evidence.

## RVV Parameter Boundary

RVV work must keep these parameter layers distinct:

- VLEN and vlenb are hardware facts / target capability evidence. They may
  constrain legality, vector capacity, and selection after provenance is
  validated, including plugin-owned `tcrv_rvv.vlenb_bytes` and
  `tcrv_rvv.base_i32_m1_lanes` variant metadata. The base lane attribute is the
  i32 M1 capacity implied by vlenb, even when the selected vector shape is
  i32m2. It is not selected m1 config metadata and is not a per-variant runtime
  value.
- SEW, LMUL, tail policy, and mask policy are compile-time variant config
  selected or proposed by the RVV plugin and checked against target
  capabilities. The current non-executable bounded RVV dataflow slice admits only SEW 32 with
  LMUL m1 or m2, tail agnostic, and mask agnostic. The m1 shape is backed by
  `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic`; the m2 shape is backed by
  `rvv.i32_m2.sew32`, `rvv.i32_m2.lmul_m2`,
  `rvv.i32_m2.tail_policy.agnostic`, and
  `rvv.i32_m2.mask_policy.agnostic`. A selected variant must require exactly
  one shape, and future EmitC/export routes must reject selected capability/body
  LMUL mismatches before artifact bytes are emitted. These config ids are not
  sufficient as standalone hardware facts without the surrounding
  RVV/profile/toolchain evidence.
- The selected vector-shape config is the target-owned serialization of that
  compile-time choice. Materialized explicit RVV variants, selected-plan
  metadata, and any future generated RVV C source comments must agree on
  `selected_vector_shape`,
  `selected_vector_sew`, `selected_vector_lmul`, `selected_tail_policy`,
  `selected_mask_policy`, `selected_vector_type`,
  `selected_vector_suffix`, and `selected_setvl_suffix`. If any selected-shape
  attribute is present, the complete group is required. An i32m2 selected path
  must not carry stale selected i32m1 config even though it may still carry the
  separate `base_i32_m1_lanes` capacity fact.
- RVV arithmetic family identity and selected vector-shape suffix are separate
  contracts. The i32 add/sub/mul family registry owns only the suffix-free
  arithmetic intrinsic prefix; the target exporter appends the selected
  vector-shape suffix after validating the selected shape group. A full
  intrinsic string such as `__riscv_vadd_vv_i32m2` is therefore an emission
  result, not a family descriptor field.
- AVL and vl are runtime SSA values / runtime control values. The current
  bounded `tcrv_rvv.setvl` surface models AVL as a real runtime SSA operand and
  vl as a real `!tcrv_rvv.vl` result. The bounded `tcrv_rvv.with_vl` surface
  models vl only when it consumes a real `!tcrv_rvv.vl` SSA operand. It must not
  imply that AVL or vl is IR-modeled unless a real op attribute, SSA value,
  region argument, or ABI parameter exists.
- `tcrv_rvv.element_count` is deleted legacy selected-path metadata for RVV.
  Active RVV dialect ops must reject local `element_count` attributes; the
  value does not describe a valid descriptor, emitted source slice, production
  input, high-level MLIR tensor shape, global problem size, AVL, or vl.
- `tcrv_rvv.required_march` string matching is a bounded plugin-owned
  compatibility bridge for the current first slice. Do not expand dependence on
  `required_march` string comparisons when structured capabilities or
  properties are available or should be added.

Emission plans and manifests for RVV paths must not claim VLEN, vlenb, SEW,
LMUL, AVL, vl, `setvl`, `with_vl`, `element_count`, or `required_march` are
IR-modeled unless the real IR has the corresponding attribute, type, SSA value,
region argument, or generated ABI parameter. The current `tcrv_rvv.setvl` and
`tcrv_rvv.with_vl` ops model only runtime AVL/VL control-plane IR; they do not
make VLEN/vlenb or descriptor-local `element_count` runtime values.
RVV emission plans may carry bounded `selected_plan_metadata` entries such as
the selected vector-shape group, `tcrv_rvv.vlenb_bytes`, and
`tcrv_rvv.base_i32_m1_lanes` only as plugin-owned selected-plan
self-description. Those entries must be generated from validated selected
variant metadata and must not become runtime ABI parameters, generated control
values, runtime shapes, or performance claims.

## Deleted Selected Lowering Boundary Route

The canonical public `tcrv-opt` pass
`--tcrv-materialize-selected-lowering-boundaries` still routes through the
generic `ExtensionPluginRegistry` interface. RVV now implements that hook as
fail-closed no-boundary behavior for selected RVV-owned direct variants or
dispatch cases. There is no RVV-specific public wrapper pass for this route.

Rules:

- RVV-specific interpretation stays in the RVV plugin/dialect implementation.
- The generic pass routes dispatch fallback references through their generic
  fallback envelope; RVV returns no boundary for direct, dispatch, and fallback
  roles until a future materialized EmitC lowering route exists.
- Kernels without a dispatch or direct selected-path diagnostic are diagnosed
  before any plugin lowering-boundary hook is invoked.
- The old boundary op must not be materialized as an unsupported placeholder.
- RVV target exporters must not consume selected-boundary route identity fields
  as artifact authority. Executable emission requires a rebuilt
  extension-family op -> MLIR EmitC module route.
- No RVV boundary result may be reported as hardware execution, correctness, or
  performance evidence.

## Future Dialect Surface

Future RVV execution dialect work may add richer types:

```text
!tcrv.rvv.vreg<dtype, lmul>
!tcrv.rvv.mask<lmul>
!tcrv.rvv.vl
!tcrv.rvv.policy<tail, mask>
```

Future RVV execution dialect work may add ops:

```text
tcrv.rvv.load
tcrv.rvv.store
tcrv.rvv.masked_load
tcrv.rvv.masked_store
tcrv.rvv.broadcast
tcrv.rvv.fma
tcrv.rvv.add / mul / max / min
tcrv.rvv.reduce
tcrv.rvv.slide / gather / compress
tcrv.rvv.convert
```

These are RVV execution ops, not high-level tensor ops.

## Variant Generation

RVV plugin may support high-level op classes such as:

```text
matmul / batched matmul
softmax
layernorm / rmsnorm
rope
elementwise + reduction fusion
attention micro-kernel fragments
```

Output must be a `tcrv.exec.variant` containing `tcrv.rvv.*` ops, not a generic `tcrv.matmul`.

## Legality Rules

RVV plugin checks:

- target supports RVV;
- required dtype has corresponding extension or fallback path;
- SEW/LMUL combination is legal;
- VL policy is expressible;
- mask/tail policy is complete;
- load/store pattern is RVV-expressible;
- reduction handles tail/mask correctly;
- selected toolchain supports the emission path.

## Tuning Space

RVV tuning is variant quality metadata:

```text
SEW
LMUL
VL policy
unroll factor
K blocking
packing of B or weights
thread partition across harts
prefetch or software pipelining option
boundary handling strategy
```

Reference metadata:

```mlir
#tcrv.tuning<
  lmul = 4,
  sew = 16,
  k_block = 64,
  unroll = 2,
  thread_partition = "row_block"
>
```

## Selection Preference

The first RVV slice returns explicit plugin-owned selection preference metadata
for legal materialized RVV variants. Its score is a heuristic ordering input
used by the target-neutral selector; it is not a runtime, correctness, or
performance claim. When structured vlenb-derived i32 lane capacity is
available, the RVV plugin may use it as a bounded heuristic input and record
that fact in the explanation. RVV-specific interpretation of preserved
capability facts stays inside the RVV plugin before the generic preference
record is returned.

## Emission Paths

Current slice:

```text
rvv-plugin explicit typed RVV variant -> readiness unsupported
rvv-plugin explicit typed RVV variant -> emission plan fail-closed
bare RVV capability/no-body input -> no RVV proposal and no RVV route
```

The unsupported readiness result records that no materialized EmitC lowering,
runtime, or executable path exists yet. It must not be upgraded into an
emission plan, runtime ABI, artifact route, hardware execution, correctness
result, or performance evidence.

Public `tcrv-opt` registers the built-in RVV plugin at the tool boundary, so
materialized variants with `origin = "rvv-plugin"` can route through
`RVVExtensionPlugin` for emission-readiness diagnostics. Emission planning
fails closed until a materialized EmitC route exists. Unknown origins must
still fail through the generic unregistered-origin registry diagnostic.
Tests that need the historical empty-registry parser surface should pass
`--tcrv-disable-builtin-plugins`.

The RVV emission-plan hook must not return generic unsupported runtime
metadata for a deleted selected-boundary route. Absence of supported runtime
ABI/glue remains a diagnostic, not code generation, hardware execution,
correctness evidence, or performance evidence.

### Deleted Standalone Smoke-Probe Target Export

The former standalone smoke-probe target export surface is deleted.
`RVVExtensionPlugin` must not turn plugin-local metadata or route records into
supported emission readiness, a supported emission plan, or a generic target
source artifact route for that standalone harness. Built-in target artifact
exporter registration must not publish the former smoke-probe route identity or
any standalone direct C source artifact kind as a supported RVV source route.

Historical standalone smoke-probe metadata must not remain as active
code/spec/test fixtures. The compiler front door must not print
`riscv_vector.h`, `__riscv_` intrinsics, probe functions, or a probe `main`.
RVV hardware/toolchain smoke evidence belongs in explicit probe tooling and
separate `ssh rvv` artifacts.

### Deleted RVV Boundary And Future EmitC Route

The current selected RVV plugin path is non-executable and fail-closed for
emission. It must not materialize selected-boundary metadata, RVV microkernel
bodies, callable ABI parameters, source/header/object artifacts, self-check
helpers, or intrinsic C/C++ output.

The intended rebuild route is:

```text
explicit TCRV RVV extension-family ops
  -> EmitC
  -> RVV intrinsic C/C++
  -> clang default, gcc compatible
```

MLIR vector, LLVM scalable vector, LLVM RVV intrinsic IR, inline asm, and
backend patches are optional future routes. They are not the current RVV system
definition and should not be described as the mainline until promoted by a
separate spec and implementation evidence.

### Scenario: Future RVV Extension-Family EmitC Intrinsic Route

#### 1. Scope / Trigger

This scenario applies only after a future rebuild introduces explicit RVV
extension-family ops and a materialized EmitC module route. The route must
consume verified RVV family ops and lower them through a shared EmitC intrinsic
route before printing RVV intrinsic C/C++.

#### 2. Signatures

- Source family body:
  `tcrv_rvv.setvl -> tcrv_rvv.with_vl -> tcrv_rvv.<dtype>_load ->
  tcrv_rvv.<dtype>_load -> tcrv_rvv.<dtype>_<add|sub|mul> ->
  tcrv_rvv.<dtype>_store`.
- Route plan: an explicit EmitC intrinsic route object with standard headers,
  source op names, `emitc.call_opaque` callee names, and one setvl callee.
- Exported route metadata comments must include `emitc_route`,
  `emitc_route_headers`, `emitc_route_source_ops`, and indexed
  `emitc.call_opaque[...]` entries.

#### 3. Contracts

- The computation op is the RVV family op in the verified body, not metadata
  tokens.
- Explicit extension-family ops select the bounded operation, dtype, shape, and
  intrinsic spelling. Metadata-only variants fail closed before source output.
- `emitc.call_opaque` callees map from verified family ops:
  setvl maps to the selected vsetvl intrinsic; load, arithmetic, and store map
  to the selected RVV load, arithmetic, and store intrinsics.
- Runtime `n` remains the IR-backed runtime-element-count ABI parameter.
- Generated source comments are compile/export evidence only; they are not
  runtime correctness, hardware execution, throughput, latency, or performance
  evidence.

#### 4. Validation & Error Matrix

- Missing or duplicate family-op body step -> fail before source output.
- Body arithmetic op disagrees with selected bounded family -> fail before
  source output.
- Missing `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` control surface -> fail before
  source output.
- Missing route callee for any body step -> fail before source output.
- Stale runtime ABI role/name/type/ownership mirror -> fail before source
  output through the target artifact preflight.

#### 5. Good/Base/Bad Cases

- Good: future rebuilt i32 add emits `tcrv_rvv.i32_add`, the route records
  `emitc.call_opaque` for `__riscv_vadd_vv_i32m1`, and generated C calls that
  intrinsic.
- Base: a hand-authored bounded RVV explicit dataflow body with the same
  verified `setvl` / `with_vl` / load-arithmetic-store sequence can use the
  same route after selected-path and ABI preflight pass.
- Bad: stale descriptor mirror metadata says i32 add but the body contains
  `tcrv_rvv.i32_sub`; export must fail instead of printing vadd C from the
  metadata.

#### 6. Tests Required

- lit/FileCheck or C++ tests must show selected-boundary materialization does
  not auto-create an RVV family body from descriptor/family records.
- Tests must show RVV selected emission planning does not build callable ABI
  parameters or supported source/header/object routes from selected metadata.
- Negative coverage must keep stale body, stale descriptor, missing boundary,
  and malformed ABI cases fail-closed before source/header/object output.

#### 7. Wrong vs Correct

Wrong:

```text
stale descriptor metadata -> generated microkernel body -> direct C/intrinsic source
```

Correct:

```text
selected RVV metadata boundary
  -> unsupported/deleted-route diagnostic
future rebuild:
explicit extension-family ops -> materialized MLIR EmitC module -> C/C++
```

## Hart Parallelism

RVV plugin does not define a thread/block model. Multi-core execution is organized by `tcrv.exec.hart_parallel`.

RVV plugin provides per-hart RVV execution behavior and lowering preferences for OpenMP, pthread, runtime thread pool, or single-thread paths.

## Diagnostics

Diagnostics must report:

- illegal LMUL/SEW;
- unsupported dtype;
- missing mask/tail policy;
- unavailable emission path;
- unsuitable memory pattern;
- unsatisfied capability.
