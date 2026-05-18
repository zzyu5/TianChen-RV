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
artifact route, or metadata-only artifact authority. A selected RVV variant is legal only when it
already contains explicit typed `tcrv_rvv` extension-family IR. That typed IR
becomes executable only through the bounded selected EmitC artifact bridge
described below; typed IR alone is not runtime or performance evidence.

Stable current names:

```text
plugin name: rvv-plugin
plugin version: 0.1.0
plugin capability id: rvv
plugin capability kind: isa-vector
preferred kernel capability symbol: @rvv
variant origin: rvv-plugin
registered plugin capability ids:
  rvv
explicit typed variant form: a hand-authored or future materialized RVV variant
  must contain real `tcrv_rvv` ops in its body before RVV plugin legality or
  selection preference can accept it
typed policy attr name: tcrv_rvv.policy
typed policy attr value: #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
optional vlenb attr name: tcrv_rvv.vlenb_bytes
```

The bounded explicit RVV i32m1 add/sub/mul executable slice is allowed to
export a relocatable object only through the selected emission-plan target
artifact handoff:

```text
selected RVV path
  -> explicit typed tcrv_rvv i32m1 add/sub/mul body
  -> validated tcrv_rvv.with_vl selected lowering boundary
  -> RVV-owned EmitC lowerable route
  -> MLIR EmitC C/C++ emitter
  -> clang RISC-V relocatable object
```

This bounded route does not authorize descriptor-driven computation, deleted
microkernel wrappers, new dtype/LMUL families, or a generic RVV source printer.
It also does not let target/export code select an RVV artifact by assuming a
module contains exactly one direct variant; the selected emission-plan
candidate is the handoff authority. The rebuilt header route is declaration
only and derives its function name and ordered callable ABI parameters from the
same selected materialized EmitC candidate as the object route. The rebuilt
bundle route packages the object and header as one selected-variant component
group. Historical header and bundle route ids remain deleted and must not be
used as compatibility aliases.

The bounded RVV i32m1 add source materializer may construct this selected route
from the exact MLIR vector/scf source pattern owned by the RVV plugin. That
source body, not stale `tcrv_rvv.lowering_seed` metadata, route ids,
descriptors, artifact names, or deleted finite-family records, is the positive
authority for this front door. Unsupported source shapes and stale pre-existing
`tcrv.exec`/`tcrv_rvv` residue must fail before emission planning or target
artifact export.

The accepted runtime-count source pattern must make tail behavior explicit.
For the bounded i32m1 arithmetic slice, the source loop body computes
`remaining = n - iv`, creates `vector.create_mask remaining : vector<4xi1>`,
uses that same mask for both `vector.transfer_read` inputs and the
`vector.transfer_write` output, and keeps the loop upper bound equal to the
runtime ABI `n` operand. The old fixed-width `vector.load` / `vector.store`
step-4 shape is not valid authority for arbitrary runtime `n` unless a future
source contract explicitly proves a multiple-of-width precondition. The current
production source-artifact path must reject that old shape.

RVV probe facts remain bounded hardware/toolchain evidence inputs. They may be
validated into raw `TargetCapabilitySet` evidence facts, but they must not
manufacture finite SEW/LMUL/tail/mask config capabilities or authorize a
compiler route unless an explicit typed RVV variant body already exists.
Proposal collection for a no-body RVV-capable kernel records a recoverable
decline and produces no RVV proposal. Materialized RVV variant legality is
strict: the variant must be owned by `origin = "rvv-plugin"` and contain real
`tcrv_rvv` operations in its body. Metadata-only RVV attributes, finite
capability facts, deleted selected-shape facts, or profile evidence alone are
not sufficient.

The RVV plugin must not create any binary-family proposal from a no-body
`tcrv.exec.kernel`, from deleted frontend metadata, from finite-family registry
metadata, or from hand-authored microkernel names alone. Kernel-based
executable planning requires a future explicit extension-family op contract
and a materialized EmitC route before selecting family, dtype, artifact route,
callable ABI, artifact kind, or emitted body.

## Scenario: Tail-Safe RVV Vector Source Front Door

### 1. Scope / Trigger

This applies when the RVV plugin source front door recognizes source MLIR and
materializes the bounded i32m1 add/sub/mul selected boundary for runtime counts
provided by the callable ABI parameter `n`.

### 2. Signatures

- Source function ABI: `(%lhs: memref<?xi32>, %rhs: memref<?xi32>,
  %out: memref<?xi32>, %n: index) -> ()`.
- Source loop: `scf.for %i = %c0 to %n step %c4`.
- Tail source ops in order:
  `arith.subi %n, %i : index`, `vector.create_mask`, two masked
  `vector.transfer_read` ops, one `arith.addi|arith.subi|arith.muli` on
  `vector<4xi32>`, and one masked `vector.transfer_write`.

### 3. Contracts

- `%n` is the runtime element-count ABI parameter and must be the loop upper
  bound.
- The tail mask must be derived from `n - iv` and have type `vector<4xi1>`.
- Both source input transfers and the output transfer must consume the same
  tail mask, use minor-identity transfer maps, avoid `in_bounds = true`, and
  operate on `vector<4xi32>`.
- The source materializer may then construct the existing selected boundary:
  `runtime_abi_value(lhs,rhs,out,n) -> setvl(n) -> with_vl ->
  i32_load/i32_load -> i32_add|i32_sub|i32_mul -> i32_store`.
- The source front door must remain RVV plugin-owned. Common source-artifact
  orchestration must only run registered plugin passes and target exporters.

### 4. Validation & Error Matrix

- Missing or reordered ABI operands -> reject before materialization.
- Loop upper bound is not `%n`, lower bound is not zero, or step is not four ->
  reject before materialization.
- Loop body lacks `remaining = n - iv`, lacks `vector.create_mask`, omits a
  transfer mask, uses a different mask, or uses `vector.load`/`vector.store` ->
  reject before selected boundary materialization.
- Stale `tcrv_rvv.lowering_seed`, pre-existing `tcrv.exec` residue, or
  pre-existing `tcrv_rvv` residue -> reject before emission planning.

### 5. Good/Base/Bad Cases

- Good: masked `vector.transfer_read`/`vector.transfer_write` source with
  `n - iv` tail mask materializes the existing RVV selected boundary and target
  artifact route.
- Base: materialized hand-authored `tcrv_rvv` selected-boundary fixtures remain
  valid for target/export tests when they already contain explicit typed RVV IR.
- Bad: fixed step-4 `vector.load`/`vector.store` source is accepted and then
  used to claim correctness for runtime counts such as 7 or 23.

### 6. Tests Required

- Positive lit coverage for add/sub/mul tail-safe source fixtures reaching
  selected dispatch, emission-plan metadata, object, header, and bundle export.
- Negative lit coverage for the old fixed `vector.load`/`vector.store` source
  shape and for stale metadata/residue, ABI order mismatch, loop bound
  mismatch, missing mask, and unsupported arithmetic.
- Runtime claims from the accepted source route require real `ssh rvv` evidence
  with at least one non-multiple-of-four count and one multiple-of-four count.

### 7. Wrong vs Correct

Wrong:

```mlir
scf.for %i = %c0 to %n step %c4 {
  %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
  %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
  %sum = arith.addi %a, %b : vector<4xi32>
  vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
}
```

Correct:

```mlir
scf.for %i = %c0 to %n step %c4 {
  %remaining = arith.subi %n, %i : index
  %mask = vector.create_mask %remaining : vector<4xi1>
  %a = vector.transfer_read %lhs[%i], %pad, %mask
      : memref<?xi32>, vector<4xi32>
  %b = vector.transfer_read %rhs[%i], %pad, %mask
      : memref<?xi32>, vector<4xi32>
  %sum = arith.addi %a, %b : vector<4xi32>
  vector.transfer_write %sum, %out[%i], %mask
      : vector<4xi32>, memref<?xi32>
}
```

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
  capability facts. Deleted local RVV element-count metadata must not be
  reintroduced as a runtime trip count or artifact descriptor.
- The selected i32m1 arithmetic EmitC route must consume one RVV-owned
  config/VL contract shared with RVV dialect verification. That contract
  validates `tcrv_rvv.setvl` and `tcrv_rvv.with_vl` as the same compile-time
  SEW32, LMUL m1, tail agnostic, mask agnostic config; verifies that
  `with_vl` consumes the visible `setvl` VL result; and keeps AVL/VL as
  runtime SSA values. Dialect-level bounded dataflow may parse non-executable
  sibling config such as i32m2 when explicitly modeled, but the selected
  i32m1 arithmetic artifact route must fail closed before route payload
  construction if the selected body is not the exact i32m1 config.
- Target artifact export may consume the selected emission-plan route and
  runtime ABI metadata, but it must not re-derive RVV SEW/LMUL/policy/VL
  semantics in common target code.

### 4. Validation & Error Matrix

- Missing `rvv` provider -> plugin decline during proposal collection.
- Bare RVV capability evidence without an explicit typed RVV variant body ->
  no RVV proposal and no selected RVV route.
- Explicit selected RVV variant without `origin = "rvv-plugin"` -> fatal
  legality error.
- Explicit selected RVV variant without `tcrv_rvv` ops in its body -> fatal
  legality error.
- Selected i32m1 arithmetic route whose `setvl` / `with_vl` compile-time
  config does not match SEW32, LMUL m1, tail agnostic, mask agnostic -> fail
  before EmitC call payload construction.
- Selected i32m1 arithmetic route whose `with_vl` does not consume the visible
  `setvl` VL SSA result -> fail before EmitC call payload construction.
- Selected route id whose expected add/sub/mul op does not match the typed
  arithmetic op in the selected body -> fail before target artifact export.
- Capability/profile facts may still be validated for replay and typed-body
  checks, but they must not create proposal, boundary, emission-plan, runtime
  ABI, artifact, or metadata-only route authority by themselves.

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
- Dialect and EmitC route tests must cover the i32m1 config/VL contract:
  missing or mismatched `with_vl` config, non-agnostic policy, unsupported
  LMUL for i32m1 artifact export, `with_vl` consuming a non-`setvl` VL token,
  and stale selected route/op combinations.
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
compiler generated-source path, materialize an executable RVV microkernel,
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

The former microkernel direct C slice and finite RVV target object/header route
slice are deleted as production authority. The only active bounded RVV target
artifact route is the materialized EmitC i32m1 arithmetic route family: object
export compiles the MLIR EmitC C/C++ emitter output, header export emits only a
callable declaration for that materialized function boundary, and bundle export
packages the object and header under one selected-variant ABI component group.
Target/export code must not synthesize RVV compute C bodies from selected
metadata, family records, route records, or deleted wrapper records. RVV routes
must fail closed instead of producing `riscv_vector.h` intrinsic source,
relocatable objects, headers, bundles, or self-check harnesses from the old
direct printer path.

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
  facts: architecture, hart count, optional raw VLENB bytes,
  ISA/vector hint string, clang and CMake
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
  `rvv.hart_count`, `rvv.vlenb_bytes`,
  `rvv.toolchain.clang`, `rvv.toolchain.cmake`, `rvv.probe.compile_run`,
  `rvv.toolchain.march`, and `rvv.toolchain.mabi`. SEW/LMUL/tail/mask config
  IDs such as `rvv.i32_m1.sew32` belong to explicit RVV config/profile fixtures
  or plugin-selected variant requirements, not to
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
agnostic|undisturbed>`, the bounded `!tcrv_rvv.runtime_abi_value` token and
`tcrv_rvv.runtime_abi_value` ABI binding op, the bounded runtime AVL-to-VL
control-plane operation `tcrv_rvv.setvl`, the bounded VL scope region
operation `tcrv_rvv.with_vl`, the finite `tcrv_rvv.i32_load`,
`tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, `tcrv_rvv.i32_mul`, and
`tcrv_rvv.i32_store` ops nested under that scope for non-executable i32
dataflow modeling. The previous plugin-local selected lowering-boundary
operation is deleted as active compiler authority. The `runtime_abi_value` op
binds one callable C ABI value by role, C name, C type, and ownership, and
produces an SSA value consumed by the RVV first-slice IR. The setvl op consumes
a runtime AVL SSA value, produces a `!tcrv_rvv.vl` token, and carries only
bounded first-slice compile-time config metadata: SEW 32, LMUL m1 or m2, and
the finite policy attribute. The with_vl op consumes one `!tcrv_rvv.vl` value
and owns one single-block region for bounded RVV control/body work. Optional
duplicated SEW/LMUL/policy metadata is limited to the same bounded first-slice
config and must agree with the visible defining setvl when present. The bounded
i32 add/sub/mul dataflow body consumes explicit `lhs`, `rhs`, and `out`
runtime ABI SSA values on the load/store operations; concrete C parameter
names are plugin-owned route inputs from the defining runtime ABI value ops,
not descriptor fields or target-side inference. It is not a generic vector
memory model. These surfaces are not generic RVV arithmetic, generic memory
operations, LLVM/RISC-V lowering, full runtime ABI glue, hardware execution,
correctness evidence, or performance evidence. `tcrv_rvv`
is the concrete MLIR dialect namespace because MLIR dialect namespaces cannot
contain `.` characters; the architectural extension family remains `tcrv.rvv`.

For the bounded i32m1 arithmetic route, runtime `n` may drive repeated RVV
chunks only through the RVV extension-family route and the common materialized
EmitC path. A supported multi-VL route must materialize a real `emitc.for`
boundary with an induction value, compute remaining AVL from `n - offset`,
derive the per-iteration VL from `tcrv_rvv.setvl` ownership, advance lhs/rhs/out
pointers or indices by the induction value, and place load/compute/store calls
inside that structured loop. Artifact metadata such as `multi_vl`, loop
induction, step, remaining-AVL, or pointer-advance keys is evidence and
preflight input only; it must match the materialized EmitC loop and must not
synthesize RVV compute bodies, intrinsic calls, loop source text, or route
semantics in target/export code. Headers, objects, and bundles must fail closed
when they claim multi-VL support without that materialized loop.

## Capability Fields

RVV plugin should register the base plugin capability and may query explicit
profile or typed-variant config facts without advertising them from a target
helper catalog:

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
  sew = 32 : i64,
  source_kernel = "example_kernel",
  selected_variant = @example_rvv_variant,
  origin = "rvv-plugin",
  selected_path_role = "direct variant",
  status = "selected-lowering-boundary",
  required_capabilities = [@rvv],
  rvv_construction_protocol = "extension-family-construction-protocol.v1",
  rvv_emitc_route_mapping = "rvv-i32m1-arithmetic-emitc-route-family"
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
token, owns one single-block region with no region arguments, and may repeat
the same bounded SEW/LMUL/policy config. When the bounded i32m1 materialized
EmitC artifact path selects this op as the construction-template boundary, the
same op must also carry selected-boundary conformance facts:
`source_kernel`, `selected_variant`, `origin`, `selected_path_role`, `status`,
`required_capabilities`, `rvv_construction_protocol`, and
`rvv_emitc_route_mapping`. These selected-boundary attrs are validation facts
only; they are not descriptors, source printers, runtime ABI synthesis,
hardware facts, or compute semantics. The with_vl op continues to reject
VLEN/vlenb, `element_count`, `required_march`,
`tcrv_rvv.required_capabilities`, and raw capability facts. The former
selected-boundary op and typed RVV microkernel wrappers are deleted as
structural authority; future executable emission must be rebuilt through
explicit extension-family IR plus a materialized MLIR EmitC module route.
These surfaces are not vector registers, masks, memory operations, RVV
intrinsics, LLVM/RISC-V lowering, runtime ABI, executable emission, correctness
evidence, or performance evidence.

## RVV Parameter Boundary

RVV work must keep these parameter layers distinct:

- VLEN and vlenb are hardware facts / target capability evidence. They may
  constrain legality and selection only after provenance is validated, but
  active profiles must keep VLENB as raw byte evidence and must not derive
  finite i32/M1 lane-capacity capability facts or selected-path metadata from
  it. Lane behavior belongs to explicit RVV config IR plus runtime AVL/VL/ABI
  surfaces.
- SEW, LMUL, tail policy, and mask policy are compile-time variant config
  selected or proposed by the RVV plugin and checked against target
  capabilities. The current non-executable bounded RVV dataflow slice admits only SEW 32 with
  LMUL m1 or m2, tail agnostic, and mask agnostic. The m1 shape is backed by
  `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic`; the m2 shape is backed by
  `rvv.i32_m2.sew32`, `rvv.i32_m2.lmul_m2`,
  `rvv.i32_m2.tail_policy.agnostic`, and
  `rvv.i32_m2.mask_policy.agnostic`. A selected variant must carry explicit
  typed RVV IR whose SEW/LMUL/policy attributes agree with its requirements,
  and future EmitC/export routes must reject capability/body LMUL mismatches
  before artifact bytes are emitted. These config ids are not
  sufficient as standalone hardware facts without the surrounding
  RVV/profile/toolchain evidence.
- Target helper catalogs must not serialize the selected RVV compile-time
  choice. Materialized explicit RVV variants and any future generated artifacts
  must derive SEW, LMUL, tail/mask policy, and intrinsic spellings from real
  `tcrv_rvv` IR plus the materialized EmitC/runtime route, not from selected
  metadata descriptors, suffix tables, or target-owned shape catalogs.
- AVL and vl are runtime SSA values / runtime control values. The current
  bounded `tcrv_rvv.setvl` surface models AVL as a real runtime SSA operand and
  vl as a real `!tcrv_rvv.vl` result. The bounded `tcrv_rvv.with_vl` surface
  models vl only when it consumes a real `!tcrv_rvv.vl` SSA operand. It must not
  imply that AVL or vl is IR-modeled unless a real op attribute, SSA value,
  region argument, or ABI parameter exists.
- The deleted legacy RVV local element-count marker is not active selected-path
  metadata. Active RVV dialect ops must reject local `element_count`
  attributes; the value does not describe a valid descriptor, emitted source
  slice, production input, high-level MLIR tensor shape, global problem size,
  AVL, or vl.
- `tcrv_rvv.required_march` string matching is a bounded plugin-owned
  compatibility bridge for the current first slice. Do not expand dependence on
  `required_march` string comparisons when structured capabilities or
  properties are available or should be added.

Emission plans and manifests for RVV paths must not claim VLEN, vlenb, SEW,
LMUL, AVL, vl, `setvl`, `with_vl`, `element_count`, or `required_march` are
IR-modeled unless the real IR has the corresponding attribute, type, SSA value,
region argument, or generated ABI parameter. The current `tcrv_rvv.setvl` and
`tcrv_rvv.with_vl` ops model only runtime AVL/VL control-plane IR; they do not
make VLEN/vlenb or deleted local RVV element-count residue runtime values.
Selected-boundary attrs on `tcrv_rvv.with_vl` are artifact-handoff
conformance facts for the selected materialized EmitC route; they do not make
capability facts, route ids, ABI records, or compute bodies executable by
themselves.
RVV emission plans must not use selected-shape metadata descriptors as
lowering, runtime ABI, or artifact authority. Bounded diagnostics may mention
validated raw hardware/profile facts such as `tcrv_rvv.vlenb_bytes`, but
executable artifacts require explicit extension-family IR plus a materialized
EmitC/runtime route.

## Selected `with_vl` Lowering Boundary Route

The canonical public `tcrv-opt` pass
`--tcrv-materialize-selected-lowering-boundaries` still routes through the
generic `ExtensionPluginRegistry` interface. For the bounded explicit i32m1
add/sub/mul route, RVV recognizes the existing `tcrv_rvv.with_vl` operation in
the selected variant body as the selected lowering boundary. There is no
RVV-specific public wrapper pass for this route.

Rules:

- RVV-specific interpretation stays in the RVV plugin/dialect implementation.
- The generic pass routes selected direct variants and dispatch cases through
  the plugin interface; the RVV plugin may report the existing
  `tcrv_rvv.with_vl` op as the materialized boundary when the selected variant
  is a legal i32m1 add/sub/mul body.
- Dispatch fallback references continue through their generic fallback
  envelope and do not create an RVV boundary.
- Kernels without a dispatch or direct selected-path diagnostic are diagnosed
  before any plugin lowering-boundary hook is invoked.
- The old boundary op must not be materialized as an unsupported placeholder.
- `tcrv_rvv.with_vl` boundary validation must use the RVV-owned config/VL
  contract and fail closed for missing, duplicate, mismatched, or unsupported
  selected boundary shapes.
- The selected `tcrv_rvv.with_vl` boundary must carry IR-owned conformance
  facts before emission planning or artifact export: source kernel, selected
  variant, origin plugin, selected-path role, status, required capabilities,
  construction protocol id, route mapping id, and bounded RVV config attrs.
  RVV plugin and target/export code must validate these attrs; they must not
  synthesize missing selected-boundary truth at planning or export time.
- RVV target exporters must not consume selected-boundary route identity fields
  as artifact authority by themselves. Executable emission requires the
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
performance claim. Raw VLENB may be considered only as validated hardware
evidence; the plugin must not preserve derived finite i32/M1 lane-capacity
facts as selection preference authority. RVV-specific interpretation of
preserved capability facts stays inside the RVV plugin before the generic
preference record is returned.

## Emission Paths

Current slice:

```text
rvv-plugin explicit typed RVV i32m1 add variant
  -> common EmitC lowerable route
  -> parseable MLIR EmitC module
rvv-plugin explicit typed RVV variant -> runtime/artifact readiness unsupported
rvv-plugin explicit typed RVV variant -> target emission plan fail-closed
bare RVV capability/no-body input -> no RVV proposal and no RVV route
```

The bounded current materialized route is MLIR EmitC IR only. It proves that
explicit RVV extension-family ops can construct a common
`TCRVEmitCLowerableRoute` and materialize `emitc.include`, `emitc.func`, and
`emitc.call_opaque` operations with interface-backed provenance. It does not
print C/C++, export source/header/object artifacts, compile, run hardware, or
prove correctness/performance. Runtime/artifact readiness remains unsupported
until runtime ABI, C/C++ emitter handoff, and target artifact routes are built.

Public `tcrv-opt` registers the built-in RVV plugin at the tool boundary, so
materialized variants with `origin = "rvv-plugin"` can route through
`RVVExtensionPlugin` for emission-readiness diagnostics and through the common
EmitC lowerable materialization pass. Target emission planning still fails
closed after the bounded EmitC module step. Unknown origins must still fail
through the generic unregistered-origin registry diagnostic.
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
generated-source route for that standalone harness. Built-in target artifact
exporter registration must not publish the former smoke-probe route identity or
any standalone direct C output artifact kind as a supported RVV output route.

Historical standalone smoke-probe metadata must not remain as active
code/spec/test fixtures. The compiler front door must not print
`riscv_vector.h`, `__riscv_` intrinsics, probe functions, or a probe `main`.
RVV hardware/toolchain smoke evidence belongs in explicit probe tooling and
separate `ssh rvv` artifacts.

### Deleted RVV Boundary And Bounded EmitC Route

The historical metadata/direct-wrapper RVV selected-boundary path remains
deleted and fail-closed for target emission. It must not materialize
selected-boundary metadata, RVV microkernel bodies, direct source/header/object
artifacts, self-check helpers, or intrinsic C/C++ output. The bounded i32m1
add/sub/mul path may materialize an MLIR EmitC module from explicit RVV ops
through the common lowerable route; a selected target artifact may package that
module only after the selected emission-plan candidate and materialized EmitC
handoff are verified.

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

### Scenario: Current Bounded RVV Extension-Family EmitC Intrinsic Route

#### 1. Scope / Trigger

This scenario applies to the current bounded RVV i32m1 add/sub/mul
materialization route. The route consumes verified RVV family ops and lowers
them through the shared EmitC lowerable route into an MLIR EmitC module.
Printing C/C++, compiling, target artifact packaging, and `ssh rvv` runtime
evidence are separate stages. A target artifact bridge may invoke them only
after this provider-owned route has been built and verified.

#### 2. Signatures

- Source family body:
  four explicit `tcrv_rvv.runtime_abi_value` bindings for `lhs`, `rhs`,
  `out`, and `n`, then
  `tcrv_rvv.setvl -> tcrv_rvv.with_vl -> tcrv_rvv.i32_load ->
  tcrv_rvv.i32_load -> tcrv_rvv.i32_add|i32_sub|i32_mul ->
  tcrv_rvv.i32_store` for SEW32 LMUL m1 with agnostic policy.
- Selected lowering boundary: the existing `tcrv_rvv.with_vl` operation in the
  selected variant body, not a synthesized wrapper. The boundary op must carry
  `source_kernel`, `selected_variant`, `origin`, `selected_path_role`,
  `status`, `required_capabilities`, `rvv_construction_protocol`, and
  `rvv_emitc_route_mapping` before the route is planned or exported.
- Route plan: an explicit EmitC intrinsic route object with standard headers,
  source op names, `emitc.call_opaque` callee names, and one setvl callee.
- Materialized EmitC provenance comments must include typed source op names,
  source roles, `TCRVEmitCLowerableOpInterface`, and `emitc.call_opaque`
  callee evidence.
- Provider header: `TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`.
- Provider route builders:
  `plugin::rvv::buildRVVI32M1ArithmeticEmitCLowerableRoute`,
  `plugin::rvv::buildRVVI32M1AddEmitCLowerableRoute`,
  `plugin::rvv::buildRVVI32M1SubEmitCLowerableRoute`, and
  `plugin::rvv::buildRVVI32M1MulEmitCLowerableRoute`; each accepts
  `const plugin::VariantEmitCLowerableRequest &` and populates
  `conversion::emitc::TCRVEmitCLowerableRoute &`.
- Provider metadata accessors own the bounded route id, emission kind,
  lowering-boundary op name, runtime ABI kind/name/glue role, and ordered
  runtime ABI parameters for this RVV i32m1 arithmetic slice.
- Selected target artifact route id:
  `rvv-i32m1-arithmetic-emitc-route-family`.
- Supported artifact kind: `riscv-elf-relocatable-object`.
- Target handoff kind:
  `materialized-emitc-cpp-rvv-intrinsic-object`.
- Target route API:
  `target::rvv::getRVVMaterializedEmitCTargetArtifactRouteID()`.

#### 3. Contracts

- The computation op is the RVV family op in the verified body, not metadata
  tokens.
- Explicit extension-family ops select the bounded operation, dtype, shape, and
  intrinsic spelling. Metadata-only variants fail closed before source output.
- `emitc.call_opaque` callees map from verified family ops:
  setvl maps to the selected vsetvl intrinsic; load, arithmetic, and store map
  to the selected RVV load, arithmetic, and store intrinsics.
- Runtime `n` remains the IR-backed runtime-element-count ABI parameter.
- Runtime `lhs`, `rhs`, `out`, and `n` must be explicit SSA values produced by
  `tcrv_rvv.runtime_abi_value` in the selected RVV body before this bounded
  route may construct EmitC. `setvl` consumes the `n` SSA value; load/store
  ops consume the buffer SSA values. Synthetic
  `builtin.unrealized_conversion_cast` placeholders and buffer-role-only
  metadata are not artifact handoff authority for this route.
- Materialized EmitC comments are compiler route evidence only; they are not
  runtime correctness, hardware execution, throughput, latency, or performance
  evidence.
- RVV intrinsic/header names, typed-body shape validation, source-op
  provenance checks, and ABI value mapping are provider-owned. Target artifact
  support may consume the provider and package source/header/object artifacts,
  but it must not duplicate or re-own these RVV route semantics.
- Common selected EmitC artifact front doors call a route-builder callback and
  remain generic; they must not contain RVV intrinsic names, RVV header names,
  or typed RVV body-shape rules.
- Target/RVV artifact support registers one family-level materialized EmitC
  exporter for the selected RVV i32m1 arithmetic object route. It must not
  register per-op target route tables, descriptor adapters, old object/header/
  bundle route ids, or compatibility wrappers.
- Target/RVV artifact support may compile the MLIR-emitted C/C++ source to a
  RISC-V relocatable object with clang. It must not synthesize the C/C++ source
  from metadata, selected-path records, route ids, family registries, or
  descriptors.
- Target/RVV artifact support must also not synthesize missing selected
  `with_vl` boundary conformance attrs. It may validate the IR-owned
  `with_vl` attrs against the selected variant, required capabilities,
  selected path role, construction protocol, route mapping, and bounded RVV
  config, then fail closed when any fact is absent or stale.

#### 4. Validation & Error Matrix

- Missing or duplicate family-op body step -> fail before source output.
- Body arithmetic op disagrees with selected bounded family -> fail before
  source output.
- Missing `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` control surface -> fail before
  source output.
- Missing, duplicate, mismatched, or unsupported selected `tcrv_rvv.with_vl`
  boundary shape -> fail before route payload construction or source output.
- Missing or stale selected `tcrv_rvv.with_vl` conformance attr
  (`source_kernel`, `selected_variant`, `origin`, `selected_path_role`,
  `status`, `required_capabilities`, `rvv_construction_protocol`,
  `rvv_emitc_route_mapping`, or bounded config attrs required by the route) ->
  fail before route payload construction, generated C++ source, object,
  header, or bundle output.
- Missing, duplicate, malformed, or unsupported explicit runtime ABI value
  binding for `lhs`, `rhs`, `out`, or `n` -> fail before source output.
- Missing route callee for any body step -> fail before source or object
  output.
- Stale runtime ABI role/name/type/ownership mirror -> fail before source
  output through the target artifact preflight.
- Target artifact code contains RVV intrinsic/header names or reconstructs the
  typed body shape instead of calling the provider -> ownership violation.
- Missing `rvv_emitc_lowerable_route` artifact metadata, stale selected
  runtime ABI identity, non-materialized EmitC handoff, missing route/call
  provenance, or clang object-packaging failure -> fail before claiming a
  target artifact.

#### 5. Good/Base/Bad Cases

- Good: rebuilt i32 add/sub/mul emits the matching
  `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, or `tcrv_rvv.i32_mul`, the route
  records the matching `emitc.call_opaque` arithmetic intrinsic, generated C
  calls that intrinsic, and the target bridge packages the MLIR-emitted source
  as a RISC-V relocatable object.
- Base: a hand-authored bounded RVV explicit dataflow body with the same
  verified `setvl` / `with_vl` / load-arithmetic-store sequence can use the
  same route after selected-path and ABI preflight pass.
- Bad: stale descriptor mirror metadata says i32 add but the body contains
  `tcrv_rvv.i32_sub`; export must fail instead of printing vadd C from the
  metadata.
- Bad: Target/RVV artifact packaging directly scans `tcrv_rvv` ops, chooses
  `riscv_vector.h`, or spells `__riscv_vadd_vv_i32m1` instead of delegating to
  the RVV provider.

#### 6. Tests Required

- lit/FileCheck or C++ tests must show selected-boundary materialization does
  not auto-create an RVV family body from descriptor/family records.
- Tests must show RVV selected emission planning does not build callable ABI
  parameters or supported source/header/object routes from selected metadata.
- Tests must show explicit typed RVV source/extension-op paths produce a
  supported family-level object artifact plan and that the generic target
  artifact front door can emit a RISC-V relocatable object through the common
  selected EmitC artifact bridge.
- Negative coverage must keep stale body, stale descriptor, missing boundary,
  and malformed ABI cases fail-closed before source/header/object output.
- Target artifact tests must prove RVV target artifact exporters use the
  materialized EmitC family route, keep deleted direct route ids absent, and
  reject missing route provenance or stale runtime ABI metadata.

#### 7. Wrong vs Correct

Wrong:

```text
stale descriptor metadata -> generated microkernel body -> direct C/intrinsic source
```

Correct:

```text
explicit extension-family ops
  -> selected family-level EmitC artifact plan
  -> provider-owned TCRVEmitCLowerableRoute
  -> materialized MLIR EmitC module
  -> MLIR EmitC C/C++ emitter
  -> clang RISC-V relocatable object
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
