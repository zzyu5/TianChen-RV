# Emission Runtime Contract

## Core Principle

Core pass does not implement extension-specific emission. Each plugin owns its lowering/emission/runtime glue.

Variant selection and emission path must be represented in capability model and verified before final lowering.

## Emission Readiness First Slice

The first compiler-visible emission slice is a target-neutral readiness check.
It does not lower IR, generate runtime ABI glue, invoke a toolchain, run a
kernel, or prove correctness/performance. It verifies that current
`tcrv.exec` structure can be routed to the origin plugin that owns each
materialized variant.

Rules:

- route by the materialized `tcrv.exec.variant` `origin` attribute;
- carry the variant, enclosing `tcrv.exec.kernel`, generic
  `TargetCapabilitySet`, and target-neutral role (`direct variant`,
  `dispatch case`, or `dispatch fallback`);
- require supported plugin results to contain a non-empty plugin-owned emission
  path identifier or description;
- require unsupported plugin results to contain a non-empty reason;
- diagnose missing/empty origin, unknown plugin, disabled plugin, malformed
  plugin result, missing kernel/variant, and non-direct variant/kernel
  relationships generically;
- validate `tcrv.exec.dispatch` case/fallback targets structurally before
  routing to plugins: every reference must resolve to a direct sibling
  `tcrv.exec.variant` in the same kernel and duplicate references are invalid;
- when no dispatch is present, consume a direct selected-path
  `tcrv.exec.diagnostic` marker before falling back to all variants; the marker
  must carry a direct sibling variant `target` and a generic `selection_kind`
  such as `static-variant` or `fallback-only`;
- avoid core branches on RVV, IME, offload, scalar, vendor, dtype, shape,
  runtime, toolchain, or microarchitecture details.

## Emission Plan First Slice

After readiness and selected-path traversal, the compiler may collect
plugin-owned emission plans. An emission plan is a target-neutral compiler
decision object, not an executable artifact.

Rules:

- route by the same materialized variant `origin` plugin as readiness;
- carry kernel symbol, variant symbol, selected-path role, support status,
  origin plugin, runtime ABI ownership metadata, required capability refs, and
  diagnostic/explanation metadata;
- carry bounded runtime ABI kind/name metadata and a bounded required runtime
  glue role chosen by the origin plugin;
- carry required capability symbol refs that are a safe subset of the selected
  variant `requires` metadata;
- for supported paths, require non-empty emission kind, lowering pipeline
  identifier, runtime ABI identifier, artifact kind, and explanation;
- for unsupported paths, require a non-empty diagnostic reason;
- reject plugin-returned empty runtime ABI kind/name, empty runtime glue role,
  missing required capability refs, or unbounded/secret-like diagnostic and
  explanation text before materialization;
- diagnose missing origin, unregistered origin, disabled origin plugin,
  malformed plugin result, mismatched variant symbol, mismatched kernel symbol,
  mismatched selected-path role, duplicate selected markers, and missing
  dispatch targets generically before treating the plan as usable;
- for selected dispatch or selected-marker paths after lowering-boundary
  materialization, require exactly one matching plugin-owned boundary operation
  before emission-plan diagnostics are materialized;
- validate selected-boundary metadata generically before plugin plan routing:
  `source_kernel` must match the enclosing kernel, `selected_variant` and
  `role` must match the selected path, `origin` must match the selected
  variant origin, `required_capabilities` must be a safe subset of the selected
  variant `requires`, and stale or duplicate competing boundaries are fatal;
- recognize selected lowering-boundary candidates only when they are actual
  plugin-local `*.lowering_boundary` operations or explicit lowering-boundary
  diagnostic metadata. A plugin-local executable attachment may also carry
  selected-path fields, but it is not the generic lowering-boundary operation
  consumed by emission planning;
- allow public tools to populate a deterministic built-in plugin registry before
  constructing registry-dependent passes, while keeping the traversal and
  selected-path routing target-neutral in shared pass code;
- keep the boundary clear: readiness says whether a selected path is
  supportable, while the plan describes the plugin-owned lowering/runtime route
  or the structured unsupported reason;
- do not claim generated code, runtime ABI glue, linked artifacts, RVV hardware
  execution, correctness, or performance from an emission plan alone.

## Target Artifact Export Route

Public artifact export tools may add a generic routing front door after
selected-path, lowering-boundary, and emission-plan metadata already exist in
MLIR. The generic layer consumes only target-neutral emission-plan fields such
as selected variant, role, origin, support status, lowering pipeline route id,
emission kind, artifact kind, lowering-boundary reference, runtime ABI metadata,
and required capability refs. It must fail closed for missing selected paths,
missing emission-plan diagnostics, unsupported or metadata-only plans, unknown
route ids, artifact-kind mismatch, origin/emission-kind mismatch, stale selected
paths, duplicate or ambiguous supported artifacts, and malformed bounded text.

Concrete artifact generation remains target-owned. Built-in tools may register
target exporters, but the registration is where extension-specific route facts
belong. Shared generic routing must not branch on RVV, IME, offload, scalar,
vendor, dtype, shape, runtime, toolchain, or microarchitecture semantics. The
currently supported source routes are only bounded explicit microkernel
attachments: the RVV i32 vector-add standalone C exporter registered by RVV
target/export code, and the scalar fallback i32 vector-add portable standalone
C exporter registered by scalar target/export code. This does not add generic
RVV or scalar lowering, runtime ABI integration, object generation, linking,
arbitrary source export, correctness evidence, or performance evidence.

## Selected Lowering Boundary First Slice

Before executable lowering exists, the compiler may materialize selected-path
lowering-boundary metadata through the generic extension plugin registry. RVV is
the first plugin that creates a plugin-local `tcrv_rvv.lowering_boundary`
operation for selected RVV direct variants or dispatch cases. Scalar fallback
creates a plugin-local `tcrv_scalar.lowering_boundary` operation for selected
portable fallback paths. These structures are attachment points for future
lowering work, not executable lowering products.

Rules:

- materialization consumes selected `tcrv.exec` dispatch or selected-marker
  structure and generic variant `origin` metadata;
- ownership and legality are routed through the `ExtensionPluginRegistry` and
  the selected variant's origin plugin before any plugin-local metadata is
  created;
- scalar or other fallback references remain selected `tcrv.exec` metadata and
  are routed to their origin plugin; scalar fallback materializes scalar
  plugin-local metadata and must not receive RVV ops;
- RVV boundary ops must carry `status = "unsupported"` and a non-empty
  unsupported reason;
- RVV and scalar fallback boundary ops must carry the generic selected-boundary
  contract fields needed by downstream emission planning: `source_kernel`,
  `selected_variant`, `origin`, `role`, `status`, and
  `required_capabilities`;
- scalar fallback boundary ops must carry `status = "metadata-only"` and
  selected variant, origin, role, and required capability reference metadata;
- selected lowering-boundary metadata must not claim intrinsics, LLVM/RISC-V
  lowering, runtime ABI glue, generated objects, hardware execution,
  correctness, or performance.

## EmissionProvider Responsibilities

Each plugin emission provider:

- registers lowering patterns;
- declares emission path;
- declares compiler flags, headers, libraries;
- generates runtime glue;
- reports unsupported path;
- emits diagnostics.

## RVV Emission

### MLIR vector / LLVM scalable vector

Path:

```text
tcrv.rvv ops
  -> MLIR vector dialect / LLVM dialect
  -> LLVM scalable vector
  -> LLVM RISC-V backend
  -> RVV machine code
```

Use for ordinary arithmetic, load/store, and reductions that LLVM reliably lowers.

### RVV intrinsic / inline asm / builtin

Path:

```text
tcrv.rvv ops
  -> LLVM RVV intrinsic / compiler builtin / inline asm
  -> native compile
```

Use for precise `vsetvl`, mask/tail policy, segment/strided ops, or key kernels where LLVM vector lowering is not stable enough.

## IME Emission

Possible paths:

```text
vendor intrinsic
compiler builtin
inline asm
external assembly stub
patched LLVM/backend adapter
```

Rules:

- IME-specific emission stays inside IME plugin.
- Core only knows that the variant has an emission path.
- Capability model records toolchain support.
- If emission is unavailable, verifier rejects the variant or preserves it as disabled diagnostic.

## Offload Emission

Offload emission is runtime glue generation, not instruction generation.

Path:

```text
tcrv.offload ops
  -> C ABI call / vendor runtime call
  -> link vendor runtime library
  -> runtime dispatch / sync / buffer management
```

Generated or declared runtime pieces:

- runtime handle initialization;
- buffer binding/allocation;
- host-device copy;
- async call;
- wait/sync;
- error handling;
- resource release;
- fallback path.

## Runtime Dispatch Lowering

`tcrv.exec.dispatch` lowers to host-side decision logic.

Inputs:

```text
capability availability
runtime probe
shape/dtype guard
cost threshold
user policy
```

Output:

```text
call selected variant or fallback
```

Dispatch logic must diagnose why a variant was selected or skipped.

## Fallback Lowering

Fallback is required for system completeness. Sources may include:

```text
MLIR default lowering
scalar/scf lowering
LLVM auto-vectorization
portable C/C++ kernel
RVV conservative kernel
```

Fallback targets correctness and coverage, not the main performance story.

## Toolchain Boundary

Rules:

- plugin declares toolchain requirement;
- capability verifier checks the requirement;
- core pass does not call vendor-specific compiler path directly;
- build system selects flags/libraries through plugin metadata;
- unsupported path produces clear diagnostics.

## Deployment Profiles

Reference profiles:

```text
rvv-native
  target: ssh rvv
  emission: LLVM/RVV intrinsic/native compile
  runtime: OpenMP/pthread optional

k3-ime
  target: K3/IME environment
  emission: IME plugin selected path
  runtime: native or vendor support

riscv-sophgo-offload
  target: RISC-V host + Sophgo runtime
  emission: C ABI/runtime glue
  runtime: vendor accelerator runtime
```

## Reproducibility Output

Each emission path should record:

- selected variant;
- required capabilities;
- selected compiler flags;
- selected runtime libraries;
- lowering path;
- fallback status;
- unsupported reason if failed.

Before real lowering/runtime work exists, plugin emission plans may be
materialized as `tcrv.exec.diagnostic {reason = "emission_plan"}` metadata.
These diagnostics are reproducibility and compiler-decision records only. A
supported or metadata-only diagnostic records plugin-owned intent such as
emission kind, lowering pipeline id, runtime ABI id, runtime ABI kind/name,
runtime glue role, required capability refs, and artifact kind; it does not
mean that the compiler emitted LLVM/RISC-V/RVV IR, assembled an object, linked
a runtime, ran hardware, proved correctness, or measured performance.
Unsupported diagnostics should carry explicit plugin diagnostic text and are
valid evidence of a boundary, not of executable support. Unsupported
diagnostics may still carry plugin-owned runtime ABI kind/name and runtime glue
role metadata so the unsupported boundary is explicit; those fields are not
runtime ABI glue implementation or executable evidence. When a selected-path
lowering boundary was consumed, the diagnostic should also carry a generic
`lowering_boundary` metadata field naming the boundary operation used by the
plan. That field is a diagnostic link only and does not imply executable
lowering.

The current public `tcrv-opt` built-in registry includes the RVV first-slice
plugin. Therefore an `origin = "rvv-plugin"` selected path can materialize an
unsupported plugin-owned emission-plan diagnostic instead of a generic
unregistered-origin failure. This remains compiler metadata only and must not be
reported as RVV lowering, runtime, correctness, or performance evidence. The
tool-level `--tcrv-disable-builtin-plugins` option preserves an explicit
empty-registry surface for negative parser/diagnostic tests.

The `tcrv_rvv.lowering_boundary` first slice is one step more concrete than an
emission-plan diagnostic because it is an RVV extension-dialect op, but it is
still pre-executable and unsupported. It is valid evidence that selected-path
metadata reached a plugin-local RVV boundary. It is not evidence that the
compiler emitted LLVM/RISC-V/RVV IR, assembled an object, linked runtime glue,
ran hardware, proved correctness, or measured performance.

## RVV Probe Evidence Boundary

`scripts/rvv_remote_probe.py` may produce `ssh rvv` evidence under
`artifacts/tmp/rvv_probe/<run-id>/`. That artifact is a remote
hardware/toolchain evidence record, not an emission artifact. Even when its
minimal hand-written RVV intrinsic program compiles and runs, the result only
proves that the remote host can compile and execute that probe program.

Do not reinterpret an RVV probe artifact as:

- plugin-supported emission readiness;
- compiler lowering to LLVM, RISC-V, or RVV intrinsics;
- generated object, executable, runtime ABI glue, or dispatch lowering;
- TianChen-RV compiler correctness;
- RVV kernel performance.

Future supported RVV emission must add plugin-local lowering/runtime work and
then cite both compiler-generated artifact evidence and separate `ssh rvv`
hardware/toolchain evidence.

## RVV Smoke-Probe Target Export Boundary

### 1. Scope / Trigger

Trigger: post-planning MLIR contains a selected RVV path and a matching
plugin-owned `tcrv_rvv.lowering_boundary`, but RVV kernel lowering is still
unsupported. A target/export tool may emit a deterministic standalone C smoke
program to exercise the RVV host toolchain and `riscv_vector.h` path.

This export is hardware/toolchain smoke evidence only. It is not RVV lowering,
LLVM/RISC-V/RVV IR emission, runtime ABI glue, generated kernel object,
TianChen-RV kernel correctness, or performance evidence.

### 2. Signatures

Public command:

```text
tcrv-translate --tcrv-export-rvv-smoke-probe-c
```

C++ entry point:

```cpp
llvm::Error exportRVVSmokeProbeC(mlir::ModuleOp module,
                                 llvm::raw_ostream &os);
```

Expected pipeline use:

```bash
tcrv-opt input.mlir --tcrv-execution-planning-pipeline \
  | tcrv-translate --tcrv-export-rvv-smoke-probe-c > rvv_smoke_probe.c
```

### 3. Contracts

- Input must be real post-planning MLIR with one or more `tcrv.exec.kernel`
  operations.
- Each exported selected path must be a selected direct variant or dispatch case
  whose variant carries `origin = "rvv-plugin"`.
- The selected variant must require an available `rvv` capability through
  structured `requires = [@...]` metadata.
- The selected variant must carry bounded single-line
  `tcrv_rvv.required_march` metadata containing RVV vector evidence.
- The kernel capability set must preserve matching selected march metadata
  through available `rvv.probe.compile_run.selected_march` or
  `rvv.toolchain.march.value`.
- A matching direct child `tcrv_rvv.lowering_boundary` must identify the same
  `source_kernel`, `selected_variant`, `origin`, `role`, `status`, and
  `required_capabilities`.
- RVV smoke export must stay under RVV target/plugin-specific code. It must not
  add RVV branches to core orchestration, generic manifest export, or
  `tcrv.exec`.
- Output is deterministic standalone C using `riscv_vector.h` and tiny RVV
  intrinsics. It may include bounded comments naming selected kernel, variant,
  role, selected march, optional selected ABI, and required capability refs.
- Output must not serialize raw probe logs, credentials, absolute build paths,
  timestamps, benchmark sizes, latency/throughput numbers, manifest success, or
  runtime-success claims.

### 4. Validation & Error Matrix

- No `tcrv.exec.kernel` -> export fails before source output.
- Missing selected surface -> export fails before source output.
- Scalar-only, offload-only, or fallback-only selected paths -> export fails as
  not an RVV smoke-probe input.
- Selected RVV-like origin other than `rvv-plugin` -> export fails through a
  bounded unknown-origin diagnostic.
- Missing, malformed, unavailable, or non-symbol `requires` metadata -> export
  fails before source output.
- Selected variant does not require available capability id `rvv` -> export
  fails.
- Missing, non-RVV, secret-like, newline-containing, or unbounded
  `tcrv_rvv.required_march` -> export fails.
- Missing or mismatched preserved selected march capability metadata -> export
  fails.
- Missing, duplicate, stale, role-mismatched, status-mismatched, or
  required-capability-mismatched `tcrv_rvv.lowering_boundary` -> export fails.
- Any validation failure must leave stdout without partial C source.

### 5. Good/Base/Bad Cases

- Good: `tcrv-opt --tcrv-execution-planning-pipeline` selects an RVV dispatch
  case, materializes `tcrv_rvv.lowering_boundary`, preserves
  `tcrv_rvv.required_march = "rv64gcv"`, and the exporter emits stable C with
  `#include <riscv_vector.h>`.
- Base: multiple kernels with selected RVV paths are exported in deterministic
  symbol order with one bounded probe function per selected RVV path.
- Bad: an offload dispatch case plus scalar fallback is selected and the RVV
  smoke exporter silently emits a generic probe. It must fail instead because no
  selected `rvv-plugin` path exists.

### 6. Tests Required

- lit/FileCheck positive coverage for pipeline-to-export C source.
- Positive checks for deterministic multi-kernel ordering, bounded probe names,
  `riscv_vector.h`, selected kernel/variant comments, selected march metadata,
  and absence of manifest/runtime-success/raw-log/performance claims.
- Negative checks for scalar-only and offload-only selection.
- Negative checks for stale selected boundary, missing/malformed selected march,
  secret-like or newline metadata, unknown RVV-like origin, and unavailable RVV
  requirement.
- Full project check must still pass through `check-tianchenrv`.
- If an RVV runtime/toolchain claim is made, record separate `ssh rvv`
  compile/run evidence under `artifacts/tmp/...` and state that it proves only
  the generated smoke program compiled and ran.

### 7. Wrong vs Correct

Wrong:

```text
Generated RVV smoke probe compiled, therefore RVV kernel emission is supported.
```

Correct:

```text
Generated standalone RVV smoke probe C compiled and ran on ssh rvv; RVV kernel
emission remains unsupported/deferred until plugin-local kernel lowering and
runtime evidence are implemented.
```

## RVV Explicit Microkernel Target Export Boundary

### 1. Scope / Trigger

Trigger: post-planning MLIR contains one selected `rvv-plugin` path, a matching
plugin-owned `tcrv_rvv.lowering_boundary`, preserved selected march metadata,
and exactly one explicit `tcrv_rvv.i32_vadd_microkernel` op for that selected
kernel/variant.

This export is the first bounded RVV executable microkernel slice. It may emit
a deterministic standalone C program that computes and self-checks a finite i32
vector add using RVV intrinsics. It is not generic high-level MLIR lowering,
arbitrary RVV kernel emission, runtime ABI glue, benchmarking, or performance
evidence.

### 2. Signatures

Public command:

```text
tcrv-translate --tcrv-export-rvv-microkernel-c
```

C++ entry point:

```cpp
llvm::Error exportRVVMicrokernelC(mlir::ModuleOp module,
                                  llvm::raw_ostream &os);
```

### 3. Contracts

- Input must be real post-planning MLIR with one selected RVV path.
- The selected variant must be owned by `origin = "rvv-plugin"` and require an
  available capability whose id is `rvv`.
- The selected variant must carry bounded `tcrv_rvv.required_march`, and the
  enclosing kernel capability set must preserve matching selected march metadata
  through `rvv.probe.compile_run.selected_march` or
  `rvv.toolchain.march.value`.
- A matching direct child `tcrv_rvv.lowering_boundary` must identify the same
  source kernel, selected variant, origin, role, status, and required
  capability refs. The boundary may remain `status = "unsupported"` because
  generic RVV emission readiness is still deferred.
- A matching direct child `tcrv_rvv.i32_vadd_microkernel` must identify the same
  selected path, required capability refs, required march, optional selected
  mabi, and bounded element count.
- Output must be deterministic standalone C with `riscv_vector.h`, RVV i32
  load/add/store intrinsics, fixed local arrays, and a self-checking `main`.
- Output must not include timestamps, absolute paths, raw logs, credentials,
  benchmark sizes, latency/throughput numbers, or performance claims.

### 4. Validation & Error Matrix

- Missing selected RVV path, scalar-only path, offload-only path, or
  fallback-only path -> export fails before source output.
- Missing, duplicate, stale, role-mismatched, status-mismatched, or
  required-capability-mismatched `tcrv_rvv.lowering_boundary` -> export fails.
- Missing, duplicate, stale, selected-variant-mismatched,
  required-capability-mismatched, invalid-element-count, malformed-march, or
  secret-like `tcrv_rvv.i32_vadd_microkernel` -> export fails.
- Missing, unavailable, non-symbol, non-RVV, or unknown selected variant
  requirements -> export fails.
- Missing or mismatched preserved selected march capability metadata -> export
  fails.
- Any validation failure must leave stdout without partial C source and must not
  fall back to `--tcrv-export-rvv-smoke-probe-c`.

### 5. Evidence Interpretation

Real `ssh rvv` compile/run evidence for generated microkernel C proves only that
the explicit generated i32 vector-add microkernel compiled and passed its
self-check on that host with the selected compiler flags. It does not prove
generic TianChen-RV lowering correctness, supported arbitrary RVV kernel
emission, runtime ABI support, or performance.

### 6. Emission Plan / Manifest Handoff

When the explicit microkernel op matches the selected RVV path, the RVV plugin
may return a supported emission plan for the standalone C source export route:

```text
status: supported
emission kind: rvv-explicit-i32-vadd-microkernel-c-source
lowering pipeline: tcrv-export-rvv-microkernel-c
runtime ABI: rvv-i32-vadd-standalone-c-self-check.v1
runtime ABI kind/name: RVV standalone C source export metadata
runtime glue role: standalone self-check main
artifact kind: standalone-c-source
```

This supported plan is a plugin-owned compiler handoff to the existing RVV
microkernel exporter. It is more concrete than the default unsupported RVV
first-slice plan, but it is still not generic RVV lowering, runtime ABI glue,
object generation, arbitrary kernel emission, correctness evidence, or
performance evidence by itself. The generic emission manifest may serialize the
record after validating the selected surface, the `tcrv_rvv.lowering_boundary`
link, and the emission-plan diagnostic. It must not reinterpret the adjacent
`tcrv_rvv.i32_vadd_microkernel` attachment as a second lowering boundary.

## Scalar Fallback Metadata Boundary

The first scalar fallback plugin slice may return a metadata-only emission
readiness result and materialize a metadata-only emission-plan diagnostic for
the portable fallback route:

```text
status: metadata-only
emission kind: portable-scalar-fallback-metadata-route
lowering pipeline: none-executable-metadata-only
runtime ABI: none-metadata-only
runtime ABI kind: host-scalar-fallback-metadata
runtime ABI name: portable-scalar-fallback-metadata-abi.v1
runtime glue role: metadata-only-host-fallback-boundary
artifact kind: metadata-diagnostic
```

This is still compiler-decision metadata. It does not prove that TianChen-RV
emitted LLVM IR, generated an object, linked a runtime, executed a scalar
kernel, proved correctness, or measured performance. Later scalar fallback
lowering must add plugin-local lowering code and validation artifacts before
reporting executable support.

The selected scalar fallback boundary is slightly more concrete than an
emission-plan diagnostic because it is a scalar extension-dialect op:
`tcrv_scalar.lowering_boundary`. It records selected fallback metadata and
required capability references, but it remains metadata-only and non-executable.
It is not evidence that the compiler emitted LLVM IR, assembled an object,
linked runtime glue, ran a scalar kernel, proved correctness, or measured
performance.

## Scalar Explicit Microkernel Target Export Boundary

Trigger: post-planning MLIR contains one selected `scalar-plugin` fallback
path, a matching plugin-owned `tcrv_scalar.lowering_boundary`, and exactly one
explicit `tcrv_scalar.i32_vadd_microkernel` op for that selected
kernel/variant/role.

This export is the first bounded scalar fallback executable source slice. It
may emit a deterministic portable standalone C program that computes and
self-checks a finite i32 vector add. It is not generic scalar lowering,
arbitrary scalar kernel emission, runtime ABI glue, object generation, linking,
benchmarking, or performance evidence.

Public route:

```text
tcrv-translate --tcrv-export-target-source-artifact
```

Route metadata:

```text
status: supported
emission kind: scalar-explicit-i32-vadd-microkernel-c-source
lowering pipeline: tcrv-export-scalar-microkernel-c
runtime ABI: scalar-i32-vadd-standalone-c-self-check.v1
runtime ABI kind/name: scalar standalone C source export metadata
runtime glue role: standalone self-check main
artifact kind: standalone-c-source
```

Contracts:

- Input must be real post-planning MLIR with one selected scalar fallback path.
- The selected variant must be owned by `origin = "scalar-plugin"` and require
  an available capability whose id is `scalar.fallback`.
- A matching direct child `tcrv_scalar.lowering_boundary` must identify the
  same source kernel, selected variant, origin, role, metadata-only status, and
  required capability refs.
- A matching direct child `tcrv_scalar.i32_vadd_microkernel` must identify the
  same selected path and required capability refs with a bounded element count.
- Output must be deterministic standalone portable C with fixed local arrays,
  scalar i32 add, and a self-checking `main`.
- Output must not include RVV headers, RVV intrinsics, route-spoof claims,
  timestamps, absolute paths, raw logs, credentials, benchmark sizes,
  latency/throughput numbers, or performance claims.

Missing selected scalar path, missing or stale scalar lowering boundary,
missing or stale scalar microkernel, unavailable fallback capability, unknown
route id, unsupported artifact kind, route spoofing, offload-only paths, and
ambiguous multiple supported artifacts must fail before source output.

## Runtime Offload Metadata Boundary

The first runtime-offload plugin slice may return a metadata-only emission
readiness result and materialize a metadata-only emission-plan diagnostic for a
generic runtime ABI handoff:

```text
status: metadata-only
emission kind: runtime-offload-handoff-metadata-route
lowering pipeline: offload-runtime-boundary-metadata-only
runtime ABI: generic-runtime-offload-c-abi-handoff.v1
runtime ABI kind: runtime-offload-c-abi-handoff
runtime ABI name: generic-runtime-offload-c-abi-handoff.v1
runtime glue role: plugin-owned-runtime-offload-glue-boundary
artifact kind: metadata-handoff-manifest
```

This is compiler handoff metadata for downstream runtime integration. It does
not emit vendor runtime calls, allocate or copy device buffers, compile
accelerator kernels, generate objects, link runtime libraries, run hardware,
prove correctness, or measure performance. The selected
`tcrv_offload.lowering_boundary` op records the plugin-local runtime-offload
handoff boundary, but it remains metadata-only and non-executable. Its
availability depends on explicit `offload.runtime` capability metadata and
plugin legality; it cannot resurrect unavailable, malformed, illegal, or
unselected variants.

## Emission Manifest Export Boundary

After planning has materialized an explicit selected surface and
`tcrv.exec.diagnostic {reason = "emission_plan"}` metadata, a target/export
tool may serialize a deterministic emission handoff manifest. The manifest is a
compiler handoff artifact for downstream lowering/runtime-glue integration. It
may include module and kernel symbols, selected variant symbols, origin plugin,
selected-path role, dispatch cases and fallback targets, lowering-boundary
diagnostic links, runtime ABI kind/name, runtime glue role, emission status,
required capability refs, preference metadata, and bounded plugin
explanation/diagnostic text.

The exporter must validate the post-planning MLIR before writing any manifest.
It must reject missing selected surfaces, stale selected references, selected
paths without exactly one complete emission-plan diagnostic, duplicate runtime
ABI ownership diagnostics, malformed runtime ABI ownership fields, unknown
diagnostic targets, and unbounded or credential-like explanatory text. Export
failure must not produce a partial manifest.

The manifest does not perform plugin registry routing, legality recovery,
variant selection, lowering-boundary materialization, LLVM lowering, RVV
intrinsic emission, scalar code generation, object generation, runtime library
linking, hardware execution, correctness validation, or performance
measurement. It must remain target-neutral: it serializes generic selected-path
and diagnostic metadata without interpreting RVV, scalar fallback, IME, offload,
Sophgo, AME, vendor, dtype, shape, microarchitecture, probe evidence, runtime,
correctness, or performance semantics.
