# TianChen-RV MLIR

TianChen-RV MLIR is a capability-driven execution layer for extensible RISC-V AI kernels after high-level MLIR. It does not introduce a new high-level tensor or tile IR; it organizes target capabilities, extension plugins, execution variants, legality, dispatch, lowering, and fallback.

## Project Spine

```text
High-level MLIR op
  -> target capability model
  -> extension plugin registry
  -> plugin-proposed execution variants
  -> legality verification
  -> capability-aware variant selection / dispatch
  -> plugin-owned lowering / emission / runtime glue
  -> RVV / IME / offload / fallback executable path
```

## Current Bootstrap

This repository is bootstrapped as a conventional MLIR project:

```text
include/TianChenRV/
lib/
tools/tcrv-opt/
test/
cmake/
CMakeLists.txt
```

The first compiler slice defines a minimal `tcrv.exec.*` operation family through TableGen/ODS and C++ registration. MLIR registers the concrete namespace as `tcrv` so operations parse and print as `tcrv.exec.kernel`, `tcrv.exec.variant`, and related core execution ops. The family is intentionally limited to execution organization concepts such as kernels, targets, capabilities, variants, dispatch, and fallback. Concrete computation belongs in extension dialects such as `tcrv.rvv`, `tcrv.ime`, `tcrv.offload`, or future plugin-local dialects.

The built-in RVV first slice registers the concrete MLIR namespace
`tcrv_rvv` through the RVV plugin path. It includes metadata/control-plane
surfaces such as `!tcrv_rvv.vl`, `#tcrv_rvv.policy`, and
`tcrv_rvv.lowering_boundary` for selected RVV variants. The lowering boundary
is pre-executable compiler metadata only: it does not emit RVV intrinsics,
lower to LLVM/RISC-V, create runtime ABI glue, generate objects, run hardware,
prove correctness, or measure performance.

Selected-path lowering-boundary materialization is routed through the generic
extension plugin registry. The public `tcrv-opt` pass
`--tcrv-materialize-selected-lowering-boundaries` delegates selected direct,
dispatch-case, and dispatch-fallback variant references to their origin plugin.
RVV materializes `tcrv_rvv.lowering_boundary` for selected RVV paths; scalar
fallback materializes `tcrv_scalar.lowering_boundary` for selected portable
fallback paths. Both surfaces are compiler metadata only and do not claim
executable lowering.

Emission-plan materialization also routes through the selected variant's origin
plugin. The resulting `tcrv.exec.diagnostic {reason = "emission_plan"}`
metadata records bounded plugin-owned runtime ABI ownership fields such as
runtime ABI kind/name, runtime glue role, selected variant, lowering boundary,
status, and required capability refs. These diagnostics are compiler-decision
metadata only: they are not executable code, runtime ABI glue, correctness
evidence, or performance evidence.

The `tcrv-translate --tcrv-export-emission-manifest` tool exports a
deterministic compiler handoff manifest from post-planning MLIR that already
contains selected-path, lowering-boundary, and plugin-owned emission-plan
metadata. The manifest is intended for downstream lowering/runtime-glue tooling
to consume the generic handoff contract. It does not emit LLVM/RISC-V/RVV IR,
generate objects, link runtime libraries, run hardware, prove correctness, or
measure performance.

The `tcrv-translate --tcrv-export-rvv-smoke-probe-c` tool exports a
deterministic standalone C RVV hardware/toolchain smoke probe from post-planning
MLIR that has selected RVV metadata and a matching
`tcrv_rvv.lowering_boundary`. The generated source uses `riscv_vector.h` and a
tiny RVV intrinsic load/add/store check so it can be compiled and run on the
`ssh rvv` host as toolchain evidence. It is not TianChen-RV kernel lowering,
kernel executable emission, runtime ABI glue, kernel correctness evidence, or
performance evidence, and it does not change the RVV first-slice unsupported
emission boundary.

The `tcrv-translate --tcrv-export-rvv-microkernel-c` tool exports a distinct
deterministic standalone C executable for exactly one explicit
`tcrv_rvv.i32_vadd_microkernel` op attached to a selected RVV path. This is the
first plugin-local RVV executable microkernel slice: the input MLIR must already
contain a selected `rvv-plugin` variant, matching `tcrv_rvv.lowering_boundary`,
preserved selected march metadata, and the explicit bounded RVV microkernel op.
The generated source uses `riscv_vector.h`, RVV i32 add intrinsics, fixed local
arrays, and a self-checking `main`. It proves only that this explicit generated
microkernel source can compile and pass its self-check when real `ssh rvv`
evidence is recorded. It is not generic high-level lowering, arbitrary RVV
kernel executable emission, runtime ABI glue, or performance evidence; default
RVV selected paths without the explicit microkernel op remain unsupported and
deferred.

The `tcrv-translate --tcrv-export-target-source-artifact` tool adds a generic
target artifact routing front door for supported post-planning emission-plan
metadata. The route is selected from compiler-owned selected-path,
lowering-boundary, and plugin-owned emission-plan diagnostics, then dispatched
through a registered target-owned exporter. Registered source routes are
bounded to explicit plugin-local microkernel attachments: the existing RVV
microkernel C exporter above and the scalar fallback explicit i32 vector-add
portable C exporter below. Unsupported metadata-only RVV/scalar paths, offload
paths, unknown routes, stale selected paths, missing boundaries, missing
microkernels, route spoofing, and ambiguous multiple supported artifacts fail
closed. This tool does not add generic RVV or scalar lowering, arbitrary source
export, runtime ABI integration, object generation, linking, correctness
evidence, or performance evidence.

The `tcrv-translate --tcrv-export-target-artifact` tool is the artifact-kind
aware generic front door. It uses the same selected-path, lowering-boundary,
and plugin-owned emission-plan route metadata, but it is not limited to source
artifacts. The first non-source route is the offload runtime handoff descriptor:
a deterministic target-owned text descriptor for a selected `offload-plugin`
path with a matching `tcrv_offload.lowering_boundary`, runtime ABI metadata,
required capability refs, and supported descriptor emission plan. This
descriptor is compiler handoff metadata only. It does not emit vendor runtime
calls, implement DMA or buffer management, generate accelerator objects, link
runtime libraries, run offload hardware, prove correctness, or measure
performance.

When the selected RVV path has that exact explicit microkernel attachment, the
RVV plugin may also materialize a supported emission-plan diagnostic and the
generic emission manifest may serialize the handoff as a deterministic
standalone C source export route. That manifest record is still compiler
handoff metadata: it points downstream tooling to
`tcrv-translate --tcrv-export-rvv-microkernel-c` and does not claim generic RVV
lowering, runtime ABI integration, arbitrary kernel emission, correctness, or
performance by itself.

## Build

Configure with an installed LLVM/MLIR package:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
cmake --build build
```

If LLVM/MLIR CMake packages or required tools are missing, configuration fails with an explicit diagnostic. The project must not replace MLIR compiler internals with Python data structures.

## Test

Run lit/FileCheck tests after building:

```bash
cmake --build build --target check-tianchenrv
```

Python is used only for support tooling such as lit configuration, runners, probes, supervision scripts, and artifact parsing. Core IR, dialects, operations, passes, plugin registries, capability models, lowering, and emission belong in C++ / MLIR / LLVM / TableGen / CMake.

## Hardware Evidence

The current real hardware mainline is RVV 1.0 via `ssh rvv`. Any RVV correctness, runtime, or performance claim must include real `ssh rvv` evidence. Local CMake, `tcrv-opt`, or lit checks are compiler/toolchain evidence only.

Use the bounded RVV probe to record hardware/toolchain availability without
claiming TianChen-RV compiler lowering or runtime support:

```bash
python3 scripts/rvv_remote_probe.py
```

The probe writes sanitized JSON evidence and command logs under
`artifacts/tmp/rvv_probe/<run-id>/`. Its tiny RVV intrinsic compile/run check
only proves remote RVV header/toolchain/program availability; it is not a
TianChen-RV-generated executable, correctness result, or performance result.
The JSON also exposes sanitized `capability_facts` for the plugin-local C++
RVV capability profile; Python remains evidence/artifact tooling and is not the
compiler capability model.

After `tcrv-opt --tcrv-execution-planning-pipeline` has selected an RVV path,
the generated smoke probe can be exported and compiled on `ssh rvv` for a
narrower source-to-toolchain evidence slice:

```bash
tcrv-opt input.mlir --tcrv-execution-planning-pipeline \
  | tcrv-translate --tcrv-export-rvv-smoke-probe-c > rvv_smoke_probe.c
```

This proves only that the exported standalone smoke program can compile and run
on the RVV host when separate `ssh rvv` evidence is recorded. It does not prove
TianChen-RV lowered a selected kernel, generated an object for that kernel,
linked runtime glue, produced a correctness result, or measured performance.

For the explicit first microkernel slice, use a post-planning fixture or
pipeline output that has been combined with `tcrv_rvv.i32_vadd_microkernel`:

```bash
tcrv-translate --tcrv-export-rvv-microkernel-c post_planning_microkernel.mlir \
  > rvv_microkernel.c
tcrv-translate --tcrv-export-target-source-artifact post_planning_microkernel.mlir \
  > rvv_microkernel.c
```

Compile and run that source on `ssh rvv` with the selected `-march` and, when
present, selected `-mabi`. The resulting evidence is bounded to the explicit
i32 vector-add microkernel self-check and must not be reported as generic
TianChen-RV RVV lowering correctness or performance.

The helper below ties the existing manifest-supported microkernel route to
source export and optional real `ssh rvv` evidence without broadening compiler
semantics:

```bash
python3 scripts/rvv_microkernel_e2e.py --dry-run
python3 scripts/rvv_microkernel_e2e.py --dry-run --generic-route
python3 scripts/rvv_microkernel_e2e.py --ssh-target rvv
```

The dry-run mode runs local compiler tools only and writes sanitized
post-planning MLIR, emission manifest, generated C source, hashes, and command
summaries under `artifacts/tmp/rvv_microkernel_e2e/<run-id>/`. Real ssh mode
adds bounded remote compile/run evidence for the explicit generated
`tcrv_rvv.i32_vadd_microkernel` self-check only. It is not generic RVV lowering,
runtime ABI integration, arbitrary kernel emission, correctness coverage, or
performance evidence.

## Scalar Fallback First Slice

The built-in plugin registry also includes a C++ `scalar-plugin` first slice.
It proposes `@scalar_fallback_first_slice` only when the target kernel declares
an available structured capability `scalar.fallback`, and it participates in
the same legality, cost, selection, emission-readiness, and emission-plan
interfaces as other plugins.

This scalar fallback path is compiler metadata for a portable fallback route.
It marks a generic conservative fallback role for dispatch synthesis and emits
metadata-only readiness/plan diagnostics by default. It does not add a new
high-level compute op, generic scalar lowering, runtime ABI integration, object
generation, correctness evidence, or performance evidence.

The scalar fallback plugin also owns the concrete `tcrv_scalar` MLIR namespace.
Its first operation, `tcrv_scalar.lowering_boundary`, records selected fallback
boundary metadata such as source kernel, selected variant, origin plugin,
selected-path role, required capability references, and metadata-only status.
It is a plugin-local attachment point for future scalar lowering work, not
scalar computation, LLVM lowering, runtime ABI glue, object generation,
correctness evidence, or performance evidence.

The scalar dialect now also has one bounded explicit microkernel attachment:
`tcrv_scalar.i32_vadd_microkernel`. It is valid only for a selected
`scalar-plugin` fallback path with a matching `tcrv_scalar.lowering_boundary`
and preserved `scalar.fallback` capability metadata. When present, the scalar
plugin may report a supported standalone C source-export emission plan routed
through `tcrv-translate --tcrv-export-target-source-artifact`. The generated C
is portable scalar i32 vector add with a self-checking `main`; it uses no RVV
headers or intrinsics. This proves only the local generated scalar C self-check
when compiled and run locally. It is not generic scalar lowering, runtime ABI
integration, arbitrary scalar source export, object/linking support,
correctness coverage beyond this explicit microkernel, or performance evidence.

## Runtime Offload First Slice

The built-in registry includes a generic C++ `offload-plugin` first slice for
runtime-offload handoff metadata. It is enabled only by an explicit
`offload.runtime` capability with `kind = "runtime-offload"` and bounded generic
handoff properties; vendor strings, Sophgo names, RVV facts, or ordinary target
attributes do not enable it.

The offload plugin owns the concrete `tcrv_offload` MLIR namespace. Its first
operation, `tcrv_offload.lowering_boundary`, records selected runtime-offload
handoff metadata such as source kernel, selected variant, origin plugin,
selected-path role, required capability references, runtime ABI handoff id, and
metadata-only status. The emission-plan and manifest paths may serialize this
generic runtime ABI handoff for downstream integration. They do not emit vendor
runtime calls, implement DMA or buffer management, generate accelerator objects,
run hardware, prove correctness, or measure performance.

When that selected offload path has the matching plugin-owned lowering boundary
and a supported descriptor emission plan, the target artifact route can export
a deterministic runtime handoff descriptor:

```bash
tcrv-opt input.mlir --tcrv-execution-planning-pipeline \
  | tcrv-translate --tcrv-export-target-artifact \
  > offload_runtime_descriptor.txt
```

The descriptor contains only sanitized compiler-visible fields such as source
kernel, selected variant, origin plugin, required capabilities, runtime ABI
kind/name, emission kind, artifact kind, lowering-boundary op name, and handoff
reason. It is not offload runtime execution, hardware correctness evidence, or
performance evidence.
