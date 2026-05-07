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

The C++ capability model consumes direct `tcrv.exec.capability` anchors and
preserves structured properties plus first-slice capability relations:
`provides`, `implies`, and `conflicts`. Relation-aware provider lookup lets a
profile capability such as `id = "rvv.profile.rv64gcv", provides = ["rvv"]`
satisfy plugin proposals that require capability id `rvv`, while an exact
capability id remains authoritative when present. These relations participate
in compiler decisions such as RVV plugin proposal, variant materialization, and
RVV legality; full conflict solving remains future work.

The built-in RVV first slice registers the concrete MLIR namespace
`tcrv_rvv` through the RVV plugin path. It includes metadata/control-plane
surfaces such as `!tcrv_rvv.vl`, `#tcrv_rvv.policy`, `tcrv_rvv.setvl` for
bounded runtime AVL-to-VL control, and `tcrv_rvv.with_vl` for the matching
bounded VL scope region, plus the finite `tcrv_rvv.i32_vadd_dataflow` marker
for the current i32-vadd microkernel export route and
`tcrv_rvv.lowering_boundary` for selected RVV variants. The setvl surface is
control-plane IR only: it consumes a runtime AVL
SSA value, returns a `!tcrv_rvv.vl` token, and carries bounded first-slice
SEW/LMUL/policy metadata. The with_vl surface consumes that runtime VL token and
creates a single-block plugin-local region. Its first bounded dataflow payload
is exactly the i32-vadd marker consumed by the RVV exporter; this is not a
generic RVV memory model, arbitrary vector lowering, full runtime ABI, or
evidence. The lowering boundary is pre-executable compiler metadata only: these
RVV surfaces do not by themselves lower to LLVM/RISC-V, create runtime ABI glue,
generate objects, run hardware, prove correctness, or measure performance.

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
status, required capability refs, and structured `runtime_abi_parameters` for
supported callable C source exports. Those parameter entries record C name, C
type spelling, semantic role, and whether the value is IR-modeled or
target/export ABI-owned. These diagnostics are compiler-decision metadata only:
they are not executable code, runtime ABI glue, correctness evidence, or
performance evidence.

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
deterministic library-style C source artifact for exactly one selected
`tcrv_rvv.i32_vadd_microkernel` op attached to a selected RVV path. This is the
first plugin-local RVV executable microkernel slice: post-planning MLIR must
contain a selected `rvv-plugin` variant, matching `tcrv_rvv.lowering_boundary`,
preserved selected march metadata, and the bounded RVV microkernel op. The op
may come from an explicit fixture or from the RVV plugin materializing the
finite `tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"` selected
variant descriptor during the execution-planning pipeline. The microkernel op
now carries a structured RVV body with one runtime index body argument for
target/export-owned `n`/AVL, one `tcrv_rvv.setvl`, one matching
`tcrv_rvv.with_vl`, and one nested finite `tcrv_rvv.i32_vadd_dataflow` marker
for the target/export-owned lhs input, rhs input, output, and runtime element
count ABI roles consumed by the exporter. Descriptor-local `element_count`
remains metadata and is not promoted to AVL or VL. The generated source uses
`riscv_vector.h` and RVV i32 add intrinsics to expose a deterministic
runtime-callable C ABI function. The default ABI metadata names those
parameters `lhs`, `rhs`, `out`, and `n`, and supported emission-plan metadata
may provide different concrete C names for the same roles:
`void <generated_name>(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)`.
The exporter validates and consumes that `setvl` / `with_vl` /
`i32_vadd_dataflow` role body before emitting the runtime-callable loop and
resolves concrete C parameter names/types structurally from runtime ABI
metadata when present, so mismatched or stale control/dataflow metadata fails
before source output. The default artifact has no embedded `main` or
self-check harness, so later runtime glue can embed it and call the ABI
boundary directly. The explicit
`tcrv-translate --tcrv-export-rvv-microkernel-self-check-c` helper emits the
same callable function plus a bounded self-check `main` for evidence
collection.
Real `ssh rvv` compile/run evidence for that harness source proves only that
this bounded generated microkernel source compiled and that the harness passed
on the selected host flags. It is not generic high-level lowering, arbitrary
RVV kernel executable emission, full runtime integration, or performance
evidence; selected RVV paths without the finite descriptor or matching
microkernel op remain unsupported and deferred.

The `tcrv-translate --tcrv-export-target-source-artifact` tool adds a generic
target artifact routing front door for supported post-planning emission-plan
metadata. The route is selected from compiler-owned selected-path,
lowering-boundary, and plugin-owned emission-plan diagnostics, then dispatched
through a registered target-owned exporter only after a generic execution-plan
coherence preflight validates that selected-path, lowering-boundary,
runtime-ABI, emission-plan, and artifact-route metadata still describe the same
path, including the structured ABI parameter roles required by the target-owned
source exporter. Registered source routes are bounded to plugin-local
microkernel attachments: the RVV i32 vector-add microkernel C exporter above
and the scalar fallback explicit i32 vector-add portable runtime-callable C
exporter below.
Unsupported
metadata-only RVV/scalar paths, offload paths, unknown routes, stale selected
paths, missing boundaries, missing microkernels, route spoofing, and ambiguous
multiple supported artifacts fail closed. This tool does not add generic RVV or
scalar lowering, arbitrary source export, full runtime ABI integration, object
generation, linking, correctness evidence, or performance evidence.

The `tcrv-translate --tcrv-export-target-artifact` tool is the artifact-kind
aware generic front door. It uses the same selected-path, lowering-boundary,
and plugin-owned emission-plan route metadata, but it is not limited to source
artifacts. It runs the same generic execution-plan coherence preflight before
artifact dispatch. The first non-source route is the offload runtime handoff
descriptor: a deterministic target-owned text descriptor for a selected
`offload-plugin` path with a matching `tcrv_offload.lowering_boundary`, runtime
ABI metadata, required capability refs, and supported descriptor emission plan.
This descriptor is compiler handoff metadata only. It does not emit vendor
runtime calls, implement DMA or buffer management, generate accelerator
objects, link runtime libraries, run offload hardware, prove correctness, or
measure performance.

When the selected RVV path has that exact microkernel attachment, either
explicitly authored or materialized by the RVV plugin from the finite descriptor,
the RVV plugin may also materialize a supported emission-plan diagnostic and
the generic emission manifest may serialize the handoff as a deterministic
runtime-callable C source export route. That manifest record is still compiler
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

For the first microkernel slice, use post-planning MLIR that contains the
selected RVV path and matching `tcrv_rvv.i32_vadd_microkernel`. The op may be
an explicit fixture attachment or a plugin-materialized op produced from the
finite RVV i32-vadd descriptor by `tcrv-opt --tcrv-execution-planning-pipeline`:

```bash
tcrv-translate --tcrv-export-rvv-microkernel-c post_planning_microkernel.mlir \
  > rvv_microkernel.c
tcrv-translate --tcrv-export-target-source-artifact post_planning_microkernel.mlir \
  > rvv_microkernel.c
```

Compile and run that source on `ssh rvv` with the selected `-march` and, when
present, selected `-mabi`. The resulting evidence is bounded to the i32
vector-add microkernel self-check and must not be reported as generic
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
uses the explicit self-check harness export and adds bounded remote compile/run
evidence for the generated `tcrv_rvv.i32_vadd_microkernel` callable ABI plus
self-check harness only. It is not generic RVV lowering, full runtime
integration, arbitrary kernel emission, broad correctness coverage, or
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

For the bounded i32 vector-add slice, the scalar proposal also carries the
finite plugin-owned descriptor
`tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"` with
descriptor-local `tcrv_scalar.element_count = 16`. When that selected path is
materialized, the scalar plugin creates both the selected
`tcrv_scalar.lowering_boundary` and one matching
`tcrv_scalar.i32_vadd_microkernel`; the element count is descriptor metadata,
not tensor shape, AVL, vl, runtime loop trip count, correctness coverage, or
performance evidence.

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
and preserved `scalar.fallback` capability metadata, and it may be generated
directly from the descriptor above. When present, the scalar plugin reports a
supported runtime-callable C source-export emission plan routed through
`tcrv-translate --tcrv-export-target-source-artifact`. The generated C is a
library-style portable scalar i32 vector-add function with the same structural
pointer-plus-length callable ABI shape used by the bounded RVV microkernel
route; the default artifact has no embedded `main` or self-check harness and
uses no RVV headers or intrinsics. This is a callable fallback source artifact
for later host dispatch glue. It is not generic scalar lowering, arbitrary
scalar source export, object/linking support, runtime integration, correctness
coverage beyond this explicit microkernel, or performance evidence. For generic
single-artifact export, a supported primary non-fallback route wins over a
supported `dispatch fallback` candidate; scalar-only selected fallback remains
exportable when it is the only supported route.

## Host RVV + Scalar Dispatch First Slice

The `tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c` tool exports
the first bounded host-side runtime dispatch C source artifact for the finite
i32 vector-add slice. The input must already contain one selected
`rvv-plugin` dispatch case with a matching `tcrv_rvv.lowering_boundary`,
runtime-callable RVV i32-vadd microkernel metadata, and supported RVV emission
plan, plus one selected `scalar-plugin` dispatch fallback with a matching
`tcrv_scalar.lowering_boundary`, runtime-callable scalar i32-vadd microkernel
metadata, and supported scalar emission plan.

The normal `--tcrv-execution-planning-pipeline` can now produce both callable
sides for the built-in RVV+scalar dispatch fixture: RVV from the finite RVV
descriptor and scalar fallback from the finite scalar descriptor. A hand-authored
scalar microkernel is no longer required for that pipeline-to-dispatch-export
path.

The generated source embeds the existing deterministic RVV runtime-callable C
function and scalar runtime-callable fallback C function, then emits a stable
dispatcher ABI:

```c
void tcrv_dispatch_i32_vadd_<kernel>(const int32_t *lhs,
                                     const int32_t *rhs,
                                     int32_t *out, size_t n,
                                     int rvv_available);
```

For the current bounded RVV, scalar, and RVV+scalar dispatch source exports,
`lhs`, `rhs`, `out`, runtime `n`, and dispatcher `rvv_available` are structured
target/export-owned ABI parameters unless a future IR surface models them
directly. Descriptor-local `element_count` remains finite microkernel metadata;
it is not high-level shape, runtime `n`, AVL, or VL. The `rvv_available`
argument is an explicit host-provided guard. This export
does not implement automatic hardware probing, object generation, dynamic
loading, linking, benchmarking, correctness measurement, or performance
measurement. RVV runtime/correctness/performance claims still require separate
real `ssh rvv` evidence.

The explicit
`tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c`
helper emits the same bounded dispatcher source plus a small `main` that calls
the dispatcher with `rvv_available = 0` and `rvv_available = 1` over fixed local
arrays. This helper exists only for bounded runtime invocation evidence of the
current RVV+scalar i32-vadd dispatch slice. A successful `ssh rvv` compile/run
of that generated source proves only that this dispatcher harness passed on the
selected host flags; it is not generic RVV lowering, object generation, dynamic
runtime integration, performance evidence, or broad correctness coverage.

The
`tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object`
tool is the first bounded object-file artifact route for that same dispatcher
self-check source. It reuses the validated RVV+scalar self-check C generation
path, then invokes `clang` with the selected RVV `-march` and optional
`-mabi` metadata carried by the selected target capabilities to produce one
ELF relocatable object on stdout. This route fails closed if the selected-path,
lowering-boundary, emission-plan, runtime ABI parameter metadata, selected
compile facts, `clang` tool, target headers, or local/native RISC-V toolchain
setup are unavailable. The object is still only the bounded dispatcher
self-check artifact; it is not generic RVV lowering, dynamic runtime
integration, linking, automatic hardware probing, performance evidence, or
broad correctness coverage.

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
