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
`provides`, `implies`, and `conflicts`. A kernel may also reference one
module-level capability-provider `tcrv.exec.target` profile with
`target = @profile`; that attached profile may declare additional module-level
providers through the generic
`capability_providers = [@provider, ...]` composition contract. Only the
explicitly attached profile, its validated composed providers, and direct
kernel-local providers enter the kernel's `TargetCapabilitySet`.
Relation-aware provider lookup lets a profile capability such as
`id = "rvv.profile.rv64gcv", provides = ["rvv"]` satisfy plugin proposals that
require capability id `rvv`, while composed providers can make fallback policy,
scalar fallback, toolchain, or build-policy capability objects visible to
planning without a frontend-specific provider list. Exact capability ids remain
authoritative when present. These relations participate in compiler
decisions such as RVV plugin proposal, variant materialization, and RVV
legality. The generic `--tcrv-check-capability-requires` pass also uses a
bounded conflict query: unprotected static variants and dispatch fallbacks fail
when a required available capability conflicts with another available
capability, while dispatch cases must carry an explicit generic guard surface
before a conflicting requirement can remain dispatch-protected. Full conflict
solving, provider ranking, and automatic conflict resolution remain future
work.

The built-in RVV first slice registers the concrete MLIR namespace
`tcrv_rvv` through the RVV plugin path. It includes metadata/control-plane
surfaces such as `!tcrv_rvv.vl`, `#tcrv_rvv.policy`, `tcrv_rvv.setvl` for
bounded runtime AVL-to-VL control, and `tcrv_rvv.with_vl` for the matching
bounded VL scope region, plus finite `tcrv_rvv.i32_load`,
`tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, `tcrv_rvv.i32_mul`, and
`tcrv_rvv.i32_store` dataflow ops for the current non-executable explicit
extension-family IR surface. The setvl surface is control-plane IR only: it
consumes a runtime AVL SSA value, returns a `!tcrv_rvv.vl` token, and carries
bounded first-slice SEW/LMUL/policy metadata. The with_vl surface consumes that
runtime VL token and creates a single-block plugin-local region. Its first
bounded dataflow payload is the i32 add/sub/mul lhs-load, rhs-load,
arithmetic, output-store sequence, but it is not a generic RVV memory model,
arbitrary vector lowering, full runtime ABI, exporter contract, or evidence.
These RVV surfaces do not by themselves lower to LLVM/RISC-V, create runtime
ABI glue, generate objects, run hardware, prove correctness, or measure
performance.

Selected-path lowering-boundary materialization is routed through the generic
extension plugin registry. The public `tcrv-opt` pass
`--tcrv-materialize-selected-lowering-boundaries` delegates selected direct,
dispatch-case, and dispatch-fallback variant references to their origin plugin.
RVV currently requires an explicit typed `tcrv_rvv` variant body before it will
accept legality, but selected RVV lowering-boundary materialization still
returns a no-boundary result until a future materialized EmitC lowering route
exists. Scalar fallback remains a generic dispatch/fallback envelope and does
not currently materialize a scalar plugin-local selected-boundary operation.

Emission-plan materialization also routes through the selected variant's origin
plugin. The resulting `tcrv.exec.diagnostic {reason = "emission_plan"}`
metadata records bounded plugin-owned runtime ABI ownership fields such as
runtime ABI kind/name, runtime glue role, selected variant, lowering boundary,
status, required capability refs, and structured `runtime_abi_parameters` when
a future plugin route supports them. RVV selected paths currently fail closed
without supported runtime ABI metadata because no materialized EmitC route
exists. These diagnostics are compiler-decision metadata only: they are not
executable code, runtime ABI glue, correctness evidence, or performance
evidence.

The `tcrv-translate --tcrv-export-emission-manifest` tool exports a
deterministic compiler handoff manifest from post-planning MLIR that already
contains selected-path, lowering-boundary, and plugin-owned emission-plan
metadata. The manifest is intended for downstream lowering/runtime-glue tooling
to consume the generic handoff contract. It does not emit LLVM/RISC-V/RVV IR,
generate objects, link runtime libraries, run hardware, prove correctness, or
measure performance.

The former public linalg/vector RVV source-to-exec pass family is deleted as a
core semantic branch. `tcrv-opt` no longer registers the historical source,
linalg RVV, linalg i32 compatibility, or vector i32 arithmetic source adapter
options, and those option spellings are not preserved as active route
contracts. Current planning starts from already materialized TianChen-RV
execution surfaces such as `tcrv.exec.kernel`, capability-provider scope,
selected-path diagnostics or future materialized boundaries,
`tcrv.exec.mem_window`, `tcrv.exec.runtime_param`, and plugin-local extension
family ops. Any future high-level frontend rebuild must be plugin/interface
owned and must not restore core transforms that inspect finite RVV
linalg/vector source bodies or query RVV family records to materialize
`tcrv.exec`.

The former RVV standalone smoke-probe compiler front doors are deleted.
Selected RVV metadata or typed RVV IR without a materialized EmitC route is not
enough to synthesize a standalone C harness through `tcrv-translate`; the
generic source-only artifact export entrypoint is deleted. Explicit RVV
hardware/toolchain probes belong in separate probe tooling and recorded `ssh
rvv` artifacts, not in a compiler source-artifact entrypoint.

The historical RVV, scalar, and RVV+scalar runtime-callable direct C semantic
exporters are deleted production routes. Selected metadata, family records,
route records, or descriptor-like records must not be translated directly into
kernel C source, headers, objects, self-check sources, or target-artifact
bundles. The removed direct translate options are absent, and surviving
generic target-artifact front doors fail closed for those deleted route ids
until a future rebuild materializes a real MLIR EmitC module and emits C/C++
through the MLIR emitter.

The generic target-artifact front doors remain coherence gates, not alternate
direct-C backdoors. They may still reject stale RVV/scalar/dispatch
runtime-callable route metadata with bounded unsupported diagnostics, and they
may continue to serve non-semantic or separately contracted routes. They must
not synthesize executable kernel bodies from selected metadata as a compatibility
path.

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
RVV capability profile, including bounded `vlenb_bytes` and derived
`i32_m1_lane_count` when the probe program can read the RVV `vlenb` CSR.
Python remains evidence/artifact tooling and is not the compiler capability
model.

The former compiler-owned RVV smoke-probe source export route is deleted.
Post-planning metadata and typed RVV IR without a materialized EmitC route must
not be used to print a standalone probe program, intrinsic C body, or
source-to-toolchain evidence artifact. RVV hardware/toolchain evidence remains
the responsibility of explicit probe tooling such as
`scripts/rvv_remote_probe.py` plus recorded `ssh rvv` artifacts.

The previous hand-written/test `linalg.generic` frontend slice and its
`tcrv-opt` lowering flags are deleted. Run the execution-planning pipeline only
on input that already contains the TianChen-RV execution anchors it consumes,
such as a `tcrv.exec.kernel` with capability-provider scope and any required
ABI boundary operations. Planning still owns plugin proposal, legality,
variant selection, selected lowering-boundary materialization, emission-plan
diagnostics, and coherence checks. Any RVV runtime or correctness claim from
artifacts exported after this point still requires separate real `ssh rvv`
evidence.

The former RVV microkernel direct source/object/self-check evidence bridge has
been removed with the direct C semantic exporters. Do not use historical
post-planning metadata or deleted route ids as RVV runtime/correctness
evidence. A future evidence bridge must start from a materialized MLIR EmitC
module and the MLIR C/C++ emitter, then record separate real `ssh rvv`
compile/run evidence for the concrete artifact under test.

## Scalar Fallback First Slice

The built-in plugin registry also includes a C++ `scalar-plugin` first slice.
It proposes `@scalar_fallback_first_slice` only when the target kernel declares
an available structured capability provider for `scalar.fallback`, either as an
exact capability or through an explicit module target/profile relation such as
`provides = ["scalar.fallback"]`, and it participates in the same legality,
cost, selection, emission-readiness, and emission-plan interfaces as other
plugins.

This scalar fallback path is a conservative portable fallback proposal and
dispatch envelope only. It does not add a new high-level compute op, generic
scalar lowering, metadata-only emission route, runtime ABI integration, object
generation, correctness evidence, or performance evidence.

The scalar plugin no longer treats a descriptor string, default family table, or
frontend marker as compute authority. A selected scalar path can preserve
`tcrv_scalar.element_count` only as bounded selected-path metadata; there is no
current typed scalar microkernel attachment authority. The element count is not
tensor shape, AVL, vl, runtime loop trip count, correctness coverage, or
performance evidence.

The scalar fallback plugin reserves the concrete `tcrv_scalar` MLIR namespace
for future scalar lowering work, but the namespace currently exposes no active
selected-boundary operation. Scalar fallback also has no supported direct
runtime-callable C source/header/object exporter. Historical scalar route
metadata must fail closed instead of synthesizing portable C bodies from family
records. A future scalar artifact route must be rebuilt through the shared
extension-family ops to EmitC module path.

## Host RVV + Scalar Dispatch First Slice

The historical RVV+scalar host dispatch direct C exporter is deleted. The
former dispatch source/header/object/self-check and target-artifact-bundle
routes must fail closed instead of embedding generated RVV and scalar callable
C bodies and then synthesizing a dispatcher body. The old e2e dry-run bridge
was also removed because it treated selected metadata plus route records as a
valid executable C evidence path.

Future dispatch executable evidence must be rebuilt through extension-family
ops, a materialized common EmitC module, and the MLIR C/C++ emitter before any
real `ssh rvv` compile/run evidence can be claimed.

## Runtime Offload First Slice

The built-in registry includes a generic C++ `offload-plugin` first slice for
runtime-offload handoff metadata. It is enabled only by an explicit structured
provider for `offload.runtime`: either an exact `offload.runtime` capability
with `kind = "runtime-offload"` or a module target/profile relation such as
`provides = ["offload.runtime"]`, with bounded generic handoff properties.
Vendor strings, Sophgo names, RVV facts, or ordinary target attributes do not
enable it unless they explicitly provide the generic offload capability and
preserve the required handoff properties.

The offload plugin owns the concrete `tcrv_offload` MLIR namespace. Its first
operation, `tcrv_offload.lowering_boundary`, records selected runtime-offload
handoff metadata such as source kernel, selected variant, origin plugin,
selected-path role, required capability references, runtime ABI handoff id, and
metadata-only status. The emission-plan and manifest paths may serialize this
generic runtime ABI handoff for downstream integration. They do not emit vendor
runtime calls, implement DMA or buffer management, generate accelerator objects,
run hardware, prove correctness, or measure performance.

The previous offload descriptor-only target artifact route is deleted. A
selected offload path may still carry capability-gated lowering-boundary and
runtime-ABI handoff metadata for diagnostics and manifests, but generic target
artifact export must fail closed until a future route is rebuilt through
extension-family ops, a materialized EmitC/runtime C ABI handoff, and
target-owned artifact generation. This metadata is not offload runtime
execution, hardware correctness evidence, or performance evidence.
