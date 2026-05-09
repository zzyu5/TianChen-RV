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
`tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store` dataflow ops for the current
i32-vadd microkernel export route and
`tcrv_rvv.lowering_boundary` for selected RVV variants. The setvl surface is
control-plane IR only: it consumes a runtime AVL
SSA value, returns a `!tcrv_rvv.vl` token, and carries bounded first-slice
SEW/LMUL/policy metadata. The with_vl surface consumes that runtime VL token and
creates a single-block plugin-local region. Its first bounded dataflow payload
is exactly the i32-vadd lhs-load, rhs-load, add, output-store sequence consumed
by the RVV exporter; this is not a generic RVV memory model, arbitrary vector
lowering, full runtime ABI, or evidence. The lowering boundary is
pre-executable compiler metadata only: these
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
supported callable C source exports. For the bounded i32-vadd RVV/scalar
callable routes, those entries are derived from direct `tcrv.exec.mem_window`
buffer ABI boundaries plus direct `tcrv.exec.runtime_param` scalar ABI/control
boundaries and are validated mirrors of that IR-backed callable plan. These
diagnostics are compiler-decision metadata only: they are not executable code,
runtime ABI glue, correctness evidence, or performance evidence.

The `tcrv-translate --tcrv-export-emission-manifest` tool exports a
deterministic compiler handoff manifest from post-planning MLIR that already
contains selected-path, lowering-boundary, and plugin-owned emission-plan
metadata. The manifest is intended for downstream lowering/runtime-glue tooling
to consume the generic handoff contract. It does not emit LLVM/RISC-V/RVV IR,
generate objects, link runtime libraries, run hardware, prove correctness, or
measure performance.

The public `tcrv-opt --tcrv-lower-linalg-i32-vadd-to-exec` pass is the first
bounded frontend lowering slice from high-level MLIR into TianChen-RV execution
surfaces. It accepts only explicitly marked hand-written/test `linalg.generic`
i32 vector-add wrappers whose region body is exactly the current two-input
`arith.addi` / `linalg.yield` shape. The pass materializes one
`tcrv.exec.kernel` with a source-selected `target = @profile` module target
reference, validates the selected target profile's generic capability-provider
composition, and creates the IR-backed i32-vadd callable ABI boundary:
`tcrv.exec.mem_window` for lhs/rhs/out buffers and
`tcrv.exec.runtime_param` for runtime `n`. The resulting kernel is directly
consumable by `--tcrv-execution-planning-pipeline`, so existing RVV/scalar
plugin proposal, legality, selection, selected-boundary, emission-plan, and
target-artifact routes remain reused. This pass is not generic linalg lowering,
does not add a `tcrv` compute op, does not infer arbitrary tensor semantics,
does not invent target capabilities, does not lower to LLVM/RISC-V, and does
not create runtime correctness or performance evidence.

The `tcrv-translate --tcrv-export-rvv-smoke-probe-c` tool exports a
deterministic standalone C RVV hardware/toolchain smoke probe from post-planning
MLIR that has selected RVV metadata and a matching
`tcrv_rvv.lowering_boundary`. The generated source uses `riscv_vector.h` and a
tiny RVV intrinsic load/add/store check so it can be compiled and run on the
`ssh rvv` host as toolchain evidence. It is not TianChen-RV kernel lowering,
kernel executable emission, runtime ABI glue, kernel correctness evidence, or
performance evidence, and it does not change the RVV first-slice unsupported
emission boundary.
When a selected RVV path is explicitly planned as a smoke-probe descriptor, the
same emitter is also reachable through the registry-driven generic
`tcrv-translate --tcrv-export-target-source-artifact` front door as a
`standalone-c-source` artifact. That generic route still emits only the bounded
toolchain smoke program and does not turn the path into kernel lowering,
runtime integration, correctness evidence, or performance evidence.

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
`tcrv_rvv.with_vl`, and a nested finite `tcrv_rvv.i32_load`,
`tcrv_rvv.i32_load`, `tcrv_rvv.i32_add`, `tcrv_rvv.i32_store` body. The load
and store ops reference the target/export-owned lhs input, rhs input, and
output buffer ABI roles; the runtime element-count role remains a
`tcrv.exec.runtime_param` ABI boundary consumed by the callable plan rather
than an RVV dataflow operand. Descriptor-local `element_count`
is selected from structured RVV i32 M1 lane capacity when available, capped as
a bounded sample, and otherwise falls back to the first-slice sample size 16;
it remains metadata and is not promoted to shape, runtime `n`, AVL, VL,
correctness coverage, or performance evidence. The generated source uses
`riscv_vector.h` and RVV i32 add intrinsics to expose a deterministic
runtime-callable C ABI function. The callable parameter roles, C type spellings,
and deterministic lhs/rhs/out/runtime-count order are derived from direct
`tcrv.exec.mem_window` buffer boundaries and direct
`tcrv.exec.runtime_param` runtime-count boundaries; supported emission-plan
metadata must mirror that IR-backed plan exactly:
`void <generated_name>(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)`.
The exporter validates and consumes that `setvl` / `with_vl` /
explicit load/add/store dataflow body before emitting the runtime-callable loop and
validates any supported emission-plan parameter metadata as an exact mirror of
the IR-backed callable plan, so mismatched or stale control/dataflow/ABI
metadata fails before source output. The default artifact has no embedded
`main` or self-check harness, so later runtime glue can embed it and call the
ABI boundary directly. The explicit
`tcrv-translate --tcrv-export-rvv-microkernel-self-check-c` helper emits the
same callable function plus a bounded self-check `main` for evidence
collection. That harness uses descriptor-local `element_count` only as bounded
local-array capacity and calls the generated ABI function with explicit
runtime `n` values, including a shorter runtime count and the bounded capacity,
so `n` remains caller-provided runtime ABI state.
The `tcrv-translate --tcrv-export-rvv-microkernel-object` helper compiles the
same validated library-style source into one RISC-V ELF relocatable object
using the structured RVV architecture capability, selected compile capability
metadata, and local `clang`. For direct selected RVV paths, the generic
`--tcrv-export-target-artifact` front door can select that bounded object route
while `--tcrv-export-target-source-artifact` remains source-only. The object
route has no hidden `main`, does not link or run, and does not perform
automatic RVV probing.
The matching `tcrv-translate --tcrv-export-rvv-microkernel-header` helper and
the generic `--tcrv-export-target-header-artifact` front door emit the bounded
runtime-callable C ABI header for the same selected RVV i32-vadd microkernel
path. The header is derived from the same selected path, validated
microkernel, structured callable ABI plan, mem_window/runtime_param boundaries,
and capability metadata as the source/object routes. It contains only an
include guard, standard integer/size includes, the `extern "C"` guard, and the
single callable prototype; it has no body, RVV intrinsics, hidden `main`,
self-check harness, runtime probing, correctness evidence, or performance
text. Header selection is routed separately from source/object artifact
selection: the source-only front door remains source-only and the default
artifact front door continues to choose the object route when that route is
requested or supported.
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
source exporter. Registered source routes are bounded to target-owned explicit
artifacts: the RVV standalone smoke-probe C source exporter above, the RVV i32
vector-add microkernel C exporter above, the scalar fallback explicit i32
vector-add portable runtime-callable C exporter below, and the RVV+scalar
i32-vadd host dispatch C composite exporter when the selected plan contains
both supported callable sides.
Unsupported
metadata-only RVV/scalar paths, offload paths, unknown routes, stale selected
paths, missing boundaries, missing microkernels, route spoofing, and ambiguous
multiple supported artifacts fail closed. For a planned selected RVV dispatch
case plus scalar dispatch fallback, the generic route delegates to the
target-owned dispatch exporter instead of silently exporting only the primary
callable. This tool does not add generic RVV or scalar lowering, arbitrary
source export, full runtime ABI integration, object generation, linking,
correctness evidence, or performance evidence.

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

The same generic artifact front door can also select target-owned bounded
RISC-V ELF relocatable object routes for scalar fallback i32-vadd microkernel
paths, direct RVV i32-vadd microkernel paths, or RVV+scalar i32-vadd dispatch
library-object composite paths. In those cases
`--tcrv-export-target-artifact` prefers the bounded non-source
runtime-callable object route, while
`--tcrv-export-target-source-artifact` remains source-only. The object routes
emit from validated library-style source plus structured target/toolchain
capability metadata. Scalar object export requires an available `rv64`
capability provider with `riscv64` architecture metadata and selected RISC-V
compile facts such as `riscv.toolchain.march`; RVV object export uses the
structured RVV architecture and selected compile capability metadata. They do
not add a hidden `main` or self-check harness, link, run hardware, auto-probe
RVV availability, prove correctness, or measure performance.

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
RVV capability profile, including bounded `vlenb_bytes` and derived
`i32_m1_lane_count` when the probe program can read the RVV `vlenb` CSR.
Python remains evidence/artifact tooling and is not the compiler capability
model.

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

For the first frontend slice, start from a hand-written/test `linalg.generic`
i32 vector-add wrapper with `tcrv_frontend_lowering = "i32-vadd"`,
`tcrv_frontend_kernel = "<kernel-symbol>"`, and
`tcrv_frontend_target = @<module-target-profile>`, then run:

```bash
tcrv-opt input_linalg.mlir \
  --tcrv-lower-linalg-i32-vadd-to-exec \
  --tcrv-execution-planning-pipeline
```

The first pass only creates the `tcrv.exec` kernel and ABI boundary. The
planning pipeline still owns plugin proposal, legality, variant selection,
selected lowering-boundary materialization, emission-plan diagnostics, and
coherence checks. Any RVV runtime or correctness claim from artifacts exported
after this point still requires separate real `ssh rvv` evidence.

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

Compile and run the explicit self-check source on `ssh rvv` with the selected
`-march` and, when present, selected `-mabi`. The resulting evidence is bounded
to the i32 vector-add microkernel self-check for the explicit runtime `n`
values reported by the generated success marker and must not be reported as
generic TianChen-RV RVV lowering correctness or performance.

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
exports the generated runtime-callable C header and RISC-V relocatable object,
creates an explicit external C caller under `artifacts/tmp`, copies those three
inputs to `ssh rvv`, compiles/links the external caller against the generated
object, runs it, and checks the finite i32-vadd result. That evidence is
bounded to this generated header plus generated object external caller
correctness check only. It is not generic RVV lowering, full runtime
integration, arbitrary kernel emission, broad correctness coverage, or
performance evidence.

## Scalar Fallback First Slice

The built-in plugin registry also includes a C++ `scalar-plugin` first slice.
It proposes `@scalar_fallback_first_slice` only when the target kernel declares
an available structured capability provider for `scalar.fallback`, either as an
exact capability or through an explicit module target/profile relation such as
`provides = ["scalar.fallback"]`, and it participates in the same legality,
cost, selection, emission-readiness, and emission-plan interfaces as other
plugins.

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
uses no RVV headers or intrinsics.

The same validated scalar callable candidate can now feed
`--tcrv-export-target-header-artifact` and
`--tcrv-export-target-artifact`. The header route emits only the external C
prototype. The object route emits the scalar library source internally and
compiles a RISC-V ELF relocatable object with local `clang`; it requires
structured `rv64` architecture metadata and selected RISC-V compile facts such
as `riscv.toolchain.march`, with optional MABI metadata from
`riscv.toolchain.mabi` or compatible preserved profile facts. These are
callable fallback artifacts for later host dispatch or external callers. They
are not generic scalar lowering, arbitrary scalar source export, linked runtime
integration, correctness coverage beyond this explicit microkernel, RVV
hardware evidence, or performance evidence. For generic single-artifact export,
a supported primary non-fallback route wins over a supported `dispatch
fallback` candidate; scalar-only selected fallback remains exportable when it
is the only supported route. The bounded RVV+scalar dispatcher is the
target-owned composite exception: when both selected callable sides are present,
generic source export emits the dispatch source.

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

The same bounded dispatch source is also reachable through the coherence-gated
generic source artifact route when the selected plan contains both supported
callable sides:

```bash
tcrv-opt input.mlir --tcrv-execution-planning-pipeline \
  | tcrv-translate --tcrv-export-target-source-artifact \
  > rvv_scalar_dispatch.c
```

The generated source embeds the existing deterministic RVV runtime-callable C
function and scalar runtime-callable fallback C function, then emits a stable
dispatcher ABI. With default IR-backed ABI boundaries this ABI is:

```c
void tcrv_dispatch_i32_vadd_<kernel>(const int32_t *lhs,
                                     const int32_t *rhs,
                                     int32_t *out, size_t n,
                                     int rvv_available);
```

The matching
`tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-header` tool emits
the bounded C ABI header for that dispatcher. The header is generated from the
same exec-IR-backed dispatch ABI plan as the dispatch source/object route: direct
`tcrv.exec.mem_window` supplies the `lhs`, `rhs`, and `out` buffer parameters,
direct `tcrv.exec.runtime_param` supplies runtime `n`, and the selected RVV
`tcrv.exec.case runtime_guard` supplies the explicit dispatch availability
parameter. It contains only the include guard, required standard integer/size
headers, the `extern "C"` guard, and the dispatcher prototype. It does not
embed callable bodies, computation, a `main`, a self-check harness, runtime
probing, linking logic, correctness evidence, or performance evidence.

For emission-plan-backed dispatch export, the RVV and scalar callable
candidates must both mirror the same IR-backed callable ABI plan: direct
`tcrv.exec.mem_window` IR supplies the callable `lhs`, `rhs`, and `out` buffer
meanings and C pointer types, while direct `tcrv.exec.runtime_param` IR supplies
the callable runtime element count. Candidate parameter metadata is accepted
only as an exact mirror of those boundaries, not as an independent source of
callable names or types. Calls to both embedded callables are emitted in the
fixed lhs/rhs/output/runtime-count role order. The dispatch wrapper then appends
the explicit `dispatch-availability-guard` `tcrv.exec.runtime_param` as
dispatch-only control through the selected RVV `tcrv.exec.case runtime_guard`
symbol reference; detached guard metadata is not the branch-control source. The
guard is not part of the RVV or scalar callable microkernel signatures. The
default guard C name remains `rvv_available`, while the symbol-linked
runtime_param may provide a different valid C name. Runtime `n` and the
dispatcher availability guard remain ABI/control parameters, not tensor shapes
or hardware facts. Descriptor-local `element_count` remains finite microkernel
metadata; it is not high-level shape, runtime `n`, AVL, or VL. This export does
not implement automatic hardware probing, object generation, dynamic loading,
linking, benchmarking, correctness measurement, or performance measurement.
RVV runtime/correctness/performance claims still require separate real
`ssh rvv` evidence.

The explicit
`tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c`
helper emits the same bounded dispatcher source plus a small `main` that calls
the dispatcher with explicit runtime element-count ABI values for both
`rvv_available = 0` and `rvv_available = 1` over bounded local arrays. The
current harness covers `n = 7` and `n = 16`, so the runtime loop bound is a
caller-provided ABI/control input rather than descriptor-local `element_count`
metadata. This helper exists only for bounded runtime invocation evidence of
the current RVV+scalar i32-vadd dispatch slice. A successful `ssh rvv`
compile/run of that generated source proves only that this dispatcher harness
passed on the selected host flags; it is not generic RVV lowering, object
generation, dynamic runtime integration, performance evidence, or broad
correctness coverage.

The `tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-object` tool is
the bounded library object route for that same dispatcher. It reuses the
validated RVV+scalar default dispatch C generation path, then invokes `clang`
with the structured RVV architecture capability as the compile target plus the
selected RVV `-march` and optional `-mabi` metadata carried by the selected
target capabilities to produce one ELF relocatable object on stdout. The object
defines the embedded RVV callable, embedded scalar callable, and dispatcher ABI
symbols, but it has no `main` or self-check helper. This route fails closed if
the selected-path, lowering-boundary, emission-plan, runtime ABI parameter
metadata, selected architecture/compile facts, `clang` tool, target headers, or
local/native RISC-V toolchain setup are unavailable. The object is still only
the bounded dispatcher library artifact; it is not generic RVV lowering,
dynamic runtime integration, linking, automatic hardware probing, performance
evidence, or broad correctness coverage.

The explicit
`tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object`
helper remains available for evidence-oriented object generation from the
self-check source. It may contain `main` and the bounded local-array harness, and
it is intentionally not the generic `--tcrv-export-target-artifact` route.

The same object route is also available through the generic artifact-kind-aware
front door:

```bash
tcrv-opt input.mlir --tcrv-execution-planning-pipeline \
  | tcrv-translate --tcrv-export-target-artifact \
  > tcrv_dispatch.o
```

The source-only front door continues to emit the library-style dispatcher C
source without an embedded `main` or self-check harness.

For a bounded input that has kernel/capability anchors but does not already
contain selected-path diagnostics, selected lowering-boundary metadata, or
emission-plan diagnostics, the bundle handoff is also reachable through a
single C++ translate front door:

```bash
tcrv-translate --tcrv-plan-and-export-target-artifact-bundle \
  --tcrv-target-artifact-bundle-output-dir=<existing-output-dir> \
  input.mlir
```

That command first runs the bounded marked-linalg i32-vadd frontend lowering
slice, then runs the existing built-in execution planning pipeline in-process,
and finally calls the same target artifact bundle exporter used by
`--tcrv-export-target-artifact-bundle`. Inputs that already contain
`tcrv.exec.kernel` anchors are unchanged by the frontend lowering pass; marked
`linalg.generic` i32-vadd inputs become the same `tcrv.exec.kernel` +
`mem_window` / `runtime_param` ABI boundary consumed by the plugin pipeline.
The existing bundle export command remains the coherence-gated exporter for
already planned input. Neither command claims generic linalg lowering, linking,
runtime success, RVV correctness, or performance by itself.

Together, the explicit dispatch header and the library-object route form the
bounded runtime-caller handoff for this finite i32-vadd dispatcher: an external
C caller can compile against the emitted prototype and link the generated
RISC-V relocatable object. That handoff remains a compiler artifact boundary;
runtime or correctness claims still require separate real `ssh rvv` evidence.

The bounded executable evidence bridge below drives the planned dispatch
pipeline, exports the default generic library-style dispatch source and the
explicit target-owned self-check source, and optionally compiles that generated
self-check source on `ssh rvv` into a relocatable object, links an executable,
and runs it:

```bash
python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run
python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle
python3 scripts/rvv_scalar_dispatch_e2e.py --ssh-target rvv
python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --ssh-target rvv
```

Dry-run mode writes sanitized post-planning MLIR, emission manifest, generated
dispatch C sources, hashes, and command summaries under
`artifacts/tmp/rvv_scalar_dispatch_e2e/<run-id>/` without making a runtime
claim. Real ssh mode copies the compiler-generated self-check source to the RVV
host, compiles it with selected `-march`/optional `-mabi`, links the executable,
and runs the harness that calls both `rvv_available = 0` and
`rvv_available = 1` branches with explicit runtime `n = 7` and `n = 16` values.
A successful run proves only that this finite RVV+scalar i32-vadd dispatcher
self-check executable passed on the selected RVV host flags for those runtime
count ABI inputs. It is not generic RVV lowering, arbitrary kernel support,
dynamic runtime integration, performance evidence, or broad correctness
coverage.

The optional target-artifact-bundle mode writes sanitized bundle evidence under
`artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/<run-id>/`. Dry-run bundle
mode proves only that the compiler exported the selected registry-derived
bundle, the bridge parsed the bundle index, discovered the generated
source/header/object files, and generated an external caller. Real ssh bundle
mode copies the generated source, generated header, generated relocatable
object, and generated external caller to `ssh rvv`; compiles the generated
dispatch source on that host, links and runs the external caller against the
source-built object, then also links and runs the same caller against the
bundle object. A successful run proves only the bounded RVV+scalar i32-vadd
bundle external ABI handoff for those compiler-produced bundle artifacts and
the explicit runtime `n = 7` and `n = 16` caller inputs.

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

When that selected offload path has the matching plugin-owned lowering boundary
and a supported descriptor emission plan, the target artifact route can export
a deterministic runtime handoff descriptor:

```bash
tcrv-opt input.mlir --tcrv-execution-planning-pipeline \
  | tcrv-translate --tcrv-export-target-artifact \
  > offload_runtime_descriptor.txt
```

The descriptor contains only sanitized compiler-visible fields such as source
kernel, selected variant, origin plugin, selected role, descriptor schema
version, descriptor kind/status, adapter contract, required capabilities,
runtime ABI kind/name, emission kind, artifact kind, lowering-boundary op name
and status, runtime-offload handoff kind, handoff reason, and explicit
non-claim metadata. Through the generic target artifact bundle exporter, the
selected non-fallback offload descriptor path materializes as one descriptor
artifact plus a bundle index entry carrying stable route, owner, runtime ABI,
and handoff kind metadata; a scalar fallback candidate is not emitted as a
second single artifact unless a target-owned composite route explicitly matches.
This handoff is not offload runtime execution, hardware correctness evidence,
or performance evidence.
