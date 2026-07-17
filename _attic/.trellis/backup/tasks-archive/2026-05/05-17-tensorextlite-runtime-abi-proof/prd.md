# TensorExtLite Materialized Bundle Runtime ABI Proof

## Goal

Prove that the existing TensorExtLite first-slice route can be consumed as a
runtime ABI artifact set: TianChen-RV MLIR source-front-door input must produce
explicit TensorExtLite role ops, a materialized EmitC module, generated C++
source, an object/header/bundle, and a local native harness that calls the
generated ABI function through the generated header.

The proof must extend the current artifact-generation evidence into ABI
consumption evidence without adding a new extension family, source language,
descriptor route, direct-C semantic exporter, source-export route, or
core/common TensorExtLite semantic branch.

## Requirements

- Keep the bounded TensorExtLite fragment-MMA first slice as the only module
  owner.
- Preserve the existing production artifact route identity and behavior:
  source front door -> explicit TensorExtLite role ops -> materialized EmitC
  route -> MLIR EmitC C/C++ emitter -> target object/header/bundle.
- Export a coherent object/header/bundle through the existing
  construction-protocol-backed materialized EmitC route.
- Generate a local native runtime proof object from the same materialized EmitC
  C++ output, include the generated declaration-only header in a small harness,
  link both locally with clang/LLVM, run the harness, and record a PASS result.
- The harness may define the external TensorExtLite role callees and verify a
  deterministic call-order/result. The route-owned compute/control sequence
  must still come from the explicit TensorExtLite role ops lowered through
  materialized EmitC, not from header or bundle metadata.
- Header and bundle content must remain non-semantic ABI packaging: declaration
  and bounded evidence metadata only, no compute body, descriptor body, source
  export, direct-C exporter, runtime log, or performance claim.
- Positive evidence must tie the PASS result to selected variant, origin
  plugin, construction protocol, EmitC route, runtime ABI name, ordered ABI
  parameter list, and bundle component group.
- Negative coverage must fail closed for missing construction protocol
  metadata, mismatched header/object route identity, stale source-front-door
  metadata used as artifact authority, unsupported artifact kinds,
  direct-C/source-export route residue, descriptor metadata as legal input, and
  header/bundle content that embeds compute bodies.

## Acceptance Criteria

- [x] A focused runtime ABI proof runner or lit test exports TensorExtLite
      post-planning MLIR, generated C++ source, generated header, generated
      object/bundle/index, harness source, compile/link command records, and
      run output under an evidence directory.
- [x] The harness compiles a native proof object from the generated
      materialized EmitC C++ output, compiles an external ABI consumer against
      the generated header, links locally with clang/LLVM, runs, and emits a
      PASS marker.
- [x] The PASS marker is checked against selected variant
      `tensorext_lite_tile_mma_first_slice`, origin plugin
      `tensorext-lite-plugin`, construction protocol
      `extension-family-construction-protocol.v1`, route
      `tensorext-lite-fragment-mma-emitc-route`, runtime ABI
      `tensorext-lite-fragment-mma-runtime-c-abi.v1`, zero ordered ABI
      parameters, and bundle component group
      `tensorext-lite-fragment-mma-materialized-emitc-bundle.v1`.
- [x] Existing TensorExtLite C++ target artifact coverage continues to reject
      missing/stale route metadata, stale role sequence/source ops,
      descriptor metadata, unexpected runtime ABI parameters, missing or wrong
      runtime ABI identity, wrong selected role, and wrong origin.
- [x] Existing lit coverage continues to reject stale source-front-door
      metadata, missing lowering boundary, unsupported target artifact input,
      and non-materialized EmitC attempts.
- [x] The generated header and bundle index are checked to be declaration /
      metadata only and not compute-body authority.
- [x] Targeted residue scans over TensorExtLite plugin/target/tests and common
      target surfaces show no descriptor route authority, direct C semantic
      exporter, source-export route, Python compiler-core path, or common/core
      TensorExtLite semantic branch introduced.
- [x] Focused TensorExtLite plugin test, target artifact export test, relevant
      lit filter, `git diff --check`, and `check-tianchenrv` if practical pass.

## Definition Of Done

- Task status, notes, and archive state are truthful.
- Evidence artifacts are present in a small `artifacts/tmp/...` directory and
  the final report names that path.
- One coherent commit records the PRD, implementation, tests, and task archive
  when the acceptance criteria are complete.

## Technical Approach

Add a bounded evidence runner for TensorExtLite runtime ABI consumption. The
runner will invoke existing MLIR/C++ compiler front doors (`tcrv-opt`,
`tcrv-translate`) to produce the post-planning MLIR, generated C++ source,
header, and target artifact bundle. It will then compile the same generated
materialized EmitC C++ source as a local native proof object, compile a harness
that includes the generated header and provides the external role callees, link
both locally with clang/LLVM, and execute the resulting binary.

The current production object in the target artifact bundle remains the
existing `riscv-elf-relocatable-object` route output. The local native proof
object is only a runtime ABI consumption proof for the same materialized EmitC
source and header; it is not a new target artifact route and not a replacement
for the target bundle object.

## Decision (ADR-lite)

Context: The existing TensorExtLite target object route intentionally packages
the materialized EmitC C++ source as a RISC-V relocatable object. The current
developer machine is `x86_64`, so that production object cannot be linked and
run by a local native harness without changing the target artifact route.

Decision: Preserve the production TensorExtLite target object/header/bundle
route unchanged, and add a local native runtime ABI proof that compiles the
same generated materialized EmitC C++ source into a host proof object, includes
the generated declaration header, links a harness, and executes it.

Consequences: The proof demonstrates ABI consumption and call sequencing for
the generated TensorExtLite runtime boundary without claiming TensorExtLite
hardware performance or changing the RISC-V artifact route. Future hardware
runtime claims still require target-appropriate evidence.

## Out Of Scope

- New TensorExtLite operations, new TensorExtLite families, RVV work,
  high-level frontend lowering, performance matrices, broad smoke reports,
  compatibility wrappers, legacy modes, descriptor adapters, direct C semantic
  exporters, source-export routes, Python compiler-core behavior, or core/common
  TensorExtLite semantic branches.
- Replacing the existing RISC-V target artifact route with a host artifact
  route.
- Claiming performance or target hardware correctness.

## Technical Notes

- Required specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-17-tensorextlite-construction-protocol-consumption/prd.md`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h`,
  `lib/Plugin/TensorExtLite/Construction/TensorExtLiteConstructionProtocol.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`, and
  `lib/Target/TargetArtifactExport.cpp`.
- Primary tests:
  `test/Plugin/TensorExtLiteExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Transforms/TensorExtLite/`,
  `test/Conversion/EmitC/tensorext-lite-first-slice-materialization*.mlir`,
  and `test/Target/TensorExtLite/`.

## Completion Summary

- Added `scripts/tensorextlite_runtime_abi_e2e.py`, an evidence runner that
  invokes the existing MLIR/C++ front doors, exports the TensorExtLite
  post-planning MLIR, generated C++ source, declaration header, standalone
  target object, object/header bundle, readobj output, harness source,
  native proof object, link command records, and run output.
- Added `test/Target/TensorExtLite/tensorext-lite-runtime-abi-harness.test`
  so the runtime ABI consumption proof is part of lit coverage.
- Added a lit feature/substitution for local native `clang++` so the harness
  test only runs when the local toolchain can compile and execute a native
  C++ binary.
- Preserved the existing production TensorExtLite target object route as a
  `riscv-elf-relocatable-object`. The local native proof object is compiled
  from the same generated materialized EmitC C++ source strictly to prove
  ABI consumption on this `x86_64` developer machine.
- The harness includes the generated declaration header, provides the external
  role callees, calls the generated ABI function, and verifies the role order
  `configure,load_frag,tile_mma,store_frag`.
- The evidence runner also checks fail-closed behavior for stale
  source-front-door metadata, unsupported artifact kind, and wrong route
  identity, and verifies header/bundle content does not embed compute bodies.
- Targeted production scans found no TensorExtLite descriptor route authority,
  direct C semantic exporter, source-export route, Python compiler-core path,
  or common/core TensorExtLite semantic branch.

## Validation

- `python3 -m py_compile scripts/tensorextlite_runtime_abi_e2e.py`
- `python3 scripts/tensorextlite_runtime_abi_e2e.py --artifact-root artifacts/tmp/tensorextlite_runtime_abi_e2e --run-id manual-smoke --tcrv-opt ./build/bin/tcrv-opt --tcrv-translate ./build/bin/tcrv-translate --clangxx /usr/lib/llvm-20/bin/clang++ --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='tensorext-lite-runtime-abi-harness' .`
- From `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='TensorExtLite|tensorext-lite' .`
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`
- Targeted residue scans over TensorExtLite plugin/target/tests and common
  target surfaces.

## Evidence

- Manual evidence directory:
  `artifacts/tmp/tensorextlite_runtime_abi_e2e/manual-smoke`
- PASS marker:
  `PASS tianchenrv.tensorext_lite.runtime_abi_e2e selected_variant=tensorext_lite_tile_mma_first_slice origin_plugin=tensorext-lite-plugin construction_protocol=extension-family-construction-protocol.v1 emitc_route=tensorext-lite-fragment-mma-emitc-route runtime_abi_name=tensorext-lite-fragment-mma-runtime-c-abi.v1 runtime_abi_parameter_count=0 component_group=tensorext-lite-fragment-mma-materialized-emitc-bundle.v1 native_call_trace=configure,load_frag,tile_mma,store_frag`
