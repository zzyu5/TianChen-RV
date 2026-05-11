# Generated TCRV EmitC lowerable op interface for RVV i32 family ops

## Goal

Add a generated MLIR/TableGen EmitC lowerable op-interface boundary for the
bounded RVV i32 add/sub/mul family dataflow ops, and make the RVV target/export
path prove that this generated interface is consumed when constructing the
common EmitC lowerable route. The existing common C++ route object remains the
route payload; the new generated interface is the IR-modeled source-op
boundary.

## Module Owner

Generated MLIR/TableGen EmitC lowerable op interface plus the RVV i32
add/sub/mul typed family-op consumer path.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start state is clean on `main`, with HEAD
  `a2c853f feat(emitc): add common lowerable interface`.
- No `.trellis/.current-task` existed at task start; this is a new task, not a
  reopened archive.
- The completed archive
  `.trellis/tasks/archive/2026-05/05-11-tcrv-emitc-lowerable-interface-rvv-i32-add/`
  added a hand-written C++ `TCRVEmitCLowerableInterface` route builder and
  adapted the direct RVV i32-add export path.
- Live inspection found no generated `*Interface*.td` under
  `include/TianChenRV` before this task.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common lowering to
  consume an EmitC lowerable interface instead of descriptor-driven
  computation.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` already has bounded
  `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul` typed
  dataflow ops with coherent ODS shapes for this slice.
- `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp` currently recognizes
  those ops with typed `dyn_cast`s and builds dataflow plan steps.
- `lib/Target/RVV/RVVMicrokernel.cpp` currently constructs the common
  `TCRVEmitCLowerableRoute` from the verified dataflow plan through a
  hand-written C++ adapter.

## Naming Decision

The generated op interface will be named `TCRVEmitCLowerableOpInterface`.
Reason: the previous task intentionally introduced the hand-written C++
`TCRVEmitCLowerableInterface` class as the route-builder adapter API. Reusing
the exact same class name for generated ODS output would create two unrelated
interface meanings. `TCRVEmitCLowerableOpInterface` makes the generated
operation-interface boundary explicit while preserving the existing
`TCRVEmitCLowerableRoute` payload and C++ adapter naming.

## Requirements

- Add a TableGen op-interface definition for `TCRVEmitCLowerableOpInterface`
  and wire generated declarations/definitions through CMake.
- Include the generated interface in the RVV dialect headers and definitions so
  RVV ODS ops can attach it.
- Attach the generated interface to bounded RVV i32 arithmetic dataflow ops:
  `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul`.
- The interface must expose bounded source-op provenance sufficient for the
  common route construction to prove it is consuming an IR-modeled EmitC
  lowerable op boundary.
- The RVV target/body-verifier/export path must query the generated interface
  on the typed arithmetic op before constructing the common
  `TCRVEmitCLowerableRoute`.
- The existing common C++ route object remains the payload carrying headers, C
  type mappings, ABI mappings, source-op provenance, and
  `emitc.call_opaque` construction data.
- Descriptor metadata remains selected-config, ABI identity, legacy id, and
  mismatch cross-check data only. It must not define arithmetic semantics.
- Descriptor/body mismatch failures must continue to fail closed before C
  source export.
- Common code must stay extension-family generic; RVV intrinsic spelling,
  vector suffixes, and family rules remain RVV target/plugin-owned.
- `tcrv.exec` must remain compute-free.

## Acceptance Criteria

- [ ] A generated TableGen op-interface definition exists and is built through
      CMake with generated declaration and definition targets.
- [ ] `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul` carry
      `TCRVEmitCLowerableOpInterface`.
- [ ] The RVV body verifier or export path queries the generated interface on
      the typed arithmetic op and propagates that provenance into route
      construction.
- [ ] RVV direct C export comments or focused tests prove the route came from
      the generated op-interface boundary, not only from descriptor metadata or
      the hand-written route helper.
- [ ] Existing descriptor/body mismatch negative behavior remains fail-closed
      before source output.
- [ ] Focused C++ route tests assert generated-interface-backed behavior.
- [ ] Focused lit/FileCheck coverage proves add/sub/mul keep their
      interface-backed route and intrinsic mappings.
- [ ] No RVV runtime, correctness, or performance claim is made unless a fresh
      `ssh rvv` run is performed through the changed path.

## Non-Goals

- No new dtype, LMUL, arithmetic family, scalar fallback, IME, TensorExt,
  Offload, vendor target, MLIR vector, LLVM scalable-vector, inline assembly,
  or backend patch route.
- No descriptor-to-C computation expansion.
- No Python compiler internals.
- No broad smoke matrix or report-only/prompt-only closeout.
- No change that turns `tcrv.exec` into a compute dialect.

## Minimal Validation Plan

- Configure/build touched TableGen and C++ targets, including RVV dialect,
  `TianChenRVConversionEmitC`, `TianChenRVRVVTarget`, `tcrv-translate`, and
  `tianchenrv-emitc-lowerable-interface-test`.
- Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`.
- Run focused lit/FileCheck for `rvv-microkernel-pipeline`,
  `rvv-microkernel-i32-add-descriptor-body-mismatch-fails`, and the affected
  i32 sub/mul family route tests.
- Run `git diff --check`.
- Run Trellis validation for this task.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if focused checks pass and the build tree is usable.

## Completion Boundary

Finish and archive only if the generated op-interface exists, the bounded RVV
i32 add/sub/mul arithmetic ops implement it, and the RVV exporter/body-verifier
consumes that generated interface while building the common EmitC lowerable
route. If the generated interface exists but route construction still does not
query it, leave the task open and record the continuation point as
route-consumer migration.
