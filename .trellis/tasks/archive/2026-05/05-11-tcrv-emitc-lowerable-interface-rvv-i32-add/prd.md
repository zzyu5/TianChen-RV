# tcrv-emitc-lowerable-interface-rvv-i32-add

## Goal

Create the first common TCRV EmitC lowerable boundary and route the default
direct RVV i32-add microkernel export through it. The structural route must
move toward:

```text
tcrv_rvv typed family ops -> common EmitC lowerable route -> RVV intrinsic C/C++
```

The descriptor remains a bounded selected-config and cross-check surface only.
It must not be the computation source for the generated arithmetic call.

## Module Owner

Common extension-family EmitC lowering interface plus the RVV i32-add first
consumer.

## What I Already Know

- The repository is at `/home/kingdom/phdworks/TianchenRV`, HEAD `b67b13e`,
  with a clean worktree at task start.
- There is no active `.trellis/.current-task`; this is a new Trellis task, not
  a reopened evidence-only archive.
- The latest archived evidence task intentionally changed Python runner/test
  evidence, while this task must be a C++/MLIR structural module.
- `.trellis/spec/lowering-runtime/emitc-route.md` names
  `TCRVEmitCLowerableInterface` as the common route consumed by shared lowering.
- The live tree currently has typed RVV family ops and local RVV target export
  route objects, but no common `TCRVEmitCLowerableInterface` source boundary.
- The existing direct RVV microkernel exporter validates typed
  `tcrv_rvv.i32_vadd_microkernel` bodies before source emission. This task must
  make that validation feed a common route object rather than keeping the route
  shape RVV-local.

## Requirements

- Add a common C++ EmitC lowerable interface/adapter layer under
  `include/TianChenRV/Conversion/EmitC/` and `lib/Conversion/EmitC/`, wired into
  CMake.
- Name the common C++ interface `TCRVEmitCLowerableInterface` and make it
  extension-family generic.
- The common route object must describe:
  - header requirements;
  - C type mapping;
  - ABI operand/result mapping;
  - intrinsic or runtime call names;
  - source-op provenance;
  - `emitc.call_opaque` construction inputs.
- Adapt the direct RVV i32-add path as the first consumer. The generated
  arithmetic call must come from the verified typed `tcrv_rvv.i32_add` source
  operation through the common route object.
- Preserve descriptor metadata only as selected config, ABI identity, legacy id,
  and mismatch cross-check. Descriptor/body mismatch must fail before C export.
- Keep `tcrv.exec` as capability/variant/dispatch/fallback/ABI envelope only;
  add no compute semantics to core.
- Do not add RVV-specific semantic branches to common orchestration.
- Keep Python tooling-only; do not implement compiler core, dialects, lowering,
  or emission in Python.

## ODS Interface Decision

The preferred long-term shape is an ODS MLIR op interface named
`TCRVEmitCLowerableInterface`. This round will implement the narrow common C++
interface first. Reason: the current verified RVV export path already consumes
generated typed ops through C++ body verification and target emission; adding a
generated ODS interface in the same round would require new TableGen interface
generation targets plus touching the RVV op declarations broadly. The common
C++ route creates the reusable boundary now while leaving ODS attachment as a
bounded follow-up.

## Acceptance Criteria

- [x] A common `TCRVEmitCLowerableInterface` C++ boundary exists in source and
      is built by CMake.
- [x] RVV i32-add builds a common EmitC lowerable route from verified typed RVV
      family ops.
- [x] The direct i32-add C artifact obtains its arithmetic call from the typed
      `tcrv_rvv.i32_add` route mapping, not from `lowering_descriptor` alone.
- [x] The route records source-op provenance for `tcrv_rvv.i32_load`,
      `tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store`.
- [x] Descriptor/body arithmetic or config mismatch fails before misleading C
      export.
- [x] Focused C++ tests cover the common route/interface shape and RVV first
      consumer boundary.
- [x] Focused lit/FileCheck coverage proves direct i32-add export prints the
      common route provenance and RVV intrinsic call boundary.
- [x] No runtime correctness, hardware execution, or performance claim is made
      without fresh `ssh rvv` evidence.

## Non-Goals

- No new RVV family matrix, dtype expansion, LMUL expansion, IME, TensorExt,
  Offload, scalar fallback, or GCC-default work.
- No MLIR vector, LLVM scalable-vector, inline-asm, or descriptor-to-C
  computation extension.
- No Python compiler-core implementation.
- No evidence-only, helper-only, prompt-only, or broad smoke-test round as the
  main deliverable.
- No changes that make `tcrv.exec` a compute IR.

## Minimal Validation Plan

- Build focused targets touched by the change:
  `TianChenRVConversionEmitC`, `TianChenRVRVVTarget`, `tcrv-translate`, and new
  or updated focused C++ tests.
- Run focused C++ tests for the common EmitC lowerable boundary and target
  artifact/RVV consumer behavior.
- Run focused lit/FileCheck for direct RVV i32-add route provenance and the
  descriptor/body mismatch failure.
- Run `git diff --check`.
- Run Trellis validation for this task.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if focused checks pass and the build tree is usable.

## Completion Boundary

Finish and archive only if the default direct RVV i32-add export path consumes
typed RVV family ops through the common EmitC lowerable boundary before
generated RVV intrinsic C output. If the source artifact still derives
computation semantics from descriptor metadata rather than typed RVV family ops
through the common route object, keep the task open and record the exact
continuation point.
