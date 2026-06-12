# TCRV EmitC route to MLIR EmitC materialization for RVV i32 family

## Goal

Add a common C++/MLIR materialization boundary that consumes a verified
`TCRVEmitCLowerableRoute` and builds real MLIR EmitC dialect IR for the
bounded RVV i32 add/sub/mul target/export path. The existing C source emitter
may remain as bounded legacy output for this round, but the RVV path must
invoke and validate real route-to-EmitC MLIR materialization before source
export.

## Module Owner

Common `TCRVEmitCLowerableRoute` to MLIR EmitC materialization in
`include/TianChenRV/Conversion/EmitC/` and `lib/Conversion/EmitC/`, consumed by
the RVV i32 add/sub/mul microkernel source/header/object export route before
bounded legacy C source generation.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start state is clean on `main`, with HEAD
  `c6ba034 feat(emitc): add generated lowerable op interface`.
- No `.trellis/.current-task` existed at task start; this is a new task, not a
  reopened archive.
- The archived task
  `.trellis/tasks/archive/2026-05/05-11-tcrv-emitc-ods-lowerable-interface-rvv-i32-family/prd.md`
  closed the generated `TCRVEmitCLowerableOpInterface` boundary for
  `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul`.
- Current `TCRVEmitCLowerableRoute` verification checks bounded text, headers,
  ABI mappings, type mappings, and `TCRVEmitCCallOpaqueStep` payloads, but it
  does not yet build MLIR EmitC ops.
- `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp` already queries
  `TCRVEmitCLowerableOpInterface` on bounded i32 arithmetic dataflow ops and
  carries source-op/interface provenance into the dataflow plan.
- `lib/Target/RVV/RVVMicrokernel.cpp` already builds an interface-backed
  `TCRVEmitCLowerableRoute` and emits C source through a bounded legacy helper
  that prints route provenance comments.
- Installed MLIR 20 provides the EmitC dialect API including
  `emitc.include`, `emitc.func`, `emitc.call_opaque`, `emitc.return`,
  `emitc.literal`, and EmitC opaque/size/pointer types.
- The architecture specs require extension family ops -> EmitC -> intrinsic /
  runtime C/C++ as the current main route, with descriptor-driven computation
  treated only as bounded implementation debt.

## Requirements

- Add a public common materializer API in `Conversion/EmitC` that takes a
  verified `TCRVEmitCLowerableRoute`, a function name, and materialization
  options, and returns/writes a parseable MLIR module containing real EmitC
  dialect operations.
- Register/use `mlir::emitc::EmitCDialect` in the materializer and link
  `TianChenRVConversionEmitC` against `MLIREmitCDialect`.
- Materialize route headers as `emitc.include` operations.
- Materialize one `emitc.func` boundary whose arguments are derived from
  route ABI mappings and whose function argument attributes preserve route ABI
  value names.
- Materialize each `TCRVEmitCCallOpaqueStep` as an `emitc.call_opaque` with
  typed operands/results when the route step has a result.
- Preserve route/source provenance in bounded MLIR attributes or locations
  because the installed EmitC ops do not expose a native provenance field for
  source operation/interface provenance.
- Fail closed before target C source export for malformed route materializer
  input, including empty callee, missing result for compute steps, unknown
  operand value names, duplicate result names, unsafe provenance text, and
  ABI/value mapping mismatches.
- Wire the RVV i32 add/sub/mul target/export path so it materializes and
  verifies the route-to-EmitC MLIR module before emitting the existing bounded
  C source.
- Keep RVV intrinsic spelling, vector suffix mapping, and family dispatch in
  RVV target-owned code; common materialization must remain family-generic.
- Keep descriptor data limited to selected config, ABI identity, legacy id, and
  mismatch cross-checks. Descriptor data must not choose computation semantics.
- Keep `tcrv.exec` compute-free.

## Acceptance Criteria

- [ ] Common C++ materializer API and implementation exist under
      `Conversion/EmitC`.
- [ ] `TianChenRVConversionEmitC` links the MLIR EmitC dialect and builds.
- [ ] Focused C++ tests prove valid add/sub/mul-style route payloads produce
      walkable MLIR EmitC IR containing includes, one function, ABI-backed
      arguments, and `emitc.call_opaque` steps.
- [ ] Focused C++ negative tests prove malformed route materialization fails
      closed for missing compute result, unknown operand value name, duplicate
      result name, unsafe provenance text, missing callee, and ABI/value mapping
      mismatch.
- [ ] RVV target/export tests prove the RVV path consumes the real materializer
      before source output and still preserves generated op-interface
      provenance.
- [ ] Lit/FileCheck coverage for the frontend-selected i32 vmul path proves the
      bounded route still emits the RVV intrinsic C evidence and now records the
      EmitC materialization boundary.
- [ ] No descriptor-to-C computation expansion is added.
- [ ] No new dtype, LMUL, arithmetic family, scalar fallback, IME, TensorExt,
      Offload, vendor target, LLVM scalable-vector route, MLIR vector route,
      inline assembly, or backend patch route is added.
- [ ] No RVV runtime/correctness/performance claim is made unless fresh
      `ssh rvv` evidence is collected through the changed path.

## Non-Goals

- Do not replace the final bounded C source emitter with MLIR's C/C++ emitter
  in this task unless the focused implementation proves small and safe.
- Do not add new arithmetic families or broader RVV lowering.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Do not route generic core orchestration through RVV-family semantic branches.
- Do not broaden validation into a smoke matrix beyond changed focused tests
  and the requested final `check-tianchenrv`.

## Minimal Validation Plan

- Build focused targets:
  `MLIRTCRVEmitCLowerableOpInterfaceIncGen`,
  `TianChenRVConversionEmitC`, `TianChenRVRVVDialect`,
  `TianChenRVRVVTarget`, `tcrv-translate`, and
  `tianchenrv-emitc-lowerable-interface-test`.
- Run
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`.
- Run focused lit tests changed by this task, including the RVV microkernel
  target/export coverage and `linalg-i32-vmul-to-rvv-artifact.mlir`.
- Run `git diff --check`.
- Run `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-tcrv-emitc-route-to-mlir-emitc-rvv-i32-family`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  after focused checks pass.
- Run `python3 -m py_compile` only if Python scripts are touched.
- Run fresh `ssh rvv` only if runtime artifacts/runners change or if making an
  RVV runtime/correctness/performance claim.

## Completion Boundary

Finish/archive only if the RVV i32 add/sub/mul export path consumes the common
route-to-EmitC MLIR materializer before C source output, focused C++ and lit
coverage pass, Trellis validation passes, and the final check target is run or
any inability to run it is recorded with exact evidence. If the materializer
compiles but RVV export consumption is not wired, leave the task open and
record the continuation point as: wire RVV target/export path to consume
route-to-EmitC materializer before C source emission.
