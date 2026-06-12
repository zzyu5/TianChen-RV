# RVV dataflow-driven source emission

## Goal

Make the bounded RVV i32-vadd C source exporter derive its emitted intrinsic
loop from the validated `tcrv_rvv` load/add/store dataflow body, rather than
treating that body only as a pre-export validation guard before printing a
fixed hard-coded sequence.

The compiler path that becomes more real is:

```text
selected rvv-plugin variant
  -> tcrv_rvv.i32_vadd_microkernel
  -> tcrv_rvv.setvl / tcrv_rvv.with_vl
  -> tcrv_rvv.i32_load / i32_load / i32_add / i32_store
  -> generated RVV intrinsic C statements
```

## What I Already Know

* Repo root is `/home/kingdom/phdworks/TianchenRV`.
* Worktree was clean before this task was created.
* Latest commit is `77eac9a feat: structure rvv i32 vadd dataflow body`.
* Latest Hermes audit/review input is under
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0093-20260509T001003Z/`.
* The previous completed round added explicit RVV dataflow ops:
  `tcrv_rvv.i32_load`, `tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store`.
* `lib/Target/RVV/RVVMicrokernel.cpp` validates that dataflow body, but
  `printMicrokernelFunction` still emits the fixed i32-vadd intrinsic body
  from ABI role bindings alone.

## Requirements

* Preserve the primary implementation stack: C++ / MLIR / TableGen / CMake /
  lit/FileCheck.
* Keep `tcrv.exec` compute-free; no new core dialect compute constructs.
* Keep finite i32-vadd semantics plugin-local to `tcrv_rvv` and target/export
  code.
* Introduce a small target/export-local representation of the validated RVV
  i32-vadd dataflow emission plan.
* Populate that plan from the real `tcrv_rvv.i32_load`, `tcrv_rvv.i32_add`,
  and `tcrv_rvv.i32_store` operations during microkernel validation.
* Emit the RVV C intrinsic loop by consuming the extracted dataflow plan and
  the IR-backed runtime ABI parameter bindings.
* Preserve parameter layering:
  runtime `n`/AVL comes from the microkernel region argument and callable ABI;
  `vl` is a target-owned local generated from `setvl`; descriptor-local
  `element_count` remains fixture metadata only; selected march/mabi remain
  compile capability metadata.
* Add only focused tests or FileCheck updates proving that output and
  diagnostics reflect the dataflow-driven emission path.

## Acceptance Criteria

* [ ] The RVV source exporter stores normalized load/add/store emission steps
      derived from the validated `tcrv_rvv` body.
* [ ] `printMicrokernelFunction` emits RVV load/add/store statements through
      those steps, not through a standalone fixed statement sequence.
* [ ] Existing malformed dataflow tests still fail before source output.
* [ ] The default RVV microkernel source artifact still has no hidden `main`.
* [ ] `cmake --build build --target check-tianchenrv -j2` passes, or the exact
      missing local dependency is reported.

## Definition Of Done

* Tests added or updated only where they validate changed compiler behavior.
* Lint/whitespace check passes.
* Relevant specs are updated only if the durable contract changes.
* The Trellis task is validated and archived or left in an accurate state.
* Repo is clean, with one coherent commit if the round completes.

## Technical Approach

Add an RVV target/export-local dataflow emission structure in
`lib/Target/RVV/RVVMicrokernel.cpp`. Validation already has direct access to
the concrete `I32LoadOp`, `I32AddOp`, and `I32StoreOp`; extend that path to
extract the semantic ABI roles for lhs load, rhs load, output store, and the
finite operation kind/order. Store the result in `RVVMicrokernelRecord`, then
have source generation bind role names to the existing IR-backed ABI parameter
plan and print statements by walking the extracted steps.

This keeps the scope intentionally bounded: it does not add a generic RVV
lowering IR, a generic vector memory model, a new high-level op, a runtime
probe, object/link evidence, or performance measurement.

## Out Of Scope

* Generic RVV lowering to LLVM/RISC-V.
* Arbitrary RVV memory/arithmetic ops beyond this finite i32-vadd slice.
* Python compiler internals.
* New smoke dashboards, broad negative matrices, or standalone evidence
  packaging.
* RVV runtime correctness/performance claims without real `ssh rvv` evidence.

## Technical Notes

Relevant specs inspected:

* `.trellis/spec/index.md`
* `.trellis/spec/capability-model/index.md`
* `.trellis/spec/capability-model/capability-contract.md`
* `.trellis/spec/core-dialect/index.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/plugin-protocol/locality-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant code inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/Builtin/RVVScalarDispatch.cpp`
