# RVV Vector-Source Arithmetic-Family Selected-Boundary Materializer

## Goal

Generalize the production RVV vector-source front-door materializer from the
current add-only matcher into an op-kind-aware bounded i32m1 arithmetic
materializer for the already-supported add/sub/mul route family.

The source path must remain the real production path:

```text
MLIR vector/scf source arithmetic
  -> RVV selected boundary
  -> explicit tcrv_rvv add/sub/mul ops
  -> materialized EmitC object/header/bundle
  -> target artifact export
```

This is a compiler boundary task, not another evidence-only runner.

## Current Repository Facts

- Session start repo root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean.
- Starting HEAD was `b99f31a test(rvv): prove generated bundle ABI on ssh rvv`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief as
  `05-17-rvv-vector-source-arithmetic-family-materializer`.
- The production front door in
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` still uses
  `BoundedI32AddSourcePattern`, `arith.addi`, `createI32Add`, and
  `_rvv_i32_add` variant naming.
- Downstream RVV construction, EmitC route provider, extension plugin emission
  planning, and target artifact bundle export already carry the bounded
  i32m1 add/sub/mul route family.
- The previous task proved the generated object/header bundle ABI for the
  existing add path on real `ssh rvv`; that proof remains evidence for
  unchanged bundle mechanics unless this task changes the ABI runner or claims
  new runtime correctness for sub/mul.

## Requirements

- Recognize exactly one source-only `func.func` with:
  - four runtime ABI operands in order: `memref<?xi32>` lhs, rhs, out and
    runtime `index` n;
  - no results;
  - one block;
  - one `scf.for` from constant index 0 to `%n` by constant index 4;
  - no loop-carried values;
  - exactly two `vector.load` ops, one supported integer vector arithmetic op,
    and one `vector.store` in the loop body.
- Supported source arithmetic ops are exactly:
  - `arith.addi` -> `tcrv_rvv.i32_add`;
  - `arith.subi` -> `tcrv_rvv.i32_sub`;
  - `arith.muli` -> `tcrv_rvv.i32_mul`.
- Carry the chosen op kind into:
  - selected RVV variant symbol naming;
  - explicit typed `tcrv_rvv` compute op construction;
  - emission-plan metadata `rvv_arithmetic_op`;
  - dynamic runtime ABI identity;
  - generated header/object/bundle component metadata through the existing
    materialized EmitC route family.
- Fail closed for:
  - stale `tcrv_rvv.lowering_seed`;
  - pre-existing `tcrv.exec` or `tcrv_rvv` input;
  - wrong ABI shape/order/type;
  - wrong vector type/shape;
  - unsupported arithmetic;
  - descriptor/direct-C/source-export residue.
- Keep Python only as evidence tooling. Do not move compiler semantics into
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Acceptance Criteria

- [x] Positive source-front-door lit coverage proves add, sub, and mul source
      MLIR each materialize to the correct selected RVV variant and typed
      `tcrv_rvv.i32_*` op.
- [x] Positive pipeline lit coverage proves add, sub, and mul source MLIR each
      produce supported emission-plan metadata with the correct
      `rvv_arithmetic_op`, per-op EmitC route id, and per-op runtime ABI name.
- [x] Target object/header/bundle lit coverage proves add, sub, and mul export
      through the existing materialized EmitC object/header/bundle route.
- [x] Negative source-front-door coverage fails closed for unsupported
      arithmetic, stale seed metadata, pre-existing `tcrv.exec`/`tcrv_rvv`
      residue, wrong ABI shape, wrong vector type, and descriptor/direct-C/
      source-export residue checks in generated headers.
- [x] Focused build for `tcrv-opt`, `tcrv-translate`, and relevant RVV/target
      tests passes.
- [x] Focused lit for the RVV source front door and RVV target artifact exports
      passes.
- [x] `check-tianchenrv` passes if practical; otherwise the exact blocker is
      recorded.
- [x] Targeted scans over touched RVV plugin/target/translate/tests show no
      descriptor route authority, direct C semantic exporter, source-export
      route, or restored legacy route ID.
- [x] `git diff --check` passes.

## Definition Of Done

- Trellis task state and journal truthfully record what changed, what checks
  ran, whether runtime evidence was refreshed, and any remaining gap.
- If complete, archive the task and create one coherent commit.
- If incomplete, leave the task open with the exact next continuation point.

## Out Of Scope

- Broader dtype/SEW/LMUL support.
- Masks, generic vector lowering, or scalar fallback compute.
- New EmitC route family or new artifact format.
- Descriptor adapter, direct C semantic exporter, source-export route, legacy
  route alias, or independent RVV backend.
- Python compiler-core behavior.
- New standalone runtime evidence campaign unless the ABI runner is touched or
  sub/mul runtime correctness is explicitly refreshed.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-rvv-generated-bundle-abi-execution-proof/prd.md`.
- Code/test surfaces inspected:
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir`,
  `test/Target/RVV/vector-source-target-artifact-object.mlir`,
  `test/Target/RVV/vector-source-target-artifact-header.mlir`,
  and `scripts/rvv_generated_bundle_abi_e2e.py`.

## Implementation Summary

- Replaced the add-only source matcher/materializer with
  `BoundedI32ArithmeticSourcePattern`, carrying an explicit
  `RVVI32M1ArithmeticOp` selected from `arith.addi`, `arith.subi`, or
  `arith.muli`.
- The selected source op kind now controls:
  - the generated selected variant suffix `_rvv_i32_add/sub/mul`;
  - the explicit typed RVV compute op `tcrv_rvv.i32_add/sub/mul`;
  - downstream existing route metadata through the already-wired RVV
    construction and EmitC route provider.
- Updated the RVV source front-door pass description from add-only to the
  bounded add/sub/mul arithmetic family.
- Added sub and mul source-front-door fixtures, covering selected boundary and
  emission-plan metadata through the default source-artifact front-door
  pipeline.
- Expanded RVV target artifact object/header/bundle lit coverage so add, sub,
  and mul each export through the existing materialized EmitC route family with
  the correct selected variant, runtime ABI identity, symbol, and route
  metadata.
- Kept `scripts/rvv_generated_bundle_abi_e2e.py` unchanged. No new runtime
  correctness claim was made for sub/mul; the prior `ssh rvv` evidence remains
  the ABI-consumption proof for unchanged bundle mechanics.

## Completion Evidence

- Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit from `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-negative.mlir Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-disabled.mlir Target/RVV/vector-source-target-artifact-object.mlir Target/RVV/vector-source-target-artifact-header.mlir`
  passed 8/8.
- Full check:
  `cmake --build build --target check-tianchenrv -j2` passed 115/115 lit
  tests.
- `git diff --check` passed.
- Targeted scan over touched RVV plugin/tests and the unchanged ABI runner
  found descriptor/direct-C/source-export strings only in stale-seed
  fail-closed diagnostics, generated-header `CHECK-NOT` assertions, or the
  existing evidence runner forbidden-token validation.

## Status

Complete. No compatibility layer was added. No descriptor-driven computation,
direct C semantic exporter, source-export route, scalar fallback compute, new
artifact route, or legacy route alias was introduced.
