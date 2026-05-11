# RVV descriptor route to TCRV RVV family ops EmitC route

## Goal

Refactor the finite RVV binary path away from descriptor-driven computation
toward the latest unified TCRV architecture: `tcrv.exec` remains the
capability/variant/dispatch/ABI envelope, RVV computation is represented by
TCRV RVV extension family ops, and executable C/C++ is produced through a
common EmitC route that maps RVV family ops to RVV intrinsics.

## Why Now

The long-term specs now define TianChen-RV as a unified TCRV RISC-V MLIR:
RVV, IME, TensorExt, Offload, and future vendor/custom targets are extension
families inside one system. The current RVV finite binary implementation still
has descriptor-era surfaces such as `lowering_descriptor`, microkernel
descriptors, descriptor-driven selected boundaries, descriptor-to-C exporters,
and descriptor-based tests. Those should stop being the main path.

## Module Owner

RVV descriptor route -> TCRV RVV family ops -> EmitC route.

## Required Scope For First Implementation Round

Start with one finite but complete RVV slice, preferably i32 add unless repo
evidence shows another finite binary member is the safer first cut.

The round should:

- locate where descriptor metadata currently owns RVV binary compute semantics;
- keep `tcrv.exec` as envelope only;
- materialize or consume RVV extension family ops for the chosen slice;
- route those RVV family ops through EmitC;
- generate RVV intrinsic C/C++ from the EmitC route;
- stop using descriptor as the computation source for that slice;
- avoid MLIR vector / LLVM scalable vector work;
- avoid new high-level lowering work;
- avoid expanding descriptor coverage.

If descriptor code cannot be deleted safely in one round, keep it as bounded
legacy support only and move the default/current slice toward family ops and
EmitC.

## Architecture Requirements

- RVV is a TCRV extension family, not an independent backend dialect.
- Common route infrastructure should be reusable by IME / TensorExt later.
- Common/core code must work through TCRV interfaces or plugin hooks, not
  RVV-specific semantic branches.
- The current main route is extension family ops -> EmitC -> intrinsic/runtime
  C/C++ -> clang/LLVM by default.
- GCC may be compatible, not default.
- Descriptor-driven computation is invalid as long-term architecture.
- Runtime/correctness/performance claims require `ssh rvv`; compile-only or
  generated-source checks must not be reported as hardware execution evidence.

## Suggested Read First

- `.trellis/spec/index.md`
- `.trellis/spec/architecture/unified-riscv-mlir.md`
- `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Plugin/RVV/`
- `lib/Target/RVV/`
- `tools/tcrv-opt/`
- `tools/tcrv-translate/`
- `test/Dialect/RVV/`
- `test/Target/RVVMicrokernel/`

## Acceptance Criteria

- [ ] A concrete finite RVV slice uses RVV family ops as the compiler-visible
      compute semantics rather than descriptor metadata.
- [ ] The slice has an EmitC lowering path from RVV family ops to RVV intrinsic
      C/C++.
- [ ] Tests prove RVV family ops exist and are consumed by the EmitC route.
- [ ] Tests prove generated C/C++ contains RVV intrinsic usage when that route
      is claimed.
- [ ] Tests prove the chosen slice no longer depends on descriptor metadata as
      its computation semantics source.
- [ ] Core pass changes, if any, are generic interface/route infrastructure,
      not RVV semantic branches.
- [ ] Descriptor-dependent paths are not expanded; any remaining descriptor
      logic is explicitly bounded legacy.

## Non-Goals

- No new prompt/spec-only work as the main result.
- No MLIR vector / LLVM scalable vector route.
- No new high-level linalg/stablehlo/tosa lowering.
- No new RVV arithmetic family beyond what is needed for the chosen slice.
- No performance experiment unless the implementation makes a runtime claim
  that requires `ssh rvv`.
- No Python implementation of compiler core, dialects, passes, capability,
  plugin registry, lowering, or emission.

## Minimal Validation

- Focused lit/FileCheck tests for RVV family op parse/verify or materialization.
- Focused lowering/export test showing RVV family op -> EmitC route.
- Generated C/C++ content check for RVV intrinsic call and required header.
- Build/test target as narrow as the changed module allows.
- `ssh rvv` only if the round claims runtime/correctness/performance evidence.

## Continuation Rule

If the full descriptor-to-family-ops migration cannot finish in one round,
leave this task open with a clear continuation point. Hermes should continue
this owner rather than switching to another micro-task.
