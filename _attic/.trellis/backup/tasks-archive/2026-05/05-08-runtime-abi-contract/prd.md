# centralize i32 vadd runtime abi contract

## Goal

Make the bounded i32-vadd runtime ABI contract a single compiler-owned C++
support/target boundary consumed by the current RVV, scalar, dispatch, and
target-artifact export paths while preserving the proven RVV+scalar selected
bundle behavior.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting HEAD is `1a62a9d feat: validate selected artifact bundle front door`.
- The starting worktree was clean.
- Previous evidence was bounded to the RVV+scalar i32-vadd target-artifact
  bundle external ABI path on real `ssh rvv`; this task does not broaden that
  claim.
- The current repeated ABI ownership surfaces are the callable parameter list,
  buffer mem-window specs, runtime element-count spec, dispatch availability
  guard spec, ABI identity strings, and route expectation strings.

## Requirements

- Add the smallest useful C++ support-layer runtime ABI contract object for the
  current bounded i32-vadd callable ABI.
- Centralize ordered callable parameters, mem-window specs, runtime count
  param spec, optional dispatch guard param spec, and stable ABI identity
  strings needed by existing consumers.
- Preserve existing wrapper APIs where they avoid churn; wrappers may delegate
  to the new contract.
- Make active compiler/exporter consumers use the contract instead of duplicate
  literals or independently assembled lists.
- Keep source, header, object, descriptor, bundle, evidence, capability,
  selected-variant, runtime SSA/control, and descriptor-local metadata layers
  separate.
- Do not add new generic compute operations, generic vector lowering, Python
  compiler logic, runtime integration, performance machinery, or offload routing
  through this ABI contract.

## Acceptance Criteria

- [ ] Runtime ABI contract lives in C++ support code.
- [ ] Existing wrapper APIs remain available.
- [ ] At least two active compiler/exporter consumers use the contract.
- [ ] Focused tests prove callable parameter order/roles and active exporter
      mismatch rejection still flow through real compiler/export behavior.
- [ ] Existing lit/FileCheck RVV+scalar dispatch source/header/object/bundle
      export behavior is preserved.
- [ ] `git diff --check` passes.
- [ ] CMake configure succeeds or an existing valid configured build is reused
      with explanation.
- [ ] `cmake --build build --target check-tianchenrv -j2` passes.
- [ ] Local bundle dry-run passes after source changes.
- [ ] Task is archived before the final commit and the repo is left clean.

## Out Of Scope

- No performance, throughput, latency, speedup, or broad correctness claims.
- No new high-level tensor/tile IR or generic compute op in `tcrv.exec`.
- No offload runtime descriptor handoff changes.
- No Python implementation of runtime ABI contracts or compiler decisions.
- No broad target-artifact exporter redesign.

## Technical Notes

- Relevant specs: `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Key code surfaces: `include/TianChenRV/Support/RuntimeABI*.h`,
  `lib/Support/RuntimeABI*.cpp`, `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`, RVV/scalar plugins, and
  `lib/Target/TargetArtifactExport.cpp`.
