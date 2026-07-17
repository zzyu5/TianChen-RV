# RVV executable construction protocol realization

## Goal

Make the existing executable RVV i32m1 add/sub/mul path the plugin-local
Extension-Family Construction Protocol exemplar. The RVV plugin must expose a
C++ construction manifest and typed role/interface realization for the current
family declaration, role graph, common interface realization, EmitC route
mapping, runtime ABI fields, and evidence profile, then consume that surface
from active RVV validation/planning/provider code.

The bounded production path for this task is:

```text
RVV extension-family declaration
  -> typed role/common-interface realization
  -> validated tcrv_rvv.with_vl selected boundary
  -> RVV-owned EmitC route mapping for i32m1 add/sub/mul
  -> target artifact/evidence profile consumed by active RVV route code
```

## What I Already Know

- Current HEAD is `577b3d6`, which made the existing `tcrv_rvv.with_vl` op the
  explicit selected lowering boundary for bounded i32m1 add/sub/mul variants.
- The current real executable RVV path is implemented in RVV plugin/provider
  code: `RVVExtensionPlugin`, `RVVEmitCRouteProvider`, `RVVConfigContract`, and
  RVV target support bundle registration.
- `ConstructionProtocol` already provides a shared C++ manifest verifier for
  family declarations, semantic role ordering, common interface realization,
  typed role/interface realization, role-op interface checks, route mapping,
  and evidence-profile requirements.
- Template, Toy, and TensorExtLite already use that shared construction model,
  but RVV does not yet have a plugin-local construction-protocol realization.
- The relevant specs require the manifest to be a machine-readable construction
  entry point, not a passive checklist or Markdown-only report.
- The existing valid RVV add/sub/mul selected variants must keep validating
  through explicit typed `tcrv_rvv` IR, `tcrv_rvv.with_vl`, and RVV-owned EmitC
  route builders.

## Design Boundaries

- Implement compiler behavior in C++/MLIR/CMake/tests, not Python compiler-core
  structures.
- Keep all RVV-specific construction, intrinsic/header spelling, typed-body
  route validation, and runtime ABI ownership inside RVV plugin/provider/target
  code.
- Do not add RVV semantic branches, RVV intrinsic names, or RVV header names to
  common/core orchestration.
- Do not add new RVV dtype/LMUL/SEW/op families, i32m2 execution, vector
  lowering, high-level tensor lowering, TensorExt/IME work, new artifact kinds,
  descriptor registries, direct C compute printers, or compatibility wrappers.
- Do not restore descriptor-driven computation, direct-C/source-export
  authority, deleted microkernel wrappers, metadata-only routes, or Python
  compiler-core logic.
- One coherent submodule is enough if it includes manifest validation plus at
  least one active consumption point in the add/sub/mul executable route path.

## Requirements

- Add an RVV plugin-local construction protocol surface, following the existing
  shared `construction::Manifest` and `construction::TypedRoleGraphRealization`
  model.
- The RVV construction manifest must declare:
  - protocol version and RVV archetype;
  - role graph for `runtime_abi_value`, `setvl`, `with_vl`, load, arithmetic,
    and store expectations;
  - family declaration for the RVV extension family;
  - common interface realization and typed role/interface realization;
  - EmitC route mapping for i32m1 add/sub/mul route ids and runtime ABI fields;
  - evidence-profile requirements for parse/verify, capability, interface,
    selected boundary, EmitC route mapping, target artifact, and RVV runtime
    evidence boundary.
- Manifest and typed-role verification must fail closed for missing family
  fields, stale route/op mapping, missing typed roles, wrong common interface
  realization, or missing evidence-profile requirements.
- Active RVV validation/planning/provider code must consume the construction
  surface before accepting the selected boundary, emission readiness/plan, or
  EmitC route construction for add/sub/mul.
- Route IDs and artifact/runtime ABI IDs used by RVV emission planning must be
  checked against construction mapping instead of remaining unrelated string
  constants.
- Existing valid add/sub/mul selected variants must still validate through
  `with_vl` and build the same supported EmitC/object/header route.
- Negative coverage must prove stale or incomplete construction data fails
  before route payload construction or target artifact export.

## Acceptance Criteria

- [ ] `verifyRVVConstructionManifest` and
      `verifyRVVTypedRoleGraphRealization` validate the real RVV construction
      declaration and fail closed on stale/missing manifest fields.
- [ ] The RVV plugin/provider consumes construction verification in active
      validation/planning/route code, not only in tests or documentation.
- [ ] RVV add/sub/mul route IDs, emission kind, lowering boundary, runtime ABI
      kind/name/glue role, and ordered ABI parameters are checked against the
      RVV construction mapping.
- [ ] Existing selected add/sub/mul `with_vl` boundary tests and selected
      artifact/export lit tests remain valid.
- [ ] Focused C++ tests cover RVV construction manifest validation and active
      consumption by RVV plugin/provider code.
- [ ] Focused lit/FileCheck checks pass for selected boundary, EmitC
      materialization, and add/sub/mul selected artifact export.
- [ ] A changed-surface scan shows descriptor/direct-C/source-export legacy
      terms were not restored and common/core code has no RVV intrinsic/header
      names or RVV semantic branch.
- [ ] If generated C/header/object payload changes, rerun `ssh rvv` for
      affected arithmetic routes. If not, record an unchanged-payload rationale.

## Definition Of Done

- Trellis task context is current and truthful.
- PRD, implementation, tests, and verification remain scoped to the RVV
  construction-protocol realization for existing i32m1 add/sub/mul.
- Focused checks are run and failures are self-repaired where practical.
- The task is finished/archived when complete.
- One coherent commit is created when complete.

## Out Of Scope

- New RVV families, i32m2 execution, MLIR vector lowering, high-level tensor
  lowering, TensorExt/IME implementation, new target artifact kinds,
  performance matrices, descriptor or binary-family registries, direct C
  compute printers, Python compiler-core logic, GCC-default routes,
  compatibility wrappers, state-machine ledgers, and extension-specific
  semantic branches in common/core orchestration.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Existing construction examples:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`,
  `lib/Plugin/Template/TemplateConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/Toy/ToyConstructionProtocol.h`,
  `lib/Plugin/Toy/ToyConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h`,
  `lib/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.cpp`.
- RVV active surfaces:
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
