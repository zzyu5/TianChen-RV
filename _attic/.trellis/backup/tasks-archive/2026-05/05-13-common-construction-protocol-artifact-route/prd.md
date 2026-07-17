# Common construction-protocol artifact route

## Goal

Factor the now-proven construction-protocol artifact path into a shared C++
production route layer consumed by Template, Toy, and TensorExtLite. The task
should reduce copied manifest / role graph / interface / EmitC mapping /
evidence-profile / generated-output validation while keeping extension-specific
semantics plugin-local.

The intended slice is:

```text
extension-local family declaration and typed role identities
  -> shared construction-protocol manifest validation
  -> shared typed-role/interface validation
  -> shared generated-output route construction
  -> extension-local selected-boundary and EmitC mapping hooks
  -> target artifact export through the existing plugin-owned routes
```

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial state for this round: worktree clean, HEAD
  `a96e2a1 feat(tensorext-lite): instantiate construction protocol extension`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief and started as current task.
* Template, Toy, and TensorExtLite are now active construction-protocol
  consumers with typed ODS role ops and target metadata artifact routes.
* The prior Template/Toy/TensorExtLite PRDs all completed the construction
  protocol path and explicitly left the current bottleneck as repeated
  per-extension protocol and metadata-artifact implementation.
* The main duplicated source owners found before implementation are:
  * `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`
  * `include/TianChenRV/Plugin/Toy/ToyConstructionProtocol.h`
  * `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h`
  * `lib/Plugin/Template/TemplateConstructionProtocol.cpp`
  * `lib/Plugin/Toy/ToyConstructionProtocol.cpp`
  * `lib/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.cpp`
  * Template/Toy/TensorExtLite target metadata artifact files under
    `lib/Target/`.
* Template and Toy share the role graph
  `configure -> load -> compute -> store`; TensorExtLite uses
  `configure -> load_frag -> tile_mma -> store_frag`.

## Module Goal

Add one coherent shared construction-protocol route module that owns common
validation and generated output construction for the existing three consumers:

* generic construction manifest data model;
* generic semantic role and typed role realization validation;
* generic generated output route / generated step construction;
* generic role-op interface validation for the selected compute-like role;
* generic helpers for route metadata and generated output emission if bounded
  without turning target-specific route selection into core logic.

Template, Toy, and TensorExtLite must actively call the shared route from their
production construction-protocol implementation. A helper-only module that is
not consumed by at least two existing extensions is not sufficient.

## Boundaries

* Scope is common construction-protocol validation and generated artifact
  support for Template, Toy, and TensorExtLite only.
* Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* Python may only run Trellis/tooling validation.
* Extension-specific constants, family names, capabilities, ODS op identities,
  role names, route ids, intrinsic/runtime call spellings, target exporter
  registration, and selected-boundary materialization stay plugin-local.
* Shared code must remain target-neutral and must not add Template/Toy/
  TensorExtLite semantic branches to `tcrv.exec`, `lib/Transforms`, or common
  orchestration passes.
* This task does not create a new extension family, RVV runtime behavior,
  descriptor cleanup, hardware evidence, correctness evidence, or performance
  evidence.

## Requirements

* Identify duplicated construction protocol surfaces across Template, Toy, and
  TensorExtLite.
* Add a shared C++ construction-protocol support surface with generic data
  structures and validation entry points.
* Rewire all three existing construction-protocol consumers onto the shared
  validation and generated-output route if bounded. If all three become too
  large, finish Template plus one concrete consumer and leave the third open
  truthfully.
* Preserve extension-local hooks for:
  * family name and plugin identity;
  * capability id/kind and first-slice variant;
  * semantic role names and role order;
  * ODS role-op identity and typed role id;
  * role-specific interface expectations;
  * EmitC role-to-call mapping and required header;
  * target artifact route id, emission kind, artifact kind, runtime ABI kind,
    runtime ABI name, and runtime glue role.
* Fail closed through the shared route for missing, stale, wrong-role,
  wrong-interface, wrong-route, malformed role-to-call map, and missing
  evidence-profile data.
* Remove or quarantine duplicated validator/generated-output code that is no
  longer the production path.
* Keep built-in plugin and target registry changes mechanical if they are
  touched.
* Prove core `tcrv.exec`, generic transforms, and common orchestration passes
  do not gain extension-specific semantic branches.

## Acceptance Criteria

* [x] A shared C++ construction-protocol route/support module exists and is
      wired into CMake.
* [x] Template construction-protocol validation and generated output route use
      the shared module in production.
* [x] Toy construction-protocol validation and generated output route use the
      shared module in production.
* [x] TensorExtLite construction-protocol validation and generated output route
      use the shared module in production, unless explicitly staged with a
      truthful blocker.
* [x] Extension-local constants and typed ODS role identities remain local to
      each extension.
* [x] Shared validation preserves fail-closed coverage for manifest, role
      graph, common interfaces, typed role realization, EmitC mapping, evidence
      profile, and selected role-op interface data.
* [x] Existing Template/Toy/TensorExtLite positive artifact routes still emit
      deterministic generated-output records.
* [x] Existing Template/Toy/TensorExtLite stale/missing/wrong-role/wrong-route
      negative routes still fail before artifact output.
* [x] Focused C++ tests prove shared validation/generated-output code is
      consumed by Template, Toy, and TensorExtLite rather than duplicated.
* [x] RVV plugin regression still passes if shared plugin/target link surfaces
      are touched.
* [x] No Template/Toy/TensorExtLite semantic branches are added to
      `tcrv.exec`, `lib/Transforms`, or common orchestration passes.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis validation passes before finish/archive and after archive.
* [x] Work is finished, archived, and committed as one coherent commit if the
      task completes.

## Definition Of Done

* Focused build targets pass for the touched common construction route,
  Template/Toy/TensorExtLite plugin and target libraries, generated headers,
  `tcrv-opt`, and `tcrv-translate`.
* Focused C++ tests pass for Template, Toy, TensorExtLite construction
  protocol behavior and any new common construction-protocol coverage.
* Focused lit/FileCheck tests pass for Template/Toy/TensorExtLite positive and
  fail-closed artifact routes.
* A bounded reference scan shows obsolete duplicated validation/generated route
  helpers are no longer the production path.
* No `ssh rvv` evidence is required unless RVV emitted artifacts change.
* Trellis task status, context, archive state, and git commit are truthful.

## Out Of Scope

* New extension-family instantiation as the main result.
* Documentation-only cleanup, checklist-only work, smoke-only coverage, or
  helper-only extraction without consumer rewiring.
* RVV vmul, i64, LMUL, dtype, descriptor cleanup, runtime, correctness, or
  performance expansion.
* Treating Template, Toy, TensorExtLite, RVV, IME, or future targets as
  independent backend dialects outside the TCRV extension-family model.
* Python compiler implementation.
* Direct descriptor-to-C exporter architecture.
* Moving computation semantics into `tcrv.exec`.
* Hardware/runtime correctness claims.

## Technical Notes

* Specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/plugin-protocol/index.md`
  * `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  * `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  * `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Prior PRDs read:
  * `.trellis/tasks/archive/2026-05/05-13-tensor-ext-lite-construction-protocol-extension/prd.md`
  * `.trellis/tasks/archive/2026-05/05-13-first-concrete-extension-family-template-instantiation/prd.md`
* Relevant journal entries read:
  * `.trellis/workspace/codex/journal-5.md` sessions for Template, Toy, and
    TensorExtLite construction-protocol work.
* Initial implementation direction:
  * add a common C++ construction-protocol support module under the plugin
    layer;
  * convert the three extension-specific construction protocol headers to
    alias the common data model while preserving public extension-local API
    names;
  * route the three extension-specific verifier/build functions through shared
    validation specs rather than copy-pasted helper logic;
  * keep plugin-selected metadata and target exporter registration local unless
    a bounded common target helper can be safely consumed in the same round.

## Completion Summary

Implemented a shared production construction-protocol route layer and rewired
all three existing consumers:

* Added `TianChenRV/Plugin/ConstructionProtocol.h` plus
  `TianChenRVPluginConstructionProtocol`.
* Moved common manifest / semantic-role / typed-role / role-op interface /
  EmitC role-to-call / evidence-profile validation into the shared C++ route.
* Moved common generated-output route construction and generated artifact
  emission helpers into the shared route.
* Converted Template, Toy, and TensorExtLite construction-protocol public data
  types to aliases of the shared data model while preserving each extension's
  public API names.
* Rewired Template, Toy, and TensorExtLite verifier/build functions to call the
  shared route with extension-local validation specs.
* Rewired Template, Toy, and TensorExtLite target metadata artifacts to emit
  typed-role and generated-output records through common emission helpers.
* Preserved extension-local family constants, capability ids, role names, ODS
  op identities, typed role ids, route ids, required headers, and EmitC call
  spellings.
* Left built-in plugin and target registration semantics unchanged.
* Removed the duplicated production validator/generated-output helper bodies
  from the three extension-specific construction protocol implementations.

## Checks Run

* `cmake --build build --target TianChenRVPluginConstructionProtocol TianChenRVTemplatePlugin TianChenRVToyPlugin TianChenRVTensorExtLitePlugin -j2`
* `cmake --build build --target TianChenRVPluginConstructionProtocol TianChenRVTemplateDialect TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test TianChenRVToyDialect TianChenRVToyPlugin TianChenRVToyTarget tianchenrv-toy-extension-plugin-test TianChenRVTensorExtLiteDialect TianChenRVTensorExtLitePlugin TianChenRVTensorExtLiteTarget tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
* `./build/bin/tianchenrv-construction-protocol-common-test`
* `./build/bin/tianchenrv-template-extension-plugin-test`
* `./build/bin/tianchenrv-toy-extension-plugin-test`
* `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='ConstructionProtocol|Template|template|Toy|toy|TensorExtLite|tensorext-lite|tensorext_lite'` from `build/test`
* Core neutrality scan:
  `rg -n "tcrv_template|template-plugin|Template|tcrv_toy|toy-plugin|Toy|tensorext|TensorExtLite|tcrv_tensorext_lite" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
* Duplicate-route ref scan:
  `rg -n "parseRoleToCallMapInManifestOrder|verifyConstructionManifest\\(|verifyTypedRoleGraphRealization\\(|verifyRoleOpInterface\\(|buildGeneratedOutputRoute\\(|emitGeneratedOutputRoute|emitTypedRoleGraphRealization" include/TianChenRV/Plugin lib/Plugin lib/Target test/Plugin`
* `git diff --check`
* `git diff --cached --check`
* Trellis validation before finish/archive and after archive.

No `ssh rvv` evidence was run because this task changes construction-protocol
metadata validation and non-executable generated artifact emission only. It
does not change RVV runtime artifacts and makes no RVV runtime, correctness,
or performance claim.
