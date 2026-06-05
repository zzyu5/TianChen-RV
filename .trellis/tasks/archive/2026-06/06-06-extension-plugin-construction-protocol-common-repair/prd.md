# Extension plugin construction protocol common repair

## Goal

Repair the shared extension-plugin construction protocol boundary so registry
registration input, common construction protocol validation, concrete plugin
construction, and selected extension-family hooks agree with production code.
This round is bounded to the current `Plugin/construction-protocol-common.test`
failure and only includes `Plugin/template-extension-plugin.test` if the
reproduced failure is part of the same construction workflow-path boundary.

## Direction Source

Hermes Direction Brief:

`Extension plugin construction protocol common repair`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `86694ff1 rvv: close runtime scalar splat-store route`.
- No `.trellis/.current-task` existed at the start of the round, so this task
  was created before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-06-rvv-stage2-runtime-scalar-splat-store-route-closure/`
  completed the runtime scalar splat-store route owner and explicitly left
  construction protocol failures out of scope.
- Relevant specs require extension plugin construction to remain C++/MLIR and
  plugin-local: manifests/templates are optional provenance, registry/common
  code only orchestrates validation and lookup, and concrete plugins own their
  dialects, legality, selected-body realization, route providers, and export
  hooks.
- Focused direct reproduction before repair:

  ```text
  build/bin/tianchenrv-construction-protocol-common-test
  RVV executable role steps are built from route operation: TianChen-RV RVV construction protocol invalid: RVV f32 clamp/select construction requires generic tcrv_rvv.select
  ```

- `Plugin/construction-protocol-common.test` is therefore not a pure stale
  FileCheck failure. It exposes drift between the shared construction protocol
  proof and current production route families: the common test enumerates RVV
  selected-body construction routes and maps each route mnemonic to the typed
  executable role surface expected by the RVV construction protocol.
- Focused direct reproduction before repair for the sibling test:

  ```text
  build/bin/tianchenrv-template-extension-plugin-test
  process terminated by signal 11
  ```

  Batch gdb shows the crash occurs in
  `template_ext::buildTemplateComputeSkeletonEmitCLowerableRoute` while moving
  a `TCRVEmitCLowerableRoute` into the output route during
  `TemplateExtensionPlugin::checkVariantEmissionReadiness`, reached from the
  registry pipeline hook test. This is production construction/registration
  workflow drift in the Template plugin path, not a FileCheck wording issue.

## Requirements

- Keep scope to the shared construction protocol workflow path:
  plugin registry/registration input -> common construction protocol
  validation -> concrete plugin construction or template-extension fixture ->
  selected plugin hooks used by pass/export/route plumbing.
- Repair production construction or test-local proof code according to the
  durable plugin contract. Do not hide production protocol drift with weakened
  tests.
- The common construction proof must show concrete plugins register only their
  own dialect/hooks/routes through registry/interface APIs, without adding core
  semantic pass branches.
- `construction-protocol-common` must keep validating current RVV selected-body
  construction route families against generic typed `tcrv_rvv` body surfaces,
  including newly present select/clamp and epilogue route mnemonics.
- `template-extension-plugin` stays in scope because the reproduced crash is in
  the Template plugin construction/registration route workflow reached through
  the registry hook path.
- Common EmitC/export and plugin registry code must remain neutral. They must
  not infer plugin semantics from manifests, templates, route ids, artifact
  names, or status metadata.
- If a failure proves unrelated to construction protocol common behavior after
  deeper inspection, record it as the exact continuation instead of merging
  unrelated cleanup.

## Acceptance Criteria

- [x] `Plugin/construction-protocol-common.test` failure is reproduced before
      repair and root cause is classified.
- [x] `Plugin/template-extension-plugin.test` is reproduced and either repaired
      as the same construction workflow owner or documented as a separate next
      owner.
- [x] The construction common proof aligns RVV route mnemonic -> typed role
      surface classification with current production route facts for f32
      clamp/select and dequant-clamp epilogue routes.
- [x] Template plugin route construction no longer crashes when the registry
      checks emission readiness and route materialization hooks.
- [x] Concrete plugin registration remains plugin-local: no new family-specific
      semantic branches are added to common/core passes.
- [x] Focused direct binaries or lit equivalents pass for
      `construction-protocol-common` and, if in scope,
      `template-extension-plugin`.
- [x] Any affected plugin/registry unit tests are run.
- [x] Bounded ref-scan over touched plugin construction files confirms no
      descriptor-driven computation, manifest/template executable authority,
      source-front-door positive route, or legacy RVV i32 route authority is
      introduced.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      final git status are clean before finish/archive and commit.

## Evidence Plan

- Run the two focused test binaries before and after repair.
- Use the direct failing diagnostic and gdb backtrace to localize source
  changes before editing.
- Run the lit test wrappers if local lit is available; otherwise the direct
  binaries are the focused equivalent because the lit files only invoke those
  binaries plus FileCheck for the Template success line.
- Run adjacent registry/plugin tests only if touched code affects registry or
  shared route construction surfaces.
- Run bounded textual scans over touched files and added diff lines.

## Definition Of Done

- The construction protocol common proof and Template plugin workflow reflect
  current production construction behavior without adding broad full-suite
  cleanup.
- Task metadata is truthful, the task is finished/archived when complete, and
  one coherent commit records the repair.

## Out Of Scope

- RVV route expansion beyond aligning construction protocol proof with current
  route families.
- Template/Toy feature growth beyond proving the shared construction workflow.
- Registry rewrite unless required by reproduced failure evidence.
- High-level frontend work, scalar fallback work, IME/offload expansion,
  dashboards, reports, or status-only completion.
- Descriptor-driven computation, manifest/template executable authority,
  source-front-door positive routes, or compatibility wrappers preserving old
  i32 route authority.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/plugin-protocol/index.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/plugin-protocol/locality-contract.md`
- `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
- `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Primary inspection surfaces:

- `test/Plugin/construction-protocol-common.test`
- `test/Plugin/template-extension-plugin.test`
- `test/Plugin/ConstructionProtocolCommonTest.cpp`
- `test/Plugin/TemplateExtensionPluginTest.cpp`
- `include/TianChenRV/Plugin/ConstructionProtocol.h`
- `lib/Plugin/Construction/ConstructionProtocol.cpp`
- `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `include/TianChenRV/Plugin/Template/TemplateExtensionPlugin.h`
- `lib/Plugin/Template/TemplateExtensionPlugin.cpp`

## Completion Evidence

Completed as a bounded construction protocol common repair.

Reproduced failures before repair:

- `build/bin/tianchenrv-construction-protocol-common-test` failed with:
  `RVV f32 clamp/select construction requires generic tcrv_rvv.select`.
  Root cause: the common proof enumerated production RVV selected-body routes
  but mapped newer clamp/dequant-clamp route mnemonics through stale fallback
  typed-op and step-count expectations.
- `build/bin/tianchenrv-template-extension-plugin-test` terminated with
  signal 11. Batch gdb showed the Template registry emission-readiness path
  crashed while assigning a nested `TCRVEmitCLowerableRoute` temporary into
  the caller-supplied output route. This was the same construction workflow
  owner, not a separate FileCheck drift.

Implementation completed:

- `TCRVEmitCLowerableRoute` now has explicit default/copy/move construction
  and assignment plus `reset(routeID, routeKind)` for provider-facing
  out-parameter construction.
- Template route provider now validates the construction protocol and fills
  the caller-supplied route in place with route id, header, function
  declaration, source provenance, and call step.
- The common construction proof now consumes the production route table's
  `typedComputeOpName` and adds route-step/runtime-ABI facts for
  `f32_clamp_select`, `dequant_clamp_f32_epilogue`, and
  `widening_product_reduce_dequant_clamp_f32`.
- Template plugin test now reports route id, source provenance, and call-step
  preservation separately.
- `.trellis/spec/lowering-runtime/emitc-route.md` records the provider-owned
  route value contract and `reset`-based out-parameter construction rule.

Checks run:

- `cmake --build build --target tianchenrv-construction-protocol-common-test tianchenrv-template-extension-plugin-test tianchenrv-emitc-lowerable-interface-test`
- `build/bin/tianchenrv-construction-protocol-common-test`
- `build/bin/tianchenrv-template-extension-plugin-test`
- `build/bin/tianchenrv-emitc-lowerable-interface-test`
- `build/bin/tianchenrv-toy-extension-plugin-test`
- `build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- `build/bin/tianchenrv-plugin-registry-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 -m lit -sv test/Plugin/construction-protocol-common.test test/Plugin/template-extension-plugin.test`
  was attempted but unavailable locally: `/usr/bin/python3: No module named
  lit`. The direct binaries are the focused equivalent for these wrappers in
  this checkout.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-extension-plugin-construction-protocol-common-repair`
- Bounded scan over touched files and added diff lines for descriptor,
  source-front-door/source-artifact, manifest/template authority, and legacy
  RVV i32 route authority markers.
