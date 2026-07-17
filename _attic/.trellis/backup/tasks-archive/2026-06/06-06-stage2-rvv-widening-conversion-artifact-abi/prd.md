# Stage2 RVV widening conversion executable artifact ABI boundary

## Goal

Make the existing widening integer conversion selected-body RVV route family
truthfully executable as generated target artifacts, or fail closed at the
precise missing artifact/ABI boundary. The owned boundary is typed
`tcrv_rvv` conversion body facts, source/result element types, source/result
SEW-LMUL relation, conversion kind/relation, LMUL/config/policy, runtime
AVL/VL, per-operand ABI/header bindings, RVV plugin-owned route validation,
common EmitC materialization, target artifact export, generated bundle ABI,
and `ssh rvv` correctness evidence.

## Why Now

Commit `dad89a62` closed the compare/select executable artifact ABI gap by
proving provider-owned compare/select facts through generated bundles and
`ssh rvv` correctness, including tail preservation. The next Stage2 bottleneck
is the conversion/dtype/SEW class, starting with widening integer conversion
because it exercises result type derivation and SEW widening rather than
memory, mask, elementwise, or MAcc semantics.

## What I Already Know

- There was no active Trellis task; this task was created from the Hermes
  Direction Brief.
- The work is Stage2 RVV coverage on corrected typed low-level `tcrv_rvv`
  selected bodies, not Stage1 legacy `i32m1` route-authority growth.
- The authority chain is `tcrv.exec` selected RVV variant -> typed or realized
  `tcrv_rvv` body -> RVV plugin legality/selected-body realization/route
  provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact ->
  generated bundle -> `ssh rvv` evidence when runtime correctness is claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only. Source/result dtype, SEW widening, LMUL, conversion relation,
  intrinsic spelling, ABI order, and route support must come from RVV provider
  facts derived from the selected typed conversion body/config/runtime facts.
- Common EmitC/export may carry provider-built payloads and mirrors, but must
  not infer widening conversion semantics from route ids, ABI strings,
  artifact metadata, test names, helper names, or exact intrinsic spellings.
- The specs already define widening conversion route-provider preflight and a
  conversion dtype-policy target validation contract. Target artifact
  validation must consume provider contracts and mirror facts before accepting
  executable artifact/header claims.
- Existing widening conversion generated-bundle tests named in the brief are
  dry-run or direct fail-closed coverage; this round must audit whether a
  non-dry-run `ssh rvv` evidence gap or a production validation gap remains.
- The archived compare/select task is a reference for evidence shape and
  harness-level tail-preservation proof. It is not proof that widening
  conversion is complete.

## Requirements

- Preserve RVV plugin ownership for widening conversion realization,
  family/provider facts, source/result dtype policy, source/result SEW-LMUL,
  conversion relation, operand binding, statement planning, runtime AVL/VL,
  header/type summaries, and target validation contracts.
- Prove or repair the pre-realized widening conversion path from selected
  boundary materialization through provider route facts, common EmitC
  materialization, target artifact export, generated bundle compile, and
  `ssh rvv` correctness when executable behavior is claimed.
- Start with pre-realized `widen_i16_to_i32`. If the same production seam
  naturally covers `widen_i32_to_i64` without broad matrix work, include it in
  the same coherent round; otherwise leave the exact continuation point.
- Harden the production seam if audit shows dry-run-only support, stale route
  validation, missing provider preflight, missing conversion dtype-policy
  target validation, wrong source/result element type, wrong SEW widening,
  stale LMUL/config, wrong ABI/header binding, wrong generated C type, or
  metadata/route-id/common-EmitC semantic authority.
- Add or retain focused fail-closed evidence for at least one stale or missing
  executable-boundary fact such as source element type, result element type,
  SEW widening, conversion kind/relation, header/type summary, ABI order,
  runtime AVL/VL, or statement-plan leaf.
- Keep support levels separate: parseable/verifier-legal is not
  route-supported; route-supported is not executable without complete
  ABI/runtime/export support and real `ssh rvv` evidence for runtime claims.

## Acceptance Criteria

- [x] PRD and Trellis context identify the widening conversion executable
  artifact ABI boundary, non-goals, and relevant specs.
- [x] Repository audit records whether current pre-realized widening
  conversion is dry-run-only, stale, under-validated, already production-valid,
  or too broad for a single round.
- [x] Production code is changed when the executable boundary is incomplete or
  under-validated; otherwise the PRD records a precise no-source-change
  justification backed by focused evidence.
- [x] Positive evidence covers the selected scope through selected boundary
  materialization, emission plan, target artifact export, generated bundle
  compile, and `ssh rvv` correctness if runtime behavior is claimed.
- [x] Focused fail-closed evidence rejects at least one stale/missing widening
  conversion executable-boundary fact before artifact acceptance or
  executable-route claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] Relevant generated-bundle dry-run and direct pre-realized fail-closed
  tests pass for the selected widening conversion scope.
- [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*` route
  authority, source-front-door route, descriptor compute path, or exact
  intrinsic spelling as route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean after commit.

## Out of Scope

- No broad conversion matrix, dtype/LMUL clone batch, high-level
  Linalg/Vector/StableHLO frontend, per-Linalg route authority,
  source-front-door positive route, performance tuning database, dashboard, or
  report-only closeout.
- No dequantize/product-reduce/clamp route expansion unless used only as a
  bounded reference for the conversion dtype-policy seam.
- No compare/select, MAcc, reduction, memory, segment2, product/dequant,
  contraction, or unrelated mask route rework except as bounded reference.
- No common EmitC invention of RVV source/result dtype, SEW widening,
  conversion relation, runtime VL, ABI, intrinsic spelling, or route support
  semantics.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, emission, or route semantics. Python
  may support generated-bundle tooling and evidence collection only.

## Technical Approach

1. Audit the current widening conversion realization owner, schedule facts,
   route-provider preflight, statement plan, target validator,
   generated-bundle script, and named tests.
2. Narrow the executable scope if needed, starting with pre-realized
   `widen_i16_to_i32`.
3. Patch production C++/MLIR/test/tooling only where the executable
   artifact/ABI seam is missing or stale.
4. Run focused C++ tests, lit/generated-bundle dry-run/fail-closed tests, and
   non-dry-run `ssh rvv` generated-bundle evidence for any claimed executable
   behavior.
5. Finish/archive the Trellis task and commit one coherent change.

## Technical Notes

- Read specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/guides/index.md`
  - `.trellis/spec/guides/capability-first-design-guide.md`
  - `.trellis/spec/guides/plugin-locality-review-guide.md`
  - `.trellis/spec/guides/compute-boundary-review-guide.md`
- Reference archive:
  - `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-compare-select-artifact-abi/`
- Bounded source/test targets from the Direction Brief:
  - `include/TianChenRV/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.h`
  - `lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`
  - `lib/Plugin/RVV/RVVGearboxSchedules.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widen-i16-to-i32-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-widen-i32-to-i64-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widen-i32-to-i64-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-widen-i16-to-i32-fail-closed.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-widen-i32-to-i64-fail-closed.test`
  - `test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-widen-i32-to-i64.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir`

## Audit Conclusion

The production widening conversion seam is already provider-owned and
fail-closed at the route and target artifact boundary. No compiler source
change was needed in this round.

- `RVVWideningConversionSelectedBodyRealizationOwner.cpp` validates the
  pre-realized body as a direct selected `tcrv.exec.variant` child, checks the
  bounded supported conversion forms (`widen_i32_to_i64` and
  `sign_extend_widen_vf2`), verifies source/result SEW-LMUL and conversion
  relation, requires tail/mask agnostic policy, validates explicit runtime ABI
  roles for `lhs`, `out`, and `n`, rejects mixed already-realized RVV body
  ops, and realizes the body into `setvl`, `with_vl`, `load`,
  `widening_convert`, and `store`.
- `RVVEmitCRouteProvider.cpp` calls
  `getRVVSelectedBodyWideningConversionRouteStatementPlan(...)` and
  `verifyRVVSelectedBodyWideningConversionRouteProviderFacts(...)` after
  top-level family/provider verification and before constructing the
  `TCRVEmitCLowerableRoute`.
- `RVVEmitCRoutePlanning.cpp` carries widening conversion family plans and the
  conversion dtype-policy owner. It verifies source/result dtype policy,
  source/result SEW-LMUL, conversion relation, intrinsic/type/header facts,
  route operand binding, runtime ABI order, runtime AVL/VL control, target
  profile, provider mirror, and rejects stale scalar-broadcast/dequant or
  non-conversion facts.
- `RVVTargetArtifactRouteFamilyValidation.cpp` dispatches widening conversion
  through the `conversion-dtype-policy` route-family validator. The validator
  obtains `RVVConversionDtypePolicyRouteValidationContract`, consumes the
  embedded `RVVRuntimeAVLVLSelectedBoundaryContract` before route payload and
  metadata mirrors, validates route-local runtime mirrors, ABI mappings,
  source/result element types, SEW/LMUL, conversion kind/relation, headers,
  type mappings, statement counts and statement leaves, and rejects stale
  non-conversion/dequantization mirrors.
- The generated-bundle harness already checks active-lane sign extension,
  mixed-sign and wide-magnitude input patterns where applicable, two input
  patterns, runtime counts including zero and multi-VL counts, and output tail
  sentinel preservation.

The current executable gap was evidence, not source behavior: the repository
had dry-run and fail-closed widening conversion coverage, but the task brief
required non-dry-run generated-bundle `ssh rvv` correctness evidence for the
pre-realized widening conversion route family. This round produced that
evidence for both `widen_i16_to_i32` and `widen_i32_to_i64`.

## Implementation Summary

- Created the Trellis task and bounded PRD from the Hermes Direction Brief.
- Audited the widening conversion realization owner, provider preflight,
  conversion dtype-policy route-family owner, target artifact validator,
  generated-bundle harness, and named dry-run/fail-closed fixtures.
- No production source files were changed because the audited seam already
  satisfies the provider-owned route/ABI/header/type/runtime validation
  contract and focused checks did not expose a stale or under-validated
  production boundary.
- No `.trellis/spec/` update was needed: the relevant widening conversion
  provider preflight, conversion dtype-policy target validation, runtime
  AVL/VL, header/type, generated-bundle evidence, and fail-closed test
  contracts are already present in the current specs.

## Evidence

Positive executable `ssh rvv` evidence:

- Command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-widening-conversion-audit --run-id pre-realized-widen-ssh --overwrite --op-kind widen_i16_to_i32 --op-kind widen_i32_to_i64 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
- Result: `rvv_generated_bundle_abi_e2e: success`.
- Artifact root:
  `artifacts/tmp/stage2-rvv-widening-conversion-audit/pre-realized-widen-ssh`.
- `widen_i16_to_i32` executed on `ssh rvv` for counts `0,1,16,23,257` and
  both input patterns. Every case reported `sign_extension_checked`,
  `two_input_patterns_checked`, and `tail_preserved`, with final marker
  `tcrv_rvv_generated_bundle_abi_widen_i16_to_i32_ok`.
- `widen_i32_to_i64` executed on `ssh rvv` for counts `0,1,16,23,257` and
  both input patterns. Every case reported `sign_extension_checked`,
  `wide_magnitude_checked`, `tail_preserved`, and
  `two_input_patterns_checked`, with final marker
  `tcrv_rvv_generated_bundle_abi_widen_i32_to_i64_ok`.

Positive dry-run evidence:

- Combined pre-realized dry-run for `widen_i16_to_i32` and `widen_i32_to_i64`
  passed with `rvv_generated_bundle_abi_e2e: dry_run_success` under
  `artifacts/tmp/stage2-rvv-widening-conversion-audit/pre-realized-widen-dry-run`.
- Explicit selected-body `widen_i32_to_i64` dry-run passed with
  `rvv_generated_bundle_abi_e2e: dry_run_success` under
  `artifacts/tmp/stage2-rvv-widening-conversion-audit/explicit-widen-i32-dry-run`.

Focused fail-closed evidence:

- Direct pre-realized route-entry shortcut for `widen_i16_to_i32` failed as
  expected with:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): widen_i16_to_i32; the direct route-entry shortcut is retired and these fixtures must use the public selected lowering-boundary producer before target bundle export`.
- Direct pre-realized route-entry shortcut for `widen_i32_to_i64` failed as
  expected with the analogous diagnostic for `widen_i32_to_i64`.
- Existing target fixtures retain stale mirror rejection for source/result
  conversion facts, including stale conversion kind, source SEW, destination
  LMUL, runtime ABI order, route operand binding summary, header summary,
  conversion plan, conversion relation, provider mirror, dequantization
  residue, and elementwise residue.

Build and local checks:

- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `./build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 -m lit --version` failed because local `lit` is not installed, so
  the relevant lit files were checked through their underlying script/tool
  commands instead of the lit runner.
