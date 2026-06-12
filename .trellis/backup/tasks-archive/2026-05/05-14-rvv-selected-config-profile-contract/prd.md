# RVV selected config profile contract

## Module Goal

Make the RVV selected vector config a first-class plugin-owned profile for the
current bounded binary-family route. The profile must be selected by the RVV
plugin and consumed by the materialized RVV lowering/emission path, generated
artifact export, RuntimeABI callable planning, and RVVScalarDispatch without
reconstructing SEW, LMUL, tail policy, mask policy, or runtime AVL/VL authority
from descriptor shape, per-family strings, or implicit route assumptions.

## Scope

This round is bounded to the existing i32 RVV binary families:

- source-frontdoor finite binary selection for i32 vadd, vsub, and vmul;
- RVV selected lowering boundary and selected emission-plan metadata;
- RVV microkernel materialization/source/header/object artifact export;
- RuntimeABICallablePlan/RVVScalarDispatch consumption of the selected RVV
  profile;
- focused local and ssh-rvv evidence for the existing generated artifact path.

The implementation may touch shared RVV helpers where the current i32 path
already shares code with i64, but this task does not claim a broad dtype,
tail/mask, LMUL, performance, or op-family expansion.

## Boundaries And Non-Goals

- Do not introduce Python compiler semantics.
- Do not add descriptor-to-C production export or descriptor-owned computation.
- Do not put RVV semantic branches into generic core orchestration.
- Do not treat helper-only, metadata-only, PRD-only, journal-only, test-only, or
  evidence-packaging-only work as completion.
- Do not expand to a dtype/LMUL/tail/mask matrix in this round.
- Do not replace clang/LLVM as the default native route.
- Do not make runtime/correctness/performance claims without focused local
  generated-artifact evidence and at least one focused `ssh rvv` run after the
  profile change.

## Required Behavior

1. The selected config profile separates three layers:
   hardware facts from the target capability profile; compile-time variant
   config including SEW, LMUL, tail policy, mask policy, vector type/suffix, and
   selected intrinsic spelling; and runtime SSA/control roles including runtime
   element count, AVL source, VL source, and source-frontdoor dynamic extent
   roles. Dispatch availability remains in the dispatch contract rather than in
   the RVV selected config profile.

2. Existing i32 vadd/vsub/vmul carry the selected profile through the same
   production chain:
   source-frontdoor marker and runtime ABI -> RVV plugin proposal/selection ->
   selected lowering boundary -> selected emission plan -> RVVMicrokernel /
   TargetArtifactExport -> RuntimeABICallablePlan / RVVScalarDispatch.

3. RVVMicrokernel, TargetArtifactExport, and RVVScalarDispatch consume the
   plugin-owned selected profile metadata and direct selected config contract.
   They must fail closed when the profile metadata is missing, stale,
   conflicting, descriptor-only, or inconsistent with the selected family,
   selected shape, runtime length roles, EmitC body mapping, or source identity.

4. Descriptor-local `element_count` may remain as bounded artifact capacity
   metadata, but it is not selected config, runtime trip-count, source identity,
   or computation authority.

5. The default source-frontdoor behavior remains the existing bounded i32
   vadd/vsub/vmul route. The change should make the selected config boundary
   explicit and consumed, not replace the route with a new frontend or a broad
   matrix.

## Acceptance Criteria

- Production C++ changes define or extend a plugin-owned RVV selected config
  profile surface and wire it into the existing i32 vadd/vsub/vmul compiler
  path.
- Selected profile metadata or typed profile contract entries are emitted by
  the RVV plugin selected-emission path and consumed by RVVMicrokernel /
  TargetArtifactExport and RVVScalarDispatch.
- SEW, LMUL, tail policy, mask policy, ABI signature, runtime element-count
  C-name, AVL/VL roles, and selected source identity are checked against the
  same selected profile/contract, not reconstructed from descriptor-only state.
- Missing or stale profile metadata, missing EmitC body mapping, unknown/stale
  family, mismatched SEW/LMUL, missing tail/mask policy, missing runtime length
  role data, stale ABI signature, descriptor-only production attempts,
  explicit-only route misuse, and source-identity mismatch fail before artifact
  or runtime claims.
- Focused lit/FileCheck and C++ tests cover vadd/vsub/vmul profile propagation
  and fail-closed behavior.
- Local generated-artifact/runtime regressions for vadd/vsub/vmul remain green.
- At least one focused `ssh rvv` generated-artifact regression is rerun after
  the profile change, or the exact blocker is recorded with no runtime claim.
- Trellis validation passes before finish and after archive.
- `git diff --check` and `git diff --cached --check` pass.
- The final worktree is clean after one coherent commit, unless an external
  blocker prevents completion.

## Evidence Plan

- Build focused C++/MLIR targets for touched support, plugin, target/export,
  and tools including generated headers, RVV plugin/target libraries,
  `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, `tcrv-translate`,
  RuntimeABI callable-plan test, target artifact export test, and affected RVV
  target tests.
- Run C++ tests for RVV plugin selected planning, RuntimeABI callable planning,
  TargetArtifactExport, and RVVScalarDispatch-relevant paths.
- Run focused lit tests for vector-dynamic i32 vadd/vsub/vmul source-frontdoor
  through bundle/artifact export and dispatch.
- Run local script regressions for generated artifacts for i32 vadd/vsub/vmul.
- Run one focused `ssh rvv` generated-artifact regression after the code change.

## Exact Next Config-Axis Owner If Unfinished

If broadening beyond the current i32 families is too large, stop after the i32
profile centralization/consumption is coherent and record the next owner as:
generalizing the selected profile to a deliberate dtype/LMUL/tail/mask matrix
with dedicated capability facts, lit coverage, and `ssh rvv` evidence for each
new axis rather than implicit reuse of the current i32 profile.

## Implementation Summary

- Added a plugin-owned `RVVBinarySelectedConfigProfile` view over the existing
  selected config contract. It emits three bounded metadata entries:
  `tcrv_rvv.selected_config_profile.hardware_facts`,
  `tcrv_rvv.selected_config_profile.variant_config`, and
  `tcrv_rvv.selected_config_profile.runtime_roles`.
- Wired the profile into RVV selected emission planning so the source-frontdoor
  vadd/vsub/vmul path publishes the same selected profile before artifact
  export.
- Made RVVMicrokernel route validation require the selected profile metadata,
  print the profile in generated source comments, and include the profile in
  object evidence.
- Made RVVScalarDispatch derive and validate dispatch bundle profile metadata
  from the same RVV selected config contract instead of reconstructing SEW,
  LMUL, tail/mask, or runtime roles from descriptor-local data.
- Extended TargetArtifactExport and focused lit tests so stale descriptor-only
  profile values and missing runtime-role profile metadata fail before artifact
  or runtime claims.

## Validation Summary

- Focused build passed:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit passed from `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-family-(sub|mul)|rvv-microkernel-auto-materialization|rvv-microkernel-runtime-abi-role-binding|rvv-scalar-i32-v(add|sub|mul)-dispatch|plan-vector-dynamic-i32-v(add|sub|mul)-and-export-target-artifact-bundle|vector-dynamic-i32-v(add|sub|mul)-to-exec'`;
  result: 22 selected tests passed.
- Script self-tests passed:
  `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Local generated-artifact bundle dry-runs passed for vadd/vsub/vmul direct
  RVVMicrokernel and RVVScalarDispatch runs under
  `artifacts/tmp/rvv-selected-config-profile-contract`.
- Focused `ssh rvv` direct RVVMicrokernel evidence passed for generated
  i32-vmul: run id `selected-config-profile-micro-vmul-ssh-rvv`, selected
  kernel `frontend_vector_dynamic_i32_vmul`, source SHA256
  `710df3874a679c4307e55f558dda38f0d8c9cafe0b29aa6a6a373220ec7cd035`.
- Focused `ssh rvv` RVVScalarDispatch bundle evidence passed for generated
  i32-vmul: run id `selected-config-profile-dispatch-vmul-ssh-rvv`,
  selected kernel `frontend_vector_dynamic_bundle_i32_vmul`,
  `ssh_evidence_verified = true`, source SHA256
  `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`.
- Artifact inspection confirmed generated source/header/object bundles and
  bundle indexes contain the selected profile metadata and dispatch profile
  metadata.
- Bounded ref-scan over changed production files found no new descriptor-to-C,
  descriptor-only runtime authority, explicit-only production authority, or
  generic core RVV semantic branch. Hits are profile notes, existing dispatch
  case emission, and existing fail-closed diagnostics.
