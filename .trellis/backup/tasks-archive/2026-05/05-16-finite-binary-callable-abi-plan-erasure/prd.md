# Finite binary callable ABI plan erasure

## Goal

Erase `RuntimeABICallablePlan` as a finite-binary callable ABI authority. This
round deletes the support-layer callable-plan metadata mirror and direct /
dispatch invocation comment contract surface so future runtime ABI evidence
must come from IR-modeled ABI surfaces and materialized artifacts, not from a
standalone finite-family support plan.

## Why

The previous deletion round removed target route-metadata authority, but the
support layer still exposes `FiniteBinaryCallableABIPlan`,
`buildFiniteBinaryCallableABIPlan`,
`validateFiniteBinaryCallableABIParameterMirror`,
`RuntimeABIInvocationContract`, `buildRuntimeABIInvocationContract`, and
`formatRuntimeABIInvocationContractCommentBody`. The active standalone test
protects metadata mirroring of an IR-backed callable plan and comment-body
invocation contracts for direct and dispatch callables. That keeps runtime ABI
evidence anchored in finite-family metadata instead of materialized
extension-family EmitC/runtime artifact paths.

## Scope

- Delete `include/TianChenRV/Support/RuntimeABICallablePlan.h`.
- Delete `lib/Support/RuntimeABICallablePlan.cpp` and remove it from
  `TianChenRVSupport`.
- Delete `test/Support/RuntimeABICallablePlanTest.cpp` and remove the
  standalone test target and `check-tianchenrv` dependency.
- Delete the standalone lit wrapper
  `test/Support/runtime-abi-callable-plan.test`.
- Rewrite directly affected spec text that currently protects finite-binary
  callable-plan construction, metadata-parameter mirroring, or
  runtime-ABI-invocation comment contracts.
- Run focused active-surface scans and targeted build/check commands after
  deletion.

## Non-goals

- No Common EmitC rebuild.
- No new runtime ABI subsystem.
- No replacement invocation-contract schema.
- No plugin template work.
- No RVV emission or source artifact route.
- No descriptor compatibility wrapper.
- No broad rewrite of neutral `RuntimeABIParam`, `RuntimeABIMemWindow`, or
  `RuntimeABIContract` helpers unless direct consumers make deletion
  impossible.
- No restoration of finite-binary callable-plan behavior to keep tests green.

## Requirements

- No active production or test surface builds a finite-binary callable ABI plan
  from plugin family contracts.
- No test protects metadata-parameter mirroring of an IR-backed callable plan.
- No direct or dispatch runtime invocation contract is emitted as comment-body
  metadata.
- CMake must no longer build or depend on a standalone runtime ABI callable-plan
  test.
- Remaining runtime ABI support may keep neutral parameter, role,
  mem-window, runtime-param, and bounded contract-shape helpers only if they do
  not reintroduce callable-plan metadata mirrors or direct-call comment
  contracts.
- Any deletion-exposed missing new-architecture failure must be reported as a
  gap instead of repaired by restoring the deleted authority.

## Acceptance Criteria

- [ ] Focused active-surface ref-scan finds no live reference to
  `RuntimeABICallablePlan`, `FiniteBinaryCallableABIPlan`,
  `buildFiniteBinaryCallableABIPlan`,
  `validateFiniteBinaryCallableABIParameterMirror`,
  `RuntimeABIInvocationContract`, `buildRuntimeABIInvocationContract`,
  `formatRuntimeABIInvocationContractCommentBody`,
  `finite binary runtime ABI callable plan`,
  `plugin-direct-invocation-contract`, or
  `runtime_abi_invocation_contract` outside archived tasks/workspace/build/git
  and allowed historical notes.
- [ ] The `TianChenRVSupport` library no longer compiles
  `RuntimeABICallablePlan.cpp`.
- [ ] `test/CMakeLists.txt` no longer defines or depends on
  `tianchenrv-runtime-abi-callable-plan-test`.
- [ ] The standalone lit wrapper for the deleted support executable is removed.
- [ ] Affected support/target test targets that still exist build and run, or
  any failure is reported as a missing new-architecture gap.
- [ ] `ninja -C build check-tianchenrv` is attempted after focused checks.
- [ ] `git diff --check`, Trellis validation, task finish/archive, clean
  status, and one coherent commit are completed if the task is complete.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `include/TianChenRV/Support/RuntimeABICallablePlan.h`
- `lib/Support/RuntimeABICallablePlan.cpp`
- `test/Support/RuntimeABICallablePlanTest.cpp`
- `lib/Support/CMakeLists.txt`
- `test/CMakeLists.txt`

## Technical Notes

- Initial repo state: `/home/kingdom/phdworks/TianchenRV`, clean worktree,
  HEAD `4e10d85 chore(target): erase route metadata authority`.
- Initial active-surface scan found callable-plan and invocation-contract
  references in `RuntimeABICallablePlan.{h,cpp}`, its standalone test/CMake
  target, and active lowering-runtime spec text.
- `RuntimeABIContract`, `RuntimeABIParam`, and `RuntimeABIMemWindow` still have
  active target/runtime support consumers. This task does not delete them unless
  required to remove direct callable-plan consumers.
- This is part of the Wrong Logic Deletion Campaign: deletion before rebuild,
  no compatibility wrappers or replacement runtime ABI schema in this round.
