# linalg-fed RVV scalar dispatch external ABI bundle

## Goal

Make the bounded marked `linalg.generic` i32-vadd frontend feed the existing
RVV plus scalar dispatch/fallback external ABI bundle path, then prove the
generated bundle can be compiled, linked, and run on the real `ssh rvv` target
for both scalar fallback and RVV branches.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Expected starting HEAD is `1a26a4f feat: export rvv bundle runtime abi artifacts`.
* Previous round made direct linalg-fed RVV external ABI bundle artifacts real.
* This round must reuse existing C++/MLIR planning, dispatch synthesis, runtime
  guard, scalar fallback, RVV microkernel, RVVScalarDispatch target exporter,
  and external ABI machinery.
* Python is allowed only for evidence orchestration and artifact consumption.
* No subagents, parallel agents, background queues, or multi-agent workflows are
  allowed for this round.

## Requirements

* Extend the durable C++ compiler/export path so marked linalg i32-vadd can
  reach the RVV plus scalar dispatch external ABI bundle route.
* Keep `tcrv.exec` compute-free; do not add target-family branches to core
  orchestration.
* Keep extension semantics plugin-local and target/exporter-owned.
* Preserve separation among target capability facts, compile-time variant
  configuration, runtime ABI/control values, and descriptor-local bounded
  parameters.
* Add focused local coverage for linalg-fed dispatch bundle metadata.
* Run real `ssh rvv` evidence if runtime correctness is claimed, exercising
  `rvv_available=0` and `rvv_available=1`, preferably with at least two runtime
  `n` values if supported by existing evidence tooling.

## Acceptance Criteria

* [ ] Linalg-fed i32-vadd reaches RVV plus scalar dispatch/fallback external ABI
      bundle export through the compiler/exporter front door.
* [ ] A focused lit/FileCheck test covers linalg-fed dispatch bundle metadata.
* [ ] `git diff --check` passes.
* [ ] `cmake --build build --target tcrv-translate` passes.
* [ ] `cmake --build build --target check-tianchenrv` passes.
* [ ] If claimed, remote `ssh rvv` compile/link/run evidence passes both scalar
      fallback and RVV branches.
* [ ] The task is archived before final commit and the final worktree is clean.

## Out Of Scope

* General linalg/stablehlo/tosa lowering beyond the bounded marked i32-vadd
  frontend.
* New Python implementations of compiler IR, dialects, passes, plugin registry,
  capability model, variant pipeline, lowering, target emission, dispatch
  semantics, or runtime ABI semantics.
* New standalone helper routes that bypass the C++ compiler/exporter path.
* Performance claims.

## Technical Notes

* Required spec and code inspection is listed in the user request and must be
  completed before editing compiler code.
* Specs may only be updated if they describe a durable compiler/export contract
  implemented in the same commit.
