# tcrv-translate export coherence gate

## Goal

Make the direct `tcrv-translate` artifact export boundary fail closed on execution-plan coherence before invoking target-owned artifact exporters.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Starting HEAD is expected and observed as `d11714d feat: add execution plan coherence preflight`.
* Starting worktree was clean.
* `predoc/` is absent and no files under `build/` are tracked.
* Previous round added the generic C++ execution-plan coherence helper and `tcrv-check-execution-plan-coherence` pass.
* This round must not create a new artifact route, runtime, object generation path, benchmark, or RVV runtime/correctness/performance claim.

## Requirements

* Wire the existing generic C++ coherence helper into both `tcrv-translate` artifact export commands:
  * `--tcrv-export-target-source-artifact`
  * `--tcrv-export-target-artifact`
* Keep the tool boundary generic:
  * built-in extension plugins and target artifact exporters may be registered deterministically;
  * no branching on RVV, scalar, offload, IME, Sophgo, AME, vendor, runtime family, dtype, shape, toolchain, or microarchitecture.
* Preserve target-owned exporter semantics and the built-in target artifact exporter registry boundary.
* Preserve the existing explicit `tcrv-check-execution-plan-coherence` pass and `tcrv-opt` behavior.
* Add focused lit/FileCheck coverage under `test/Target/ArtifactExport`.
* Positive coverage must show legal RVV source, scalar source, and offload descriptor exports still succeed.
* Negative coverage must show direct export rejects:
  * stale selected-path variant symbol;
  * selected-path origin mismatch or unregistered origin plugin;
  * lowering-boundary kernel/variant/origin mismatch;
  * emission-plan route/origin mismatch;
  * offload descriptor route missing runtime ABI metadata;
  * source/descriptor route cross-spoofing through contradictory metadata.
* Preserve or update route diagnostics to assert the stronger preflight reason and ensure failed export does not leak source/descriptor content.

## Acceptance Criteria

* `git diff --check` passes.
* CMake configure passes:
  `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes.
* Trellis task is archived and validates if task state is updated.
* One coherent commit is created and the final worktree is clean.

## Out of Scope

* No Python compiler internals.
* No target-specific logic in generic transform, target-neutral support, or `tcrv-translate`.
* No generic compute ops inside `tcrv.exec`.
* No new target exporter route or new execution runtime.
* No SSH RVV run unless a new RVV runtime/correctness/performance claim is intentionally made.

## Technical Notes

* Required inspection commands were run before edits.
* Required code/spec files are listed in the user request and will be read before compiler edits.
