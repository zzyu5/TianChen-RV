# RVV Remote Evidence Replay Into Capability Planning

## Goal

Add a smallest reproducible replay path from sanitized `ssh rvv` probe evidence
into parseable `tcrv.exec` capability MLIR so the existing C++/MLIR execution
planning pipeline can be driven by bounded RVV hardware/toolchain facts.

## Requirements

* Add or extend a Python helper under `scripts/` for artifact parsing only. The
  helper reads `rvv_probe_evidence.json` produced by `scripts/rvv_remote_probe.py`
  and emits bounded MLIR containing existing `tcrv.exec.capability` ops.
* The helper must not implement plugin decisions, legality, selection,
  lowering, emission, runtime ABI, or compiler internals.
* Emitted RVV capability fields must match the property names consumed by the
  C++ RVV plugin/profile: `architecture`, `isa_vector_hints`, hart `count`,
  compile-run `selected_march`, optional `selected_mabi`, optional source/binary
  digests, `status`, and the stable RVV capability IDs.
* Scalar fallback capability may be emitted only when explicitly requested for
  deterministic fallback coverage; it must remain explicit capability metadata.
* Add deterministic sanitized JSON fixtures and script/lit coverage. Fixtures
  must not contain secrets, credentials, private keys, tokens, unbounded logs, or
  raw command dumps.
* Add positive coverage showing replayed MLIR parses and the existing
  `--tcrv-execution-planning-pipeline` materializes RVV proposal metadata,
  `tcrv_rvv.lowering_boundary`, scalar fallback boundary when included, and
  boundary-linked emission-plan diagnostics.
* Add negative coverage for missing/non-riscv64 architecture, missing vector
  hints or selected march, failed compile-run evidence, secret-like/unbounded
  fields, and malformed MLIR not being swallowed by the pipeline.
* Update durable specs only for the replay contract: remote evidence may feed
  capability metadata, but it is bounded capability-decision evidence only, not
  generated-code correctness or performance.
* Run the real `ssh rvv` probe once if reachable, storing output only under
  `artifacts/tmp/rvv_probe/<run-id>/`.

## Acceptance Criteria

* [ ] `python3 scripts/rvv_remote_probe.py --self-test` passes.
* [ ] New evidence-to-MLIR helper self-test or fixture test passes directly.
* [ ] lit/FileCheck covers positive replay-to-MLIR syntax and planning pipeline
      behavior.
* [ ] lit/FileCheck covers negative replay/parser cases without inventing RVV
      support.
* [ ] `git diff --check` passes.
* [ ] CMake configure with LLVM/MLIR 20 paths succeeds.
* [ ] `cmake --build build --target check-tianchenrv -j2` passes.
* [ ] Any real probe artifact stays under `artifacts/tmp/rvv_probe/` and is not
      committed.
* [ ] Completed Trellis task is archived before the final commit.

## Definition of Done

* Compiler structure remains C++/MLIR/TableGen/CMake/lit.
* Python remains evidence/artifact parsing and fixture tooling only.
* `tcrv.exec` stays compute-free.
* RVV interpretation remains plugin-local in C++.
* Scalar fallback remains available when RVV evidence is missing or invalid and
  scalar fallback capability is present.
* No executable lowering, RVV runtime correctness, generated object, or
  performance claim is made.

## Technical Approach

Implement `scripts/rvv_probe_to_mlir.py` as a bounded translator from sanitized
probe `capability_facts` into existing `tcrv.exec.capability` syntax. Keep
validation focused on schema shape, bounded/redacted fact strings, and required
facts needed by the existing C++ RVV plugin. Then wire deterministic lit tests
that pipe fixture JSON through the helper into `tcrv-opt`, proving that the
existing C++ plugin/pipeline consumes the emitted MLIR.

## Out of Scope

* No new core dialect compute ops or high-level tensor/tile IR.
* No Python implementation of capability relations, plugin decisions, variant
  selection, legality, lowering, emission, or runtime behavior.
* No LLVM/RISC-V/RVV lowering, inline asm emission, runtime ABI glue, object
  generation, benchmarks, correctness claims, or performance claims.
* No hard-coded RVV/IME/Sophgo/AME branches in core transforms/support.

## Technical Notes

* Initial repo inspection confirmed clean worktree and `HEAD = 0a93ee8 feat:
  add boundary-aware emission planning diagnostics`.
* Existing C++ RVV plugin consumes `rvv`, `rvv.hart_count`,
  `rvv.probe.compile_run`, and optional `rvv.toolchain.march` capability
  properties from `TargetCapabilitySet`.
* Existing public planning pipeline is
  `--tcrv-execution-planning-pipeline`, composing plugin variant materialization,
  capability checks, selection, selected lowering-boundary materialization, and
  emission-plan diagnostics.
