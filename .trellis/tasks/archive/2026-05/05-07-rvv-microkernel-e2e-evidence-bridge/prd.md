# RVV microkernel end-to-end evidence bridge

## Goal

Add a bounded, reproducible evidence helper for the existing explicit RVV i32
vector-add microkernel route. The helper must drive the current MLIR/C++
compiler tools from a checked-in microkernel fixture through lowering-boundary
and emission-plan materialization, emission-manifest recognition, deterministic
C source export, and optional real `ssh rvv` compile/run self-check evidence.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD before this task is `ce522c5 feat: add RVV microkernel emission manifest handoff`.
* Worktree was clean at takeover, `predoc/` was absent, and transform tests are
  already organized into readable subdirectories.
* The compiler spine is high-level MLIR op to capability model, plugin registry,
  plugin variants, legality, capability-aware selection/dispatch, plugin-owned
  lowering/emission/runtime glue, then RVV/IME/offload/fallback executable path.
* Python is allowed here only as runner/probe/artifact orchestration. Compiler
  IR, dialects, operations, passes, plugin registry decisions, capability model,
  lowering, emission, and runtime ABI must remain C++/MLIR/TableGen/CMake/lit.
* Existing toolchain surfaces include `tcrv-opt`,
  `tcrv-translate --tcrv-export-emission-manifest`, and
  `tcrv-translate --tcrv-export-rvv-microkernel-c`.
* Existing fixture coverage includes explicit
  `tcrv_rvv.i32_vadd_microkernel`, supported RVV microkernel emission plans,
  and manifest serialization of the bounded source-export handoff.

## Requirements

* Add a script helper under `scripts/` that starts from a checked-in MLIR input
  containing the explicit `tcrv_rvv.i32_vadd_microkernel` and matching selected
  RVV path.
* The helper must run existing compiler tools, not reimplement compiler
  decisions:
  * `tcrv-opt` pass pipeline to materialize selected lowering boundaries and
    emission plans.
  * `tcrv-translate --tcrv-export-emission-manifest`.
  * `tcrv-translate --tcrv-export-rvv-microkernel-c`.
* The helper must fail closed unless the manifest identifies the bounded
  supported RVV microkernel source-export handoff.
* The helper must write generated source, hashes, command summaries, exit
  codes, stdout, stderr, and optional remote evidence only under
  `artifacts/tmp/...`.
* The helper must support a local no-ssh/dry-run/self-test mode suitable for lit
  coverage that validates command construction, manifest parsing, sanitization,
  and artifact layout without contacting the remote host.
* The helper must redact or reject secret-like strings and must not write
  credentials, environment dumps, private keys, raw SSH config, tokens, or proxy
  URLs into artifacts.
* Optional real `ssh rvv` mode must compile generated source with selected
  `rv64gcv`/`lp64d` style flags, run the self-check, and require stdout marker
  `tcrv_rvv_microkernel_ok`.
* Do not add generic RVV lowering, arbitrary RVV export, object/linking
  pipeline, performance benchmark, generic runtime ABI, or new compute ops.

## Acceptance Criteria

* [x] Local dry-run lit coverage covers manifest handoff recognition, command
  summary generation without secrets, artifact paths nested under
  `artifacts/tmp`, missing handoff failure, and secret-like input rejection or
  redaction.
* [x] Existing RVV dialect/export/readiness/manifest/plugin/selection/dispatch/
  fallback/offload tests remain green under `check-tianchenrv`.
* [x] `git diff --check` passes.
* [x] CMake configure to `artifacts/tmp/tianchenrv-build` succeeds.
* [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  succeeds or an exact toolchain blocker is recorded.
* [x] The helper is run in local dry-run mode and the command is recorded.
* [x] If `ssh rvv` is reachable, helper real mode records sanitized compile/run
  evidence; otherwise no new RVV runtime/correctness/performance claim is made.
* [x] Trellis task is finished, archived, validated, and included in the final
  coherent commit.

## Out of Scope

* Python compiler internals or Python data structures standing in for MLIR/C++
  decisions.
* Generic RVV lowering, arbitrary RVV kernel emission, runtime ABI integration,
  object/linking pipeline, bufferization, or performance measurement.
* Broad workspace cleanup, `predoc/` changes, stale prompt/design packs, or
  moving tracked tests into `artifacts/`.

## Technical Notes

* Required takeover reads covered `AGENTS.md`, `README.md`,
  `.trellis/spec/index.md`, RVV plugin, lowering/runtime, plugin protocol,
  testing, validation specs, RVV microkernel exporter, emission manifest,
  RVV plugin, emission readiness, translate tool, RVV probe helpers, and the
  relevant lit fixtures.
* The existing bounded source-export handoff strings are:
  `rvv-explicit-i32-vadd-microkernel-c-source`,
  `tcrv-export-rvv-microkernel-c`,
  `rvv-i32-vadd-standalone-c-self-check.v1`,
  `rvv-standalone-c-source-export`,
  `rvv-i32-vadd-microkernel-standalone-c.v1`, and
  `standalone-self-check-main`.
* Implementation completed:
  * Added `scripts/rvv_microkernel_e2e.py` as runner/evidence tooling only.
  * Added local lit coverage in `test/Scripts/rvv-microkernel-e2e.test`.
  * Added a non-test text fixture for missing supported-handoff failure.
  * Updated README and testing spec wording for the helper and its evidence
    boundary.
* Validation completed:
  * `python3 scripts/rvv_microkernel_e2e.py --self-test`
  * `python3 scripts/rvv_microkernel_e2e.py --dry-run --run-id final-local-dry-run --overwrite`
  * `python3 scripts/rvv_microkernel_e2e.py --run-id final-ssh-rvv --overwrite --ssh-target rvv --connect-timeout 10 --timeout 60`
  * `git diff --check`
  * `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
  * `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
* `ssh rvv` evidence scope: bounded explicit generated
  `tcrv_rvv.i32_vadd_microkernel` i32 vector-add standalone C self-check only.
  Artifact root: `artifacts/tmp/rvv_microkernel_e2e/final-ssh-rvv/`; source
  SHA-256 `04856f3ad00285fc6efd4c171883b0151d1fa67fad54394534734285e70df589`;
  remote binary SHA-256
  `9027d4548f91651632b4433e5ecdd87dbeef2b1c05e0b19ad6fbb4ff270a18e7`;
  compile exit 0; run exit 0; expected stdout marker
  `tcrv_rvv_microkernel_ok elements=16`.
