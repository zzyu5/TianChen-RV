# RVV Smoke Probe Target Exporter

## Goal

Add a bounded C++ target/export surface that emits a deterministic standalone
RVV smoke-probe C source from post-planning MLIR with a selected RVV path and a
matching `tcrv_rvv.lowering_boundary`. The exported source is hardware/toolchain
evidence plumbing only: it must not be described as TianChen-RV kernel lowering,
kernel correctness, runtime support, or performance evidence.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current `HEAD` is `9575e3d feat: add runtime offload plugin first slice`.
* The worktree was clean before this task was created.
* Existing target export surface is `tcrv-translate --tcrv-export-emission-manifest`
  backed by `TianChenRVTarget`.
* RVV planning already materializes `@rvv_first_slice`,
  `tcrv_rvv.required_march`, `tcrv_rvv.lowering_boundary`, and unsupported
  RVV emission-plan diagnostics.
* Existing RVV emission readiness and emission plans are explicitly unsupported;
  this task must preserve that boundary.

## Requirements

* Add a C++ exporter under an RVV-specific target package, preferred surface:
  `include/TianChenRV/Target/RVV/RVVSmokeProbe.h` and
  `lib/Target/RVV/RVVSmokeProbe.cpp`.
* Register a public `tcrv-translate` translation named
  `--tcrv-export-rvv-smoke-probe-c`.
* Parse post-planning MLIR and verify before emitting:
  * at least one `tcrv.exec.kernel` exists;
  * selected path exists and is owned by `origin = "rvv-plugin"`;
  * selected variant requires available RVV capability refs;
  * selected `tcrv_rvv.required_march` or equivalent selected march metadata is
    present, bounded, single-line, RVV-like, and secret-free;
  * matching direct `tcrv_rvv.lowering_boundary` exists for the selected kernel,
    variant, origin, role, status, and required capabilities;
  * boundary status is explicit and bounded;
  * scalar, offload-only, unknown-origin, stale-boundary, malformed metadata, and
    secret-like metadata cases fail before source emission.
* Emit deterministic tiny C using `riscv_vector.h`, `vsetvl`, load/store/add
  intrinsics, and a small correctness check internal to the smoke program.
* Generated C must label itself as an RVV hardware/toolchain smoke probe derived
  from selected RVV metadata and must not claim kernel lowering, runtime support,
  correctness evidence, or performance evidence.
* Keep RVV-specific logic target/plugin-local; do not add RVV branches to core
  orchestration, generic manifest export, or `tcrv.exec`.
* Keep Python limited to probe/artifact helper usage if needed. Do not implement
  compiler decisions in Python.

## Acceptance Criteria

* Positive lit coverage proves deterministic C source export from planned
  RVV-selected MLIR.
* Positive coverage proves `tcrv-opt --tcrv-execution-planning-pipeline` can
  pipe into `tcrv-translate --tcrv-export-rvv-smoke-probe-c`.
* Multiple selected RVV paths are handled deterministically.
* Generated output includes `riscv_vector.h`, bounded probe symbols,
  selected kernel/variant comments, selected march metadata, and no runtime,
  manifest, timestamp, credential, raw-log, absolute-path, or performance claim.
* Negative lit coverage rejects missing RVV selection, scalar/offload-only
  selection, stale boundary, missing/malformed selected march, unsafe metadata,
  and unregistered/unknown RVV-like origin.
* Normal local checks pass:
  `git diff --check`,
  `python3 scripts/rvv_remote_probe.py --self-test`,
  `python3 scripts/rvv_probe_to_mlir.py --self-test`,
  CMake configure, and `cmake --build build --target check-tianchenrv -j2`.
* If `ssh rvv` is reachable, generated C is compiled with `/usr/bin/clang` and
  selected `-march` plus appropriate ABI on the remote RVV host, run there, and
  sanitized bounded evidence is written under `artifacts/tmp/rvv_smoke_probe/`.

## Out Of Scope

* RVV kernel lowering, LLVM/RISC-V/RVV IR emission, object generation for
  TianChen-RV kernels, runtime ABI glue, benchmark harnesses, and performance
  measurement.
* Marking RVV kernel emission as supported.
* Sophgo/offload, IME, scalar, vendor, dtype, tensor shape, benchmark, or
  microarchitecture interpretation inside the smoke exporter.
* Resurrecting `predoc/` or flattening the organized test layout.

## Technical Notes

* Required specs read before implementation include architecture boundaries,
  capability contract, core dialect contract, plugin protocol and locality,
  variant pipeline, lowering/runtime contract, RVV plugin, offload runtime
  plugin, MLIR testing contract, and validation reference.
* Existing exporter validation style is in `lib/Target/EmissionManifest.cpp`;
  the RVV smoke exporter should reuse its fail-before-write discipline but stay
  RVV-target-local.
