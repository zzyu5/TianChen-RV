# TianChen-RV MLIR

TianChen-RV is an MLIR-based compiler layer for extensible RISC-V execution
paths. The current contribution focus for this branch is bounded RVV slice work:
typed low-level `tcrv_rvv` IR, RVV plugin-owned legality and route derivation,
common EmitC materialization, and RVV intrinsic C/C++ output.

This branch is intentionally contributor-facing. It removes internal automation
and steering artifacts and keeps the compiler tree itself as the working
surface.

## Current RVV Path

The accepted RVV contribution path is:

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv body
  -> RVV plugin legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV intrinsic C/C++ or equivalent backend representation
  -> optional local RVV QEMU proof for the contributed slice
```

`tcrv.exec` owns kernel organization, variants, dispatch/fallback, capability
scope, diagnostics, and ABI role declarations. RVV computation belongs in the
typed `tcrv_rvv` body and the RVV plugin route provider.

Do not add new legacy helper families such as `tcrv_rvv.i32_*`, route-id-driven
semantics, source-front-door paths, or common EmitC branches that choose RVV
types/intrinsics.

## Repository Layout

```text
include/TianChenRV/        Public headers, TableGen dialect/op definitions
lib/                       Compiler implementation
tools/tcrv-opt/            Optimization/pass driver
tools/tcrv-translate/      Translation/export driver
test/                      lit/FileCheck and C++ tests
scripts/                   Support/evidence helpers
docs/                      Contributor documentation
assignments/               RVV slice module backlog for external contributors
examples/qemu/             Optional local QEMU proof fragment
```

## Main Contributor Docs

- [RVV Slice Contribution Guide](docs/rvv-slice-contribution-guide.md)
- [RVV Slice Module Backlog](assignments/rvv-slices-25.md)
- [Build And RVV Proof](docs/build-and-rvv-proof.md)

## Build

Configure with an installed LLVM/MLIR package:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
cmake --build build
```

Run the project tests:

```bash
cmake --build build --target check-tianchenrv
```

See [Build And RVV Proof](docs/build-and-rvv-proof.md) for a minimal command
sequence that materializes one existing RVV fixture into RVV intrinsic C++.

## Contribution Rule

Each RVV PR should own one bounded slice. A useful slice normally includes:

- typed `tcrv_rvv` IR or a constrained extension of an existing generic op;
- verifier/negative coverage;
- RVV plugin route derivation from typed body/config/runtime/capability facts;
- common EmitC output checks;
- optional but preferred local RVV QEMU correctness proof.

Keep changes narrow. Do not introduce internal automation files, source-front
doors, frontend projects, or broad runtime infrastructure as part of a slice.
