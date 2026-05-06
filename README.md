# TianChen-RV MLIR

TianChen-RV MLIR is a capability-driven execution layer for extensible RISC-V AI kernels after high-level MLIR. It does not introduce a new high-level tensor or tile IR; it organizes target capabilities, extension plugins, execution variants, legality, dispatch, lowering, and fallback.

## Project Spine

```text
High-level MLIR op
  -> target capability model
  -> extension plugin registry
  -> plugin-proposed execution variants
  -> legality verification
  -> capability-aware variant selection / dispatch
  -> plugin-owned lowering / emission / runtime glue
  -> RVV / IME / offload / fallback executable path
```

## Current Bootstrap

This repository is bootstrapped as a conventional MLIR project:

```text
include/TianChenRV/
lib/
tools/tcrv-opt/
test/
cmake/
CMakeLists.txt
```

The first compiler slice defines a minimal `tcrv.exec.*` operation family through TableGen/ODS and C++ registration. MLIR registers the concrete namespace as `tcrv` so operations parse and print as `tcrv.exec.kernel`, `tcrv.exec.variant`, and related core execution ops. The family is intentionally limited to execution organization concepts such as kernels, targets, capabilities, variants, dispatch, and fallback. Concrete computation belongs in extension dialects such as `tcrv.rvv`, `tcrv.ime`, `tcrv.offload`, or future plugin-local dialects.

## Build

Configure with an installed LLVM/MLIR package:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
cmake --build build
```

If LLVM/MLIR CMake packages or required tools are missing, configuration fails with an explicit diagnostic. The project must not replace MLIR compiler internals with Python data structures.

## Test

Run lit/FileCheck tests after building:

```bash
cmake --build build --target check-tianchenrv
```

Python is used only for support tooling such as lit configuration, runners, probes, supervision scripts, and artifact parsing. Core IR, dialects, operations, passes, plugin registries, capability models, lowering, and emission belong in C++ / MLIR / LLVM / TableGen / CMake.

## Hardware Evidence

The current real hardware mainline is RVV 1.0 via `ssh rvv`. Any RVV correctness, runtime, or performance claim must include real `ssh rvv` evidence. Local CMake, `tcrv-opt`, or lit checks are compiler/toolchain evidence only.
