# Implementation Stack Specs

This layer defines the durable implementation stack for TianChen-RV MLIR.

## Pre-Development Checklist

- [ ] Is compiler functionality implemented in C++ with MLIR/LLVM APIs?
- [ ] Are dialects and operations declared through TableGen / ODS where MLIR expects it?
- [ ] Are build targets wired through CMake?
- [ ] Are MLIR behavior tests written with lit / FileCheck, with C++ tests where appropriate?
- [ ] Is Python limited to tooling, probes, runners, supervisors, artifact parsing, or small support scripts?
- [ ] If MLIR tools are missing locally, does the change add detection and diagnostics instead of replacing MLIR with Python data structures?
- [ ] Are selected-body realization, plugin route authority, capability mapping, and typed body generation implemented in C++/MLIR rather than Python compiler-core code?
- [ ] 这一步推进的是哪个 Novelty 主张（N1/N2/N3）？还是只是相邻枝节？（见 [trunk-discipline](../guides/trunk-discipline.md)）

## Guidelines Index

| Spec | Description |
|---|---|
| [Compiler Stack Contract](./compiler-stack-contract.md) | Required stack and Python boundary（= core-invariants I6）|

## Quality Check

- Core compiler objects must not be modeled as Python dictionaries/classes as the implementation of record.
- Python scripts may inspect, launch, parse, or orchestrate compiler runs; they may not become the compiler stack.
- Any fallback caused by missing local MLIR/LLVM tools must be explicit diagnostics, not an alternate Python compiler implementation.
- RVV route support must flow through typed `tcrv_rvv` bodies, plugin-owned selected-body realization/route provider, and common EmitC materialization; Python may only orchestrate or probe around that compiler path.
