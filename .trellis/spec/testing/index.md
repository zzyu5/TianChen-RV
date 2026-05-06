# Testing Specs

This layer defines required tests and evidence for TianChen-RV MLIR.

## Pre-Development Checklist

- [ ] Does MLIR syntax, parsing, printing, verification, or pass behavior have lit/FileCheck coverage?
- [ ] Are C++ tests added for registry, capability helper, or non-textual compiler APIs when lit/FileCheck is insufficient?
- [ ] Does CMake configure/build include the relevant compiler libraries, dialects, passes, tools, and tests?
- [ ] Does any RVV runtime/correctness/performance claim include `ssh rvv` probe or run output?
- [ ] If local MLIR tools are unavailable, is the missing toolchain reported explicitly?

## Guidelines Index

| Spec | Description |
|---|---|
| [MLIR Testing Contract](./mlir-testing-contract.md) | lit/FileCheck, C++ tests, CMake checks, RVV evidence |

## Quality Check

- Dialect syntax, parser/printer, verifier, pass rewrite, and diagnostics need lit/FileCheck tests.
- C++ tests are appropriate for compiler APIs that are not naturally visible in textual MLIR.
- Python tests may validate tooling scripts, but they do not replace MLIR behavior tests.
