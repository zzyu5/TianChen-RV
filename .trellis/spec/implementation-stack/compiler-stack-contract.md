# Compiler Stack Contract

## Primary Stack

TianChen-RV MLIR is a real MLIR compiler project. The primary implementation stack is:

```text
C++
MLIR
LLVM
TableGen / ODS
CMake
lit / FileCheck
```

This stack is mandatory for:

- core IR and dialect implementation;
- `tcrv.exec` operation definitions and verifiers;
- TCRV extension families such as RVV, IME, TensorExt, Offload, and future vendor/custom families;
- compiler passes;
- plugin registry and interfaces;
- capability model compiler objects;
- variant generation, legality, selection, dispatch, lowering, and emission;
- MLIR parser/printer/verification behavior;
- build integration and tests.

## Python Boundary

Python is allowed only for tooling:

```text
runner
supervisor
remote probe
artifact parsing
report generation
small support scripts
developer convenience wrappers
```

Python is disallowed as the implementation language for:

```text
core IR
dialects
passes
plugin registry
capability model
variant pipeline
lowering pipeline
emission pipeline
compiler legality model
compiler decision objects
```

## Wrong vs Correct

Wrong:

```text
Implement `tcrv.exec.variant` as a Python class and run legality with Python dictionaries.
Represent capabilities as JSON-only objects consumed by Python pass simulators.
Use Python to lower pseudo-IR into strings and call that the compiler pipeline.
```

Correct:

```text
Define dialect ops in TableGen / ODS.
Implement verifiers, passes, registries, and lowering in C++ using MLIR/LLVM APIs.
Use Python only to run `mlir-opt`, launch `ssh rvv` probes, parse artifacts, or supervise runs.
```

## Missing Toolchain Rule

If local MLIR tools are unavailable:

- add detection for required tools such as `mlir-opt`, `mlir-translate`, `FileCheck`, `llvm-lit`, `clang`, and `cmake`;
- emit clear diagnostics and mark MLIR behavior tests as blocked or unavailable;
- do not replace MLIR compiler internals with Python structures to keep moving.

## Build Integration

Compiler components must be buildable through CMake. New compiler libraries, dialects, passes, tools, and tests should have CMake ownership from the start, even if the first slice is small.

## Test Integration

MLIR syntax, parsing, verification, pass behavior, and diagnostics should be validated through lit / FileCheck. C++ unit tests are appropriate for registry APIs, capability model helpers, and non-textual compiler utilities.
