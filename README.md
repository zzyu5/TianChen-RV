# TianChen-RV MLIR

TianChen-RV is a **capability-driven, unified RISC-V MLIR execution layer** that sits *after* high-level MLIR. It does not introduce a new high-level tensor/tile IR, and it is not "one independent backend dialect per target." Instead it models RISC-V target capabilities (ISA extensions, VLEN/uarch, toolchain, runtime/offload) as first-class, queryable MLIR objects, and uses them to drive plugin-local variant generation, legality, selection, dispatch, tuning, and lowering across a single common pipeline.

The long-term spec lives in [`.trellis/spec/`](.trellis/spec/index.md); read [`spec/index.md`](.trellis/spec/index.md) before changing design or code.

## Project spine

```text
high-level MLIR op
  -> target capability model            (capabilities as queryable MLIR objects)
  -> extension plugin proposes variants (RVV / IME / offload / scalar-fallback)
  -> capability-driven legality + selection + dispatch
  -> Gearbox: resource-aware tuning / selected-body realization
  -> plugin-built TCRVEmitCLowerableRoute -> common EmitC
  -> intrinsic / vendor builtin / runtime C/C++ -> clang -> target artifact
  -> ssh rvv evidence when runtime/correctness/performance is claimed
```

## Research contributions (what makes this more than plumbing)

- **N1 — RISC-V extension heterogeneity as first-class capability IR.** RISC-V's à-la-carte ISA + vendor extensions + accelerators are combinatorial in a way x86/ARM are not. TianChen-RV models them as queryable capability objects with `provides`/`implies`/`conflicts` relations that drive variant legality and selection.
- **N2 — zero-core-branch plugin generalization.** Adding an extension family is local: core/common passes route through capability queries and plugin interfaces, never `if RVV` / `if IME` branches. RVV is the first family; IME (matrix) and offload are intended to prove the same common path generalizes.
- **N3 — capability/resource-aware cross-family tuning (Gearbox).** Capability objects don't only gate legality; they parameterize a resource-aware tuning space that turns a selected extension body into a tuned executable body — one mechanism reused across families.

These are claims to be *demonstrated*, not assumed. "Variant containers" and "plugins" are architecture, not contributions — MLIR already provides them.

## Current status (honest)

- **Real:** the `tcrv.exec` core dialect; a real C++ capability model (single-hop `provides`/`implies`/`conflicts`, queried by passes); a substantial typed `tcrv_rvv` vector dialect with verifiers; the RVV plugin (legality, selected-body realization, route provider); and a **live end-to-end RVV path** — `tcrv.exec` → typed `tcrv_rvv` body → Gearbox → MLIR EmitC → C/C++ → clang cross-compile → RISC-V object → `ssh rvv`. A packed-i4 dequant-contraction kernel has been generated, run on RVV hardware, and checked correct against a scalar oracle.
- **Open / weak:** measured RVV speedups are currently **at or below scalar** — performance is the open N3 problem, not a solved feature. The capability model is real but shallow (no transitive closure / provider ranking yet). The Gearbox enumerates only a small fixed candidate set; making it genuinely resource-aware is in progress.
- **Stub / future:** only RVV is a real family. `tcrv_scalar` is a reserved namespace with no active op; `tcrv_offload` is a single fail-closed handoff marker; IME does not exist yet. There is no high-level frontend (linalg→tcrv) — current input is hand-written TianChen-RV MLIR.

## Repository layout

```text
include/TianChenRV/   ODS/TableGen + headers (dialects, capability model, plugin interfaces)
lib/                  C++ implementation (dialects, passes, plugins, EmitC, target export)
tools/                tcrv-opt, tcrv-translate
test/                 lit/FileCheck + C++ tests
scripts/              Python tooling: probes, runners, ssh-rvv evidence harnesses (tooling only)
.trellis/             project spec, tasks, and developer workspace
```

## Build

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
cmake --build build
```

Missing LLVM/MLIR CMake packages or tools fail configuration with an explicit diagnostic. The project must not replace MLIR compiler internals with Python data structures.

## Test

```bash
cmake --build build --target check-tianchenrv
```

In-tree lit/FileCheck + C++ tests cover dialect syntax, verification, pass behavior, plugin interfaces, route materialization, and fail-closed diagnostics. They are compiler/toolchain evidence — they do **not** prove hardware correctness or performance.

## Hardware evidence

The real hardware mainline is RVV 1.0 via `ssh rvv`. Any RVV correctness, runtime, or performance claim requires real `ssh rvv` evidence (correctness checked before timing; baseline and generated artifact on the same named target). Local CMake / `tcrv-opt` / lit checks are not runtime evidence.

```bash
python3 scripts/rvv_remote_probe.py   # records sanitized RVV host/toolchain capability facts
```

Python is restricted to tooling (probes, runners, evidence harnesses, artifact parsing). Core IR, dialects, passes, the plugin registry, the capability model, lowering, and emission are C++/MLIR/LLVM/TableGen/CMake.
