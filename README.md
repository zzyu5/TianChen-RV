# Weft-RV MLIR

Weft-RV is a **reference template for a capability-driven, extensible MLIR execution-layer software stack** — with **RISC-V quantized LLM inference as its first high-performance instance**. The headline is the *extensibility*: a reproducible way to organize the stack so that admission cost is predictable, correctness is machine-checked, and selection is attributable. On-silicon wins over hand-written shipped kernels are the *proof of the template's quality, not the goal itself*.

It is **not** a general-purpose compiler: "extensible" means the stack-organization is reproducible; the input side stops at a kernel-level interface and the load domain stays locked to ggml-style quantized inference kernels.

Concretely, it is a **capability-driven, unified RISC-V MLIR execution layer** that sits *after* high-level MLIR. It does not introduce a new high-level tensor/tile IR, and it is not "one independent backend dialect per target." Instead it models RISC-V target capabilities (ISA extensions, VLEN/uarch, toolchain, runtime/offload) as first-class, queryable MLIR objects, and uses them to drive plugin-local variant generation, legality, selection, dispatch, tuning, and lowering across a single common pipeline.

## Where everything is: `.trellis/`

**`.trellis/` is the single authority** for this project — queue, tasks, spec, issues, and report deliverables. There is no second source.

**Start at [`.trellis/spec/index.md`](.trellis/spec/index.md)** (the root map): positioning, the six-layer table, the reading order for a new contributor, and the N1/N2/N3 ↔ C1/C2/C3′ bridge.

| Layer | Answers |
|---|---|
| [canon](.trellis/spec/canon/index.md) | The **law**: may a claim stand, may a number be reported, is a cell a win. Includes the core invariants I1–I9 |
| [measurement](.trellis/spec/measurement/index.md) | **How to measure**: what counts as measurement, which board, which pipeline, against whom, where results land |
| [architecture](.trellis/spec/architecture/index.md) | **Structural law**: what each station *actually is* in code today, what it should become, what is off-limits |
| [evidence](.trellis/spec/evidence/index.md) | **Evidence map**: which artifact backs which claim; how to read a result and which readings are forbidden |
| [governance](.trellis/spec/governance/index.md) | **How to work**: decision-authority card, deferred-ruling rule, queue and briefing, hygiene, thinking guides |
| [issues](.trellis/spec/issues/index.md) | The **single issue register**: every known gap / pending ruling / debt, as `ISSUE-NNN` |

Read the root map, then the layer your task belongs to; attach the work to a Trellis task with its scope pre-registered before starting.

> **Research claims are not restated here.** The positioning red line and the three contributions C1/C2/C3′ are canonical in the root map and in [`canon/暂定-科研主张.md`](.trellis/spec/canon/暂定-科研主张.md), marked **【暂定 · 随论文侧更新 · 非定论】** (provisional, tracks the paper side, not settled). Their supporting artifacts and honest boundaries are mapped in the [evidence](.trellis/spec/evidence/index.md) layer. Performance figures are not quoted in this file: numbers live in the result tables with their run-ids, under the measurement layer's rules.

## Project spine

```text
high-level MLIR op
  -> target capability model            (capabilities as queryable MLIR objects)
  -> extension plugin proposes variants (RVV / IME / offload / scalar-fallback)
  -> capability-driven legality + selection + dispatch
  -> Gearbox: resource-aware tuning / selected-body realization
  -> plugin-built WEFTEmitCLowerableRoute -> common EmitC
  -> intrinsic / vendor builtin / runtime C/C++ -> clang -> target artifact
  -> hardware evidence when runtime/correctness/performance is claimed
```

The normative version of this spine, the station-to-code map, and the Non-Goals live in [`architecture/系统定位与边界.md`](.trellis/spec/architecture/系统定位与边界.md). Per-family status — with re-runnable predicates separating **what the code does today** from **what is designed but not yet built** — lives in [`architecture/家族现状.md`](.trellis/spec/architecture/家族现状.md). Treat those as the status source; a prose summary in a README goes stale, the predicates do not.

## Repository layout

```text
include/Weft/   ODS/TableGen + headers (dialects, capability model, plugin interfaces)
lib/            C++ implementation (dialects, passes, plugins, EmitC, target export)
  lib/Plugin/   per-family plugins: RVV, IME, Scalar, Offload, Template (reference family), ...
tools/          weft-opt, weft-translate, gates/, oracle/, bench/ (build helpers), ...
test/           lit/FileCheck + C++ tests
scripts/        Python tooling: probes, runners, ssh-hardware evidence harnesses (tooling only)
schema/         capability schema data
experiments/    measurement data + artifacts (consumed by scripts; see measurement layer)
.trellis/       spec, tasks, issues, workspace — the project's single authority
_attic/         archive (git-ignored except ATTIC_INDEX.md)
```

**`docs/` and `docs/ROADMAP.md` are retired and archived** (大重构 §一.1 / §4.2.7). Their content lives in `.trellis/spec/`; the originals are in `_attic/docs/` (git-ignored — see `_attic/ATTIC_INDEX.md`). Do not read them as current, and do not create any new governance/knowledge file under `docs/`.

Two carve-outs, both deliberate:
- **`.trellis/事故档案/`** — accident casefiles (misdiagnosis / reversal / self-correction records), moved **verbatim** as recurrence-prevention assets (§4.2.7). In-repo and tracked, *not* archived. Read these before repeating an old mistake.
- **`docs/` still holds 6 files** — the sealed-Win registry + its evidence legs + the C2 ledger. Their destination is a **pending user ruling** (`ISSUE-072`, gaps **G-1 / G-3**); agents must not relocate them.

Python is restricted to tooling (probes, runners, evidence harnesses, artifact parsing). Core IR, dialects, passes, the plugin registry, the capability model, lowering, and emission are C++/MLIR/LLVM/TableGen/CMake.

## Extending the stack: add a family

The headline claim is *extensibility* — that a new capability family (a new ISA extension, matrix engine, or offload target) can be admitted through one branch-free core. The admission protocol is canonical in the **architecture** layer:

- **Integration contract + the [P-2] five-piece acceptance set + registration** — [`architecture/插件协议.md`](.trellis/spec/architecture/插件协议.md) (this is C1's implementation: interface freeze [P-1], registry, locality [F-3], the family template and its real touch-set).
- **Capability model** (fact shape [S-1]/[S-2], relations, verifier duties) — [`architecture/能力模型.md`](.trellis/spec/architecture/能力模型.md).
- **Admission boundary** (which capabilities may be admitted at all) and per-family status — [`architecture/家族现状.md`](.trellis/spec/architecture/家族现状.md).
- **Machine-checked acceptance** — the falsifier gates [F-1..F-6]; the checkers live in `tools/gates/`, and the [evidence](.trellis/spec/evidence/index.md) layer maps each gate to its script.
- **Hard rules the core must keep** (zero family-name branching, etc.) — [`canon/核心不变量.md`](.trellis/spec/canon/核心不变量.md) I1–I9.

Reference family to copy: `lib/Plugin/Template/`.

> **Honest note on the gates:** the checker scripts exist under `tools/gates/`, but there is **no CI orchestrator in this repo** — `.github/` does not exist (`ls -d .github` → absent). "Gate is green" therefore means *someone ran the script*, not *CI enforces it*. See the [issues](.trellis/spec/issues/index.md) register for the gate-orchestration debt.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

The top-level `CMakeLists.txt` searches `/usr/lib/llvm-{20..14}` for the LLVM and MLIR CMake packages; pass `-DLLVM_DIR=/path/to/lib/cmake/llvm -DMLIR_DIR=/path/to/lib/cmake/mlir` to override. Missing LLVM/MLIR CMake packages fail configuration with an explicit diagnostic. The project must not replace MLIR compiler internals with Python data structures.

## Test

```bash
cmake --build build --target check-weft
```

In-tree lit/FileCheck + C++ tests cover dialect syntax, verification, pass behavior, plugin interfaces, route materialization, and fail-closed diagnostics. They are **compiler/toolchain evidence** — they do **not** prove hardware correctness or performance.

## Hardware evidence and measurement

RISC-V correctness / runtime / performance claims require real on-device evidence: correctness is checked **before** timing, and the baseline and the generated artifact must be built for the same named board. Local CMake / `weft-opt` / lit checks are not runtime evidence.

The board register — board identity, VLEN, which performance counters exist, and the per-board constraints that make a measurement legal or illegal — is canonical in [`measurement/板册.md`](.trellis/spec/measurement/板册.md) (SSH aliases `rvv`, `k1`, `scalar`; `rvv07` is registered-pending). **Do not infer board facts from this README** — that file is the source, and per-board rules (e.g. `k1` has no usable PMU, so no performance-counter claims; the `scalar` board must be built with vectorization explicitly off) decide whether a number may be reported at all.

> **★ Current state, stated plainly: there is no legal formal-measurement channel today.**
> The measurement layer establishes `bench <格> --board <板>` as the **only legal measurement action**, but **that runner does not exist**: `tools/bench/` contains four build-helper scripts and no timing / cross-check / row-writing logic, and the three destinations it pins (`experiments/master/`, `experiments/runs/`, `experiments/runs.log`) do not exist either. Live tables are under `experiments/active/result-tables/`. This is tracked as **`ISSUE-067`** (with re-runnable predicates in the entry) and is the single hard prerequisite before measurement restarts. Read [measurement](.trellis/spec/measurement/index.md) before attempting to produce any number — **an action that layer does not authorize is an illegal action**.

```bash
python3 scripts/rvv_remote_probe.py   # records sanitized RVV host/toolchain capability facts
```
