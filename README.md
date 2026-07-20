# Weft-RV MLIR

Weft-RV is a capability-driven, extensible MLIR execution layer for heterogeneous RISC-V inference targets. Its current workload domain is kernel-level ggml/llama.cpp-style quantized inference.

The project is not a general-purpose tensor compiler and does not introduce a new high-level tensor/tile IR. It organizes low-level format, capability, scheduling, legality, selection and emission knowledge so new formats, target capabilities and extension families can be integrated without returning to per-format × per-board handwritten backends.

## Research direction: two pillars

### Pillar 1: a typed, capability-driven extension template

- format and mechanism changes enter typed facts and family-local plans;
- target and board changes enter canonical capability objects;
- new extension families use the plugin protocol and typed bodies;
- core/common code does not branch on family names;
- new performance knowledge does not require editing old emitters;
- correctness and locality are machine-checkable.

### Pillar 2: executable high-performance knowledge

- analytic knowledge constructs typed candidates and plans;
- legality and resource bounds are checked before selection;
- capability/context rules provide an analytic prior;
- qualified offline measurements may correct ranking only inside the legal candidate set;
- the selected result is stamped into a typed body;
- route providers and emitters mechanically realize that body;
- misses, stale data, unsupported inputs and negative results have named behavior.

The current two-pillar, six-law research framing lives in [canon/暂定-科研主张.md](.trellis/spec/canon/暂定-科研主张.md). The executable formula and selector contract lives in [architecture/变体流水线.md](.trellis/spec/architecture/变体流水线.md).

## Core design

~~~text
kernel-level input
  → plugin-local typed facts g + bounded static context ω
  → canonical capability c
  → plugin-local formula / decision provider
      · typed candidate or plan
      · legality/resource bounds
      · analytic prior
      · optional measurement key
  → bounded selector
      · qualified measured winner if still legal
      · otherwise analytic prior or named fallback
  → selected typed extension body
  → plugin route provider
  → common EmitC / target artifact
~~~

This is an incremental organization of existing Construction, Selection, Schedule, BodyRealization and EmitC code. It is not a new Formula IR, a universal expression DSL or a runtime autotuner.

## Current project assets

The repository already contains:

- RVV, IME, Scalar, Offload, Template and other plugin families;
- typed construction and extension bodies;
- five dequant mechanism plans: Nibble, Codebook, KQuant, GridLookup and Ternary;
- capability consumption for VLEN, RVV version and register-count-related decisions;
- measured selection paths for selected LMUL, SP4 and loop-order decisions;
- clean-room reuse-emitter and own-emitter integration records;
- physical no-V Scalar evidence;
- deployed ggml, representative strong-opponent and end-to-end result ledgers;
- an official bench runner, master table and run lineage directories.

These assets do not mean the project is finished. The main remaining engineering work is:

- centralize scattered g/c/ω decisions behind a small plugin-local decision contract;
- remove selector/emitter double authority;
- unify measurement schema, qualification and compiled winner views;
- complete capability fields and per-board instances;
- close remaining strong-construction gaps;
- make a second extension family use the same minimal formula/selection contract;
- continue representative strong-opponent and end-to-end performance work.

The live migration record is [formula-layer-migration/LEDGER.md](experiments/active/formula-layer-migration/LEDGER.md).

## Repository layout

~~~text
include/Weft/       ODS/TableGen and public C++ headers
lib/                dialects, passes, plugins, realization, EmitC and target export
  lib/Plugin/       RVV, IME, Scalar, Offload, Template and other families
tools/              weft tools, bench runner, cell harnesses, gates and oracles
test/               lit/FileCheck and C++ tests
scripts/            probes and support tooling
schema/             capability and measurement schemas
experiments/master/ canonical master tables
experiments/runs/   raw run artifacts keyed by run-id
experiments/runs.log append-only run ledger
.trellis/spec/      durable contracts
.trellis/tasks/     optional planning and campaign records
.trellis/事故档案/   recurrence-prevention casefiles
~~~

Root AGENTS.md has been retired. This README and [.trellis/spec/index.md](.trellis/spec/index.md) are the project entry points.

## Working with the repository

Trellis is used as a spec, issue and optional task system. It is not a mandatory GPT workflow.

For a normal change:

1. inspect git status and the current implementation;
2. read [.trellis/spec/index.md](.trellis/spec/index.md);
3. read only the relevant canon/architecture/measurement files;
4. implement the smallest coherent change;
5. verify correctness and relevant gates;
6. update spec or issues when the stable contract or known boundary changed.

Create a Trellis task when it is useful for:

- multi-stage or multi-day work;
- multiple agents/worktrees;
- cross-layer interface migrations;
- formal measurement campaigns;
- user-requested task-tree tracking.

Do not create a task merely to obtain permission for a small, reversible, clearly scoped change.

Governance details are in [governance/index.md](.trellis/spec/governance/index.md).

## Build

~~~bash
cmake -S . -B build -G Ninja
cmake --build build
~~~

The top-level CMake configuration searches common LLVM/MLIR installations. Override with LLVM_DIR and MLIR_DIR when needed.

## Test

~~~bash
cmake --build build --target check-weft
~~~

Tests cover dialect syntax, verification, pass behavior, plugin interfaces, selected-body realization, route materialization and fail-closed diagnostics.

When modifying emitter/verifier code, make sure the tools are actually relinked before trusting lit. When changing shared C++ struct layouts, use a clean rebuild; see [governance/思维准则.md](.trellis/spec/governance/思维准则.md).

## Extending the stack

The plugin protocol is defined in [architecture/插件协议.md](.trellis/spec/architecture/插件协议.md).

A family supplies the five-piece acceptance set:

1. capability facts/schema;
2. plugin legality;
3. typed mechanism/body and emission;
4. tests/falsifiers;
5. ledger/docs/evidence.

If the family uses analytic or measured performance knowledge, it also supplies the corresponding decision contract:

~~~text
typed g/c/ω inputs
candidate or plan
legality/resource verdict
analytic prior
reason/domain/fallback
optional measurement key
selected typed result
~~~

Reference family: lib/Plugin/Template/.

## Measurement

Official runner:

~~~bash
tools/bench/bench --help
tools/bench/bench --self-test
tools/bench/bench <op> <format> --board <board> --engine <engine> --regime <regime>
~~~

Persistent outputs:

- experiments/master/
- experiments/runs/<run-id>/
- experiments/runs.log

Cell harnesses live under tools/bench/cells/. The runner fails closed for unsupported or ambiguous combinations.

Measurement rules:

- correctness before timing;
- explicit four-part row key: op, format, engine, regime;
- real named board and toolchain lineage;
- paired baseline and generated artifact;
- raw artifacts referenced by run-id;
- no ad-hoc official numbers outside the registered pipeline.

Read [measurement/index.md](.trellis/spec/measurement/index.md) before producing reportable hardware numbers.

## Performance evidence

Weft-RV keeps three complementary views:

| Evidence | What it proves |
|---|---|
| deployed ggml path | the generated/selected path integrates into the real stack and can exploit coverage/routing differences |
| representative strong opponent | generated kernel quality against meaningful expert code |
| end-to-end | whether a kernel improvement survives dispatch, packing, memory and system overhead |

None of these replaces the others. A deployed-path win is not automatically a kernel microarchitecture win; a kernel win is not automatically an end-to-end win.

Current tables and run lineage are under experiments/master/, experiments/runs/ and experiments/active/result-tables/.

## Important boundaries

- weft.exec is an execution envelope, not a compute dialect.
- Computation belongs to typed extension-family bodies.
- Core/common paths do not branch on RVV, IME, Scalar or vendor names.
- Metadata, reason traces and artifacts are mirrors, not compute authority.
- Measurement rows cannot create candidates or bypass legality.
- Emitters do not redo formula or selector decisions.
- Python is tooling; core compiler implementation remains C++/MLIR/LLVM/TableGen.
- Runtime sparse/MoE observation is future work until a real observer, policy, overhead model and workload exist.

## Documentation map

- [spec root](.trellis/spec/index.md)
- [canon](.trellis/spec/canon/index.md)
- [architecture](.trellis/spec/architecture/index.md)
- [measurement](.trellis/spec/measurement/index.md)
- [evidence](.trellis/spec/evidence/index.md)
- [governance](.trellis/spec/governance/index.md)
- [issues](.trellis/spec/issues/index.md)
