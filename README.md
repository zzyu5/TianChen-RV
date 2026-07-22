# Weft-RV MLIR

Weft is an extensible, MLIR-based automatic operator-to-kernel compiler and execution-layer software stack. It accepts a semantically complete but execution-undetermined operator problem after graph-level compilation, binds a target construction family, and uses family-local capability- and context-conditioned executable knowledge to construct specialized kernels. Fragmented RISC-V quantized inference is the flagship reference realization and primary stress domain; GPU is the second execution paradigm introduced by the V2 architecture, not a currently implemented backend.

The project is not a general-purpose graph/tensor compiler and does not introduce a new high-level tensor/tile IR. It owns the post-graph, pre-schedule operator execution layer: canonical problem intake, target/family binding, construction, legality, selection, typed bodies, artifact realization, ABI/runtime integration and evidence. The design goal is an ecosystem in which new operators and targets remain local without giving up expert-quality specialization.

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
- the selected result is constructed as the final typed body;
- plugin-local realization and emitters mechanically realize that body;
- misses, stale data, unsupported inputs and negative results have named behavior.

The current two-pillar, six-law research framing lives in [canon/暂定-科研主张.md](.trellis/spec/canon/暂定-科研主张.md). The executable formula and selector contract lives in [architecture/变体流水线.md](.trellis/spec/architecture/变体流水线.md).

## Core design

~~~text
canonical operator problem P=(S,g,ω)
  → typed target/profile binding (family f, capability c_f)
  → catalogued plugin-local formula / construction
      · typed candidate or plan
      · legality/resource bounds
      · analytic prior
      · optional measurement key
  → bounded selector
      · qualified measured winner if still legal
      · otherwise analytic prior or named fallback
  → selected typed extension body
  → family-local realization / artifact driver
      · current EmitC/native object
      · future family-specific artifact
~~~

Every production operator entry follows this path, including deterministic single-candidate construction with honest-null axes. Quantize, dequantize, contraction, elementwise, reduction and different backend families do not keep separate hidden decision worlds. This is not a new Formula IR, a universal expression DSL or a runtime autotuner.

The canonical problem, family-binding and artifact-neutral contract is defined in
[architecture/执行问题与家族边界.md](.trellis/spec/architecture/执行问题与家族边界.md).

## Current project assets

The repository already contains:

- RVV, IME, Scalar, Demo, Toy, Template, TensorExtLite and Offload plugin families;
- typed construction and extension bodies;
- seven dequant mechanism formulas: Int8Scale, NibbleDecode, BinarySign,
  KQuantScaleMin, CodebookGather, GridLookup and TernaryDecode;
- capability consumption for VLEN, RVV version and register-count-related decisions;
- measured selection paths for selected LMUL and loop-order decisions, with
  deterministic singleton construction where SP4 has only one real body;
- clean-room reuse-emitter and own-emitter integration records;
- physical no-V Scalar evidence;
- deployed ggml, representative strong-opponent and end-to-end result ledgers;
- an official bench runner, master table and run lineage directories.

These assets do not mean the project is finished. The current production authority
boundary has completed its horizontal cutover: registered/direct construction entries
are enumerated by the lightweight catalog while evaluation remains in family-local
typed formulas; generic and source schedules use the same construction lifecycle;
lower-quant outputs complete legal schedules; composite realization is a real registry
owner; and obsolete Q40/GEMM compatibility passes and non-semantic decision mirrors
have been removed.

The construction boundary is now artifact-neutral. Registry clone, public
materialization, direct RVV conversion, translate and artifact export all invoke a
family-owned construction seam before the construction-blind backend registry.
`TypedBackendEmissionDriver` no longer owns a construction hook, and `emitc.func` is
only the success gate for the current EmitC artifact. RVV, IME, Scalar, Demo, Toy,
Template and TensorExtLite are construction-qualified; Offload remains explicitly
unsupported. Scalar q2/dequant and IME MAC/tile decisions are frozen into family-local
final plans before emission, while deterministic small families qualify a complete
typed body for their current mechanical artifact path. Catalog/backend inventories are
checked separately and neither is compute authority. See
[ISSUE-129 and ISSUE-131](.trellis/spec/issues/发射器与架构.md).

That lifecycle cutover is a structural prerequisite, not the end of the research
refactor. Some code-affecting knowledge and legacy route/manifest protocols still live
across leaves, front doors, schedules and conversions; `ConstructedWeak` entries have
not thereby passed delete-leaf reconstruction. The next project-wide task therefore
closes the A/B lines horizontally across the current RISC-V realization: factor
mechanisms and formulas, remove provider/replay/mirror authority, prove multi-topology
reconstruction, and re-establish current-artifact correctness and performance
causality. GPU implementation starts only after this closure and will not be
registered as another EmitC emitter or consume an RVV body/`flat_*` plan. See the
[V2 method baseline](docs/method/项目全景与Spec重构前方法基线v2.md) and the
[A/B horizontal closure task](.trellis/tasks/07-23-executable-knowledge-ab-horizontal-closure/prd.md).

For flat block-dot kernels, formula construction now produces the final `flat_*`
computation plan—body family, decode, fold, block length, activation offset, scale
source, table identity where one really exists, and bias. EmitC consumes that plan
directly and fails closed on missing, partial, unknown, or mechanism-conflicting input;
it does not recover a second decision from `kind`, `format`, or historical
`fold_model` fields. See
[formula/construction architecture](.trellis/spec/architecture/公式层与覆盖.md) and
[ISSUE-128](.trellis/spec/issues/发射器与架构.md).

This authority convergence is not strong reconstruction. Entries marked
`ConstructedWeak` still depend on complete mechanical leaves and must not be counted
as passing the delete-leaf criterion: after deleting a point implementation,
`g/c/ω + mechanisms + formula` must independently rebuild the same instance.

The current contract is the registry-derived formula catalog plus
[formula/construction architecture](.trellis/spec/architecture/公式层与覆盖.md).
The former eight-row migration ledger is retained only as
[archived investigation](experiments/archive/formula-layer-migration-2026-07-22/LEDGER.md).

## Repository layout

~~~text
include/Weft/       ODS/TableGen and public C++ headers
lib/                dialects, passes, plugins, realization, EmitC and target export
  lib/Plugin/       RVV, IME, Scalar, Demo, Toy, Template, TensorExtLite, Offload
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

1. problem applicability plus capability facts/schema;
2. plugin legality;
3. family-local construction, typed mechanism/body and artifact lowering;
4. tests/falsifiers;
5. ledger/docs/evidence.

Every production family also supplies a catalogued, artifact-neutral formula/construction contract.
A family with no choice uses a deterministic single-candidate construction and honest-null axes; it
does not bypass the stage:

~~~text
canonical problem S/g/ω + bound family capability c_f
candidate or plan
legality/resource verdict
analytic prior
reason/domain/fallback
optional measurement key
selected typed result
~~~

Reference family: lib/Plugin/Template/.

The stable formula-layer and coverage contract is in [architecture/公式层与覆盖.md](.trellis/spec/architecture/公式层与覆盖.md).
The first V2 implementation task is
[artifact-neutral family construction rebase](.trellis/tasks/07-23-artifact-neutral-family-construction-rebase/prd.md);
it deliberately does not implement GPU.

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
- Target/profile binds a construction family before formula evaluation; artifact code does not choose the family.
- Construction completion is artifact-neutral; `emitc.func` is only an EmitC artifact gate.
- GPU is a V2 architecture target, not a currently supported family or performance claim.
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
