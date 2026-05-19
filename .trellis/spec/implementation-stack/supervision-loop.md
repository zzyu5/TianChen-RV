# Supervision Loop Contract

This spec defines durable invariants for the Hermes-reviewed Codex supervision loop. It is a policy contract, not a step-by-step operating script.

## Scope

The supervision loop may use Python for:

- launching Codex worker turns;
- packaging run evidence;
- prompting Hermes review;
- rendering next prompts;
- recording loop artifacts;
- sending optional notifications.

The supervision loop must not implement TianChen-RV compiler internals. Core IR, dialects, operations, types, attributes, passes, plugin registries, capability model objects, variant pipelines, lowering, and emission remain in the primary C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck stack.

## Executable Contract

### 1. Scope / Trigger

This contract applies when changing supervisor runner behavior, Hermes review prompt text, Codex worker base prompts, prompt rendering, no-exec packaging, or loop evidence packaging.

The trigger is infrastructure-level prompt orchestration. The prompt contract crosses runner, Hermes, Codex, Trellis specs, local repository state, and generated artifacts, so it must be documented with concrete signatures and validation behavior.

### 2. Signatures

The runner must preserve these command surfaces unless a task explicitly authorizes a CLI change:

```bash
python3 scripts/codex_serial_supervisor.py prompt --repo <repo> --artifact-root <path> --run-id <id>
python3 scripts/codex_serial_supervisor.py run --repo <repo> --artifact-root <path> --run-id <id> --no-exec
python3 scripts/codex_serial_supervisor.py loop --repo <repo> --artifact-root <path> [--review-no-llm] [--hermes-review-mode chat|oneshot] [--resume-review-run-dir <run-dir>]
python3 scripts/codex_serial_supervisor.py status --repo <repo> --artifact-root <path>
python3 scripts/codex_serial_supervisor.py stop --repo <repo> --artifact-root <path> --reason <text>
python3 scripts/codex_serial_supervisor.py ask-hermes --repo <repo> --artifact-root <path> --question <text>
```

The embedded Hermes review prompt is produced by:

```text
build_review_prompt(run_dir, loop_dir, base_prompt, round_index, previous_delta, previous_prompt_mode, max_chars, manual_steering="") -> str
```

### 3. Contracts

The Codex worker prompt input is Markdown. The canonical Codex base prompt is a
short execution prompt, not a full architecture manual. It must state the
repository root, single-worker execution, real-repo inspection, current Trellis
task handling, the one-round Trellis flow, stack discipline, short red lines,
validation expectations, commit expectations, and final report requirements.
Detailed architecture constraints are injected through the current Trellis task,
Hermes Direction Brief, and relevant `.trellis/spec/` files.

Hermes review output is strict JSON. It must contain exactly these keys:

```json
{
  "continue": true,
  "next_prompt": "Hermes Direction Brief appended under the base prompt; required when continue is true",
  "base_prompt_edits": [],
  "delta": "",
  "reason": "brief audit conclusion and why the next prompt is shaped this way",
  "telegram_note": "very short optional user-facing note"
}
```

When `continue=true`, `next_prompt` must be a non-empty Hermes Direction Brief.
The worker turns that brief into or repairs a Trellis PRD before implementation.
The runner prepends the canonical Codex base prompt to that brief. The runner
may keep legacy `base_prompt_edits` and `delta` handling for fallback
compatibility, but the normal path is base prompt plus Hermes-selected
direction brief.

Manual steering is a first-class supervisor input. The runner may read a durable
manual steering file, defaulting to the supervisor artifact root, and include it
in every Hermes review prompt. This durable steering shapes the next owner and
next prompt unless it conflicts with repository evidence, user safety, or
architecture invariants. It is not itself proof of repository state.

The runner may also read a one-shot manual steering file, defaulting to
`manual_steering_once.md` under the supervisor artifact root. One-shot steering
is included in the next official Hermes review prompt, then archived under the
loop artifact directory and cleared only after Hermes returns parseable strict
JSON and the review/next-prompt artifacts are written. If Hermes times out,
exits non-zero, returns malformed JSON, or returns `continue=true` without a
non-empty `next_prompt`, the one-shot steering must remain for the next
official review. Ask-only Hermes self-checks may inspect durable steering, but
they must not consume one-shot steering intended for the next official review.

The runner may resume the latest saved Hermes chat session across loop restarts. A user-provided `--hermes-session-id` remains the strongest selector. Auto-resume must be visible in loop, start, and status artifacts and must have an explicit disable path.

Hermes chat access must avoid concurrent writes to the same session. Official review and ask-only self-check commands should share a supervisor-local Hermes session lock.

`ask-hermes` is a read-only self-check command. It packages current repository evidence, optional durable steering, and a user question for Hermes without launching Codex and without writing a next worker prompt. It may write ask artifacts under the supervisor artifact root.

### 4. Validation & Error Matrix

| Condition | Required behavior |
|---|---|
| Hermes returns malformed JSON | runner marks parse error and stops/fails closed after any short JSON repair attempt; it must not fall back to the base prompt |
| Hermes returns `continue=true` with empty `next_prompt` | runner must not silently treat that as a successful Direction Brief review |
| Hermes evidence conflicts with Codex final summary | Hermes must trust repository state, then `repo_audit.md`, then `review_input.md`, then Codex summary |
| Codex claims RVV runtime/correctness/performance without `ssh rvv` evidence | Hermes must redirect or block that claim in the next prompt |
| Codex implements compiler internals in Python | Hermes must redirect back to C++ / MLIR / LLVM / TableGen / CMake |
| Codex blurs hardware facts, compile-time variant config, runtime SSA/control values, or descriptor-local fixture parameters | Hermes must redirect the next prompt to restore the parameter layering contract |
| Prompt rendering fails or base prompt is empty | runner command must fail rather than launch a worker with an empty prompt |
| No-exec rendering is requested | runner must package evidence without launching Codex |
| Durable steering exists | Hermes review prompt must include it as control-plane steering, not repository proof |
| One-shot steering exists | Hermes review prompt must include it and clear it only after a successful strict-JSON official review |
| ask-only self-check is requested | runner must not launch Codex or mutate source files |
| Loop is resumed from an existing Codex run directory | runner must run official Hermes review on that run before launching another Codex worker; it must not require a human-authored initial delta |
| Codex exits with a transient model/API/stream/network failure | runner may retry the worker once regardless of whether `HEAD` or git status changed; if repository state changed, the retry must be a continuation over the live repo state and previous run artifacts, not a fresh unrelated task |
| Codex exits with a transient failure after committing and leaving the worktree clean | retry prompt must let Codex verify the landed commit/task state and report completion without creating a duplicate task or commit |

### 5. Good / Base / Bad Cases

Good: Hermes audits live file contents, notices a Python-only compiler drift,
and returns one Direction Brief assigning one C++/MLIR owner with required
checks.

Base: Hermes sees aligned progress and returns a Direction Brief that continues
or expands the current module direction without repeating the canonical base
prompt.

Bad: Hermes returns only `"continue": true` and an empty `next_prompt`, trusts the Codex summary over the checked-out files, or asks the worker to implement `tcrv.exec` as Python dictionaries.

### 6. Tests Required

Prompt and runner changes require:

- `python3 -m py_compile scripts/codex_serial_supervisor.py`;
- `python3 scripts/codex_serial_supervisor.py prompt ...` to render a worker prompt;
- `python3 scripts/codex_serial_supervisor.py run ... --no-exec` when evidence packaging or run orchestration might be affected;
- a direct or command-level render check for the Hermes review prompt when `build_review_prompt` changes;
- a status or render check showing durable and one-shot steering paths when steering handling changes;
- `git diff --check` before commit.

Compiler changes generated by later workers require the normal MLIR/C++/CMake/lit/FileCheck/RVV evidence checks defined by the relevant compiler specs.

### 7. Wrong vs Correct

Wrong:

```text
Use Hermes as an implementation agent that edits files directly.
Let Hermes continue with an empty next prompt.
Replace MLIR compiler objects with Python prompt-side structures.
Treat a local no-exec runner smoke as RVV runtime evidence.
```

Correct:

```text
Use Hermes as a read-only reviewer that emits a strict JSON next-prompt contract.
Require a non-empty Direction Brief whenever continue is true.
Keep Python limited to supervision, probing, artifact parsing, and support tooling.
Require real ssh rvv evidence before accepting RVV runtime, correctness, or performance claims.
```

## Hermes Review Invariants

Hermes reviews completed Codex worker runs. Hermes must not modify the repository.

Hermes may use read-only inspection commands when tool access is available. Read-only inspection may include `pwd`, `git status --short`, `git log`, `git show`, `find`, and text search over source, tests, specs, and docs. It must not run destructive commands or write files.

Hermes must interpret evidence in this order:

1. Real repository state and file contents from the live checkout.
2. Runner-generated `repo_audit.md`.
3. `review_input.md`, run manifest, stderr, and last Codex message.
4. Codex final summary.

If Codex's summary disagrees with repository evidence, Hermes must trust repository evidence.

## Required Hermes Audit Surface

Each review must check whether the worker preserved the TianChen-RV project boundary:

- TianChen-RV remains a capability-driven RISC-V execution layer after high-level MLIR.
- The primary engineering stack remains C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.
- Python remains limited to runner, supervisor, remote probe, artifact parsing, and helper tooling.
- `tcrv.exec` remains focused on execution organization, capability, variants, dispatch, fallback, and diagnostics.
- Concrete computation stays in TCRV extension families such as RVV, IME,
  TensorExt, Offload, scalar fallback, or future vendor/custom families.
- The current main route is extension family ops -> EmitC -> intrinsic/vendor
  builtin/runtime C/C++; descriptor-driven computation and descriptor-driven
  C/source export are implementation debt, not architecture.
- RVV routes are provider-built lowering payloads over selected typed
  `tcrv_rvv` bodies, not decorators over old `i32_*` op names, route ids,
  descriptors, or artifact labels. Common EmitC/export materializes the
  provider route and must not select RVV semantics itself.
- Capability objects participate in compiler decisions rather than appearing only as comments or strings.
- RVV hardware facts, compile-time variant config, runtime SSA/control values,
  and descriptor-local bounded parameters remain explicitly separated.
- Extension-specific behavior stays plugin-local through registries and interfaces.
- RVV correctness, runtime, or performance claims use real `ssh rvv` evidence.
- Sophgo-style accelerator work is treated as runtime-offload capability, not a custom RISC-V ISA extension.
- IME remains a later extension plugin path until hardware and toolchain evidence exists.
- AME and future custom ISA paths remain future plugin slots until real target facts exist.

Hermes must redirect the next worker if a run implements compiler internals in Python, adds generic compute operations to `tcrv.exec`, hard-codes concrete extensions in core passes, claims RVV progress without `ssh rvv` evidence, delivers only metadata/status/report work when active compiler structure is needed, or confuses VLEN/vlenb as runtime values, SEW/LMUL as hardware facts only, AVL/vl as capabilities, `setvl`/`with_vl` as modeled without real IR/ABI surfaces, `element_count` as shape/AVL, or grows `required_march` string-comparison dependence when structured capabilities or properties are available.

## Next Codex Prompt Contract

Hermes must return strict JSON with exactly these keys:

```json
{
  "continue": true,
  "next_prompt": "Hermes Direction Brief appended under the base prompt; required when continue is true",
  "base_prompt_edits": [],
  "delta": "",
  "reason": "brief audit conclusion and why the next prompt is shaped this way",
  "telegram_note": "very short optional user-facing note"
}
```

When `continue=true`, `next_prompt` must be a non-empty Hermes Direction Brief
for the next worker turn. It must not be empty and must not depend on an
unstated delta. Hermes must not repeat the whole canonical base prompt inside
`next_prompt`; the runner composes the short base prompt with the direction
brief, and Codex creates or repairs the Trellis PRD before implementation.

Hermes is the planner/reviewer. It chooses one coherent engineering owner and
does not ask Codex to choose among several candidates. Internally Hermes should
decide whether to continue the current module, expand the current module, or
switch modules because the current one has converged or stalled. Good owners
include:

- CMake and MLIR project integration;
- capability model;
- `tcrv.exec` ODS contract;
- plugin registry interfaces;
- RVV Stage 1 route-authority replacement and RVV extension-family surface;
- variant generation, legality, and selection;
- lowering and emission diagnostics;
- `ssh rvv` evidence-producing RVV probe;
- offload runtime boundary.

Smoke tests, tiny guardrails, small harnesses, broad negative fixture matrices,
extra evidence packaging, export wrappers, registries, and dashboard-like
reports are not default engineering owners. They consume loop tokens and review
attention, so Hermes must skip them unless the review can name a concrete
necessity. A necessity exists only when the work directly verifies a real
lowering or runtime semantic change in the same round, prevents a specific
observed regression in an already implemented path, or is the single blocker to
the next real compiler implementation step. If that standard is not met, Hermes
must choose the next real compiler implementation owner instead.

For migration and architecture cleanup tasks, Hermes must also ask whether the
production/default path actually changed. Adding a replacement helper,
metadata, coverage, evidence, or wrapper next to the old path is not sufficient
when the old path remains the source of compute semantics. If a round builds a
replacement path, the next owner should normally switch the default path to it,
delete or bypass obsolete code, or make the replacement route the production
route.

Deletion and rewrite are allowed during architecture cleanup. Hermes must not
reward preserving descriptor-driven tests when the active task is to remove
descriptor authority from a default path. For RVV migration work, Hermes should
prefer extension family ops as source of truth, common EmitC route use,
production/default path rewiring, and deletion of obsolete
descriptor-driven behavior over finite-family coverage, descriptor checks,
helper-only tests, smoke tests, or standalone ssh-evidence rounds.

## Wrong Logic Cleanup And RVV Stage Gates

Hermes and Codex must still support deletion-only cleanup when a live task or
human steering explicitly names a wrong-logic deletion owner. That mode is a
bounded cleanup tool, not the standing state of the project and not a separate
runner state machine.

The current RVV-first workflow is no longer the old deletion-only campaign.
Current Stage 1 is RVV route-authority replacement: remove, rewrite, replace,
or fail-close active paths that still treat bounded `i32m1` arithmetic,
source-front-door patterns, route ids, artifact names, descriptor residue,
intrinsic spellings, or common/export code as RVV authority. A Stage 1 owner
may therefore build the corrected typed `tcrv_rvv` body authority, RVV-owned
legality/realization hook, or route-builder consumption path while deleting or
fail-closing the old authority. It must not preserve old logic through
compatibility wrappers. This Stage 1 realization wording means boundary/hook
authority and faithful selected-body consumption only; performance-sensitive
selected-body realization and tuning belong to Stage 2 RVV completion.
Adding another `tcrv_rvv.i32_*` helper, route-table case, intrinsic wrapper, or
source-pattern recognizer is not route-authority replacement. Stage 1 must
delete or fail-close that legacy authority instead of converting it into a
retained executable compatibility route. If a proposed owner would add names
such as `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`,
`tcrv_rvv.i32_macc`, or analogous i32/LMUL clones, Hermes must treat it as
drift and redirect to deleting/fail-closing the legacy i32 route surface first.

Deletion-only cleanup remains valid for residue that is not yet part of a
replacement owner. In that mode, Hermes must not choose finite RVV feature
expansion, helper/wrapper work, compatibility preservation, descriptor tests,
or "delete and rebuild everything" mixed work. If removal breaks build or
tests, Codex records the breakage as a missing new-architecture gap and does
not restore the wrong path to make checks pass.

Campaign deletion targets are:

- descriptor-driven computation, including `lowering_descriptor`, descriptor
  microkernel selection, descriptor intrinsic or artifact route authority,
  descriptor C generation, descriptor-as-plugin/template/RAG basis, finite
  descriptor family production paths, `i32-vadd-microkernel.v1`-style
  descriptor semantics, old RVV binary descriptor/family-registry names,
  descriptor-based dispatch, and descriptor-to-C exporters;
- direct C semantic exporters that synthesize compute bodies from metadata,
  descriptors, selected routes, or family registries instead of consuming
  extension family ops and EmitC ops;
- durable wording that describes RVV, IME, TensorExt, Offload, scalar
  fallback, or future vendor targets as independent backends rather than TCRV
  extension families;
- core orchestration branches that know extension-specific compute semantics,
  including RVV intrinsic names, microkernel descriptor semantics, scalar loop
  semantics, offload runtime call semantics, TensorExt/IME fragment semantics,
  or `if RVV` / `if IME` / `if TensorExt` compute branches;
- Python compiler-core implementations of IR semantics, lowering, plugin
  registry, codegen, route selection, or capability model behavior.

Allowed campaign owners are:

- Descriptor Erasure Owner;
- Direct C Semantic Exporter Erasure Owner;
- Independent Backend/Dialect Cleanup Owner;
- Core Semantic Branch Erasure Owner;
- Legacy Tests and Artifact Cleanup Owner.

Deletion-only cleanup is not the correct-architecture implementation phase.
When a task is explicitly deletion-only, Hermes must not ask Codex to implement
new general RVV lowering, common lower-to-EmitC pass, executable plugin
template, TensorExt/IME extension, new EmitC route, new capability model
features, or a performance/evidence matrix in the same round. When the task is
RVV Stage 1 route-authority replacement, however, this deletion-only restriction
does not block the bounded replacement work needed to make typed `tcrv_rvv`
body authority real.

Hermes may exit the campaign only when live repository evidence shows:

- descriptor is no longer compute semantics;
- direct C exporter is no longer a semantic route;
- core passes no longer contain extension-specific compute branches;
- specs, prompts, docs, comments, and tests do not describe extension families
  as independent backends;
- legacy tests no longer protect old paths;
- remaining failures are missing new architecture, not old-path compatibility.

Only after deletion-only cleanup exits may Hermes choose unrelated rebuild
owners. Only after RVV Stage 1 evidence shows no active compiler path uses
`i32m1` or source/artifact/route metadata as RVV authority, and no active RVV
route provider is still organized around `RVVI32M1*` specs/slices, finite
`i32_*` route cases, route ids, or exact `__riscv_*_i32m1` spellings as the
family architecture, may Hermes enter Stage 2 coverage/performance work.
There should be no retained executable i32 compatibility route in Stage 1. Any
future i32 behavior must be reintroduced only after the corrected typed
vector-level route surface exists, and not by keeping the old route table or
dtype-prefixed op namespace alive.

### Grill Consensus And Mature Path Steering

Human grill/consensus notes under `artifacts/` are control-plane interpretation
notes. They may help Hermes understand human intent, but they are not source of
truth, acceptance evidence, Trellis task state, or replacement specification.
Durable rules from those notes must be promoted into `.trellis/spec/` or the
canonical supervisor prompt before they steer future rounds.

A maturity-path note such as "RVV is the first executable plugin path" does not
authorize skipping Stage 1. While live evidence still shows old RVV authority,
Hermes must choose a Stage 1 route-authority replacement owner or a narrowly
named deletion-only owner for residue that blocks Stage 1. It must not
interpret maturity discussion as permission to jump to high-level frontend
lowering, later plugin mainlines, global autotuning systems, dashboards, or
Stage 2 coverage before Stage 1 evidence is clean.

After campaign exit, the intended mature route is:

```text
explicit extension-family ops
-> selected-body realization when needed by the origin plugin
-> materialized common EmitC module
-> MLIR C/C++ emitter
-> intrinsic/vendor runtime ABI
-> target export validation and packaging
-> ssh rvv evidence for the RVV plugin path
```

This is a route order, not a new state machine, bundle index, artifact ledger,
dashboard, readiness marker, or checkpoint protocol. Target export validates
and packages supported emitted artifacts; it must not synthesize compute
semantics, scheduling, dtype/LMUL policy, or RVV body shape. One-shot steering
may name the next bounded owner, but it must not create durable architecture
outside the specs and committed prompt.

Stage 2 is the RVV completion stage. It expands route-supported RVV coverage on
the corrected vector-level `tcrv_rvv` surface and includes RVV plugin-local
selected-body realization for performance-sensitive bodies. It should be judged
against structured-kernel math and data-movement classes such as
elementwise/broadcast, reduction/accumulation, contraction-like
multiply-accumulate, mask/tail-safe memory movement, conversion/dtype policy,
runtime AVL/VL control, and supported movement/layout forms. It must not be
turned into per-Linalg-op frontend work, high-level kernel ops, one-intrinsic
wrappers, dtype/LMUL clone batches, global autotuning databases, dashboards, or
readiness state machines.
It also must not introduce new dtype-prefixed `tcrv_rvv.i32_*` operation
families for reductions, accumulators, multiply-accumulate, conversion, or
memory forms. Those concepts must be modeled on the corrected vector-level
value/config/body surface.

If the just-finished run added broadcast, compare/select, reduction,
conversion, dtype, LMUL, source-shape, or intrinsic cases by extending a legacy
`RVVI32M1*` route table, Hermes must treat that as a redirect case, not as a
Stage 2 milestone. The next owner must replace the route surface so dtype,
SEW, LMUL, policy, memory form, operation kind, runtime ABI use, and intrinsic
mapping are derived from typed `tcrv_rvv` body/config structure by the RVV
plugin. Common EmitC/export may only materialize and package the selected
route.

Hermes should prefer owners that are large enough to remove a real compiler
spine bottleneck in one round. A good owner may span several tightly related
surfaces, such as ODS/verifier/materialization/exporter/tests, when those
surfaces are all required for one semantic step. A round is too small when its
primary deliverable is only a helper function, one extra negative test, a
single guardrail, a registry wrapper, or evidence packaging that does not make a
new compiler behavior available. Hermes must not decompose an obvious coherent
compiler step into a sequence of helper-only or test-only rounds merely because
each micro-step can be validated independently.

The next prompt should name the compiler path or artifact that becomes more
real after the round. For example, it should say which IR boundary,
plugin-owned lowering path, capability decision, variant selection behavior, or
runtime/emission path will be advanced. If Hermes cannot state that outcome, it
should choose a different owner.

Anti-stall rule: if several recent rounds do not make an end-to-end path closer
to completion, Hermes must stop refining the same micro-surface and choose a
larger module owner. It should repair or create a module-level Trellis PRD when
the current task is too small, stale, or unclear.

The next prompt must state the Trellis task handling, module owner, why that
owner is now the bottleneck, files or directories to inspect first, the
implementation area to modify, explicit non-goals, minimal tests or evidence,
invalid work patterns for the round, unfinished-task continuation rules, final
reporting requirements, and whether a clean commit is expected.

## Codex Worker Invariants

Each Codex worker round is a single-owner Trellis engineering round. It should
execute the current Trellis task brief rather than choose its own direction or
split into unrelated micro-tasks.

Worker prompts must preserve these rules:

- use the actual `/home/kingdom/phdworks/TianchenRV` checkout as the source of truth;
- inspect current repository state before editing;
- read the current Trellis task, PRD, relevant specs, and workspace journal;
- if no current task exists, create or repair a Trellis task from the Hermes
  brief before editing source files;
- follow one Trellis round: brainstorm/research, PRD repair if needed,
  implementation, check/self-repair, minimal validation, task status update,
  finish/archive when complete, and coherent commit;
- keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake;
- use Python only for tooling, probes, runners, artifact parsing, and small support scripts;
- keep `tcrv.exec` compute-free;
- keep extension behavior plugin-local;
- obey hardware-facts / compile-time-variant-config / runtime-values /
  descriptor-local-boundary layering, including bounded descriptor-local
  intrinsic parameters;
- require `ssh rvv` evidence for RVV runtime, correctness, or performance claims;
- keep tests minimal and proportional to changed compiler behavior rather than
  using broad smoke coverage as a default completion signal;
- avoid helper-only, guardrail-only, fixture-only, probe-only, wrapper-only, or
  registry-only rounds unless the prompt states the concrete necessity under
  the work-selection rule;
- prefer a bounded but meaningful compiler owner over a micro-owner: combine
  related ODS, C++ verifier/materialization, plugin/exporter integration, and
  minimal tests when that is the natural size of one semantic compiler step;
- treat generated loop artifacts as artifacts, not durable compiler source;
- avoid committing temporary prompt packs or raw run scratch data.

## Stop Conditions

Hermes should set `continue=false` only when the user requested stop, evidence is missing or corrupted enough that useful continuation is unsafe, credentials or hardware access require human intervention, or repeated failures make another automatic round likely harmful.

Otherwise Hermes should continue the loop with a complete `next_prompt` shaped by current repository evidence.
