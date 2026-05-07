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
python3 scripts/codex_serial_supervisor.py loop --repo <repo> --artifact-root <path> [--review-no-llm] [--hermes-review-mode chat|oneshot]
python3 scripts/codex_serial_supervisor.py status --repo <repo> --artifact-root <path>
python3 scripts/codex_serial_supervisor.py stop --repo <repo> --artifact-root <path> --reason <text>
python3 scripts/codex_serial_supervisor.py ask-hermes --repo <repo> --artifact-root <path> --question <text>
```

The embedded Hermes review prompt is produced by:

```text
build_review_prompt(run_dir, loop_dir, base_prompt, round_index, previous_delta, previous_prompt_mode, max_chars, manual_steering="") -> str
```

### 3. Contracts

The Codex worker prompt input is Markdown. It must state repository root, single-worker execution, stack discipline, architecture boundaries, capability model requirements, plugin-locality requirements, RVV evidence requirements, validation requirements, commit expectations, and final report requirements.

Hermes review output is strict JSON. It must contain exactly these keys:

```json
{
  "continue": true,
  "next_prompt": "full prompt for the next Codex worker turn; required when continue is true",
  "base_prompt_edits": [],
  "delta": "",
  "reason": "brief audit conclusion and why the next prompt is shaped this way",
  "telegram_note": "very short optional user-facing note"
}
```

When `continue=true`, `next_prompt` must be a complete prompt. The runner may keep legacy `base_prompt_edits` and `delta` handling for fallback compatibility, but the normal path is full-prompt rewrite through `next_prompt`.

Durable human steering is a first-class supervisor input. The runner may read a manual steering file, defaulting to the supervisor artifact root, and include it in every Hermes review prompt. This steering shapes the next owner and next prompt unless it conflicts with repository evidence, user safety, or architecture invariants. It is not itself proof of repository state.

The runner may resume the latest saved Hermes chat session across loop restarts. A user-provided `--hermes-session-id` remains the strongest selector. Auto-resume must be visible in loop, start, and status artifacts and must have an explicit disable path.

Hermes chat access must avoid concurrent writes to the same session. Official review and ask-only self-check commands should share a supervisor-local Hermes session lock.

`ask-hermes` is a read-only self-check command. It packages current repository evidence, optional durable steering, and a user question for Hermes without launching Codex and without writing a next worker prompt. It may write ask artifacts under the supervisor artifact root.

### 4. Validation & Error Matrix

| Condition | Required behavior |
|---|---|
| Hermes returns malformed JSON | runner marks parse error and uses the existing safe fallback behavior |
| Hermes returns `continue=true` with empty `next_prompt` | runner must not silently treat that as a successful full-prompt review |
| Hermes evidence conflicts with Codex final summary | Hermes must trust repository state, then `repo_audit.md`, then `review_input.md`, then Codex summary |
| Codex claims RVV runtime/correctness/performance without `ssh rvv` evidence | Hermes must redirect or block that claim in the next prompt |
| Codex implements compiler internals in Python | Hermes must redirect back to C++ / MLIR / LLVM / TableGen / CMake |
| Prompt rendering fails or base prompt is empty | runner command must fail rather than launch a worker with an empty prompt |
| No-exec rendering is requested | runner must package evidence without launching Codex |
| Durable steering exists | Hermes review prompt must include it as control-plane steering, not repository proof |
| ask-only self-check is requested | runner must not launch Codex or mutate source files |

### 5. Good / Base / Bad Cases

Good: Hermes audits live file contents, notices a Python-only compiler drift, and returns a complete next prompt assigning one C++/MLIR owner with required checks.

Base: Hermes sees aligned progress and returns a complete next prompt that remains close to the canonical base prompt while naming the next coherent engineering owner.

Bad: Hermes returns only `"continue": true` and an empty `next_prompt`, trusts the Codex summary over the checked-out files, or asks the worker to implement `tcrv.exec` as Python dictionaries.

### 6. Tests Required

Prompt and runner changes require:

- `python3 -m py_compile scripts/codex_serial_supervisor.py`;
- `python3 scripts/codex_serial_supervisor.py prompt ...` to render a worker prompt;
- `python3 scripts/codex_serial_supervisor.py run ... --no-exec` when evidence packaging or run orchestration might be affected;
- a direct or command-level render check for the Hermes review prompt when `build_review_prompt` changes;
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
Require a complete next_prompt whenever continue is true.
Keep Python limited to supervision, probing, artifact parsing, and support tooling.
Require real ssh rvv evidence before accepting RVV runtime, correctness, or performance claims.
```

## Hermes Review Invariants

Hermes reviews completed Codex worker runs. Hermes must not modify the repository.

Hermes may use read-only inspection commands when tool access is available. Read-only inspection may include `pwd`, `git status --short`, `git log`, `git show`, `find`, and text search over source, tests, specs, predoc, and docs. It must not run destructive commands or write files.

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
- Concrete computation stays in extension dialects such as `tcrv.rvv`, `tcrv.ime`, `tcrv.offload`, scalar fallback, or future plugin dialects.
- Capability objects participate in compiler decisions rather than appearing only as comments or strings.
- Extension-specific behavior stays plugin-local through registries and interfaces.
- RVV correctness, runtime, or performance claims use real `ssh rvv` evidence.
- Sophgo-style accelerator work is treated as runtime-offload capability, not a custom RISC-V ISA extension.
- IME remains a later extension plugin path until hardware and toolchain evidence exists.
- AME and future custom ISA paths remain future plugin slots until real target facts exist.

Hermes must redirect the next worker if a run implements compiler internals in Python, adds generic compute operations to `tcrv.exec`, hard-codes concrete extensions in core passes, claims RVV progress without `ssh rvv` evidence, or delivers only metadata/status/report work when active compiler structure is needed.

## Next Codex Prompt Contract

Hermes must return strict JSON with exactly these keys:

```json
{
  "continue": true,
  "next_prompt": "full prompt for the next Codex worker turn; required when continue is true",
  "base_prompt_edits": [],
  "delta": "",
  "reason": "brief audit conclusion and why the next prompt is shaped this way",
  "telegram_note": "very short optional user-facing note"
}
```

When `continue=true`, `next_prompt` must be a complete prompt for the next worker turn. It must not be empty and must not depend on an unstated delta.

The next prompt must choose one coherent engineering owner. Good owners include:

- CMake and MLIR project integration;
- capability model;
- `tcrv.exec` ODS contract;
- plugin registry interfaces;
- RVV extension first slice;
- variant generation, legality, and selection;
- lowering and emission diagnostics;
- `ssh rvv` evidence-producing RVV probe;
- offload runtime boundary.

The next prompt must state the files or directories to inspect first, the implementation area to modify, required tests or evidence, invalid work patterns for the round, final reporting requirements, and whether a clean commit is expected.

## Codex Worker Invariants

Each Codex worker round is a single-owner engineering round. It should move one coherent compiler owner forward rather than split into unrelated micro-tasks.

Worker prompts must preserve these rules:

- use the actual `/home/kingdom/phdworks/TianchenRV` checkout as the source of truth;
- inspect current repository state before editing;
- keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake;
- use Python only for tooling, probes, runners, artifact parsing, and small support scripts;
- keep `tcrv.exec` compute-free;
- keep extension behavior plugin-local;
- require `ssh rvv` evidence for RVV runtime, correctness, or performance claims;
- treat generated loop artifacts as artifacts, not durable compiler source;
- avoid committing temporary prompt packs or raw run scratch data.

## Stop Conditions

Hermes should set `continue=false` only when the user requested stop, evidence is missing or corrupted enough that useful continuation is unsafe, credentials or hardware access require human intervention, or repeated failures make another automatic round likely harmful.

Otherwise Hermes should continue the loop with a complete `next_prompt` shaped by current repository evidence.
