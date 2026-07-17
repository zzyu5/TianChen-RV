# Supervisor extension-family construction protocol

## Goal

Update long-term plugin/protocol specs, the Hermes/Codex supervisor prompts,
and one-shot steering so the next loop moves from a plugin component checklist
to an executable Extension-Family Plugin Construction Protocol.

This is a workflow/spec round only. It must not implement compiler
functionality, add plugin code, or change lowering code.

## What I already know

* TianChen-RV is unified TCRV RISC-V MLIR.
* RVV, IME, TensorExt, Offload, scalar fallback, and future targets are
  extension families, not independent backends.
* Existing spec already has an Extension Family Plugin Template and manifest,
  but it reads mainly as a component checklist.
* The requested protocol must define the construction flow:

```text
extension archetype
  -> semantic role graph
  -> extension family declaration
  -> common interface realization
  -> EmitC route mapping
  -> evidence profile
```

* One-shot steering should point Hermes at an executable construction template
  owner for the next loop.
* The loop is currently stopped and the worktree starts clean.

## Requirements

* Update `.trellis/spec/` to define the Extension-Family Plugin Construction
  Protocol.
* Treat Extension Manifest as the machine-readable entry to the construction
  protocol, not a passive document checklist.
* Explain that "fast plugin addition" means less architecture decision search,
  not less extension-specific code.
* Preserve descriptor-driven computation as invalid architecture.
* Use RVV finite/vector binary as the first exemplar without making it the
  final upper bound.
* Update Hermes prompt checks minimally.
* Update Codex base prompt minimally.
* Write one-shot steering for the next official Hermes review:
  `Executable Extension-Family Plugin Construction Template Owner`.
* Do not create compiler functionality or modify compiler code/lowering.
* Commit the spec/prompt/task changes coherently.
* Restart the existing supervisor workflow after commit using the runner,
  one-shot steering, latest Hermes session, and normal loop settings.

## Acceptance Criteria

* [x] Spec contains `Extension-Family Plugin Construction Protocol`.
* [x] Spec contains `extension archetype`, `semantic role graph`,
      `Extension Manifest`, `common interface realization`,
      `EmitC route mapping`, and `evidence profile`.
* [x] Spec says new extensions are TCRV extension families, not independent
      backends.
* [x] Spec says quick addition reduces decision search, not required code.
* [x] Spec says descriptors cannot define new extension computation semantics.
* [x] Hermes prompt checks construction protocol, independent backend drift,
      descriptor-driven compute, core orchestration special cases, and
      executable-template progress.
* [x] Codex prompt mentions the construction protocol without becoming a long
      architecture manual.
* [x] One-shot steering exists and names
      `Executable Extension-Family Plugin Construction Template Owner`.
* [x] No compiler code is modified.
* [x] Markdown/prompt grep checks pass; Python compile passes if runner changes.
* [x] Task is archived and committed.
* [x] Loop is restarted after commit without creating a duplicate active loop.

## Out of Scope

* No new plugin implementation.
* No lowering or RVV route code changes.
* No performance or RVV hardware validation.
* No broad supervisor rewrite.
* No durable manual steering; this round uses one-shot steering.

## Technical Notes

Likely files:

* `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
* `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
* `.trellis/spec/plugin-protocol/index.md`
* `scripts/codex_serial_supervisor.py`
* `scripts/codex_serial_supervisor_prompt.md`
* `artifacts/tmp/hermes_codex_supervisor/manual_steering_once.md`

## Completion Notes

Spec changes:

* Added the Extension-Family Plugin Construction Protocol to
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`.
* Defined the construction sequence:
  `extension archetype -> semantic role graph -> extension family declaration
  -> common interface realization -> EmitC route mapping -> evidence profile`.
* Promoted Extension Manifest to the machine-readable entry point for the
  construction protocol, including archetype, semantic roles, interface
  realization, EmitC route, and evidence fields.
* Clarified that fast plugin addition reduces architecture decision search,
  not family-specific code.
* Added RVV finite/vector binary as the first exemplar without making it the
  final upper bound.
* Reaffirmed that descriptor-driven computation is outside the construction
  protocol.

Supervisor changes:

* Hermes review prompt now checks construction protocol drift, metadata-only
  template drift, and extension-specific core orchestration branches.
* Codex base prompt now points new extension work at the construction protocol
  without embedding the full spec.

One-shot steering:

```text
artifacts/tmp/hermes_codex_supervisor/manual_steering_once.md
```

Next owner:

```text
Executable Extension-Family Plugin Construction Template Owner
```

Validation:

```bash
rg -n "Extension-Family Plugin Construction Protocol|extension archetype|semantic role graph|Extension Manifest|common interface realization|EmitC route mapping|evidence profile" .trellis/spec/plugin-protocol scripts/codex_serial_supervisor.py scripts/codex_serial_supervisor_prompt.md artifacts/tmp/hermes_codex_supervisor/manual_steering_once.md
rg -n "construction protocol|not independent backend|independent backend|descriptor-driven compute|descriptor-driven computation|Executable Extension-Family Plugin Construction Template Owner" scripts/codex_serial_supervisor.py scripts/codex_serial_supervisor_prompt.md artifacts/tmp/hermes_codex_supervisor/manual_steering_once.md .trellis/spec/plugin-protocol
python3 -m py_compile scripts/codex_serial_supervisor.py
python3 scripts/codex_serial_supervisor.py run --repo /home/kingdom/phdworks/TianchenRV --artifact-root artifacts/tmp/hermes_codex_supervisor --run-id construction-protocol-noexec --no-exec --prompt-override 'Construction protocol render check'
```
