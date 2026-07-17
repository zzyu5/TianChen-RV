# install supervisor prompts

## Goal

Install the stricter Hermes/Codex supervision prompt workflow for TianChen-RV without implementing compiler functionality. The runner's Hermes review prompt should audit real repository evidence and produce the next full Codex prompt as strict JSON. The Codex worker base prompt should keep future worker rounds on the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck engineering path. The workspace should be clean enough to start the loop afterward.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current runner is `scripts/codex_serial_supervisor.py`; it contains `build_review_prompt`.
* Current worker base prompt is `scripts/codex_serial_supervisor_prompt.md`.
* Prompt pack exists at `tianchen_rv_supervisor_prompt_pack/`.
* The relevant prompt pack files are `03_hermes_review_prompt_template.md` and `04_codex_worker_base_prompt.md`.
* Existing specs already enforce the primary stack, `tcrv.exec` boundary, capability model, plugin locality, RVV evidence, offload boundary, and validation policy.
* This task is supervisor/prompt/spec work only. It must not implement TianChen-RV dialects, passes, lowering, or compiler functionality.

## Requirements

* Update the Hermes review prompt embedded in `scripts/codex_serial_supervisor.py` to match the intent of `tianchen_rv_supervisor_prompt_pack/03_hermes_review_prompt_template.md`.
* Ensure Hermes treats real repo state and file contents as highest-priority evidence, with `repo_audit.md` and `review_input.md` as supporting evidence.
* Permit Hermes to use read-only inspection commands when tool access is available, while forbidding repository modifications.
* Ensure Hermes audits MLIR/C++/TableGen/CMake progress, `tcrv.exec` boundary, capability model, plugin-locality, RVV evidence, offload boundary, IME status, and AME evidence requirements.
* Ensure Hermes returns strict JSON with keys `continue`, `next_prompt`, `base_prompt_edits`, `delta`, `reason`, and `telegram_note`.
* Ensure Hermes provides a complete `next_prompt` whenever `continue=true` and chooses one coherent next engineering owner for Codex.
* Replace `scripts/codex_serial_supervisor_prompt.md` with the content of `tianchen_rv_supervisor_prompt_pack/04_codex_worker_base_prompt.md`.
* Preserve the worker base prompt requirements around the primary C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck stack, Python-only tooling boundary, `tcrv.exec` as execution/capability/variant core, extension dialect ownership of concrete computation, plugin-local behavior, capability-driven compiler decisions, `ssh rvv` evidence for RVV claims, Sophgo/offload as runtime-offload, IME as later plugin, and AME only with real evidence.
* Update or create a durable supervision-loop spec under `.trellis/spec/` that records loop invariants without becoming a step-by-step script.
* Remove the temporary prompt pack directory once its content is installed, so the workspace is ready for starting the loop.

## Acceptance Criteria

* [ ] `scripts/codex_serial_supervisor.py` contains the stricter Hermes review prompt and still parses/compiles as Python.
* [ ] `scripts/codex_serial_supervisor_prompt.md` matches the intended worker base prompt content.
* [ ] `.trellis/spec/` has a durable supervision-loop spec covering Hermes evidence review, single-owner Codex rounds, MLIR/C++/TableGen/CMake stack discipline, Python boundary, and RVV `ssh rvv` evidence requirements.
* [ ] Prompt pack directory `tianchen_rv_supervisor_prompt_pack/` is removed after installation.
* [ ] Relevant checks are run, including `python3 -m py_compile scripts/codex_serial_supervisor.py` and an available prompt-render/no-exec style runner command if supported.
* [ ] A coherent commit is created if the workflow expects commits.
* [ ] Final report covers changed runner files, base prompt files, Trellis specs, Hermes behavior change, Codex worker behavior change, checks, unresolved risks, worktree cleanliness, and commit status.

## Out of Scope

* No TianChen-RV compiler-core implementation.
* No dialect, pass, lowering, or runtime implementation.
* No new Python compiler-core modules.
* No replacement of MLIR with JSON/Python structures.
* No removal of useful runner evidence packaging.
* No breaking existing runner CLI options without a strong reason.

## Technical Notes

* Initial inspection commands requested by the user were run: `pwd`, `git status --short`, `git log --oneline -8`, repository file listing, and grep for prompt/supervisor terms.
* Initial `git status --short` showed `?? tianchen_rv_supervisor_prompt_pack/`.
* `scripts/codex_serial_supervisor.py` currently embeds the Hermes prompt in a Python f-string, so literal JSON braces must be escaped.
* `scripts/codex_serial_supervisor.py` already packages `review_input.md`, `repo_audit.md`, manifest, last message, stderr tail, and loop history for Hermes.
