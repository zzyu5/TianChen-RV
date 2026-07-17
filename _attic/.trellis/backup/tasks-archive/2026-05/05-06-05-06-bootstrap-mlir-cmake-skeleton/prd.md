# bootstrap MLIR CMake project skeleton

## Goal

Create the first reviewable TianChen-RV compiler skeleton in the required C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck stack. This round should make the repository recognizably buildable as an MLIR project, with explicit local toolchain diagnostics when MLIR/LLVM tools are missing, and with a minimal `tcrv.exec` dialect contract smoke test.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current git branch is `main`; the worktree was clean before this task began.
* There is no root `README.md`, no root `CMakeLists.txt`, and no existing `include/`, `lib/`, `tools/`, `test/`, `tests/`, or `cmake/` compiler layout.
* `.trellis/.current-task` did not exist before this task; an older supervisor prompt task exists but is not current and its commit already appears in recent git history.
* Specs require TianChen-RV to stay a capability-driven RISC-V execution layer after high-level MLIR, not a Python or high-level compute IR.
* `tcrv.exec` must own execution organization, capabilities, variants, dispatch, and fallback; concrete computation must remain in extension dialects/plugins.
* RVV runtime/performance/correctness claims require real `ssh rvv` evidence; this bootstrap task should not make RVV runtime claims.

## Requirements

* Add a conventional MLIR project skeleton:
  * root `CMakeLists.txt`;
  * `cmake/` support modules;
  * `include/TianChenRV/` public headers and TableGen declarations;
  * `lib/` C++ implementation;
  * `tools/tcrv-opt/` driver;
  * `test/` lit/FileCheck tests.
* Define a minimal `tcrv.exec` dialect with ODS/TableGen as the first core dialect slice.
* Keep the dialect focused on execution/capability/variant organization; do not add high-level compute ops such as matmul, softmax, reduce, or generic tile ops.
* Provide enough C++ registration code for `tcrv-opt` to load the dialect when MLIR is available.
* Add CMake diagnostics that explicitly report missing MLIR/LLVM packages or missing testing tools rather than silently replacing the compiler with Python data structures.
* Add lit/FileCheck coverage for minimal syntax/parsing of the `tcrv.exec` dialect when the MLIR toolchain is available.
* Add a short `README.md` describing the project spine, build command, test command, and the boundary that Python is only for support tooling.
* Update durable Trellis specs only if implementation reveals a new invariant. Do not put task sequencing into specs.

## Acceptance Criteria

* [ ] `cmake -S . -B <build-dir>` either configures against local MLIR/LLVM or fails with an explicit diagnostic naming the missing package/tool.
* [ ] The repository contains a conventional MLIR/C++/TableGen/CMake/lit layout.
* [ ] `tcrv.exec` is introduced via ODS/TableGen and C++ dialect registration.
* [ ] The first test uses lit/FileCheck against `tcrv-opt`, not a Python compiler-core substitute.
* [ ] No concrete computation op is added to `tcrv.exec`.
* [ ] Local validation commands are run and recorded, including CMake configure behavior.
* [ ] No `ssh rvv` runtime claim is made unless real `ssh rvv` evidence is collected.
* [ ] A coherent commit is created if the round completes cleanly.

## Out of Scope

* No RVV/IME/offload lowering implementation.
* No plugin registry implementation beyond what is necessary for this first dialect/project skeleton.
* No Python representation of compiler IR, dialects, operations, passes, registries, capability model, lowering, or emission.
* No runtime/performance/correctness claim for RVV without real hardware evidence.
* No high-level compute dialect or generic compute op in `tcrv.exec`.

## Technical Notes

* Initial inspection commands requested by the worker prompt were run before compiler edits: `pwd`, `git status --short`, `git log --oneline -8`, and repository file listing.
* `include/`, `lib/`, `tools/`, `test/`, `tests/`, and `cmake/` were absent at the start of the task.
* The highest-value coherent owner for an empty compiler repo is CMake + MLIR project integration because it enables later capability model, dialect, plugin, lowering, and test work to land in the required stack.
