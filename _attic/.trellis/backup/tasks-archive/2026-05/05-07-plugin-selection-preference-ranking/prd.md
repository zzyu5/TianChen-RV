# plugin selection preference ranking

## Goal

Add a durable, target-neutral C++/MLIR plugin preference/ranking contract for
variant selection. Selection must ask origin plugins for generic heuristic
preference metadata after generic requires and plugin-local legality have passed,
then deterministically choose a viable variant before dispatch, lowering-boundary,
and emission planning.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD is expected to include `f719e6a feat: verify plugin variant legality before selection`.
* The previous slice added `--tcrv-verify-plugin-variant-legality` and kept core
  orchestration target-neutral.
* The current live-code gap is selection: legal variants still need a first-class
  plugin-provided preference/ranking contract instead of relying only on proposal
  order or hard-coded fallback behavior.
* The implementation stack remains C++ / MLIR / LLVM / TableGen / ODS / CMake /
  lit / FileCheck. Python is allowed only for support/probe scripts.

## Requirements

* Add or extend a C++ plugin interface for selection preference/ranking metadata.
* Keep dependency direction as tool/plugin loader -> registry -> core selection
  orchestration -> abstract plugin interface -> concrete plugin implementation.
* The core selection pass must remain target-neutral: it may consume
  `TargetCapabilitySet`, `tcrv.exec.variant` metadata, legality-verified status,
  and plugin-returned generic preference records only.
* Plugin-specific preference interpretation belongs inside the owning plugin.
* RVV should prefer its legal variant when RVV capability facts are valid.
* Scalar fallback should expose a conservative fallback preference that wins only
  when no better legal plugin-specific variant is preferred.
* Preference is not legality. It must not resurrect illegal variants or bypass
  generic requires, plugin-local legality, or invalid selected-path diagnostics.
* Selection must be deterministic with an explicit target-neutral tie-break order.
* Diagnostics should use generic terms: kernel symbol, variant symbol, origin
  plugin, preference rank/cost/category, fallback role, and tie-break reason.
* `tcrv.exec` must remain compute-free and focused on execution/capability/
  variant/dispatch/fallback/diagnostic metadata.
* Public `tcrv-opt` should use the deterministic built-in registry at the tool
  boundary. Empty/default registry construction must remain an honest diagnostic
  surface.

## Acceptance Criteria

* Positive coverage:
  * Valid RVV replay capabilities plus scalar fallback capability selects RVV
    because RVV plugin preference is stronger.
  * RVV unavailable, declined, or not materialized selects scalar fallback when
    scalar fallback capability is present.
  * Scalar fallback-only fixture still selects scalar fallback and continues
    through dispatch/lowering-boundary/emission metadata.
  * Equal preference metadata has stable deterministic tie-breaking.
  * The composed execution-planning pipeline uses plugin preference after
    plugin-variant legality and before dispatch/lowering-boundary/emission.
  * Public `tcrv-opt` built-in registry behavior works for preference selection.
* Negative coverage:
  * Preference cannot select a variant that failed generic capability requires or
    plugin-local legality.
  * Unknown/unregistered variant origin fails before or at preference collection
    with a clear generic diagnostic.
  * Malformed plugin preference results are rejected or normalized
    deterministically without crashing.
  * Missing preference hook/no-preference plugin falls back to documented
    target-neutral tie-break behavior.
  * Invalid RVV selected variants still fail before downstream materialization.
  * No dispatch/lowering-boundary/emission diagnostics are produced when no legal
    viable selection exists.

## Required Checks

* `git diff --check`
* `python3 scripts/rvv_remote_probe.py --self-test`
* `python3 scripts/rvv_probe_to_mlir.py --self-test`
* `cmake -S . -B build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
* `cmake --build build --target check-tianchenrv -j2`

## Out Of Scope

* No Python implementation of compiler internals.
* No target-specific branches in core transforms/support.
* No lowering, emission artifacts, runtime ABI, executable generation,
  correctness runs, benchmarking, or performance measurement.
* No new RVV hardware/toolchain evidence claim and no `ssh rvv` unless an
  intentional hardware/runtime/correctness/performance claim is made.
