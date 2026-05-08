# MLIR Testing Contract

## Required Test Forms

TianChen-RV MLIR compiler behavior must be tested through the standard MLIR/LLVM stack:

```text
lit
FileCheck
C++ tests where appropriate
CMake configure/build checks
```

## lit / FileCheck

Use lit/FileCheck for:

- dialect syntax;
- parser/printer round trips;
- operation verification;
- expected verifier diagnostics;
- pass pipelines;
- rewrite/lowering behavior visible in textual IR;
- capability-driven variant generation and rejection diagnostics;
- capability conflict legality diagnostics, including static variant
  rejection, guarded dispatch-case allowance, unguarded dispatch-case
  rejection, and fallback rejection without target-family-specific branches;
- public plugin-variant materialization pass behavior, including built-in
  registry proposal routing, deterministic proposal order, parseable downstream
  pass pipelines, no-built-in-registry diagnostics, plugin-local proposal
  recoverable declines, fatal malformed proposal failures, no-viable-proposals
  diagnostics, fallback preservation after extension-local evidence declines,
  and safe repeated-run behavior;
- dispatch/fallback IR structure;
- default public pass diagnostics when origin plugins are not registered.
- emission-plan diagnostic verifier behavior and default public
  materialization-pass diagnostics when origin plugins are not registered.
- public tool built-in plugin routing, including `tcrv-opt` RVV first-slice
  unsupported readiness/plan diagnostics, unknown-origin diagnostics, selected
  marker traversal, dispatch case/fallback ordering, and
  `--tcrv-disable-builtin-plugins` negative coverage for empty-registry plugin
  dialect behavior.
- generic selected lowering-boundary pass routing, RVV lowering-boundary syntax,
  scalar fallback lowering-boundary syntax, verifier diagnostics, selected
  dispatch-case materialization, scalar fallback preservation, scalar-only
  metadata-boundary behavior, missing selected-surface diagnostics, old RVV pass
  compatibility, and unsupported RVV evidence wording.
- public `--tcrv-execution-planning-pipeline` coverage through `tcrv-opt`,
  including deterministic built-in RVV/scalar proposal order, selected
  dispatch/fallback materialization, selected lowering-boundary metadata,
  boundary-linked emission-plan diagnostics, parseable pipeline output,
  fallback preservation after recoverable RVV proposal declines,
  no-viable-proposal diagnostics, pre-existing mismatched variant diagnostics,
  deterministic rerun diagnostics, and fatal invalid selected RVV metadata.
- boundary-aware emission planning diagnostics, including missing selected
  boundary, selected-boundary origin mismatch, selected-variant mismatch, stale
  boundary after selection changes, duplicate competing boundaries,
  required-capability mismatch, and no diagnostic append after failed
  materialization.
- plugin-owned runtime ABI/emission ownership metadata on materialized emission
  plans, including bounded runtime ABI kind/name, runtime glue role, required
  capability refs, unsupported RVV metadata boundaries, scalar metadata-only
  fallback routes, and malformed plugin-returned metadata rejection.
- target/export emission handoff manifests, including public tool coverage,
  deterministic symbol/path ordering, scalar fallback supported C source routes
  when the finite descriptor materializes a matching
  `tcrv_scalar.i32_vadd_microkernel`, unsupported/deferred RVV emission
  metadata, pipeline-to-export coverage, and negative tests proving invalid
  selected surfaces or malformed runtime ABI ownership metadata produce no
  manifest.
- RVV smoke-probe C target export, including pipeline-to-export coverage,
  deterministic multi-kernel ordering, selected RVV kernel/variant/march
  comments, `riscv_vector.h` use, fail-before-source diagnostics for malformed
  selected RVV metadata, and no manifest/runtime-success/raw-log/performance
  claims.
- RVV i32 vector-add microkernel materialization and C target export, including
  dialect parse/verify for `tcrv_rvv.i32_vadd_microkernel`, execution-planning
  coverage proving the RVV plugin materializes the op and its structured
  `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` /
  `tcrv_rvv.i32_vadd_dataflow` body from the finite selected descriptor without
  a hand-authored input microkernel, pipeline-to-export
  coverage, selected kernel/variant/march comments, control/dataflow body
  consumption comments, `riscv_vector.h` use, a
  runtime-callable C ABI function with `const int32_t *` inputs, `int32_t *`
  output, and `size_t` length, RVV i32 load/add/store intrinsics in that
  function, default library-style source without an embedded `main`, and an
  explicit self-check harness export that calls the ABI function,
  fail-before-source diagnostics for missing selected RVV paths,
  missing/stale boundaries, missing/duplicate microkernel ops, missing or
  malformed finite descriptor metadata, malformed selected march metadata,
  malformed structured control/dataflow bodies, exporter-visible
  setvl/with_vl-policy/dataflow mismatches, and no manifest/runtime-success/
  raw-log/performance numbers.
- RVV microkernel emission-plan and emission-manifest handoff,
  including supported runtime-callable C source export metadata only when a
  selected RVV path has exactly one matching
  `tcrv_rvv.i32_vadd_microkernel`, manifest serialization of the bounded
  handoff fields, stale/duplicate microkernel diagnostics, and tests proving
  the microkernel attachment is not counted as a duplicate selected lowering
  boundary.
- generic target source artifact export routing, including post-planning
  selected-path/emission-plan consumption, deterministic agreement with the
  direct RVV microkernel C exporter for the checked-in explicit microkernel
  fixture, generic route coverage for the target-owned RVV+scalar dispatch
  composite exporter when a planned selected dispatch has both callable sides,
  fail-closed diagnostics for unsupported metadata-only paths, missing
  plan/route metadata, unknown route ids, unsupported artifact kinds, stale
  selected paths, missing lowering boundaries, missing microkernels, ambiguous
  multiple supported artifacts, and scalar/offload paths not being routed
  through the RVV exporter.
- execution-plan/export preflight coherence checks, including legal RVV explicit
  microkernel, scalar fallback microkernel, and offload descriptor planned paths;
  existing execution-planning pipeline output followed by the preflight pass;
  and fail-closed diagnostics for stale selected paths, selected origin
  mismatch, lowering-boundary kernel/variant/origin mismatch, emission-plan
  route/origin mismatch, missing runtime ABI ownership metadata, source versus
  descriptor artifact route spoofing, unregistered origins, and ambiguous
  supported artifact candidates.
- scalar explicit microkernel C target export through the generic artifact
  route, including dialect parse/verify for
  `tcrv_scalar.i32_vadd_microkernel`, selected scalar fallback plus matching
  `tcrv_scalar.lowering_boundary`, deterministic portable runtime-callable C
  output with `const int32_t *` inputs, `int32_t *` output, and `size_t`
  length, scalar i32 addition inside that callable function, absence of RVV
  headers/intrinsics/route claims, absence of an embedded `main` or self-check
  success marker in the default artifact, runtime ABI kind/name and runtime
  glue role metadata, execution-planning coverage proving the scalar plugin
  materializes the microkernel from
  `tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"` without a
  hand-authored input microkernel, and fail-closed diagnostics for
  missing/stale scalar boundaries, missing/stale scalar microkernels, malformed
  finite scalar descriptors, route spoofing, unsupported metadata-only scalar
  fallback paths, offload-only paths, and ambiguous supported artifacts.
- RVV + scalar host dispatch C export for the finite i32 vector-add slice,
  including one selected RVV dispatch case plus one scalar dispatch fallback,
  matching lowering boundaries, supported runtime-callable C source
  emission-plan routes for both paths, deterministic dispatcher signature from
  an IR-backed callable ABI plan rather than detached callable parameter
  metadata, explicit host-provided availability guard metadata, RVV callable
  branch, scalar fallback branch, RVV intrinsic code preservation, scalar i32
  addition preservation, bounded metadata comments, pipeline-to-dispatch-export
  coverage where the scalar fallback callable is descriptor-materialized rather
  than hand-authored, `tcrv.exec.mem_window` parse/verify and pipeline
  materialized lhs/rhs/out buffer-window roles consumed by RVV, scalar, and
  dispatch source exporters, `tcrv.exec.runtime_param` parse/verify and pipeline
  materialized runtime-element-count / dispatch-availability-guard scalar roles
  consumed by the exporters, `tcrv.exec.case runtime_guard` symbol linkage from
  the selected RVV dispatch case to the dispatch-availability runtime_param, a
  scalar callable branch link through the selected `tcrv.exec.fallback` target
  rather than detached route metadata alone, diagnostics naming both symbols
  when the scalar callable route and `tcrv.exec.fallback` target mismatch, a
  role-binding fixture where a non-default runtime `len` name and non-default
  dispatch guard name are emitted in the generated callable and dispatcher body
  through that symbol-linked parameter, an explicit self-check harness export
  that calls the generated dispatcher with both guard false and guard true, and
  fail-closed diagnostics when scalar callable fallback metadata is missing,
  unsupported, not an exact mirror of IR-backed callable ABI
  role/name/type/ownership, runtime_guard is missing/malformed/stale for the
  selected dispatch case, or required mem_window buffer roles or runtime_param
  scalar roles are missing/duplicated/inconsistently described,
  and generic `--tcrv-export-target-source-artifact` coverage that proves a
  pipeline-synthesized selected dispatch is exported through the target-owned
  composite dispatch source route rather than a single callable shortcut.
  When the bounded object export routes are present, lit coverage must also
  prove the public library-object and self-check-object routes are visible,
  preserve the source/self-check split, fail closed before object creation for
  missing or malformed selected-path/runtime ABI metadata, and, when
  local/native RVV clang support is detected, emit non-empty tool-readable
  object files without committing binary artifacts. The generic
  `--tcrv-export-target-artifact` front door must select the library-style
  dispatch object with dispatcher/callable symbols and no `main`; the explicit
  self-check object helper may define `main`.
- offload runtime descriptor target artifact export through the artifact-kind
  aware generic route, including selected offload path plus matching
  `tcrv_offload.lowering_boundary`, runtime ABI kind/name, required capability
  refs, descriptor route id, non-source artifact kind, deterministic descriptor
  output, source-only route filtering, and fail-closed diagnostics for missing
  boundaries, missing runtime ABI metadata, stale selected variants, stale
  lowering boundaries, unknown route ids, unsupported artifact kinds,
  RVV/scalar/offload route spoofing, URLs, raw credentials, and secret-like
  metadata.

Example test intent:

```text
RUN: tcrv-opt %s --verify-diagnostics --tcrv-some-pass | FileCheck %s
```

The exact tool name may evolve, but the test must exercise real MLIR tools or project MLIR tools, not Python pseudo-IR.

## C++ Tests

Use C++ tests for:

- plugin registry APIs;
- capability query helper semantics;
- plugin-local RVV capability-profile validation, including positive sanitized
  probe facts, missing vector hints, non-`riscv64` architecture, compile/run
  failure, missing clang/CMake availability, deterministic sanitized capability
  identities, and preservation of unsupported RVV emission readiness/plans;
- cost model helper logic;
- concrete first-slice plugin registration, proposal metadata, legality, and
  registry-backed selection consumption;
- built-in plugin registration helpers, including safe registration lifetime,
  lookup-visible plugin names/origins, and deterministic duplicate handling;
- registry-injected pass behavior that requires in-process mock plugins;
- non-textual MLIR interfaces;
- C++ utility behavior that is not well covered through textual IR.
- emission-plan materialization helpers that need injected mock registries,
  deterministic selected-path ordering checks, no-partial-mutation checks, and
  unsupported first-slice plugin behavior.
- selected lowering-boundary materialization helpers that need an in-process
  registry, origin routing, disabled or unknown plugin diagnostics, malformed
  selected-path rejection, plugin result mismatch checks, RVV
  dispatch-case/direct-marker coverage, scalar fallback preservation, malformed
  RVV legality rejection, and bounded unsupported emission-plan status checks.
- plugin-local scalar fallback validation, including proposal gating on an
  available `scalar.fallback` capability, materialization, legality rejection
  for missing/unavailable fallback capability, conservative cost metadata,
  generic conservative fallback role metadata, finite
  `tcrv_scalar.lowering_descriptor` and descriptor-local element-count metadata,
  descriptor-driven `tcrv_scalar.i32_vadd_microkernel` materialization, stable
  metadata-only emission-plan fields when no matching microkernel exists, and
  supported source-export readiness/plan fields only when a matching scalar
  microkernel and scalar lowering boundary are present.

## CMake Checks

Every compiler component should be included in CMake ownership:

- dialect libraries;
- pass libraries;
- plugin libraries;
- command-line tools;
- lit test suites;
- C++ unit tests where present.

At minimum, validation for compiler changes should include CMake configure/build checks or a clear report that the local MLIR/LLVM toolchain is unavailable.

## RVV Evidence

RVV runtime, correctness, and performance claims require real `ssh rvv` evidence.

Acceptable evidence examples:

```text
ssh rvv probe output showing host/arch/core/toolchain facts
remote compile/run logs from the RVV machine
remote correctness output with input/profile metadata
remote performance output with command, target profile, and selected variant
```

Not acceptable as RVV runtime evidence:

```text
local compile-only check
local markdown/spec review
Python-only smoke test
mocked probe output
unattributed artifact copied from an older run
```

## Python Tool Tests

Python tooling may have Python tests, shell smoke tests, or script-level checks. These only validate tooling. They cannot be used as the sole evidence for MLIR dialects, passes, capability decisions, lowering, or emission.

The RVV remote probe must keep a local self-test path that requires no
hardware:

```text
python3 scripts/rvv_remote_probe.py --self-test
```

`check-tianchenrv` should cover that self-test through lit so parser, schema,
and sanitizer behavior are guarded without depending on `ssh rvv`. Passing this
local self-test proves only the Python evidence tooling contract; it is not RVV
runtime, correctness, performance, lowering, or emission evidence. The self-test
must validate the sanitized `capability_facts` artifact section and redaction
behavior, but compiler-facing capability mapping remains covered by C++ plugin
tests rather than Python.

If the repository provides an RVV probe replay helper, it must also be tested as
Python tooling and as MLIR pipeline input. Required coverage:

- a sanitized probe fixture converts to parseable `tcrv.exec` MLIR capability
  ops with the field names consumed by `RVVExtensionPlugin`;
- the replayed MLIR can drive `tcrv-opt --tcrv-execution-planning-pipeline` to
  materialize RVV proposal metadata, selected lowering-boundary metadata, and
  boundary-linked emission-plan diagnostics when the required RVV facts are
  present;
- missing, malformed, failed, secret-like, or unbounded probe facts do not
  invent RVV support;
- scalar fallback remains available when an explicit `scalar.fallback`
  capability is included and RVV evidence is unavailable or declined;
- helper tests must not use raw credentials, private keys, tokens, unbounded
  command dumps, correctness claims, runtime claims, or performance claims.

If the repository exports generated RVV smoke-probe C from post-planning MLIR,
local lit tests must cover the exporter without requiring `ssh rvv`. Any remote
compile/run of that generated source must be reported separately as
hardware/toolchain evidence only, with bounded sanitized artifacts under
`artifacts/tmp/...`; it cannot be used as a TianChen-RV kernel correctness,
supported RVV emission, runtime ABI, or performance result.

If the repository exports generated C for an explicit
`tcrv_rvv.i32_vadd_microkernel`, local lit tests must cover the dialect op and
exporter without requiring `ssh rvv`, including the runtime-callable C ABI
function, absence of an embedded `main` in the default source artifact, and the
explicit harness call into that function when using the harness export. Any
remote compile/run of the harness source must be reported separately as bounded
microkernel correctness evidence only for that explicit source and selected
flags. It must not be reported as generic TianChen-RV lowering correctness,
arbitrary RVV emission support, full runtime integration, or performance
evidence.

If the repository exports a generated RVV+scalar dispatch self-check C harness,
local lit tests must cover the harness structure without requiring `ssh rvv`.
Any remote compile/run of that generated harness must be reported separately as
bounded correctness/runtime-invocation evidence for the explicit i32-vadd
dispatcher only. It must not be reported as generic high-level lowering
correctness, arbitrary RVV emission support, object generation, dynamic runtime
integration, or performance evidence.

If the repository exports a generated RVV+scalar dispatch self-check object
artifact, local object-byte tests may be conditional on a detected local/native
RVV clang setup. If that setup is unavailable, local tests should cover
fail-closed diagnostics only and the final report must name the exact missing
tool/header/flag/sysroot condition. Any runtime/correctness claim still
requires separate `ssh rvv` evidence for the bounded dispatcher self-check
source/object path and must not be widened to generic RVV lowering, dynamic
runtime integration, or performance.

If the repository provides an RVV+scalar dispatch executable evidence bridge,
it remains Python runner/evidence tooling only. Local lit coverage must exercise
the bridge without contacting `ssh rvv`, including self-test coverage,
execution-planning pipeline use, generic library dispatch source export,
explicit self-check source export, deterministic artifact layout below
`artifacts/tmp`, command-summary redaction, and failure on secret-like evidence
metadata. Passing dry-run proves only planned dispatch handoff and source
export. Any runtime/correctness claim must use real `ssh rvv` evidence where
the generated self-check dispatch source is compiled to an object, linked to an
executable, and run with the bounded success marker observed. That claim must
remain limited to the finite RVV+scalar i32-vadd dispatcher executable path and
must not be reported as generic RVV lowering, arbitrary kernel support, dynamic
runtime integration, broad correctness, or performance evidence.

If the repository provides an end-to-end helper for that explicit microkernel
route, it remains Python runner/evidence tooling only. Local lit coverage must
exercise the helper without contacting `ssh rvv`, including manifest handoff
recognition, deterministic command-summary/artifact layout below
`artifacts/tmp`, missing supported-handoff failure, and secret-like metadata
redaction or rejection. Passing this helper dry-run proves only that existing
compiler tools can materialize the selected-boundary/emission-plan surface,
export the bounded manifest handoff, and export deterministic RVV C source;
the source is standalone only when explicit harness mode is requested. It is
not RVV runtime, correctness, generic lowering, runtime ABI, or performance
evidence.
