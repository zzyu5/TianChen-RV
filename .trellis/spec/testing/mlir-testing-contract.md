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
- core capability identity verifier diagnostics, including rejection of
  duplicate direct capability-provider ids within one `tcrv.exec.kernel`,
  target-profile provider `id`/`kind` pairing diagnostics, and preservation of
  relation-provider lookup behavior for unique profile ids, plus explicit
  module-level `tcrv.exec.target` profile attachment through
  `tcrv.exec.kernel target = @profile`;
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
  deterministic rerun diagnostics, fatal invalid selected RVV metadata, and
  final execution-plan coherence gate behavior against the built-in target
  artifact exporter registry.
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
  `tcrv_rvv.i32_load` / `tcrv_rvv.i32_add` / `tcrv_rvv.i32_store` body from
  the finite selected descriptor without
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
  and generic target artifact coverage for the target-owned direct RVV
  microkernel object exporter when a direct selected RVV path has one supported
  callable side,
  fail-closed diagnostics for unsupported metadata-only paths, missing
  plan/route metadata, unknown route ids, unsupported artifact kinds, stale
  selected paths, missing lowering boundaries, missing microkernels, ambiguous
  multiple supported artifacts, and scalar/offload paths not being routed
  through the RVV exporter.
- target artifact bundle export routing, including an explicit output-directory
  front door, deterministic source/header/object file names derived from
  registry artifact route metadata, direct RVV microkernel bundle coverage,
  RVV+scalar dispatch composite bundle coverage that preserves dispatch route
  identity, a deterministic bundle index recording file name, artifact kind,
  route, owner, runtime ABI kind/name, component selected paths, and
  conservative evidence role, preservation of existing single-artifact
  source/header/object front doors, explicit dispatch external ABI component
  metadata (`component_group`, `component_role`, and `external_abi_name`) for
  RVV+scalar source/header/object bundle records, explicit ordered
  `runtime_abi_parameter[index]` signature metadata for the dispatch external
  ABI boundary, generic rejection of incoherent grouped bundle records such as
  missing source/header/object roles, duplicate component roles, missing
  external ABI identity, missing runtime ABI signature, duplicate parameter
  roles, mismatched runtime ABI parameter name/type/ownership, mismatched
  parameter order, mismatched runtime ABI metadata, or mismatched selected
  component paths, fail-closed diagnostics for missing or invalid output
  directories, and fail-closed behavior for unsupported or metadata-only
  selected paths without a fake complete bundle.
  Bundle tests must not commit generated binary artifacts or treat object
  creation as link, runtime, correctness, or performance evidence.
- plan-and-export target artifact bundle front-door coverage through
  `tcrv-translate`, including an input fixture that contains only kernel and
  capability anchors rather than hand-authored selected-path diagnostics,
  lowering-boundary metadata, or emission-plan diagnostics; successful
  RVV+scalar dispatch bundle index checks for selected dispatch surface,
  component_group/component_role/external_abi_name metadata, and ordered
  runtime_abi_parameter signature fields plus bounded selected-plan metadata
  self-description when RVV capacity facts are present; focused bounded linalg
  i32-vadd frontend input coverage proving the same front door first lowers the
  marked source into the exec ABI boundary and then exports a registry-derived
  bundle without hand-authored selected-path or emission metadata; and a focused
  fail-closed negative case proving planning failure does not print bundle
  completion or emit a complete bundle index.
- bounded linalg frontend lowering coverage, including a hand-written/test
  `linalg.generic` i32 vector-add wrapper that lowers through
  `--tcrv-lower-linalg-i32-vadd-to-exec` to a parseable `tcrv.exec.kernel`
  with a module target profile reference plus lhs/rhs/out `mem_window` roles
  and runtime `n` `runtime_param`; pipeline coverage proving the generated
  kernel can feed existing RVV/scalar proposal, selection, selected-boundary,
  and supported emission-plan materialization; and fail-closed negative
  coverage proving unsupported marked linalg bodies do not create an exec
  kernel. These tests must not claim generic linalg lowering, LLVM/RISC-V
  lowering, runtime success, correctness, or performance.
- execution-plan/export preflight coherence checks, including legal RVV explicit
  microkernel, scalar fallback microkernel, and offload descriptor planned paths;
  existing execution-planning pipeline output followed by the preflight pass;
  and fail-closed diagnostics for stale selected paths, selected origin
  mismatch, lowering-boundary kernel/variant/origin mismatch, emission-plan
  route/origin mismatch, missing runtime ABI ownership metadata, source versus
  descriptor artifact route spoofing, stale or mismatched selected RVV capacity
  metadata, unregistered origins, and ambiguous supported artifact candidates.
  Runtime ABI role-contract coverage must include the generic target-artifact
  front-door preflight for direct scalar/RVV callable source routes and the
  route-local composite preflight callbacks for scalar header/object helpers,
  RVV header/object helpers, and RVV+scalar dispatch source/header/object
  helpers; malformed component callable ABI role/type/name/ownership metadata
  must fail before artifact or bundle output.
- scalar explicit microkernel C target export through the generic artifact
  route, including dialect parse/verify for
  `tcrv_scalar.i32_vadd_microkernel`, selected scalar fallback plus matching
  `tcrv_scalar.lowering_boundary`, deterministic portable runtime-callable C
  output with `const int32_t *` inputs, `int32_t *` output, and `size_t`
  length, scalar i32 addition inside that callable function, absence of RVV
  headers/intrinsics/route claims, absence of an embedded `main` or self-check
  success marker in the default artifact, scalar runtime-callable header export
  with a declaration-only C prototype, scalar RISC-V relocatable object export
  from structured `rv64` / `riscv.toolchain.*` capability metadata and local
  clang, scalar source/header/object bundle index coverage, runtime ABI
  kind/name and runtime glue role metadata, execution-planning coverage proving
  the scalar plugin materializes the microkernel from
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
  consumed by the exporters, typed `tcrv.exec.case runtime_guard_required`
  metadata plus `runtime_guard` symbol linkage from the selected RVV dispatch
  case to the dispatch-availability runtime_param, a
  scalar callable branch link through the selected `tcrv.exec.fallback` target
  rather than detached route metadata alone, diagnostics naming both symbols
  when the scalar callable route and `tcrv.exec.fallback` target mismatch, a
  role-binding fixture where a non-default runtime `len` name and non-default
  dispatch guard name are emitted in the generated callable and dispatcher body
  through that symbol-linked parameter, an explicit self-check harness export
  that calls the generated dispatcher with both guard false and guard true, and
  fail-closed diagnostics when scalar callable fallback metadata is missing,
  unsupported, not an exact mirror of IR-backed callable ABI
  role/name/type/ownership, typed runtime guard requirement/linkage is
  missing/malformed/stale for the selected dispatch case, or required
  mem_window buffer roles or runtime_param scalar roles are
  missing/duplicated/inconsistently described,
  and generic `--tcrv-export-target-source-artifact` coverage that proves a
  pipeline-synthesized selected dispatch is exported through the target-owned
  composite dispatch source route rather than a single callable shortcut.
  Compiler-owned dispatch runtime-guard materialization tests must include a
  positive transform case where a selected or synthesized guarded
  `tcrv.exec.case` first carries typed `runtime_guard_required = true`, then
  receives a same-kernel `dispatch-availability-guard`
  `tcrv.exec.runtime_param` plus `runtime_guard` symbol reference, and negative
  coverage where arbitrary printable condition/guard/policy annotations alone
  do not create a runtime guard and where missing, stale, duplicate,
  wrong-kind, or conflicting runtime-guard linkage fails before a target bundle
  can report success. Fallback cases must be checked not to receive
  `runtime_guard_required` or `runtime_guard` metadata unless the core exec
  dialect contract is explicitly extended.
  Direct RVV microkernel object coverage must prove that the public
  `--tcrv-export-rvv-microkernel-object` route and the generic
  `--tcrv-export-target-artifact` front door, when local/native RVV clang
  object support is detected, emit non-empty tool-readable RISC-V relocatable
  objects exposing the runtime-callable RVV microkernel symbol and no `main`.
  When the bounded object export routes are present, lit coverage must also
  prove the public library-object and self-check-object routes are visible,
  preserve the source/self-check split, fail closed before object creation for
  missing or malformed selected-path/runtime ABI metadata, and, when
  local/native RVV clang support is detected, emit non-empty tool-readable
  object files without committing binary artifacts. The generic
  `--tcrv-export-target-artifact` front door must select the library-style
  dispatch object with dispatcher/callable symbols and no `main`; the explicit
  self-check object helper may define `main`.
  When the bounded ABI header export is present, lit/FileCheck coverage must
  prove both the direct header command and the generic
  `--tcrv-export-target-header-artifact` registry front door for the
  header-only runtime-caller surface: deterministic include guard, standard
  integer/size includes, `extern "C"` guard, and a single prototype with the
  same parameter order and C type spellings as the generated callable or
  dispatcher definition. Direct RVV microkernel header tests must cover the
  generic header front door separately from the generic source/object artifact
  front doors. Negative coverage must prove the header exporter fails closed on
  shared exec-IR boundary errors such as missing selected-case `runtime_guard`
  linkage or scalar fallback mismatch for dispatch headers, and missing/stale
  selected RVV path or callable ABI boundary errors for direct microkernel
  headers. Header tests must also prove that no callable bodies, RVV
  intrinsics, self-check helper, hidden `main`, runtime success marker,
  performance text, raw logs, credentials, or artifact paths are emitted. C++
  registry tests must prove that adding `runtime-callable-c-header` does not
  make source export choose a header and does not make the generic target
  artifact route choose the header instead of the relocatable object. If
  object-link evidence is added, it must remain bounded to compiling an
  external C caller against the generated header and linking that caller with
  the generated RISC-V relocatable library object; only real `ssh rvv` run
  evidence may support the bounded external ABI correctness claim, and it is
  not performance or generic lowering evidence.
- offload runtime descriptor target artifact export through the artifact-kind
  aware generic route, including selected offload path plus matching
  `tcrv_offload.lowering_boundary`, runtime ABI kind/name, required capability
  refs, descriptor route id, non-source artifact kind, descriptor schema
  version/kind/status, external adapter contract id, explicit non-claim
  metadata, deterministic ABI role entries for host buffers and runtime
  scalar/control parameters, deterministic descriptor output, source-only route
  filtering, a one-artifact offload descriptor bundle index with route/owner/
  runtime ABI signature/handoff kind metadata, and fail-closed diagnostics for
  missing boundaries, missing runtime ABI or handoff kind metadata, missing or
  conflicting ABI roles, stale selected variants, stale lowering boundaries,
  unknown route ids, unsupported artifact kinds, RVV/scalar/offload route
  spoofing, URLs, raw credentials, raw logs, and secret-like metadata.

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
  probe facts, vlenb-derived i32 lane capacity preservation and rejection,
  missing vector hints, non-`riscv64` architecture, compile/run failure,
  missing clang/CMake availability, deterministic sanitized capability
  identities, `rvv.hart_count` providing `target.hart_count`, and preservation
  of unsupported RVV emission readiness/plans;
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
  ops with the field names consumed by `RVVExtensionPlugin`, including optional
  vlenb bytes and derived i32 m1 lane-count capabilities when present;
- the replayed MLIR can drive `tcrv-opt --tcrv-execution-planning-pipeline` to
  materialize RVV proposal metadata, selected lowering-boundary metadata, and
  boundary-linked emission-plan diagnostics when the required RVV facts are
  present, including selected RVV capacity metadata preserved as diagnostic
  self-description rather than runtime input or performance evidence;
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
microkernel correctness evidence only for that explicit source, selected flags,
and generated runtime `n` cases. It must not be reported as generic TianChen-RV
lowering correctness, arbitrary RVV emission support, full runtime integration,
or performance
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

If that bridge also exposes a target-artifact-bundle mode, local lit coverage
must exercise the mode without contacting `ssh rvv`, including bundle export
through `tcrv-translate --tcrv-export-target-artifact-bundle`, parsing
`tianchenrv-target-artifact-bundle.index`, discovery of generated source,
header, and object file names from index metadata, required consumption of the
compiler-emitted `component_group`, `component_role`, `external_abi_name`, and
`runtime_abi_parameter[index]` fields instead of route/file-name or hardcoded C
signature heuristics, malformed or incomplete index rejection, external caller
generation from the emitted header prototype plus the emitted ordered ABI
signature, command-summary redaction, and absence of any dry-run
runtime/correctness claim. Passing bundle dry-run proves only compiler bundle
export, typed index parsing, file discovery, and caller construction. Any
bundle external ABI runtime/correctness claim must use real `ssh rvv` evidence
where only the generated source, generated header, generated object, and
generated caller are copied to the RVV host, the generated source and caller are
compiled there, the caller is linked and run against the source-built object and
the generated bundle object, and the bounded success marker is observed.

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
