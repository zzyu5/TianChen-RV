# MLIR Testing Contract

## Required Test Forms

TianChen-RV MLIR compiler behavior must be tested through the standard MLIR/LLVM stack:

```text
lit
FileCheck
C++ tests where appropriate
CMake configure/build checks
```

## Scenario: Deleted Direct C Exporter Test Ownership

### 1. Scope / Trigger

Use this contract for tests that touch RVV microkernel, scalar microkernel, or
RVV+scalar dispatch runtime-callable C source/header/object/self-check/bundle
routes after the direct C semantic exporter deletion campaign.

### 2. Signatures

- Removed direct options:
  `--tcrv-export-rvv-microkernel-c` and
  `--tcrv-export-rvv-scalar-dispatch-c`.
- Fail-closed diagnostic text must name the deleted direct C source exporter
  and the missing materialized MLIR EmitC module route.
- Generic target artifact front doors remain valid test commands, but they
  must report absence of a supported route for deleted direct-C artifacts.

### 3. Contracts

- Do not keep lit or script tests whose only assertion is successful direct C
  source, header, object, self-check, bundle, or e2e dry-run output from the
  old printers.
- Tests may keep MLIR planning/source fixtures only when they assert
  unsupported/deleted-route behavior or serve as parseable input to another
  fail-closed test.
- C++ tests for route registries must assert deleted route absence rather than
  positive registration.

### 4. Validation & Error Matrix

- Old emitted RVV intrinsic body expected -> delete the test or rewrite it to
  assert unsupported/no route.
- Old scalar arithmetic loop expected -> delete or rewrite to unsupported.
- Old dispatch source/header/object/bundle expected -> delete or rewrite to
  unsupported/no supported artifact route.
- Script dry-run depends on direct source artifacts -> remove that lit
  expectation until the Common EmitC rebuild supplies a new artifact authority.

### 5. Good/Base/Bad Cases

- Good: a lit test proves a selected RVV path materializes typed IR and then
  receives an unsupported emission-plan diagnostic.
- Base: RVV smoke-probe route tests may remain only as deleted-route coverage
  proving no C source is emitted.
- Bad: a test checks `runtime-callable-c-source`, a direct route id, or an
  `__riscv_` intrinsic as proof that kernel semantic source export or
  smoke-probe source export succeeded.

### 6. Tests Required

- Focused fail-closed lit coverage for RVV direct source, scalar source, and
  RVV+scalar dispatch generic front doors.
- C++ plugin/registry coverage for unsupported emission plans and absent
  target artifact route registrations.
- Full `check-tianchenrv` after deleting stale positive fixtures.

### 7. Wrong vs Correct

Wrong:

```text
RUN: tcrv-translate --old-direct-c-route | FileCheck generated C body
```

Correct:

```text
RUN: not tcrv-translate --old-direct-c-route | FileCheck deleted-route error
RUN: tcrv-opt ... | FileCheck unsupported emission-plan diagnostic
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
  deterministic symbol/path ordering, scalar fallback deleted-route
  diagnostics when legacy finite microkernel metadata is present,
  unsupported/deferred RVV emission metadata, pipeline-to-export coverage, and
  negative tests proving invalid selected surfaces or malformed runtime ABI
  ownership metadata produce no manifest.
- deleted RVV smoke-probe C target export coverage, proving
  `--tcrv-export-rvv-smoke-probe-c` is absent or fail-closed, built-in target
  artifact exporters do not publish the smoke route, historical smoke
  descriptor metadata is not a supported emission plan, and no
  `riscv_vector.h` / `__riscv_` C source is emitted.
- RVV selected microkernel descriptor materialization is deleted. Tests must
  assert that selected-boundary materialization does not auto-create
  `tcrv_rvv.i32_vadd_microkernel` or structured `setvl` / `with_vl` /
  load/arithmetic/store bodies from finite selected descriptors, and that RVV
  selected emission planning does not build callable ABI parameters or
  supported source/header/object routes from descriptor/family records.
  Dialect parse/verify for hand-authored `tcrv_rvv.*_microkernel` ops may
  remain only as syntax coverage or fail-closed input coverage until the
  rebuild supplies explicit extension-family IR plus a materialized MLIR EmitC
  module route. Tests must expect no generated RVV C source, header, object,
  self-check harness, runtime-success, raw-log, correctness, or performance
  claim from descriptor-selected RVV microkernel paths.
- RVV microkernel emission-plan and emission-manifest handoff,
  now deletion-campaign fail-closed: selected RVV paths with or without matching
  `tcrv_rvv.*_microkernel` attachments must not produce supported
  runtime-callable C source metadata, descriptor-derived ABI parameters, or
  manifest entries until the Common EmitC rebuild supplies a materialized
  source authority. Tests should cover unsupported metadata-only diagnostics,
  stale/duplicate microkernel inputs as deleted-route failures, and the fact
  that microkernel attachments are not replacement lowering boundaries.
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
  front door, deterministic metadata-only file names derived from registry
  artifact route metadata, deterministic bundle index fields for any current
  metadata-only route, preservation of existing single-artifact front doors,
  generic rejection of incoherent grouped bundle records such as missing
  component roles, missing external ABI identity, missing runtime ABI
  signature, duplicate parameter roles, mismatched runtime ABI parameter
  name/type/ownership, mismatched parameter order, mismatched runtime ABI
  metadata, or mismatched selected component paths, fail-closed diagnostics for
  missing or invalid output directories, and fail-closed behavior for deleted
  RVV/scalar/dispatch source/header/object routes or metadata-only selected
  paths without a fake complete executable bundle.
  Bundle tests must not commit generated binary artifacts or treat object
  creation as link, runtime, correctness, or performance evidence.
- plan-and-export target artifact bundle front-door coverage through
  `tcrv-translate`, including an input fixture that contains only kernel and
  capability anchors rather than hand-authored selected-path diagnostics,
  lowering-boundary metadata, or emission-plan diagnostics; fail-closed
  deleted-route coverage for RVV/scalar/dispatch executable bundles; and a
  focused fail-closed negative case proving planning failure does not print
  bundle completion or emit a complete executable bundle index. The front door
  must not first lower marked linalg/vector source into the exec ABI boundary.
- deleted core RVV source-to-exec pass coverage, proving the removed
  `--tcrv-lower-source-rvv-binary-to-exec`,
  `--tcrv-lower-linalg-rvv-binary-to-exec`,
  `--tcrv-lower-linalg-i32-binary-to-exec`,
  `--tcrv-lower-linalg-i32-vadd-to-exec`,
  `--tcrv-lower-vector-rvv-i32-vadd-to-exec`,
  `--tcrv-lower-vector-rvv-i32-vsub-to-exec`, and
  `--tcrv-lower-vector-rvv-i32-vmul-to-exec` options are absent or fail closed
  as deleted routes without creating `tcrv.exec.kernel`. Tests that only
  protected old linalg/vector source lowering success, source-body semantic
  diagnostics, or compatibility alias delegation should be deleted rather than
  kept alive as production coverage.
- execution-plan/export preflight coherence checks, including legal RVV explicit
  microkernel, scalar fallback microkernel, and unsupported Offload selected
  paths after descriptor deletion;
  existing execution-planning pipeline output followed by the preflight pass;
  and fail-closed diagnostics for stale selected paths, selected origin
  mismatch, lowering-boundary kernel/variant/origin mismatch, emission-plan
  route/origin mismatch, missing runtime ABI ownership metadata, stale or
  mismatched selected RVV capacity
  metadata, unregistered origins, and ambiguous supported artifact candidates.
  Runtime ABI role-contract coverage must include absence/fail-closed checks
  for deleted direct scalar/RVV callable source routes and deleted
  scalar/RVV/dispatch header/object helpers. Any remaining metadata-only
  route preflight must reject malformed runtime ABI role/type/name/ownership
  metadata before artifact output.
- scalar explicit microkernel runtime-callable C target export is currently
  deletion-campaign fail-closed. Tests may still cover dialect parse/verify,
  selected scalar fallback, and matching `tcrv_scalar.lowering_boundary`
  materialization, but they must not expect source/header/object/bundle bytes
  or portable scalar compute loops from `tcrv-export-scalar-*-microkernel-c`
  routes. Until a real materialized MLIR EmitC module route exists, tests must
  expect unsupported emission-plan diagnostics and generic source front-door
  failure with no generated C body.
- RVV + scalar host dispatch runtime-callable C export is currently
  deletion-campaign fail-closed. Tests may still cover selected RVV dispatch
  case, scalar dispatch fallback, runtime guard metadata, `tcrv.exec.mem_window`
  and `tcrv.exec.runtime_param` materialization, but they must not expect
  dispatcher source/header/object/bundle bytes, RVV intrinsic code, scalar
  compute loops, or self-check harnesses from direct target printers. Until the
  Common EmitC rebuild supplies materialized callable component source routes,
  tests must expect unsupported RVV/scalar emission-plan diagnostics and generic
  target artifact front-door failure with no supported source/header/object
  route.
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
  Direct RVV/scalar/dispatch object coverage must currently prove route absence
  or fail-closed behavior rather than object creation. No test should require
  public source/object/header translate commands or generic artifact front
  doors to emit runtime-callable microkernel or dispatch artifacts from the old
  direct C printer path.
  When a future bounded ABI header export is rebuilt on materialized EmitC
  source authority, lit/FileCheck coverage must prove both the direct header
  command and the generic `--tcrv-export-target-header-artifact` registry front
  door for the header-only runtime-caller surface. Negative coverage must prove
  the header exporter fails closed on
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
- Offload selected-path fail-closed coverage after descriptor deletion,
  including selected Offload path plus matching `tcrv_offload.lowering_boundary`,
  unsupported emission-plan diagnostics, absence of target artifact route
  registration, source/default/bundle front doors reporting no supported Offload
  artifact, and negative route-spoofing coverage proving Offload metadata cannot
  claim RVV or scalar routes. These tests must not restore descriptor artifacts,
  selected-plan export scopes, ABI role mirrors, or bundle outputs.

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
  generic conservative fallback role metadata, finite selected-path
  element-count metadata, bounded typed `tcrv_scalar.i32_vadd_microkernel`
  materialization, stable metadata-only emission-plan fields when no matching
  microkernel exists, and supported source-export readiness/plan fields only
  when a matching scalar microkernel and scalar lowering boundary are present.

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

The repository must not export generated RVV smoke-probe C from post-planning
MLIR. Tests should prove the old command and target artifact route are absent
or fail-closed and that no `riscv_vector.h` / `__riscv_` source text is printed.
Any future RVV hardware/toolchain smoke evidence belongs in explicit probe
tooling, not a compiler source artifact frontdoor.

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
bounded correctness/runtime-invocation evidence for the explicit finite i32
add/sub/mul dispatcher family only. Vsub tests must prove subtract semantics,
and vmul tests must prove multiply semantics; both must reject stale vadd
intrinsic, route, ABI-name, callable-stem, success marker, or `lhs + rhs`
expectations. It must not be reported as generic
high-level lowering correctness, arbitrary RVV emission support, object
generation, dynamic runtime integration, or performance evidence.

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
metadata. If the bridge supports more than one arithmetic family, local lit
coverage must use already materialized execution-plan inputs or an explicitly
future plugin-owned frontend, not deleted core linalg/vector source-to-exec
flags. Passing dry-run proves only planned dispatch handoff and source export.
Any runtime/correctness claim must use real `ssh rvv` evidence where the
generated self-check dispatch source is compiled to an object, linked to an
executable, and run with the bounded success marker observed. That claim must
remain limited to the finite family-selected RVV+scalar i32 add/sub/mul
dispatcher executable path and must not be reported as generic RVV lowering,
arbitrary kernel support, dynamic runtime integration, broad correctness, or
performance evidence.

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
runtime/correctness claim. Family-specific bundle dry-runs must verify the
selected route/component group and generated external caller arithmetic; for
vsub the caller must check `lhs - rhs`, and for vmul the caller must check
`lhs * rhs`, not stale vadd semantics. Passing bundle dry-run proves only
compiler bundle export, typed index parsing, file
discovery, and caller construction. Any bundle external ABI runtime/correctness
claim must use real `ssh rvv` evidence
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

## Scenario: Deleted RVV Direct Microkernel Evidence Bridge

### 1. Scope / Trigger

Use this scenario for the former direct RVV microkernel e2e evidence bridge.
That runner and its lit coverage were removed with the direct C semantic
exporter deletion campaign.

### 2. Signatures

There is no supported script command surface for selected metadata to direct
RVV runtime-callable C source/header/object/bundle evidence.

### 3. Contracts

- Do not restore a dry-run runner that accepts direct route ids or generated
  C source as evidence.
- New executable evidence must start from a rebuilt extension-family ops to
  materialized EmitC module path.
- Runtime/correctness claims still require separate real `ssh rvv` evidence
  over the rebuilt artifact.

### 4. Validation & Error Matrix

- A lit test depends on old direct source/header/object/bundle dry-run output
  -> delete it or rewrite it to fail closed.
- A helper validates RVV intrinsics in generated direct C source -> delete it.

### 5. Good/Base/Bad Cases

- Good: fail-closed tests prove the old direct route is unsupported and emits
  no RVV intrinsic body.
- Base: RVV smoke-probe tests remain allowed only as deleted-route command or
  route-absence coverage.
- Bad: a script treats route metadata plus selected-family fields as enough to
  produce executable C evidence.

### 6. Tests Required

- Keep focused fail-closed lit/C++ coverage for deleted route families.
- Do not require script self-tests for the removed direct e2e runner.
