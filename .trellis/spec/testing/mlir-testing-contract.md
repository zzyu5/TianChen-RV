# MLIR Testing Contract

## Required Test Forms

TianChen-RV MLIR compiler behavior must be tested through the standard MLIR/LLVM stack:

```text
lit
FileCheck
C++ tests where appropriate
CMake configure/build checks
```

## Scenario: Current Target Artifact Export Test Ownership

### 1. Scope / Trigger

Use this contract for tests that touch RVV microkernel, scalar microkernel, or
RVV+scalar dispatch source/header/object/self-check/bundle surfaces while no
materialized MLIR EmitC route exists.

### 2. Signatures

- Removed direct options include the historical RVV microkernel, self-check,
  scalar microkernel, and RVV+scalar dispatch C-output helpers.
- Fail-closed diagnostic text must explain that no materialized MLIR EmitC
  module route is available.
- Generic target artifact front doors remain valid test commands, but they
  must report absence of a currently supported route.

### 3. Contracts

- Do not keep lit or script tests whose only assertion is successful direct C
  source, header, object, self-check, bundle, or e2e dry-run output from the
  old printers.
- Tests may keep MLIR planning/source fixtures only when they assert current
  unsupported behavior or serve as parseable input to another fail-closed test.
- C++ tests for route registries must assert current generic registry behavior
  rather than old route-family absence.

### 4. Validation & Error Matrix

- Old emitted RVV intrinsic body expected -> delete the test or rewrite it to
  assert unsupported/no route.
- Old scalar arithmetic loop expected -> delete or rewrite to unsupported.
- Old dispatch source/header/object/bundle expected -> delete or rewrite to
  unsupported/no supported artifact route.
- Script dry-run depends on direct generated-source outputs -> remove that lit
  expectation until the Common EmitC rebuild supplies a new artifact authority.

### 5. Good/Base/Bad Cases

- Good: a lit test proves a selected RVV path materializes typed IR and then
  receives an unsupported emission-plan diagnostic.
- Base: RVV smoke-probe tests may remain only when they exercise selected RVV
  metadata through current unsupported emission-plan diagnostics without naming
  an old route-name fixture.
- Bad: a test checks raw runtime-callable C output, a direct route id, or an
  `__riscv_` intrinsic as proof that kernel semantic source export or
  smoke-probe source export succeeded.

### 6. Tests Required

- Focused fail-closed lit coverage for RVV source, scalar source, and
  RVV+scalar dispatch generic front doors when no materialized EmitC route is
  available.
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
RUN: not tcrv-translate --old-direct-source-route | FileCheck missing-EmitC error
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
- public tool built-in plugin routing, including `tcrv-opt` RVV plugin
  selected-body realization / emission-plan fail-closed diagnostics,
  unknown-origin diagnostics, selected marker traversal, dispatch
  case/fallback ordering, and
  `--tcrv-disable-builtin-plugins` negative coverage for empty-registry plugin
  dialect behavior.
- generic selected lowering-boundary pass routing through
  `--tcrv-materialize-selected-lowering-boundaries`, RVV lowering-boundary
  syntax, scalar fallback lowering-boundary syntax, verifier diagnostics,
  selected dispatch-case materialization, scalar fallback preservation,
  scalar-only metadata-boundary behavior, missing selected-surface diagnostics,
  and unsupported RVV evidence wording.
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
  capability refs, unsupported RVV metadata boundaries, scalar fail-closed
  fallback routes, and malformed plugin-returned metadata rejection.
- target/export emission handoff manifests, including public tool coverage,
  deterministic symbol/path ordering, scalar fallback deleted-route
  diagnostics when legacy finite microkernel metadata is present,
  unsupported/deferred RVV emission metadata, pipeline-to-export coverage, and
  negative tests proving invalid selected surfaces or malformed runtime ABI
  ownership metadata produce no manifest.
- deleted RVV smoke-probe C target export coverage, proving built-in target
  artifact exporters only expose the current allowed route set, historical
  smoke descriptor metadata is not a supported emission plan, and no
  `riscv_vector.h` / `__riscv_` C source is emitted.
- RVV selected microkernel descriptor materialization is deleted. Tests must
  assert that selected-boundary materialization does not auto-create deleted
  RVV wrapper ops or structured `setvl` / `with_vl` /
  load/arithmetic/store bodies from finite selected descriptor residue, and
  that RVV selected emission planning does not build callable ABI parameters or
  supported source/header/object routes from descriptor/family records.
  Dialect parse/verify coverage must target the surviving explicit
  `setvl` / `with_vl` / dataflow ops; hand-authored
  `tcrv_rvv.*_microkernel` wrapper syntax is no longer active coverage.
  Tests must expect no generated RVV C source, header, object, self-check
  harness, runtime-success, raw-log, correctness, or performance claim from
  descriptor-selected legacy RVV microkernel paths.
- RVV microkernel emission-plan and emission-manifest handoff is
  deletion-campaign fail-closed: selected RVV metadata paths must not produce
  supported runtime-callable C source metadata, descriptor-derived ABI
  parameters, or manifest entries until the Common EmitC rebuild supplies a
  materialized source authority. Tests should cover unsupported diagnostics,
  stale deleted-route inputs, and the fact that deleted wrapper attachments are
  not replacement lowering boundaries.
- generic target generated-source export routing, including post-planning
  selected-path/emission-plan consumption, fail-closed coverage for deleted RVV
  direct artifact routes, and fail-closed coverage for deleted non-RVV metadata
  artifact paths, fail-closed diagnostics for unsupported paths, missing
  plan diagnostics, unknown route ids, unsupported artifact kinds, stale
  selected paths, missing lowering boundaries, missing microkernels, ambiguous
  multiple supported artifacts, and scalar/offload paths not being routed
  through the RVV exporter.
- target artifact bundle export routing, including an explicit output-directory
  front door, deterministic file names derived from registry artifact route
  metadata, deterministic bundle index fields for any future supported route,
  preservation of existing single-artifact front doors,
  generic rejection of incoherent grouped bundle records such as missing
  component roles, missing external ABI identity, one-sided empty or otherwise
  mismatched runtime ABI signatures, duplicate parameter roles, mismatched
  runtime ABI parameter name/type/ownership, mismatched parameter order,
  mismatched runtime ABI metadata, or mismatched selected component paths,
  fail-closed diagnostics for
  missing or invalid output directories, and fail-closed behavior for obsolete
  metadata/source route inputs or unsupported selected paths without a fake
  complete executable bundle.
  A shared empty runtime ABI signature is valid coverage for zero-argument
  materialized EmitC object/header bundles only when both component records
  agree and the generated header declaration uses `(void)`.
  Bundle tests must not commit generated binary artifacts or treat object
  creation as link, runtime, correctness, or performance evidence.
  When a task claims live RVV generated-bundle ABI correctness, the evidence
  must use a bounded external ABI consumer on `ssh rvv`: generate the bundle
  through the public target artifact exporter, verify that the index ties the
  generated header and object to the same selected variant, route, component
  group, runtime ABI identity, ordered ABI parameters, materialized EmitC
  provenance, and RVV runtime AVL/VL metadata, then compile/link/run a small C
  harness against only the generated header and object. Such harnesses and
  Python runners are evidence tooling only; they must not become compiler
  semantic paths, descriptor adapters, direct C exporters, fallback compute
  implementations, artifact ledgers, or replacements for MLIR/C++ lowering.
- deleted plan-and-export target artifact bundle route coverage: active tests
  must not protect a `tcrv-translate` wrapper that runs execution planning
  before target artifact bundle export. Target bundle exporter tests should
  consume already planned/materialized MLIR. Source-level positive coverage
  may use the plugin source artifact bundle front door only when the plugin
  source-front-door registration is default-artifact-front-door eligible. For
  explicit-only legacy RVV source seeds, positive generated-bundle evidence
  must first invoke the explicit RVV materializer to produce selected typed
  `tcrv_rvv` IR, then export through the selected typed-body target artifact
  bundle route; default source-artifact front doors must remain fail-closed for
  source-only RVV input. Deletion rounds may use focused help/unknown-option
  probes to prove the wrapper is absent, but should not keep a durable named
  absence fixture as a production workflow.
- deleted core RVV source-to-exec fixture erasure, proving active tests no
  longer invoke the historical source, linalg RVV, linalg i32 compatibility, or
  vector i32 arithmetic public options as named absence fixtures. Tests that
  only protected old linalg/vector source lowering success, source-body semantic
  diagnostics, compatibility alias delegation, or historical option absence
  should be deleted rather than kept alive as production coverage.
- execution-plan/export preflight coherence checks, including legal RVV explicit
  microkernel, scalar fallback fail-closed path, and unsupported Offload
  selected paths after descriptor deletion;
  existing execution-planning pipeline output followed by the preflight pass;
  and fail-closed diagnostics for stale selected paths, selected origin
  mismatch, lowering-boundary kernel/variant/origin mismatch, emission-plan
  route/origin mismatch, missing runtime ABI ownership metadata, stale or
  mismatched selected RVV capacity
  metadata, unregistered origins, and ambiguous supported artifact candidates.
  Runtime ABI role-contract coverage must include fail-closed checks for
  missing materialized scalar/RVV callable source routes and scalar/RVV/
  dispatch header/object helpers. Any remaining no-route preflight must reject
  malformed runtime ABI role/type/name/ownership metadata before artifact
  output.
- scalar explicit microkernel runtime-callable C target export is deleted.
  Tests may still cover scalar fallback fail-closed boundary diagnostics and
  unsupported diagnostics, but they must not expect source/header/object/
  bundle bytes or portable scalar compute loops from
  `tcrv-export-scalar-*-microkernel-c` routes. Until a real materialized MLIR
  EmitC module route exists, tests must expect unsupported emission-plan
  diagnostics and generic source front-door failure with no generated C body.
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
  RVV/scalar/dispatch object coverage must currently prove generic fail-closed
  behavior rather than object creation. No test should require
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
  probe facts, raw VLENB preservation without derived lane-capacity authority,
  missing vector hints, non-`riscv64` architecture, compile/run failure,
  missing clang/CMake availability, deterministic sanitized capability
  identities, `rvv.hart_count` providing `target.hart_count`, and preservation
  of unsupported RVV selected-body realization / emission-plan fail-closed
  behavior;
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
  element-count metadata fail-closed behavior when no typed scalar body exists,
  absence of descriptorless default microkernel materialization, stable
  unsupported emission-plan fields when no matching microkernel exists, and
  deleted-route fail-closed fields only when a matching explicit scalar
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

The Python RVV probe-to-MLIR replay route is deleted. Tests must not keep a
Python helper, fixture, or lit pipeline whose purpose is to convert sanitized
RVV probe JSON into `tcrv.exec` MLIR capability, target, kernel, selected
route, or scalar fallback modeling. In particular, no active lit test may pipe
Python-generated RVV replay MLIR into `tcrv-opt` as compiler-path evidence.

Coverage for retained Python RVV tooling is limited to evidence-tool behavior:
schema shape, bounded hardware/toolchain facts, sanitizer/redaction behavior,
artifact layout, and fail-closed diagnostics. Compiler-facing RVV capability
profile validation, relation construction, legality, selection, lowering, and
emission behavior must be covered by C++/MLIR plugin tests and lit/FileCheck
compiler tests, not by Python-generated compiler IR.

The repository must not export generated RVV smoke-probe C from post-planning
MLIR. Tests should prove current unsupported RVV emission-plan diagnostics and
target artifact route shape without preserving the old command or route-name
fixtures. A bounded MLIR EmitC materialization test may contain
`emitc.include <"riscv_vector.h">` and `emitc.call_opaque "__riscv_*"`
evidence derived from explicit RVV ops, but no C/C++ source artifact, header,
object, harness, runtime success, or performance text may be printed from the
old direct exporter path.
Any future RVV hardware/toolchain smoke evidence belongs in explicit probe
tooling, not a compiler generated-source front door.

For the current bounded RVV explicit-op EmitC materialization route, local lit
tests must cover the source IR, the materialized MLIR EmitC module, and
fail-closed unsupported shapes without requiring `ssh rvv`. Any future
generated C/C++ export or remote compile/run of that source must be reported
separately as bounded route evidence only for that explicit source, selected
flags, and generated runtime `n` cases. It must not be reported as generic
TianChen-RV lowering correctness, arbitrary RVV emission support, full runtime
integration, or performance evidence.

The historical RVV+scalar dispatch self-check harness, self-check object, and
direct executable evidence bridge are deleted test surfaces. Active tests must
not require generated self-check C bodies, embedded success markers, explicit
self-check source export, self-check object bytes, route-name dry-runs, or
bundle-mode caller generation from the old direct printer path. Current
coverage must instead prove fail-closed route absence or exercise rebuilt
materialized EmitC object/header bundles that do not synthesize self-check
compute bodies. Any future runtime/correctness claim must start from a rebuilt
extension-family op to materialized EmitC route plus separate real `ssh rvv`
evidence for the concrete artifact under test.

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
- Base: RVV smoke-probe tests remain allowed only as current unsupported RVV
  emission-plan coverage without old route-name fixtures.
- Bad: a script treats descriptor metadata plus selected-family fields as enough to
  produce executable C evidence.

### 6. Tests Required

- Keep focused fail-closed lit/C++ coverage for obsolete route-family inputs
  without making their names active test APIs.
- Do not require script self-tests for the removed direct e2e runner.
