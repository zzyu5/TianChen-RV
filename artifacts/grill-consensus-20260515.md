# Grill Consensus - 2026-05-15

This note records the read-only grill thread about TianChen-RV project state,
artifact meaning, and Hermes/Codex loop control. It is a discussion checkpoint,
not a compiler artifact, task artifact, evidence artifact, or source of truth.

## Scope

- Repository discussed: `/home/kingdom/phdworks/TianchenRV`.
- No code changes are implied by this note.
- This note was updated from a read-only grill discussion, not from a Trellis
  implementation task. It may later inform spec or prompt updates, but it is not
  itself a spec, prompt, source of truth, or acceptance artifact.
- `artifacts/tmp/` contents are mostly transient outputs, old probes, build
  products, supervisor records, or historical task evidence; the default stance
  is to ignore them unless a specific question needs a specific run output.

## Agreed Framing

1. Artifact classification should be based on whether the output participates in
   the real execution chain, not on whether it exists, has a task archive, or is
   tied to a git commit.

2. The main execution chain is:

   ```text
   TCRV MLIR
     -> extension-family ops
     -> common EmitC/lowering
     -> target artifact bundle
     -> ssh rvv compile/run
   ```

3. Files under `test/**/*.mlir` are normally compiler test fixtures. They should
   be ignored in project-mainline discussion unless the question is specifically
   about verifier/pass/compiler-rule coverage.

4. The cleanup/rebuild context is best understood as two architectural phases,
   not as an artifact status machine or workflow state machine:

   ```text
   Stage A: Wrong Logic Deletion Campaign
     - remove descriptor/direct-C/fake-runtime authority and old tests that
       protect it
     - do not rebuild the correct executable route in the same campaign round

   Stage B: Correct Architecture Rebuild
     - Common Extension Interface Foundation
     - Common Lower-To-EmitC Pass
     - Executable Plugin Construction Template
     - General RVV Extension Family Rebuild
     - real ssh rvv runtime evidence
   ```

5. The deletion campaign exit condition is structural, not "all artifacts are
   clean" or "all tests are green":

   ```text
   descriptor is no longer compute semantics
   direct C exporter is no longer a semantic route
   core passes no longer contain extension-specific compute branches
   docs/comments/tests do not describe extension families as independent backends
   legacy tests no longer protect old paths
   remaining failures are missing new architecture, not old-path compatibility
   ```

## Artifact Meaning

Use these labels when reading output directories:

- `machine probe record`: output from tools like `rvv_remote_probe.py`; records
  remote RVV hardware/toolchain facts, not compiler/runtime success.
- `capability replay input`: historical Python probe-to-MLIR output; this route
  is deleted as active compiler authority and must not be treated as a live
  source of `tcrv.exec` capability MLIR input.
- `compiler test fixture`: most `test/**/*.mlir`; input used to exercise parser,
  verifier, pass, exporter, or fail-closed behavior.
- `boundary note / diagnostic artifact`: metadata or diagnostic output that
  says a route is unsupported, deleted, or fail-closed.
- `control-plane log`: Hermes/Codex supervisor prompts, repo audits, snapshots,
  review inputs, last messages, and loop event records.
- `mainline runtime artifact`: only something that can be placed in the chain
  from TCRV MLIR through extension ops, EmitC/lowering, target artifact bundle,
  and `ssh rvv` compile/run.

## Specificity Rule

Agreed rule:

```text
specificity is allowed inside extension ownership;
specificity is forbidden as generic authority.
```

Concrete interpretation:

- RVV-specific names such as `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`,
  `tcrv_rvv.i32_mul`, `i32_m1`, or `i32_m2` are not automatically wrong.
- They are legitimate when they stay inside RVV dialect, RVV plugin ownership,
  RVV capability profiles, bounded probe facts, or negative tests.
- They become wrong-logic residue when Support/Core/generic Target/front-door
  code uses them to decide runtime ABI identity, artifact kind, route id,
  C function name, intrinsic name, dispatch guard default, or emitted body.

## Three Confusing Layers

1. RVV dialect

   Owns concrete RVV IR surfaces such as `setvl`, `with_vl`, `i32_load`,
   `i32_add`, `i32_sub`, `i32_mul`, and stores. It may express extension-local
   specificity. It must not by itself claim executable artifact, runtime ABI,
   `ssh rvv` success, or target export authority.

2. Support ABI primitives

   May own extension-agnostic shapes such as finite-binary ABI roles,
   `lhs/rhs/out/n`, mem-window specs, runtime-param specs, and role binding.
   It must not own RVV/I32 selected-route defaults such as hard-coded
   `i32-vadd` contracts, `rvv_available`, RVV/scalar dispatch C ABI names, or
   route/artifact identities.

3. Target artifact export

   Owns front-door validation, artifact kind checks, route metadata checks,
   bundle construction, and fail-closed behavior. It must not synthesize compute
   from metadata, descriptors, selected routes, finite family records, or route
   ids.

## Current Active Cleanup Reading

The active cleanup target discussed was the Support-layer I32/RVV runtime ABI
residue: deleting hard-coded I32/RVV runtime ABI authority from Support while
keeping only extension-agnostic finite-binary ABI primitives.

This should be understood as removal of leaked generic authority, not removal
of RVV-specific extension ops or bounded RVV capability facts.

## Runtime Value Boundary

The discussion settled the parameter/value boundary as:

```text
MLIR:
  owns compile-time semantic facts and runtime value roles.

ABI:
  owns callable signature shape, parameter roles, C names, C types, ownership,
  and ordering.

Target deliverable:
  owns packaged payloads and the minimum consumer manifest needed to call them
  without guessing.

Caller / harness / runtime:
  owns concrete runtime values such as pointer addresses, input arrays, runtime
  n, dispatch guard values, and target invocation.
```

Consequences:

- MLIR may contain compile-time facts such as SEW, LMUL, policy, capability
  facts, and IR-modeled runtime roles such as `tcrv.exec.runtime_param`.
- ABI does not "receive concrete values"; it defines slots such as `size_t n`
  and ordered buffer parameters.
- Concrete values are introduced by the generated caller, harness, runtime, or
  external application at call time.
- Target artifact export validates and packages ABI metadata. It must not
  synthesize compute semantics or runtime values.

## Target Manifest Boundary

`target artifact bundle/index` must be treated as an optional target-export
manifest only when multiple payloads or an automatic consumer need to discover
files and ABI information.

Allowed meaning:

```text
packing list / receipt / minimum consumer manifest
```

Disallowed meanings:

```text
project status file
progress ledger
agent coordination protocol
test acceptance database
artifacts/tmp management layer
```

If a single payload or human-explicit command is enough, no bundle/index should
be forced. If a generated caller or harness must discover source/header/object
files, callable names, external ABI names, or ordered ABI parameters, it should
read the manifest rather than guess from filenames, route strings, or historical
conventions.

## Reuse Model

The project reuses the skeleton, not hardware semantics.

Three reuse layers were agreed:

1. Construction-protocol reuse

   ```text
   archetype
     -> semantic role graph
     -> extension family declaration
     -> common interface realization
     -> EmitC route mapping
     -> evidence profile
   ```

   This is the spec-level template for adding a plugin family.

2. Common lowering reuse

   Common EmitC/lowering owns mechanical translation surfaces such as headers,
   function boundaries, ABI mappings, C type mappings, and `emitc.call_opaque`
   materialization. It answers "how to lower", not "what the hardware semantics
   are".

3. Target delivery reuse

   Target export owns candidate validation, route/artifact metadata checks,
   ABI parameter recording, payload writing, and optional consumer manifests. It
   does not own compute synthesis.

Concrete plugin semantics are not reused across families:

```text
RVV owns vadd/vsub/vmul, setvl/with_vl, SEW/LMUL/policy, intrinsic suffixes,
and ssh-rvv evidence.

Future IME/TensorExt/Offload/vendor plugins own their own typed ops,
capabilities, intrinsic/runtime-call mappings, runtime ABI details, and target
evidence.
```

`Template` has two distinct meanings:

- spec-level construction protocol: reusable by all plugin families;
- concrete `tcrv_template` plugin: one implementation specimen/skeleton of that
  protocol, not the semantic parent of RVV.

## Plugin Mainline Staging

The total project mainline is:

```text
capability-driven, plugin-based full-lower execution layer
```

The current active executable phase is:

```text
RVV as the first real executable plugin
  -> plugin-owned typed ops
  -> plugin-owned mapping
  -> common EmitC/lowering
  -> target deliverable
  -> caller/runtime invocation
  -> real ssh rvv compile/link/run correctness evidence
```

RVV is the phase-1 executable mainline, not the permanent whole-project
boundary. After RVV proves the full-lower skeleton, later plugins such as IME,
Offload, TensorExt, or vendor/custom extensions should become their own plugin
mainlines while reusing the same construction protocol, common interfaces,
common EmitC/lowering shell, target delivery model, and evidence boundary.

Scalar is currently:

```text
metadata-only conservative fallback plugin
```

Scalar proves that fallback participation can enter through the plugin registry,
capability, proposal, legality, selection, and selected lowering-boundary
protocol rather than through a core fallback special case. It is not currently an
executable scalar baseline, scalar runtime proof, or scalar performance
baseline. Real scalar fallback lowering would require a later plugin-local
rebuild through typed scalar ops and a materialized EmitC route.

## Core And Plugin Lowering Boundary

The active executable lowering input should not be understood as "naked RVV IR"
or as "metadata-only tcrv.exec". It must combine:

```text
tcrv.exec envelope
  + ABI/runtime boundary
  + plugin-owned typed realization
```

Agreed ownership:

```text
tcrv core:
  owns execution envelope and orchestration:
  kernel, target, capabilities, variants, mem windows, runtime params,
  hart/dispatch/fallback organization, diagnostics, and generic ABI boundaries.

plugin dialects:
  own concrete compute/control realization:
  RVV setvl/with_vl/load/compute/store today, and future IME/Offload/TensorExt
  family-specific typed ops later.

common EmitC/lowering:
  owns generic mechanical lowering from common interfaces and plugin-provided
  mappings to EmitC.

target export:
  owns validation, packaging, and optional consumer manifests.
```

The full-lower input for the first RVV executable path should therefore look
conceptually like:

```text
tcrv.exec.kernel
  -> explicit capabilities
  -> selected rvv-plugin variant
  -> tcrv.exec.mem_window for lhs/rhs/out
  -> tcrv.exec.runtime_param for runtime n
  -> RVV typed body under the selected variant
```

This avoids two invalid shortcuts:

```text
metadata-only compute:
  tcrv.exec.variant has route_id = i32-vadd but no typed RVV body,
  then target export or helper emits the body from metadata.

naked plugin IR:
  tcrv_rvv ops lower directly without tcrv.exec envelope, capabilities,
  variants, ABI roles, or runtime boundary.
```

The first executable proof should lower an already-selected direct RVV variant
before proving the whole proposal/selection/dispatch/fallback pipeline. This is
not a new workflow state. It separates two questions:

```text
Can a selected plugin-owned realization lower and run?
Can planner/selection later produce that selected realization?
```

Dispatch/fallback executable wrapping should come after the direct selected RVV
path. Scalar stays metadata-only during that first executable proof.

## Generic Mechanism, Specific Sample

The first RVV executable sample may be small and specific, for example an
`i32_add`-like RVV op, but the implementation mechanism must be generic.

Agreed rule:

```text
Specific sample, generic mechanism.
No kernel-type lowerer.
```

Common lowering must dispatch on interfaces or registry-provided lowering
mappings, not family names or op names:

```text
Allowed:
  walk extension ops
  query common lowerable interfaces / plugin mapping providers
  get headers, callee names, operand/result mappings, ABI mappings,
  compile requirements
  materialize EmitC mechanically

Not allowed:
  if RVV then ...
  if tcrv_rvv.i32_add then ...
  if route_id == i32-vadd then emit vadd body
```

RVV-specific mapping is still allowed, but it must stay RVV-owned:

```text
tcrv_rvv.i32_add owns add semantics.
RVV mapping owns the RVV intrinsic spelling.
Route metadata owns delivery identity only.
```

Variant metadata can select, describe, cross-check, and package. It must not
replace the typed body:

```text
Variant metadata may route and describe;
variant body must realize compute.
```

Names and tests should be treated as suspicious if they imply kernel-type or
descriptor-route lowering as the implementation unit:

```text
lowerI32BinaryKernel
emitRVVAddKernel
buildFiniteBinaryRVVSource
i32-vadd route authority
rvv-binary-family lowerer
microkernel descriptor lowerer
direct C source exporter
```

The healthy unit of lowering is:

```text
plugin-owned ops through common interfaces
```

## Semantic Realization vs Variant Selection

Important correction:

```text
Do not collapse "how compute semantics are formed" into
"how a variant is selected".
```

These are two different problems.

Semantic realization construction:

```text
source/high-level intent, or a hand-authored first executable slice
  -> tcrv.exec envelope
  -> plugin-owned typed body such as tcrv_rvv ops
```

This decides what computation is being realized: add/sub/mul, load/store shape,
loop/control structure, vector policy, and other concrete semantics. Core,
common lowering, Support ABI, and target export must not infer these semantics
from `route_id`, artifact names, descriptors, test filenames, or packaging
metadata.

For phase 1, this does not require proving a full high-level-to-TCRV frontend.
The first executable proof may start from an already-selected or hand-authored
`tcrv.exec + tcrv_rvv` body, as long as the body is typed and plugin-owned.

Variant selection:

```text
set of semantically valid typed variants
  + target capabilities
  + policy/cost/availability
  -> selected variant
```

This chooses which already-valid realization should run on the target. It may
choose RVV now, scalar fallback later, or future IME/Offload/TensorExt/vendor
routes later. It must not decide the compute semantics themselves.

Therefore, "selecting the correct variant through semantics" means:

```text
Each candidate already has explicit typed semantics.
Selection checks and chooses among legal candidates.
Selection does not synthesize add/sub/mul/RVV behavior from route metadata.
```

A selected variant is correct only if both conditions hold:

```text
1. Its plugin-owned typed body matches the intended computation.
2. Its capability, ABI, and target requirements are legal for the target.
```

Why not start by proving high-level lowering?

```text
High-level -> TCRV/plugin realization is upstream completeness.
Selected TCRV/plugin realization -> EmitC/target/runtime is first executable proof.
```

The first proof should start at the selected `tcrv.exec + tcrv_rvv` realization
because that isolates the active blocker: can a plugin-owned RVV body pass
through common lowering, target export, and real `ssh rvv` execution without
descriptor/direct-C/kernel-type shortcuts?

Once that works, later work can prove that a high-level or planner path produces
the same selected realization. Starting with high-level proof too early would
mix two failures together:

```text
Did we fail to construct/select the right realization?
Did we fail to lower/run an already-selected realization?
```

## No Semantic Recognition Layer

Terminology correction from the later grill:

```text
Do not call the future high-level path "semantic recognition".
Call it source-to-TCRV semantic-preserving conversion.
```

The project should not add a new system that guesses computation semantics from
route ids, descriptor strings, artifact names, test names, or metadata bundles.
If a future high-level MLIR path is added, source dialects such as linalg,
arith, memref, stablehlo, or tosa already carry computation semantics. A
conversion pass may preserve those semantics into TCRV surfaces:

```text
high-level MLIR
  -> tcrv.exec envelope
  -> plugin-owned typed body
```

That future pass must still produce TCRV MLIR. It must not directly produce:

```text
C source
target artifact
route_id-as-semantics
descriptor-shaped compute record
runtime evidence
```

Current active scope does not require proving the high-level path. High-level
MLIR remains a future upstream integration point. The current backend/plugin
work may start from hand-authored or already materialized TianChen-RV MLIR.

## Current Project Reality

The later code/spec check corrected an over-abstract grill direction:

```text
The project is not missing a whole new boundary system.
Many boundaries are already defined in spec, prompt, and code.
```

Already-defined and currently meaningful skeleton:

```text
tcrv.exec core envelope:
  kernel, target, capability, variant, mem_window, runtime_param,
  dispatch, fallback, diagnostics, and ABI envelope.

plugin protocol:
  registry, proposal, legality, selection, selected lowering boundary,
  emission readiness, emission plan, and plugin-owned metadata.

RVV dialect:
  plugin-owned typed control/dataflow surfaces such as setvl, with_vl,
  load, add/sub/mul, and store.

Scalar plugin:
  metadata-only conservative fallback participation through the plugin protocol.

target artifact export:
  selected-path and supported-emission-plan validation, fail-closed candidate
  collection, route metadata checks, payload packaging, and optional manifest.
```

The current gap is therefore not "invent a variant-boundary state machine". The
current maturity gap is:

```text
after deletion completes, rebuild the first executable plugin path
through explicit extension-family ops and a materialized EmitC route.
```

For the first real executable plugin path, that means RVV emission rebuild. But
that is a maturity-path milestone after the relevant deletion rounds, not a
license to implement replacement RVV emission inside a deletion campaign round.

## Spec And Prompt Ownership

Important correction:

```text
This grill note must not become a parallel spec.
```

The durable source of truth is still `.trellis/spec/` plus the active supervisor
prompt/task context when an implementation round is running. This grill note is
only a compact interpretation layer for the human discussion. It may record:

```text
we understand spec X as meaning Y;
therefore we should not do Z.
```

It must not create:

```text
new architecture rules;
new workflow states;
new acceptance gates;
new artifact management layers;
new progress ledgers;
new Trellis task substitutes.
```

If a later grill discovers a real missing contract, the right outcome is:

```text
identify it as a spec/prompt gap;
update the owning spec or prompt explicitly in a separate authorized step.
```

Do not treat this file as acceptance evidence or as an override for specs.

## Mature Path Reading

The mature project path, as clarified against the existing specs and prompt, is:

```text
1. Delete old authority.
2. Keep the existing core/plugin/variant/lowering/export skeleton.
3. Rebuild the first executable plugin path.
4. Use RVV to prove the executable plugin skeleton.
5. Later add high-level frontend conversion and additional plugin mainlines.
```

Deletion means removing active authority from:

```text
descriptor-driven computation;
descriptor-driven C/source or route-to-C exporters;
microkernel wrapper ops as executable semantic authority;
route metadata as add/sub/mul/RVV semantic source;
target export as compute synthesis;
tests that preserve old generated C/header/object/self-check success.
```

Deletion does not mean deleting the project skeleton:

```text
tcrv.exec remains;
capability model remains;
plugin registry/proposal/legality/selection remains;
RVV typed dialect surfaces remain when they are not executable wrapper authority;
Scalar fallback plugin remains metadata-only;
common EmitC interfaces/materialization remain as future rebuild skeleton;
target export remains fail-closed consumer of supported emission plans.
```

After deletion, the first executable maturity milestone is:

```text
RVV selected plugin realization
  -> plugin-owned typed RVV body
  -> materialized common EmitC module route
  -> C/C++ emitter / native compiler
  -> target artifact
  -> real ssh rvv compile/run evidence
```

The point of RVV is not to make RVV the whole project forever. RVV is the first
real hardware proof that the extension-family plugin skeleton can become
executable. Later IME, offload, TensorExt, or vendor/custom plugins should reuse
the same skeleton while owning their own semantics, legality, mappings, runtime
ABI, and evidence.

## What To Do Next Conceptually

For project maturity discussion, the useful question is no longer "what new
boundary should we invent?" The useful question is:

```text
Which existing spec-defined boundary is still protected by old code, tests,
fixtures, or wording that make descriptor/direct-C/metadata routes act as
compute authority?
```

During deletion rounds:

```text
remove or rewrite old authority;
do not add replacement executable architecture in the same round;
report remaining build/test failures as missing rebuild architecture when true;
keep fail-closed behavior honest.
```

During rebuild rounds after deletion:

```text
connect explicit plugin-owned typed ops to common EmitC;
make plugin emission plans supported only when typed bodies and ABI boundaries
are real;
let target export consume supported plans instead of synthesizing compute;
use ssh rvv evidence only for rebuilt RVV artifacts.
```

Default handling of noisy outputs remains:

```text
ignore artifacts/tmp broadly;
inspect only a specific artifact when a specific question requires it;
watch tracked tests/specs/prompts/code more carefully than tmp logs.
```

## Testing Discipline

Tracked tests are more dangerous than `artifacts/tmp` when they protect the
wrong behavior, because they can force future agents to preserve old paths.

Agreed testing stance:

```text
Tests are not project progress.
Tests enforce already-chosen contracts; tests must not author contracts.
```

Useful test categories:

- main path tests that directly exercise the active full-lower route;
- positive contract tests for syntax, verifier, pass behavior, ABI mirrors,
  selection, materialization, and target export behavior that the architecture
  already owns;
- negative guardrails for real risks such as descriptor/direct-C resurrection,
  empty plugin registry materialization, stale metadata, invalid ABI mirrors,
  unsupported route claims, and route-string guessing;
- fixture or tooling tests only when they protect a bounded parser/tooling
  contract and do not claim runtime, correctness, performance, or architecture
  progress.

Before the active executable path works, tests should either build that path or
block a path-killing wrong route. They should not expand into parameter
matrices, dry-run helper coverage, artifact layout coverage, report formatting,
or compatibility tests unless the test blocks a concrete regression risk.

The active completion target is not "one RVV demo" and not "all future plugins
at once". It is:

```text
RVV as first real plugin fully lowers and runs.
```

The architecture implication is:

```text
future plugins reuse the same plugin/full-lower skeleton.
```

## Current Grill Position

The next useful question is not whether to inspect `artifacts/tmp/` broadly.
The default answer is to ignore it. The more important question is:

```text
For any remaining confusing path, which layer owns it?

RVV dialect?
Support ABI primitive?
Target artifact export?
Common EmitC/lowering?
Control-plane/tooling?

And has concrete extension specificity leaked into a generic authority layer?
```

After the later grill updates, the current working consensus can be shortened
to:

```text
One framework mainline; one active plugin mainline at a time.

RVV is first, not final.
Scalar proves plugin fallback participation, not executable baseline status.

Evidence proves architecture; evidence must not become architecture.
Tests protect contracts; tests must not become contracts.
```

## Operationalization

This grill note is a discussion checkpoint, not the durable project authority.
The durable update path is:

```text
promote stable rules into .trellis/spec/;
mirror supervision-critical rules into the Hermes/Codex prompt;
inject only one bounded next-round steering note into the ignored supervisor
artifact root;
restart the loop only after confirming no active worker owns the worktree.
```

The next system action should keep the current deletion campaign intact. It may
steer Hermes toward the strongest remaining descriptor/direct-C/microkernel
wrapper/route-metadata/test-fixture/prompt residue, but it must not turn the
mature RVV executable plugin path into immediate rebuild work before deletion
exit evidence exists.
