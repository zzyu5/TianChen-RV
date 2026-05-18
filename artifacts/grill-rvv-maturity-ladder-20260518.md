# Grill RVV Maturity Ladder - 2026-05-18

This note records the current read-only grill discussion about the RVV-first
phase scope. It is a control-plane interpretation note, not a Trellis spec, not
compiler evidence, not an acceptance artifact, and not a replacement for
`.trellis/spec/**` or `scripts/codex_serial_supervisor_prompt.md`.

The purpose of this file is to avoid losing the discussion during context
compaction. Durable rules must later be promoted explicitly into the owning
specs and the canonical supervisor prompt before they steer future Hermes/Codex
rounds.

## Current Context

- The previous grill consensus is `artifacts/grill-consensus-20260515.md`.
- The latest stopped supervisor loop was halted by a safe STOP file, not by
  killing the active worker.
- The immediate concern is that the loop may treat the bounded i32m1
  add/sub/mul path as the whole RVV-first phase, or may switch too early to
  other plugins / high-level frontends.
- The user clarified that the goal is a complete TianChen-RV-to-RVV lowering
  toolchain, not a tiny permanent demo.

## Corrected Direction

The direction is:

```text
TianChen-RV / tcrv.exec envelope
  + selected RVV variant
  + explicit tcrv_rvv typed body
  -> RVV-owned lowering / mapping / legality
  -> common EmitC or equivalent materialization shell
  -> RVV intrinsic C/C++ or another promoted RVV backend representation
  -> target artifact
  -> runtime invocation
  -> ssh rvv evidence when correctness/runtime claims are made
```

The direction is not:

```text
external RVV C / RVV intrinsic C++ / assembly program
  -> import into TianChen-RV
```

Hand-authored TianChen-RV MLIR is explicitly allowed during this phase. Valid
low-level phase inputs include:

```text
hand-authored tcrv.exec envelopes
hand-authored selected RVV variants
hand-authored tcrv_rvv typed bodies
agent-generated or fixture-generated TianChen-RV MLIR
later, generated TianChen-RV MLIR from high-level frontend conversion
```

The phase does not import or analyze arbitrary external RVV C programs, RVV
intrinsic C/C++ programs, or assembly as source authority.

## Low-Level RVV MLIR Toolchain

The RVV-first phase should be understood as building TianChen-RV's low-level
RVV MLIR toolchain. It is not yet the high-level MLIR frontend phase.

The low-level toolchain should make `tcrv_rvv` a compositional extension-family
IR for RVV-backed kernels:

```text
tcrv.exec owns the execution envelope:
  kernel, capability, selected variant, ABI/runtime boundary, dispatch/fallback
  organization, diagnostics, and target-facing metadata.

tcrv_rvv owns RVV typed realization:
  RVV control, VL, memory, arithmetic, reduction, movement, conversion, policy,
  and other RVV-specific execution surfaces as they are introduced.

RVV plugin/provider owns route support:
  legality contracts, route ids, lowering mappings, intrinsic names, ABI
  bindings where needed, and route-local fail-closed diagnostics.

Common lowering/export owns neutral mechanics:
  interface dispatch, EmitC/materialization shell, selected-path validation,
  artifact packaging, and generic fail-closed routing without RVV semantic
  branches.
```

The bounded i32m1 add/sub/mul path is a historical narrow implementation
artifact and evidence of the old narrow route shape. It is not the terminal RVV
scope, and it should not be preserved as the architecture. The intended work is
to modify or replace the i32-specific route structure with a more general typed
RVV route surface; unsupported residue may fail closed while that correction is
rebuilt.

## Linalg-Calibrated RVV Scope

The RVV scope should be informed by upstream structured-computation dialects
such as MLIR Linalg, even though the current phase does not implement a
high-level Linalg-to-TianChen-RV frontend.

Linalg is useful here as a coverage reference, not as the current input
contract. Its structured-op model covers elementwise maps, broadcasts,
contractions, reductions, matmul-like kernels, loop mapping, vectorization, and
lowering toward loops, intrinsics, library calls, or ISA-specific forms.

The practical consequence for TianChen-RV is:

```text
Use Linalg-like structured computation classes to decide what the low-level
tcrv_rvv route-supported set must eventually be able to express and lower.

Do not start the current implementation from Linalg frontend conversion.
Do not let Linalg op names become route ids or RVV compute authority.
Do not implement per-Linalg-op special lowerers as the RVV architecture.
```

The RVV-first phase should therefore expand toward route-supported units needed
by structured AI kernels, for example:

```text
elementwise maps:
  unary, binary, ternary/select-like vector operations

broadcast and shape-compatible maps:
  scalar/vector broadcast and repeated operand patterns

contraction / matmul-like kernels:
  multiply-add / FMA, accumulation, inner reduction structure, memory tiling

reductions:
  sum/max/min and masked/tail-safe horizontal accumulation

movement and layout support:
  contiguous, masked, strided, gathered, slid, or compressed data movement when
  required by supported structured kernels

conversion and dtype policy:
  SEW/LMUL/dtype conversions, widening/narrowing, and legal policy selection

runtime and parallel structure:
  runtime sizes, tail policy, masks, ABI roles, and later hart/thread partition
```

This does not mean that every Linalg op is immediately supported, or that the
project is building a Linalg frontend now. It means the RVV route-supported
roadmap should be calibrated against the math and data-movement classes that a
future Linalg/high-level path will need to lower into `tcrv_rvv`.

The RVV route-supported scope should be defined as a complete coverage matrix,
not as a small "first batch" that can become another permanent stopping point.
Implementation may still follow dependency order, but the target scope must be
declared up front.

Grill question 15 consensus: do not define RVV maturity as batches. A batch
sounds like a legitimate stopping point, which can recreate the earlier
bounded-i32 trap. The durable rule is:

```text
Define the complete route-supported RVV coverage target first.
Use dependency ordering only to decide implementation sequence.
Treat early executable units as plumbing proof, not phase completion.
Do not let bounded i32m1 arithmetic, or any other early subset, stand in for
the RVV-first maturity target.
```

This is a scope rule, not a progress ledger. It should guide future spec and
supervisor-prompt wording so agents keep expanding the RVV-owned lowering
surface until the declared route-supported coverage target is closed.

```text
Coverage axes:
  VL/control:
    setvl, with_vl, runtime AVL/VL, mask/tail policy, multi-VL loops.

  Memory:
    contiguous load/store, masked load/store, tail-safe accesses, and the
    strided/indexed/gather/scatter surfaces required by supported structured
    kernels.

  Arithmetic and comparison:
    add/sub/mul, FMA-ready arithmetic, max/min, compare/select, and scalar-vector
    or broadcast operand forms.

  Reduction and accumulation:
    horizontal accumulation, masked/tail-safe reduction, accumulator state,
    sum/max/min-style route-supported units.

  Movement and layout:
    slide, gather, compress, layout/tile data movement, and other movement
    surfaces needed by supported low-level body shapes.

  Conversion and dtype policy:
    dtype/SEW/LMUL policy, widening/narrowing, conversion/cast surfaces, and
    route legality across supported element types and vector shapes.

  Runtime and parallel structure:
    runtime sizes, ABI roles, boundary conditions, and later hart/thread
    partitioning when route-supported executable units require it.
```

Reduction and matmul-like contraction are part of the required maturity target,
not optional later ideas:

```text
reduction:
  horizontal accumulation, masked/tail-safe reduction, accumulator state,
  reduction ABI/evidence, sum/max/min-style route-supported units.

matmul / contraction:
  multiply-add/FMA composition, accumulation loops, memory layout and packing
  assumptions, tile/block structure, runtime dimensions, and later hart/thread
  partitioning.
```

The rule is dependency ordering, not batch scoping:

```text
Do not define a tiny batch as the goal.
Define the complete RVV route-supported coverage matrix as the goal.
Execute dependent implementation steps in a sensible order.
Never treat completion of an early subset as completion of the RVV-first phase.
```

## Low-Level Boundary: Not A High-Level Math IR

TianChen-RV must stay a low-level capability-driven execution layer. It must
not drift into a new high-level mathematical IR or kernel IR.

The roadmap therefore should not be phrased as "add high-level matmul ops" or
"make softmax/layernorm kernels first-class core IR semantics." Those names may
be useful as evidence scenarios or future frontend use cases, but the IR owned
by TianChen-RV remains lower-level:

```text
tcrv.exec:
  execution envelope, selected variants, capabilities, ABI/runtime boundaries,
  dispatch/fallback organization, diagnostics, and target-facing metadata.

tcrv_rvv:
  low-level RVV typed control/dataflow units, route-supported lowering units,
  legality/config surfaces, and RVV-owned mappings.
```

The mature roadmap should be expressed as low-level route-supported layers:

```text
Layer 1: route-supported primitive RVV units
  VL/control, masked/tail-safe memory, elementwise operations, broadcast or
  scalar-vector forms, FMA-ready arithmetic, dtype/SEW/LMUL policy.

Layer 2: route-supported low-level RVV body shapes
  legal compositions of Layer 1 units inside explicit selected tcrv_rvv bodies:
  map-shaped bodies, map-reduce-shaped bodies, reduction-shaped bodies,
  contraction-shaped multiply-accumulate bodies, and tile/layout/runtime-size
  assumptions expressed through low-level RVV/TCRV surfaces.

Layer 3: evidence scenarios / future frontend targets
  softmax, layernorm/rmsnorm, rope, attention fragments, matmul-like kernels,
  and other AI kernels used to check whether the low-level route-supported set
  is sufficient.
```

Layer 2 is not a new high-level IR layer. It is not directly expressed by
`tcrv.exec` core as matmul, reduction, softmax, or other high-level kernel ops.
Layer 2 is an RVV-plugin-owned legality/lowering contract over explicit
`tcrv_rvv` bodies.

Layer 3 names must not become TianChen-RV core ops, route ids, or compute
authority. A future high-level frontend may lower such kernels into
`tcrv.exec + tcrv_rvv`, but the current RVV-first work should build the
low-level route-supported units and composition patterns.

## Route-Supported Lowering Unit

The key rule agreed in this grill is:

```text
Every route-supported tcrv_rvv lowering unit must legalize through its declared
RVV-owned route.
```

`route-supported` is a TianChen-RV project term, not an upstream MLIR term. It
is intended to map to normal MLIR ideas such as legality contracts, conversion
targets, rewrite/conversion patterns, op or dialect interfaces, and
legalization.

A route-supported `tcrv_rvv` lowering unit is an op, op pattern, region, or
selected variant body for which the RVV plugin explicitly declares:

```text
a legality contract
a route or mapping id
lowering/conversion patterns or an equivalent RVV-owned lowering provider
operand/result/ABI mapping rules when needed
target/emission expectations when executable
fail-closed diagnostics for unsupported shapes
```

This implies three levels:

```text
parseable / verifier-legal:
  The IR is well-formed for the dialect. This is not an executable promise.

route-supported:
  The RVV plugin declares a lowering route for this op, pattern, region, or
  selected body. Route-contract-valid units must legalize through that route.
  Route-contract-invalid units must fail closed with targeted diagnostics.

executable:
  The route-supported unit is inside a selected tcrv.exec envelope with complete
  ABI, materialization, target artifact, and runtime/evidence boundary.
```

Verifier-legal but not route-supported `tcrv_rvv` IR must not claim artifact or
runtime support.

## Scope Boundary

Allowed during the RVV-first low-level phase:

```text
hand-authored TianChen-RV MLIR
selected tcrv.exec + tcrv_rvv bodies
RVV plugin-owned typed op expansion
RVV plugin-owned legality and route mappings
EmitC / RVV intrinsic C/C++ materialization for declared routes
target artifact export for executable routes
ssh rvv evidence for runtime/correctness claims
```

Not allowed as a shortcut:

```text
external RVV C/intrinsic-C/assembly import as source authority
descriptor-driven compute
route_id or artifact name as compute semantics
target export synthesizing compute bodies
common/core if-RVV semantic branches
per-kernel special lowerers as the main architecture
test-only or evidence-only rounds as active compiler progress by default
```

The project may later add high-level MLIR frontend conversion, but that future
path should produce TianChen-RV MLIR with explicit `tcrv_rvv` typed
realization. It must not bypass the low-level RVV toolchain by emitting C,
artifacts, descriptors, or route ids as compute authority.

## Test-Only Round Concern

The current prompt already discourages tiny helper/test/smoke/report work as
the main achievement, but the loop has still performed rounds where the main
delta was tests, evidence, or fixture closure.

This grill records a sharper desired rule for later prompt/spec updates:

```text
Test-only rounds are not active compiler progress by default.
```

Allowed test-only or evidence-only rounds should be narrow exceptions:

```text
1. A production behavior just landed in the immediately preceding round.
2. The single blocker is focused proof of that behavior.
3. The round does not expand scope.
4. It is not repeated consecutively as a progress substitute.
5. It does not replace lowering, artifact, or runtime implementation.
6. It must either close a specific evidence gap or stop; it must not become
   the next mainline owner.
```

If no production behavior changed, a test-only commit is evidence bookkeeping,
not active RVV toolchain progress.

## Grill Questions 16-33 Consensus

This section preserves the continued grill discussion after question 15. It is
still a discussion note, not a spec update and not a progress/status ledger.

### Q16-Q17: Maturity Closure Uses Structured-Kernel Capability Classes

The RVV maturity closure point should not be phrased as "calibrated by Linalg"
if that sounds like an external reference or optional validation point. The
stronger consensus is:

```text
RVV-first maturity closes when TianChen-RV's low-level RVV route-supported
surface can cover the math and data-movement classes represented by Linalg
structured kernels.
```

Linalg is not the current input contract, route authority, or frontend phase.
However, Linalg structured kernels describe the relevant capability envelope:
elementwise maps, broadcasts, reductions, contractions/matmul-like
accumulation, movement/layout, dtype policy, runtime shapes, masks, tails, and
boundary-safe memory behavior.

This does not mean arbitrary `linalg.generic` scalar regions are accepted as
current inputs. A scalar region is in scope only when it can decompose into
declared low-level `tcrv_rvv` semantic units. Unsupported shapes must fail
closed rather than becoming per-kernel special lowerers or source-pattern
exceptions.

### Q18-Q20: Active Progress Is Production-Path Work

The agent is allowed to judge whether RVV maturity is complete. That judgment
should guide implementation choices; it must not become a persistent status
table, dashboard, coverage ledger, or validation state machine.

Active RVV progress defaults to production compiler/runtime path work:

```text
tcrv_rvv selected-variant semantic surface
  -> RVV plugin route support
  -> lowering / materialization
  -> ABI / runtime / target export when needed by the route
  -> minimal attached verification
```

Test-only, report-only, index-only, dashboard-only, artifact-catalog-only, and
status-ledger-only rounds are not active RVV progress. A test is acceptable only
as a thin guard attached to a real lowering/runtime change, or as a focused
reproduction for a concrete blocker. It must not become the work product.

Downstream ABI/runtime/export work may happen earlier only when it directly
blocks the currently missing RVV-owned semantic surface. It must stay tied to
that route, not become general artifact/runtime management work.

### Q21-Q22: RVV-First Is A Workflow Gate

"Next loop should focus on RVV" is too weak. The durable workflow rule is:

```text
RVV-first maturity is a workflow gate.
Do not enter later plugin mainlines while the RVV route-supported coverage
target is still incomplete.
```

Scalar, IME, offload, high-level frontend generalization, and future plugin
mainlines remain future work until RVV is complete enough to serve as the first
executable plugin path. Exceptions are limited to removing historical residue
that blocks RVV production-path work or correcting specs/prompts to prevent
workflow drift.

The generic plugin contract is already defined elsewhere in `.trellis/spec`.
The remaining issue is not to invent a new plugin rule, but to prevent the
bounded RVV slice from being mistaken for RVV maturity.

### Q23-Q24: Bounded I32 Source Path Is Legacy Narrow Path, Not Maturity Unit

The bounded `i32m1 add/sub/mul` path is an artifact of the old narrow
implementation, not a model to preserve. It only showed that the following
production path can exist:

```text
selected RVV variant
  -> explicit typed tcrv_rvv body
  -> RVV-owned route
  -> EmitC / materialization
  -> target artifact handoff
```

It is not the RVV maturity unit. It must not become the template for repeatedly
adding exact source-front-door scenarios such as one source recognizer for each
operation, dtype, or structured kernel. Future work should modify this narrow
i32-specific route into a more general RVV-owned typed surface and route
support, not preserve it with compatibility glue or multiply one-off source
pattern contracts.

The current vector/scf source front door should be treated as a bounded
compatibility path and historical narrow path. RVV maturity's main entry is the
explicit typed `tcrv_rvv` body inside the selected RVV variant. Future
high-level, source, Vector, SCF, or Linalg-like inputs may lower into that typed
body, but they must
not bypass it by going directly to artifacts, route ids, descriptors, or C.

### Q25-Q27: TianChen-RV MLIR Is The Subject

TianChen-RV MLIR is the subject. `tcrv_rvv` is not an independent plugin dialect
or standalone backend system. More precise terminology:

```text
TianChen-RV MLIR:
  the unified IR suite and system.

tcrv.exec:
  the execution envelope: kernel, capability, selected variant, dispatch,
  runtime boundary, and diagnostics; not compute semantics.

tcrv_rvv:
  TianChen-RV MLIR's typed RVV extension-family IR used inside the selected
  RVV variant body after the tcrv.exec envelope selects the RVV path.

RVV plugin:
  the provider/locality for RVV capability, legality, route support, lowering,
  emission, and runtime glue for that TianChen-RV RVV extension-family path.
```

Formal RVV maturity is not "a naked `tcrv_rvv` file lowers." It is:

```text
TianChen-RV MLIR
  -> tcrv.exec envelope
  -> selected RVV variant
  -> typed tcrv_rvv body
  -> RVV plugin route / lowering / emission
  -> common materialization / export / runtime boundary
```

Naked `tcrv_rvv` or pass-level `.mlir` fixtures are local development aids for
parser, verifier, pass, or lowering-pattern behavior. They do not prove the
selected-variant executable path and must not become architecture authority.

### Q28-Q31: Low-Level RVV Unit Taxonomy Is The Real Missing Spec Surface

The next core question is how the selected RVV variant body is corrected away
from the bounded `i32m1 add/sub/mul` path toward the math/data-movement classes
required for structured kernels.

The wrong routes are:

```text
exact source-front-door recognizers for every new case
per-kernel variant templates
high-level kernel ops such as matmul_kernel / softmax_kernel / layernorm_kernel
one tcrv_rvv op per RVV intrinsic spelling
```

The intended route is a composable typed low-level RVV unit taxonomy:

```text
VL / control
mask / tail policy
memory access shape
arithmetic / compare / select
reduction / accumulation
movement / layout
conversion / dtype / SEW / LMUL
runtime boundary
contraction-supporting accumulation and movement
```

These units should be low-level enough to lower through RVV routes into
EmitC/materialized RVV intrinsic C/C++ and structured enough to support
legality, route support, ABI/runtime checks, and fail-closed diagnostics. They
should not be high-level kernel ops and should not mechanically mirror every
intrinsic name.

MLIR Vector and Linalg are taxonomy references. They can inform which semantic
categories are needed, but they are not the current input contract, route
authority, or completion proof. Their relevant categories should map into
explicit TianChen-RV `tcrv_rvv` units inside selected RVV variants.

The current spec gap is therefore not "define plugin contract again." It is to
make the `tcrv_rvv` selected-variant low-level unit taxonomy explicit enough
that future agents expand the RVV semantic surface instead of orbiting the
bounded i32 source front door, fixtures, tests, or artifacts.

### Q32-Q33: Landing Plan For Later Durable Updates

When this grill converges, durable updates should land in the owning surfaces:

```text
.trellis/spec/extension-plugins/rvv-plugin.md:
  RVV maturity target, tcrv_rvv low-level unit taxonomy, bounded i32
  historical-narrow-path role, and source-front-door demotion.

.trellis/spec/plugin-protocol/*
.trellis/spec/lowering-runtime/*:
  extension-owned typed units first; common lowering/export owns neutral
  mechanics only; target/export does not synthesize compute.

scripts/codex_serial_supervisor_prompt.md:
  RVV incomplete means do not enter later plugin mainlines; production-path-only
  progress; no test/report/index/dashboard-only rounds; no copying exact source
  front-door scenarios as maturity work.
```

The user asked to save this before continuing the grill. The next grill branch
should be concrete and repo-informed, especially:

```text
Q34 candidate:
  Is the current i32m1 op/unit surface too low-level or too dtype-specific?

Q35 candidate:
  How exactly should TianChen-RV selected tcrv_rvv IR lower into RVV intrinsic
  C/C++ or another promoted RVV backend representation?

Q36 candidate:
  Where should the pass pipeline live, and how should the pipeline sequence be
  organized from tcrv.exec selected variant to intrinsic C/materialized target?
```

## Grill Questions 34-42 Consensus

This section corrects the first attempt at questions 34-42. The important
correction is that the discussion must track the real repository pipeline, not
invent a detached "semantic unit" layer or pretend the route provider directly
prints C.

### Q34: Current I32M1 Is Legacy Narrow Structure, Not Something To Preserve

Current repo facts:

```text
RVVOps.td:
  !tcrv_rvv.i32m1 / !tcrv_rvv.i32m2
  tcrv_rvv.i32_load
  tcrv_rvv.i32_add / i32_sub / i32_mul
  tcrv_rvv.i32_store

RVVEmitCRouteProvider.cpp:
  RVVI32M1ArithmeticRouteSpec
  collectRVVI32M1ArithmeticSlice()
  exactly two i32_load ops
  exactly one i32_add/i32_sub/i32_mul op
  exactly one i32_store op
  fixed op-count-10 shape
  direct mapping to __riscv_*_i32m1 intrinsics
```

The bounded i32m1 path is historical evidence that an executable route once
existed, but it is not a route structure to preserve. It is too specific as the
maturity architecture because dtype, LMUL, operation family, body shape, route
ids, and intrinsic spelling are currently coupled too early.

The desired direction is not to delete low-level RVV IR or raise the IR into
kernel ops. It is to modify the current narrow i32-specific route into a typed
RVV route structure where the selected `tcrv_rvv` body is not organized around
`i32m1 add/sub/mul` as route authority. If the old i32m1 route cannot fit that
structure cleanly during Stage 1, it should fail closed or be removed while the
correct route structure is rebuilt, rather than being preserved through
compatibility glue.

### Q35-Q40: Stage Discipline

The project may use stages, but not feature batches:

```text
Stage:
  dependency and architecture-hygiene ordering.

Batch:
  false completion unit such as "i32m1 add/sub/mul is done, therefore RVV is
  done enough."
```

Stage 1 is a short but real cleanup/correction stage. It is allowed to be a
large architectural refactor. Its purpose is to remove or fail-close
`i32m1-as-architecture-authority`.

Stage 1 may allow the bounded i32m1 executable route to fail closed or disappear
while the correct structure is rebuilt. This is not a loss of the intended
architecture; it is an acceptable correction of the old narrow route. Build-level
coherence still matters: commits should not leave the repository unbuildable or
APIs half-broken. What is not allowed is preserving the wrong route through
compatibility glue.

Stage 1 is not a long deletion campaign. It must not drift into full RVV
feature implementation, reduction/matmul, high-level frontend work, exhaustive
test rewrite, report/dashboard/index cleanup, or test-only progress.

After Stage 1 cleanup/correction, the workflow should move directly into Stage
2 RVV expansion. Do not insert report-only summary, broad test cleanup,
artifact dashboard, acceptance-status, or loop-health paperwork between Stage 1
and Stage 2. Minimal build/coherence checks are fine; runtime evidence is needed
only when a runtime/executable claim is made.

### Q41-Q42: Do Not Invent A New Decorator Layer

The phrase "foundation semantic units" is suspicious and should not become a
new decorator system. The user clarified the intended direction:

```text
Keep the real working path.
Replace the current i32-specific route architecture with a typed RVV route
surface that no longer hard-codes i32m1 as the route architecture.
Add the correct pass/route pipeline around the selected typed RVV body.
```

The corrected Stage 2 direction is:

```text
selected tcrv.exec RVV variant
  -> typed tcrv_rvv body
  -> plugin-owned route provider builds a TCRVEmitCLowerableRoute
  -> common materializer lowers that route to MLIR EmitC dialect
  -> EmitC C/C++ emission / target artifact export handles downstream delivery
```

This is the repo's actual shape. In current code:

```text
RVVEmitCRouteProvider.cpp:
  constructs TCRVEmitCLowerableRoute;
  adds headers such as riscv_vector.h;
  adds type mappings such as !tcrv_rvv.i32m1 -> vint32m1_t;
  adds ABI mappings;
  adds call-opaque steps and structured EmitC loop steps with RVV intrinsic
  callee names.

lib/Transforms/EmitCLowerableMaterialization.cpp:
  asks the origin plugin registry to build the TCRVEmitCLowerableRoute;
  calls materializeTCRVEmitCLowerableRoute().

lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp:
  creates MLIR emitc.include / emitc.func / emitc.for / emitc.call_opaque.
```

Therefore a precise wording is:

```text
The RVV route provider does not directly print C source and should not bypass
MLIR EmitC. It provides plugin-owned route semantics and intrinsic mapping by
building a TCRVEmitCLowerableRoute. The common materializer lowers that route
into MLIR EmitC dialect. Later translation/export turns EmitC into C/C++ or
target artifacts.
```

Stage 2 should rebuild the selected typed RVV path around the corrected typed
RVV route structure, not merely keep the old i32m1 path and add more cases:

```text
Current narrow shape to modify or replace:
  !tcrv_rvv.i32m1
  tcrv_rvv.i32_load
  tcrv_rvv.i32_add / i32_sub / i32_mul
  tcrv_rvv.i32_store
  RVVI32M1ArithmeticSlice
  __riscv_*_i32m1 spelling selected by exact route

Desired corrected direction:
  typed RVV value/config/body representation with enough abstraction for dtype,
  SEW, LMUL, policy, memory shape, and operation kind;
  route provider derives the concrete intrinsic mapping from that structure;
  common materializer still emits MLIR EmitC dialect;
  unsupported combinations fail closed.
```

This is not a new high-level kernel IR, not a metadata/decorator layer, and not
a one-op-per-intrinsic dialect. It is a correction of the current i32 path's
abstraction boundary so the real selected-variant-to-EmitC pipeline can support
RVV broadly without keeping i32m1 as the architectural center.

## Grill Questions 43-60 Consensus

### Q43-Q44: Stage 1 And Stage 2 Are Now Named By Architecture Role

The useful stage names are not "cleanup" and "more RVV"; those are too vague
and let future agents drift into test cleanup, artifact accounting, or another
small feature batch. The corrected stage definitions are:

```text
Stage 1:
  RVV route-authority replacement.

Stage 2:
  route-supported RVV coverage expansion on the corrected typed surface.
```

Stage 1 replaces the old i32m1-as-architecture-authority. It is allowed to
fail-close or delete the old i32m1-specific route while replacing that
authority. The replacement authority is:

```text
typed low-level tcrv_rvv execution surface
RVV-owned legality
RVV-owned route provider
fail-closed unsupported combinations
common EmitC/export as neutral mechanics only
```

Stage 1 is not a first-case demo, not a request to preserve a minimal i32 path,
and not a hidden requirement to restore a particular executable case before the
architecture is allowed to move forward. No single restored case, i32m1 or
otherwise, defines Stage 1 completion. Compatibility wrappers that preserve the
old i32m1 route as the architecture are explicitly the wrong direction.

Stage 2 then expands route-supported RVV coverage on the corrected typed
low-level `tcrv_rvv` surface. It should use dependency order for
implementation, but it must not turn dependency order into small completion
batches such as "i32 is done" or "f32 is done".

### Q45-Q47: Provider, EmitC, And Product Shape

The term "RVV provider" means the RVV plugin-owned route builder/lowering
authority. It is not a new IR, dialect, state machine, or dashboard. Its job is
to read the selected typed `tcrv_rvv` body and decide, under RVV legality:

```text
supported or fail-closed
C/RVV vector type mappings
RVV intrinsic spelling
headers
ABI mappings
TCRVEmitCLowerableRoute payload
```

MLIR EmitC dialect is not the RVV semantic layer. It is the C-like MLIR
materialization target after the RVV provider has already decided the route.
The common materializer should consume the provider-built
`TCRVEmitCLowerableRoute` and produce MLIR EmitC operations such as includes,
functions, loops, and opaque calls. EmitC/common/export must not choose RVV
intrinsics or infer RVV semantics.

The 5/18 project shape is:

```text
TianChen-RV MLIR:
  capability-driven execution layer after high-level MLIR.

tcrv.exec:
  execution envelope, selected variants, ABI/runtime boundary, dispatch,
  fallback, diagnostics, and target/capability organization.

tcrv_rvv:
  typed low-level RVV execution body inside selected RVV variants.

RVV plugin/provider:
  RVV capability, legality, route support, intrinsic mapping, emission, and
  runtime glue for the RVV extension-family path.
```

This project is not a high-level Linalg/Vector frontend, not an RVV intrinsic
wrapper dialect, not a new kernel/math IR, and not an i32m1 add/sub/mul demo.
Linalg and Vector are coverage/reference inputs for the maturity target and
future frontend work; they are not current route authority.

### Q48-Q52: Maturity Definition And Short Steering Form

The earlier coverage discussion and the later representation discussion are one
definition:

```text
RVV maturity =
  coverage target
  + representation discipline
  + executable route discipline.
```

Coverage target:

```text
route-supported low-level RVV units cover structured-kernel math and
data-movement needs such as elementwise, broadcast, reduction, contraction-like
accumulation, memory movement, dtype conversion, and runtime shape/control.
```

Representation discipline:

```text
coverage must be represented as typed low-level tcrv_rvv execution bodies, not
high-level kernel ops, one-intrinsic wrappers, metadata mega-ops, or i32m1-style
clone batches.
```

Executable route discipline:

```text
RVV plugin/provider owns legality and intrinsic route selection. Common
EmitC/export only materializes and packages selected routes.
```

The durable spec can use the three-part definition above. Hermes/Codex steering
should be shorter:

```text
Current work is RVV Stage 1: route-authority replacement.

Replace or fail-close i32m1-as-architecture-authority with typed low-level
tcrv_rvv execution bodies plus RVV-owned legality and route provider logic.

Do not preserve the old i32 route through compatibility glue, clone it into
dtype/LMUL batches, add high-level kernel ops, wrap intrinsics one-for-one,
move RVV semantics into common/export code, or spend the round mainly on
tests/reports/artifact indexes.
```

Stage 1 ends when no active compiler path treats i32m1 arithmetic as RVV route
authority. At that point, stop cleanup/deletion work and enter Stage 2.
Stage 2 expands route-supported RVV coverage on the corrected typed low-level
`tcrv_rvv` execution surface, using dependency order but not small completion
batches.

### Q53-Q55: Migration Strategy And Source/Frontend Boundary

The accepted Stage 1 migration strategy is replacement first:

```text
B preferred:
  introduce the corrected typed low-level tcrv_rvv surface and RVV-owned route
  authority;
  let old i32m1-specific routes fail closed or be deleted when they do not fit;
  later reconnect i32m1 add/sub/mul as ordinary instances of the corrected
  surface.

A allowed locally:
  modify existing code in place when it genuinely serves the replacement.

C rejected:
  preserve the old i32m1 route with compatibility wrappers.
```

Stage 1 also does not mature high-level/source frontend lowering. Existing
bounded Vector/SCF/i32m1 source front doors must not remain route authority.
They may be disabled, fail-closed, or later reconnected through the corrected
typed `tcrv_rvv` surface. Source/frontend work must not define RVV maturity and
must not preserve the old i32m1 architecture.

### Q56-Q60: Exec, Variant, Parameters, And Dtype/Config Flow

The corrected responsibility split is:

```text
tcrv.exec envelope:
  declares and binds ABI/runtime roles with mem_window/runtime_param;
  organizes kernel, capability, selected variants, dispatch, fallback, and
  diagnostics;
  does not infer compute semantics from parameter names, C type strings, route
  ids, or artifact metadata.

tcrv_rvv body:
  imports those ABI/runtime values through explicit binding ops;
  uses them in typed RVV dataflow/control/config ops;
  carries or consumes dtype, SEW, LMUL, policy, VL/AVL, memory form, op kind,
  and body shape.

RVV provider:
  validates the selected tcrv_rvv dataflow/config against capability;
  derives the RVV intrinsic route, C vector type, headers, ABI mapping, and
  EmitC route payload.
```

In the current RVV-first phase, "start from `tcrv.exec`" means starting from a
complete TianChen-RV execution surface:

```text
tcrv.exec.kernel
  -> selected tcrv.exec.variant { origin = "rvv-plugin", requires = [...] }
  -> explicit typed tcrv_rvv body inside that variant
```

It does not mean a bare `tcrv.exec.kernel` may infer RVV compute semantics from
capability, route ids, ABI strings, parameter names, or artifact metadata.

For future high-level paths, source IR such as Linalg or Vector does not lower
to a generic/bare `tcrv.exec` compute op. It lowers into TianChen-RV execution
surfaces: the `tcrv.exec` envelope plus plugin-proposed typed extension-family
variant bodies. For RVV, source dtype and semantics must be preserved into an
explicit typed `tcrv_rvv` body; RVV-specific SEW/LMUL/policy are lowering,
legality, and tuning choices constrained by source semantics and target
capability, and must be explicit in the `tcrv_rvv` body before route selection.

Parameter flow can be summarized as:

```text
tcrv.exec envelope
  uses mem_window / runtime_param to bind parameter roles
        |
        v
tcrv_rvv body
  imports those ABI values and uses them in real RVV typed dataflow
        |
        v
RVV provider
  lowers the selected typed RVV body to a route
```

ABI strings and parameter names may be consistency/provenance inputs, but they
are not compute authority. The `tcrv_rvv` body is the canonical downstream
authority for dtype/config/dataflow once the RVV variant exists.

## Grill Questions 61-65 Consensus

### Q61: Stage 3 Is Extension-Family Generalization, Not Exec Genericity

`tcrv.exec` genericity is an always-on invariant, not a future stage. Stage 1
and Stage 2 must already preserve it. Stage 3 should therefore not be named
"make exec generic"; that would imply the earlier stages are allowed to make
`tcrv.exec` RVV-specific.

The accepted Stage 3 meaning is:

```text
Stage 3:
  extension-family generalization / second-family integration after RVV
  maturity.
```

Stage 3 validates that a real non-RVV family can use the same TianChen-RV
execution architecture:

```text
tcrv.exec envelope
  -> selected variant with origin/requires
  -> plugin-owned typed extension body or selected boundary
  -> plugin-owned legality and route/provider mapping
  -> common selected-boundary / EmitC route / export mechanics
```

Template and Toy are scaffold/protocol proofs. They are useful because they
show how a family can register capabilities, dialects, selected boundaries,
role ops, and EmitC routes without changing core orchestration. They are not,
by themselves, proof of Stage 3 maturity.

### Q62-Q64: How Exec Is Filled Across Plugins

For different plugins, `tcrv.exec` is filled only with generic execution
structure:

```text
kernel
target/capability scope
mem_window
runtime_param
variant
requires
dispatch
fallback
diagnostics
```

Plugin-specific filling happens inside selected variant bodies or selected
plugin boundaries:

```text
RVV:
  typed tcrv_rvv dataflow/config/body.

TensorExtLite / IME:
  fragment/MMA-style typed extension body or selected boundary.

Offload:
  runtime-offload bind/call/wait style body or boundary.

Scalar:
  fallback-local body or fail-closed boundary.

Template / Toy:
  scaffold/protocol-proof boundary and role-op surfaces.
```

Variant generation has two separate responsibilities:

```text
core materializes variant envelope from plugin proposal:
  tcrv.exec.variant { origin, requires, condition, guard, policy, plugin attrs }

plugin-owned construction/materialization provides:
  typed extension body or selected boundary.
```

The current public pipeline reflects that split. The proposal materialization
step creates `tcrv.exec.variant` envelopes. Later registry-mediated calls ask
the origin plugin to verify legality, estimate cost, materialize or validate
selected boundaries, build emission plans, and build EmitC-lowerable routes.
Core passes route through plugin interfaces; they must not branch on family
names to synthesize compute.

Parameter flow remains:

```text
exec.mem_window / exec.runtime_param
  declare ABI roles
        |
        v
plugin-owned binding ops or selected boundary
  import those roles into the extension-family body
        |
        v
extension-family typed ops
  use them for real dataflow/config/compute
        |
        v
provider
  lowers selected body to route
```

For future high-level frontend work, the full system model is:

```text
source semantics
  -> plugin/interface-owned construction of candidate extension bodies
  -> common tcrv.exec envelope selection/dispatch
  -> selected plugin-owned lowering route
```

For an elementwise add, future plugin construction may produce:

```text
RVV plugin:
  setvl, load lhs, load rhs, add, store out.

Scalar plugin:
  scalar loop or scalar fallback body.

Offload plugin:
  runtime call / bind / wait body.
```

This is a later full-system model, not current Stage 1/2 implementation scope.
Current Stage 1/2 must not implement high-level source-to-multi-plugin
construction, Linalg/Vector frontend generalization, or generation of
RVV/Scalar/Offload candidates from one source. Current work matures:

```text
selected tcrv.exec RVV variant
  + explicit typed tcrv_rvv body
  -> RVV-owned legality/provider route
  -> common EmitC/export
```

Future high-level construction can plug into this matured low-level RVV surface
later.

### Q65: Stage 3 Reuse Proof

Stage 3 is not "add another plugin directory." It must prove real reuse of the
TCRV common surfaces. A valid second-family plugin should reach execution
through the same shared architecture:

```text
tcrv.exec envelope
ExtensionPluginRegistry hooks
plugin-local typed extension body or selected boundary
plugin-owned legality and route provider
selected lowering-boundary framework
TCRVEmitCLowerableRoute
common EmitC materializer
target export/package framework
```

New family logic must stay local to family/plugin/target-support code. Core or
common changes are allowed only when they extend a generic public interface for
all families. They are not allowed when they add family-specific branches such
as:

```text
if RVV
if TensorExtLite
if Offload
if IME
```

Stage 3 maturity requires at least one real non-RVV family, such as
TensorExtLite, Scalar, Offload, or IME, to use the common path without
RVV-specific or family-specific leakage into core/common. Template/Toy remain
scaffolds and protocol proofs unless paired with such real second-family
integration.

### Q66: Stage 3 Generality, Usability, And RISC-V Extension Fit

Stage 3 has a clearer target now. It is not "make `tcrv.exec` more generic,"
because `tcrv.exec` is already the generic execution envelope. It is also not
"add one more template plugin," because Template/Toy/TensorExtLite-style slices
are construction proofs unless a real non-RVV extension family reaches the same
TCRV path.

Stage 3 should be defined as:

```text
Make adding a new RISC-V extension family local, repeatable, and
common-path-verifiable.
```

Generality comes from shared TCRV surfaces:

```text
tcrv.exec envelope
  -> capability / variant / requires / dispatch / fallback / ABI roles
ExtensionPluginRegistry
  -> capability, dialect, source-front-door, proposal, legality, cost,
     selected-boundary, emission-plan, EmitC-route, target-support hooks
common TCRV interfaces
  -> extension ops expose config/resource/memory/compute/EmitC roles without
     core branching on concrete family names
common EmitC/export route
  -> extension-family ops or selected boundary become EmitC, then C/C++,
     then intrinsic / vendor builtin / runtime C ABI / object artifact
```

Usability means the extension author follows a known local construction path
rather than redesigning the compiler:

```text
classify the RISC-V extension archetype
  -> define capability ids and availability
  -> define family ops/types/attrs
  -> implement common TCRV interfaces
  -> register the plugin and dialects
  -> provide proposal / legality / cost / selected-boundary hooks
  -> provide plugin-owned EmitC/runtime/target route
  -> reuse common exec, registry, EmitC, runtime ABI, and artifact-export
     mechanics
```

"Easy to add" therefore means local and repeatable, not zero work. A new
extension normally adds files under plugin/family/target/test ownership. It
should not require a new family-specific branch in core variant selection,
capability checking, dispatch planning, selected-boundary orchestration, or the
generic artifact front door. If core/common changes are needed, they must extend
a public generic interface for all families rather than introduce a branch such
as `if RVV`, `if IME`, `if TensorExt`, or `if Offload`.

This matches RISC-V because RISC-V extensions do not all share one execution
shape. RVV, matrix/fragment extensions, custom opcodes, vendor builtins,
runtime-offload paths, sparse/DMA/cluster resources, and future accelerators may
need different typed ops, resource models, legality checks, and runtime routes.
TCRV should not force them into RVV, IME, or Offload unless the semantics truly
match. The common abstraction is capability-scoped extension execution, not a
single mathematical kernel op and not one independent backend dialect per
hardware target.

The Stage 3 maturity proof is:

```text
one real non-RVV extension family
  -> uses tcrv.exec for envelope and ABI roles
  -> owns its typed body or selected boundary
  -> owns capability / legality / cost / route construction
  -> lowers through TCRVEmitCLowerableRoute and common EmitC/export machinery
  -> avoids family-name leakage into core/common
```

Template/Toy/TensorExtLite can remain useful scaffolds, but they are not enough
by themselves to claim Stage 3 maturity. Stage 3 maturity needs at least one
real non-RVV family path proving the same common TCRV route works beyond RVV.

### Q67-Q87: Final Performance Layer And Exec-To-Intrinsic Flow

This section records the corrected performance-layer consensus. It supersedes
any earlier phrasing that made RVV realization sound like a state machine, a
high-level kernel IR, a dashboard, or an EmitC-side scheduling trick.

Performance discussion must not drift back into high-level frontend work. The
current Stage 1 and Stage 2 system starts from TianChen-RV execution IR, not
from Linalg/Vector/Transform dialect lowering. Linalg/Vector-style structured
computation remains a coverage reference and future frontend input class, not
the current route authority.

The accepted role is:

```text
high-level source semantics        future only, not current Stage 1/2 authority
        |
        v
tcrv.exec envelope                 stable execution envelope and parameter roles
        |
        v
selected tcrv_rvv variant body     RVV-owned vector-level execution IR
        |
        v
RVV plugin-local realization       one linear target-performance lowering step
        |
        v
realized tcrv_rvv selected body    operative vector-level RVV structure
        |
        v
TCRVEmitCLowerableRoute / EmitC    faithful emission shell
        |
        v
RVV intrinsic C/C++                riscv_vector.h / native compiler surface
        |
        v
native RVV compile/run evidence
```

`tcrv_rvv` should be defined as RVV-owned vector-level execution IR. Its
abstraction level is close to MLIR Vector: higher than a list of RVV intrinsic C
calls, lower than Linalg/tensor/kernel semantics. It is not a high-level matmul,
softmax, reduction-kernel, tensor/tile, or per-kernel IR layer, and it is not a
one-op-per-intrinsic wrapper dialect.

Mature `tcrv_rvv` bodies should express vector-level execution categories:

```text
vector typed dataflow
vector load/store and memory access forms
vector arithmetic, compare, select, FMA/update
vector reduction primitives and accumulator values
mask/tail/policy
dynamic VL / AVL relation
SEW / LMUL / vtype constraints
runtime ABI value use
low-level control around vector operations
realization hints only when they are consumed by RVV realization
```

They must not introduce high-level ops such as:

```text
tcrv_rvv.matmul
tcrv_rvv.softmax
tcrv_rvv.reduction_kernel
generic tensor tile ops
route-id-as-compute ops
```

For matmul-like or reduction-like work, the pre-realized `tcrv_rvv` body should
already be lowered to vector-level operations: vector loads, broadcast or
movement forms when needed, vector FMA/update, vector accumulator values,
vector reduction primitives, mask/tail handling, and stores. RVV realization
then chooses the RVV execution structure. It does not discover matmul or
reduction semantics from names, routes, artifacts, or C.

The performance optimization product is the realized `tcrv_rvv` selected body.
It is not EmitC, not intrinsic C, not a route string, not an artifact index, not
a status marker, and not a dashboard. EmitC and intrinsic C are faithful
emission products after the RVV body has already been shaped.

Current code reality: today the concrete RVV path is still too direct.
`RVVExtensionPlugin` is the plugin object registered through the extension
registry. `RVVEmitCRouteProvider.cpp` is a route-building helper, not the mature
performance layer. `tcrv-materialize-emitc-lowerable-routes` is a common pass
that finds the selected variant, asks the registry for the origin plugin, calls
`buildVariantEmitCLowerableRoute`, verifies the returned route, and materializes
EmitC. The current RVV route builder directly constructs a bounded i32m1 route:
include `riscv_vector.h`, map RVV types, emit `__riscv_vsetvl_e32m1`, vector
loads, arithmetic, and vector store. That is useful as route-shape evidence, but
it is not the mature performance architecture.

The corrected pipeline needs a selected-body realization slot after selected
lowering-boundary materialization and before emission planning:

```text
materialize selected lowering boundaries
  -> realize selected extension bodies
  -> materialize emission plans
```

In terms of the current public planning pipeline, the new slot belongs between
`createMaterializeSelectedLoweringBoundariesPass(registry)` and
`createMaterializeEmissionPlansPass(registry)`.

The common realization pass/hook is only orchestration:

```text
find the selected variant / selected path
read the variant origin
route to the origin plugin through ExtensionPluginRegistry
propagate diagnostics and fail closed
```

It must not know RVV-specific concepts such as SEW, LMUL, mask/tail policy,
`vsetvl` placement, RVV prefetch form, RVV unroll shape, vector register
pressure, or reduction accumulator layout.

The RVV plugin owns the RVV-specific internal tuning and realization:

```text
consume target capability facts
consume the selected vector-level tcrv_rvv body
consume runtime SSA / ABI values such as AVL, n, pointers, and guards
consume optional hand-written, frontend, user-policy, profile, or autotune hints
choose legal SEW / LMUL / policy choices
place dynamic VL / vsetvl control
choose memory forms allowed by the body and target
realize mask/tail behavior
choose register-pressure-safe unroll / pipeline / prefetch structure
choose accumulator / reduction layout when the vector-level body requires it
rewrite the selected body into realized vector-level tcrv_rvv structure
```

The config source is RVV plugin-local tuning. Hints may come from hand-written
TianChen-RV MLIR, a future frontend, profile/autotune data, user policy, or
capability facts, but none of those hints is route authority. The RVV plugin
must check and consume them. `tcrv.exec`, common passes, EmitC route builders,
artifacts, route ids, and test names must not invent performance config.

The current scope only needs internal tuning for the selected RVV variant. It
does not need a multi-candidate global autotuning system, a tuning database, or
extra competing `rvv_m1_unroll1` / `rvv_m2_pipeline` variants. Cross-variant and
cross-plugin tuning can be future work after the selected RVV body path matures.

RVV realization is a one-time linear transformation:

```text
selected pre-realized tcrv_rvv body
  -> RVV plugin-local realization
  -> realized tcrv_rvv body
```

It is not a repeated optimization loop, not an idempotence/state-machine design,
and not a progress/status model. Normal compilation runs this step once. If an
implementation later protects against accidental double-running, that is only
defensive hygiene, not an architectural maturity claim.

Realization changes execution structure, not computation semantics. It may
change or add RVV body structure such as:

```text
dynamic VL / setvl placement
SEW / LMUL / policy legalization
low-level vector control around vector ops
memory form selection when semantics and capability allow it
prefetch ops or explicit no-support failure/no-op
unroll and software-pipeline organization over vector-level operations
accumulator and reduction realization
mask/tail realization
```

It must not change:

```text
kernel computation semantics
parameter roles
dtype semantics
variant origin
required capabilities
target capability facts
dispatch / fallback semantics
runtime n / AVL into a fake compile-time constant
the body into direct EmitC or C
```

Config is not the final product. If a hint affects generated code, RVV
realization must consume it into operative `tcrv_rvv` vector-level structure. If
a hint does not affect generated code, it is only hint/provenance and must not
become route authority. Later emission validates and lowers the body structure;
it does not consult a separate readiness state, dashboard, status field, or
acceptance label.

Full exec-envelope to intrinsic-C flow after the intended refactor:

1. `tcrv.exec` envelope declares the kernel, target/capability requirements,
   variant candidates, dispatch/fallback, and parameter roles such as memory
   windows and runtime parameters. It does not infer computation from parameter
   names and does not invent dtype/shape semantics.
2. A selected RVV variant exists under the envelope. Its body is vector-level
   typed `tcrv_rvv` IR. Dtype, memory access roles, vector operation semantics,
   VL/AVL use, and policy constraints must be present in or derivable from this
   body plus target capability. They do not appear from artifacts, route names,
   or test files.
3. Common planning passes verify plugin legality, select the variant,
   materialize dispatch guards, check capability requirements, and request
   selected lowering boundaries through the registry.
4. The selected-body realization slot calls the origin plugin. For RVV, the RVV
   plugin consumes capability, body facts, runtime values, and hints into a
   realized vector-level `tcrv_rvv` body.
5. Emission planning and EmitC route materialization consume the realized body
   structurally. They may reject unsupported structures or produce route
   metadata, but they must not synthesize missing RVV computation or invent
   performance scheduling.
6. The RVV route builder converts the realized `tcrv_rvv` body into a structured
   `TCRVEmitCLowerableRoute`.
7. The common route materializer lowers that route to EmitC: includes, ABI
   bindings, loop/control structure, `emitc.call_opaque` or equivalent calls for
   `__riscv_*` intrinsics, and ordinary C control flow.
8. EmitC export emits C/C++ source using `riscv_vector.h`. The source is then
   compiled on the RVV target and executed through the runtime/hardware evidence
   path.

Boundary rule: route builders may be transitional implementation locations
while Stage 1 cleanup is in progress, but the stable architecture must not make
route construction the place where target schedule and body semantics are
invented. The mature flow is vector-level selected body first, RVV realization
second, faithful route/emission third.

## Durable Promotion Plan And Status

The grill converged on the following durable promotion plan. In the authorized
follow-up step, these rules were promoted into the owning specs and canonical
supervisor prompt rather than being left only in this artifact note:

```text
1. `.trellis/spec/extension-plugins/rvv-plugin.md` carries:
   - RVV-owned vector-level execution IR definition;
   - complete route-supported RVV coverage target calibrated by structured
     kernel math/data-movement classes;
   - bounded i32m1 route as historical narrow path, not architecture;
   - selected-body realization as one linear RVV plugin-local lowering step;
   - performance config as plugin-local tuning consumed into real tcrv_rvv
     structure, not route/status/artifact authority.

2. `.trellis/spec/variant-pipeline/*` carries:
   - realization slot placement after selected lowering-boundary materialization
     and before emission planning;
   - common pass responsibility limited to registry/origin-plugin orchestration;
   - current Stage 2 scope limited to selected-variant internal realization,
     not global multi-variant autotuning.

3. `.trellis/spec/lowering-runtime/*` carries route-supported / executable
   artifact boundary wording:
   - EmitC / TCRVEmitCLowerableRoute is faithful emission, not performance
     decision source;
   - route builders consume realized tcrv_rvv structure and must not invent
     missing RVV semantics or schedule.

4. `scripts/codex_serial_supervisor_prompt.md` carries the loop steering rule:
   - Stage 1: replace/fail-close i32m1-as-route-authority;
   - Stage 2: expand route-supported RVV coverage on the corrected vector-level
     tcrv_rvv surface and implement selected-body realization;
   - do not switch early to Scalar/IME/Offload/high-level frontend work;
   - do not spend rounds on tests/reports/dashboards/status indexes as progress;
   - do not build high-level kernel ops, intrinsic wrappers, global autotune
     dashboards, or state-machine readiness markers.

5. One-shot steering before restarting the loop should point to the updated
   specs/prompt and the compact summary below.
```

The compact expression to preserve is:

```text
Stage 1:
  RVV route-authority replacement.

Stage 2:
  route-supported RVV coverage expansion on the corrected typed low-level
  tcrv_rvv execution surface, including RVV plugin-local selected-body
  realization for performance-sensitive vector-level bodies.

Stage 3:
  extension-family generalization / second-family integration after RVV
  maturity, proving a real non-RVV family uses the same common TCRV path.
  Its target is local, repeatable, common-path-verifiable extension-family
  integration: new families own their typed ops/boundary, capability, legality,
  cost, EmitC/runtime/target route, while core/common stays family-neutral.

Key boundary:
  tcrv.exec binds execution envelope and parameter roles;
  tcrv_rvv owns RVV-owned vector-level execution IR inside the selected variant,
  close to MLIR Vector abstraction level but bound to RVV execution;
  tcrv_rvv is not high-level Linalg/tensor/kernel IR and not an intrinsic
  wrapper dialect;
  RVVExtensionPlugin plus RVV route/body builders own RVV-specific legality,
  selected-body realization, and EmitC/intrinsic route construction;
  selected-body realization is one linear RVV plugin-local lowering step from
  pre-realized tcrv_rvv body to realized tcrv_rvv body, not a repeated
  optimization loop, status machine, dashboard, or readiness marker system;
  RVV performance config comes from RVV plugin-local tuning over capability,
  typed body facts, runtime SSA/ABI values, and optional hints/policy/profile;
  hints must be consumed into operative tcrv_rvv vector-level structure before
  they affect emission;
  current Stage 2 needs selected-variant internal realization, not global
  multi-variant autotuning or a tuning database;
  common passes own only target-neutral orchestration through registry hooks;
  selected-body realization is a reusable common slot with plugin-local
  semantics, not a common RVV pass;
  plugin construction fills extension-family bodies, not tcrv.exec core;
  common registry/boundary/EmitC/export are neutral materialization and
  packaging mechanics.
```
