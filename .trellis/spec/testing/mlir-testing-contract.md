# MLIR Testing Contract

## Scope

Testing must validate real TianChen-RV compiler behavior: dialect syntax,
verification, pass behavior, plugin interfaces, route-provider materialization,
common EmitC lowering, target artifact packaging, and runtime evidence when
claimed.

Tests are not dashboards, readiness states, artifact ledgers, or substitute
architecture authority.

## RVV Stage Rules

During RVV Stage 1:

- no positive legacy `RVVI32M1*` route-table tests;
- no positive `rvv-i32m1-*` object/header/bundle tests;
- no positive generated artifacts from `!tcrv_rvv.i32m1` or
  `tcrv_rvv.i32_*` helper-family authority;
- no positive RVV source-front-door/source-artifact bundle route tests;
- negative/fail-closed tests are required for stale legacy authority.

Positive RVV generated artifact tests are allowed only for corrected generic
typed `tcrv_rvv` routes after Stage1 reset:

```text
selected RVV variant
  -> explicit typed vector-level tcrv_rvv body
  -> RVV plugin legality / realization
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC
  -> target artifact
```

Use "explicit typed vector-level `tcrv_rvv` body"; do not use stale phrases
such as "explicit microkernel" or "selected RVV capacity metadata" as
authority.

## Required Test Types

### lit / FileCheck

Use for:

- dialect syntax and parsing;
- verifier success/failure;
- pass rewrite behavior;
- fail-closed diagnostics;
- selected-body realization visible in IR;
- route-provider handoff visible through emitted IR/mirrors;
- common EmitC materialization.

For positive selected-body target artifact fixtures, keep kernel and variant
symbol names concise. The generated C/EmitC function name is derived from the
kernel and selected variant symbols, and target export must fail closed when
that derived identifier is outside the bounded C/EmitC identifier contract.
Do not treat a shortened fixture symbol as route authority; it is only a
packaging constraint around an already provider-built route.

For generated-bundle dry-run tests, a `HARNESS` FileCheck prefix normally reads
the generated C harness source, not remote/runtime stdout. Assert source-level
facts such as function calls, expected expressions, pattern arrays, prototype
arguments, and `printf` format strings. Assert concrete runtime lines such as
`pattern=1 ok ...` only against remote output or evidence JSON fields that
actually contain stdout.

### C++ Tests

Use for:

- plugin registry APIs;
- route provider APIs;
- selected-body realization result objects;
- capability helper APIs;
- non-textual compiler utilities.

### Runtime / Hardware Evidence

Runtime, correctness, or performance claims require actual execution evidence.
For RVV claims, use real `ssh rvv` output. Local compile-only, static MLIR
checks, or Python smoke tests are not RVV runtime evidence.

Performance-comparison claims require more than executable artifact ABI proof.
For RVV comparisons against a handwritten or external baseline such as
llama.cpp, evidence must include:

- the baseline implementation identity and version/path;
- the generated TianChen-RV artifact identity;
- the same named `ssh rvv` target/profile for both sides;
- compile flags, input sizes, data initialization, warmup/repetition policy, and
  timing method;
- correctness checks before timing;
- raw output or evidence JSON that contains both correctness and timing results.

An artifact ABI closeout, generated-bundle dry-run, local compile-only test, or
single success marker is not performance evidence and must not be described as
llama.cpp parity.

## RVV Same-Target Measurement Evidence

### 1. Scope / Trigger

Use this contract when RVV work claims a measured same-target comparison for a
generated TianChen-RV RVV artifact against a handwritten, scalar, or external
baseline. This includes the bounded Gate 4 production-kernel campaign workflow
for `widening_product_reduce_dequantize_f32` and
`widening_product_reduce_dequant_clamp_f32`.

The measurement harness is evidence tooling only. It must reuse the production
compiler/export path for the generated object/header and must not become route,
dtype, compute, artifact-name, or q8/q4 authority.

### 2. Signatures

The current bounded command shape is:

```bash
python3 scripts/rvv_generated_bundle_same_target_measure.py \
  --op-kind widening_product_reduce_dequantize_f32 \
  --op-kind widening_product_reduce_dequant_clamp_f32 \
  --measure-count 257 \
  --measure-count 4096 \
  --measure-count 65536 \
  --ssh-target rvv
```

The command records evidence under an artifact root containing:

```text
evidence.json
<op>/same_target_measurement_evidence.json
<op>/remote_target_profile_stdout.txt
<op>/remote_measure_compile_stdout.txt
<op>/remote_measure_run_stdout.txt
<op>/rvv_generated_bundle_same_target_measure_<op>.c
```

Root evidence must expose:

```json
{
  "same_target_measurement": true,
  "ssh_evidence": true,
  "timing_method": "clock_gettime(CLOCK_MONOTONIC_RAW)",
  "measurement_config": {
    "input_sizes": [257, 4096, 65536],
    "warmup_count": 2,
    "repeat_count": 5,
    "measure_iterations": 8,
    "compile_flags": ["-O2", "-march=rv64gcv", "-mabi=lp64d", "-I."],
    "baseline_identities": {
      "...": "scalar-c-reference/..."
    }
  }
}
```

### 3. Contracts

- The generated side must be identified by selected input, selected variant,
  generated function name, object/header paths, and object/header hashes.
- The baseline side must have an explicit identity such as
  `scalar-c-reference/product-reduction-dequant-v1`; baseline names are
  comparator/oracle identities only, not RVV route authority.
- Both sides must compile and run on the same named `ssh rvv` target in the same
  measurement run.
- The harness must run correctness checks before timing every measured case and
  print a `CORRECTNESS_GUARD_BEFORE_TIMING` record before `MEASURE` records.
- Timing must use a declared monotonic timing method, explicit input sizes,
  warmups, repeats, and per-repeat raw timing records for both generated and
  baseline calls.
- Evidence must include target profile stdout, compile flags, raw run stdout,
  parsed timing records, and parsed summary records. A root `success` status is
  valid only when every requested op has remote compile/run success and parsed
  timing summaries.

### 4. Validation & Error Matrix

- Missing generated object/header identity or hash -> evidence is incomplete.
- Missing baseline identity -> comparison evidence is incomplete.
- Generated and baseline paths are not run through the same `ssh rvv` target and
  compile command family -> not same-target evidence.
- Missing target profile, compile flags, timing method, input sizes, warmups,
  repeats, or iterations -> not performance-comparison evidence.
- A `MEASURE` record appears without a preceding correctness guard for that case
  -> timing is invalid for performance claims.
- Remote compile/run failure or missing `PASS op=... measurement` marker ->
  evidence is blocked.
- Parsed timing records or summary records are empty -> evidence is blocked even
  if raw stdout exists.

### 5. Good/Base/Bad Cases

- Good: generated pre-realized RVV artifact -> same-target scalar C baseline ->
  correctness guard -> raw `CLOCK_MONOTONIC_RAW` timing records -> evidence JSON
  with target profile and parsed summaries.
- Base: dry-run measurement tests generate and FileCheck the measurement harness
  and evidence schema without claiming runtime or performance evidence.
- Bad: generated-bundle ABI success, dry-run harness generation, or a single
  remote PASS marker is described as same-target performance evidence.

### 6. Tests Required

- Script self-test must check harness generation keeps baseline identity,
  generated function calls, timing method, warmup/repeat/iteration loops,
  correctness-before-timing, and stdout parsing.
- lit/FileCheck dry-run coverage must assert root/per-op evidence fields and the
  generated C harness structure for each measured op kind.
- Non-dry-run `ssh rvv` evidence must be collected for any runtime or timing
  claim and must preserve raw target profile, compile stdout, and run stdout.

### 7. Wrong vs Correct

Wrong:

```text
generated bundle runs once on ssh rvv
  -> evidence JSON says success
  -> claim same-target measurement or llama.cpp parity
```

Correct:

```text
provider-validated generated RVV object/header
  -> same-target baseline with explicit identity
  -> correctness guard before timing
  -> raw generated and baseline timings on ssh rvv
  -> parsed summary records in evidence JSON
```

For RVV generated-bundle evidence over runtime `n`, memory-writing routes must
check both active-lane arithmetic/data movement and guard/tail preservation.
Harnesses should initialize output storage beyond `n` (or inactive/passthrough
lanes when applicable) with sentinels and fail if the generated route writes
outside the runtime element count. These sentinel checks are evidence quality
guards only; they do not become route, dtype, or artifact authority.

When a generated-bundle harness snapshots input, accumulator, passthrough, or
seed buffers before calling the generated route, the harness must compare those
snapshots after execution, fail with a route-specific diagnostic if any
snapshotted value changed unexpectedly, and free the snapshot buffers on every
return path. The corresponding dry-run `HARNESS` FileCheck should assert both
the mutation diagnostic path and the success marker such as
`source_preserved` or `accumulator_preserved`. Allocating `*_before` buffers
without comparing them is not preservation evidence.

For routes whose expected result depends on snapshotted inputs, accumulators,
passthrough values, gather sources, payload buffers, or seed buffers, the
expected-value computation and mismatch diagnostics must read the pre-call
snapshots, not the post-call live buffers. A harness that first computes
`expected` from post-call `src`, `payload`, or `acc` and only later compares
those arrays against `*_before` snapshots can mask a generated route that
mutates an input and then produces output consistent with the mutated value.
Dry-run `HARNESS` FileCheck should assert the snapshot-backed expression, for
example `acc_before[index] + gather_src_before[index] * payload_before[index]`,
whenever preservation is part of the evidence claim.

For multi-pattern generated-bundle harnesses, the pattern dimension is part of
the evidence surface. If `run_case` accepts a pattern argument, `main` must
iterate every required pattern, pass the pattern into `run_case`, and print the
pattern set in the final success marker, such as `patterns=0,1`. Dry-run
FileCheck and script self-tests must check that pattern loop shape, so a
single-count call such as `run_case(counts[index])` cannot pass local dry-run
while failing real `ssh rvv` compilation.

## Selected Dispatch Generated-Bundle Evidence

### 1. Scope / Trigger

Use this contract when generated-bundle evidence claims executable correctness
for a selected RVV dispatch case with a `tcrv.exec.dispatch` fallback envelope.
It applies after the RVV provider and target artifact validator already accept
the selected route.

### 2. Signatures

Per-op evidence JSON should expose:

```json
"selected_dispatch_bundle_boundary": {
  "artifact_metadata_role": "mirror-only-after-provider-route-and-selected-dispatch-validation",
  "selected_variant": "<rvv selected variant symbol without @>",
  "selected_dispatch_case_mirror": "selected_dispatch_case_mirror:@<rvv case>;...",
  "selected_dispatch_fallback_mirror": "selected_dispatch_fallback_mirror:@<fallback>;...",
  "exec_abi_bindings": "<selected-envelope ABI binding summary>",
  "runtime_abi_order": "<provider runtime ABI order>",
  "route_operand_binding_plan": "<provider operand binding plan>",
  "route_operand_binding_operands": "<provider operand binding summary>",
  "provider_supported_mirror": "<provider support mirror>",
  "object_header_metadata_agree": true,
  "runtime_counts_are_execution_cases_not_dispatch_authority": true
}
```

### 3. Contracts

- The selected dispatch case/fallback values must be read from object/header
  bundle metadata that already passed target artifact validation.
- Object and header records must agree on selected dispatch case, selected
  fallback, exec ABI bindings, runtime ABI order, route operand binding plan and
  operands, and provider support mirror.
- The evidence field is a mirror-only runtime evidence summary. It must not
  define route support, fallback semantics, ABI order, dtype, or computation.
- Runtime counts, scalar thresholds, patterns, and remote PASS markers are
  execution cases only; they do not authorize dispatch selection.

### 4. Validation & Error Matrix

- Missing selected dispatch case/fallback expectation for a route that claims
  selected-dispatch evidence -> evidence failure.
- Object/header metadata missing selected dispatch case or fallback mirror ->
  evidence failure before `ssh rvv` correctness is claimed.
- Object/header selected dispatch/fallback mirrors disagree with expected
  provider/target facts -> evidence failure.
- Object/header disagree on ABI binding, ABI order, provider support, or route
  operand binding facts -> evidence failure.
- Remote compile/run failure -> evidence is blocked; do not claim executable
  correctness.

### 5. Good/Base/Bad Cases

- Good: actual `tcrv.exec.dispatch` case/fallback -> selected typed RVV body ->
  provider-built route -> target artifact validation -> generated bundle ->
  per-op evidence records selected dispatch/fallback mirrors and `ssh rvv`
  PASS output.
- Base: dry-run generated-bundle evidence validates the same object/header
  metadata and harness source without claiming runtime correctness.
- Bad: a generated-bundle evidence file reports remote PASS but omits selected
  dispatch/fallback mirrors while the task claims selected-dispatch executable
  coverage.

### 6. Tests Required

- Dry-run FileCheck must assert the selected dispatch bundle boundary fields
  for each selected-dispatch generated-bundle route under test.
- Script self-tests or equivalent local checks must ensure fake bundle metadata
  includes selected dispatch/fallback mirrors when the expectation declares
  them.
- Existing target artifact negative tests must continue to reject missing or
  stale selected dispatch case, fallback, runtime guard, ABI binding, ABI order,
  provider support, or route operand binding mirrors.
- Runtime correctness claims must include non-dry-run `ssh rvv` evidence with
  the generated object, header, and external ABI harness.

### 7. Wrong vs Correct

Wrong:

```text
selected_dispatch_case_mirror string in a test name
  -> generated bundle runs
  -> claim selected-dispatch executable boundary
```

Correct:

```text
actual tcrv.exec.dispatch case/fallback
  -> RVV provider route and target artifact validation
  -> object/header bundle metadata agree on selected dispatch/fallback facts
  -> generated harness compiles and runs on ssh rvv
  -> evidence records PASS output and mirror-only selected_dispatch_bundle_boundary
```

## F32 Clamp/Select Generated-Bundle Evidence

### 1. Scope / Trigger

Use this contract when generated-bundle evidence claims executable correctness
for the selected pre-realized RVV f32 clamp/select route.

### 2. Signatures

The bounded command shape is:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --op-kind f32_clamp_select
```

The generated external ABI must remain:

```c
void tcrv_emitc_pre_realized_f32_clamp_select_kernel_pre_realized_rvv_f32_clamp_select(
    const float *input, float lower_bound, float upper_bound,
    float *out, size_t n);
```

### 3. Contracts

- Authority is the selected typed f32 `tcrv_rvv` body/config/runtime-bound
  facts, not route ids, artifact names, ABI strings, metadata mirrors, or
  harness constants.
- Materialized IR must prove the pre-realized body was consumed before
  emission and realized into `setvl`, input `load`, lower/upper `splat`,
  lower/upper `compare`, lower/upper `select`, and `store`.
- Evidence JSON must expose the lower/upper bound ABI roles, ordered bound
  relation, provider mirror, route operand binding, emitted C++ boundary, and
  runtime AVL/VL boundary.
- Target artifact/header metadata must mirror the provider-derived route facts
  exactly for `tcrv_rvv.runtime_abi_order`,
  `tcrv_rvv.route_operand_binding_plan`,
  `tcrv_rvv.route_operand_binding_operands`,
  `tcrv_rvv.computed_mask_select_route_family_plan`,
  `tcrv_rvv.computed_mask_select_mask_producer_source`,
  `tcrv_rvv.provider_supported_mirror`,
  `tcrv_rvv.required_header_declarations`, `tcrv_rvv.c_type_mapping`,
  `tcrv_rvv.lower_bound_role`, `tcrv_rvv.upper_bound_role`,
  `tcrv_rvv.lower_bound_c_type`, `tcrv_rvv.upper_bound_c_type`,
  `tcrv_rvv.bound_order`, `tcrv_rvv.clamp_relation`, and
  `tcrv_rvv.secondary_compare_predicate_kind`.
- Runtime bound pairs and counts are execution cases only; they must not define
  dtype, SEW, LMUL, policy, bound semantics, or route support.

### 4. Validation & Error Matrix

- Missing lower/upper runtime ABI roles -> evidence failure.
- Missing pre-realized-body consumption -> evidence failure.
- Generated C/C++ omits either bound splat, compare, select, or runtime VL
  operand -> evidence failure.
- Object/header metadata disagree on bound roles, bound order, provider mirror,
  or route operand binding -> evidence failure.
- Object/header metadata disagree on bound C types, route-family plan, computed
  mask producer source, secondary compare predicate, required headers, or C type
  mapping -> evidence failure.
- `ssh rvv` compile/run failure -> report blocked evidence and do not claim
  executable correctness.

### 5. Good/Base/Bad Cases

- Good: selected f32 body/config/runtime-bound facts -> RVV plugin-local
  realization -> provider-built clamp/select route -> common EmitC -> generated
  artifact -> harness compares against a host clamp oracle.
- Base: dry-run evidence proves materialization, route/export metadata, emitted
  C++ shape, and harness source without claiming runtime correctness.
- Bad: harness chooses semantics from `f32_clamp_select` as a string or accepts
  metadata/header success without running the generated object on `ssh rvv`.

### 6. Tests Required

- Script self-test must cover fake-bundle bound metadata and generated harness
  source checks.
- Dry-run evidence must generate the bundle and harness before remote
  execution.
- Target artifact dry-run or lit/FileCheck must include stale-mirror negative
  checks for provider mirror, runtime ABI order, route operand binding summary,
  route-family plan, lower/upper bound role or C type, bound order, secondary
  predicate, and required header/type-mapping mirrors.
- Runtime evidence must run on `ssh rvv` with multiple counts, at least two
  ordered nontrivial bound pairs, below/in/above-bound source patterns, explicit
  f32 tolerance, source preservation, and output tail sentinel preservation.

### 7. Wrong vs Correct

Wrong:

```text
artifact/header says f32_clamp_select
  -> claim clamp/select executable support
```

Correct:

```text
selected typed f32 clamp/select body
  -> RVV realization before emission
  -> provider-built route and common EmitC
  -> generated bundle dry-run
  -> ssh rvv harness with host/reference tolerance and source/tail checks
```

## Mask/Tail Policy Generated-Bundle Evidence

### 1. Scope / Trigger

Use mask/tail policy generated-bundle evidence whenever an RVV generated-bundle
test claims executable masked behavior, inactive-lane behavior, or tail-policy
correctness for a route-supported typed `tcrv_rvv` path.

### 2. Signatures

The per-op evidence JSON should expose a bounded summary key such as:

```json
"mask_tail_policy_boundary": {
  "authority": "provider-derived typed tcrv_rvv mask/policy body/config/runtime facts",
  "artifact_metadata_role": "mirror-only-after-provider-route",
  "tail_policy": "undisturbed",
  "mask_policy": "undisturbed",
  "selected_mask_abi": {"c_name": "mask", "role": "mask-input-buffer"},
  "materialized_body": {},
  "emitted_cpp": {},
  "route_metadata": {},
  "runtime_counts_are_execution_cases_not_policy_authority": true
}
```

### 3. Contracts

- `authority` must identify typed `tcrv_rvv` body/config/runtime facts as the
  source of route support.
- `artifact_metadata_role` must make mirror-only status explicit.
- `selected_mask_abi` must identify the runtime mask operand and role when the
  route consumes an external mask.
- `materialized_body` must prove the typed/pre-realized body was consumed into
  realized mask/policy-carrying `tcrv_rvv` ops.
- `emitted_cpp` must prove generated RVV C/C++ uses the expected mask operand,
  predicate construction, masked intrinsic form, and runtime VL operands.
- `route_metadata` may mirror provider/export metadata only after provider route
  construction. It must not be used as route authority.
- Runtime counts are execution cases only. They must not define mask/tail
  policy, dtype, route support, or artifact authority.
- For `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`,
  `mask_tail_policy_boundary` must also expose a
  `composite_resource_selection` object that mirrors provider-owned
  `tcrv_rvv.composite_resource.*` facts: candidate set, selected candidate,
  selection reason, legality scope, operation, memory form, SEW, LMUL,
  tail/mask policy, VL policy, accumulator layout, unroll factor,
  pipeline/prefetch intent, `vsetvl` region count, peak live vector groups,
  vector register budget, runtime AVL source, runtime ABI order, target
  capability mirrors, legality, and rejection reason. These fields are
  evidence mirrors only after provider route construction; they must not become
  route authority.

### 4. Validation & Error Matrix

- Missing selected mask ABI for an externally masked route -> evidence failure.
- Missing realized mask op or masked compute/store op -> evidence failure.
- Generated C/C++ omits the mask operand in the masked intrinsic -> evidence
  failure.
- Object/header mirror metadata disagree on tail/mask policy or mask role ->
  evidence failure.
- For `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`, object/header
  mirror metadata missing or disagreeing on any
  `tcrv_rvv.composite_resource.*` key -> evidence failure.
- Descriptor, direct-C/source-export, route-id, artifact-name, or test-name
  residue is required to explain policy -> evidence failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: typed selected body carries mask/policy facts -> RVV plugin realizes or
  validates those facts -> provider emits masked route -> evidence mirrors the
  facts and harness checks inactive/tail preservation.
- Base: non-masked routes omit `mask_tail_policy_boundary` and keep ordinary
  runtime AVL/VL or typed-config evidence.
- Bad: evidence infers mask/tail policy from route id, ABI string, artifact
  name, exact intrinsic spelling, or harness constants.

### 6. Tests Required

- lit/FileCheck for the generated-bundle dry-run must check representative
  `mask_tail_policy_boundary` fields and mirror metadata.
- lit/FileCheck for the composite gather-MAcc-scatter generated-bundle dry-run
  must check representative `composite_resource_selection` fields and raw
  `tcrv_rvv.composite_resource.*` mirror metadata.
- Provider or C++ API tests must check the route facts before common EmitC
  materialization when textual MLIR cannot fully prove the boundary.
- Runtime RVV claims must include `ssh rvv` output and inactive/tail sentinel
  checks for the claimed masked route.

### 7. Wrong vs Correct

Wrong:

```text
artifact metadata says tail_policy=undisturbed
  -> claim masked route support
```

Correct:

```text
typed body/config/runtime mask facts
  -> RVV plugin legality/realization
  -> provider-built masked route
  -> common EmitC
  -> mirror metadata plus executable harness evidence
```

## Computed-Mask Standalone Reduction Generated-Bundle Evidence

### 1. Scope / Trigger

Use computed-mask standalone reduction generated-bundle evidence whenever an
RVV generated-bundle test claims executable correctness for
`computed_mask_standalone_reduce_add`, `computed_mask_standalone_reduce_min`,
or `computed_mask_standalone_reduce_max`.

### 2. Signatures

The bounded command shape is:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --op-kind computed_mask_standalone_reduce_add
```

The provider-owned runtime ABI order for the vector computed-mask standalone
reduction family is:

```text
cmp_lhs,cmp_rhs,src,acc,out,n
```

### 3. Contracts

- Authority is the selected typed `tcrv_rvv` body/config/runtime facts, not
  route ids, artifact names, ABI strings, metadata mirrors, exact intrinsic
  spelling, or harness constants.
- Evidence must mirror the provider-derived `reduction_kind`, compare
  predicate, mask role/source/memory form, inactive-lane requirement,
  accumulator/result layout, runtime AVL/VL boundary, route operand binding,
  required headers, C type mapping, and statement plan.
- For `add`, inactive lanes must be neutralized with zero before the horizontal
  reduction. For `min`/`max`, inactive lanes must use the operation-specific
  min/max neutral literal.
- The generated C/C++ must show compare-mask construction, inactive neutral
  splat, mask merge, scalar seed splat, standalone reduction intrinsic, and
  scalar lane-0 store using runtime VL facts.
- The harness must check ordinary mixed active/inactive cases and an
  all-inactive-mask oracle where `out[0]` remains equal to the scalar seed.
- Runtime counts are execution cases only. They must not define mask policy,
  reduction kind, dtype, route support, or artifact authority.

### 4. Validation & Error Matrix

- Missing compare-mask producer facts -> evidence failure.
- Missing or stale `reduction_kind` -> evidence failure.
- Missing inactive-lane requirement or wrong inactive neutral literal ->
  evidence failure.
- Missing compare/source/accumulator/output/runtime `n` ABI role or wrong ABI
  order -> evidence failure.
- Generated C/C++ omits runtime VL in compare, merge, reduction, or scalar
  store -> evidence failure.
- A multi-lane runtime evidence case lacks both active and inactive mask lanes
  -> evidence failure.
- The all-inactive oracle changes `out[0]` away from the scalar seed ->
  evidence failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: selected typed body carries compare-mask, source, scalar seed/result,
  policy, and runtime facts -> RVV plugin realizes or validates those facts ->
  provider emits computed-mask standalone reduction route -> generated evidence
  mirrors provider facts -> harness checks active lanes, inactive neutral lanes,
  all-inactive seed preservation, source preservation, and tail sentinels.
- Base: dry-run evidence proves provider facts, generated bundle, emitted C++
  shape, and harness source without claiming runtime correctness.
- Bad: evidence infers `reduction_kind`, mask behavior, inactive behavior, or
  runtime ABI order from op kind strings, artifact filenames, exact intrinsic
  spelling, or harness constants.

### 6. Tests Required

- lit/FileCheck for route/artifact fixtures must check representative
  `reduction_kind`, mask role/source/memory form, inactive-lane requirement,
  runtime ABI order, route operand binding, and header/type mirrors.
- Target artifact tests must reject stale provider or candidate mirrors for
  reduction kind, mask facts, inactive-lane contract, runtime AVL/order,
  operand binding, header/type facts, and statement-plan facts before artifact
  acceptance.
- Runtime RVV claims must include `ssh rvv` output over multiple runtime
  counts, including `n=0`, at least one tail case, and mixed active/inactive
  mask cases. Harness output must also include an all-inactive-mask oracle and
  source/tail preservation checks.

### 7. Wrong vs Correct

Wrong:

```text
artifact name says computed_mask_standalone_reduce_add
  -> harness assumes add reduction and inactive zeroing
```

Correct:

```text
typed body/config/runtime compare-mask and reduction facts
  -> RVV provider derives reduction kind and inactive neutralization
  -> Common EmitC carries provider payload
  -> target mirrors are checked
  -> ssh rvv harness checks mixed masks and all-inactive seed preservation
```

## Plain MAcc Generated-Bundle Evidence

### 1. Scope / Trigger

Use plain MAcc generated-bundle evidence whenever an RVV generated-bundle test
claims route-supported or executable correctness for a selected typed
`tcrv_rvv.macc` vector-vector multiply-accumulate body.

This contract applies to the bounded plain route:

```text
macc_add
```

Scalar-broadcast, computed-mask, runtime-scalar computed-mask, widening MAcc,
widening dot-reduction, and high-level matmul/frontend routes use their own
contracts.

### 2. Signatures

The bounded command shape is:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --op-kind macc_add
```

For pre-realized selected bodies:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --op-kind macc_add
```

The provider-owned runtime ABI order for the plain vector-vector MAcc family
is:

```text
lhs,rhs,acc,out,n
```

### 3. Contracts

- Authority is the selected typed `tcrv_rvv.macc` body/config/runtime facts,
  not route ids, artifact names, ABI strings, metadata mirrors, exact
  intrinsic spelling, fixture names, or harness constants.
- Evidence must mirror source/result dtype, SEW, LMUL, tail/mask policy,
  arithmetic kind, accumulator layout, result layout, runtime AVL/VL
  boundary, route operand binding, required headers, C type mapping, target
  leaf profile, and `provider_supported_mirror`.
- The selected ABI role summary must bind `lhs`, `rhs`, accumulator input,
  output, and runtime `n` in the provider-owned ABI order.
- The generated C/C++ must show runtime-VL-controlled lhs load, rhs load,
  accumulator load, multiply-accumulate compute, and output store. The MAcc
  operand order and store pointer must be checked as emitted code facts, not
  inferred from the op kind string.
- The harness must check vector-vector multiply, accumulator contribution,
  source preservation, output tail sentinel preservation, and `n = 0` loop
  skip behavior.
- Runtime counts and data patterns are execution cases only. They must not
  define arithmetic kind, dtype, route support, accumulator layout, result
  layout, policy, or artifact authority.

### 4. Validation & Error Matrix

- Missing `lhs`, `rhs`, accumulator, output, or runtime `n` ABI role ->
  evidence failure.
- Wrong ABI order or missing `abi`/`hdr` binding marker for any exported
  parameter -> evidence failure.
- Missing or stale arithmetic kind, accumulator layout, result layout,
  runtime AVL/VL boundary, header/type mapping, target profile, or
  provider-supported mirror -> evidence failure.
- Generated C/C++ omits runtime VL in lhs load, rhs load, accumulator load,
  MAcc, or store -> evidence failure.
- Harness does not distinguish add-only, multiply-only, missing-accumulator, or
  wrong-operand-order behavior -> evidence failure.
- A route id, artifact name, ABI string, exact intrinsic spelling, descriptor,
  source-front-door marker, or legacy i32 helper is required to explain MAcc
  semantics -> evidence failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: typed selected body carries lhs/rhs/acc/out/n roles, `kind = add`,
  SEW/LMUL/policy, accumulator/result layout, and runtime facts -> RVV plugin
  realizes or validates the body -> provider emits the plain MAcc route ->
  generated evidence mirrors provider facts -> harness checks
  `acc + lhs * rhs` and tail preservation.
- Base: dry-run evidence proves materialization, route/export metadata,
  emitted C++ shape, and harness source without claiming runtime correctness.
- Bad: evidence infers arithmetic kind, accumulator behavior, operand order,
  or runtime ABI order from `macc_add`, artifact filenames, exact intrinsic
  spelling, or harness constants.

### 6. Tests Required

- lit/FileCheck for route/artifact fixtures must check representative
  arithmetic kind, accumulator/result layout, runtime ABI order, route operand
  binding, provider-supported mirror, and header/type mirrors.
- Target artifact tests must reject stale provider or candidate facts for
  arithmetic kind, accumulator/result layout, runtime AVL/order, operand
  binding, header/type facts, provider-supported mirror, and cross-family MAcc
  residue before artifact acceptance.
- Generated-bundle dry-run coverage must check `multiply_accumulate_boundary`
  fields and provider route facts, including `macc_arithmetic_kind`.
- Runtime RVV claims must include `ssh rvv` output over multiple runtime
  counts, including `n = 0`, at least one tail case, and at least two data
  patterns that distinguish multiply, accumulator contribution, source
  preservation, and tail preservation.

### 7. Wrong vs Correct

Wrong:

```text
artifact name or route id says macc_add
  -> harness expects acc + lhs * rhs behavior
```

Correct:

```text
selected typed tcrv_rvv.macc body/config/runtime facts
  -> RVV provider derives arithmetic kind, layout, ABI, type, and route facts
  -> Common EmitC carries provider payload
  -> target mirrors are checked
  -> generated/ssh rvv harness checks acc + lhs * rhs behavior
```

## Computed-Mask MAcc Generated-Bundle Evidence

### 1. Scope / Trigger

Use computed-mask MAcc generated-bundle evidence whenever an RVV
generated-bundle test claims executable multiply-accumulate correctness under a
compare-produced mask. This evidence is required when active lanes compute
`acc + lhs * rhs` and inactive lanes preserve an explicit pass-through value
for a route-supported typed `tcrv_rvv` path.

### 2. Signatures

The per-op evidence JSON should expose a bounded summary key such as:

```json
"computed_masked_macc_boundary": {
  "authority": "provider-derived typed tcrv_rvv body/config/runtime facts",
  "artifact_metadata_role": "mirror-only-after-provider-route",
  "compare_predicate_kind": "slt",
  "mask_role_source": {
    "role": "predicate-mask-produced-by-compare",
    "source": "compare-lhs-rhs-same-vl-scope",
    "memory_form": "computed-mask"
  },
  "active_lane_contract": "out[i] = acc[i] + lhs[i] * rhs[i]",
  "inactive_lane_contract": "out[i] preserves accumulator/pass-through",
  "selected_abi_roles": {},
  "statement_plan": {},
  "materialized_body": {},
  "emitted_cpp": {},
  "route_metadata": {},
  "runtime_counts_are_execution_cases_not_macc_or_mask_authority": true
}
```

### 3. Contracts

- `authority` must identify typed `tcrv_rvv` body/config/runtime facts as the
  source of route support.
- `compare_predicate_kind`, mask role/source, mask memory form, MAcc operation,
  accumulator layout, result layout, dtype, SEW, LMUL, policy, and runtime
  AVL/VL must be mirrored from provider-validated route facts.
- `selected_abi_roles` must identify compare operands, payload operands,
  accumulator input, output, and runtime `n` in the selected ABI order.
- For runtime-scalar computed-mask MAcc, the selected ABI role summary must
  identify `rhs_scalar` as the `rhs-scalar-value` runtime ABI input, scalar
  splat source, and compare RHS for the same-VL mask producer. Evidence must
  not infer this scalar from a constant, ABI name, route id, artifact name, or
  harness convention.
- `statement_plan` and `emitted_cpp` must prove the compare mask, active MAcc,
  merge/pass-through, store, loop induction, and runtime VL operands use the
  provider-selected operand bindings.
- `materialized_body` must prove the typed or pre-realized selected body was
  consumed into the realized `tcrv_rvv` body before route construction.
- `route_metadata` may mirror provider/export metadata only after provider
  route construction. It must not be used as route authority.
- Runtime counts are execution cases only. They must not define mask predicate,
  accumulator layout, MAcc operation, dtype, route support, or artifact
  authority.

### 4. Validation & Error Matrix

- Missing compare-mask producer facts -> evidence failure.
- Missing payload lhs/rhs, accumulator, output, or runtime `n` ABI role ->
  evidence failure.
- Generated C/C++ omits runtime VL in compare, MAcc, merge, or store sequence
  -> evidence failure.
- Generated C/C++ uses MAcc operands in an order that does not match the
  provider-owned operand-binding facts -> evidence failure.
- Generated C/C++ stores active MAcc directly and drops inactive
  pass-through/merge semantics -> evidence failure.
- Descriptor, direct-C/source-export, route-id, artifact-name, ABI-string,
  script, test-name, common EmitC, or legacy i32 residue is required to explain
  mask, accumulator, or MAcc semantics -> evidence failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: typed selected body carries compare-mask, payload, accumulator,
  result-layout, policy, and runtime facts -> RVV plugin realizes or validates
  those facts -> provider emits computed-mask MAcc route -> evidence mirrors
  the facts and harness checks active lanes, inactive pass-through, and tail
  preservation.
- Base: plain MAcc routes omit `computed_masked_macc_boundary` and keep their
  own accumulator/runtime generated-bundle evidence.
- Bad: evidence infers predicate kind, accumulator behavior, operand order, or
  inactive-lane behavior from route id, ABI string, artifact name, exact
  intrinsic spelling, script constants, or harness constants.

### 6. Tests Required

- lit/FileCheck for the generated-bundle dry-run must check representative
  `computed_masked_macc_boundary` fields and mirror metadata.
- Runtime-scalar computed-mask MAcc dry-run coverage must also check the
  runtime scalar values, route-operand binding for `rhs_scalar` as
  `abi|splat|cmp-rhs`, and the generated harness call that passes the scalar
  ABI argument into the same generated artifact.
- Runtime-scalar computed-mask MAcc runtime and dry-run harnesses must exercise
  at least two runtime scalar values and at least two lhs/rhs/accumulator data
  patterns. The scalar loop proves the scalar/splat compare boundary; the data
  pattern loop proves vector-vector MAcc payload and accumulator behavior are
  not inferred from one fixed harness convention.
- Target artifact tests must check provider-owned predicate, mask role/source,
  accumulator/result layout, runtime AVL/VL, typed config, ABI roles, and
  provider-supported mirror fields before relying on generated artifacts.
- Runtime RVV claims must include `ssh rvv` output over multiple runtime counts,
  including `n=0`, at least one tail case, and mixed active/inactive mask cases.
- Harness checks must distinguish `acc + lhs * rhs` from add-only or mul-only
  behavior on active lanes and must verify inactive accumulator/pass-through and
  tail sentinel preservation.

### 7. Wrong vs Correct

Wrong:

```text
artifact name or route id says computed_masked_macc_add
  -> harness expects masked MAcc behavior
```

Correct:

```text
typed body/config/runtime compare-mask and accumulator facts
  -> RVV plugin legality/realization
  -> provider-built computed-mask MAcc route
  -> common EmitC
  -> mirror metadata plus executable harness evidence
```

## Widening-MAcc Generated-Bundle Evidence

### 1. Scope / Trigger

Use widening-MAcc generated-bundle evidence whenever an RVV generated-bundle
test claims executable `widening_macc_add` correctness for a route-supported
typed `tcrv_rvv` path. This evidence is required for both explicit selected
bodies and selected pre-realized bodies after RVV plugin-local realization.

### 2. Signatures

The per-op evidence JSON should expose a bounded summary key such as:

```json
"widening_macc_boundary": {
  "authority": "provider-derived typed tcrv_rvv widening-MAcc body/config/runtime facts",
  "target_artifact_validator": "target-owned widening-macc-contraction route-family validator",
  "artifact_metadata_role": "mirror-only-after-provider-route",
  "direct_pre_realized_route_entry_supported": false,
  "selected_source_abi": {},
  "provider_route_facts": {},
  "target_validator_consumed_facts": [],
  "route_metadata": {},
  "runtime_counts_are_execution_cases_not_widening_macc_authority": true
}
```

### 3. Contracts

- `authority` must identify typed `tcrv_rvv` body/config/runtime facts as the
  source of route support.
- `selected_source_abi` must identify `lhs`, `rhs`, `acc`, `out`, and runtime
  `n` in provider-validated ABI order. `lhs` and `rhs` are source-width inputs;
  `acc` and `out` are widened accumulator/result buffers.
- The evidence must mirror source dtype, accumulator/result dtype, SEW, LMUL,
  policy, memory form, runtime AVL/VL, setvl placement, operand binding,
  accumulator layout, result layout, widening relation, and artifact ABI order
  from provider route facts after route construction.
- `target_validator_consumed_facts` must identify the target-owned
  `widening-macc-contraction` validator as the consumer of provider facts and
  candidate mirror checks.
- `route_metadata` may mirror provider/export metadata only after provider route
  construction. It must not be used as route authority.
- Runtime counts are execution cases only. They must not define dtype,
  widening relation, route support, artifact authority, or direct route-entry
  support.

### 4. Validation & Error Matrix

- Missing or stale source/result dtype relation -> evidence failure.
- Missing `lhs`, `rhs`, accumulator, output, or runtime `n` ABI role ->
  evidence failure.
- Missing widening relation, accumulator layout, result layout, runtime
  AVL/VL, route operand binding, provider-supported mirror, or target-validator
  consumption summary -> evidence failure.
- Generated C/C++ omits runtime VL on source loads, accumulator load,
  widening-MAcc compute, or store -> evidence failure.
- Generated C/C++ uses MAcc operands in an order that disagrees with
  provider-owned operand-binding facts -> evidence failure.
- A multi-lane runtime evidence case lacks positive products, negative
  products, nonzero accumulators, or products outside the source-width range ->
  evidence failure.
- Descriptor, direct-C/source-export, route-id, artifact-name, ABI-string,
  script, test-name, common EmitC, exact intrinsic spelling, direct route entry,
  or legacy i32 residue is required to explain widening-MAcc semantics ->
  evidence failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: typed selected body carries i16 source, i32 accumulator/result,
  widening relation, runtime `n`, and policy facts -> RVV plugin validates or
  realizes the body -> provider emits the widening-MAcc route -> target-owned
  validator consumes provider facts -> generated evidence mirrors the facts and
  harness checks signed widening products, accumulation, and tail preservation.
- Base: unrelated MAcc, reduction, conversion, or dot-reduction routes omit
  `widening_macc_boundary` and keep their own generated-bundle evidence.
- Bad: evidence infers widening behavior from route id, artifact filename,
  exact intrinsic spelling, script constants, or the pre-realized fixture name.

### 6. Tests Required

- lit/FileCheck for explicit and pre-realized generated-bundle dry-runs must
  check representative `widening_macc_boundary` fields and mirror metadata.
- Target artifact tests must check provider-owned runtime ABI order/roles,
  operand binding, type relation, accumulator/result layout, provider-supported
  mirror, and stale non-widening-MAcc fact rejection before artifact acceptance.
- Runtime RVV claims must include `ssh rvv` output over multiple runtime counts,
  including `n=0`, `n=1`, exact-VL, tail, and stress cases.
- Harness checks must distinguish signed widening multiply-accumulate from
  add-only, non-widening, missing-accumulator, wrong sign-extension, and
  tail-overwrite behavior. Explicit and pre-realized selected-body modes must
  use input patterns that produce source-width-overflowing products in
  multi-lane runtime cases.

### 7. Wrong vs Correct

Wrong:

```text
pre-realized fixture name says widening_macc_add
  -> harness uses small i16 products and claims widening correctness
```

Correct:

```text
typed body/config/runtime widening-MAcc facts
  -> RVV plugin legality/realization
  -> provider-built contraction route
  -> target-owned widening-MAcc validator
  -> mirror metadata plus executable harness evidence with widening products
```

## Widening Dot-Reduction Generated-Bundle Evidence

### 1. Scope / Trigger

Use widening dot-reduction generated-bundle evidence whenever an RVV
generated-bundle test claims executable `widening_dot_reduce_add` or
`strided_input_widening_dot_reduce_add` correctness for a route-supported typed
`tcrv_rvv` path. This evidence is required for explicit selected bodies and
selected pre-realized bodies after RVV plugin-local realization. Computed-mask
widening dot-reduction routes use their own computed-mask evidence boundary.

### 2. Signatures

The per-op evidence JSON should expose a bounded summary key such as:

```json
"widening_dot_reduction_boundary": {
  "authority": "provider-derived typed tcrv_rvv widening dot-reduction body/config/runtime facts",
  "target_artifact_validator": "RVVTargetArtifactRouteFamilyValidation.cpp:widening-dot-reduction target-owned consumer",
  "artifact_metadata_role": "mirror-only-after-provider-route",
  "direct_pre_realized_route_entry_supported": false,
  "selected_source_abi": {},
  "provider_route_facts": {},
  "target_validator_consumed_facts": [],
  "route_metadata": {},
  "runtime_counts_are_execution_cases_not_widening_dot_authority": true
}
```

### 3. Contracts

- `authority` must identify typed `tcrv_rvv` body/config/runtime facts as the
  source of route support.
- `selected_source_abi` must identify `lhs`, `rhs`, `acc`, `out`, and runtime
  `n` in provider-validated ABI order. Strided-input routes must also identify
  `lhs_stride` and `rhs_stride`.
- The evidence must mirror source dtype, accumulator/result dtype, SEW, LMUL,
  policy, runtime AVL/VL, scalar store VL, route operand binding, dot-product
  relation, accumulator layout, result layout, required headers, C type mapping,
  provider support mirror, and artifact ABI order from provider route facts
  after route construction.
- Strided-input evidence must mirror source memory form, destination memory
  form, strided memory layout, stride ABI sources, and strided load intrinsic.
  Unit-stride evidence must make stale strided-input facts fail evidence or
  target validation.
- `target_validator_consumed_facts` must identify the target-owned
  widening-dot-reduction validator as the consumer of provider facts and
  candidate mirror checks.
- `route_metadata` may mirror provider/export metadata only after provider route
  construction. It must not be used as route authority.
- Runtime counts are execution cases only. They must not define dtype,
  dot-reduction semantics, route support, artifact authority, strided support,
  or direct route-entry support.

### 4. Validation & Error Matrix

- Missing or stale source/accumulator/result dtype relation -> evidence
  failure.
- Missing `lhs`, `rhs`, accumulator, output, runtime `n`, or required stride ABI
  role -> evidence failure.
- Missing route operand binding plan, exact binding summary,
  provider-supported mirror, target-validator consumption summary, dot-product
  relation, accumulator/result layout, scalar store VL, or source load form ->
  evidence failure.
- Strided-input route missing source/destination memory form, strided memory
  layout, stride source, or strided load intrinsic -> evidence failure.
- Unit-stride route carrying stale strided-input facts -> evidence failure.
- Generated C/C++ omits runtime VL on source loads, widening product, scalar
  seed, reduction, or scalar result store -> evidence failure.
- Generated C/C++ uses dot-product operands in an order that disagrees with
  provider-owned operand-binding facts -> evidence failure.
- A multi-lane runtime evidence case lacks positive products, negative products,
  source-width-overflowing widening products, nonzero seed, scalar output check,
  or tail sentinel preservation -> evidence failure.
- Descriptor, direct-C/source-export, route-id, artifact-name, ABI-string,
  script, test-name, common EmitC, exact intrinsic spelling, direct route entry,
  pre-realized fixture name, or legacy i32 residue is required to explain
  widening dot-reduction semantics -> evidence failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: typed selected body carries i16 dot sources, i32 scalar seed/result,
  dot-product relation, source load form, runtime `n`, optional stride ABI, and
  policy facts -> RVV plugin validates or realizes the body -> provider emits
  the widening dot-reduction route -> target-owned validator consumes provider
  facts -> generated evidence mirrors the facts and harness checks signed
  widening products, horizontal reduction, scalar output, and tail preservation.
- Base: computed-mask widening dot-reduction routes omit
  `widening_dot_reduction_boundary` and use the computed-mask widening
  dot-reduction evidence boundary.
- Bad: evidence infers dot-reduction behavior, strided behavior, dtype, or
  route support from route id, artifact filename, ABI string, exact intrinsic
  spelling, script constants, or the pre-realized fixture name.

### 6. Tests Required

- lit/FileCheck for explicit and pre-realized generated-bundle dry-runs must
  check representative `widening_dot_reduction_boundary` fields and mirror
  metadata for both unit-stride and strided-input routes.
- Target artifact tests must check provider-owned runtime ABI order/roles,
  operand binding, dtype relation, dot relation/layout, source load form,
  strided fact presence/absence, provider-supported mirror, and stale
  non-widening-dot fact rejection before artifact acceptance.
- Runtime RVV claims must include `ssh rvv` output over multiple runtime counts,
  including `n=0`, `n=1`, exact-VL, tail, and stress cases.
- Harness checks must distinguish signed widening horizontal dot-reduction from
  add-only, mul-only, non-widening, missing-seed, wrong sign-extension,
  unit-stride-vs-strided confusion, scalar-output overwrite, and tail-overwrite
  behavior.

### 7. Wrong vs Correct

Wrong:

```text
pre-realized fixture name says strided_input_widening_dot_reduce_add
  -> harness expects strided widening dot behavior
```

Correct:

```text
typed body/config/runtime widening dot-reduction and optional stride facts
  -> RVV plugin legality/realization
  -> provider-built contraction route
  -> target-owned widening-dot validator
  -> mirror metadata plus executable harness evidence with signed widening dot products
```

## Computed-Mask Widening Dot-Reduction Generated-Bundle Evidence

### 1. Scope / Trigger

Use computed-mask widening dot-reduction generated-bundle evidence whenever an
RVV generated-bundle test claims executable
`computed_masked_widening_dot_reduce_add` or
`computed_masked_strided_input_widening_dot_reduce_add` correctness for a
route-supported typed `tcrv_rvv` path. This evidence is required for explicit
selected bodies and selected pre-realized bodies after RVV plugin-local
realization.

### 2. Signatures

The per-op evidence JSON should expose a bounded summary key:

```json
"computed_masked_widening_dot_reduce_boundary": {
  "authority": "provider-derived typed tcrv_rvv computed-mask widening dot-reduce body/config/runtime facts",
  "target_artifact_validator": "RVVTargetArtifactRouteFamilyValidation.cpp:widening-dot-reduction target-owned consumer",
  "artifact_metadata_role": "mirror-only-after-provider-route",
  "direct_pre_realized_route_entry_supported": false,
  "selected_source_abi": {},
  "provider_route_facts": {},
  "target_validator_consumed_facts": [],
  "materialized_body": {},
  "emitted_cpp": {},
  "route_metadata": {},
  "runtime_counts_are_execution_cases_not_dot_or_mask_authority": true
}
```

### 3. Contracts

- `authority` must identify typed `tcrv_rvv` body/config/runtime facts as the
  source of route support.
- `selected_source_abi` must identify `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`,
  `acc`, `out`, and runtime `n` in provider-validated ABI order. Strided-input
  routes must also identify `lhs_stride` and `rhs_stride`.
- `provider_route_facts` must mirror provider support, target leaf profile,
  runtime ABI order, route operand binding plan and exact binding operands,
  contraction family plan, headers, C type mapping, predicate, mask role/source,
  mask memory form, inactive-lane zeroing, masked product/merge intrinsics,
  source/accumulator/result SEW and LMUL, accumulator/result layout, dot-product
  relation, and scalar store VL after route construction.
- Unit-stride evidence must report strided-input facts as rejected if present.
  Strided-input evidence must mirror source/destination memory forms, strided
  memory layout, stride ABI sources, and strided load intrinsic.
- `target_validator_consumed_facts` must identify the target-owned
  widening-dot-reduction validator as the consumer of provider facts and
  candidate mirror checks.
- `route_metadata` may mirror provider/export metadata only after provider
  route construction. It must not be used as route authority.
- Runtime counts are execution cases only. They must not define predicate kind,
  mask behavior, dtype, dot-reduction semantics, strided support, route support,
  artifact authority, or direct route-entry support.

### 4. Validation & Error Matrix

- Missing or stale runtime ABI order, role, or parameter count -> evidence
  failure.
- Missing compare lhs/rhs, dot lhs/rhs, accumulator, output, runtime `n`, or
  required stride ABI role -> evidence failure.
- Missing or stale route operand binding plan or exact binding summary ->
  evidence failure.
- Missing mask role/source/form, compare predicate, compare intrinsic, masked
  product intrinsic, merge intrinsic, mask type, inactive-lane zeroing, or
  mask/tail policy mirror -> evidence failure.
- Missing source/accumulator/result dtype relation, dot-product relation,
  accumulator/result layout, source load form, scalar store VL, provider support
  mirror, headers, C type mapping, or target-validator consumption summary ->
  evidence failure.
- Strided-input route missing source/destination memory form, stride source,
  strided layout, or strided load intrinsic -> evidence failure.
- Unit-stride route carrying stale strided-input facts -> evidence failure.
- Generated C/C++ omits runtime VL on compare, source loads, masked widening
  product, merge, scalar seed, reduction, or scalar result store -> evidence
  failure.
- Generated C/C++ uses compare, product, merge, or dot operands in an order that
  disagrees with provider-owned operand-binding facts -> evidence failure.
- A multi-lane runtime evidence case lacks active and inactive mask lanes,
  positive products, negative products, nonzero inactive products, nonzero seed,
  scalar output check, strided skipped-source checks where applicable, or tail
  sentinel preservation -> evidence failure.
- Descriptor, direct-C/source-export, route-id, artifact-name, ABI-string,
  script, test-name, common EmitC, exact intrinsic spelling, direct route entry,
  pre-realized fixture name, or legacy i32 residue is required to explain
  computed-mask widening dot semantics -> evidence failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: typed selected body carries compare-produced mask, i16 dot sources, i32
  scalar seed/result, dot-product relation, source load form, runtime `n`,
  optional stride ABI, and policy facts -> RVV plugin validates or realizes the
  body -> provider emits computed-mask widening dot route -> target-owned
  validator consumes provider facts -> generated evidence mirrors the facts and
  harness checks active lanes, inactive zeroing before reduction, signed widening
  products, scalar output, strided source selection when applicable, and tail
  preservation.
- Base: non-computed widening dot-reduction routes omit
  `computed_masked_widening_dot_reduce_boundary` and use
  `widening_dot_reduction_boundary`.
- Bad: evidence infers predicate kind, mask behavior, dot-reduction behavior,
  strided behavior, dtype, or route support from route id, artifact filename,
  ABI string, exact intrinsic spelling, script constants, or the pre-realized
  fixture name.

### 6. Tests Required

- lit/FileCheck for explicit and pre-realized generated-bundle dry-runs must
  check representative `computed_masked_widening_dot_reduce_boundary` fields,
  provider route facts, target-validator consumed facts, mirror metadata, and
  direct pre-realized route-entry support remaining false.
- Target artifact tests must check provider-owned runtime ABI order/roles,
  route operand binding, mask role/source/form, predicate, masked product/merge
  facts, dtype relation, dot relation/layout, source load form, strided fact
  presence/absence, provider-supported mirror, and stale non-family fact
  rejection before artifact acceptance.
- Runtime RVV claims must include `ssh rvv` output over multiple runtime counts,
  including `n=0`, `n=1`, exact-VL, tail, and stress cases.
- Strided-input runtime evidence must include more than one stride pattern or an
  equivalent oracle proving source elements are consumed through runtime stride
  ABI values rather than unit-stride addresses.
- Harness checks must distinguish signed computed-mask widening horizontal
  dot-reduction from add-only, mul-only, unmasked, non-widening, missing-seed,
  wrong sign-extension, unit-stride-vs-strided confusion, scalar-output
  overwrite, inactive-lane contribution, and tail-overwrite behavior.

### 7. Wrong vs Correct

Wrong:

```text
pre-realized fixture name says computed_masked_strided_input_widening_dot_reduce_add
  -> harness expects computed-mask strided widening dot behavior
```

Correct:

```text
typed body/config/runtime compare-mask, widening dot, and optional stride facts
  -> RVV plugin legality/realization
  -> provider-built contraction route
  -> target-owned widening-dot validator
  -> mirror metadata plus executable harness evidence with active/inactive mask and signed widening dot products
```

## Compare/Select Predicate Generated-Bundle Evidence

### 1. Scope / Trigger

Use compare/select predicate generated-bundle evidence whenever an RVV
generated-bundle test claims executable compare/select correctness for a
route-supported typed `tcrv_rvv` path. This evidence is required for bounded
plain compare/select closure and should be extended to computed-mask or
runtime-scalar compare/select only when those paths are the selected task
scope.

### 2. Signatures

The per-op evidence JSON should expose a bounded summary key such as:

```json
"compare_select_predicate_boundary": {
  "authority": "provider-derived typed tcrv_rvv compare/select body/config/runtime facts",
  "artifact_metadata_role": "mirror-only-after-provider-route",
  "compare_predicate_kind": "eq",
  "predicate_source": "compare-produced-mask-same-vl-scope",
  "predicate_role": "predicate-mask-produced-by-compare",
  "select_layout": "select-lhs-when-mask-else-rhs",
  "selected_value_operands": {"true_value": "lhs", "false_value": "rhs"},
  "materialized_body": {},
  "emitted_cpp": {},
  "route_metadata": {},
  "runtime_counts_are_execution_cases_not_predicate_authority": true
}
```

### 3. Contracts

- `authority` must identify typed `tcrv_rvv` body/config/runtime facts as the
  source of route support.
- `artifact_metadata_role` must make mirror-only status explicit.
- `compare_predicate_kind`, `predicate_source`, and `predicate_role` must come
  from realized typed `tcrv_rvv.compare` / route-family facts, not from route
  ids, test names, artifact names, ABI strings, or harness constants.
- `select_layout` and `selected_value_operands` must identify the true/false
  value inputs and output that feed the generated select/merge operation.
- `materialized_body` must prove the typed/pre-realized body was consumed into
  realized `setvl`, `with_vl`, `load`, `compare`, `select`, and `store`
  structure.
- `emitted_cpp` must prove generated RVV C/C++ uses the expected setvl, load,
  compare, select/merge, store intrinsics, runtime loop VL, predicate variable,
  and true/false operands.
- `route_metadata` may mirror provider/export metadata only after provider
  route construction. It must not be used as route authority.
- Runtime counts are execution cases only. They must not define predicate kind,
  select semantics, dtype, route support, or artifact authority.

### 4. Validation & Error Matrix

- Missing realized compare or select op -> evidence failure.
- Missing compare predicate kind, predicate source, or select layout ->
  evidence failure.
- Generated C/C++ omits the compare predicate, uses the wrong predicate
  intrinsic, omits the select/merge intrinsic, reverses true/false operands for
  the declared layout, or omits runtime loop VL operands -> evidence failure.
- Object/header mirror metadata disagree on compare predicate, operand binding,
  mask source/role/form, select layout, dtype/config, or provider-supported
  mirror -> evidence failure.
- Descriptor, direct-C/source-export, route-id, artifact-name, ABI-string, or
  test-name residue is required to explain predicate/select semantics ->
  evidence failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: typed selected body carries compare/select facts -> RVV plugin realizes
  or validates those facts -> route-family/operand-binding/statement-plan facts
  feed provider emission -> evidence mirrors the facts and harness checks both
  predicate-true and predicate-false lanes.
- Base: non compare/select routes omit `compare_select_predicate_boundary` and
  keep ordinary runtime AVL/VL, typed-config, or family-specific evidence.
- Bad: evidence infers predicate kind or select true/false operands from route
  id, ABI string, artifact name, exact intrinsic spelling, or harness expected
  expression.

### 6. Tests Required

- lit/FileCheck for the generated-bundle dry-run must check representative
  `compare_select_predicate_boundary` fields and mirror metadata.
- Provider or C++ API tests must check route-family facts, operand-binding
  facts, and statement-plan facts before common EmitC materialization when
  textual MLIR cannot fully prove the boundary.
- Runtime RVV claims must include `ssh rvv` output and harness checks that
  multi-lane cases cover both predicate-true and predicate-false lanes.

### 7. Wrong vs Correct

Wrong:

```text
artifact metadata says compare_predicate_kind=eq
  -> claim compare/select route support
```

Correct:

```text
typed body/config/runtime compare/select facts
  -> RVV plugin legality/realization
  -> route-family facts + operand-binding facts + statement plan
  -> provider-built compare/select route
  -> common EmitC
  -> mirror metadata plus executable harness evidence
```

## Conversion/SEW Policy Generated-Bundle Evidence

### 1. Scope / Trigger

Use conversion/SEW policy generated-bundle evidence whenever an RVV
generated-bundle test claims route-supported or executable conversion behavior
for a typed `tcrv_rvv` path. This evidence is for bounded conversion routes
such as widening integer conversion; it must not be generalized into broad
dtype/LMUL clone matrices or one-op-per-intrinsic wrapper growth.

### 2. Signatures

The per-op evidence JSON should expose a bounded summary key such as:

```json
"conversion_sew_policy_boundary": {
  "authority": "provider-derived typed tcrv_rvv conversion body/config/runtime facts",
  "artifact_metadata_role": "mirror-only-after-provider-route",
  "conversion_kind": "widen_signed_integer",
  "conversion_relation": "widening",
  "source_type_policy": {"element_type": "i16", "sew": "16", "lmul": "mf2"},
  "result_type_policy": {"element_type": "i32", "sew": "32", "lmul": "m1"},
  "materialized_body": {},
  "emitted_cpp": {},
  "route_metadata": {},
  "runtime_counts_are_execution_cases_not_conversion_authority": true
}
```

### 3. Contracts

- `authority` must identify typed `tcrv_rvv` body/config/runtime facts as the
  source of conversion route support.
- `artifact_metadata_role` must make mirror-only status explicit.
- `conversion_kind` and `conversion_relation` must come from the typed
  conversion op and RVV plugin route-family facts, not from route ids, test
  names, artifact names, ABI strings, or harness constants.
- `source_type_policy` and `result_type_policy` must identify element type,
  SEW, LMUL, and vector type facts consumed by the provider route. These facts
  may be mirrored in generated artifacts only after provider route construction.
- `materialized_body` must prove the selected typed body was consumed into
  realized `setvl`, `with_vl`, source `load`, conversion op, and `store`
  structure.
- `emitted_cpp` must prove generated RVV C/C++ uses the expected setvl, source
  load, conversion, and store intrinsic forms with runtime loop VL operands.
- Runtime counts are execution cases only. They must not define conversion kind,
  source/result dtype, SEW, LMUL, intrinsic spelling, route support, or artifact
  authority.

### 4. Validation & Error Matrix

- Missing realized conversion op or inconsistent source/result vector type
  policy -> evidence failure.
- Generated C/C++ omits the source load, conversion intrinsic, store intrinsic,
  runtime loop VL, or uses a result type not produced by the conversion ->
  evidence failure.
- Object/header mirror metadata disagree on conversion kind, relation,
  source/result dtype, SEW, LMUL, runtime AVL/VL, or provider-supported mirror ->
  evidence failure.
- Unsupported widening, narrowing, sign/unsigned, float/integer, or LMUL policy
  combinations must fail closed with targeted diagnostics before common EmitC
  chooses conversion semantics.
- Descriptor, direct-C/source-export, route-id, artifact-name, ABI-string, or
  test-name residue is required to explain conversion semantics -> evidence
  failure.
- `ssh rvv` compile/run failure -> report blocked/failed evidence and do not
  claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: typed selected body carries conversion and source/result type-policy
  facts -> RVV plugin validates or realizes those facts -> route-family,
  operand-binding, and statement-plan facts feed provider emission -> evidence
  mirrors the facts and harness checks converted lane values.
- Base: non-conversion routes omit `conversion_sew_policy_boundary` and keep
  ordinary runtime AVL/VL, typed-config, or family-specific evidence.
- Bad: evidence infers conversion kind, source/result dtype, SEW, LMUL, or
  intrinsic spelling from route id, ABI string, artifact name, exact intrinsic
  spelling alone, or harness expected expression.

### 6. Tests Required

- lit/FileCheck for the generated-bundle dry-run must check representative
  `conversion_sew_policy_boundary` fields and mirror metadata.
- Provider or C++ API tests must check route-family facts, operand-binding
  facts, and statement-plan facts before common EmitC materialization when
  textual MLIR cannot fully prove the boundary.
- Runtime RVV claims must include `ssh rvv` output and harness checks across
  multiple runtime counts that exercise full-vector and tail-vector execution.

### 7. Wrong vs Correct

Wrong:

```text
artifact metadata says conversion_kind=widen_signed_integer
  -> claim conversion route support
```

Correct:

```text
typed body/config/runtime conversion facts
  -> RVV plugin legality/realization
  -> route-family facts + operand-binding facts + statement plan
  -> provider-built conversion route
  -> common EmitC
  -> mirror metadata plus executable harness evidence
```

## Pre-Realized Selected-Boundary Generated-Bundle Evidence

### 1. Scope / Trigger

Current RVV pre-realized selected-body generated-bundle evidence must use the
public selected lowering-boundary materialization path before route facts are
collected. It proves:

```text
selected pre-realized tcrv_rvv body
  -> RVV selected-body realization owner
  -> realized typed tcrv_rvv body
  -> provider-built route
  -> target artifact bundle
  -> external C ABI harness
```

The direct pre-realized route-entry shortcut is retired for current selected
pre-realized families. `--direct-pre-realized-route-entry` is a negative
evidence mode and must fail closed before bundle generation unless a later spec
explicitly reopens a bounded family.

### 2. Signatures

The durable evidence command shape is:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --op-kind <bounded-supported-kind> \
  [--op-kind <bounded-supported-kind> ...] \
  [--dry-run] \
  --runtime-count <n> --runtime-count <n> \
  [--stride-bytes <bytes> ...]
```

`--direct-pre-realized-route-entry` requires
`--pre-realized-selected-body`, but in current RVV it is only a fail-closed
negative check:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --direct-pre-realized-route-entry \
  --op-kind <selected-pre-realized-kind> \
  [--dry-run] \
  --runtime-count <n> --runtime-count <n>
```

### 3. Contracts

- The selected input remains a pre-realized selected `tcrv.exec` RVV fixture.
- The local bundle generation pipeline must contain
  `--tcrv-materialize-selected-lowering-boundaries`,
  `--tcrv-materialize-emission-plans` and target artifact bundle export.
- Evidence must label the input mode as `pre-realized-selected-body`.
- Evidence must not claim direct route-entry realization support for selected
  pre-realized bodies.
- Materialized body checks must prove the pre-realized op was consumed and the
  realized typed `tcrv_rvv` body remains.
- Bundle checks must validate provider-owned route metadata, runtime ABI order,
  generated header/object paths, and absence of descriptor/direct-C/source
  export residue.
- Runtime/correctness claims still require non-dry-run `ssh rvv` compile/run
  evidence.

### 4. Validation & Error Matrix

- `--direct-pre-realized-route-entry` without
  `--pre-realized-selected-body` -> fail before bundle generation.
- `--direct-pre-realized-route-entry` with any current selected pre-realized
  op kind -> fail before bundle generation with the retired direct route-entry
  diagnostic.
- Materialized output still contains the selected pre-realized body -> fail.
- Pipeline omits `--tcrv-materialize-selected-lowering-boundaries` in positive
  pre-realized selected-body mode -> fail.
- Bundle metadata contains descriptor/direct-C/source-export residue -> fail.
- Non-dry-run remote compile or run fails -> report blocked/failed evidence
  with command output; do not claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: pre-realized selected-body fixture ->
  selected lowering-boundary realization -> provider-built route -> generated
  bundle -> ABI harness, with optional `ssh rvv` run evidence.
- Base: future work may reopen a bounded direct route-entry artifact path only
  through a new spec and focused tests.
- Bad: all pre-realized selected-body fixtures silently skip the selected
  boundary pass and claim direct route-entry support.
- Bad: script evidence treats the artifact name, route id, status field,
  runtime ABI string, or test name as route support authority.

### 6. Tests Required

- Generated-bundle dry-run or runtime evidence for at least one pre-realized
  selected-body fixture.
- The test must check selected-boundary materialized body consumption, provider
  route metadata, generated harness invocation, runtime ABI order, and absence
  of descriptor/direct-C/source-export/source-front-door authority.
- Negative CLI checks must cover missing `--pre-realized-selected-body` and at
  least one current selected pre-realized op kind rejected by
  `--direct-pre-realized-route-entry`.
- Keep at least one existing pre-realized selected-boundary materializer test.

### 7. Wrong vs Correct

Wrong:

```text
--pre-realized-selected-body for any op
  -> skip selected-boundary pass
  -> infer route support from generated artifact metadata
```

Correct:

```text
--pre-realized-selected-body
  -> selected lowering-boundary realization
  -> provider-built route and target bundle
```

## Source Front Door Tests

Default source-front-door policy is explicit-only or disabled. During RVV
Stage 1, RVV source-front-door tests must prove fail-closed behavior unless an
explicit future task has enabled a mature corrected typed route.

Toy, Template, TensorExtLite, IME, Offload, and future plugin positive
source-front-door tests are Stage3/later examples and must not be introduced as
current RVV progress.

## Metadata-Only Tests

Metadata-only surfaces may be tested only as mirrors or diagnostics. A test
must not assert that these alone authorize executable output:

```text
emission_plan status/result
route id
artifact kind/name
manifest
semantic role graph
construction template
source-front-door marker
selected-path metadata
descriptor
```

## Good / Bad Cases

Good:

```text
typed tcrv_rvv body -> plugin route provider -> common EmitC
```

Good:

```text
legacy RVVI32M1 route table -> unsupported diagnostic
```

Good:

```text
source-artifact RVV Stage1 marker -> fail-closed diagnostic
```

Bad:

```text
rvv-i32m1 route id -> supported object artifact
```

Bad:

```text
emission_plan status -> readiness/progress acceptance
```

Bad:

```text
test name or artifact name implies dtype/operation
```

## Review Checklist

- [ ] Does each positive test exercise a production compiler path?
- [ ] Does each RVV positive artifact test use corrected typed `tcrv_rvv` body authority?
- [ ] Are legacy i32m1 and source-front-door RVV paths negative/fail-closed?
- [ ] Are C ABI strings, parameter names, route ids, and artifact names treated as mirrors only?
- [ ] Are runtime/correctness/performance claims backed by appropriate run evidence?
- [ ] Is broad smoke/dashboard/report coverage avoided unless it verifies a real touched behavior?
