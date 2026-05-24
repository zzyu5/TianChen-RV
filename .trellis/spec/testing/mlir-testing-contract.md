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

For RVV generated-bundle evidence over runtime `n`, memory-writing routes must
check both active-lane arithmetic/data movement and guard/tail preservation.
Harnesses should initialize output storage beyond `n` (or inactive/passthrough
lanes when applicable) with sentinels and fail if the generated route writes
outside the runtime element count. These sentinel checks are evidence quality
guards only; they do not become route, dtype, or artifact authority.

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

### 4. Validation & Error Matrix

- Missing selected mask ABI for an externally masked route -> evidence failure.
- Missing realized mask op or masked compute/store op -> evidence failure.
- Generated C/C++ omits the mask operand in the masked intrinsic -> evidence
  failure.
- Object/header mirror metadata disagree on tail/mask policy or mask role ->
  evidence failure.
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

## Direct Pre-Realized Route-Entry Generated-Bundle Evidence

### 1. Scope / Trigger

Use direct pre-realized route-entry generated-bundle evidence only for bounded
RVV pre-realized selected-body families whose production emission-plan entry is
already specified to realize before route facts are collected. It proves:

```text
selected pre-realized tcrv_rvv body
  -> RVV route-entry realization bridge during emission-plan construction
  -> provider-built route
  -> target artifact bundle
  -> external C ABI harness
```

It must not be used to imply direct route-entry support for every pre-realized
family.

### 2. Signatures

The durable evidence command shape is:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --direct-pre-realized-route-entry \
  --op-kind <bounded-supported-kind> \
  [--op-kind <bounded-supported-kind> ...] \
  [--dry-run] \
  --runtime-count <n> --runtime-count <n> \
  [--stride-bytes <bytes> ...]
```

`--direct-pre-realized-route-entry` requires
`--pre-realized-selected-body`. The option means the script must not insert
`--tcrv-materialize-selected-lowering-boundaries`; the production
`--tcrv-materialize-emission-plans` route-entry path must consume the
pre-realized body.

### 3. Contracts

- The selected input remains a pre-realized selected `tcrv.exec` RVV fixture.
- The local bundle generation pipeline must contain
  `--tcrv-materialize-emission-plans` and target artifact bundle export.
- The local bundle generation pipeline must not contain
  `--tcrv-materialize-selected-lowering-boundaries`.
- Evidence must label `materializer` as
  `rvv-route-entry-selected-body-realization`.
- Evidence must set `route_entry_realization` to `true`.
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
- Direct route-entry option with an op kind outside the bounded supported set
  -> fail before bundle generation.
- Materialized output still contains the selected pre-realized body -> fail.
- Pipeline contains `--tcrv-materialize-selected-lowering-boundaries` while in
  direct route-entry mode -> fail.
- Bundle metadata contains descriptor/direct-C/source-export residue -> fail.
- Non-dry-run remote compile or run fails -> report blocked/failed evidence
  with command output; do not claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: pre-realized compare/select or base-memory route-entry fixture ->
  direct emission-plan route-entry realization -> generated bundle -> ABI
  harness, with optional `ssh rvv` run evidence.
- Base: pre-realized families outside the bounded direct route-entry set keep
  using the explicit selected-boundary materialization mode until a later task
  adds route-entry artifact evidence and tests.
- Bad: all pre-realized selected-body fixtures silently skip the selected
  boundary pass and claim direct route-entry support.
- Bad: script evidence treats the artifact name, route id, status field,
  runtime ABI string, or test name as route support authority.

### 6. Tests Required

- lit/FileCheck for at least one direct pre-realized route-entry generated
  bundle dry-run.
- The test must check `materializer`,
  `route_entry_realization`, materialized body consumption, provider route
  metadata, generated harness invocation, and absence of descriptor/direct-C/
  source-export/source-front-door authority.
- Negative CLI checks must cover missing `--pre-realized-selected-body` and an
  unsupported direct route-entry op kind when the option changes.
- Keep at least one existing pre-realized selected-boundary materializer test
  for families that still require the explicit boundary path.

### 7. Wrong vs Correct

Wrong:

```text
--pre-realized-selected-body for any op
  -> skip selected-boundary pass
  -> infer route support from generated artifact metadata
```

Correct:

```text
--pre-realized-selected-body --direct-pre-realized-route-entry
  -> bounded supported route-entry op
  -> emission-plan route-entry realization
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
