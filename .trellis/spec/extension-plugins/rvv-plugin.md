# RVV Plugin

## Role

RVV plugin is the current primary real hardware extension family for
TianChen-RV MLIR. It is part of the unified TCRV system, not an independent
backend dialect.

Environment:

```text
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: primary development, primary performance, primary correctness
```

## Responsibilities

RVV plugin provides:

- RVV capability registration;
- `tcrv.rvv` dialect registration;
- RVV variant generation;
- RVV legality verification;
- RVV tuning space;
- RVV cost model;
- RVV emission paths;
- RVV runtime/threading integration.

It does not define high-level matmul semantics, generic tensor/tile IR, IME internals, offload internals, or custom ISA internals.

## First C++ Plugin Slice

The first concrete C++ RVV slice is intentionally narrower than the full RVV
plugin above. It proves plugin identity, capability participation, proposal
metadata, materialization, legality routing, and selection consumption through
`ExtensionPluginRegistry`.

Stable first-slice names:

```text
plugin name: rvv-plugin
plugin version: 0.1.0
plugin capability id: rvv
plugin capability kind: isa-vector
preferred kernel capability symbol: @rvv
first-slice proposal / variant symbol: @rvv_first_slice
variant origin: rvv-plugin
required capability ids:
  rvv
  rvv.i32_m1.sew32
  rvv.i32_m1.lmul_m1
  rvv.i32_m1.tail_policy.agnostic
  rvv.i32_m1.mask_policy.agnostic
finite i32m2 capability ids:
  rvv.i32_m2.sew32
  rvv.i32_m2.lmul_m2
  rvv.i32_m2.tail_policy.agnostic
  rvv.i32_m2.mask_policy.agnostic
finite i64m1 capability ids consumed by plugin-local profile/replay and i64
binary family routes:
  rvv.i64_m1.sew64
  rvv.i64_m1.lmul_m1
  rvv.i64_m1.tail_policy.agnostic
  rvv.i64_m1.mask_policy.agnostic
optional finite i32 binary selected vector-shape selector capability id:
  rvv.i32_binary.selected_vector_shape
selector property:
  shape = "i32m1" | "i32m2"
preferred first-slice config symbols:
  @rvv_i32_m1_sew32
  @rvv_i32_m1_lmul_m1
  @rvv_i32_m1_tail_agnostic
  @rvv_i32_m1_mask_agnostic
  @rvv_i32_m2_sew32
  @rvv_i32_m2_lmul_m2
  @rvv_i32_m2_tail_agnostic
  @rvv_i32_m2_mask_agnostic
materialized requires form: requires may be [@rvv] only when @rvv is a
  relation-provider for rvv plus exactly one finite config shape above
  (i32m1 or i32m2); otherwise the variant must require the exact provider
  symbols for rvv plus each selected config/policy id
typed policy attr name: tcrv_rvv.policy
typed policy attr value: #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
property requirement attr name: tcrv_rvv.required_march
finite lowering descriptor attr name: tcrv_rvv.lowering_descriptor
finite lowering descriptor values:
  i32-vadd-microkernel.v1
  i32-vsub-microkernel.v1
  i32-vmul-microkernel.v1
finite element-count attr name: tcrv_rvv.element_count
smoke-probe descriptor attr name: tcrv_rvv.smoke_probe_descriptor
smoke-probe descriptor value: standalone-c-toolchain-smoke-probe.v1
optional vlenb attr name: tcrv_rvv.vlenb_bytes
optional base i32 m1 lane attr name: tcrv_rvv.base_i32_m1_lanes
selected vector-shape attr names:
  tcrv_rvv.selected_vector_shape
  tcrv_rvv.selected_vector_sew
  tcrv_rvv.selected_vector_lmul
  tcrv_rvv.selected_tail_policy
  tcrv_rvv.selected_mask_policy
  tcrv_rvv.selected_vector_type
  tcrv_rvv.selected_vector_suffix
  tcrv_rvv.selected_setvl_suffix
```

The first-slice RVV plugin may propose `@rvv_first_slice` only when the request
contains a real high-level MLIR operation, a `tcrv.exec.kernel`, and a
`TargetCapabilitySet` where capability id `rvv` is available either as an exact
capability id or through a structured relation-provider capability whose
`provides` or `implies` list satisfies id `rvv`. The satisfying capability must
also carry the preserved RVV capability properties that provide bounded
plugin-local evidence.

The plugin-local profile/replay contract may additionally surface the finite
i64m1 capability ids above as structured capability facts for i64 add/sub/mul
family routes. These facts remain capability/profile inputs, not runtime SSA
values, selected vector shape metadata, or descriptor-local element counts.

The current minimal proposal gate is:

- capability id `rvv` has bounded `architecture` and `isa_vector_hints`
  properties, and `isa_vector_hints` contains RVV vector evidence;
- capability id `rvv.hart_count` is available and has a positive integer
  `count` property;
- exactly one finite i32 config shape is selected by structured target
  capability facts. If an available provider satisfies capability id
  `rvv.i32_binary.selected_vector_shape`, its bounded string property `shape`
  must be exactly `i32m1` or `i32m2`; the plugin then validates only that
  selected shape's finite config/policy capability ids and must not silently
  fall back to another shape when the requested shape is incomplete. If no
  selector capability is present, the bounded default remains availability
  based and chooses the first valid finite config shape in deterministic order,
  preserving existing `i32m1` fixtures before trying `i32m2`. The finite shapes
  are either the m1 shape
  (`rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`,
  `rvv.i32_m1.mask_policy.agnostic`) or the m2 shape
  (`rvv.i32_m2.sew32`, `rvv.i32_m2.lmul_m2`,
  `rvv.i32_m2.tail_policy.agnostic`,
  `rvv.i32_m2.mask_policy.agnostic`);
- the selected shape's SEW capability is available and has integer property
  `sew_bits = 32`;
- the selected shape's LMUL capability is available and has string property
  `lmul = "m1"` or `lmul = "m2"` according to the selected shape;
- the selected shape's tail-policy capability is available and has string
  property `tail_policy = "agnostic"`;
- the selected shape's mask-policy capability is available and has string
  property `mask_policy = "agnostic"`;
- when capability ids `rvv.vlenb_bytes` and `rvv.i32_m1_lane_count` are
  available, either as exact capability providers or through the explicit
  kernel capability-provider relation scope such as a module-level
  `target = @profile`, they must appear as a pair, preserve positive integer
  `bytes`/`lanes` properties, and satisfy `lanes = bytes / 4` for the current
  i32 m1 baseline capacity fact; an m2 selected config still preserves this
  target/profile fact as m1 lanes and derives m2 intrinsic spelling from
  selected LMUL metadata, not from a new runtime shape claim;
- capability id `rvv.probe.compile_run` is available, either as an exact
  capability provider or through the explicit kernel capability-provider
  relation scope such as a module-level `target = @profile`, and has a bounded
  `selected_march` property containing RVV vector evidence;
- if capability id `rvv.toolchain.march` is present and available, either as an
  exact capability provider or through the same explicit relation scope, its
  `value` property must agree with `rvv.probe.compile_run.selected_march`;
- all consumed property text must be bounded, single-line, and free of
  secret-like/raw-log text.

If the `rvv` capability id or any required first-slice config/policy capability
id is not satisfied by an exact or relation-provider capability, or if a
satisfying capability is generically unavailable (`status = "disabled"`,
`"missing"`, or `"unavailable"`), the plugin proposes no variant. If the
required RVV property evidence is missing, malformed, secret-like, internally
conflicting, or mismatched with the bounded i32/m1/agnostic policy slice,
proposal collection records a recoverable RVV plugin decline diagnostic and
produces no RVV proposal, rather than synthesizing a partial variant or
aborting later plugins.
The diagnostic must name bounded property/capability evidence categories and
must not echo raw property values or secret-like text. Materialized RVV variant
legality remains strict: malformed explicit RVV metadata such as
`tcrv_rvv.required_march` is still fatal in legality and selected-boundary APIs.

The first slice carries generic decision metadata (`condition`, `guard`, and
`policy`), one typed non-compute RVV policy attribute
(`tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>`), a
plugin-owned `tcrv_rvv.required_march` string attribute derived from the
validated `rvv.probe.compile_run.selected_march` property, optional
capability-derived `tcrv_rvv.vlenb_bytes` and
`tcrv_rvv.base_i32_m1_lanes` integer attributes when structured vlenb/lane
capabilities are available, a complete selected vector-shape metadata group,
and typed finite RVV binary family/body authority for the bounded RVV
C-intrinsic microkernel family. A legacy `tcrv_rvv.lowering_descriptor`, when
present beside that typed authority, is non-authoritative mirror metadata only.
The selected vector-shape group records the
plugin/target-owned compile-time choice: shape id, SEW, LMUL, tail policy,
mask policy, C vector type, intrinsic suffix, setvl suffix, and the backing
capability ids through selected-plan metadata and generated source comments.
It is required to match exactly one finite config shape and is distinct from
base i32 M1 lane capacity. The current supported family members are
`i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`,
all using the runtime-callable binary ABI shape for their dtype and the bounded
selected-path integer `tcrv_rvv.element_count`. The default automatic
first-slice proposal still requests typed `i32-vadd` body materialization;
bounded frontend lowering may preserve `tcrv_frontend_lowering = "i32-vsub"`,
`"i32-vmul"`, `"i64-vadd"`, `"i64-vsub"`, or `"i64-vmul"` on the generated
`tcrv.exec.kernel`, in which case the RVV plugin derives the matching proposal
identity from that typed frontend family and later typed family/body evidence.
Hand-authored or test TianChen-RV MLIR may also provide typed direct
`tcrv_rvv.i32_*_microkernel` or `tcrv_rvv.i64_*_microkernel` bodies to
exercise the same contract. Direct descriptor-only RVV variants must fail
closed or remain explicitly unsupported metadata; they must not select family,
dtype, shape, route id, artifact kind, callable ABI, required capabilities,
selected vector config, or emitted body. When structured
`rvv.i32_m1_lane_count` capacity evidence is present through an exact
capability provider or a relation-provider target profile, the RVV plugin
chooses that selected-path sample size as four M1 i32 vectors, capped at 64
elements; without structured capacity evidence it uses the fixed first-slice
fallback sample size `16`. This is a compiler selected-plan decision, not a
runtime trip count, shape, AVL, VL, correctness coverage, or performance claim.
The generic string policy remains the input for core selection/dispatch. The
typed policy, required march, optional capacity metadata, selected vector-shape
metadata, and element count are default plugin-local metadata preserved by the
generic proposal/materialization path and validated by `RVVExtensionPlugin`
when present on a materialized variant. Default typed RVV binary proposals must
not attach `tcrv_rvv.lowering_descriptor`; if that attribute is present on
hand-authored or legacy MLIR, it is optional legacy mirror metadata validated
only after typed family/body authority is known. The optional legacy descriptor
mirror is not a default production authority and is not emitted by typed RVV
proposal planning. These fields are compiler-visible metadata for the existing
generic materialization, legality, capability, and selection helpers. They are
not generic tensor semantics, arbitrary RVV lowering, runtime correctness
evidence, or performance evidence.

## Scenario: Capability-Backed RVV i32m1 Config Policy Slice

### 1. Scope / Trigger

This scenario applies when the RVV plugin proposes, materializes, verifies, or
exports the bounded i32 add/sub/mul first slice. It is a cross-layer contract:
target capability profile facts, variant `requires`, selected
lowering-boundary validation, emission readiness, and probe replay must agree
on the same SEW/LMUL/tail/mask policy ids.

### 2. Signatures

- C++ capability ids: `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic`.
- Preferred MLIR symbols from replay/profile fixtures:
  `@rvv_i32_m1_sew32`, `@rvv_i32_m1_lmul_m1`,
  `@rvv_i32_m1_tail_agnostic`, and `@rvv_i32_m1_mask_agnostic`.
- Variant metadata: `requires` must satisfy `rvv` plus all four first-slice
  config/policy ids; `tcrv_rvv.policy` must be
  `#tcrv_rvv.policy<tail = agnostic, mask = agnostic>`.

### 3. Contracts

- `rvv.i32_m1.sew32` requires `sew_bits = 32 : i64`.
- `rvv.i32_m1.lmul_m1` requires `lmul = "m1"`.
- `rvv.i32_m1.tail_policy.agnostic` requires
  `tail_policy = "agnostic"`.
- `rvv.i32_m1.mask_policy.agnostic` requires
  `mask_policy = "agnostic"`.
- Runtime `n`, AVL, VL, and dispatch guard values remain runtime SSA/control
  or ABI values. They must not be encoded as these compile-time config
  capability facts. Descriptor-local `tcrv_rvv.element_count` remains a bounded
  artifact descriptor, not a runtime trip count.

### 4. Validation & Error Matrix

- Missing `rvv` provider -> plugin decline during proposal; fatal legality
  error for an explicit selected RVV variant.
- Missing first-slice config/policy provider in a materialized `requires` list
  -> fatal legality error naming the missing capability id.
- Disabled/unavailable first-slice config/policy provider -> proposal decline
  or fatal selected-path legality error naming the unavailable capability id.
- `sew_bits != 32`, `lmul != "m1"`, tail policy not `agnostic`, or mask policy
  not `agnostic` -> proposal decline or fatal selected-path legality error
  naming the mismatched property.
- Selected RVV path with `tcrv_rvv.required_march`, finite lowering descriptor,
  smoke descriptor, or capacity metadata must re-run RVV plugin legality before
  selected lowering-boundary or emission artifact success can be reported.

### 5. Good/Base/Bad Cases

- Good: a probe-derived profile emits separate config providers and a
  materialized variant requires all five ids explicitly.
- Base: a hand-authored fixture keeps `requires = [@rvv]` only when `@rvv`
  structurally `provides` all four first-slice config/policy ids in addition
  to exact id `rvv`.
- Bad: `@rvv` is available but the selected variant does not require
  `rvv.i32_m1.lmul_m1`, or the only tail-policy provider is
  `status = "disabled"`.

### 6. Tests Required

- RVV plugin unit/lit tests must assert proposal `requiredCapabilityIDs`
  contain all five ids in deterministic order.
- Legality tests must cover missing requirement, disabled provider, and
  mismatched property for the first-slice config/policy ids.
- Lowering-boundary or emission-readiness tests must show a selected RVV path
  fails closed before source/object/bundle output when required first-slice
  config/policy capability evidence is absent or unavailable.
- Probe and probe-to-MLIR self-tests must assert replayed capability facts
  preserve the four first-slice config/policy facts without moving legality
  into Python.

### 7. Wrong vs Correct

Wrong:

```mlir
tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector",
                           status = "available"}
tcrv.exec.variant @rvv_first_slice attributes {
  origin = "rvv-plugin",
  requires = [@rvv],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {}
```

Correct for a relation-provider fixture:

```mlir
tcrv.exec.capability @rvv {
  id = "rvv",
  kind = "isa-vector",
  provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1",
              "rvv.i32_m1.tail_policy.agnostic",
              "rvv.i32_m1.mask_policy.agnostic"],
  sew_bits = 32 : i64,
  lmul = "m1",
  tail_policy = "agnostic",
  mask_policy = "agnostic",
  status = "available"
}
tcrv.exec.variant @rvv_first_slice attributes {
  origin = "rvv-plugin",
  requires = [@rvv],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {}
```

Correct for a replay/profile fixture:

```mlir
tcrv.exec.variant @rvv_first_slice attributes {
  origin = "rvv-plugin",
  requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1,
              @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
  tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
} {}
```

When the explicit capability scope also makes plugin-local capability id
`rvv.smoke_probe` available, the first-slice proposal may choose the
plugin-local `tcrv_rvv.smoke_probe_descriptor` instead of the finite
i32 microkernel descriptor. That descriptor selects the standalone
smoke-probe source artifact route only. It must preserve the same selected
RVV march/profile validation and lowering-boundary metadata as the direct
smoke exporter, and it must not materialize an executable RVV microkernel,
runtime callable ABI parameters, kernel lowering, correctness evidence, or
performance evidence.

When a relation-provider capability satisfies id `rvv`, generic variant
materialization records the provider symbol in `requires`, for example
`requires = [@rvv_profile]`. RVV legality must treat that provider as satisfying
the RVV id through the C++ `CapabilityDescriptor::satisfiesID("rvv")` relation
API; it must not require an exact `id = "rvv"` capability when a structured
provider relation is present.

When the selected `rvv-plugin` path carries the finite descriptor above,
`RVVExtensionPlugin` owns the lowering-boundary materialization step that
creates the matching plugin-local direct kernel-child microkernel op:
`tcrv_rvv.i32_vadd_microkernel` for `i32-vadd-microkernel.v1`, or
`tcrv_rvv.i32_vsub_microkernel` for `i32-vsub-microkernel.v1`, or
`tcrv_rvv.i32_vmul_microkernel` for `i32-vmul-microkernel.v1`. The
materialized op must preserve source kernel, selected
variant, origin, selected role, required capability refs, required march,
optional selected mabi, bounded element count, and one structured RVV
dataflow body. That body has a runtime index block argument for the
target/export-owned runtime `n`/AVL value, one `tcrv_rvv.setvl`, one matching
`tcrv_rvv.with_vl`, and a nested finite `tcrv_rvv.i32_load`,
`tcrv_rvv.i32_load`, one family-selected arithmetic op
(`tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, or `tcrv_rvv.i32_mul`), and
`tcrv_rvv.i32_store`
dataflow sequence. The load/store ops reference the target/export-owned lhs
input, rhs input, and output buffer ABI roles consumed by the exporter; the runtime
element-count role remains a direct `tcrv.exec.runtime_param` ABI boundary in
the callable plan. If descriptor metadata is missing or
malformed, required march is missing, the structured body is malformed, or an
explicit matching microkernel would make the selected path ambiguous, the
plugin must fail before claiming a supported emission path.

The microkernel slice is the first narrow exception to the metadata-only RVV
unsupported boundary. If the selected `rvv-plugin` path has exactly one matching
RVV i32 microkernel direct kernel-child attachment, either explicitly authored
or plugin-materialized from the finite descriptor, `RVVExtensionPlugin` may
report a supported plugin-owned emission path for deterministic
runtime-callable C source export through
`tcrv-translate --tcrv-export-rvv-microkernel-c`. This support is bounded to
the explicit i32 add/sub/mul RVV C-intrinsic microkernel artifact routes. When
the selected variant carries the finite descriptor, supported readiness/plan
metadata must also validate that the attached microkernel's `element_count`
matches the selected variant's `tcrv_rvv.element_count` descriptor metadata. It
must also validate the microkernel's `setvl` / `with_vl` / explicit
load/arithmetic/store body against the selected RVV first-slice policy and the
finite i32 runtime ABI role contract before reporting the supported handoff. The
target-owned exporter must emit riscv_vector.h C intrinsic source from the
structured body/config by combining the family-owned arithmetic intrinsic
prefix (`__riscv_vadd_vv_`, `__riscv_vsub_vv_`, or `__riscv_vmul_vv_`) with
the selected vector-shape suffix (`i32m1` or `i32m2`). The family registry must
not bake a selected shape suffix into the arithmetic intrinsic prefix; selected
shape metadata is the only owner of the `i32m1`/`i32m2` suffix. It
does not provide generic RVV lowering or runtime ABI integration, and it does
not create correctness or performance evidence without separate `ssh rvv`
compile/run artifacts.

The target layer may also compile that same validated library-style
microkernel source into a bounded RISC-V ELF relocatable object through the
direct `tcrv-translate --tcrv-export-rvv-microkernel-object` command or
through the generic artifact-kind-aware front door when it selects the RVV
microkernel object route. This remains target-owned emission: it consumes the
already selected callable source candidate, structured RVV architecture
capability metadata, selected march/mabi capability metadata from either exact
RVV capability providers or explicit relation-provider target profiles, and
local toolchain diagnostics. The generic object helper route must preflight the
matched RVV callable source candidate through the same typed runtime ABI role
contract used by the direct source route before compiling generated source.
It does not introduce a new lowering dialect, a hidden self-check `main`,
linking, runtime probing, correctness evidence, or performance evidence.

## Remote Evidence Probe Contract

The repo-owned RVV evidence probe is `scripts/rvv_remote_probe.py`. It is
Python runner/evidence tooling and must not be used as the implementation of
capabilities, plugin registry behavior, legality, lowering, emission, runtime
ABI, or compiler-generated executable paths.

Probe artifacts live under:

```text
artifacts/tmp/rvv_probe/<run-id>/
  rvv_probe_evidence.json
  logs/*.log
```

The JSON artifact records a bounded schema:

- `schema_version`, `probe_name`, `run_id`, timestamp, ssh target, artifact dir,
  status, and success boolean;
- hardware/toolchain facts for `uname`, architecture, hart count, clang,
  cmake, bounded RISC-V/vector hints from `/proc/cpuinfo`, and non-interactive
  sudo availability as a boolean capability fact;
- a sanitized `capability_facts` section containing only bounded
  compiler-facing facts: architecture, hart count, optional vlenb bytes and
  derived i32 m1 lane count, ISA/vector hint string, clang and CMake
  availability/version facts, minimal RVV compile/run success, selected
  march/mabi, and optional source/binary digests;
- a minimal RVV intrinsic compile/run probe result with command references,
  exit status, diagnostics, compiler path/version, source digest, selected
  compiler flags, and binary digest when a binary was produced;
- sanitized command logs that avoid secrets, credentials, private keys,
  environment tokens, and unrelated raw environment dumps.

Interpretation rules:

- Successful `ssh rvv` probe output is hardware/toolchain/probe-program
  evidence only.
- The transition from probe evidence to compiler capabilities is
  `sanitized capability_facts -> plugin-local C++ RVV capability profile ->
  TargetCapabilitySet`. The Python probe may emit artifacts, but it must not
  implement capability relations, legality, selection, lowering, or emission.
- The RVV C++ capability profile must validate facts before producing any
  `TargetCapabilitySet`. Required gates include `riscv64`, positive hart count,
  RVV ISA/vector hints, clang and CMake availability, and minimal RVV
  compile/run success. Negative cases must return structured diagnostics rather
  than partial target capabilities.
- Profile-derived capability identities are stable and plugin-local. Current
  first profile IDs include `rvv`, `rvv.hart_count`, `rvv.vlenb_bytes`,
  `rvv.i32_m1_lane_count`, `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, `rvv.i32_m1.mask_policy.agnostic`,
  `rvv.toolchain.clang`, `rvv.toolchain.cmake`, `rvv.probe.compile_run`,
  `rvv.toolchain.march`, and `rvv.toolchain.mabi`. These identities must not
  include ssh/provider names, raw command logs, secrets, benchmark names, or
  performance measurements.
- The probe does not prove that TianChen-RV generated RVV IR, lowered a
  `tcrv.exec` variant, emitted an object, linked runtime glue, proved compiler
  correctness, or measured performance.
- If clang, RVV headers, candidate flags, or remote execution are unavailable,
  the artifact must record failure with exact non-secret command diagnostics
  rather than synthesizing success.
- Future RVV emission beyond the finite explicit i32 add/sub/mul microkernel
  routes requires plugin-local lowering/runtime implementation and successful
  named `ssh rvv` evidence; probe artifacts alone do not create broader
  supported RVV emission.

## Remote Evidence Replay Contract

The repo may provide a Python artifact parser such as
`scripts/rvv_probe_to_mlir.py` to replay sanitized `rvv_probe_evidence.json`
facts into a bounded `tcrv.exec` MLIR fixture. This helper is allowed to parse
the probe JSON, reject secret-like or unbounded compiler-facing facts, and emit
existing `tcrv.exec.capability` ops with the stable plugin-local capability IDs
used by the C++ RVV plugin:

```text
rvv
rvv.hart_count
rvv.vlenb_bytes
rvv.i32_m1_lane_count
rvv.i32_m1.sew32
rvv.i32_m1.lmul_m1
rvv.i32_m1.tail_policy.agnostic
rvv.i32_m1.mask_policy.agnostic
rvv.toolchain.clang
rvv.toolchain.cmake
rvv.probe.compile_run
rvv.toolchain.march
rvv.toolchain.mabi
```

The replayed MLIR is a capability fixture, not a compiler decision. The helper
must not decide RVV proposal availability, legality, selection, lowering,
emission, runtime ABI, correctness, or performance. Those decisions remain in
`RVVExtensionPlugin`, the plugin-local C++ RVV capability profile, and the
generic C++/MLIR planning pipeline. Malformed or failed RVV probe facts may
therefore produce replay MLIR that the C++ RVV plugin declines or diagnoses,
and an explicitly present `scalar.fallback` capability may still allow the
scalar fallback plugin to provide coverage. This is the expected boundary: the
Python helper preserves sanitized evidence facts; it does not invent RVV
support.

`registerDialects` now registers the minimal RVV dialect skeleton through the
RVV plugin path. The default `registerAllDialects` path remains core-only; RVV
dialect availability is proven by populating an `ExtensionPluginRegistry` with
the RVV plugin and calling `registerPluginDialects`.

The first RVV dialect slice started as metadata/control-plane only and now has
bounded i32 add/sub/mul dataflow exceptions for the finite microkernel export. It
introduces the vector-length token type `!tcrv_rvv.vl`, the finite policy
attribute
`#tcrv_rvv.policy<tail = agnostic|undisturbed, mask =
agnostic|undisturbed>`, the bounded runtime AVL-to-VL control-plane operation
`tcrv_rvv.setvl`, the bounded VL scope region operation `tcrv_rvv.with_vl`,
the finite `tcrv_rvv.i32_load`, `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`,
`tcrv_rvv.i32_mul`, and `tcrv_rvv.i32_store` ops nested under that scope for
the current i32 add/sub/mul export routes, and the pre-executable
`tcrv_rvv.lowering_boundary` operation. The setvl op
consumes a runtime AVL SSA value, produces a `!tcrv_rvv.vl` token, and carries
only bounded first-slice compile-time config metadata: SEW 32, LMUL m1 or m2,
and the finite policy attribute. The with_vl op consumes one `!tcrv_rvv.vl` value and
owns one single-block region for bounded RVV control/body work. Optional
duplicated SEW/LMUL/policy metadata is limited to the same bounded first-slice
config and must agree with the visible defining setvl when present. The bounded
i32 add/sub/mul dataflow body carries finite runtime ABI role references on
the explicit lhs load, rhs load, and output store operations; concrete C
parameter names are resolved from runtime ABI metadata by the target exporter
when that metadata is present. It is not a generic vector memory model. The
boundary
op records selected RVV source/variant/role/status metadata for a future
lowering attachment point. These surfaces are not generic RVV arithmetic,
generic memory operations, LLVM/RISC-V lowering, full runtime ABI glue,
hardware execution, correctness evidence, or performance evidence. `tcrv_rvv`
is the concrete MLIR dialect namespace because MLIR dialect namespaces cannot
contain `.` characters; the architectural extension family remains `tcrv.rvv`.

## Capability Fields

RVV plugin should register and query:

```text
rvv
rvv.version
rvv.i32_m1.sew32
rvv.i32_m1.lmul_m1
rvv.i32_m1.tail_policy.agnostic
rvv.i32_m1.mask_policy.agnostic
rvv.i32_m2.sew32
rvv.i32_m2.lmul_m2
rvv.i32_m2.tail_policy.agnostic
rvv.i32_m2.mask_policy.agnostic
rvv.i32_binary.selected_vector_shape
vlen
elen
supported SEW
supported LMUL
mask support
tail policy support
Zvfh / Zvfbfmin / other vector dtype extensions
LLVM scalable vector support
RVV intrinsic support
inline asm policy
thread runtime availability
```

Reference attribute:

```mlir
#tcrv.ext<"rvv",
          kind = "isa-vector",
          version = "1.0",
          vlen = 128,
          supports_mask = true,
          supports_tail_policy = true>
```

## Current Dialect Skeleton

Architectural family:

```text
tcrv.rvv
```

Concrete MLIR namespace:

```text
tcrv_rvv
```

Current first-slice type:

```text
!tcrv_rvv.vl
!tcrv_rvv.i32m1
!tcrv_rvv.i32m2
```

Current first-slice policy attribute:

```text
#tcrv_rvv.policy<tail = agnostic, mask = agnostic>
```

Current first-slice runtime VL control-plane op:

```mlir
%vl = tcrv_rvv.setvl %avl {
  lmul = "m1",  // or "m2" for the finite i32m2 slice
  policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
  sew = 32 : i64
} : index -> !tcrv_rvv.vl
```

Current first-slice VL scope control-plane op:

```mlir
tcrv_rvv.with_vl %vl attributes {
  lmul = "m1",  // or "m2" for the finite i32m2 slice
  policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
  sew = 32 : i64
} {
} : !tcrv_rvv.vl
```

Current first-slice lowering boundary op:

```mlir
tcrv_rvv.lowering_boundary {
  source_kernel = "kernel_symbol",
  selected_variant = @rvv_first_slice,
  origin = "rvv-plugin",
  role = "dispatch case",
  status = "unsupported",
  required_capabilities = [@rvv],
  capability_summary = "rvv",
  unsupported_reason = "RVV lowering boundary is pre-executable metadata only"
}
```

The type is a non-compute vector-length token used by the bounded setvl and
with_vl surfaces and parser/printer ownership tests. The policy attribute is
finite non-compute metadata for proposal preservation, setvl/with_vl
configuration, and RVV plugin-local legality. The setvl op is the first bounded
runtime VL control-plane surface: AVL is a runtime SSA operand, vl is a
`!tcrv_rvv.vl` result, SEW/LMUL/policy are compile-time config metadata for
this first slice, and VLEN/vlenb, `element_count`, `required_march`, and
`required_capabilities` are explicitly not accepted on the op. The with_vl op
is the bounded structural companion to setvl: it consumes one runtime VL SSA
token, owns one single-block region with no region arguments, may repeat only
the same bounded SEW/LMUL/policy config, and rejects VLEN/vlenb,
`element_count`, `required_march`, `required_capabilities`, and raw capability
facts. The lowering-boundary op is a direct child of a `tcrv.exec.kernel`,
references a direct sibling selected RVV
`tcrv.exec.variant`, and only admits `status = "unsupported"` plus direct
variant or dispatch-case roles. It also carries generic selected-boundary
contract metadata (`origin = "rvv-plugin"` and `required_capabilities`
matching the selected variant requirement references) so target-neutral
emission planning can validate the boundary before materializing diagnostics.
These surfaces are not vector registers, masks, memory operations, RVV
intrinsics, LLVM/RISC-V lowering, runtime ABI, executable emission, correctness
evidence, or performance evidence.

## RVV Parameter Boundary

RVV work must keep these parameter layers distinct:

- VLEN and vlenb are hardware facts / target capability evidence. They may
  constrain legality, vector capacity, and selection after provenance is
  validated, including plugin-owned `tcrv_rvv.vlenb_bytes` and
  `tcrv_rvv.base_i32_m1_lanes` variant metadata. The base lane attribute is the
  i32 M1 capacity implied by vlenb, even when the selected vector shape is
  i32m2. It is not selected m1 config metadata and is not a per-variant runtime
  value.
- Selected RVV lowering-boundary capacity metadata, when present, is a
  plugin-owned copy of selected variant metadata on `tcrv_rvv.lowering_boundary`
  using `vlenb_bytes` and `base_i32_m1_lanes`. The pair must be integer,
  present together, positive, ratio-valid for base i32 M1 lanes, and validated
  against both the selected variant and the preserved target capability facts
  by the RVV plugin. It is diagnostic selected-plan metadata, not runtime IR,
  selected vector shape, AVL, VL, runtime `n`, or performance evidence.
- SEW, LMUL, tail policy, and mask policy are compile-time variant config
  selected or proposed by the RVV plugin and checked against target
  capabilities. The current executable first slice admits only SEW 32 with
  LMUL m1 or m2, tail agnostic, and mask agnostic. The m1 shape is backed by
  `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic`; the m2 shape is backed by
  `rvv.i32_m2.sew32`, `rvv.i32_m2.lmul_m2`,
  `rvv.i32_m2.tail_policy.agnostic`, and
  `rvv.i32_m2.mask_policy.agnostic`. A selected variant must require exactly
  one shape, and the target exporter must reject selected capability/body LMUL
  mismatches before artifact bytes are emitted. These config ids are not
  sufficient as standalone hardware facts without the surrounding
  RVV/profile/toolchain evidence.
- The selected vector-shape config is the target-owned serialization of that
  compile-time choice. Materialized RVV variants, `tcrv_rvv.lowering_boundary`,
  the i32 add/sub/mul microkernel ops, selected-plan metadata, and generated
  RVV C source comments must agree on `selected_vector_shape`,
  `selected_vector_sew`, `selected_vector_lmul`, `selected_tail_policy`,
  `selected_mask_policy`, `selected_vector_type`,
  `selected_vector_suffix`, and `selected_setvl_suffix`. If any selected-shape
  attribute is present, the complete group is required. An i32m2 selected path
  must not carry stale selected i32m1 config even though it may still carry the
  separate `base_i32_m1_lanes` capacity fact.
- RVV arithmetic family identity and selected vector-shape suffix are separate
  contracts. The i32 add/sub/mul family registry owns only the suffix-free
  arithmetic intrinsic prefix; the target exporter appends the selected
  vector-shape suffix after validating the selected shape group. A full
  intrinsic string such as `__riscv_vadd_vv_i32m2` is therefore an emission
  result, not a family descriptor field.
- AVL and vl are runtime SSA values / runtime control values. The current
  bounded `tcrv_rvv.setvl` surface models AVL as a real runtime SSA operand and
  vl as a real `!tcrv_rvv.vl` result. The bounded `tcrv_rvv.with_vl` surface
  models vl only when it consumes a real `!tcrv_rvv.vl` SSA operand. It must not
  imply that AVL or vl is IR-modeled unless a real op attribute, SSA value,
  region argument, or ABI parameter exists.
- `tcrv_rvv.element_count` in the current i32-vadd descriptor, microkernel op,
  and C export path is descriptor-local and bounded. It describes the finite
  emitted source slice or test descriptor, not high-level MLIR tensor shape,
  global problem size, AVL, or vl.
- `tcrv_rvv.required_march` string matching is a bounded plugin-owned
  compatibility bridge for the current first slice. Do not expand dependence on
  `required_march` string comparisons when structured capabilities or
  properties are available or should be added.

Emission plans and manifests for RVV paths must not claim VLEN, vlenb, SEW,
LMUL, AVL, vl, `setvl`, `with_vl`, `element_count`, or `required_march` are
IR-modeled unless the real IR has the corresponding attribute, type, SSA value,
region argument, or generated ABI parameter. The current `tcrv_rvv.setvl` and
`tcrv_rvv.with_vl` ops model only runtime AVL/VL control-plane IR; they do not
make VLEN/vlenb or descriptor-local `element_count` runtime values.
RVV emission plans may carry bounded `selected_plan_metadata` entries such as
the selected vector-shape group, `tcrv_rvv.vlenb_bytes`, and
`tcrv_rvv.base_i32_m1_lanes` only as plugin-owned selected-plan
self-description. Those entries must be generated from validated selected
variant metadata and must not become runtime ABI parameters, generated control
values, runtime shapes, or performance claims.

## First Lowering Boundary Slice

The canonical public `tcrv-opt` pass
`--tcrv-materialize-selected-lowering-boundaries` materializes selected-path
lowering-boundary metadata through the generic `ExtensionPluginRegistry`
interface. RVV implements that plugin hook by creating
`tcrv_rvv.lowering_boundary` for selected RVV-owned direct variants or dispatch
cases. The older `--tcrv-materialize-rvv-lowering-boundary` entry remains only a
compatibility wrapper around the same generic path.

Rules:

- RVV-specific interpretation stays in the RVV plugin/dialect implementation.
- The generic pass routes dispatch fallback references to their origin plugin;
  the RVV first slice returns no boundary for fallback role, and scalar fallback
  materializes `tcrv_scalar.lowering_boundary` without receiving RVV ops.
- Kernels without a dispatch or direct selected-path diagnostic are diagnosed
  before any plugin lowering-boundary hook is invoked.
- The boundary op remains `status = "unsupported"` until a later RVV lowering
  and runtime slice adds executable evidence.
- If a selected RVV variant carries capacity metadata, the RVV boundary hook
  must copy that metadata into the plugin-local boundary and plugin-local
  validation must reject missing, stale, unpaired, non-integer, ratio-invalid,
  or target-capability-mismatched capacity metadata before emission planning or
  target bundle export can report success.
- If a selected RVV variant carries `tcrv_rvv.required_march`, a finite
  lowering descriptor, a smoke descriptor, or first-slice capacity metadata, the
  RVV boundary hook must re-run RVV plugin legality before materializing a
  successful selected boundary. Missing, unavailable, or mismatched
  SEW/LMUL/tail/mask capability evidence must fail there before any source,
  object, or bundle export route can claim support.
- The boundary is a compiler structure/evidence boundary only; it must not be
  reported as hardware execution, correctness, or performance evidence.

## Future Dialect Surface

Future RVV execution dialect work may add richer types:

```text
!tcrv.rvv.vreg<dtype, lmul>
!tcrv.rvv.mask<lmul>
!tcrv.rvv.vl
!tcrv.rvv.policy<tail, mask>
```

Future RVV execution dialect work may add ops:

```text
tcrv.rvv.load
tcrv.rvv.store
tcrv.rvv.masked_load
tcrv.rvv.masked_store
tcrv.rvv.broadcast
tcrv.rvv.fma
tcrv.rvv.add / mul / max / min
tcrv.rvv.reduce
tcrv.rvv.slide / gather / compress
tcrv.rvv.convert
```

These are RVV execution ops, not high-level tensor ops.

## Variant Generation

RVV plugin may support high-level op classes such as:

```text
matmul / batched matmul
softmax
layernorm / rmsnorm
rope
elementwise + reduction fusion
attention micro-kernel fragments
```

Output must be a `tcrv.exec.variant` containing `tcrv.rvv.*` ops, not a generic `tcrv.matmul`.

## Legality Rules

RVV plugin checks:

- target supports RVV;
- required dtype has corresponding extension or fallback path;
- SEW/LMUL combination is legal;
- VL policy is expressible;
- mask/tail policy is complete;
- load/store pattern is RVV-expressible;
- reduction handles tail/mask correctly;
- selected toolchain supports the emission path.

## Tuning Space

RVV tuning is variant quality metadata:

```text
SEW
LMUL
VL policy
unroll factor
K blocking
packing of B or weights
thread partition across harts
prefetch or software pipelining option
boundary handling strategy
```

Reference metadata:

```mlir
#tcrv.tuning<
  lmul = 4,
  sew = 16,
  k_block = 64,
  unroll = 2,
  thread_partition = "row_block"
>
```

## Selection Preference

The first RVV slice returns explicit plugin-owned selection preference metadata
for legal materialized RVV variants. Its score is a heuristic ordering input
used by the target-neutral selector; it is not a runtime, correctness, or
performance claim. When structured vlenb-derived i32 lane capacity is
available, the RVV plugin may use it as a bounded heuristic input and record
that fact in the explanation. RVV-specific interpretation of preserved
capability facts stays inside the RVV plugin before the generic preference
record is returned.

## Emission Paths

Current first slice:

```text
rvv-plugin @rvv_first_slice -> unsupported emission readiness
reason: metadata/control-plane only; no RVV lowering/runtime/executable path
emission plan status: unsupported
runtime ABI kind: rvv-plugin-deferred-runtime-abi
runtime ABI name: rvv-executable-runtime-abi-deferred
runtime glue role: deferred-rvv-runtime-glue
required capabilities: selected variant required capability refs
```

This unsupported readiness result is required. It is not a failure of the
architecture and is not RVV hardware, toolchain, runtime, correctness, or
performance evidence.

Public `tcrv-opt` registers the built-in RVV plugin at the tool boundary, so
materialized variants with `origin = "rvv-plugin"` can route through
`RVVExtensionPlugin` for emission-readiness and emission-plan diagnostics. The
result remains explicitly unsupported metadata: the plugin reports no RVV
lowering pipeline, runtime ABI, artifact contract, executable emission path,
hardware execution, correctness result, or performance result. Unknown origins
must still fail through the generic unregistered-origin registry diagnostic.
Tests that need the historical empty-registry parser surface should pass
`--tcrv-disable-builtin-plugins`.

The RVV emission plan may still return bounded runtime ABI ownership metadata
for the selected unsupported boundary. That metadata explains which plugin owns
the future RVV executable/runtime ABI slice; it is not runtime ABI glue, code
generation, hardware execution, correctness evidence, or performance evidence.

### Smoke-Probe Target Export

`tcrv-translate --tcrv-export-rvv-smoke-probe-c` is an RVV-specific target
export surface for post-planning MLIR. It consumes selected `rvv-plugin`
metadata, `tcrv_rvv.required_march`, and the matching
`tcrv_rvv.lowering_boundary` to emit a standalone C program that includes
`riscv_vector.h` and performs a tiny RVV intrinsic smoke check.

The same emitter may be registered as the target-owned standalone C source
artifact route `tcrv-export-rvv-smoke-probe-c` with emission kind
`rvv-smoke-probe-standalone-c-source` and artifact kind
`standalone-c-source`. The generic
`tcrv-translate --tcrv-export-target-source-artifact` front door may select
that route only from normal emission-plan and target artifact exporter registry
metadata; shared generic front-door code must not branch on RVV or target
profile names to discover the smoke exporter.

This target export does not make ordinary RVV kernel emission supported:
metadata-only `@rvv_first_slice` paths without the explicit smoke descriptor
still report unsupported executable emission and a deferred runtime ABI
boundary. A smoke-descriptor path may report support only for the standalone
smoke-probe C source artifact route. Successful `ssh rvv` compile/run of the
generated smoke C only proves the generated smoke program can use the RVV
toolchain on that host.
It does not prove TianChen-RV lowered a selected kernel, emitted LLVM/RISC-V/RVV
IR, generated an object, linked runtime glue, proved kernel correctness, or
measured performance.

### I32 Vector-Add Microkernel Materialization And Export

`tcrv_rvv.i32_vadd_microkernel` is the first RVV extension-dialect executable
microkernel op. It represents exactly one bounded i32 vector-add executable
microkernel body for a selected RVV path. The op is plugin-local under the
`tcrv_rvv` dialect and must carry only selected-path metadata: source kernel,
selected variant, origin, selected role, required capability refs, required
march, optional selected mabi, a tiny element count, and one structured
dataflow region. The region has one runtime index block argument for
target/export-owned `n`/AVL, one `tcrv_rvv.setvl` consuming that argument, one
`tcrv_rvv.with_vl` consuming the resulting `!tcrv_rvv.vl` value, and exactly
a finite lhs-load, rhs-load, add, output-store dataflow sequence nested under
the `with_vl` body. It must
reject generic tensor/tile/benchmark attributes, unbounded or secret-like
strings, invalid element counts, stale selected variants, missing or
unavailable RVV capability refs, required-march mismatches, descriptor-local
`element_count` masquerading as AVL/VL, and malformed control/dataflow bodies.

During selected-boundary materialization, `RVVExtensionPlugin` may create this
op automatically from the selected variant's finite
`tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"` and bounded
`tcrv_rvv.element_count`. This materialization is plugin-owned and must remain
local to `tcrv_rvv`; generic transforms and `tcrv.exec` must not learn RVV
compute semantics, dtype dispatch, or microarchitecture-specific branches.

`tcrv-translate --tcrv-export-rvv-microkernel-c` consumes post-planning MLIR
that has exactly one selected `rvv-plugin` path, a matching
`tcrv_rvv.lowering_boundary`, preserved selected march metadata, and exactly one
matching `tcrv_rvv.i32_vadd_microkernel`. The generated C includes
`riscv_vector.h` and exposes a deterministic runtime-callable i32 vadd C ABI
function with `const int32_t *` input pointers, an `int32_t *` output pointer,
and a `size_t` runtime element count. The callable ABI plan must be built from
direct `tcrv.exec.mem_window` IR for the lhs input, rhs input, and output buffer
roles plus direct `tcrv.exec.runtime_param` IR for the runtime element-count
role. Supported emission-plan `runtime_abi_parameters` entries are a validated
mirror of that IR-backed plan, not an independent source of callable C names or
types. For this first slice those parameters are target/export ABI-owned, not
IR-modeled RVV operands or descriptor-local `element_count`. The exporter must
validate and consume the
microkernel's structured `setvl` / `with_vl` / load-add-store dataflow body
before emitting the RVV i32 load/add/store-intrinsic loop, so stale or
mismatched control/dataflow metadata cannot be silently ignored. The default
export is a library-style callable source artifact with no embedded `main` or
self-check harness. If a bounded self-check executable is
needed for evidence
collection, `tcrv-translate --tcrv-export-rvv-microkernel-self-check-c` is the
explicit harness export. It calls the same ABI function over bounded local
arrays with explicit runtime `n` values; descriptor-local `element_count`
provides only the bounded local-array capacity and is not shape, AVL, VL, or
the only runtime trip count.
Successful `ssh rvv` compile/run evidence for the harness source supports only
the bounded microkernel correctness claim for the generated callable ABI plus
harness. It is not generic high-level lowering, arbitrary RVV executable
emission, full runtime integration, or performance evidence.
The target layer may also emit a matching runtime-callable C ABI header through
the direct `tcrv-translate --tcrv-export-rvv-microkernel-header` command or the
generic `--tcrv-export-target-header-artifact` front door. This header route is
matched from the same validated callable source candidate as the object route
and derives its single prototype from the same selected RVV path, validated
microkernel body, IR-backed callable ABI plan, and capability metadata. The
generic header helper route must validate stale or conflicting
`runtime_abi_parameters` through the same typed target-artifact candidate
preflight surface used by the direct source route before header output. It is
an external caller ABI surface only: no RVV body, hidden `main`, self-check
harness, runtime probe, correctness evidence, or performance text belongs in
the header artifact. Successful `ssh rvv` compile/link/run evidence for an
explicit external caller that includes the generated header and links the
generated relocatable object supports only the bounded external ABI handoff
claim for this finite i32-vadd microkernel. It is not generic high-level
lowering, arbitrary RVV executable emission, full runtime integration, broad
correctness, or performance evidence.

This target export does not change the default metadata-only RVV unsupported
boundary. A selected RVV path without the finite descriptor or exactly one
matching microkernel op remains unsupported/deferred for executable RVV kernel
emission. Generic manifest export and core orchestration must not add
RVV-specific branches for this microkernel path.

### Current EmitC RVV intrinsic route

The current main route is:

```text
TCRV RVV family ops
  -> EmitC
  -> RVV intrinsic C/C++
  -> clang default, gcc compatible
```

MLIR vector, LLVM scalable vector, LLVM RVV intrinsic IR, inline asm, and
backend patches are optional future routes. They are not the current RVV system
definition and should not be described as the mainline until promoted by a
separate spec and implementation evidence.

### Scenario: RVV finite binary family-op EmitC intrinsic route

#### 1. Scope / Trigger

This scenario applies when a selected finite RVV binary microkernel path exports
runtime-callable C/C++ source. The route must consume the verified
`tcrv_rvv.*` family-op body and lower it through an explicit EmitC intrinsic
route before printing RVV intrinsic C/C++.

#### 2. Signatures

- Source family body:
  `tcrv_rvv.setvl -> tcrv_rvv.with_vl -> tcrv_rvv.<dtype>_load ->
  tcrv_rvv.<dtype>_load -> tcrv_rvv.<dtype>_<add|sub|mul> ->
  tcrv_rvv.<dtype>_store`.
- Route plan: `RVVEmitCIntrinsicRoute`, with standard headers, source op
  names, `emitc.call_opaque` callee names, and one setvl callee.
- Exported route metadata comments must include `emitc_route`,
  `emitc_route_headers`, `emitc_route_source_ops`, and indexed
  `emitc.call_opaque[...]` entries.

#### 3. Contracts

- The computation op is the RVV family op in the verified body, not the
  lowering descriptor string.
- Typed family/body authority selects the bounded family, dtype, shape, and
  intrinsic spelling before descriptor text is considered. Descriptor metadata
  may only mirror that already-selected typed authority; stale or descriptor-only
  metadata fails closed before source output.
- `emitc.call_opaque` callees map from verified family ops:
  setvl maps to the selected vsetvl intrinsic; load, arithmetic, and store map
  to the selected RVV load, arithmetic, and store intrinsics.
- Runtime `n` remains the IR-backed runtime-element-count ABI parameter.
- Generated source comments are compile/export evidence only; they are not
  runtime correctness, hardware execution, throughput, latency, or performance
  evidence.

#### 4. Validation & Error Matrix

- Missing or duplicate family-op body step -> fail before source output.
- Body arithmetic op disagrees with selected bounded family -> fail before
  source output.
- Missing `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` control surface -> fail before
  source output.
- Missing route callee for any body step -> fail before source output.
- Stale runtime ABI role/name/type/ownership mirror -> fail before source
  output through the target artifact preflight.

#### 5. Good/Base/Bad Cases

- Good: auto-materialized i32 add emits `tcrv_rvv.i32_add`, the route records
  `emitc.call_opaque` for `__riscv_vadd_vv_i32m1`, and generated C calls that
  intrinsic.
- Base: a hand-authored bounded RVV microkernel with the same verified body can
  use the same route after selected-path and ABI preflight pass.
- Bad: a selected descriptor says i32 add but the body contains
  `tcrv_rvv.i32_sub`; export must fail instead of printing vadd C from the
  descriptor.

#### 6. Tests Required

- lit/FileCheck must show auto materialization creates the RVV family body.
- lit/FileCheck must show generated source includes the EmitC route metadata
  and the expected RVV intrinsic call.
- Negative coverage must keep stale body, stale descriptor, missing boundary,
  and malformed ABI cases fail-closed before source output.

#### 7. Wrong vs Correct

Wrong:

```text
selected descriptor -> direct C string intrinsic emission
```

Correct:

```text
selected descriptor/config metadata
  -> verified tcrv_rvv family-op body
  -> RVVEmitCIntrinsicRoute / emitc.call_opaque mapping
  -> RVV intrinsic C/C++ source
```

## Hart Parallelism

RVV plugin does not define a thread/block model. Multi-core execution is organized by `tcrv.exec.hart_parallel`.

RVV plugin provides per-hart RVV execution behavior and lowering preferences for OpenMP, pthread, runtime thread pool, or single-thread paths.

## Diagnostics

Diagnostics must report:

- illegal LMUL/SEW;
- unsupported dtype;
- missing mask/tail policy;
- unavailable emission path;
- unsuitable memory pattern;
- unsatisfied capability.
