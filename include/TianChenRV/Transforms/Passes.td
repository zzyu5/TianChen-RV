#ifndef TIANCHENRV_TRANSFORMS_PASSES_TD
#define TIANCHENRV_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def CheckCapabilityRequires
    : Pass<"tcrv-check-capability-requires", "::mlir::ModuleOp"> {
  let summary = "Check tcrv.exec.variant requirements against generic capability availability";
  let description = [{
    Builds a target-independent capability query object from each
    tcrv.exec.kernel and rejects unguarded variants whose declared capability
    requirements are present but generically marked unavailable, disabled, or
    missing. Structured tcrv.exec.dispatch cases may guard unavailable
    non-fallback variants only when they carry the typed generic
    runtime_guard_required marker, while dispatch fallback targets must be
    generically available.
    Malformed requires attributes and unknown symbols remain the responsibility
    of the tcrv.exec verifier.
  }];
  let constructor =
      "::tianchenrv::transforms::createCheckCapabilityRequiresPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def CheckHartParallelCapabilities
    : Pass<"tcrv-check-hart-parallel-capabilities", "::mlir::ModuleOp"> {
  let summary = "Check tcrv.exec.hart_parallel against target hart count capabilities";
  let description = [{
    Builds a target-independent capability query object from each
    tcrv.exec.kernel and rejects tcrv.exec.hart_parallel operations whose
    explicit harts attribute exceeds the available structured target hart/core
    count capability.

    The check is target-neutral. It queries the generic capability relation id
    target.hart_count and its positive integer count property. Extension
    plugins may expose plugin-local facts such as rvv.hart_count through that
    relation, but this pass does not branch on RVV, scalar fallback, IME,
    offload, Sophgo, AME, vendor, runtime, dtype, shape, or microkernel
    semantics. A hart_parallel op without an explicit harts attribute remains
    structurally valid and is not treated as a concrete capacity request.
  }];
  let constructor =
      "::tianchenrv::transforms::createCheckHartParallelCapabilitiesPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def SynthesizeVariantDispatch
    : Pass<"tcrv-synthesize-variant-dispatch", "::mlir::ModuleOp"> {
  let summary = "Synthesize generic tcrv.exec.dispatch from materialized variants";
  let description = [{
    Scans each tcrv.exec.kernel for direct sibling tcrv.exec.variant operations
    and, when no dispatch already exists, builds one structurally valid
    tcrv.exec.dispatch. The first direct conservative-fallback variant whose
    requirements are generically available and conflict-free under the kernel
    TargetCapabilitySet becomes the fallback target. Remaining variants are
    emitted as deterministic dispatch cases in original kernel order, with a
    non-empty generic policy attached to cases whose requirements are
    unavailable or conflicting and therefore require runtime dispatch guarding;
    those cases also receive the typed runtime_guard_required marker.
    Kernels without a conflict-free generically available fallback are
    diagnosed and left unmodified.
  }];
  let constructor =
      "::tianchenrv::transforms::createSynthesizeVariantDispatchPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def MaterializePluginVariants
    : Pass<"tcrv-materialize-plugin-variants", "::mlir::ModuleOp"> {
  let summary = "Collect plugin proposals and materialize tcrv.exec variants";
  let description = [{
    Scans existing tcrv.exec.kernel operations, builds a generic
    TargetCapabilitySet from each kernel's capability-provider scope,
    asks an injected ExtensionPluginRegistry for validated plugin proposals, and
    materializes those proposals as typed tcrv.exec.variant operations through
    the shared materialization API.

    The pass is target-neutral. It does not branch on RVV, scalar fallback, IME,
    offload, vendor, runtime, dtype, shape, or microarchitecture semantics.
    Public tcrv-opt registration owns a deterministic built-in registry at the
    tool boundary, while the default factory constructor owns an empty registry
    for embedded negative tests and diagnoses kernels instead of inventing core
    variants.

    Repeated runs skip existing direct variants only when they exactly match the
    current plugin proposal metadata. Mismatched existing symbols are diagnosed
    so the pass never silently appends duplicates or overwrites hand-authored IR.
  }];
  let constructor =
      "::tianchenrv::transforms::createMaterializePluginVariantsPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def VerifyPluginVariantLegality
    : Pass<"tcrv-verify-plugin-variant-legality", "::mlir::ModuleOp"> {
  let summary = "Verify materialized variants through origin plugin legality";
  let description = [{
    Verifies each direct materialized tcrv.exec.variant child of every
    tcrv.exec.kernel through the variant's origin plugin before generic
    selection, dispatch synthesis, selected lowering-boundary materialization,
    or emission-plan diagnostics.

    The pass builds a target-neutral TargetCapabilitySet from each kernel's
    capability-provider scope and delegates legality to the injected
    ExtensionPluginRegistry using the existing VariantLegalityRequest protocol.
    It does not mutate IR and does not branch on RVV, scalar fallback, IME,
    offload, Sophgo, AME, vendor, runtime, dtype, shape, toolchain,
    microarchitecture, or probe-evidence semantics.

    Public tcrv-opt registration owns a deterministic built-in registry at the
    tool boundary, while the default factory constructor owns an empty registry
    for embedded negative tests and diagnoses unregistered origins instead of
    inventing core-side legality.
  }];
  let constructor =
      "::tianchenrv::transforms::createVerifyPluginVariantLegalityPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def SelectVariants
    : Pass<"tcrv-select-variants", "::mlir::ModuleOp"> {
  let summary = "Run generic capability-aware variant selection planning";
  let description = [{
    Runs the target-neutral TianChen-RV variant selection planner over direct
    tcrv.exec.variant children of each tcrv.exec.kernel. Production tools should
    construct this pass with a populated ExtensionPluginRegistry so variant
    preference/cost metadata and legality stay origin-plugin-owned. The pass
    verifies plugin-local legality before ranking and never lets preference act
    as soft legality. Public tcrv-opt registration owns a deterministic built-in
    registry at the tool boundary, while the default factory constructor still
    uses an empty registry for embedded negative tests and therefore diagnoses
    kernels whose variants name unregistered origin plugins rather than
    inventing core-side preference semantics.

    Runtime-dispatch plans materialize typed tcrv.exec.dispatch,
    tcrv.exec.case, and tcrv.exec.fallback operations only when a plugin-provided
    generic conservative fallback candidate is available and conflict-free.
    Unavailable or conflicting non-fallback variants are retained only as
    guarded dispatch cases with typed runtime_guard_required metadata. Generic
    condition, guard, and policy strings may still be copied as printable
    decision annotations when a capability-guarded dispatch case is
    materialized, but they do not independently create dispatch plans or typed
    runtime guard requirements. Static and fallback-only plans materialize a
    generic selected-path tcrv.exec.diagnostic marker with target and
    selection_kind metadata; static plans without fallback coverage also
    materialize a missing-fallback diagnostic instead of inventing an implicit
    fallback. No-variant plans do not erase variants or inject target-specific
    IR.
  }];
  let constructor =
      "::tianchenrv::transforms::createSelectVariantsPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def MaterializeDispatchRuntimeGuards
    : Pass<"tcrv-materialize-dispatch-runtime-guards", "::mlir::ModuleOp"> {
  let summary = "Materialize compiler-owned runtime guard links for selected dispatch cases";
  let description = [{
    Traverses selected tcrv.exec.dispatch surfaces after variant selection or
    dispatch synthesis and materializes the generic dispatch-availability
    runtime ABI/control parameter required by guarded dispatch cases. The pass
    attaches tcrv.exec.case runtime_guard symbol references to a direct
    same-kernel tcrv.exec.runtime_param with ABI role
    dispatch-availability-guard.

    The pass is target-neutral. It may inspect generic capability availability,
    conflicts, and the typed tcrv.exec.case runtime_guard_required marker, but
    it does not branch on RVV,
    scalar fallback, IME, offload, Sophgo, AME, vendor, dtype, shape,
    microarchitecture, runtime probe implementation, or plugin microkernel
    semantics. It does not lower, emit code, probe hardware, or claim runtime,
    correctness, or performance evidence.
  }];
  let constructor =
      "::tianchenrv::transforms::createMaterializeDispatchRuntimeGuardsPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def CheckEmissionPaths
    : Pass<"tcrv-check-emission-paths", "::mlir::ModuleOp"> {
  let summary = "Check plugin-owned emission readiness for selected execution paths";
  let description = [{
    Verifies that every compiler-visible execution path selected by current
    tcrv.exec structure can be routed to its origin plugin for emission
    readiness before final lowering. The check is target-neutral: it validates
    dispatch/fallback references structurally, routes each materialized
    tcrv.exec.variant by its generic origin plugin attribute, and requires the
    origin plugin to return a well-formed supported emission path result.

    Kernels with a direct tcrv.exec.dispatch are checked through every
    tcrv.exec.case target and the tcrv.exec.fallback target. Kernels without a
    dispatch but with a direct selected-path diagnostic marker are checked
    through that marker target. Kernels without a dispatch or selected-marker
    surface are conservatively checked through all direct tcrv.exec.variant
    children. Production tools should construct this pass with a populated
    ExtensionPluginRegistry. Public tcrv-opt registration owns a deterministic
    built-in registry at the tool boundary, while the default factory constructor
    still uses an empty registry for embedded negative tests and diagnoses
    unregistered origins.
  }];
  let constructor =
      "::tianchenrv::transforms::createCheckEmissionPathsPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def MaterializeEmissionPlans
    : Pass<"tcrv-materialize-emission-plans", "::mlir::ModuleOp"> {
  let summary = "Materialize plugin-owned emission plans as structured diagnostics";
  let description = [{
    Collects VariantEmissionPlan results for the currently selected
    compiler-visible execution paths and materializes them as structured
    tcrv.exec.diagnostic metadata. The pass is target-neutral and does not
    lower, emit, invoke a runtime, or claim executable correctness or
    performance. It reuses the same selected-path traversal as emission
    readiness: dispatch cases in deterministic order followed by fallback,
    a direct selected-path marker target when present, or all direct variants
    in IR order when no selected-path surface exists.

    Production tools should construct this pass with a populated
    ExtensionPluginRegistry. Public tcrv-opt registration owns a deterministic
    built-in registry at the tool boundary, while the default factory constructor
    still uses an empty registry for embedded negative tests and diagnoses
    unregistered origins instead of inventing core-side emission semantics.
    Existing materialized emission-plan diagnostics are rejected so stale or
    repeated diagnostics are not silently appended.
  }];
  let constructor =
      "::tianchenrv::transforms::createMaterializeEmissionPlansPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def MaterializeSelectedLoweringBoundaries
    : Pass<"tcrv-materialize-selected-lowering-boundaries", "::mlir::ModuleOp"> {
  let summary = "Materialize selected-path lowering boundaries through plugins";
  let description = [{
    Traverses the already selected tcrv.exec dispatch or selected-marker
    surface and routes each selected direct, dispatch-case, and dispatch-fallback
    variant reference to the owning enabled extension plugin through the generic
    ExtensionPluginRegistry lowering-boundary interface. The traversal and pass
    are target-neutral: they do not branch on RVV, scalar, IME, offload, vendor,
    runtime, dtype, shape, or microarchitecture semantics.

    Plugins decide whether a selected path materializes a plugin-local
    lowering-boundary operation or is an explicit no-boundary fail-closed route.
    The pass itself does not lower to LLVM/RISC-V, emit intrinsics, create
    runtime ABI glue, generate objects, run hardware, prove correctness, or
    claim performance. Public tcrv-opt registration owns a deterministic
    built-in registry at the tool boundary, while the default factory
    constructor owns an empty registry for negative registration tests.
  }];
  let constructor =
      "::tianchenrv::transforms::createMaterializeSelectedLoweringBoundariesPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def MaterializeRVVProbedCapabilityAxes
    : Pass<"tcrv-rvv-materialize-probed-capability-axes", "::mlir::ModuleOp"> {
  let summary = "Materialize probe/profile-derived RVV divergence axes onto in-IR capability providers";
  let description = [{
    Mirrors the RVV plugin-local capability authority's derived target-support
    axes onto the in-kernel tcrv.exec.capability / tcrv.exec.target provider ops
    that the legality gate already queries. Given a selected RVV -march (and an
    optional probed isa/vector-hint string), the pass calls the same plugin-local
    authority the probe->capability conversion uses
    (deriveSupportedSEWAllowList / deriveSupportedLMULAllowList) and stamps the
    derived supported_sew / supported_lmul allow-lists onto each RVV-kind
    capability provider that does not already carry them.

    This closes the live probe->gate seam: a real profile selection (e.g.
    --march=rv64gcv vs --march=rv64gc_zve32x) drives the in-IR capability gate
    divergence automatically, with no hand-authored supported_sew / supported_lmul
    fixture attributes. The materialized values are a TARGET-CAPABILITY support
    allow-list, not a plugin-selected compile-time SEW/LMUL (I5): the typed body
    still owns its single chosen config, and the gate queries the typed
    capability object the pass writes (I1/I4), never inferring from ABI strings
    or family names (I5). The pass derives nothing from clang/cmake/compile-run
    toolchain facts and probes no hardware; it does not lower, emit code, claim
    runtime or performance evidence, or fabricate selected config. Capability
    providers that already declare an axis are left untouched, so it never
    clobbers hand-authored fixtures.
  }];
  let options = [
    Option<"march", "march", "std::string", /*default=*/"\"\"",
           "Selected RISC-V -march whose RVV ISA tier the support allow-lists "
           "are derived from (e.g. rv64gcv, rv64gc_zve32x).">,
    Option<"isaVectorHints", "isa-vector-hints", "std::string",
           /*default=*/"\"\"",
           "Optional probed isa/vector-hint string augmenting the -march "
           "evidence for axis derivation.">
  ];
  let constructor =
      "::tianchenrv::transforms::createMaterializeRVVProbedCapabilityAxesPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def MaterializeRVVQ40Schedule
    : Pass<"tcrv-rvv-materialize-q4-0-schedule", "::mlir::ModuleOp"> {
  let summary = "Capability/resource-aware autotuner: select + stamp the ggml Q4_0 x Q8_0 block-dot shape";
  let description = [{
    The N3 capability/resource-aware autotuner for the ggml Q4_0 x Q8_0 block
    dot-product. Given a selected RVV -march (a profile selection), it derives
    the Zvl128b capability fact through the plugin-local capability authority
    (deriveHasZvl128b), enumerates the bounded Q4_0 shape candidate space
    (integer_core_lmul {mf4,m1} x multi_block_factor {1,2,4} x strip_elision
    {robust,elided}), PRUNES each candidate by legality (strip_elision "elided"
    is correct only at the m1 anchor on a Zvl128b target) and by the
    vector-register-file budget, RANKS the legal candidates by a principled,
    CAPABILITY-BLIND structural cost (per-block reduction count, inner strip-loop
    serialization, outer-loop latency-overlap amortized by the multi-block
    factor), SELECTS the min-cost legal shape, and STAMPS the chosen shape knobs
    (integer_core_lmul / multi_block_factor / strip_elision) plus a
    "tcrv_rvv.q4_0_schedule.*" resource-provenance audit trail onto each
    tcrv_rvv.q4_0_q8_0_block_dot op that does not already carry a hand-authored
    shape knob (additive / no-clobber).

    This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
    resource facts -- NOT a lookup table: the only place the capability enters is
    the legality prune; the cost model is a pure structural function. The SAME
    argmin therefore DIVERGES by capability -- a full-V (rv64gcv, Zvl128b)
    profile selects the strip-elided shape (the ~13% ggml beat), while a
    constrained zve32x profile has the elided shapes pruned and the same argmin
    selects the robust strip-loop shape. One capability fact -> N1 legality
    divergence -> N3 win, on a real llama.cpp kernel.

    Per core-invariants: the stamped knobs MIRROR the plugin-local C++ schedule
    authority (I4); the schedule is derived from the validated ISA tier and the
    structural resource facts, never inferred from ABI strings / family names /
    route ids, and no hardware is probed (I5); a candidate that fails the
    capability/anchor/budget prune is dropped fail-closed and if every candidate
    is pruned the op is left untouched (I7). The pass does not lower, emit code,
    benchmark, or claim runtime evidence; it selects the *how*, never the *what*
    (the dot-product is byte-exact for every legal shape).
  }];
  let options = [
    Option<"march", "march", "std::string", /*default=*/"\"\"",
           "Selected RISC-V -march whose RVV ISA tier the Zvl128b capability "
           "fact (and thus the strip-elision legality) is derived from (e.g. "
           "rv64gcv => Zvl128b => VLEN >= 128; rv64gc_zve32x => not Zvl128b).">,
    Option<"isaVectorHints", "isa-vector-hints", "std::string",
           /*default=*/"\"\"",
           "Optional probed isa/vector-hint string augmenting the -march "
           "evidence for the Zvl128b derivation.">,
    Option<"tuneRecord", "tune-record", "std::string", /*default=*/"\"\"",
           "Optional path to a MEASUREMENT tuning record. When set and it "
           "carries an entry for (kernel, capability-march) whose shape is still "
           "legal in the current candidate set, the pass stamps the "
           "MEASURED-fastest shape (the genuine N3 '实测胜出') instead of the "
           "static cost-model argmin. Absent / stale / unreadable => the pass "
           "falls back to the static cost model (the record is an advisory "
           "cache, never a correctness authority).">,
    Option<"dumpCandidates", "dump-candidates", "bool", /*default=*/"false",
           "Dump the LEGAL candidate set (the single source of truth for the "
           "capability+budget prune) for the offline tune driver to consume, "
           "then exit without stamping. Includes every budget+capability-legal "
           "shape (incl. the mf4 anchors the cost model deems dominated) -- the "
           "driver MEASURES the full legal set; the board ranks, not the cost "
           "model.">
  ];
  let constructor =
      "::tianchenrv::transforms::createMaterializeRVVQ40SchedulePass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect"
  ];
}

def MaterializeRVVQ80Schedule
    : Pass<"tcrv-rvv-materialize-q8-0-schedule", "::mlir::ModuleOp"> {
  let summary = "Capability/resource-aware autotuner: select + stamp the ggml Q8_0 x Q8_0 block-dot shape";
  let description = [{
    The Family-A SIBLING of the Q4_0 autotuner (--tcrv-rvv-materialize-q4-0-
    schedule), for the ggml Q8_0 x Q8_0 block dot-product. Given a selected RVV
    -march (a profile selection), it derives the Zvl128b capability fact through
    the SAME plugin-local capability authority (deriveHasZvl128b), enumerates the
    bounded Q8_0 shape candidate space (integer_core_lmul {mf4,m1,m2} x
    multi_block_factor {1,2,4} x strip_elision {robust,elided}), PRUNES each
    candidate by legality (strip_elision "elided" is correct only at the m2 anchor
    on a Zvl128b target -- q8_0's anchor is one LMUL step up from q4_0's because
    its block is 32 contiguous int8, not a 16-lane nibble half-block) and by the
    vector-register-file budget, RANKS the legal candidates by the SAME
    principled, CAPABILITY-BLIND structural cost FORMULA the Q4_0 autotuner uses
    (fed q8_0's per-anchor reduction count -- the structural difference, not a new
    cost branch), SELECTS the min-cost legal shape, and STAMPS the chosen shape
    knobs (integer_core_lmul / multi_block_factor / strip_elision) plus a
    "tcrv_rvv.q8_0_schedule.*" resource-provenance audit trail onto each
    tcrv_rvv.q8_0_q8_0_block_dot op that does not already carry a hand-authored
    shape knob (additive / no-clobber).

    This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
    resource facts -- NOT a lookup table: the only place the capability enters is
    the legality prune; the cost model is a pure structural function shared with
    the Q4_0 sibling. The SAME argmin therefore DIVERGES by capability -- a
    full-V (rv64gcv, Zvl128b) profile selects the strip-elided m2 shape (the shape
    ggml hand-wrote, plus the multi-block unroll ggml does not use), while a
    constrained zve32x profile has the elided shapes pruned and the same argmin
    selects the robust strip-loop shape. One capability fact -> N1 legality
    divergence -> the q8_0 breadth proof that the template + autotuner generalize
    to a SECOND real llama.cpp kernel.

    Per core-invariants: the stamped knobs MIRROR the plugin-local C++ schedule
    authority (I4); the schedule is derived from the validated ISA tier and the
    structural resource facts (I5); a candidate that fails the prune is dropped
    fail-closed and if every candidate is pruned the op is left untouched (I7).
    The pass selects the *how*, never the *what* (the dot-product is byte-exact
    for every legal shape).
  }];
  let options = [
    Option<"march", "march", "std::string", /*default=*/"\"\"",
           "Selected RISC-V -march whose RVV ISA tier the Zvl128b capability "
           "fact (and thus the strip-elision legality) is derived from (e.g. "
           "rv64gcv => Zvl128b => VLEN >= 128; rv64gc_zve32x => not Zvl128b).">,
    Option<"isaVectorHints", "isa-vector-hints", "std::string",
           /*default=*/"\"\"",
           "Optional probed isa/vector-hint string augmenting the -march "
           "evidence for the Zvl128b derivation.">,
    Option<"tuneRecord", "tune-record", "std::string", /*default=*/"\"\"",
           "Optional path to a MEASUREMENT tuning record; when it carries a "
           "still-legal entry for (kernel, capability-march) the pass stamps the "
           "MEASURED-fastest shape, else falls back to the static cost model "
           "(see the q4-0 pass for the full contract).">,
    Option<"dumpCandidates", "dump-candidates", "bool", /*default=*/"false",
           "Dump the LEGAL candidate set for the offline tune driver, then exit "
           "without stamping (the single source of truth for the prune).">
  ];
  let constructor =
      "::tianchenrv::transforms::createMaterializeRVVQ80SchedulePass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect"
  ];
}

def MaterializeRVVQ41Schedule
    : Pass<"tcrv-rvv-materialize-q4-1-schedule", "::mlir::ModuleOp"> {
  let summary = "Capability/resource-aware autotuner: select + stamp the ggml Q4_1 x Q8_1 block-dot shape";
  let description = [{
    The Family-B SIBLING of the Q4_0 autotuner (--tcrv-rvv-materialize-q4-0-
    schedule), for the ggml Q4_1 x Q8_1 block dot-product (scale+MIN, asymmetric).
    Given a selected RVV -march (a profile selection), it derives the Zvl128b
    capability fact through the SAME plugin-local capability authority
    (deriveHasZvl128b), enumerates the bounded Q4_1 shape candidate space
    (integer_core_lmul {mf4,m1} x multi_block_factor {1,2,4} x strip_elision
    {robust,elided} -- q4_1's nibble half-block matches q4_0's shape, so its
    anchor set is q4_0's, NOT q8_0's), PRUNES each candidate by legality
    (strip_elision "elided" is correct only at the m1 anchor on a Zvl128b target)
    and by the vector-register-file budget, RANKS the legal candidates by the SAME
    principled, CAPABILITY-BLIND structural cost FORMULA the Q4_0 autotuner uses
    (fed q4_1's per-anchor reduction count AND its DERIVED latency depth -- q4_1's
    UNSIGNED-nibble decode is a SHORTER dependent chain than q4_0's offset-binary
    one, a structural fact of the format, not a new cost branch), SELECTS the
    min-cost legal shape, and STAMPS the chosen shape knobs (integer_core_lmul /
    multi_block_factor / strip_elision) plus a "tcrv_rvv.q4_1_schedule.*"
    resource-provenance audit trail onto each tcrv_rvv.q4_1_q8_1_block_dot op that
    does not already carry a hand-authored shape knob (additive / no-clobber).

    This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
    resource facts -- NOT a lookup table: the only place the capability enters is
    the legality prune; the cost model is a pure structural function shared with
    the Q4_0 sibling. The SAME argmin therefore DIVERGES by capability -- a
    full-V (rv64gcv, Zvl128b) profile selects the strip-elided m1 shape, while a
    constrained zve32x profile has the elided shapes pruned and the same argmin
    selects the robust strip-loop shape. One capability fact -> N1 legality
    divergence -> the Family-B breadth proof that the template + autotuner
    generalize to a structurally NEW (scale+MIN) real llama.cpp kernel.

    Per core-invariants: the stamped knobs MIRROR the plugin-local C++ schedule
    authority (I4); the schedule is derived from the validated ISA tier and the
    structural resource facts (I5); a candidate that fails the prune is dropped
    fail-closed and if every candidate is pruned the op is left untouched (I7).
    The pass selects the *how*, never the *what* (the dot-product is byte-exact
    for every legal shape).
  }];
  let options = [
    Option<"march", "march", "std::string", /*default=*/"\"\"",
           "Selected RISC-V -march whose RVV ISA tier the Zvl128b capability "
           "fact (and thus the strip-elision legality) is derived from (e.g. "
           "rv64gcv => Zvl128b => VLEN >= 128; rv64gc_zve32x => not Zvl128b).">,
    Option<"isaVectorHints", "isa-vector-hints", "std::string",
           /*default=*/"\"\"",
           "Optional probed isa/vector-hint string augmenting the -march "
           "evidence for the Zvl128b derivation.">,
    Option<"tuneRecord", "tune-record", "std::string", /*default=*/"\"\"",
           "Optional path to a MEASUREMENT tuning record; when it carries a "
           "still-legal entry for (kernel, capability-march) the pass stamps the "
           "MEASURED-fastest shape, else falls back to the static cost model. "
           "For q4_1 this is the SEAM that FIXES the static mis-pick (the static "
           "model ranks q4_1's slowest legal shape first; measurement picks the "
           "real optimum). See the q4-0 pass for the full contract.">,
    Option<"dumpCandidates", "dump-candidates", "bool", /*default=*/"false",
           "Dump the LEGAL candidate set for the offline tune driver, then exit "
           "without stamping (the single source of truth for the prune).">
  ];
  let constructor =
      "::tianchenrv::transforms::createMaterializeRVVQ41SchedulePass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect"
  ];
}

def MaterializeRVVQ50Schedule
    : Pass<"tcrv-rvv-materialize-q5-0-schedule", "::mlir::ModuleOp"> {
  let summary = "Capability/resource-aware autotuner: select + stamp the ggml Q5_0 x Q8_0 block-dot shape";
  let description = [{
    The Family-A SIBLING of the Q4_0 autotuner (--tcrv-rvv-materialize-q4-0-
    schedule), for the ggml Q5_0 x Q8_0 block dot-product (single-scale, 5-bit
    weight). Given a selected RVV -march (a profile selection), it derives the
    Zvl128b capability fact through the SAME plugin-local capability authority
    (deriveHasZvl128b), enumerates the bounded Q5_0 shape candidate space
    (integer_core_lmul {mf4,m1} x multi_block_factor {1,2,4} x strip_elision
    {robust,elided} -- q5_0's nibble half-block matches q4_0's shape, so its
    anchor set is q4_0's, NOT q8_0's), PRUNES each candidate by legality
    (strip_elision "elided" is correct only at the m1 anchor on a Zvl128b target)
    and by the vector-register-file budget, RANKS the legal candidates by the SAME
    principled, CAPABILITY-BLIND structural cost FORMULA the Q4_0 autotuner uses
    (fed q5_0's per-anchor reduction count AND its DERIVED latency depth -- q5_0's
    unsigned-nibble decode plus 5th-bit injection plus offset-binary `-16` bias is
    the LONGEST dependent decode chain, a structural fact of the format, not a new
    cost branch), SELECTS the min-cost legal shape, and STAMPS the chosen shape
    knobs (integer_core_lmul / multi_block_factor / strip_elision) plus a
    "tcrv_rvv.q5_0_schedule.*" resource-provenance audit trail onto each
    tcrv_rvv.q5_0_q8_0_block_dot op that does not already carry a hand-authored
    shape knob (additive / no-clobber).

    This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
    resource facts -- NOT a lookup table: the only place the capability enters is
    the legality prune; the cost model is a pure structural function shared with
    the Q4_0 sibling. The SAME argmin therefore DIVERGES by capability -- a
    full-V (rv64gcv, Zvl128b) profile selects the strip-elided m1 shape, while a
    constrained zve32x profile has the elided shapes pruned and the same argmin
    selects the robust strip-loop shape. One capability fact -> N1 legality
    divergence -> the Family-A 5-bit breadth proof that the template + autotuner
    generalize to a structurally NEW (5-bit weight) real llama.cpp kernel.

    Per core-invariants: the stamped knobs MIRROR the plugin-local C++ schedule
    authority (I4); the schedule is derived from the validated ISA tier and the
    structural resource facts (I5); a candidate that fails the prune is dropped
    fail-closed and if every candidate is pruned the op is left untouched (I7).
    The pass selects the *how*, never the *what* (the dot-product is byte-exact
    for every legal shape).
  }];
  let options = [
    Option<"march", "march", "std::string", /*default=*/"\"\"",
           "Selected RISC-V -march whose RVV ISA tier the Zvl128b capability "
           "fact (and thus the strip-elision legality) is derived from (e.g. "
           "rv64gcv => Zvl128b => VLEN >= 128; rv64gc_zve32x => not Zvl128b).">,
    Option<"isaVectorHints", "isa-vector-hints", "std::string",
           /*default=*/"\"\"",
           "Optional probed isa/vector-hint string augmenting the -march "
           "evidence for the Zvl128b derivation.">,
    Option<"tuneRecord", "tune-record", "std::string", /*default=*/"\"\"",
           "Optional path to a MEASUREMENT tuning record; when it carries a "
           "still-legal entry for (kernel, capability-march) the pass stamps the "
           "MEASURED-fastest shape, else falls back to the static cost model. "
           "See the q4-0 pass for the full contract.">,
    Option<"dumpCandidates", "dump-candidates", "bool", /*default=*/"false",
           "Dump the LEGAL candidate set for the offline tune driver, then exit "
           "without stamping (the single source of truth for the prune).">
  ];
  let constructor =
      "::tianchenrv::transforms::createMaterializeRVVQ50SchedulePass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect"
  ];
}

def MaterializeRVVQ51Schedule
    : Pass<"tcrv-rvv-materialize-q5-1-schedule", "::mlir::ModuleOp"> {
  let summary = "Capability/resource-aware autotuner: select + stamp the ggml Q5_1 x Q8_1 block-dot shape";
  let description = [{
    The Family-B 5-bit SIBLING of the Q4_0 autotuner (--tcrv-rvv-materialize-q4-0-
    schedule), for the ggml Q5_1 x Q8_1 block dot-product (scale+MIN, asymmetric,
    5-bit UNSIGNED weight) -- it COMBINES q5_0's 5-bit reconstruction and q4_1's
    scale+MIN fold. Given a selected RVV -march (a profile selection), it derives
    the Zvl128b capability fact through the SAME plugin-local capability authority
    (deriveHasZvl128b), enumerates the bounded Q5_1 shape candidate space
    (integer_core_lmul {mf4,m1} x multi_block_factor {1,2,4} x strip_elision
    {robust,elided} -- q5_1's nibble half-block matches q4_0's shape, so its
    anchor set is q4_0's, NOT q8_0's), PRUNES each candidate by legality
    (strip_elision "elided" is correct only at the m1 anchor on a Zvl128b target)
    and by the vector-register-file budget, RANKS the legal candidates by the SAME
    principled, CAPABILITY-BLIND structural cost FORMULA the Q4_0 autotuner uses
    (fed q5_1's per-anchor reduction count AND its DERIVED latency depth -- q5_1's
    unsigned 5-bit decode without the `-16` bias is ONE op shorter than q5_0's
    offset-binary 5-bit chain, a structural fact of the format, not a new cost
    branch), SELECTS the min-cost legal shape, and STAMPS the chosen shape knobs
    (integer_core_lmul / multi_block_factor / strip_elision) plus a
    "tcrv_rvv.q5_1_schedule.*" resource-provenance audit trail onto each
    tcrv_rvv.q5_1_q8_1_block_dot op that does not already carry a hand-authored
    shape knob (additive / no-clobber).

    This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
    resource facts -- NOT a lookup table: the only place the capability enters is
    the legality prune; the cost model is a pure structural function shared with
    the Q4_0 sibling. The SAME argmin therefore DIVERGES by capability -- a
    full-V (rv64gcv, Zvl128b) profile selects the strip-elided m1 shape, while a
    constrained zve32x profile has the elided shapes pruned and the same argmin
    selects the robust strip-loop shape. One capability fact -> N1 legality
    divergence -> the Family-B 5-bit breadth proof that the template + autotuner
    generalize to a structurally NEW (scale+MIN 5-bit) real llama.cpp kernel.

    Per core-invariants: the stamped knobs MIRROR the plugin-local C++ schedule
    authority (I4); the schedule is derived from the validated ISA tier and the
    structural resource facts (I5); a candidate that fails the prune is dropped
    fail-closed and if every candidate is pruned the op is left untouched (I7).
    The pass selects the *how*, never the *what* (the dot-product is byte-exact
    for every legal shape).
  }];
  let options = [
    Option<"march", "march", "std::string", /*default=*/"\"\"",
           "Selected RISC-V -march whose RVV ISA tier the Zvl128b capability "
           "fact (and thus the strip-elision legality) is derived from (e.g. "
           "rv64gcv => Zvl128b => VLEN >= 128; rv64gc_zve32x => not Zvl128b).">,
    Option<"isaVectorHints", "isa-vector-hints", "std::string",
           /*default=*/"\"\"",
           "Optional probed isa/vector-hint string augmenting the -march "
           "evidence for the Zvl128b derivation.">,
    Option<"tuneRecord", "tune-record", "std::string", /*default=*/"\"\"",
           "Optional path to a MEASUREMENT tuning record; when it carries a "
           "still-legal entry for (kernel, capability-march) the pass stamps the "
           "MEASURED-fastest shape, else falls back to the static cost model. "
           "See the q4-0 pass for the full contract.">,
    Option<"dumpCandidates", "dump-candidates", "bool", /*default=*/"false",
           "Dump the LEGAL candidate set for the offline tune driver, then exit "
           "without stamping (the single source of truth for the prune).">
  ];
  let constructor =
      "::tianchenrv::transforms::createMaterializeRVVQ51SchedulePass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect"
  ];
}

def MaterializeRVVGemmSchedule
    : Pass<"tcrv-rvv-materialize-gemm-schedule", "::mlir::ModuleOp"> {
  let summary = "Measurement autotuner: select + stamp the ggml Q4_0 x Q8_0 GEMM M-block (activation-column block)";
  let description = [{
    The N3 MEASUREMENT-backed autotuner for the ggml Q4_0 x Q8_0 FULL GEMM
    (tcrv_rvv.q4_0_q8_0_gemm) M-block -- the inner activation-column block M, the
    one knob that sets how many activation columns share each hoisted weight
    decode. It REUSES the EXACT tune-once -> cache -> read architecture the
    block-dot measurement tuner (--tcrv-rvv-materialize-q4-0-schedule, INC-10)
    established: it ENUMERATES the bounded M band {1,2,4,6,8}, PRUNES by the
    vreg-ceiling resource bound (M <= 8; never a capability gate -- every M is
    byte-exact on any RVV target), then either stamps the MEASURED-fastest M read
    from a cached tuning record (the genuine N3 '实测胜出') or falls back to the
    static default M (the safe cache-friendly tile M=4). It STAMPS activation_cols
    plus a "tcrv_rvv.q4_0_gemm_schedule.*" resource-provenance audit trail onto
    each tcrv_rvv.q4_0_q8_0_gemm op that does not already carry a hand-authored
    activation_cols knob (additive / no-clobber).

    Unlike the block-dot autotuner, M is NOT capability-gated -- it is a cache/L1
    + vreg/unroll resource knob whose optimum is NOISY and analytically
    UNPREDICTABLE (gemv-perf-scoping.md section 2b: M_opt proportional 1/K; INC-25
    G2: M=4 wins ~1.04x while M=6 REGRESSES ~0.857x on register pressure at
    K=4096). A static cost model cannot reliably predict that M6 vreg-pressure
    cliff, so the win is the MEASURED pick (M=4 over M=6), NOT a capability
    divergence. The dump-candidates mode emits the FULL legal band (incl. the
    over-blocking M6/M8 the static proxy deems suboptimal) so the offline tune
    driver MEASURES the whole band -- the BOARD ranks, not the cost proxy.

    Per core-invariants: the stamped M MIRRORS the plugin-local C++ schedule
    authority (I4); the M is derived from the validated resource facts + the
    cached measurement, never inferred from ABI strings / family names / route
    ids, and no hardware is probed at compile time (I5); a stale record naming a
    now-illegal M is fail-closed (revalidated against the current band) and if
    every candidate is pruned the op is left untouched (I7). The pass selects the
    *how* (the column-block), never the *what* (every output is byte-exact for
    every legal M).
  }];
  let options = [
    Option<"march", "march", "std::string", /*default=*/"\"\"",
           "Selected RISC-V -march the tuning record is keyed on (the GEMM M-block "
           "is not capability-gated; the -march keys the record entry for reuse + "
           "the stale/revalidate fail-closed demo).">,
    Option<"isaVectorHints", "isa-vector-hints", "std::string",
           /*default=*/"\"\"",
           "Optional probed isa/vector-hint string (reserved for parity with the "
           "block-dot passes; the M-block legality does not consume it).">,
    Option<"tuneRecord", "tune-record", "std::string", /*default=*/"\"\"",
           "Optional path to a MEASUREMENT tuning record. When set and it carries "
           "a still-legal entry for (kernel, capability-march), the pass stamps the "
           "MEASURED-fastest M (the genuine N3 '实测胜出') instead of the static "
           "default. Absent / stale / unreadable => the pass falls back to the "
           "static default M (the record is an advisory cache, never a correctness "
           "authority).">,
    Option<"dumpCandidates", "dump-candidates", "bool", /*default=*/"false",
           "Dump the LEGAL M band (the single source of truth for the vreg-ceiling "
           "prune) for the offline tune driver to consume, then exit without "
           "stamping. Includes the over-blocking M6/M8 -- the driver MEASURES the "
           "full band; the board ranks, not the cost proxy.">
  ];
  let constructor =
      "::tianchenrv::transforms::createMaterializeRVVGemmSchedulePass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect"
  ];
}

def MaterializeRVVGearboxSchedules
    : Pass<"tcrv-rvv-materialize-gearbox-schedules", "::mlir::ModuleOp"> {
  let summary = "Materialize bounded RVV Gearbox schedule facts on selected typed RVV bodies";
  let description = [{
    Runs one RVV plugin-local Stage 2 Gearbox MVP over already selected typed
    tcrv_rvv bodies. The pass recognizes the existing bounded
    dequantize_i32_to_f32 body shape and the bounded low-precision
    widening_product_reduce_dequantize_f32 representative, derives a bounded
    legal candidate inventory and deterministic selected schedule/resource
    candidate from typed body/config/runtime facts, and materializes explicit
    tcrv_rvv.gearbox.* or tcrv_rvv.low_precision_resource.* facts on the
    selected body boundary.

    The pass does not benchmark, profile, query assembly, persist tuning
    caches, infer from artifact names or route strings, change computation
    semantics, change dtype semantics, mutate runtime ABI roles, alter dispatch
    or fallback behavior, or choose target metadata. RVV provider route
    planning remains responsible for consuming these facts before constructing
    a lowerable route.
  }];
  let constructor =
      "::tianchenrv::transforms::createMaterializeRVVGearboxSchedulesPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect",
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect"
  ];
}

def MaterializeEmitCLowerableRoutes
    : Pass<"tcrv-materialize-emitc-lowerable-routes", "::mlir::ModuleOp"> {
  let summary = "Materialize plugin-owned EmitC lowerable routes through the common route interface";
  let description = [{
    Finds a bounded direct tcrv.exec.variant, builds target capabilities from
    the enclosing kernel, asks the variant origin plugin for an
    extension-family EmitC lowerable route, and replaces the module with the
    materialized MLIR EmitC module produced by the common
    TCRVEmitCLowerableRoute materializer.

    The pass is extension-neutral. It routes by the generic origin plugin and
    does not branch on RVV, scalar, IME, offload, vendor, dtype, LMUL,
    intrinsic spelling, runtime ABI roles, or artifact route metadata. Family
    semantics and intrinsic/runtime-call mapping stay in the origin plugin.
    This pass materializes EmitC IR only; it does not print C/C++, export
    target artifacts, compile objects, run hardware, prove correctness, or
    claim performance.
  }];
  let constructor =
      "::tianchenrv::transforms::createMaterializeEmitCLowerableRoutesPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect",
    "::mlir::emitc::EmitCDialect"
  ];
}

def CheckExecutionPlanCoherence
    : Pass<"tcrv-check-execution-plan-coherence", "::mlir::ModuleOp"> {
  let summary = "Check selected execution-plan metadata before target artifact export";
  let description = [{
    Verifies that post-planning compiler-visible handoff metadata is coherent
    before target artifact export. The pass checks that selected-path markers or
    dispatch/fallback references resolve to direct selected variants; variant
    origins are registered plugins; selected lowering-boundary metadata matches
    the selected kernel, variant, role, origin, and required capabilities;
    emission-plan diagnostics match the same selected path and carry bounded
    runtime ABI ownership metadata; and supported artifact route id, artifact
    kind, callback, and runtime ABI parameter contracts agree with the
    registered target artifact exporter.

    The pass is target-neutral and selected-path based. It does not export
    artifacts, lower to LLVM/RISC-V, run a runtime, or claim correctness or
    performance. Plugin-specific emission and lowering semantics stay behind
    ExtensionPluginRegistry and target-owned exporter registrations.
  }];
  let constructor =
      "::tianchenrv::transforms::createCheckExecutionPlanCoherencePass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

def RVVLowerToEmitC
    : Pass<"tcrv-rvv-lower-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower selected generic tcrv_rvv dataflow bodies to MLIR EmitC via DialectConversion";
  let description = [{
    Stands up the real MLIR DialectConversion infrastructure (TypeConverter +
    ConversionTarget + RewritePatternSet + applyPartialConversion) over the
    existing generic typed tcrv_rvv dataflow ops, the typed-IR replacement for
    the legacy RVV emitted-C string machine.

    In this initial state the pass is an additive structural no-op: it builds
    the beachhead !tcrv_rvv.* -> emitc C-type conversions, marks the emitc
    dialect legal, marks every op (including tcrv_rvv ops) legal, and registers
    zero conversion patterns. Running it on any module leaves the IR unchanged.
    Real per-op conversion patterns for the unit-stride elementwise add
    beachhead family land in a subsequent step.

    The pass is plugin-neutral conversion scaffolding. It does not print C/C++,
    export target artifacts, compile objects, run hardware, prove correctness,
    or claim performance.
  }];
  let constructor =
      "::tianchenrv::transforms::createRVVLowerToEmitCPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect",
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect",
    "::mlir::emitc::EmitCDialect"
  ];
}

#endif // TIANCHENRV_TRANSFORMS_PASSES_TD
