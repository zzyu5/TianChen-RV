#ifndef WEFT_TRANSFORMS_PASSES_TD
#define WEFT_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def CheckCapabilityRequires
    : Pass<"weft-check-capability-requires", "::mlir::ModuleOp"> {
  let summary = "Check weft.exec.variant requirements against generic capability availability";
  let description = [{
    Builds a target-independent capability query object from each
    weft.exec.kernel and rejects unguarded variants whose declared capability
    requirements are present but generically marked unavailable, disabled, or
    missing. Structured weft.exec.dispatch cases may guard unavailable
    non-fallback variants only when they carry the typed generic
    runtime_guard_required marker, while dispatch fallback targets must be
    generically available.
    This gate is self-sufficient: a requires symbol that is not present in the
    kernel's capability set is rejected here as an unknown (unresolvable)
    requirement (default-deny), rather than deferred to a later selection or
    proposal pass. Because an unknown requirement is unresolvable, it is never
    dispatch-guardable. Malformed requires attributes (non-symbol-ref) remain
    the responsibility of the weft.exec verifier.
  }];
  let constructor =
      "::weft::transforms::createCheckCapabilityRequiresPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def CheckHartParallelCapabilities
    : Pass<"weft-check-hart-parallel-capabilities", "::mlir::ModuleOp"> {
  let summary = "Check weft.exec.hart_parallel against target hart count capabilities";
  let description = [{
    Builds a target-independent capability query object from each
    weft.exec.kernel and rejects weft.exec.hart_parallel operations whose
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
      "::weft::transforms::createCheckHartParallelCapabilitiesPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def SynthesizeVariantDispatch
    : Pass<"weft-synthesize-variant-dispatch", "::mlir::ModuleOp"> {
  let summary = "Synthesize generic weft.exec.dispatch from materialized variants";
  let description = [{
    Scans each weft.exec.kernel for direct sibling weft.exec.variant operations
    and, when no dispatch already exists, builds one structurally valid
    weft.exec.dispatch. The first direct conservative-fallback variant whose
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
      "::weft::transforms::createSynthesizeVariantDispatchPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def MaterializePluginVariants
    : Pass<"weft-materialize-plugin-variants", "::mlir::ModuleOp"> {
  let summary = "Collect plugin proposals and materialize weft.exec variants";
  let description = [{
    Scans existing weft.exec.kernel operations, builds a generic
    TargetCapabilitySet from each kernel's capability-provider scope,
    asks an injected ExtensionPluginRegistry for validated plugin proposals, and
    materializes those proposals as typed weft.exec.variant operations through
    the shared materialization API.

    The pass is target-neutral. It does not branch on RVV, scalar fallback, IME,
    offload, vendor, runtime, dtype, shape, or microarchitecture semantics.
    Public weft-opt registration owns a deterministic built-in registry at the
    tool boundary, while the default factory constructor owns an empty registry
    for embedded negative tests and diagnoses kernels instead of inventing core
    variants.

    Repeated runs skip existing direct variants only when they exactly match the
    current plugin proposal metadata. Mismatched existing symbols are diagnosed
    so the pass never silently appends duplicates or overwrites hand-authored IR.
  }];
  let constructor =
      "::weft::transforms::createMaterializePluginVariantsPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def VerifyPluginVariantLegality
    : Pass<"weft-verify-plugin-variant-legality", "::mlir::ModuleOp"> {
  let summary = "Verify materialized variants through origin plugin legality";
  let description = [{
    Verifies each direct materialized weft.exec.variant child of every
    weft.exec.kernel through the variant's origin plugin before generic
    selection, dispatch synthesis, selected lowering-boundary materialization,
    or emission-plan diagnostics.

    The pass builds a target-neutral TargetCapabilitySet from each kernel's
    capability-provider scope and delegates legality to the injected
    ExtensionPluginRegistry using the existing VariantLegalityRequest protocol.
    It does not mutate IR and does not branch on RVV, scalar fallback, IME,
    offload, Sophgo, AME, vendor, runtime, dtype, shape, toolchain,
    microarchitecture, or probe-evidence semantics.

    Public weft-opt registration owns a deterministic built-in registry at the
    tool boundary, while the default factory constructor owns an empty registry
    for embedded negative tests and diagnoses unregistered origins instead of
    inventing core-side legality.
  }];
  let constructor =
      "::weft::transforms::createVerifyPluginVariantLegalityPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def SelectVariants
    : Pass<"weft-select-variants", "::mlir::ModuleOp"> {
  let summary = "Run generic capability-aware variant selection planning";
  let description = [{
    Runs the target-neutral Weft-RV variant selection planner over direct
    weft.exec.variant children of each weft.exec.kernel. Production tools should
    construct this pass with a populated ExtensionPluginRegistry so variant
    preference/cost metadata and legality stay origin-plugin-owned. The pass
    verifies plugin-local legality before ranking and never lets preference act
    as soft legality. Public weft-opt registration owns a deterministic built-in
    registry at the tool boundary, while the default factory constructor still
    uses an empty registry for embedded negative tests and therefore diagnoses
    kernels whose variants name unregistered origin plugins rather than
    inventing core-side preference semantics.

    Runtime-dispatch plans materialize typed weft.exec.dispatch,
    weft.exec.case, and weft.exec.fallback operations only when a plugin-provided
    generic conservative fallback candidate is available and conflict-free.
    Unavailable or conflicting non-fallback variants are retained only as
    guarded dispatch cases with typed runtime_guard_required metadata. Generic
    condition, guard, and policy strings may still be copied as printable
    decision annotations when a capability-guarded dispatch case is
    materialized, but they do not independently create dispatch plans or typed
    runtime guard requirements. Static and fallback-only plans materialize a
    generic selected-path weft.exec.diagnostic marker with target and
    selection_kind metadata; static plans without fallback coverage also
    materialize a missing-fallback diagnostic instead of inventing an implicit
    fallback. No-variant plans do not erase variants or inject target-specific
    IR.

    Compile-time selection attribution (science target [D-4] (1)). When the
    attribution-jsonl option is set the pass appends, for EVERY planned kernel
    (all four VariantSelectionKinds, incl. NoViableVariant => chosen=null), one
    canonical-JSON line to the given file:
      {kernel, candidates[], keys_evaluated{}, chosen, reason,
       declared_instance_hash, ts}
    reason is a NEWLY DERIVED side field carrying the distinction on the PRIMARY
    key: only_feasible = exactly one feasible candidate; static_order = >=2
    feasible chosen by today's capability-blind constant-score cold-start
    ordering. prior (capability-derived prior) and measured (memoized measurement)
    are reserved enum values NOT emitted at this selection stage today (prior
    awaits [SEL-1]/G3, measured awaits [SEL-3]). Every feasible candidate always
    records its constant ranking score so a static_order decision is fully
    reconstructible. The in-IR reason attribute (variant-selected) is NOT
    remapped -- the JSONL sink is purely additive and defaults OFF, so existing
    IR / lit expectations are byte-identical when the option is absent.
  }];
  let options = [
    Option<"attributionJsonl", "attribution-jsonl", "std::string",
           /*default=*/"\"\"",
           "Output file path for the compile-time selection attribution JSONL "
           "([D-4] (1)). Empty (default) => the attribution sink is DISABLED and "
           "the pass behaves byte-identically to today. When set, one canonical "
           "JSON object per planned kernel is appended.">,
    Option<"attributionJsonlNoTimestamp", "attribution-jsonl-no-timestamp",
           "bool", /*default=*/"false",
           "Emit a fixed sentinel timestamp (ts of 0) instead of a wall-clock "
           "one so the attribution JSONL is byte-deterministic for lit/FileCheck. "
           "Production runs leave this false to record a real ISO-8601 ts.">
  ];
  let constructor =
      "::weft::transforms::createSelectVariantsPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def MaterializeDispatchRuntimeGuards
    : Pass<"weft-materialize-dispatch-runtime-guards", "::mlir::ModuleOp"> {
  let summary = "Materialize compiler-owned runtime guard links for selected dispatch cases";
  let description = [{
    Traverses selected weft.exec.dispatch surfaces after variant selection or
    dispatch synthesis and materializes the generic dispatch-availability
    runtime ABI/control parameter required by guarded dispatch cases. The pass
    attaches weft.exec.case runtime_guard symbol references to a direct
    same-kernel weft.exec.runtime_param with ABI role
    dispatch-availability-guard.

    The pass is target-neutral. It may inspect generic capability availability,
    conflicts, and the typed weft.exec.case runtime_guard_required marker, but
    it does not branch on RVV,
    scalar fallback, IME, offload, Sophgo, AME, vendor, dtype, shape,
    microarchitecture, runtime probe implementation, or plugin microkernel
    semantics. It does not lower, emit code, probe hardware, or claim runtime,
    correctness, or performance evidence.
  }];
  let constructor =
      "::weft::transforms::createMaterializeDispatchRuntimeGuardsPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def CheckEmissionPaths
    : Pass<"weft-check-emission-paths", "::mlir::ModuleOp"> {
  let summary = "Check plugin-owned emission readiness for selected execution paths";
  let description = [{
    Verifies that every compiler-visible execution path selected by current
    weft.exec structure can be routed to its origin plugin for emission
    readiness before final lowering. The check is target-neutral: it validates
    dispatch/fallback references structurally, routes each materialized
    weft.exec.variant by its generic origin plugin attribute, and requires the
    origin plugin to return a well-formed supported emission path result.

    Kernels with a direct weft.exec.dispatch are checked through every
    weft.exec.case target and the weft.exec.fallback target. Kernels without a
    dispatch but with a direct selected-path diagnostic marker are checked
    through that marker target. Kernels without a dispatch or selected-marker
    surface are conservatively checked through all direct weft.exec.variant
    children. Production tools should construct this pass with a populated
    ExtensionPluginRegistry. Public weft-opt registration owns a deterministic
    built-in registry at the tool boundary, while the default factory constructor
    still uses an empty registry for embedded negative tests and diagnoses
    unregistered origins.
  }];
  let constructor =
      "::weft::transforms::createCheckEmissionPathsPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def MaterializeEmissionPlans
    : Pass<"weft-materialize-emission-plans", "::mlir::ModuleOp"> {
  let summary = "Materialize plugin-owned emission plans as structured diagnostics";
  let description = [{
    Collects VariantEmissionPlan results for the currently selected
    compiler-visible execution paths and materializes them as structured
    weft.exec.diagnostic metadata. The pass is target-neutral and does not
    lower, emit, invoke a runtime, or claim executable correctness or
    performance. It reuses the same selected-path traversal as emission
    readiness: dispatch cases in deterministic order followed by fallback,
    a direct selected-path marker target when present, or all direct variants
    in IR order when no selected-path surface exists.

    Production tools should construct this pass with a populated
    ExtensionPluginRegistry. Public weft-opt registration owns a deterministic
    built-in registry at the tool boundary, while the default factory constructor
    still uses an empty registry for embedded negative tests and diagnoses
    unregistered origins instead of inventing core-side emission semantics.
    Existing materialized emission-plan diagnostics are rejected so stale or
    repeated diagnostics are not silently appended.
  }];
  let constructor =
      "::weft::transforms::createMaterializeEmissionPlansPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def MaterializeSelectedLoweringBoundaries
    : Pass<"weft-materialize-selected-lowering-boundaries", "::mlir::ModuleOp"> {
  let summary = "Materialize selected-path lowering boundaries through plugins";
  let description = [{
    Traverses the already selected weft.exec dispatch or selected-marker
    surface and routes each selected direct, dispatch-case, and dispatch-fallback
    variant reference to the owning enabled extension plugin through the generic
    ExtensionPluginRegistry lowering-boundary interface. The traversal and pass
    are target-neutral: they do not branch on RVV, scalar, IME, offload, vendor,
    runtime, dtype, shape, or microarchitecture semantics.

    Plugins decide whether a selected path materializes a plugin-local
    lowering-boundary operation or is an explicit no-boundary fail-closed route.
    The pass itself does not lower to LLVM/RISC-V, emit intrinsics, create
    runtime ABI glue, generate objects, run hardware, prove correctness, or
    claim performance. Public weft-opt registration owns a deterministic
    built-in registry at the tool boundary, while the default factory
    constructor owns an empty registry for negative registration tests.
  }];
  let constructor =
      "::weft::transforms::createMaterializeSelectedLoweringBoundariesPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def MaterializeRVVProbedCapabilityAxes
    : Pass<"weft-rvv-materialize-probed-capability-axes", "::mlir::ModuleOp"> {
  let summary = "Materialize probe/profile-derived RVV divergence axes onto in-IR capability providers";
  let description = [{
    Mirrors the RVV plugin-local capability authority's derived target-support
    axes onto the in-kernel weft.exec.capability / weft.exec.target provider ops
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
      "::weft::transforms::createMaterializeRVVProbedCapabilityAxesPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

// The 4 pass options shared by ALL FIVE block-dot schedule autotuners
// (q4_0/q8_0/q4_1/q5_0/q5_1): march -> Zvl128b derivation, the optional probed
// isa/vector hints, the optional measurement tuning-record path, and the
// dump-candidates single-source-of-truth seam. Declared ONCE here (formerly the
// 4-option block was duplicated verbatim in all five pass defs).
defvar kRVVBlockDotScheduleOptions = [
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
         "carries a still-legal entry for (kernel, capability-march) the pass "
         "selects that candidate instead of the analytic/static prior. Absent, "
         "stale, or unreadable records leave the prior in control; the record "
         "cannot create candidates or bypass legality.">,
  Option<"dumpCandidates", "dump-candidates", "bool", /*default=*/"false",
         "Dump the LEGAL candidate set (the single source of truth for the "
         "capability+budget prune) for the offline tune driver to consume, "
         "then exit without mutating IR. Includes every budget+capability-legal "
         "shape (incl. the mf4 anchors the cost model deems dominated) -- the "
         "driver MEASURES the full legal set; the board ranks, not the cost "
         "model.">
];

def MaterializeRVVSchedule
    : Pass<"weft-rvv-materialize-schedule", "::mlir::ModuleOp"> {
  let summary = "Construct or validate every tunable RVV op's complete final schedule in one walk";
  let description = [{
    Walk every TunableScheduleOpInterface, look up its plugin-local formula,
    construct the bounded candidate set from capability/resource inputs, apply
    legality, and select only within that set. A valid tuning-record winner may
    override the analytic/static prior; it cannot create an implementation.

    An op with no final fields receives all final fields atomically. An op that
    already carries every field is accepted only when it exactly matches a
    currently legal candidate. Partial, unknown, or illegal final plans fail the
    pass. Candidate counts, costs, reasons, producer names and measurement times
    remain transient; only the final schedule fields and semantic minimum_vlen
    enter IR. Dump mode exposes legal candidates without mutating IR.
  }];
  let options = kRVVBlockDotScheduleOptions;
  let constructor =
      "::weft::transforms::createMaterializeRVVSchedulePass()";
  let dependentDialects = [
    "::weft::rvv::WEFTRVVDialect"
  ];
}

def MaterializeRVVQ40Schedule
    : Pass<"weft-rvv-materialize-q4-0-schedule", "::mlir::ModuleOp"> {
  let summary = "Construct or validate the ggml Q4_0 x Q8_0 block-dot final schedule";
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
    factor), and SELECTS the min-cost legal shape. It writes exactly the final
    integer_core_lmul / multi_block_factor / strip_elision fields. A complete
    explicit plan must match a legal candidate; a partial plan fails.

    This is the COMPILER SELECTING the kernel shape, DERIVED from capability +
    resource facts -- NOT a lookup table: the only place the capability enters is
    the legality prune; the cost model is a pure structural function. The SAME
    argmin therefore DIVERGES by capability -- a full-V (rv64gcv, Zvl128b)
    profile selects the strip-elided shape, while a
    constrained zve32x profile has the elided shapes pruned and the same argmin
    selects the robust strip-loop shape. One capability fact -> N1 legality
    divergence -> N3 capability-keyed shape divergence, on a real llama.cpp
    kernel. (No performance claim: any vs-ggml delta is un-sealed until [PERF-1]
    eight-gate passes; see NG-4.)

    Per core-invariants, the schedule is derived from the validated ISA tier and the
    structural resource facts, never inferred from ABI strings / family names /
    route ids, and no hardware is probed (I5); a candidate that fails the
    capability/anchor/budget prune is dropped fail-closed and if every candidate
    is pruned the pass fails (I7). The pass does not lower, emit code,
    benchmark, or claim runtime evidence; it selects the *how*, never the *what*
    (the dot-product is byte-exact for every legal shape).
  }];
  let options = kRVVBlockDotScheduleOptions;
  let constructor =
      "::weft::transforms::createMaterializeRVVQ40SchedulePass()";
  let dependentDialects = [
    "::weft::rvv::WEFTRVVDialect"
  ];
}

def MaterializeRVVGemmSchedule
    : Pass<"weft-rvv-materialize-gemm-schedule", "::mlir::ModuleOp"> {
  let summary = "Construct or validate the ggml Q4_0 x Q8_0 GEMM activation-column schedule";
  let description = [{
    The N3 MEASUREMENT-backed autotuner for the ggml Q4_0 x Q8_0 FULL GEMM
    (weft_rvv.q4_0_q8_0_gemm) M-block -- the inner activation-column block M, the
    one knob that sets how many activation columns share each hoisted weight
    decode. It REUSES the EXACT tune-once -> cache -> read architecture the
    block-dot measurement tuner (--weft-rvv-materialize-q4-0-schedule, INC-10)
    established: it ENUMERATES the bounded M band {1,2,4,6,8}, PRUNES by the
    vreg-ceiling resource bound (M <= 8; never a capability gate -- every M is
    byte-exact on any RVV target), then either selects the MEASURED-fastest M read
    from a cached tuning record (the genuine N3 '实测胜出') or falls back to the
    static prior M (the safe cache-friendly tile M=4). Only the final
    activation_cols field enters IR; a partial or illegal explicit plan fails.

    Unlike the block-dot autotuner, M is NOT capability-gated -- it is a cache/L1
    + vreg/unroll resource knob whose optimum is NOISY and analytically
    UNPREDICTABLE (gemv-perf-scoping.md section 2b: M_opt proportional 1/K; INC-25
    G2: M=4 wins ~1.04x while M=6 REGRESSES ~0.857x on register pressure at
    K=4096). A static cost model cannot reliably predict that M6 vreg-pressure
    cliff, so the win is the MEASURED pick (M=4 over M=6), NOT a capability
    divergence. The dump-candidates mode emits the FULL legal band (incl. the
    over-blocking M6/M8 the static proxy deems suboptimal) so the offline tune
    driver MEASURES the whole band -- the BOARD ranks, not the cost proxy.

    Per core-invariants, M is derived from the validated resource facts + the
    cached measurement, never inferred from ABI strings / family names / route
    ids, and no hardware is probed at compile time (I5); a stale record naming a
    now-illegal M is fail-closed (revalidated against the current band) and if
    every candidate is pruned the pass fails (I7). The pass selects the
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
           "a still-legal entry for (kernel, capability-march), the pass selects the "
           "MEASURED-fastest M (the genuine N3 '实测胜出') instead of the static "
           "default. Absent / stale / unreadable => the pass falls back to the "
           "static default M (the record is an advisory cache, never a correctness "
           "authority).">,
    Option<"dumpCandidates", "dump-candidates", "bool", /*default=*/"false",
           "Dump the LEGAL M band (the single source of truth for the vreg-ceiling "
           "prune) for the offline tune driver to consume, then exit without "
           "mutating IR. Includes the over-blocking M6/M8 -- the driver MEASURES the "
           "full band; the board ranks, not the cost proxy.">
  ];
  let constructor =
      "::weft::transforms::createMaterializeRVVGemmSchedulePass()";
  let dependentDialects = [
    "::weft::rvv::WEFTRVVDialect"
  ];
}

def MaterializeEmitCLowerableRoutes
    : Pass<"weft-materialize-emitc-lowerable-routes", "::mlir::ModuleOp"> {
  let summary = "Materialize plugin-owned EmitC lowerable routes through the common route interface";
  let description = [{
    Finds a bounded direct weft.exec.variant, builds target capabilities from
    the enclosing kernel, asks the variant origin plugin for an
    extension-family EmitC lowerable route, and replaces the module with the
    materialized MLIR EmitC module produced by the common
    WEFTEmitCLowerableRoute materializer.

    The pass is extension-neutral. It routes by the generic origin plugin and
    does not branch on RVV, scalar, IME, offload, vendor, dtype, LMUL,
    intrinsic spelling, runtime ABI roles, or artifact route metadata. Family
    semantics and intrinsic/runtime-call mapping stay in the origin plugin.
    This pass materializes EmitC IR only; it does not print C/C++, export
    target artifacts, compile objects, run hardware, prove correctness, or
    claim performance.
  }];
  let constructor =
      "::weft::transforms::createMaterializeEmitCLowerableRoutesPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect",
    "::mlir::emitc::EmitCDialect"
  ];
}

def CheckExecutionPlanCoherence
    : Pass<"weft-check-execution-plan-coherence", "::mlir::ModuleOp"> {
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
      "::weft::transforms::createCheckExecutionPlanCoherencePass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect"
  ];
}

def RVVLowerToEmitC
    : Pass<"weft-rvv-lower-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower selected generic weft_rvv dataflow bodies to MLIR EmitC via DialectConversion";
  let description = [{
    Stands up the real MLIR DialectConversion infrastructure (TypeConverter +
    ConversionTarget + RewritePatternSet + applyPartialConversion) over the
    existing generic typed weft_rvv dataflow ops, the typed-IR replacement for
    the legacy RVV emitted-C string machine.

    In this initial state the pass is an additive structural no-op: it builds
    the beachhead !weft_rvv.* -> emitc C-type conversions, marks the emitc
    dialect legal, marks every op (including weft_rvv ops) legal, and registers
    zero conversion patterns. Running it on any module leaves the IR unchanged.
    Real per-op conversion patterns for the unit-stride elementwise add
    beachhead family land in a subsequent step.

    The pass is plugin-neutral conversion scaffolding. It does not print C/C++,
    export target artifacts, compile objects, run hardware, prove correctness,
    or claim performance.
  }];
  let constructor =
      "::weft::transforms::createRVVLowerToEmitCPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect",
    "::weft::rvv::WEFTRVVDialect",
    "::mlir::emitc::EmitCDialect"
  ];
}

def RVVLowerQuantContraction
    : Pass<"weft-rvv-lower-quant-contraction", "::mlir::ModuleOp"> {
  let summary = "Select + lower the abstract weft_rvv.quant_contraction op to a concrete contraction op (stage-B capability-fact-driven path selection)";
  let description = [{
    The option-2 front-of-pipeline pass that lowers the abstract,
    algorithm-UNCOMMITTED weft_rvv.quant_contraction op to a CONCRETE
    contraction op, running BEFORE the EmitC lowering.

    STAGE B makes "the COMPILER itself selects repack-vs-block-dot from
    capability facts" actually true. Per quant_contraction request the pass
    derives the target VLEN from the selected -march (deriveMinimumVLEN -- the
    SAME capability authority every other capability-gated pass uses; the op's
    advisory min_vlen attr is NOT the source), lifts the committed WHAT attrs
    (quant, m_regime), and calls the pure, branch-free, capability-fact-driven
    selectContractionAlgorithm. The decision is stamped on the lowered op as
    three INERT audit attrs (weft_rvv.contraction_algorithm /
    path_selection_reason / path_materialization) so it is provable in-IR.

    THE CRUX (Option (i)): the concrete repack target needs pre-interleaved
    block_q4_0x16 weights the abstract op's PLAIN stride-18 weights cannot
    supply -- that plain->x16 weight materialization is stage C. So this pass
    emits the BYTE-IDENTICAL weft_rvv.q4_0_q8_0_block_dot body for BOTH the
    Repack-SELECTED and the BlockDot-SELECTED branches, differentiating ONLY via
    the inert audit attrs (Repack-selected => path_materialization =
    "deferred-stage-c"). So the emitted C is byte-identical on EVERY path: the
    compiler MAKES + RECORDS the selection in-compiler (the N3 novelty) with NO
    e2e algorithm switch and NO perf claim; the e2e materialization is stage C.

    The block-dot branch DROPS the abstract op's column_count (nc) operand: the
    block-dot kernel (ggml_vec_dot_q4_0_q8_0) writes ONE fp32 and delegates the
    M/N loops to ggml's mul_mat caller, so it is a bare 4-operand vec_dot. The
    abstract op carries nc ALWAYS so a later repack branch can reach it; here it
    is dropped to keep the emitted C byte-identical. NO schedule attrs are
    stamped (integer_core_lmul / multi_block_factor / strip_elision) -- those
    remain the downstream MaterializeRVVQ40Schedule pass's job. On any module
    with no quant_contraction op the pass is a structural no-op.
  }];
  let options = kRVVBlockDotScheduleOptions;
  let constructor =
      "::weft::transforms::createRVVLowerQuantContractionPass()";
  let dependentDialects = [
    "::weft::rvv::WEFTRVVDialect"
  ];
}

#endif // WEFT_TRANSFORMS_PASSES_TD
