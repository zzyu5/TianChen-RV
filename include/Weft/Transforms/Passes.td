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

def ConstructRVVFormulaPlans
    : Pass<"weft-rvv-construct-formula-plans", "::mlir::ModuleOp"> {
  let summary = "Run bound RVV family construction before artifact projection";
  let description = [{
    Binds every explicitly owned RVV variant to its enclosing kernel and typed
    target capability set, then invokes the RVV family construction contract.
    Formula construction produces final schedules and computation plans such as
    `flat_*`, and returns the exact final typed root. This pass does not inspect
    artifact routes, emit EmitC, or infer a body from kind/format metadata.

    An RVV operation without an owning `rvv-plugin` variant is not adopted by
    this pass; the subsequent artifact projection therefore rejects it. This is
    a lifecycle stage, not a compatibility repair path.
  }];
  let constructor =
      "::weft::transforms::createConstructRVVFormulaPlansPass()";
  let dependentDialects = [
    "::weft::exec::WEFTExecDialect",
    "::weft::rvv::WEFTRVVDialect"
  ];
}

def RVVLowerToEmitC
    : Pass<"weft-rvv-project-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Project construction-qualified weft_rvv bodies to MLIR EmitC";
  let description = [{
    Mechanically lowers an already construction-qualified RVV typed body to
    EmitC. Formula construction and schedule selection are deliberately absent
    from this pass: `flat_*` and all other final computation plans must already
    exist, and the emitter consumes them directly. A module with no RVV op/type
    is a true no-op; once an RVV body is present, anything short of complete
    legalization is a pass failure.

    The user-facing `weft-rvv-lower-to-emitc` pipeline composes the separate
    bound construction pass with this artifact-only projection. This pass does
    not print C/C++, export target artifacts, compile objects, run hardware,
    prove correctness, or claim performance.
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
  let summary = "Construct a complete typed contraction body and final RVV schedule from formula inputs";
  let description = [{
    Consumes typed contraction geometry, canonical RVV capability facts and the
    bounded GEMV/GEMM regime. The family-local formula constructs the complete
    legal algorithm set and analytic prior; qualified measurement may select
    only an existing legal candidate. The selected result is realized directly
    as the final typed contraction/repack body.

    Before the pass succeeds, the same unified schedule formula used by the
    project-level construction lifecycle constructs or validates every tunable
    operation in the module. Consequently no successful output contains a
    missing or partial schedule tuple, and no downstream Q4/GEMM compatibility
    materializer is required. Formula reasons and measurement keys remain
    transient; only code-affecting typed fields and the real x16 layout contract
    enter IR. Unsupported geometry or an empty legal set fails closed.
  }];
  let options = kRVVBlockDotScheduleOptions;
  let constructor =
      "::weft::transforms::createRVVLowerQuantContractionPass()";
  let dependentDialects = [
    "::weft::rvv::WEFTRVVDialect"
  ];
}

#endif // WEFT_TRANSFORMS_PASSES_TD
