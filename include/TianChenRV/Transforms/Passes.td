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

#endif // TIANCHENRV_TRANSFORMS_PASSES_TD
