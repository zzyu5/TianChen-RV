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
    non-fallback variants when they carry a non-empty generic condition, guard,
    or policy, while dispatch fallback targets must be generically available.
    Malformed requires attributes and unknown symbols remain the responsibility
    of the tcrv.exec verifier.
  }];
  let constructor =
      "::tianchenrv::transforms::createCheckCapabilityRequiresPass()";
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
    tcrv.exec.dispatch. The first direct variant whose requirements are
    generically available under the kernel TargetCapabilitySet becomes the
    fallback target. Remaining variants are emitted as deterministic dispatch
    cases in original kernel order, with a non-empty generic policy attached to
    cases whose requirements are unavailable and therefore require runtime
    dispatch guarding. Kernels without a generically available fallback are
    diagnosed and left unmodified.
  }];
  let constructor =
      "::tianchenrv::transforms::createSynthesizeVariantDispatchPass()";
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
    construct this pass with a populated ExtensionPluginRegistry so variant costs
    stay origin-plugin-owned. Public tcrv-opt registration owns a deterministic
    built-in registry at the tool boundary, while the default factory constructor
    still uses an empty registry for embedded negative tests and therefore
    diagnoses kernels whose variants name unregistered origin plugins rather than
    inventing core-side cost semantics.

    Runtime-dispatch plans materialize typed tcrv.exec.dispatch,
    tcrv.exec.case, and tcrv.exec.fallback operations only when a plugin-provided
    generic conservative fallback candidate is available. Static and
    fallback-only plans materialize a generic selected-path tcrv.exec.diagnostic
    marker with target and selection_kind metadata; static plans without fallback
    coverage also materialize a missing-fallback diagnostic instead of inventing
    an implicit fallback. No-variant plans do not erase variants or inject
    target-specific IR.
  }];
  let constructor =
      "::tianchenrv::transforms::createSelectVariantsPass()";
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

def MaterializeRVVLoweringBoundary
    : Pass<"tcrv-materialize-rvv-lowering-boundary", "::mlir::ModuleOp"> {
  let summary = "Materialize pre-executable RVV lowering-boundary metadata";
  let description = [{
    Materializes plugin-owned `tcrv_rvv.lowering_boundary` operations for
    selected RVV dispatch-case or direct selected variants. The pass consumes
    generic `tcrv.exec` selected-path structure and routes RVV-owned variants
    through the extension plugin registry before creating RVV dialect metadata.
    It does not materialize dispatch fallback boundaries, lower to LLVM/RISC-V,
    emit RVV intrinsics, create runtime ABI glue, generate objects, run
    hardware, prove correctness, or claim performance.

    Public tools should construct this pass with a populated
    ExtensionPluginRegistry. The default constructor owns an empty registry and
    is useful for negative registration tests.
  }];
  let constructor =
      "::tianchenrv::plugin::rvv::createMaterializeRVVLoweringBoundaryPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect",
    "::tianchenrv::tcrv::rvv::TCRVRVVDialect"
  ];
}

#endif // TIANCHENRV_TRANSFORMS_PASSES_TD
