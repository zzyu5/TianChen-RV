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
    stay origin-plugin-owned. The default public registration uses an empty
    registry and therefore diagnoses kernels whose variants name unregistered
    origin plugins rather than inventing core-side cost semantics.

    Runtime-dispatch plans materialize typed tcrv.exec.dispatch,
    tcrv.exec.case, and tcrv.exec.fallback operations. Static, fallback-only,
    and no-variant plans do not erase variants or inject target-specific IR.
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
    dispatch or selected-marker surface are conservatively checked through all
    direct tcrv.exec.variant children. Production tools should construct this
    pass with a populated ExtensionPluginRegistry. The default public
    registration uses an empty registry and diagnoses unregistered origins.
  }];
  let constructor =
      "::tianchenrv::transforms::createCheckEmissionPathsPass()";
  let dependentDialects = [
    "::tianchenrv::tcrv::exec::TCRVExecDialect"
  ];
}

#endif // TIANCHENRV_TRANSFORMS_PASSES_TD
