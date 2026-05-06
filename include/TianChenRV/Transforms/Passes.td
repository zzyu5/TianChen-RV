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

#endif // TIANCHENRV_TRANSFORMS_PASSES_TD
