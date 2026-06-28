#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Conversion/RVV/RVVToEmitC.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {
namespace {

// Stage 3 换心 retirement (I7). The RVV body-emission string machine (the
// per-family + residual statement-plan owners that synthesized std::string C
// expressions, plus the route-CONSTRUCTION half that dispatched into them) has
// been deleted: every covered RVV family now lowers through the real
// RVV->emitc DialectConversion (`conversion::rvv::convertRVVModuleToEmitC`),
// which is the authority. A selected RVV body that does NOT fully convert (a
// retired legacy form such as the two-scope Gearbox cross-region-handoff dequant
// body) therefore has no legal materialized route. Refuse it fail-closed with a
// bounded diagnostic carrying the op token, instead of resurrecting any string
// synthesis. This is the single chokepoint both the public route builder and
// the description's verified-route branch funnel through.
llvm::Error refuseRetiredRVVSelectedBodyStringRouteForOperation(
    RVVSelectedBodyOperationKind operation, llvm::StringRef context) {
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " selected RVV body does not fully lower through the RVV->emitc "
      "DialectConversion and the legacy statement-plan string route is "
      "retired; no legal materialized route remains for operation '" +
      stringifyRVVSelectedBodyOperationKind(operation) + "'");
}

} // namespace


bool rvvSelectedBodyFullyConvertsToEmitC(
    const VariantEmitCLowerableRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return false;
  auto module = variant->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return false;

  // Probe a clone so the live IR is never mutated: convertRVVModuleToEmitC
  // rewrites in place and, on a not-yet-covered family, fails partway. The
  // speculative failure is the normal strangler-fig fall-back signal, so
  // swallow its diagnostics rather than leak a spurious "failed to legalize"
  // to stderr; the real conversion seam (and the --tcrv-rvv-lower-to-emitc
  // pass) still surface diagnostics normally.
  mlir::OwningOpRef<mlir::ModuleOp> probe(module.clone());
  mlir::ScopedDiagnosticHandler quietTry(
      probe->getContext(),
      [](mlir::Diagnostic &) { return mlir::success(); });
  return conversion::rvv::convertRVVModuleToEmitC(*probe);
}

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const VariantEmitCLowerableRequest &request) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();

  return analysis->description;
}

llvm::Error refuseRetiredRVVSelectedBodyStringRoute(
    const VariantEmitCLowerableRequest &request) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();
  return refuseRetiredRVVSelectedBodyStringRouteForOperation(
      analysis->description.operation,
      "selected RVV EmitC route construction");
}

} // namespace tianchenrv::plugin::rvv
