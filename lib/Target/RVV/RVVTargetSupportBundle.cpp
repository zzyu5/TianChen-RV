#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::target::rvv {
namespace {

constexpr llvm::StringLiteral kRVVEmitCToCppRouteID(
    "tcrv-rvv-emitc-to-cpp");

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportMaterializedEmitCModuleToCpp(
      module, os, "RVV EmitC C/C++ translate route");
}

} // namespace

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  (void)bundle;
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (registry.lookup(kRVVEmitCToCppRouteID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      kRVVEmitCToCppRouteID,
      "export a materialized RVV EmitC module through the MLIR EmitC "
      "C/C++ emitter",
      exportMaterializedRVVEmitCToCpp));
}

} // namespace tianchenrv::target::rvv
