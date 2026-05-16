#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/Operation.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::target::rvv {
namespace {

constexpr llvm::StringLiteral kRVVEmitCToCppRouteID(
    "tcrv-rvv-emitc-to-cpp");

llvm::Error makeRVVEmitCToCppRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV EmitC C/C++ translate route failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error requireMaterializedEmitCModule(mlir::ModuleOp module) {
  bool foundEmitCOp = false;
  mlir::Operation *unsupported = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (op == module.getOperation())
      return mlir::WalkResult::advance();

    llvm::StringRef dialectNamespace = op->getName().getDialectNamespace();
    if (dialectNamespace == "emitc") {
      foundEmitCOp = true;
      return mlir::WalkResult::advance();
    }

    unsupported = op;
    return mlir::WalkResult::interrupt();
  });

  if (unsupported)
    return makeRVVEmitCToCppRouteError(
        llvm::Twine("requires an already materialized EmitC module; found "
                    "non-EmitC op '") +
        unsupported->getName().getStringRef() + "'");
  if (!foundEmitCOp)
    return makeRVVEmitCToCppRouteError(
        "requires an already materialized EmitC module with at least one "
        "EmitC op");
  return llvm::Error::success();
}

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  if (llvm::Error error = requireMaterializedEmitCModule(module))
    return error;
  if (mlir::failed(mlir::emitc::translateToCpp(module.getOperation(), os)))
    return makeRVVEmitCToCppRouteError(
        "upstream MLIR EmitC C/C++ emitter rejected the materialized EmitC "
        "module");
  return llvm::Error::success();
}

} // namespace

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (registry.lookup(kRVVEmitCToCppRouteID))
    return llvm::Error::success();
  return registry.registerRoute(TargetTranslateRoute(
      kRVVEmitCToCppRouteID,
      "export a materialized RVV EmitC module through the MLIR EmitC C/C++ "
      "emitter",
      exportMaterializedRVVEmitCToCpp));
}

} // namespace tianchenrv::target::rvv
