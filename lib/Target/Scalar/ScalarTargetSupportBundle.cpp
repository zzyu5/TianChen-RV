#include "Weft/Target/Scalar/ScalarTargetSupportBundle.h"

#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/Scalar/ScalarEmitCRouteProvider.h"
#include "Weft/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace weft::target::scalar_ext {
namespace {

llvm::Error makeScalarTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV portable-scalar materialized EmitC target "
                  "translate route failed: ") +
          message,
      llvm::errc::invalid_argument);
}

/// Lowers a selected `weft_scalar.compute_skeleton` body to a standalone EmitC
/// module through the shared typed-emission backend registry (which runs the
/// scalar backend emission driver on a clone), then renders it as pure-scalar
/// C/C++ through the upstream MLIR EmitC C/C++ emitter. The live module is
/// never mutated.
llvm::Error exportScalarEmitCToCpp(mlir::ModuleOp module,
                                   const plugin::ExtensionPluginRegistry &plugins,
                                   llvm::raw_ostream &os) {
  mlir::OwningOpRef<mlir::ModuleOp> constructed(module.clone());
  weft::exec::VariantOp selectedVariant;
  unsigned scalarVariants = 0;
  constructed->walk([&](weft::exec::VariantOp variant) {
    auto origin = variant->getAttrOfType<mlir::StringAttr>("origin");
    if (!origin || origin.getValue() != "scalar-plugin")
      return;
    selectedVariant = variant;
    ++scalarVariants;
  });
  if (scalarVariants != 1)
    return makeScalarTargetRouteError(
        "requires exactly one bound scalar variant before translation");
  plugin::FamilyConstructionResult construction;
  if (llvm::Error error = plugins.constructFormulaPlansForVariant(
          *constructed, selectedVariant, construction))
    return makeScalarTargetRouteError(
        llvm::Twine("Scalar family construction rejected the selected typed "
                    "body: ") +
        llvm::toString(std::move(error)));
  if (!construction.hasFinalBody())
    return makeScalarTargetRouteError(
        "selected scalar variant has no executable final typed body");
  mlir::OwningOpRef<mlir::ModuleOp> emitcModule =
      conversion::emitc::
          tryConvertConstructedModuleWithRegisteredBackend(*constructed);
  if (!emitcModule)
    return makeScalarTargetRouteError(
        "no registered backend emission driver fully legalizes the selected "
        "portable-scalar body to EmitC");
  if (mlir::failed(mlir::emitc::translateToCpp(emitcModule->getOperation(), os)))
    return makeScalarTargetRouteError(
        "upstream MLIR EmitC C/C++ emitter rejected the materialized "
        "portable-scalar EmitC module");
  return llvm::Error::success();
}

} // namespace

llvm::StringRef getScalarEmitCToCppTranslateRouteID() {
  return plugin::scalar::getScalarEmitCConstructionRoute().translateRouteID;
}

llvm::Error registerScalarTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  const plugin::scalar::ScalarEmitCConstructionRoute &route =
      plugin::scalar::getScalarEmitCConstructionRoute();
  if (registry.lookup(route.translateRouteID))
    return llvm::Error::success();
  return registry.registerRoute(TargetTranslateRoute(
      route.translateRouteID, route.translateRouteDescription,
      exportScalarEmitCToCpp));
}

} // namespace weft::target::scalar_ext
