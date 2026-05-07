#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/EmissionManifest.h"
#include "TianChenRV/Target/RVV/RVVSmokeProbe.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace {

void registerTianChenRVTranslateDialects(mlir::DialectRegistry &registry) {
  tianchenrv::registerAllDialects(registry);

  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error =
          tianchenrv::plugin::registerBuiltinExtensionPlugins(plugins)) {
    llvm::report_fatal_error(
        llvm::Twine("failed to register TianChen-RV built-in extension "
                    "plugins for tcrv-translate: ") +
        llvm::toString(std::move(error)));
  }
  tianchenrv::registerPluginDialects(plugins, registry);
}

mlir::LogicalResult exportEmissionManifest(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  if (llvm::Error error =
          tianchenrv::target::exportEmissionManifest(module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportRVVSmokeProbeC(mlir::ModuleOp module,
                                         llvm::raw_ostream &os) {
  if (llvm::Error error =
          tianchenrv::target::rvv::exportRVVSmokeProbeC(module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

void registerTianChenRVTranslations() {
  static mlir::TranslateFromMLIRRegistration emissionManifest(
      "tcrv-export-emission-manifest",
      "export TianChen-RV selected emission handoff manifest",
      exportEmissionManifest, registerTianChenRVTranslateDialects);
  (void)emissionManifest;

  static mlir::TranslateFromMLIRRegistration rvvSmokeProbeC(
      "tcrv-export-rvv-smoke-probe-c",
      "export a bounded RVV hardware/toolchain smoke-probe C source",
      exportRVVSmokeProbeC, registerTianChenRVTranslateDialects);
  (void)rvvSmokeProbeC;
}

} // namespace

int main(int argc, char **argv) {
  registerTianChenRVTranslations();
  return mlir::failed(mlir::mlirTranslateMain(
      argc, argv, "TianChen-RV translation driver\n"));
}
