#include "Weft/InitWeftDialects.h"
#include "Weft/Plugin/BuiltinExtensionPlugins.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Target/BuiltinTargetArtifactExporters.h"
#include "Weft/Target/TargetArtifactExport.h"
#include "Weft/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace {

llvm::cl::opt<bool> disableBuiltinPlugins(
    "weft-disable-builtin-plugins",
    llvm::cl::desc("Run weft-opt with an empty extension plugin registry"),
    llvm::cl::init(false));

bool shouldDisableBuiltinPlugins(int argc, char **argv) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef argument(argv[index]);
    if (argument == "--weft-disable-builtin-plugins")
      return true;

    if (!argument.consume_front("--weft-disable-builtin-plugins="))
      continue;
    return argument == "1" || argument == "true" || argument == "TRUE";
  }

  return false;
}

llvm::Error registerWeftOptPasses(
    const weft::plugin::ExtensionPluginRegistry &plugins,
    const weft::target::TargetArtifactExporterRegistry
        &targetExporters) {
  mlir::registerPass(
      [] { return weft::transforms::createCheckCapabilityRequiresPass(); });
  mlir::registerPass([] {
    return weft::transforms::createCheckHartParallelCapabilitiesPass();
  });
  mlir::registerPass(
      [] { return weft::transforms::createSynthesizeVariantDispatchPass(); });
  mlir::registerPass([&plugins] {
    return weft::transforms::createMaterializePluginVariantsPass(plugins);
  });
  mlir::registerPass([&plugins] {
    return weft::transforms::createVerifyPluginVariantLegalityPass(
        plugins);
  });
  mlir::registerPass([&plugins] {
    return weft::transforms::createSelectVariantsPass(plugins);
  });
  mlir::registerPass([] {
    return weft::transforms::createMaterializeDispatchRuntimeGuardsPass();
  });
  mlir::registerPass([&plugins] {
    return weft::transforms::createCheckEmissionPathsPass(plugins);
  });
  mlir::registerPass([&plugins] {
    return weft::transforms::createMaterializeEmissionPlansPass(plugins);
  });
  mlir::registerPass([&plugins] {
    return weft::transforms::
        createMaterializeSelectedLoweringBoundariesPass(plugins);
  });
  mlir::registerPass([] {
    return weft::transforms::
        createMaterializeRVVProbedCapabilityAxesPass();
  });
  mlir::registerPass([] {
    return weft::transforms::createMaterializeRVVSchedulePass();
  });
  mlir::registerPass([&plugins] {
    return weft::transforms::createConstructRVVFormulaPlansPass(plugins);
  });
  mlir::registerPass([&plugins] {
    return weft::transforms::createMaterializeEmitCLowerableRoutesPass(
        plugins);
  });
  mlir::registerPass([] {
    return weft::transforms::createRVVLowerToEmitCPass();
  });
  mlir::registerPass([] {
    return weft::transforms::createRVVLowerQuantContractionPass();
  });
  llvm::SmallVector<weft::plugin::SourceFrontDoorPassRegistration, 4>
      sourceFrontDoorPasses;
  if (llvm::Error error =
          plugins.collectSourceFrontDoorPasses(sourceFrontDoorPasses))
    return error;
  for (const weft::plugin::SourceFrontDoorPassRegistration
           &sourceFrontDoorPass : sourceFrontDoorPasses) {
    mlir::registerPass([sourceFrontDoorPass] {
      return sourceFrontDoorPass.getFactory()();
    });
  }
  mlir::registerPass([&plugins, &targetExporters] {
    return weft::transforms::createCheckExecutionPlanCoherencePass(
        plugins, targetExporters);
  });
  weft::transforms::registerSourceArtifactFrontDoorPipeline(
      sourceFrontDoorPasses, plugins, targetExporters);
  weft::transforms::registerRVVLowerToEmitCPipeline(plugins);
  weft::transforms::registerExecutionPlanningPipeline(plugins,
                                                            targetExporters);
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  bool useBuiltinPlugins = !shouldDisableBuiltinPlugins(argc, argv);
  weft::plugin::ExtensionBundleRegistry bundles;
  weft::plugin::ExtensionPluginRegistry plugins;
  weft::target::TargetArtifactExporterRegistry targetExporters;
  if (useBuiltinPlugins) {
    if (llvm::Error error =
            weft::plugin::registerBuiltinExtensionBundlePlugins(
                bundles, plugins)) {
      llvm::errs() << "failed to register Weft-RV built-in extension "
                      "bundle front door: "
                   << llvm::toString(std::move(error)) << "\n";
      return 1;
    }
    if (llvm::Error error =
            weft::target::registerBuiltinTargetArtifactExporters(
                targetExporters, bundles, plugins)) {
      llvm::errs() << "failed to register Weft-RV built-in target "
                      "artifact exporters: "
                   << llvm::toString(std::move(error)) << "\n";
      return 1;
    }
  }
  if (llvm::Error error =
          registerWeftOptPasses(plugins, targetExporters)) {
    llvm::errs() << "failed to register Weft-RV optimizer passes: "
                 << llvm::toString(std::move(error)) << "\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  weft::registerAllDialects(registry);
  weft::registerPluginDialects(plugins, registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Weft-RV optimizer driver\n", registry));
}
