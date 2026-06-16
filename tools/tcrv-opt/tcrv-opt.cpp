#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Transforms/Passes.h"

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
    "tcrv-disable-builtin-plugins",
    llvm::cl::desc("Run tcrv-opt with an empty extension plugin registry"),
    llvm::cl::init(false));

bool shouldDisableBuiltinPlugins(int argc, char **argv) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef argument(argv[index]);
    if (argument == "--tcrv-disable-builtin-plugins")
      return true;

    if (!argument.consume_front("--tcrv-disable-builtin-plugins="))
      continue;
    return argument == "1" || argument == "true" || argument == "TRUE";
  }

  return false;
}

llvm::Error registerTianChenRVOptPasses(
    const tianchenrv::plugin::ExtensionPluginRegistry &plugins,
    const tianchenrv::target::TargetArtifactExporterRegistry
        &targetExporters) {
  mlir::registerPass(
      [] { return tianchenrv::transforms::createCheckCapabilityRequiresPass(); });
  mlir::registerPass([] {
    return tianchenrv::transforms::createCheckHartParallelCapabilitiesPass();
  });
  mlir::registerPass(
      [] { return tianchenrv::transforms::createSynthesizeVariantDispatchPass(); });
  mlir::registerPass([&plugins] {
    return tianchenrv::transforms::createMaterializePluginVariantsPass(plugins);
  });
  mlir::registerPass([&plugins] {
    return tianchenrv::transforms::createVerifyPluginVariantLegalityPass(
        plugins);
  });
  mlir::registerPass([&plugins] {
    return tianchenrv::transforms::createSelectVariantsPass(plugins);
  });
  mlir::registerPass([] {
    return tianchenrv::transforms::createMaterializeDispatchRuntimeGuardsPass();
  });
  mlir::registerPass([&plugins] {
    return tianchenrv::transforms::createCheckEmissionPathsPass(plugins);
  });
  mlir::registerPass([&plugins] {
    return tianchenrv::transforms::createMaterializeEmissionPlansPass(plugins);
  });
  mlir::registerPass([&plugins] {
    return tianchenrv::transforms::
        createMaterializeSelectedLoweringBoundariesPass(plugins);
  });
  mlir::registerPass([] {
    return tianchenrv::transforms::
        createMaterializeRVVProbedCapabilityAxesPass();
  });
  mlir::registerPass([] {
    return tianchenrv::transforms::createMaterializeRVVQ40SchedulePass();
  });
  mlir::registerPass([] {
    return tianchenrv::transforms::createMaterializeRVVQ80SchedulePass();
  });
  mlir::registerPass([] {
    return tianchenrv::transforms::createMaterializeRVVQ41SchedulePass();
  });
  mlir::registerPass([] {
    return tianchenrv::transforms::createMaterializeRVVQ50SchedulePass();
  });
  mlir::registerPass([] {
    return tianchenrv::transforms::createMaterializeRVVGearboxSchedulesPass();
  });
  mlir::registerPass([&plugins] {
    return tianchenrv::transforms::createMaterializeEmitCLowerableRoutesPass(
        plugins);
  });
  mlir::registerPass([] {
    return tianchenrv::transforms::createRVVLowerToEmitCPass();
  });
  llvm::SmallVector<tianchenrv::plugin::SourceFrontDoorPassRegistration, 4>
      sourceFrontDoorPasses;
  if (llvm::Error error =
          plugins.collectSourceFrontDoorPasses(sourceFrontDoorPasses))
    return error;
  for (const tianchenrv::plugin::SourceFrontDoorPassRegistration
           &sourceFrontDoorPass : sourceFrontDoorPasses) {
    mlir::registerPass([sourceFrontDoorPass] {
      return sourceFrontDoorPass.getFactory()();
    });
  }
  mlir::registerPass([&plugins, &targetExporters] {
    return tianchenrv::transforms::createCheckExecutionPlanCoherencePass(
        plugins, targetExporters);
  });
  tianchenrv::transforms::registerSourceArtifactFrontDoorPipeline(
      sourceFrontDoorPasses, plugins, targetExporters);
  tianchenrv::transforms::registerExecutionPlanningPipeline(plugins,
                                                            targetExporters);
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  bool useBuiltinPlugins = !shouldDisableBuiltinPlugins(argc, argv);
  tianchenrv::plugin::ExtensionBundleRegistry bundles;
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  tianchenrv::target::TargetArtifactExporterRegistry targetExporters;
  if (useBuiltinPlugins) {
    if (llvm::Error error =
            tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                bundles, plugins)) {
      llvm::errs() << "failed to register TianChen-RV built-in extension "
                      "bundle front door: "
                   << llvm::toString(std::move(error)) << "\n";
      return 1;
    }
    if (llvm::Error error =
            tianchenrv::target::registerBuiltinTargetArtifactExporters(
                targetExporters, bundles, plugins)) {
      llvm::errs() << "failed to register TianChen-RV built-in target "
                      "artifact exporters: "
                   << llvm::toString(std::move(error)) << "\n";
      return 1;
    }
  }
  if (llvm::Error error =
          registerTianChenRVOptPasses(plugins, targetExporters)) {
    llvm::errs() << "failed to register TianChen-RV optimizer passes: "
                 << llvm::toString(std::move(error)) << "\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  tianchenrv::registerAllDialects(registry);
  tianchenrv::registerPluginDialects(plugins, registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TianChen-RV optimizer driver\n", registry));
}
