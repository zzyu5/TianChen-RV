#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"
#include "TianChenRV/Target/EmissionManifest.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"
#include "TianChenRV/Transforms/ExecutionPlanCoherence.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>
#include <vector>

namespace {

using TargetArtifactExportFn = llvm::Error (*)(
    mlir::ModuleOp, const tianchenrv::target::TargetArtifactExporterRegistry &,
    llvm::raw_ostream &);

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

llvm::cl::opt<std::string> targetArtifactBundleOutputDirectory(
    "tcrv-target-artifact-bundle-output-dir",
    llvm::cl::desc("output directory for target artifact bundle export"),
    llvm::cl::value_desc("directory"), llvm::cl::init(""));

llvm::cl::opt<bool> disableBuiltinPlugins(
    "tcrv-disable-builtin-plugins",
    llvm::cl::desc("Run tcrv-translate with an empty extension plugin registry"),
    llvm::cl::init(false));

llvm::Error populateBuiltinExtensionFrontDoor(
    tianchenrv::plugin::ExtensionBundleRegistry &bundles,
    tianchenrv::plugin::ExtensionPluginRegistry &plugins) {
  if (disableBuiltinPlugins)
    return llvm::Error::success();
  return tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(bundles,
                                                                   plugins);
}

mlir::LogicalResult populateBuiltinPlanningRegistries(
    mlir::ModuleOp module, tianchenrv::plugin::ExtensionPluginRegistry &plugins,
    tianchenrv::target::TargetArtifactExporterRegistry &exporters) {
  tianchenrv::plugin::ExtensionBundleRegistry bundles;
  if (llvm::Error error = populateBuiltinExtensionFrontDoor(bundles, plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error =
          tianchenrv::target::registerBuiltinTargetArtifactExporters(
              exporters, bundles, plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  return mlir::success();
}

void registerTianChenRVTranslateDialects(mlir::DialectRegistry &registry) {
  tianchenrv::registerAllDialects(registry);
  registry.insert<mlir::emitc::EmitCDialect>();

  tianchenrv::plugin::ExtensionBundleRegistry bundles;
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error = populateBuiltinExtensionFrontDoor(bundles, plugins)) {
    llvm::report_fatal_error(
        llvm::Twine("failed to register TianChen-RV built-in extension "
                    "bundle front door for tcrv-translate: ") +
        llvm::toString(std::move(error)));
  }
  tianchenrv::registerPluginDialects(plugins, registry);
}

mlir::LogicalResult exportEmissionManifest(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  tianchenrv::plugin::ExtensionBundleRegistry bundles;
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error = populateBuiltinExtensionFrontDoor(bundles, plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  tianchenrv::target::TargetArtifactExporterRegistry exporters;
  if (llvm::Error error =
          tianchenrv::target::registerBuiltinTargetArtifactExporters(
              exporters, bundles, plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error =
          tianchenrv::target::exportEmissionManifest(module, exporters, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

bool hasEmissionPlanDiagnostics(mlir::ModuleOp module);
bool hasMaterializedExecutionVariants(mlir::ModuleOp module);
bool hasMaterializedSelectedLoweringBoundaries(mlir::ModuleOp module);

mlir::LogicalResult exportCoherenceGatedTargetArtifactRoute(
    mlir::ModuleOp module, llvm::raw_ostream &os, llvm::StringRef routeID);

mlir::LogicalResult planAndExportTargetTranslateArtifactRoute(
    mlir::ModuleOp module, const tianchenrv::target::TargetTranslateRoute &route,
    llvm::raw_ostream &os);

mlir::LogicalResult runSourceArtifactFrontDoorPipeline(
    mlir::ModuleOp module,
    const tianchenrv::plugin::ExtensionPluginRegistry &plugins,
    const tianchenrv::target::TargetArtifactExporterRegistry &exporters,
    llvm::StringRef routeDescription);

mlir::LogicalResult
exportTargetTranslateRoute(mlir::ModuleOp module,
                           const tianchenrv::target::TargetTranslateRoute &route,
                           llvm::raw_ostream &os) {
  if (route.requiresBinaryStdout()) {
    if (std::error_code error = llvm::sys::ChangeStdoutToBinary()) {
      module.emitError()
          << "failed to switch stdout to binary mode for object export: "
          << error.message();
      return mlir::failure();
    }
  }

  if (route.hasTargetArtifactRouteID() && hasEmissionPlanDiagnostics(module))
    return exportCoherenceGatedTargetArtifactRoute(
        module, os, route.getTargetArtifactRouteID());
  if (route.hasTargetArtifactRouteID())
    return planAndExportTargetTranslateArtifactRoute(module, route, os);

  if (llvm::Error error = route.getExportFn()(module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult runSourceArtifactFrontDoorPipeline(
    mlir::ModuleOp module,
    const tianchenrv::plugin::ExtensionPluginRegistry &plugins,
    const tianchenrv::target::TargetArtifactExporterRegistry &exporters,
    llvm::StringRef routeDescription) {
  llvm::SmallVector<tianchenrv::plugin::SourceFrontDoorPassRegistration, 4>
      sourceFrontDoorPasses;
  if (llvm::Error error =
          plugins.collectSourceFrontDoorPasses(sourceFrontDoorPasses)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (sourceFrontDoorPasses.empty()) {
    module.emitError()
        << "TianChen-RV " << routeDescription
        << " requires at least one registered source front-door pass";
    return mlir::failure();
  }

  mlir::PassManager pm(module.getContext());
  tianchenrv::transforms::buildSourceArtifactFrontDoorPipeline(
      pm, sourceFrontDoorPasses, plugins, exporters);
  if (mlir::failed(pm.run(module))) {
    module.emitError()
        << "TianChen-RV " << routeDescription
        << " failed during source-artifact front-door pipeline";
    return mlir::failure();
  }

  return mlir::success();
}

void registerBuiltinTargetTranslateRouteTranslations() {
  static tianchenrv::target::TargetTranslateRouteRegistry routeRegistry;
  static std::vector<std::unique_ptr<mlir::TranslateFromMLIRRegistration>>
      registrations;
  static bool initialized = false;
  if (!registrations.empty())
    return;

  if (!initialized) {
    if (llvm::Error error =
            tianchenrv::target::registerBuiltinTargetTranslateRoutes(
                routeRegistry)) {
      llvm::report_fatal_error(
          llvm::Twine("failed to register TianChen-RV built-in target "
                      "translate routes for tcrv-translate: ") +
          llvm::toString(std::move(error)));
    }
    initialized = true;
  }

  for (const tianchenrv::target::TargetTranslateRoute &route :
       routeRegistry.getRoutes()) {
    const tianchenrv::target::TargetTranslateRoute *routePtr = &route;
    mlir::TranslateFromMLIRFunction translate =
        [routePtr](mlir::Operation *op,
                   llvm::raw_ostream &os) -> mlir::LogicalResult {
      auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
      if (!module)
        return op->emitError()
               << "expected a 'builtin.module' op for TianChen-RV target "
                  "translate route '"
               << routePtr->getRouteID() << "'";
      return exportTargetTranslateRoute(module, *routePtr, os);
    };

    registrations.push_back(
        std::make_unique<mlir::TranslateFromMLIRRegistration>(
            route.getRouteID(), route.getDescription(), translate,
            registerTianChenRVTranslateDialects));
  }
}

bool hasEmissionPlanDiagnostics(mlir::ModuleOp module) {
  bool found = false;
  module->walk([&](mlir::Operation *op) {
    auto reason =
        op->getAttrOfType<mlir::StringAttr>(execDiagnostic::kReasonAttrName);
    if (!reason || !execDiagnostic::isEmissionPlanReason(reason.getValue()))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

bool hasMaterializedExecutionVariants(mlir::ModuleOp module) {
  bool found = false;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "tcrv.exec.variant")
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

bool hasMaterializedSelectedLoweringBoundaries(mlir::ModuleOp module) {
  bool found = false;
  module->walk([&](mlir::Operation *op) {
    if (!op->getName().getStringRef().ends_with(".lowering_boundary"))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

mlir::LogicalResult
exportCoherenceGatedTargetArtifact(mlir::ModuleOp module, llvm::raw_ostream &os,
                                   TargetArtifactExportFn exportFn) {
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  tianchenrv::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  if (llvm::Error error =
          tianchenrv::transforms::checkExecutionPlanCoherence(module, plugins,
                                                              exporters)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error = exportFn(module, exporters, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportCoherenceGatedTargetArtifactRoute(
    mlir::ModuleOp module, llvm::raw_ostream &os, llvm::StringRef routeID) {
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  tianchenrv::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  if (llvm::Error error =
          tianchenrv::transforms::checkExecutionPlanCoherence(module, plugins,
                                                              exporters)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error = tianchenrv::target::exportTargetArtifactRoute(
          module, exporters, routeID, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult planAndExportTargetTranslateArtifactRoute(
    mlir::ModuleOp module, const tianchenrv::target::TargetTranslateRoute &route,
    llvm::raw_ostream &os) {
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  tianchenrv::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  mlir::PassManager pm(module.getContext());
  if (hasMaterializedSelectedLoweringBoundaries(module)) {
    pm.addPass(tianchenrv::transforms::createMaterializeEmissionPlansPass(
        plugins));
    pm.addPass(tianchenrv::transforms::createCheckExecutionPlanCoherencePass(
        plugins, exporters));
  } else if (hasMaterializedExecutionVariants(module)) {
    pm.addPass(
        tianchenrv::transforms::createMaterializeSelectedLoweringBoundariesPass(
            plugins));
    pm.addPass(tianchenrv::transforms::createMaterializeEmissionPlansPass(
        plugins));
    pm.addPass(tianchenrv::transforms::createCheckExecutionPlanCoherencePass(
        plugins, exporters));
  } else {
    tianchenrv::transforms::buildExecutionPlanningPipeline(pm, plugins,
                                                           exporters);
  }

  if (mlir::failed(pm.run(module))) {
    module.emitError()
        << "TianChen-RV artifact-backed direct translate route '"
        << route.getRouteID()
        << "' failed during execution planning before exact target artifact "
           "export";
    return mlir::failure();
  }

  if (llvm::Error error = tianchenrv::target::exportTargetArtifactRoute(
          module, exporters, route.getTargetArtifactRouteID(), os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportTargetArtifact(mlir::ModuleOp module,
                                         llvm::raw_ostream &os) {
  return exportCoherenceGatedTargetArtifact(
      module, os, tianchenrv::target::exportTargetArtifact);
}

mlir::LogicalResult exportTargetHeaderArtifact(mlir::ModuleOp module,
                                               llvm::raw_ostream &os) {
  return exportCoherenceGatedTargetArtifact(
      module, os, tianchenrv::target::exportTargetHeaderArtifact);
}

mlir::LogicalResult exportTargetArtifactBundle(mlir::ModuleOp module,
                                               llvm::raw_ostream &os) {
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  tianchenrv::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  if (llvm::Error error =
          tianchenrv::transforms::checkExecutionPlanCoherence(module, plugins,
                                                              exporters)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error = tianchenrv::target::exportTargetArtifactBundle(
          module, exporters, targetArtifactBundleOutputDirectory)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  os << "tianchenrv.target_artifact_bundle_export: complete\n";
  os << "index_file: \"tianchenrv-target-artifact-bundle.index\"\n";
  return mlir::success();
}

mlir::LogicalResult
sourceArtifactBundleFrontDoor(mlir::ModuleOp module, llvm::raw_ostream &os) {
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  tianchenrv::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  constexpr llvm::StringLiteral routeDescription(
      "source-artifact bundle front door");
  if (mlir::failed(runSourceArtifactFrontDoorPipeline(
          module, plugins, exporters, routeDescription)))
    return mlir::failure();

  if (llvm::Error error = tianchenrv::target::exportTargetArtifactBundle(
          module, exporters, targetArtifactBundleOutputDirectory)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  os << "tianchenrv.target_artifact_bundle_export: complete\n";
  os << "index_file: \"tianchenrv-target-artifact-bundle.index\"\n";
  return mlir::success();
}

mlir::LogicalResult planAndExportTargetArtifactBundle(mlir::ModuleOp module,
                                                      llvm::raw_ostream &os) {
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  tianchenrv::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  mlir::PassManager pm(module.getContext());
  tianchenrv::transforms::buildExecutionPlanningPipeline(pm, plugins,
                                                         exporters);
  if (mlir::failed(pm.run(module))) {
    module.emitError()
        << "TianChen-RV plan-and-export target artifact bundle failed during "
           "execution planning pipeline";
    return mlir::failure();
  }

  if (llvm::Error error = tianchenrv::target::exportTargetArtifactBundle(
          module, exporters, targetArtifactBundleOutputDirectory)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  os << "tianchenrv.target_artifact_bundle_export: complete\n";
  os << "index_file: \"tianchenrv-target-artifact-bundle.index\"\n";
  return mlir::success();
}

void registerTianChenRVTranslations() {
  static mlir::TranslateFromMLIRRegistration emissionManifest(
      "tcrv-export-emission-manifest",
      "export TianChen-RV selected emission handoff manifest",
      exportEmissionManifest, registerTianChenRVTranslateDialects);
  (void)emissionManifest;

  registerBuiltinTargetTranslateRouteTranslations();

  static mlir::TranslateFromMLIRRegistration targetArtifact(
      "tcrv-export-target-artifact",
      "export one supported TianChen-RV target artifact route",
      exportTargetArtifact, registerTianChenRVTranslateDialects);
  (void)targetArtifact;

  static mlir::TranslateFromMLIRRegistration targetHeaderArtifact(
      "tcrv-export-target-header-artifact",
      "export one supported TianChen-RV target C header artifact route",
      exportTargetHeaderArtifact, registerTianChenRVTranslateDialects);
  (void)targetHeaderArtifact;

  static mlir::TranslateFromMLIRRegistration targetArtifactBundle(
      "tcrv-export-target-artifact-bundle",
      "export selected TianChen-RV target artifacts into an output directory",
      exportTargetArtifactBundle, registerTianChenRVTranslateDialects);
  (void)targetArtifactBundle;

  static mlir::TranslateFromMLIRRegistration
      planAndExportTargetArtifactBundleRegistration(
      "tcrv-plan-and-export-target-artifact-bundle",
      "run TianChen-RV execution planning and export selected target artifacts "
      "into an output directory",
      planAndExportTargetArtifactBundle, registerTianChenRVTranslateDialects);
  (void)planAndExportTargetArtifactBundleRegistration;

  static mlir::TranslateFromMLIRRegistration
      sourceArtifactBundleFrontDoorRegistration(
      "tcrv-source-artifact-bundle-front-door",
      "run plugin-registered TianChen-RV source front doors and export selected "
      "target artifacts into an output directory",
      sourceArtifactBundleFrontDoor, registerTianChenRVTranslateDialects);
  (void)sourceArtifactBundleFrontDoorRegistration;
}

} // namespace

int main(int argc, char **argv) {
  registerTianChenRVTranslations();
  return mlir::failed(mlir::mlirTranslateMain(
      argc, argv, "TianChen-RV translation driver\n"));
}
