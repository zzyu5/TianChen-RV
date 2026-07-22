#include "Weft/InitWeftDialects.h"
#include "Weft/Dialect/Exec/IR/DiagnosticConventions.h"
#include "Weft/Plugin/BuiltinExtensionPlugins.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Target/BuiltinTargetArtifactExporters.h"
#include "Weft/Target/BuiltinTargetTranslateRoutes.h"
#include "Weft/Target/EmissionManifest.h"
#include "Weft/Target/TargetArtifactExport.h"
#include "Weft/Target/TargetTranslateRegistration.h"
#include "Weft/Transforms/ExecutionPlanCoherence.h"
#include "Weft/Transforms/Passes.h"

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
    mlir::ModuleOp, const weft::target::TargetArtifactExporterRegistry &,
    const weft::plugin::ExtensionPluginRegistry &,
    llvm::raw_ostream &);

namespace execDiagnostic = weft::exec::diagnostic;

llvm::cl::opt<std::string> targetArtifactBundleOutputDirectory(
    "weft-target-artifact-bundle-output-dir",
    llvm::cl::desc("output directory for target artifact bundle export"),
    llvm::cl::value_desc("directory"), llvm::cl::init(""));

llvm::cl::opt<bool> disableBuiltinPlugins(
    "weft-disable-builtin-plugins",
    llvm::cl::desc("Run weft-translate with an empty extension plugin registry"),
    llvm::cl::init(false));

llvm::Error populateBuiltinExtensionFrontDoor(
    weft::plugin::ExtensionBundleRegistry &bundles,
    weft::plugin::ExtensionPluginRegistry &plugins) {
  if (disableBuiltinPlugins)
    return llvm::Error::success();
  return weft::plugin::registerBuiltinExtensionBundlePlugins(bundles,
                                                                   plugins);
}

mlir::LogicalResult populateBuiltinPlanningRegistries(
    mlir::ModuleOp module, weft::plugin::ExtensionPluginRegistry &plugins,
    weft::target::TargetArtifactExporterRegistry &exporters) {
  weft::plugin::ExtensionBundleRegistry bundles;
  if (llvm::Error error = populateBuiltinExtensionFrontDoor(bundles, plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error =
          weft::target::registerBuiltinTargetArtifactExporters(
              exporters, bundles, plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  return mlir::success();
}

void registerWeftTranslateDialects(mlir::DialectRegistry &registry) {
  weft::registerAllDialects(registry);
  registry.insert<mlir::emitc::EmitCDialect>();

  weft::plugin::ExtensionBundleRegistry bundles;
  weft::plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error = populateBuiltinExtensionFrontDoor(bundles, plugins)) {
    llvm::report_fatal_error(
        llvm::Twine("failed to register Weft-RV built-in extension "
                    "bundle front door for weft-translate: ") +
        llvm::toString(std::move(error)));
  }
  weft::registerPluginDialects(plugins, registry);
}

mlir::LogicalResult exportEmissionManifest(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  weft::plugin::ExtensionBundleRegistry bundles;
  weft::plugin::ExtensionPluginRegistry plugins;
  if (llvm::Error error = populateBuiltinExtensionFrontDoor(bundles, plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  weft::target::TargetArtifactExporterRegistry exporters;
  if (llvm::Error error =
          weft::target::registerBuiltinTargetArtifactExporters(
              exporters, bundles, plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error =
          weft::target::exportEmissionManifest(module, exporters, os)) {
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
    mlir::ModuleOp module, const weft::target::TargetTranslateRoute &route,
    llvm::raw_ostream &os);

mlir::LogicalResult runSourceArtifactFrontDoorPipeline(
    mlir::ModuleOp module,
    const weft::plugin::ExtensionPluginRegistry &plugins,
    const weft::target::TargetArtifactExporterRegistry &exporters,
    llvm::StringRef routeDescription);

mlir::LogicalResult
exportTargetTranslateRoute(mlir::ModuleOp module,
                           const weft::target::TargetTranslateRoute &route,
                           const weft::plugin::ExtensionPluginRegistry &plugins,
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

  if (llvm::Error error = route.getExportFn()(module, plugins, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult runSourceArtifactFrontDoorPipeline(
    mlir::ModuleOp module,
    const weft::plugin::ExtensionPluginRegistry &plugins,
    const weft::target::TargetArtifactExporterRegistry &exporters,
    llvm::StringRef routeDescription) {
  llvm::SmallVector<weft::plugin::SourceFrontDoorPassRegistration, 4>
      sourceFrontDoorPasses;
  if (llvm::Error error =
          plugins.collectSourceFrontDoorPasses(sourceFrontDoorPasses)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (sourceFrontDoorPasses.empty()) {
    module.emitError()
        << "Weft-RV " << routeDescription
        << " requires at least one registered source front-door pass";
    return mlir::failure();
  }

  mlir::PassManager pm(module.getContext());
  weft::transforms::buildSourceArtifactFrontDoorPipeline(
      pm, sourceFrontDoorPasses, plugins, exporters);
  if (mlir::failed(pm.run(module))) {
    module.emitError()
        << "Weft-RV " << routeDescription
        << " failed during source-artifact front-door pipeline";
    return mlir::failure();
  }

  return mlir::success();
}

void registerBuiltinTargetTranslateRouteTranslations() {
  static weft::target::TargetTranslateRouteRegistry routeRegistry;
  static weft::plugin::ExtensionBundleRegistry bundles;
  static weft::plugin::ExtensionPluginRegistry plugins;
  static std::vector<std::unique_ptr<mlir::TranslateFromMLIRRegistration>>
      registrations;
  static bool initialized = false;
  if (!registrations.empty())
    return;

  if (!initialized) {
    if (llvm::Error error = populateBuiltinExtensionFrontDoor(bundles, plugins)) {
      llvm::report_fatal_error(
          llvm::Twine("failed to register Weft-RV built-in extension "
                      "bundle front door for target translate routes: ") +
          llvm::toString(std::move(error)));
    }
    if (llvm::Error error =
            weft::target::registerBuiltinTargetTranslateRoutes(
                routeRegistry, bundles, plugins)) {
      llvm::report_fatal_error(
          llvm::Twine("failed to register Weft-RV built-in target "
                      "translate routes for weft-translate: ") +
          llvm::toString(std::move(error)));
    }
    initialized = true;
  }

  for (const weft::target::TargetTranslateRoute &route :
       routeRegistry.getRoutes()) {
    const weft::target::TargetTranslateRoute *routePtr = &route;
    const weft::plugin::ExtensionPluginRegistry *pluginsPtr = &plugins;
    mlir::TranslateFromMLIRFunction translate =
        [routePtr, pluginsPtr](mlir::Operation *op,
                              llvm::raw_ostream &os) -> mlir::LogicalResult {
      auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
      if (!module)
        return op->emitError()
               << "expected a 'builtin.module' op for Weft-RV target "
                  "translate route '"
               << routePtr->getRouteID() << "'";
      return exportTargetTranslateRoute(module, *routePtr, *pluginsPtr, os);
    };

    registrations.push_back(
        std::make_unique<mlir::TranslateFromMLIRRegistration>(
            route.getRouteID(), route.getDescription(), translate,
            registerWeftTranslateDialects));
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
    if (op->getName().getStringRef() != "weft.exec.variant")
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
  weft::plugin::ExtensionPluginRegistry plugins;
  weft::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  if (llvm::Error error =
          weft::transforms::checkExecutionPlanCoherence(module, plugins,
                                                              exporters)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error = exportFn(module, exporters, plugins, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportCoherenceGatedTargetArtifactRoute(
    mlir::ModuleOp module, llvm::raw_ostream &os, llvm::StringRef routeID) {
  weft::plugin::ExtensionPluginRegistry plugins;
  weft::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  if (llvm::Error error =
          weft::transforms::checkExecutionPlanCoherence(module, plugins,
                                                              exporters)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error = weft::target::exportTargetArtifactRoute(
          module, exporters, plugins, routeID, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult planAndExportTargetTranslateArtifactRoute(
    mlir::ModuleOp module, const weft::target::TargetTranslateRoute &route,
    llvm::raw_ostream &os) {
  weft::plugin::ExtensionPluginRegistry plugins;
  weft::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  mlir::PassManager pm(module.getContext());
  if (hasMaterializedSelectedLoweringBoundaries(module)) {
    pm.addPass(weft::transforms::createMaterializeEmissionPlansPass(
        plugins));
    pm.addPass(weft::transforms::createCheckExecutionPlanCoherencePass(
        plugins, exporters));
  } else if (hasMaterializedExecutionVariants(module)) {
    pm.addPass(
        weft::transforms::createMaterializeSelectedLoweringBoundariesPass(
            plugins));
    pm.addPass(weft::transforms::createMaterializeEmissionPlansPass(
        plugins));
    pm.addPass(weft::transforms::createCheckExecutionPlanCoherencePass(
        plugins, exporters));
  } else {
    weft::transforms::buildExecutionPlanningPipeline(pm, plugins,
                                                           exporters);
  }

  if (mlir::failed(pm.run(module))) {
    module.emitError()
        << "Weft-RV artifact-backed direct translate route '"
        << route.getRouteID()
        << "' failed during execution planning before exact target artifact "
           "export";
    return mlir::failure();
  }

  if (llvm::Error error = weft::target::exportTargetArtifactRoute(
          module, exporters, plugins, route.getTargetArtifactRouteID(), os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportTargetArtifact(mlir::ModuleOp module,
                                         llvm::raw_ostream &os) {
  return exportCoherenceGatedTargetArtifact(
      module, os, weft::target::exportTargetArtifact);
}

mlir::LogicalResult exportTargetHeaderArtifact(mlir::ModuleOp module,
                                               llvm::raw_ostream &os) {
  return exportCoherenceGatedTargetArtifact(
      module, os, weft::target::exportTargetHeaderArtifact);
}

mlir::LogicalResult exportTargetArtifactBundle(mlir::ModuleOp module,
                                               llvm::raw_ostream &os) {
  weft::plugin::ExtensionPluginRegistry plugins;
  weft::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  if (llvm::Error error =
          weft::transforms::checkExecutionPlanCoherence(module, plugins,
                                                              exporters)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error = weft::target::exportTargetArtifactBundle(
          module, exporters, plugins,
          targetArtifactBundleOutputDirectory)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  os << "weft.target_artifact_bundle_export: complete\n";
  os << "index_file: \"weft-target-artifact-bundle.index\"\n";
  return mlir::success();
}

mlir::LogicalResult
sourceArtifactBundleFrontDoor(mlir::ModuleOp module, llvm::raw_ostream &os) {
  weft::plugin::ExtensionPluginRegistry plugins;
  weft::target::TargetArtifactExporterRegistry exporters;
  if (mlir::failed(
          populateBuiltinPlanningRegistries(module, plugins, exporters)))
    return mlir::failure();

  constexpr llvm::StringLiteral routeDescription(
      "source-artifact bundle front door");
  if (mlir::failed(runSourceArtifactFrontDoorPipeline(
          module, plugins, exporters, routeDescription)))
    return mlir::failure();

  if (llvm::Error error = weft::target::exportTargetArtifactBundle(
          module, exporters, plugins,
          targetArtifactBundleOutputDirectory)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  os << "weft.target_artifact_bundle_export: complete\n";
  os << "index_file: \"weft-target-artifact-bundle.index\"\n";
  return mlir::success();
}

void registerWeftTranslations() {
  static mlir::TranslateFromMLIRRegistration emissionManifest(
      "weft-export-emission-manifest",
      "export Weft-RV selected emission handoff manifest",
      exportEmissionManifest, registerWeftTranslateDialects);
  (void)emissionManifest;

  registerBuiltinTargetTranslateRouteTranslations();

  static mlir::TranslateFromMLIRRegistration targetArtifact(
      "weft-export-target-artifact",
      "export one supported Weft-RV target artifact route",
      exportTargetArtifact, registerWeftTranslateDialects);
  (void)targetArtifact;

  static mlir::TranslateFromMLIRRegistration targetHeaderArtifact(
      "weft-export-target-header-artifact",
      "export one supported Weft-RV target C header artifact route",
      exportTargetHeaderArtifact, registerWeftTranslateDialects);
  (void)targetHeaderArtifact;

  static mlir::TranslateFromMLIRRegistration targetArtifactBundle(
      "weft-export-target-artifact-bundle",
      "export selected Weft-RV target artifacts into an output directory",
      exportTargetArtifactBundle, registerWeftTranslateDialects);
  (void)targetArtifactBundle;

  static mlir::TranslateFromMLIRRegistration
      sourceArtifactBundleFrontDoorRegistration(
      "weft-source-artifact-bundle-front-door",
      "run plugin-registered Weft-RV source front doors and export selected "
      "target artifacts into an output directory",
      sourceArtifactBundleFrontDoor, registerWeftTranslateDialects);
  (void)sourceArtifactBundleFrontDoorRegistration;
}

} // namespace

int main(int argc, char **argv) {
  registerWeftTranslations();
  return mlir::failed(mlir::mlirTranslateMain(
      argc, argv, "Weft-RV translation driver\n"));
}
