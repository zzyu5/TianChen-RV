#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/EmissionManifest.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVSmokeProbe.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Transforms/ExecutionPlanCoherence.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace {

using TargetArtifactExportFn = llvm::Error (*)(
    mlir::ModuleOp, const tianchenrv::target::TargetArtifactExporterRegistry &,
    llvm::raw_ostream &);

llvm::cl::opt<std::string> targetArtifactBundleOutputDirectory(
    "tcrv-target-artifact-bundle-output-dir",
    llvm::cl::desc("output directory for target artifact bundle export"),
    llvm::cl::value_desc("directory"), llvm::cl::init(""));

mlir::LogicalResult populateBuiltinPlanningRegistries(
    mlir::ModuleOp module, tianchenrv::plugin::ExtensionPluginRegistry &plugins,
    tianchenrv::target::TargetArtifactExporterRegistry &exporters) {
  if (llvm::Error error =
          tianchenrv::plugin::registerBuiltinExtensionPlugins(plugins)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  if (llvm::Error error =
          tianchenrv::target::registerBuiltinTargetArtifactExporters(
              exporters)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }

  return mlir::success();
}

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
  tianchenrv::target::TargetArtifactExporterRegistry exporters;
  if (llvm::Error error =
          tianchenrv::target::registerBuiltinTargetArtifactExporters(
              exporters)) {
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

mlir::LogicalResult exportRVVMicrokernelC(mlir::ModuleOp module,
                                          llvm::raw_ostream &os) {
  if (llvm::Error error =
          tianchenrv::target::rvv::exportRVVMicrokernelC(module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportRVVMicrokernelSelfCheckC(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  if (llvm::Error error =
          tianchenrv::target::rvv::exportRVVMicrokernelSelfCheckC(module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportRVVMicrokernelHeader(mlir::ModuleOp module,
                                               llvm::raw_ostream &os) {
  if (llvm::Error error =
          tianchenrv::target::rvv::exportRVVMicrokernelHeader(module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportRVVMicrokernelObject(mlir::ModuleOp module,
                                               llvm::raw_ostream &os) {
  if (std::error_code error = llvm::sys::ChangeStdoutToBinary()) {
    module.emitError()
        << "failed to switch stdout to binary mode for object export: "
        << error.message();
    return mlir::failure();
  }

  if (llvm::Error error =
          tianchenrv::target::rvv::exportRVVMicrokernelObject(module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult exportRVVScalarI32VAddDispatchC(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os) {
  if (llvm::Error error =
          tianchenrv::target::rvv_scalar::exportRVVScalarI32VAddDispatchC(
              module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
exportRVVScalarI32VAddDispatchHeader(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  if (llvm::Error error = tianchenrv::target::rvv_scalar::
                              exportRVVScalarI32VAddDispatchHeader(module,
                                                                   os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
exportRVVScalarI32VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                         llvm::raw_ostream &os) {
  if (llvm::Error error = tianchenrv::target::rvv_scalar::
                              exportRVVScalarI32VAddDispatchSelfCheckC(module,
                                                                       os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
exportRVVScalarI32VAddDispatchObject(mlir::ModuleOp module,
                                     llvm::raw_ostream &os) {
  if (std::error_code error = llvm::sys::ChangeStdoutToBinary()) {
    module.emitError()
        << "failed to switch stdout to binary mode for object export: "
        << error.message();
    return mlir::failure();
  }

  if (llvm::Error error =
          tianchenrv::target::rvv_scalar::exportRVVScalarI32VAddDispatchObject(
              module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
exportRVVScalarI32VAddDispatchSelfCheckObject(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  if (std::error_code error = llvm::sys::ChangeStdoutToBinary()) {
    module.emitError()
        << "failed to switch stdout to binary mode for object export: "
        << error.message();
    return mlir::failure();
  }

  if (llvm::Error error = tianchenrv::target::rvv_scalar::
                              exportRVVScalarI32VAddDispatchSelfCheckObject(
                                  module, os)) {
    std::string message = llvm::toString(std::move(error));
    module.emitError() << message;
    return mlir::failure();
  }
  return mlir::success();
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

mlir::LogicalResult exportTargetSourceArtifact(mlir::ModuleOp module,
                                               llvm::raw_ostream &os) {
  return exportCoherenceGatedTargetArtifact(
      module, os, tianchenrv::target::exportTargetSourceArtifact);
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

mlir::LogicalResult lowerBoundedLinalgFrontendForPlanning(
    mlir::ModuleOp module, llvm::StringRef frontDoorName) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(tianchenrv::transforms::createLowerLinalgI32VAddToExecPass());
  if (mlir::succeeded(pm.run(module)))
    return mlir::success();

  module.emitError() << "TianChen-RV " << frontDoorName
                     << " failed during bounded linalg i32 add/sub frontend "
                        "lowering";
  return mlir::failure();
}

mlir::LogicalResult planAndExportTargetArtifactBundle(mlir::ModuleOp module,
                                                      llvm::raw_ostream &os) {
  if (mlir::failed(lowerBoundedLinalgFrontendForPlanning(
          module, "plan-and-export target artifact bundle")))
    return mlir::failure();

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

  static mlir::TranslateFromMLIRRegistration rvvSmokeProbeC(
      "tcrv-export-rvv-smoke-probe-c",
      "export a bounded RVV hardware/toolchain smoke-probe C source",
      exportRVVSmokeProbeC, registerTianChenRVTranslateDialects);
  (void)rvvSmokeProbeC;

  static mlir::TranslateFromMLIRRegistration rvvMicrokernelC(
      "tcrv-export-rvv-microkernel-c",
      "export one runtime-callable RVV i32 add/sub microkernel C source",
      exportRVVMicrokernelC, registerTianChenRVTranslateDialects);
  (void)rvvMicrokernelC;

  static mlir::TranslateFromMLIRRegistration rvvMicrokernelSelfCheckC(
      "tcrv-export-rvv-microkernel-self-check-c",
      "export one RVV i32 add/sub microkernel C source with self-check "
      "harness",
      exportRVVMicrokernelSelfCheckC, registerTianChenRVTranslateDialects);
  (void)rvvMicrokernelSelfCheckC;

  static mlir::TranslateFromMLIRRegistration rvvMicrokernelHeader(
      "tcrv-export-rvv-microkernel-header",
      "export one RVV i32 add/sub microkernel runtime-callable C ABI header",
      exportRVVMicrokernelHeader, registerTianChenRVTranslateDialects);
  (void)rvvMicrokernelHeader;

  static mlir::TranslateFromMLIRRegistration rvvMicrokernelObject(
      "tcrv-export-rvv-microkernel-object",
      "export one RVV i32 add/sub microkernel library object file",
      exportRVVMicrokernelObject, registerTianChenRVTranslateDialects);
  (void)rvvMicrokernelObject;

  static mlir::TranslateFromMLIRRegistration rvvScalarDispatchC(
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-c",
      "export one host RVV+scalar i32 vector-add dispatch C source",
      exportRVVScalarI32VAddDispatchC, registerTianChenRVTranslateDialects);
  (void)rvvScalarDispatchC;

  static mlir::TranslateFromMLIRRegistration rvvScalarDispatchHeader(
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-header",
      "export one host RVV+scalar i32 vector-add dispatch C ABI header",
      exportRVVScalarI32VAddDispatchHeader,
      registerTianChenRVTranslateDialects);
  (void)rvvScalarDispatchHeader;

  static mlir::TranslateFromMLIRRegistration rvvScalarDispatchSelfCheckC(
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c",
      "export one host RVV+scalar i32 vector-add dispatch C source with "
      "self-check harness",
      exportRVVScalarI32VAddDispatchSelfCheckC,
      registerTianChenRVTranslateDialects);
  (void)rvvScalarDispatchSelfCheckC;

  static mlir::TranslateFromMLIRRegistration rvvScalarDispatchObject(
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-object",
      "export one host RVV+scalar i32 vector-add dispatch library object file",
      exportRVVScalarI32VAddDispatchObject,
      registerTianChenRVTranslateDialects);
  (void)rvvScalarDispatchObject;

  static mlir::TranslateFromMLIRRegistration rvvScalarDispatchSelfCheckObject(
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object",
      "export one host RVV+scalar i32 vector-add dispatch self-check object "
      "file",
      exportRVVScalarI32VAddDispatchSelfCheckObject,
      registerTianChenRVTranslateDialects);
  (void)rvvScalarDispatchSelfCheckObject;

  static mlir::TranslateFromMLIRRegistration targetSourceArtifact(
      "tcrv-export-target-source-artifact",
      "export one supported TianChen-RV target source artifact route",
      exportTargetSourceArtifact, registerTianChenRVTranslateDialects);
  (void)targetSourceArtifact;

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
}

} // namespace

int main(int argc, char **argv) {
  registerTianChenRVTranslations();
  return mlir::failed(mlir::mlirTranslateMain(
      argc, argv, "TianChen-RV translation driver\n"));
}
