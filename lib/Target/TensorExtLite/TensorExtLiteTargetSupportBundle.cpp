#include "TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h"

#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::target::tensorext_lite {
namespace {

constexpr llvm::StringLiteral kTensorExtLiteHeaderHandoffKind(
    "materialized-emitc-cpp-tensorext-lite-fragment-header");
constexpr llvm::StringLiteral kTensorExtLiteRouteMetadataKey(
    "tensorext_lite_emitc_lowerable_route");
constexpr llvm::StringLiteral kTensorExtLiteRoleSequenceMetadataKey(
    "tensorext_lite_role_sequence");

llvm::Error makeTensorExtLiteTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TensorExtLite materialized EmitC target "
                  "artifact bridge failed: ") +
          message,
      llvm::errc::invalid_argument);
}

const plugin::tensorext_lite::TensorExtLiteConstructionManifest &
getTensorExtLiteManifest() {
  return plugin::tensorext_lite::getTensorExtLiteConstructionManifest();
}

const plugin::tensorext_lite::TensorExtLiteFragmentMmaEmitCConstructionRoute &
getTensorExtLiteRoute() {
  return plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
}

llvm::Error requireCandidateField(llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeTensorExtLiteTargetRouteError(
      llvm::Twine("candidate ") + fieldName + " must be '" + expected +
      "' but was '" + actual + "'");
}

llvm::StringRef lookupArtifactMetadataValue(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef key) {
  for (const support::ArtifactMetadataEntry &entry : metadata)
    if (entry.key == key)
      return entry.value;
  return {};
}

llvm::Error rejectForbiddenTensorExtLiteArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    std::string combined = (llvm::Twine(entry.key) + "=" + entry.value).str();
    std::string lowerStorage = llvm::StringRef(combined).lower();
    llvm::StringRef lower(lowerStorage);
    if (lower.contains("descriptor") ||
        lower.contains("metadata-diagnostic") ||
        lower.contains("source-export") ||
        lower.contains("source_export") || lower.contains("direct-c") ||
        lower.contains("direct_c") || lower.contains("compute-body") ||
        lower.contains("compute_body"))
      return makeTensorExtLiteTargetRouteError(
          llvm::Twine("candidate artifact metadata key '") + entry.key +
          "' attempts to reintroduce metadata-driven compute or direct source "
          "artifact authority");
  }
  return llvm::Error::success();
}

llvm::Error validateTensorExtLiteTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;

  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();
  if (llvm::Error error =
          requireCandidateField("route id", candidate.routeID, route.routeID))
    return error;
  if (llvm::Error error = requireCandidateField(
          "origin", candidate.origin, manifest.family.pluginName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "emission kind", candidate.emissionKind, route.emissionKind))
    return error;
  if (llvm::Error error = requireCandidateField(
          "artifact kind", candidate.artifactKind, route.artifactKind))
    return error;
  if (llvm::Error error = requireCandidateField(
          "lowering boundary", candidate.loweringBoundary,
          route.loweringBoundaryOpName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI", candidate.runtimeABI, route.runtimeABI))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI kind", candidate.runtimeABIKind, route.runtimeABIKind))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI name", candidate.runtimeABIName, route.runtimeABIName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime glue role", candidate.runtimeGlueRole,
          route.runtimeGlueRole))
    return error;

  llvm::StringRef routeMetadata = lookupArtifactMetadataValue(
      candidate.artifactMetadata, kTensorExtLiteRouteMetadataKey);
  if (routeMetadata.empty())
    return makeTensorExtLiteTargetRouteError(
        "candidate metadata must carry "
        "tensorext_lite_emitc_lowerable_route provenance");
  if (routeMetadata != route.routeID)
    return makeTensorExtLiteTargetRouteError(
        llvm::Twine("candidate TensorExtLite EmitC route metadata must be '") +
        route.routeID + "'");

  llvm::StringRef roleSequence = lookupArtifactMetadataValue(
      candidate.artifactMetadata, kTensorExtLiteRoleSequenceMetadataKey);
  if (roleSequence.empty())
    return makeTensorExtLiteTargetRouteError(
        "candidate metadata must carry TensorExtLite role-sequence "
        "provenance");
  if (roleSequence != manifest.semanticRoleGraph)
    return makeTensorExtLiteTargetRouteError(
        "candidate TensorExtLite role-sequence metadata must match the "
        "construction manifest");

  return rejectForbiddenTensorExtLiteArtifactMetadata(candidate);
}

SelectedEmitCArtifactRouteConfig getTensorExtLiteHeaderArtifactConfig() {
  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = route.routeID;
  config.artifactKind = route.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription =
      "TensorExtLite fragment MMA materialized EmitC header artifact bridge";
  config.candidateValidationFn = validateTensorExtLiteTargetArtifactCandidate;
  config.routeBuilderFn =
      plugin::tensorext_lite::buildTensorExtLiteFragmentMmaEmitCLowerableRoute;
  return config;
}

llvm::Expected<mlir::emitc::FuncOp>
getSingleMaterializedEmitCFunction(mlir::ModuleOp module,
                                   llvm::StringRef expectedFunctionName) {
  mlir::emitc::FuncOp selectedFunc;
  unsigned functionCount = 0;
  module->walk([&](mlir::emitc::FuncOp func) {
    ++functionCount;
    if (func.getSymName() == expectedFunctionName)
      selectedFunc = func;
  });

  if (!selectedFunc)
    return makeTensorExtLiteTargetRouteError(
        llvm::Twine("materialized EmitC header route requires EmitC function "
                    "boundary '") +
        expectedFunctionName + "'");
  if (functionCount != 1)
    return makeTensorExtLiteTargetRouteError(
        "materialized EmitC header route requires exactly one EmitC function "
        "boundary");
  return selectedFunc;
}

void printTensorExtLiteMaterializedEmitCHeaderDeclaration(
    llvm::raw_ostream &os, llvm::StringRef functionName,
    const TargetArtifactCandidate &candidate) {
  os << "#ifndef TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H\n";
  os << "#define TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H\n";
  os << "\n";
  os << "#include <stdint.h>\n";
  os << "\n";
  os << "/* tianchenrv.tensorext_lite.materialized_emitc_header.version: 1 "
        "*/\n";
  os << "/* tianchenrv.tensorext_lite.selected_route: "
     << candidate.routeID << " */\n";
  os << "/* tianchenrv.tensorext_lite.runtime_abi_name: "
     << candidate.runtimeABIName << " */\n";
  os << "/* tianchenrv.tensorext_lite.emitc_lowerable_route: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kTensorExtLiteRouteMetadataKey)
     << " */\n";
  os << "/* tianchenrv.tensorext_lite.role_sequence: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kTensorExtLiteRoleSequenceMetadataKey)
     << " */\n";
  os << "\n";
  os << "void " << functionName << "(void);\n";
  os << "\n";
  os << "#endif /* TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H */\n";
}

llvm::Error exportTensorExtLiteHeaderArtifact(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  SelectedEmitCArtifactRouteConfig config =
      getTensorExtLiteHeaderArtifactConfig();

  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config);
  if (!target)
    return target.takeError();

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> emitcModule =
      materializeSelectedEmitCArtifactModule(module, config);
  if (!emitcModule)
    return emitcModule.takeError();

  llvm::Expected<std::string> functionName =
      getSelectedEmitCArtifactFunctionName(module, config);
  if (!functionName)
    return functionName.takeError();

  llvm::Expected<mlir::emitc::FuncOp> func =
      getSingleMaterializedEmitCFunction(**emitcModule, *functionName);
  if (!func)
    return func.takeError();
  if ((*func).getFunctionType().getNumInputs() !=
      target->candidate.runtimeABIParameters.size())
    return makeTensorExtLiteTargetRouteError(
        "materialized EmitC header route function boundary arity must match "
        "the selected ordered runtime ABI parameter signature");

  printTensorExtLiteMaterializedEmitCHeaderDeclaration(
      os, *functionName, target->candidate);
  return llvm::Error::success();
}

llvm::Error registerTensorExtLiteHeaderTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error =
          plugin::tensorext_lite::verifyTensorExtLiteConstructionProtocolReady())
    return error;

  const auto &manifest = getTensorExtLiteManifest();
  const auto &route = getTensorExtLiteRoute();
  if (registry.lookup(route.routeID))
    return llvm::Error::success();

  return registry.registerExporter(TargetArtifactExporter(
      route.routeID, route.artifactKind, manifest.family.pluginName,
      route.emissionKind, exportTensorExtLiteHeaderArtifact,
      /*requiredRuntimeABIParameters=*/{}, kTensorExtLiteHeaderHandoffKind,
      validateTensorExtLiteTargetArtifactCandidate));
}

} // namespace

llvm::StringRef getTensorExtLiteMaterializedEmitCHeaderArtifactRouteID() {
  return getTensorExtLiteRoute().routeID;
}

llvm::Error registerTensorExtLiteTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getTensorExtLiteManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerTensorExtLiteHeaderTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerTensorExtLiteHeaderTargetArtifactExporter));
}

llvm::Error
configureTensorExtLiteTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerTensorExtLiteTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

} // namespace tianchenrv::target::tensorext_lite
