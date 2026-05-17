#include "TianChenRV/Target/Toy/ToyTargetSupportBundle.h"

#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::target::toy {
namespace {

constexpr llvm::StringLiteral kToyHeaderHandoffKind(
    "materialized-emitc-cpp-toy-template-header");
constexpr llvm::StringLiteral kToyRouteMetadataKey(
    "toy_emitc_lowerable_route");
constexpr llvm::StringLiteral kToySourceOpMetadataKey("toy_source_op");
constexpr llvm::StringLiteral kToySourceRoleMetadataKey("toy_source_role");
constexpr llvm::StringLiteral kToySourceOpInterfaceMetadataKey(
    "toy_source_op_interface");
constexpr llvm::StringLiteral kToyConstructionProtocolMetadataKey(
    "toy_construction_protocol");
constexpr llvm::StringLiteral kToySemanticRoleGraphMetadataKey(
    "toy_semantic_role_graph");
constexpr llvm::StringLiteral kToyTypedRoleRealizationMetadataKey(
    "toy_typed_role_realization");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

llvm::Error makeToyTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Toy materialized EmitC target artifact bridge "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

const plugin::toy::ToyConstructionManifest &getToyManifest() {
  return plugin::toy::getToyConstructionManifest();
}

const plugin::toy::ToyTemplateEmitCConstructionRoute &getToyRoute() {
  return plugin::toy::getToyTemplateEmitCConstructionRoute();
}

llvm::Error requireCandidateField(llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeToyTargetRouteError(llvm::Twine("candidate ") + fieldName +
                                 " must be '" + expected + "' but was '" +
                                 actual + "'");
}

llvm::StringRef lookupArtifactMetadataValue(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef key) {
  for (const support::ArtifactMetadataEntry &entry : metadata)
    if (entry.key == key)
      return entry.value;
  return {};
}

llvm::Error requireArtifactMetadata(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef expected, llvm::StringRef description) {
  llvm::StringRef value =
      lookupArtifactMetadataValue(candidate.artifactMetadata, key);
  if (value.empty())
    return makeToyTargetRouteError(llvm::Twine("candidate metadata must "
                                               "carry '") +
                                   key + "' " + description);
  if (value != expected)
    return makeToyTargetRouteError(llvm::Twine("candidate metadata '") + key +
                                   "' must be '" + expected + "'");
  return llvm::Error::success();
}

llvm::Error rejectForbiddenToyArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    std::string combined = (llvm::Twine(entry.key) + "=" + entry.value).str();
    std::string lowerStorage = llvm::StringRef(combined).lower();
    llvm::StringRef lower(lowerStorage);
    if (lower.contains("descriptor") ||
        lower.contains("metadata-diagnostic") ||
        lower.contains("source-export") || lower.contains("source_export") ||
        lower.contains("direct-c") || lower.contains("direct_c") ||
        lower.contains("compute-body") || lower.contains("compute_body"))
      return makeToyTargetRouteError(
          llvm::Twine("candidate artifact metadata key '") + entry.key +
          "' attempts to reintroduce descriptor-driven compute or direct "
          "source artifact authority");
  }
  return llvm::Error::success();
}

llvm::Error validateToyTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = plugin::toy::verifyToyConstructionProtocolReady())
    return error;

  const auto &manifest = getToyManifest();
  const auto &route = getToyRoute();
  if (llvm::Error error = requireCandidateField(
          "selected variant", candidate.selectedVariant,
          manifest.family.firstSliceVariantName))
    return error;
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

  if (!support::runtimeABIParametersEqual(
          candidate.runtimeABIParameters,
          plugin::toy::getToyTemplateRuntimeABIParameters()))
    return makeToyTargetRouteError(
        "candidate runtime ABI parameter signature must match the Toy "
        "construction route");

  if (llvm::Error error = requireArtifactMetadata(
          candidate, kToyRouteMetadataKey, route.routeID,
          "Toy EmitC lowerable route provenance"))
    return error;
  if (llvm::Error error = requireArtifactMetadata(
          candidate, kToySourceOpMetadataKey, route.loweringBoundaryOpName,
          "Toy source op provenance"))
    return error;
  if (llvm::Error error = requireArtifactMetadata(
          candidate, kToySourceRoleMetadataKey, "compute",
          "Toy source role provenance"))
    return error;
  if (llvm::Error error = requireArtifactMetadata(
          candidate, kToySourceOpInterfaceMetadataKey,
          kEmitCLowerableOpInterfaceName,
          "Toy source op-interface provenance"))
    return error;
  if (llvm::Error error = requireArtifactMetadata(
          candidate, kToyConstructionProtocolMetadataKey,
          manifest.protocolVersion, "Toy construction protocol provenance"))
    return error;
  if (llvm::Error error = requireArtifactMetadata(
          candidate, kToySemanticRoleGraphMetadataKey,
          manifest.semanticRoleGraph, "Toy semantic role graph provenance"))
    return error;
  if (llvm::Error error = requireArtifactMetadata(
          candidate, kToyTypedRoleRealizationMetadataKey,
          plugin::toy::getToyTypedRoleRealizationSummary(),
          "Toy typed role realization provenance"))
    return error;

  return rejectForbiddenToyArtifactMetadata(candidate);
}

SelectedEmitCArtifactRouteConfig getToyHeaderArtifactConfig() {
  const auto &manifest = getToyManifest();
  const auto &route = getToyRoute();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = route.routeID;
  config.artifactKind = route.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription =
      "Toy template materialized EmitC header artifact bridge";
  config.candidateValidationFn = validateToyTargetArtifactCandidate;
  config.routeBuilderFn = plugin::toy::buildToyTemplateEmitCLowerableRoute;
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
    return makeToyTargetRouteError(
        llvm::Twine("materialized EmitC header route requires EmitC function "
                    "boundary '") +
        expectedFunctionName + "'");
  if (functionCount != 1)
    return makeToyTargetRouteError(
        "materialized EmitC header route requires exactly one EmitC function "
        "boundary");
  return selectedFunc;
}

void printRuntimeABIParameterEvidence(
    llvm::raw_ostream &os, const support::RuntimeABIParameter &parameter,
    unsigned index) {
  os << "/* tianchenrv.toy.runtime_abi_parameter[" << index << "]: ";
  support::printRuntimeABIParameterCDeclaration(os, parameter);
  os << " role="
     << support::stringifyRuntimeABIParameterRole(parameter.role)
     << " ownership="
     << support::stringifyRuntimeABIParameterOwnership(parameter.ownership)
     << " */\n";
}

void printToyMaterializedEmitCHeaderDeclaration(
    llvm::raw_ostream &os, llvm::StringRef functionName,
    const TargetArtifactCandidate &candidate) {
  os << "#ifndef TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H\n";
  os << "#define TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H\n";
  os << "\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "\n";
  os << "/* tianchenrv.toy.materialized_emitc_header.version: 1 */\n";
  os << "/* tianchenrv.toy.origin_plugin: " << candidate.origin << " */\n";
  os << "/* tianchenrv.toy.selected_variant: @"
     << candidate.selectedVariant << " */\n";
  os << "/* tianchenrv.toy.selected_route: " << candidate.routeID << " */\n";
  os << "/* tianchenrv.toy.runtime_abi_kind: "
     << candidate.runtimeABIKind << " */\n";
  os << "/* tianchenrv.toy.runtime_abi_name: "
     << candidate.runtimeABIName << " */\n";
  for (auto [index, parameter] :
       llvm::enumerate(candidate.runtimeABIParameters))
    printRuntimeABIParameterEvidence(os, parameter, index);
  os << "/* tianchenrv.toy.emitc_lowerable_route: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kToyRouteMetadataKey)
     << " */\n";
  os << "/* tianchenrv.toy.source_op: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kToySourceOpMetadataKey)
     << " */\n";
  os << "/* tianchenrv.toy.source_role: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kToySourceRoleMetadataKey)
     << " */\n";
  os << "/* tianchenrv.toy.source_op_interface: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kToySourceOpInterfaceMetadataKey)
     << " */\n";
  os << "/* tianchenrv.toy.construction_protocol: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kToyConstructionProtocolMetadataKey)
     << " */\n";
  os << "/* tianchenrv.toy.semantic_role_graph: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kToySemanticRoleGraphMetadataKey)
     << " */\n";
  os << "/* tianchenrv.toy.typed_role_realization: "
     << lookupArtifactMetadataValue(candidate.artifactMetadata,
                                    kToyTypedRoleRealizationMetadataKey)
     << " */\n";
  os << "\n";
  os << "void " << functionName << "(";
  llvm::ArrayRef<support::RuntimeABIParameter> parameters =
      candidate.runtimeABIParameters;
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ");\n";
  os << "\n";
  os << "#endif /* TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H */\n";
}

llvm::Error exportToyHeaderArtifact(mlir::ModuleOp module,
                                    llvm::raw_ostream &os) {
  SelectedEmitCArtifactRouteConfig config = getToyHeaderArtifactConfig();

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
    return makeToyTargetRouteError(
        "materialized EmitC header route function boundary arity must match "
        "the selected ordered runtime ABI parameter signature");

  printToyMaterializedEmitCHeaderDeclaration(os, *functionName,
                                             target->candidate);
  return llvm::Error::success();
}

llvm::Error registerToyHeaderTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = plugin::toy::verifyToyConstructionProtocolReady())
    return error;

  const auto &manifest = getToyManifest();
  const auto &route = getToyRoute();
  if (registry.lookup(route.routeID))
    return llvm::Error::success();

  return registry.registerExporter(TargetArtifactExporter(
      route.routeID, route.artifactKind, manifest.family.pluginName,
      route.emissionKind, exportToyHeaderArtifact,
      plugin::toy::getToyTemplateRuntimeABIParameters(), kToyHeaderHandoffKind,
      validateToyTargetArtifactCandidate));
}

} // namespace

llvm::StringRef getToyMaterializedEmitCHeaderArtifactRouteID() {
  return getToyRoute().routeID;
}

llvm::Error registerToyTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getToyManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() == registerToyHeaderTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerToyHeaderTargetArtifactExporter));
}

llvm::Error
configureToyTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(getToyRoute().loweringBoundaryOpName);
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerToyTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

} // namespace tianchenrv::target::toy
