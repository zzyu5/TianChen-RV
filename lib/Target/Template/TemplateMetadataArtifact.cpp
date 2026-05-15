#include "TianChenRV/Target/Template/TemplateMetadataArtifact.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Template/IR/TemplateDialect.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv::target::template_ext {
namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;
namespace construction = tianchenrv::plugin::construction;
namespace pluginTemplate = tianchenrv::plugin::template_ext;

using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::target::SelectedPlanMetadataEntry;
using tianchenrv::target::TargetArtifactCandidate;
using tianchenrv::target::TargetArtifactExporter;
using tianchenrv::tcrv::template_ext::ComputeSkeletonOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::template_ext::LoweringBoundaryOp;

constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kIntegrationContractAttrName("integration_contract");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kEmitCCallAttrName("emitc_call");
constexpr llvm::StringLiteral kMetadataArtifactVersion("1");
constexpr llvm::StringLiteral kArtifactStatus(
    "compiler-construction-template-artifact");
constexpr llvm::StringLiteral kNoRuntimeClaim("none");

struct TemplateCapabilityRecord {
  std::string symbolName;
};

llvm::Error makeTemplateArtifactError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV Template metadata artifact export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleTemplateArtifactError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Template metadata artifact export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::Error requireCandidateField(KernelOp kernel, llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual != expected)
    return makeTemplateArtifactError(
        kernel, llvm::Twine("Template artifact candidate ") + fieldName + " '" +
                    actual + "' does not match expected '" + expected + "'");
  return llvm::Error::success();
}

llvm::Error requireBoundaryStringAttr(LoweringBoundaryOp boundary,
                                      llvm::StringRef attrName,
                                      llvm::StringRef expectedValue) {
  auto attr = getStringAttr(boundary.getOperation(), attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeTemplateArtifactError(
        boundary->getParentOfType<KernelOp>(),
        llvm::Twine("Template lowering boundary requires non-empty attribute '") +
            attrName + "'");
  if (attr.getValue() != expectedValue)
    return makeTemplateArtifactError(
        boundary->getParentOfType<KernelOp>(),
        llvm::Twine("Template lowering boundary attribute '") + attrName + "' is '" +
            attr.getValue() + "' but expected '" + expectedValue + "'");
  return llvm::Error::success();
}

VariantOp findDirectVariant(KernelOp kernel, llvm::StringRef symbol) {
  if (!kernel || kernel.getBody().empty())
    return VariantOp();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<VariantOp>(op);
    if (variant && variant.getSymName() == symbol)
      return variant;
  }
  return VariantOp();
}

llvm::Expected<TemplateCapabilityRecord> getRequiredTemplateCapability(KernelOp kernel) {
  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();

  const CapabilityDescriptor *capability =
      capabilities->lookupProviderByID(pluginTemplate::getTemplateExtensionCapabilityID());
  if (!capability)
    return makeTemplateArtifactError(
        kernel, llvm::Twine("requires capability provider for id '") +
                    pluginTemplate::getTemplateExtensionCapabilityID() + "'");
  if (!capability->isAvailable())
    return makeTemplateArtifactError(
        kernel, llvm::Twine("requires available capability provider for id '") +
                    pluginTemplate::getTemplateExtensionCapabilityID() + "'");
  if (capability->getKind() != pluginTemplate::getTemplateExtensionCapabilityKind())
    return makeTemplateArtifactError(
        kernel, llvm::Twine("capability id '") +
                    pluginTemplate::getTemplateExtensionCapabilityID() +
                    "' kind must be '" +
                    pluginTemplate::getTemplateExtensionCapabilityKind() + "'");
  if (capability->getProperty("integration_contract") !=
      pluginTemplate::getTemplateExpectedIntegrationContract())
    return makeTemplateArtifactError(
        kernel,
        "Template capability integration_contract does not match the selected "
        "metadata artifact contract");
  if (capability->getProperty("handoff_kind") !=
      pluginTemplate::getTemplateExpectedHandoffKind())
    return makeTemplateArtifactError(
        kernel, "Template capability handoff_kind does not match the selected "
                "metadata artifact handoff");

  TemplateCapabilityRecord record;
  record.symbolName = capability->getSymbolName().str();
  return record;
}

llvm::Expected<std::string> validateVariantRequiresTemplateCapability(
    KernelOp kernel, VariantOp variant,
    const TemplateCapabilityRecord &templateCapability) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requires || requires.empty())
    return makeTemplateArtifactError(
        kernel, llvm::Twine("selected Template variant @") + variant.getSymName() +
                    " requires non-empty 'requires' capability metadata");

  std::string requiredTemplateSymbol;
  for (mlir::Attribute attr : requires) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeTemplateArtifactError(
          kernel, llvm::Twine("selected Template variant @") +
                      variant.getSymName() +
                      " requires only non-empty capability symbol references");

    if (symbol.getValue() == templateCapability.symbolName) {
      requiredTemplateSymbol = symbol.getValue().str();
      continue;
    }

    return makeTemplateArtifactError(
        kernel, llvm::Twine("selected Template variant @") + variant.getSymName() +
                    " has non-Template required capability @" + symbol.getValue());
  }

  if (requiredTemplateSymbol.empty())
    return makeTemplateArtifactError(
        kernel, llvm::Twine("selected Template variant @") + variant.getSymName() +
                    " must require the Template extension capability");
  return requiredTemplateSymbol;
}

llvm::Expected<LoweringBoundaryOp>
findAndValidateTemplateBoundary(const TargetArtifactCandidate &candidate,
                           VariantOp variant) {
  KernelOp kernel = candidate.kernel;
  if (!kernel || kernel.getBody().empty())
    return makeTemplateArtifactError(
        kernel, "requires selected Template kernel with a materialized body");

  LoweringBoundaryOp matched;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != candidate.loweringBoundary)
      continue;

    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      return makeTemplateArtifactError(
          kernel, llvm::Twine("lowering boundary operation '") +
                      candidate.loweringBoundary +
                      "' is not a tcrv_template.lowering_boundary");

    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = getStringAttr(boundary.getOperation(), kRoleAttrName);
    if (!selectedVariant || selectedVariant.getValue() != variant.getSymName() ||
        !role || role.getValue() != candidate.role)
      continue;

    if (matched)
      return makeTemplateArtifactError(
          kernel, llvm::Twine("duplicate Template lowering boundaries for @") +
                      candidate.selectedVariant + " as " + candidate.role);
    matched = boundary;
  }

  if (!matched)
    return makeTemplateArtifactError(
        kernel, llvm::Twine("selected Template candidate requires one ") +
                    candidate.loweringBoundary + " for @" +
                    candidate.selectedVariant + " as " + candidate.role);

  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kSourceKernelAttrName,
                                    kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kOriginAttrName,
                                    pluginTemplate::getTemplateExtensionPluginName()))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kRoleAttrName, candidate.role))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kStatusAttrName,
                                    kMetadataOnlyStatusValue))
    return std::move(error);
  if (llvm::Error error = requireBoundaryStringAttr(
          matched, kIntegrationContractAttrName, pluginTemplate::getTemplateExpectedIntegrationContract()))
    return std::move(error);
  if (llvm::Error error = requireBoundaryStringAttr(
          matched, kHandoffKindAttrName, pluginTemplate::getTemplateExpectedHandoffKind()))
    return std::move(error);

  auto boundaryCapabilities =
      matched->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!boundaryCapabilities || boundaryCapabilities != variantRequires)
    return makeTemplateArtifactError(
        kernel, "Template lowering boundary required_capabilities must match "
                "selected variant requires metadata");

  return matched;
}

llvm::Error requireComputeRoleStringAttr(ComputeSkeletonOp computeRole,
                                         llvm::StringRef attrName,
                                         llvm::StringRef expectedValue) {
  auto attr = getStringAttr(computeRole.getOperation(), attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeTemplateArtifactError(
        computeRole->getParentOfType<KernelOp>(),
        llvm::Twine("Template compute role op requires non-empty attribute '") +
            attrName + "'");
  if (attr.getValue() != expectedValue)
    return makeTemplateArtifactError(
        computeRole->getParentOfType<KernelOp>(),
        llvm::Twine("Template compute role op attribute '") + attrName +
            "' is '" + attr.getValue() + "' but expected '" + expectedValue +
            "'");
  return llvm::Error::success();
}

llvm::Expected<ComputeSkeletonOp>
findAndValidateTemplateComputeRoleOp(const TargetArtifactCandidate &candidate,
                                     VariantOp variant) {
  KernelOp kernel = candidate.kernel;
  if (!kernel || kernel.getBody().empty())
    return makeTemplateArtifactError(
        kernel, "requires selected Template kernel with a materialized body");

  ComputeSkeletonOp matched;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != ComputeSkeletonOp::getOperationName())
      continue;

    auto computeRole = llvm::dyn_cast<ComputeSkeletonOp>(op);
    if (!computeRole)
      return makeTemplateArtifactError(
          kernel, llvm::Twine("Template role operation '") +
                      ComputeSkeletonOp::getOperationName() +
                      "' does not materialize as ComputeSkeletonOp");

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = getStringAttr(computeRole.getOperation(), kRoleAttrName);
    if (!selectedVariant || selectedVariant.getValue() != variant.getSymName() ||
        !role || role.getValue() != candidate.role)
      continue;

    if (matched)
      return makeTemplateArtifactError(
          kernel, llvm::Twine("duplicate Template compute role ops for @") +
                      candidate.selectedVariant + " as " + candidate.role);
    matched = computeRole;
  }

  if (!matched)
    return makeTemplateArtifactError(
        kernel, llvm::Twine("selected Template candidate requires one ") +
                    ComputeSkeletonOp::getOperationName() + " for @" +
                    candidate.selectedVariant + " as " + candidate.role);

  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kSourceKernelAttrName,
                                       kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(
              matched, kOriginAttrName,
              pluginTemplate::getTemplateExtensionPluginName()))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kRoleAttrName, candidate.role))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kStatusAttrName,
                                       kRoleOpBoundaryStatusValue))
    return std::move(error);

  auto roleOrder =
      matched->getAttrOfType<mlir::IntegerAttr>("role_order");
  if (!roleOrder || roleOrder.getInt() != 2)
    return makeTemplateArtifactError(
        kernel, "Template compute role op role_order must be 2");

  const pluginTemplate::TemplateTypedRoleGraphRealization &realization =
      pluginTemplate::getTemplateTypedRoleGraphRealization();
  const pluginTemplate::TemplateTypedRoleInterfaceRealization &compute =
      realization.roles[2];
  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kTypedRoleAttrName,
                                       compute.typedRoleID))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kSourceRoleAttrName,
                                       compute.role))
    return std::move(error);
  if (llvm::Error error = requireComputeRoleStringAttr(
          matched, kRoleSpecificInterfaceAttrName,
          compute.roleSpecificInterface))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kEmitCCallAttrName,
                                       compute.emitCCall))
    return std::move(error);

  auto requiredCapabilities =
      matched->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiredCapabilities || requiredCapabilities != variantRequires)
    return makeTemplateArtifactError(
        kernel, "Template compute role op required_capabilities must match "
                "selected variant requires metadata");

  if (llvm::Error error = pluginTemplate::verifyTemplateComputeRoleOpInterface(
          pluginTemplate::getTemplateConstructionManifest(), realization,
          matched.getOperation()))
    return std::move(error);

  return matched;
}

const SelectedPlanMetadataEntry *
findSelectedPlanMetadata(llvm::ArrayRef<SelectedPlanMetadataEntry> metadata,
                         llvm::StringRef name) {
  for (const SelectedPlanMetadataEntry &entry : metadata)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

llvm::Error requireSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate, llvm::StringRef name,
    llvm::StringRef expectedValue, llvm::StringRef expectedRole) {
  const SelectedPlanMetadataEntry *entry =
      findSelectedPlanMetadata(candidate.selectedPlanMetadata, name);
  if (!entry)
    return makeTemplateArtifactError(
        candidate.kernel,
        llvm::Twine("Template artifact candidate requires selected_plan_metadata '") +
            name + "'");
  if (entry->value != expectedValue || entry->role != expectedRole ||
      entry->note.empty())
    return makeTemplateArtifactError(
        candidate.kernel,
        llvm::Twine("Template artifact candidate selected_plan_metadata '") + name +
            "' does not match expected value/role");
  return llvm::Error::success();
}

llvm::Error validateTemplateSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate) {
  if (candidate.selectedPlanMetadata.size() != 10)
    return makeTemplateArtifactError(
        candidate.kernel,
        "Template artifact candidate requires exactly ten selected plan metadata "
        "entries");

  const pluginTemplate::TemplateConstructionManifest &manifest =
      pluginTemplate::getTemplateConstructionManifest();
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "template_extension_capability_id",
          pluginTemplate::getTemplateExtensionCapabilityID(), "capability-requirement"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "template_extension_integration_contract",
          pluginTemplate::getTemplateExpectedIntegrationContract(),
          "integration-contract"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "template_extension_scope", "zero-core-integration",
          "evidence-scope"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate,
          pluginTemplate::getTemplateConstructionProtocolMetadataName(),
          manifest.protocolVersion,
          pluginTemplate::getTemplateConstructionProtocolMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate,
          pluginTemplate::getTemplateConstructionArchetypeMetadataName(),
          manifest.archetype,
          pluginTemplate::getTemplateConstructionArchetypeMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTemplate::getTemplateSemanticRoleGraphMetadataName(),
          manifest.semanticRoleGraph,
          pluginTemplate::getTemplateSemanticRoleGraphMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate,
          pluginTemplate::getTemplateCommonInterfaceRealizationMetadataName(),
          pluginTemplate::getTemplateConstructionInterfaceRealization(),
          pluginTemplate::getTemplateCommonInterfaceRealizationMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate,
          pluginTemplate::getTemplateTypedRoleRealizationMetadataName(),
          pluginTemplate::getTemplateTypedRoleRealizationSummary(),
          pluginTemplate::getTemplateTypedRoleRealizationMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTemplate::getTemplateEmitCRouteMappingMetadataName(),
          manifest.emitcRoute.routeID,
          pluginTemplate::getTemplateEmitCRouteMappingMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTemplate::getTemplateEvidenceProfileMetadataName(),
          manifest.evidenceProfile,
          pluginTemplate::getTemplateEvidenceProfileMetadataRole()))
    return error;
  return llvm::Error::success();
}

llvm::Error validateTemplateMetadataCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = pluginTemplate::verifyTemplateConstructionManifest(
          pluginTemplate::getTemplateConstructionManifest()))
    return error;
  if (llvm::Error error = pluginTemplate::verifyTemplateTypedRoleGraphRealization(
          pluginTemplate::getTemplateConstructionManifest(),
          pluginTemplate::getTemplateTypedRoleGraphRealization()))
    return error;
  if (llvm::Expected<pluginTemplate::TemplateGeneratedOutputRoute> route =
          pluginTemplate::buildTemplateGeneratedOutputRoute(
              pluginTemplate::getTemplateConstructionManifest(),
              pluginTemplate::getTemplateTypedRoleGraphRealization());
      !route)
    return route.takeError();

  KernelOp kernel = candidate.kernel;
  if (!kernel)
    return makeTemplateArtifactError(
        kernel, "requires a selected Template kernel from target artifact "
                "candidate collection");

  if (llvm::Error error = requireCandidateField(
          kernel, "origin", candidate.origin,
          pluginTemplate::getTemplateExtensionPluginName()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "route id", candidate.routeID,
          pluginTemplate::getTemplateMetadataRouteID()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "emission kind", candidate.emissionKind,
          pluginTemplate::getTemplateMetadataEmissionKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "artifact kind", candidate.artifactKind,
          pluginTemplate::getTemplateMetadataArtifactKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI", candidate.runtimeABI,
          pluginTemplate::getTemplateExpectedIntegrationContract()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI kind", candidate.runtimeABIKind,
          pluginTemplate::getTemplateMetadataRuntimeABIKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI name", candidate.runtimeABIName,
          pluginTemplate::getTemplateExpectedIntegrationContract()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime glue role", candidate.runtimeGlueRole,
          pluginTemplate::getTemplateMetadataRuntimeGlueRole()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "lowering boundary", candidate.loweringBoundary,
          LoweringBoundaryOp::getOperationName()))
    return error;
  if (!candidate.runtimeABIParameters.empty())
    return makeTemplateArtifactError(
        kernel, "Template metadata artifact route must not carry executable "
                "runtime ABI parameters");

  if (candidate.selectedVariant !=
      pluginTemplate::getTemplateExtensionFirstSliceVariantName())
    return makeTemplateArtifactError(
        kernel, llvm::Twine("Template artifact candidate selected variant @") +
                    candidate.selectedVariant + " does not match @" +
                    pluginTemplate::getTemplateExtensionFirstSliceVariantName());

  VariantOp variant = findDirectVariant(kernel, candidate.selectedVariant);
  if (!variant)
    return makeTemplateArtifactError(
        kernel, llvm::Twine("selected Template variant @") +
                    candidate.selectedVariant +
                    " must resolve to a direct tcrv.exec.variant");
  auto origin = getStringAttr(variant.getOperation(), kOriginAttrName);
  if (!origin || origin.getValue() != pluginTemplate::getTemplateExtensionPluginName())
    return makeTemplateArtifactError(
        kernel, llvm::Twine("selected Template variant @") +
                    candidate.selectedVariant +
                    " must be owned by origin 'template-plugin'");

  llvm::Expected<TemplateCapabilityRecord> capability =
      getRequiredTemplateCapability(kernel);
  if (!capability)
    return capability.takeError();
  llvm::Expected<std::string> requiredTemplateSymbol =
      validateVariantRequiresTemplateCapability(kernel, variant, *capability);
  if (!requiredTemplateSymbol)
    return requiredTemplateSymbol.takeError();

  if (llvm::Error error = validateTemplateSelectedPlanMetadata(candidate))
    return error;

  llvm::Expected<LoweringBoundaryOp> boundary =
      findAndValidateTemplateBoundary(candidate, variant);
  if (!boundary)
    return boundary.takeError();

  llvm::Expected<ComputeSkeletonOp> computeRole =
      findAndValidateTemplateComputeRoleOp(candidate, variant);
  if (!computeRole)
    return computeRole.takeError();

  return llvm::Error::success();
}

llvm::Expected<TargetArtifactCandidate>
selectTemplateMetadataCandidate(mlir::ModuleOp module) {
  llvm::SmallVector<TargetArtifactCandidate, 4> candidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, candidates))
    return std::move(error);

  llvm::SmallVector<TargetArtifactCandidate, 2> templateCandidates;
  for (const TargetArtifactCandidate &candidate : candidates) {
    bool templateShaped =
        candidate.origin == pluginTemplate::getTemplateExtensionPluginName() ||
        candidate.routeID == pluginTemplate::getTemplateMetadataRouteID() ||
        candidate.emissionKind == pluginTemplate::getTemplateMetadataEmissionKind() ||
        candidate.artifactKind == pluginTemplate::getTemplateMetadataArtifactKind();
    if (!templateShaped)
      continue;

    if (llvm::Error error = validateTemplateMetadataCandidate(candidate))
      return std::move(error);
    templateCandidates.push_back(candidate);
  }

  if (templateCandidates.empty())
    return makeModuleTemplateArtifactError(
        "requires exactly one selected Template metadata artifact candidate; found "
        "none");
  if (templateCandidates.size() > 1)
    return makeModuleTemplateArtifactError(
        "requires exactly one selected Template metadata artifact candidate; found "
        "multiple");
  return templateCandidates.front();
}

llvm::Expected<std::string>
getRequiredTemplateCapabilitySymbol(const TargetArtifactCandidate &candidate) {
  llvm::Expected<TemplateCapabilityRecord> capability =
      getRequiredTemplateCapability(candidate.kernel);
  if (!capability)
    return capability.takeError();
  VariantOp variant = findDirectVariant(candidate.kernel,
                                        candidate.selectedVariant);
  return validateVariantRequiresTemplateCapability(candidate.kernel, variant,
                                             *capability);
}

void printQuoted(llvm::raw_ostream &os, llvm::StringRef value) {
  os << "\"";
  for (char character : value) {
    switch (character) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << character;
      break;
    }
  }
  os << "\"";
}

void printField(llvm::raw_ostream &os, llvm::StringRef name,
                llvm::StringRef value) {
  os << name << ": ";
  printQuoted(os, value);
  os << "\n";
}

void printSelectedPlanMetadata(
    llvm::raw_ostream &os,
    llvm::ArrayRef<SelectedPlanMetadataEntry> metadata) {
  for (auto [index, entry] : llvm::enumerate(metadata)) {
    os << "selected_plan_metadata[" << index << "]:\n";
    os << "  name: ";
    printQuoted(os, entry.name);
    os << "\n";
    os << "  value: ";
    printQuoted(os, entry.value);
    os << "\n";
    os << "  role: ";
    printQuoted(os, entry.role);
    os << "\n";
    os << "  note: ";
    printQuoted(os, entry.note);
    os << "\n";
  }
}

void printValidatedRoleOpBoundary(llvm::raw_ostream &os,
                                  ComputeSkeletonOp computeRole) {
  printField(os, "validated_role_op", ComputeSkeletonOp::getOperationName());
  printField(os, "validated_role_op_interface",
             "TCRVEmitCLowerableOpInterface");
  printField(os, "validated_role_op_source",
             computeRole.getTCRVEmitCLowerableSourceOpName());
  printField(os, "validated_role_op_source_role",
             computeRole.getTCRVEmitCLowerableSourceRole());
}

} // namespace

static TargetArtifactRouteMetadata buildTemplateMetadataArtifactRouteMetadata() {
  const pluginTemplate::TemplateConstructionManifest &manifest =
      pluginTemplate::getTemplateConstructionManifest();
  TargetArtifactRouteMetadata metadata(
      pluginTemplate::getTemplateExpectedIntegrationContract(),
      pluginTemplate::getTemplateMetadataRuntimeABIKind(),
      pluginTemplate::getTemplateExpectedIntegrationContract(),
      pluginTemplate::getTemplateMetadataRuntimeGlueRole());
  metadata.addSelectedPlanMetadataRequirement(
      "template_extension_capability_id", pluginTemplate::getTemplateExtensionCapabilityID(),
      "capability-requirement");
  metadata.addSelectedPlanMetadataRequirement(
      "template_extension_integration_contract",
      pluginTemplate::getTemplateExpectedIntegrationContract(),
      "integration-contract");
  metadata.addSelectedPlanMetadataRequirement("template_extension_scope",
                                              "zero-core-integration",
                                              "evidence-scope");
  metadata.addSelectedPlanMetadataRequirement(
      pluginTemplate::getTemplateConstructionProtocolMetadataName(),
      manifest.protocolVersion,
      pluginTemplate::getTemplateConstructionProtocolMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTemplate::getTemplateConstructionArchetypeMetadataName(),
      manifest.archetype,
      pluginTemplate::getTemplateConstructionArchetypeMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTemplate::getTemplateSemanticRoleGraphMetadataName(),
      manifest.semanticRoleGraph,
      pluginTemplate::getTemplateSemanticRoleGraphMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTemplate::getTemplateCommonInterfaceRealizationMetadataName(),
      pluginTemplate::getTemplateConstructionInterfaceRealization(),
      pluginTemplate::getTemplateCommonInterfaceRealizationMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTemplate::getTemplateTypedRoleRealizationMetadataName(),
      pluginTemplate::getTemplateTypedRoleRealizationSummary(),
      pluginTemplate::getTemplateTypedRoleRealizationMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTemplate::getTemplateEmitCRouteMappingMetadataName(),
      manifest.emitcRoute.routeID,
      pluginTemplate::getTemplateEmitCRouteMappingMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTemplate::getTemplateEvidenceProfileMetadataName(),
      manifest.evidenceProfile,
      pluginTemplate::getTemplateEvidenceProfileMetadataRole());
  metadata.addClaimField("artifact_status", kArtifactStatus);
  metadata.addClaimField("runtime_execution_claim", kNoRuntimeClaim);
  metadata.addClaimField("hardware_execution_claim", kNoRuntimeClaim);
  metadata.addClaimField("correctness_claim", kNoRuntimeClaim);
  metadata.addClaimField("performance_claim", kNoRuntimeClaim);
  return metadata;
}

llvm::Error exportTemplateMetadataArtifact(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  llvm::Expected<TargetArtifactCandidate> candidate =
      selectTemplateMetadataCandidate(module);
  if (!candidate)
    return candidate.takeError();

  llvm::Expected<std::string> requiredCapabilitySymbol =
      getRequiredTemplateCapabilitySymbol(*candidate);
  if (!requiredCapabilitySymbol)
    return requiredCapabilitySymbol.takeError();

  VariantOp selectedVariant =
      findDirectVariant(candidate->kernel, candidate->selectedVariant);
  llvm::Expected<ComputeSkeletonOp> computeRole =
      findAndValidateTemplateComputeRoleOp(*candidate, selectedVariant);
  if (!computeRole)
    return computeRole.takeError();

  const pluginTemplate::TemplateConstructionManifest &manifest =
      pluginTemplate::getTemplateConstructionManifest();
  const pluginTemplate::TemplateTypedRoleGraphRealization &realization =
      pluginTemplate::getTemplateTypedRoleGraphRealization();
  llvm::Expected<pluginTemplate::TemplateGeneratedOutputRoute> route =
      pluginTemplate::buildTemplateGeneratedOutputRoute(manifest, realization);
  if (!route)
    return route.takeError();

  os << "tianchenrv.template_metadata_artifact.version: "
     << kMetadataArtifactVersion << "\n";
  printField(os, "artifact_status", kArtifactStatus);
  printField(os, "artifact_description",
             "Template extension construction manifest artifact");
  printField(os, "runtime_execution_claim", kNoRuntimeClaim);
  printField(os, "hardware_execution_claim", kNoRuntimeClaim);
  printField(os, "correctness_claim", kNoRuntimeClaim);
  printField(os, "performance_claim", kNoRuntimeClaim);
  os << "kernel: @" << candidate->kernel.getSymName() << "\n";
  os << "selected_variant: @" << candidate->selectedVariant << "\n";
  printField(os, "role", candidate->role);
  printField(os, "origin_plugin", candidate->origin);
  printField(os, "route", candidate->routeID);
  printField(os, "emission_kind", candidate->emissionKind);
  printField(os, "artifact_kind", candidate->artifactKind);
  printField(os, "runtime_abi", candidate->runtimeABI);
  printField(os, "runtime_abi_kind", candidate->runtimeABIKind);
  printField(os, "runtime_abi_name", candidate->runtimeABIName);
  printField(os, "runtime_glue_role", candidate->runtimeGlueRole);
  printField(os, "lowering_boundary", candidate->loweringBoundary);
  printField(os, "integration_contract", pluginTemplate::getTemplateExpectedIntegrationContract());
  printField(os, "handoff_kind", pluginTemplate::getTemplateExpectedHandoffKind());
  os << "required_capability: @" << *requiredCapabilitySymbol << "\n";
  printField(os, "required_capability_id",
             pluginTemplate::getTemplateExtensionCapabilityID());
  printField(os, "required_capability_kind",
             pluginTemplate::getTemplateExtensionCapabilityKind());
  printField(os, "construction_protocol", manifest.protocolVersion);
  printField(os, "extension_archetype", manifest.archetype);
  printField(os, "semantic_role_graph", manifest.semanticRoleGraph);
  printField(os, "family_name", manifest.family.familyName);
  printField(os, "family_architectural_namespace",
             manifest.family.architecturalNamespace);
  printField(os, "family_concrete_namespace",
             manifest.family.concreteNamespace);
  printField(os, "family_plugin", manifest.family.pluginName);
  printField(os, "family_first_slice_variant",
             manifest.family.firstSliceVariantName);
  for (auto [index, role] : llvm::enumerate(manifest.semanticRoles)) {
    os << "semantic_role[" << index << "]:\n";
    printField(os, "  role", role.role);
    os << "  order: " << role.order << "\n";
    printField(os, "  operation", role.operationName);
    printField(os, "  common_interfaces", role.commonInterfaces);
  }
  printField(os, "common_interface_realization",
             pluginTemplate::getTemplateConstructionInterfaceRealization());
  printValidatedRoleOpBoundary(os, *computeRole);
  construction::emitTypedRoleGraphRealization(os, realization);
  printField(os, "emitc_route_id", manifest.emitcRoute.routeID);
  printField(os, "emitc_emission_kind", manifest.emitcRoute.emissionKind);
  printField(os, "emitc_artifact_kind", manifest.emitcRoute.artifactKind);
  printField(os, "emitc_required_header", manifest.emitcRoute.requiredHeader);
  printField(os, "emitc_role_to_call_map", manifest.emitcRoute.roleToCallMap);
  printField(os, "evidence_profile", manifest.evidenceProfile);
  construction::emitGeneratedOutputRoute(os, *route);
  printSelectedPlanMetadata(os, candidate->selectedPlanMetadata);
  return llvm::Error::success();
}

llvm::Error registerTemplateMetadataArtifactTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerExporter(TargetArtifactExporter(
      pluginTemplate::getTemplateMetadataRouteID(),
      pluginTemplate::getTemplateMetadataArtifactKind(),
      pluginTemplate::getTemplateExtensionPluginName(),
      pluginTemplate::getTemplateMetadataEmissionKind(), exportTemplateMetadataArtifact,
      /*requiredRuntimeABIParameters=*/{},
      pluginTemplate::getTemplateExpectedHandoffKind(),
      validateTemplateMetadataCandidate,
      /*componentGroup=*/{}, /*externalABIName=*/{},
      buildTemplateMetadataArtifactRouteMetadata()));
}

llvm::Error registerTemplateMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginTemplate::getTemplateExtensionPluginName(),
      registerTemplateMetadataArtifactTargetExporters));
}

} // namespace tianchenrv::target::template_ext
