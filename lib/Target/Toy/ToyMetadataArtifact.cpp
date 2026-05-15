#include "TianChenRV/Target/Toy/ToyMetadataArtifact.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv::target::toy {
namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;
namespace construction = tianchenrv::plugin::construction;
namespace pluginToy = tianchenrv::plugin::toy;

using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::target::SelectedPlanMetadataEntry;
using tianchenrv::target::TargetArtifactCandidate;
using tianchenrv::target::TargetArtifactExporter;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::toy::ComputeSkeletonOp;
using tianchenrv::tcrv::toy::LoweringBoundaryOp;

constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kTemplateABIAttrName("template_abi");
constexpr llvm::StringLiteral kHandoffKindAttrName("handoff_kind");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kEmitCCallAttrName("emitc_call");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kMetadataArtifactVersion("1");
constexpr llvm::StringLiteral kArtifactStatus(
    "non-executable-metadata-evidence");
constexpr llvm::StringLiteral kNoRuntimeClaim("none");

struct ToyCapabilityRecord {
  std::string symbolName;
};

llvm::Error makeToyArtifactError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV Toy metadata artifact export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleToyArtifactError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Toy metadata artifact export failed: ") +
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
    return makeToyArtifactError(
        kernel, llvm::Twine("Toy artifact candidate ") + fieldName + " '" +
                    actual + "' does not match expected '" + expected + "'");
  return llvm::Error::success();
}

llvm::Error requireBoundaryStringAttr(LoweringBoundaryOp boundary,
                                      llvm::StringRef attrName,
                                      llvm::StringRef expectedValue) {
  auto attr = getStringAttr(boundary.getOperation(), attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeToyArtifactError(
        boundary->getParentOfType<KernelOp>(),
        llvm::Twine("Toy lowering boundary requires non-empty attribute '") +
            attrName + "'");
  if (attr.getValue() != expectedValue)
    return makeToyArtifactError(
        boundary->getParentOfType<KernelOp>(),
        llvm::Twine("Toy lowering boundary attribute '") + attrName + "' is '" +
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

llvm::Expected<ToyCapabilityRecord> getRequiredToyCapability(KernelOp kernel) {
  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();

  const CapabilityDescriptor *capability =
      capabilities->lookupProviderByID(pluginToy::getToyTemplateCapabilityID());
  if (!capability)
    return makeToyArtifactError(
        kernel, llvm::Twine("requires capability provider for id '") +
                    pluginToy::getToyTemplateCapabilityID() + "'");
  if (!capability->isAvailable())
    return makeToyArtifactError(
        kernel, llvm::Twine("requires available capability provider for id '") +
                    pluginToy::getToyTemplateCapabilityID() + "'");
  if (capability->getKind() != pluginToy::getToyTemplateCapabilityKind())
    return makeToyArtifactError(
        kernel, llvm::Twine("capability id '") +
                    pluginToy::getToyTemplateCapabilityID() +
                    "' kind must be '" +
                    pluginToy::getToyTemplateCapabilityKind() + "'");
  if (capability->getProperty("template_abi") !=
      pluginToy::getToyExpectedTemplateABI())
    return makeToyArtifactError(
        kernel, "Toy capability template_abi does not match the selected "
                "metadata artifact ABI");
  if (capability->getProperty("handoff_kind") !=
      pluginToy::getToyExpectedHandoffKind())
    return makeToyArtifactError(
        kernel, "Toy capability handoff_kind does not match the selected "
                "metadata artifact handoff");

  ToyCapabilityRecord record;
  record.symbolName = capability->getSymbolName().str();
  return record;
}

llvm::Expected<std::string> validateVariantRequiresToyCapability(
    KernelOp kernel, VariantOp variant,
    const ToyCapabilityRecord &toyCapability) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requires || requires.empty())
    return makeToyArtifactError(
        kernel, llvm::Twine("selected Toy variant @") + variant.getSymName() +
                    " requires non-empty 'requires' capability metadata");

  std::string requiredToySymbol;
  for (mlir::Attribute attr : requires) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeToyArtifactError(
          kernel, llvm::Twine("selected Toy variant @") +
                      variant.getSymName() +
                      " requires only non-empty capability symbol references");

    if (symbol.getValue() == toyCapability.symbolName) {
      requiredToySymbol = symbol.getValue().str();
      continue;
    }

    return makeToyArtifactError(
        kernel, llvm::Twine("selected Toy variant @") + variant.getSymName() +
                    " has non-Toy required capability @" + symbol.getValue());
  }

  if (requiredToySymbol.empty())
    return makeToyArtifactError(
        kernel, llvm::Twine("selected Toy variant @") + variant.getSymName() +
                    " must require the Toy template capability");
  return requiredToySymbol;
}

llvm::Expected<LoweringBoundaryOp>
findAndValidateToyBoundary(const TargetArtifactCandidate &candidate,
                           VariantOp variant) {
  KernelOp kernel = candidate.kernel;
  if (!kernel || kernel.getBody().empty())
    return makeToyArtifactError(
        kernel, "requires selected Toy kernel with a materialized body");

  LoweringBoundaryOp matched;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != candidate.loweringBoundary)
      continue;

    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      return makeToyArtifactError(
          kernel, llvm::Twine("lowering boundary operation '") +
                      candidate.loweringBoundary +
                      "' is not a tcrv_toy.lowering_boundary");

    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = getStringAttr(boundary.getOperation(), kRoleAttrName);
    if (!selectedVariant || selectedVariant.getValue() != variant.getSymName() ||
        !role || role.getValue() != candidate.role)
      continue;

    if (matched)
      return makeToyArtifactError(
          kernel, llvm::Twine("duplicate Toy lowering boundaries for @") +
                      candidate.selectedVariant + " as " + candidate.role);
    matched = boundary;
  }

  if (!matched)
    return makeToyArtifactError(
        kernel, llvm::Twine("selected Toy candidate requires one ") +
                    candidate.loweringBoundary + " for @" +
                    candidate.selectedVariant + " as " + candidate.role);

  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kSourceKernelAttrName,
                                    kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kOriginAttrName,
                                    pluginToy::getToyExtensionPluginName()))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kRoleAttrName, candidate.role))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kStatusAttrName,
                                    kMetadataOnlyStatusValue))
    return std::move(error);
  if (llvm::Error error = requireBoundaryStringAttr(
          matched, kTemplateABIAttrName, pluginToy::getToyExpectedTemplateABI()))
    return std::move(error);
  if (llvm::Error error = requireBoundaryStringAttr(
          matched, kHandoffKindAttrName, pluginToy::getToyExpectedHandoffKind()))
    return std::move(error);

  auto boundaryCapabilities =
      matched->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!boundaryCapabilities || boundaryCapabilities != variantRequires)
    return makeToyArtifactError(
        kernel, "Toy lowering boundary required_capabilities must match "
                "selected variant requires metadata");

  return matched;
}

llvm::Error requireComputeRoleStringAttr(ComputeSkeletonOp computeRole,
                                         llvm::StringRef attrName,
                                         llvm::StringRef expectedValue) {
  auto attr = getStringAttr(computeRole.getOperation(), attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeToyArtifactError(
        computeRole->getParentOfType<KernelOp>(),
        llvm::Twine("Toy compute role op requires non-empty attribute '") +
            attrName + "'");
  if (attr.getValue() != expectedValue)
    return makeToyArtifactError(
        computeRole->getParentOfType<KernelOp>(),
        llvm::Twine("Toy compute role op attribute '") + attrName + "' is '" +
            attr.getValue() + "' but expected '" + expectedValue + "'");
  return llvm::Error::success();
}

llvm::Expected<ComputeSkeletonOp>
findAndValidateToyComputeRoleOp(const TargetArtifactCandidate &candidate,
                                VariantOp variant) {
  KernelOp kernel = candidate.kernel;
  if (!kernel || kernel.getBody().empty())
    return makeToyArtifactError(
        kernel, "requires selected Toy kernel with a materialized body");

  ComputeSkeletonOp matched;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != ComputeSkeletonOp::getOperationName())
      continue;

    auto computeRole = llvm::dyn_cast<ComputeSkeletonOp>(op);
    if (!computeRole)
      return makeToyArtifactError(
          kernel, llvm::Twine("Toy role operation '") +
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
      return makeToyArtifactError(
          kernel, llvm::Twine("duplicate Toy compute role ops for @") +
                      candidate.selectedVariant + " as " + candidate.role);
    matched = computeRole;
  }

  if (!matched)
    return makeToyArtifactError(
        kernel, llvm::Twine("selected Toy candidate requires one ") +
                    ComputeSkeletonOp::getOperationName() + " for @" +
                    candidate.selectedVariant + " as " + candidate.role);

  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kSourceKernelAttrName,
                                       kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(
              matched, kOriginAttrName,
              pluginToy::getToyExtensionPluginName()))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kRoleAttrName, candidate.role))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kStatusAttrName,
                                       kRoleOpBoundaryStatusValue))
    return std::move(error);

  auto roleOrder =
      matched->getAttrOfType<mlir::IntegerAttr>(kRoleOrderAttrName);
  if (!roleOrder || roleOrder.getInt() != 2)
    return makeToyArtifactError(kernel,
                                "Toy compute role op role_order must be 2");

  const pluginToy::ToyTypedRoleGraphRealization &realization =
      pluginToy::getToyTypedRoleGraphRealization();
  const pluginToy::ToyTypedRoleInterfaceRealization &compute =
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
    return makeToyArtifactError(
        kernel, "Toy compute role op required_capabilities must match "
                "selected variant requires metadata");

  if (llvm::Error error = pluginToy::verifyToyComputeRoleOpInterface(
          pluginToy::getToyConstructionManifest(), realization,
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
    return makeToyArtifactError(
        candidate.kernel,
        llvm::Twine("Toy artifact candidate requires selected_plan_metadata '") +
            name + "'");
  if (entry->value != expectedValue || entry->role != expectedRole ||
      entry->note.empty())
    return makeToyArtifactError(
        candidate.kernel,
        llvm::Twine("Toy artifact candidate selected_plan_metadata '") + name +
            "' does not match expected value/role");
  return llvm::Error::success();
}

llvm::Error validateToySelectedPlanMetadata(
    const TargetArtifactCandidate &candidate) {
  if (candidate.selectedPlanMetadata.size() != 10)
    return makeToyArtifactError(
        candidate.kernel,
        "Toy artifact candidate requires exactly ten selected plan metadata "
        "entries");

  const pluginToy::ToyConstructionManifest &manifest =
      pluginToy::getToyConstructionManifest();
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "toy_template_capability_id",
          pluginToy::getToyTemplateCapabilityID(), "capability-requirement"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "toy_template_abi", pluginToy::getToyExpectedTemplateABI(),
          "template-abi"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "toy_template_scope", "metadata-only", "evidence-scope"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginToy::getToyConstructionProtocolMetadataName(),
          manifest.protocolVersion,
          pluginToy::getToyConstructionProtocolMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginToy::getToyConstructionArchetypeMetadataName(),
          manifest.archetype,
          pluginToy::getToyConstructionArchetypeMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginToy::getToySemanticRoleGraphMetadataName(),
          manifest.semanticRoleGraph,
          pluginToy::getToySemanticRoleGraphMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginToy::getToyCommonInterfaceRealizationMetadataName(),
          pluginToy::getToyConstructionInterfaceRealization(),
          pluginToy::getToyCommonInterfaceRealizationMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginToy::getToyTypedRoleRealizationMetadataName(),
          pluginToy::getToyTypedRoleRealizationSummary(),
          pluginToy::getToyTypedRoleRealizationMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginToy::getToyEmitCRouteMappingMetadataName(),
          manifest.emitcRoute.routeID,
          pluginToy::getToyEmitCRouteMappingMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginToy::getToyEvidenceProfileMetadataName(),
          manifest.evidenceProfile,
          pluginToy::getToyEvidenceProfileMetadataRole()))
    return error;
  return llvm::Error::success();
}

llvm::Error validateToyMetadataCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = pluginToy::verifyToyConstructionManifest(
          pluginToy::getToyConstructionManifest()))
    return error;
  if (llvm::Error error = pluginToy::verifyToyTypedRoleGraphRealization(
          pluginToy::getToyConstructionManifest(),
          pluginToy::getToyTypedRoleGraphRealization()))
    return error;
  if (llvm::Expected<pluginToy::ToyGeneratedOutputRoute> route =
          pluginToy::buildToyGeneratedOutputRoute(
              pluginToy::getToyConstructionManifest(),
              pluginToy::getToyTypedRoleGraphRealization());
      !route)
    return route.takeError();

  KernelOp kernel = candidate.kernel;
  if (!kernel)
    return makeToyArtifactError(
        kernel, "requires a selected Toy kernel from target artifact "
                "candidate collection");

  if (llvm::Error error = requireCandidateField(
          kernel, "origin", candidate.origin,
          pluginToy::getToyExtensionPluginName()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "route id", candidate.routeID,
          pluginToy::getToyMetadataRouteID()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "emission kind", candidate.emissionKind,
          pluginToy::getToyMetadataEmissionKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "artifact kind", candidate.artifactKind,
          pluginToy::getToyMetadataArtifactKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI", candidate.runtimeABI,
          pluginToy::getToyExpectedTemplateABI()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI kind", candidate.runtimeABIKind,
          pluginToy::getToyMetadataRuntimeABIKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI name", candidate.runtimeABIName,
          pluginToy::getToyExpectedTemplateABI()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime glue role", candidate.runtimeGlueRole,
          pluginToy::getToyMetadataRuntimeGlueRole()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "lowering boundary", candidate.loweringBoundary,
          LoweringBoundaryOp::getOperationName()))
    return error;
  if (!candidate.runtimeABIParameters.empty())
    return makeToyArtifactError(
        kernel, "Toy metadata artifact route must not carry executable "
                "runtime ABI parameters");

  if (candidate.selectedVariant !=
      pluginToy::getToyTemplateFirstSliceVariantName())
    return makeToyArtifactError(
        kernel, llvm::Twine("Toy artifact candidate selected variant @") +
                    candidate.selectedVariant + " does not match @" +
                    pluginToy::getToyTemplateFirstSliceVariantName());

  VariantOp variant = findDirectVariant(kernel, candidate.selectedVariant);
  if (!variant)
    return makeToyArtifactError(
        kernel, llvm::Twine("selected Toy variant @") +
                    candidate.selectedVariant +
                    " must resolve to a direct tcrv.exec.variant");
  auto origin = getStringAttr(variant.getOperation(), kOriginAttrName);
  if (!origin || origin.getValue() != pluginToy::getToyExtensionPluginName())
    return makeToyArtifactError(
        kernel, llvm::Twine("selected Toy variant @") +
                    candidate.selectedVariant +
                    " must be owned by origin 'toy-plugin'");

  llvm::Expected<ToyCapabilityRecord> capability =
      getRequiredToyCapability(kernel);
  if (!capability)
    return capability.takeError();
  llvm::Expected<std::string> requiredToySymbol =
      validateVariantRequiresToyCapability(kernel, variant, *capability);
  if (!requiredToySymbol)
    return requiredToySymbol.takeError();

  if (llvm::Error error = validateToySelectedPlanMetadata(candidate))
    return error;

  llvm::Expected<LoweringBoundaryOp> boundary =
      findAndValidateToyBoundary(candidate, variant);
  if (!boundary)
    return boundary.takeError();

  llvm::Expected<ComputeSkeletonOp> computeRole =
      findAndValidateToyComputeRoleOp(candidate, variant);
  if (!computeRole)
    return computeRole.takeError();

  return llvm::Error::success();
}

llvm::Expected<TargetArtifactCandidate>
selectToyMetadataCandidate(mlir::ModuleOp module) {
  llvm::SmallVector<TargetArtifactCandidate, 4> candidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, candidates))
    return std::move(error);

  llvm::SmallVector<TargetArtifactCandidate, 2> toyCandidates;
  for (const TargetArtifactCandidate &candidate : candidates) {
    bool toyShaped =
        candidate.origin == pluginToy::getToyExtensionPluginName() ||
        candidate.routeID == pluginToy::getToyMetadataRouteID() ||
        candidate.emissionKind == pluginToy::getToyMetadataEmissionKind() ||
        candidate.artifactKind == pluginToy::getToyMetadataArtifactKind();
    if (!toyShaped)
      continue;

    if (llvm::Error error = validateToyMetadataCandidate(candidate))
      return std::move(error);
    toyCandidates.push_back(candidate);
  }

  if (toyCandidates.empty())
    return makeModuleToyArtifactError(
        "requires exactly one selected Toy metadata artifact candidate; found "
        "none");
  if (toyCandidates.size() > 1)
    return makeModuleToyArtifactError(
        "requires exactly one selected Toy metadata artifact candidate; found "
        "multiple");
  return toyCandidates.front();
}

llvm::Expected<std::string>
getRequiredToyCapabilitySymbol(const TargetArtifactCandidate &candidate) {
  llvm::Expected<ToyCapabilityRecord> capability =
      getRequiredToyCapability(candidate.kernel);
  if (!capability)
    return capability.takeError();
  VariantOp variant = findDirectVariant(candidate.kernel,
                                        candidate.selectedVariant);
  return validateVariantRequiresToyCapability(candidate.kernel, variant,
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

static TargetArtifactRouteMetadata buildToyMetadataArtifactRouteMetadata() {
  const pluginToy::ToyConstructionManifest &manifest =
      pluginToy::getToyConstructionManifest();
  TargetArtifactRouteMetadata metadata(
      pluginToy::getToyExpectedTemplateABI(),
      pluginToy::getToyMetadataRuntimeABIKind(),
      pluginToy::getToyExpectedTemplateABI(),
      pluginToy::getToyMetadataRuntimeGlueRole());
  metadata.addSelectedPlanMetadataRequirement(
      "toy_template_capability_id", pluginToy::getToyTemplateCapabilityID(),
      "capability-requirement");
  metadata.addSelectedPlanMetadataRequirement(
      "toy_template_abi", pluginToy::getToyExpectedTemplateABI(),
      "template-abi");
  metadata.addSelectedPlanMetadataRequirement("toy_template_scope",
                                              "metadata-only",
                                              "evidence-scope");
  metadata.addSelectedPlanMetadataRequirement(
      pluginToy::getToyConstructionProtocolMetadataName(),
      manifest.protocolVersion,
      pluginToy::getToyConstructionProtocolMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginToy::getToyConstructionArchetypeMetadataName(),
      manifest.archetype,
      pluginToy::getToyConstructionArchetypeMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginToy::getToySemanticRoleGraphMetadataName(),
      manifest.semanticRoleGraph,
      pluginToy::getToySemanticRoleGraphMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginToy::getToyCommonInterfaceRealizationMetadataName(),
      pluginToy::getToyConstructionInterfaceRealization(),
      pluginToy::getToyCommonInterfaceRealizationMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginToy::getToyTypedRoleRealizationMetadataName(),
      pluginToy::getToyTypedRoleRealizationSummary(),
      pluginToy::getToyTypedRoleRealizationMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginToy::getToyEmitCRouteMappingMetadataName(),
      manifest.emitcRoute.routeID,
      pluginToy::getToyEmitCRouteMappingMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginToy::getToyEvidenceProfileMetadataName(),
      manifest.evidenceProfile,
      pluginToy::getToyEvidenceProfileMetadataRole());
  metadata.addClaimField("artifact_status", kArtifactStatus);
  metadata.addClaimField("runtime_execution_claim", kNoRuntimeClaim);
  metadata.addClaimField("hardware_execution_claim", kNoRuntimeClaim);
  metadata.addClaimField("correctness_claim", kNoRuntimeClaim);
  metadata.addClaimField("performance_claim", kNoRuntimeClaim);
  return metadata;
}

llvm::Error exportToyMetadataArtifact(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  llvm::Expected<TargetArtifactCandidate> candidate =
      selectToyMetadataCandidate(module);
  if (!candidate)
    return candidate.takeError();

  llvm::Expected<std::string> requiredCapabilitySymbol =
      getRequiredToyCapabilitySymbol(*candidate);
  if (!requiredCapabilitySymbol)
    return requiredCapabilitySymbol.takeError();

  VariantOp selectedVariant =
      findDirectVariant(candidate->kernel, candidate->selectedVariant);
  llvm::Expected<ComputeSkeletonOp> computeRole =
      findAndValidateToyComputeRoleOp(*candidate, selectedVariant);
  if (!computeRole)
    return computeRole.takeError();

  const pluginToy::ToyConstructionManifest &manifest =
      pluginToy::getToyConstructionManifest();
  const pluginToy::ToyTypedRoleGraphRealization &realization =
      pluginToy::getToyTypedRoleGraphRealization();
  llvm::Expected<pluginToy::ToyGeneratedOutputRoute> route =
      pluginToy::buildToyGeneratedOutputRoute(manifest, realization);
  if (!route)
    return route.takeError();

  os << "tianchenrv.toy_metadata_artifact.version: "
     << kMetadataArtifactVersion << "\n";
  printField(os, "artifact_status", kArtifactStatus);
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
  printField(os, "template_abi", pluginToy::getToyExpectedTemplateABI());
  printField(os, "handoff_kind", pluginToy::getToyExpectedHandoffKind());
  os << "required_capability: @" << *requiredCapabilitySymbol << "\n";
  printField(os, "required_capability_id",
             pluginToy::getToyTemplateCapabilityID());
  printField(os, "required_capability_kind",
             pluginToy::getToyTemplateCapabilityKind());
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
             pluginToy::getToyConstructionInterfaceRealization());
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

llvm::Error registerToyMetadataArtifactTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerExporter(TargetArtifactExporter(
      pluginToy::getToyMetadataRouteID(),
      pluginToy::getToyMetadataArtifactKind(),
      pluginToy::getToyExtensionPluginName(),
      pluginToy::getToyMetadataEmissionKind(), exportToyMetadataArtifact,
      /*requiredRuntimeABIParameters=*/{},
      pluginToy::getToyExpectedHandoffKind(), validateToyMetadataCandidate,
      /*componentGroup=*/{}, /*externalABIName=*/{},
      buildToyMetadataArtifactRouteMetadata()));
}

llvm::Error registerToyMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginToy::getToyExtensionPluginName(),
      registerToyMetadataArtifactTargetExporters));
}

} // namespace tianchenrv::target::toy
