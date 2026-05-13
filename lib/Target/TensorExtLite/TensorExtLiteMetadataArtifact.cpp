#include "TianChenRV/Target/TensorExtLite/TensorExtLiteMetadataArtifact.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv::target::tensorext_lite {
namespace {

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;
namespace construction = tianchenrv::plugin::construction;
namespace pluginTensorExtLite = tianchenrv::plugin::tensorext_lite;

using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::target::SelectedPlanMetadataEntry;
using tianchenrv::target::TargetArtifactCandidate;
using tianchenrv::target::TargetArtifactExporter;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::tensorext_lite::TileMmaSkeletonOp;
using tianchenrv::tcrv::tensorext_lite::LoweringBoundaryOp;

constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kFragmentABIAttrName("fragment_abi");
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

struct TensorExtLiteCapabilityRecord {
  std::string symbolName;
};

llvm::Error makeTensorExtLiteArtifactError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV TensorExtLite metadata artifact export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleTensorExtLiteArtifactError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TensorExtLite metadata artifact export failed: ") +
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
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("TensorExtLite artifact candidate ") + fieldName + " '" +
                    actual + "' does not match expected '" + expected + "'");
  return llvm::Error::success();
}

llvm::Error requireBoundaryStringAttr(LoweringBoundaryOp boundary,
                                      llvm::StringRef attrName,
                                      llvm::StringRef expectedValue) {
  auto attr = getStringAttr(boundary.getOperation(), attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeTensorExtLiteArtifactError(
        boundary->getParentOfType<KernelOp>(),
        llvm::Twine("TensorExtLite lowering boundary requires non-empty attribute '") +
            attrName + "'");
  if (attr.getValue() != expectedValue)
    return makeTensorExtLiteArtifactError(
        boundary->getParentOfType<KernelOp>(),
        llvm::Twine("TensorExtLite lowering boundary attribute '") + attrName + "' is '" +
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

llvm::Expected<TensorExtLiteCapabilityRecord> getRequiredTensorExtLiteCapability(KernelOp kernel) {
  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();

  const CapabilityDescriptor *capability =
      capabilities->lookupProviderByID(pluginTensorExtLite::getTensorExtLiteFragmentCapabilityID());
  if (!capability)
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("requires capability provider for id '") +
                    pluginTensorExtLite::getTensorExtLiteFragmentCapabilityID() + "'");
  if (!capability->isAvailable())
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("requires available capability provider for id '") +
                    pluginTensorExtLite::getTensorExtLiteFragmentCapabilityID() + "'");
  if (capability->getKind() != pluginTensorExtLite::getTensorExtLiteFragmentCapabilityKind())
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("capability id '") +
                    pluginTensorExtLite::getTensorExtLiteFragmentCapabilityID() +
                    "' kind must be '" +
                    pluginTensorExtLite::getTensorExtLiteFragmentCapabilityKind() + "'");
  if (capability->getProperty("fragment_abi") !=
      pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI())
    return makeTensorExtLiteArtifactError(
        kernel, "TensorExtLite capability fragment_abi does not match the selected "
                "metadata artifact ABI");
  if (capability->getProperty("handoff_kind") !=
      pluginTensorExtLite::getTensorExtLiteExpectedHandoffKind())
    return makeTensorExtLiteArtifactError(
        kernel, "TensorExtLite capability handoff_kind does not match the selected "
                "metadata artifact handoff");

  TensorExtLiteCapabilityRecord record;
  record.symbolName = capability->getSymbolName().str();
  return record;
}

llvm::Expected<std::string> validateVariantRequiresTensorExtLiteCapability(
    KernelOp kernel, VariantOp variant,
    const TensorExtLiteCapabilityRecord &tensorext_liteCapability) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requires || requires.empty())
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("selected TensorExtLite variant @") + variant.getSymName() +
                    " requires non-empty 'requires' capability metadata");

  std::string requiredTensorExtLiteSymbol;
  for (mlir::Attribute attr : requires) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol || symbol.getValue().trim().empty())
      return makeTensorExtLiteArtifactError(
          kernel, llvm::Twine("selected TensorExtLite variant @") +
                      variant.getSymName() +
                      " requires only non-empty capability symbol references");

    if (symbol.getValue() == tensorext_liteCapability.symbolName) {
      requiredTensorExtLiteSymbol = symbol.getValue().str();
      continue;
    }

    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("selected TensorExtLite variant @") + variant.getSymName() +
                    " has non-TensorExtLite required capability @" + symbol.getValue());
  }

  if (requiredTensorExtLiteSymbol.empty())
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("selected TensorExtLite variant @") + variant.getSymName() +
                    " must require the TensorExtLite fragment capability");
  return requiredTensorExtLiteSymbol;
}

llvm::Expected<LoweringBoundaryOp>
findAndValidateTensorExtLiteBoundary(const TargetArtifactCandidate &candidate,
                           VariantOp variant) {
  KernelOp kernel = candidate.kernel;
  if (!kernel || kernel.getBody().empty())
    return makeTensorExtLiteArtifactError(
        kernel, "requires selected TensorExtLite kernel with a materialized body");

  LoweringBoundaryOp matched;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != candidate.loweringBoundary)
      continue;

    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      return makeTensorExtLiteArtifactError(
          kernel, llvm::Twine("lowering boundary operation '") +
                      candidate.loweringBoundary +
                      "' is not a tcrv_tensorext_lite.lowering_boundary");

    auto selectedVariant =
        boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = getStringAttr(boundary.getOperation(), kRoleAttrName);
    if (!selectedVariant || selectedVariant.getValue() != variant.getSymName() ||
        !role || role.getValue() != candidate.role)
      continue;

    if (matched)
      return makeTensorExtLiteArtifactError(
          kernel, llvm::Twine("duplicate TensorExtLite lowering boundaries for @") +
                      candidate.selectedVariant + " as " + candidate.role);
    matched = boundary;
  }

  if (!matched)
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("selected TensorExtLite candidate requires one ") +
                    candidate.loweringBoundary + " for @" +
                    candidate.selectedVariant + " as " + candidate.role);

  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kSourceKernelAttrName,
                                    kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kOriginAttrName,
                                    pluginTensorExtLite::getTensorExtLiteExtensionPluginName()))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kRoleAttrName, candidate.role))
    return std::move(error);
  if (llvm::Error error =
          requireBoundaryStringAttr(matched, kStatusAttrName,
                                    kMetadataOnlyStatusValue))
    return std::move(error);
  if (llvm::Error error = requireBoundaryStringAttr(
          matched, kFragmentABIAttrName, pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI()))
    return std::move(error);
  if (llvm::Error error = requireBoundaryStringAttr(
          matched, kHandoffKindAttrName, pluginTensorExtLite::getTensorExtLiteExpectedHandoffKind()))
    return std::move(error);

  auto boundaryCapabilities =
      matched->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!boundaryCapabilities || boundaryCapabilities != variantRequires)
    return makeTensorExtLiteArtifactError(
        kernel, "TensorExtLite lowering boundary required_capabilities must match "
                "selected variant requires metadata");

  return matched;
}

llvm::Error requireComputeRoleStringAttr(TileMmaSkeletonOp computeRole,
                                         llvm::StringRef attrName,
                                         llvm::StringRef expectedValue) {
  auto attr = getStringAttr(computeRole.getOperation(), attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeTensorExtLiteArtifactError(
        computeRole->getParentOfType<KernelOp>(),
        llvm::Twine("TensorExtLite tile_mma role op requires non-empty attribute '") +
            attrName + "'");
  if (attr.getValue() != expectedValue)
    return makeTensorExtLiteArtifactError(
        computeRole->getParentOfType<KernelOp>(),
        llvm::Twine("TensorExtLite tile_mma role op attribute '") + attrName + "' is '" +
            attr.getValue() + "' but expected '" + expectedValue + "'");
  return llvm::Error::success();
}

llvm::Expected<TileMmaSkeletonOp>
findAndValidateTensorExtLiteComputeRoleOp(const TargetArtifactCandidate &candidate,
                                VariantOp variant) {
  KernelOp kernel = candidate.kernel;
  if (!kernel || kernel.getBody().empty())
    return makeTensorExtLiteArtifactError(
        kernel, "requires selected TensorExtLite kernel with a materialized body");

  TileMmaSkeletonOp matched;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != TileMmaSkeletonOp::getOperationName())
      continue;

    auto computeRole = llvm::dyn_cast<TileMmaSkeletonOp>(op);
    if (!computeRole)
      return makeTensorExtLiteArtifactError(
          kernel, llvm::Twine("TensorExtLite role operation '") +
                      TileMmaSkeletonOp::getOperationName() +
                      "' does not materialize as TileMmaSkeletonOp");

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(
            kSelectedVariantAttrName);
    auto role = getStringAttr(computeRole.getOperation(), kRoleAttrName);
    if (!selectedVariant || selectedVariant.getValue() != variant.getSymName() ||
        !role || role.getValue() != candidate.role)
      continue;

    if (matched)
      return makeTensorExtLiteArtifactError(
          kernel, llvm::Twine("duplicate TensorExtLite tile_mma role ops for @") +
                      candidate.selectedVariant + " as " + candidate.role);
    matched = computeRole;
  }

  if (!matched)
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("selected TensorExtLite candidate requires one ") +
                    TileMmaSkeletonOp::getOperationName() + " for @" +
                    candidate.selectedVariant + " as " + candidate.role);

  if (llvm::Error error =
          requireComputeRoleStringAttr(matched, kSourceKernelAttrName,
                                       kernel.getSymName()))
    return std::move(error);
  if (llvm::Error error =
          requireComputeRoleStringAttr(
              matched, kOriginAttrName,
              pluginTensorExtLite::getTensorExtLiteExtensionPluginName()))
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
    return makeTensorExtLiteArtifactError(kernel,
                                "TensorExtLite tile_mma role op role_order must be 2");

  const pluginTensorExtLite::TensorExtLiteTypedRoleGraphRealization &realization =
      pluginTensorExtLite::getTensorExtLiteTypedRoleGraphRealization();
  const pluginTensorExtLite::TensorExtLiteTypedRoleInterfaceRealization &compute =
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
    return makeTensorExtLiteArtifactError(
        kernel, "TensorExtLite tile_mma role op required_capabilities must match "
                "selected variant requires metadata");

  if (llvm::Error error = pluginTensorExtLite::verifyTensorExtLiteComputeRoleOpInterface(
          pluginTensorExtLite::getTensorExtLiteConstructionManifest(), realization,
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
    return makeTensorExtLiteArtifactError(
        candidate.kernel,
        llvm::Twine("TensorExtLite artifact candidate requires selected_plan_metadata '") +
            name + "'");
  if (entry->value != expectedValue || entry->role != expectedRole ||
      entry->note.empty())
    return makeTensorExtLiteArtifactError(
        candidate.kernel,
        llvm::Twine("TensorExtLite artifact candidate selected_plan_metadata '") + name +
            "' does not match expected value/role");
  return llvm::Error::success();
}

llvm::Error validateTensorExtLiteSelectedPlanMetadata(
    const TargetArtifactCandidate &candidate) {
  if (candidate.selectedPlanMetadata.size() != 10)
    return makeTensorExtLiteArtifactError(
        candidate.kernel,
        "TensorExtLite artifact candidate requires exactly ten selected plan metadata "
        "entries");

  const pluginTensorExtLite::TensorExtLiteConstructionManifest &manifest =
      pluginTensorExtLite::getTensorExtLiteConstructionManifest();
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "tensorext_lite_tile_mma_capability_id",
          pluginTensorExtLite::getTensorExtLiteFragmentCapabilityID(), "capability-requirement"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "tensorext_lite_tile_mma_abi", pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI(),
          "fragment-abi"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, "tensorext_lite_tile_mma_scope", "metadata-only", "evidence-scope"))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTensorExtLite::getTensorExtLiteConstructionProtocolMetadataName(),
          manifest.protocolVersion,
          pluginTensorExtLite::getTensorExtLiteConstructionProtocolMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTensorExtLite::getTensorExtLiteConstructionArchetypeMetadataName(),
          manifest.archetype,
          pluginTensorExtLite::getTensorExtLiteConstructionArchetypeMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTensorExtLite::getTensorExtLiteSemanticRoleGraphMetadataName(),
          manifest.semanticRoleGraph,
          pluginTensorExtLite::getTensorExtLiteSemanticRoleGraphMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTensorExtLite::getTensorExtLiteCommonInterfaceRealizationMetadataName(),
          pluginTensorExtLite::getTensorExtLiteConstructionInterfaceRealization(),
          pluginTensorExtLite::getTensorExtLiteCommonInterfaceRealizationMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTensorExtLite::getTensorExtLiteTypedRoleRealizationMetadataName(),
          pluginTensorExtLite::getTensorExtLiteTypedRoleRealizationSummary(),
          pluginTensorExtLite::getTensorExtLiteTypedRoleRealizationMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTensorExtLite::getTensorExtLiteEmitCRouteMappingMetadataName(),
          manifest.emitcRoute.routeID,
          pluginTensorExtLite::getTensorExtLiteEmitCRouteMappingMetadataRole()))
    return error;
  if (llvm::Error error = requireSelectedPlanMetadata(
          candidate, pluginTensorExtLite::getTensorExtLiteEvidenceProfileMetadataName(),
          manifest.evidenceProfile,
          pluginTensorExtLite::getTensorExtLiteEvidenceProfileMetadataRole()))
    return error;
  return llvm::Error::success();
}

llvm::Error validateTensorExtLiteMetadataCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = pluginTensorExtLite::verifyTensorExtLiteConstructionManifest(
          pluginTensorExtLite::getTensorExtLiteConstructionManifest()))
    return error;
  if (llvm::Error error = pluginTensorExtLite::verifyTensorExtLiteTypedRoleGraphRealization(
          pluginTensorExtLite::getTensorExtLiteConstructionManifest(),
          pluginTensorExtLite::getTensorExtLiteTypedRoleGraphRealization()))
    return error;
  if (llvm::Expected<pluginTensorExtLite::TensorExtLiteGeneratedOutputRoute> route =
          pluginTensorExtLite::buildTensorExtLiteGeneratedOutputRoute(
              pluginTensorExtLite::getTensorExtLiteConstructionManifest(),
              pluginTensorExtLite::getTensorExtLiteTypedRoleGraphRealization());
      !route)
    return route.takeError();

  KernelOp kernel = candidate.kernel;
  if (!kernel)
    return makeTensorExtLiteArtifactError(
        kernel, "requires a selected TensorExtLite kernel from target artifact "
                "candidate collection");

  if (llvm::Error error = requireCandidateField(
          kernel, "origin", candidate.origin,
          pluginTensorExtLite::getTensorExtLiteExtensionPluginName()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "route id", candidate.routeID,
          pluginTensorExtLite::getTensorExtLiteMetadataRouteID()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "emission kind", candidate.emissionKind,
          pluginTensorExtLite::getTensorExtLiteMetadataEmissionKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "artifact kind", candidate.artifactKind,
          pluginTensorExtLite::getTensorExtLiteMetadataArtifactKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI", candidate.runtimeABI,
          pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI kind", candidate.runtimeABIKind,
          pluginTensorExtLite::getTensorExtLiteMetadataRuntimeABIKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime ABI name", candidate.runtimeABIName,
          pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "runtime glue role", candidate.runtimeGlueRole,
          pluginTensorExtLite::getTensorExtLiteMetadataRuntimeGlueRole()))
    return error;
  if (llvm::Error error = requireCandidateField(
          kernel, "lowering boundary", candidate.loweringBoundary,
          LoweringBoundaryOp::getOperationName()))
    return error;
  if (!candidate.runtimeABIParameters.empty())
    return makeTensorExtLiteArtifactError(
        kernel, "TensorExtLite metadata artifact route must not carry executable "
                "runtime ABI parameters");

  if (candidate.selectedVariant !=
      pluginTensorExtLite::getTensorExtLiteFragmentFirstSliceVariantName())
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("TensorExtLite artifact candidate selected variant @") +
                    candidate.selectedVariant + " does not match @" +
                    pluginTensorExtLite::getTensorExtLiteFragmentFirstSliceVariantName());

  VariantOp variant = findDirectVariant(kernel, candidate.selectedVariant);
  if (!variant)
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("selected TensorExtLite variant @") +
                    candidate.selectedVariant +
                    " must resolve to a direct tcrv.exec.variant");
  auto origin = getStringAttr(variant.getOperation(), kOriginAttrName);
  if (!origin || origin.getValue() != pluginTensorExtLite::getTensorExtLiteExtensionPluginName())
    return makeTensorExtLiteArtifactError(
        kernel, llvm::Twine("selected TensorExtLite variant @") +
                    candidate.selectedVariant +
                    " must be owned by origin 'tensorext-lite-plugin'");

  llvm::Expected<TensorExtLiteCapabilityRecord> capability =
      getRequiredTensorExtLiteCapability(kernel);
  if (!capability)
    return capability.takeError();
  llvm::Expected<std::string> requiredTensorExtLiteSymbol =
      validateVariantRequiresTensorExtLiteCapability(kernel, variant, *capability);
  if (!requiredTensorExtLiteSymbol)
    return requiredTensorExtLiteSymbol.takeError();

  if (llvm::Error error = validateTensorExtLiteSelectedPlanMetadata(candidate))
    return error;

  llvm::Expected<LoweringBoundaryOp> boundary =
      findAndValidateTensorExtLiteBoundary(candidate, variant);
  if (!boundary)
    return boundary.takeError();

  llvm::Expected<TileMmaSkeletonOp> computeRole =
      findAndValidateTensorExtLiteComputeRoleOp(candidate, variant);
  if (!computeRole)
    return computeRole.takeError();

  return llvm::Error::success();
}

llvm::Expected<TargetArtifactCandidate>
selectTensorExtLiteMetadataCandidate(mlir::ModuleOp module) {
  llvm::SmallVector<TargetArtifactCandidate, 4> candidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, candidates))
    return std::move(error);

  llvm::SmallVector<TargetArtifactCandidate, 2> tensorext_liteCandidates;
  for (const TargetArtifactCandidate &candidate : candidates) {
    bool tensorext_liteShaped =
        candidate.origin == pluginTensorExtLite::getTensorExtLiteExtensionPluginName() ||
        candidate.routeID == pluginTensorExtLite::getTensorExtLiteMetadataRouteID() ||
        candidate.emissionKind == pluginTensorExtLite::getTensorExtLiteMetadataEmissionKind() ||
        candidate.artifactKind == pluginTensorExtLite::getTensorExtLiteMetadataArtifactKind();
    if (!tensorext_liteShaped)
      continue;

    if (llvm::Error error = validateTensorExtLiteMetadataCandidate(candidate))
      return std::move(error);
    tensorext_liteCandidates.push_back(candidate);
  }

  if (tensorext_liteCandidates.empty())
    return makeModuleTensorExtLiteArtifactError(
        "requires exactly one selected TensorExtLite metadata artifact candidate; found "
        "none");
  if (tensorext_liteCandidates.size() > 1)
    return makeModuleTensorExtLiteArtifactError(
        "requires exactly one selected TensorExtLite metadata artifact candidate; found "
        "multiple");
  return tensorext_liteCandidates.front();
}

llvm::Expected<std::string>
getRequiredTensorExtLiteCapabilitySymbol(const TargetArtifactCandidate &candidate) {
  llvm::Expected<TensorExtLiteCapabilityRecord> capability =
      getRequiredTensorExtLiteCapability(candidate.kernel);
  if (!capability)
    return capability.takeError();
  VariantOp variant = findDirectVariant(candidate.kernel,
                                        candidate.selectedVariant);
  return validateVariantRequiresTensorExtLiteCapability(candidate.kernel, variant,
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
                                  TileMmaSkeletonOp computeRole) {
  printField(os, "validated_role_op", TileMmaSkeletonOp::getOperationName());
  printField(os, "validated_role_op_interface",
             "TCRVEmitCLowerableOpInterface");
  printField(os, "validated_role_op_source",
             computeRole.getTCRVEmitCLowerableSourceOpName());
  printField(os, "validated_role_op_source_role",
             computeRole.getTCRVEmitCLowerableSourceRole());
}

} // namespace

static TargetArtifactRouteMetadata buildTensorExtLiteMetadataArtifactRouteMetadata() {
  const pluginTensorExtLite::TensorExtLiteConstructionManifest &manifest =
      pluginTensorExtLite::getTensorExtLiteConstructionManifest();
  TargetArtifactRouteMetadata metadata(
      pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI(),
      pluginTensorExtLite::getTensorExtLiteMetadataRuntimeABIKind(),
      pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI(),
      pluginTensorExtLite::getTensorExtLiteMetadataRuntimeGlueRole());
  metadata.addSelectedPlanMetadataRequirement(
      "tensorext_lite_tile_mma_capability_id", pluginTensorExtLite::getTensorExtLiteFragmentCapabilityID(),
      "capability-requirement");
  metadata.addSelectedPlanMetadataRequirement(
      "tensorext_lite_tile_mma_abi", pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI(),
      "fragment-abi");
  metadata.addSelectedPlanMetadataRequirement("tensorext_lite_tile_mma_scope",
                                              "metadata-only",
                                              "evidence-scope");
  metadata.addSelectedPlanMetadataRequirement(
      pluginTensorExtLite::getTensorExtLiteConstructionProtocolMetadataName(),
      manifest.protocolVersion,
      pluginTensorExtLite::getTensorExtLiteConstructionProtocolMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTensorExtLite::getTensorExtLiteConstructionArchetypeMetadataName(),
      manifest.archetype,
      pluginTensorExtLite::getTensorExtLiteConstructionArchetypeMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTensorExtLite::getTensorExtLiteSemanticRoleGraphMetadataName(),
      manifest.semanticRoleGraph,
      pluginTensorExtLite::getTensorExtLiteSemanticRoleGraphMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTensorExtLite::getTensorExtLiteCommonInterfaceRealizationMetadataName(),
      pluginTensorExtLite::getTensorExtLiteConstructionInterfaceRealization(),
      pluginTensorExtLite::getTensorExtLiteCommonInterfaceRealizationMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTensorExtLite::getTensorExtLiteTypedRoleRealizationMetadataName(),
      pluginTensorExtLite::getTensorExtLiteTypedRoleRealizationSummary(),
      pluginTensorExtLite::getTensorExtLiteTypedRoleRealizationMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTensorExtLite::getTensorExtLiteEmitCRouteMappingMetadataName(),
      manifest.emitcRoute.routeID,
      pluginTensorExtLite::getTensorExtLiteEmitCRouteMappingMetadataRole());
  metadata.addSelectedPlanMetadataRequirement(
      pluginTensorExtLite::getTensorExtLiteEvidenceProfileMetadataName(),
      manifest.evidenceProfile,
      pluginTensorExtLite::getTensorExtLiteEvidenceProfileMetadataRole());
  metadata.addClaimField("artifact_status", kArtifactStatus);
  metadata.addClaimField("runtime_execution_claim", kNoRuntimeClaim);
  metadata.addClaimField("hardware_execution_claim", kNoRuntimeClaim);
  metadata.addClaimField("correctness_claim", kNoRuntimeClaim);
  metadata.addClaimField("performance_claim", kNoRuntimeClaim);
  return metadata;
}

llvm::Error exportTensorExtLiteMetadataArtifact(mlir::ModuleOp module,
                                      llvm::raw_ostream &os) {
  llvm::Expected<TargetArtifactCandidate> candidate =
      selectTensorExtLiteMetadataCandidate(module);
  if (!candidate)
    return candidate.takeError();

  llvm::Expected<std::string> requiredCapabilitySymbol =
      getRequiredTensorExtLiteCapabilitySymbol(*candidate);
  if (!requiredCapabilitySymbol)
    return requiredCapabilitySymbol.takeError();

  VariantOp selectedVariant =
      findDirectVariant(candidate->kernel, candidate->selectedVariant);
  llvm::Expected<TileMmaSkeletonOp> computeRole =
      findAndValidateTensorExtLiteComputeRoleOp(*candidate, selectedVariant);
  if (!computeRole)
    return computeRole.takeError();

  const pluginTensorExtLite::TensorExtLiteConstructionManifest &manifest =
      pluginTensorExtLite::getTensorExtLiteConstructionManifest();
  const pluginTensorExtLite::TensorExtLiteTypedRoleGraphRealization &realization =
      pluginTensorExtLite::getTensorExtLiteTypedRoleGraphRealization();
  llvm::Expected<pluginTensorExtLite::TensorExtLiteGeneratedOutputRoute> route =
      pluginTensorExtLite::buildTensorExtLiteGeneratedOutputRoute(manifest, realization);
  if (!route)
    return route.takeError();

  os << "tianchenrv.tensorext_lite_metadata_artifact.version: "
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
  printField(os, "fragment_abi", pluginTensorExtLite::getTensorExtLiteExpectedFragmentABI());
  printField(os, "handoff_kind", pluginTensorExtLite::getTensorExtLiteExpectedHandoffKind());
  os << "required_capability: @" << *requiredCapabilitySymbol << "\n";
  printField(os, "required_capability_id",
             pluginTensorExtLite::getTensorExtLiteFragmentCapabilityID());
  printField(os, "required_capability_kind",
             pluginTensorExtLite::getTensorExtLiteFragmentCapabilityKind());
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
             pluginTensorExtLite::getTensorExtLiteConstructionInterfaceRealization());
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

llvm::Error registerTensorExtLiteMetadataArtifactTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerExporter(TargetArtifactExporter(
      pluginTensorExtLite::getTensorExtLiteMetadataRouteID(),
      pluginTensorExtLite::getTensorExtLiteMetadataArtifactKind(),
      pluginTensorExtLite::getTensorExtLiteExtensionPluginName(),
      pluginTensorExtLite::getTensorExtLiteMetadataEmissionKind(), exportTensorExtLiteMetadataArtifact,
      /*requiredRuntimeABIParameters=*/{},
      /*directHelperRoute=*/false, pluginTensorExtLite::getTensorExtLiteExpectedHandoffKind(),
      validateTensorExtLiteMetadataCandidate,
      /*componentGroup=*/{}, /*externalABIName=*/{},
      buildTensorExtLiteMetadataArtifactRouteMetadata()));
}

llvm::Error registerTensorExtLiteMetadataArtifactPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginTensorExtLite::getTensorExtLiteExtensionPluginName(),
      registerTensorExtLiteMetadataArtifactTargetExporters));
}

} // namespace tianchenrv::target::tensorext_lite
