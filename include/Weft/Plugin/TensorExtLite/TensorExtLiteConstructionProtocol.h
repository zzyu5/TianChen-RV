#ifndef WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H
#define WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H

#include "Weft/Plugin/ConstructionProtocol.h"
#include "Weft/Support/ArtifactMetadata.h"
#include "Weft/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace weft::exec {
class VariantOp;
class KernelOp;
} // namespace weft::exec

namespace weft::support {
class TargetCapabilitySet;
} // namespace weft::support

namespace weft::plugin::tensorext_lite {

/// Verifies that a selected TensorExtLite variant is legal to emit: capability
/// conformance (provider id/kind/properties) AND variant metadata-vs-manifest
/// conformance (incl. the `emitc_route_mapping` eligibility declaration). This
/// is the single legality authority shared by the extension plugin (which emits
/// the fail-closed diagnostic) and the typed-emission backend driver (which
/// declines on error so the plugin still owns the diagnostic) — keeping the
/// driver's convert-set equal to the plugin route-build's success-set.
llvm::Error verifyTensorExtLiteSelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities);

using TensorExtLiteConstructionSemanticRole =
    weft::plugin::construction::SemanticRole;
using TensorExtLiteConstructionFamilyDeclaration =
    weft::plugin::construction::FamilyDeclaration;
using TensorExtLiteConstructionEmitCMapping =
    weft::plugin::construction::EmitCMapping;
using TensorExtLiteConstructionManifest =
    weft::plugin::construction::Manifest;
using TensorExtLiteTypedRoleInterfaceRealization =
    weft::plugin::construction::TypedRoleInterfaceRealization;
using TensorExtLiteTypedRoleGraphRealization =
    weft::plugin::construction::TypedRoleGraphRealization;
using TensorExtLiteFragmentMmaRoleStep =
    weft::plugin::construction::ExecutableRoleStep;

struct TensorExtLiteFragmentMmaEmitCConstructionRoute {
  llvm::StringRef routeID;
  llvm::StringRef emissionKind;
  llvm::StringRef artifactKind;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef loweringBoundaryOpName;
  llvm::StringRef headerRouteID;
  llvm::StringRef headerArtifactKind;
  llvm::StringRef bundleComponentGroup;
  llvm::StringRef objectHandoffKind;
  llvm::StringRef emitCToCppTranslateRouteID;
  llvm::StringRef configCallee;
  llvm::StringRef loadFragCallee;
  llvm::StringRef tileMmaCallee;
  llvm::StringRef storeFragCallee;
};

llvm::StringRef getTensorExtLiteConstructionInterfaceRealization();
llvm::StringRef getTensorExtLiteTypedRoleRealizationSummary();
llvm::StringRef getTensorExtLiteFragmentMmaSourceOps();
llvm::StringRef getTensorExtLiteFragmentMmaSourceRoles();
llvm::StringRef getTensorExtLiteEmitCLowerableOpInterfaceName();

llvm::StringRef getTensorExtLiteEmitCLowerableRouteMetadataName();
llvm::StringRef getTensorExtLiteRoleSequenceMetadataName();
llvm::StringRef getTensorExtLiteSourceOpsMetadataName();
llvm::StringRef getTensorExtLiteSourceRolesMetadataName();
llvm::StringRef getTensorExtLiteSourceOpInterfaceMetadataName();
llvm::StringRef getTensorExtLiteConstructionProtocolMetadataName();
llvm::StringRef getTensorExtLiteConstructionArchetypeMetadataName();
llvm::StringRef getTensorExtLiteSemanticRoleGraphMetadataName();
llvm::StringRef getTensorExtLiteCommonInterfaceRealizationMetadataName();
llvm::StringRef getTensorExtLiteTypedRoleRealizationMetadataName();
llvm::StringRef getTensorExtLiteEmitCRouteMappingMetadataName();
llvm::StringRef getTensorExtLiteEvidenceProfileMetadataName();

llvm::StringRef getTensorExtLiteConstructionProtocolMetadataRole();
llvm::StringRef getTensorExtLiteConstructionArchetypeMetadataRole();
llvm::StringRef getTensorExtLiteSemanticRoleGraphMetadataRole();
llvm::StringRef getTensorExtLiteCommonInterfaceRealizationMetadataRole();
llvm::StringRef getTensorExtLiteTypedRoleRealizationMetadataRole();
llvm::StringRef getTensorExtLiteEmitCRouteMappingMetadataRole();
llvm::StringRef getTensorExtLiteEvidenceProfileMetadataRole();

const TensorExtLiteConstructionManifest &getTensorExtLiteConstructionManifest();
const TensorExtLiteTypedRoleGraphRealization &getTensorExtLiteTypedRoleGraphRealization();
const TensorExtLiteFragmentMmaEmitCConstructionRoute
    &getTensorExtLiteFragmentMmaEmitCConstructionRoute();
llvm::ArrayRef<TensorExtLiteFragmentMmaRoleStep>
getTensorExtLiteFragmentMmaRoleSteps();
llvm::ArrayRef<weft::support::ArtifactMetadataEntry>
getTensorExtLiteFragmentMmaArtifactMetadata();
llvm::ArrayRef<weft::support::RuntimeABIParameter>
getTensorExtLiteFragmentMmaRuntimeABIParameters();

llvm::Error
verifyTensorExtLiteConstructionManifest(const TensorExtLiteConstructionManifest &manifest);
llvm::Error verifyTensorExtLiteTypedRoleGraphRealization(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization);
llvm::Error verifyTensorExtLiteConstructionProtocolReady();
llvm::Error verifyTensorExtLiteFragmentMmaEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef runtimeABI,
    llvm::StringRef runtimeABIKind, llvm::StringRef runtimeABIName,
    llvm::StringRef runtimeGlueRole);
llvm::Error verifyTensorExtLiteFragmentMmaTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID);
llvm::Error verifyTensorExtLiteFragmentMmaArtifactMetadata(
    llvm::ArrayRef<weft::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);
llvm::Error verifyTensorExtLiteRoleOpInterface(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization,
    mlir::Operation *roleOp, llvm::StringRef sourceRole);
llvm::Error verifyTensorExtLiteComputeRoleOpInterface(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);

} // namespace weft::plugin::tensorext_lite

#endif // WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H
