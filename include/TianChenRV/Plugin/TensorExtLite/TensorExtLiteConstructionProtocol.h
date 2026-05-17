#ifndef TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Plugin/ConstructionProtocol.h"
#include "TianChenRV/Support/ArtifactMetadata.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace tianchenrv::plugin::tensorext_lite {

using TensorExtLiteConstructionSemanticRole =
    tianchenrv::plugin::construction::SemanticRole;
using TensorExtLiteConstructionFamilyDeclaration =
    tianchenrv::plugin::construction::FamilyDeclaration;
using TensorExtLiteConstructionEmitCMapping =
    tianchenrv::plugin::construction::EmitCMapping;
using TensorExtLiteConstructionManifest =
    tianchenrv::plugin::construction::Manifest;
using TensorExtLiteTypedRoleInterfaceRealization =
    tianchenrv::plugin::construction::TypedRoleInterfaceRealization;
using TensorExtLiteTypedRoleGraphRealization =
    tianchenrv::plugin::construction::TypedRoleGraphRealization;

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

struct TensorExtLiteFragmentMmaRoleStep {
  llvm::StringRef sourceRole;
  llvm::StringRef operationName;
  llvm::StringRef typedRoleID;
  llvm::StringRef roleSpecificInterface;
  llvm::StringRef emitCLowerableInterface;
  llvm::StringRef callee;
  unsigned order = 0;
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
llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry>
getTensorExtLiteFragmentMmaArtifactMetadata();
llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter>
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
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);
llvm::Error verifyTensorExtLiteRoleOpInterface(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization,
    mlir::Operation *roleOp, llvm::StringRef sourceRole);
llvm::Error verifyTensorExtLiteComputeRoleOpInterface(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);

} // namespace tianchenrv::plugin::tensorext_lite

#endif // TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H
