#ifndef TIANCHENRV_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Plugin/ConstructionProtocol.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace tianchenrv::plugin::toy {

using ToyConstructionSemanticRole =
    tianchenrv::plugin::construction::SemanticRole;
using ToyConstructionFamilyDeclaration =
    tianchenrv::plugin::construction::FamilyDeclaration;
using ToyConstructionEmitCMapping =
    tianchenrv::plugin::construction::EmitCMapping;
using ToyConstructionManifest = tianchenrv::plugin::construction::Manifest;
using ToyTypedRoleInterfaceRealization =
    tianchenrv::plugin::construction::TypedRoleInterfaceRealization;
using ToyTypedRoleGraphRealization =
    tianchenrv::plugin::construction::TypedRoleGraphRealization;

struct ToyTemplateEmitCConstructionRoute {
  llvm::StringRef routeID;
  llvm::StringRef emissionKind;
  llvm::StringRef artifactKind;
  llvm::StringRef loweringBoundaryOpName;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef headerRouteID;
  llvm::StringRef headerArtifactKind;
  llvm::StringRef bundleComponentGroup;
  llvm::StringRef objectHandoffKind;
  llvm::StringRef callee;
  llvm::StringRef resultName;
  llvm::StringRef resultCType;
};

llvm::StringRef getToyConstructionInterfaceRealization();
llvm::StringRef getToyTypedRoleRealizationSummary();

llvm::StringRef getToyConstructionProtocolMetadataName();
llvm::StringRef getToyConstructionArchetypeMetadataName();
llvm::StringRef getToySemanticRoleGraphMetadataName();
llvm::StringRef getToyCommonInterfaceRealizationMetadataName();
llvm::StringRef getToyTypedRoleRealizationMetadataName();
llvm::StringRef getToyEmitCRouteMappingMetadataName();
llvm::StringRef getToyEvidenceProfileMetadataName();

llvm::StringRef getToyConstructionProtocolMetadataRole();
llvm::StringRef getToyConstructionArchetypeMetadataRole();
llvm::StringRef getToySemanticRoleGraphMetadataRole();
llvm::StringRef getToyCommonInterfaceRealizationMetadataRole();
llvm::StringRef getToyTypedRoleRealizationMetadataRole();
llvm::StringRef getToyEmitCRouteMappingMetadataRole();
llvm::StringRef getToyEvidenceProfileMetadataRole();

const ToyConstructionManifest &getToyConstructionManifest();
const ToyTypedRoleGraphRealization &getToyTypedRoleGraphRealization();
const ToyTemplateEmitCConstructionRoute
    &getToyTemplateEmitCConstructionRoute();
llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry>
getToyTemplateConstructionArtifactMetadata();
llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter>
getToyTemplateRuntimeABIParameters();

llvm::Error
verifyToyConstructionManifest(const ToyConstructionManifest &manifest);
llvm::Error verifyToyTypedRoleGraphRealization(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization);
llvm::Error verifyToyConstructionProtocolReady();
llvm::Error verifyToyTemplateEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef loweringBoundaryOpName,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole);
llvm::Error verifyToyTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind);
llvm::Error verifyToyTemplateConstructionArtifactMetadata(
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);
llvm::Error verifyToyComputeRoleOpInterface(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);

} // namespace tianchenrv::plugin::toy

#endif // TIANCHENRV_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H
