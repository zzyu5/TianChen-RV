#ifndef TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace tianchenrv::plugin::tensorext_lite {

struct TensorExtLiteConstructionSemanticRole {
  llvm::StringRef role;
  unsigned order = 0;
  llvm::StringRef operationName;
  llvm::StringRef commonInterfaces;
  llvm::StringRef description;
};

struct TensorExtLiteConstructionFamilyDeclaration {
  llvm::StringRef familyName;
  llvm::StringRef architecturalNamespace;
  llvm::StringRef concreteNamespace;
  llvm::StringRef pluginName;
  llvm::StringRef capabilityID;
  llvm::StringRef capabilityKind;
  llvm::StringRef firstSliceVariantName;
};

struct TensorExtLiteConstructionEmitCMapping {
  llvm::StringRef routeID;
  llvm::StringRef emissionKind;
  llvm::StringRef artifactKind;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef requiredHeader;
  llvm::StringRef roleToCallMap;
};

struct TensorExtLiteConstructionManifest {
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  TensorExtLiteConstructionFamilyDeclaration family;
  llvm::ArrayRef<TensorExtLiteConstructionSemanticRole> semanticRoles;
  TensorExtLiteConstructionEmitCMapping emitcRoute;
  llvm::StringRef evidenceProfile;
};

struct TensorExtLiteTypedRoleInterfaceRealization {
  llvm::StringRef typedRoleID;
  llvm::StringRef role;
  unsigned order = 0;
  llvm::StringRef operationName;
  llvm::StringRef commonInterfaces;
  llvm::StringRef roleSpecificInterface;
  llvm::StringRef emitCLowerableInterface;
  llvm::StringRef emitCCall;
};

struct TensorExtLiteTypedRoleGraphRealization {
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  llvm::StringRef familyName;
  llvm::StringRef realizationSummary;
  llvm::ArrayRef<TensorExtLiteTypedRoleInterfaceRealization> roles;
  llvm::StringRef evidenceProfile;
};

struct TensorExtLiteGeneratedOutputStep {
  std::string typedRoleID;
  std::string role;
  unsigned order = 0;
  std::string operationName;
  std::string commonInterfaces;
  std::string roleSpecificInterface;
  std::string emitCLowerableInterface;
  std::string emitCCall;
  std::string sourceLine;
};

struct TensorExtLiteGeneratedOutputRoute {
  std::string functionName;
  std::string requiredHeader;
  std::vector<TensorExtLiteGeneratedOutputStep> steps;
};

llvm::StringRef getTensorExtLiteConstructionInterfaceRealization();
llvm::StringRef getTensorExtLiteTypedRoleRealizationSummary();

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

llvm::Error
verifyTensorExtLiteConstructionManifest(const TensorExtLiteConstructionManifest &manifest);
llvm::Error verifyTensorExtLiteTypedRoleGraphRealization(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization);
llvm::Error verifyTensorExtLiteComputeRoleOpInterface(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);
llvm::Expected<TensorExtLiteGeneratedOutputRoute>
buildTensorExtLiteGeneratedOutputRoute(const TensorExtLiteConstructionManifest &manifest);
llvm::Expected<TensorExtLiteGeneratedOutputRoute> buildTensorExtLiteGeneratedOutputRoute(
    const TensorExtLiteConstructionManifest &manifest,
    const TensorExtLiteTypedRoleGraphRealization &realization);

} // namespace tianchenrv::plugin::tensorext_lite

#endif // TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H
