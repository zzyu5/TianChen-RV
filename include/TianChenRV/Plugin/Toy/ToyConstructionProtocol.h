#ifndef TIANCHENRV_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace tianchenrv::plugin::toy {

struct ToyConstructionSemanticRole {
  llvm::StringRef role;
  unsigned order = 0;
  llvm::StringRef operationName;
  llvm::StringRef commonInterfaces;
  llvm::StringRef description;
};

struct ToyConstructionFamilyDeclaration {
  llvm::StringRef familyName;
  llvm::StringRef architecturalNamespace;
  llvm::StringRef concreteNamespace;
  llvm::StringRef pluginName;
  llvm::StringRef capabilityID;
  llvm::StringRef capabilityKind;
  llvm::StringRef firstSliceVariantName;
};

struct ToyConstructionEmitCMapping {
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

struct ToyConstructionManifest {
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  ToyConstructionFamilyDeclaration family;
  llvm::ArrayRef<ToyConstructionSemanticRole> semanticRoles;
  ToyConstructionEmitCMapping emitcRoute;
  llvm::StringRef evidenceProfile;
};

struct ToyTypedRoleInterfaceRealization {
  llvm::StringRef typedRoleID;
  llvm::StringRef role;
  unsigned order = 0;
  llvm::StringRef operationName;
  llvm::StringRef commonInterfaces;
  llvm::StringRef roleSpecificInterface;
  llvm::StringRef emitCLowerableInterface;
  llvm::StringRef emitCCall;
};

struct ToyTypedRoleGraphRealization {
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  llvm::StringRef familyName;
  llvm::StringRef realizationSummary;
  llvm::ArrayRef<ToyTypedRoleInterfaceRealization> roles;
  llvm::StringRef evidenceProfile;
};

struct ToyGeneratedOutputStep {
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

struct ToyGeneratedOutputRoute {
  std::string functionName;
  std::string requiredHeader;
  std::vector<ToyGeneratedOutputStep> steps;
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

llvm::Error
verifyToyConstructionManifest(const ToyConstructionManifest &manifest);
llvm::Error verifyToyTypedRoleGraphRealization(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization);
llvm::Error verifyToyComputeRoleOpInterface(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);
llvm::Expected<ToyGeneratedOutputRoute>
buildToyGeneratedOutputRoute(const ToyConstructionManifest &manifest);
llvm::Expected<ToyGeneratedOutputRoute> buildToyGeneratedOutputRoute(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization);

} // namespace tianchenrv::plugin::toy

#endif // TIANCHENRV_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H
