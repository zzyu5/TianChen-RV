#ifndef TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace tianchenrv::plugin::template_ext {

struct TemplateConstructionSemanticRole {
  llvm::StringRef role;
  unsigned order = 0;
  llvm::StringRef operationName;
  llvm::StringRef commonInterfaces;
  llvm::StringRef description;
};

struct TemplateConstructionFamilyDeclaration {
  llvm::StringRef familyName;
  llvm::StringRef architecturalNamespace;
  llvm::StringRef concreteNamespace;
  llvm::StringRef pluginName;
  llvm::StringRef capabilityID;
  llvm::StringRef capabilityKind;
  llvm::StringRef firstSliceVariantName;
};

struct TemplateConstructionEmitCMapping {
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

struct TemplateConstructionManifest {
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  TemplateConstructionFamilyDeclaration family;
  llvm::ArrayRef<TemplateConstructionSemanticRole> semanticRoles;
  TemplateConstructionEmitCMapping emitcRoute;
  llvm::StringRef evidenceProfile;
};

struct TemplateTypedRoleInterfaceRealization {
  llvm::StringRef typedRoleID;
  llvm::StringRef role;
  unsigned order = 0;
  llvm::StringRef operationName;
  llvm::StringRef commonInterfaces;
  llvm::StringRef roleSpecificInterface;
  llvm::StringRef emitCLowerableInterface;
  llvm::StringRef emitCCall;
};

struct TemplateTypedRoleGraphRealization {
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  llvm::StringRef familyName;
  llvm::StringRef realizationSummary;
  llvm::ArrayRef<TemplateTypedRoleInterfaceRealization> roles;
  llvm::StringRef evidenceProfile;
};

struct TemplateGeneratedOutputStep {
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

struct TemplateGeneratedOutputRoute {
  std::string functionName;
  std::string requiredHeader;
  std::vector<TemplateGeneratedOutputStep> steps;
};

llvm::StringRef getTemplateConstructionProtocolVersion();
llvm::StringRef getTemplateConstructionArchetype();
llvm::StringRef getTemplateConstructionSemanticRoleGraph();
llvm::StringRef getTemplateConstructionInterfaceRealization();
llvm::StringRef getTemplateTypedRoleRealizationSummary();
llvm::StringRef getTemplateConstructionEvidenceProfile();

llvm::StringRef getTemplateConstructionProtocolMetadataName();
llvm::StringRef getTemplateConstructionArchetypeMetadataName();
llvm::StringRef getTemplateSemanticRoleGraphMetadataName();
llvm::StringRef getTemplateCommonInterfaceRealizationMetadataName();
llvm::StringRef getTemplateTypedRoleRealizationMetadataName();
llvm::StringRef getTemplateEmitCRouteMappingMetadataName();
llvm::StringRef getTemplateEvidenceProfileMetadataName();

llvm::StringRef getTemplateConstructionProtocolMetadataRole();
llvm::StringRef getTemplateConstructionArchetypeMetadataRole();
llvm::StringRef getTemplateSemanticRoleGraphMetadataRole();
llvm::StringRef getTemplateCommonInterfaceRealizationMetadataRole();
llvm::StringRef getTemplateTypedRoleRealizationMetadataRole();
llvm::StringRef getTemplateEmitCRouteMappingMetadataRole();
llvm::StringRef getTemplateEvidenceProfileMetadataRole();

const TemplateConstructionManifest &getTemplateConstructionManifest();
const TemplateTypedRoleGraphRealization &
getTemplateTypedRoleGraphRealization();
llvm::Error
verifyTemplateConstructionManifest(const TemplateConstructionManifest &manifest);
llvm::Error verifyTemplateTypedRoleGraphRealization(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization);
llvm::Error verifyTemplateComputeRoleOpInterface(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);
llvm::Expected<TemplateGeneratedOutputRoute>
buildTemplateGeneratedOutputRoute(const TemplateConstructionManifest &manifest);
llvm::Expected<TemplateGeneratedOutputRoute> buildTemplateGeneratedOutputRoute(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization);

} // namespace tianchenrv::plugin::template_ext

#endif // TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
