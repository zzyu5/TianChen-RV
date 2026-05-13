#ifndef TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

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

llvm::StringRef getTemplateConstructionProtocolVersion();
llvm::StringRef getTemplateConstructionArchetype();
llvm::StringRef getTemplateConstructionSemanticRoleGraph();
llvm::StringRef getTemplateConstructionInterfaceRealization();
llvm::StringRef getTemplateConstructionEvidenceProfile();

llvm::StringRef getTemplateConstructionProtocolMetadataName();
llvm::StringRef getTemplateConstructionArchetypeMetadataName();
llvm::StringRef getTemplateSemanticRoleGraphMetadataName();
llvm::StringRef getTemplateCommonInterfaceRealizationMetadataName();
llvm::StringRef getTemplateEmitCRouteMappingMetadataName();
llvm::StringRef getTemplateEvidenceProfileMetadataName();

llvm::StringRef getTemplateConstructionProtocolMetadataRole();
llvm::StringRef getTemplateConstructionArchetypeMetadataRole();
llvm::StringRef getTemplateSemanticRoleGraphMetadataRole();
llvm::StringRef getTemplateCommonInterfaceRealizationMetadataRole();
llvm::StringRef getTemplateEmitCRouteMappingMetadataRole();
llvm::StringRef getTemplateEvidenceProfileMetadataRole();

const TemplateConstructionManifest &getTemplateConstructionManifest();
llvm::Error
verifyTemplateConstructionManifest(const TemplateConstructionManifest &manifest);

} // namespace tianchenrv::plugin::template_ext

#endif // TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
