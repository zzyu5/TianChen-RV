#ifndef TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Plugin/ConstructionProtocol.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace tianchenrv::plugin::template_ext {

using TemplateConstructionSemanticRole =
    tianchenrv::plugin::construction::SemanticRole;
using TemplateConstructionFamilyDeclaration =
    tianchenrv::plugin::construction::FamilyDeclaration;
using TemplateConstructionEmitCMapping =
    tianchenrv::plugin::construction::EmitCMapping;
using TemplateConstructionManifest = tianchenrv::plugin::construction::Manifest;
using TemplateTypedRoleInterfaceRealization =
    tianchenrv::plugin::construction::TypedRoleInterfaceRealization;
using TemplateTypedRoleGraphRealization =
    tianchenrv::plugin::construction::TypedRoleGraphRealization;

struct TemplateEmitCConstructionRoute {
  llvm::StringRef routeID;
  llvm::StringRef emissionKind;
  llvm::StringRef artifactKind;
  llvm::StringRef loweringBoundaryOpName;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef callee;
  llvm::StringRef resultName;
  llvm::StringRef resultCType;
  llvm::StringRef emitCToCppTranslateRouteID;
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
llvm::StringRef getTemplateSourceOpMetadataName();
llvm::StringRef getTemplateSourceRoleMetadataName();
llvm::StringRef getTemplateSourceOpInterfaceMetadataName();

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
const TemplateEmitCConstructionRoute &getTemplateEmitCConstructionRoute();
llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter>
getTemplateRuntimeABIParameters();
llvm::Error
verifyTemplateConstructionManifest(const TemplateConstructionManifest &manifest);
llvm::Error verifyTemplateTypedRoleGraphRealization(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization);
llvm::Error verifyTemplateConstructionProtocolReady();
llvm::Error verifyTemplateEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef loweringBoundaryOpName,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole);
llvm::Error verifyTemplateComputeRoleOpInterface(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);

} // namespace tianchenrv::plugin::template_ext

#endif // TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
