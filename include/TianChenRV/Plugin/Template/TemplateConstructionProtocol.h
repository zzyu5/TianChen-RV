#ifndef TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Plugin/ConstructionProtocol.h"

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
using TemplateGeneratedOutputStep =
    tianchenrv::plugin::construction::GeneratedOutputStep;
using TemplateGeneratedOutputRoute =
    tianchenrv::plugin::construction::GeneratedOutputRoute;

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
