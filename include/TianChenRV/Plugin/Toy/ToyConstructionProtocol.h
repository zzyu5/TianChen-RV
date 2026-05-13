#ifndef TIANCHENRV_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Plugin/ConstructionProtocol.h"

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
using ToyGeneratedOutputStep =
    tianchenrv::plugin::construction::GeneratedOutputStep;
using ToyGeneratedOutputRoute =
    tianchenrv::plugin::construction::GeneratedOutputRoute;

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
