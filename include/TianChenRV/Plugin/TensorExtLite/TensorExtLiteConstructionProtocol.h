#ifndef TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Plugin/ConstructionProtocol.h"

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

} // namespace tianchenrv::plugin::tensorext_lite

#endif // TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITECONSTRUCTIONPROTOCOL_H
