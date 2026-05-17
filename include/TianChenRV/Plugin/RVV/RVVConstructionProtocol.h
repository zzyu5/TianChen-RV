#ifndef TIANCHENRV_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Plugin/ConstructionProtocol.h"
#include "TianChenRV/Support/ArtifactMetadata.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace tianchenrv::plugin::rvv {

using RVVConstructionSemanticRole =
    tianchenrv::plugin::construction::SemanticRole;
using RVVConstructionFamilyDeclaration =
    tianchenrv::plugin::construction::FamilyDeclaration;
using RVVConstructionEmitCMapping =
    tianchenrv::plugin::construction::EmitCMapping;
using RVVConstructionManifest = tianchenrv::plugin::construction::Manifest;
using RVVTypedRoleInterfaceRealization =
    tianchenrv::plugin::construction::TypedRoleInterfaceRealization;
using RVVTypedRoleGraphRealization =
    tianchenrv::plugin::construction::TypedRoleGraphRealization;
using RVVI32M1ArithmeticExecutableRoleStep =
    tianchenrv::plugin::construction::ExecutableRoleStep;

struct RVVI32M1ArithmeticConstructionRoute {
  llvm::StringRef mnemonic;
  llvm::StringRef operationName;
  llvm::StringRef typedRoleID;
  llvm::StringRef emitCRouteID;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeABIContractName;
};

struct RVVI32M1ArithmeticTargetArtifactMapping {
  llvm::StringRef headerRouteID;
  llvm::StringRef headerArtifactKind;
  llvm::StringRef bundleComponentGroup;
  llvm::StringRef objectHandoffKind;
  llvm::StringRef emitCToCppTranslateRouteID;
};

llvm::StringRef getRVVConstructionProtocolVersion();
llvm::StringRef getRVVConstructionArchetype();
llvm::StringRef getRVVConstructionSemanticRoleGraph();
llvm::StringRef getRVVConstructionInterfaceRealization();
llvm::StringRef getRVVTypedRoleRealizationSummary();
llvm::StringRef getRVVConstructionArtifactInterfaceRealization();
llvm::StringRef getRVVArtifactTypedRoleRealizationSummary();
llvm::StringRef getRVVConstructionEvidenceProfile();
llvm::StringRef getRVVI32M1ArithmeticSourceOps();
llvm::StringRef getRVVI32M1ArithmeticSourceRoles();
llvm::StringRef getRVVEmitCLowerableOpInterfaceName();

llvm::StringRef getRVVEmitCLowerableRouteMetadataName();
llvm::StringRef getRVVArithmeticOpMetadataName();
llvm::StringRef getRVVSourceOpsMetadataName();
llvm::StringRef getRVVSourceRolesMetadataName();
llvm::StringRef getRVVSourceOpInterfaceMetadataName();
llvm::StringRef getRVVConstructionProtocolMetadataName();
llvm::StringRef getRVVConstructionArchetypeMetadataName();
llvm::StringRef getRVVSemanticRoleGraphMetadataName();
llvm::StringRef getRVVCommonInterfaceRealizationMetadataName();
llvm::StringRef getRVVTypedRoleRealizationMetadataName();
llvm::StringRef getRVVEmitCRouteMappingMetadataName();
llvm::StringRef getRVVEvidenceProfileMetadataName();
llvm::StringRef getRVVRuntimeABIContractMetadataName();
llvm::StringRef getRVVBundleComponentGroupMetadataName();
llvm::StringRef getRVVObjectHandoffMetadataName();
llvm::StringRef getRVVSourceKernelAttrName();
llvm::StringRef getRVVSelectedVariantAttrName();
llvm::StringRef getRVVOriginAttrName();
llvm::StringRef getRVVSelectedPathRoleAttrName();
llvm::StringRef getRVVStatusAttrName();
llvm::StringRef getRVVRequiredCapabilitiesAttrName();
llvm::StringRef getRVVTypedRoleAttrName();
llvm::StringRef getRVVRoleOrderAttrName();
llvm::StringRef getRVVSourceRoleAttrName();
llvm::StringRef getRVVRoleSpecificInterfaceAttrName();
llvm::StringRef getRVVRoleOpBoundaryStatus();
llvm::StringRef getRVVLoweringBoundaryStatus();

const RVVConstructionManifest &getRVVConstructionManifest();
const RVVTypedRoleGraphRealization &getRVVTypedRoleGraphRealization();
llvm::ArrayRef<RVVI32M1ArithmeticConstructionRoute>
getRVVI32M1ArithmeticConstructionRoutes();
const RVVI32M1ArithmeticTargetArtifactMapping &
getRVVI32M1ArithmeticTargetArtifactMapping();

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVI32M1ArithmeticConstructionRuntimeABIParameters();
llvm::Expected<llvm::SmallVector<tianchenrv::support::ArtifactMetadataEntry, 16>>
getRVVI32M1ArithmeticConstructionArtifactMetadata(
    llvm::StringRef emitCRouteID);
llvm::Expected<llvm::SmallVector<RVVI32M1ArithmeticExecutableRoleStep, 10>>
getRVVI32M1ArithmeticExecutableRoleSteps(llvm::StringRef operationName);

llvm::Error verifyRVVConstructionManifest(
    const RVVConstructionManifest &manifest);
llvm::Error verifyRVVTypedRoleGraphRealization(
    const RVVConstructionManifest &manifest,
    const RVVTypedRoleGraphRealization &realization);
llvm::Error verifyRVVConstructionProtocolReady();
llvm::Error verifyRVVI32M1ArithmeticConstructionRuntimeABIParameters(
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter> parameters);
llvm::Error verifyRVVI32M1ArithmeticTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID);
llvm::Error verifyRVVI32M1ArithmeticConstructionArtifactMetadata(
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);
llvm::Error verifyRVVI32M1ArithmeticSelectedRoleSequence(
    llvm::ArrayRef<mlir::Operation *> orderedRoleOperations,
    llvm::ArrayRef<unsigned> orderedRoleOperationOrders,
    llvm::StringRef selectedVariantSymbol, llvm::StringRef pathRole,
    llvm::StringRef operationName, llvm::StringRef context);

llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByMnemonic(llvm::StringRef mnemonic);
llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByOperationName(
    llvm::StringRef operationName);
llvm::Expected<const RVVI32M1ArithmeticConstructionRoute *>
lookupRVVI32M1ArithmeticConstructionRouteByEmitCRouteID(
    llvm::StringRef emitCRouteID);

llvm::Error verifyRVVRoleOperationInterface(mlir::Operation *roleOp,
                                            llvm::StringRef role);
llvm::Error verifyRVVRuntimeABIValueRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVSetVLRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVWithVLRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVLoadRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVArithmeticRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVStoreRoleOpInterface(mlir::Operation *roleOp);

llvm::Error verifyRVVI32M1ArithmeticConstructionRouteMapping(
    llvm::StringRef mnemonic, llvm::StringRef operationName,
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName);
llvm::Error verifyRVVI32M1ArithmeticConstructionPlanMapping(
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName,
    llvm::StringRef emissionKind,
    llvm::StringRef loweringBoundaryOpName, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeGlueRole);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H
