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
using RVVSelectedBodyExecutableRoleStep =
    tianchenrv::plugin::construction::ExecutableRoleStep;

struct RVVSelectedBodyConstructionRoute {
  llvm::StringRef operationMnemonic;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef typedRoleID;
  llvm::StringRef emitCRouteID;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeABIContractName;
};

struct RVVSelectedBodyTargetArtifactMapping {
  llvm::StringRef headerRouteID;
  llvm::StringRef headerArtifactKind;
  llvm::StringRef bundleComponentGroup;
  llvm::StringRef objectHandoffKind;
  llvm::StringRef emitCToCppTranslateRouteID;
};

struct RVVSelectedBodyConstructionMetadataFacts {
  llvm::StringRef operationMnemonic;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef emitCRouteID;
  llvm::StringRef targetArtifactRouteID;
  llvm::StringRef targetArtifactKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeABIContractName;
  llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter>
      runtimeABIParameters;
};

llvm::StringRef getRVVConstructionProtocolVersion();
llvm::StringRef getRVVConstructionArchetype();
llvm::StringRef getRVVConstructionSemanticRoleGraph();
llvm::StringRef getRVVConstructionInterfaceRealization();
llvm::StringRef getRVVTypedRoleRealizationSummary();
llvm::StringRef getRVVConstructionArtifactInterfaceRealization();
llvm::StringRef getRVVArtifactTypedRoleRealizationSummary();
llvm::StringRef getRVVConstructionEvidenceProfile();
llvm::StringRef getRVVSelectedBodySourceOps();
llvm::StringRef getRVVSelectedBodySourceRoles();
llvm::StringRef getRVVEmitCLowerableOpInterfaceName();

llvm::StringRef getRVVEmitCLowerableRouteMetadataName();
llvm::StringRef getRVVSelectedBodyOperationMetadataName();
llvm::StringRef getRVVSelectedBodyTypedComputeOpMetadataName();
llvm::StringRef getRVVSourceOpsMetadataName();
llvm::StringRef getRVVSourceRolesMetadataName();
llvm::StringRef getRVVSourceOpInterfaceMetadataName();
llvm::StringRef getRVVConstructionProtocolMetadataName();
llvm::StringRef getRVVConstructionArchetypeMetadataName();
llvm::StringRef getRVVSemanticRoleGraphMetadataName();
llvm::StringRef getRVVCommonInterfaceRealizationMetadataName();
llvm::StringRef getRVVTypedRoleRealizationMetadataName();
llvm::StringRef getRVVEmitCRouteMappingMetadataName();
llvm::StringRef getRVVTargetArtifactRouteMetadataName();
llvm::StringRef getRVVTargetArtifactKindMetadataName();
llvm::StringRef getRVVEvidenceProfileMetadataName();
llvm::StringRef getRVVRuntimeABINameMetadataName();
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
llvm::ArrayRef<RVVSelectedBodyConstructionRoute>
getRVVSelectedBodyConstructionRoutes();
const RVVSelectedBodyTargetArtifactMapping &
getRVVSelectedBodyTargetArtifactMapping();

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyConstructionRuntimeABIParameters();
llvm::Expected<llvm::SmallVector<tianchenrv::support::ArtifactMetadataEntry, 16>>
getRVVSelectedBodyConstructionArtifactMetadata(
    const RVVSelectedBodyConstructionMetadataFacts &facts);
llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
getRVVSelectedBodyExecutableRoleSteps(llvm::StringRef typedComputeOpName);
llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
getRVVSelectedBodyExecutableRoleSteps(llvm::StringRef typedComputeOpName,
                                      llvm::StringRef rhsSourceOperationName);

llvm::Error verifyRVVConstructionManifest(
    const RVVConstructionManifest &manifest);
llvm::Error verifyRVVTypedRoleGraphRealization(
    const RVVConstructionManifest &manifest,
    const RVVTypedRoleGraphRealization &realization);
llvm::Error verifyRVVConstructionProtocolReady();
llvm::Error verifyRVVSelectedBodyConstructionRuntimeABIParameters(
    llvm::ArrayRef<tianchenrv::support::RuntimeABIParameter> parameters);
llvm::Error verifyRVVSelectedBodyTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID);
llvm::Error verifyRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyConstructionMetadataFacts &facts,
    llvm::StringRef context);
llvm::Error verifyRVVSelectedBodyConstructionArtifactMetadata(
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata,
    const RVVSelectedBodyConstructionMetadataFacts &facts,
    llvm::StringRef context);
llvm::Error verifyRVVSelectedBodySelectedRoleSequence(
    llvm::ArrayRef<mlir::Operation *> orderedRoleOperations,
    llvm::ArrayRef<unsigned> orderedRoleOperationOrders,
    llvm::StringRef selectedVariantSymbol, llvm::StringRef pathRole,
    llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName,
    llvm::StringRef context);

llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
    llvm::StringRef operationMnemonic);
llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRVVSelectedBodyConstructionRouteByTypedComputeOpName(
    llvm::StringRef typedComputeOpName);
llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRVVSelectedBodyConstructionRouteByEmitCRouteID(
    llvm::StringRef emitCRouteID);

llvm::Error verifyRVVRoleOperationInterface(mlir::Operation *roleOp,
                                            llvm::StringRef role);
llvm::Error verifyRVVRuntimeABIValueRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVSetVLRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVWithVLRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVLoadRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVComputeRoleOpInterface(mlir::Operation *roleOp);
llvm::Error verifyRVVStoreRoleOpInterface(mlir::Operation *roleOp);

llvm::Error verifyRVVSelectedBodyConstructionRouteMapping(
    llvm::StringRef operationMnemonic, llvm::StringRef typedComputeOpName,
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName);
llvm::Error verifyRVVSelectedBodyConstructionPlanMapping(
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName,
    llvm::StringRef emissionKind,
    llvm::StringRef loweringBoundaryOpName, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeGlueRole);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H
