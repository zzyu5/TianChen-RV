#ifndef WEFT_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H
#define WEFT_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H

#include "Weft/Plugin/ConstructionProtocol.h"
#include "Weft/Support/ArtifactMetadata.h"
#include "Weft/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace weft::plugin::rvv {

// One descriptor per N-operand product-reduction contraction route (defined in
// RVVContractionRouteIdentity.h). The role-sequence verifier takes it by pointer
// so the N runtime_abi + N load role steps + their orders derive from the route's
// arity/c-names, not from per-route boolean gates.
struct ContractionRouteIdentity;

using RVVConstructionSemanticRole =
    weft::plugin::construction::SemanticRole;
using RVVConstructionFamilyDeclaration =
    weft::plugin::construction::FamilyDeclaration;
using RVVConstructionEmitCMapping =
    weft::plugin::construction::EmitCMapping;
using RVVConstructionManifest = weft::plugin::construction::Manifest;
using RVVTypedRoleInterfaceRealization =
    weft::plugin::construction::TypedRoleInterfaceRealization;
using RVVTypedRoleGraphRealization =
    weft::plugin::construction::TypedRoleGraphRealization;
using RVVSelectedBodyExecutableRoleStep =
    weft::plugin::construction::ExecutableRoleStep;

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
  llvm::ArrayRef<weft::support::RuntimeABIParameter>
      runtimeABIParameters;
};

llvm::StringRef getRVVConstructionProtocolVersion();
llvm::StringRef getRVVConstructionArtifactInterfaceRealization();
llvm::StringRef getRVVArtifactTypedRoleRealizationSummary();
llvm::StringRef getRVVSelectedBodySourceOps();
llvm::StringRef getRVVSelectedBodySourceRoles();
llvm::StringRef getRVVEmitCLowerableOpInterfaceName();

llvm::StringRef getRVVEmitCLowerableRouteMetadataName();
llvm::StringRef getRVVSelectedBodyOperationMetadataName();
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
llvm::StringRef getRVVBundleComponentGroupMetadataName();
llvm::StringRef getRVVObjectHandoffMetadataName();
llvm::StringRef getRVVSourceKernelAttrName();
llvm::StringRef getRVVSelectedVariantAttrName();
llvm::StringRef getRVVOriginAttrName();
llvm::StringRef getRVVSelectedPathRoleAttrName();
llvm::StringRef getRVVStatusAttrName();
llvm::StringRef getRVVRequiredCapabilitiesAttrName();
llvm::StringRef getRVVLoweringBoundaryStatus();

const RVVConstructionManifest &getRVVConstructionManifest();
const RVVTypedRoleGraphRealization &getRVVTypedRoleGraphRealization();
llvm::StringRef getRVVSelectedBodyTargetArtifactRouteID();
llvm::StringRef getRVVSelectedBodyTargetArtifactKind();
llvm::ArrayRef<RVVSelectedBodyConstructionRoute>
getRVVSelectedBodyConstructionRoutes();
const RVVSelectedBodyTargetArtifactMapping &
getRVVSelectedBodyTargetArtifactMapping();

llvm::SmallVector<weft::support::RuntimeABIParameter, 4>
getRVVSelectedBodyConstructionRuntimeABIParameters();
llvm::Expected<llvm::SmallVector<weft::support::ArtifactMetadataEntry, 16>>
getRVVSelectedBodyConstructionArtifactMetadata(
    const RVVSelectedBodyConstructionMetadataFacts &facts);
llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
getRVVSelectedBodyExecutableRoleSteps(llvm::StringRef typedComputeOpName);
llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
getRVVSelectedBodyExecutableRoleSteps(llvm::StringRef typedComputeOpName,
                                      llvm::StringRef rhsSourceOperationName);
llvm::Expected<llvm::SmallVector<RVVSelectedBodyExecutableRoleStep, 10>>
getRVVSelectedBodyExecutableRoleSteps(llvm::StringRef operationMnemonic,
                                      llvm::StringRef typedComputeOpName,
                                      llvm::StringRef rhsSourceOperationName);

llvm::Error verifyRVVConstructionManifest(
    const RVVConstructionManifest &manifest);
llvm::Error verifyRVVTypedRoleGraphRealization(
    const RVVConstructionManifest &manifest,
    const RVVTypedRoleGraphRealization &realization);
llvm::Error verifyRVVConstructionProtocolReady();
llvm::Error verifyRVVSelectedBodyConstructionRuntimeABIParameters(
    llvm::ArrayRef<weft::support::RuntimeABIParameter> parameters);
llvm::Error verifyRVVSelectedBodyTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID);
llvm::Error verifyRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyConstructionMetadataFacts &facts,
    llvm::StringRef context);
llvm::Error verifyRVVSelectedBodyConstructionArtifactMetadata(
    llvm::ArrayRef<weft::support::ArtifactMetadataEntry> metadata,
    const RVVSelectedBodyConstructionMetadataFacts &facts,
    llvm::StringRef context);
llvm::Error verifyRVVSelectedBodySelectedRoleSequence(
    llvm::ArrayRef<mlir::Operation *> orderedRoleOperations,
    llvm::ArrayRef<unsigned> orderedRoleOperationOrders,
    llvm::StringRef selectedVariantSymbol, llvm::StringRef pathRole,
    llvm::StringRef operationMnemonic, llvm::StringRef typedComputeOpName,
    llvm::StringRef rhsSourceOperationName,
    llvm::StringRef context,
    const ContractionRouteIdentity *productRouteIdentity = nullptr);

llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
    llvm::StringRef operationMnemonic);
llvm::Expected<const RVVSelectedBodyConstructionRoute *>
lookupRVVSelectedBodyConstructionRouteByEmitCRouteID(
    llvm::StringRef emitCRouteID);

llvm::Error verifyRVVRoleOperationInterface(mlir::Operation *roleOp,
                                            llvm::StringRef role);
llvm::Error verifyRVVRuntimeABIValueRoleOpInterface(mlir::Operation *roleOp);

llvm::Error verifyRVVSelectedBodyConstructionRouteMapping(
    llvm::StringRef operationMnemonic, llvm::StringRef typedComputeOpName,
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName);
llvm::Error verifyRVVSelectedBodyConstructionPlanMapping(
    llvm::StringRef emitCRouteID, llvm::StringRef runtimeABIName,
    llvm::StringRef emissionKind,
    llvm::StringRef loweringBoundaryOpName, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeGlueRole);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVCONSTRUCTIONPROTOCOL_H
