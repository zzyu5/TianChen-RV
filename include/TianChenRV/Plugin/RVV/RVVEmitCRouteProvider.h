#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Support/ArtifactMetadata.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class VariantEmitCLowerableRequest;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::rvv {

enum class RVVSelectedBodyOperationKind {
  Add,
  Sub,
  Mul,
  CmpSelect,
  ComputedMaskSelect,
  ReduceAdd,
  StandaloneReduceAdd,
  MaskedAdd,
  MAccAdd,
  StridedAdd,
  StridedLoadUnitStore,
  UnitLoadStridedStore,
  IndexedGatherUnitStore,
  IndexedScatterUnitLoad,
  MaskedUnitLoadStore,
  MaskedUnitStore,
  ComputedMaskUnitLoadStore,
  ComputedMaskStridedStore,
  Segment2DeinterleaveUnitStore,
  Segment2InterleaveUnitLoad,
  ScalarBroadcastAdd,
  WidenI32ToI64,
  WidenI16ToI32,
  WideningMAccAdd,
  WideningDotReduceAdd,
  StridedInputWideningDotReduceAdd,
  ComputedMaskWideningDotReduceAdd,
  ComputedMaskStridedInputWideningDotReduceAdd,
};

enum class RVVSelectedBodyMemoryForm {
  VectorRHSLoad,
  RHSBroadcastLoad,
  RHSScalarBroadcast,
  StridedLoadStore,
  StridedLoadUnitStore,
  UnitLoadStridedStore,
  IndexedLoadUnitStore,
  UnitLoadIndexedStore,
  MaskedUnitLoadStore,
  MaskedUnitStore,
  ComputedMaskUnitLoadStore,
  ComputedMaskVectorSelect,
  ComputedMaskUnitLoadStridedStore,
  Segment2LoadUnitStore,
  UnitLoadSegment2Store,
  UnitStrideConversion,
  ComputedMaskUnitStrideWideningDotReduce,
  StridedInputWideningDotReduce,
  ComputedMaskStridedInputWideningDotReduce,
  UnitStrideStandaloneReduction,
};

struct RVVSelectedBodyEmitCRouteDescription {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef runtimeControlPlanID;
  llvm::StringRef configContractID;
  llvm::StringRef runtimeVLContractID;
  llvm::StringRef runtimeAVLASource;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlDefOpName;
  llvm::StringRef vlScopeOpName;
  llvm::StringRef vlUses;
  llvm::StringRef emitCLoopKind;
  llvm::StringRef emitCLoopInductionName;
  llvm::StringRef emitCFullChunkVLName;
  llvm::StringRef emitCLoopVLName;
  llvm::StringRef remainingAVLMetadata;
  llvm::StringRef pointerAdvanceMetadata;
  llvm::StringRef boundedSlice;
  llvm::StringRef multiVL;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef destSEW;
  llvm::StringRef destLMUL;
  llvm::StringRef conversionRelation;
  llvm::StringRef typedComputeOpName;
  llvm::StringRef boundaryOpName;
  llvm::StringRef emitCRouteID;
  llvm::StringRef targetArtifactRouteID;
  llvm::StringRef targetArtifactKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeABIContractName;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef maskTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef indexedLoadIntrinsic;
  llvm::StringRef indexedStoreIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
  llvm::StringRef intrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef reductionAccumulatorLayout;
  llvm::StringRef reductionResultLayout;
  llvm::StringRef reductionStoreVL;
  llvm::StringRef maccAccumulatorLayout;
  llvm::StringRef maccResultLayout;
  llvm::StringRef wideningMAccAccumulatorLayout;
  llvm::StringRef wideningMAccResultLayout;
  llvm::StringRef wideningMAccRelation;
  llvm::StringRef wideningDotProductAccumulatorLayout;
  llvm::StringRef wideningDotProductResultLayout;
  llvm::StringRef wideningDotProductRelation;
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef maskedWideningProductIntrinsic;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef inactiveLaneZeroingRequirement;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef segmentMemoryLayout;
  std::int64_t segmentCount = 0;
  llvm::StringRef segmentTupleCType;
  llvm::StringRef segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic;
  llvm::StringRef field0Role;
  llvm::StringRef field1Role;
  llvm::StringRef field0Name;
  llvm::StringRef field1Name;
  llvm::StringRef field0SourceMemoryForm;
  llvm::StringRef field1SourceMemoryForm;
  llvm::StringRef field0DestinationMemoryForm;
  llvm::StringRef field1DestinationMemoryForm;
  std::int64_t indexEEW = 0;
  llvm::StringRef offsetUnit;
  llvm::StringRef indexUniqueness;
  llvm::StringRef indexSource;
  llvm::StringRef indexedDataMemoryForm;
  llvm::StringRef indexedDestinationMemoryForm;
  llvm::StringRef lhsStrideSource;
  llvm::StringRef rhsStrideSource;
  llvm::StringRef outStrideSource;
  llvm::StringRef sourceStrideSource;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 8>
      runtimeABIParameters;
};

llvm::ArrayRef<RVVSelectedBodyOperationKind> getRVVSelectedBodyOperationKinds();

llvm::StringRef
stringifyRVVSelectedBodyOperationKind(RVVSelectedBodyOperationKind op);
llvm::StringRef
stringifyRVVSelectedBodyMemoryForm(RVVSelectedBodyMemoryForm form);
llvm::StringRef getRVVSelectedBodyEmitCRouteID(RVVSelectedBodyOperationKind op);
llvm::StringRef getRVVSelectedBodyEmissionKind();
llvm::StringRef getRVVSelectedBodyLoweringBoundaryOpName();
llvm::StringRef getRVVSelectedBodyRuntimeABIKind();
llvm::StringRef
getRVVSelectedBodyRuntimeABIName(RVVSelectedBodyOperationKind op);
llvm::StringRef getRVVSelectedBodyRuntimeGlueRole();

llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters();

RVVSelectedBodyConstructionMetadataFacts
getRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::SmallVector<tianchenrv::support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute *verifiedRoute =
        nullptr);

llvm::Error verifyRVVSelectedBodyEmitCRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

llvm::Error buildRVVSelectedBodyEmitCLowerableRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPROVIDER_H
