#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPLANNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPLANNING_H

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"
#include "TianChenRV/Support/RuntimeABIContract.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

struct RVVRouteOperandBinding {
  std::string logicalOperand;
  support::RuntimeABIParameter parameter;
  llvm::SmallVector<std::string, 4> materializedUses;
};

struct RVVRouteOperandBindingPlan {
  std::string planID;
  llvm::SmallVector<RVVRouteOperandBinding, 8> bindings;
};

struct RVVSelectedBodyRouteSlice {
  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  tcrv::rvv::LoadOp lhsGenericLoad;
  tcrv::rvv::LoadOp rhsGenericLoad;
  tcrv::rvv::LoadOp secondaryCompareLhsGenericLoad;
  tcrv::rvv::LoadOp trueValueGenericLoad;
  tcrv::rvv::LoadOp falseValueGenericLoad;
  tcrv::rvv::LoadOp dotLHSGenericLoad;
  tcrv::rvv::LoadOp dotRHSGenericLoad;
  tcrv::rvv::LoadOp sourceGenericLoad;
  tcrv::rvv::StridedLoadOp lhsStridedLoad;
  tcrv::rvv::StridedLoadOp rhsStridedLoad;
  tcrv::rvv::IndexLoadOp indexLoad;
  tcrv::rvv::IndexedLoadOp indexedLoad;
  tcrv::rvv::IndexedStoreOp indexedStore;
  tcrv::rvv::MaskLoadOp maskLoad;
  tcrv::rvv::MaskedLoadOp maskedLoadOp;
  tcrv::rvv::MaskedStridedLoadOp maskedStridedLoadOp;
  tcrv::rvv::MaskedIndexedLoadOp maskedIndexedLoadOp;
  tcrv::rvv::MaskedIndexedStoreOp maskedIndexedStore;
  tcrv::rvv::MaskedSegment2LoadOp maskedSegment2LoadOp;
  tcrv::rvv::Segment2LoadOp segment2Load;
  tcrv::rvv::Segment2StoreOp segment2Store;
  tcrv::rvv::MaskedSegment2StoreOp maskedSegment2Store;
  tcrv::rvv::BroadcastLoadOp rhsBroadcastLoad;
  tcrv::rvv::SplatOp rhsScalarSplat;
  tcrv::rvv::SplatOp rhsSecondaryScalarSplat;
  tcrv::rvv::CompareOp compareOp;
  tcrv::rvv::CompareOp secondaryCompareOp;
  tcrv::rvv::MaskAndOp maskAndOp;
  tcrv::rvv::SelectOp selectOp;
  tcrv::rvv::ReduceOp reduceOp;
  tcrv::rvv::StandaloneReduceOp standaloneReduceOp;
  tcrv::rvv::MaskedStandaloneReduceOp maskedStandaloneReduceOp;
  tcrv::rvv::MaskedBinaryOp maskedBinaryOp;
  tcrv::rvv::MAccOp maccOp;
  tcrv::rvv::MaskedMAccOp maskedMAccOp;
  tcrv::rvv::WideningMAccOp wideningMAccOp;
  tcrv::rvv::WideningDotReduceOp wideningDotReduceOp;
  tcrv::rvv::MaskedWideningDotReduceOp maskedWideningDotReduceOp;
  tcrv::rvv::WideningConvertOp wideningConvertOp;
  tcrv::rvv::MoveOp moveOp;
  tcrv::rvv::MoveOp field0MoveOp;
  tcrv::rvv::MoveOp field1MoveOp;
  tcrv::rvv::MaskedMoveOp maskedMoveOp;
  tcrv::rvv::MaskedStoreOp maskedStore;
  tcrv::rvv::MaskedStridedStoreOp maskedStridedStore;
  mlir::Operation *arithmeticOp = nullptr;
  mlir::Value arithmeticLhs;
  mlir::Value arithmeticRhs;
  mlir::Value arithmeticAccumulator;
  mlir::Value arithmeticResult;
  mlir::Value compareLhs;
  mlir::Value compareRhs;
  mlir::Value compareMask;
  mlir::Value secondaryCompareLhs;
  mlir::Value secondaryCompareRhs;
  mlir::Value secondaryCompareMask;
  mlir::Value composedMask;
  mlir::Value maskValue;
  mlir::Value maskedPassthrough;
  mlir::Value maskedActiveValue;
  mlir::Value maskedInactivePassthrough;
  mlir::Value conversionSource;
  mlir::Value dotLHSValue;
  mlir::Value dotRHSValue;
  RVVSelectedBodyOperationKind arithmeticKind;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  tcrv::rvv::StoreOp genericStore;
  tcrv::rvv::StoreOp field0Store;
  tcrv::rvv::StoreOp field1Store;
  tcrv::rvv::StridedStoreOp stridedStore;
  mlir::Operation *lhsLoadOperation = nullptr;
  mlir::Operation *rhsLoadOperation = nullptr;
  mlir::Operation *secondaryCompareLhsLoadOperation = nullptr;
  mlir::Operation *trueValueLoadOperation = nullptr;
  mlir::Operation *falseValueLoadOperation = nullptr;
  mlir::Operation *sourceLoadOperation = nullptr;
  mlir::Operation *field0LoadOperation = nullptr;
  mlir::Operation *field1LoadOperation = nullptr;
  mlir::Operation *indexLoadOperation = nullptr;
  mlir::Operation *indexedLoadOperation = nullptr;
  mlir::Operation *indexedStoreOperation = nullptr;
  mlir::Operation *maskLoadOperation = nullptr;
  mlir::Operation *maskedLoadOperation = nullptr;
  mlir::Operation *maskedStridedLoadOperation = nullptr;
  mlir::Operation *maskedIndexedLoadOperation = nullptr;
  mlir::Operation *maskedIndexedStoreOperation = nullptr;
  mlir::Operation *maskedSegment2LoadOperation = nullptr;
  mlir::Operation *segment2LoadOperation = nullptr;
  mlir::Operation *segment2StoreOperation = nullptr;
  mlir::Operation *maskedSegment2StoreOperation = nullptr;
  mlir::Operation *field0MoveOperation = nullptr;
  mlir::Operation *field1MoveOperation = nullptr;
  mlir::Operation *field0StoreOperation = nullptr;
  mlir::Operation *field1StoreOperation = nullptr;
  mlir::Operation *accumulatorLoadOperation = nullptr;
  mlir::Operation *dotLHSLoadOperation = nullptr;
  mlir::Operation *dotRHSLoadOperation = nullptr;
  mlir::Operation *storeOperation = nullptr;
  mlir::Operation *maskedStoreOperation = nullptr;
  mlir::Operation *maskedStridedStoreOperation = nullptr;
  mlir::Value lhsBuffer;
  mlir::Value rhsBuffer;
  mlir::Value secondaryCompareLhsBuffer;
  mlir::Value trueValueBuffer;
  mlir::Value falseValueBuffer;
  mlir::Value sourceBuffer;
  mlir::Value indexBuffer;
  mlir::Value maskBuffer;
  mlir::Value field0Buffer;
  mlir::Value field1Buffer;
  mlir::Value indexValue;
  mlir::Value indexedDataBuffer;
  mlir::Value indexedDestinationBuffer;
  mlir::Value accumulatorBuffer;
  mlir::Value dotLHSBuffer;
  mlir::Value dotRHSBuffer;
  mlir::Value outBuffer;
  mlir::Value lhsStride;
  mlir::Value rhsStride;
  mlir::Value sourceStride;
  mlir::Value outStride;
  mlir::Value lhsValue;
  mlir::Value rhsValue;
  mlir::Value trueValue;
  mlir::Value falseValue;
  mlir::Value sourceValue;
  mlir::Value field0LoadedValue;
  mlir::Value field1LoadedValue;
  mlir::Value field0PassthroughValue;
  mlir::Value field1PassthroughValue;
  mlir::Value field0Value;
  mlir::Value field1Value;
  mlir::Value accumulatorValue;
  mlir::Value storeValue;
  support::RuntimeABIParameter lhsABI;
  support::RuntimeABIParameter rhsABI;
  support::RuntimeABIParameter secondaryCompareLhsABI;
  support::RuntimeABIParameter secondaryCompareRhsScalarABI;
  support::RuntimeABIParameter trueValueABI;
  support::RuntimeABIParameter falseValueABI;
  support::RuntimeABIParameter sourceABI;
  support::RuntimeABIParameter indexABI;
  support::RuntimeABIParameter maskABI;
  support::RuntimeABIParameter field0ABI;
  support::RuntimeABIParameter field1ABI;
  support::RuntimeABIParameter accumulatorABI;
  support::RuntimeABIParameter dotLHSABI;
  support::RuntimeABIParameter dotRHSABI;
  support::RuntimeABIParameter outABI;
  support::RuntimeABIParameter runtimeElementCountABI;
  support::RuntimeABIParameter lhsStrideABI;
  support::RuntimeABIParameter rhsStrideABI;
  support::RuntimeABIParameter sourceStrideABI;
  support::RuntimeABIParameter outStrideABI;
};

struct RVVSelectedBodyContractionRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesWideningMAcc = false;
  bool usesDotReduction = false;
  bool usesComputedMask = false;
  bool usesStridedInputs = false;
  bool usesScalarSeed = false;
  bool usesVectorAccumulator = false;
  llvm::StringRef familyPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef sourceVectorLoadIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef accumulatorLayout;
  llvm::StringRef resultLayout;
  llvm::StringRef relation;
  llvm::StringRef contractionComputeIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef wideningProductIntrinsic;
  llvm::StringRef maskedWideningProductIntrinsic;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef reductionStoreVL;
  llvm::StringRef inactiveLaneZeroingRequirement;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef lhsStrideSource;
  llvm::StringRef rhsStrideSource;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::SmallVector<support::RuntimeABIParameter, 8> runtimeABIParameters;
};

struct RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef arithmeticIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef resultName;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
};

struct RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef resultName;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
};

struct RVVSelectedBodyWideningConversionRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef sourceVectorLoadIntrinsic;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef conversionIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef conversionRelation;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
};

struct RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesStridedLoad = false;
  bool usesStridedStore = false;
  bool usesIndexedGather = false;
  bool usesIndexedScatter = false;
  bool usesStaticMaskLoad = false;
  bool usesStaticMaskStore = false;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef indexedLoadIntrinsic;
  llvm::StringRef indexedStoreIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef maskedLoadIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef indexedMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef sourceStrideSource;
  llvm::StringRef destinationStrideSource;
  std::int64_t indexEEW = 0;
  llvm::StringRef offsetUnit;
  llvm::StringRef indexSource;
  llvm::StringRef indexUniqueness;
  llvm::StringRef indexedDataMemoryForm;
  llvm::StringRef indexedDestinationMemoryForm;
  llvm::SmallVector<support::RuntimeABIParameter, 6> runtimeABIParameters;
};

struct RVVSelectedBodyComputedMaskSelectRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesVectorCompareProducer = false;
  bool usesRuntimeScalarProducer = false;
  bool usesDualCompareMaskAnd = false;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef maskProducerSource;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef secondaryCompareIntrinsic;
  llvm::StringRef maskAndIntrinsic;
  llvm::StringRef selectIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef maskComposition;
  llvm::StringRef selectLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef indexedMemoryLayout;
  llvm::SmallVector<support::RuntimeABIParameter, 8> runtimeABIParameters;
};

struct RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesRuntimeScalarProducer = false;
  bool usesVectorCompareProducer = false;
  bool usesStoreOnly = false;
  bool usesLoadMerge = false;
  bool usesIndexedGather = false;
  bool usesIndexedScatter = false;
  bool usesSegment2Load = false;
  bool usesSegment2Store = false;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef maskProducerSource;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef indexLoadIntrinsic;
  llvm::StringRef indexScaleIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedLoadIntrinsic;
  llvm::StringRef maskedStoreIntrinsic;
  llvm::StringRef stridedStoreIntrinsic;
  llvm::StringRef indexedStoreIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef maskedMemoryLayout;
  llvm::StringRef stridedMemoryLayout;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef sourceStrideSource;
  llvm::StringRef destinationStrideSource;
  std::int64_t indexEEW = 0;
  llvm::StringRef offsetUnit;
  llvm::StringRef indexSource;
  llvm::StringRef indexUniqueness;
  llvm::StringRef indexedDataMemoryForm;
  llvm::StringRef indexedDestinationMemoryForm;
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
  llvm::SmallVector<support::RuntimeABIParameter, 6> runtimeABIParameters;
};

struct RVVSelectedBodySegment2MemoryRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesDeinterleaveLoad = false;
  bool usesInterleaveStore = false;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
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
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
};

struct RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesVectorMAccSuffix = false;
  bool usesScalarHorizontalReductionSuffix = false;
  bool usesVectorCompareProducer = false;
  bool usesRuntimeScalarProducer = false;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef computeSuffix;
  llvm::StringRef maskProducerSource;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
  llvm::StringRef accumulatorContract;
  llvm::StringRef resultContract;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef scalarCarryContract;
  llvm::SmallVector<support::RuntimeABIParameter, 8> runtimeABIParameters;
};

struct RVVSelectedBodyStandaloneReductionRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesComputedMask = false;
  bool usesRuntimeScalarThreshold = false;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
  llvm::StringRef familyPlanID;
  llvm::StringRef runtimeABIOrder;
  llvm::StringRef targetLeafProfile;
  llvm::StringRef providerSupportedMirror;
  llvm::SmallVector<llvm::StringRef, 4> requiredHeaders;
  llvm::StringRef requiredHeaderDeclarations;
  llvm::StringRef cTypeMappingSummary;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic;
  llvm::StringRef scalarSeedSplatIntrinsic;
  llvm::StringRef reductionIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef accumulatorLayout;
  llvm::StringRef resultLayout;
  llvm::StringRef reductionStoreVL;
  llvm::StringRef inactiveLaneZeroingRequirement;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef resultName;
  llvm::SmallVector<support::RuntimeABIParameter, 6> runtimeABIParameters;
};

struct RVVSelectedBodyRouteAnalysis {
  RVVSelectedBodyRouteSlice slice;
  const RVVSelectedBodyConstructionRoute *constructionRoute = nullptr;
  RVVSelectedBodyEmitCRouteDescription description;
  RVVRouteOperandBindingPlan routeOperandBindingPlan;
  std::optional<RVVSelectedBodyContractionRouteFamilyPlan>
      contractionRouteFamilyPlan;
  std::optional<RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan>
      scalarBroadcastElementwiseRouteFamilyPlan;
  std::optional<RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan>
      runtimeScalarSplatStoreRouteFamilyPlan;
  std::optional<RVVSelectedBodyWideningConversionRouteFamilyPlan>
      wideningConversionRouteFamilyPlan;
  std::optional<RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan>
      baseMemoryMovementRouteFamilyPlan;
  std::optional<RVVSelectedBodyComputedMaskSelectRouteFamilyPlan>
      computedMaskSelectRouteFamilyPlan;
  std::optional<RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan>
      computedMaskMemoryRouteFamilyPlan;
  std::optional<RVVSelectedBodySegment2MemoryRouteFamilyPlan>
      segment2MemoryRouteFamilyPlan;
  std::optional<
      RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan>
      computedMaskAccumulationRouteFamilyPlan;
  std::optional<RVVSelectedBodyStandaloneReductionRouteFamilyPlan>
      standaloneReductionRouteFamilyPlan;
};

bool isRVVSelectedBodyComputedMaskMemoryRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyPlainSegment2MemoryRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyBaseMemoryMovementRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyMemoryRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyMemoryRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyBaseMemoryMovementRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error
verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyComputedMaskSelectRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyWideningMAccContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyWideningDotReductionContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyContractionRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyPlainStandaloneReductionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyComputedMaskStandaloneReductionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyStandaloneReductionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyStandaloneReductionRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyComputedMaskMAccAccumulationRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyComputedMaskAccumulationRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error
verifyRVVSelectedBodyComputedMaskAccumulationRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message);

llvm::Expected<const support::RuntimeABIParameter *>
getRVVRouteOperandBindingParameter(
    const RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
    llvm::StringRef materializedUse, llvm::StringRef context);

llvm::Error verifyRVVRouteOperandBindingPlan(
    const RVVRouteOperandBindingPlan &plan, llvm::StringRef expectedPlanID,
    llvm::StringRef expectedRuntimeABIOrder, llvm::StringRef context);

llvm::StringRef getExpectedRVVRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVRouteOperandBindingClosure(
    const RVVRouteOperandBindingPlan &plan,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context);

std::string
stringifyRVVRouteOperandBindingPlan(const RVVRouteOperandBindingPlan &plan);

llvm::Expected<RVVSelectedBodyRouteAnalysis>
analyzeRVVSelectedBodyRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPLANNING_H
