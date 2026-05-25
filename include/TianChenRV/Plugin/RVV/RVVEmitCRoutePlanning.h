#ifndef TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPLANNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVEMITCROUTEPLANNING_H

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABIContract.h"

#include "llvm/ADT/ArrayRef.h"
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

struct RVVSelectedBodyTypedConfigFacts {
  llvm::StringRef factsID;
  llvm::StringRef elementTypeName;
  std::int64_t elementBitWidth = 0;
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef configContractID;
  llvm::StringRef vectorTypeName;
  llvm::StringRef indexVectorTypeName;
  llvm::StringRef maskTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef indexVectorCType;
  llvm::StringRef maskCType;
  llvm::StringRef vlCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef storeIntrinsic;

  bool hasFacts() const { return !factsID.empty(); }
};

struct RVVSelectedTargetCapabilityFacts {
  std::string selectedProviderSymbol;
  std::string selectedProviderID;
  std::string selectedProviderKind;
  std::string rvvSatisfactionKind;
  std::string supportedSEW;
  std::string supportedLMUL;
  std::string requiredTailPolicy;
  std::string requiredMaskPolicy;
  std::string providerMirror;
  std::string legalityMirror;

  bool hasFacts() const { return !selectedProviderSymbol.empty(); }
};

llvm::Expected<RVVSelectedTargetCapabilityFacts>
collectRVVSelectedTargetCapabilityFacts(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    llvm::StringRef context);

llvm::Error verifyRVVSelectedTargetCapabilityForTypedConfig(
    RVVSelectedTargetCapabilityFacts &facts,
    const RVVSelectedBodyTypedConfigFacts &typedConfigFacts,
    llvm::StringRef context);

struct RVVSelectedBodyContractionRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesWideningMAcc = false;
  bool usesDotReduction = false;
  bool usesComputedMask = false;
  bool usesStridedInputs = false;
  bool usesScalarSeed = false;
  bool usesVectorAccumulator = false;
  RVVRuntimeAVLVLControlPlan runtimeControlPlan;
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

struct RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan {
  RVVSelectedBodyOperationKind operation;
  RVVSelectedBodyMemoryForm memoryForm;
  bool usesPlainVector = false;
  bool usesMaskedArithmetic = false;
  bool usesStridedInputs = false;
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
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef stridedLoadIntrinsic;
  llvm::StringRef arithmeticIntrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
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
  llvm::StringRef lhsStrideSource;
  llvm::StringRef rhsStrideSource;
  llvm::StringRef outStrideSource;
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

struct RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan {
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
  llvm::StringRef maccIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef accumulatorLayout;
  llvm::StringRef resultLayout;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
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

struct RVVSelectedBodyPlainCompareSelectRouteFamilyPlan {
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
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef comparePredicateKind;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef selectIntrinsic;
  llvm::StringRef storeIntrinsic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;
  llvm::StringRef inactiveLaneContract;
  llvm::StringRef maskedPassthroughLayout;
  llvm::StringRef selectLayout;
  llvm::StringRef trueValueRole;
  llvm::StringRef falseValueRole;
  llvm::StringRef selectedResultRole;
  llvm::StringRef sourceMemoryForm;
  llvm::StringRef destinationMemoryForm;
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
  RVVSelectedBodyTypedConfigFacts typedConfigFacts;
  RVVSelectedTargetCapabilityFacts selectedTargetCapabilityFacts;
  RVVRouteOperandBindingPlan routeOperandBindingPlan;
  std::optional<RVVSelectedBodyContractionRouteFamilyPlan>
      contractionRouteFamilyPlan;
  std::optional<RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan>
      elementwiseArithmeticRouteFamilyPlan;
  std::optional<RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan>
      scalarBroadcastElementwiseRouteFamilyPlan;
  std::optional<RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan>
      scalarBroadcastMAccRouteFamilyPlan;
  std::optional<RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan>
      runtimeScalarSplatStoreRouteFamilyPlan;
  std::optional<RVVSelectedBodyPlainCompareSelectRouteFamilyPlan>
      plainCompareSelectRouteFamilyPlan;
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

struct RVVSelectedBodyRouteMaterializationFacts {
  RVVSelectedBodyTypedConfigFacts typedConfigFacts;

  const RVVSelectedBodyContractionRouteFamilyPlan *contractionPlan = nullptr;
  const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan
      *elementwiseArithmeticPlan = nullptr;
  const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan
      *scalarBroadcastPlan = nullptr;
  const RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan
      *scalarBroadcastMAccPlan = nullptr;
  const RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan
      *runtimeScalarSplatStorePlan = nullptr;
  const RVVSelectedBodyPlainCompareSelectRouteFamilyPlan
      *plainCompareSelectPlan = nullptr;
  const RVVSelectedBodyWideningConversionRouteFamilyPlan
      *wideningConversionPlan = nullptr;
  const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan
      *baseMemoryMovementPlan = nullptr;
  const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan
      *computedMaskSelectPlan = nullptr;
  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan
      *computedMaskMemoryPlan = nullptr;
  const RVVSelectedBodySegment2MemoryRouteFamilyPlan *segment2MemoryPlan =
      nullptr;
  const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan
      *computedMaskAccumulationPlan = nullptr;
  const RVVSelectedBodyStandaloneReductionRouteFamilyPlan
      *standaloneReductionPlan = nullptr;

  bool emitsContractionDotReduction = false;
  bool emitsContractionWideningMAcc = false;
  bool emitsComputedMaskContraction = false;
  bool emitsStridedInputContraction = false;
  bool emitsStandaloneReduction = false;
  bool emitsComputedMaskStandaloneReduction = false;
  bool emitsRuntimeScalarComputedMaskStandaloneReduction = false;
  bool emitsWideningConversion = false;
  bool emitsPlainStandaloneReduction = false;
  bool emitsComputedMaskAccumulation = false;

  llvm::ArrayRef<llvm::StringRef> requiredHeaders;
  llvm::StringRef vlCType;
  llvm::StringRef resultVectorTypeName;
  llvm::StringRef resultVectorCType;
  llvm::StringRef sourceVectorTypeName;
  llvm::StringRef sourceVectorCType;
  llvm::StringRef maskTypeName;
  llvm::StringRef maskCType;
  llvm::StringRef setVLLeaf;
  llvm::StringRef sourceLoadLeaf;
  llvm::StringRef vectorLoadLeaf;
  llvm::StringRef stridedSourceLoadLeaf;
  llvm::StringRef storeLeaf;
  llvm::StringRef stridedStoreLeaf;
  llvm::StringRef contractionComputeLeaf;
  llvm::StringRef elementwiseComputeLeaf;
  llvm::StringRef wideningProductLeaf;
  llvm::StringRef maskedWideningProductLeaf;
  llvm::StringRef scalarSeedSplatLeaf;
  llvm::StringRef rhsScalarBroadcastLeaf;
  llvm::StringRef compareLeaf;
  llvm::StringRef maskedMergeLeaf;
};

struct RVVSelectedBodyRouteControlProviderPlan {
  const RVVSelectedBodyTypedConfigFacts *typedConfigFacts = nullptr;
  const RVVSelectedTargetCapabilityFacts *selectedTargetCapabilityFacts =
      nullptr;
  const RVVRuntimeAVLVLControlPlan *runtimeControlPlan = nullptr;

  bool plansRouteControl = false;
  bool controlsBaseMemoryMovement = false;
  bool controlsStandaloneReduction = false;
  bool controlsScalarBroadcastMAcc = false;

  llvm::StringRef controlPlanIDMirror;
  llvm::StringRef configContractIDMirror;
  llvm::StringRef runtimeVLContractIDMirror;
  llvm::StringRef runtimeAVLASourceMirror;
  llvm::StringRef runtimeABIOrderMirror;
  llvm::StringRef tailPolicyMirror;
  llvm::StringRef maskPolicyMirror;
  llvm::StringRef selectedProviderMirror;
  llvm::StringRef selectedLegalityMirror;
};

struct RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts {
  const RVVRouteOperandBindingPlan *bindingPlan = nullptr;

  bool bindsElementwiseSelectCluster = false;
  bool bindsOrdinaryElementwiseArithmetic = false;
  bool bindsScalarBroadcastElementwise = false;
  bool bindsPlainCompareSelect = false;
  bool bindsComputedMaskSelect = false;
  bool bindsRuntimeScalarComputedMaskSelect = false;
  bool bindsRuntimeScalarDualCompareMaskAndSelect = false;

  const support::RuntimeABIParameter *lhsABI = nullptr;
  const support::RuntimeABIParameter *rhsABI = nullptr;
  const support::RuntimeABIParameter *secondaryCompareLhsABI = nullptr;
  const support::RuntimeABIParameter *secondaryCompareRhsScalarABI = nullptr;
  const support::RuntimeABIParameter *trueValueABI = nullptr;
  const support::RuntimeABIParameter *falseValueABI = nullptr;
  const support::RuntimeABIParameter *outABI = nullptr;
  const support::RuntimeABIParameter *runtimeElementCountABI = nullptr;
};

struct RVVSelectedBodyMemoryRouteOperandBindingFacts {
  const RVVRouteOperandBindingPlan *bindingPlan = nullptr;

  bool bindsMemoryCluster = false;
  bool bindsBaseMemoryMovement = false;
  bool bindsComputedMaskMemory = false;
  bool bindsRuntimeScalarComputedMaskMemory = false;
  bool bindsPlainSegment2Memory = false;
  bool bindsSegment2Memory = false;

  const support::RuntimeABIParameter *compareLhsABI = nullptr;
  const support::RuntimeABIParameter *compareRhsABI = nullptr;
  const support::RuntimeABIParameter *rhsScalarABI = nullptr;
  const support::RuntimeABIParameter *sourceABI = nullptr;
  const support::RuntimeABIParameter *destinationABI = nullptr;
  const support::RuntimeABIParameter *passthroughABI = nullptr;
  const support::RuntimeABIParameter *indexABI = nullptr;
  const support::RuntimeABIParameter *maskABI = nullptr;
  const support::RuntimeABIParameter *field0ABI = nullptr;
  const support::RuntimeABIParameter *field1ABI = nullptr;
  const support::RuntimeABIParameter *runtimeElementCountABI = nullptr;
  const support::RuntimeABIParameter *sourceStrideABI = nullptr;
  const support::RuntimeABIParameter *destinationStrideABI = nullptr;
};

struct RVVSelectedBodyMathRouteOperandBindingFacts {
  const RVVRouteOperandBindingPlan *bindingPlan = nullptr;

  bool bindsMathCluster = false;
  bool bindsReduceAdd = false;
  bool bindsPlainMAcc = false;
  bool bindsComputedMaskMAcc = false;
  bool bindsStandaloneReduction = false;
  bool bindsComputedMaskStandaloneReduction = false;
  bool bindsRuntimeScalarComputedMaskStandaloneReduction = false;
  bool bindsWideningMAcc = false;
  bool bindsWideningConversion = false;
  bool bindsWideningDotReduction = false;
  bool bindsStridedInputWideningDotReduction = false;
  bool bindsComputedMaskWideningDotReduction = false;
  bool bindsComputedMaskStridedInputWideningDotReduction = false;

  const support::RuntimeABIParameter *lhsABI = nullptr;
  const support::RuntimeABIParameter *rhsABI = nullptr;
  const support::RuntimeABIParameter *sourceABI = nullptr;
  const support::RuntimeABIParameter *accumulatorABI = nullptr;
  const support::RuntimeABIParameter *dotLHSABI = nullptr;
  const support::RuntimeABIParameter *dotRHSABI = nullptr;
  const support::RuntimeABIParameter *outABI = nullptr;
  const support::RuntimeABIParameter *runtimeElementCountABI = nullptr;
  const support::RuntimeABIParameter *lhsStrideABI = nullptr;
  const support::RuntimeABIParameter *rhsStrideABI = nullptr;
};

struct RVVSelectedBodyResidualRouteOperandBindingFacts {
  const RVVRouteOperandBindingPlan *bindingPlan = nullptr;

  bool bindsResidualCluster = false;
  bool bindsMaskedElementwiseArithmetic = false;
  bool bindsStridedElementwiseAdd = false;
  bool bindsRuntimeScalarSplatStore = false;

  const support::RuntimeABIParameter *lhsABI = nullptr;
  const support::RuntimeABIParameter *rhsABI = nullptr;
  const support::RuntimeABIParameter *rhsScalarABI = nullptr;
  const support::RuntimeABIParameter *outABI = nullptr;
  const support::RuntimeABIParameter *runtimeElementCountABI = nullptr;
  const support::RuntimeABIParameter *lhsStrideABI = nullptr;
  const support::RuntimeABIParameter *rhsStrideABI = nullptr;
  const support::RuntimeABIParameter *outStrideABI = nullptr;
};

struct RVVSelectedBodyElementwiseArithmeticRouteStatementPlan {
  const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan
      *elementwiseArithmeticPlan = nullptr;
  const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan
      *scalarBroadcastPlan = nullptr;

  bool plansElementwiseArithmeticRoute = false;
  bool plansOrdinaryElementwiseArithmetic = false;
  bool plansScalarBroadcastElementwise = false;
  bool plansMaskedElementwiseArithmetic = false;
  bool plansStridedElementwiseAdd = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodyCompareSelectRouteStatementPlan {
  const RVVSelectedBodyPlainCompareSelectRouteFamilyPlan
      *plainCompareSelectPlan = nullptr;
  const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan
      *computedMaskSelectPlan = nullptr;

  bool plansCompareSelectRoute = false;
  bool plansPlainCompareSelect = false;
  bool plansComputedMaskSelect = false;
  bool plansRuntimeScalarComputedMaskSelect = false;
  bool plansRuntimeScalarDualCompareMaskAndSelect = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodyWideningConversionRouteStatementPlan {
  const RVVSelectedBodyWideningConversionRouteFamilyPlan
      *wideningConversionPlan = nullptr;

  bool plansWideningConversionRoute = false;
  bool plansWidenI16ToI32 = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodyStandaloneReductionRouteStatementPlan {
  const RVVSelectedBodyStandaloneReductionRouteFamilyPlan
      *standaloneReductionPlan = nullptr;

  bool plansStandaloneReductionRoute = false;
  bool plansStandaloneReduceAdd = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 3>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodyPlainMAccRouteStatementPlan {
  const RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan
      *scalarBroadcastMAccPlan = nullptr;
  const RVVRouteOperandBindingPlan *bindingPlan = nullptr;

  bool plansPlainMAccRoute = false;
  bool plansMAccAdd = false;
  bool plansScalarBroadcastMAccAdd = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodyBaseMemoryMovementRouteStatementPlan {
  const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan
      *baseMemoryMovementPlan = nullptr;

  bool plansBaseMemoryMovementRoute = false;
  bool plansStridedLoadUnitStore = false;
  bool plansUnitLoadStridedStore = false;
  bool plansIndexedGatherUnitStore = false;
  bool plansIndexedScatterUnitLoad = false;
  bool plansStaticMaskUnitLoadStore = false;
  bool plansStaticMaskUnitStore = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodyBaseMemoryMovementRouteProviderPlan {
  const RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan
      *baseMemoryMovementPlan = nullptr;
  const RVVRouteOperandBindingPlan *bindingPlan = nullptr;

  bool plansBaseMemoryMovementRoute = false;

  llvm::StringRef familyPlanIDMirror;
  llvm::StringRef providerSupportedMirror;
  llvm::StringRef targetLeafProfileMirror;
  llvm::StringRef runtimeABIOrderMirror;
  llvm::StringRef routeOperandBindingPlanIDMirror;
  llvm::StringRef routeOperandBindingSummaryMirror;
  llvm::StringRef requiredHeaderDeclarationsMirror;
  llvm::StringRef cTypeMappingSummaryMirror;

  RVVSelectedBodyBaseMemoryMovementRouteStatementPlan statementPlan;
};

struct RVVSelectedBodyComputedMaskMemoryRouteStatementPlan {
  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan
      *computedMaskMemoryPlan = nullptr;

  bool plansComputedMaskMemoryRoute = false;
  bool plansRuntimeScalarComputedMaskStore = false;
  bool plansRuntimeScalarComputedMaskLoadStore = false;
  bool plansComputedMaskUnitLoadStore = false;
  bool plansComputedMaskStridedStore = false;
  bool plansComputedMaskStridedLoadUnitStore = false;
  bool plansComputedMaskIndexedGatherLoadUnitStore = false;
  bool plansComputedMaskIndexedScatterStoreUnitLoad = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodySegment2MemoryRouteStatementPlan {
  const RVVSelectedBodySegment2MemoryRouteFamilyPlan *segment2MemoryPlan =
      nullptr;
  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan
      *computedMaskMemoryPlan = nullptr;

  bool plansSegment2MemoryRoute = false;
  bool plansPlainSegment2DeinterleaveUnitStore = false;
  bool plansPlainSegment2InterleaveUnitLoad = false;
  bool plansComputedMaskSegment2LoadUnitStore = false;
  bool plansComputedMaskSegment2StoreUnitLoad = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan {
  const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan
      *computedMaskAccumulationPlan = nullptr;

  bool plansComputedMaskAccumulationRoute = false;
  bool plansComputedMaskedMAccAdd = false;
  bool plansRuntimeScalarComputedMaskedMAccAdd = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

enum class RVVSelectedBodyMigratedRouteStatementPlanFamily {
  None,
  ElementwiseArithmetic,
  CompareSelect,
  WideningConversion,
  StandaloneReduction,
  PlainMAcc,
  BaseMemoryMovement,
  ComputedMaskMemory,
  Segment2Memory,
  ComputedMaskAccumulation,
};

struct RVVSelectedBodyMigratedRouteStatementPlan {
  RVVSelectedBodyMigratedRouteStatementPlanFamily family =
      RVVSelectedBodyMigratedRouteStatementPlanFamily::None;

  bool plansMigratedRoute = false;

  llvm::SmallVector<conversion::emitc::TCRVEmitCCallOpaqueStep, 2>
      preLoopSteps;
  conversion::emitc::TCRVEmitCForLoop loop;
};

struct RVVSelectedBodyMemoryRouteFamilyOwner {
  using ConsumerPredicate = bool (*)(RVVSelectedBodyOperationKind);
  using ProviderPlanVerifier = llvm::Error (*)(
      const RVVSelectedBodyRouteAnalysis &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanVerifier verifyProviderPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyMemoryRouteFamilyOwner>
getRVVSelectedBodyMemoryRouteFamilyOwners();

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

llvm::Error
verifyRVVSelectedBodyComputedMaskMemoryRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error verifyRVVSelectedBodyBaseMemoryMovementRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Error verifyRVVSelectedBodySegment2MemoryRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error
verifyRVVSelectedBodyElementwiseArithmeticRouteFamilyProviderPlans(
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

bool isRVVSelectedBodyPlainCompareSelectRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyPlainCompareSelectRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyComputedMaskSelectRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

struct RVVSelectedBodyElementwiseSelectRouteFamilyOwner {
  using ConsumerPredicate = bool (*)(RVVSelectedBodyOperationKind);
  using ProviderPlanVerifier = llvm::Error (*)(
      const RVVSelectedBodyRouteAnalysis &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanVerifier verifyProviderPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyElementwiseSelectRouteFamilyOwner>
getRVVSelectedBodyElementwiseSelectRouteFamilyOwners();

bool isRVVSelectedBodyElementwiseSelectRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyElementwiseSelectRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyWideningMAccContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyWideningDotReductionContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);
bool isRVVSelectedBodyContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyContractionRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

bool isRVVSelectedBodyScalarBroadcastMAccRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error
verifyRVVSelectedBodyScalarBroadcastMAccRouteFamilyProviderPlans(
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

struct RVVSelectedBodyReductionAccumulationContractionRouteFamilyOwner {
  using ConsumerPredicate = bool (*)(RVVSelectedBodyOperationKind);
  using ProviderPlanVerifier = llvm::Error (*)(
      const RVVSelectedBodyRouteAnalysis &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanVerifier verifyProviderPlan = nullptr;
};

llvm::ArrayRef<
    RVVSelectedBodyReductionAccumulationContractionRouteFamilyOwner>
getRVVSelectedBodyReductionAccumulationContractionRouteFamilyOwners();

bool isRVVSelectedBodyReductionAccumulationContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error
verifyRVVSelectedBodyReductionAccumulationContractionRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

struct RVVSelectedBodyRouteFamilyProviderOwner {
  using ConsumerPredicate = bool (*)(RVVSelectedBodyOperationKind);
  using ProviderPlanVerifier = llvm::Error (*)(
      const RVVSelectedBodyRouteAnalysis &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanVerifier verifyProviderPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyRouteFamilyProviderOwner>
getRVVSelectedBodyRouteFamilyProviderOwners();

bool isRVVSelectedBodyRouteFamilyProviderConsumer(
    RVVSelectedBodyOperationKind operation);

llvm::Error verifyRVVSelectedBodyRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Expected<RVVSelectedBodyRouteMaterializationFacts>
getRVVSelectedBodyRouteMaterializationFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Expected<RVVSelectedBodyRouteControlProviderPlan>
getRVVSelectedBodyRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts>
getRVVSelectedBodyElementwiseSelectRouteOperandBindingFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Expected<RVVSelectedBodyMemoryRouteOperandBindingFacts>
getRVVSelectedBodyMemoryRouteOperandBindingFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Expected<RVVSelectedBodyMathRouteOperandBindingFacts>
getRVVSelectedBodyMathRouteOperandBindingFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Expected<RVVSelectedBodyResidualRouteOperandBindingFacts>
getRVVSelectedBodyResidualRouteOperandBindingFacts(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context);

llvm::Expected<RVVSelectedBodyElementwiseArithmeticRouteStatementPlan>
getRVVSelectedBodyElementwiseArithmeticRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyCompareSelectRouteStatementPlan>
getRVVSelectedBodyCompareSelectRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyWideningConversionRouteStatementPlan>
getRVVSelectedBodyWideningConversionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts
        &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyStandaloneReductionRouteStatementPlan>
getRVVSelectedBodyStandaloneReductionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyPlainMAccRouteStatementPlan>
getRVVSelectedBodyPlainMAccRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteStatementPlan>
getRVVSelectedBodyBaseMemoryMovementRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteProviderPlan>
getRVVSelectedBodyBaseMemoryMovementRouteProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyComputedMaskMemoryRouteStatementPlan>
getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodySegment2MemoryRouteStatementPlan>
getRVVSelectedBodySegment2MemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyComputedMaskAccumulationRouteStatementPlan>
getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context);

llvm::Expected<RVVSelectedBodyMigratedRouteStatementPlan>
getRVVSelectedBodyMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context);

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
