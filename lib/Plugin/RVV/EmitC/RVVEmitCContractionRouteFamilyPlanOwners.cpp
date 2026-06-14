#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"

#include "RVVEmitCContractionRouteFamilyInternal.h"

#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

static std::optional<RVVWideningProductRouteFacts>
buildRVVWideningProductRouteFacts(RVVSelectedBodyOperationKind operation,
                                  bool isUnsigned) {
  if (operation != RVVSelectedBodyOperationKind::WideningProduct)
    return std::nullopt;

  constexpr std::int64_t kSourceSEW = 8;
  constexpr std::int64_t kResultSEW = 16;
  constexpr llvm::StringLiteral kSourceLMUL("mf4");
  constexpr llvm::StringLiteral kResultLMUL("mf2");
  const llvm::StringRef relation = getContractionWideningProductRelation(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, isUnsigned);

  RVVWideningProductRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
  facts.sourceElementTypeName =
      getContractionIntegerElementTypeName(kSourceSEW, isUnsigned);
  facts.resultElementTypeName =
      getContractionIntegerElementTypeName(kResultSEW, isUnsigned);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = kRVVWideningProductRuntimeABIOrder;
  facts.targetLeafProfile =
      getContractionTargetLeafProfile(kSourceSEW, kSourceLMUL, kResultSEW,
                                      kResultLMUL, isUnsigned);
  facts.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  facts.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      getContractionCTypeMappingSummary(kSourceSEW, kSourceLMUL, kResultSEW,
                                        kResultLMUL, isUnsigned);
  facts.routeOperandBindingPlanID =
      isUnsigned
          ? llvm::StringRef(kRVVUnsignedWideningProductOperandBindingPlanID)
          : llvm::StringRef(kRVVWideningProductOperandBindingPlanID);
  facts.contractionRouteFamilyPlanID = kRVVContractionRouteFamilyPlanID;
  facts.typedComputeOpName = "tcrv_rvv.widening_product";
  facts.lhsRole = "lhs-input-buffer";
  facts.rhsRole = "rhs-input-buffer";
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL;
  facts.resultSEW = kResultSEW;
  facts.resultLMUL = kResultLMUL;
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.wideningProductRelation = relation;
  facts.wideningProductMultiplicandRoleSummary =
      isUnsigned ? llvm::StringRef(
                       kRVVLowPrecisionUnsignedWideningProductMultiplicandRoles)
                 : llvm::StringRef(
                       kRVVLowPrecisionSignedWideningProductMultiplicandRoles);
  facts.wideningProductExtensionPolicy =
      isUnsigned ? llvm::StringRef(
                       kRVVLowPrecisionUnsignedWideningProductExtensionPolicy)
                 : llvm::StringRef(
                       kRVVLowPrecisionSignedWideningProductExtensionPolicy);
  facts.sourceVectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(kSourceSEW, kSourceLMUL, isUnsigned);
  facts.wideningProductIntrinsic = getContractionWideningProductIntrinsic(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
  facts.lowPrecisionPrimitiveContractID = kRVVLowPrecisionPrimitiveContractID;
  facts.lowPrecisionPrimitiveKind =
      isUnsigned ? llvm::StringRef(kRVVLowPrecisionPrimitiveUnsignedProductKind)
                 : llvm::StringRef(kRVVLowPrecisionPrimitiveSignedProductKind);
  facts.lowPrecisionPrimitiveSourceElementTypeName =
      facts.sourceElementTypeName;
  facts.lowPrecisionPrimitiveSourceSignedness =
      isUnsigned ? llvm::StringRef(kRVVLowPrecisionResourceSourceSignednessUnsigned)
                 : llvm::StringRef(kRVVLowPrecisionResourceSourceSignednessSigned);
  facts.lowPrecisionPrimitiveSourceLoadKind =
      kRVVLowPrecisionPrimitiveSourceLoadKind;
  facts.lowPrecisionPrimitiveSourceExtensionKind =
      isUnsigned ? llvm::StringRef(
                       kRVVLowPrecisionPrimitiveUnsignedSourceExtensionKind)
                 : llvm::StringRef(
                       kRVVLowPrecisionPrimitiveSignedSourceExtensionKind);
  facts.lowPrecisionPrimitiveProductElementTypeName =
      facts.resultElementTypeName;
  facts.lowPrecisionPrimitiveAccumulatorElementTypeName = "";
  facts.lowPrecisionPrimitiveResultElementTypeName =
      facts.resultElementTypeName;
  facts.storeIntrinsic =
      getContractionStoreIntrinsic(kResultSEW, kResultLMUL, isUnsigned);
  facts.setVLIntrinsic = getContractionSetVLIntrinsic(kResultSEW, kResultLMUL);
  facts.vlCType = "size_t";
  facts.sourceVectorTypeName =
      getContractionVectorTypeName(kSourceSEW, kSourceLMUL, isUnsigned);
  facts.sourceVectorCType =
      getContractionVectorCType(kSourceSEW, kSourceLMUL, isUnsigned);
  facts.resultVectorTypeName =
      getContractionVectorTypeName(kResultSEW, kResultLMUL, isUnsigned);
  facts.resultVectorCType =
      getContractionVectorCType(kResultSEW, kResultLMUL, isUnsigned);
  facts.logicalOperands.push_back("lhs");
  facts.logicalOperands.push_back("rhs");
  facts.logicalOperands.push_back("out");
  facts.logicalOperands.push_back("n");
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "lhs", isUnsigned ? llvm::StringRef(kRVVContractionU8PointerCType)
                         : llvm::StringRef(kRVVContractionI8PointerCType),
      support::RuntimeABIParameterRole::LHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "rhs", isUnsigned ? llvm::StringRef(kRVVContractionU8PointerCType)
                         : llvm::StringRef(kRVVContractionI8PointerCType),
      support::RuntimeABIParameterRole::RHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "out",
      isUnsigned ? llvm::StringRef(kRVVContractionOutputU16PointerCType)
                 : llvm::StringRef(kRVVContractionOutputI16PointerCType),
      support::RuntimeABIParameterRole::OutputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.routeOperandBindingSummary =
      (llvm::Twine(facts.routeOperandBindingPlanID) +
       (isUnsigned
            ? ";lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-u8mf4|hdr;"
              "rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-u8mf4|hdr;"
              "out=output-buffer:out:abi|res-store|res-u16mf2|hdr;"
            : ";lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-i8mf4|hdr;"
              "rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-i8mf4|hdr;"
              "out=output-buffer:out:abi|res-store|res-i16mf2|hdr;") +
       "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
          .str();
  return facts;
}

std::optional<RVVWideningProductRouteFacts>
getRVVWideningProductRouteFacts(RVVSelectedBodyOperationKind operation) {
  return buildRVVWideningProductRouteFacts(operation, /*isUnsigned=*/false);
}

std::optional<RVVWideningProductRouteFacts>
getRVVWideningProductRouteFacts(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.operation != RVVSelectedBodyOperationKind::WideningProduct)
    return std::nullopt;
  const bool isUnsigned =
      description.wideningProductRelation ==
      getContractionWideningProductRelation(
          tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
          tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
          /*isUnsigned=*/true);
  return buildRVVWideningProductRouteFacts(description.operation, isUnsigned);
}






std::optional<RVVWideningMAccRouteFacts>
getRVVWideningMAccRouteFacts(RVVSelectedBodyOperationKind operation) {
  if (operation != RVVSelectedBodyOperationKind::WideningMAccAdd)
    return std::nullopt;

  constexpr std::int64_t kSourceSEW = 16;
  constexpr std::int64_t kResultSEW = 32;
  constexpr llvm::StringLiteral kSourceLMUL("mf2");
  constexpr llvm::StringLiteral kResultLMUL("m1");
  const llvm::StringRef relation = getContractionWideningMAccRelation(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL);

  RVVWideningMAccRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
  facts.sourceElementTypeName = getContractionIntegerElementTypeName(kSourceSEW);
  facts.accumulatorElementTypeName =
      getContractionIntegerElementTypeName(kResultSEW);
  facts.resultElementTypeName =
      getContractionIntegerElementTypeName(kResultSEW);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = kRVVWideningMAccRuntimeABIOrder;
  facts.targetLeafProfile =
      getContractionTargetLeafProfile(kSourceSEW, kSourceLMUL, kResultSEW,
                                      kResultLMUL);
  facts.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  facts.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      getContractionCTypeMappingSummary(kSourceSEW, kSourceLMUL, kResultSEW,
                                        kResultLMUL);
  facts.routeOperandBindingPlanID = kRVVWideningMAccOperandBindingPlanID;
  facts.contractionRouteFamilyPlanID = kRVVContractionRouteFamilyPlanID;
  facts.typedComputeOpName = "tcrv_rvv.widening_macc";
  facts.wideningMAccArithmeticKind = kRVVPreRealizedWideningMAccOpKind;
  facts.lhsRole = "lhs-input-buffer";
  facts.rhsRole = "rhs-input-buffer";
  facts.accumulatorRole = kRVVPreRealizedAccumulatorRole;
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL;
  facts.accumulatorSEW = kResultSEW;
  facts.accumulatorLMUL = kResultLMUL;
  facts.resultSEW = kResultSEW;
  facts.resultLMUL = kResultLMUL;
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.rhsMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.accumulatorMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.wideningMAccAccumulatorLayout = kRVVWideningMAccAccumulatorLayout;
  facts.wideningMAccResultLayout = kRVVWideningMAccResultLayout;
  facts.wideningMAccRelation = relation;
  facts.sourceVectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(kSourceSEW, kSourceLMUL);
  facts.accumulatorVectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(kResultSEW, kResultLMUL);
  facts.wideningMAccIntrinsic = getContractionWideningMAccIntrinsic(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
  facts.storeIntrinsic = getContractionStoreIntrinsic(kResultSEW, kResultLMUL);
  facts.setVLIntrinsic = getContractionSetVLIntrinsic(kResultSEW, kResultLMUL);
  facts.vlCType = "size_t";
  facts.sourceVectorTypeName =
      getContractionVectorTypeName(kSourceSEW, kSourceLMUL);
  facts.sourceVectorCType =
      getContractionSignedVectorCType(kSourceSEW, kSourceLMUL);
  facts.resultVectorTypeName =
      getContractionVectorTypeName(kResultSEW, kResultLMUL);
  facts.resultVectorCType =
      getContractionSignedVectorCType(kResultSEW, kResultLMUL);
  facts.logicalOperands.push_back("lhs");
  facts.logicalOperands.push_back("rhs");
  facts.logicalOperands.push_back("acc");
  facts.logicalOperands.push_back("out");
  facts.logicalOperands.push_back("n");
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "lhs", kRVVContractionI16PointerCType,
      support::RuntimeABIParameterRole::LHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "rhs", kRVVContractionI16PointerCType,
      support::RuntimeABIParameterRole::RHSInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "acc", kRVVContractionI32PointerCType,
      support::RuntimeABIParameterRole::AccumulatorInputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "out", kRVVContractionOutputI32PointerCType,
      support::RuntimeABIParameterRole::OutputBuffer,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount,
      support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  facts.routeOperandBindingSummary =
      (llvm::Twine(facts.routeOperandBindingPlanID) +
       ";lhs=lhs-input-buffer:lhs:abi|src-load|wmacc-lhs|src-i16mf2|hdr;"
       "rhs=rhs-input-buffer:rhs:abi|src-load|wmacc-rhs|src-i16mf2|hdr;"
       "acc=accumulator-input-buffer:acc:abi|acc-load|wmacc-acc|acc-i32m1|hdr;"
       "out=output-buffer:out:abi|res-store|res-i32m1|hdr;"
       "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
          .str();
  return facts;
}

std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
getRVVLowPrecisionWideningReductionPrimitiveFacts(
    RVVSelectedBodyOperationKind operation) {
  return getRVVLowPrecisionWideningReductionPrimitiveFacts(
      operation, /*isUnsignedProductReduction=*/false);
}

std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
getRVVLowPrecisionWideningReductionPrimitiveFacts(
    RVVSelectedBodyOperationKind operation,
    bool isUnsignedProductReduction) {
  const bool isProductReductionDequantClamp =
      operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantization =
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      isProductReductionDequantClamp;
  const bool isProductReductionChain =
      operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isProductReductionDequantization;
  if (!isProductReductionChain)
    return std::nullopt;
  if (isUnsignedProductReduction && isProductReductionDequantization)
    return std::nullopt;

  constexpr std::int64_t kSourceSEW = 8;
  constexpr llvm::StringLiteral kSourceLMUL("mf4");
  constexpr std::int64_t kProductSEW = 16;
  constexpr llvm::StringLiteral kProductLMUL("mf2");
  constexpr std::int64_t kAccumulatorSEW = 32;
  constexpr llvm::StringLiteral kAccumulatorLMUL("m1");

  const llvm::StringRef productRelation = getContractionWideningProductRelation(
      kSourceSEW, kSourceLMUL, kProductSEW, kProductLMUL,
      isUnsignedProductReduction);
  const llvm::StringRef chainRelation =
      getContractionProductReductionChainRelation(
          kSourceSEW, kSourceLMUL, kProductSEW, kProductLMUL, kAccumulatorSEW,
          kAccumulatorLMUL, isUnsignedProductReduction);

  RVVLowPrecisionWideningReductionPrimitiveFacts facts;
  facts.hasFacts = true;
  facts.contractID = kRVVLowPrecisionWideningReductionPrimitiveContractID.str();
  facts.lowPrecisionPrimitiveContractID =
      kRVVLowPrecisionPrimitiveContractID.str();
  facts.lowPrecisionPrimitiveKind =
      (isUnsignedProductReduction
           ? llvm::StringRef(
                 kRVVLowPrecisionPrimitiveUnsignedProductReductionKind)
       : isProductReductionDequantClamp
           ? llvm::StringRef(
                 kRVVLowPrecisionPrimitiveSignedProductReductionDequantClampKind)
       : isProductReductionDequantization
           ? llvm::StringRef(
                 kRVVLowPrecisionPrimitiveSignedProductReductionDequantKind)
           : llvm::StringRef(
                 kRVVLowPrecisionPrimitiveSignedProductReductionKind))
          .str();
  facts.kind =
      (isUnsignedProductReduction
           ? llvm::StringRef(kRVVLowPrecisionUnsignedWideningReductionPrimitiveKind)
           : llvm::StringRef(kRVVLowPrecisionWideningReductionPrimitiveKind))
          .str();

  facts.sourceElementTypeName =
      getContractionIntegerElementTypeName(kSourceSEW,
                                           isUnsignedProductReduction)
          .str();
  facts.sourceSignedness =
      (isUnsignedProductReduction
           ? llvm::StringRef(kRVVLowPrecisionResourceSourceSignednessUnsigned)
           : llvm::StringRef(kRVVLowPrecisionResourceSourceSignednessSigned))
          .str();
  facts.sourceLoadKind = kRVVLowPrecisionPrimitiveSourceLoadKind.str();
  facts.sourceExtensionKind =
      (isUnsignedProductReduction
           ? llvm::StringRef(kRVVLowPrecisionPrimitiveUnsignedSourceExtensionKind)
           : llvm::StringRef(kRVVLowPrecisionPrimitiveSignedSourceExtensionKind))
          .str();
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL.str();
  facts.sourceVectorTypeName =
      getContractionVectorTypeName(kSourceSEW, kSourceLMUL,
                                   isUnsignedProductReduction)
          .str();
  facts.sourceVectorCType =
      getContractionVectorCType(kSourceSEW, kSourceLMUL,
                                isUnsignedProductReduction)
          .str();

  facts.productElementTypeName =
      getContractionIntegerElementTypeName(kProductSEW,
                                           isUnsignedProductReduction)
          .str();
  facts.productSEW = kProductSEW;
  facts.productLMUL = kProductLMUL.str();
  facts.productVectorTypeName =
      getContractionVectorTypeName(kProductSEW, kProductLMUL,
                                   isUnsignedProductReduction)
          .str();
  facts.productVectorCType =
      getContractionVectorCType(kProductSEW, kProductLMUL,
                                isUnsignedProductReduction)
          .str();

  facts.accumulatorElementTypeName =
      getContractionIntegerElementTypeName(kAccumulatorSEW,
                                           isUnsignedProductReduction)
          .str();
  facts.accumulatorSEW = kAccumulatorSEW;
  facts.accumulatorLMUL = kAccumulatorLMUL.str();
  facts.accumulatorVectorTypeName =
      getContractionVectorTypeName(kAccumulatorSEW, kAccumulatorLMUL,
                                   isUnsignedProductReduction)
          .str();
  facts.accumulatorVectorCType =
      getContractionVectorCType(kAccumulatorSEW, kAccumulatorLMUL,
                                isUnsignedProductReduction)
          .str();

  facts.reductionResultElementTypeName = facts.accumulatorElementTypeName;
  facts.reductionResultSEW = kAccumulatorSEW;
  facts.reductionResultLMUL = kAccumulatorLMUL.str();
  facts.finalResultElementTypeName =
      isProductReductionDequantization
          ? getContractionFloatElementTypeName(kAccumulatorSEW).str()
          : facts.reductionResultElementTypeName;

  facts.wideningProductRelation = productRelation.str();
  facts.productReductionChainRelation = chainRelation.str();
  facts.wideningProductIntrinsic =
      getContractionWideningProductIntrinsic(
          kSourceSEW, kSourceLMUL, kProductSEW, kProductLMUL, productRelation)
          .str();
  facts.reductionIntrinsic =
      getContractionWideningReductionIntrinsic(
          kProductSEW, kProductLMUL, kAccumulatorSEW, kAccumulatorLMUL,
          chainRelation)
          .str();
  facts.scalarSeedSplatIntrinsic =
      getContractionScalarSeedSplatIntrinsic(kAccumulatorSEW,
                                             kAccumulatorLMUL,
                                             isUnsignedProductReduction)
          .str();
  facts.accumulatorLayout = kRVVWideningDotProductAccumulatorLayout.str();
  facts.resultLayout = kRVVProductReductionResultLayout.str();
  facts.reductionStoreVL = kRVVWideningDotProductStoreVL.str();
  facts.wideningProductCandidateFact =
      (llvm::Twine("resource-candidate-widening-product:") +
       llvm::Twine(facts.wideningProductRelation) + ":" +
       llvm::Twine(facts.wideningProductIntrinsic))
          .str();
  facts.reductionCandidateFact =
      (llvm::Twine("resource-candidate-widening-reduction:") +
       llvm::Twine(facts.productReductionChainRelation) + ":" +
       llvm::Twine(facts.reductionIntrinsic) + ":store-vl=" +
       llvm::Twine(facts.reductionStoreVL))
          .str();
  return facts;
}

static std::optional<RVVWideningDotReduceRouteFacts>
buildRVVWideningDotReduceRouteFacts(RVVSelectedBodyOperationKind operation,
                                    bool isUnsignedProductReduction) {
  if (!isContractionDotReductionOperation(operation))
    return std::nullopt;

  const bool isProductReductionChain =
      operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantClamp =
      operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionDequantization =
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      isProductReductionDequantClamp;
  if (isUnsignedProductReduction &&
      operation != RVVSelectedBodyOperationKind::WideningProductReduceAdd)
    return std::nullopt;
  constexpr std::int64_t kProductSourceSEW = 8;
  constexpr llvm::StringLiteral kProductSourceLMUL("mf4");
  constexpr std::int64_t kProductSEW = 16;
  constexpr llvm::StringLiteral kProductLMUL("mf2");
  constexpr std::int64_t kDotSourceSEW = 16;
  constexpr llvm::StringLiteral kDotSourceLMUL("mf2");
  constexpr std::int64_t kResultSEW = 32;
  constexpr llvm::StringLiteral kResultLMUL("m1");
  const std::int64_t kSourceSEW =
      isProductReductionChain ? kProductSourceSEW : kDotSourceSEW;
  const llvm::StringRef kSourceLMUL =
      isProductReductionChain ? llvm::StringRef(kProductSourceLMUL)
                              : llvm::StringRef(kDotSourceLMUL);
  const llvm::StringRef dotRelation = getContractionWideningDotProductRelation(
      kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL);
  const llvm::StringRef productRelation = getContractionWideningProductRelation(
      kProductSourceSEW, kProductSourceLMUL, kProductSEW, kProductLMUL,
      isUnsignedProductReduction);
  const llvm::StringRef chainRelation =
      getContractionProductReductionChainRelation(
          kProductSourceSEW, kProductSourceLMUL, kProductSEW, kProductLMUL,
          kResultSEW, kResultLMUL, isUnsignedProductReduction);
  const llvm::StringRef relation =
      isProductReductionChain ? chainRelation : dotRelation;

  const bool isComputedMask = isRVVSelectedBodyContractionComputedMask(operation);
  const bool isStrided = isRVVSelectedBodyContractionStridedInputs(operation);

  RVVWideningDotReduceRouteFacts facts;
  facts.operation = operation;
  if (isProductReductionChain) {
    std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
        primitiveFacts =
            getRVVLowPrecisionWideningReductionPrimitiveFacts(
                operation, isUnsignedProductReduction);
    if (!primitiveFacts)
      return std::nullopt;
    facts.lowPrecisionWideningReductionPrimitiveFacts =
        std::move(*primitiveFacts);
  }
  if (isProductReductionDequantClamp) {
    facts.memoryForm =
        RVVSelectedBodyMemoryForm::
            UnitStrideWideningProductReduceDequantClampF32;
  } else if (isComputedMask && isStrided) {
    facts.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskStridedInputWideningDotReduce;
  } else if (isComputedMask) {
    facts.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideWideningDotReduce;
  } else if (isStrided) {
    facts.memoryForm =
        RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce;
  } else {
    facts.memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
  }
  const bool usesUnsignedIntegerRoute =
      isProductReductionChain && isUnsignedProductReduction;
  facts.sourceElementTypeName =
      getContractionIntegerElementTypeName(kSourceSEW,
                                           usesUnsignedIntegerRoute);
  facts.accumulatorElementTypeName =
      getContractionIntegerElementTypeName(kResultSEW,
                                           usesUnsignedIntegerRoute);
  facts.resultElementTypeName =
      isProductReductionDequantization
          ? getContractionFloatElementTypeName(kResultSEW)
          : getContractionIntegerElementTypeName(kResultSEW,
                                                 usesUnsignedIntegerRoute);
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getContractionRuntimeABIOrder(operation);
  facts.targetLeafProfile =
      isProductReductionDequantClamp
          ? internContractionDerivedText(
                (llvm::Twine("rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-"
                             "reduction-dequant-clamp-leaf-profile.v1"))
                    .str())
      : isProductReductionDequantization
          ? internContractionDerivedText(
                (llvm::Twine("rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-"
                             "reduction-dequantization-leaf-profile.v1"))
                    .str())
      : isProductReductionChain
          ? (usesUnsignedIntegerRoute
                 ? internContractionDerivedText(
                       (llvm::Twine("rvv-v1-u8mf4-u16mf2-u32m1-product-"
                                    "reduction-contraction-leaf-profile.v1"))
                           .str())
                 : internContractionDerivedText(
                       (llvm::Twine("rvv-v1-i8mf4-i16mf2-i32m1-product-"
                                    "reduction-contraction-leaf-profile.v1"))
                           .str()))
          : getContractionTargetLeafProfile(kSourceSEW, kSourceLMUL,
                                            kResultSEW, kResultLMUL);
  facts.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  facts.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  facts.cTypeMappingSummary =
      isProductReductionDequantClamp
          ? internContractionDerivedText(
                (llvm::Twine("vl:size_t,source:signed-e8mf4,product:signed-"
                             "e16mf2,seed:signed-i32,accumulator:signed-"
                             "e32m1,dequant-splat/clamped:float-e32m1,"
                             "scale:float,lower:float,upper:float"))
                    .str())
      : isProductReductionDequantization
          ? internContractionDerivedText(
                (llvm::Twine("vl:size_t,source:signed-e8mf4,product:signed-"
                             "e16mf2,seed:signed-i32,accumulator:signed-"
                             "e32m1,dequant-splat:float-e32m1,scale:float"))
                    .str())
      : isProductReductionChain
          ? getContractionProductReductionChainCTypeMappingSummary(
                kProductSourceSEW, kProductSourceLMUL, kProductSEW,
                kProductLMUL, kResultSEW, kResultLMUL,
                usesUnsignedIntegerRoute)
          : getContractionCTypeMappingSummary(kSourceSEW, kSourceLMUL,
                                              kResultSEW, kResultLMUL);
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    facts.routeOperandBindingPlanID =
        kRVVWideningProductReductionChainOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    facts.routeOperandBindingPlanID =
        kRVVWideningProductReductionDequantizeOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    facts.routeOperandBindingPlanID =
        kRVVWideningProductReductionDequantClampF32OperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    facts.routeOperandBindingPlanID = kRVVWideningDotReduceOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    facts.routeOperandBindingPlanID =
        kRVVStridedInputWideningDotReduceOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    facts.routeOperandBindingPlanID =
        kRVVComputedMaskWideningDotReduceOperandBindingPlanID;
    break;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    facts.routeOperandBindingPlanID =
        kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID;
    break;
  default:
    return std::nullopt;
  }
  facts.contractionRouteFamilyPlanID = kRVVContractionRouteFamilyPlanID;
  facts.typedComputeOpName =
      isProductReductionDequantClamp
          ? "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize+tcrv_rvv.compare+tcrv_rvv.select"
      : isProductReductionDequantization
          ? "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize"
      : isProductReductionChain
          ? "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce"
      : isComputedMask ? "tcrv_rvv.masked_widening_dot_reduce"
                       : "tcrv_rvv.widening_dot_reduce";
  if (isComputedMask) {
    facts.comparePredicateKind = kRVVPreRealizedPredicateKind;
    facts.maskRole = kRVVMaskedPredicateMaskRole;
    facts.maskSource = kRVVMaskedCompareMaskSource;
    facts.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  }
  facts.lhsRole = "lhs-input-buffer";
  facts.rhsRole = "rhs-input-buffer";
  facts.dotLHSRole = "dot-lhs-input-buffer";
  facts.dotRHSRole = "dot-rhs-input-buffer";
  facts.accumulatorRole = kRVVPreRealizedAccumulatorRole;
  facts.outputRole = "output-buffer";
  facts.runtimeCountRole = "runtime-element-count";
  facts.lhsStrideRole = "lhs-input-stride";
  facts.rhsStrideRole = "rhs-input-stride";
  if (isProductReductionDequantClamp) {
    facts.lowerBoundRole = kRVVContractionLowerBoundRole;
    facts.upperBoundRole = kRVVContractionUpperBoundRole;
    facts.lowerBoundCType = kRVVContractionF32ScalarCType;
    facts.upperBoundCType = kRVVContractionF32ScalarCType;
    facts.boundOrder = kRVVContractionClampBoundOrder;
    facts.clampRelation = kRVVContractionProductReductionDequantClampRelation;
    facts.selectLayout = kRVVContractionClampSelectLayout;
    facts.comparePredicateKind = kRVVPreRealizedPredicateKind;
    facts.secondaryComparePredicateKind = kRVVPreRealizedPredicateKind;
  }
  facts.sourceSEW = kSourceSEW;
  facts.sourceLMUL = kSourceLMUL;
  if (isProductReductionChain) {
    facts.productSEW = kProductSEW;
    facts.productLMUL = kProductLMUL;
  }
  facts.accumulatorSEW = kResultSEW;
  facts.accumulatorLMUL = kResultLMUL;
  facts.resultSEW = kResultSEW;
  facts.resultLMUL = kResultLMUL;
  if (isProductReductionChain) {
    facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  } else if (isStrided) {
    facts.sourceMemoryForm = kRVVStridedInputDotSourceMemoryForm;
    facts.destinationMemoryForm = kRVVDestinationMemoryForm;
    facts.stridedMemoryLayout =
        isComputedMask ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
                       : kRVVStridedInputWideningDotMemoryLayout;
    facts.lhsStrideSource = kRVVLHSStrideSource;
    facts.rhsStrideSource = kRVVRHSStrideSource;
  }
  facts.wideningDotProductAccumulatorLayout =
      kRVVWideningDotProductAccumulatorLayout;
  facts.wideningDotProductResultLayout =
      isProductReductionChain ? kRVVProductReductionResultLayout
                              : kRVVWideningDotProductResultLayout;
  if (isProductReductionChain) {
    facts.productReductionChainRelation = relation;
    facts.wideningProductIntrinsic = getContractionWideningProductIntrinsic(
        kProductSourceSEW, kProductSourceLMUL, kProductSEW, kProductLMUL,
        productRelation);
  } else {
    facts.wideningDotProductRelation = relation;
    if (operation == RVVSelectedBodyOperationKind::
                         ComputedMaskStridedInputWideningDotReduceAdd)
      facts.wideningDotSourceAccumulatorResultContract =
          kRVVComputedMaskStridedInputWideningDotSourceAccumulatorResultContract;
    facts.wideningProductIntrinsic = getContractionWideningProductIntrinsic(
        kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
  }
  if (isComputedMask)
    facts.maskedWideningProductIntrinsic =
        getContractionMaskedWideningProductIntrinsic(
            kSourceSEW, kSourceLMUL, kResultSEW, kResultLMUL, relation);
  facts.scalarSeedSplatIntrinsic =
      getContractionScalarSeedSplatIntrinsic(kResultSEW, kResultLMUL,
                                             usesUnsignedIntegerRoute);
  if (isStrided)
    facts.stridedLoadIntrinsic =
        getContractionStridedLoadIntrinsic(kSourceSEW, kSourceLMUL);
  facts.sourceVectorLoadIntrinsic =
      getContractionVectorLoadIntrinsic(kSourceSEW, kSourceLMUL,
                                        usesUnsignedIntegerRoute);
  facts.compareVectorLoadIntrinsic =
      isProductReductionDequantClamp
          ? getContractionFloatVectorLoadIntrinsic(kResultSEW, kResultLMUL)
          : getContractionVectorLoadIntrinsic(kResultSEW, kResultLMUL,
                                              usesUnsignedIntegerRoute);
  facts.reductionIntrinsic =
      isProductReductionChain
          ? getContractionWideningReductionIntrinsic(kProductSEW, kProductLMUL,
                                                     kResultSEW, kResultLMUL,
                                                     relation)
          : getContractionReductionIntrinsic(kResultSEW, kResultLMUL);
  facts.storeIntrinsic = getContractionStoreIntrinsic(
      kResultSEW, kResultLMUL, usesUnsignedIntegerRoute);
  facts.setVLIntrinsic = getContractionSetVLIntrinsic(kResultSEW, kResultLMUL);
  if (isComputedMask || isProductReductionDequantClamp) {
    facts.compareIntrinsic =
        isProductReductionDequantClamp
            ? getContractionFloatLessThanCompareIntrinsic(kResultSEW,
                                                          kResultLMUL)
            : getContractionSignedLessThanCompareIntrinsic(kResultSEW,
                                                           kResultLMUL);
    facts.maskedMergeIntrinsic =
        isProductReductionDequantClamp
            ? getContractionFloatSelectIntrinsic(kResultSEW, kResultLMUL)
            : getContractionSelectIntrinsic(kResultSEW, kResultLMUL);
  }
  if (isProductReductionDequantization)
    facts.rhsBroadcastIntrinsic =
        getContractionFloatScalarSplatIntrinsic(kResultSEW, kResultLMUL);
  if (isProductReductionDequantClamp) {
    facts.secondaryCompareIntrinsic =
        getContractionFloatLessThanCompareIntrinsic(kResultSEW, kResultLMUL);
  }
  facts.reductionStoreVL = kRVVWideningDotProductStoreVL;
  if (isComputedMask)
    facts.inactiveLaneZeroingRequirement =
        kRVVContractionMaskedInactiveLaneZeroingRequirement;
  facts.vlCType = "size_t";
  facts.sourceVectorTypeName =
      getContractionVectorTypeName(kSourceSEW, kSourceLMUL,
                                   usesUnsignedIntegerRoute);
  facts.sourceVectorCType =
      getContractionVectorCType(kSourceSEW, kSourceLMUL,
                                usesUnsignedIntegerRoute);
  if (isProductReductionChain) {
    facts.productVectorTypeName =
        getContractionVectorTypeName(kProductSEW, kProductLMUL,
                                     usesUnsignedIntegerRoute);
    facts.productVectorCType =
        getContractionVectorCType(kProductSEW, kProductLMUL,
                                  usesUnsignedIntegerRoute);
  }
  facts.resultVectorTypeName =
      isProductReductionDequantization
          ? getContractionFloatVectorTypeName(kResultSEW, kResultLMUL)
          : getContractionVectorTypeName(kResultSEW, kResultLMUL,
                                         usesUnsignedIntegerRoute);
  facts.resultVectorCType =
      isProductReductionDequantization
          ? getContractionFloatVectorCType(kResultSEW, kResultLMUL)
          : getContractionVectorCType(kResultSEW, kResultLMUL,
                                      usesUnsignedIntegerRoute);
  if (isProductReductionDequantization) {
    facts.dequantizationRelation = kRVVContractionDequantizationRelation;
    facts.dequantScaleRole = kRVVContractionDequantScaleRole;
    facts.dequantScaleCType = kRVVContractionDequantScaleCType;
    facts.dequantScaleName = kRVVContractionDequantScaleName;
    facts.storeIntrinsic = getContractionFloatStoreIntrinsic(kResultSEW,
                                                             kResultLMUL);
  }
  if (isComputedMask || isProductReductionDequantClamp) {
    llvm::StringRef maskElementTypeName =
        isProductReductionDequantClamp
            ? getContractionFloatElementTypeName(kResultSEW)
            : getContractionIntegerElementTypeName(kResultSEW);
    facts.maskTypeName = internContractionDerivedText(
        (llvm::Twine("!tcrv_rvv.mask<") +
         maskElementTypeName + ", \"" + kResultLMUL + "\">")
            .str());
    facts.maskCType = getContractionMaskCType(kResultSEW, kResultLMUL);
  }

  auto addRuntimeABI = [&](llvm::StringRef cName, llvm::StringRef cType,
                           support::RuntimeABIParameterRole role) {
    facts.runtimeABIParameters.push_back(support::RuntimeABIParameter(
        cName, cType, role,
        support::RuntimeABIParameterOwnership::TargetExportABIOwned));
  };
  if (isProductReductionChain) {
    facts.logicalOperands.push_back("lhs");
    facts.logicalOperands.push_back("rhs");
    facts.logicalOperands.push_back("acc");
    if (isProductReductionDequantization)
      facts.logicalOperands.push_back("scale");
    if (isProductReductionDequantClamp) {
      facts.logicalOperands.push_back("lower_bound");
      facts.logicalOperands.push_back("upper_bound");
    }
    facts.logicalOperands.push_back("out");
    facts.logicalOperands.push_back("n");
    addRuntimeABI("lhs",
                  usesUnsignedIntegerRoute
                      ? llvm::StringRef(kRVVContractionU8PointerCType)
                      : llvm::StringRef(kRVVContractionI8PointerCType),
                  support::RuntimeABIParameterRole::LHSInputBuffer);
    addRuntimeABI("rhs",
                  usesUnsignedIntegerRoute
                      ? llvm::StringRef(kRVVContractionU8PointerCType)
                      : llvm::StringRef(kRVVContractionI8PointerCType),
                  support::RuntimeABIParameterRole::RHSInputBuffer);
  } else if (isComputedMask) {
    facts.logicalOperands.push_back("cmp_lhs");
    facts.logicalOperands.push_back("cmp_rhs");
    facts.logicalOperands.push_back("dot_lhs");
    facts.logicalOperands.push_back("dot_rhs");
    facts.logicalOperands.push_back("acc");
    facts.logicalOperands.push_back("out");
    facts.logicalOperands.push_back("n");
    addRuntimeABI("cmp_lhs", kRVVContractionI32PointerCType,
                  support::RuntimeABIParameterRole::LHSInputBuffer);
    addRuntimeABI("cmp_rhs", kRVVContractionI32PointerCType,
                  support::RuntimeABIParameterRole::RHSInputBuffer);
    addRuntimeABI("lhs", kRVVContractionI16PointerCType,
                  support::RuntimeABIParameterRole::DotLHSInputBuffer);
    addRuntimeABI("rhs", kRVVContractionI16PointerCType,
                  support::RuntimeABIParameterRole::DotRHSInputBuffer);
  } else {
    facts.logicalOperands.push_back("lhs");
    facts.logicalOperands.push_back("rhs");
    facts.logicalOperands.push_back("acc");
    facts.logicalOperands.push_back("out");
    facts.logicalOperands.push_back("n");
    addRuntimeABI("lhs", kRVVContractionI16PointerCType,
                  support::RuntimeABIParameterRole::LHSInputBuffer);
    addRuntimeABI("rhs", kRVVContractionI16PointerCType,
                  support::RuntimeABIParameterRole::RHSInputBuffer);
  }
    addRuntimeABI("acc",
                  usesUnsignedIntegerRoute
                      ? llvm::StringRef(kRVVContractionU32PointerCType)
                      : llvm::StringRef(kRVVContractionI32PointerCType),
                  support::RuntimeABIParameterRole::AccumulatorInputBuffer);
    if (isProductReductionDequantization)
      addRuntimeABI("scale", kRVVContractionDequantScaleCType,
                    support::RuntimeABIParameterRole::DequantScaleValue);
    if (isProductReductionDequantClamp) {
      addRuntimeABI(kRVVContractionLowerBoundName,
                    kRVVContractionF32ScalarCType,
                    support::RuntimeABIParameterRole::LowerBoundScalarValue);
      addRuntimeABI(kRVVContractionUpperBoundName,
                    kRVVContractionF32ScalarCType,
                    support::RuntimeABIParameterRole::UpperBoundScalarValue);
    }
    addRuntimeABI("out",
                  isProductReductionDequantization
                      ? llvm::StringRef(kRVVContractionF32PointerCType)
                  : usesUnsignedIntegerRoute
                      ? llvm::StringRef(kRVVContractionOutputU32PointerCType)
                      : llvm::StringRef(kRVVContractionOutputI32PointerCType),
                  support::RuntimeABIParameterRole::OutputBuffer);
  addRuntimeABI("n", "size_t",
                support::RuntimeABIParameterRole::RuntimeElementCount);
  if (isStrided) {
    facts.logicalOperands.push_back("lhs_stride");
    facts.logicalOperands.push_back("rhs_stride");
    addRuntimeABI("lhs_stride", "size_t",
                  support::RuntimeABIParameterRole::LHSInputStride);
    addRuntimeABI("rhs_stride", "size_t",
                  support::RuntimeABIParameterRole::RHSInputStride);
  }

  if (isProductReductionDequantClamp) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|ld|wpl|i8mf4|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|ld|wpr|i8mf4|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;"
         "scale=dequant-scale-value:scale:abi|scale|f32|deq|hdr;"
         "lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr;"
         "upper_bound=upper-bound-scalar-value:upper_bound:abi|up|splat|cmp|sel|hdr;"
         "out=output-buffer:out:abi|cdeq|store|f32m1|hdr;"
         "n=runtime-element-count:n:abi|setvl|loop|hdr")
            .str();
  } else if (isProductReductionDequantization) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|src-i8mf4|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|src-i8mf4|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;"
         "scale=dequant-scale-value:scale:abi|runtime-scale|scale-f32|dequant|hdr;"
         "out=output-buffer:out:abi|dequant-result|store|res-f32m1|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
            .str();
  } else if (isProductReductionChain) {
    const llvm::StringRef sourceWidthUse =
        usesUnsignedIntegerRoute ? "src-u8mf4" : "src-i8mf4";
    const llvm::StringRef accumulatorWidthUse =
        usesUnsignedIntegerRoute ? "u32" : "i32";
    const llvm::StringRef resultWidthUse =
        usesUnsignedIntegerRoute ? "res-u32m1" : "res-i32m1";
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|src-load|wprod-lhs|" +
         sourceWidthUse +
         "|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|src-load|wprod-rhs|" +
         sourceWidthUse +
         "|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|wred|" +
         accumulatorWidthUse +
         "|hdr;"
         "out=output-buffer:out:abi|acc-state|store|" +
         resultWidthUse +
         "|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
            .str();
  } else if (isComputedMask && isStrided) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;"
         "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;"
         "dot_lhs=dot-lhs-input-buffer:lhs:abi|sld|mlhs|i16|hdr;"
         "dot_rhs=dot-rhs-input-buffer:rhs:abi|sld|mrhs|i16|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
         "out=output-buffer:out:abi|store|i32|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr;"
         "lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;"
         "rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr")
            .str();
  } else if (isComputedMask) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;"
         "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;"
         "dot_lhs=dot-lhs-input-buffer:lhs:abi|ld|mlhs|i16|hdr;"
         "dot_rhs=dot-rhs-input-buffer:rhs:abi|ld|mrhs|i16|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
         "out=output-buffer:out:abi|store|i32|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
            .str();
  } else if (isStrided) {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|sld|dot-lhs|i16|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|sld|dot-rhs|i16|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
         "out=output-buffer:out:abi|store|i32|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr;"
         "lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr;"
         "rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr")
            .str();
  } else {
    facts.routeOperandBindingSummary =
        (llvm::Twine(facts.routeOperandBindingPlanID) +
         ";lhs=lhs-input-buffer:lhs:abi|ld|dot-lhs|i16|hdr;"
         "rhs=rhs-input-buffer:rhs:abi|ld|dot-rhs|i16|hdr;"
         "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
         "out=output-buffer:out:abi|store|i32|hdr;"
         "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
            .str();
  }
  return facts;
}

static bool isUnsignedProductReductionRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.operation !=
      RVVSelectedBodyOperationKind::WideningProductReduceAdd)
    return false;
  return description.wideningProductRelation ==
             "unsigned-u8mf4xu8mf4-to-u16mf2" ||
         description.productReductionChainRelation ==
             "unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32" ||
         description.lowPrecisionPrimitiveSourceSignedness ==
             kRVVLowPrecisionResourceSourceSignednessUnsigned ||
         description.sourceVectorCType == "vuint8mf4_t" ||
         description.productVectorCType == "vuint16mf2_t" ||
         description.vectorCType == "vuint32m1_t";
}

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(RVVSelectedBodyOperationKind operation) {
  return buildRVVWideningDotReduceRouteFacts(
      operation, /*isUnsignedProductReduction=*/false);
}

std::optional<RVVWideningDotReduceRouteFacts>
getRVVWideningDotReduceRouteFacts(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return buildRVVWideningDotReduceRouteFacts(
      description.operation,
      isUnsignedProductReductionRouteDescription(description));
}








bool isRVVSelectedBodyContractionRouteOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
  case RVVSelectedBodyOperationKind::WideningProduct:
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyContractionDotReduction(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
         operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
         operation == RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         operation ==
             RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionComputedMask(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionStridedInputs(
    RVVSelectedBodyOperationKind operation) {
  return operation ==
             RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         operation == RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyWideningMAccContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
}

bool isRVVSelectedBodyWideningProductContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::WideningProduct;
}

bool isRVVSelectedBodyWideningDotReductionContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isContractionDotReductionOperation(operation);
}

bool isRVVSelectedBodyContractionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  // Stage 3 换心: the contraction route-family provider-plan registry (the
  // retired route-construction layer) is deleted; this live
  // family-classification predicate is decoupled from the dead table and
  // expressed directly as the disjunction of its leaf consumer classifiers
  // (byte-faithful to the former isConsumer column).
  return isRVVSelectedBodyWideningMAccContractionRouteFamilyConsumer(
             operation) ||
         isRVVSelectedBodyWideningProductContractionRouteFamilyConsumer(
             operation) ||
         isRVVSelectedBodyWideningDotReductionContractionRouteFamilyConsumer(
             operation);
}

llvm::StringRef getRVVSelectedBodyContractionRuntimeABIOrder(
    RVVSelectedBodyOperationKind operation) {
  return getContractionRuntimeABIOrder(operation);
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccAccumulatorLayout() {
  return kRVVWideningMAccAccumulatorLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccResultLayout() {
  return kRVVWideningMAccResultLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningMAccRelation() {
  return getContractionWideningMAccRelation(
      tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
      tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductAccumulatorLayout() {
  return kRVVWideningDotProductAccumulatorLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductResultLayout() {
  return kRVVWideningDotProductResultLayout;
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedWideningDotProductRelation() {
  return getContractionWideningDotProductRelation(
      tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
      tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1());
}

llvm::StringRef
getRVVSelectedBodyContractionExpectedMaskedInactiveLaneZeroingRequirement() {
  return kRVVContractionMaskedInactiveLaneZeroingRequirement;
}

} // namespace tianchenrv::plugin::rvv
