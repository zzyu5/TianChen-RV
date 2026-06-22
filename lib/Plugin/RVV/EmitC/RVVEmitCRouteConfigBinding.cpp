//===- RVVEmitCRouteConfigBinding.cpp - RVV typed-config + ABI binding ----===//
//
// Behavior-preserving split out of RVVEmitCRoutePlanning.cpp: the target-leaf
// profile derivation, typed-config-facts derivation + type validation, the
// runtime-ABI exec-binding assignment set (assignRVVGeneric*Binding), and the
// route operand-binding-plan builders + deriveRVVRouteOperandBindingPlan. These
// were file-local helpers in the monolith anonymous namespace; they are promoted
// to external linkage (named namespace) and the cross-TU subset is declared in
// the co-located implementation-private RVVEmitCRoutePlanningInternal.h. Pure
// relocation -- bodies are byte-identical; derived content is unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "RVVEmitCRoutePlanningInternal.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"
#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>



namespace tianchenrv::plugin::rvv {

llvm::Expected<RVVSelectedBodyTargetLeafProfile>
deriveRVVSelectedBodyTargetLeafProfile(
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedBodyOperationProfile &operationProfile,
    const RVVSelectedBodyConfigProfile &configProfile) {
  if (configProfile.sew == tcrv::rvv::getRVVSEW64Bits() &&
      description.operation != RVVSelectedBodyOperationKind::WidenI32ToI64) {
    const bool supportsPlainI64Add =
        description.operation == RVVSelectedBodyOperationKind::Add &&
        description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    const bool supportsMaskedI64Arithmetic =
        operationProfile.isMaskedArithmetic &&
        description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    const bool supportsPlainI64CompareSelect =
        description.operation == RVVSelectedBodyOperationKind::CmpSelect &&
        description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    const bool supportsComputedMaskI64CompareSelect =
        description.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskSelect &&
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    const bool supportsRuntimeScalarI64CompareSelect =
        description.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect &&
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    const bool supportsRuntimeScalarI64DualCompareMaskAndSelect =
        description.operation == RVVSelectedBodyOperationKind::
                                     RuntimeScalarDualCompareMaskAndSelect &&
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    const bool supportsRuntimeScalarI64ComputedMaskStore =
        description.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore &&
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    const bool supportsRuntimeScalarI64ComputedMaskLoadStore =
        description.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore &&
        description.memoryForm ==
            RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    const bool supportsRuntimeScalarI64ComputedMaskStandaloneReduction =
        description.operation == RVVSelectedBodyOperationKind::
                                     RuntimeScalarComputedMaskStandaloneReduceAdd &&
        description.memoryForm == RVVSelectedBodyMemoryForm::
                                      RuntimeScalarComputedMaskUnitStrideStandaloneReduction &&
        configProfile.lmul == tcrv::rvv::getRVVLMULM1();
    if (!supportsPlainI64Add && !supportsMaskedI64Arithmetic &&
        !supportsPlainI64CompareSelect &&
        !supportsComputedMaskI64CompareSelect &&
        !supportsRuntimeScalarI64CompareSelect &&
        !supportsRuntimeScalarI64DualCompareMaskAndSelect &&
        !supportsRuntimeScalarI64ComputedMaskStore &&
        !supportsRuntimeScalarI64ComputedMaskLoadStore &&
        !supportsRuntimeScalarI64ComputedMaskStandaloneReduction)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
  }

  if (isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
          description.operation, description.memoryForm) ||
      isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          description.operation))
    return RVVSelectedBodyTargetLeafProfile{};

  if (operationProfile.isWideningConversion) {
    if (description.memoryForm !=
        RVVSelectedBodyMemoryForm::UnitStrideConversion)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef intrinsic =
        getRVVSelectedBodyWideningConversionIntrinsic(description);
    if (intrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{intrinsic, "", "", ""};
  }

  if (isRVVSelectedBodyDequantizationRouteOperation(description.operation)) {
    if (description.memoryForm !=
        RVVSelectedBodyMemoryForm::UnitStrideDequantization)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef intrinsic =
        getRVVSelectedBodyI32ToF32DequantConvertIntrinsic();
    if (intrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{intrinsic, "", "", ""};
  }

  if (operationProfile.isCompareSelect) {
    const bool isF32ClampSelect =
        description.operation == RVVSelectedBodyOperationKind::F32ClampSelect;
    const bool isDequantClampF32Epilogue =
        description.operation ==
        RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
    const bool isComputedMaskSelect =
        description.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect;
    const bool isRuntimeScalarCompareSelect =
        description.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;
    const bool isRuntimeScalarDualCompareMaskAndSelect =
        description.operation == RVVSelectedBodyOperationKind::
                                     RuntimeScalarDualCompareMaskAndSelect;
    if (isF32ClampSelect || isDequantClampF32Epilogue) {
      if ((isF32ClampSelect &&
           description.memoryForm !=
               RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect) ||
          (isDequantClampF32Epilogue &&
           description.memoryForm !=
               RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue) ||
          configProfile.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
          configProfile.lmul != tcrv::rvv::getRVVLMULM1())
        return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
      llvm::StringRef compareIntrinsic =
          getRVVSelectedBodyFloatCompareIntrinsicForPredicate(
              description.comparePredicateKind, configProfile.sew,
              configProfile.lmul);
      llvm::StringRef secondaryCompareIntrinsic =
          getRVVSelectedBodyFloatCompareIntrinsicForPredicate(
              description.secondaryComparePredicateKind, configProfile.sew,
              configProfile.lmul);
      llvm::StringRef selectIntrinsic =
          getRVVSelectedBodyFloatSelectIntrinsic(configProfile.sew,
                                                 configProfile.lmul);
      llvm::StringRef scalarSplatIntrinsic =
          getRVVSelectedBodyFloatScalarSplatIntrinsic(configProfile.sew,
                                                      configProfile.lmul);
      if (compareIntrinsic.empty() || secondaryCompareIntrinsic.empty() ||
          selectIntrinsic.empty() || scalarSplatIntrinsic.empty())
        return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
      return RVVSelectedBodyTargetLeafProfile{
          selectIntrinsic, compareIntrinsic, "", scalarSplatIntrinsic, ""};
    }

    if ((!isComputedMaskSelect &&
         !isRuntimeScalarCompareSelect &&
         !isRuntimeScalarDualCompareMaskAndSelect &&
         description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad) ||
        (isComputedMaskSelect &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect) ||
        (isRuntimeScalarCompareSelect &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect) ||
        (isRuntimeScalarDualCompareMaskAndSelect &&
         description.memoryForm != RVVSelectedBodyMemoryForm::
                                       RuntimeScalarDualCompareMaskAndSelect))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef compareIntrinsic = getRVVSelectedBodyCompareIntrinsic(
        description.comparePredicateKind, configProfile);
    if (compareIntrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef secondaryCompareIntrinsic =
        isRuntimeScalarDualCompareMaskAndSelect
            ? getRVVSelectedBodyCompareIntrinsic(
                  description.secondaryComparePredicateKind, configProfile)
            : llvm::StringRef();
    llvm::StringRef maskAndIntrinsic =
        isRuntimeScalarDualCompareMaskAndSelect
            ? getRVVSelectedBodyMaskAndIntrinsic(configProfile)
            : llvm::StringRef();
    if (isRuntimeScalarDualCompareMaskAndSelect &&
        (secondaryCompareIntrinsic.empty() || maskAndIntrinsic.empty()))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodySelectIntrinsic(configProfile),
        compareIntrinsic, "", (isRuntimeScalarCompareSelect ||
                               isRuntimeScalarDualCompareMaskAndSelect)
                                  ? configProfile.rhsBroadcastIntrinsic
                                  : llvm::StringRef(),
        maskAndIntrinsic};
  }

  if (description.operation ==
          RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
      // The deferred-wide i16 dot-reduce terminal kind (2nd kernel family) shares
      // the narrow dot-reduce leaf profile: it carries NO scalar arithmetic /
      // compare / select / reduce leaf intrinsic -- the widening_product +
      // deferred_accumulate + standalone_reduce chain is emitted from the
      // dedicated structural ops, not from a derived leaf.
      description.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredDotAccumulateReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              ComputedMaskStridedInputWideningDotReduceAdd) {
    const bool isComputedMaskDotReduce =
        description.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
        description.operation ==
            RVVSelectedBodyOperationKind::
                ComputedMaskStridedInputWideningDotReduceAdd;
    const bool isStridedInputDotReduce =
        description.operation ==
            RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
        description.operation ==
            RVVSelectedBodyOperationKind::
                ComputedMaskStridedInputWideningDotReduceAdd;
    const bool isComputedMaskStridedInputDotReduce =
        description.operation ==
        RVVSelectedBodyOperationKind::
            ComputedMaskStridedInputWideningDotReduceAdd;
    if ((!isComputedMaskDotReduce && !isStridedInputDotReduce &&
         description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad) ||
        (isStridedInputDotReduce && !isComputedMaskStridedInputDotReduce &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce) ||
        (isComputedMaskStridedInputDotReduce &&
         description.memoryForm != RVVSelectedBodyMemoryForm::
                                       ComputedMaskStridedInputWideningDotReduce) ||
        (isComputedMaskDotReduce && !isStridedInputDotReduce &&
         description.memoryForm != RVVSelectedBodyMemoryForm::
                                       ComputedMaskUnitStrideWideningDotReduce))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{};
  }

  if (operationProfile.isReduction) {
    const bool isStandalone =
        isRVVSelectedBodyPlainStandaloneReductionRouteOperation(
            description.operation);
    const bool isComputedMaskStandalone =
        isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
            description.operation);
    const bool isRuntimeScalarComputedMaskStandalone =
        isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
            description.operation);
    if ((!isStandalone && !isComputedMaskStandalone &&
         !isRuntimeScalarComputedMaskStandalone &&
         description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad) ||
        (isStandalone &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction) ||
        (isComputedMaskStandalone &&
         description.memoryForm != RVVSelectedBodyMemoryForm::
                                       ComputedMaskUnitStrideStandaloneReduction) ||
        (isRuntimeScalarComputedMaskStandalone &&
         description.memoryForm != RVVSelectedBodyMemoryForm::
                                       RuntimeScalarComputedMaskUnitStrideStandaloneReduction))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef reductionIntrinsic =
        (isStandalone || isComputedMaskStandalone ||
         isRuntimeScalarComputedMaskStandalone)
            ? getRVVSelectedBodyStandaloneReductionIntrinsic(
                  description.operation, configProfile)
            : getRVVSelectedBodyReductionIntrinsic(configProfile);
    llvm::StringRef compareIntrinsic =
        (isComputedMaskStandalone || isRuntimeScalarComputedMaskStandalone)
            ? getRVVSelectedBodyCompareIntrinsic(
                  description.comparePredicateKind, configProfile)
            : llvm::StringRef();
    llvm::StringRef maskedMergeIntrinsic =
        (isComputedMaskStandalone || isRuntimeScalarComputedMaskStandalone)
            ? getRVVSelectedBodySelectIntrinsic(configProfile)
            : llvm::StringRef();
    if (reductionIntrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if ((isComputedMaskStandalone || isRuntimeScalarComputedMaskStandalone) &&
        (compareIntrinsic.empty() || maskedMergeIntrinsic.empty()))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        reductionIntrinsic, compareIntrinsic, maskedMergeIntrinsic,
        isRuntimeScalarComputedMaskStandalone
            ? configProfile.rhsBroadcastIntrinsic
            : llvm::StringRef()};
  }

  if (operationProfile.isMaskedArithmetic) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                             configProfile),
        getRVVSelectedBodyEqualCompareIntrinsic(configProfile),
        getRVVSelectedBodySelectIntrinsic(configProfile), ""};
  }

  if (description.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    if (description.memoryForm !=
        RVVSelectedBodyMemoryForm::
            RuntimeScalarComputedMaskIndexedGatherMAccScatter)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef intrinsic =
        getRVVSelectedBodyMAccIntrinsic(configProfile.sew, configProfile.lmul);
    llvm::StringRef compareIntrinsic = getRVVSelectedBodyCompareIntrinsic(
        description.comparePredicateKind, configProfile.lmul);
    llvm::StringRef maskedMergeIntrinsic =
        getRVVSelectedBodySelectIntrinsic(configProfile.lmul);
    if (intrinsic.empty() || compareIntrinsic.empty() ||
        maskedMergeIntrinsic.empty() ||
        configProfile.indexLoadIntrinsic.empty() ||
        configProfile.indexedLoadIntrinsic.empty() ||
        configProfile.indexedStoreIntrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        intrinsic, compareIntrinsic, maskedMergeIntrinsic,
        configProfile.rhsBroadcastIntrinsic};
  }

  if (operationProfile.isMultiplyAccumulate) {
    const bool isScalarBroadcastMAcc =
        description.operation ==
        RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
    const bool isComputedMaskedMAcc =
        description.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd;
    const bool isRuntimeScalarComputedMaskedMAcc =
        description.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
    if ((!isScalarBroadcastMAcc &&
         !isComputedMaskedMAcc &&
         !isRuntimeScalarComputedMaskedMAcc &&
         description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad) ||
        (isScalarBroadcastMAcc &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc) ||
        (isComputedMaskedMAcc &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc) ||
        (isRuntimeScalarComputedMaskedMAcc &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskUnitStrideMAcc))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd) {
      return RVVSelectedBodyTargetLeafProfile{};
    }
    if (isScalarBroadcastMAcc)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMAccIntrinsic(configProfile.sew,
                                          configProfile.lmul),
          "", "",
          configProfile.rhsBroadcastIntrinsic};
    if (isComputedMaskedMAcc || isRuntimeScalarComputedMaskedMAcc) {
      llvm::StringRef intrinsic =
          getRVVSelectedBodyMAccIntrinsic(configProfile.sew,
                                          configProfile.lmul);
      llvm::StringRef compareIntrinsic = getRVVSelectedBodyCompareIntrinsic(
          description.comparePredicateKind, configProfile.lmul);
      llvm::StringRef maskedMergeIntrinsic =
          getRVVSelectedBodySelectIntrinsic(configProfile.lmul);
      if (intrinsic.empty() || compareIntrinsic.empty() ||
          maskedMergeIntrinsic.empty())
        return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
      return RVVSelectedBodyTargetLeafProfile{
          intrinsic, compareIntrinsic, maskedMergeIntrinsic,
          isRuntimeScalarComputedMaskedMAcc
              ? configProfile.rhsBroadcastIntrinsic
              : llvm::StringRef()};
    }
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyMAccIntrinsic(configProfile.sew, configProfile.lmul),
        "", "", ""};
  }

  if (operationProfile.isMemoryMovement) {
    const bool isStridedLoadUnitStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::StridedLoadUnitStore;
    const bool isUnitLoadStridedStore =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::UnitLoadStridedStore;
    if ((isStridedLoadUnitStore &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) ||
        (isUnitLoadStridedStore &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::UnitLoadStridedStore))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (operationProfile.isSegmentedMemoryMovement) {
    const bool isComputedMaskSegment2Load =
        description.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
    const bool isRuntimeScalarComputedMaskSegment2Load =
        description.operation == RVVSelectedBodyOperationKind::
                                     RuntimeScalarComputedMaskSegment2LoadUnitStore;
    const bool isComputedMaskSegment2Store =
        description.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
    const bool isRuntimeScalarComputedMaskSegment2Store =
        description.operation == RVVSelectedBodyOperationKind::
                                     RuntimeScalarComputedMaskSegment2StoreUnitLoad;
    const bool isComputedMaskSegment2Update =
        description.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
    if (description.memoryForm !=
            RVVSelectedBodyMemoryForm::Segment2LoadUnitStore &&
        description.memoryForm !=
            RVVSelectedBodyMemoryForm::UnitLoadSegment2Store &&
        description.memoryForm !=
            RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore &&
        description.memoryForm !=
            RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1() ||
        configProfile.sew != tcrv::rvv::getRVVFirstSliceSEWBits())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (isComputedMaskSegment2Load)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMaskedSegmentLoadIntrinsic(
              configProfile.sew, configProfile.lmul, 2),
          getRVVSelectedBodySignedLessThanCompareIntrinsic(configProfile.lmul),
          "", ""};
    if (isRuntimeScalarComputedMaskSegment2Load)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMaskedSegmentLoadIntrinsic(
              configProfile.sew, configProfile.lmul, 2),
          getRVVSelectedBodySignedLessEqualCompareIntrinsic(
              configProfile.lmul),
          "", configProfile.rhsBroadcastIntrinsic, ""};
    if (isRuntimeScalarComputedMaskSegment2Store)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMaskedSegmentStoreIntrinsic(
              configProfile.sew, configProfile.lmul, 2),
          getRVVSelectedBodySignedLessEqualCompareIntrinsic(
              configProfile.lmul),
          "", configProfile.rhsBroadcastIntrinsic, ""};
    if (isComputedMaskSegment2Store || isComputedMaskSegment2Update)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMaskedSegmentStoreIntrinsic(
              configProfile.sew, configProfile.lmul, 2),
          getRVVSelectedBodySignedLessThanCompareIntrinsic(configProfile.lmul),
          "", ""};
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (operationProfile.isIndexedMemoryMovement) {
    const bool isGather =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
    const bool isScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
    const bool isComputedMaskIndexedGather =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGather =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
    const bool isComputedMaskIndexedScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
    const bool isRuntimeScalarComputedMaskIndexedScatter =
        operationProfile.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
    if ((isGather &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::IndexedLoadUnitStore) ||
        (isScatter &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::UnitLoadIndexedStore) ||
        (isComputedMaskIndexedGather &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::
                 ComputedMaskIndexedGatherLoadUnitStore) ||
        (isRuntimeScalarComputedMaskIndexedGather &&
         description.memoryForm != RVVSelectedBodyMemoryForm::
                                       ComputedMaskIndexedGatherLoadUnitStore) ||
        (isComputedMaskIndexedScatter &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::
                 ComputedMaskUnitLoadIndexedScatterStore) ||
        (isRuntimeScalarComputedMaskIndexedScatter &&
         description.memoryForm !=
             RVVSelectedBodyMemoryForm::
                 ComputedMaskUnitLoadIndexedScatterStore))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1() ||
        configProfile.indexLoadIntrinsic.empty() ||
        configProfile.indexScaleIntrinsic.empty() ||
        (isGather && configProfile.indexedLoadIntrinsic.empty()) ||
        (isScatter && configProfile.indexedStoreIntrinsic.empty()) ||
        ((isComputedMaskIndexedGather || isComputedMaskIndexedScatter) &&
         getRVVSelectedBodySignedLessThanCompareIntrinsic(
             configProfile.lmul)
             .empty()) ||
        (isRuntimeScalarComputedMaskIndexedGather &&
         (configProfile.rhsBroadcastIntrinsic.empty() ||
          getRVVSelectedBodySignedLessEqualCompareIntrinsic(
              configProfile.lmul)
              .empty())) ||
        (isRuntimeScalarComputedMaskIndexedScatter &&
         (configProfile.rhsBroadcastIntrinsic.empty() ||
          getRVVSelectedBodySignedLessEqualCompareIntrinsic(
              configProfile.lmul)
              .empty())))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (isComputedMaskIndexedGather)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMaskedIndexedLoadIntrinsic(configProfile.sew,
                                                       configProfile.lmul),
          getRVVSelectedBodySignedLessThanCompareIntrinsic(configProfile.lmul),
          "", ""};
    if (isRuntimeScalarComputedMaskIndexedGather)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMaskedIndexedLoadIntrinsic(configProfile.sew,
                                                       configProfile.lmul),
          getRVVSelectedBodySignedLessEqualCompareIntrinsic(
              configProfile.lmul),
          "", configProfile.rhsBroadcastIntrinsic, ""};
    if (isComputedMaskIndexedScatter)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMaskedIndexedStoreIntrinsic(configProfile.sew,
                                                        configProfile.lmul),
          getRVVSelectedBodySignedLessThanCompareIntrinsic(configProfile.lmul),
          "", ""};
    if (isRuntimeScalarComputedMaskIndexedScatter)
      return RVVSelectedBodyTargetLeafProfile{
          getRVVSelectedBodyMaskedIndexedStoreIntrinsic(configProfile.sew,
                                                        configProfile.lmul),
          getRVVSelectedBodySignedLessEqualCompareIntrinsic(
              configProfile.lmul),
          "", configProfile.rhsBroadcastIntrinsic, ""};
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (operationProfile.isMaskedMemoryMovement) {
    const bool isRuntimeMask =
        description.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
    const bool isMaskedStore =
        description.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore;
    const bool isRuntimeScalarComputedMaskStore =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore;
    const bool isRuntimeScalarComputedMaskLoadStore =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore;
    const bool isComputedMask =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
    const bool isComputedMaskStridedStore =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
    const bool isComputedMaskStridedLoad =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore;
    const bool isComputedMaskIndexedGather =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
    const bool isComputedMaskIndexedScatter =
        description.memoryForm ==
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
    if (!isRuntimeMask && !isMaskedStore &&
        !isRuntimeScalarComputedMaskStore &&
        !isRuntimeScalarComputedMaskLoadStore && !isComputedMask &&
        !isComputedMaskStridedStore && !isComputedMaskStridedLoad &&
        !isComputedMaskIndexedGather && !isComputedMaskIndexedScatter)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    const bool isRuntimeScalarComputedMaskMemory =
        isRuntimeScalarComputedMaskStore ||
        isRuntimeScalarComputedMaskLoadStore;
    if (isRuntimeScalarComputedMaskMemory &&
        !isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(
            configProfile.sew, configProfile.lmul))
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (!isRuntimeScalarComputedMaskMemory &&
        configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
      isComputedMaskIndexedScatter
          ? getRVVSelectedBodyMaskedIndexedStoreIntrinsic(configProfile.sew,
                                                          configProfile.lmul)
        : isComputedMaskIndexedGather
            ? getRVVSelectedBodyMaskedIndexedLoadIntrinsic(configProfile.sew,
                                                           configProfile.lmul)
        : isComputedMaskStridedStore
            ? getRVVSelectedBodyMaskedStridedStoreIntrinsic(configProfile.sew,
                                                            configProfile.lmul)
        : isComputedMaskStridedLoad
            ? getRVVSelectedBodyMaskedStridedLoadIntrinsic(configProfile.sew,
                                                           configProfile.lmul)
      : isRuntimeScalarComputedMaskLoadStore
          ? getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(configProfile)
      : (isRuntimeMask || isComputedMask)
          ? getRVVSelectedBodyMaskedLoadIntrinsic(configProfile.sew,
                                                  configProfile.lmul)
      : isRuntimeScalarComputedMaskStore
          ? getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(configProfile)
          : llvm::StringRef(),
      (isRuntimeScalarComputedMaskStore ||
       isRuntimeScalarComputedMaskLoadStore)
          ? getRVVSelectedBodyCompareIntrinsic(
                description.comparePredicateKind, configProfile)
        : (isComputedMask || isComputedMaskStridedStore ||
           isComputedMaskStridedLoad || isComputedMaskIndexedGather ||
           isComputedMaskIndexedScatter)
            ? getRVVSelectedBodySignedLessThanCompareIntrinsic(
                  configProfile.lmul)
            : getRVVSelectedBodyMaskFromI32Intrinsic(configProfile.lmul),
        "",
        (isRuntimeScalarComputedMaskStore ||
         isRuntimeScalarComputedMaskLoadStore)
            ? configProfile.rhsBroadcastIntrinsic
            : llvm::StringRef()};
  }

  if (operationProfile.isStridedMemory) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::StridedLoadStore)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (configProfile.lmul != tcrv::rvv::getRVVLMULM1())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(
            RVVSelectedBodyOperationKind::StridedAdd, configProfile),
        "", "", ""};
  }

  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad ||
      description.memoryForm == RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                             configProfile),
        "", "", configProfile.rhsBroadcastIntrinsic};

  if (description.memoryForm ==
      RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore)
    return RVVSelectedBodyTargetLeafProfile{
        "", "", "", configProfile.rhsBroadcastIntrinsic};

  if (description.operation == RVVSelectedBodyOperationKind::WideningProduct) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32) {
    const bool hasExpectedMemoryForm =
        description.operation ==
                RVVSelectedBodyOperationKind::
                    WideningProductReduceDequantClampF32
            ? description.memoryForm ==
                  RVVSelectedBodyMemoryForm::
                      UnitStrideWideningProductReduceDequantClampF32
            : description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
    if (!hasExpectedMemoryForm)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    if (description.operation ==
        RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32)
      return RVVSelectedBodyTargetLeafProfile{
          "",
          getRVVSelectedBodyFloatCompareIntrinsicForPredicate(
              description.comparePredicateKind, configProfile.sew,
              configProfile.lmul),
          getRVVSelectedBodyFloatSelectIntrinsic(configProfile.sew,
                                                 configProfile.lmul),
          configProfile.rhsBroadcastIntrinsic};
    // The deferred-wide dequant terminal kind shares the narrow dequant leaf
    // profile: the dequant scale splat (rhsBroadcastIntrinsic), no arithmetic /
    // compare / select leaf (those are dedicated structural ops).
    if (description.operation ==
            RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
        description.operation ==
            RVVSelectedBodyOperationKind::
                WideningProductDeferredAccumulateReduceDequantizeF32)
      return RVVSelectedBodyTargetLeafProfile{
          "", "", "", configProfile.rhsBroadcastIntrinsic};
    return RVVSelectedBodyTargetLeafProfile{"", "", "", ""};
  }

  if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);

  return RVVSelectedBodyTargetLeafProfile{
      getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                           configProfile),
      "", "", ""};
}

llvm::Expected<RVVSelectedBodyRouteProfile>
deriveRVVSelectedBodyRouteProfile(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyRouteProfile profile;
  profile.operation = getRVVSelectedBodyOperationProfile(description.operation);

  llvm::Expected<RVVSelectedBodyConfigProfile> config =
      deriveRVVSelectedBodyConfigProfile(description);
  if (!config)
    return config.takeError();
  profile.config = *config;

  llvm::Expected<RVVSelectedBodyTargetLeafProfile> targetLeaves =
      deriveRVVSelectedBodyTargetLeafProfile(description, profile.operation,
                                            profile.config);
  if (!targetLeaves)
    return targetLeaves.takeError();
  profile.targetLeaves = *targetLeaves;
  return profile;
}

llvm::Error requireRouteDescriptionText(llvm::StringRef context,
                                        llvm::StringRef field,
                                        llvm::StringRef value) {
  if (!value.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(llvm::Twine(context) + " " + field +
                                        " must be provider-derived and "
                                        "non-empty");
}

llvm::Error requireRouteDescriptionField(llvm::StringRef context,
                                         llvm::StringRef field,
                                         llvm::StringRef actual,
                                         llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must mirror selected-body route profile fact '" + expected +
      "' but was '" + actual + "'");
}

llvm::Expected<std::string> requireRVVCompositeResourceStringFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires realized composite resource facts on a concrete tcrv_rvv "
        "operation");
  mlir::StringAttr attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires realized composite resource string fact '" + attrName +
        "' before provider route construction");
  return attr.getValue().str();
}

llvm::Expected<std::int64_t> requireRVVCompositeResourceIntegerFact(
    mlir::Operation *op, llvm::StringRef context, llvm::StringRef attrName) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires realized composite resource facts on a concrete tcrv_rvv "
        "operation");
  mlir::IntegerAttr attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires realized composite resource integer fact '" + attrName +
        "' before provider route construction");
  return attr.getInt();
}

llvm::Error requireRVVCompositeResourceStringField(
    llvm::StringRef context, llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " composite resource " + field +
      " must mirror realized/provider-derived fact '" + expected +
      "' but was '" + actual + "'");
}

llvm::Error requireRVVCompositeResourceIntegerField(
    llvm::StringRef context, llvm::StringRef field, std::int64_t actual,
    std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " composite resource " + field +
      " must mirror realized/provider-derived fact " + llvm::Twine(expected) +
      " but was " + llvm::Twine(actual));
}

RVVCompositeGatherMAccScatterResourceSelection
deriveExpectedRVVCompositeGatherMAccScatterResourceSelection(
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedTargetCapabilityFacts &targetFacts) {
  RVVCompositeGatherMAccScatterResourceSelection selection;
  selection.hasSelection = true;
  selection.candidateSetID = kRVVCompositeResourceCandidateSet.str();
  selection.selectedCandidateID =
      kRVVCompositeResourceSelectedCandidate.str();
  selection.selectionReason = kRVVCompositeResourceSelectionReason.str();
  selection.legalityScope = kRVVCompositeResourceLegalityScope.str();
  selection.operation = kRVVCompositeResourceOperation.str();
  selection.memoryForm = kRVVCompositeResourceMemoryForm.str();
  selection.sew = description.sew;
  selection.lmul = description.lmul.str();
  selection.tailPolicy = description.tailPolicy.str();
  selection.maskPolicy = description.maskPolicy.str();
  selection.vlPolicy = kRVVGearboxRuntimeAVLSingleSetVLPolicy.str();
  selection.accumulatorLayout =
      kRVVCompositeResourceAccumulatorLayout.str();
  selection.unrollFactor = kRVVCompositeResourceStaticUnroll;
  selection.pipelineIntent = kRVVCompositeResourcePipelineIntent.str();
  selection.prefetchIntent = kRVVCompositeResourcePrefetchIntent.str();
  selection.vsetvlRegionCount = kRVVCompositeResourceVSetVLRegions;
  selection.peakLiveVectorGroups =
      kRVVCompositeResourcePeakLiveVectorGroups;
  selection.vectorRegisterBudget =
      kRVVCompositeResourceVectorRegisterBudget;
  selection.runtimeAVLSource = description.runtimeAVLASource.str();
  selection.runtimeABIOrder = description.runtimeABIOrder.str();
  selection.targetCapabilityProviderMirror = targetFacts.providerMirror;
  selection.targetCapabilityLegalityMirror = targetFacts.legalityMirror;
  selection.isLegal = true;
  selection.rejectionReason =
      kRVVCompositeResourceNoRejectionReason.str();
  return selection;
}

llvm::Expected<RVVCompositeGatherMAccScatterResourceSelection>
deriveRVVCompositeGatherMAccScatterResourceSelectionFromRealizedFacts(
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedTargetCapabilityFacts &targetFacts, mlir::Operation *op,
    llvm::StringRef context) {
  RVVCompositeGatherMAccScatterResourceSelection expected =
      deriveExpectedRVVCompositeGatherMAccScatterResourceSelection(
          description, targetFacts);
  RVVCompositeGatherMAccScatterResourceSelection selection;
  selection.hasSelection = true;

#define TCRV_READ_COMPOSITE_STRING(Field, AttrName)                           \
  {                                                                           \
    llvm::Expected<std::string> value =                                       \
        requireRVVCompositeResourceStringFact(op, context, AttrName);         \
    if (!value)                                                               \
      return value.takeError();                                               \
    selection.Field = std::move(*value);                                      \
  }
#define TCRV_READ_COMPOSITE_INTEGER(Field, AttrName)                          \
  {                                                                           \
    llvm::Expected<std::int64_t> value =                                      \
        requireRVVCompositeResourceIntegerFact(op, context, AttrName);        \
    if (!value)                                                               \
      return value.takeError();                                               \
    selection.Field = *value;                                                 \
  }

  TCRV_READ_COMPOSITE_STRING(candidateSetID,
                             kRVVCompositeResourceCandidateSetAttrName);
  TCRV_READ_COMPOSITE_STRING(selectedCandidateID,
                             kRVVCompositeResourceSelectedCandidateAttrName);
  TCRV_READ_COMPOSITE_STRING(selectionReason,
                             kRVVCompositeResourceSelectionReasonAttrName);
  TCRV_READ_COMPOSITE_STRING(legalityScope,
                             kRVVCompositeResourceLegalityScopeAttrName);
  TCRV_READ_COMPOSITE_STRING(operation,
                             kRVVCompositeResourceOperationAttrName);
  TCRV_READ_COMPOSITE_STRING(memoryForm,
                             kRVVCompositeResourceMemoryFormAttrName);
  TCRV_READ_COMPOSITE_INTEGER(sew, kRVVCompositeResourceSEWAttrName);
  TCRV_READ_COMPOSITE_STRING(lmul, kRVVCompositeResourceLMULAttrName);
  TCRV_READ_COMPOSITE_STRING(tailPolicy,
                             kRVVCompositeResourceTailPolicyAttrName);
  TCRV_READ_COMPOSITE_STRING(maskPolicy,
                             kRVVCompositeResourceMaskPolicyAttrName);
  TCRV_READ_COMPOSITE_STRING(vlPolicy,
                             kRVVCompositeResourceVLPolicyAttrName);
  TCRV_READ_COMPOSITE_STRING(
      accumulatorLayout, kRVVCompositeResourceAccumulatorLayoutAttrName);
  TCRV_READ_COMPOSITE_INTEGER(unrollFactor,
                              kRVVCompositeResourceUnrollFactorAttrName);
  TCRV_READ_COMPOSITE_STRING(pipelineIntent,
                             kRVVCompositeResourcePipelineIntentAttrName);
  TCRV_READ_COMPOSITE_STRING(prefetchIntent,
                             kRVVCompositeResourcePrefetchIntentAttrName);
  TCRV_READ_COMPOSITE_INTEGER(
      vsetvlRegionCount, kRVVCompositeResourceVSetVLRegionCountAttrName);
  TCRV_READ_COMPOSITE_INTEGER(
      peakLiveVectorGroups,
      kRVVCompositeResourcePeakLiveVectorGroupsAttrName);
  TCRV_READ_COMPOSITE_INTEGER(
      vectorRegisterBudget,
      kRVVCompositeResourceVectorRegisterBudgetAttrName);
  TCRV_READ_COMPOSITE_STRING(
      runtimeAVLSource, kRVVCompositeResourceRuntimeAVLSourceAttrName);
  TCRV_READ_COMPOSITE_STRING(
      runtimeABIOrder, kRVVCompositeResourceRuntimeABIOrderAttrName);
  TCRV_READ_COMPOSITE_STRING(
      targetCapabilityProviderMirror,
      kRVVCompositeResourceTargetCapabilityProviderMirrorAttrName);
  TCRV_READ_COMPOSITE_STRING(
      targetCapabilityLegalityMirror,
      kRVVCompositeResourceTargetCapabilityLegalityMirrorAttrName);
  std::string legality;
  {
    llvm::Expected<std::string> value = requireRVVCompositeResourceStringFact(
        op, context, kRVVCompositeResourceLegalityAttrName);
    if (!value)
      return value.takeError();
    legality = std::move(*value);
  }
  selection.isLegal = legality == kRVVCompositeResourceLegal;
  TCRV_READ_COMPOSITE_STRING(
      rejectionReason, kRVVCompositeResourceRejectionReasonAttrName);

#undef TCRV_READ_COMPOSITE_STRING
#undef TCRV_READ_COMPOSITE_INTEGER

  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "candidate set", selection.candidateSetID,
          expected.candidateSetID))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "selected candidate", selection.selectedCandidateID,
          expected.selectedCandidateID))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "selection reason", selection.selectionReason,
          expected.selectionReason))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "legality scope", selection.legalityScope,
          expected.legalityScope))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "operation", selection.operation, expected.operation))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "memory form", selection.memoryForm, expected.memoryForm))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceIntegerField(
          context, "SEW", selection.sew, expected.sew))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "LMUL", selection.lmul, expected.lmul))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "tail policy", selection.tailPolicy,
          expected.tailPolicy))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "mask policy", selection.maskPolicy,
          expected.maskPolicy))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "VL policy", selection.vlPolicy, expected.vlPolicy))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "accumulator layout", selection.accumulatorLayout,
          expected.accumulatorLayout))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceIntegerField(
          context, "unroll factor", selection.unrollFactor,
          expected.unrollFactor))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "pipeline intent", selection.pipelineIntent,
          expected.pipelineIntent))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "prefetch intent", selection.prefetchIntent,
          expected.prefetchIntent))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceIntegerField(
          context, "vsetvl region count", selection.vsetvlRegionCount,
          expected.vsetvlRegionCount))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceIntegerField(
          context, "peak live vector groups",
          selection.peakLiveVectorGroups,
          expected.peakLiveVectorGroups))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceIntegerField(
          context, "vector register budget",
          selection.vectorRegisterBudget,
          expected.vectorRegisterBudget))
    return std::move(error);
  if (selection.peakLiveVectorGroups > selection.vectorRegisterBudget)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " composite resource peak live vector-group estimate " +
        llvm::Twine(selection.peakLiveVectorGroups) +
        " exceeds vector register budget " +
        llvm::Twine(selection.vectorRegisterBudget));
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "runtime AVL source", selection.runtimeAVLSource,
          expected.runtimeAVLSource))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "runtime ABI order", selection.runtimeABIOrder,
          expected.runtimeABIOrder))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "target capability provider mirror",
          selection.targetCapabilityProviderMirror,
          expected.targetCapabilityProviderMirror))
    return std::move(error);
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "target capability legality mirror",
          selection.targetCapabilityLegalityMirror,
          expected.targetCapabilityLegalityMirror))
    return std::move(error);
  if (legality != kRVVCompositeResourceLegal)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " composite resource legality must be 'legal' before provider route "
        "construction but was '" +
        legality + "'");
  if (llvm::Error error = requireRVVCompositeResourceStringField(
          context, "rejection reason", selection.rejectionReason,
          expected.rejectionReason))
    return std::move(error);
  return selection;
}

llvm::Error verifyRVVCompositeGatherMAccScatterResourceDescriptionSelection(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const RVVCompositeGatherMAccScatterResourceSelection &selection =
      description.compositeGatherMAccScatterResourceSelection;
  if (description.operation !=
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    if (!selection.hasSelection)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " non-composite route must not carry composite gather-MAcc-scatter "
        "resource selection facts");
  }
  if (!selection.hasSelection || !selection.isLegal ||
      selection.selectedCandidateID.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime-scalar computed-mask indexed gather-MAcc-scatter route "
        "requires a provider-consumed legal composite resource candidate "
        "before route construction");
  if (selection.peakLiveVectorGroups > selection.vectorRegisterBudget)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " runtime-scalar computed-mask indexed gather-MAcc-scatter route "
        "rejects composite resource peak live vector-group estimate " +
        llvm::Twine(selection.peakLiveVectorGroups) +
        " above vector register budget " +
        llvm::Twine(selection.vectorRegisterBudget));
  return llvm::Error::success();
}

llvm::StringRef stringifyRVVTailPolicy(tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV tail policy");
}

llvm::StringRef stringifyRVVMaskPolicy(tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV mask policy");
}

llvm::Expected<llvm::StringRef>
getRVVSelectedBodyElementTypeNameForSEW(std::int64_t sew,
                                        llvm::StringRef context) {
  if (sew == tcrv::rvv::getRVVSEW16Bits())
    return llvm::StringRef("i16");
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return llvm::StringRef("i32");
  if (sew == tcrv::rvv::getRVVSEW64Bits())
    return llvm::StringRef("i64");
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " typed config facts require a supported integer element type for SEW " +
      llvm::Twine(sew));
}

llvm::Expected<RVVSelectedBodyTypedConfigFacts>
deriveRVVSelectedBodyTypedConfigFacts(
    const tcrv::rvv::RVVCompileTimeConfig &config,
    const RVVSelectedBodyConfigProfile &configProfile,
    llvm::StringRef context) {
  if (configProfile.sew != config.sew || configProfile.lmul != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " typed config facts require the provider config profile to mirror "
        "realized tcrv_rvv.setvl SEW/LMUL");
  if (!config.policy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " typed config facts require an explicit realized RVV policy");
  if (!configProfile.configContract)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " typed config facts require a selected-body config contract");
  if (configProfile.vectorTypeName.empty() ||
      configProfile.vectorCType.empty() || configProfile.vlCType.empty() ||
      configProfile.setVLIntrinsic.empty() ||
      configProfile.vectorLoadIntrinsic.empty() ||
      configProfile.rhsBroadcastIntrinsic.empty() ||
      configProfile.storeIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " typed config facts require provider-derived vector type, C type, "
        "VL type, setvl, load, scalar-splat, and store leaves");

  llvm::StringRef elementTypeNameStorage;
  if (configProfile.vectorTypeName.contains("!tcrv_rvv.vector<f32")) {
    elementTypeNameStorage = getRVVSelectedBodyFloatElementTypeName(config.sew);
    if (elementTypeNameStorage.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " typed config facts require a supported f32 element type for SEW " +
          llvm::Twine(config.sew));
  } else if (configProfile.vectorTypeName.contains("!tcrv_rvv.vector<u")) {
    elementTypeNameStorage =
        getRVVSelectedBodyUnsignedIntegerElementTypeName(config.sew);
    if (elementTypeNameStorage.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " typed config facts require a supported unsigned integer element "
          "type for SEW " +
          llvm::Twine(config.sew));
  } else {
    llvm::Expected<llvm::StringRef> elementTypeName =
        getRVVSelectedBodyElementTypeNameForSEW(config.sew, context);
    if (!elementTypeName)
      return elementTypeName.takeError();
    elementTypeNameStorage = *elementTypeName;
  }

  RVVSelectedBodyTypedConfigFacts facts;
  facts.factsID = "rvv-selected-body-typed-config-facts.v1";
  facts.elementTypeName = elementTypeNameStorage;
  facts.elementBitWidth = config.sew;
  facts.sew = config.sew;
  facts.lmul = config.lmul;
  facts.tailPolicy = stringifyRVVTailPolicy(config.policy.getTail());
  facts.maskPolicy = stringifyRVVMaskPolicy(config.policy.getMask());
  facts.configContractID = configProfile.configContract->configContractID;
  facts.vectorTypeName = configProfile.vectorTypeName;
  facts.indexVectorTypeName = configProfile.indexVectorTypeName;
  facts.maskTypeName = configProfile.maskTypeName;
  facts.vectorCType = configProfile.vectorCType;
  facts.indexVectorCType = configProfile.indexVectorCType;
  facts.maskCType = configProfile.maskCType;
  facts.vlCType = configProfile.vlCType;
  facts.setVLIntrinsic = configProfile.setVLIntrinsic;
  facts.vectorLoadIntrinsic = configProfile.vectorLoadIntrinsic;
  facts.indexLoadIntrinsic = configProfile.indexLoadIntrinsic;
  facts.indexScaleIntrinsic = configProfile.indexScaleIntrinsic;
  facts.indexedLoadIntrinsic = configProfile.indexedLoadIntrinsic;
  facts.indexedStoreIntrinsic = configProfile.indexedStoreIntrinsic;
  facts.stridedLoadIntrinsic = configProfile.stridedLoadIntrinsic;
  facts.scalarSplatIntrinsic = configProfile.rhsBroadcastIntrinsic;
  facts.maskedLoadIntrinsic =
      getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(configProfile);
  facts.storeIntrinsic = configProfile.storeIntrinsic;
  facts.maskedStoreIntrinsic =
      getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(configProfile);
  facts.stridedStoreIntrinsic = configProfile.stridedStoreIntrinsic;
  return facts;
}

const RVVSelectedBodyConstructionRoute &
getRVVSelectedBodyConstructionRouteOrDie(RVVSelectedBodyOperationKind op) {
  const RVVSelectedBodyOperationProfile &profile =
      getRVVSelectedBodyOperationProfile(op);
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          profile.operationMnemonic);
  if (!route) {
    std::string message = llvm::toString(route.takeError());
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return **route;
}

bool variantContainsExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getDialectNamespace() == "tcrv_rvv")
      found = true;
  });
  return found;
}

llvm::Error requireRVVVariantLegality(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVEmitCRouteProviderError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();

  return makeRVVEmitCRouteProviderError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

llvm::Error validateRVVSelectedBodyVectorTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  if (!value)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be present before route construction");
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.vector value before route construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != config.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element width " + llvm::Twine(integerElementType.getWidth()) +
        " to match selected config SEW " + llvm::Twine(config.sew));
  if (vectorType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + vectorType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyF32VectorTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  if (!value)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be present before route construction");
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.vector value before route construction");

  auto floatElementType =
      llvm::dyn_cast<mlir::FloatType>(vectorType.getElementType());
  if (!floatElementType || !floatElementType.isF32())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be f32");
  if (config.sew != tcrv::rvv::getRVVFirstSliceSEWBits())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " selected config SEW32 for f32 dequantization vectors");
  if (vectorType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + vectorType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyMaskTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto maskType = llvm::dyn_cast<tcrv::rvv::MaskType>(value.getType());
  if (!maskType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.mask value before route construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(maskType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != config.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element width " + llvm::Twine(integerElementType.getWidth()) +
        " to match selected config SEW " + llvm::Twine(config.sew));
  if (maskType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + maskType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyF32MaskTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto maskType = llvm::dyn_cast<tcrv::rvv::MaskType>(value.getType());
  if (!maskType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.mask value before route construction");

  auto floatElementType =
      llvm::dyn_cast<mlir::FloatType>(maskType.getElementType());
  if (!floatElementType || !floatElementType.isF32())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be f32");
  if (config.sew != tcrv::rvv::getRVVFirstSliceSEWBits())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " selected config SEW32 for f32 compare/select masks");
  if (maskType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + maskType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto vectorType =
      llvm::dyn_cast<tcrv::rvv::IndexVectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.index_vector value before route "
        "construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != 32)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " index element width 32 for the bounded indexed gather route");
  if (vectorType.getLmul() != config.lmul ||
      vectorType.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + vectorType.getLmul() +
        "' to match selected config LMUL m1 for indexed gather");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyTypedConfigFacts(
    RVVSelectedBodyRouteSlice &slice,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarSplatStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "runtime scalar splat result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "runtime scalar splat stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI16ToI32) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    if (slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI16ToI32) {
      sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
      sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    } else {
      sourceConfig.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
      sourceConfig.lmul = tcrv::rvv::getRVVLMULM1();
    }
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "conversion source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "conversion result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "conversion stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::DequantizeI32ToF32) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "dequantization source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.arithmeticResult, "dequantization result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.storeValue, "dequantization stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "dequant-clamp source i32 vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.dequantizeOp.getResult(),
            "dequant-clamp dequantized f32 vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.lowerBoundValue,
            "dequant-clamp lower-bound splat vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.upperBoundValue,
            "dequant-clamp upper-bound splat vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32MaskTypeAgainstConfig(
            slice.compareMask, "dequant-clamp lower predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.lowerClampedValue,
            "dequant-clamp lower-clamped f32 vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32MaskTypeAgainstConfig(
            slice.secondaryCompareMask,
            "dequant-clamp upper predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.arithmeticResult,
            "dequant-clamp selected f32 result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.storeValue, "dequant-clamp stored f32 vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "widening macc lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "widening macc rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue, "widening macc accumulator vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "widening macc result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "widening macc stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningProduct) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW8Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF4();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "widening product lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "widening product rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "widening product result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "widening product stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32) {
    const bool isWideningProductReduceDequantize =
        slice.arithmeticKind ==
        RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32;
    const bool isWideningProductReduceDequantClamp =
        slice.arithmeticKind ==
        RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
    const bool hasProductReductionDequantization =
        isWideningProductReduceDequantize ||
        isWideningProductReduceDequantClamp;
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW8Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF4();
    sourceConfig.policy = config.policy;
    tcrv::rvv::RVVCompileTimeConfig productConfig;
    productConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    productConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    productConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "product-reduction lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "product-reduction rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            productSlotResult(slice),
            "product-reduction i16 product vector", productConfig))
      return error;
    mlir::Value reductionResult = slice.standaloneReduceOp
                                      ? slice.standaloneReduceOp.getOperation()
                                            ->getResult(0)
                                      : slice.arithmeticResult;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            reductionResult, "product-reduction scalar result vector", config))
      return error;
    if (hasProductReductionDequantization) {
      if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
              slice.dequantizeOp.getResult(),
              "product-reduction dequantized f32 vector", config))
        return error;
      if (isWideningProductReduceDequantClamp) {
        if (llvm::Error error =
                validateRVVSelectedBodyF32VectorTypeAgainstConfig(
                    slice.lowerBoundValue,
                    "product-reduction dequant-clamp lower-bound splat vector",
                    config))
          return error;
        if (llvm::Error error =
                validateRVVSelectedBodyF32VectorTypeAgainstConfig(
                    slice.upperBoundValue,
                    "product-reduction dequant-clamp upper-bound splat vector",
                    config))
          return error;
        if (llvm::Error error = validateRVVSelectedBodyF32MaskTypeAgainstConfig(
                slice.compareMask,
                "product-reduction dequant-clamp lower predicate mask", config))
          return error;
        if (llvm::Error error =
                validateRVVSelectedBodyF32VectorTypeAgainstConfig(
                    slice.lowerClampedValue,
                    "product-reduction dequant-clamp lower-clamped f32 vector",
                    config))
          return error;
        if (llvm::Error error = validateRVVSelectedBodyF32MaskTypeAgainstConfig(
                slice.secondaryCompareMask,
                "product-reduction dequant-clamp upper predicate mask", config))
          return error;
        if (llvm::Error error =
                validateRVVSelectedBodyF32VectorTypeAgainstConfig(
                    slice.secondarySelectOp.getSelected(),
                    "product-reduction dequant-clamp selected f32 result vector",
                    config))
          return error;
        if (llvm::Error error =
                validateRVVSelectedBodyF32VectorTypeAgainstConfig(
                    slice.storeValue,
                    "product-reduction dequant-clamp stored f32 vector",
                    config))
          return error;
      } else {
        if (llvm::Error error =
                validateRVVSelectedBodyF32VectorTypeAgainstConfig(
                    slice.arithmeticResult,
                    "product-reduction dequantized result vector", config))
          return error;
        if (llvm::Error error =
                validateRVVSelectedBodyF32VectorTypeAgainstConfig(
                    slice.storeValue,
                    "product-reduction dequantized stored vector", config))
          return error;
      }
    } else if (llvm::Error error =
                   validateRVVSelectedBodyVectorTypeAgainstConfig(
                       slice.storeValue,
                       "product-reduction stored scalar result vector",
                       config)) {
      return error;
    }
    return llvm::Error::success();
  }

  // The deferred-wide (N3 resource-aware max-legal-LMUL winner) product-reduce-
  // dequant realization of the SAME logical op. UNLIKE the narrow path (whose
  // setvl carries the i32m1/f32m1 RESULT config), the deferred-wide setvl carries
  // the SOURCE strip config (sew8/m2), so the result config (sew32/m1) is derived
  // explicitly here. The wide LMUL ladder is fixed by the structural ops: source
  // i8m2, product i16m4, deferred accumulate i32m8, ONE trailing reduce to i32m1,
  // dequant to f32m1. Every config below mirrors those typed result types (I5) --
  // nothing is inferred from a name/route id.
  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          WideningProductDeferredAccumulateReduceDequantizeF32) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW8Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULM2();
    sourceConfig.policy = config.policy;
    tcrv::rvv::RVVCompileTimeConfig productConfig;
    productConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    productConfig.lmul = tcrv::rvv::getRVVLMULM4();
    productConfig.policy = config.policy;
    tcrv::rvv::RVVCompileTimeConfig accumulateConfig;
    accumulateConfig.sew = tcrv::rvv::getRVVSEW32Bits();
    accumulateConfig.lmul = tcrv::rvv::getRVVLMULM8();
    accumulateConfig.policy = config.policy;
    tcrv::rvv::RVVCompileTimeConfig resultConfig;
    resultConfig.sew = tcrv::rvv::getRVVSEW32Bits();
    resultConfig.lmul = tcrv::rvv::getRVVLMULM1();
    resultConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "deferred-wide product-reduction lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "deferred-wide product-reduction rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            productSlotResult(slice),
            "deferred-wide product-reduction i16m4 product vector",
            productConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.wideningAccumulateOp.getResult(),
            "deferred-wide product-reduction i32m8 accumulate vector",
            accumulateConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.standaloneReduceOp.getResult(),
            "deferred-wide product-reduction i32m1 reduce result vector",
            resultConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.dequantizeOp.getResult(),
            "deferred-wide product-reduction dequantized f32m1 vector",
            resultConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.arithmeticResult,
            "deferred-wide product-reduction dequantized result vector",
            resultConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.storeValue,
            "deferred-wide product-reduction dequantized stored vector",
            resultConfig))
      return error;
    return llvm::Error::success();
  }

  // The deferred-wide (N3 resource-aware max-legal-LMUL winner) realization of the
  // SAME logical widening_dot_reduce_add op (2nd kernel family, signed i16
  // dot-reduce). UNLIKE the narrow path (whose setvl carries the i32m1 RESULT
  // config), the deferred-wide setvl carries the SOURCE strip config (sew16/<L>),
  // so the result config (sew32/m1) is derived explicitly here. The wide LMUL
  // ladder is fixed by the structural ops: source i16<L>, product i32<W>, deferred
  // accumulate i32<W> (same-width vadd.vv), ONE trailing reduce to i32m1, store
  // i32m1. The source/accumulate LMUL is READ from the realized typed ops (the
  // budget-selected rung), NOT pinned to m4/m8: the budget-driven LMUL-width
  // ablation realizes the wide m4/m8 rung at the default budget and a narrower
  // m2/m4 or mf2/m1 rung at a constrained budget. Every config below mirrors the
  // typed result types (I5) -- nothing is inferred from a name/route id.
  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          WideningProductDeferredDotAccumulateReduceAdd) {
    // Read the realized source/accumulator LMUL from the deferred-accumulate
    // result type (the budget-selected i32 accumulator <W>); the i16 source <L>
    // is <W> un-widened one step (the i32 product is one EMUL step wider).
    auto accumulateType = llvm::dyn_cast_or_null<tcrv::rvv::VectorType>(
        slice.deferredAccumulateOp
            ? slice.deferredAccumulateOp.getResult().getType()
            : mlir::Type());
    if (!accumulateType)
      return makeRVVEmitCRouteProviderError(
          "deferred-wide dot-reduction requires a typed i32 accumulate result");
    llvm::StringRef accumulatorLMUL = accumulateType.getLmul();
    llvm::StringRef sourceLMUL;
    for (llvm::StringRef candidate : {"mf2", "m1", "m2", "m4"}) {
      if (getRVVNextWiderLMUL(candidate) == accumulatorLMUL) {
        sourceLMUL = candidate;
        break;
      }
    }
    if (sourceLMUL.empty())
      return makeRVVEmitCRouteProviderError(
          "deferred-wide dot-reduction accumulator LMUL is not a legal i16->i32 "
          "single-widening rung");
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = sourceLMUL;
    sourceConfig.policy = config.policy;
    tcrv::rvv::RVVCompileTimeConfig accumulateConfig;
    accumulateConfig.sew = tcrv::rvv::getRVVSEW32Bits();
    accumulateConfig.lmul = accumulatorLMUL;
    accumulateConfig.policy = config.policy;
    tcrv::rvv::RVVCompileTimeConfig resultConfig;
    resultConfig.sew = tcrv::rvv::getRVVSEW32Bits();
    resultConfig.lmul = tcrv::rvv::getRVVLMULM1();
    resultConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "deferred-wide dot-reduction lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "deferred-wide dot-reduction rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            productSlotResult(slice),
            "deferred-wide dot-reduction i32 product vector",
            accumulateConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.deferredAccumulateOp.getResult(),
            "deferred-wide dot-reduction i32 accumulate vector",
            accumulateConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.standaloneReduceOp.getResult(),
            "deferred-wide dot-reduction i32m1 reduce result vector",
            resultConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "deferred-wide dot-reduction result vector", resultConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "deferred-wide dot-reduction stored vector", resultConfig))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "widening dot-reduction lhs source vector",
            sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "widening dot-reduction rhs source vector",
            sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "widening dot-reduction result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "widening dot-reduction stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "strided-input dot-reduction lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "strided-input dot-reduction rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "strided-input dot-reduction result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "strided-input dot-reduction stored vector",
            config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "computed-mask dot-reduction compare lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "computed-mask dot-reduction compare rhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "computed-mask dot-reduction predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotLHSValue,
            "computed-mask dot-reduction lhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotRHSValue,
            "computed-mask dot-reduction rhs source vector", sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "computed-mask dot-reduction result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "computed-mask dot-reduction stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd) {
    tcrv::rvv::RVVCompileTimeConfig sourceConfig;
    sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
    sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
    sourceConfig.policy = config.policy;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "computed-mask strided-input dot-reduction compare lhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "computed-mask strided-input dot-reduction compare rhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "computed-mask strided-input dot-reduction predicate mask",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotLHSValue,
            "computed-mask strided-input dot-reduction lhs source vector",
            sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotRHSValue,
            "computed-mask strided-input dot-reduction rhs source vector",
            sourceConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "computed-mask strided-input dot-reduction result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "computed-mask strided-input dot-reduction stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "computed-mask macc compare lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "computed-mask macc compare rhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "computed-mask macc predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotLHSValue,
            "computed-mask macc lhs payload vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotRHSValue,
            "computed-mask macc rhs payload vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue,
            "computed-mask macc accumulator vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "computed-mask macc result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "computed-mask macc stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "scalar-broadcast macc lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "scalar-broadcast macc RHS splat vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue,
            "scalar-broadcast macc accumulator vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "scalar-broadcast macc result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "scalar-broadcast macc stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "runtime scalar computed-mask macc compare lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "runtime scalar computed-mask macc splatted threshold vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "runtime scalar computed-mask macc predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotLHSValue,
            "runtime scalar computed-mask macc lhs payload vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotRHSValue,
            "runtime scalar computed-mask macc rhs payload vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue,
            "runtime scalar computed-mask macc accumulator vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "runtime scalar computed-mask macc result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "runtime scalar computed-mask macc stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "runtime scalar composite gather-MAcc-scatter compare lhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "runtime scalar composite gather-MAcc-scatter splatted threshold "
            "vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "runtime scalar composite gather-MAcc-scatter predicate mask",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
            slice.indexValue,
            "runtime scalar composite gather-MAcc-scatter index vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.sourceValue,
            "runtime scalar composite gather-MAcc-scatter gathered source "
            "vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.dotRHSValue,
            "runtime scalar composite gather-MAcc-scatter payload RHS vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue,
            "runtime scalar composite gather-MAcc-scatter accumulator vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "runtime scalar composite gather-MAcc-scatter MAcc result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "runtime scalar composite gather-MAcc-scatter scatter payload "
            "vector",
            config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedLoadUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "strided source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::UnitLoadStridedStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "unit-load source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "strided-store stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::IndexedGatherUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
            slice.indexValue, "indexed gather index vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "indexed gather loaded data vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::IndexedScatterUnitLoad) {
    if (llvm::Error error = validateRVVSelectedBodyIndexVectorTypeAgainstConfig(
            slice.indexValue, "indexed scatter index vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "indexed scatter source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "indexed-store stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::MaskedUnitLoadStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "masked memory source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue, "masked memory old destination vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.maskValue, "masked memory predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "masked movement result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "masked unit-stride stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "masked store payload source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.maskValue, "masked store predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "masked store payload vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "computed-mask compare lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "computed-mask compare rhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.sourceValue, "computed-mask active source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue,
            "computed-mask old destination vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "computed-mask predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "computed-mask movement result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "computed-mask unit-stride stored vector",
            config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskSelect) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "computed-mask select compare lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "computed-mask select compare rhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.trueValue, "computed-mask select true-value vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.falseValue, "computed-mask select false-value vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "computed-mask select predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult, "computed-mask selected result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "computed-mask select stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "runtime scalar compare/select lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "runtime scalar compare/select splatted threshold vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.trueValue,
            "runtime scalar compare/select true-value vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.falseValue,
            "runtime scalar compare/select false-value vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "runtime scalar compare/select predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "runtime scalar compare/select selected result vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "runtime scalar compare/select stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "runtime scalar dual-compare mask-and select lhs A vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "runtime scalar dual-compare mask-and select splatted threshold A "
            "vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.secondaryCompareLhs,
            "runtime scalar dual-compare mask-and select lhs B vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.secondaryCompareRhs,
            "runtime scalar dual-compare mask-and select splatted threshold B "
            "vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.trueValue,
            "runtime scalar dual-compare mask-and select true-value vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.falseValue,
            "runtime scalar dual-compare mask-and select false-value vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "runtime scalar dual-compare mask-and select predicate mask A",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.secondaryCompareMask,
            "runtime scalar dual-compare mask-and select predicate mask B",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.composedMask,
            "runtime scalar dual-compare mask-and select composed mask",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "runtime scalar dual-compare mask-and select selected result "
            "vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "runtime scalar dual-compare mask-and select stored vector",
            config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::F32ClampSelect) {
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.lhsValue, "f32 clamp/select input vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.lowerBoundValue,
            "f32 clamp/select lower-bound splat vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.upperBoundValue,
            "f32 clamp/select upper-bound splat vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32MaskTypeAgainstConfig(
            slice.compareMask, "f32 clamp/select lower predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.lowerClampedValue,
            "f32 clamp/select lower-clamped vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32MaskTypeAgainstConfig(
            slice.secondaryCompareMask,
            "f32 clamp/select upper predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.arithmeticResult, "f32 clamp/select selected result vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyF32VectorTypeAgainstConfig(
            slice.storeValue, "f32 clamp/select stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue,
            "runtime scalar computed-mask store lhs vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue,
            "runtime scalar computed-mask store splatted threshold vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.sourceValue,
            "runtime scalar computed-mask store payload source vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask,
            "runtime scalar computed-mask store predicate mask", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "runtime scalar computed-mask store payload vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.lhsValue, "computed-mask strided-store compare lhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.rhsValue, "computed-mask strided-store compare rhs vector",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.sourceValue,
            "computed-mask strided-store active source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "computed-mask strided-store predicate mask",
            config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "computed-mask strided-store payload vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue,
            "computed-mask strided-store stored vector", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field0LoadedValue, "segment2 field0 loaded vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field1LoadedValue, "segment2 field1 loaded vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field0Value, "segment2 field0 movement result", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field1Value, "segment2 field1 movement result", config))
      return error;
    return llvm::Error::success();
  }

  if (slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad) {
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field0Value, "segment2 field0 source vector", config))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.field1Value, "segment2 field1 source vector", config))
      return error;
    return llvm::Error::success();
  }

  if (isRVVSelectedBodyStandaloneReductionRouteOperation(
          slice.arithmeticKind)) {
    tcrv::rvv::RVVCompileTimeConfig scalarResultConfig = config;
    scalarResultConfig.lmul = tcrv::rvv::getRVVLMULM1();
    if (isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
            slice.arithmeticKind) ||
        isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
            slice.arithmeticKind)) {
      if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
              slice.lhsValue,
              "standalone reduction compare lhs vector", config))
        return error;
      if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
              slice.rhsValue,
              "standalone reduction compare rhs vector", config))
        return error;
      if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
              slice.sourceValue,
              "standalone reduction active source vector", config))
        return error;
      if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
              slice.compareMask,
              "standalone reduction predicate mask", config))
        return error;
    } else {
      if (slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd) {
        tcrv::rvv::RVVCompileTimeConfig sourceConfig;
        sourceConfig.sew = tcrv::rvv::getRVVSEW16Bits();
        sourceConfig.lmul = tcrv::rvv::getRVVLMULMF2();
        sourceConfig.policy = config.policy;
        if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
                slice.lhsValue,
                "widening standalone reduction source vector", sourceConfig))
          return error;
      } else if (llvm::Error error =
                     validateRVVSelectedBodyVectorTypeAgainstConfig(
                         slice.lhsValue,
                         "standalone reduction input vector", config)) {
        return error;
      }
    }
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.arithmeticResult,
            "standalone reduction scalar result vector", scalarResultConfig))
      return error;
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.storeValue, "standalone reduction stored scalar result vector",
            scalarResultConfig))
      return error;
    return llvm::Error::success();
  }

  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.lhsValue, "lhs vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.rhsValue, "rhs vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.arithmeticResult, "compute result vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.storeValue, "stored vector", config))
    return error;
  if (slice.compareOp)
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "compare mask", config))
      return error;
  if (slice.maskedBinaryOp)
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.maskedPassthrough, "masked passthrough vector", config))
      return error;
  if (slice.maccOp)
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.accumulatorValue, "multiply-accumulate accumulator vector",
            config))
      return error;
  return llvm::Error::success();
}

std::string formatRuntimeABIExpectedRoles(
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](support::RuntimeABIParameterRole role) {
        stream << "'" << support::stringifyRuntimeABIParameterRole(role) << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return expected;
}

bool variantRequiresRVVExecABIBindings(tcrv::exec::VariantOp variant) {
  if (!variant)
    return false;
  auto required =
      variant->getAttrOfType<mlir::BoolAttr>(kRVVRequireExecABIBindingsAttrName);
  return required && required.getValue();
}

bool runtimeABIParametersMatchRuntimeABIValue(
    const support::RuntimeABIParameter &parameter,
    tcrv::rvv::RuntimeABIValueOp binding) {
  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  return role && ownership && parameter.role == *role &&
         parameter.ownership == *ownership &&
         parameter.cName == binding.getCName() &&
         parameter.cType == binding.getCType();
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
findRuntimeABIValueForParameter(tcrv::exec::VariantOp variant,
                                const support::RuntimeABIParameter &parameter,
                                llvm::StringRef context) {
  tcrv::rvv::RuntimeABIValueOp matched;
  unsigned count = 0;
  variant.getBody().walk([&](tcrv::rvv::RuntimeABIValueOp binding) {
    if (!runtimeABIParametersMatchRuntimeABIValue(parameter, binding))
      return;
    matched = binding;
    ++count;
  });

  if (count == 1)
    return matched;

  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " requires exactly one tcrv_rvv.runtime_abi_value matching ABI "
      "parameter '" +
      parameter.cName + "' with role '" +
      support::stringifyRuntimeABIParameterRole(parameter.role) +
      "' before exec-envelope ABI binding validation; found " +
      llvm::Twine(count));
}

llvm::Expected<std::string>
summarizeRVVSelectedBodyExecABIBindings(
    tcrv::exec::VariantOp variant,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    llvm::StringRef context) {
  const bool required = variantRequiresRVVExecABIBindings(variant);
  llvm::SmallVector<std::string, 8> entries;

  for (const support::RuntimeABIParameter &parameter : runtimeABIParameters) {
    llvm::Expected<tcrv::rvv::RuntimeABIValueOp> binding =
        findRuntimeABIValueForParameter(variant, parameter, context);
    if (!binding)
      return binding.takeError();

    auto execBinding = (*binding)->getAttrOfType<mlir::FlatSymbolRefAttr>(
        kRVVRuntimeABIExecBindingAttrName);
    if (!execBinding) {
      if (required)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " requires tcrv_rvv.runtime_abi_value '" + parameter.cName +
            "' with role '" +
            support::stringifyRuntimeABIParameterRole(parameter.role) +
            "' to carry exec_binding to a tcrv.exec ABI declaration");
      continue;
    }

    std::string entry;
    llvm::raw_string_ostream stream(entry);
    stream << parameter.cName << "="
           << support::stringifyRuntimeABIParameterRole(parameter.role)
           << "->@" << execBinding.getValue();
    stream.flush();
    entries.push_back(std::move(entry));
  }

  std::string summary;
  llvm::raw_string_ostream stream(summary);
  llvm::interleave(
      entries, [&](const std::string &entry) { stream << entry; },
      [&] { stream << ";"; });
  stream.flush();
  return summary;
}

llvm::Expected<support::RuntimeABIParameter>
getRuntimeABIParameterBindingFromValue(
    mlir::Value value, llvm::StringRef context,
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must be defined by explicit tcrv_rvv.runtime_abi_value before "
        "RVV EmitC route construction");
  if (llvm::Error error =
          verifyRVVRuntimeABIValueRoleOpInterface(binding.getOperation()))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI role '" +
        binding.getRole() + "'");
  if (!llvm::is_contained(expectedRoles, *role)) {
    std::string expectedRolesText =
        formatRuntimeABIExpectedRoles(expectedRoles);
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role " +
        expectedRolesText + "; got '" + binding.getRole() + "'");
  }

  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  if (!ownership)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI ownership '" +
        binding.getOwnership() + "'");

  return support::RuntimeABIParameter(binding.getCName(), binding.getCType(),
                                      *role, *ownership);
}

llvm::Error
assignRVVGenericLoadBinding(RVVSelectedBodyRouteSlice &slice,
                            tcrv::rvv::LoadOp load,
                            const support::RuntimeABIParameter &parameter) {
  if (parameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique lhs-input-buffer load");
    slice.lhsGenericLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    if (slice.rhsLoadOperation && slice.rhsScalarSplat &&
        !slice.secondaryCompareLhsLoadOperation) {
      slice.secondaryCompareLhsGenericLoad = load;
      slice.secondaryCompareLhsLoadOperation = load.getOperation();
      slice.secondaryCompareLhsBuffer = load.getBuffer();
      slice.secondaryCompareLhs = load.getLoaded();
      slice.secondaryCompareLhsABI = parameter;
      return llvm::Error::success();
    }
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique rhs-input-buffer load");
    slice.rhsGenericLoad = load;
    slice.rhsLoadOperation = load.getOperation();
    slice.rhsBuffer = load.getBuffer();
    slice.rhsValue = load.getLoaded();
    slice.rhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::TrueValueInputBuffer) {
    if (slice.trueValueLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask select route requires a unique "
          "true-value-input-buffer load");
    slice.trueValueGenericLoad = load;
    slice.trueValueLoadOperation = load.getOperation();
    slice.trueValueBuffer = load.getBuffer();
    slice.trueValue = load.getLoaded();
    slice.trueValueABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::FalseValueInputBuffer) {
    if (slice.falseValueLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask select route requires a unique "
          "false-value-input-buffer load");
    slice.falseValueGenericLoad = load;
    slice.falseValueLoadOperation = load.getOperation();
    slice.falseValueBuffer = load.getBuffer();
    slice.falseValue = load.getLoaded();
    slice.falseValueABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::DotLHSInputBuffer) {
    if (slice.dotLHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask dot-reduction route requires a unique "
          "dot-lhs-input-buffer load");
    slice.dotLHSGenericLoad = load;
    slice.dotLHSLoadOperation = load.getOperation();
    slice.dotLHSBuffer = load.getBuffer();
    slice.dotLHSValue = load.getLoaded();
    slice.dotLHSABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::DotRHSInputBuffer) {
    if (slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask dot-reduction route requires a unique "
          "dot-rhs-input-buffer load");
    slice.dotRHSGenericLoad = load;
    slice.dotRHSLoadOperation = load.getOperation();
    slice.dotRHSBuffer = load.getBuffer();
    slice.dotRHSValue = load.getLoaded();
    slice.dotRHSABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::SourceInputBuffer) {
    if (slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique source-input-buffer "
          "load");
    slice.sourceGenericLoad = load;
    slice.sourceLoadOperation = load.getOperation();
    slice.sourceBuffer = load.getBuffer();
    slice.sourceValue = load.getLoaded();
    slice.sourceABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::OutputBuffer) {
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique output-buffer "
          "accumulator load");
    slice.accumulatorLoadOperation = load.getOperation();
    slice.accumulatorBuffer = load.getBuffer();
    slice.accumulatorValue = load.getLoaded();
    slice.accumulatorABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::AccumulatorInputBuffer) {
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique "
          "accumulator-input-buffer load");
    slice.accumulatorLoadOperation = load.getOperation();
    slice.accumulatorBuffer = load.getBuffer();
    slice.accumulatorValue = load.getLoaded();
    slice.accumulatorABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::SegmentField0InputBuffer) {
    if (slice.field0LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV segment2 route requires a unique field0 source load");
    slice.field0LoadOperation = load.getOperation();
    slice.field0Buffer = load.getBuffer();
    slice.field0LoadedValue = load.getLoaded();
    slice.field0Value = load.getLoaded();
    slice.field0ABI = parameter;
    if (!slice.lhsLoadOperation) {
      slice.lhsGenericLoad = load;
      slice.lhsLoadOperation = load.getOperation();
      slice.lhsBuffer = load.getBuffer();
      slice.lhsValue = load.getLoaded();
      slice.lhsABI = parameter;
    }
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::SegmentField1InputBuffer) {
    if (slice.field1LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV segment2 route requires a unique field1 source load");
    slice.field1LoadOperation = load.getOperation();
    slice.field1Buffer = load.getBuffer();
    slice.field1LoadedValue = load.getLoaded();
    slice.field1Value = load.getLoaded();
    slice.field1ABI = parameter;
    if (!slice.rhsLoadOperation) {
      slice.rhsGenericLoad = load;
      slice.rhsLoadOperation = load.getOperation();
      slice.rhsBuffer = load.getBuffer();
      slice.rhsValue = load.getLoaded();
      slice.rhsABI = parameter;
    }
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::SegmentField0OutputBuffer) {
    if (slice.field0LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask segment2 load route requires a unique "
          "field0 output-buffer passthrough load");
    slice.field0LoadOperation = load.getOperation();
    slice.field0Buffer = load.getBuffer();
    slice.field0PassthroughValue = load.getLoaded();
    slice.field0ABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::SegmentField1OutputBuffer) {
    if (slice.field1LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask segment2 load route requires a unique "
          "field1 output-buffer passthrough load");
    slice.field1LoadOperation = load.getOperation();
    slice.field1Buffer = load.getBuffer();
    slice.field1PassthroughValue = load.getLoaded();
    slice.field1ABI = parameter;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV generic load runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(parameter.role) +
      "' for bounded EmitC route");
}

llvm::Error assignRVVGenericBroadcastBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::BroadcastLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::RHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV generic broadcast runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique RHS source load or "
        "broadcast");
  slice.rhsBroadcastLoad = load;
  slice.rhsLoadOperation = load.getOperation();
  slice.rhsBuffer = load.getBuffer();
  slice.rhsValue = load.getBroadcast();
  slice.rhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericScalarSplatBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::SplatOp splat,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role ==
      support::RuntimeABIParameterRole::LowerBoundScalarValue) {
    if (slice.lowerBoundScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV f32 clamp/select route requires a unique lower bound "
          "scalar splat");
    slice.lowerBoundScalarSplat = splat;
    slice.lowerBoundValue = splat.getBroadcast();
    slice.lowerBoundABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::UpperBoundScalarValue) {
    if (slice.upperBoundScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV f32 clamp/select route requires a unique upper bound "
          "scalar splat");
    slice.upperBoundScalarSplat = splat;
    slice.upperBoundValue = splat.getBroadcast();
    slice.upperBoundABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role ==
      support::RuntimeABIParameterRole::RHSSecondaryScalarValue) {
    if (slice.rhsSecondaryScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV dual runtime scalar compare route requires a unique "
          "secondary RHS scalar splat");
    slice.rhsSecondaryScalarSplat = splat;
    slice.secondaryCompareRhs = splat.getBroadcast();
    slice.secondaryCompareRhsScalarABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role != support::RuntimeABIParameterRole::RHSScalarValue)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV generic scalar splat runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.rhsLoadOperation && slice.rhsGenericLoad && !slice.rhsScalarSplat &&
      !slice.secondaryCompareLhsLoadOperation &&
      slice.rhsABI.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    slice.secondaryCompareLhsGenericLoad = slice.rhsGenericLoad;
    slice.secondaryCompareLhsLoadOperation = slice.rhsLoadOperation;
    slice.secondaryCompareLhsBuffer = slice.rhsBuffer;
    slice.secondaryCompareLhs = slice.rhsValue;
    slice.secondaryCompareLhsABI = slice.rhsABI;
    slice.rhsGenericLoad = {};
    slice.rhsLoadOperation = nullptr;
    slice.rhsBuffer = {};
    slice.rhsValue = {};
    slice.rhsABI = {};
  }
  if (slice.rhsLoadOperation && slice.field1LoadOperation &&
      slice.rhsLoadOperation == slice.field1LoadOperation &&
      !slice.rhsScalarSplat &&
      slice.rhsABI.role ==
          support::RuntimeABIParameterRole::SegmentField1InputBuffer) {
    slice.rhsGenericLoad = {};
    slice.rhsLoadOperation = nullptr;
    slice.rhsBuffer = {};
    slice.rhsValue = {};
    slice.rhsABI = {};
  }
  if (slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique RHS source load, "
        "broadcast, or scalar splat");
  slice.rhsScalarSplat = splat;
  slice.rhsLoadOperation = splat.getOperation();
  slice.rhsBuffer = splat.getScalar();
  slice.rhsValue = splat.getBroadcast();
  slice.rhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::RHSScalarBroadcast;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericStridedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::StridedLoadOp load,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter) {
  if (bufferParameter.role ==
      support::RuntimeABIParameterRole::SourceInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::SourceByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV source byte-strided load requires "
          "source-byte-stride runtime ABI value");
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-load to unit-stride-store route "
          "requires a unique source strided load");
    slice.lhsStridedLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsStride = load.getStride();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = bufferParameter;
    slice.lhsStrideABI = strideParameter;
    return llvm::Error::success();
  }

  if (bufferParameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::LHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided lhs load requires "
          "lhs-input-stride runtime ABI value");
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided route requires a unique lhs load");
    slice.lhsStridedLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsStride = load.getStride();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = bufferParameter;
    slice.lhsStrideABI = strideParameter;
    return llvm::Error::success();
  }

  if (bufferParameter.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::RHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided rhs load requires "
          "rhs-input-stride runtime ABI value");
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided route requires a unique rhs load");
    slice.rhsStridedLoad = load;
    slice.rhsLoadOperation = load.getOperation();
    slice.rhsBuffer = load.getBuffer();
    slice.rhsStride = load.getStride();
    slice.rhsValue = load.getLoaded();
    slice.rhsABI = bufferParameter;
    slice.rhsStrideABI = strideParameter;
    slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadStore;
    return llvm::Error::success();
  }

  if (bufferParameter.role ==
      support::RuntimeABIParameterRole::DotLHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::LHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided dot lhs load requires "
          "lhs-input-stride runtime ABI value");
    if (slice.dotLHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided dot route requires a "
          "unique dot lhs load");
    slice.lhsStridedLoad = load;
    slice.dotLHSLoadOperation = load.getOperation();
    slice.dotLHSBuffer = load.getBuffer();
    slice.lhsStride = load.getStride();
    slice.dotLHSValue = load.getLoaded();
    slice.dotLHSABI = bufferParameter;
    slice.lhsStrideABI = strideParameter;
    return llvm::Error::success();
  }

  if (bufferParameter.role ==
      support::RuntimeABIParameterRole::DotRHSInputBuffer) {
    if (strideParameter.role !=
        support::RuntimeABIParameterRole::RHSInputStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided dot rhs load requires "
          "rhs-input-stride runtime ABI value");
    if (slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided dot route requires a "
          "unique dot rhs load");
    slice.rhsStridedLoad = load;
    slice.dotRHSLoadOperation = load.getOperation();
    slice.dotRHSBuffer = load.getBuffer();
    slice.rhsStride = load.getStride();
    slice.dotRHSValue = load.getLoaded();
    slice.dotRHSABI = bufferParameter;
    slice.rhsStrideABI = strideParameter;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskStridedInputWideningDotReduce;
    return llvm::Error::success();
  }

  if (bufferParameter.role == support::RuntimeABIParameterRole::OutputBuffer) {
    if (strideParameter.role !=
            support::RuntimeABIParameterRole::OutputStride &&
        strideParameter.role !=
            support::RuntimeABIParameterRole::DestinationByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided old-destination load requires "
          "output-stride or destination-byte-stride runtime ABI value");
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided route requires a unique "
          "old-destination strided load");
    slice.lhsStridedLoad = load;
    slice.accumulatorLoadOperation = load.getOperation();
    slice.accumulatorBuffer = load.getBuffer();
    slice.accumulatorValue = load.getLoaded();
    slice.outBuffer = load.getBuffer();
    slice.outStride = load.getStride();
    slice.accumulatorABI = bufferParameter;
    slice.outABI = bufferParameter;
    slice.outStrideABI = strideParameter;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV strided load buffer runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(bufferParameter.role) + "'");
}

llvm::Error assignRVVGenericStridedStoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::StridedStoreOp store,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter,
    bool requiresDestinationByteStride) {
  if (bufferParameter.role != support::RuntimeABIParameterRole::OutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV strided store buffer runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(bufferParameter.role) + "'");
  const support::RuntimeABIParameterRole expectedStrideRole =
      requiresDestinationByteStride
          ? support::RuntimeABIParameterRole::DestinationByteStride
          : support::RuntimeABIParameterRole::OutputStride;
  if (strideParameter.role != expectedStrideRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded generic RVV strided store requires ") +
        support::stringifyRuntimeABIParameterRole(expectedStrideRole) +
        " runtime ABI value");
  slice.stridedStore = store;
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getBuffer();
  slice.storeValue = store.getValue();
  slice.outStride = store.getStride();
  slice.outABI = bufferParameter;
  slice.outStrideABI = strideParameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::IndexInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV index load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.indexLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires a unique "
        "tcrv_rvv.index_load op");
  if (static_cast<std::int64_t>(load.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only index_eew 32");

  slice.indexLoad = load;
  slice.indexLoadOperation = load.getOperation();
  slice.indexBuffer = load.getIndex();
  slice.indexValue = load.getLoaded();
  slice.indexABI = parameter;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexedLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::LHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV indexed data load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.lhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires a unique data "
        "source tcrv_rvv.indexed_load op");
  if (static_cast<std::int64_t>(load.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only index_eew 32");
  if (load.getOffsetUnit() != kRVVIndexedGatherOffsetUnit)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route supports only offset_unit "
        "'element'");

  slice.indexedLoad = load;
  slice.indexedLoadOperation = load.getOperation();
  slice.lhsLoadOperation = load.getOperation();
  slice.lhsBuffer = load.getData();
  slice.indexedDataBuffer = load.getData();
  slice.lhsValue = load.getLoaded();
  slice.lhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericIndexedStoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::IndexedStoreOp store,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::OutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV indexed data store runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.indexedStoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires a unique "
        "tcrv_rvv.indexed_store op");
  if (static_cast<std::int64_t>(store.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route supports only index_eew "
        "32");
  if (store.getOffsetUnit() != kRVVIndexedGatherOffsetUnit)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route supports only offset_unit "
        "'element'");

  slice.indexedStore = store;
  slice.indexedStoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getDestination();
  slice.indexedDestinationBuffer = store.getDestination();
  slice.storeValue = store.getValue();
  slice.outABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericMaskLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::MaskInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV mask load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.maskLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires a unique "
        "tcrv_rvv.mask_load op");
  if (load.getMaskRole() != kRVVMaskRole)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires mask_role "
        "'predicate-mask-input-buffer'");
  if (load.getMaskMemoryForm() != kRVVMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires mask_memory_form "
        "'unit-stride-mask-load'");

  slice.maskLoad = load;
  slice.maskLoadOperation = load.getOperation();
  slice.maskBuffer = load.getMask();
  slice.maskValue = load.getLoaded();
  slice.maskABI = parameter;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericMaskedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (load.getMemoryForm() != kRVVMaskedLoadMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV masked load memory_form '") +
        load.getMemoryForm() + "' for bounded EmitC route");
  if (load.getInactiveLanePolicy() != kRVVMaskedLoadInactiveLanePolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV masked load inactive_lane_policy '") +
        load.getInactiveLanePolicy() + "' for bounded EmitC route");

  if (parameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV masked unit-load-store route requires a unique "
          "lhs-input-buffer tcrv_rvv.masked_load");
    slice.maskedLoadOp = load;
    slice.maskedLoadOperation = load.getOperation();
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = parameter;
    return llvm::Error::success();
  }

  if (parameter.role == support::RuntimeABIParameterRole::SourceInputBuffer) {
    if (slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV computed-mask unit-load-store route requires a unique "
          "source-input-buffer tcrv_rvv.masked_load");
    slice.maskedLoadOp = load;
    slice.maskedLoadOperation = load.getOperation();
    slice.sourceLoadOperation = load.getOperation();
    slice.sourceBuffer = load.getBuffer();
    slice.sourceValue = load.getLoaded();
    slice.sourceABI = parameter;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV masked load runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(parameter.role) +
      "' for bounded EmitC route");
}

llvm::Error assignRVVGenericMaskedStridedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedStridedLoadOp load,
    const support::RuntimeABIParameter &bufferParameter,
    const support::RuntimeABIParameter &strideParameter) {
  if (load.getMemoryForm() != kRVVMaskedStridedLoadSourceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_strided_load "
                    "memory_form '") +
        load.getMemoryForm() +
        "' for bounded RVV computed-mask strided load route");
  if (load.getStrideUnit() != "byte")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_strided_load "
                    "stride_unit '") +
        load.getStrideUnit() +
        "' for bounded RVV computed-mask strided load route");
  if (load.getInactiveLanePolicy() != kRVVMaskedLoadInactiveLanePolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_strided_load "
                    "inactive_lane_policy '") +
        load.getInactiveLanePolicy() +
        "' for bounded RVV computed-mask strided load route");
  if (bufferParameter.role !=
      support::RuntimeABIParameterRole::SourceInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV masked strided load buffer runtime ABI "
                    "role '") +
        support::stringifyRuntimeABIParameterRole(bufferParameter.role) +
        "' for bounded EmitC route");
  if (strideParameter.role !=
      support::RuntimeABIParameterRole::SourceByteStride)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV masked strided load route requires source-byte-stride "
        "runtime ABI value");
  if (slice.sourceLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask strided-load route requires a unique "
        "source-input-buffer tcrv_rvv.masked_strided_load");

  slice.maskedStridedLoadOp = load;
  slice.maskedStridedLoadOperation = load.getOperation();
  slice.sourceLoadOperation = load.getOperation();
  slice.sourceBuffer = load.getBuffer();
  slice.sourceStride = load.getStride();
  slice.sourceValue = load.getLoaded();
  slice.sourceABI = bufferParameter;
  slice.sourceStrideABI = strideParameter;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericMaskedIndexedLoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedIndexedLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (load.getMemoryForm() != kRVVMaskedIndexedLoadSourceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_indexed_load "
                    "memory_form '") +
        load.getMemoryForm() +
        "' for bounded RVV computed-mask indexed gather-load route");
  if (load.getOffsetUnit() != kRVVIndexedGatherOffsetUnit)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_indexed_load "
                    "offset_unit '") +
        load.getOffsetUnit() +
        "' for bounded RVV computed-mask indexed gather-load route");
  if (static_cast<std::int64_t>(load.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed gather-load route "
        "supports only index_eew 32");
  if (load.getInactiveLanePolicy() != kRVVMaskedLoadInactiveLanePolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_indexed_load "
                    "inactive_lane_policy '") +
        load.getInactiveLanePolicy() +
        "' for bounded RVV computed-mask indexed gather-load route");
  if (parameter.role != support::RuntimeABIParameterRole::SourceInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV masked indexed load source runtime ABI "
                    "role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.sourceLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask indexed gather-load route requires a "
        "unique source-input-buffer tcrv_rvv.masked_indexed_load");

  slice.maskedIndexedLoadOp = load;
  slice.maskedIndexedLoadOperation = load.getOperation();
  slice.sourceLoadOperation = load.getOperation();
  slice.sourceBuffer = load.getData();
  slice.indexedDataBuffer = load.getData();
  slice.sourceValue = load.getLoaded();
  slice.sourceABI = parameter;
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericSegment2LoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::Segment2LoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::LHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV segment2 load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.segment2LoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires a unique "
        "tcrv_rvv.segment2_load op");
  if (static_cast<std::int64_t>(load.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires "
        "segment_count 2");
  if (load.getSourceMemoryForm() != kRVVSegment2SourceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires "
        "source_memory_form 'segment2-interleaved-unit-stride-load'");
  if (load.getField0Role() != kRVVSegment2Field0Role ||
      load.getField1Role() != kRVVSegment2Field1Role)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires field0_role "
        "'segment-field0-output-buffer' and field1_role "
        "'segment-field1-output-buffer'");
  if (load.getField0Role() == load.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires distinct "
        "field roles");

  slice.segment2Load = load;
  slice.segment2LoadOperation = load.getOperation();
  slice.lhsLoadOperation = load.getOperation();
  slice.lhsBuffer = load.getSource();
  slice.lhsABI = parameter;
  slice.field0LoadedValue = load.getField0();
  slice.field1LoadedValue = load.getField1();
  slice.memoryForm = RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericMaskedSegment2LoadBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedSegment2LoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::SourceInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV masked segment2 load runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.maskedSegment2LoadOperation &&
      slice.maskedSegment2LoadOperation != load.getOperation())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires a "
        "unique tcrv_rvv.masked_segment2_load op");
  if (slice.sourceLoadOperation &&
      slice.sourceLoadOperation != load.getOperation())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires a "
        "unique source-input-buffer tcrv_rvv.masked_segment2_load");
  if (static_cast<std::int64_t>(load.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires "
        "segment_count 2");
  if (load.getSourceMemoryForm() != kRVVSegment2SourceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires "
        "source_memory_form 'segment2-interleaved-unit-stride-load'");
  if (load.getField0Role() != kRVVSegment2Field0Role ||
      load.getField1Role() != kRVVSegment2Field1Role)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires "
        "field0_role 'segment-field0-output-buffer' and field1_role "
        "'segment-field1-output-buffer'");
  if (load.getField0Role() == load.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires "
        "distinct field roles");
  if (load.getInactiveLanePolicy() != kRVVMaskedLoadInactiveLanePolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_segment2_load "
                    "inactive_lane_policy '") +
        load.getInactiveLanePolicy() +
        "' for bounded RVV computed-mask segment2 load route");

  slice.maskedSegment2LoadOp = load;
  slice.maskedSegment2LoadOperation = load.getOperation();
  slice.sourceLoadOperation = load.getOperation();
  slice.sourceBuffer = load.getSource();
  slice.sourceABI = parameter;
  slice.maskValue = load.getMask();
  slice.field0LoadedValue = load.getField0();
  slice.field1LoadedValue = load.getField1();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericSegment2StoreBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::Segment2StoreOp store,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role !=
      support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV segment2 store runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.segment2StoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires a unique "
        "tcrv_rvv.segment2_store op");
  if (static_cast<std::int64_t>(store.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires "
        "segment_count 2");
  if (store.getDestinationMemoryForm() !=
      kRVVSegment2InterleavedDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires "
        "destination_memory_form "
        "'segment2-interleaved-unit-stride-store'");
  if (store.getField0Role() != kRVVSegment2Field0InputRole ||
      store.getField1Role() != kRVVSegment2Field1InputRole)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires field0_role "
        "'segment-field0-input-buffer' and field1_role "
        "'segment-field1-input-buffer'");
  if (store.getField0Role() == store.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires distinct "
        "field roles");

  slice.segment2Store = store;
  slice.segment2StoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getDestination();
  slice.outABI = parameter;
  slice.storeValue = store.getField0();
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  return llvm::Error::success();
}

llvm::Error assignRVVGenericMaskedSegment2StoreBinding(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedSegment2StoreOp store,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role !=
      support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV masked segment2 store runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) + "'");
  if (slice.maskedSegment2StoreOperation &&
      slice.maskedSegment2StoreOperation != store.getOperation())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store route requires a "
        "unique tcrv_rvv.masked_segment2_store op");
  if (slice.storeOperation && slice.storeOperation != store.getOperation())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store route requires a "
        "unique interleaved destination store op");
  if (static_cast<std::int64_t>(store.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store route requires "
        "segment_count 2");
  if (store.getDestinationMemoryForm() !=
      kRVVSegment2InterleavedDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store route requires "
        "destination_memory_form "
        "'segment2-interleaved-unit-stride-store'");
  if (store.getField0Role() != kRVVSegment2Field0InputRole ||
      store.getField1Role() != kRVVSegment2Field1InputRole)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store route requires "
        "field0_role 'segment-field0-input-buffer' and field1_role "
        "'segment-field1-input-buffer'");
  if (store.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store route requires "
        "inactive_lane_policy 'preserve-output-on-false-lanes'");

  slice.maskedSegment2Store = store;
  slice.maskedSegment2StoreOperation = store.getOperation();
  slice.segment2StoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.outBuffer = store.getDestination();
  slice.outABI = parameter;
  slice.storeValue = store.getField0();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyRuntimeABIParameters(
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedBodyConstructionRoute &constructionRoute,
    const RVVSelectedBodyConfigProfile &configProfile,
    const support::RuntimeABIParameter &runtimeElementCountABI,
    const support::RuntimeABIParameter &outABI) {
  slice.runtimeElementCountABI = runtimeElementCountABI;
  slice.outABI = outABI;
  const bool isWideningMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningProductReduceDequantize =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32;
  const bool isWideningProductReduceDequantClamp =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isWideningProductReduce =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isWideningProductReduceDequantize;
  // The deferred-wide i16 dot-reduce terminal kind (2nd kernel family) shares the
  // narrow dot-reduce runtime-ABI shape (lhs/rhs/acc/out/n, source i16) -- its ABI
  // is verified through the dot-reduce ABI verifier, NOT the finite-binary
  // elementwise contract (whose source c-type is derived from the result config
  // and would wrongly expect const int32_t *).
  const bool isWideningDotReduce =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredDotAccumulateReduceAdd;
  const bool isWideningProduct =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningProduct;
  const bool isStridedInputWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
  const bool isComputedMaskWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  const bool isComputedMaskStridedInputWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          ComputedMaskStridedInputWideningDotReduceAdd;
  const bool isStandaloneReduction =
      isRVVSelectedBodyPlainStandaloneReductionRouteOperation(
          slice.arithmeticKind);
  const bool isComputedMaskStandaloneReduction =
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          slice.arithmeticKind);
  const bool isRuntimeScalarComputedMaskStandaloneReduction =
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          slice.arithmeticKind);
  const bool isComputedMaskedMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd;
  const bool isRuntimeScalarComputedMaskedMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isScalarBroadcastMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
  const bool isRuntimeScalarSplatStore =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarSplatStore;
  const bool isRuntimeScalarCompareSelect =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;
  const bool isF32ClampSelect =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::F32ClampSelect;
  const bool isDequantClampF32Epilogue =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
  const bool isDequantization =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::DequantizeI32ToF32;

  llvm::SmallVector<support::RuntimeABIParameter, 9> ordered;
  if (isRuntimeScalarSplatStore) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit runtime scalar "
                "splat-store runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                  RuntimeScalarDualCompareMaskAndSelect) {
    ordered.push_back(slice.lhsABI);
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.secondaryCompareLhsABI);
    ordered.push_back(slice.secondaryCompareRhsScalarABI);
    ordered.push_back(slice.trueValueABI);
    ordered.push_back(slice.falseValueABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit runtime scalar "
                "dual-compare mask-and select runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  ordered.push_back(slice.lhsABI);
  if (isF32ClampSelect) {
    ordered.push_back(slice.lowerBoundABI);
    ordered.push_back(slice.upperBoundABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit f32 clamp/select runtime "
                "ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isDequantClampF32Epilogue) {
    ordered.push_back(slice.dequantScaleABI);
    ordered.push_back(slice.lowerBoundABI);
    ordered.push_back(slice.upperBoundABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit dequant-clamp f32 "
                "epilogue runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isDequantization) {
    ordered.push_back(slice.dequantScaleABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit i32-to-f32 "
                "dequantization runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isWideningProductReduceDequantize) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.dequantScaleABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit low-precision "
                "product-reduction dequantization runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isWideningProductReduceDequantClamp) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.dequantScaleABI);
    ordered.push_back(slice.lowerBoundABI);
    ordered.push_back(slice.upperBoundABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit low-precision "
                "product-reduction dequant-clamp runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isRuntimeScalarCompareSelect) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.trueValueABI);
    ordered.push_back(slice.falseValueABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit runtime scalar "
                "compare/select runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isComputedMaskedMAcc) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.dotLHSABI);
    ordered.push_back(slice.dotRHSABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit computed-mask "
                "multiply-accumulate runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isScalarBroadcastMAcc) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit scalar-broadcast "
                "multiply-accumulate composition runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isRuntimeScalarComputedMaskedMAcc) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.dotLHSABI);
    ordered.push_back(slice.dotRHSABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit runtime scalar computed-mask "
                "multiply-accumulate runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.sourceABI);
    ordered.push_back(slice.dotRHSABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.indexABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit runtime scalar "
                "computed-mask indexed gather-MAcc-scatter runtime ABI "
                "values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isComputedMaskWideningDotReduce) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.dotLHSABI);
    ordered.push_back(slice.dotRHSABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit computed-mask widening "
                "dot-product reduction runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isComputedMaskStridedInputWideningDotReduce) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.dotLHSABI);
    ordered.push_back(slice.dotRHSABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    ordered.push_back(slice.lhsStrideABI);
    ordered.push_back(slice.rhsStrideABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit computed-mask "
                "strided-input widening dot-product reduction runtime ABI "
                "values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isComputedMaskStandaloneReduction) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.sourceABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit computed-mask standalone "
                "reduction runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (isRuntimeScalarComputedMaskStandaloneReduction) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.sourceABI);
    ordered.push_back(slice.accumulatorABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit runtime scalar computed-mask "
                "standalone reduction runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.sourceABI);
    ordered.push_back(slice.field0ABI);
    ordered.push_back(slice.field1ABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit computed-mask segment2 "
                "load runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.field0ABI);
    ordered.push_back(slice.field1ABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit computed-mask segment2 "
                "store runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::Segment2LoadUnitStore) {
    ordered.push_back(slice.field0ABI);
    ordered.push_back(slice.field1ABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit segment2 runtime ABI "
                "values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadSegment2Store) {
    ordered.push_back(slice.rhsABI);
    ordered.push_back(slice.outABI);
    ordered.push_back(slice.runtimeElementCountABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit segment2 interleave "
                "runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore ||
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore)
    ordered.push_back(slice.indexABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore ||
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore)
    ordered.push_back(slice.maskABI);
  if (slice.memoryForm != RVVSelectedBodyMemoryForm::UnitStrideConversion &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::UnitLoadStridedStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitLoadStore &&
      slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitStore &&
      slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction)
    ordered.push_back(slice.rhsABI);
  if (isWideningMAcc || isWideningDotReduce || isWideningProductReduce ||
      isStridedInputWideningDotReduce || isStandaloneReduction)
    ordered.push_back(slice.accumulatorABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect) {
    ordered.push_back(slice.trueValueABI);
    ordered.push_back(slice.falseValueABI);
  }
  if (slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore ||
      slice.memoryForm ==
          RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore ||
      slice.memoryForm ==
          RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore ||
      slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore ||
      slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore ||
      slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore ||
      slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore)
    ordered.push_back(slice.sourceABI);
  if (slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore ||
      slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore)
    ordered.push_back(slice.indexABI);
  ordered.push_back(slice.outABI);
  ordered.push_back(slice.runtimeElementCountABI);
  if (slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore) {
    ordered.push_back(slice.lhsStrideABI);
    ordered.push_back(slice.rhsStrideABI);
    ordered.push_back(slice.outStrideABI);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce) {
    ordered.push_back(slice.lhsStrideABI);
    ordered.push_back(slice.rhsStrideABI);
    if (llvm::Error error =
            tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                ordered,
                "selected RVV EmitC route explicit strided-input widening "
                "dot-product reduction runtime ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    ordered.push_back(slice.lhsStrideABI);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitLoadStridedStore) {
    ordered.push_back(slice.outStrideABI);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore) {
    ordered.push_back(slice.outStrideABI);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore) {
    ordered.push_back(slice.sourceStrideABI);
  } else if (slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::RHSScalarBroadcast &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitStrideConversion &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitLoadStridedStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::
                     RuntimeScalarComputedMaskLoadStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::
                     ComputedMaskIndexedGatherLoadUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::
                     ComputedMaskUnitLoadIndexedScatterStore &&
             slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitLoadStore &&
             slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitStore &&
             slice.memoryForm !=
                 RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction) {
    if (isWideningMAcc || isWideningDotReduce || isWideningProduct ||
        isWideningProductReduce) {
      if (llvm::Error error =
              tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
                  ordered,
                  isWideningDotReduce
                      ? "selected RVV EmitC route explicit widening "
                        "dot-product reduction runtime ABI values"
                  : isWideningProductReduce
                      ? "selected RVV EmitC route explicit low-precision "
                        "product-reduction runtime ABI values"
                  : isWideningProduct
                      ? "selected RVV EmitC route explicit low-precision "
                        "widening-product runtime ABI values"
                      : "selected RVV EmitC route explicit widening macc "
                        "runtime ABI values"))
        return makeRVVEmitCRouteProviderError(
            llvm::toString(std::move(error)));
      return llvm::Error::success();
    }
    support::FiniteBinaryRuntimeABIContract contract(
        support::FiniteBinaryRuntimeABIContractSpec{
            constructionRoute.runtimeABIContractName,
            configProfile.constInputPointerCType,
            configProfile.outputPointerCType});
    llvm::Expected<support::FiniteBinaryCallableRuntimeABIParameterBindings>
        bindings = support::bindFiniteBinaryCallableRuntimeABIParametersByRole(
            ordered, "RVV selected-body explicit runtime ABI values",
            contract);
    if (!bindings)
      return bindings.takeError();
  }

  if (isStandaloneReduction) {
    if (llvm::Error error = tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
            ordered,
            "selected RVV EmitC route explicit standalone reduction runtime "
            "ABI values"))
      return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
    return llvm::Error::success();
  }

  if (llvm::Error error = tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
          ordered, "selected RVV EmitC route explicit runtime ABI values"))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

void addRouteOperandBinding(
    RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
    const support::RuntimeABIParameter &parameter,
    llvm::ArrayRef<llvm::StringRef> materializedUses) {
  RVVRouteOperandBinding binding;
  binding.logicalOperand = logicalOperand.str();
  binding.parameter = parameter;
  for (llvm::StringRef use : materializedUses)
    binding.materializedUses.push_back(use.str());
  plan.bindings.push_back(std::move(binding));
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVComputedMaskIndexedMemoryRuntimeABIParameters(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return tcrv::rvv::
        getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return tcrv::rvv::
        getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return tcrv::rvv::
        getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return tcrv::rvv::
        getRVVSelectedBodyRuntimeScalarComputedMaskIndexedScatterRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return tcrv::rvv::
        getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIParameters();
  default:
    return {};
  }
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVComputedMaskStridedMemoryRuntimeABIParameters(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return tcrv::rvv::
        getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return tcrv::rvv::
        getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters();
  default:
    return {};
  }
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVComputedMaskSegment2MemoryRuntimeABIParameters(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return tcrv::rvv::
        getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return tcrv::rvv::
        getRVVSelectedBodyRuntimeScalarComputedMaskSegment2LoadRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return tcrv::rvv::
        getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return tcrv::rvv::
        getRVVSelectedBodyRuntimeScalarComputedMaskSegment2StoreRuntimeABIParameters();
  default:
    return {};
  }
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVPlainSegment2MemoryRuntimeABIParameters(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return tcrv::rvv::
        getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters();
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return tcrv::rvv::
        getRVVSelectedBodySegment2InterleaveRuntimeABIParameters();
  default:
    return {};
  }
}

RVVRouteOperandBindingPlan
buildUnitStrideMaskedMemoryRouteOperandBindingPlanFromFacts(
    const RVVUnitStrideMaskedMemoryRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  auto parameter =
      [&](std::size_t index) -> const support::RuntimeABIParameter & {
    return facts.runtimeABIParameters[index];
  };

  switch (facts.operation) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    addRouteOperandBinding(
        plan, "lhs", parameter(0),
        {"runtime-abi-mirror", "materialized-load-base", "compare-lhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", parameter(1),
        {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
         "compare-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "src", parameter(2),
        {"runtime-abi-mirror", "materialized-source-load-base",
         "masked-store-source-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst", parameter(3),
        {"runtime-abi-mirror", "materialized-masked-store-base",
         "masked-store-destination-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(4),
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    addRouteOperandBinding(
        plan, "lhs", parameter(0),
        {"runtime-abi-mirror", "materialized-load-base", "compare-lhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", parameter(1),
        {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
         "compare-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "src", parameter(2),
        {"runtime-abi-mirror", "materialized-masked-load-base",
         "masked-load-source-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst", parameter(3),
        {"runtime-abi-mirror", "materialized-old-destination-load-base",
         "masked-load-passthrough-call", "materialized-store-base",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(4),
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    addRouteOperandBinding(
        plan, "cmp_lhs", parameter(0),
        {"abi-mirror", "cmp-lhs-load", "compare-lhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", parameter(1),
        {"abi-mirror", "cmp-rhs-load", "compare-rhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "src", parameter(2),
        {"abi-mirror", "materialized-masked-load-base",
         "masked-load-source-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst", parameter(3),
        {"abi-mirror", "old-dst-load", "masked-load-passthrough-call",
         "materialized-store-base", "header-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(4),
        {"abi-mirror", "setvl-avl", "loop-control", "header-mirror"});
    break;
  default:
    break;
  }

  return plan;
}

RVVRouteOperandBindingPlan
buildComputedMaskStridedMemoryRouteOperandBindingPlanFromFacts(
    const RVVComputedMaskStridedMemoryRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  auto parameter =
      [&](std::size_t index) -> const support::RuntimeABIParameter & {
    return facts.runtimeABIParameters[index];
  };

  switch (facts.operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    addRouteOperandBinding(
        plan, "cmp_lhs", parameter(0),
        {"abi-mirror", "cmp-lhs-load", "cmp-lhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", parameter(1),
        {"abi-mirror", "cmp-rhs-load", "cmp-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "src", parameter(2),
        {"abi-mirror", "src-load", "mstr-store-src-call", "header-mirror"});
    addRouteOperandBinding(plan, "dst", parameter(3),
                           {"abi-mirror", "mstr-store-base",
                            "header-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(4),
        {"abi-mirror", "setvl-avl", "loop-control", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst_stride_bytes", parameter(5),
        {"abi-mirror", "mstr-store-stride", facts.destinationStrideUnit,
         "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    addRouteOperandBinding(plan, "cmp_lhs", parameter(0),
                           {"abi", "cmp-lhs-load", "lhs-call",
                            "hdr-mirror"});
    addRouteOperandBinding(plan, "cmp_rhs", parameter(1),
                           {"abi", "cmp-rhs-load", "rhs-call",
                            "hdr-mirror"});
    addRouteOperandBinding(plan, "src", parameter(2),
                           {"abi", "mstr-base", "mstr-load-call",
                            "hdr-mirror"});
    addRouteOperandBinding(plan, "dst", parameter(3),
                           {"abi", "old-dst-load", "passthru-call",
                            "store-base", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(4),
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src_stride_bytes", parameter(5),
        {"abi", "mstr-stride", facts.sourceStrideUnit, "hdr-mirror"});
    break;
  default:
    break;
  }

  return plan;
}

RVVRouteOperandBindingPlan
buildComputedMaskIndexedMemoryRouteOperandBindingPlanFromFacts(
    const RVVComputedMaskIndexedMemoryRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  auto parameter =
      [&](std::size_t index) -> const support::RuntimeABIParameter & {
    return facts.runtimeABIParameters[index];
  };

  switch (facts.operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    addRouteOperandBinding(
        plan, "cmp_lhs", parameter(0),
        {"abi", "cmp-lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", parameter(1),
        {"abi", "cmp-rhs-load", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(plan, "src", parameter(2),
                           {"abi", "midx-base", "midx-load-call",
                            "hdr-mirror"});
    addRouteOperandBinding(
        plan, "index", parameter(3),
        {"abi", "materialized-index-load-base", "index-offset-scale",
         "index-source-mirror", "hdr-mirror"});
    addRouteOperandBinding(plan, "dst", parameter(4),
                           {"abi", "old-dst-load", "passthru-call",
                            "store-base", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(5),
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
    break;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    addRouteOperandBinding(
        plan, "lhs", parameter(0),
        {"abi", "lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", parameter(1),
        {"abi", "splat", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(plan, "src", parameter(2),
                           {"abi", "midx-base", "midx-load-call",
                            "hdr-mirror"});
    addRouteOperandBinding(
        plan, "index", parameter(3),
        {"abi", "materialized-index-load-base", "index-offset-scale",
         "index-source-mirror", "hdr-mirror"});
    addRouteOperandBinding(plan, "dst", parameter(4),
                           {"abi", "old-dst-load", "passthru-call",
                            "store-base", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(5),
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    addRouteOperandBinding(
        plan, "cmp_lhs", parameter(0),
        {"abi", "cmp-lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", parameter(1),
        {"abi", "cmp-rhs-load", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src", parameter(2),
        {"abi", "src-load", "mistore-src-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "index", parameter(3),
        {"abi", "materialized-index-load-base", "index-offset-scale",
         "index-source-mirror", "hdr-mirror"});
    addRouteOperandBinding(plan, "dst", parameter(4),
                           {"abi", "mistore-base", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(5),
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
    break;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    addRouteOperandBinding(
        plan, "lhs", parameter(0),
        {"abi", "lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", parameter(1),
        {"abi", "splat", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src", parameter(2),
        {"abi", "src-load", "mistore-src-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "index", parameter(3),
        {"abi", "materialized-index-load-base", "index-offset-scale",
         "index-source-mirror", "hdr-mirror"});
    addRouteOperandBinding(plan, "dst", parameter(4),
                           {"abi", "mistore-base", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", parameter(5),
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
    break;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    addRouteOperandBinding(plan, "cmp_lhs", parameter(0),
                           {"abi", "ld", "cmp-lhs", "hdr"});
    addRouteOperandBinding(
        plan, "rhs_scalar", parameter(1),
        {"abi", "splat", "cmp-rhs", "hdr"});
    addRouteOperandBinding(
        plan, "gather_src", parameter(2),
        {"abi", "gbase", "gather", "hdr"});
    addRouteOperandBinding(plan, "payload", parameter(3),
                           {"abi", "ld", "macc-rhs", "hdr"});
    addRouteOperandBinding(plan, "acc", parameter(4),
                           {"abi", "ld", "macc-acc", "hdr"});
    addRouteOperandBinding(
        plan, "index", parameter(5),
        {"abi", "idxld", "idxscale", "idxmirror", "hdr"});
    addRouteOperandBinding(plan, "dst", parameter(6),
                           {"abi", "olddst", "gpass", "scatter", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(7),
                           {"abi", "setvl-avl", "loop", "hdr"});
    break;
  default:
    break;
  }

  return plan;
}

RVVRouteOperandBindingPlan buildPlainSegment2MemoryRouteOperandBindingPlanFromFacts(
    const RVVPlainSegment2MemoryRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  auto parameter =
      [&](std::size_t index) -> const support::RuntimeABIParameter & {
    return facts.runtimeABIParameters[index];
  };

  switch (facts.operation) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    addRouteOperandBinding(plan, "src", parameter(0),
                           {"abi", "seg-load-base", "src-mem", "hdr"});
    addRouteOperandBinding(plan, "out0", parameter(1),
                           {"abi", "field0-store-base", "field0-role",
                            "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "out1", parameter(2),
                           {"abi", "field1-store-base", "field1-role",
                            "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(3),
                           {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    addRouteOperandBinding(plan, "src0", parameter(0),
                           {"abi", "field0-load-base", "field0-role",
                            "src0-mem", "tuple-field0", "hdr"});
    addRouteOperandBinding(plan, "src1", parameter(1),
                           {"abi", "field1-load-base", "field1-role",
                            "src1-mem", "tuple-field1", "hdr"});
    addRouteOperandBinding(plan, "dst", parameter(2),
                           {"abi", "seg-store-base", "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(3),
                           {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  default:
    break;
  }

  return plan;
}

RVVRouteOperandBindingPlan
buildComputedMaskSegment2MemoryRouteOperandBindingPlanFromFacts(
    const RVVComputedMaskSegment2MemoryRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  auto parameter =
      [&](std::size_t index) -> const support::RuntimeABIParameter & {
    return facts.runtimeABIParameters[index];
  };

  switch (facts.operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    addRouteOperandBinding(plan, "cmp_lhs", parameter(0),
                           {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addRouteOperandBinding(plan, "cmp_rhs", parameter(1),
                           {"abi", "cmp-rhs-load", "rhs-call", "hdr"});
    addRouteOperandBinding(plan, "src", parameter(2),
                           {"abi", "mseg-base", "mseg-call", "src-mem",
                            "hdr"});
    addRouteOperandBinding(plan, "out0", parameter(3),
                           {"abi", "old0-load", "f0-pass", "f0-store",
                            "f0-role", "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "out1", parameter(4),
                           {"abi", "old1-load", "f1-pass", "f1-store",
                            "f1-role", "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(5),
                           {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    addRouteOperandBinding(plan, "lhs", parameter(0),
                           {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addRouteOperandBinding(plan, "rhs_scalar", parameter(1),
                           {"abi", "splat", "rhs-call", "hdr"});
    addRouteOperandBinding(plan, "src", parameter(2),
                           {"abi", "mseg-base", "mseg-call", "src-mem",
                            "hdr"});
    addRouteOperandBinding(plan, "out0", parameter(3),
                           {"abi", "old0-load", "f0-pass", "f0-store",
                            "f0-role", "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "out1", parameter(4),
                           {"abi", "old1-load", "f1-pass", "f1-store",
                            "f1-role", "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(5),
                           {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    addRouteOperandBinding(plan, "cmp_lhs", parameter(0),
                           {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addRouteOperandBinding(plan, "cmp_rhs", parameter(1),
                           {"abi", "cmp-rhs-load", "rhs-call", "hdr"});
    addRouteOperandBinding(plan, "src0", parameter(2),
                           {"abi", "f0-load", "f0-payload", "tuple0",
                            "f0-role", "src0-mem", "hdr"});
    addRouteOperandBinding(plan, "src1", parameter(3),
                           {"abi", "f1-load", "f1-payload", "tuple1",
                            "f1-role", "src1-mem", "hdr"});
    addRouteOperandBinding(plan, "dst", parameter(4),
                           {"abi", "mseg-store", "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(5),
                           {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    addRouteOperandBinding(plan, "lhs", parameter(0),
                           {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addRouteOperandBinding(plan, "rhs_scalar", parameter(1),
                           {"abi", "splat", "rhs-call", "hdr"});
    addRouteOperandBinding(plan, "src0", parameter(2),
                           {"abi", "f0-load", "f0-payload", "tuple0",
                            "f0-role", "src0-mem", "hdr"});
    addRouteOperandBinding(plan, "src1", parameter(3),
                           {"abi", "f1-load", "f1-payload", "tuple1",
                            "f1-role", "src1-mem", "hdr"});
    addRouteOperandBinding(plan, "dst", parameter(4),
                           {"abi", "mseg-store", "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(5),
                           {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    addRouteOperandBinding(plan, "cmp_lhs", parameter(0),
                           {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addRouteOperandBinding(plan, "cmp_rhs", parameter(1),
                           {"abi", "cmp-rhs-load", "rhs-call", "hdr"});
    addRouteOperandBinding(plan, "src0", parameter(2),
                           {"abi", "f0-load", "f0-payload", "add-lhs",
                            "tuple0", "f0-role", "src0-mem", "hdr"});
    addRouteOperandBinding(plan, "src1", parameter(3),
                           {"abi", "f1-load", "f1-payload", "add-rhs",
                            "tuple1", "f1-role", "src1-mem", "hdr"});
    addRouteOperandBinding(plan, "dst", parameter(4),
                           {"abi", "mseg-store", "dst-mem", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(5),
                           {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  default:
    break;
  }

  return plan;
}

RVVRouteOperandBindingPlan
buildCompareSelectRouteOperandBindingPlanFromFacts(
    const RVVCompareSelectRouteFacts &facts) {
  RVVRouteOperandBindingPlan plan;
  plan.planID = facts.routeOperandBindingPlanID.str();
  auto parameter =
      [&](std::size_t index) -> const support::RuntimeABIParameter & {
    return facts.runtimeABIParameters[index];
  };

  switch (facts.operation) {
  case RVVSelectedBodyOperationKind::CmpSelect:
    addRouteOperandBinding(
        plan, "lhs", parameter(0),
        {"abi", "load-base", "compare-lhs-call", "select-true-call"});
    addRouteOperandBinding(
        plan, "rhs", parameter(1),
        {"abi", "load-base", "compare-rhs-call", "select-false-call"});
    addRouteOperandBinding(plan, "out", parameter(2),
                           {"abi", "store-base", "header"});
    addRouteOperandBinding(plan, "n", parameter(3),
                           {"abi", "setvl-avl", "loop-control", "header"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
    addRouteOperandBinding(plan, "cmp_lhs", parameter(0),
                           {"abi", "cmp-lhs", "cmp-call", "hdr"});
    addRouteOperandBinding(plan, "cmp_rhs", parameter(1),
                           {"abi", "cmp-rhs", "cmp-call", "hdr"});
    addRouteOperandBinding(plan, "true_value", parameter(2),
                           {"abi", "true-load", "sel-true", "hdr"});
    addRouteOperandBinding(plan, "false_value", parameter(3),
                           {"abi", "false-load", "sel-false", "hdr"});
    addRouteOperandBinding(plan, "out", parameter(4),
                           {"abi", "store", "hdr"});
    addRouteOperandBinding(plan, "n", parameter(5),
                           {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    addRouteOperandBinding(
        plan, "lhs", parameter(0),
        {"runtime-abi-mirror", "materialized-load-base", "compare-lhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", parameter(1),
        {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
         "compare-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "true_value", parameter(2),
        {"runtime-abi-mirror", "materialized-true-load-base",
         "select-true-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "false_value", parameter(3),
        {"runtime-abi-mirror", "materialized-false-load-base",
         "select-false-call", "header-mirror"});
    addRouteOperandBinding(plan, "out", parameter(4),
                           {"runtime-abi-mirror", "materialized-store-base",
                            "header-mirror"});
    addRouteOperandBinding(plan, "n", parameter(5),
                           {"runtime-abi-mirror", "setvl-avl",
                            "loop-control", "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect:
    addRouteOperandBinding(
        plan, "cmp_lhs_a", parameter(0),
        {"runtime-abi-mirror", "materialized-load-base", "compare-lhs-call",
         "mask-and-lhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar_a", parameter(1),
        {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
         "compare-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "cmp_lhs_b", parameter(2),
        {"runtime-abi-mirror", "materialized-secondary-load-base",
         "secondary-compare-lhs-call", "mask-and-rhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar_b", parameter(3),
        {"runtime-abi-mirror", "secondary-scalar-broadcast-rhs-call",
         "secondary-compare-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "true_value", parameter(4),
        {"runtime-abi-mirror", "materialized-true-load-base",
         "select-true-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "false_value", parameter(5),
        {"runtime-abi-mirror", "materialized-false-load-base",
         "select-false-call", "header-mirror"});
    addRouteOperandBinding(plan, "out", parameter(6),
                           {"runtime-abi-mirror", "materialized-store-base",
                            "header-mirror"});
    addRouteOperandBinding(plan, "n", parameter(7),
                           {"runtime-abi-mirror", "setvl-avl",
                            "loop-control", "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::F32ClampSelect:
    addRouteOperandBinding(
        plan, "input", parameter(0),
        {"runtime-abi-mirror", "materialized-f32-input-load-base",
         "lower-compare-lhs-call", "lower-select-false-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "lower_bound", parameter(1),
        {"runtime-abi-mirror", "lower-bound-splat-call",
         "lower-compare-rhs-call", "lower-select-true-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "upper_bound", parameter(2),
        {"runtime-abi-mirror", "upper-bound-splat-call",
         "upper-compare-lhs-call", "upper-select-true-call",
         "header-mirror"});
    addRouteOperandBinding(plan, "out", parameter(3),
                           {"runtime-abi-mirror", "materialized-store-base",
                            "header-mirror"});
    addRouteOperandBinding(plan, "n", parameter(4),
                           {"runtime-abi-mirror", "setvl-avl",
                            "loop-control", "header-mirror"});
    break;
  case RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    addRouteOperandBinding(
        plan, "lhs", parameter(0),
        {"runtime-abi-mirror", "materialized-source-load-base",
         "dequant-src", "src-i32m1",
         "relation-signed-i32m1-to-f32m1-scale-f32",
         "lower-compare-lhs-call", "lower-select-false-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "scale", parameter(1),
        {"runtime-abi-mirror", "runtime-scale", "scale-f32",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "lower_bound", parameter(2),
        {"runtime-abi-mirror", "lower-bound-splat-call",
         "lower-compare-rhs-call", "lower-select-true-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "upper_bound", parameter(3),
        {"runtime-abi-mirror", "upper-bound-splat-call",
         "upper-compare-lhs-call", "upper-select-true-call",
         "header-mirror"});
    addRouteOperandBinding(plan, "out", parameter(4),
                           {"runtime-abi-mirror", "materialized-store-base",
                            "dequant-result", "res-f32m1",
                            "relation-signed-i32m1-to-f32m1-scale-f32",
                            "header-mirror"});
    addRouteOperandBinding(plan, "n", parameter(5),
                           {"runtime-abi-mirror", "setvl-avl",
                            "loop-control", "header-mirror"});
    break;
  default:
    break;
  }

  return plan;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVRouteOperandBindingPlan(const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyOperationKind operation =
      analysis.description.operation;
  const RVVSelectedBodyMemoryForm memoryForm = analysis.description.memoryForm;
  RVVRouteOperandBindingPlan plan;
  llvm::StringRef expectedRuntimeABIOrder;
  llvm::StringRef context;
  std::string dynamicContext;

  if (isRVVSelectedBodyMAccRouteFamilyConsumer(operation))
    return deriveRVVSelectedBodyMAccRouteOperandBindingPlan(analysis);
  if (isRVVSelectedBodyContractionRouteFamilyConsumer(operation))
    return deriveRVVSelectedBodyContractionRouteOperandBindingPlan(analysis);
  if (operation == RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskSegment2LoadUnitStore ||
      operation == RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskSegment2StoreUnitLoad ||
      operation == RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad ||
      operation == RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore ||
      operation == RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad)
    return deriveRVVSelectedBodySegment2RouteOperandBindingPlan(analysis);
  if (isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
          operation, memoryForm) ||
      isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer(operation))
    return deriveRVVSelectedBodyElementwiseRouteOperandBindingPlan(analysis);

  if (isRVVSelectedBodyBaseMemoryMovementRouteFamilyConsumer(operation))
    return deriveRVVSelectedBodyBaseMemoryRouteOperandBindingPlan(analysis);

  if (operation == RVVSelectedBodyOperationKind::CmpSelect) {
    expectedRuntimeABIOrder = kRVVGenericBinaryRuntimeABIOrder;
    context = "cmp_select route";
    std::optional<RVVCompareSelectRouteFacts> routeFacts =
        getRVVCompareSelectRouteFacts(operation);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "cmp_select route operand binding requires provider-owned "
          "compare/select canonical route facts");
    routeFacts->runtimeABIParameters.clear();
    routeFacts->runtimeABIParameters.push_back(slice.lhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.rhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.outABI);
    routeFacts->runtimeABIParameters.push_back(slice.runtimeElementCountABI);
    plan = buildCompareSelectRouteOperandBindingPlanFromFacts(*routeFacts);
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ComputedMaskSelect) {
    expectedRuntimeABIOrder = kRVVComputedMaskSelectRuntimeABIOrder;
    context = "computed_mask_select route";
    std::optional<RVVCompareSelectRouteFacts> routeFacts =
        getRVVCompareSelectRouteFacts(slice.arithmeticKind);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "computed_mask_select route operand binding requires provider-owned "
          "compare/select canonical route facts");
    routeFacts->runtimeABIParameters.clear();
    routeFacts->runtimeABIParameters.push_back(slice.lhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.rhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.trueValueABI);
    routeFacts->runtimeABIParameters.push_back(slice.falseValueABI);
    routeFacts->runtimeABIParameters.push_back(slice.outABI);
    routeFacts->runtimeABIParameters.push_back(slice.runtimeElementCountABI);
    plan = buildCompareSelectRouteOperandBindingPlanFromFacts(*routeFacts);
  } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                      RuntimeScalarDualCompareMaskAndSelect) {
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarDualCompareMaskAndSelectRuntimeABIOrder;
    context = "runtime_scalar_dual_cmp_mask_and_select route";
    std::optional<RVVCompareSelectRouteFacts> routeFacts =
        getRVVCompareSelectRouteFacts(slice.arithmeticKind);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "runtime_scalar_dual_cmp_mask_and_select route operand binding "
          "requires provider-owned compare/select canonical route facts");
    routeFacts->runtimeABIParameters.clear();
    routeFacts->runtimeABIParameters.push_back(slice.lhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.rhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.secondaryCompareLhsABI);
    routeFacts->runtimeABIParameters.push_back(
        slice.secondaryCompareRhsScalarABI);
    routeFacts->runtimeABIParameters.push_back(slice.trueValueABI);
    routeFacts->runtimeABIParameters.push_back(slice.falseValueABI);
    routeFacts->runtimeABIParameters.push_back(slice.outABI);
    routeFacts->runtimeABIParameters.push_back(slice.runtimeElementCountABI);
    plan = buildCompareSelectRouteOperandBindingPlanFromFacts(*routeFacts);
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::F32ClampSelect) {
    expectedRuntimeABIOrder = kRVVF32ClampSelectRuntimeABIOrder;
    context = "f32_clamp_select route";
    std::optional<RVVCompareSelectRouteFacts> routeFacts =
        getRVVCompareSelectRouteFacts(slice.arithmeticKind);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "f32_clamp_select route operand binding requires provider-owned "
          "compare/select canonical route facts");
    routeFacts->runtimeABIParameters.clear();
    routeFacts->runtimeABIParameters.push_back(slice.lhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.lowerBoundABI);
    routeFacts->runtimeABIParameters.push_back(slice.upperBoundABI);
    routeFacts->runtimeABIParameters.push_back(slice.outABI);
    routeFacts->runtimeABIParameters.push_back(slice.runtimeElementCountABI);
    plan = buildCompareSelectRouteOperandBindingPlanFromFacts(*routeFacts);
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
    expectedRuntimeABIOrder = kRVVDequantClampF32EpilogueRuntimeABIOrder;
    context = "dequant_clamp_f32_epilogue route";
    std::optional<RVVCompareSelectRouteFacts> routeFacts =
        getRVVCompareSelectRouteFacts(slice.arithmeticKind);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "dequant_clamp_f32_epilogue route operand binding requires "
          "provider-owned compare/select canonical route facts");
    routeFacts->runtimeABIParameters.clear();
    routeFacts->runtimeABIParameters.push_back(slice.lhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.dequantScaleABI);
    routeFacts->runtimeABIParameters.push_back(slice.lowerBoundABI);
    routeFacts->runtimeABIParameters.push_back(slice.upperBoundABI);
    routeFacts->runtimeABIParameters.push_back(slice.outABI);
    routeFacts->runtimeABIParameters.push_back(slice.runtimeElementCountABI);
    plan = buildCompareSelectRouteOperandBindingPlanFromFacts(*routeFacts);
  } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::ReduceAdd) {
    plan.planID = kRVVReduceAddOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVGenericBinaryRuntimeABIOrder;
    context = "reduce_add route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base",
         "reduction-input-call"});
    addRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"runtime-abi-mirror", "materialized-accumulator-load-base",
         "reduction-accumulator-call"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base",
         "reduction-result-store", "header-mirror"});
    addRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                           {"runtime-abi-mirror", "setvl-avl",
                            "loop-control", "header-mirror"});
  } else if (slice.arithmeticKind ==
                 RVVSelectedBodyOperationKind::WidenI32ToI64 ||
             slice.arithmeticKind ==
                 RVVSelectedBodyOperationKind::WidenI16ToI32) {
    const bool isI16ToI32 =
        slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI16ToI32;
    plan.planID = isI16ToI32 ? kRVVWidenI16ToI32OperandBindingPlanID.str()
                             : kRVVWidenI32ToI64OperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVWideningConversionRuntimeABIOrder;
    context = isI16ToI32 ? "widen_i16_to_i32 conversion route"
                         : "widen_i32_to_i64 conversion route";
    const llvm::StringRef sourceConfigUse =
        isI16ToI32 ? "src-i16mf2" : "src-i32m1";
    const llvm::StringRef resultConfigUse =
        isI16ToI32 ? "res-i32m1" : "res-i64m2";
    const llvm::StringRef relationUse =
        isI16ToI32 ? "relation-signed-i16mf2-to-i32m1"
                   : "relation-signed-i32m1-to-i64m2";
    addRouteOperandBinding(plan, "lhs", slice.lhsABI,
                           {"abi", "src-load", "convert-src",
                            sourceConfigUse, relationUse, "hdr"});
    addRouteOperandBinding(plan, "out", slice.outABI,
                           {"abi", "res-store", "convert-result",
                            resultConfigUse, relationUse, "hdr"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::DequantizeI32ToF32) {
    plan.planID = kRVVDequantizeI32ToF32OperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVDequantizeI32ToF32RuntimeABIOrder;
    context = "dequantize_i32_to_f32 route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "dequant-src", "src-i32m1",
         "relation-signed-i32m1-to-f32m1-scale-f32", "hdr"});
    addRouteOperandBinding(plan, "scale", slice.dequantScaleABI,
                           {"abi", "runtime-scale", "scale-f32", "hdr"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "res-store", "dequant-result", "res-f32m1",
         "relation-signed-i32m1-to-f32m1-scale-f32", "hdr"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore) {
    plan.planID = kRVVRuntimeScalarSplatStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVRuntimeScalarSplatStoreRuntimeABIOrder;
    context = "runtime_scalar_splat_store route";
    addRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"runtime-abi-mirror", "runtime-scalar-splat-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"runtime-abi-mirror", "materialized-store-base",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect) {
    expectedRuntimeABIOrder = kRVVRuntimeScalarCompareSelectRuntimeABIOrder;
    context = "runtime_scalar_cmp_select route";
    std::optional<RVVCompareSelectRouteFacts> routeFacts =
        getRVVCompareSelectRouteFacts(slice.arithmeticKind);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "runtime_scalar_cmp_select route operand binding requires "
          "provider-owned compare/select canonical route facts");
    routeFacts->runtimeABIParameters.clear();
    routeFacts->runtimeABIParameters.push_back(slice.lhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.rhsABI);
    routeFacts->runtimeABIParameters.push_back(slice.trueValueABI);
    routeFacts->runtimeABIParameters.push_back(slice.falseValueABI);
    routeFacts->runtimeABIParameters.push_back(slice.outABI);
    routeFacts->runtimeABIParameters.push_back(slice.runtimeElementCountABI);
    plan = buildCompareSelectRouteOperandBindingPlanFromFacts(*routeFacts);
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore) {
    plan.planID =
        kRVVRuntimeScalarComputedMaskStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarComputedMaskStoreRuntimeABIOrder;
    context = "runtime_scalar_cmp_masked_store route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "compare-lhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
         "compare-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"runtime-abi-mirror", "materialized-source-load-base",
         "masked-store-source-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"runtime-abi-mirror", "materialized-masked-store-base",
         "masked-store-destination-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore) {
    plan.planID =
        kRVVRuntimeScalarComputedMaskLoadStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarComputedMaskLoadStoreRuntimeABIOrder;
    context = "runtime_scalar_cmp_masked_load_store route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"runtime-abi-mirror", "materialized-load-base", "compare-lhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
         "compare-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"runtime-abi-mirror", "materialized-masked-load-base",
         "masked-load-source-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"runtime-abi-mirror", "materialized-old-destination-load-base",
         "masked-load-passthrough-call", "materialized-store-base",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"runtime-abi-mirror", "setvl-avl", "loop-control",
         "header-mirror"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::
                 RuntimeScalarComputedMaskUnitStrideStandaloneReduction) {
    std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
        getRVVStandaloneReductionRouteFacts(slice.arithmeticKind);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "standalone reduction route operand binding requires provider-owned "
          "add/min/max canonical route facts");
    plan.planID = routeFacts->routeOperandBindingPlanID.str();
    expectedRuntimeABIOrder = routeFacts->runtimeABIOrder;
    dynamicContext =
        (stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) + " route")
            .str();
    context = dynamicContext;
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi", "cmp-lhs-load", "cmp-lhs-call", "hdr"});
    addRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"abi", "splat", "cmp-rhs-call", "hdr"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi", "src-load", "masked-reduce-input",
         routeFacts->inactiveLaneUse.str(), "hdr"});
    addRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "initial-seed", "acc-state", "masked-reduce-acc", "hdr"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "acc-state", "store-base", "hdr"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
  } else if (slice.memoryForm == RVVSelectedBodyMemoryForm::
                                    ComputedMaskUnitStrideStandaloneReduction) {
    std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
        getRVVStandaloneReductionRouteFacts(slice.arithmeticKind);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "standalone reduction route operand binding requires provider-owned "
          "add/min/max canonical route facts");
    plan.planID = routeFacts->routeOperandBindingPlanID.str();
    expectedRuntimeABIOrder = routeFacts->runtimeABIOrder;
    dynamicContext =
        (stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) + " route")
            .str();
    context = dynamicContext;
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi", "cmp-lhs-load", "cmp-lhs-call", "hdr"});
    addRouteOperandBinding(
        plan, "cmp_rhs", slice.rhsABI,
        {"abi", "cmp-rhs-load", "cmp-rhs-call", "hdr"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi", "src-load", "masked-reduce-input",
         routeFacts->inactiveLaneUse.str(), "hdr"});
    addRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "initial-seed", "acc-state", "masked-reduce-acc", "hdr"});
    addRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "acc-state", "store-base", "hdr"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
  } else if (slice.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction) {
    std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
        getRVVStandaloneReductionRouteFacts(slice.arithmeticKind);
    if (!routeFacts)
      return makeRVVEmitCRouteProviderError(
          "standalone reduction route operand binding requires provider-owned "
          "add/min/max canonical route facts");
    plan.planID = routeFacts->routeOperandBindingPlanID.str();
    expectedRuntimeABIOrder = routeFacts->runtimeABIOrder;
    dynamicContext =
        (stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) + " route")
            .str();
    context = dynamicContext;
    const bool isWideningStandalone =
        slice.arithmeticKind ==
        RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd;
    llvm::SmallVector<llvm::StringRef, 5> lhsUses = {"abi", "load",
                                                     "reduce-input"};
    if (isWideningStandalone)
      lhsUses.push_back("src-i16mf2");
    lhsUses.push_back("hdr");
    addRouteOperandBinding(plan, "lhs", slice.lhsABI, lhsUses);
    llvm::SmallVector<llvm::StringRef, 5> accUses = {"abi", "seed",
                                                     "acc-state"};
    if (isWideningStandalone)
      accUses.push_back("acc-i32");
    accUses.push_back("hdr");
    addRouteOperandBinding(plan, "acc", slice.accumulatorABI, accUses);
    llvm::SmallVector<llvm::StringRef, 5> outUses = {"abi", "acc-state",
                                                     "store"};
    if (isWideningStandalone)
      outUses.push_back("res-i32m1");
    outUses.push_back("hdr");
    addRouteOperandBinding(plan, "out", slice.outABI, outUses);
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore) {
    plan.planID = kRVVComputedMaskUnitLoadStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskMemoryRuntimeABIOrder;
    context = "computed_masked_unit_load_store route";
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi-mirror", "cmp-lhs-load", "compare-lhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", slice.rhsABI,
        {"abi-mirror", "cmp-rhs-load", "compare-rhs-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi-mirror", "materialized-masked-load-base",
         "masked-load-source-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"abi-mirror", "old-dst-load", "masked-load-passthrough-call",
         "materialized-store-base", "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi-mirror", "setvl-avl", "loop-control", "header-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedStore) {
    plan.planID = kRVVComputedMaskStridedStoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskStridedStoreRuntimeABIOrder;
    context = "computed_masked_strided_store route";
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi-mirror", "cmp-lhs-load", "cmp-lhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", slice.rhsABI,
        {"abi-mirror", "cmp-rhs-load", "cmp-rhs-call", "header-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi-mirror", "src-load", "mstr-store-src-call",
         "header-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"abi-mirror", "mstr-store-base", "header-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi-mirror", "setvl-avl", "loop-control", "header-mirror"});
    addRouteOperandBinding(
        plan, "dst_stride_bytes", slice.outStrideABI,
        {"abi-mirror", "mstr-store-stride", "byte", "header-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore) {
    plan.planID = kRVVComputedMaskStridedLoadOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskStridedLoadRuntimeABIOrder;
    context = "computed_masked_strided_load_unit_store route";
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi", "cmp-lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", slice.rhsABI,
        {"abi", "cmp-rhs-load", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi", "mstr-base", "mstr-load-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"abi", "old-dst-load", "passthru-call", "store-base",
         "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src_stride_bytes", slice.sourceStrideABI,
        {"abi", "mstr-stride", "byte", "hdr-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 ComputedMaskIndexedGatherLoadUnitStore) {
    plan.planID = kRVVComputedMaskIndexedGatherOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskIndexedGatherRuntimeABIOrder;
    context = "computed_masked_indexed_gather_load_unit_store route";
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi", "cmp-lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", slice.rhsABI,
        {"abi", "cmp-rhs-load", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi", "midx-base", "midx-load-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "index", slice.indexABI,
        {"abi", "materialized-index-load-base", "index-offset-scale",
         "index-source-mirror", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"abi", "old-dst-load", "passthru-call", "store-base",
         "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedGatherLoadUnitStore) {
    plan.planID =
        kRVVRuntimeScalarComputedMaskIndexedGatherOperandBindingPlanID.str();
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarComputedMaskIndexedGatherRuntimeABIOrder;
    context =
        "runtime_scalar_cmp_masked_indexed_gather_load_unit_store route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"abi", "splat", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi", "midx-base", "midx-load-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "index", slice.indexABI,
        {"abi", "materialized-index-load-base", "index-offset-scale",
         "index-source-mirror", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"abi", "old-dst-load", "passthru-call", "store-base",
         "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 ComputedMaskIndexedScatterStoreUnitLoad) {
    plan.planID = kRVVComputedMaskIndexedScatterOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskIndexedScatterRuntimeABIOrder;
    context = "computed_masked_indexed_scatter_store_unit_load route";
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI,
        {"abi", "cmp-lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "cmp_rhs", slice.rhsABI,
        {"abi", "cmp-rhs-load", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi", "src-load", "mistore-src-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "index", slice.indexABI,
        {"abi", "materialized-index-load-base", "index-offset-scale",
         "index-source-mirror", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"abi", "mistore-base", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad) {
    plan.planID =
        kRVVRuntimeScalarComputedMaskIndexedScatterOperandBindingPlanID.str();
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarComputedMaskIndexedScatterRuntimeABIOrder;
    context =
        "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load route";
    addRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "lhs-load", "lhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"abi", "splat", "rhs-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "src", slice.sourceABI,
        {"abi", "src-load", "mistore-src-call", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "index", slice.indexABI,
        {"abi", "materialized-index-load-base", "index-offset-scale",
         "index-source-mirror", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "dst", slice.outABI,
        {"abi", "mistore-base", "hdr-mirror"});
    addRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr-mirror"});
  } else if (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    plan.planID =
        kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterOperandBindingPlanID
            .str();
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIOrder;
    context = "runtime_scalar_cmp_masked_indexed_gather_macc_scatter route";
    addRouteOperandBinding(
        plan, "cmp_lhs", slice.lhsABI, {"abi", "ld", "cmp-lhs", "hdr"});
    addRouteOperandBinding(
        plan, "rhs_scalar", slice.rhsABI,
        {"abi", "splat", "cmp-rhs", "hdr"});
    addRouteOperandBinding(
        plan, "gather_src", slice.sourceABI,
        {"abi", "gbase", "gather", "hdr"});
    addRouteOperandBinding(plan, "payload", slice.dotRHSABI,
                           {"abi", "ld", "macc-rhs", "hdr"});
    addRouteOperandBinding(plan, "acc", slice.accumulatorABI,
                           {"abi", "ld", "macc-acc", "hdr"});
    addRouteOperandBinding(
        plan, "index", slice.indexABI,
        {"abi", "idxld", "idxscale", "idxmirror", "hdr"});
    addRouteOperandBinding(plan, "dst", slice.outABI,
                           {"abi", "olddst", "gpass", "scatter", "hdr"});
    addRouteOperandBinding(plan, "n", slice.runtimeElementCountABI,
                           {"abi", "setvl-avl", "loop", "hdr"});
  }

  if (plan.planID.empty())
    return plan;

  if (llvm::Error error = verifyRVVRouteOperandBindingPlan(
          plan, plan.planID, expectedRuntimeABIOrder, context))
    return std::move(error);
  if (expectedRuntimeABIOrder != analysis.description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires description runtime ABI order '" +
        expectedRuntimeABIOrder + "' but found '" +
        analysis.description.runtimeABIOrder + "'");
  llvm::SmallVector<support::RuntimeABIParameter, 8> planParameters;
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    planParameters.push_back(binding.parameter);
  if (!support::runtimeABIParametersEqual(
          planParameters, analysis.description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context +
        " requires runtime ABI parameter mirrors to match the binding plan; "
        "plan has '" +
        stringifyRVVRuntimeABIParametersForDiagnostic(planParameters) +
        "' while description has '" +
        stringifyRVVRuntimeABIParametersForDiagnostic(
            analysis.description.runtimeABIParameters) +
        "'; slice operation '" +
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
        "', slice memory form '" +
        stringifyRVVSelectedBodyMemoryForm(slice.memoryForm) +
        "', description operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "', description memory form '" +
        stringifyRVVSelectedBodyMemoryForm(analysis.description.memoryForm) +
        "'");
  return plan;
}

} // namespace tianchenrv::plugin::rvv
