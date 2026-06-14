//===- RVVEmitCRouteAnalysis.cpp - RVV route slice collection + roles -----===//
//
// Behavior-preserving split out of RVVEmitCRoutePlanning.cpp: the route-slice
// collection machinery -- the recordRVVSelectedBody* op-recording helpers,
// collectRVVSelectedBodyRouteSlice (the slice gatherer), and the canonical
// role-ordering + role-sequence verification (getRVVCanonicalRoleOrder /
// collectRVVRoleOperationsInBodyOrder / verifySelectedRVVRoleSequence). The
// file-local record/role helpers stay in anonymous namespaces (with the
// pre-split using-declarations that restore the relocated-primitive overload
// set); only collectRVVSelectedBodyRouteSlice and verifySelectedRVVRoleSequence
// are promoted to external linkage (declared in the co-located
// RVVEmitCRoutePlanningInternal.h). Pure relocation -- bodies byte-identical.
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

namespace {
// Restore the pre-split single-namespace overload set so unqualified calls to
// the relocated (sew, lmul) primitive builders resolve identically.
using ::tianchenrv::plugin::rvv::getRVVSelectedBodyArithmeticIntrinsic;
using ::tianchenrv::plugin::rvv::getRVVSelectedBodyMaskAndIntrinsic;
using ::tianchenrv::plugin::rvv::getRVVSelectedBodySelectIntrinsic;

llvm::Error recordRVVSelectedBodyOperation(RVVSelectedBodyRouteSlice &slice,
                                           mlir::Operation *op,
                                           RVVSelectedBodyOperationKind kind,
                                           mlir::Value lhs, mlir::Value rhs,
                                           mlir::Value result) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one supported "
        "generic tcrv_rvv.binary op");
  slice.arithmeticOp = op;
  slice.arithmeticKind = kind;
  slice.arithmeticLhs = lhs;
  slice.arithmeticRhs = rhs;
  slice.arithmeticResult = result;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyCompare(RVVSelectedBodyRouteSlice &slice,
                                         tcrv::rvv::CompareOp compare) {
  if (slice.compareOp && slice.secondaryCompareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route supports at most two generic "
        "tcrv_rvv.compare ops for the dual runtime scalar mask-and route");
  if (compare.getKind() != "eq" && compare.getKind() != "slt" &&
      compare.getKind() != "sle")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.compare kind '") +
        compare.getKind() + "' for bounded RVV EmitC route");
  if (slice.compareOp) {
    slice.secondaryCompareOp = compare;
    slice.secondaryCompareLhs = compare.getLhs();
    slice.secondaryCompareRhs = compare.getRhs();
    slice.secondaryCompareMask = compare.getMask();
    return llvm::Error::success();
  }
  slice.compareOp = compare;
  slice.compareLhs = compare.getLhs();
  slice.compareRhs = compare.getRhs();
  slice.compareMask = compare.getMask();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskAnd(RVVSelectedBodyRouteSlice &slice,
                                         tcrv::rvv::MaskAndOp maskAnd) {
  if (slice.maskAndOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.mask_and op "
        "for the dual runtime scalar mask composition route");
  if (maskAnd.getKind() != "and")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.mask_and kind '") +
        maskAnd.getKind() + "' for bounded RVV EmitC route");
  slice.maskAndOp = maskAnd;
  slice.composedMask = maskAnd.getMask();
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyMaskedBinaryKind(llvm::StringRef kind);

llvm::Error
recordRVVSelectedBodyMaskedBinary(RVVSelectedBodyRouteSlice &slice,
                                  tcrv::rvv::MaskedBinaryOp maskedBinary) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  llvm::Expected<RVVSelectedBodyOperationKind> kind =
      parseRVVSelectedBodyMaskedBinaryKind(maskedBinary.getKind());
  if (!kind)
    return kind.takeError();
  slice.maskedBinaryOp = maskedBinary;
  slice.arithmeticOp = maskedBinary.getOperation();
  slice.arithmeticKind = *kind;
  slice.compareMask = maskedBinary.getMask();
  slice.maskedPassthrough = maskedBinary.getPassthrough();
  slice.arithmeticLhs = maskedBinary.getLhs();
  slice.arithmeticRhs = maskedBinary.getRhs();
  slice.arithmeticResult = maskedBinary.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodySelect(RVVSelectedBodyRouteSlice &slice,
                                        tcrv::rvv::SelectOp select) {
  if (slice.selectOp && slice.secondarySelectOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route supports at most two tcrv_rvv.select ops "
        "for the f32 clamp/select route");
  if (slice.selectOp) {
    const bool extendsProductReductionDequantClamp =
        (slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
         slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 WideningProductReduceDequantClampF32) &&
        slice.dequantizeOp && slice.standaloneReduceOp &&
        hasProductHead(slice);
    slice.secondarySelectOp = select;
    slice.arithmeticOp = select.getOperation();
    slice.arithmeticKind =
        extendsProductReductionDequantClamp
            ? RVVSelectedBodyOperationKind::
                  WideningProductReduceDequantClampF32
        : slice.dequantizeOp ? RVVSelectedBodyOperationKind::
                                   DequantClampF32Epilogue
                             : RVVSelectedBodyOperationKind::F32ClampSelect;
    slice.arithmeticLhs = select.getTrueValue();
    slice.arithmeticRhs = select.getFalseValue();
    slice.arithmeticResult = select.getSelected();
    slice.lowerClampedValue = slice.selectOp.getSelected();
    return llvm::Error::success();
  }
  const bool extendsDequantClampEpilogue =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::DequantizeI32ToF32 &&
      slice.dequantizeOp;
  const bool extendsProductReductionDequantClamp =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 &&
      slice.dequantizeOp && slice.standaloneReduceOp && hasProductHead(slice);
  if (slice.arithmeticOp && !extendsDequantClampEpilogue &&
      !extendsProductReductionDequantClamp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  slice.selectOp = select;
  slice.arithmeticOp = select.getOperation();
  slice.arithmeticKind =
      extendsProductReductionDequantClamp
          ? RVVSelectedBodyOperationKind::
                WideningProductReduceDequantClampF32
      : extendsDequantClampEpilogue
          ? RVVSelectedBodyOperationKind::DequantClampF32Epilogue
          : RVVSelectedBodyOperationKind::CmpSelect;
  slice.arithmeticLhs = select.getTrueValue();
  slice.arithmeticRhs = select.getFalseValue();
  slice.arithmeticResult = select.getSelected();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyReduction(RVVSelectedBodyRouteSlice &slice,
                                           tcrv::rvv::ReduceOp reduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (reduce.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce kind '") +
        reduce.getKind() + "' for bounded RVV reduction route");
  std::optional<llvm::StringRef> accumulatorLayout =
      reduce.getAccumulatorLayout();
  if (!accumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV reduction route requires tcrv_rvv.reduce to carry "
        "accumulator_layout 'rhs-vector-seed-lane0-per-vl-chunk'");
  if (*accumulatorLayout != kRVVReductionAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce accumulator_layout '") +
        *accumulatorLayout + "' for bounded RVV reduction route");
  std::optional<llvm::StringRef> resultLayout = reduce.getResultLayout();
  if (!resultLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV reduction route requires tcrv_rvv.reduce to carry "
        "result_layout 'store-reduction-lane0-to-output-chunk-base'");
  if (*resultLayout != kRVVReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce result_layout '") +
        *resultLayout + "' for bounded RVV reduction route");
  slice.reduceOp = reduce;
  slice.arithmeticOp = reduce.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::ReduceAdd;
  slice.arithmeticLhs = reduce.getInput();
  slice.arithmeticRhs = reduce.getAccumulator();
  slice.arithmeticResult = reduce.getResult();
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyStandaloneReductionKind(llvm::StringRef kind) {
  if (kind == "add")
    return RVVSelectedBodyOperationKind::StandaloneReduceAdd;
  if (kind == "min")
    return RVVSelectedBodyOperationKind::StandaloneReduceMin;
  if (kind == "max")
    return RVVSelectedBodyOperationKind::StandaloneReduceMax;
  if (kind == "signed_widening_reduce_add" ||
      kind == "unsigned_widening_reduce_add")
    return RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd;
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported generic tcrv_rvv.standalone_reduce kind '") +
      kind + "' for bounded RVV standalone reduction route");
}

llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyComputedMaskStandaloneReductionKind(llvm::StringRef kind) {
  if (kind == "add")
    return RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd;
  if (kind == "min")
    return RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin;
  if (kind == "max")
    return RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax;
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported generic tcrv_rvv.masked_standalone_reduce kind '") +
      kind + "' for bounded RVV computed-mask standalone reduction route");
}

llvm::Error recordRVVSelectedBodyStandaloneReduction(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::StandaloneReduceOp standaloneReduce) {
  llvm::Expected<RVVSelectedBodyOperationKind> reductionKind =
      parseRVVSelectedBodyStandaloneReductionKind(standaloneReduce.getKind());
  if (!reductionKind)
    return reductionKind.takeError();
  if (standaloneReduce.getAccumulatorLayout() !=
          kRVVStandaloneReductionAccumulatorLayout &&
      standaloneReduce.getAccumulatorLayout() !=
          kRVVStandaloneReductionI64AccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.standalone_reduce "
                    "accumulator_layout '") +
        standaloneReduce.getAccumulatorLayout() +
        "' for bounded RVV standalone reduction route");
  if (standaloneReduce.getResultLayout() !=
      kRVVStandaloneReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.standalone_reduce "
                    "result_layout '") +
        standaloneReduce.getResultLayout() +
        "' for bounded RVV standalone reduction route");
  if (slice.arithmeticOp) {
    if (slice.arithmeticKind != RVVSelectedBodyOperationKind::WideningProduct ||
        !hasProductHead(slice) ||
        *reductionKind !=
            RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires exactly one selected compute op "
          "unless a typed tcrv_rvv.widening_product or "
          "tcrv_rvv.packed_i4_nibble_unpack_product feeds a matching "
          "signed or unsigned tcrv_rvv.standalone_reduce widening add chain");
    const bool isSignedProductReduction =
        productSlotIsSigned(slice) &&
        standaloneReduce.getKind() == "signed_widening_reduce_add";
    const bool isUnsignedProductReduction =
        productSlotIsUnsigned(slice) &&
        standaloneReduce.getKind() == "unsigned_widening_reduce_add";
    if (!isSignedProductReduction && !isUnsignedProductReduction)
      return makeRVVEmitCRouteProviderError(
          "low-precision product-reduction RVV route requires widening "
          "product signedness to match standalone widening-reduction "
          "signedness");
    if (standaloneReduce.getInput() != productSlotResult(slice))
      return makeRVVEmitCRouteProviderError(
          "low-precision product-reduction RVV route requires "
          "tcrv_rvv.standalone_reduce input to consume the selected "
          "tcrv_rvv.widening_product result");
    slice.standaloneReduceOp = standaloneReduce;
    slice.arithmeticOp = standaloneReduce.getOperation();
    slice.arithmeticKind = RVVSelectedBodyOperationKind::WideningProductReduceAdd;
    slice.arithmeticLhs = productSlotLhs(slice);
    slice.arithmeticRhs = productSlotRhs(slice);
    slice.arithmeticAccumulator = standaloneReduce.getAccumulatorSeed();
    slice.accumulatorBuffer = standaloneReduce.getAccumulatorSeed();
    slice.arithmeticResult = standaloneReduce.getResult();
    slice.memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
    return llvm::Error::success();
  }
  slice.standaloneReduceOp = standaloneReduce;
  slice.arithmeticOp = standaloneReduce.getOperation();
  slice.arithmeticKind = *reductionKind;
  slice.arithmeticLhs = standaloneReduce.getInput();
  slice.arithmeticAccumulator = standaloneReduce.getAccumulatorSeed();
  slice.accumulatorBuffer = standaloneReduce.getAccumulatorSeed();
  slice.arithmeticResult = standaloneReduce.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedStandaloneReduction(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedStandaloneReduceOp maskedStandaloneReduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  llvm::Expected<RVVSelectedBodyOperationKind> reductionKind =
      parseRVVSelectedBodyComputedMaskStandaloneReductionKind(
          maskedStandaloneReduce.getKind());
  if (!reductionKind)
    return reductionKind.takeError();
  if (maskedStandaloneReduce.getMaskRole() != kRVVMaskedPredicateMaskRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_standalone_reduce "
                    "mask_role '") +
        maskedStandaloneReduce.getMaskRole() +
        "' for bounded RVV computed-mask standalone reduction route");
  if (maskedStandaloneReduce.getMaskSource() != kRVVMaskedCompareMaskSource)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_standalone_reduce "
                    "mask_source '") +
        maskedStandaloneReduce.getMaskSource() +
        "' for bounded RVV computed-mask standalone reduction route");
  if (maskedStandaloneReduce.getMaskMemoryForm() !=
      kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_standalone_reduce "
                    "mask_memory_form '") +
        maskedStandaloneReduce.getMaskMemoryForm() +
        "' for bounded RVV computed-mask standalone reduction route");
  if (maskedStandaloneReduce.getAccumulatorLayout() !=
          kRVVStandaloneReductionAccumulatorLayout &&
      maskedStandaloneReduce.getAccumulatorLayout() !=
          kRVVStandaloneReductionI64AccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_standalone_reduce "
                    "accumulator_layout '") +
        maskedStandaloneReduce.getAccumulatorLayout() +
        "' for bounded RVV computed-mask standalone reduction route");
  if (maskedStandaloneReduce.getResultLayout() !=
      kRVVStandaloneReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_standalone_reduce "
                    "result_layout '") +
        maskedStandaloneReduce.getResultLayout() +
        "' for bounded RVV computed-mask standalone reduction route");
  slice.maskedStandaloneReduceOp = maskedStandaloneReduce;
  slice.arithmeticOp = maskedStandaloneReduce.getOperation();
  slice.arithmeticKind = *reductionKind;
  slice.compareMask = maskedStandaloneReduce.getMask();
  slice.maskValue = maskedStandaloneReduce.getMask();
  slice.arithmeticLhs = maskedStandaloneReduce.getInput();
  slice.arithmeticAccumulator =
      maskedStandaloneReduce.getAccumulatorSeed();
  slice.accumulatorBuffer = maskedStandaloneReduce.getAccumulatorSeed();
  slice.arithmeticResult = maskedStandaloneReduce.getResult();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideStandaloneReduction;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMAcc(RVVSelectedBodyRouteSlice &slice,
                                      tcrv::rvv::MAccOp macc) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (macc.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.macc kind '") +
        macc.getKind() + "' for bounded RVV multiply-accumulate route");
  std::optional<llvm::StringRef> accumulatorLayout =
      macc.getAccumulatorLayout();
  if (!accumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV multiply-accumulate route requires tcrv_rvv.macc to "
        "carry accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (*accumulatorLayout != getRVVSelectedBodyMAccAccumulatorLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.macc accumulator_layout '") +
        *accumulatorLayout + "' for bounded RVV multiply-accumulate route");
  std::optional<llvm::StringRef> resultLayout = macc.getResultLayout();
  if (!resultLayout)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV multiply-accumulate route requires tcrv_rvv.macc to "
        "carry result_layout "
        "'store-multiply-accumulate-result-to-output-buffer'");
  if (*resultLayout != getRVVSelectedBodyMAccResultLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.macc result_layout '") +
        *resultLayout + "' for bounded RVV multiply-accumulate route");
  slice.maccOp = macc;
  slice.arithmeticOp = macc.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MAccAdd;
  slice.arithmeticLhs = macc.getLhs();
  slice.arithmeticRhs = macc.getRhs();
  slice.arithmeticAccumulator = macc.getAccumulator();
  slice.arithmeticResult = macc.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedMAcc(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedMAccOp macc) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (macc.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_macc kind '") +
        macc.getKind() + "' for bounded RVV computed-mask macc route");
  if (macc.getMaskRole() != kRVVMaskedPredicateMaskRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_macc mask_role '") +
        macc.getMaskRole() + "' for bounded RVV computed-mask macc route");
  if (macc.getMaskSource() != kRVVMaskedCompareMaskSource)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_macc mask_source '") +
        macc.getMaskSource() + "' for bounded RVV computed-mask macc route");
  if (macc.getMaskMemoryForm() != kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.masked_macc mask_memory_form '") +
        macc.getMaskMemoryForm() +
        "' for bounded RVV computed-mask macc route");
  if (macc.getAccumulatorLayout() != getRVVSelectedBodyMAccAccumulatorLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.masked_macc accumulator_layout '") +
        macc.getAccumulatorLayout() +
        "' for bounded RVV computed-mask macc route");
  if (macc.getResultLayout() != getRVVSelectedBodyMAccResultLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_macc "
                    "result_layout '") +
        macc.getResultLayout() +
        "' for bounded RVV computed-mask macc route");
  slice.maskedMAccOp = macc;
  slice.arithmeticOp = macc.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd;
  slice.compareMask = macc.getMask();
  slice.maskValue = macc.getMask();
  slice.arithmeticLhs = macc.getLhs();
  slice.arithmeticRhs = macc.getRhs();
  slice.arithmeticAccumulator = macc.getAccumulator();
  slice.arithmeticResult = macc.getResult();
  slice.memoryForm = RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc;
  return llvm::Error::success();
}

llvm::Error
recordRVVSelectedBodyWideningMAcc(RVVSelectedBodyRouteSlice &slice,
                                  tcrv::rvv::WideningMAccOp macc) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (macc.getKind() != "signed_widening_macc_add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_macc kind '") +
        macc.getKind() +
        "' for bounded RVV widening multiply-accumulate route");
  if (macc.getAccumulatorLayout() !=
      getRVVSelectedBodyContractionExpectedWideningMAccAccumulatorLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_macc "
                    "accumulator_layout '") +
        macc.getAccumulatorLayout() +
        "' for bounded RVV widening multiply-accumulate route");
  if (macc.getResultLayout() !=
      getRVVSelectedBodyContractionExpectedWideningMAccResultLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_macc "
                    "result_layout '") +
        macc.getResultLayout() +
        "' for bounded RVV widening multiply-accumulate route");
  if (macc.getMaccRelation() !=
      getRVVSelectedBodyContractionExpectedWideningMAccRelation())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_macc "
                    "macc_relation '") +
        macc.getMaccRelation() +
        "' for bounded RVV widening multiply-accumulate route");
  slice.wideningMAccOp = macc;
  slice.arithmeticOp = macc.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::WideningMAccAdd;
  slice.arithmeticLhs = macc.getLhs();
  slice.arithmeticRhs = macc.getRhs();
  slice.arithmeticAccumulator = macc.getAccumulator();
  slice.arithmeticResult = macc.getResult();
  return llvm::Error::success();
}

llvm::Error
recordRVVSelectedBodyWideningProduct(RVVSelectedBodyRouteSlice &slice,
                                     tcrv::rvv::WideningProductOp product) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  const bool isUnsignedProduct =
      product.getKind() == "unsigned_widening_product" &&
      product.getProductRelation() == "unsigned-u8mf4xu8mf4-to-u16mf2";
  const bool isSignedProduct =
      product.getKind() == "signed_widening_product" &&
      product.getProductRelation() == "signed-i8mf4xi8mf4-to-i16mf2";
  if (!isSignedProduct && !isUnsignedProduct)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_product kind '") +
        product.getKind() +
        "' for bounded RVV low-precision widening-product route");
  slice.wideningProductOp = product;
  slice.arithmeticOp = product.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::WideningProduct;
  slice.arithmeticLhs = product.getLhs();
  slice.arithmeticRhs = product.getRhs();
  slice.arithmeticResult = product.getResult();
  return llvm::Error::success();
}

// Record a typed signed packed-i4 nibble-unpack widening product as the selected
// product head of a low-precision dequant chain. Mirrors
// recordRVVSelectedBodyWideningProduct but stores into slice.nibbleProductOp; the
// downstream standalone-reduce / dequant hatches read the product head through
// productSlotResult / hasProductHead so a single-scope nibble body threads the
// same WideningProduct->StandaloneReduce->Dequantize promotion path.
llvm::Error recordRVVSelectedBodyNibbleUnpackProduct(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::PackedI4NibbleUnpackProductOp product) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (product.getKind() != "signed_packed_i4_nibble_unpack_product" ||
      product.getProductRelation() != "signed-i8mf4xi8mf4-to-i16mf2")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.packed_i4_nibble_unpack_product kind '") +
        product.getKind() +
        "' for bounded RVV low-precision packed-i4 nibble-unpack product route");
  slice.nibbleProductOp = product;
  slice.arithmeticOp = product.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::WideningProduct;
  slice.arithmeticLhs = product.getLhs();
  slice.arithmeticRhs = product.getRhs();
  slice.arithmeticResult = product.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyWideningDotReduce(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::WideningDotReduceOp dotReduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (dotReduce.getKind() != "signed_widening_dot_reduce_add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.widening_dot_reduce kind '") +
        dotReduce.getKind() +
        "' for bounded RVV widening dot-product reduction route");
  if (dotReduce.getAccumulatorLayout() !=
      getRVVSelectedBodyContractionExpectedWideningDotProductAccumulatorLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_dot_reduce "
                    "accumulator_layout '") +
        dotReduce.getAccumulatorLayout() +
        "' for bounded RVV widening dot-product reduction route");
  if (dotReduce.getResultLayout() !=
      getRVVSelectedBodyContractionExpectedWideningDotProductResultLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_dot_reduce "
                    "result_layout '") +
        dotReduce.getResultLayout() +
        "' for bounded RVV widening dot-product reduction route");
  if (dotReduce.getDotProductRelation() !=
      getRVVSelectedBodyContractionExpectedWideningDotProductRelation())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_dot_reduce "
                    "dot_product_relation '") +
        dotReduce.getDotProductRelation() +
        "' for bounded RVV widening dot-product reduction route");
  slice.wideningDotReduceOp = dotReduce;
  slice.arithmeticOp = dotReduce.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::WideningDotReduceAdd;
  slice.arithmeticLhs = dotReduce.getLhs();
  slice.arithmeticRhs = dotReduce.getRhs();
  slice.arithmeticAccumulator = dotReduce.getAccumulatorSeed();
  slice.accumulatorBuffer = dotReduce.getAccumulatorSeed();
  slice.arithmeticResult = dotReduce.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedWideningDotReduce(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedWideningDotReduceOp dotReduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (dotReduce.getKind() != "signed_masked_widening_dot_reduce_add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(
            "unsupported generic tcrv_rvv.masked_widening_dot_reduce kind '") +
        dotReduce.getKind() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getMaskRole() != kRVVMaskedPredicateMaskRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce mask_role '") +
        dotReduce.getMaskRole() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getMaskSource() != kRVVMaskedCompareMaskSource)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce mask_source '") +
        dotReduce.getMaskSource() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getMaskMemoryForm() != kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce mask_memory_form '") +
        dotReduce.getMaskMemoryForm() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getAccumulatorLayout() !=
      getRVVSelectedBodyContractionExpectedWideningDotProductAccumulatorLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce "
                    "accumulator_layout '") +
        dotReduce.getAccumulatorLayout() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getResultLayout() !=
      getRVVSelectedBodyContractionExpectedWideningDotProductResultLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce result_layout '") +
        dotReduce.getResultLayout() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  if (dotReduce.getDotProductRelation() !=
      getRVVSelectedBodyContractionExpectedWideningDotProductRelation())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic "
                    "tcrv_rvv.masked_widening_dot_reduce "
                    "dot_product_relation '") +
        dotReduce.getDotProductRelation() +
        "' for bounded RVV computed-mask widening dot-product reduction "
        "route");
  slice.maskedWideningDotReduceOp = dotReduce;
  slice.arithmeticOp = dotReduce.getOperation();
  slice.arithmeticKind =
      RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  slice.compareMask = dotReduce.getMask();
  slice.maskValue = dotReduce.getMask();
  slice.arithmeticLhs = dotReduce.getLhs();
  slice.arithmeticRhs = dotReduce.getRhs();
  slice.arithmeticAccumulator = dotReduce.getAccumulatorSeed();
  slice.accumulatorBuffer = dotReduce.getAccumulatorSeed();
  slice.arithmeticResult = dotReduce.getResult();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideWideningDotReduce;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyWideningConvert(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::WideningConvertOp conversion) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (conversion.getKind() != "widen_i32_to_i64" &&
      conversion.getKind() != "sign_extend_widen_vf2")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.widening_convert kind '") +
        conversion.getKind() + "' for bounded RVV widening conversion route");
  slice.wideningConvertOp = conversion;
  slice.arithmeticOp = conversion.getOperation();
  slice.arithmeticKind =
      conversion.getKind() == "sign_extend_widen_vf2"
          ? RVVSelectedBodyOperationKind::WidenI16ToI32
          : RVVSelectedBodyOperationKind::WidenI32ToI64;
  slice.conversionSource = conversion.getSource();
  slice.arithmeticLhs = conversion.getSource();
  slice.arithmeticResult = conversion.getResult();
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitStrideConversion;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyGearboxCrossRegionHandoff(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::GearboxCrossRegionHandoffOp handoff,
    const support::RuntimeABIParameter &runtimeElementCountABI) {
  if (slice.gearboxCrossRegionHandoffOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one "
        "tcrv_rvv.gearbox_cross_region_handoff op for the low-precision "
        "product-reduction dequantization cross-region boundary");
  if (slice.arithmeticKind !=
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      !slice.standaloneReduceOp || !hasProductHead(slice))
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff to follow the selected "
        "tcrv_rvv.widening_product -> tcrv_rvv.standalone_reduce chain");
  if (handoff.getInput() != slice.standaloneReduceOp.getResult())
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff input to consume the selected "
        "tcrv_rvv.standalone_reduce i32 result");
  if (handoff.getVl() != slice.setvl.getVl())
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff to consume the selected "
        "!tcrv_rvv.vl token");
  if (handoff.getRuntimeAvl() != slice.setvl.getAvl())
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff runtime AVL operand to be the "
        "same runtime n/AVL SSA value consumed by tcrv_rvv.setvl");
  if (runtimeElementCountABI.role !=
      support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff runtime AVL source to bind "
        "runtime-element-count");
  const std::int64_t expectedLowPrecisionRegionCount =
      getRVVLowPrecisionResourceExpectedVSetVLRegionCountForRealizationDecision(
          handoff.getResourceDecision());
  const llvm::StringRef expectedLowPrecisionFromPhase =
      getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
          handoff.getResourceDecision());
  const bool hasSupportedLowPrecisionDecision =
      isRVVLowPrecisionResourceSupportedRealizationDecision(
          handoff.getResourceDecision());
  if (handoff.getContract() !=
          "gearbox-product-reduce-to-dequant-cross-region-handoff.v1" ||
      handoff.getFromPhase() != expectedLowPrecisionFromPhase ||
      handoff.getToPhase() != "dequant-store" ||
      static_cast<std::int64_t>(handoff.getRegionCount()) !=
          expectedLowPrecisionRegionCount ||
      handoff.getRuntimeAvlSource() != "runtime_abi:n" ||
      !hasSupportedLowPrecisionDecision ||
      handoff.getProducerScope() != kRVVGearboxProducerScope ||
      handoff.getConsumerScope() != kRVVGearboxConsumerScope ||
      handoff.getProducerScope() == handoff.getConsumerScope())
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff contract/from_phase/to_phase/"
        "region_count/runtime_avl_source/resource_decision/producer_scope/"
        "consumer_scope to match the "
        "RVV-owned Gearbox cross-region handoff contract");

  if (!isRVVLowPrecisionResourceCandidateSetMember(
          handoff.getResourceCandidateSet(),
          handoff.getResourceSelectedCandidate()))
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff resource_selected_candidate "
        "to belong to the provider-owned resource_candidate_set");
  auto planningContract =
      handoff->getAttrOfType<mlir::StringAttr>(
          kGearboxHandoffPlanningContractAttrName);
  if (!planningContract)
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff planning_contract from the "
        "selected resource plan");
  if (planningContract.getValue() != kRVVLowPrecisionResourcePlanningContract)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("low-precision product-reduction dequantization RVV "
                    "route requires tcrv_rvv.gearbox_cross_region_handoff "
                    "planning_contract to match provider-owned resource "
                    "planning contract '") +
        kRVVLowPrecisionResourcePlanningContract + "' but found '" +
        planningContract.getValue() + "'");
  const llvm::StringRef expectedDecisionFromCandidate =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          handoff.getResourceSelectedCandidate());
  if (expectedDecisionFromCandidate.empty() ||
      expectedDecisionFromCandidate != handoff.getResourceDecision())
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff resource_decision to match "
        "the selected resource candidate");
  const bool isPackedI4Resource =
      isRVVLowPrecisionResourcePackedI4CandidateID(
          handoff.getResourceSelectedCandidate());
  const llvm::StringRef expectedOperandForm =
      isPackedI4Resource ? kRVVLowPrecisionResourceOperandFormPackedI4Nibbles
                         : kRVVLowPrecisionResourceOperandFormUnpackedByte;
  const llvm::StringRef expectedPackingLayout =
      isPackedI4Resource
          ? kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles
          : kRVVLowPrecisionResourcePackingLayoutByte;
  const llvm::StringRef expectedUnpackIntent =
      isPackedI4Resource
          ? kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles
          : kRVVLowPrecisionResourceUnpackIntentNone;
  if (handoff.getOperandForm() != expectedOperandForm ||
      handoff.getPackingLayout() != expectedPackingLayout ||
      handoff.getUnpackIntent() != expectedUnpackIntent)
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff operand form, packing layout, "
        "and unpack intent to match the selected resource candidate");

  auto requireOptionalPackedI4LoadUnpackFact =
      [&](llvm::StringRef attrName, llvm::StringRef expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine("low-precision product-reduction dequantization RVV "
                        "route requires packed-i4 load/unpack fact '") +
            attrName + "' to be absent for unpacked-byte resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("low-precision product-reduction dequantization RVV "
                      "route requires tcrv_rvv.gearbox_cross_region_handoff "
                      "packed-i4 load/unpack fact '") +
          attrName + "' before route support");
    if (attr.getValue() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("low-precision product-reduction dequantization RVV "
                    "route requires tcrv_rvv.gearbox_cross_region_handoff "
                    "packed-i4 load/unpack fact '") +
        attrName + "' to match provider-owned resource fact '" + expected +
        "' but found '" + attr.getValue() + "'");
  };
  if (llvm::Error error = requireOptionalPackedI4LoadUnpackFact(
          "packed_load_unpack_contract",
          kRVVLowPrecisionResourcePackedI4LoadUnpackContract))
    return error;
  if (llvm::Error error = requireOptionalPackedI4LoadUnpackFact(
          "packed_storage_load", kRVVLowPrecisionResourcePackedI4StorageLoad))
    return error;
  if (llvm::Error error = requireOptionalPackedI4LoadUnpackFact(
          "packed_unpack_plan", kRVVLowPrecisionResourcePackedI4UnpackPlan))
    return error;
  if (llvm::Error error = requireOptionalPackedI4LoadUnpackFact(
          "packed_unpacked_source",
          kRVVLowPrecisionResourcePackedI4UnpackedSource))
    return error;

  if (static_cast<std::int64_t>(handoff.getPeakLiveVectorGroups()) !=
      getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
          handoff.getResourceSelectedCandidate()))
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff peak_live_vector_groups to "
        "match the selected resource candidate");
  if (static_cast<std::int64_t>(handoff.getVectorRegisterBudget()) !=
          kRVVLowPrecisionResourceVectorRegisterBudget ||
      handoff.getPeakLiveVectorGroups() > handoff.getVectorRegisterBudget())
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff vector_register_budget to "
        "contain the selected peak live vector-group estimate");
  if (static_cast<std::int64_t>(handoff.getProductRegionIndex()) !=
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              handoff.getResourceDecision()) ||
      static_cast<std::int64_t>(handoff.getDequantRegionIndex()) !=
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              handoff.getResourceDecision()) ||
      handoff.getProductRegionIndex() <= 0 ||
      handoff.getProductRegionIndex() >= handoff.getDequantRegionIndex() ||
      handoff.getDequantRegionIndex() > handoff.getRegionCount())
    return makeRVVEmitCRouteProviderError(
        "low-precision product-reduction dequantization RVV route requires "
        "tcrv_rvv.gearbox_cross_region_handoff product/dequant region indexes "
        "to match the selected resource decision and realized region count");

  auto requireHandoffPrimitiveFact =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("low-precision product-reduction dequantization RVV "
                    "route requires tcrv_rvv.gearbox_cross_region_handoff "
                    "primitive-chain fact '") +
        field + "' to match provider-owned resource fact '" + expected +
        "' but found '" + actual + "'");
  };
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_chain_contract", handoff.getPrimitiveChainContract(),
          kRVVLowPrecisionResourcePrimitiveChainContract))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_chain_kind", handoff.getPrimitiveChainKind(),
          kRVVLowPrecisionResourcePrimitiveChainKind))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_source_signedness", handoff.getPrimitiveSourceSignedness(),
          kRVVLowPrecisionResourceSourceSignednessSigned))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_source_load", handoff.getPrimitiveSourceLoad(),
          kRVVLowPrecisionResourcePrimitiveSourceLoad))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_source_extension", handoff.getPrimitiveSourceExtension(),
          kRVVLowPrecisionResourcePrimitiveSourceExtension))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "widening_product_multiplicand_roles",
          handoff.getWideningProductMultiplicandRoles(),
          kRVVLowPrecisionResourceWideningProductMultiplicandRoles))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "widening_product_extension_policy",
          handoff.getWideningProductExtensionPolicy(),
          kRVVLowPrecisionResourceWideningProductExtensionPolicy))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "widening_product_candidate_fact",
          handoff.getWideningProductCandidateFact(),
          kRVVLowPrecisionResourceWideningProductCandidateFact))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "reduction_candidate_fact", handoff.getReductionCandidateFact(),
          kRVVLowPrecisionResourceReductionCandidateFact))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_widening_product_relation",
          handoff.getPrimitiveWideningProductRelation(),
          kRVVLowPrecisionResourcePrimitiveWideningProductRelation))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_product_reduction_chain_relation",
          handoff.getPrimitiveProductReductionChainRelation(),
          kRVVLowPrecisionResourcePrimitiveProductReductionChainRelation))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_widening_product_intrinsic",
          handoff.getPrimitiveWideningProductIntrinsic(),
          kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_reduction_intrinsic",
          handoff.getPrimitiveReductionIntrinsic(),
          kRVVLowPrecisionResourcePrimitiveReductionIntrinsic))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_scalar_seed_splat_intrinsic",
          handoff.getPrimitiveScalarSeedSplatIntrinsic(),
          kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_accumulator_layout",
          handoff.getPrimitiveAccumulatorLayout(),
          kRVVLowPrecisionResourcePrimitiveAccumulatorLayout))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_result_layout", handoff.getPrimitiveResultLayout(),
          kRVVLowPrecisionResourcePrimitiveResultLayout))
    return error;
  if (llvm::Error error = requireHandoffPrimitiveFact(
          "primitive_reduction_store_vl",
          handoff.getPrimitiveReductionStoreVl(),
          kRVVLowPrecisionResourcePrimitiveReductionStoreVL))
    return error;

  auto requireOptionalRemediationFact =
      [&](llvm::StringRef attrName, llvm::StringRef expected) -> llvm::Error {
    auto attr = handoff->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine("low-precision product-reduction dequantization RVV "
                        "route requires packed-i4 remediation fact '") +
            attrName + "' to be absent for unpacked-byte resource candidates");
      return llvm::Error::success();
    }
    if (!attr)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("low-precision product-reduction dequantization RVV "
                      "route requires tcrv_rvv.gearbox_cross_region_handoff "
                      "packed-i4 remediation fact '") +
          attrName + "' before route support");
    if (attr.getValue() == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("low-precision product-reduction dequantization RVV "
                    "route requires tcrv_rvv.gearbox_cross_region_handoff "
                    "packed-i4 remediation fact '") +
        attrName + "' to match provider-owned resource fact '" + expected +
        "' but found '" + attr.getValue() + "'");
  };
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_plan_contract",
          kRVVLowPrecisionResourcePackedI4RemediationPlanContract))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_plan", kRVVLowPrecisionResourcePackedI4RemediationPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_statement_strategy",
          kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_vector_budget",
          kRVVLowPrecisionResourcePackedI4RemediationVectorBudget))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_schedule_contract",
          kRVVLowPrecisionResourcePackedI4RemediationScheduleContract))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_unpack_plan",
          kRVVLowPrecisionResourcePackedI4RemediationUnpackPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_product_plan",
          kRVVLowPrecisionResourcePackedI4RemediationProductPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_reduction_plan",
          kRVVLowPrecisionResourcePackedI4RemediationReductionPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "remediation_vl_plan",
          kRVVLowPrecisionResourcePackedI4RemediationVLPlan))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "schedule_decision_contract",
          kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "schedule_decision", kRVVLowPrecisionResourcePackedI4ScheduleDecision))
    return error;
  if (llvm::Error error = requireOptionalRemediationFact(
          "schedule_decision_reason",
          kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason))
    return error;

  slice.gearboxCrossRegionHandoffOp = handoff;
  slice.conversionSource = handoff.getOutput();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyDequantize(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::DequantizeOp dequantize) {
  if (dequantize.getKind() != kRVVDequantizeI32ToF32Kind)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.dequantize kind '") +
        dequantize.getKind() +
        "' for bounded RVV i32-to-f32 dequantization route");
  if (dequantize.getDequantRelation() != kRVVDequantizeI32ToF32Relation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.dequantize "
                    "dequant_relation '") +
        dequantize.getDequantRelation() +
        "' for bounded RVV i32-to-f32 dequantization route");

  llvm::Expected<support::RuntimeABIParameter> scaleABI =
      getRuntimeABIParameterBindingFromValue(
          dequantize.getScale(), "tcrv_rvv.dequantize runtime scale operand",
          {support::RuntimeABIParameterRole::DequantScaleValue});
  if (!scaleABI)
    return scaleABI.takeError();

  if (slice.arithmeticOp) {
    if (slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
        !slice.standaloneReduceOp || !hasProductHead(slice))
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires exactly one selected compute op "
          "unless a low-precision product-reduction result feeds a "
          "tcrv_rvv.dequantize i32-to-f32 chain");
    // Two carriers of the i32 product-reduction result feed the dequant:
    //   - legacy two-scope: the gearbox_cross_region_handoff output (carrier);
    //   - single-scope typed body (Stage 3 flip): the standalone_reduce result
    //     directly (no handoff op is present).
    if (slice.gearboxCrossRegionHandoffOp) {
      if (dequantize.getSource() !=
          slice.gearboxCrossRegionHandoffOp.getOutput())
        return makeRVVEmitCRouteProviderError(
            "low-precision product-reduction dequantization RVV route requires "
            "tcrv_rvv.dequantize source to consume the selected "
            "tcrv_rvv.gearbox_cross_region_handoff output");
    } else if (dequantize.getSource() !=
               slice.standaloneReduceOp.getResult()) {
      return makeRVVEmitCRouteProviderError(
          "low-precision product-reduction dequantization RVV route requires "
          "tcrv_rvv.dequantize source to consume the selected "
          "tcrv_rvv.standalone_reduce i32 result");
    }
    slice.dequantizeOp = dequantize;
    slice.arithmeticOp = dequantize.getOperation();
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32;
    slice.conversionSource = dequantize.getSource();
    slice.dequantScale = dequantize.getScale();
    slice.dequantScaleABI = *scaleABI;
    slice.arithmeticResult = dequantize.getResult();
    slice.memoryForm = RVVSelectedBodyMemoryForm::VectorRHSLoad;
    return llvm::Error::success();
  }

  slice.dequantizeOp = dequantize;
  slice.arithmeticOp = dequantize.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::DequantizeI32ToF32;
  slice.conversionSource = dequantize.getSource();
  slice.arithmeticLhs = dequantize.getSource();
  slice.dequantScale = dequantize.getScale();
  slice.dequantScaleABI = *scaleABI;
  slice.arithmeticResult = dequantize.getResult();
  slice.memoryForm = RVVSelectedBodyMemoryForm::UnitStrideDequantization;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMove(RVVSelectedBodyRouteSlice &slice,
                                      tcrv::rvv::MoveOp move);
llvm::Error recordRVVSelectedBodyMaskedMove(RVVSelectedBodyRouteSlice &slice,
                                            tcrv::rvv::MaskedMoveOp move);
llvm::Error recordRVVSelectedBodyMaskedLoad(RVVSelectedBodyRouteSlice &slice,
                                            tcrv::rvv::MaskedLoadOp load);
llvm::Error
recordRVVSelectedBodyMaskedStridedLoad(RVVSelectedBodyRouteSlice &slice,
                                       tcrv::rvv::MaskedStridedLoadOp load);
llvm::Error
recordRVVSelectedBodyMaskedIndexedLoad(RVVSelectedBodyRouteSlice &slice,
                                       tcrv::rvv::MaskedIndexedLoadOp load);
llvm::Error
recordRVVSelectedBodyMaskedIndexedStore(RVVSelectedBodyRouteSlice &slice,
                                        tcrv::rvv::MaskedIndexedStoreOp store);
llvm::Error
recordRVVSelectedBodyMaskedSegment2Load(RVVSelectedBodyRouteSlice &slice,
                                        tcrv::rvv::MaskedSegment2LoadOp load);
llvm::Error
recordRVVSelectedBodyMaskedSegment2Store(RVVSelectedBodyRouteSlice &slice,
                                         tcrv::rvv::MaskedSegment2StoreOp store);
llvm::Error recordRVVSelectedBodyMaskedStore(RVVSelectedBodyRouteSlice &slice,
                                             tcrv::rvv::MaskedStoreOp store);
llvm::Error
recordRVVSelectedBodyMaskedStridedStore(RVVSelectedBodyRouteSlice &slice,
                                        tcrv::rvv::MaskedStridedStoreOp store);
llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyBinaryKind(llvm::StringRef kind);

bool isRVVGearboxProductReduceDequantConsumerScope(
    tcrv::rvv::WithVLOp producerWithVL, tcrv::rvv::WithVLOp candidate) {
  if (!producerWithVL || !candidate || producerWithVL == candidate ||
      candidate.getVl() != producerWithVL.getVl() ||
      !producerWithVL->isProperAncestor(candidate.getOperation()))
    return false;

  auto isHandoffConsumingDequantize = [&](tcrv::rvv::DequantizeOp dequantize) {
    auto handoff = dequantize.getSource()
                       .getDefiningOp<tcrv::rvv::GearboxCrossRegionHandoffOp>();
    return handoff && handoff->getParentOp() == producerWithVL.getOperation() &&
           dequantize->getParentOp() == candidate.getOperation() &&
           dequantize.getVl() == producerWithVL.getVl();
  };

  auto valueUsesHandoffDequantize = [&](mlir::Value value) {
    llvm::SmallVector<mlir::Value, 4> worklist{value};
    llvm::SmallPtrSet<mlir::Value, 4> seen;
    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();
      if (!seen.insert(current).second)
        continue;
      if (auto dequantize = current.getDefiningOp<tcrv::rvv::DequantizeOp>()) {
        if (isHandoffConsumingDequantize(dequantize))
          return true;
        continue;
      }
      auto select = current.getDefiningOp<tcrv::rvv::SelectOp>();
      if (!select || select->getParentOp() != candidate.getOperation() ||
          select.getVl() != producerWithVL.getVl())
        continue;
      worklist.push_back(select.getTrueValue());
      worklist.push_back(select.getFalseValue());
    }
    return false;
  };

  bool hasRegionMarker = false;
  bool hasHandoffDequantize = false;
  bool hasStore = false;
  for (mlir::Operation &op : candidate.getBody().front()) {
    if (auto marker = llvm::dyn_cast<tcrv::rvv::VSetVLRegionMarkerOp>(op)) {
      hasRegionMarker =
          marker.getVl() == producerWithVL.getVl() &&
          marker.getPhase() == "dequant-store" &&
          marker->getAttrOfType<mlir::StringAttr>(
              kGearboxHandoffPlanningContractAttrName) &&
          marker->getAttrOfType<mlir::StringAttr>(
              kGearboxHandoffPlanningContractAttrName)
                  .getValue() == kRVVLowPrecisionResourcePlanningContract &&
          static_cast<std::int64_t>(marker.getRegionIndex()) ==
              getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
                  marker.getResourceDecision()) &&
          static_cast<std::int64_t>(marker.getRegionCount()) ==
              getRVVLowPrecisionResourceExpectedVSetVLRegionCountForRealizationDecision(
                  marker.getResourceDecision()) &&
          isRVVLowPrecisionResourceSupportedRealizationDecision(
              marker.getResourceDecision());
      continue;
    }
    if (auto dequantize = llvm::dyn_cast<tcrv::rvv::DequantizeOp>(op)) {
      if (isHandoffConsumingDequantize(dequantize))
        hasHandoffDequantize = true;
      continue;
    }
    if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(op)) {
      if (store.getVl() == producerWithVL.getVl() &&
          valueUsesHandoffDequantize(store.getValue()))
        hasStore = true;
      continue;
    }
  }
  return hasRegionMarker && hasHandoffDequantize && hasStore;
}

tcrv::rvv::GearboxCrossRegionHandoffOp
findDirectRVVGearboxCrossRegionHandoff(tcrv::rvv::WithVLOp withVL) {
  tcrv::rvv::GearboxCrossRegionHandoffOp handoff;
  if (!withVL)
    return handoff;
  for (mlir::Operation &op : withVL.getBody().front()) {
    auto candidate =
        llvm::dyn_cast<tcrv::rvv::GearboxCrossRegionHandoffOp>(op);
    if (!candidate)
      continue;
    if (handoff)
      return {};
    handoff = candidate;
  }
  return handoff;
}

bool isRVVGearboxConsumerScopeOrderedAfterHandoff(
    tcrv::rvv::WithVLOp producerWithVL, tcrv::rvv::WithVLOp consumerWithVL,
    tcrv::rvv::GearboxCrossRegionHandoffOp handoff) {
  if (!producerWithVL || !consumerWithVL || !handoff)
    return false;
  bool sawHandoff = false;
  for (mlir::Operation &op : producerWithVL.getBody().front()) {
    if (&op == handoff.getOperation()) {
      sawHandoff = true;
      continue;
    }
    if (&op == consumerWithVL.getOperation())
      return sawHandoff;
  }
  return false;
}

llvm::Error recordRVVSelectedBodyScopedRouteOp(
    RVVSelectedBodyRouteSlice &slice, mlir::Operation &op,
    llvm::SmallVectorImpl<tcrv::rvv::LoadOp> &genericLoads,
    llvm::SmallVectorImpl<tcrv::rvv::StridedLoadOp> &genericStridedLoads,
    llvm::SmallVectorImpl<tcrv::rvv::IndexLoadOp> &genericIndexLoads,
    llvm::SmallVectorImpl<tcrv::rvv::IndexedLoadOp> &genericIndexedLoads,
    llvm::SmallVectorImpl<tcrv::rvv::IndexedStoreOp> &genericIndexedStores,
    llvm::SmallVectorImpl<tcrv::rvv::MaskLoadOp> &genericMaskLoads,
    llvm::SmallVectorImpl<tcrv::rvv::MaskedLoadOp> &genericMaskedLoads,
    llvm::SmallVectorImpl<tcrv::rvv::MaskedStridedLoadOp>
        &genericMaskedStridedLoads,
    llvm::SmallVectorImpl<tcrv::rvv::MaskedIndexedLoadOp>
        &genericMaskedIndexedLoads,
    llvm::SmallVectorImpl<tcrv::rvv::MaskedIndexedStoreOp>
        &genericMaskedIndexedStores,
    llvm::SmallVectorImpl<tcrv::rvv::MaskedSegment2LoadOp>
        &genericMaskedSegment2Loads,
    llvm::SmallVectorImpl<tcrv::rvv::MaskedSegment2StoreOp>
        &genericMaskedSegment2Stores,
    llvm::SmallVectorImpl<tcrv::rvv::Segment2LoadOp> &genericSegment2Loads,
    llvm::SmallVectorImpl<tcrv::rvv::Segment2StoreOp> &genericSegment2Stores,
    llvm::SmallVectorImpl<tcrv::rvv::VSetVLRegionMarkerOp>
        &vsetvlRegionMarkers,
    llvm::SmallVectorImpl<tcrv::rvv::BroadcastLoadOp> &genericBroadcastLoads,
    llvm::SmallVectorImpl<tcrv::rvv::SplatOp> &genericScalarSplats,
    llvm::SmallVectorImpl<tcrv::rvv::StoreOp> &genericStores,
    unsigned &storeCount, unsigned &stridedStoreCount,
    const support::RuntimeABIParameter &runtimeElementCountABI) {
  if (auto marker = llvm::dyn_cast<tcrv::rvv::VSetVLRegionMarkerOp>(op)) {
    vsetvlRegionMarkers.push_back(marker);
    return llvm::Error::success();
  }
  if (auto handoff =
          llvm::dyn_cast<tcrv::rvv::GearboxCrossRegionHandoffOp>(op))
    return recordRVVSelectedBodyGearboxCrossRegionHandoff(
        slice, handoff, runtimeElementCountABI);
  if (auto load = llvm::dyn_cast<tcrv::rvv::LoadOp>(op)) {
    genericLoads.push_back(load);
    return llvm::Error::success();
  }
  if (auto stridedLoad = llvm::dyn_cast<tcrv::rvv::StridedLoadOp>(op)) {
    genericStridedLoads.push_back(stridedLoad);
    return llvm::Error::success();
  }
  if (auto indexLoad = llvm::dyn_cast<tcrv::rvv::IndexLoadOp>(op)) {
    genericIndexLoads.push_back(indexLoad);
    return llvm::Error::success();
  }
  if (auto indexedLoad = llvm::dyn_cast<tcrv::rvv::IndexedLoadOp>(op)) {
    genericIndexedLoads.push_back(indexedLoad);
    return llvm::Error::success();
  }
  if (auto indexedStore = llvm::dyn_cast<tcrv::rvv::IndexedStoreOp>(op)) {
    genericIndexedStores.push_back(indexedStore);
    return llvm::Error::success();
  }
  if (auto maskLoad = llvm::dyn_cast<tcrv::rvv::MaskLoadOp>(op)) {
    genericMaskLoads.push_back(maskLoad);
    return llvm::Error::success();
  }
  if (auto maskedLoad = llvm::dyn_cast<tcrv::rvv::MaskedLoadOp>(op)) {
    genericMaskedLoads.push_back(maskedLoad);
    return recordRVVSelectedBodyMaskedLoad(slice, maskedLoad);
  }
  if (auto maskedStridedLoad =
          llvm::dyn_cast<tcrv::rvv::MaskedStridedLoadOp>(op)) {
    genericMaskedStridedLoads.push_back(maskedStridedLoad);
    return recordRVVSelectedBodyMaskedStridedLoad(slice, maskedStridedLoad);
  }
  if (auto maskedIndexedLoad =
          llvm::dyn_cast<tcrv::rvv::MaskedIndexedLoadOp>(op)) {
    genericMaskedIndexedLoads.push_back(maskedIndexedLoad);
    return recordRVVSelectedBodyMaskedIndexedLoad(slice, maskedIndexedLoad);
  }
  if (auto maskedIndexedStore =
          llvm::dyn_cast<tcrv::rvv::MaskedIndexedStoreOp>(op)) {
    genericMaskedIndexedStores.push_back(maskedIndexedStore);
    return recordRVVSelectedBodyMaskedIndexedStore(slice, maskedIndexedStore);
  }
  if (auto maskedSegment2Load =
          llvm::dyn_cast<tcrv::rvv::MaskedSegment2LoadOp>(op)) {
    genericMaskedSegment2Loads.push_back(maskedSegment2Load);
    return recordRVVSelectedBodyMaskedSegment2Load(slice, maskedSegment2Load);
  }
  if (auto maskedSegment2Store =
          llvm::dyn_cast<tcrv::rvv::MaskedSegment2StoreOp>(op)) {
    genericMaskedSegment2Stores.push_back(maskedSegment2Store);
    return recordRVVSelectedBodyMaskedSegment2Store(slice, maskedSegment2Store);
  }
  if (auto segment2Load = llvm::dyn_cast<tcrv::rvv::Segment2LoadOp>(op)) {
    genericSegment2Loads.push_back(segment2Load);
    return llvm::Error::success();
  }
  if (auto segment2Store = llvm::dyn_cast<tcrv::rvv::Segment2StoreOp>(op)) {
    genericSegment2Stores.push_back(segment2Store);
    return llvm::Error::success();
  }
  if (auto broadcast = llvm::dyn_cast<tcrv::rvv::BroadcastLoadOp>(op)) {
    genericBroadcastLoads.push_back(broadcast);
    return llvm::Error::success();
  }
  if (auto splat = llvm::dyn_cast<tcrv::rvv::SplatOp>(op)) {
    genericScalarSplats.push_back(splat);
    return llvm::Error::success();
  }
  if (auto binary = llvm::dyn_cast<tcrv::rvv::BinaryOp>(op)) {
    llvm::Expected<RVVSelectedBodyOperationKind> kind =
        parseRVVSelectedBodyBinaryKind(binary.getKind());
    if (!kind)
      return kind.takeError();
    return recordRVVSelectedBodyOperation(slice, binary.getOperation(), *kind,
                                          binary.getLhs(), binary.getRhs(),
                                          binary.getResult());
  }
  if (auto compare = llvm::dyn_cast<tcrv::rvv::CompareOp>(op))
    return recordRVVSelectedBodyCompare(slice, compare);
  if (auto maskAnd = llvm::dyn_cast<tcrv::rvv::MaskAndOp>(op))
    return recordRVVSelectedBodyMaskAnd(slice, maskAnd);
  if (auto maskedBinary = llvm::dyn_cast<tcrv::rvv::MaskedBinaryOp>(op))
    return recordRVVSelectedBodyMaskedBinary(slice, maskedBinary);
  if (auto select = llvm::dyn_cast<tcrv::rvv::SelectOp>(op))
    return recordRVVSelectedBodySelect(slice, select);
  if (auto reduce = llvm::dyn_cast<tcrv::rvv::ReduceOp>(op))
    return recordRVVSelectedBodyReduction(slice, reduce);
  if (auto standaloneReduce =
          llvm::dyn_cast<tcrv::rvv::StandaloneReduceOp>(op))
    return recordRVVSelectedBodyStandaloneReduction(slice, standaloneReduce);
  if (auto maskedStandaloneReduce =
          llvm::dyn_cast<tcrv::rvv::MaskedStandaloneReduceOp>(op))
    return recordRVVSelectedBodyMaskedStandaloneReduction(slice,
                                                          maskedStandaloneReduce);
  if (auto macc = llvm::dyn_cast<tcrv::rvv::MAccOp>(op))
    return recordRVVSelectedBodyMAcc(slice, macc);
  if (auto maskedMAcc = llvm::dyn_cast<tcrv::rvv::MaskedMAccOp>(op))
    return recordRVVSelectedBodyMaskedMAcc(slice, maskedMAcc);
  if (auto wideningMAcc = llvm::dyn_cast<tcrv::rvv::WideningMAccOp>(op))
    return recordRVVSelectedBodyWideningMAcc(slice, wideningMAcc);
  if (auto product = llvm::dyn_cast<tcrv::rvv::WideningProductOp>(op))
    return recordRVVSelectedBodyWideningProduct(slice, product);
  if (auto nibbleProduct =
          llvm::dyn_cast<tcrv::rvv::PackedI4NibbleUnpackProductOp>(op))
    return recordRVVSelectedBodyNibbleUnpackProduct(slice, nibbleProduct);
  if (auto dotReduce = llvm::dyn_cast<tcrv::rvv::WideningDotReduceOp>(op))
    return recordRVVSelectedBodyWideningDotReduce(slice, dotReduce);
  if (auto maskedDotReduce =
          llvm::dyn_cast<tcrv::rvv::MaskedWideningDotReduceOp>(op))
    return recordRVVSelectedBodyMaskedWideningDotReduce(slice, maskedDotReduce);
  if (auto conversion = llvm::dyn_cast<tcrv::rvv::WideningConvertOp>(op))
    return recordRVVSelectedBodyWideningConvert(slice, conversion);
  if (auto dequantize = llvm::dyn_cast<tcrv::rvv::DequantizeOp>(op))
    return recordRVVSelectedBodyDequantize(slice, dequantize);
  if (auto move = llvm::dyn_cast<tcrv::rvv::MoveOp>(op))
    return recordRVVSelectedBodyMove(slice, move);
  if (auto maskedMove = llvm::dyn_cast<tcrv::rvv::MaskedMoveOp>(op))
    return recordRVVSelectedBodyMaskedMove(slice, maskedMove);
  if (auto maskedStore = llvm::dyn_cast<tcrv::rvv::MaskedStoreOp>(op))
    return recordRVVSelectedBodyMaskedStore(slice, maskedStore);
  if (auto maskedStridedStore =
          llvm::dyn_cast<tcrv::rvv::MaskedStridedStoreOp>(op))
    return recordRVVSelectedBodyMaskedStridedStore(slice, maskedStridedStore);
  if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(op)) {
    slice.genericStore = store;
    genericStores.push_back(store);
    ++storeCount;
    return llvm::Error::success();
  }
  if (auto stridedStore = llvm::dyn_cast<tcrv::rvv::StridedStoreOp>(op)) {
    slice.stridedStore = stridedStore;
    ++stridedStoreCount;
    return llvm::Error::success();
  }
  if (op.getName().getStringRef().starts_with("tcrv_rvv.i32_"))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("legacy selected-body op '") + op.getName().getStringRef() +
        "' is fail-closed during RVV Stage1; Stage2 routes must use generic "
        "tcrv_rvv.load, tcrv_rvv.broadcast_load, "
        "tcrv_rvv.splat, tcrv_rvv.strided_load, tcrv_rvv.binary, "
        "tcrv_rvv.index_load, tcrv_rvv.indexed_load, tcrv_rvv.segment2_load, "
        "tcrv_rvv.segment2_store, "
        "tcrv_rvv.indexed_store, tcrv_rvv.mask_load, tcrv_rvv.compare, "
        "tcrv_rvv.masked_binary, tcrv_rvv.select, tcrv_rvv.reduce, "
        "tcrv_rvv.standalone_reduce, tcrv_rvv.masked_standalone_reduce, "
        "tcrv_rvv.macc, tcrv_rvv.masked_macc, "
        "tcrv_rvv.widening_product, tcrv_rvv.widening_convert, "
        "tcrv_rvv.gearbox_cross_region_handoff, "
        "tcrv_rvv.move, "
        "tcrv_rvv.widening_dot_reduce, "
        "tcrv_rvv.masked_widening_dot_reduce, "
        "tcrv_rvv.masked_move, tcrv_rvv.masked_load, "
        "tcrv_rvv.masked_strided_load, tcrv_rvv.masked_indexed_load, "
        "tcrv_rvv.masked_indexed_store, "
        "tcrv_rvv.masked_segment2_load, tcrv_rvv.masked_store, "
        "tcrv_rvv.masked_strided_store, "
        "tcrv_rvv.store, and "
        "tcrv_rvv.strided_store body structure");
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("bounded RVV EmitC route does not support op '") +
      op.getName().getStringRef() +
      "' inside tcrv_rvv.with_vl; expected generic load, broadcast_load, "
      "splat, strided_load, index_load, indexed_load, indexed_store, "
      "mask_load, segment2_load, segment2_store, binary, compare, "
      "masked_binary, select, reduce, standalone_reduce, "
      "masked_standalone_reduce, macc, masked_macc, "
      "widening_product, packed_i4_nibble_unpack_product, widening_convert, "
      "gearbox_cross_region_handoff, "
      "widening_dot_reduce, "
      "masked_widening_dot_reduce, move, masked_move, masked_load, "
      "masked_strided_load, masked_indexed_load, masked_indexed_store, "
      "masked_segment2_load, "
      "masked_store, masked_strided_store, store, and strided_store only");
}

llvm::Error recordRVVSelectedBodyMove(RVVSelectedBodyRouteSlice &slice,
                                      tcrv::rvv::MoveOp move) {
  if (auto segment2Load =
          move.getSource().getDefiningOp<tcrv::rvv::Segment2LoadOp>()) {
    if (move.getKind() != "copy")
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("unsupported generic tcrv_rvv.move kind '") +
          move.getKind() +
          "' for bounded RVV segment2 deinterleave route");
    if (move.getSource() == segment2Load.getField0()) {
      if (slice.field0MoveOperation)
        return makeRVVEmitCRouteProviderError(
            "bounded RVV segment2 deinterleave route requires a unique "
            "field0 tcrv_rvv.move");
      slice.field0MoveOp = move;
      slice.field0MoveOperation = move.getOperation();
      slice.field0LoadedValue = move.getSource();
      slice.field0Value = move.getResult();
    } else if (move.getSource() == segment2Load.getField1()) {
      if (slice.field1MoveOperation)
        return makeRVVEmitCRouteProviderError(
            "bounded RVV segment2 deinterleave route requires a unique "
            "field1 tcrv_rvv.move");
      slice.field1MoveOp = move;
      slice.field1MoveOperation = move.getOperation();
      slice.field1LoadedValue = move.getSource();
      slice.field1Value = move.getResult();
    } else {
      return makeRVVEmitCRouteProviderError(
          "bounded RVV segment2 deinterleave route requires tcrv_rvv.move to "
          "consume field0 or field1 from tcrv_rvv.segment2_load");
    }
    if (!slice.arithmeticOp)
      slice.arithmeticOp = move.getOperation();
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
    slice.arithmeticLhs = slice.field0LoadedValue;
    slice.arithmeticResult = slice.field0Value;
    return llvm::Error::success();
  }

  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement op");
  if (move.getKind() != "copy")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.move kind '") +
        move.getKind() + "' for bounded RVV strided memory movement route");
  slice.moveOp = move;
  slice.arithmeticOp = move.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  slice.arithmeticLhs = move.getSource();
  slice.arithmeticResult = move.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedMove(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedMoveOp move) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement op");
  if (move.getKind() != "active-source-preserve-old-destination")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_move kind '") +
        move.getKind() + "' for bounded RVV masked memory route");
  slice.maskedMoveOp = move;
  slice.arithmeticOp = move.getOperation();
  slice.arithmeticKind = move.getMask().getDefiningOp<tcrv::rvv::CompareOp>()
                             ? RVVSelectedBodyOperationKind::
                                   ComputedMaskUnitLoadStore
                             : RVVSelectedBodyOperationKind::
                                   MaskedUnitLoadStore;
  slice.maskValue = move.getMask();
  slice.maskedActiveValue = move.getActiveValue();
  slice.maskedInactivePassthrough = move.getInactivePassthrough();
  slice.arithmeticLhs = move.getActiveValue();
  slice.arithmeticRhs = move.getInactivePassthrough();
  slice.arithmeticResult = move.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedLoad(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedLoadOp load) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement/load op");
  if (load.getMemoryForm() != kRVVMaskedLoadMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_load memory_form '") +
        load.getMemoryForm() + "' for bounded RVV masked memory route");
  if (load.getInactiveLanePolicy() != kRVVMaskedLoadInactiveLanePolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_load "
                    "inactive_lane_policy '") +
        load.getInactiveLanePolicy() +
        "' for bounded RVV masked memory route");
  slice.maskedLoadOp = load;
  slice.maskedLoadOperation = load.getOperation();
  slice.arithmeticOp = load.getOperation();
  slice.arithmeticKind = load.getMask().getDefiningOp<tcrv::rvv::CompareOp>()
                             ? RVVSelectedBodyOperationKind::
                                   ComputedMaskUnitLoadStore
                             : RVVSelectedBodyOperationKind::
                                   MaskedUnitLoadStore;
  slice.maskValue = load.getMask();
  slice.maskedActiveValue = load.getLoaded();
  slice.maskedInactivePassthrough = load.getPassthrough();
  slice.arithmeticLhs = load.getBuffer();
  slice.arithmeticRhs = load.getPassthrough();
  slice.arithmeticResult = load.getLoaded();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedStridedLoad(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedStridedLoadOp load) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement/load op");
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
  if (!load.getMask().getDefiningOp<tcrv::rvv::CompareOp>())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask strided load route requires "
        "tcrv_rvv.masked_strided_load to consume a tcrv_rvv.compare mask");
  slice.maskedStridedLoadOp = load;
  slice.maskedStridedLoadOperation = load.getOperation();
  slice.arithmeticOp = load.getOperation();
  slice.arithmeticKind =
      RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
  slice.maskValue = load.getMask();
  slice.maskedActiveValue = load.getLoaded();
  slice.maskedInactivePassthrough = load.getPassthrough();
  slice.arithmeticLhs = load.getBuffer();
  slice.arithmeticRhs = load.getPassthrough();
  slice.arithmeticResult = load.getLoaded();
  slice.memoryForm = RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedIndexedLoad(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedIndexedLoadOp load) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement/load op");
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
        "bounded RVV computed-mask indexed gather-load route supports only "
        "index_eew 32");
  if (load.getInactiveLanePolicy() != kRVVMaskedLoadInactiveLanePolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_indexed_load "
                    "inactive_lane_policy '") +
        load.getInactiveLanePolicy() +
        "' for bounded RVV computed-mask indexed gather-load route");
  if (!load.getMask().getDefiningOp<tcrv::rvv::CompareOp>())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask indexed gather-load route requires "
        "tcrv_rvv.masked_indexed_load to consume a tcrv_rvv.compare mask");
  slice.maskedIndexedLoadOp = load;
  slice.maskedIndexedLoadOperation = load.getOperation();
  slice.arithmeticOp = load.getOperation();
  slice.arithmeticKind =
      RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  slice.maskValue = load.getMask();
  slice.maskedActiveValue = load.getLoaded();
  slice.maskedInactivePassthrough = load.getPassthrough();
  slice.arithmeticLhs = load.getData();
  slice.arithmeticRhs = load.getPassthrough();
  slice.arithmeticResult = load.getLoaded();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedIndexedStore(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedIndexedStoreOp store) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement/store op");
  if (store.getMemoryForm() != kRVVMaskedIndexedStoreDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_indexed_store "
                    "memory_form '") +
        store.getMemoryForm() +
        "' for bounded RVV computed-mask indexed scatter-store route");
  if (store.getOffsetUnit() != kRVVIndexedGatherOffsetUnit)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_indexed_store "
                    "offset_unit '") +
        store.getOffsetUnit() +
        "' for bounded RVV computed-mask indexed scatter-store route");
  if (static_cast<std::int64_t>(store.getIndexEew()) != 32)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask indexed scatter-store route supports only "
        "index_eew 32");
  if (store.getIndexUniqueness() != kRVVIndexedScatterIndexUniqueness)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_indexed_store "
                    "index_uniqueness '") +
        store.getIndexUniqueness() +
        "' for bounded RVV computed-mask indexed scatter-store route");
  if (store.getInactiveLanePolicy() !=
      "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_indexed_store "
                    "inactive_lane_policy '") +
        store.getInactiveLanePolicy() +
        "' for bounded RVV computed-mask indexed scatter-store route");
  if (!store.getMask().getDefiningOp<tcrv::rvv::CompareOp>())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask indexed scatter-store route requires "
        "tcrv_rvv.masked_indexed_store to consume a tcrv_rvv.compare mask");
  slice.maskedIndexedStore = store;
  slice.maskedIndexedStoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.arithmeticOp = store.getOperation();
  slice.arithmeticKind =
      RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
  slice.maskValue = store.getMask();
  slice.maskedActiveValue = store.getValue();
  slice.arithmeticLhs = store.getValue();
  slice.arithmeticResult = store.getValue();
  slice.indexedDestinationBuffer = store.getDestination();
  slice.outBuffer = store.getDestination();
  slice.storeValue = store.getValue();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedSegment2Load(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedSegment2LoadOp load) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement/load op");
  if (load.getSourceMemoryForm() != kRVVSegment2SourceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_segment2_load "
                    "source_memory_form '") +
        load.getSourceMemoryForm() +
        "' for bounded RVV computed-mask segment2 load route");
  if (static_cast<std::int64_t>(load.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask segment2 load route supports only "
        "segment_count 2");
  if (load.getField0Role() != kRVVSegment2Field0Role ||
      load.getField1Role() != kRVVSegment2Field1Role)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask segment2 load route requires field roles "
        "to mirror segment output buffers");
  if (load.getInactiveLanePolicy() != kRVVMaskedLoadInactiveLanePolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_segment2_load "
                    "inactive_lane_policy '") +
        load.getInactiveLanePolicy() +
        "' for bounded RVV computed-mask segment2 load route");
  if (!load.getMask().getDefiningOp<tcrv::rvv::CompareOp>())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask segment2 load route requires "
        "tcrv_rvv.masked_segment2_load to consume a tcrv_rvv.compare mask");

  slice.maskedSegment2LoadOp = load;
  slice.maskedSegment2LoadOperation = load.getOperation();
  slice.arithmeticOp = load.getOperation();
  slice.arithmeticKind =
      RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
  slice.maskValue = load.getMask();
  slice.maskedInactivePassthrough = load.getPassthrough0();
  slice.arithmeticLhs = load.getSource();
  slice.arithmeticRhs = load.getPassthrough0();
  slice.arithmeticResult = load.getField0();
  slice.field0LoadedValue = load.getField0();
  slice.field1LoadedValue = load.getField1();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedSegment2Store(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedSegment2StoreOp store) {
  const bool composesBinaryUpdate =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::Add &&
      store.getField0() == slice.arithmeticResult;
  if (slice.arithmeticOp && !composesBinaryUpdate)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask segment2 store/update route requires "
        "masked_segment2_store either to be the selected movement/store op or "
        "to consume the add result as field0 payload for the bounded composed "
        "update route");
  if (store.getDestinationMemoryForm() !=
      kRVVSegment2InterleavedDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_segment2_store "
                    "destination_memory_form '") +
        store.getDestinationMemoryForm() +
        "' for bounded RVV computed-mask segment2 store route");
  if (static_cast<std::int64_t>(store.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask segment2 store route supports only "
        "segment_count 2");
  if (store.getField0Role() != kRVVSegment2Field0InputRole ||
      store.getField1Role() != kRVVSegment2Field1InputRole)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask segment2 store route requires field roles "
        "to mirror segment input buffers");
  if (store.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_segment2_store "
                    "inactive_lane_policy '") +
        store.getInactiveLanePolicy() +
        "' for bounded RVV computed-mask segment2 store route");
  if (!store.getMask().getDefiningOp<tcrv::rvv::CompareOp>())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask segment2 store route requires "
        "tcrv_rvv.masked_segment2_store to consume a tcrv_rvv.compare mask");

  slice.maskedSegment2Store = store;
  slice.maskedSegment2StoreOperation = store.getOperation();
  slice.segment2StoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  if (!composesBinaryUpdate) {
    slice.arithmeticOp = store.getOperation();
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
    slice.arithmeticLhs = store.getField0();
    slice.arithmeticRhs = store.getField1();
    slice.arithmeticResult = store.getField0();
  } else {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  }
  slice.maskValue = store.getMask();
  slice.field0Value = store.getField0();
  slice.field1Value = store.getField1();
  slice.maskedActiveValue = store.getField0();
  slice.outBuffer = store.getDestination();
  slice.storeValue = store.getField0();
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedStore(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::MaskedStoreOp store) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement/store op");
  if (store.getMemoryForm() != kRVVMaskedStoreDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_store memory_form '") +
        store.getMemoryForm() + "' for bounded RVV masked store route");
  if (store.getInactiveLanePolicy() !=
      "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_store "
                    "inactive_lane_policy '") +
        store.getInactiveLanePolicy() +
        "' for bounded RVV masked store route");
  slice.maskedStore = store;
  slice.maskedStoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.arithmeticOp = store.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MaskedUnitStore;
  slice.maskValue = store.getMask();
  slice.maskedActiveValue = store.getValue();
  slice.arithmeticLhs = store.getValue();
  slice.arithmeticResult = store.getValue();
  slice.outBuffer = store.getBuffer();
  slice.storeValue = store.getValue();
  slice.memoryForm = RVVSelectedBodyMemoryForm::MaskedUnitStore;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyMaskedStridedStore(
    RVVSelectedBodyRouteSlice &slice,
    tcrv::rvv::MaskedStridedStoreOp store) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute or "
        "movement/store op");
  if (store.getMemoryForm() != kRVVMaskedStridedStoreDestinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_strided_store "
                    "memory_form '") +
        store.getMemoryForm() +
        "' for bounded RVV computed-mask strided store route");
  if (store.getStrideUnit() != "byte")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_strided_store "
                    "stride_unit '") +
        store.getStrideUnit() +
        "' for bounded RVV computed-mask strided store route");
  if (store.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_strided_store "
                    "inactive_lane_policy '") +
        store.getInactiveLanePolicy() +
        "' for bounded RVV computed-mask strided store route");
  if (!store.getMask().getDefiningOp<tcrv::rvv::CompareOp>())
    return makeRVVEmitCRouteProviderError(
        "bounded RVV computed-mask strided store route requires "
        "tcrv_rvv.masked_strided_store to consume a tcrv_rvv.compare mask");
  slice.maskedStridedStore = store;
  slice.maskedStridedStoreOperation = store.getOperation();
  slice.storeOperation = store.getOperation();
  slice.arithmeticOp = store.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  slice.maskValue = store.getMask();
  slice.maskedActiveValue = store.getValue();
  slice.arithmeticLhs = store.getValue();
  slice.arithmeticResult = store.getValue();
  slice.outBuffer = store.getBuffer();
  slice.outStride = store.getStride();
  slice.storeValue = store.getValue();
  slice.memoryForm = RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyBinaryKind(llvm::StringRef kind) {
  if (kind == "add")
    return RVVSelectedBodyOperationKind::Add;
  if (kind == "sub")
    return RVVSelectedBodyOperationKind::Sub;
  if (kind == "mul")
    return RVVSelectedBodyOperationKind::Mul;
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported generic tcrv_rvv.binary kind '") + kind +
      "' for bounded RVV EmitC route");
}

llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyMaskedBinaryKind(llvm::StringRef kind) {
  if (kind == "add")
    return RVVSelectedBodyOperationKind::MaskedAdd;
  if (kind == "sub")
    return RVVSelectedBodyOperationKind::MaskedSub;
  if (kind == "mul")
    return RVVSelectedBodyOperationKind::MaskedMul;
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported generic tcrv_rvv.masked_binary kind '") +
      kind + "' for bounded RVV masked route");
}

llvm::Expected<RVVSelectedBodyOperationKind>
getRVVScalarBroadcastOperationKind(RVVSelectedBodyOperationKind binaryKind) {
  switch (binaryKind) {
  case RVVSelectedBodyOperationKind::Add:
    return RVVSelectedBodyOperationKind::ScalarBroadcastAdd;
  case RVVSelectedBodyOperationKind::Sub:
    return RVVSelectedBodyOperationKind::ScalarBroadcastSub;
  case RVVSelectedBodyOperationKind::Mul:
    return RVVSelectedBodyOperationKind::ScalarBroadcastMul;
  default:
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded generic RVV scalar-broadcast route does not "
                    "support operation '") +
        stringifyRVVSelectedBodyOperationKind(binaryKind) + "'");
  }
}

bool isRuntimeScalarComputedMaskIndexedGatherMAccScatterCompositeCandidate(
    tcrv::rvv::WithVLOp withVL) {
  tcrv::rvv::SplatOp runtimeScalarSplat;
  tcrv::rvv::CompareOp compare;
  tcrv::rvv::IndexLoadOp indexLoad;
  tcrv::rvv::MaskedIndexedLoadOp indexedGather;
  tcrv::rvv::MaskedMAccOp maskedMAcc;
  tcrv::rvv::MaskedIndexedStoreOp indexedScatter;

  for (mlir::Operation &op : withVL.getBody().front()) {
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::SplatOp>(op))
      runtimeScalarSplat = candidate;
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::CompareOp>(op))
      compare = candidate;
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::IndexLoadOp>(op))
      indexLoad = candidate;
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::MaskedIndexedLoadOp>(op))
      indexedGather = candidate;
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::MaskedMAccOp>(op))
      maskedMAcc = candidate;
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::MaskedIndexedStoreOp>(op))
      indexedScatter = candidate;
  }

  if (!runtimeScalarSplat || !compare || !indexLoad || !indexedGather ||
      !maskedMAcc || !indexedScatter)
    return false;
  return true;
}

llvm::Error requireRVVCompositeSelectedVL(mlir::Value actual,
                                         mlir::Value expected,
                                         llvm::StringRef context) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("Stage2 RVV composite gather-MAcc-scatter route requires ") +
      context + " to consume the selected !tcrv_rvv.vl token");
}

llvm::Error
recordRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherMAccScatter(
    RVVSelectedBodyRouteSlice &slice,
    const support::RuntimeABIParameter &runtimeElementCountABI) {
  tcrv::rvv::SplatOp runtimeScalarSplat;
  tcrv::rvv::CompareOp compare;
  tcrv::rvv::IndexLoadOp indexLoad;
  tcrv::rvv::MaskedIndexedLoadOp indexedGather;
  tcrv::rvv::MaskedMAccOp maskedMAcc;
  tcrv::rvv::MaskedIndexedStoreOp indexedScatter;
  llvm::SmallVector<tcrv::rvv::LoadOp, 4> genericLoads;

  for (mlir::Operation &op : slice.withVL.getBody().front()) {
    if (auto load = llvm::dyn_cast<tcrv::rvv::LoadOp>(op)) {
      genericLoads.push_back(load);
      continue;
    }
    if (auto splat = llvm::dyn_cast<tcrv::rvv::SplatOp>(op)) {
      if (runtimeScalarSplat)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "tcrv_rvv.splat runtime scalar threshold");
      runtimeScalarSplat = splat;
      continue;
    }
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::CompareOp>(op)) {
      if (compare)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "tcrv_rvv.compare mask producer");
      compare = candidate;
      continue;
    }
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::IndexLoadOp>(op)) {
      if (indexLoad)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "tcrv_rvv.index_load");
      indexLoad = candidate;
      continue;
    }
    if (auto candidate =
            llvm::dyn_cast<tcrv::rvv::MaskedIndexedLoadOp>(op)) {
      if (indexedGather)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "tcrv_rvv.masked_indexed_load gather");
      indexedGather = candidate;
      continue;
    }
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::MaskedMAccOp>(op)) {
      if (maskedMAcc)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "tcrv_rvv.masked_macc");
      maskedMAcc = candidate;
      continue;
    }
    if (auto candidate =
            llvm::dyn_cast<tcrv::rvv::MaskedIndexedStoreOp>(op)) {
      if (indexedScatter)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "tcrv_rvv.masked_indexed_store scatter");
      indexedScatter = candidate;
      continue;
    }
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("Stage2 RVV composite gather-MAcc-scatter route rejects "
                    "unsupported selected-body op '") +
        op.getName().getStringRef() + "' in the composite with_vl body");
  }

  if (genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires exactly four "
        "tcrv_rvv.load ops for compare lhs, payload RHS, accumulator, and old "
        "scatter destination passthrough");
  if (!runtimeScalarSplat || !compare || !indexLoad || !indexedGather ||
      !maskedMAcc || !indexedScatter)
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.splat, compare, index_load, masked_indexed_load, "
        "masked_macc, and masked_indexed_store in one selected with_vl body");

  tcrv::rvv::LoadOp compareLhsLoad;
  tcrv::rvv::LoadOp payloadLoad;
  tcrv::rvv::LoadOp accumulatorLoad;
  tcrv::rvv::LoadOp oldDestinationLoad;
  support::RuntimeABIParameter compareLhsABI;
  support::RuntimeABIParameter payloadABI;
  support::RuntimeABIParameter accumulatorABI;
  support::RuntimeABIParameter oldDestinationABI;

  for (tcrv::rvv::LoadOp load : genericLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(),
            "Stage2 RVV composite gather-MAcc-scatter tcrv_rvv.load buffer "
            "operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::DotRHSInputBuffer,
             support::RuntimeABIParameterRole::AccumulatorInputBuffer,
             support::RuntimeABIParameterRole::OutputBuffer});
    if (!parameter)
      return parameter.takeError();

    switch (parameter->role) {
    case support::RuntimeABIParameterRole::LHSInputBuffer:
      if (compareLhsLoad)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "compare lhs load bound to lhs-input-buffer");
      compareLhsLoad = load;
      compareLhsABI = *parameter;
      break;
    case support::RuntimeABIParameterRole::DotRHSInputBuffer:
      if (payloadLoad)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "payload RHS load bound to dot-rhs-input-buffer");
      payloadLoad = load;
      payloadABI = *parameter;
      break;
    case support::RuntimeABIParameterRole::AccumulatorInputBuffer:
      if (accumulatorLoad)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "accumulator load bound to accumulator-input-buffer");
      accumulatorLoad = load;
      accumulatorABI = *parameter;
      break;
    case support::RuntimeABIParameterRole::OutputBuffer:
      if (oldDestinationLoad)
        return makeRVVEmitCRouteProviderError(
            "Stage2 RVV composite gather-MAcc-scatter route requires a unique "
            "old destination passthrough load bound to output-buffer");
      oldDestinationLoad = load;
      oldDestinationABI = *parameter;
      break;
    default:
      llvm_unreachable("unexpected composite load runtime ABI role");
    }
  }

  if (!compareLhsLoad || !payloadLoad || !accumulatorLoad ||
      !oldDestinationLoad)
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires compare lhs, "
        "payload RHS, accumulator, and old destination passthrough load roles");

  if (llvm::Error error = assignRVVGenericLoadBinding(
          slice, compareLhsLoad, compareLhsABI))
    return error;
  if (llvm::Error error =
          assignRVVGenericLoadBinding(slice, payloadLoad, payloadABI))
    return error;
  if (llvm::Error error = assignRVVGenericLoadBinding(
          slice, accumulatorLoad, accumulatorABI))
    return error;

  llvm::Expected<support::RuntimeABIParameter> rhsABI =
      getRuntimeABIParameterBindingFromValue(
          runtimeScalarSplat.getScalar(),
          "Stage2 RVV composite gather-MAcc-scatter tcrv_rvv.splat scalar "
          "operand",
          {support::RuntimeABIParameterRole::RHSScalarValue});
  if (!rhsABI)
    return rhsABI.takeError();
  if (llvm::Error error =
          assignRVVGenericScalarSplatBinding(slice, runtimeScalarSplat,
                                             *rhsABI))
    return error;

  llvm::Expected<support::RuntimeABIParameter> indexABI =
      getRuntimeABIParameterBindingFromValue(
          indexLoad.getIndex(),
          "Stage2 RVV composite gather-MAcc-scatter tcrv_rvv.index_load index "
          "operand",
          {support::RuntimeABIParameterRole::IndexInputBuffer});
  if (!indexABI)
    return indexABI.takeError();
  if (llvm::Error error =
          assignRVVGenericIndexLoadBinding(slice, indexLoad, *indexABI))
    return error;

  llvm::Expected<support::RuntimeABIParameter> sourceABI =
      getRuntimeABIParameterBindingFromValue(
          indexedGather.getData(),
          "Stage2 RVV composite gather-MAcc-scatter "
          "tcrv_rvv.masked_indexed_load source operand",
          {support::RuntimeABIParameterRole::SourceInputBuffer});
  if (!sourceABI)
    return sourceABI.takeError();
  if (llvm::Error error = assignRVVGenericMaskedIndexedLoadBinding(
          slice, indexedGather, *sourceABI))
    return error;

  llvm::Expected<support::RuntimeABIParameter> scatterABI =
      getRuntimeABIParameterBindingFromValue(
          indexedScatter.getDestination(),
          "Stage2 RVV composite gather-MAcc-scatter "
          "tcrv_rvv.masked_indexed_store destination operand",
          {support::RuntimeABIParameterRole::OutputBuffer});
  if (!scatterABI)
    return scatterABI.takeError();

  if (oldDestinationABI.role != scatterABI->role ||
      oldDestinationABI.ownership != scatterABI->ownership ||
      oldDestinationABI.cName != scatterABI->cName ||
      oldDestinationABI.cType != scatterABI->cType)
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires the old "
        "destination passthrough load and masked indexed scatter destination "
        "to bind the same output-buffer ABI value");
  if (oldDestinationLoad.getBuffer() != indexedScatter.getDestination())
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires the old "
        "destination passthrough load to read the same output buffer used by "
        "tcrv_rvv.masked_indexed_store");
  if (indexedGather.getPassthrough() != oldDestinationLoad.getLoaded())
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.masked_indexed_load passthrough to consume the old "
        "destination load result");

  if (llvm::Error error = recordRVVSelectedBodyCompare(slice, compare))
    return error;
  if (indexedGather.getMask() != slice.compareMask ||
      maskedMAcc.getMask() != slice.compareMask ||
      indexedScatter.getMask() != slice.compareMask)
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires gather, "
        "MAcc, and scatter to consume the mask produced by tcrv_rvv.compare");
  if (compare.getLhs() != compareLhsLoad.getLoaded() ||
      compare.getRhs() != runtimeScalarSplat.getBroadcast())
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.compare to consume the compare lhs load and runtime scalar "
        "splat result");
  if (indexedGather.getIndices() != slice.indexValue ||
      indexedScatter.getIndices() != slice.indexValue)
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires gather and "
        "scatter to consume the index vector produced by tcrv_rvv.index_load");
  if (maskedMAcc.getLhs() != indexedGather.getLoaded())
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.masked_macc lhs to consume the indexed gather result");
  if (maskedMAcc.getRhs() != payloadLoad.getLoaded())
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.masked_macc rhs to consume the payload RHS load result");
  if (maskedMAcc.getAccumulator() != accumulatorLoad.getLoaded())
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.masked_macc accumulator to consume the accumulator input "
        "load result");
  if (indexedScatter.getValue() != maskedMAcc.getResult())
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.masked_indexed_store to consume the tcrv_rvv.masked_macc "
        "result");

  if (maskedMAcc.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_macc kind '") +
        maskedMAcc.getKind() +
        "' for Stage2 RVV composite gather-MAcc-scatter route");
  if (maskedMAcc.getMaskRole() != kRVVMaskedPredicateMaskRole ||
      maskedMAcc.getMaskSource() != kRVVMaskedCompareMaskSource ||
      maskedMAcc.getMaskMemoryForm() != kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.masked_macc mask role/source/memory form to mirror the "
        "compare-produced predicate mask");
  if (maskedMAcc.getAccumulatorLayout() !=
      getRVVSelectedBodyMAccAccumulatorLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_macc "
                    "accumulator_layout '") +
        maskedMAcc.getAccumulatorLayout() +
        "' for Stage2 RVV composite gather-MAcc-scatter route");
  if (maskedMAcc.getResultLayout() != getRVVSelectedBodyMAccResultLayout())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_macc result_layout '") +
        maskedMAcc.getResultLayout() +
        "' for Stage2 RVV composite gather-MAcc-scatter route");
  if (indexedScatter.getMemoryForm() !=
          kRVVMaskedIndexedStoreDestinationMemoryForm ||
      indexedScatter.getOffsetUnit() != kRVVIndexedGatherOffsetUnit ||
      static_cast<std::int64_t>(indexedScatter.getIndexEew()) != 32 ||
      indexedScatter.getIndexUniqueness() !=
          kRVVIndexedScatterIndexUniqueness ||
      indexedScatter.getInactiveLanePolicy() !=
          "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "Stage2 RVV composite gather-MAcc-scatter route requires "
        "tcrv_rvv.masked_indexed_store to carry indexed scatter destination "
        "memory form, element offsets, i32 indices, unique index policy, and "
        "preserve-output inactive-lane policy");

  if (llvm::Error error = requireRVVCompositeSelectedVL(
          compareLhsLoad.getVl(), slice.setvl.getVl(), "compare lhs load"))
    return error;
  if (llvm::Error error = requireRVVCompositeSelectedVL(
          payloadLoad.getVl(), slice.setvl.getVl(), "payload RHS load"))
    return error;
  if (llvm::Error error = requireRVVCompositeSelectedVL(
          accumulatorLoad.getVl(), slice.setvl.getVl(), "accumulator load"))
    return error;
  if (llvm::Error error =
          requireRVVCompositeSelectedVL(oldDestinationLoad.getVl(),
                                        slice.setvl.getVl(),
                                        "old destination passthrough load"))
    return error;
  if (llvm::Error error = requireRVVCompositeSelectedVL(
          runtimeScalarSplat.getVl(), slice.setvl.getVl(),
          "runtime scalar splat"))
    return error;
  if (llvm::Error error = requireRVVCompositeSelectedVL(
          compare.getVl(), slice.setvl.getVl(), "compare"))
    return error;
  if (llvm::Error error = requireRVVCompositeSelectedVL(
          indexLoad.getVl(), slice.setvl.getVl(), "index_load"))
    return error;
  if (llvm::Error error = requireRVVCompositeSelectedVL(
          indexedGather.getVl(), slice.setvl.getVl(), "masked_indexed_load"))
    return error;
  if (llvm::Error error = requireRVVCompositeSelectedVL(
          maskedMAcc.getVl(), slice.setvl.getVl(), "masked_macc"))
    return error;
  if (llvm::Error error = requireRVVCompositeSelectedVL(
          indexedScatter.getVl(), slice.setvl.getVl(),
          "masked_indexed_store"))
    return error;

  slice.runtimeElementCountABI = runtimeElementCountABI;
  slice.maskedMAccOp = maskedMAcc;
  slice.maskedIndexedStore = indexedScatter;
  slice.maskedIndexedStoreOperation = indexedScatter.getOperation();
  slice.storeOperation = indexedScatter.getOperation();
  slice.arithmeticOp = maskedMAcc.getOperation();
  slice.arithmeticKind =
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  slice.memoryForm =
      RVVSelectedBodyMemoryForm::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  slice.maskValue = slice.compareMask;
  slice.maskedPassthrough = indexedGather.getPassthrough();
  slice.maskedActiveValue = maskedMAcc.getResult();
  slice.maskedInactivePassthrough = indexedGather.getPassthrough();
  slice.oldDestinationLoadOperation = oldDestinationLoad.getOperation();
  slice.oldDestinationValue = oldDestinationLoad.getLoaded();
  slice.arithmeticLhs = indexedGather.getLoaded();
  slice.arithmeticRhs = payloadLoad.getLoaded();
  slice.arithmeticAccumulator = accumulatorLoad.getLoaded();
  slice.arithmeticResult = maskedMAcc.getResult();
  slice.dotLHSBuffer = indexedGather.getData();
  slice.dotLHSValue = indexedGather.getLoaded();
  slice.indexedDestinationBuffer = indexedScatter.getDestination();
  slice.outBuffer = indexedScatter.getDestination();
  slice.outABI = *scatterABI;
  slice.storeValue = indexedScatter.getValue();
  return llvm::Error::success();
}
} // namespace

llvm::Expected<RVVSelectedBodyRouteSlice>
collectRVVSelectedBodyRouteSlice(tcrv::exec::VariantOp variant) {
  llvm::SmallVector<tcrv::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<tcrv::rvv::WithVLOp, 2> withVLs;
  unsigned rvvOpCount = 0;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    ++rvvOpCount;
    if (auto setvl = llvm::dyn_cast<tcrv::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.setvl op");

  RVVSelectedBodyRouteSlice slice;
  slice.setvl = setvls.front();
  if (withVLs.size() == 1) {
    slice.withVL = withVLs.front();
  } else if (withVLs.size() == 2) {
    tcrv::rvv::GearboxCrossRegionHandoffOp producerHandoff;
    for (tcrv::rvv::WithVLOp withVL : withVLs) {
      tcrv::rvv::GearboxCrossRegionHandoffOp handoff =
          findDirectRVVGearboxCrossRegionHandoff(withVL);
      if (!handoff)
        continue;
      if (producerHandoff)
        return makeRVVEmitCRouteProviderError(
            "bounded Gearbox multi-with_vl RVV route requires a unique "
            "producer tcrv_rvv.with_vl with a direct "
            "tcrv_rvv.gearbox_cross_region_handoff");
      slice.gearboxProducerWithVL = withVL;
      producerHandoff = handoff;
    }
    if (!slice.gearboxProducerWithVL)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route supports multiple tcrv_rvv.with_vl ops "
          "only for Gearbox product-reduction/dequant producer-consumer "
          "route collection with a direct "
          "tcrv_rvv.gearbox_cross_region_handoff");

    for (tcrv::rvv::WithVLOp withVL : withVLs) {
      if (withVL == slice.gearboxProducerWithVL)
        continue;
      if (!isRVVGearboxProductReduceDequantConsumerScope(
              slice.gearboxProducerWithVL, withVL))
        continue;
      if (slice.gearboxConsumerWithVL)
        return makeRVVEmitCRouteProviderError(
            "bounded Gearbox multi-with_vl RVV route requires a unique "
            "consumer tcrv_rvv.with_vl carrying the dequant-store scope");
      slice.gearboxConsumerWithVL = withVL;
    }
    if (!slice.gearboxConsumerWithVL)
      return makeRVVEmitCRouteProviderError(
          "bounded Gearbox multi-with_vl RVV route requires a nested consumer "
          "tcrv_rvv.with_vl that consumes the same VL and carries "
          "dequant-store marker/dequant/store facts");
    if (!isRVVGearboxConsumerScopeOrderedAfterHandoff(
            slice.gearboxProducerWithVL, slice.gearboxConsumerWithVL,
            producerHandoff))
      return makeRVVEmitCRouteProviderError(
          "bounded Gearbox multi-with_vl RVV route requires the consumer "
          "tcrv_rvv.with_vl to be structurally ordered after the producer "
          "tcrv_rvv.gearbox_cross_region_handoff");
    slice.withVL = slice.gearboxProducerWithVL;
  } else {
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.with_vl op, "
        "or exactly two Gearbox producer/consumer tcrv_rvv.with_vl ops");
  }

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(slice.setvl,
                                                          slice.withVL);
  if (!configDiagnostic.ok)
    return makeRVVEmitCRouteProviderError(configDiagnostic.message);
  if (slice.gearboxConsumerWithVL) {
    tcrv::rvv::RVVConfigContractDiagnostic consumerConfigDiagnostic =
        tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(
            slice.setvl, slice.gearboxConsumerWithVL);
    if (!consumerConfigDiagnostic.ok)
      return makeRVVEmitCRouteProviderError(consumerConfigDiagnostic.message);
  }

  llvm::Expected<support::RuntimeABIParameter> runtimeElementCountABI =
      getRuntimeABIParameterBindingFromValue(
          slice.setvl.getAvl(), "tcrv_rvv.setvl AVL operand",
          {support::RuntimeABIParameterRole::RuntimeElementCount});
  if (!runtimeElementCountABI)
    return runtimeElementCountABI.takeError();

  if (!slice.gearboxConsumerWithVL &&
      isRuntimeScalarComputedMaskIndexedGatherMAccScatterCompositeCandidate(
          slice.withVL)) {
    if (llvm::Error error =
            recordRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherMAccScatter(
                slice, *runtimeElementCountABI))
      return std::move(error);
    return slice;
  }

  llvm::SmallVector<tcrv::rvv::LoadOp, 2> genericLoads;
  llvm::SmallVector<tcrv::rvv::StridedLoadOp, 2> genericStridedLoads;
  llvm::SmallVector<tcrv::rvv::IndexLoadOp, 1> genericIndexLoads;
  llvm::SmallVector<tcrv::rvv::IndexedLoadOp, 1> genericIndexedLoads;
  llvm::SmallVector<tcrv::rvv::IndexedStoreOp, 1> genericIndexedStores;
  llvm::SmallVector<tcrv::rvv::MaskLoadOp, 1> genericMaskLoads;
  llvm::SmallVector<tcrv::rvv::MaskedLoadOp, 1> genericMaskedLoads;
  llvm::SmallVector<tcrv::rvv::MaskedStridedLoadOp, 1>
      genericMaskedStridedLoads;
  llvm::SmallVector<tcrv::rvv::MaskedIndexedLoadOp, 1>
      genericMaskedIndexedLoads;
  llvm::SmallVector<tcrv::rvv::MaskedIndexedStoreOp, 1>
      genericMaskedIndexedStores;
  llvm::SmallVector<tcrv::rvv::MaskedSegment2LoadOp, 1>
      genericMaskedSegment2Loads;
  llvm::SmallVector<tcrv::rvv::MaskedSegment2StoreOp, 1>
      genericMaskedSegment2Stores;
  llvm::SmallVector<tcrv::rvv::Segment2LoadOp, 1> genericSegment2Loads;
  llvm::SmallVector<tcrv::rvv::Segment2StoreOp, 1> genericSegment2Stores;
  llvm::SmallVector<tcrv::rvv::VSetVLRegionMarkerOp, 4>
      vsetvlRegionMarkers;
  llvm::SmallVector<tcrv::rvv::BroadcastLoadOp, 1> genericBroadcastLoads;
  llvm::SmallVector<tcrv::rvv::SplatOp, 2> genericScalarSplats;
  llvm::SmallVector<tcrv::rvv::StoreOp, 2> genericStores;
  unsigned storeCount = 0;
  unsigned stridedStoreCount = 0;
  auto recordScopedRouteOp = [&](mlir::Operation &op) -> llvm::Error {
    return recordRVVSelectedBodyScopedRouteOp(
        slice, op, genericLoads, genericStridedLoads, genericIndexLoads,
        genericIndexedLoads, genericIndexedStores, genericMaskLoads,
        genericMaskedLoads, genericMaskedStridedLoads,
        genericMaskedIndexedLoads, genericMaskedIndexedStores,
        genericMaskedSegment2Loads, genericMaskedSegment2Stores,
        genericSegment2Loads, genericSegment2Stores, vsetvlRegionMarkers,
        genericBroadcastLoads, genericScalarSplats, genericStores, storeCount,
        stridedStoreCount, *runtimeElementCountABI);
  };
  if (slice.gearboxConsumerWithVL) {
    bool sawHandoff = false;
    bool sawConsumerScope = false;
    for (mlir::Operation &op : slice.gearboxProducerWithVL.getBody().front()) {
      if (auto nestedWithVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op)) {
        if (nestedWithVL != slice.gearboxConsumerWithVL)
          return makeRVVEmitCRouteProviderError(
              "bounded Gearbox multi-with_vl RVV route does not support "
              "unrelated nested tcrv_rvv.with_vl scopes in the producer body");
        if (!sawHandoff)
          return makeRVVEmitCRouteProviderError(
              "bounded Gearbox multi-with_vl RVV route requires the consumer "
              "tcrv_rvv.with_vl to appear after the producer handoff");
        sawConsumerScope = true;
        continue;
      }
      if (llvm::Error error = recordScopedRouteOp(op))
        return std::move(error);
      if (llvm::isa<tcrv::rvv::GearboxCrossRegionHandoffOp>(op))
        sawHandoff = true;
    }
    if (!sawHandoff || !sawConsumerScope)
      return makeRVVEmitCRouteProviderError(
          "bounded Gearbox multi-with_vl RVV route requires direct producer "
          "handoff and nested consumer scope in structural order");
    for (mlir::Operation &op : slice.gearboxConsumerWithVL.getBody().front()) {
      if (llvm::isa<tcrv::rvv::WithVLOp>(op))
        return makeRVVEmitCRouteProviderError(
            "bounded Gearbox multi-with_vl RVV route does not support nested "
            "tcrv_rvv.with_vl below the consumer dequant-store scope");
      if (llvm::Error error = recordScopedRouteOp(op))
        return std::move(error);
    }
  } else {
  for (mlir::Operation &op : slice.withVL.getBody().front()) {
    if (auto marker = llvm::dyn_cast<tcrv::rvv::VSetVLRegionMarkerOp>(op)) {
      vsetvlRegionMarkers.push_back(marker);
      continue;
    }
    if (auto handoff =
            llvm::dyn_cast<tcrv::rvv::GearboxCrossRegionHandoffOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyGearboxCrossRegionHandoff(
                  slice, handoff, *runtimeElementCountABI))
        return std::move(error);
      continue;
    }
    if (auto load = llvm::dyn_cast<tcrv::rvv::LoadOp>(op)) {
      genericLoads.push_back(load);
      continue;
    }
    if (auto stridedLoad = llvm::dyn_cast<tcrv::rvv::StridedLoadOp>(op)) {
      genericStridedLoads.push_back(stridedLoad);
      continue;
    }
    if (auto indexLoad = llvm::dyn_cast<tcrv::rvv::IndexLoadOp>(op)) {
      genericIndexLoads.push_back(indexLoad);
      continue;
    }
    if (auto indexedLoad = llvm::dyn_cast<tcrv::rvv::IndexedLoadOp>(op)) {
      genericIndexedLoads.push_back(indexedLoad);
      continue;
    }
    if (auto indexedStore = llvm::dyn_cast<tcrv::rvv::IndexedStoreOp>(op)) {
      genericIndexedStores.push_back(indexedStore);
      continue;
    }
    if (auto maskLoad = llvm::dyn_cast<tcrv::rvv::MaskLoadOp>(op)) {
      genericMaskLoads.push_back(maskLoad);
      continue;
    }
    if (auto maskedLoad = llvm::dyn_cast<tcrv::rvv::MaskedLoadOp>(op)) {
      genericMaskedLoads.push_back(maskedLoad);
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedLoad(slice, maskedLoad))
        return std::move(error);
      continue;
    }
    if (auto maskedStridedLoad =
            llvm::dyn_cast<tcrv::rvv::MaskedStridedLoadOp>(op)) {
      genericMaskedStridedLoads.push_back(maskedStridedLoad);
      if (llvm::Error error = recordRVVSelectedBodyMaskedStridedLoad(
              slice, maskedStridedLoad))
        return std::move(error);
      continue;
    }
    if (auto maskedIndexedLoad =
            llvm::dyn_cast<tcrv::rvv::MaskedIndexedLoadOp>(op)) {
      genericMaskedIndexedLoads.push_back(maskedIndexedLoad);
      if (llvm::Error error = recordRVVSelectedBodyMaskedIndexedLoad(
              slice, maskedIndexedLoad))
        return std::move(error);
      continue;
    }
    if (auto maskedIndexedStore =
            llvm::dyn_cast<tcrv::rvv::MaskedIndexedStoreOp>(op)) {
      genericMaskedIndexedStores.push_back(maskedIndexedStore);
      if (llvm::Error error = recordRVVSelectedBodyMaskedIndexedStore(
              slice, maskedIndexedStore))
        return std::move(error);
      continue;
    }
    if (auto maskedSegment2Load =
            llvm::dyn_cast<tcrv::rvv::MaskedSegment2LoadOp>(op)) {
      genericMaskedSegment2Loads.push_back(maskedSegment2Load);
      if (llvm::Error error = recordRVVSelectedBodyMaskedSegment2Load(
              slice, maskedSegment2Load))
        return std::move(error);
      continue;
    }
    if (auto maskedSegment2Store =
            llvm::dyn_cast<tcrv::rvv::MaskedSegment2StoreOp>(op)) {
      genericMaskedSegment2Stores.push_back(maskedSegment2Store);
      if (llvm::Error error = recordRVVSelectedBodyMaskedSegment2Store(
              slice, maskedSegment2Store))
        return std::move(error);
      continue;
    }
    if (auto segment2Load = llvm::dyn_cast<tcrv::rvv::Segment2LoadOp>(op)) {
      genericSegment2Loads.push_back(segment2Load);
      continue;
    }
    if (auto segment2Store = llvm::dyn_cast<tcrv::rvv::Segment2StoreOp>(op)) {
      genericSegment2Stores.push_back(segment2Store);
      continue;
    }
    if (auto broadcast = llvm::dyn_cast<tcrv::rvv::BroadcastLoadOp>(op)) {
      genericBroadcastLoads.push_back(broadcast);
      continue;
    }
    if (auto splat = llvm::dyn_cast<tcrv::rvv::SplatOp>(op)) {
      genericScalarSplats.push_back(splat);
      continue;
    }
    if (auto binary = llvm::dyn_cast<tcrv::rvv::BinaryOp>(op)) {
      llvm::Expected<RVVSelectedBodyOperationKind> kind =
          parseRVVSelectedBodyBinaryKind(binary.getKind());
      if (!kind)
        return kind.takeError();
      if (llvm::Error error = recordRVVSelectedBodyOperation(
              slice, binary.getOperation(), *kind, binary.getLhs(),
              binary.getRhs(), binary.getResult()))
        return std::move(error);
      continue;
    }
    if (auto compare = llvm::dyn_cast<tcrv::rvv::CompareOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyCompare(slice, compare))
        return std::move(error);
      continue;
    }
    if (auto maskAnd = llvm::dyn_cast<tcrv::rvv::MaskAndOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMaskAnd(slice, maskAnd))
        return std::move(error);
      continue;
    }
    if (auto maskedBinary = llvm::dyn_cast<tcrv::rvv::MaskedBinaryOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedBinary(slice, maskedBinary))
        return std::move(error);
      continue;
    }
    if (auto select = llvm::dyn_cast<tcrv::rvv::SelectOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodySelect(slice, select))
        return std::move(error);
      continue;
    }
    if (auto reduce = llvm::dyn_cast<tcrv::rvv::ReduceOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyReduction(slice, reduce))
        return std::move(error);
      continue;
    }
    if (auto standaloneReduce =
            llvm::dyn_cast<tcrv::rvv::StandaloneReduceOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyStandaloneReduction(slice,
                                                       standaloneReduce))
        return std::move(error);
      continue;
    }
    if (auto maskedStandaloneReduce =
            llvm::dyn_cast<tcrv::rvv::MaskedStandaloneReduceOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedStandaloneReduction(
                  slice, maskedStandaloneReduce))
        return std::move(error);
      continue;
    }
    if (auto macc = llvm::dyn_cast<tcrv::rvv::MAccOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMAcc(slice, macc))
        return std::move(error);
      continue;
    }
    if (auto maskedMAcc = llvm::dyn_cast<tcrv::rvv::MaskedMAccOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedMAcc(slice, maskedMAcc))
        return std::move(error);
      continue;
    }
    if (auto wideningMAcc =
            llvm::dyn_cast<tcrv::rvv::WideningMAccOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyWideningMAcc(slice, wideningMAcc))
        return std::move(error);
      continue;
    }
    if (auto product = llvm::dyn_cast<tcrv::rvv::WideningProductOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyWideningProduct(slice, product))
        return std::move(error);
      continue;
    }
    if (auto nibbleProduct =
            llvm::dyn_cast<tcrv::rvv::PackedI4NibbleUnpackProductOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyNibbleUnpackProduct(slice, nibbleProduct))
        return std::move(error);
      continue;
    }
    if (auto dotReduce =
            llvm::dyn_cast<tcrv::rvv::WideningDotReduceOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyWideningDotReduce(slice, dotReduce))
        return std::move(error);
      continue;
    }
    if (auto maskedDotReduce =
            llvm::dyn_cast<tcrv::rvv::MaskedWideningDotReduceOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMaskedWideningDotReduce(
              slice, maskedDotReduce))
        return std::move(error);
      continue;
    }
    if (auto conversion = llvm::dyn_cast<tcrv::rvv::WideningConvertOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyWideningConvert(slice, conversion))
        return std::move(error);
      continue;
    }
    if (auto dequantize = llvm::dyn_cast<tcrv::rvv::DequantizeOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyDequantize(slice, dequantize))
        return std::move(error);
      continue;
    }
    if (auto move = llvm::dyn_cast<tcrv::rvv::MoveOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMove(slice, move))
        return std::move(error);
      continue;
    }
    if (auto maskedMove = llvm::dyn_cast<tcrv::rvv::MaskedMoveOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedMove(slice, maskedMove))
        return std::move(error);
      continue;
    }
    if (auto maskedStore = llvm::dyn_cast<tcrv::rvv::MaskedStoreOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedStore(slice, maskedStore))
        return std::move(error);
      continue;
    }
    if (auto maskedStridedStore =
            llvm::dyn_cast<tcrv::rvv::MaskedStridedStoreOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyMaskedStridedStore(
              slice, maskedStridedStore))
        return std::move(error);
      continue;
    }
    if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(op)) {
      slice.genericStore = store;
      genericStores.push_back(store);
      ++storeCount;
      continue;
    }
    if (auto stridedStore = llvm::dyn_cast<tcrv::rvv::StridedStoreOp>(op)) {
      slice.stridedStore = stridedStore;
      ++stridedStoreCount;
      continue;
    }
    if (op.getName().getStringRef().starts_with("tcrv_rvv.i32_"))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("legacy selected-body op '") +
          op.getName().getStringRef() +
          "' is fail-closed during RVV Stage1; Stage2 routes must use generic "
          "tcrv_rvv.load, tcrv_rvv.broadcast_load, "
          "tcrv_rvv.splat, tcrv_rvv.strided_load, tcrv_rvv.binary, "
          "tcrv_rvv.index_load, tcrv_rvv.indexed_load, tcrv_rvv.segment2_load, "
          "tcrv_rvv.segment2_store, "
          "tcrv_rvv.indexed_store, tcrv_rvv.mask_load, tcrv_rvv.compare, "
          "tcrv_rvv.masked_binary, tcrv_rvv.select, tcrv_rvv.reduce, "
          "tcrv_rvv.standalone_reduce, tcrv_rvv.masked_standalone_reduce, "
          "tcrv_rvv.macc, tcrv_rvv.masked_macc, "
          "tcrv_rvv.widening_product, tcrv_rvv.widening_convert, "
          "tcrv_rvv.gearbox_cross_region_handoff, "
          "tcrv_rvv.move, "
          "tcrv_rvv.widening_dot_reduce, "
          "tcrv_rvv.masked_widening_dot_reduce, "
          "tcrv_rvv.masked_move, tcrv_rvv.masked_load, "
          "tcrv_rvv.masked_strided_load, tcrv_rvv.masked_indexed_load, "
          "tcrv_rvv.masked_indexed_store, "
          "tcrv_rvv.masked_segment2_load, tcrv_rvv.masked_store, "
          "tcrv_rvv.masked_strided_store, "
          "tcrv_rvv.store, and "
          "tcrv_rvv.strided_store body structure");
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded RVV EmitC route does not support op '") +
        op.getName().getStringRef() +
        "' inside tcrv_rvv.with_vl; expected generic load, broadcast_load, "
        "splat, strided_load, index_load, indexed_load, indexed_store, "
        "mask_load, segment2_load, segment2_store, binary, compare, "
        "masked_binary, select, reduce, standalone_reduce, "
        "masked_standalone_reduce, macc, masked_macc, "
        "widening_product, packed_i4_nibble_unpack_product, widening_convert, "
      "gearbox_cross_region_handoff, "
        "widening_dot_reduce, "
        "masked_widening_dot_reduce, move, masked_move, masked_load, "
        "masked_strided_load, masked_indexed_load, masked_indexed_store, "
        "masked_segment2_load, "
        "masked_store, masked_strided_store, store, and strided_store only");
  }
  }
  slice.vsetvlRegionMarkers = std::move(vsetvlRegionMarkers);

  const bool hasIndexedMemory =
      !genericIndexLoads.empty() || !genericIndexedLoads.empty() ||
      !genericIndexedStores.empty() || !genericMaskedIndexedLoads.empty() ||
      !genericMaskedIndexedStores.empty();
  const bool hasMaskedMemory =
      !genericMaskLoads.empty() || !genericMaskedLoads.empty() ||
      !genericMaskedStridedLoads.empty() || !genericMaskedIndexedLoads.empty() ||
      !genericMaskedIndexedStores.empty() ||
      !genericMaskedSegment2Loads.empty() ||
      !genericMaskedSegment2Stores.empty() ||
      static_cast<bool>(slice.maskedMoveOp) ||
      static_cast<bool>(slice.maskedStore) ||
      static_cast<bool>(slice.maskedStridedStore);
  const bool hasSegmentedMemory = !genericSegment2Loads.empty() ||
                                  !genericMaskedSegment2Loads.empty() ||
                                  !genericMaskedSegment2Stores.empty() ||
                                  !genericSegment2Stores.empty() ||
                                  static_cast<bool>(slice.field0MoveOp) ||
                                  static_cast<bool>(slice.field1MoveOp);
  const bool hasScalarBroadcast = !genericScalarSplats.empty();
  if (!genericMaskedIndexedLoads.empty() && storeCount == 1 &&
      slice.compareOp && hasScalarBroadcast && genericLoads.size() == 2) {
    slice.arithmeticKind = RVVSelectedBodyOperationKind::
        RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  }
  if (!genericMaskedIndexedLoads.empty() && storeCount == 1 &&
      slice.compareOp && genericLoads.size() == 3) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  }
  if (!genericMaskedIndexedStores.empty() && slice.compareOp &&
      genericLoads.size() == 3) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  }
  if (!genericMaskedIndexedStores.empty() && slice.compareOp &&
      hasScalarBroadcast && genericLoads.size() == 2) {
    slice.arithmeticKind = RVVSelectedBodyOperationKind::
        RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  }
  if (!genericMaskedSegment2Loads.empty() && slice.compareOp &&
      hasScalarBroadcast && genericLoads.size() == 3 && storeCount == 2) {
    slice.arithmeticKind = RVVSelectedBodyOperationKind::
        RuntimeScalarComputedMaskSegment2LoadUnitStore;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  }
  if (!genericMaskedSegment2Loads.empty() && slice.compareOp &&
      genericLoads.size() == 4 && storeCount == 2) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  }
  if (!genericMaskedSegment2Stores.empty() && slice.compareOp &&
      hasScalarBroadcast && genericLoads.size() == 3 &&
      slice.arithmeticKind !=
          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad) {
    slice.arithmeticKind = RVVSelectedBodyOperationKind::
        RuntimeScalarComputedMaskSegment2StoreUnitLoad;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  }
  if (!genericMaskedSegment2Stores.empty() && slice.compareOp &&
      genericLoads.size() == 4 &&
      slice.arithmeticKind !=
          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  }
  if (!genericSegment2Loads.empty()) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
    slice.memoryForm = RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  }
  if (hasIndexedMemory && genericMaskedIndexedLoads.empty() &&
      genericMaskedIndexedStores.empty()) {
    if (!slice.moveOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed memory route requires exactly one "
          "tcrv_rvv.move {kind = \"copy\"} movement op");
    slice.arithmeticKind =
        genericIndexedStores.empty()
            ? RVVSelectedBodyOperationKind::IndexedGatherUnitStore
            : RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  }
  if (!genericSegment2Stores.empty()) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
    slice.arithmeticOp = genericSegment2Stores.front().getOperation();
  }
  const bool hasStridedMemory =
      !genericStridedLoads.empty() || static_cast<bool>(slice.stridedStore) ||
      static_cast<bool>(slice.maskedStridedStore) ||
      !genericMaskedStridedLoads.empty();
  if (hasStridedMemory && genericStridedLoads.empty() &&
      stridedStoreCount == 1 && slice.moveOp)
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::UnitLoadStridedStore;
  if (hasStridedMemory && genericStridedLoads.empty() &&
      stridedStoreCount == 0 && slice.maskedStridedStore && slice.compareOp &&
      genericLoads.size() == 3)
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  if (hasStridedMemory && genericMaskedStridedLoads.size() == 1 &&
      stridedStoreCount == 0 && storeCount == 1 && slice.compareOp &&
      genericLoads.size() == 3)
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
  if (hasStridedMemory && genericStridedLoads.size() == 2 &&
      stridedStoreCount == 0 && storeCount == 1 &&
      slice.wideningDotReduceOp && genericLoads.empty()) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce;
  }
  if (hasStridedMemory && genericStridedLoads.size() == 2 &&
      stridedStoreCount == 0 && storeCount == 1 &&
      slice.maskedWideningDotReduceOp && slice.compareOp &&
      genericLoads.size() == 2) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::
            ComputedMaskStridedInputWideningDotReduceAdd;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::
            ComputedMaskStridedInputWideningDotReduce;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.selectOp && slice.compareOp && genericLoads.size() == 4 &&
      storeCount == 1) {
    slice.arithmeticKind = RVVSelectedBodyOperationKind::ComputedMaskSelect;
    slice.memoryForm = RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.selectOp && slice.compareOp && slice.secondaryCompareOp &&
      slice.maskAndOp && genericScalarSplats.size() == 2 &&
      genericLoads.size() == 4 && storeCount == 1) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.selectOp && slice.secondarySelectOp && slice.compareOp &&
      slice.secondaryCompareOp && !slice.maskAndOp &&
      slice.dequantizeOp && genericScalarSplats.size() == 2 &&
      genericLoads.size() == 1 && storeCount == 1) {
    slice.arithmeticKind = RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.selectOp && slice.secondarySelectOp && slice.compareOp &&
      slice.secondaryCompareOp && !slice.maskAndOp && !slice.dequantizeOp &&
      genericScalarSplats.size() == 2 && genericLoads.size() == 1 &&
      storeCount == 1) {
    slice.arithmeticKind = RVVSelectedBodyOperationKind::F32ClampSelect;
    slice.memoryForm = RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.selectOp && slice.compareOp && hasScalarBroadcast &&
      genericLoads.size() == 3 && storeCount == 1) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;
    slice.memoryForm = RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.maskedStore && slice.compareOp && hasScalarBroadcast &&
      genericLoads.size() == 2 && storeCount == 0) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.maskedLoadOp && slice.compareOp && hasScalarBroadcast &&
      genericLoads.size() == 2 && storeCount == 1) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.maskedMAccOp && slice.compareOp && hasScalarBroadcast &&
      genericLoads.size() == 4 && storeCount == 1) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskUnitStrideMAcc;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.maskedStandaloneReduceOp && slice.compareOp &&
      genericLoads.size() == 3 && storeCount == 1) {
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideStandaloneReduction;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      slice.maskedStandaloneReduceOp && slice.compareOp && hasScalarBroadcast &&
      genericLoads.size() == 2 && storeCount == 1) {
    llvm::StringRef reduceKind = slice.maskedStandaloneReduceOp.getKind();
    if (reduceKind == "add")
      slice.arithmeticKind = RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskStandaloneReduceAdd;
    else if (reduceKind == "min")
      slice.arithmeticKind = RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskStandaloneReduceMin;
    else if (reduceKind == "max")
      slice.arithmeticKind = RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskStandaloneReduceMax;
    else
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route currently supports only "
          "tcrv_rvv.masked_standalone_reduce {kind = \"add\", \"min\", or "
          "\"max\"}");
    slice.memoryForm = RVVSelectedBodyMemoryForm::
        RuntimeScalarComputedMaskUnitStrideStandaloneReduction;
  }
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          slice.arithmeticKind) &&
      genericLoads.size() == 2 && storeCount == 1) {
    switch (slice.arithmeticKind) {
    case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
      slice.arithmeticKind = RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskStandaloneReduceAdd;
      break;
    case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
      slice.arithmeticKind = RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskStandaloneReduceMin;
      break;
    case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
      slice.arithmeticKind = RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskStandaloneReduceMax;
      break;
    default:
      break;
    }
    slice.memoryForm = RVVSelectedBodyMemoryForm::
        RuntimeScalarComputedMaskUnitStrideStandaloneReduction;
  }

  const bool isCompareSelect =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect;
  const bool isComputedMaskSelect =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::ComputedMaskSelect;
  const bool isRuntimeScalarCompareSelect =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;
  const bool isRuntimeScalarDualCompareMaskAndSelect =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                  RuntimeScalarDualCompareMaskAndSelect;
  const bool isF32ClampSelect =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::F32ClampSelect;
  const bool isDequantClampF32Epilogue =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
  const bool isRuntimeScalarComputedMaskStore =
      slice.maskedStore && slice.compareOp && hasScalarBroadcast &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
  const bool isRuntimeScalarComputedMaskLoadStore =
      slice.maskedLoadOp && slice.compareOp && hasScalarBroadcast &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
  const bool isReduction =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::ReduceAdd;
  const bool isStandaloneReduction =
      slice.arithmeticOp &&
      isRVVSelectedBodyPlainStandaloneReductionRouteOperation(
          slice.arithmeticKind);
  const bool isComputedMaskStandaloneReduction =
      slice.arithmeticOp &&
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          slice.arithmeticKind);
  const bool isRuntimeScalarComputedMaskStandaloneReduction =
      slice.arithmeticOp &&
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          slice.arithmeticKind);
  const bool isMaskedArithmetic =
      slice.arithmeticOp &&
      getRVVSelectedBodyOperationProfile(slice.arithmeticKind)
          .isMaskedArithmetic;
  bool isMAccAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd;
  const bool isComputedMaskedMAccAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd;
  const bool isRuntimeScalarComputedMaskedMAccAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isWideningMAccAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningProduct =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningProduct;
  const bool isWideningProductReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd;
  const bool isWideningProductReduceDequantize =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32;
  const bool isWideningProductReduceDequantClamp =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isWideningProductReductionChain =
      isWideningProductReduceAdd || isWideningProductReduceDequantize ||
      isWideningProductReduceDequantClamp;
  const bool isWideningProductReduceDequantGearboxRoute =
      isWideningProductReduceDequantize ||
      isWideningProductReduceDequantClamp;
  const bool isWideningDotReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningDotReduceAdd;
  const bool isStridedInputWideningDotReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd;
  const bool isComputedMaskWideningDotReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd;
  const bool isComputedMaskStridedInputWideningDotReduceAdd =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              ComputedMaskStridedInputWideningDotReduceAdd;
  const bool isStridedLoadUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::StridedLoadUnitStore;
  const bool isUnitLoadStridedStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::UnitLoadStridedStore;
  const bool isIndexedGatherUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::IndexedGatherUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::IndexedScatterUnitLoad;
  const bool isMaskedUnitLoadStore =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isMaskedUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isComputedMaskUnitLoadStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore;
  const bool isComputedMaskStridedStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  const bool isComputedMaskStridedLoadUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
  const bool isComputedMaskIndexedGatherLoadUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              ComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                  RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                  RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskIndexedGatherLike =
      isComputedMaskIndexedGatherLoadUnitStore ||
      isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskIndexedScatterStoreUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              ComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                  RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
  const bool isComputedMaskIndexedScatterLike =
      isComputedMaskIndexedScatterStoreUnitLoad ||
      isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
      isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskSegment2LoadUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
  const bool isRuntimeScalarComputedMaskSegment2LoadUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                  RuntimeScalarComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2StoreUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
  const bool isRuntimeScalarComputedMaskSegment2StoreUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                  RuntimeScalarComputedMaskSegment2StoreUnitLoad;
  const bool isComputedMaskSegment2UpdateUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isComputedMaskSegment2StoreLike =
      isComputedMaskSegment2StoreUnitLoad ||
      isRuntimeScalarComputedMaskSegment2StoreUnitLoad ||
      isComputedMaskSegment2UpdateUnitLoad;
  const bool isComputedMaskSegment2LoadLike =
      isComputedMaskSegment2LoadUnitStore ||
      isRuntimeScalarComputedMaskSegment2LoadUnitStore;
  const bool isSegment2DeinterleaveUnitStore =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
  const bool isSegment2InterleaveUnitLoad =
      slice.arithmeticOp &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
  const bool isWideningConversion =
      slice.arithmeticOp &&
      (slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI32ToI64 ||
       slice.arithmeticKind == RVVSelectedBodyOperationKind::WidenI16ToI32);
  const bool isDequantization =
      slice.arithmeticOp &&
      slice.arithmeticKind == RVVSelectedBodyOperationKind::DequantizeI32ToF32;
  const bool isRuntimeScalarSplatStore =
      hasScalarBroadcast && !slice.arithmeticOp && genericLoads.empty() &&
      storeCount == 1 && stridedStoreCount == 0 && !hasStridedMemory &&
      !hasIndexedMemory && !hasSegmentedMemory && !hasMaskedMemory;
  if (isRuntimeScalarSplatStore) {
    tcrv::rvv::SplatOp splat = genericScalarSplats.front();
    slice.arithmeticKind = RVVSelectedBodyOperationKind::RuntimeScalarSplatStore;
    slice.memoryForm = RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore;
    slice.arithmeticOp = splat.getOperation();
    slice.arithmeticResult = splat.getBroadcast();
  }
  if (!slice.vsetvlRegionMarkers.empty() &&
      !isWideningProductReduceDequantGearboxRoute)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route accepts tcrv_rvv.vsetvl_region_marker only "
        "for low-precision product-reduction dequantization/dequant-clamp "
        "selected-body realization");
  if (slice.gearboxCrossRegionHandoffOp &&
      !isWideningProductReduceDequantGearboxRoute)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route accepts "
        "tcrv_rvv.gearbox_cross_region_handoff only for low-precision "
        "product-reduction dequantization/dequant-clamp selected-body "
        "realization");
  // The dequant/dequant-clamp selected body has two structurally legal forms:
  //   - legacy two-scope: a gearbox_cross_region_handoff carrier + a nested
  //     consumer with_vl (producer/consumer split). BOTH must be present.
  //   - single-scope typed body (Stage 3 flip): no handoff, no consumer scope --
  //     the i32 carry feeds tcrv_rvv.dequantize directly inside the one with_vl.
  // A handoff present without a consumer scope (or vice versa) is a malformed
  // partial two-scope body and stays fail-closed.
  if (isWideningProductReduceDequantGearboxRoute &&
      static_cast<bool>(slice.gearboxCrossRegionHandoffOp) !=
          static_cast<bool>(slice.gearboxConsumerWithVL))
    return makeRVVEmitCRouteProviderError(
        "bounded Gearbox product-reduction dequantization/dequant-clamp RVV "
        "route requires either a complete two-scope producer/consumer body "
        "(both tcrv_rvv.gearbox_cross_region_handoff and the nested consumer "
        "tcrv_rvv.with_vl) or a single-scope typed body with neither; a "
        "partial two-scope body is not route authority");
  if (slice.gearboxConsumerWithVL && !isWideningProductReduceDequantGearboxRoute)
    return makeRVVEmitCRouteProviderError(
        "bounded Gearbox multi-with_vl RVV route collection is currently "
        "supported only for low-precision product-reduction dequantization/"
        "dequant-clamp");
  if (hasIndexedMemory && isIndexedGatherUnitStore &&
      (!genericStridedLoads.empty() || stridedStoreCount != 0 ||
       !genericLoads.empty() || !genericBroadcastLoads.empty() ||
       hasScalarBroadcast))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route cannot mix indexed memory "
        "ops with unit-stride input loads, broadcast, scalar splat, strided "
        "loads, or strided stores");
  if (hasIndexedMemory && isIndexedScatterUnitLoad &&
      (!genericStridedLoads.empty() || stridedStoreCount != 0 ||
       !genericBroadcastLoads.empty() || hasScalarBroadcast || storeCount != 0))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route cannot mix indexed store "
        "memory ops with broadcast, scalar splat, unit-stride stores, "
        "strided loads, or strided stores");
  if (genericScalarSplats.size() > 1 &&
      !isRuntimeScalarDualCompareMaskAndSelect && !isF32ClampSelect &&
      !isDequantClampF32Epilogue && !isWideningProductReduceDequantClamp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires at most one "
        "tcrv_rvv.splat op");
  if (genericBroadcastLoads.size() > 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires at most one "
        "tcrv_rvv.broadcast_load op");
  const bool hasRHSBroadcastLike =
      !genericBroadcastLoads.empty() || hasScalarBroadcast;
  if (!genericBroadcastLoads.empty() && hasScalarBroadcast)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route cannot mix RHS buffer broadcast and "
        "RHS scalar splat in one slice");
  bool isScalarBroadcastMAccAdd = false;
  if (isMAccAdd && hasScalarBroadcast && genericBroadcastLoads.empty()) {
    slice.arithmeticKind =
        RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
    slice.memoryForm = RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc;
    isMAccAdd = false;
    isScalarBroadcastMAccAdd = true;
  }
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      (!genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericLoads.empty() || slice.genericStore))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route cannot mix strided memory ops with "
        "unit-stride load/store, broadcast, or scalar-splat memory forms");
  if (hasStridedMemory && isUnitLoadStridedStore &&
      (!genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericStridedLoads.empty() || slice.genericStore))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route cannot mix "
        "unit-stride source memory with strided source loads, unit-stride "
        "stores, broadcast, or scalar-splat memory forms");
  if (hasStridedMemory && isStridedLoadUnitStore &&
      (!genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericLoads.empty()))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route cannot "
        "mix strided source memory with unit-stride input loads, broadcast, or "
        "scalar-splat memory forms");
  if (hasStridedMemory &&
      (isMAccAdd || isScalarBroadcastMAccAdd || isComputedMaskedMAccAdd ||
       isRuntimeScalarComputedMaskedMAccAdd || isWideningMAccAdd ||
       isWideningProduct || isWideningProductReductionChain ||
       isWideningDotReduceAdd ||
       isComputedMaskWideningDotReduceAdd))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route supports only add in this slice, "
        "not multiply-accumulate, widening product, or dot-product reduction");
  if (hasStridedMemory && (isWideningConversion || isDequantization))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion/dequantization route "
        "requires unit-stride source load and destination store");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      (!slice.arithmeticOp ||
                           slice.arithmeticKind !=
                               RVVSelectedBodyOperationKind::Add))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly one "
        "tcrv_rvv.binary {kind = \"add\"} compute op");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      genericStridedLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly two "
        "tcrv_rvv.strided_load ops for lhs and rhs");
  if (hasStridedMemory && isUnitLoadStridedStore &&
      genericStridedLoads.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route must use "
        "unit-stride tcrv_rvv.load source, not tcrv_rvv.strided_load");
  if (hasStridedMemory && isUnitLoadStridedStore && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route requires "
        "exactly one unit-stride tcrv_rvv.load source op");
  if (hasStridedMemory && isStridedLoadUnitStore &&
      genericStridedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route requires "
        "exactly one tcrv_rvv.strided_load source op");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isUnitLoadStridedStore && !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      stridedStoreCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route requires exactly one "
        "tcrv_rvv.strided_store op");
  if (hasStridedMemory && isUnitLoadStridedStore && stridedStoreCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route requires "
        "exactly one tcrv_rvv.strided_store op");
  if (hasStridedMemory && isStridedLoadUnitStore && stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route must use "
        "unit-stride tcrv_rvv.store, not tcrv_rvv.strided_store");
  if (hasStridedMemory && isStridedLoadUnitStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-load to unit-stride-store route requires "
        "exactly one unit-stride tcrv_rvv.store op");
  if ((isStridedInputWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd) &&
      stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route must use unit-stride scalar tcrv_rvv.store, not "
        "tcrv_rvv.strided_store");
  if ((isStridedInputWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd) &&
      storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route requires exactly one unit-stride scalar tcrv_rvv.store op");
  if (hasStridedMemory && isUnitLoadStridedStore && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV unit-load to strided-store route must use "
        "tcrv_rvv.strided_store, not unit-stride tcrv_rvv.store");
  if (hasIndexedMemory && genericIndexLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed memory route requires exactly one "
        "tcrv_rvv.index_load op");
  if (isIndexedGatherUnitStore && genericIndexedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires exactly one "
        "tcrv_rvv.indexed_load op");
  if (isIndexedGatherUnitStore && genericIndexedStores.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route must use unit-stride "
        "tcrv_rvv.store, not tcrv_rvv.indexed_store");
  if (isIndexedGatherUnitStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed gather route requires exactly one "
        "unit-stride tcrv_rvv.store op");
  if (isIndexedScatterUnitLoad && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires exactly one "
        "unit-stride tcrv_rvv.load source op");
  if (isIndexedScatterUnitLoad && genericIndexedLoads.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route must use unit-stride "
        "tcrv_rvv.load source, not tcrv_rvv.indexed_load");
  if (isIndexedScatterUnitLoad && genericIndexedStores.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed scatter route requires exactly one "
        "tcrv_rvv.indexed_store op");
  if (hasIndexedMemory &&
      (isMAccAdd || isScalarBroadcastMAccAdd || isComputedMaskedMAccAdd ||
       isRuntimeScalarComputedMaskedMAccAdd || isWideningMAccAdd ||
       isWideningProduct || isWideningProductReductionChain ||
       isWideningConversion || isDequantization ||
       isWideningDotReduceAdd || isComputedMaskWideningDotReduceAdd ||
       isCompareSelect || isComputedMaskSelect || isF32ClampSelect ||
       isMaskedArithmetic ||
       isReduction || isStandaloneReduction ||
       isComputedMaskStandaloneReduction))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV indexed memory route supports only "
        "index_load/indexed_load_or_unit_load/move/store_or_indexed_store "
        "memory movement in this slice");
  if (isComputedMaskIndexedGatherLike &&
      (!genericIndexedLoads.empty() || !genericIndexedStores.empty() ||
       !genericMaskedIndexedStores.empty() || slice.moveOp ||
       stridedStoreCount != 0))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed gather-load route must use "
        "index_load/masked_indexed_load/unit-stride store, not indexed_load, "
        "indexed_store, masked_indexed_store, move, or strided_store fallback");
  if (isComputedMaskIndexedScatterLike &&
      (!genericIndexedLoads.empty() || !genericIndexedStores.empty() ||
       !genericMaskedIndexedLoads.empty() || slice.moveOp ||
       stridedStoreCount != 0 || storeCount != 0))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed scatter-store route must use "
        "index_load/unit-stride payload load/masked_indexed_store, not "
        "indexed_load, indexed_store, masked_indexed_load, move, unit-stride "
        "store, or strided_store fallback");
  if (hasMaskedMemory &&
      ((!isMaskedUnitLoadStore && !isMaskedUnitStore &&
        !isRuntimeScalarComputedMaskStore &&
        !isRuntimeScalarComputedMaskLoadStore &&
        !isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
        !isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
        !isComputedMaskUnitLoadStore &&
        !isComputedMaskStridedStore &&
        !isComputedMaskStridedLoadUnitStore &&
        !isComputedMaskIndexedGatherLike &&
        !isComputedMaskIndexedScatterLike &&
        !isComputedMaskSegment2LoadUnitStore &&
        !isComputedMaskSegment2StoreLike) ||
       (hasIndexedMemory && !isComputedMaskIndexedGatherLike &&
        !isComputedMaskIndexedScatterLike) ||
       (hasStridedMemory && !isComputedMaskStridedStore &&
        !isComputedMaskStridedLoadUnitStore) ||
       !genericBroadcastLoads.empty() ||
       (hasScalarBroadcast && !isRuntimeScalarComputedMaskStore &&
        !isRuntimeScalarComputedMaskLoadStore &&
        !isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
        !isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
        !isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
        !isRuntimeScalarComputedMaskSegment2StoreUnitLoad) ||
      (hasSegmentedMemory && !isComputedMaskSegment2LoadLike &&
        !isComputedMaskSegment2StoreLike) ||
       slice.maskedBinaryOp ||
       slice.selectOp ||
       slice.reduceOp || slice.standaloneReduceOp ||
       slice.maskedStandaloneReduceOp || slice.maccOp ||
       slice.wideningMAccOp ||
       hasProductHead(slice) ||
       slice.wideningDotReduceOp ||
       isWideningConversion || isDequantization))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route supports only runtime-mask "
        "unit-stride mask_load/old-destination-load/masked_load/store, "
        "runtime-mask unit-stride load/mask_load/masked_store, "
        "computed-mask compare/source/old-destination/masked_load/store, "
        "computed-mask compare/source/masked_strided_store, or "
        "computed-mask compare/byte-strided-source/old-destination/"
        "masked_strided_load/store, computed-mask compare/indexed-source/"
        "old-destination/masked_indexed_load/store, or computed-mask "
        "compare/unit-source/indexed-destination/masked_indexed_store, or "
        "computed-mask compare/interleaved-source/old-field passthrough/"
        "masked_segment2_load/field stores, or computed-mask compare/"
        "field-payloads/interleaved-destination masked_segment2_store memory "
        "movement in this slice");
  if (hasSegmentedMemory && isComputedMaskSegment2LoadLike &&
      (hasIndexedMemory || hasStridedMemory || !genericSegment2Loads.empty() ||
       !genericSegment2Stores.empty() || !genericMaskedSegment2Stores.empty() ||
       !genericBroadcastLoads.empty() ||
       (hasScalarBroadcast &&
        !isRuntimeScalarComputedMaskSegment2LoadUnitStore) ||
       stridedStoreCount != 0 ||
       genericIndexedStores.size() != 0 || slice.maskedMoveOp ||
       slice.maskedLoadOp || slice.maskedStridedLoadOp ||
       slice.maskedIndexedLoadOp || slice.maskedIndexedStore ||
       slice.maskedStore || slice.maskedStridedStore || slice.maskedBinaryOp ||
       slice.selectOp || slice.reduceOp || slice.standaloneReduceOp ||
       slice.maskedStandaloneReduceOp || slice.maccOp ||
       slice.wideningMAccOp || slice.wideningDotReduceOp ||
       hasProductHead(slice) || slice.maskedWideningDotReduceOp ||
       isWideningConversion || isDequantization))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route supports only "
        "compare lhs load, vector rhs load or runtime scalar splat, "
        "field0/field1 old passthrough loads, masked_segment2_load, and "
        "field0/field1 stores");
  if (hasSegmentedMemory && isComputedMaskSegment2StoreLike &&
      (hasIndexedMemory || hasStridedMemory || !genericSegment2Loads.empty() ||
       !genericSegment2Stores.empty() || !genericMaskedSegment2Loads.empty() ||
       !genericBroadcastLoads.empty() ||
       (hasScalarBroadcast &&
        !isRuntimeScalarComputedMaskSegment2StoreUnitLoad) ||
       stridedStoreCount != 0 || storeCount != 0 ||
       genericIndexedStores.size() != 0 || slice.maskedMoveOp ||
       slice.maskedLoadOp || slice.maskedStridedLoadOp ||
       slice.maskedIndexedLoadOp || slice.maskedIndexedStore ||
       slice.maskedStore || slice.maskedStridedStore || slice.maskedBinaryOp ||
       slice.selectOp || slice.reduceOp || slice.standaloneReduceOp ||
       slice.maskedStandaloneReduceOp || slice.maccOp ||
       slice.wideningMAccOp || slice.wideningDotReduceOp ||
       hasProductHead(slice) || slice.maskedWideningDotReduceOp ||
       isWideningConversion || isDequantization))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store/update route "
        "supports only compare lhs/rhs loads, field0/field1 payload loads, "
        "optional add, and masked_segment2_store memory movement");
  if (hasSegmentedMemory && isSegment2DeinterleaveUnitStore &&
      (hasIndexedMemory ||
       hasStridedMemory || hasMaskedMemory || !genericLoads.empty() ||
       !genericBroadcastLoads.empty() || hasScalarBroadcast ||
       !genericSegment2Stores.empty() ||
       stridedStoreCount != 0 || genericIndexedStores.size() != 0 ||
       slice.maskedBinaryOp || slice.selectOp || slice.reduceOp ||
       slice.standaloneReduceOp || slice.maskedStandaloneReduceOp ||
       slice.maccOp || slice.wideningMAccOp || slice.wideningDotReduceOp ||
       hasProductHead(slice) || slice.maskedWideningDotReduceOp ||
       slice.compareOp ||
       isWideningConversion || isDequantization))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route supports only "
        "segment2_load/field0 move/field1 move/field0 store/field1 store "
        "memory movement in this slice");
  if (hasSegmentedMemory && isSegment2InterleaveUnitLoad &&
      (hasIndexedMemory || hasStridedMemory || hasMaskedMemory ||
       !genericSegment2Loads.empty() || !genericBroadcastLoads.empty() ||
       hasScalarBroadcast || stridedStoreCount != 0 ||
       genericIndexedStores.size() != 0 || storeCount != 0 ||
       slice.maskedBinaryOp || slice.selectOp || slice.reduceOp ||
       slice.standaloneReduceOp || slice.maskedStandaloneReduceOp ||
       slice.maccOp || slice.wideningMAccOp || slice.wideningDotReduceOp ||
       hasProductHead(slice) || slice.maskedWideningDotReduceOp ||
       slice.compareOp ||
       isWideningConversion || isDequantization))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route supports only "
        "field0 load/field1 load/segment2_store memory movement in this "
        "slice");
  if (hasSegmentedMemory && !isSegment2DeinterleaveUnitStore &&
      !isSegment2InterleaveUnitLoad && !isComputedMaskSegment2LoadLike &&
      !isComputedMaskSegment2StoreLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 route requires either deinterleave "
        "segment2_load, interleave segment2_store, or computed-mask "
        "masked_segment2_load/masked_segment2_store typed body structure");
  if (isComputedMaskSegment2LoadLike &&
      genericMaskedSegment2Loads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires "
        "exactly one tcrv_rvv.masked_segment2_load op");
  if (!isComputedMaskSegment2LoadLike &&
      !genericMaskedSegment2Loads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked_segment2_load is supported only for "
        "computed_masked_segment2_load_unit_store or "
        "runtime_scalar_cmp_masked_segment2_load_unit_store");
  if (isComputedMaskSegment2StoreLike &&
      genericMaskedSegment2Stores.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store route requires "
        "exactly one tcrv_rvv.masked_segment2_store op");
 if (!isComputedMaskSegment2StoreLike &&
      !genericMaskedSegment2Stores.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked_segment2_store is supported only for "
        "computed_masked_segment2_store_unit_load or "
        "runtime_scalar_cmp_masked_segment2_store_unit_load or "
        "computed_masked_segment2_update_unit_load");
  if (isComputedMaskSegment2LoadLike && storeCount != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires "
        "exactly two unit-stride tcrv_rvv.store ops for field0 and field1");
  if (isComputedMaskSegment2LoadUnitStore && genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 load route requires "
        "exactly four tcrv_rvv.load ops for compare lhs, compare rhs, "
        "field0 old passthrough, and field1 old passthrough");
  if (isRuntimeScalarComputedMaskSegment2LoadUnitStore &&
      genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask segment2 load route "
        "requires exactly three tcrv_rvv.load ops for compare lhs, field0 "
        "old passthrough, and field1 old passthrough; compare rhs must be "
        "tcrv_rvv.splat");
  if (isComputedMaskSegment2StoreLike &&
      !isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 store route requires "
        "exactly four tcrv_rvv.load ops for compare lhs, compare rhs, "
        "field0 payload, and field1 payload");
  if (isRuntimeScalarComputedMaskSegment2StoreUnitLoad &&
      genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask segment2 store route "
        "requires exactly three tcrv_rvv.load ops for compare lhs, field0 "
        "payload, and field1 payload; compare rhs must be tcrv_rvv.splat");
  if (isSegment2DeinterleaveUnitStore && genericSegment2Loads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires exactly "
        "one tcrv_rvv.segment2_load op");
  if (isSegment2DeinterleaveUnitStore &&
      (!slice.field0MoveOp || !slice.field1MoveOp))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires exactly one "
        "field0 move and one field1 move from tcrv_rvv.segment2_load");
  if (isSegment2DeinterleaveUnitStore && storeCount != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 deinterleave route requires exactly two "
        "unit-stride tcrv_rvv.store ops for field0 and field1");
  if (isSegment2InterleaveUnitLoad && genericSegment2Stores.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires exactly one "
        "tcrv_rvv.segment2_store op");
  if (isSegment2InterleaveUnitLoad && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV segment2 interleave route requires exactly two "
        "unit-stride tcrv_rvv.load ops for field0 and field1");
  if (isMaskedUnitLoadStore && slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime-mask memory route must not contain "
        "tcrv_rvv.compare; computed masks require "
        "computed_masked_unit_load_store");
  if (isMaskedUnitStore && slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime-mask store route must not contain "
        "tcrv_rvv.compare; this slice requires explicit mask_load authority");
  if (isComputedMaskUnitLoadStore && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask memory route requires one "
        "tcrv_rvv.compare producer before tcrv_rvv.masked_load");
  if (isRuntimeScalarComputedMaskStore && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask store route requires "
        "one tcrv_rvv.compare producer before tcrv_rvv.masked_store");
  if (isRuntimeScalarComputedMaskLoadStore && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask load-store route "
        "requires one tcrv_rvv.compare producer before "
        "tcrv_rvv.masked_load");
  if (isComputedMaskStridedStore && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route requires one "
        "tcrv_rvv.compare producer before tcrv_rvv.masked_strided_store");
  if (isComputedMaskStridedLoadUnitStore && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-load route requires one "
        "tcrv_rvv.compare producer before tcrv_rvv.masked_strided_load");
  if (isComputedMaskIndexedGatherLike && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed gather-load route "
        "requires one tcrv_rvv.compare producer before "
        "tcrv_rvv.masked_indexed_load");
  if (isComputedMaskIndexedScatterLike && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed scatter-store route "
        "requires one tcrv_rvv.compare producer before "
        "tcrv_rvv.masked_indexed_store");
  if ((isComputedMaskSegment2LoadLike ||
       isComputedMaskSegment2StoreLike) &&
      !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 route requires one "
        "tcrv_rvv.compare producer before masked segment2 memory movement");
  if ((isMaskedUnitLoadStore || isMaskedUnitStore) &&
      genericMaskLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly one "
        "tcrv_rvv.mask_load op");
  if (isComputedMaskUnitLoadStore && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask memory route must not consume "
        "tcrv_rvv.mask_load; the mask must be produced by tcrv_rvv.compare");
  if (isRuntimeScalarComputedMaskStore && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask store route must not "
        "consume tcrv_rvv.mask_load; the mask must be produced by "
        "tcrv_rvv.compare");
  if (isRuntimeScalarComputedMaskLoadStore && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask load-store route "
        "must not consume tcrv_rvv.mask_load; the mask must be produced by "
        "tcrv_rvv.compare");
  if (isComputedMaskStridedStore && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route must not "
        "consume tcrv_rvv.mask_load; the mask must be produced by "
        "tcrv_rvv.compare");
  if (isComputedMaskStridedLoadUnitStore && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-load route must not "
        "consume tcrv_rvv.mask_load; the mask must be produced by "
        "tcrv_rvv.compare");
  if (isComputedMaskIndexedGatherLike && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed gather-load route must not "
        "consume tcrv_rvv.mask_load; the mask must be produced by "
        "tcrv_rvv.compare");
  if (isComputedMaskIndexedScatterLike && !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed scatter-store route must not "
        "consume tcrv_rvv.mask_load; the mask must be produced by "
        "tcrv_rvv.compare");
  if ((isComputedMaskSegment2LoadLike ||
       isComputedMaskSegment2StoreLike) &&
      !genericMaskLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 route must not "
        "consume tcrv_rvv.mask_load; the mask must be produced by "
        "tcrv_rvv.compare");
  if (isMaskedUnitLoadStore && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly one "
        "tcrv_rvv.load op for old destination; source must be "
        "tcrv_rvv.masked_load");
  if (isMaskedUnitStore && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked store route requires exactly one "
        "tcrv_rvv.load op for the payload source");
  if (isRuntimeScalarComputedMaskStore && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask store route requires "
        "exactly two tcrv_rvv.load ops for lhs and active payload source");
  if (isRuntimeScalarComputedMaskLoadStore && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask load-store route "
        "requires exactly two tcrv_rvv.load ops for lhs and old destination; "
        "active source must be tcrv_rvv.masked_load");
  if (isComputedMaskUnitLoadStore && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask memory route requires exactly "
        "three tcrv_rvv.load ops for compare lhs, compare rhs, and old "
        "destination; active source must be tcrv_rvv.masked_load");
  if (isComputedMaskStridedStore && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route requires "
        "exactly three unit-stride tcrv_rvv.load ops for compare lhs, "
        "compare rhs, and active source");
  if (isComputedMaskStridedLoadUnitStore && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-load route requires "
        "exactly three unit-stride tcrv_rvv.load ops for compare lhs, "
        "compare rhs, and old destination passthrough");
  if (isComputedMaskIndexedGatherLoadUnitStore && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed gather-load route requires "
        "exactly three unit-stride tcrv_rvv.load ops for compare lhs, "
        "compare rhs, and old destination passthrough");
  if (isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
      genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask indexed gather-load "
        "route requires exactly two unit-stride tcrv_rvv.load ops for lhs "
        "and old destination passthrough; compare rhs must be tcrv_rvv.splat");
  if (isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
      genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask indexed "
        "scatter-store route requires exactly two unit-stride tcrv_rvv.load "
        "ops for lhs and active payload source; compare rhs must be "
        "tcrv_rvv.splat");
  if (isComputedMaskIndexedScatterStoreUnitLoad && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed scatter-store route requires "
        "exactly three unit-stride tcrv_rvv.load ops for compare lhs, "
        "compare rhs, and active payload source");
  if (isComputedMaskStridedStore && !genericStridedLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route must not load "
        "the old destination; false lanes are preserved by masked_strided_store");
  if (isComputedMaskStridedLoadUnitStore && !genericStridedLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-load route must use "
        "tcrv_rvv.masked_strided_load, not tcrv_rvv.strided_load");
  if (isComputedMaskIndexedGatherLike &&
      !genericStridedLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed gather-load route must use "
        "tcrv_rvv.masked_indexed_load, not tcrv_rvv.strided_load");
  if (isComputedMaskIndexedScatterLike &&
      !genericStridedLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed scatter-store route must use "
        "tcrv_rvv.masked_indexed_store, not tcrv_rvv.strided_load");
  if ((isComputedMaskSegment2LoadLike ||
       isComputedMaskSegment2StoreLike) &&
      !genericStridedLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask segment2 route must use unit-stride "
        "segment2 memory movement, not tcrv_rvv.strided_load");
  if (isMaskedUnitStore && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked store route must use "
        "tcrv_rvv.masked_store, not unit-stride tcrv_rvv.store");
  if (isRuntimeScalarComputedMaskStore && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask store route must use "
        "tcrv_rvv.masked_store, not unit-stride tcrv_rvv.store");
  if (isRuntimeScalarComputedMaskLoadStore && storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask load-store route "
        "requires exactly one unit-stride tcrv_rvv.store op");
  if (isMaskedUnitStore && !slice.maskedStoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked store route requires exactly one "
        "tcrv_rvv.masked_store op");
  if (isRuntimeScalarComputedMaskStore && !slice.maskedStoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask store route requires "
        "exactly one tcrv_rvv.masked_store op");
  if ((isMaskedUnitLoadStore || isComputedMaskUnitLoadStore ||
       isRuntimeScalarComputedMaskLoadStore ||
       isComputedMaskStridedLoadUnitStore ||
       isComputedMaskIndexedGatherLike) &&
      storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly one "
        "unit-stride tcrv_rvv.store op");
  if ((isMaskedUnitLoadStore || isComputedMaskUnitLoadStore ||
       isRuntimeScalarComputedMaskLoadStore) &&
      genericMaskedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked memory route requires exactly one "
        "tcrv_rvv.masked_load op");
  if ((!isMaskedUnitLoadStore && !isComputedMaskUnitLoadStore &&
       !isRuntimeScalarComputedMaskLoadStore) &&
      !genericMaskedLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked_load is supported only for "
        "masked_unit_load_store, computed_masked_unit_load_store, and "
        "runtime_scalar_cmp_masked_load_store");
  if (isComputedMaskStridedLoadUnitStore &&
      genericMaskedStridedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-load route requires "
        "exactly one tcrv_rvv.masked_strided_load op");
  if (!isComputedMaskStridedLoadUnitStore &&
      !genericMaskedStridedLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked_strided_load is supported only for "
        "computed_masked_strided_load_unit_store");
  if (isComputedMaskIndexedGatherLike &&
      genericMaskedIndexedLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed gather-load route requires "
        "exactly one tcrv_rvv.masked_indexed_load op");
  if (!isComputedMaskIndexedGatherLike &&
      !genericMaskedIndexedLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked_indexed_load is supported only for "
        "computed_masked_indexed_gather_load_unit_store");
  if (isComputedMaskIndexedScatterLike &&
      genericMaskedIndexedStores.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed scatter-store route requires "
        "exactly one tcrv_rvv.masked_indexed_store op");
  if (!isComputedMaskIndexedScatterLike &&
      !genericMaskedIndexedStores.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked_indexed_store is supported only for "
        "computed_masked_indexed_scatter_store_unit_load");
  if (isComputedMaskStridedStore && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route must use "
        "tcrv_rvv.strided_store, not unit-stride tcrv_rvv.store");
  if (isComputedMaskStridedLoadUnitStore && stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-load route must use "
        "unit-stride tcrv_rvv.store, not tcrv_rvv.strided_store");
  if (isComputedMaskIndexedGatherLike && stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed gather-load route must use "
        "unit-stride tcrv_rvv.store, not tcrv_rvv.strided_store");
  if (isComputedMaskIndexedScatterLike && stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed scatter-store route must use "
        "tcrv_rvv.masked_indexed_store, not tcrv_rvv.strided_store");
  if (isComputedMaskIndexedScatterLike &&
      !slice.maskedIndexedStoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask indexed scatter-store route requires "
        "exactly one tcrv_rvv.masked_indexed_store op");
  if (isComputedMaskStridedStore && stridedStoreCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route must use "
        "tcrv_rvv.masked_strided_store, not tcrv_rvv.strided_store");
  if (isComputedMaskStridedStore && !slice.maskedStridedStoreOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-store route requires "
        "exactly one tcrv_rvv.masked_strided_store op");
  if (isMAccAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV multiply-accumulate route requires explicit "
        "vector lhs, rhs, and accumulator loads; broadcast/splat macc is not "
        "in this bounded slice");
  if (isComputedMaskedMAccAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask multiply-accumulate route requires "
        "explicit compare lhs/rhs, payload lhs/rhs, and accumulator loads; "
        "broadcast/splat macc is not in this bounded slice");
  if (isWideningMAccAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening multiply-accumulate route requires "
        "explicit vector lhs, rhs, and accumulator loads; broadcast/splat "
        "macc is not in this bounded slice");
  if (isWideningDotReduceAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening dot-product reduction route requires "
        "explicit lhs/rhs unit-stride vector loads and an accumulator seed "
        "runtime ABI boundary; broadcast/splat dot-reduction is not in this "
        "bounded slice");
  if (isWideningProductReductionChain &&
      (!genericBroadcastLoads.empty() ||
       (hasScalarBroadcast && !isWideningProductReduceDequantClamp)))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV low-precision product-reduction route requires "
        "explicit lhs/rhs i8 source vector loads and an accumulator seed "
        "runtime ABI boundary; broadcast/splat product-reduction is not in "
        "this bounded slice");
  if (isStridedInputWideningDotReduceAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route requires explicit lhs/rhs strided vector loads and an "
        "accumulator seed runtime ABI boundary; broadcast/splat "
        "dot-reduction is not in this bounded slice");
  if (isComputedMaskWideningDotReduceAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask widening dot-product reduction "
        "route requires explicit compare lhs/rhs and dot lhs/rhs unit-stride "
        "loads; broadcast/splat dot-reduction is not in this bounded slice");
  if (isComputedMaskStridedInputWideningDotReduceAdd && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-input widening "
        "dot-product reduction route requires explicit compare lhs/rhs "
        "unit-stride loads and dot lhs/rhs strided loads; broadcast/splat "
        "dot-reduction is not in this bounded slice");
  if (isWideningConversion && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route does not consume an "
        "RHS broadcast or scalar splat");
  if (isDequantization && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV dequantization route does not consume an RHS "
        "broadcast or scalar splat");
  if (isMAccAdd && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV multiply-accumulate route requires exactly three "
        "tcrv_rvv.load ops for lhs, rhs, and accumulator-input-buffer");
  if (isScalarBroadcastMAccAdd && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV scalar-broadcast multiply-accumulate composition "
        "route requires exactly two tcrv_rvv.load ops for lhs and "
        "accumulator-input-buffer plus one RHS scalar splat");
  if (isComputedMaskedMAccAdd && genericLoads.size() != 5)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask multiply-accumulate route requires "
        "exactly five tcrv_rvv.load ops for compare lhs, compare rhs, "
        "payload lhs, payload rhs, and accumulator-input-buffer");
  if (isRuntimeScalarComputedMaskedMAccAdd && genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask "
        "multiply-accumulate route requires exactly four tcrv_rvv.load ops "
        "for compare lhs, payload lhs, payload rhs, and "
        "accumulator-input-buffer plus one runtime scalar splat threshold");
  if (isWideningMAccAdd && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening multiply-accumulate route requires "
        "exactly three tcrv_rvv.load ops for lhs, rhs, and "
        "accumulator-input-buffer");
  if (isWideningDotReduceAdd && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening dot-product reduction route requires "
        "exactly two tcrv_rvv.load ops for lhs and rhs; the accumulator seed "
        "is a scalar runtime ABI boundary");
  if (isWideningProductReductionChain && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV low-precision product-reduction route requires "
        "exactly two tcrv_rvv.load ops for i8 lhs and rhs; the accumulator "
        "seed is a scalar runtime ABI boundary");
  if (isStandaloneReduction && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV standalone reduction route requires exactly one "
        "tcrv_rvv.load op for lhs; the accumulator seed is a scalar runtime "
        "ABI boundary");
  if (isComputedMaskStandaloneReduction && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask standalone reduction route requires "
        "exactly three tcrv_rvv.load ops for compare lhs, compare rhs, and "
        "source input; the accumulator seed is a scalar runtime ABI boundary");
  if (isRuntimeScalarComputedMaskStandaloneReduction &&
      genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask standalone "
        "reduction route requires exactly two tcrv_rvv.load ops for compare "
        "lhs and source input plus one runtime scalar splat threshold; the "
        "accumulator seed is a scalar runtime ABI boundary");
  if (isStridedInputWideningDotReduceAdd && genericLoads.size() != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route must use tcrv_rvv.strided_load for lhs/rhs sources, not "
        "unit-stride tcrv_rvv.load");
  if (isStridedInputWideningDotReduceAdd && genericStridedLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided-input widening dot-product reduction "
        "route requires exactly two tcrv_rvv.strided_load ops for lhs and rhs; "
        "the accumulator seed is a scalar runtime ABI boundary");
  if (isComputedMaskWideningDotReduceAdd && genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask widening dot-product reduction "
        "route requires exactly four tcrv_rvv.load ops for compare lhs, "
        "compare rhs, dot lhs, and dot rhs; the accumulator seed is a scalar "
        "runtime ABI boundary");
  if (isComputedMaskStridedInputWideningDotReduceAdd &&
      genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-input widening "
        "dot-product reduction route requires exactly two unit-stride "
        "tcrv_rvv.load ops for compare lhs and compare rhs");
  if (isComputedMaskStridedInputWideningDotReduceAdd &&
      genericStridedLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask strided-input widening "
        "dot-product reduction route requires exactly two tcrv_rvv.strided_load "
        "ops for dot lhs and dot rhs; the accumulator seed is a scalar "
        "runtime ABI boundary");
  if (isComputedMaskSelect && genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask select route requires exactly four "
        "tcrv_rvv.load ops for compare lhs, compare rhs, true value, and "
        "false value");
  if (isRuntimeScalarCompareSelect && genericLoads.size() != 3)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar compare/select route requires "
        "exactly three tcrv_rvv.load ops for lhs, true value, and false value");
  if (isRuntimeScalarDualCompareMaskAndSelect && genericLoads.size() != 4)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar dual-compare mask-and select "
        "route requires exactly four tcrv_rvv.load ops for compare lhs A, "
        "compare lhs B, true value, and false value");
  if (isF32ClampSelect && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV f32 clamp/select route requires exactly one "
        "tcrv_rvv.load op for the f32 input plus lower and upper runtime "
        "scalar splats");
  if (isDequantClampF32Epilogue && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV dequant-clamp f32 epilogue route requires "
        "exactly one tcrv_rvv.load op for the i32 input plus runtime scale "
        "dequantize and lower/upper runtime scalar splats");
  if (isRuntimeScalarComputedMaskStore && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar computed-mask store route requires "
        "exactly two tcrv_rvv.load ops for lhs and payload source");
  if (hasScalarBroadcast &&
      (!slice.arithmeticOp ||
       (slice.arithmeticKind != RVVSelectedBodyOperationKind::Add &&
        slice.arithmeticKind != RVVSelectedBodyOperationKind::Sub &&
        slice.arithmeticKind != RVVSelectedBodyOperationKind::Mul &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarDualCompareMaskAndSelect &&
        slice.arithmeticKind != RVVSelectedBodyOperationKind::F32ClampSelect &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::DequantClampF32Epilogue &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                WideningProductReduceDequantClampF32 &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskLoadStore &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedGatherLoadUnitStore &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskSegment2LoadUnitStore &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskSegment2StoreUnitLoad &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskedMAccAdd &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskStandaloneReduceAdd &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskStandaloneReduceMin &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskStandaloneReduceMax &&
        slice.arithmeticKind !=
            RVVSelectedBodyOperationKind::RuntimeScalarSplatStore)))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV scalar-broadcast route currently requires "
        "either pure runtime_scalar_splat_store splat/store structure, exactly "
        "one tcrv_rvv.binary {kind = \"add\", \"sub\", or \"mul\"} compute "
        "op, a bounded runtime scalar compare/select structure, or a bounded "
        "runtime scalar computed-mask masked-store, masked-load-store, "
        "indexed gather/scatter, or segment2 load/store structure");
  if (isWideningConversion && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route requires exactly one "
        "tcrv_rvv.load source op");
  if (isDequantization && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV dequantization route requires exactly one "
        "tcrv_rvv.load source op");
  if (isWideningConversion &&
      (slice.compareOp || slice.maskedBinaryOp || slice.selectOp ||
       slice.reduceOp || slice.maccOp || slice.wideningMAccOp ||
       slice.wideningDotReduceOp))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV widening conversion route cannot mix "
        "compare/select/masked/reduce/macc compute ops");
  if (isDequantization &&
      (slice.compareOp || slice.maskedBinaryOp || slice.selectOp ||
       slice.reduceOp || slice.maccOp || slice.wideningMAccOp ||
       hasProductHead(slice) || slice.wideningDotReduceOp ||
       slice.maskedWideningDotReduceOp || slice.wideningConvertOp))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV dequantization route cannot mix "
        "compare/select/masked/reduce/macc/widening compute ops");
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      !isMAccAdd &&
      !isScalarBroadcastMAccAdd &&
      !isComputedMaskedMAccAdd &&
      !isRuntimeScalarComputedMaskedMAccAdd &&
      !isWideningMAccAdd &&
      !isWideningProduct &&
      !isWideningProductReductionChain &&
      !isWideningDotReduceAdd &&
      !isStandaloneReduction &&
      !isComputedMaskStandaloneReduction &&
      !isRuntimeScalarComputedMaskStandaloneReduction &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskSelect &&
      !isRuntimeScalarCompareSelect &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isF32ClampSelect &&
      !isDequantClampF32Epilogue &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isMaskedUnitLoadStore &&
      !isMaskedUnitStore &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedScatterLike &&
      !hasRHSBroadcastLike &&
      !isWideningConversion &&
      !isDequantization &&
      genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV vector-load route requires exactly two "
        "tcrv_rvv.load ops");
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      !isMAccAdd &&
      !isScalarBroadcastMAccAdd &&
      !isComputedMaskedMAccAdd &&
      !isRuntimeScalarComputedMaskedMAccAdd &&
      !isWideningMAccAdd &&
      !isWideningProduct &&
      !isWideningProductReductionChain &&
      !isWideningDotReduceAdd &&
      !isStandaloneReduction &&
      !isComputedMaskStandaloneReduction &&
      !isRuntimeScalarComputedMaskStandaloneReduction &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskSelect &&
      !isRuntimeScalarCompareSelect &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isF32ClampSelect &&
      !isDequantClampF32Epilogue &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isMaskedUnitLoadStore &&
      !isMaskedUnitStore &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedScatterLike &&
      !isDequantization &&
      !genericBroadcastLoads.empty() && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV broadcast route requires exactly one "
        "tcrv_rvv.load op and one tcrv_rvv.broadcast_load op");
  if (!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
      !isMAccAdd &&
      !isScalarBroadcastMAccAdd &&
      !isComputedMaskedMAccAdd &&
      !isRuntimeScalarComputedMaskedMAccAdd &&
      !isWideningMAccAdd &&
      !isWideningProduct &&
      !isWideningProductReductionChain &&
      !isWideningDotReduceAdd &&
      !isStandaloneReduction &&
      !isComputedMaskStandaloneReduction &&
      !isRuntimeScalarComputedMaskStandaloneReduction &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskSelect &&
      !isRuntimeScalarCompareSelect &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isF32ClampSelect &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isMaskedUnitLoadStore &&
      !isMaskedUnitStore &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedScatterLike &&
      !isDequantization &&
      hasScalarBroadcast && !isRuntimeScalarSplatStore &&
      genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV scalar-broadcast route requires exactly one "
        "tcrv_rvv.load op and one tcrv_rvv.splat op");
  if (isRuntimeScalarSplatStore && !genericLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar splat-store route must not contain "
        "lhs loads or binary fallback");
  if (((!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
        !isMaskedUnitStore && !isRuntimeScalarComputedMaskStore &&
        !isRuntimeScalarDualCompareMaskAndSelect &&
        !isF32ClampSelect &&
        !isDequantClampF32Epilogue &&
        !isRuntimeScalarComputedMaskLoadStore) ||
       isComputedMaskStridedLoadUnitStore ||
       isComputedMaskIndexedGatherLike) &&
      !slice.genericStore)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one "
        "tcrv_rvv.store op");
  if (!slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one supported "
        "tcrv_rvv.binary, tcrv_rvv.select, tcrv_rvv.reduce, tcrv_rvv.macc, "
        "tcrv_rvv.masked_macc, "
        "tcrv_rvv.standalone_reduce, tcrv_rvv.masked_standalone_reduce, "
        "tcrv_rvv.widening_macc, "
        "tcrv_rvv.widening_product, tcrv_rvv.widening_dot_reduce, "
        "tcrv_rvv.masked_widening_dot_reduce, tcrv_rvv.widening_convert, "
        "tcrv_rvv.dequantize, tcrv_rvv.move, tcrv_rvv.masked_move, "
        "tcrv_rvv.masked_load, tcrv_rvv.masked_strided_load, "
        "tcrv_rvv.masked_indexed_load, tcrv_rvv.masked_segment2_load, "
        "tcrv_rvv.masked_segment2_store, tcrv_rvv.masked_store, or "
        "tcrv_rvv.masked_strided_store op");
  if (((!hasStridedMemory && !hasIndexedMemory && !hasSegmentedMemory &&
        !isMaskedUnitStore && !isRuntimeScalarComputedMaskStore &&
        !isRuntimeScalarDualCompareMaskAndSelect &&
        !isF32ClampSelect &&
        !isDequantClampF32Epilogue &&
        !isRuntimeScalarComputedMaskLoadStore) ||
       isComputedMaskStridedLoadUnitStore ||
       isComputedMaskIndexedGatherLike) &&
      storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one tcrv_rvv.store "
        "op");
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedGatherLike && storeCount != 0)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV strided route must use tcrv_rvv.strided_store "
        "instead of tcrv_rvv.store");
  if ((isCompareSelect || isComputedMaskSelect ||
       isRuntimeScalarCompareSelect ||
       isRuntimeScalarDualCompareMaskAndSelect ||
       isF32ClampSelect ||
       isDequantClampF32Epilogue ||
       isWideningProductReduceDequantClamp ||
       isRuntimeScalarComputedMaskStore ||
       isRuntimeScalarComputedMaskLoadStore ||
       isMaskedArithmetic ||
       isComputedMaskedMAccAdd ||
       isRuntimeScalarComputedMaskedMAccAdd ||
       isComputedMaskUnitLoadStore ||
       isComputedMaskStridedStore || isComputedMaskStridedLoadUnitStore ||
       isComputedMaskIndexedGatherLike ||
       isComputedMaskIndexedScatterLike ||
       isComputedMaskSegment2LoadLike ||
       isComputedMaskSegment2StoreLike ||
       isComputedMaskWideningDotReduceAdd ||
       isComputedMaskStridedInputWideningDotReduceAdd ||
       isComputedMaskStandaloneReduction ||
       isRuntimeScalarComputedMaskStandaloneReduction) &&
      !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
      "bounded generic RVV mask-consuming route requires one "
      "tcrv_rvv.compare op before the mask-consuming compute op");
  if (isRuntimeScalarDualCompareMaskAndSelect &&
      (!slice.secondaryCompareOp || !slice.maskAndOp))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV runtime scalar dual-compare mask-and select "
        "route requires two compare ops and one tcrv_rvv.mask_and op");
  if (!isCompareSelect && !isComputedMaskSelect &&
      !isRuntimeScalarCompareSelect &&
      !isRuntimeScalarDualCompareMaskAndSelect &&
      !isF32ClampSelect &&
      !isDequantClampF32Epilogue &&
      !isWideningProductReduceDequantClamp &&
      !isRuntimeScalarComputedMaskStore &&
      !isRuntimeScalarComputedMaskLoadStore &&
      !isMaskedArithmetic &&
      !isComputedMaskedMAccAdd &&
      !isRuntimeScalarComputedMaskedMAccAdd &&
      !isComputedMaskUnitLoadStore &&
      !isComputedMaskStridedStore && !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedGatherLike &&
      !isComputedMaskIndexedScatterLike &&
      !isComputedMaskSegment2LoadLike &&
      !isComputedMaskSegment2StoreLike &&
      !isComputedMaskWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskStandaloneReduction &&
      !isRuntimeScalarComputedMaskStandaloneReduction &&
      slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV non-mask route does not support a standalone "
        "tcrv_rvv.compare op");
  if (!isRuntimeScalarDualCompareMaskAndSelect && slice.maskAndOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV non-mask-composition route does not support a "
        "standalone tcrv_rvv.mask_and op");
  if ((isCompareSelect || isComputedMaskSelect) && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV compare/select route requires an explicit RHS "
        "vector load; broadcast/splat compare/select is not in this bounded "
        "slice");
  if (isMaskedArithmetic && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked elementwise route requires an explicit "
        "RHS vector load; broadcast/splat masked elementwise is not in this "
        "bounded slice");
  if (isReduction && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV reduction route requires explicit vector input "
        "and accumulator loads; broadcast/splat reduction is not in this "
        "bounded slice");
  if (isStandaloneReduction && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV standalone reduction route requires explicit "
        "input load and accumulator-input-buffer scalar seed boundary; "
        "broadcast/splat reduction is not in this bounded slice");
  if (isComputedMaskStandaloneReduction && hasRHSBroadcastLike)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV computed-mask standalone reduction route requires "
        "explicit compare lhs/rhs and source unit-stride loads; "
        "broadcast/splat reduction is not in this bounded slice");
  const unsigned expectedRVVOps =
      isRuntimeScalarSplatStore
          ? 7
      : isDequantization
          ? 9
      : isWideningConversion
          ? 8
      : isRuntimeScalarComputedMaskStandaloneReduction
          ? 14
      : isComputedMaskStandaloneReduction
          ? 14
      : isStandaloneReduction
          ? 9
      : isSegment2DeinterleaveUnitStore
          ? 11
      : isSegment2InterleaveUnitLoad
          ? 9
      : isWideningDotReduceAdd
          ? 11
      : isWideningProductReduceDequantClamp
          ? 23
      : isWideningProductReduceDequantize
          ? 15
      : isWideningProductReductionChain
          ? 12
      : isStridedInputWideningDotReduceAdd
          ? 13
      : isComputedMaskWideningDotReduceAdd
          ? 16
      : isComputedMaskStridedInputWideningDotReduceAdd
          ? 18
      : isMAccAdd
          ? 12
      : isScalarBroadcastMAccAdd
          ? 12
      : isComputedMaskedMAccAdd
          ? 17
      : isRuntimeScalarComputedMaskedMAccAdd
          ? 17
      : isWideningMAccAdd
          ? 12
      : isComputedMaskUnitLoadStore
          ? 13
      : isComputedMaskStridedStore
          ? 13
      : isComputedMaskStridedLoadUnitStore
          ? 14
      : isComputedMaskIndexedGatherLoadUnitStore
          ? 15
      : isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore
          ? 15
      : isComputedMaskIndexedScatterStoreUnitLoad
          ? 14
      : isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad
          ? 14
      : isRuntimeScalarComputedMaskSegment2LoadUnitStore
          ? 16
      : isComputedMaskSegment2LoadUnitStore
          ? 16
      : isComputedMaskSegment2UpdateUnitLoad
          ? 15
      : isComputedMaskSegment2StoreUnitLoad
          ? 14
      : isRuntimeScalarComputedMaskSegment2StoreUnitLoad
          ? 14
      : isComputedMaskSelect
          ? 15
      : isRuntimeScalarDualCompareMaskAndSelect
          ? 21
      : isDequantClampF32Epilogue
          ? 17
      : isF32ClampSelect
          ? 15
      : isRuntimeScalarCompareSelect
          ? 15
      : isRuntimeScalarComputedMaskStore
          ? 12
      : isRuntimeScalarComputedMaskLoadStore
          ? 13
      : isMaskedUnitStore
          ? 9
      : isMaskedUnitLoadStore
          ? 10
          : (isIndexedGatherUnitStore || isIndexedScatterUnitLoad
                 ? 10
                 : (isStridedLoadUnitStore || isUnitLoadStridedStore
                        ? 9
          : (hasStridedMemory
                 ? 13
                 : ((isCompareSelect || isMaskedArithmetic) ? 11 : 10))));
  // The isWideningProductReduceDequantize/Clamp base counts (15/23) include the
  // legacy two-scope gearbox_cross_region_handoff op. A single-scope typed body
  // (Stage 3 flip) drops the handoff (and its markers + consumer with_vl), so the
  // expected op count is one fewer when no handoff op is present.
  const unsigned expectedHandoffAdjust =
      (isWideningProductReduceDequantGearboxRoute &&
       !slice.gearboxCrossRegionHandoffOp)
          ? 1u
          : 0u;
  const unsigned expectedRVVOpsWithMarkers =
      expectedRVVOps - expectedHandoffAdjust +
      slice.vsetvlRegionMarkers.size() +
      (slice.gearboxConsumerWithVL ? 1 : 0);
  if (rvvOpCount != expectedRVVOpsWithMarkers)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route supports only runtime_abi_value/"
        "runtime_abi_value/runtime_abi_value plus optional runtime_abi_value "
        "and optional strided runtime_abi_value/runtime_abi_value/"
        "runtime_abi_value, "
        "setvl/with_vl, and generic load/broadcast_load/splat/strided_load/"
        "index_load/indexed_load/mask_load/segment2_load/segment2_store/"
        "binary/compare/select/masked_binary/reduce/macc/masked_macc/"
        "standalone_reduce/masked_standalone_reduce/widening_product/"
        "widening_dot_reduce/widening_convert/"
        "gearbox_cross_region_handoff/dequantize/vsetvl_region_marker/move/"
        "masked_move/masked_load/masked_strided_load/"
        "masked_indexed_load/masked_indexed_store/masked_store/"
        "masked_segment2_store/masked_strided_store/store/strided_store body "
        "structure");

  for (tcrv::rvv::LoadOp load : genericLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::RHSInputBuffer,
             support::RuntimeABIParameterRole::SourceInputBuffer,
             support::RuntimeABIParameterRole::TrueValueInputBuffer,
             support::RuntimeABIParameterRole::FalseValueInputBuffer,
             support::RuntimeABIParameterRole::OutputBuffer,
             support::RuntimeABIParameterRole::AccumulatorInputBuffer,
             support::RuntimeABIParameterRole::DotLHSInputBuffer,
             support::RuntimeABIParameterRole::DotRHSInputBuffer,
             support::RuntimeABIParameterRole::SegmentField0InputBuffer,
             support::RuntimeABIParameterRole::SegmentField1InputBuffer,
             support::RuntimeABIParameterRole::SegmentField0OutputBuffer,
             support::RuntimeABIParameterRole::SegmentField1OutputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error = assignRVVGenericLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::BroadcastLoadOp broadcast : genericBroadcastLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            broadcast.getBuffer(), "tcrv_rvv.broadcast_load buffer operand",
            {support::RuntimeABIParameterRole::RHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericBroadcastBinding(slice, broadcast, *parameter))
      return error;
  }
  for (tcrv::rvv::SplatOp splat : genericScalarSplats) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            splat.getScalar(), "tcrv_rvv.splat scalar operand",
            {support::RuntimeABIParameterRole::RHSScalarValue,
             support::RuntimeABIParameterRole::RHSSecondaryScalarValue,
             support::RuntimeABIParameterRole::LowerBoundScalarValue,
             support::RuntimeABIParameterRole::UpperBoundScalarValue});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericScalarSplatBinding(slice, splat, *parameter))
      return error;
  }
  if (isScalarBroadcastMAccAdd)
    slice.memoryForm = RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc;
  for (tcrv::rvv::StridedLoadOp load : genericStridedLoads) {
    llvm::Expected<support::RuntimeABIParameter> bufferParameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.strided_load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::RHSInputBuffer,
             support::RuntimeABIParameterRole::DotLHSInputBuffer,
             support::RuntimeABIParameterRole::DotRHSInputBuffer,
             support::RuntimeABIParameterRole::SourceInputBuffer,
             support::RuntimeABIParameterRole::OutputBuffer});
    if (!bufferParameter)
      return bufferParameter.takeError();
    llvm::Expected<support::RuntimeABIParameter> strideParameter =
        getRuntimeABIParameterBindingFromValue(
            load.getStride(), "tcrv_rvv.strided_load stride operand",
            {support::RuntimeABIParameterRole::LHSInputStride,
             support::RuntimeABIParameterRole::RHSInputStride,
             support::RuntimeABIParameterRole::SourceByteStride,
             support::RuntimeABIParameterRole::DestinationByteStride,
             support::RuntimeABIParameterRole::OutputStride});
    if (!strideParameter)
      return strideParameter.takeError();
    if (llvm::Error error = assignRVVGenericStridedLoadBinding(
            slice, load, *bufferParameter, *strideParameter))
      return error;
  }
  for (tcrv::rvv::IndexLoadOp load : genericIndexLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getIndex(), "tcrv_rvv.index_load index operand",
            {support::RuntimeABIParameterRole::IndexInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::IndexedLoadOp load : genericIndexedLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getData(), "tcrv_rvv.indexed_load data operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexedLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::IndexedStoreOp store : genericIndexedStores) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            store.getDestination(), "tcrv_rvv.indexed_store destination operand",
            {support::RuntimeABIParameterRole::OutputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericIndexedStoreBinding(slice, store, *parameter))
      return error;
  }
  for (tcrv::rvv::MaskLoadOp load : genericMaskLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getMask(), "tcrv_rvv.mask_load mask operand",
            {support::RuntimeABIParameterRole::MaskInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericMaskLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::MaskedLoadOp load : genericMaskedLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.masked_load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::SourceInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericMaskedLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::MaskedStridedLoadOp load : genericMaskedStridedLoads) {
    llvm::Expected<support::RuntimeABIParameter> bufferParameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.masked_strided_load buffer operand",
            {support::RuntimeABIParameterRole::SourceInputBuffer});
    if (!bufferParameter)
      return bufferParameter.takeError();
    llvm::Expected<support::RuntimeABIParameter> strideParameter =
        getRuntimeABIParameterBindingFromValue(
            load.getStride(), "tcrv_rvv.masked_strided_load stride operand",
            {support::RuntimeABIParameterRole::SourceByteStride});
    if (!strideParameter)
      return strideParameter.takeError();
    if (llvm::Error error = assignRVVGenericMaskedStridedLoadBinding(
            slice, load, *bufferParameter, *strideParameter))
      return error;
  }
  for (tcrv::rvv::MaskedIndexedLoadOp load : genericMaskedIndexedLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getData(), "tcrv_rvv.masked_indexed_load source operand",
            {support::RuntimeABIParameterRole::SourceInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericMaskedIndexedLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::MaskedIndexedStoreOp store : genericMaskedIndexedStores) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            store.getDestination(),
            "tcrv_rvv.masked_indexed_store destination operand",
            {support::RuntimeABIParameterRole::OutputBuffer});
    if (!parameter)
      return parameter.takeError();
    slice.maskedIndexedStore = store;
    slice.maskedIndexedStoreOperation = store.getOperation();
    slice.indexedDestinationBuffer = store.getDestination();
    slice.outABI = *parameter;
  }
  for (tcrv::rvv::MaskedSegment2LoadOp load : genericMaskedSegment2Loads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getSource(), "tcrv_rvv.masked_segment2_load source operand",
            {support::RuntimeABIParameterRole::SourceInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericMaskedSegment2LoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::MaskedSegment2StoreOp store :
       genericMaskedSegment2Stores) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            store.getDestination(),
            "tcrv_rvv.masked_segment2_store destination operand",
            {support::RuntimeABIParameterRole::
                 SegmentInterleavedOutputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error = assignRVVGenericMaskedSegment2StoreBinding(
            slice, store, *parameter))
      return error;
  }
  for (tcrv::rvv::Segment2LoadOp load : genericSegment2Loads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getSource(), "tcrv_rvv.segment2_load source operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericSegment2LoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::Segment2StoreOp store : genericSegment2Stores) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            store.getDestination(),
            "tcrv_rvv.segment2_store destination operand",
            {support::RuntimeABIParameterRole::
                 SegmentInterleavedOutputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericSegment2StoreBinding(slice, store, *parameter))
      return error;
  }
  if (isIndexedGatherUnitStore) {
    if (slice.indexedLoad.getIndices() != slice.indexValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires "
          "tcrv_rvv.indexed_load to consume the index vector produced by "
          "tcrv_rvv.index_load");
    if (slice.indexLoad.getVl() != slice.setvl.getVl() ||
        slice.indexedLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires index_load and "
          "indexed_load to consume the selected !tcrv_rvv.vl token");
  }
  if (isIndexedScatterUnitLoad) {
    if (slice.indexedStore.getIndices() != slice.indexValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires "
          "tcrv_rvv.indexed_store to consume the index vector produced by "
          "tcrv_rvv.index_load");
    if (slice.indexLoad.getVl() != slice.setvl.getVl() ||
        slice.indexedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires index_load and "
          "indexed_store to consume the selected !tcrv_rvv.vl token");
  }
  if (isMaskedUnitLoadStore) {
    if (slice.maskedLoadOp.getMask() != slice.maskValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires "
          "tcrv_rvv.masked_load to consume the mask produced by "
          "tcrv_rvv.mask_load");
    if (slice.maskLoad.getVl() != slice.setvl.getVl() ||
        slice.maskedLoadOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires mask_load and "
          "masked_load to consume the selected !tcrv_rvv.vl token");
  }
  if (isMaskedUnitStore) {
    if (slice.maskedStore.getMask() != slice.maskValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route requires "
          "tcrv_rvv.masked_store to consume the mask produced by "
          "tcrv_rvv.mask_load");
    if (slice.maskLoad.getVl() != slice.setvl.getVl() ||
        slice.maskedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route requires mask_load and "
          "masked_store to consume the selected !tcrv_rvv.vl token");
  }
  if (isComputedMaskUnitLoadStore) {
    if (slice.maskedLoadOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires "
          "tcrv_rvv.masked_load to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedLoadOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires compare "
          "and masked_load to consume the selected !tcrv_rvv.vl token");
  }
  if (isComputedMaskStridedStore) {
    if (slice.maskedStridedStore.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "tcrv_rvv.masked_strided_store to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedStridedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "compare and masked_strided_store to consume the selected "
          "!tcrv_rvv.vl token");
  }
  if (isComputedMaskStridedLoadUnitStore) {
    if (slice.maskedStridedLoadOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route requires "
          "tcrv_rvv.masked_strided_load to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedStridedLoadOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route requires "
          "compare and masked_strided_load to consume the selected "
          "!tcrv_rvv.vl token");
  }
  if (isComputedMaskIndexedGatherLike) {
    if (slice.maskedIndexedLoadOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires tcrv_rvv.masked_indexed_load to consume the mask produced "
          "by tcrv_rvv.compare");
    if (slice.maskedIndexedLoadOp.getIndices() != slice.indexValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires tcrv_rvv.masked_indexed_load to consume the index vector "
          "produced by tcrv_rvv.index_load");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.indexLoad.getVl() != slice.setvl.getVl() ||
        slice.maskedIndexedLoadOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires compare, index_load, and masked_indexed_load to consume "
          "the selected !tcrv_rvv.vl token");
  }
  if (isComputedMaskIndexedScatterLike) {
    if (slice.maskedIndexedStore.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "requires tcrv_rvv.masked_indexed_store to consume the mask produced "
          "by tcrv_rvv.compare");
    if (slice.maskedIndexedStore.getIndices() != slice.indexValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "requires tcrv_rvv.masked_indexed_store to consume the index vector "
          "produced by tcrv_rvv.index_load");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.indexLoad.getVl() != slice.setvl.getVl() ||
        slice.maskedIndexedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "requires compare, index_load, and masked_indexed_store to consume "
          "the selected !tcrv_rvv.vl token");
  }
  if (isComputedMaskSegment2LoadLike) {
    if (slice.maskedSegment2LoadOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route requires "
          "tcrv_rvv.masked_segment2_load to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.maskedSegment2LoadOp.getPassthrough0() !=
            slice.field0PassthroughValue ||
        slice.maskedSegment2LoadOp.getPassthrough1() !=
            slice.field1PassthroughValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route requires "
          "tcrv_rvv.masked_segment2_load to consume field0 and field1 old "
          "destination loads as inactive passthrough vectors");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedSegment2LoadOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route requires "
          "compare and masked_segment2_load to consume the selected "
          "!tcrv_rvv.vl token");
  }
  if (isComputedMaskSegment2StoreLike) {
    if (slice.maskedSegment2Store.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store/update route "
          "requires "
          "tcrv_rvv.masked_segment2_store to consume the mask produced by "
          "tcrv_rvv.compare");
    if (isComputedMaskSegment2UpdateUnitLoad) {
      if (slice.arithmeticKind !=
              RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad ||
          slice.arithmeticLhs != slice.field0LoadedValue ||
          slice.arithmeticRhs != slice.field1LoadedValue ||
          slice.maskedSegment2Store.getField0() != slice.arithmeticResult ||
          slice.maskedSegment2Store.getField1() != slice.field1LoadedValue)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV computed-mask segment2 update route requires "
            "tcrv_rvv.binary {kind = \"add\"} to consume field0/field1 source "
            "loads and masked_segment2_store to consume the add result as "
            "field0 plus the original field1 load");
      if (slice.arithmeticOp->getName().getStringRef() != "tcrv_rvv.binary")
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV computed-mask segment2 update route requires "
            "the selected arithmetic producer to be tcrv_rvv.binary");
    } else if (slice.maskedSegment2Store.getField0() != slice.field0Value ||
               slice.maskedSegment2Store.getField1() != slice.field1Value) {
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store route requires "
          "tcrv_rvv.masked_segment2_store to consume matching field0 and "
          "field1 payload load results");
    }
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        !slice.field0LoadOperation || !slice.field1LoadOperation ||
        slice.maskedSegment2Store.getVl() != slice.setvl.getVl() ||
        (isComputedMaskSegment2UpdateUnitLoad &&
         llvm::cast<tcrv::rvv::BinaryOp>(slice.arithmeticOp).getVl() !=
             slice.setvl.getVl()))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store/update route "
          "requires compare, field0/field1 payload loads, optional add, and "
          "masked_segment2_store to consume the selected !tcrv_rvv.vl token");
  }
  if (isSegment2DeinterleaveUnitStore) {
    if (slice.segment2Load.getVl() != slice.setvl.getVl() ||
        slice.field0MoveOp.getVl() != slice.setvl.getVl() ||
        slice.field1MoveOp.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route requires "
          "segment2_load and both move ops to consume the selected "
          "!tcrv_rvv.vl token");
    if (slice.field0MoveOp.getSource() != slice.segment2Load.getField0() ||
        slice.field1MoveOp.getSource() != slice.segment2Load.getField1())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route requires field0 "
          "and field1 moves to consume matching segment2_load field results");
  }
  if (isSegment2InterleaveUnitLoad) {
    if (slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.rhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.segment2Store.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route requires field0 "
          "load, field1 load, and segment2_store to consume the selected "
          "!tcrv_rvv.vl token");
    if (slice.segment2Store.getField0() != slice.field0Value ||
        slice.segment2Store.getField1() != slice.field1Value)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route requires "
          "segment2_store to consume matching field0 and field1 load results");
  }

  if ((!slice.lhsLoadOperation && !isRuntimeScalarSplatStore) ||
      (!isWideningConversion && !isStridedLoadUnitStore &&
       !isRuntimeScalarSplatStore &&
       !isDequantization &&
       !isUnitLoadStridedStore &&
       !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
       !isMaskedUnitLoadStore &&
       !isMaskedUnitStore &&
       !isComputedMaskUnitLoadStore &&
       !isComputedMaskStridedStore &&
       !isComputedMaskStridedLoadUnitStore &&
      !isComputedMaskIndexedGatherLike &&
      !isComputedMaskIndexedScatterLike &&
       !isComputedMaskSegment2LoadLike &&
       !isComputedMaskSegment2StoreLike &&
       !isSegment2DeinterleaveUnitStore &&
       !isSegment2InterleaveUnitLoad &&
       !isStandaloneReduction &&
       !isRuntimeScalarComputedMaskStore &&
       !isF32ClampSelect &&
       !isDequantClampF32Epilogue &&
       !slice.rhsLoadOperation))
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires lhs-input-buffer and "
        "rhs-input-buffer or rhs-scalar-value generic load, broadcast, or "
        "scalar-splat dataflow");
  support::RuntimeABIParameter resolvedOutABI;
  if (isSegment2DeinterleaveUnitStore || isComputedMaskSegment2LoadLike) {
    for (tcrv::rvv::StoreOp store : genericStores) {
      llvm::Expected<support::RuntimeABIParameter> parameter =
          getRuntimeABIParameterBindingFromValue(
              store.getBuffer(), "tcrv_rvv.segment2 store buffer operand",
              {support::RuntimeABIParameterRole::SegmentField0OutputBuffer,
               support::RuntimeABIParameterRole::SegmentField1OutputBuffer});
      if (!parameter)
        return parameter.takeError();
      if (parameter->role ==
          support::RuntimeABIParameterRole::SegmentField0OutputBuffer) {
        if (slice.field0StoreOperation)
          return makeRVVEmitCRouteProviderError(
              "bounded generic RVV segment2 deinterleave route requires a "
              "unique field0 tcrv_rvv.store");
        slice.field0Store = store;
        slice.field0StoreOperation = store.getOperation();
        if (isComputedMaskSegment2LoadLike &&
            slice.field0LoadOperation && slice.field0Buffer != store.getBuffer())
          return makeRVVEmitCRouteProviderError(
              "bounded generic RVV computed-mask segment2 load route requires "
              "field0 old passthrough load and field0 store to use the same "
              "segment-field0 output buffer");
        slice.field0Buffer = store.getBuffer();
        slice.field0ABI = *parameter;
      } else {
        if (slice.field1StoreOperation)
          return makeRVVEmitCRouteProviderError(
              "bounded generic RVV segment2 deinterleave route requires a "
              "unique field1 tcrv_rvv.store");
        slice.field1Store = store;
        slice.field1StoreOperation = store.getOperation();
        if (isComputedMaskSegment2LoadLike &&
            slice.field1LoadOperation && slice.field1Buffer != store.getBuffer())
          return makeRVVEmitCRouteProviderError(
              "bounded generic RVV computed-mask segment2 load route requires "
              "field1 old passthrough load and field1 store to use the same "
              "segment-field1 output buffer");
        slice.field1Buffer = store.getBuffer();
        slice.field1ABI = *parameter;
      }
    }
    if (!slice.field0StoreOperation || !slice.field1StoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 route requires field0 and field1 "
          "destination stores with explicit segment field ABI roles");
    resolvedOutABI = slice.field0ABI;
  } else if (isSegment2InterleaveUnitLoad) {
    if (!slice.segment2StoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route requires "
          "interleaved destination segment2_store with explicit ABI role");
    resolvedOutABI = slice.outABI;
  } else if (isComputedMaskSegment2StoreLike) {
    if (!slice.maskedSegment2StoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store route requires "
          "interleaved destination masked_segment2_store with explicit ABI "
          "role");
    resolvedOutABI = slice.outABI;
  } else {
    llvm::Expected<support::RuntimeABIParameter> outABI =
        isComputedMaskStridedStore
            ? getRuntimeABIParameterBindingFromValue(
                  slice.maskedStridedStore.getBuffer(),
                  "tcrv_rvv.masked_strided_store buffer operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
        : isComputedMaskStridedLoadUnitStore
            ? getRuntimeABIParameterBindingFromValue(
                  slice.genericStore.getBuffer(),
                  "tcrv_rvv.computed_masked_strided_load store buffer operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
        : hasStridedMemory && !isStridedLoadUnitStore &&
                !isStridedInputWideningDotReduceAdd &&
                !isComputedMaskStridedInputWideningDotReduceAdd
            ? getRuntimeABIParameterBindingFromValue(
                  slice.stridedStore.getBuffer(),
                  "tcrv_rvv.strided_store buffer operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
        : isIndexedScatterUnitLoad
            ? getRuntimeABIParameterBindingFromValue(
                  slice.indexedStore.getDestination(),
                  "tcrv_rvv.indexed_store destination operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
        : isComputedMaskIndexedScatterLike
            ? getRuntimeABIParameterBindingFromValue(
                  slice.maskedIndexedStore.getDestination(),
                  "tcrv_rvv.masked_indexed_store destination operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
        : isMaskedUnitStore
            ? getRuntimeABIParameterBindingFromValue(
                  slice.maskedStore.getBuffer(),
                  "tcrv_rvv.masked_store destination operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
        : isRuntimeScalarComputedMaskStore
            ? getRuntimeABIParameterBindingFromValue(
                  slice.maskedStore.getBuffer(),
                  "tcrv_rvv.runtime_scalar_computed_mask_store destination "
                  "operand",
                  {support::RuntimeABIParameterRole::OutputBuffer})
            : getRuntimeABIParameterBindingFromValue(
                  slice.genericStore.getBuffer(),
                  "tcrv_rvv.store buffer operand",
                  {support::RuntimeABIParameterRole::OutputBuffer});
    if (!outABI)
      return outABI.takeError();
    resolvedOutABI = *outABI;
  }
  if (hasStridedMemory && !isStridedLoadUnitStore &&
      !isStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedInputWideningDotReduceAdd &&
      !isComputedMaskStridedStore &&
      !isComputedMaskStridedLoadUnitStore) {
    llvm::Expected<support::RuntimeABIParameter> outStrideABI =
        getRuntimeABIParameterBindingFromValue(
            slice.stridedStore.getStride(),
            "tcrv_rvv.strided_store stride operand",
            {support::RuntimeABIParameterRole::OutputStride,
             support::RuntimeABIParameterRole::DestinationByteStride});
    if (!outStrideABI)
      return outStrideABI.takeError();
    if (llvm::Error error = assignRVVGenericStridedStoreBinding(
            slice, slice.stridedStore, resolvedOutABI, *outStrideABI,
            isUnitLoadStridedStore || isComputedMaskStridedStore))
      return error;
    if (isUnitLoadStridedStore)
      slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
    else if (isComputedMaskStridedStore)
      slice.memoryForm =
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
    else
      slice.arithmeticKind = RVVSelectedBodyOperationKind::StridedAdd;
  } else if (isComputedMaskStridedStore) {
    llvm::Expected<support::RuntimeABIParameter> outStrideABI =
        getRuntimeABIParameterBindingFromValue(
            slice.maskedStridedStore.getStride(),
            "tcrv_rvv.masked_strided_store stride operand",
            {support::RuntimeABIParameterRole::DestinationByteStride});
    if (!outStrideABI)
      return outStrideABI.takeError();
    if (outStrideABI->role !=
        support::RuntimeABIParameterRole::DestinationByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "destination-byte-stride runtime ABI value");
    slice.storeOperation = slice.maskedStridedStore.getOperation();
    slice.outBuffer = slice.maskedStridedStore.getBuffer();
    slice.storeValue = slice.maskedStridedStore.getValue();
    slice.outStride = slice.maskedStridedStore.getStride();
    slice.outABI = resolvedOutABI;
    slice.outStrideABI = *outStrideABI;
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  } else if (isComputedMaskStridedLoadUnitStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore;
  } else if (isComputedMaskIndexedGatherLike) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  } else if (isComputedMaskIndexedScatterLike) {
    slice.storeOperation = slice.maskedIndexedStore.getOperation();
    slice.outBuffer = slice.maskedIndexedStore.getDestination();
    slice.storeValue = slice.maskedIndexedStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  } else if (isComputedMaskSegment2LoadLike) {
    slice.storeOperation = slice.field0StoreOperation;
    slice.outBuffer = slice.field0Buffer;
    slice.storeValue = slice.field0Store.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  } else if (isComputedMaskSegment2StoreLike) {
    slice.storeOperation = slice.maskedSegment2StoreOperation;
    slice.outBuffer = slice.maskedSegment2Store.getDestination();
    slice.storeValue = slice.maskedSegment2Store.getField0();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  } else if (isStridedLoadUnitStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  } else if (isStridedInputWideningDotReduceAdd) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce;
  } else if (isComputedMaskStridedInputWideningDotReduceAdd) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::
            ComputedMaskStridedInputWideningDotReduce;
  } else if (isIndexedGatherUnitStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  } else if (isIndexedScatterUnitLoad) {
    slice.storeOperation = slice.indexedStore.getOperation();
    slice.outBuffer = slice.indexedStore.getDestination();
    slice.storeValue = slice.indexedStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  } else if (isMaskedUnitLoadStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  } else if (isMaskedUnitStore) {
    slice.storeOperation = slice.maskedStore.getOperation();
    slice.outBuffer = slice.maskedStore.getBuffer();
    slice.storeValue = slice.maskedStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::MaskedUnitStore;
  } else if (isRuntimeScalarComputedMaskStore) {
    slice.storeOperation = slice.maskedStore.getOperation();
    slice.outBuffer = slice.maskedStore.getBuffer();
    slice.storeValue = slice.maskedStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore;
  } else if (isComputedMaskUnitLoadStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  } else if (isRuntimeScalarComputedMaskLoadStore) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore;
  } else if (isSegment2DeinterleaveUnitStore) {
    slice.storeOperation = slice.field0StoreOperation;
    slice.outBuffer = slice.field0Buffer;
    slice.storeValue = slice.field0Store.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  } else if (isSegment2InterleaveUnitLoad) {
    slice.storeOperation = slice.segment2StoreOperation;
    slice.outBuffer = slice.segment2Store.getDestination();
    slice.storeValue = slice.segment2Store.getField0();
    slice.memoryForm = RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  } else if (isComputedMaskStandaloneReduction) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideStandaloneReduction;
  } else if (isRuntimeScalarComputedMaskStandaloneReduction) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm = RVVSelectedBodyMemoryForm::
        RuntimeScalarComputedMaskUnitStrideStandaloneReduction;
  } else if (isStandaloneReduction) {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    slice.memoryForm =
        RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction;
  } else {
    slice.storeOperation = slice.genericStore.getOperation();
    slice.outBuffer = slice.genericStore.getBuffer();
    slice.storeValue = slice.genericStore.getValue();
    if (isRuntimeScalarSplatStore) {
      slice.memoryForm = RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore;
      slice.arithmeticKind = RVVSelectedBodyOperationKind::RuntimeScalarSplatStore;
      slice.arithmeticResult = slice.rhsValue;
    } else if (isRuntimeScalarDualCompareMaskAndSelect) {
      slice.memoryForm =
          RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect;
    } else if (isF32ClampSelect) {
      slice.memoryForm = RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect;
    } else if (isDequantClampF32Epilogue) {
      slice.memoryForm =
          RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue;
    } else if (isWideningProductReduceDequantClamp) {
      slice.memoryForm =
          RVVSelectedBodyMemoryForm::
              UnitStrideWideningProductReduceDequantClampF32;
    } else if (isRuntimeScalarCompareSelect) {
      slice.memoryForm = RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect;
    } else if (isRuntimeScalarComputedMaskStore ||
               isRuntimeScalarComputedMaskLoadStore ||
               isRuntimeScalarComputedMaskedMAccAdd ||
               isRuntimeScalarComputedMaskStandaloneReduction) {
      // Runtime-threshold computed-mask consumers use tcrv_rvv.splat for the
      // compare RHS, but their route identity is not scalar-broadcast
      // elementwise arithmetic.
      if (isRuntimeScalarComputedMaskedMAccAdd)
        slice.memoryForm =
            RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskUnitStrideMAcc;
      if (isRuntimeScalarComputedMaskStandaloneReduction)
        slice.memoryForm = RVVSelectedBodyMemoryForm::
            RuntimeScalarComputedMaskUnitStrideStandaloneReduction;
    } else if (hasScalarBroadcast && !isScalarBroadcastMAccAdd) {
      llvm::Expected<RVVSelectedBodyOperationKind> scalarBroadcastKind =
          getRVVScalarBroadcastOperationKind(slice.arithmeticKind);
      if (!scalarBroadcastKind)
        return scalarBroadcastKind.takeError();
      slice.arithmeticKind = *scalarBroadcastKind;
    }
  }
  llvm::StringRef operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind);
  slice.runtimeElementCountABI = *runtimeElementCountABI;
  slice.outABI = resolvedOutABI;
  if (isF32ClampSelect) {
    if (!slice.lowerBoundScalarSplat || !slice.upperBoundScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires explicit "
          "lower-bound-scalar-value and upper-bound-scalar-value splats");
    if (slice.lowerBoundABI.role !=
            support::RuntimeABIParameterRole::LowerBoundScalarValue ||
        slice.upperBoundABI.role !=
            support::RuntimeABIParameterRole::UpperBoundScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires unambiguous "
          "lower-bound and upper-bound runtime ABI roles");
    if (slice.lowerBoundABI.cType != "float" ||
        slice.upperBoundABI.cType != "float")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires lower and "
          "upper bound runtime ABI C type 'float'");
    if (slice.lhsABI.role !=
            support::RuntimeABIParameterRole::LHSInputBuffer ||
        slice.lhsABI.cType != "const float *" ||
        slice.outABI.cType != "float *")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires input C type "
          "'const float *' and output C type 'float *'");
    if (slice.compareOp.getKind() != "slt" ||
        slice.secondaryCompareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route currently supports only "
          "two tcrv_rvv.compare {kind = \"slt\"} comparisons");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.lowerBoundValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires the lower "
          "compare to consume input vector and lower-bound splat");
    if (slice.selectOp.getMask() != slice.compareMask ||
        slice.selectOp.getTrueValue() != slice.lowerBoundValue ||
        slice.selectOp.getFalseValue() != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires the first "
          "select to choose lower-bound vector when input < lower, otherwise "
          "input");
    if (slice.lowerClampedValue != slice.selectOp.getSelected())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires first select "
          "result to be the lower-clamped intermediate");
    if (slice.secondaryCompareLhs != slice.upperBoundValue ||
        slice.secondaryCompareRhs != slice.lowerClampedValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires the upper "
          "compare to consume upper-bound splat and lower-clamped vector");
    if (slice.secondarySelectOp.getMask() != slice.secondaryCompareMask ||
        slice.secondarySelectOp.getTrueValue() != slice.upperBoundValue ||
        slice.secondarySelectOp.getFalseValue() != slice.lowerClampedValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires the second "
          "select to choose upper-bound vector when upper < lower-clamped, "
          "otherwise the lower-clamped vector");
    if (slice.storeValue != slice.secondarySelectOp.getSelected())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires store to "
          "consume the second select result");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.secondaryCompareOp.getVl() != slice.setvl.getVl() ||
        slice.selectOp.getVl() != slice.setvl.getVl() ||
        slice.secondarySelectOp.getVl() != slice.setvl.getVl() ||
        slice.lowerBoundScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.upperBoundScalarSplat.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires load, splats, "
          "compares, selects, and store to share the selected VL token");
    auto inputVector =
        llvm::dyn_cast<tcrv::rvv::VectorType>(slice.lhsValue.getType());
    if (!inputVector || !inputVector.getElementType().isF32() ||
        inputVector.getLmul() != tcrv::rvv::getRVVLMULM1())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV f32 clamp/select route requires f32 LMUL m1 "
          "input vector type");
  }
  if (isDequantClampF32Epilogue) {
    if (!slice.dequantizeOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires one "
          "tcrv_rvv.dequantize producer before clamp/select");
    if (!slice.lowerBoundScalarSplat || !slice.upperBoundScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires "
          "explicit lower-bound-scalar-value and upper-bound-scalar-value "
          "splats");
    if (slice.dequantScaleABI.role !=
            support::RuntimeABIParameterRole::DequantScaleValue ||
        slice.lowerBoundABI.role !=
            support::RuntimeABIParameterRole::LowerBoundScalarValue ||
        slice.upperBoundABI.role !=
            support::RuntimeABIParameterRole::UpperBoundScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires "
          "unambiguous dequant-scale, lower-bound, and upper-bound runtime "
          "ABI roles");
    if (slice.lhsABI.role !=
            support::RuntimeABIParameterRole::LHSInputBuffer ||
        slice.lhsABI.cType != "const int32_t *" ||
        slice.dequantScaleABI.cType != "float" ||
        slice.lowerBoundABI.cType != "float" ||
        slice.upperBoundABI.cType != "float" ||
        slice.outABI.cType != "float *")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires source "
          "C type 'const int32_t *', scale/bounds C type 'float', and "
          "output C type 'float *'");
    if (slice.dequantizeOp.getSource() != slice.lhsValue ||
        slice.dequantizeOp.getScale() != slice.dequantScale)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires "
          "tcrv_rvv.dequantize to consume the i32 source load and runtime "
          "scale ABI value");
    if (slice.dequantizeOp.getKind() != "i32_to_f32_scaled" ||
        slice.dequantizeOp.getDequantRelation() !=
            "signed-i32m1-to-f32m1-scale-f32")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires the "
          "signed i32m1 to f32m1 runtime-scale dequantization relation");
    mlir::Value dequantized = slice.dequantizeOp.getResult();
    if (slice.compareOp.getKind() != "slt" ||
        slice.secondaryCompareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route currently "
          "supports only two tcrv_rvv.compare {kind = \"slt\"} comparisons");
    if (slice.compareLhs != dequantized ||
        slice.compareRhs != slice.lowerBoundValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires the "
          "lower compare to consume the f32 dequant result and lower-bound "
          "splat");
    if (slice.selectOp.getMask() != slice.compareMask ||
        slice.selectOp.getTrueValue() != slice.lowerBoundValue ||
        slice.selectOp.getFalseValue() != dequantized)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires the "
          "first select to choose lower-bound vector when dequantized < "
          "lower, otherwise the dequantized vector");
    if (slice.lowerClampedValue != slice.selectOp.getSelected())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires first "
          "select result to be the lower-clamped f32 intermediate");
    if (slice.secondaryCompareLhs != slice.upperBoundValue ||
        slice.secondaryCompareRhs != slice.lowerClampedValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires the "
          "upper compare to consume upper-bound splat and lower-clamped f32 "
          "vector");
    if (slice.secondarySelectOp.getMask() != slice.secondaryCompareMask ||
        slice.secondarySelectOp.getTrueValue() != slice.upperBoundValue ||
        slice.secondarySelectOp.getFalseValue() != slice.lowerClampedValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires the "
          "second select to choose upper-bound vector when upper < "
          "lower-clamped, otherwise the lower-clamped vector");
    if (slice.storeValue != slice.secondarySelectOp.getSelected())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires store "
          "to consume the second select result");
    if (slice.dequantizeOp.getVl() != slice.setvl.getVl() ||
        slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.secondaryCompareOp.getVl() != slice.setvl.getVl() ||
        slice.selectOp.getVl() != slice.setvl.getVl() ||
        slice.secondarySelectOp.getVl() != slice.setvl.getVl() ||
        slice.lowerBoundScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.upperBoundScalarSplat.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires load, "
          "dequantize, splats, compares, selects, and store to share the "
          "selected VL token");
    auto sourceVector =
        llvm::dyn_cast<tcrv::rvv::VectorType>(slice.lhsValue.getType());
    auto resultVector =
        llvm::dyn_cast<tcrv::rvv::VectorType>(dequantized.getType());
    if (!sourceVector || !sourceVector.getElementType().isInteger(32) ||
        sourceVector.getLmul() != tcrv::rvv::getRVVLMULM1() ||
        !resultVector || !resultVector.getElementType().isF32() ||
        resultVector.getLmul() != tcrv::rvv::getRVVLMULM1())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV dequant-clamp epilogue route requires i32 "
          "LMUL m1 source vector and f32 LMUL m1 dequant/clamp vector type");
  }
  if (isWideningProductReductionChain) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.standaloneReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.standalone_reduce accumulator_seed operand in "
            "product-reduction chain",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.standaloneReduceOp.getAccumulatorSeed();
  }
  if (isWideningDotReduceAdd || isStridedInputWideningDotReduceAdd) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.wideningDotReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.widening_dot_reduce accumulator_seed operand",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.wideningDotReduceOp.getAccumulatorSeed();
  }
  if (isComputedMaskWideningDotReduceAdd ||
      isComputedMaskStridedInputWideningDotReduceAdd) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.maskedWideningDotReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.masked_widening_dot_reduce accumulator_seed operand",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.maskedWideningDotReduceOp.getAccumulatorSeed();
  }
  if (isStandaloneReduction) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.standaloneReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.standalone_reduce accumulator_seed operand",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.standaloneReduceOp.getAccumulatorSeed();
  }
  if (isComputedMaskStandaloneReduction) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.maskedStandaloneReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.masked_standalone_reduce accumulator_seed operand",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.maskedStandaloneReduceOp.getAccumulatorSeed();
  }
  if (isRuntimeScalarComputedMaskStandaloneReduction) {
    llvm::Expected<support::RuntimeABIParameter> accumulatorABI =
        getRuntimeABIParameterBindingFromValue(
            slice.maskedStandaloneReduceOp.getAccumulatorSeed(),
            "tcrv_rvv.masked_standalone_reduce accumulator_seed operand",
            {support::RuntimeABIParameterRole::AccumulatorInputBuffer});
    if (!accumulatorABI)
      return accumulatorABI.takeError();
    slice.accumulatorABI = *accumulatorABI;
    slice.accumulatorBuffer =
        slice.maskedStandaloneReduceOp.getAccumulatorSeed();
  }
  if (isComputedMaskSelect) {
    if (!slice.trueValueLoadOperation || !slice.falseValueLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires "
          "true-value-input-buffer and false-value-input-buffer loads");
    if (slice.compareOp.getKind() != "slt" &&
        slice.compareOp.getKind() != "sle")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route currently supports "
          "only tcrv_rvv.compare {kind = \"slt\"} or {kind = \"sle\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires "
          "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    if (slice.selectOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires "
          "tcrv_rvv.select to consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.trueValue ||
        slice.arithmeticRhs != slice.falseValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires "
          "tcrv_rvv.select to consume true-value load as true value and "
          "false-value load as false value");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.selectOp.getVl() != slice.setvl.getVl() ||
        slice.trueValueGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.falseValueGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route requires compare, "
          "select, and true/false loads to consume the selected "
          "!tcrv_rvv.vl token");
    if (slice.sourceLoadOperation || slice.accumulatorLoadOperation ||
        slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask select route does not support "
          "source, accumulator, or runtime mask_load inputs");
  } else if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (!slice.trueValueLoadOperation || !slice.falseValueLoadOperation ||
        !slice.secondaryCompareLhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires compare lhs B, true-value, and false-value loads");
    if (!slice.rhsScalarSplat || !slice.rhsSecondaryScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires primary and secondary runtime scalar splats");
    if (slice.compareOp.getKind() != "sle" ||
        slice.secondaryCompareOp.getKind() != "sle")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route currently supports only tcrv_rvv.compare {kind = \"sle\"} "
          "for both predicates");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires compare A to consume lhs A load and rhs scalar A "
          "splat results");
    if (slice.secondaryCompareLhs !=
            slice.secondaryCompareLhsGenericLoad.getLoaded() ||
        slice.secondaryCompareRhs !=
            slice.rhsSecondaryScalarSplat.getBroadcast())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires compare B to consume lhs B load and rhs scalar B "
          "splat results");
    if (slice.maskAndOp.getLhs() != slice.compareMask ||
        slice.maskAndOp.getRhs() != slice.secondaryCompareMask ||
        slice.composedMask != slice.maskAndOp.getMask())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires tcrv_rvv.mask_and to compose the two compare masks");
    if (slice.selectOp.getMask() != slice.composedMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires tcrv_rvv.select to consume the composed mask");
    if (slice.arithmeticLhs != slice.trueValue ||
        slice.arithmeticRhs != slice.falseValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires tcrv_rvv.select to consume true-value load as true "
          "value and false-value load as false value");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.secondaryCompareOp.getVl() != slice.setvl.getVl() ||
        slice.maskAndOp.getVl() != slice.setvl.getVl() ||
        slice.selectOp.getVl() != slice.setvl.getVl() ||
        slice.rhsScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.rhsSecondaryScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.secondaryCompareLhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.trueValueGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.falseValueGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires splats, compares, mask_and, select, and true/false "
          "loads to consume the selected !tcrv_rvv.vl token");
    if (slice.rhsABI.role != support::RuntimeABIParameterRole::RHSScalarValue ||
        slice.secondaryCompareLhsABI.role !=
            support::RuntimeABIParameterRole::RHSInputBuffer ||
        slice.secondaryCompareRhsScalarABI.role !=
            support::RuntimeABIParameterRole::RHSSecondaryScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route requires rhs-scalar-value, rhs-input-buffer, and "
          "rhs-secondary-scalar-value ABI roles for the composed predicate");
    if (slice.sourceLoadOperation || slice.accumulatorLoadOperation ||
        slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar dual-compare mask-and select "
          "route does not support source, accumulator, or runtime mask_load "
          "inputs");
  } else if (isRuntimeScalarCompareSelect) {
    if (!slice.trueValueLoadOperation || !slice.falseValueLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar compare/select route requires "
          "true-value-input-buffer and false-value-input-buffer loads");
    if (slice.compareOp.getKind() != "sle")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar compare/select route currently "
          "supports only tcrv_rvv.compare {kind = \"sle\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar compare/select route requires "
          "tcrv_rvv.compare to consume lhs load and rhs scalar splat results");
    if (slice.selectOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar compare/select route requires "
          "tcrv_rvv.select to consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.trueValue ||
        slice.arithmeticRhs != slice.falseValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar compare/select route requires "
          "tcrv_rvv.select to consume true-value load as true value and "
          "false-value load as false value");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.selectOp.getVl() != slice.setvl.getVl() ||
        slice.rhsScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.trueValueGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.falseValueGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar compare/select route requires "
          "splat, compare, select, and true/false loads to consume the "
          "selected !tcrv_rvv.vl token");
    if (slice.rhsABI.role != support::RuntimeABIParameterRole::RHSScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar compare/select route requires "
          "the splat scalar threshold to bind rhs-scalar-value");
    if (slice.sourceLoadOperation || slice.accumulatorLoadOperation ||
        slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar compare/select route does not "
          "support source, accumulator, or runtime mask_load inputs");
  } else if (isRuntimeScalarComputedMaskStore) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route "
          "requires a source-input-buffer payload load");
    if (slice.compareOp.getKind() != "sle")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route "
          "currently supports only tcrv_rvv.compare {kind = \"sle\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route "
          "requires tcrv_rvv.compare to consume lhs load and rhs scalar "
          "splat results");
    if (slice.maskedStore.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route "
          "requires tcrv_rvv.masked_store to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.maskedActiveValue != slice.sourceValue ||
        slice.storeValue != slice.sourceValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route "
          "requires tcrv_rvv.masked_store to consume the source payload load "
          "as its active store value");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.rhsScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.sourceGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.maskedStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route "
          "requires splat, compare, payload source load, and masked_store to "
          "consume the selected !tcrv_rvv.vl token");
    if (slice.rhsABI.role != support::RuntimeABIParameterRole::RHSScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route "
          "requires the splat scalar threshold to bind rhs-scalar-value");
    if (slice.sourceABI.role !=
        support::RuntimeABIParameterRole::SourceInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route "
          "requires the payload load to bind source-input-buffer");
    if (slice.trueValueLoadOperation || slice.falseValueLoadOperation ||
        slice.accumulatorLoadOperation || slice.maskLoadOperation ||
        slice.genericStore || slice.selectOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask store route does "
          "not support true/false select loads, accumulator, runtime "
          "mask_load, unit-stride store, or compare-select fallback inputs");
  } else if (isRuntimeScalarComputedMaskLoadStore) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires one source-input-buffer tcrv_rvv.masked_load for active "
          "source values");
    if (!slice.maskedLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires one tcrv_rvv.masked_load");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires an output-buffer tcrv_rvv.load for old-destination "
          "preservation");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires the old-destination load to consume the same output "
          "buffer as tcrv_rvv.store");
    if (slice.compareOp.getKind() != "sle")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "currently supports only tcrv_rvv.compare {kind = \"sle\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires tcrv_rvv.compare to consume lhs load and rhs scalar "
          "splat results");
    if (slice.maskedLoadOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires tcrv_rvv.masked_load to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.maskedActiveValue != slice.sourceValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires tcrv_rvv.masked_load to load source-input lanes and "
          "consume old-destination load as inactive passthrough");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.rhsScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.maskedLoadOp.getVl() != slice.setvl.getVl() ||
        slice.accumulatorLoadOperation == nullptr)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires splat, compare, old-destination load, and masked_load to "
          "consume the selected !tcrv_rvv.vl token");
    if (slice.rhsABI.role != support::RuntimeABIParameterRole::RHSScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires the splat scalar threshold to bind rhs-scalar-value");
    if (slice.sourceABI.role !=
        support::RuntimeABIParameterRole::SourceInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "requires the masked_load source to bind source-input-buffer");
    if (slice.trueValueLoadOperation || slice.falseValueLoadOperation ||
        slice.maskLoadOperation || slice.maskedStore || slice.selectOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask load-store route "
          "does not support true/false select loads, runtime mask_load, "
          "masked_store, or compare-select fallback inputs");
  } else if (isCompareSelect) {
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.compare "
          "to consume lhs/rhs generic load results");
    if (slice.selectOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.select "
          "to consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.select "
          "to consume lhs as true value and rhs as false value");
  } else if (isMaskedArithmetic) {
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked elementwise route requires "
          "tcrv_rvv.compare to consume lhs/rhs generic load results");
    if (slice.maskedBinaryOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked elementwise route requires "
          "tcrv_rvv.masked_binary to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.maskedPassthrough != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked elementwise route requires "
          "tcrv_rvv.masked_binary passthrough to consume the lhs load result");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked elementwise route requires "
          "tcrv_rvv.masked_binary to consume lhs/rhs generic load results");
  } else if (isComputedMaskedMAccAdd) {
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires one accumulator-input-buffer tcrv_rvv.load");
    if (!slice.dotLHSLoadOperation || !slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires dot-lhs-input-buffer and dot-rhs-input-buffer payload "
          "loads");
    if (slice.dotLHSABI.role !=
            support::RuntimeABIParameterRole::DotLHSInputBuffer ||
        slice.dotRHSABI.role !=
            support::RuntimeABIParameterRole::DotRHSInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires payload lhs/rhs loads to bind dedicated dot input ABI "
          "roles");
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires the accumulator load to bind accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires separate accumulator input and output destination ABI "
          "values");
    if (!slice.compareOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires one tcrv_rvv.compare producer");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "currently supports only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires tcrv_rvv.compare to consume compare lhs/rhs generic load "
          "results");
    if (slice.maskedMAccOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires tcrv_rvv.masked_macc to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.dotLHSValue ||
        slice.arithmeticRhs != slice.dotRHSValue ||
        slice.arithmeticAccumulator != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires tcrv_rvv.masked_macc to consume payload lhs/rhs loads and "
          "the accumulator-input-buffer load result");
    auto accumulatorLoad =
        slice.accumulatorLoadOperation
            ? llvm::dyn_cast<tcrv::rvv::LoadOp>(
                  slice.accumulatorLoadOperation)
            : tcrv::rvv::LoadOp();
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedMAccOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.rhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.dotLHSGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.dotRHSGenericLoad.getVl() != slice.setvl.getVl() ||
        !accumulatorLoad ||
        accumulatorLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route "
          "requires compare, masked_macc, compare loads, payload loads, and "
          "accumulator load to consume the selected !tcrv_rvv.vl token");
    if (slice.maskLoadOperation || slice.sourceLoadOperation ||
        slice.indexLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask multiply-accumulate route does "
          "not support runtime mask_load, source-input, or index inputs");
  } else if (isRuntimeScalarComputedMaskedMAccAdd) {
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires one accumulator-input-buffer "
          "tcrv_rvv.load");
    if (!slice.dotLHSLoadOperation || !slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires dot-lhs-input-buffer and "
          "dot-rhs-input-buffer payload loads");
    if (!slice.rhsScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires one tcrv_rvv.splat from the "
          "rhs-scalar-value threshold");
    if (slice.dotLHSABI.role !=
            support::RuntimeABIParameterRole::DotLHSInputBuffer ||
        slice.dotRHSABI.role !=
            support::RuntimeABIParameterRole::DotRHSInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires payload lhs/rhs loads to bind "
          "dedicated dot input ABI roles");
    if (slice.rhsABI.role != support::RuntimeABIParameterRole::RHSScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires rhs_scalar to bind "
          "rhs-scalar-value");
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires the accumulator load to bind "
          "accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires separate accumulator input and "
          "output destination ABI values");
    if (!slice.compareOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires one tcrv_rvv.compare producer");
    if (slice.compareOp.getKind() != "sle")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route currently supports only "
          "tcrv_rvv.compare {kind = \"sle\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires tcrv_rvv.compare to consume "
          "the compare lhs load and splatted rhs scalar threshold");
    if (slice.maskedMAccOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires tcrv_rvv.masked_macc to "
          "consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.dotLHSValue ||
        slice.arithmeticRhs != slice.dotRHSValue ||
        slice.arithmeticAccumulator != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires tcrv_rvv.masked_macc to "
          "consume payload lhs/rhs loads and the accumulator-input-buffer "
          "load result");
    auto accumulatorLoad =
        slice.accumulatorLoadOperation
            ? llvm::dyn_cast<tcrv::rvv::LoadOp>(
                  slice.accumulatorLoadOperation)
            : tcrv::rvv::LoadOp();
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.rhsScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.maskedMAccOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.dotLHSGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.dotRHSGenericLoad.getVl() != slice.setvl.getVl() ||
        !accumulatorLoad ||
        accumulatorLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route requires compare, scalar splat, "
          "masked_macc, compare load, payload loads, and accumulator load to "
          "consume the selected !tcrv_rvv.vl token");
    if (slice.maskLoadOperation || slice.sourceLoadOperation ||
        slice.indexLoadOperation || slice.rhsGenericLoad)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask "
          "multiply-accumulate route does not support runtime mask_load, "
          "source-input, index inputs, or compare rhs vector-load fallback");
  } else if (isScalarBroadcastMAccAdd) {
    if (!slice.rhsScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route requires one tcrv_rvv.splat producer for the "
          "RHS scalar vector");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route requires one accumulator-input-buffer "
          "tcrv_rvv.load");
    if (slice.rhsGenericLoad || slice.rhsBroadcastLoad)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route must not use a vector RHS load or broadcast_load "
          "fallback");
    if (slice.rhsABI.role != support::RuntimeABIParameterRole::RHSScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route requires rhs_scalar to bind rhs-scalar-value");
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route requires the accumulator load to bind "
          "accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route requires separate accumulator input and output "
          "destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route requires tcrv_rvv.macc to consume the lhs load, "
          "the RHS scalar splat result, and the accumulator-input-buffer load "
          "result");
    auto accumulatorLoad =
        slice.accumulatorLoadOperation
            ? llvm::dyn_cast<tcrv::rvv::LoadOp>(
                  slice.accumulatorLoadOperation)
            : tcrv::rvv::LoadOp();
    if (slice.rhsScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.maccOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        !accumulatorLoad ||
        accumulatorLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route requires lhs load, RHS scalar splat, "
          "accumulator load, and macc to consume the selected "
          "!tcrv_rvv.vl token");
    if (slice.compareOp || slice.maskLoadOperation || slice.sourceLoadOperation ||
        slice.indexLoadOperation || slice.maskedMAccOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV scalar-broadcast multiply-accumulate "
          "composition route does not support compare, mask_load, "
          "source-input, index inputs, or masked_macc fallback");
  } else if (isMAccAdd) {
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires one "
          "accumulator-input-buffer tcrv_rvv.load");
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires the "
          "accumulator load to bind accumulator-input-buffer, not output "
          "buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV multiply-accumulate route requires separate "
          "accumulator input and output destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
        "bounded generic RVV multiply-accumulate route requires "
        "tcrv_rvv.macc to consume lhs/rhs generic load results and the "
        "accumulator-input-buffer load result");
  } else if (isWideningMAccAdd) {
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening multiply-accumulate route requires "
          "one accumulator-input-buffer tcrv_rvv.load");
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening multiply-accumulate route requires "
          "the accumulator load to bind accumulator-input-buffer, not output "
          "buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening multiply-accumulate route requires "
          "separate accumulator input and output destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening multiply-accumulate route requires "
          "tcrv_rvv.widening_macc to consume lhs/rhs source loads and the "
          "accumulator-input-buffer load result");
  } else if (isWideningProduct) {
    if (slice.accumulatorLoadOperation || slice.accumulatorBuffer ||
        slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision widening-product route does not "
          "support accumulator vector loads or scalar seed boundaries");
    if (slice.lhsABI.role != support::RuntimeABIParameterRole::LHSInputBuffer ||
        slice.rhsABI.role != support::RuntimeABIParameterRole::RHSInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision widening-product route requires "
          "lhs/rhs source loads to bind lhs-input-buffer and rhs-input-buffer");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.storeValue != slice.arithmeticResult)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision widening-product route requires "
          "tcrv_rvv.widening_product to consume lhs/rhs source loads and "
          "tcrv_rvv.store to store the product result");
    if (slice.wideningProductOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.rhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.genericStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision widening-product route requires "
          "source loads, widening_product, and store to consume the selected "
          "!tcrv_rvv.vl token");
  } else if (isWideningProductReductionChain) {
    const bool hasProductReductionDequantization =
        isWideningProductReduceDequantize ||
        isWideningProductReduceDequantClamp;
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction route requires "
          "the accumulator seed to bind accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction route requires "
          "separate accumulator seed input and scalar output destination ABI "
          "values");
    if (slice.lhsABI.role != support::RuntimeABIParameterRole::LHSInputBuffer ||
        slice.rhsABI.role != support::RuntimeABIParameterRole::RHSInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction route requires "
          "lhs/rhs i8 source loads to bind lhs-input-buffer and "
          "rhs-input-buffer");
    if (hasProductReductionDequantization && !slice.dequantizeOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction "
          "dequantization route requires a tcrv_rvv.dequantize consumer");
    // Single-scope typed dequant bodies (Stage 3 flip) carry the i32 carry on
    // the standalone_reduce result directly and have no handoff op; only the
    // legacy two-scope body has the gearbox_cross_region_handoff carrier.
    if (hasProductReductionDequantization &&
        slice.dequantScaleABI.role !=
            support::RuntimeABIParameterRole::DequantScaleValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction "
          "dequantization route requires the runtime scale to bind "
          "dequant-scale-value");
    if (isWideningProductReduceDequantClamp &&
        (!slice.lowerBoundScalarSplat || !slice.upperBoundScalarSplat ||
         !slice.compareOp || !slice.secondaryCompareOp ||
         !slice.selectOp || !slice.secondarySelectOp))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction "
          "dequant-clamp route requires lower/upper splats and two "
          "compare/select stages");
    const mlir::Value expectedProductReductionStoreValue =
        isWideningProductReduceDequantClamp
            ? slice.secondarySelectOp.getSelected()
        : isWideningProductReduceDequantize
            ? slice.dequantizeOp.getResult()
            : slice.standaloneReduceOp.getResult();
    // The dequant consumer reads either the handoff carrier (legacy two-scope)
    // or the standalone_reduce result directly (single-scope typed body).
    const mlir::Value dequantI32Carrier =
        slice.gearboxCrossRegionHandoffOp
            ? slice.gearboxCrossRegionHandoffOp.getOutput()
            : slice.standaloneReduceOp.getResult();
    if (productSlotLhs(slice) != slice.lhsValue ||
        productSlotRhs(slice) != slice.rhsValue ||
        slice.standaloneReduceOp.getInput() != productSlotResult(slice) ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer ||
        slice.storeValue != expectedProductReductionStoreValue ||
        (hasProductReductionDequantization &&
         ((slice.gearboxCrossRegionHandoffOp &&
           slice.gearboxCrossRegionHandoffOp.getInput() !=
               slice.standaloneReduceOp.getResult()) ||
          dequantI32Carrier != slice.dequantizeOp.getSource())))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction route requires "
          "tcrv_rvv.widening_product to consume lhs/rhs i8 source loads, "
          "tcrv_rvv.standalone_reduce to consume the product and scalar "
          "accumulator seed boundary, optional "
          "tcrv_rvv.gearbox_cross_region_handoff to forward the i32 "
          "reduction result into tcrv_rvv.dequantize, and tcrv_rvv.store to "
          "store the final chain result");
    if (isWideningProductReduceDequantClamp) {
      if (slice.lowerBoundABI.role !=
              support::RuntimeABIParameterRole::LowerBoundScalarValue ||
          slice.upperBoundABI.role !=
              support::RuntimeABIParameterRole::UpperBoundScalarValue ||
          slice.lowerBoundABI.cType != "float" ||
          slice.upperBoundABI.cType != "float")
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV low-precision product-reduction "
            "dequant-clamp route requires lower/upper bound runtime ABI "
            "roles with C type 'float'");
      if (slice.compareOp.getKind() != "slt" ||
          slice.secondaryCompareOp.getKind() != "slt" ||
          slice.compareLhs != slice.dequantizeOp.getResult() ||
          slice.compareRhs != slice.lowerBoundValue ||
          slice.selectOp.getMask() != slice.compareMask ||
          slice.selectOp.getTrueValue() != slice.lowerBoundValue ||
          slice.selectOp.getFalseValue() != slice.dequantizeOp.getResult() ||
          slice.lowerClampedValue != slice.selectOp.getSelected() ||
          slice.secondaryCompareLhs != slice.upperBoundValue ||
          slice.secondaryCompareRhs != slice.lowerClampedValue ||
          slice.secondarySelectOp.getMask() != slice.secondaryCompareMask ||
          slice.secondarySelectOp.getTrueValue() != slice.upperBoundValue ||
          slice.secondarySelectOp.getFalseValue() != slice.lowerClampedValue)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV low-precision product-reduction "
            "dequant-clamp route requires dequant result -> lower clamp -> "
            "upper clamp structural dataflow");
    }
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction route does not "
          "load the accumulator seed as a vector; it must remain a scalar "
          "runtime ABI boundary");
    if (productSlotVL(slice) != slice.setvl.getVl() ||
        slice.standaloneReduceOp.getVl() != slice.setvl.getVl() ||
        (slice.gearboxCrossRegionHandoffOp &&
         slice.gearboxCrossRegionHandoffOp.getVl() != slice.setvl.getVl()) ||
        (hasProductReductionDequantization &&
         slice.dequantizeOp.getVl() != slice.setvl.getVl()) ||
        (isWideningProductReduceDequantClamp &&
         (slice.lowerBoundScalarSplat.getVl() != slice.setvl.getVl() ||
          slice.upperBoundScalarSplat.getVl() != slice.setvl.getVl() ||
          slice.compareOp.getVl() != slice.setvl.getVl() ||
          slice.secondaryCompareOp.getVl() != slice.setvl.getVl() ||
          slice.selectOp.getVl() != slice.setvl.getVl() ||
          slice.secondarySelectOp.getVl() != slice.setvl.getVl())) ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.rhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.genericStore.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction route requires "
          "source loads, widening_product, standalone_reduce, optional "
          "gearbox_cross_region_handoff, dequantize, and store to consume "
          "the selected !tcrv_rvv.vl token");
    if (hasProductReductionDequantization &&
        slice.gearboxCrossRegionHandoffOp &&
        slice.gearboxCrossRegionHandoffOp.getRuntimeAvl() !=
            slice.setvl.getAvl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV low-precision product-reduction "
          "dequantization route requires tcrv_rvv.gearbox_cross_region_handoff "
          "runtime AVL operand to match the selected tcrv_rvv.setvl AVL "
          "runtime SSA value");
  } else if (isWideningDotReduceAdd || isStridedInputWideningDotReduceAdd) {
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening dot-product reduction route requires "
          "the accumulator seed to bind accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening dot-product reduction route requires "
          "separate accumulator seed input and output destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening dot-product reduction route requires "
          "tcrv_rvv.widening_dot_reduce to consume lhs/rhs source loads and "
          "the accumulator-input-buffer scalar seed boundary");
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening dot-product reduction route does not "
          "load the accumulator seed as a vector; it must remain a scalar "
          "runtime ABI boundary");
    if (isStridedInputWideningDotReduceAdd &&
        (slice.lhsStridedLoad.getVl() != slice.setvl.getVl() ||
         slice.rhsStridedLoad.getVl() != slice.setvl.getVl()))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-input widening dot-product reduction "
          "route requires both strided source loads to consume the selected "
          "!tcrv_rvv.vl token");
  } else if (isStandaloneReduction) {
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route requires the "
          "accumulator seed to bind accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route requires separate "
          "accumulator seed input and scalar output destination ABI values");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route requires "
          "tcrv_rvv.standalone_reduce to consume the input vector load and "
          "the accumulator-input-buffer scalar seed boundary");
    if (slice.accumulatorLoadOperation || slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route does not load RHS "
          "or accumulator seed as vectors; the seed must remain a scalar "
          "runtime ABI boundary");
    if (slice.standaloneReduceOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV standalone reduction route requires input load "
          "and standalone_reduce to consume the selected !tcrv_rvv.vl token");
  } else if (isComputedMaskStandaloneReduction) {
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires the accumulator seed to bind accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires separate accumulator seed input and scalar output "
          "destination ABI values");
    if (!slice.compareOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires one tcrv_rvv.compare producer");
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires one source-input-buffer tcrv_rvv.load");
    if (slice.sourceABI.role !=
        support::RuntimeABIParameterRole::SourceInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires the source load to bind source-input-buffer");
    if (slice.compareOp.getKind() != "sle")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "currently supports only tcrv_rvv.compare {kind = \"sle\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires tcrv_rvv.compare to consume compare lhs/rhs generic load "
          "results");
    if (slice.maskedStandaloneReduceOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires tcrv_rvv.masked_standalone_reduce to consume the mask "
          "produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.sourceValue ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires tcrv_rvv.masked_standalone_reduce to consume the "
          "source-input load and accumulator-input-buffer scalar seed "
          "boundary");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedStandaloneReduceOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.rhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.sourceGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route "
          "requires compare, source load, and masked_standalone_reduce to "
          "consume the selected !tcrv_rvv.vl token");
    if (slice.accumulatorLoadOperation || slice.maskLoadOperation ||
        slice.dotLHSLoadOperation || slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask standalone reduction route does "
          "not support vector accumulator loads, runtime mask_load input, or "
          "dot-input-buffer loads");
  } else if (isRuntimeScalarComputedMaskStandaloneReduction) {
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires the accumulator seed to bind "
          "accumulator-input-buffer");
    if (slice.rhsABI.role != support::RuntimeABIParameterRole::RHSScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires rhs_scalar to bind rhs-scalar-value");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires separate accumulator seed input and "
          "scalar output destination ABI values");
    if (!slice.compareOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires one tcrv_rvv.compare producer");
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires one source-input-buffer tcrv_rvv.load");
    if (!slice.rhsScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires one tcrv_rvv.splat from the "
          "rhs-scalar-value threshold");
    if (slice.sourceABI.role !=
        support::RuntimeABIParameterRole::SourceInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires the source load to bind "
          "source-input-buffer");
    if (slice.compareOp.getKind() != "sle")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route currently supports only "
          "tcrv_rvv.compare {kind = \"sle\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires tcrv_rvv.compare to consume compare lhs "
          "load and splatted rhs scalar threshold");
    if (slice.maskedStandaloneReduceOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires tcrv_rvv.masked_standalone_reduce to "
          "consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.sourceValue ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires tcrv_rvv.masked_standalone_reduce to "
          "consume the source-input load and accumulator-input-buffer scalar "
          "seed boundary");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.rhsScalarSplat.getVl() != slice.setvl.getVl() ||
        slice.maskedStandaloneReduceOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.sourceGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route requires compare, scalar splat, source load, and "
          "masked_standalone_reduce to consume the selected !tcrv_rvv.vl "
          "token");
    if (slice.accumulatorLoadOperation || slice.maskLoadOperation ||
        slice.dotLHSLoadOperation || slice.dotRHSLoadOperation ||
        slice.rhsGenericLoad)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask standalone "
          "reduction route does not support vector accumulator loads, runtime "
          "mask_load input, dot-input-buffer loads, or compare rhs "
          "vector-load fallback");
  } else if (isComputedMaskWideningDotReduceAdd ||
             isComputedMaskStridedInputWideningDotReduceAdd) {
    if (slice.accumulatorABI.role !=
        support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires the accumulator seed to bind "
          "accumulator-input-buffer");
    if (slice.accumulatorBuffer == slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires separate accumulator seed input and output "
          "destination ABI values");
    if (!slice.compareOp)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires one tcrv_rvv.compare producer");
    if (!slice.dotLHSLoadOperation || !slice.dotRHSLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires dot-lhs-input-buffer and dot-rhs-input-buffer "
          "loads");
    if (slice.dotLHSABI.role !=
            support::RuntimeABIParameterRole::DotLHSInputBuffer ||
        slice.dotRHSABI.role !=
            support::RuntimeABIParameterRole::DotRHSInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires dot lhs/rhs loads to bind dedicated dot input ABI "
          "roles");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route currently supports only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires tcrv_rvv.compare to consume compare lhs/rhs "
          "generic load results");
    if (slice.maskedWideningDotReduceOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires tcrv_rvv.masked_widening_dot_reduce to consume the "
          "mask produced by tcrv_rvv.compare");
    if (slice.compareOp.getVl() != slice.setvl.getVl() ||
        slice.maskedWideningDotReduceOp.getVl() != slice.setvl.getVl() ||
        slice.lhsGenericLoad.getVl() != slice.setvl.getVl() ||
        slice.rhsGenericLoad.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires compare, masked dot-reduction, and compare loads "
          "to consume the selected !tcrv_rvv.vl token");
    if (isComputedMaskWideningDotReduceAdd &&
        (slice.dotLHSGenericLoad.getVl() != slice.setvl.getVl() ||
         slice.dotRHSGenericLoad.getVl() != slice.setvl.getVl()))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires dot lhs/rhs unit-stride loads to consume the "
          "selected !tcrv_rvv.vl token");
    if (isComputedMaskStridedInputWideningDotReduceAdd &&
        (slice.lhsStridedLoad.getVl() != slice.setvl.getVl() ||
         slice.rhsStridedLoad.getVl() != slice.setvl.getVl()))
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-input widening "
          "dot-product reduction route requires dot lhs/rhs strided loads to "
          "consume the selected !tcrv_rvv.vl token");
    if (slice.arithmeticLhs != slice.dotLHSValue ||
        slice.arithmeticRhs != slice.dotRHSValue ||
        slice.arithmeticAccumulator != slice.accumulatorBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route requires tcrv_rvv.masked_widening_dot_reduce to consume "
          "dot lhs/rhs source loads and the accumulator-input-buffer scalar "
          "seed boundary");
    if (slice.accumulatorLoadOperation || slice.maskLoadOperation ||
        slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask widening dot-product reduction "
          "route does not support vector accumulator loads, runtime mask_load "
          "input, or source-input-buffer loads");
  } else if (isWideningConversion) {
    if (slice.conversionSource != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening conversion route requires "
          "tcrv_rvv.widening_convert to consume the lhs source load result");
    if (slice.accumulatorLoadOperation || slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV widening conversion route does not support RHS "
          "or accumulator loads");
  } else if (isStridedLoadUnitStore) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-load to unit-stride-store route "
          "requires tcrv_rvv.move to consume the strided source load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV strided-load to unit-stride-store route does "
          "not support RHS or accumulator loads");
  } else if (isUnitLoadStridedStore) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV unit-load to strided-store route requires "
          "tcrv_rvv.move to consume the unit-stride source load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV unit-load to strided-store route does not "
          "support RHS or accumulator loads");
  } else if (isIndexedGatherUnitStore) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route requires tcrv_rvv.move "
          "to consume the indexed data load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed gather route does not support RHS or "
          "accumulator loads");
  } else if (isIndexedScatterUnitLoad) {
    if (slice.arithmeticLhs != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route requires tcrv_rvv.move "
          "to consume the unit-stride source load result");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV indexed scatter route does not support RHS or "
          "accumulator loads");
  } else if (isMaskedUnitLoadStore) {
    if (!slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires one "
          "tcrv_rvv.mask_load");
    if (!slice.maskedLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires one "
          "tcrv_rvv.masked_load");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires an "
          "output-buffer tcrv_rvv.load for old-destination preservation");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires the "
          "old-destination load to consume the same output buffer as "
          "tcrv_rvv.store");
    if (slice.maskedActiveValue != slice.lhsValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route requires "
          "tcrv_rvv.masked_load to load source lanes and consume the "
          "old-destination load as inactive passthrough");
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked memory route does not support an RHS "
          "data load");
  } else if (isMaskedUnitStore) {
    if (!slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route requires one "
          "tcrv_rvv.mask_load");
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route must not load the old "
          "destination; false lanes are preserved by masked_store");
    if (slice.maskedActiveValue != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route requires "
          "tcrv_rvv.masked_store to consume the payload source load value");
    if (slice.rhsLoadOperation || slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked store route does not support RHS or "
          "extra source data loads");
  } else if (isComputedMaskUnitLoadStore) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires one "
          "source-input-buffer tcrv_rvv.masked_load for active source values");
    if (!slice.maskedLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires one "
          "tcrv_rvv.masked_load");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires an "
          "output-buffer tcrv_rvv.load for old-destination preservation");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires the "
          "old-destination load to consume the same output buffer as "
          "tcrv_rvv.store");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route currently supports "
          "only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires "
          "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    if (slice.maskedActiveValue != slice.sourceValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route requires "
          "tcrv_rvv.masked_load to load source-input lanes and consume "
          "old-destination load as inactive passthrough");
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask memory route does not support "
          "runtime mask_load input");
  } else if (isComputedMaskStridedStore) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires one "
          "source-input-buffer tcrv_rvv.load for active source values");
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route must not "
          "load the old destination; false lanes are preserved by "
          "masked_strided_store");
    if (!slice.maskedStridedStoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires one "
          "tcrv_rvv.masked_strided_store op");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route currently "
          "supports only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    if (slice.maskedActiveValue != slice.sourceValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route requires "
          "tcrv_rvv.masked_strided_store to consume the source-input load "
          "as payload value");
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-store route does not "
          "support runtime mask_load input");
  } else if (isComputedMaskStridedLoadUnitStore) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route requires one "
          "source-input-buffer tcrv_rvv.masked_strided_load for active source "
          "values");
    if (!slice.maskedStridedLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route requires one "
          "tcrv_rvv.masked_strided_load op");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route requires an "
          "output-buffer tcrv_rvv.load for old-destination passthrough");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route requires the "
          "old-destination load to consume the same output buffer as "
          "tcrv_rvv.store");
    if (slice.compareOp.getKind() != "slt")
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route currently "
          "supports only tcrv_rvv.compare {kind = \"slt\"}");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route requires "
          "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    if (slice.maskedActiveValue != slice.sourceValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route requires "
          "tcrv_rvv.masked_strided_load to load source-input lanes and "
          "consume old-destination load as inactive passthrough");
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask strided-load route does not "
          "support runtime mask_load input");
  } else if (isComputedMaskIndexedGatherLike) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires one source-input-buffer tcrv_rvv.masked_indexed_load for "
          "active source values");
    if (!slice.maskedIndexedLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires one tcrv_rvv.masked_indexed_load op");
    if (!slice.indexLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires one tcrv_rvv.index_load op for the explicit index vector");
    if (!slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires an output-buffer tcrv_rvv.load for old-destination "
          "passthrough");
    if (slice.accumulatorBuffer != slice.outBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires the old-destination load to consume the same output "
          "buffer as tcrv_rvv.store");
    llvm::StringRef expectedIndexedGatherPredicate =
        isRuntimeScalarComputedMaskIndexedGatherLoadUnitStore ? "sle" : "slt";
    if (slice.compareOp.getKind() != expectedIndexedGatherPredicate)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "currently supports only the operation-specific bounded "
          "tcrv_rvv.compare predicate");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires tcrv_rvv.compare to consume compare lhs/rhs generic load "
          "results");
    if (slice.maskedActiveValue != slice.sourceValue ||
        slice.maskedInactivePassthrough != slice.accumulatorValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route "
          "requires tcrv_rvv.masked_indexed_load to load source-input lanes "
          "and consume old-destination load as inactive passthrough");
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed gather-load route does "
          "not support runtime mask_load input");
  } else if (isComputedMaskIndexedScatterLike) {
    if (!slice.sourceLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "requires one source-input-buffer tcrv_rvv.load for active payload "
          "values");
    if (!slice.maskedIndexedStoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "requires one tcrv_rvv.masked_indexed_store op");
    if (!slice.indexLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "requires one tcrv_rvv.index_load op for the explicit index vector");
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route must "
          "not load the old destination; false lanes are preserved by "
          "masked_indexed_store");
    llvm::StringRef expectedIndexedScatterPredicate =
        isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ? "sle" : "slt";
    if (isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
        !slice.rhsScalarSplat)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask indexed "
          "scatter-store route requires one tcrv_rvv.splat from the "
          "rhs-scalar-value threshold");
    if (isRuntimeScalarComputedMaskIndexedScatterStoreUnitLoad &&
        slice.rhsGenericLoad)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar computed-mask indexed "
          "scatter-store route does not support a vector compare RHS load "
          "fallback");
    if (slice.compareOp.getKind() != expectedIndexedScatterPredicate)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "currently supports only the operation-specific bounded "
          "tcrv_rvv.compare predicate");
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "requires tcrv_rvv.compare to consume compare lhs and selected RHS "
          "producer results");
    if (slice.maskedActiveValue != slice.sourceValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route "
          "requires tcrv_rvv.masked_indexed_store to consume the source-input "
          "load as payload value");
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask indexed scatter-store route does "
          "not support runtime mask_load input");
  } else if (isComputedMaskSegment2LoadLike) {
    if (!slice.maskedSegment2LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route requires one "
          "tcrv_rvv.masked_segment2_load op");
    if (!slice.field0LoadOperation || !slice.field1LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route requires "
          "field0 and field1 old-destination passthrough loads");
    if (!slice.field0StoreOperation || !slice.field1StoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route requires "
          "field0 and field1 stores");
    if (isRuntimeScalarComputedMaskSegment2LoadUnitStore) {
      if (!slice.rhsScalarSplat)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV runtime scalar computed-mask segment2 load "
            "route requires one tcrv_rvv.splat from the rhs-scalar-value "
            "threshold");
      if (slice.rhsGenericLoad)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV runtime scalar computed-mask segment2 load "
            "route does not support a vector compare RHS load fallback");
      if (slice.compareOp.getKind() != "sle")
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV runtime scalar computed-mask segment2 load "
            "route currently supports only tcrv_rvv.compare {kind = \"sle\"}");
      if (slice.compareLhs != slice.lhsValue ||
          slice.compareRhs != slice.rhsValue)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV runtime scalar computed-mask segment2 load "
            "route requires tcrv_rvv.compare to consume lhs load and "
            "rhs_scalar splat results");
    } else {
      if (slice.compareOp.getKind() != "slt")
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV computed-mask segment2 load route currently "
            "supports only tcrv_rvv.compare {kind = \"slt\"}");
      if (slice.compareLhs != slice.lhsValue ||
          slice.compareRhs != slice.rhsValue)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV computed-mask segment2 load route requires "
            "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    }
    if (slice.maskedSegment2LoadOp.getPassthrough0() !=
            slice.field0PassthroughValue ||
        slice.maskedSegment2LoadOp.getPassthrough1() !=
            slice.field1PassthroughValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route requires "
          "masked_segment2_load passthrough operands to come from field0 and "
          "field1 old-destination loads");
    if (slice.field0Store.getValue() != slice.field0LoadedValue ||
        slice.field1Store.getValue() != slice.field1LoadedValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route requires "
          "field stores to consume matching masked_segment2_load field "
          "results");
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 load route does not "
          "support runtime mask_load input");
  } else if (isComputedMaskSegment2StoreLike) {
    if (!slice.maskedSegment2StoreOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store/update route "
          "requires one tcrv_rvv.masked_segment2_store op");
    if (!slice.field0LoadOperation || !slice.field1LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store/update route "
          "requires field0 and field1 payload loads");
    if (slice.accumulatorLoadOperation || slice.sourceLoadOperation ||
        slice.indexLoadOperation || slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store/update route does "
          "not support source, index, mask, or accumulator loads");
    if (isRuntimeScalarComputedMaskSegment2StoreUnitLoad) {
      if (!slice.rhsScalarSplat)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV runtime scalar computed-mask segment2 store "
            "route requires one tcrv_rvv.splat from the rhs-scalar-value "
            "threshold");
      if (slice.rhsGenericLoad)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV runtime scalar computed-mask segment2 store "
            "route does not support a vector compare RHS load fallback");
      if (slice.compareOp.getKind() != "sle")
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV runtime scalar computed-mask segment2 store "
            "route currently supports only tcrv_rvv.compare {kind = \"sle\"}");
    } else if (slice.compareOp.getKind() != "slt") {
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store/update route "
          "currently supports only tcrv_rvv.compare {kind = \"slt\"}");
    }
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store/update route "
          "requires "
          "tcrv_rvv.compare to consume compare lhs/rhs generic load results");
    if (isComputedMaskSegment2UpdateUnitLoad) {
      if (slice.arithmeticLhs != slice.field0LoadedValue ||
          slice.arithmeticRhs != slice.field1LoadedValue ||
          slice.maskedSegment2Store.getField0() != slice.arithmeticResult ||
          slice.maskedSegment2Store.getField1() != slice.field1LoadedValue)
        return makeRVVEmitCRouteProviderError(
            "bounded generic RVV computed-mask segment2 update route requires "
            "the add producer to consume field0/field1 loads and the "
            "masked_segment2_store payload to use add(field0, field1) plus "
            "the original field1 load");
    } else if (slice.maskedSegment2Store.getField0() != slice.field0Value ||
               slice.maskedSegment2Store.getField1() != slice.field1Value) {
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store route requires "
          "masked_segment2_store payload operands to come from field0 and "
          "field1 source loads");
    }
    if (slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask segment2 store/update route does "
          "not support runtime mask_load input");
  } else if (isSegment2DeinterleaveUnitStore) {
    if (slice.field0Store.getValue() != slice.field0Value ||
        slice.field1Store.getValue() != slice.field1Value)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route requires field0 "
          "and field1 stores to consume their matching field move results");
    if (slice.rhsLoadOperation || slice.accumulatorLoadOperation ||
        slice.sourceLoadOperation || slice.indexLoadOperation ||
        slice.maskLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 deinterleave route does not support "
          "RHS, source, index, mask, or accumulator loads");
  } else if (isSegment2InterleaveUnitLoad) {
    if (slice.segment2Store.getField0() != slice.field0Value ||
        slice.segment2Store.getField1() != slice.field1Value)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route requires "
          "segment2_store to consume its matching field source load results");
    if (slice.accumulatorLoadOperation || slice.sourceLoadOperation ||
        slice.indexLoadOperation || slice.maskLoadOperation ||
        slice.segment2LoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV segment2 interleave route does not support "
          "source, index, mask, accumulator, or segment2_load inputs");
  } else if (isRuntimeScalarSplatStore) {
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar splat-store route must not load "
          "lhs or substitute a binary fallback");
    if (!slice.rhsScalarSplat || slice.rhsValue != slice.storeValue ||
        slice.rhsScalarSplat.getVl() != slice.setvl.getVl())
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar splat-store route requires "
          "tcrv_rvv.store to consume the vector produced by tcrv_rvv.splat in "
          "the selected !tcrv_rvv.vl scope");
    if (slice.rhsABI.role != support::RuntimeABIParameterRole::RHSScalarValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV runtime scalar splat-store route requires the "
          "splat scalar to bind rhs-scalar-value");
  } else if (!isF32ClampSelect && !isDequantClampF32Epilogue &&
             (slice.arithmeticLhs != slice.lhsValue ||
              slice.arithmeticRhs != slice.rhsValue)) {
    if (slice.accumulatorLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV non-multiply-accumulate route does not support "
          "an output-buffer accumulator load");
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded generic RVV EmitC route requires selected-body ") +
        operationMnemonic +
        " to consume lhs/rhs generic load or broadcast results");
  }
  if (!isSegment2DeinterleaveUnitStore && !isSegment2InterleaveUnitLoad &&
      !isComputedMaskSegment2LoadUnitStore &&
      !isComputedMaskSegment2StoreLike &&
      !isRuntimeScalarSplatStore &&
      slice.storeValue != slice.arithmeticResult)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires selected-body store to "
        "consume the selected compute result");

  return slice;
}

namespace {
// Restore the pre-split single-namespace overload set so unqualified calls to
// the relocated (sew, lmul) primitive builders resolve identically.
using ::tianchenrv::plugin::rvv::getRVVSelectedBodyArithmeticIntrinsic;
using ::tianchenrv::plugin::rvv::getRVVSelectedBodyMaskAndIntrinsic;
using ::tianchenrv::plugin::rvv::getRVVSelectedBodySelectIntrinsic;

struct RVVOrderedRoleOperations {
  llvm::SmallVector<mlir::Operation *, 12> operations;
  llvm::SmallVector<unsigned, 12> constructionOrders;
};

unsigned getRVVCanonicalRoleOrder(RVVSelectedBodyRouteSlice &slice,
                                  mlir::Operation *op) {
  auto getRuntimeABI =
      [](mlir::Value value) -> tcrv::rvv::RuntimeABIValueOp {
    if (!value)
      return nullptr;
    return value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  };
  auto lhsABI =
      getRuntimeABI(slice.lhsBuffer);
  auto rhsABI =
      getRuntimeABI(slice.rhsBuffer);
  auto dequantScaleABI =
      getRuntimeABI(slice.dequantScale);
  auto lowerBoundABI =
      slice.lowerBoundScalarSplat
          ? getRuntimeABI(slice.lowerBoundScalarSplat.getScalar())
          : nullptr;
  auto upperBoundABI =
      slice.upperBoundScalarSplat
          ? getRuntimeABI(slice.upperBoundScalarSplat.getScalar())
          : nullptr;
  auto secondaryCompareLhsABI =
      getRuntimeABI(slice.secondaryCompareLhsBuffer);
  auto secondaryCompareRhsScalarABI =
      slice.rhsSecondaryScalarSplat
          ? getRuntimeABI(slice.rhsSecondaryScalarSplat.getScalar())
          : nullptr;
  auto trueValueABI =
      getRuntimeABI(slice.trueValueBuffer);
  auto falseValueABI =
      getRuntimeABI(slice.falseValueBuffer);
  auto sourceABI =
      getRuntimeABI(slice.sourceBuffer);
  auto dotLHSABI =
      getRuntimeABI(slice.dotLHSBuffer);
  auto dotRHSABI =
      getRuntimeABI(slice.dotRHSBuffer);
  auto indexABI =
      getRuntimeABI(slice.indexBuffer);
  auto maskABI =
      getRuntimeABI(slice.maskBuffer);
  auto field0ABI =
      getRuntimeABI(slice.field0Buffer);
  auto field1ABI =
      getRuntimeABI(slice.field1Buffer);
  auto accumulatorABI =
      getRuntimeABI(slice.accumulatorBuffer);
  auto outABI =
      getRuntimeABI(slice.outBuffer);
  auto nABI =
      getRuntimeABI(slice.setvl ? slice.setvl.getAvl() : mlir::Value());
  auto lhsStrideABI =
      slice.lhsStride
          ? slice.lhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto rhsStrideABI =
      slice.rhsStride
          ? slice.rhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto sourceStrideABI =
      slice.sourceStride
          ? slice.sourceStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto outStrideABI =
      slice.outStride
          ? slice.outStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  const bool isStrided =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore;
  const bool isStridedLoadUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  const bool isUnitLoadStridedStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
  const bool isIndexedGatherUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  const bool isMaskedUnitLoadStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  const bool isMaskedUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore;
  const bool isComputedMaskUnitLoadStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  const bool isComputedMaskSelect =
      slice.memoryForm == RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect;
  const bool isRuntimeScalarCompareSelect =
      slice.memoryForm == RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect;
  const bool isRuntimeScalarDualCompareMaskAndSelect =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect;
  const bool isRuntimeScalarComputedMaskStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore;
  const bool isRuntimeScalarComputedMaskLoadStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore;
  const bool isComputedMaskStridedStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  const bool isComputedMaskStridedLoad =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore;
  const bool isComputedMaskIndexedGatherLoad =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isComputedMaskIndexedScatterStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskSegment2Load =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2Store =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  const bool isComputedMaskSegment2Update =
      isComputedMaskSegment2Store &&
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isSegment2DeinterleaveUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  const bool isSegment2InterleaveUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  const bool isConversion =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideConversion;
  const bool isDequantization =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideDequantization;
  const bool isRuntimeScalarSplatStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore;
  const bool isF32ClampSelect =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::F32ClampSelect;
  const bool isDequantClampF32Epilogue =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
  const bool isMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd;
  const bool isScalarBroadcastMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
  const bool isComputedMaskedMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd;
  const bool isRuntimeScalarComputedMaskedMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isWideningMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningProductReduceDequantClamp =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isWideningProductReduceDequantize =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      isWideningProductReduceDequantClamp;
  const bool isWideningProductReduce =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isWideningProductReduceDequantize;
  const bool isWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd;
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
  if (lhsABI && op == lhsABI.getOperation())
    return 0;
  if (isRuntimeScalarSplatStore) {
    if (rhsABI && op == rhsABI.getOperation())
      return 0;
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (op == slice.setvl.getOperation())
      return 3;
    if (op == slice.withVL.getOperation())
      return 4;
    if (op == slice.rhsLoadOperation || op == slice.arithmeticOp)
      return 5;
    if (op == slice.storeOperation)
      return 6;
    return 7;
  }
  if (isDequantization) {
    if (dequantScaleABI && op == dequantScaleABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.arithmeticOp)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isDequantClampF32Epilogue) {
    if (dequantScaleABI && op == dequantScaleABI.getOperation())
      return 1;
    if (lowerBoundABI && op == lowerBoundABI.getOperation())
      return 2;
    if (upperBoundABI && op == upperBoundABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (slice.dequantizeOp && op == slice.dequantizeOp.getOperation())
      return 9;
    if (slice.lowerBoundScalarSplat &&
        op == slice.lowerBoundScalarSplat.getOperation())
      return 10;
    if (slice.upperBoundScalarSplat &&
        op == slice.upperBoundScalarSplat.getOperation())
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (slice.selectOp && op == slice.selectOp.getOperation())
      return 13;
    if (slice.secondaryCompareOp &&
        op == slice.secondaryCompareOp.getOperation())
      return 14;
    if (slice.secondarySelectOp &&
        op == slice.secondarySelectOp.getOperation())
      return 15;
    if (op == slice.storeOperation)
      return 16;
    return 17;
  }
  if (isComputedMaskedMAcc) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (dotLHSABI && op == dotLHSABI.getOperation())
      return 2;
    if (dotRHSABI && op == dotRHSABI.getOperation())
      return 3;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 4;
    if (outABI && op == outABI.getOperation())
      return 5;
    if (nABI && op == nABI.getOperation())
      return 6;
    if (op == slice.setvl.getOperation())
      return 7;
    if (op == slice.withVL.getOperation())
      return 8;
    if (op == slice.lhsLoadOperation)
      return 9;
    if (op == slice.rhsLoadOperation)
      return 10;
    if (op == slice.dotLHSLoadOperation)
      return 11;
    if (op == slice.dotRHSLoadOperation)
      return 12;
    if (op == slice.accumulatorLoadOperation)
      return 13;
    if (op == slice.compareOp.getOperation())
      return 14;
    if (op == slice.arithmeticOp)
      return 15;
    if (op == slice.storeOperation)
      return 16;
    return 17;
  }
  if (isRuntimeScalarComputedMaskedMAcc) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (dotLHSABI && op == dotLHSABI.getOperation())
      return 2;
    if (dotRHSABI && op == dotRHSABI.getOperation())
      return 3;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 4;
    if (outABI && op == outABI.getOperation())
      return 5;
    if (nABI && op == nABI.getOperation())
      return 6;
    if (op == slice.setvl.getOperation())
      return 7;
    if (op == slice.withVL.getOperation())
      return 8;
    if (op == slice.lhsLoadOperation)
      return 9;
    if (op == slice.rhsLoadOperation)
      return 10;
    if (op == slice.dotLHSLoadOperation)
      return 11;
    if (op == slice.dotRHSLoadOperation)
      return 12;
    if (op == slice.accumulatorLoadOperation)
      return 13;
    if (op == slice.compareOp.getOperation())
      return 14;
    if (op == slice.arithmeticOp)
      return 15;
    if (op == slice.storeOperation)
      return 16;
    return 17;
  }
  if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (dotRHSABI && op == dotRHSABI.getOperation())
      return 3;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 4;
    if (indexABI && op == indexABI.getOperation())
      return 5;
    if (outABI && op == outABI.getOperation())
      return 6;
    if (nABI && op == nABI.getOperation())
      return 7;
    if (op == slice.setvl.getOperation())
      return 8;
    if (op == slice.withVL.getOperation())
      return 9;
    if (op == slice.lhsLoadOperation)
      return 10;
    if (op == slice.rhsLoadOperation)
      return 11;
    if (op == slice.dotRHSLoadOperation)
      return 12;
    if (op == slice.accumulatorLoadOperation)
      return 13;
    if (op == slice.oldDestinationLoadOperation)
      return 14;
    if (op == slice.indexLoadOperation)
      return 15;
    if (op == slice.compareOp.getOperation())
      return 16;
    if (op == slice.sourceLoadOperation)
      return 17;
    if (op == slice.arithmeticOp)
      return 18;
    if (op == slice.storeOperation)
      return 19;
    return 20;
  }
  if (isComputedMaskSelect) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (trueValueABI && op == trueValueABI.getOperation())
      return 2;
    if (falseValueABI && op == falseValueABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.trueValueLoadOperation)
      return 10;
    if (op == slice.falseValueLoadOperation)
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (op == slice.arithmeticOp)
      return 13;
    if (op == slice.storeOperation)
      return 14;
    return 15;
  }
  if (isRuntimeScalarCompareSelect) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (trueValueABI && op == trueValueABI.getOperation())
      return 2;
    if (falseValueABI && op == falseValueABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.trueValueLoadOperation)
      return 10;
    if (op == slice.falseValueLoadOperation)
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (op == slice.arithmeticOp)
      return 13;
    if (op == slice.storeOperation)
      return 14;
    return 15;
  }
  if (isRuntimeScalarDualCompareMaskAndSelect) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (secondaryCompareLhsABI && op == secondaryCompareLhsABI.getOperation())
      return 2;
    if (secondaryCompareRhsScalarABI &&
        op == secondaryCompareRhsScalarABI.getOperation())
      return 3;
    if (trueValueABI && op == trueValueABI.getOperation())
      return 4;
    if (falseValueABI && op == falseValueABI.getOperation())
      return 5;
    if (outABI && op == outABI.getOperation())
      return 6;
    if (nABI && op == nABI.getOperation())
      return 7;
    if (op == slice.setvl.getOperation())
      return 8;
    if (op == slice.withVL.getOperation())
      return 9;
    if (op == slice.lhsLoadOperation)
      return 10;
    if (op == slice.rhsLoadOperation)
      return 11;
    if (op == slice.secondaryCompareLhsLoadOperation)
      return 12;
    if (slice.rhsSecondaryScalarSplat &&
        op == slice.rhsSecondaryScalarSplat.getOperation())
      return 13;
    if (op == slice.trueValueLoadOperation)
      return 14;
    if (op == slice.falseValueLoadOperation)
      return 15;
    if (op == slice.compareOp.getOperation())
      return 16;
    if (op == slice.secondaryCompareOp.getOperation())
      return 17;
    if (op == slice.maskAndOp.getOperation())
      return 18;
    if (op == slice.arithmeticOp)
      return 19;
    if (op == slice.storeOperation)
      return 20;
    return 21;
  }
  if (isF32ClampSelect) {
    if (lowerBoundABI && op == lowerBoundABI.getOperation())
      return 1;
    if (upperBoundABI && op == upperBoundABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (slice.lowerBoundScalarSplat &&
        op == slice.lowerBoundScalarSplat.getOperation())
      return 8;
    if (slice.upperBoundScalarSplat &&
        op == slice.upperBoundScalarSplat.getOperation())
      return 9;
    if (op == slice.compareOp.getOperation())
      return 10;
    if (slice.selectOp && op == slice.selectOp.getOperation())
      return 11;
    if (slice.secondaryCompareOp &&
        op == slice.secondaryCompareOp.getOperation())
      return 12;
    if (slice.secondarySelectOp &&
        op == slice.secondarySelectOp.getOperation())
      return 13;
    if (op == slice.storeOperation)
      return 14;
    return 15;
  }
  if (isRuntimeScalarComputedMaskStore) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.sourceLoadOperation)
      return 9;
    if (op == slice.compareOp.getOperation())
      return 10;
    if (op == slice.storeOperation)
      return 11;
    return 12;
  }
  if (isRuntimeScalarComputedMaskLoadStore) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.accumulatorLoadOperation)
      return 9;
    if (op == slice.compareOp.getOperation())
      return 10;
    if (op == slice.sourceLoadOperation)
      return 11;
    if (op == slice.storeOperation)
      return 12;
    return 13;
  }
  if (isComputedMaskStridedInputWideningDotReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (dotLHSABI && op == dotLHSABI.getOperation())
      return 2;
    if (dotRHSABI && op == dotRHSABI.getOperation())
      return 3;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 4;
    if (outABI && op == outABI.getOperation())
      return 5;
    if (nABI && op == nABI.getOperation())
      return 6;
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 7;
    if (rhsStrideABI && op == rhsStrideABI.getOperation())
      return 8;
    if (op == slice.setvl.getOperation())
      return 9;
    if (op == slice.withVL.getOperation())
      return 10;
    if (op == slice.lhsLoadOperation)
      return 11;
    if (op == slice.rhsLoadOperation)
      return 12;
    if (op == slice.dotLHSLoadOperation)
      return 13;
    if (op == slice.dotRHSLoadOperation)
      return 14;
    if (op == slice.compareOp.getOperation())
      return 15;
    if (op == slice.arithmeticOp)
      return 16;
    if (op == slice.storeOperation)
      return 17;
    return 18;
  }
  if (isComputedMaskWideningDotReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (dotLHSABI && op == dotLHSABI.getOperation())
      return 2;
    if (dotRHSABI && op == dotRHSABI.getOperation())
      return 3;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 4;
    if (outABI && op == outABI.getOperation())
      return 5;
    if (nABI && op == nABI.getOperation())
      return 6;
    if (op == slice.setvl.getOperation())
      return 7;
    if (op == slice.withVL.getOperation())
      return 8;
    if (op == slice.lhsLoadOperation)
      return 9;
    if (op == slice.rhsLoadOperation)
      return 10;
    if (op == slice.dotLHSLoadOperation)
      return 11;
    if (op == slice.dotRHSLoadOperation)
      return 12;
    if (op == slice.compareOp.getOperation())
      return 13;
    if (op == slice.arithmeticOp)
      return 14;
    if (op == slice.storeOperation)
      return 15;
    return 16;
  }
  if (isWideningProductReduceDequantClamp) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (dequantScaleABI && op == dequantScaleABI.getOperation())
      return 3;
    if (lowerBoundABI && op == lowerBoundABI.getOperation())
      return 4;
    if (upperBoundABI && op == upperBoundABI.getOperation())
      return 5;
    if (outABI && op == outABI.getOperation())
      return 6;
    if (nABI && op == nABI.getOperation())
      return 7;
    if (op == slice.setvl.getOperation())
      return 8;
    if (op == slice.withVL.getOperation())
      return 9;
    if (op == slice.lhsLoadOperation)
      return 10;
    if (op == slice.rhsLoadOperation)
      return 11;
    if (op == productSlotOperation(slice))
      return 12;
    if (op == slice.standaloneReduceOp.getOperation())
      return 13;
    if (slice.gearboxCrossRegionHandoffOp &&
        op == slice.gearboxCrossRegionHandoffOp.getOperation())
      return 14;
    if (slice.gearboxConsumerWithVL &&
        op == slice.gearboxConsumerWithVL.getOperation())
      return 15;
    // Single-scope typed dequant-clamp body (Stage 3 flip): no handoff op and no
    // consumer with_vl scope, so the post-reduce clamp role orders close the 14/15
    // gap the deleted two-scope ops would have occupied (contiguous with the
    // chain-driven expected step sequence).
    const unsigned clampTwoScopeExtra =
        slice.gearboxCrossRegionHandoffOp ? 2u : 0u;
    if (slice.dequantizeOp && op == slice.dequantizeOp.getOperation())
      return 14u + clampTwoScopeExtra;
    if (slice.lowerBoundScalarSplat &&
        op == slice.lowerBoundScalarSplat.getOperation())
      return 15u + clampTwoScopeExtra;
    if (slice.upperBoundScalarSplat &&
        op == slice.upperBoundScalarSplat.getOperation())
      return 16u + clampTwoScopeExtra;
    if (op == slice.compareOp.getOperation())
      return 17u + clampTwoScopeExtra;
    if (slice.selectOp && op == slice.selectOp.getOperation())
      return 18u + clampTwoScopeExtra;
    if (slice.secondaryCompareOp &&
        op == slice.secondaryCompareOp.getOperation())
      return 19u + clampTwoScopeExtra;
    if (slice.secondarySelectOp &&
        op == slice.secondarySelectOp.getOperation())
      return 20u + clampTwoScopeExtra;
    if (op == slice.storeOperation)
      return 21u + clampTwoScopeExtra;
    return 22u + clampTwoScopeExtra;
  }
  if (isWideningProductReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (isWideningProductReduceDequantize && dequantScaleABI &&
        op == dequantScaleABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return isWideningProductReduceDequantize ? 4 : 3;
    if (nABI && op == nABI.getOperation())
      return isWideningProductReduceDequantize ? 5 : 4;
    if (op == slice.setvl.getOperation())
      return isWideningProductReduceDequantize ? 6 : 5;
    if (op == slice.withVL.getOperation())
      return isWideningProductReduceDequantize ? 7 : 6;
    if (op == slice.lhsLoadOperation)
      return isWideningProductReduceDequantize ? 8 : 7;
    if (op == slice.rhsLoadOperation)
      return isWideningProductReduceDequantize ? 9 : 8;
    if (op == productSlotOperation(slice))
      return isWideningProductReduceDequantize ? 10 : 9;
    if (isWideningProductReduceDequantize &&
        op == slice.standaloneReduceOp.getOperation())
      return 11;
    if (isWideningProductReduceDequantize &&
        slice.gearboxCrossRegionHandoffOp &&
        op == slice.gearboxCrossRegionHandoffOp.getOperation())
      return 12;
    if (isWideningProductReduceDequantize && slice.gearboxConsumerWithVL &&
        op == slice.gearboxConsumerWithVL.getOperation())
      return 13;
    // Single-scope typed dequant body (Stage 3 flip): no handoff op and no
    // consumer with_vl scope, so the dequantize/store role orders close up the
    // gap (12/13) the deleted two-scope ops would have occupied, staying
    // contiguous with the chain-driven expected step sequence.
    const unsigned dequantTwoScopeExtra =
        slice.gearboxCrossRegionHandoffOp ? 2u : 0u;
    if (op == slice.arithmeticOp)
      return isWideningProductReduceDequantize ? (12u + dequantTwoScopeExtra)
                                               : 10;
    if (op == slice.storeOperation)
      return isWideningProductReduceDequantize ? (13u + dequantTwoScopeExtra)
                                               : 11;
    return isWideningProductReduceDequantize ? (14u + dequantTwoScopeExtra) : 12;
  }
  if (isComputedMaskStandaloneReduction) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.sourceLoadOperation)
      return 10;
    if (op == slice.compareOp.getOperation())
      return 11;
    if (op == slice.arithmeticOp)
      return 12;
    if (op == slice.storeOperation)
      return 13;
    return 14;
  }
  if (isRuntimeScalarComputedMaskStandaloneReduction) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.sourceLoadOperation)
      return 10;
    if (op == slice.compareOp.getOperation())
      return 11;
    if (op == slice.arithmeticOp)
      return 12;
    if (op == slice.storeOperation)
      return 13;
    return 14;
  }
  if (isComputedMaskStridedLoad) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (sourceStrideABI && op == sourceStrideABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.accumulatorLoadOperation)
      return 10;
    if (op == slice.compareOp.getOperation())
      return 11;
    if (op == slice.sourceLoadOperation)
      return 12;
    if (op == slice.storeOperation)
      return 13;
    return 14;
  }
  if (isComputedMaskIndexedGatherLoad) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (indexABI && op == indexABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.accumulatorLoadOperation)
      return 10;
    if (op == slice.indexLoadOperation)
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (op == slice.sourceLoadOperation)
      return 13;
    if (op == slice.storeOperation)
      return 14;
    return 15;
  }
  if (isComputedMaskIndexedScatterStore) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (indexABI && op == indexABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.sourceLoadOperation)
      return 10;
    if (op == slice.indexLoadOperation)
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (op == slice.storeOperation)
      return 13;
    return 14;
  }
  if (isComputedMaskSegment2Load) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (field0ABI && op == field0ABI.getOperation())
      return 3;
    if (field1ABI && op == field1ABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.field0LoadOperation)
      return 10;
    if (op == slice.field1LoadOperation)
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (op == slice.maskedSegment2LoadOperation)
      return 13;
    if (op == slice.field0StoreOperation)
      return 14;
    if (op == slice.field1StoreOperation)
      return 15;
    return 16;
  }
  if (isComputedMaskSegment2Store) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (field0ABI && op == field0ABI.getOperation())
      return 2;
    if (field1ABI && op == field1ABI.getOperation())
      return 3;
    if (outABI && op == outABI.getOperation())
      return 4;
    if (nABI && op == nABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.field0LoadOperation)
      return 10;
    if (op == slice.field1LoadOperation)
      return 11;
    if (op == slice.compareOp.getOperation())
      return 12;
    if (isComputedMaskSegment2Update && op == slice.arithmeticOp)
      return 13;
    if (op == slice.maskedSegment2StoreOperation)
      return isComputedMaskSegment2Update ? 14 : 13;
    return isComputedMaskSegment2Update ? 15 : 14;
  }
  if (isScalarBroadcastMAcc) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.accumulatorLoadOperation)
      return 9;
    if (op == slice.arithmeticOp)
      return 10;
    if (op == slice.storeOperation)
      return 11;
    return 12;
  }
  if (isMAcc || isWideningMAcc) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.accumulatorLoadOperation)
      return 9;
    if (op == slice.arithmeticOp)
      return 10;
    if (op == slice.storeOperation)
      return 11;
    return 12;
  }
  if (isWideningDotReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.arithmeticOp)
      return 9;
    if (op == slice.storeOperation)
      return 10;
    return 11;
  }
  if (isStridedInputWideningDotReduce) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 5;
    if (rhsStrideABI && op == rhsStrideABI.getOperation())
      return 6;
    if (op == slice.setvl.getOperation())
      return 7;
    if (op == slice.withVL.getOperation())
      return 8;
    if (op == slice.lhsLoadOperation)
      return 9;
    if (op == slice.rhsLoadOperation)
      return 10;
    if (op == slice.arithmeticOp)
      return 11;
    if (op == slice.storeOperation)
      return 12;
    return 13;
  }
  if (isStandaloneReduction) {
    if (accumulatorABI && op == accumulatorABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.arithmeticOp)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isSegment2InterleaveUnitLoad) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.rhsLoadOperation)
      return 7;
    if (op == slice.segment2StoreOperation)
      return 8;
    return 9;
  }
  if (isSegment2DeinterleaveUnitStore) {
    if (field0ABI && op == field0ABI.getOperation())
      return 1;
    if (field1ABI && op == field1ABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.segment2LoadOperation)
      return 6;
    if (op == slice.field0MoveOperation)
      return 7;
    if (op == slice.field1MoveOperation)
      return 8;
    if (op == slice.field0StoreOperation)
      return 9;
    if (op == slice.field1StoreOperation)
      return 10;
    return 11;
  }
  if (isComputedMaskUnitLoadStore) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (op == slice.setvl.getOperation())
      return 5;
    if (op == slice.withVL.getOperation())
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.rhsLoadOperation)
      return 8;
    if (op == slice.accumulatorLoadOperation)
      return 9;
    if (op == slice.compareOp.getOperation())
      return 10;
    if (op == slice.maskedLoadOperation || op == slice.arithmeticOp)
      return 11;
    if (op == slice.storeOperation)
      return 12;
    return 13;
  }
  if (isComputedMaskStridedStore) {
    if (rhsABI && op == rhsABI.getOperation())
      return 1;
    if (sourceABI && op == sourceABI.getOperation())
      return 2;
    if (outABI && op == outABI.getOperation())
      return 3;
    if (nABI && op == nABI.getOperation())
      return 4;
    if (outStrideABI && op == outStrideABI.getOperation())
      return 5;
    if (op == slice.setvl.getOperation())
      return 6;
    if (op == slice.withVL.getOperation())
      return 7;
    if (op == slice.lhsLoadOperation)
      return 8;
    if (op == slice.rhsLoadOperation)
      return 9;
    if (op == slice.sourceLoadOperation)
      return 10;
    if (op == slice.compareOp.getOperation())
      return 11;
    if (op == slice.arithmeticOp || op == slice.storeOperation)
      return 12;
    return 13;
  }
  if (isMaskedUnitLoadStore) {
    if (maskABI && op == maskABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.maskLoadOperation)
      return 6;
    if (op == slice.accumulatorLoadOperation)
      return 7;
    if (op == slice.maskedLoadOperation || op == slice.arithmeticOp)
      return 8;
    if (op == slice.storeOperation)
      return 9;
    return 10;
  }
  if (isMaskedUnitStore) {
    if (maskABI && op == maskABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.maskLoadOperation)
      return 6;
    if (op == slice.lhsLoadOperation)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isIndexedGatherUnitStore) {
    if (indexABI && op == indexABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.indexLoadOperation)
      return 6;
    if (op == slice.indexedLoadOperation)
      return 7;
    if (op == slice.arithmeticOp)
      return 8;
    if (op == slice.storeOperation)
      return 9;
    return 10;
  }
  if (isIndexedScatterUnitLoad) {
    if (indexABI && op == indexABI.getOperation())
      return 1;
    if (outABI && op == outABI.getOperation())
      return 2;
    if (nABI && op == nABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.indexLoadOperation)
      return 7;
    if (op == slice.arithmeticOp)
      return 8;
    if (op == slice.indexedStoreOperation)
      return 9;
    return 10;
  }
  if (isStridedLoadUnitStore) {
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.arithmeticOp)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isUnitLoadStridedStore) {
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (outStrideABI && op == outStrideABI.getOperation())
      return 3;
    if (op == slice.setvl.getOperation())
      return 4;
    if (op == slice.withVL.getOperation())
      return 5;
    if (op == slice.lhsLoadOperation)
      return 6;
    if (op == slice.arithmeticOp)
      return 7;
    if (op == slice.storeOperation)
      return 8;
    return 9;
  }
  if (isConversion) {
    if (outABI && op == outABI.getOperation())
      return 1;
    if (nABI && op == nABI.getOperation())
      return 2;
    if (op == slice.setvl.getOperation())
      return 3;
    if (op == slice.withVL.getOperation())
      return 4;
    if (op == slice.lhsLoadOperation)
      return 5;
    if (op == slice.arithmeticOp)
      return 6;
    if (op == slice.storeOperation)
      return 7;
    return 8;
  }
  if (rhsABI && op == rhsABI.getOperation())
    return 1;
  if (outABI && op == outABI.getOperation())
    return 2;
  if (nABI && op == nABI.getOperation())
    return 3;
  if (isStrided) {
    if (lhsStrideABI && op == lhsStrideABI.getOperation())
      return 4;
    if (rhsStrideABI && op == rhsStrideABI.getOperation())
      return 5;
    if (outStrideABI && op == outStrideABI.getOperation())
      return 6;
  }
  if (op == slice.setvl.getOperation())
    return isStrided ? 7 : 4;
  if (op == slice.withVL.getOperation())
    return isStrided ? 8 : 5;
  if (op == slice.lhsLoadOperation)
    return isStrided ? 9 : 6;
  if (op == slice.rhsLoadOperation)
    return isStrided ? 10 : 7;
  if (isStrided && op == slice.arithmeticOp)
    return 11;
  if (isStrided && op == slice.storeOperation)
    return 12;
  if (slice.accumulatorLoadOperation && op == slice.accumulatorLoadOperation)
    return 8;
  if (slice.compareOp && op == slice.compareOp.getOperation())
    return 8;
  if (op == slice.arithmeticOp)
    return (slice.compareOp || slice.accumulatorLoadOperation) ? 9 : 8;
  if (op == slice.storeOperation)
    return (slice.compareOp || slice.accumulatorLoadOperation) ? 10 : 9;
  return (slice.compareOp || slice.accumulatorLoadOperation) ? 11 : 10;
}

RVVOrderedRoleOperations
collectRVVRoleOperationsInBodyOrder(tcrv::exec::VariantOp variant,
                                    RVVSelectedBodyRouteSlice &slice) {
  RVVOrderedRoleOperations ordered;
  if (!variant || variant.getBody().empty())
    return ordered;

  auto recordRoleOp = [&](mlir::Operation &op) {
    if (op.getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::VSetVLRegionMarkerOp>(op))
      return;
    ordered.operations.push_back(&op);
    ordered.constructionOrders.push_back(getRVVCanonicalRoleOrder(slice, &op));
  };

  for (mlir::Operation &op : variant.getBody().front()) {
    if (op.getName().getDialectNamespace() != "tcrv_rvv")
      continue;
    if (llvm::isa<tcrv::rvv::VSetVLRegionMarkerOp>(op))
      continue;
    recordRoleOp(op);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op)) {
      for (mlir::Operation &nested : withVL.getBody().front()) {
        recordRoleOp(nested);
        if (slice.gearboxConsumerWithVL &&
            &nested == slice.gearboxConsumerWithVL.getOperation())
          for (mlir::Operation &consumerNested :
               slice.gearboxConsumerWithVL.getBody().front())
            recordRoleOp(consumerNested);
      }
    }
  }
  return ordered;
}

} // namespace

llvm::Error verifySelectedRVVRoleSequence(
    RVVSelectedBodyRouteSlice &slice,
    const VariantEmitCLowerableRequest &request,
    const RVVSelectedBodyConstructionRoute &constructionRoute) {
  auto getRuntimeABI =
      [](mlir::Value value) -> tcrv::rvv::RuntimeABIValueOp {
    if (!value)
      return nullptr;
    return value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  };
  auto lhsABI =
      getRuntimeABI(slice.lhsBuffer);
  auto rhsABI =
      getRuntimeABI(slice.rhsBuffer);
  auto dequantScaleABI =
      getRuntimeABI(slice.dequantScale);
  auto lowerBoundABI =
      slice.lowerBoundScalarSplat
          ? getRuntimeABI(slice.lowerBoundScalarSplat.getScalar())
          : nullptr;
  auto upperBoundABI =
      slice.upperBoundScalarSplat
          ? getRuntimeABI(slice.upperBoundScalarSplat.getScalar())
          : nullptr;
  auto secondaryCompareLhsABI =
      getRuntimeABI(slice.secondaryCompareLhsBuffer);
  auto secondaryCompareRhsScalarABI =
      slice.rhsSecondaryScalarSplat
          ? getRuntimeABI(slice.rhsSecondaryScalarSplat.getScalar())
          : nullptr;
  auto trueValueABI =
      getRuntimeABI(slice.trueValueBuffer);
  auto falseValueABI =
      getRuntimeABI(slice.falseValueBuffer);
  auto sourceABI =
      getRuntimeABI(slice.sourceBuffer);
  auto dotLHSABI =
      getRuntimeABI(slice.dotLHSBuffer);
  auto dotRHSABI =
      getRuntimeABI(slice.dotRHSBuffer);
  auto indexABI =
      getRuntimeABI(slice.indexBuffer);
  auto maskABI =
      getRuntimeABI(slice.maskBuffer);
  auto field0ABI =
      getRuntimeABI(slice.field0Buffer);
  auto field1ABI =
      getRuntimeABI(slice.field1Buffer);
  auto accumulatorABI =
      getRuntimeABI(slice.accumulatorBuffer);
  auto outABI =
      getRuntimeABI(slice.outBuffer);
  auto nABI =
      getRuntimeABI(slice.setvl ? slice.setvl.getAvl() : mlir::Value());
  auto lhsStrideABI =
      slice.lhsStride
          ? slice.lhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto rhsStrideABI =
      slice.rhsStride
          ? slice.rhsStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto sourceStrideABI =
      slice.sourceStride
          ? slice.sourceStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  auto outStrideABI =
      slice.outStride
          ? slice.outStride.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>()
          : nullptr;
  const bool isStrided =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore;
  const bool isStridedLoadUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  const bool isUnitLoadStridedStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
  const bool isIndexedGatherUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  const bool isIndexedScatterUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  const bool isMaskedUnitLoadStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  const bool isMaskedUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore;
  const bool isComputedMaskUnitLoadStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  const bool isComputedMaskSelect =
      slice.memoryForm == RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect;
  const bool isRuntimeScalarCompareSelect =
      slice.memoryForm == RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect;
  const bool isRuntimeScalarDualCompareMaskAndSelect =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect;
  const bool isRuntimeScalarComputedMaskStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore;
  const bool isRuntimeScalarComputedMaskLoadStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore;
  const bool isComputedMaskStridedStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  const bool isComputedMaskStridedLoad =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore;
  const bool isComputedMaskIndexedGatherLoad =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isComputedMaskIndexedScatterStore =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  const bool isComputedMaskSegment2Load =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2Store =
      slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  const bool isSegment2DeinterleaveUnitStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  const bool isSegment2InterleaveUnitLoad =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  const bool isConversion =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideConversion;
  const bool isDequantization =
      slice.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideDequantization;
  const bool isRuntimeScalarSplatStore =
      slice.memoryForm == RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore;
  const bool isF32ClampSelect =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::F32ClampSelect;
  const bool isDequantClampF32Epilogue =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
  const bool isMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd;
  const bool isScalarBroadcastMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
  const bool isComputedMaskedMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd;
  const bool isRuntimeScalarComputedMaskedMAcc =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  const bool isWideningMAcc =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningProductReduceDequantClamp =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isWideningProductReduceDequantize =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      isWideningProductReduceDequantClamp;
  const bool isWideningProductReduce =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isWideningProductReduceDequantize;
  const bool isWideningDotReduce =
      slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::WideningDotReduceAdd;
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
  if ((!lhsABI && !isRuntimeScalarSplatStore) ||
      (!isConversion && !isDequantization && !isStridedLoadUnitStore &&
       !isIndexedGatherUnitStore && !isIndexedScatterUnitLoad &&
       !isMaskedUnitLoadStore && !isMaskedUnitStore &&
       !isComputedMaskUnitLoadStore &&
       !isComputedMaskSelect &&
       !isRuntimeScalarCompareSelect &&
       !isRuntimeScalarDualCompareMaskAndSelect &&
       !isRuntimeScalarComputedMaskStore &&
       !isRuntimeScalarComputedMaskLoadStore &&
       !isComputedMaskStridedStore &&
       !isComputedMaskStridedLoad &&
       !isComputedMaskIndexedGatherLoad &&
       !isComputedMaskIndexedScatterStore &&
       !isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
       !isComputedMaskSegment2Load &&
       !isComputedMaskSegment2Store &&
       !isComputedMaskedMAcc &&
       !isRuntimeScalarComputedMaskedMAcc &&
       !isScalarBroadcastMAcc &&
       !isSegment2DeinterleaveUnitStore && !isSegment2InterleaveUnitLoad &&
       !isUnitLoadStridedStore &&
       !isRuntimeScalarSplatStore &&
       !isF32ClampSelect &&
       !isDequantClampF32Epilogue &&
       !isStandaloneReduction &&
       !isComputedMaskStandaloneReduction &&
       !isRuntimeScalarComputedMaskStandaloneReduction &&
       !rhsABI) ||
      (!isSegment2DeinterleaveUnitStore && !outABI) ||
      !nABI || (isStrided && (!lhsStrideABI || !rhsStrideABI || !outStrideABI)) ||
      (isStridedLoadUnitStore && !lhsStrideABI) ||
      (isUnitLoadStridedStore && !outStrideABI) ||
      ((isIndexedGatherUnitStore || isIndexedScatterUnitLoad) && !indexABI) ||
      ((isMaskedUnitLoadStore || isMaskedUnitStore) && !maskABI) ||
      (isComputedMaskUnitLoadStore && (!rhsABI || !sourceABI)) ||
      (isComputedMaskSelect &&
       (!rhsABI || !trueValueABI || !falseValueABI)) ||
      (isRuntimeScalarCompareSelect &&
       (!rhsABI || !trueValueABI || !falseValueABI)) ||
      (isRuntimeScalarDualCompareMaskAndSelect &&
       (!rhsABI || !secondaryCompareLhsABI || !secondaryCompareRhsScalarABI ||
        !trueValueABI || !falseValueABI)) ||
      (isRuntimeScalarComputedMaskStore && (!rhsABI || !sourceABI)) ||
      (isRuntimeScalarComputedMaskLoadStore && (!rhsABI || !sourceABI)) ||
      (isComputedMaskStridedStore &&
       (!rhsABI || !sourceABI || !outStrideABI)) ||
      (isComputedMaskStridedLoad &&
       (!rhsABI || !sourceABI || !sourceStrideABI)) ||
      (isComputedMaskIndexedGatherLoad &&
       (!rhsABI || !sourceABI || !indexABI)) ||
      (isComputedMaskIndexedScatterStore &&
       (!rhsABI || !sourceABI || !indexABI)) ||
      (isComputedMaskSegment2Load &&
       (!rhsABI || !sourceABI || !field0ABI || !field1ABI)) ||
      (isComputedMaskSegment2Store &&
       (!rhsABI || !field0ABI || !field1ABI || !outABI)) ||
      (isMAcc && (!rhsABI || !accumulatorABI)) ||
      (isScalarBroadcastMAcc && (!rhsABI || !accumulatorABI)) ||
      (isComputedMaskedMAcc &&
       (!rhsABI || !dotLHSABI || !dotRHSABI || !accumulatorABI)) ||
      (isRuntimeScalarComputedMaskedMAcc &&
       (!rhsABI || !dotLHSABI || !dotRHSABI || !accumulatorABI)) ||
      (isRuntimeScalarComputedMaskIndexedGatherMAccScatter &&
       (!rhsABI || !sourceABI || !dotRHSABI || !accumulatorABI || !indexABI ||
        !outABI)) ||
      (isWideningMAcc && (!rhsABI || !accumulatorABI)) ||
      (isWideningProductReduce &&
       (!rhsABI || !accumulatorABI ||
        (isWideningProductReduceDequantize && !dequantScaleABI))) ||
      (isWideningDotReduce && (!rhsABI || !accumulatorABI)) ||
      (isStandaloneReduction && !accumulatorABI) ||
      (isComputedMaskStandaloneReduction &&
       (!rhsABI || !sourceABI || !accumulatorABI)) ||
      (isRuntimeScalarComputedMaskStandaloneReduction &&
       (!rhsABI || !sourceABI || !accumulatorABI)) ||
      (isStridedInputWideningDotReduce &&
       (!rhsABI || !accumulatorABI || !lhsStrideABI || !rhsStrideABI)) ||
      (isComputedMaskWideningDotReduce &&
       (!rhsABI || !dotLHSABI || !dotRHSABI || !accumulatorABI)) ||
      (isComputedMaskStridedInputWideningDotReduce &&
       (!rhsABI || !dotLHSABI || !dotRHSABI || !accumulatorABI ||
        !lhsStrideABI || !rhsStrideABI)) ||
      (isRuntimeScalarSplatStore && (!rhsABI || !outABI)) ||
      (isF32ClampSelect && (!lowerBoundABI || !upperBoundABI || !outABI)) ||
      (isDequantClampF32Epilogue &&
       (!dequantScaleABI || !lowerBoundABI || !upperBoundABI || !outABI)) ||
      (isDequantization && (!dequantScaleABI || !outABI)) ||
      (isSegment2DeinterleaveUnitStore && (!field0ABI || !field1ABI)) ||
      (isSegment2InterleaveUnitLoad && (!rhsABI || !outABI)))
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires runtime ABI values "
        "to be explicit tcrv_rvv.runtime_abi_value ops");

  if (isStridedLoadUnitStore) {
    if (slice.lhsABI.role !=
        support::RuntimeABIParameterRole::SourceInputBuffer)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV byte-strided-load to unit-stride-store route "
          "requires source-input-buffer runtime ABI value");
    if (slice.lhsStrideABI.role !=
        support::RuntimeABIParameterRole::SourceByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV byte-strided-load to unit-stride-store route "
          "requires source-byte-stride runtime ABI value");
  }
  if (isUnitLoadStridedStore) {
    if (slice.outStrideABI.role !=
        support::RuntimeABIParameterRole::DestinationByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV unit-load to byte-strided-store route "
          "requires destination-byte-stride runtime ABI value");
  }
  if (isComputedMaskStridedStore) {
    if (slice.outStrideABI.role !=
        support::RuntimeABIParameterRole::DestinationByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask byte-strided-store route "
          "requires destination-byte-stride runtime ABI value");
  }
  if (isComputedMaskStridedLoad) {
    if (slice.sourceStrideABI.role !=
        support::RuntimeABIParameterRole::SourceByteStride)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV computed-mask byte-strided-load route "
          "requires source-byte-stride runtime ABI value");
  }

  mlir::ArrayAttr requires =
      request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires non-empty selected "
        "variant requires metadata");

  RVVOrderedRoleOperations ordered =
      collectRVVRoleOperationsInBodyOrder(request.getVariant(), slice);
  const bool isWideningProductReduceConstructionRoute =
      constructionRoute.operationMnemonic == "widening_product_reduce_add" ||
      constructionRoute.operationMnemonic ==
          "widening_product_reduce_dequantize_f32" ||
      constructionRoute.operationMnemonic ==
          "widening_product_reduce_dequant_clamp_f32";
  const bool isRuntimeScalarCompositeConstructionRoute =
      constructionRoute.operationMnemonic ==
      "runtime_scalar_cmp_masked_indexed_gather_macc_scatter";
  // The dequant(/clamp) routes verify the realized op sequence against a
  // candidate-aware chain (actual head op + handoff presence) so a single-scope
  // nibble/grouped body matches; the role-sequence verifier still rejects any IR
  // that does not match the derived chain (fail-closed, not relaxed).
  const bool isDequantConstructionRoute =
      constructionRoute.operationMnemonic ==
          "widening_product_reduce_dequantize_f32" ||
      constructionRoute.operationMnemonic ==
          "widening_product_reduce_dequant_clamp_f32";
  llvm::StringRef selectedTypedComputeOpName =
      isDequantConstructionRoute
          ? getRVVSelectedBodyDequantTypedComputeOpChain(
                slice, constructionRoute.operationMnemonic ==
                           "widening_product_reduce_dequant_clamp_f32")
      : (isWideningProductReduceConstructionRoute ||
         isRuntimeScalarCompositeConstructionRoute)
          ? constructionRoute.typedComputeOpName
          : slice.arithmeticOp->getName().getStringRef();
  std::string routeSequenceContext =
      (llvm::Twine("selected RVV EmitC route '") +
       constructionRoute.operationMnemonic + "' typed '" +
       selectedTypedComputeOpName + "'")
          .str();
  return verifyRVVSelectedBodySelectedRoleSequence(
      ordered.operations, ordered.constructionOrders,
      request.getVariant().getSymName(),
      stringifyVariantEmissionRole(request.getRole()),
      constructionRoute.operationMnemonic,
      selectedTypedComputeOpName,
      isIndexedGatherUnitStore
          ? slice.indexedLoadOperation->getName().getStringRef()
      : isIndexedScatterUnitLoad
          ? slice.indexedStoreOperation->getName().getStringRef()
      : isMaskedUnitLoadStore
          ? slice.maskLoadOperation->getName().getStringRef()
      : isMaskedUnitStore
          ? slice.maskLoadOperation->getName().getStringRef()
      : isComputedMaskUnitLoadStore
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskIndexedGatherLoad
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskIndexedScatterStore
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskSegment2Load
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskSegment2Store
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskSelect
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isRuntimeScalarCompareSelect
          ? slice.rhsLoadOperation->getName().getStringRef()
      : isRuntimeScalarDualCompareMaskAndSelect
          ? slice.maskAndOp.getOperation()->getName().getStringRef()
      : isRuntimeScalarComputedMaskStore
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isRuntimeScalarComputedMaskLoadStore
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskStridedStore
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskStridedLoad
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskWideningDotReduce
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskStridedInputWideningDotReduce
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isComputedMaskedMAcc
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isRuntimeScalarComputedMaskedMAcc
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isRuntimeScalarComputedMaskIndexedGatherMAccScatter
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isStridedInputWideningDotReduce
          ? slice.rhsLoadOperation->getName().getStringRef()
      : isSegment2DeinterleaveUnitStore
          ? slice.segment2LoadOperation->getName().getStringRef()
      : isSegment2InterleaveUnitLoad
          ? slice.segment2StoreOperation->getName().getStringRef()
      : isStandaloneReduction
          ? slice.lhsLoadOperation->getName().getStringRef()
      : isComputedMaskStandaloneReduction
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isRuntimeScalarComputedMaskStandaloneReduction
          ? slice.compareOp.getOperation()->getName().getStringRef()
      : isDequantClampF32Epilogue
          ? slice.dequantizeOp.getOperation()->getName().getStringRef()
      : isStridedLoadUnitStore
          ? slice.lhsLoadOperation->getName().getStringRef()
      : isUnitLoadStridedStore
          ? slice.storeOperation->getName().getStringRef()
          : (slice.rhsLoadOperation
                 ? slice.rhsLoadOperation->getName().getStringRef()
                 : llvm::StringRef()),
      routeSequenceContext);
}


} // namespace tianchenrv::plugin::rvv
