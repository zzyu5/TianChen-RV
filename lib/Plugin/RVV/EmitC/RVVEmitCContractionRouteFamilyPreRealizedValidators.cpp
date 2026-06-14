//===- RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp - body checks =//
//
// Behavior-preserving split out of RVVEmitCContractionRouteFamilyPlanOwners.cpp:
// the pre-realized / explicit typed-body validators for the contraction route
// families -- validatePreRealizedRVVSelected{WideningMAcc, WideningDotReduce,
// StridedInputWideningDotReduce, ComputedMaskWideningDotReduce,
// ComputedMaskStridedInputWideningDotReduce, WideningProductReduce,
// WideningProductReduceDequantize, WideningProductReduceDequantClampF32}Body
// plus the explicit DequantClampF32 body validator. The cross-TU helpers they
// rely on (the contraction common substrate + the rejectMixed* template) are
// declared in the co-located implementation-private
// RVVEmitCContractionRouteFamilyInternal.h. Pure relocation -- bodies are
// byte-identical.
//
//===----------------------------------------------------------------------===//

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

llvm::Error validatePreRealizedRVVSelectedWideningMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV widening macc realization requires a pre-realized "
        "widening macc body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc realization requires a "
        "selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body must be a direct child "
        "of the selected variant");
  if (!isPreRealizedWideningMAccOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only op_kind 'signed_widening_macc_add'");
  if (body.getMemoryForm() != kRVVPreRealizedWideningMAccMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only memory_form 'unit-stride-widening-macc'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only accumulator_role 'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningMAccAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only accumulator_layout 'separate-i32-vector-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningMAccResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only result_layout "
        "'store-widening-multiply-accumulate-result-to-output-buffer'");
  llvm::StringRef expectedMAccRelation = getContractionWideningMAccRelation(
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul());
  if (expectedMAccRelation.empty() ||
      body.getMaccRelation() != expectedMAccRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body currently supports "
        "only provider-derived signed widening macc_relation matching "
        "source/result typed config facts");
  if (!isPreRealizedWideningMAccSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getMaccRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc config/relation must match "
        "op_kind 'signed_widening_macc_add' with supported source, "
        "accumulator/result, and provider-derived relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(), "pre-realized RVV widening macc lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(), "pre-realized RVV widening macc rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(), "pre-realized RVV widening macc accumulator operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(), "pre-realized RVV widening macc out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening macc body requires lhs/rhs "
        "const int16_t *, accumulator const int32_t *, and out int32_t * "
        "runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(), "pre-realized RVV widening macc runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningMAccOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "widening macc"))
    return error;
  return requireContractionSelectedVariantRequires(variant,
                                                   "widening macc");
}

llvm::Error validatePreRealizedRVVSelectedWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV widening dot-product reduction realization requires a "
        "pre-realized widening dot-product reduction body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction realization "
        "requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body must "
        "be a direct child of the selected variant");
  if (!isPreRealizedWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only op_kind "
        "'signed_widening_dot_reduce_add'");
  if (body.getMemoryForm() != kRVVPreRealizedWideningDotReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only memory_form "
        "'unit-stride-widening-dot-reduce'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(
          static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul());
  if (expectedDotRelation.empty() ||
      body.getDotProductRelation() != expectedDotRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "currently supports only provider-derived dot_product_relation "
        "matching source/result typed config facts");
  if (!isPreRealizedWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction "
        "config/relation must match op_kind "
        "'signed_widening_dot_reduce_add' with supported source, "
        "accumulator/result, and provider-derived relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV widening dot-product reduction lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV widening dot-product reduction rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV widening dot-product reduction accumulator seed "
          "operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV widening dot-product reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening dot-product reduction body "
        "requires lhs/rhs const int16_t *, accumulator seed const int32_t *, "
        "and out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening dot-product reduction runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningDotReduceOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "widening dot-product reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "widening dot-product reduction");
}

llvm::Error validatePreRealizedRVVSelectedStridedInputWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV strided-input widening dot-product reduction realization "
        "requires a pre-realized strided-input widening dot-product reduction "
        "body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product "
        "reduction realization requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body must be a direct child of the selected variant");
  if (!isPreRealizedWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only op_kind 'signed_widening_dot_reduce_add'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedStridedInputWideningDotReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only memory_form "
        "'strided-input-widening-dot-reduce'");
  if (body.getStrideUnit() != "element")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only stride_unit 'element'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(
          static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul());
  if (expectedDotRelation.empty() ||
      body.getDotProductRelation() != expectedDotRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body currently supports only provider-derived dot_product_relation "
        "matching source/result typed config facts");
  if (!isPreRealizedWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "config/relation must match op_kind 'signed_widening_dot_reduce_add' "
        "with supported source, accumulator/result, and provider-derived "
        "relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV strided-input widening dot-product reduction lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV strided-input widening dot-product reduction rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV strided-input widening dot-product reduction "
          "accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV strided-input widening dot-product reduction out "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected strided-input widening dot-product reduction "
        "body requires lhs/rhs const int16_t *, accumulator seed const int32_t *, "
        "and out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV strided-input widening dot-product reduction "
          "runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhsStride =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhsStride(),
          "pre-realized RVV strided-input widening dot-product reduction lhs "
          "stride operand",
          support::RuntimeABIParameterRole::LHSInputStride);
  if (!lhsStride)
    return lhsStride.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsStride =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhsStride(),
          "pre-realized RVV strided-input widening dot-product reduction rhs "
          "stride operand",
          support::RuntimeABIParameterRole::RHSInputStride);
  if (!rhsStride)
    return rhsStride.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::StridedLoadOp, tcrv::rvv::WideningDotReduceOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "strided-input widening dot-product "
                                  "reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "strided-input widening dot-product reduction");
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask widening dot-product reduction "
        "realization requires a pre-realized computed-mask widening "
        "dot-product reduction body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction realization requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body must be a direct child of the selected variant");
  if (!isPreRealizedComputedMaskWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only op_kind "
        "'signed_masked_widening_dot_reduce_add'");
  if (body.getPredicateKind() != kRVVPreRealizedPredicateKind)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only predicate_kind 'slt'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedComputedMaskWideningDotReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only memory_form "
        "'computed-mask-unit-stride-widening-dot-reduce'");
  if (body.getMaskRole() != kRVVMaskedPredicateMaskRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_role "
        "'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != kRVVMaskedCompareMaskSource)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires mask_memory_form 'compare-produced-mask'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(
          static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul());
  if (expectedDotRelation.empty() ||
      body.getDotProductRelation() != expectedDotRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body currently supports only provider-derived "
        "dot_product_relation matching source/result typed config facts");
  if (!isPreRealizedComputedMaskWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction config/relation must match op_kind "
        "'signed_masked_widening_dot_reduce_add' with supported compare, "
        "dot source, accumulator/result, and provider-derived relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLHS =
      requirePreRealizedContractionRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLHS)
    return compareLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRHS =
      requirePreRealizedContractionRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRHS)
    return compareRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "dot lhs operand",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "dot rhs operand",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*compareLHS).getCType() != kRVVContractionI32PointerCType ||
      (*compareRHS).getCType() != kRVVContractionI32PointerCType ||
      (*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask widening dot-product "
        "reduction body requires compare lhs/rhs const int32_t *, dot "
        "lhs/rhs const int16_t *, accumulator seed const int32_t *, and out "
        "int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask widening dot-product reduction "
          "runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::CompareOp,
              tcrv::rvv::MaskedWideningDotReduceOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "computed-mask widening dot-product "
                                  "reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "computed-mask widening dot-product reduction");
}

llvm::Error
validatePreRealizedRVVSelectedComputedMaskStridedInputWideningDotReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::
        TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask strided-input widening dot-product "
        "reduction realization requires a pre-realized computed-mask "
        "strided-input widening dot-product reduction body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction realization requires a selected "
        "tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body must be a direct child of the selected "
        "variant");
  if (!isPreRealizedComputedMaskWideningDotReduceOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only op_kind "
        "'signed_masked_widening_dot_reduce_add'");
  if (body.getPredicateKind() != kRVVPreRealizedPredicateKind)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only predicate_kind "
        "'slt'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedComputedMaskStridedInputWideningDotReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only memory_form "
        "'computed-mask-strided-input-widening-dot-reduce'");
  if (body.getStrideUnit() != "element")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only stride_unit "
        "'element'");
  if (body.getMaskRole() != kRVVMaskedPredicateMaskRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_role "
        "'predicate-mask-produced-by-compare'");
  if (body.getMaskSource() != kRVVMaskedCompareMaskSource)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (body.getMaskMemoryForm() != kRVVComputedMaskMemoryMaskMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires mask_memory_form "
        "'compare-produced-mask'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "accumulator_role 'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "accumulator_layout 'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVWideningDotProductResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only result_layout "
        "'store-dot-reduction-lane0-to-output-scalar'");
  llvm::StringRef expectedDotRelation =
      getContractionWideningDotProductRelation(
          static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul());
  if (expectedDotRelation.empty() ||
      body.getDotProductRelation() != expectedDotRelation)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body currently supports only "
        "provider-derived dot_product_relation matching source/result typed "
        "config facts");
  if (!isPreRealizedComputedMaskWideningDotReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getDotProductRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction config/relation must match op_kind "
        "'signed_masked_widening_dot_reduce_add' with supported compare, dot "
        "source, accumulator/result, and provider-derived relation facts");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires tail agnostic, mask agnostic "
        "policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLHS =
      requirePreRealizedContractionRuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLHS)
    return compareLHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRHS =
      requirePreRealizedContractionRuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRHS)
    return compareRHS.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction dot lhs operand",
          support::RuntimeABIParameterRole::DotLHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction dot rhs operand",
          support::RuntimeABIParameterRole::DotRHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*compareLHS).getCType() != kRVVContractionI32PointerCType ||
      (*compareRHS).getCType() != kRVVContractionI32PointerCType ||
      (*lhs).getCType() != kRVVContractionI16PointerCType ||
      (*rhs).getCType() != kRVVContractionI16PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*out).getCType() != kRVVContractionOutputI32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask strided-input widening "
        "dot-product reduction body requires compare lhs/rhs const int32_t *, "
        "dot lhs/rhs const int16_t *, accumulator seed const int32_t *, and "
        "out int32_t * runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhsStride =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhsStride(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction lhs stride operand",
          support::RuntimeABIParameterRole::LHSInputStride);
  if (!lhsStride)
    return lhsStride.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsStride =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhsStride(),
          "pre-realized RVV computed-mask strided-input widening "
          "dot-product reduction rhs stride operand",
          support::RuntimeABIParameterRole::RHSInputStride);
  if (!rhsStride)
    return rhsStride.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::StridedLoadOp,
              tcrv::rvv::CompareOp,
              tcrv::rvv::MaskedWideningDotReduceOp,
              tcrv::rvv::StoreOp>(variant, body.getOperation(),
                                  "computed-mask strided-input widening "
                                  "dot-product reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "computed-mask strided-input widening dot-product reduction");
}

llvm::Error validatePreRealizedRVVSelectedWideningProductReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningProductReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV widening product reduction realization requires a "
        "pre-realized product-reduction body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction realization "
        "requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction body must be a "
        "direct child of the selected variant");
  if (body.getOpKind() != kRVVPreRealizedWideningProductReduceOpKind)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction body currently "
        "supports only op_kind 'widening_product_reduce_add'");
  if (body.getMemoryForm() != kRVVPreRealizedWideningProductReduceMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction body currently "
        "supports only memory_form 'unit-stride-widening-product-reduce-add'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction body currently "
        "supports only accumulator_role 'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction body currently "
        "supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVProductReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction body currently "
        "supports only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
  const bool isUnsignedProductReduction =
      body.getSourceSignedness() ==
      kRVVLowPrecisionResourceSourceSignednessUnsigned;
  if (body.getSourceSignedness() !=
          kRVVLowPrecisionResourceSourceSignednessSigned &&
      !isUnsignedProductReduction)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction body currently "
        "supports only source_signedness 'signed' or 'unsigned'");
  if (!isPreRealizedWideningProductReduceSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getProductSew()),
          body.getProductLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
          body.getProductRelation(), body.getProductReductionChainRelation(),
          isUnsignedProductReduction))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction config/relation "
        "must match op_kind 'widening_product_reduce_add' with source SEW8 "
        "LMUL mf4, product SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1, "
        "and provider-derived signed or unsigned product and reduction "
        "relations");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV widening product reduction lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV widening product reduction rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV widening product reduction accumulator seed operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV widening product reduction out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::StringRef expectedInputCType =
      isUnsignedProductReduction
          ? llvm::StringRef(kRVVContractionU8PointerCType)
          : llvm::StringRef(kRVVContractionI8PointerCType);
  llvm::StringRef expectedAccumulatorCType =
      isUnsignedProductReduction
          ? llvm::StringRef(kRVVContractionU32PointerCType)
          : llvm::StringRef(kRVVContractionI32PointerCType);
  llvm::StringRef expectedOutputCType =
      isUnsignedProductReduction
          ? llvm::StringRef(kRVVContractionOutputU32PointerCType)
          : llvm::StringRef(kRVVContractionOutputI32PointerCType);
  if ((*lhs).getCType() != expectedInputCType ||
      (*rhs).getCType() != expectedInputCType ||
      (*acc).getCType() != expectedAccumulatorCType ||
      (*out).getCType() != expectedOutputCType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected widening product reduction "
                    "body requires lhs/rhs ") +
        expectedInputCType + ", accumulator seed " +
        expectedAccumulatorCType + ", and out " + expectedOutputCType +
        " runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening product reduction runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningProductOp,
              tcrv::rvv::StandaloneReduceOp, tcrv::rvv::StoreOp>(
              variant, body.getOperation(), "widening product reduction"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "widening product reduction");
}

llvm::Error validatePreRealizedRVVSelectedWideningProductReduceDequantizeBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningProductReduceDequantizePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV widening product reduction dequantization realization "
        "requires a pre-realized product-reduction-dequantization body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization realization requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body must be a direct child of the selected variant");
  if (!isPreRealizedWideningProductReduceDequantizeOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only op_kind "
        "'widening_product_reduce_dequantize_f32'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedWideningProductReduceDequantizeMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only memory_form "
        "'unit-stride-widening-product-reduce-dequantize-f32'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVProductReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body currently supports only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
  if (body.getAccumulatorCarryBoundary() !=
      kRVVProductReductionDequantVectorCarryBoundary)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires the vector i32m1 carry boundary that "
        "feeds one final scalar extract and f32 store");
  if (body.getScaleRole() != kRVVContractionDequantScaleRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires scale_role 'dequant-scale-value'");
  if (body.getDequantStoreBoundary() !=
      "store-dequantized-f32-vector-to-output-buffer")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires the f32 output store boundary");
  if (!isPreRealizedWideningProductReduceDequantizeSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getProductSew()),
          body.getProductLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getProductRelation(),
          body.getProductReductionChainRelation(), body.getDequantRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization config/relation must match op_kind "
        "'widening_product_reduce_dequantize_f32' with source SEW8 LMUL mf4, "
        "product SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1, and "
        "provider-derived product, reduction, and dequantization relations");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV widening product reduction dequantization lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV widening product reduction dequantization rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV widening product reduction dequantization "
          "accumulator seed/carry operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> scale =
      requirePreRealizedContractionRuntimeABIValue(
          body.getScale(),
          "pre-realized RVV widening product reduction dequantization "
          "runtime scale operand",
          support::RuntimeABIParameterRole::DequantScaleValue);
  if (!scale)
    return scale.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV widening product reduction dequantization f32 "
          "out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI8PointerCType ||
      (*rhs).getCType() != kRVVContractionI8PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*scale).getCType() != kRVVContractionDequantScaleCType ||
      (*out).getCType() != kRVVContractionF32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequantization body requires lhs/rhs const int8_t *, accumulator "
        "seed/carry const int32_t *, runtime scale float, and out float * "
        "runtime ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening product reduction dequantization runtime "
          "n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningProductOp,
              tcrv::rvv::StandaloneReduceOp,
              tcrv::rvv::GearboxCrossRegionHandoffOp,
              tcrv::rvv::DequantizeOp, tcrv::rvv::StoreOp>(
              variant, body.getOperation(),
              "widening product reduction dequantization"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "widening product reduction dequantization");
}

template <typename BodyOp>
llvm::Error validateRVVSelectedWideningProductReduceDequantClampF32BodyImpl(
    const VariantLoweringBoundaryRequest &request, BodyOp body,
    llvm::StringRef bodyKind) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV widening product reduction dequant-clamp "
                    "realization requires a ") +
        bodyKind + " product-reduction-dequant-clamp body op");
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction "
        "dequant-clamp realization requires a selected tcrv.exec.variant");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body must be a direct child of the selected variant");
  if (!isPreRealizedWideningProductReduceDequantClampF32OpKind(
          body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only op_kind "
        "'widening_product_reduce_dequant_clamp_f32'");
  if (body.getMemoryForm() !=
      kRVVPreRealizedWideningProductReduceDequantClampF32MemoryForm)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only memory_form "
        "'unit-stride-widening-product-reduce-dequant-clamp-f32'");
  if (body.getAccumulatorRole() != kRVVPreRealizedAccumulatorRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only accumulator_role "
        "'accumulator-input-buffer'");
  if (body.getAccumulatorLayout() != kRVVWideningDotProductAccumulatorLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only accumulator_layout "
        "'scalar-i32-seed-lane0-from-accumulator-input'");
  if (body.getResultLayout() != kRVVProductReductionResultLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body currently supports only result_layout "
        "'store-standalone-reduction-lane0-to-output-scalar'");
  if (body.getAccumulatorCarryBoundary() !=
      kRVVProductReductionDequantVectorCarryBoundary)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires the vector i32m1 carry boundary that feeds one final "
        "scalar extract and f32 store");
  if (body.getScaleRole() != kRVVContractionDequantScaleRole)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires scale_role 'dequant-scale-value'");
  if (body.getLowerPredicateKind() != kRVVPreRealizedPredicateKind ||
      body.getUpperPredicateKind() != kRVVPreRealizedPredicateKind)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires lower/upper predicate kind 'slt'");
  if (body.getBoundOrder() != kRVVContractionClampBoundOrder ||
      body.getSelectLayout() != kRVVContractionClampSelectLayout)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires lower-bound-before-upper-bound and "
        "clamp-lower-then-upper structure");
  if (body.getDequantStoreBoundary() !=
      kRVVProductReductionDequantClampStoreBoundary)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires the clamped f32 output store boundary");
  if (!isPreRealizedWideningProductReduceDequantClampF32Signature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(),
          static_cast<std::int64_t>(body.getProductSew()),
          body.getProductLmul(),
          static_cast<std::int64_t>(body.getAccumulatorSew()),
          body.getAccumulatorLmul(),
          static_cast<std::int64_t>(body.getResultSew()),
          body.getResultLmul(), body.getProductRelation(),
          body.getProductReductionChainRelation(), body.getDequantRelation()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "config/relation must match op_kind "
        "'widening_product_reduce_dequant_clamp_f32' with source SEW8 LMUL "
        "mf4, product SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1, and "
        "provider-derived product, reduction, and dequantization relations");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV widening product reduction dequant-clamp lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedContractionRuntimeABIValue(
          body.getRhs(),
          "pre-realized RVV widening product reduction dequant-clamp rhs "
          "operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> acc =
      requirePreRealizedContractionRuntimeABIValue(
          body.getAcc(),
          "pre-realized RVV widening product reduction dequant-clamp "
          "accumulator seed/carry operand",
          support::RuntimeABIParameterRole::AccumulatorInputBuffer);
  if (!acc)
    return acc.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> scale =
      requirePreRealizedContractionRuntimeABIValue(
          body.getScale(),
          "pre-realized RVV widening product reduction dequant-clamp runtime "
          "scale operand",
          support::RuntimeABIParameterRole::DequantScaleValue);
  if (!scale)
    return scale.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lower =
      requirePreRealizedContractionRuntimeABIValue(
          body.getLowerBound(),
          "pre-realized RVV widening product reduction dequant-clamp lower "
          "bound operand",
          support::RuntimeABIParameterRole::LowerBoundScalarValue);
  if (!lower)
    return lower.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> upper =
      requirePreRealizedContractionRuntimeABIValue(
          body.getUpperBound(),
          "pre-realized RVV widening product reduction dequant-clamp upper "
          "bound operand",
          support::RuntimeABIParameterRole::UpperBoundScalarValue);
  if (!upper)
    return upper.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedContractionRuntimeABIValue(
          body.getOut(),
          "pre-realized RVV widening product reduction dequant-clamp f32 out "
          "operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  if ((*lhs).getCType() != kRVVContractionI8PointerCType ||
      (*rhs).getCType() != kRVVContractionI8PointerCType ||
      (*acc).getCType() != kRVVContractionI32PointerCType ||
      (*scale).getCType() != kRVVContractionDequantScaleCType ||
      (*lower).getCType() != kRVVContractionF32ScalarCType ||
      (*upper).getCType() != kRVVContractionF32ScalarCType ||
      (*out).getCType() != kRVVContractionF32PointerCType)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected widening product reduction dequant-clamp "
        "body requires lhs/rhs const int8_t *, accumulator seed/carry const "
        "int32_t *, runtime scale/lower/upper float, and out float * runtime "
        "ABI bindings");
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedContractionRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening product reduction dequant-clamp runtime "
          "n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedContractionBody<
              tcrv::rvv::SetVLOp, tcrv::rvv::WithVLOp,
              tcrv::rvv::LoadOp, tcrv::rvv::WideningProductOp,
              tcrv::rvv::StandaloneReduceOp, tcrv::rvv::DequantizeOp,
              tcrv::rvv::SplatOp, tcrv::rvv::CompareOp,
              tcrv::rvv::SelectOp, tcrv::rvv::StoreOp>(
              variant, body.getOperation(),
              "widening product reduction dequant-clamp"))
    return error;
  return requireContractionSelectedVariantRequires(
      variant, "widening product reduction dequant-clamp");
}

llvm::Error
validatePreRealizedRVVSelectedWideningProductReduceDequantClampF32Body(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::
        TypedWideningProductReduceDequantClampF32PreRealizedBodyOp body) {
  return validateRVVSelectedWideningProductReduceDequantClampF32BodyImpl(
      request, body, "pre-realized");
}

llvm::Error validateExplicitRVVSelectedWideningProductReduceDequantClampF32Body(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningProductReduceDequantClampF32BodyOp body) {
  return validateRVVSelectedWideningProductReduceDequantClampF32BodyImpl(
      request, body, "explicit");
}

} // namespace tianchenrv::plugin::rvv
