//===- RVVEmitCContractionRouteFamilyValidation.cpp - plan validate/apply ===//
//
// Behavior-preserving split out of RVVEmitCContractionRouteFamilyPlanOwners.cpp:
// the contraction route-family plan validation / derivation / application /
// description-mirror verification half, plus the operand-binding plan accessors.
// The cross-TU helpers it relies on are declared in the co-located
// implementation-private RVVEmitCContractionRouteFamilyInternal.h. Pure
// relocation -- bodies are byte-identical; route content is unchanged.
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

llvm::Error validateRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "contraction route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyContractionRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan supports only active widening_macc_add, "
        "widening_product, and widening dot-reduce routes");
  const bool isWideningMAcc =
      plan.operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isWideningProduct =
      plan.operation == RVVSelectedBodyOperationKind::WideningProduct;
  const bool isProductReductionDequantization =
      plan.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      plan.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
      plan.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32;
  const bool isProductReductionDequantClamp =
      plan.operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isProductReductionChain =
      plan.operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isProductReductionDequantization;
  const bool isDotReduction =
      isRVVSelectedBodyContractionDotReduction(plan.operation);
  const bool isComputedMask =
      isRVVSelectedBodyContractionComputedMask(plan.operation);
  const bool isStridedInput =
      isRVVSelectedBodyContractionStridedInputs(plan.operation);
  const bool isUnsignedWideningProduct =
      isWideningProduct &&
      plan.wideningProductRelation ==
          getContractionWideningProductRelation(
              tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
              tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true);
  const bool isUnsignedProductReductionChain =
      isProductReductionChain &&
      (plan.lowPrecisionPrimitiveSourceSignedness == "unsigned" ||
       plan.wideningProductRelation ==
           getContractionWideningProductRelation(
               tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
               tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
               /*isUnsigned=*/true) ||
       plan.productReductionChainRelation ==
           getContractionProductReductionChainRelation(
               tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
               tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
               tcrv::rvv::getRVVFirstSliceSEWBits(),
               tcrv::rvv::getRVVLMULM1(), /*isUnsigned=*/true));
  const bool isUnsignedLowPrecisionIntegerResult =
      isUnsignedWideningProduct || isUnsignedProductReductionChain;
  const bool isUnsignedLowPrecisionSourceProduct =
      isUnsignedWideningProduct || isUnsignedProductReductionChain;
  const RVVSelectedBodyMemoryForm expectedMemoryForm =
      isWideningMAcc
      ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : isWideningProduct ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : isProductReductionDequantClamp
          ? RVVSelectedBodyMemoryForm::
                UnitStrideWideningProductReduceDequantClampF32
      : isProductReductionChain ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : (plan.operation == RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         plan.operation ==
             RVVSelectedBodyOperationKind::
                 WideningProductDeferredDotAccumulateReduceAdd)
          // The deferred-wide i16 dot-reduce shares the narrow dot-reduce unit-
          // stride VectorRHSLoad memory form (both load two i16 source vectors).
          ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : plan.operation ==
              RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd
          ? RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce
      : plan.operation ==
              RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd
          ? RVVSelectedBodyMemoryForm::
                ComputedMaskUnitStrideWideningDotReduce
          : RVVSelectedBodyMemoryForm::
                ComputedMaskStridedInputWideningDotReduce;
  if (plan.usesWideningMAcc != isWideningMAcc ||
      plan.usesWideningProduct != isWideningProduct ||
      plan.usesProductReductionChain != isProductReductionChain ||
      plan.usesProductReductionDequantization !=
          isProductReductionDequantization ||
      plan.usesProductReductionDequantClamp != isProductReductionDequantClamp ||
      plan.usesDotReduction != isDotReduction ||
      plan.usesComputedMask != isComputedMask ||
      plan.usesStridedInputs != isStridedInput ||
      plan.usesScalarSeed != isDotReduction ||
      plan.usesVectorAccumulator != isWideningMAcc)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan stale operation classification "
        "markers");
  if (plan.memoryForm != expectedMemoryForm)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family plan requires the operation-specific memory "
        "form");
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "typed config facts", plan.typedConfigFactsID,
          "rvv-selected-body-typed-config-facts.v1"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "element type", plan.elementTypeName,
          isProductReductionDequantization
              ? getContractionFloatElementTypeName(plan.sew)
              : getContractionIntegerElementTypeName(
                    plan.sew, isUnsignedLowPrecisionIntegerResult)))
    return error;
  if (plan.elementBitWidth != plan.sew)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family typed config snapshot requires element bit "
        "width to match result SEW");
  const bool isDeferredWideProductReductionDequantization =
      plan.operation ==
      RVVSelectedBodyOperationKind::
          WideningProductDeferredAccumulateReduceDequantizeF32;
  const bool isDeferredWideDotReduce =
      plan.operation ==
      RVVSelectedBodyOperationKind::
          WideningProductDeferredDotAccumulateReduceAdd;
  if (isDeferredWideProductReductionDequantization) {
    // The deferred-wide route legitimately carries TWO structural configs: the
    // loop/strip config (sew8/m2, the setvl driving the runtime AVL/VL control
    // plan) and the i32m1/f32m1 RESULT config (plan.sew/lmul, driving result
    // typing). The narrow route asserts these are equal; the wide route asserts
    // each against its own structural source (I5: both are realized configs).
    if (plan.runtimeControlPlan.sew != tcrv::rvv::getRVVSEW8Bits() ||
        plan.runtimeControlPlan.lmul != tcrv::rvv::getRVVLMULM2())
      return makeRVVEmitCRouteProviderError(
          "deferred-wide contraction route-family plan requires the runtime "
          "AVL/VL control plan to run at the realized i8m2 strip config");
    if (plan.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
        plan.lmul != tcrv::rvv::getRVVLMULM1())
      return makeRVVEmitCRouteProviderError(
          "deferred-wide contraction route-family plan requires the result "
          "config to be the realized i32m1/f32m1 result config");
    if (plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
        plan.maskPolicy != plan.runtimeControlPlan.maskPolicy)
      return makeRVVEmitCRouteProviderError(
          "deferred-wide contraction route-family plan requires the result and "
          "strip configs to share the realized tail/mask policy");
  } else if (isDeferredWideDotReduce) {
    // The deferred-wide i16 dot-reduce route (2nd kernel family) carries TWO
    // structural configs: the loop/strip config (sew16/m4, the setvl driving the
    // runtime AVL/VL control plan over the i16m4 -> i32m8 winner) and the i32m1
    // RESULT config (plan.sew/lmul, driving result typing/header). Assert each
    // against its own structural source (I5: both are realized configs).
    if (plan.runtimeControlPlan.sew != tcrv::rvv::getRVVSEW16Bits() ||
        plan.runtimeControlPlan.lmul != tcrv::rvv::getRVVLMULM4())
      return makeRVVEmitCRouteProviderError(
          "deferred-wide i16 dot-reduce contraction route-family plan requires "
          "the runtime AVL/VL control plan to run at the realized i16m4 strip "
          "config");
    if (plan.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
        plan.lmul != tcrv::rvv::getRVVLMULM1())
      return makeRVVEmitCRouteProviderError(
          "deferred-wide i16 dot-reduce contraction route-family plan requires "
          "the result config to be the realized i32m1 result config");
    if (plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
        plan.maskPolicy != plan.runtimeControlPlan.maskPolicy)
      return makeRVVEmitCRouteProviderError(
          "deferred-wide i16 dot-reduce contraction route-family plan requires "
          "the result and strip configs to share the realized tail/mask policy");
  } else if (plan.sew != plan.runtimeControlPlan.sew ||
             plan.lmul != plan.runtimeControlPlan.lmul ||
             plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
             plan.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
             plan.configContractID != plan.runtimeControlPlan.configContractID)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family typed config snapshot must mirror runtime "
        "AVL/VL control plan result SEW/LMUL/policy/config facts");
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "source element type", plan.sourceElementTypeName,
          getContractionIntegerElementTypeName(plan.sourceSEW,
                                               isUnsignedLowPrecisionSourceProduct)))
    return error;
  if (plan.sourceElementBitWidth != plan.sourceSEW)
    return makeRVVEmitCRouteProviderError(
        "contraction route-family source typed snapshot requires source "
        "element bit width to match source SEW");
  const bool supportsNarrowProductReductionChain =
      plan.sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      plan.sourceLMUL == tcrv::rvv::getRVVLMULMF4() &&
      plan.productSEW == tcrv::rvv::getRVVSEW16Bits() &&
      plan.productLMUL == tcrv::rvv::getRVVLMULMF2() &&
      plan.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      plan.lmul == tcrv::rvv::getRVVLMULM1();
  // The deferred-wide (N3) chain runs the parallel wide ladder: source i8m2,
  // product i16m4, result i32m1 (the i32m8 deferred accumulate is the structural
  // intermediate between product and reduce). Derived structurally (I5).
  const bool supportsDeferredWideProductReductionChain =
      isDeferredWideProductReductionDequantization &&
      plan.sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      plan.sourceLMUL == tcrv::rvv::getRVVLMULM2() &&
      plan.productSEW == tcrv::rvv::getRVVSEW16Bits() &&
      plan.productLMUL == tcrv::rvv::getRVVLMULM4() &&
      plan.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      plan.lmul == tcrv::rvv::getRVVLMULM1();
  const bool supportsProductReductionChain =
      supportsNarrowProductReductionChain ||
      supportsDeferredWideProductReductionChain;
  if (isProductReductionChain && !supportsProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("contraction route-family plan does not support "
                    "product-reduction source/product/result SEW/LMUL chain '") +
        llvm::Twine(plan.sourceSEW) + "/" + plan.sourceLMUL + " -> " +
        llvm::Twine(plan.productSEW) + "/" + plan.productLMUL + " -> " +
        llvm::Twine(plan.sew) + "/" + plan.lmul + "'");
  // The deferred-wide i16 dot-reduce realized source is the wide i16m4 strip (vs
  // the narrow i16mf2 dot-reduce); its result is i32m1. This is the structural
  // realized wide ladder of the SAME logical dot-reduce op (I5), admitted as a
  // parallel config (it does NOT loosen isSupportedContractionSourceResultConfig
  // for any other route).
  const bool supportsDeferredWideDotReduceConfig =
      isDeferredWideDotReduce &&
      plan.sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      plan.sourceLMUL == tcrv::rvv::getRVVLMULM4() &&
      plan.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      plan.lmul == tcrv::rvv::getRVVLMULM1();
  if (!isProductReductionChain && !supportsDeferredWideDotReduceConfig &&
      !isSupportedContractionSourceResultConfig(plan.sourceSEW,
                                                plan.sourceLMUL, plan.sew,
                                                plan.lmul))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("contraction route-family plan does not support source "
                    "SEW/LMUL '") +
        llvm::Twine(plan.sourceSEW) + "/" + plan.sourceLMUL +
        "' with result SEW/LMUL '" + llvm::Twine(plan.sew) + "/" + plan.lmul +
        "'");
  const bool usesLowPrecisionPrimitive =
      plan.usesWideningProduct || plan.usesProductReductionChain;
  if (usesLowPrecisionPrimitive) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive contract",
            plan.lowPrecisionPrimitiveContractID,
            kRVVLowPrecisionPrimitiveContractID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive kind",
            plan.lowPrecisionPrimitiveKind,
            getRVVLowPrecisionPrimitiveKind(plan)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source dtype",
            plan.lowPrecisionPrimitiveSourceElementTypeName,
            plan.sourceElementTypeName))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source signedness",
            plan.lowPrecisionPrimitiveSourceSignedness,
            getRVVLowPrecisionPrimitiveSourceSignedness(plan)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source load",
            plan.lowPrecisionPrimitiveSourceLoadKind,
            getRVVLowPrecisionPrimitiveSourceLoadKind(plan)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source extension",
            plan.lowPrecisionPrimitiveSourceExtensionKind,
            getRVVLowPrecisionPrimitiveSourceExtensionKind(plan)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive product dtype",
            plan.lowPrecisionPrimitiveProductElementTypeName,
            plan.usesWideningProduct ? plan.elementTypeName
                                     : plan.productElementTypeName))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive accumulator dtype",
            plan.lowPrecisionPrimitiveAccumulatorElementTypeName,
            plan.usesWideningProduct
                ? llvm::StringRef()
                : getContractionIntegerElementTypeName(
                      plan.sew, isUnsignedLowPrecisionIntegerResult)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive result dtype",
            plan.lowPrecisionPrimitiveResultElementTypeName,
            plan.elementTypeName))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive contract",
            plan.lowPrecisionPrimitiveContractID, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive kind",
            plan.lowPrecisionPrimitiveKind, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source dtype",
            plan.lowPrecisionPrimitiveSourceElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source signedness",
            plan.lowPrecisionPrimitiveSourceSignedness, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source load",
            plan.lowPrecisionPrimitiveSourceLoadKind, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive source extension",
            plan.lowPrecisionPrimitiveSourceExtensionKind, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive product dtype",
            plan.lowPrecisionPrimitiveProductElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive accumulator dtype",
            plan.lowPrecisionPrimitiveAccumulatorElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "low-precision primitive result dtype",
            plan.lowPrecisionPrimitiveResultElementTypeName, ""))
      return error;
  }
  if (usesLowPrecisionPrimitive) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product multiplicand roles",
            plan.wideningProductMultiplicandRoleSummary,
            getRVVWideningProductMultiplicandRoleSummary(plan)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product extension policy",
            plan.wideningProductExtensionPolicy,
            getRVVWideningProductExtensionPolicy(plan)))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product multiplicand roles",
            plan.wideningProductMultiplicandRoleSummary, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product extension policy",
            plan.wideningProductExtensionPolicy, ""))
      return error;
  }
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "runtime control plan",
          plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "family plan", plan.familyPlanID,
          kRVVContractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          getRVVSelectedBodyContractionRuntimeABIOrder(plan.operation)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          isProductReductionDequantClamp
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-"
                    "dequant-clamp-leaf-profile.v1")
      : isProductReductionDequantization
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-"
                    "dequantization-leaf-profile.v1")
          : isProductReductionChain
              ? (isUnsignedProductReductionChain
                     ? llvm::StringRef(
                           "rvv-v1-u8mf4-u16mf2-u32m1-product-reduction-"
                           "contraction-leaf-profile.v1")
                     : llvm::StringRef(
                           "rvv-v1-i8mf4-i16mf2-i32m1-product-reduction-"
                           "contraction-leaf-profile.v1"))
              : getContractionTargetLeafProfile(plan.sourceSEW,
                                                plan.sourceLMUL, plan.sew,
                                                plan.lmul,
                                                isUnsignedWideningProduct)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          kRVVContractionProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          kRVVContractionRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          isProductReductionDequantClamp
              ? llvm::StringRef(
                    "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,"
                    "seed:signed-i32,accumulator:signed-e32m1,"
                    "dequant-splat/clamped:float-e32m1,scale:float,"
                    "lower:float,upper:float")
      : isProductReductionDequantization
              ? llvm::StringRef(
                    "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,"
                    "seed:signed-i32,accumulator:signed-e32m1,"
                    "dequant-splat:float-e32m1,scale:float")
          : isProductReductionChain
              ? getContractionProductReductionChainCTypeMappingSummary(
                    plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
                    plan.productLMUL, plan.sew, plan.lmul,
                    isUnsignedProductReductionChain)
              : getContractionCTypeMappingSummary(plan.sourceSEW,
                                                  plan.sourceLMUL, plan.sew,
                                                  plan.lmul,
                                                  isUnsignedWideningProduct)))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("contraction route-family target-leaf/profile "
                    "validation for operation '") +
        stringifyRVVSelectedBodyOperationKind(plan.operation) +
        "' requires provider-owned header declarations "
        "'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "VL C type", plan.vlCType, "size_t"))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "result vector type", plan.resultVectorTypeName,
          isProductReductionDequantization
              ? getContractionFloatVectorTypeName(plan.sew, plan.lmul)
              : getContractionVectorTypeName(
                    plan.sew, plan.lmul, isUnsignedLowPrecisionIntegerResult)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "result vector C type", plan.resultVectorCType,
          isProductReductionDequantization
              ? getContractionFloatVectorCType(plan.sew, plan.lmul)
              : getContractionVectorCType(
                    plan.sew, plan.lmul, isUnsignedLowPrecisionIntegerResult)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "setvl leaf", plan.setVLIntrinsic,
          getContractionSetVLIntrinsic(plan.sew, plan.lmul)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "source vector-load leaf", plan.sourceVectorLoadIntrinsic,
          getContractionVectorLoadIntrinsic(plan.sourceSEW, plan.sourceLMUL,
                                            isUnsignedLowPrecisionSourceProduct)))
    return error;
  if (isProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product element type", plan.productElementTypeName,
            getContractionIntegerElementTypeName(
                plan.productSEW, isUnsignedProductReductionChain)))
      return error;
    if (plan.productElementBitWidth != plan.productSEW)
      return makeRVVEmitCRouteProviderError(
          "product-reduction contraction plan requires product element bit "
          "width to match product SEW");
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product vector type", plan.productVectorTypeName,
            getContractionVectorTypeName(plan.productSEW, plan.productLMUL,
                                         isUnsignedProductReductionChain)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product vector C type", plan.productVectorCType,
            getContractionVectorCType(plan.productSEW, plan.productLMUL,
                                      isUnsignedProductReductionChain)))
      return error;
  }
  if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
          plan, "store leaf", plan.storeIntrinsic,
          isProductReductionDequantization
              ? getContractionFloatStoreIntrinsic(plan.sew, plan.lmul)
              : getContractionStoreIntrinsic(
                    plan.sew, plan.lmul, isUnsignedLowPrecisionIntegerResult)))
    return error;

  if (plan.usesStridedInputs) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided source-load leaf", plan.stridedLoadIntrinsic,
            getContractionStridedLoadIntrinsic(plan.sourceSEW,
                                               plan.sourceLMUL)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided memory layout", plan.stridedMemoryLayout,
            plan.usesComputedMask
                ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
                : kRVVStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "lhs stride source", plan.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "rhs stride source", plan.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm,
            kRVVDestinationMemoryForm))
      return error;
  } else if (llvm::Error error =
                 requireRVVSelectedBodyContractionPlanField(
                     plan, "strided source-load leaf",
                     plan.stridedLoadIntrinsic, ""))
    return error;
  if (plan.usesWideningMAcc || plan.usesWideningProduct ||
      plan.usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm,
            kRVVDestinationMemoryForm))
      return error;
  }
  if (!plan.usesStridedInputs && !plan.usesWideningMAcc &&
      !plan.usesWideningProduct && !plan.usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "strided memory layout", plan.stridedMemoryLayout, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "lhs stride source", plan.lhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "rhs stride source", plan.rhsStrideSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "source memory form", plan.sourceMemoryForm, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "destination memory form", plan.destinationMemoryForm, ""))
      return error;
  }

  if (plan.usesWideningMAcc) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening macc arithmetic kind",
            plan.wideningMAccArithmeticKind,
            kRVVPreRealizedWideningMAccOpKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout,
            kRVVWideningMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout,
            kRVVWideningMAccResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening macc relation", plan.relation,
            getContractionWideningMAccRelation(
                plan.sourceSEW, plan.sourceLMUL, plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening macc leaf", plan.contractionComputeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product leaf", plan.wideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL, ""))
      return error;
  } else if (plan.usesWideningProduct) {
    const llvm::StringRef expectedWideningProductRelation =
        getContractionWideningProductRelation(
            plan.sourceSEW, plan.sourceLMUL, plan.sew, plan.lmul,
            isUnsignedWideningProduct);
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product relation", plan.relation,
            expectedWideningProductRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product relation mirror",
            plan.wideningProductRelation, plan.relation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening product leaf", plan.wideningProductIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction leaf", plan.contractionComputeIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL, ""))
      return error;
  } else if (plan.usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout,
            kRVVProductReductionResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening product relation mirror",
            plan.wideningProductRelation,
            getContractionWideningProductRelation(
                plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
                plan.productLMUL, isUnsignedProductReductionChain)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product-reduction relation", plan.productReductionChainRelation,
            getContractionProductReductionChainRelation(
                plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
                plan.productLMUL, plan.sew, plan.lmul,
                isUnsignedProductReductionChain)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "product-reduction relation mirror", plan.relation,
            plan.productReductionChainRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening product leaf", plan.wideningProductIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening reduction leaf", plan.contractionComputeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL,
            kRVVWideningDotProductStoreVL))
      return error;
    if (isProductReductionDequantization) {
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "dequantization relation", plan.dequantizationRelation,
              kRVVContractionDequantizationRelation))
        return error;
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "dequant scale role", plan.dequantScaleRole,
              kRVVContractionDequantScaleRole))
        return error;
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "dequant scale C type", plan.dequantScaleCType,
              kRVVContractionDequantScaleCType))
        return error;
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "dequant scale name", plan.dequantScaleName,
              kRVVContractionDequantScaleName))
        return error;
      if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
              plan, "post-loop scalar dequant splat leaf",
              plan.rhsBroadcastIntrinsic,
              getContractionFloatScalarSplatIntrinsic(plan.sew, plan.lmul)))
        return error;
      if (isProductReductionDequantClamp) {
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "lower bound role", plan.lowerBoundRole,
                kRVVContractionLowerBoundRole))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "upper bound role", plan.upperBoundRole,
                kRVVContractionUpperBoundRole))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "lower bound C type", plan.lowerBoundCType,
                kRVVContractionF32ScalarCType))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "upper bound C type", plan.upperBoundCType,
                kRVVContractionF32ScalarCType))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "bound order", plan.boundOrder,
                kRVVContractionClampBoundOrder))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "clamp relation", plan.clampRelation,
                kRVVContractionProductReductionDequantClampRelation))
          return error;
        if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
                plan, "select layout", plan.selectLayout,
                kRVVContractionClampSelectLayout))
          return error;
      } else if (!plan.lowerBoundRole.empty() ||
                 !plan.upperBoundRole.empty() ||
                 !plan.lowerBoundCType.empty() ||
                 !plan.upperBoundCType.empty() || !plan.boundOrder.empty() ||
                 !plan.clampRelation.empty() || !plan.selectLayout.empty()) {
        return makeRVVEmitCRouteProviderError(
            "product-reduction dequantization plan must not carry clamp "
            "mirrors unless it is the dequant-clamp operation");
      }
    } else if (!plan.dequantizationRelation.empty() ||
               !plan.dequantizeConvertIntrinsic.empty() ||
               !plan.dequantizeScaleIntrinsic.empty() ||
               !plan.dequantScaleRole.empty() ||
               !plan.dequantScaleCType.empty() ||
               !plan.dequantScaleName.empty() ||
               !plan.rhsBroadcastIntrinsic.empty()) {
      return makeRVVEmitCRouteProviderError(
          "plain product-reduction contraction plan must not carry "
          "dequantization mirrors");
    }
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "accumulator layout", plan.accumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "result layout", plan.resultLayout,
            kRVVWideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "widening dot relation", plan.relation,
            getContractionWideningDotProductRelation(
                plan.sourceSEW, plan.sourceLMUL, plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "widening product leaf", plan.wideningProductIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "reduction leaf", plan.contractionComputeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "scalar seed splat leaf", plan.scalarSeedSplatIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "reduction store VL", plan.reductionStoreVL,
            kRVVWideningDotProductStoreVL))
      return error;
  }

  if (plan.usesComputedMask) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask type", plan.maskTypeName,
            internContractionDerivedText(
                (llvm::Twine("!tcrv_rvv.mask<") + plan.elementTypeName +
                 ", \"" + plan.lmul + "\">")
                    .str())))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask C type", plan.maskCType,
            getContractionMaskCType(plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "compare leaf", plan.compareIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement,
            kRVVContractionMaskedInactiveLaneZeroingRequirement))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask role", plan.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask source", plan.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask memory form", plan.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
  } else if (plan.usesProductReductionDequantClamp) {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask type", plan.maskTypeName,
            internContractionDerivedText(
                (llvm::Twine("!tcrv_rvv.mask<") +
                 getContractionFloatElementTypeName(plan.sew) + ", \"" +
                 plan.lmul + "\">")
                    .str())))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask C type", plan.maskCType,
            getContractionMaskCType(plan.sew, plan.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "lower compare predicate", plan.comparePredicateKind,
            kRVVPreRealizedPredicateKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "upper compare predicate",
            plan.secondaryComparePredicateKind, kRVVPreRealizedPredicateKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "lower compare leaf", plan.compareIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "upper compare leaf", plan.secondaryCompareIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDerivedLeaf(
            plan, "clamp select leaf", plan.maskedMergeIntrinsic,
            plan.typedConfigFactsID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask role", plan.maskRole, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask source", plan.maskSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask memory form", plan.maskMemoryForm, ""))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask type", plan.maskTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask C type", plan.maskCType, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "compare leaf", plan.compareIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "masked widening product leaf",
            plan.maskedWideningProductIntrinsic, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "inactive-lane zeroing requirement",
            plan.inactiveLaneZeroingRequirement, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask role", plan.maskRole, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask source", plan.maskSource, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionPlanField(
            plan, "mask memory form", plan.maskMemoryForm, ""))
      return error;
  }

  if (llvm::Error error =
          verifyRVVLowPrecisionContractionResourceSelection(
              plan, "contraction route-family target-leaf/profile validation"))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionMeasurementDispositionEvidence(
              "contraction route-family target-leaf/profile validation "
              "low-precision measurement-disposition policy boundary",
              plan.lowPrecisionResourceSelection))
    return error;
  return llvm::Error::success();
}

void applyContractionRuntimeAVLVLControlPlanToDescription(
    const RVVRuntimeAVLVLControlPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  description.sew = plan.sew;
  description.lmul = plan.lmul;
  description.tailPolicy = plan.tailPolicy;
  description.maskPolicy = plan.maskPolicy;
  description.runtimeControlPlanID = plan.controlPlanID;
  description.configContractID = plan.configContractID;
  description.runtimeVLContractID = plan.runtimeVLContractID;
  description.runtimeAVLASource = plan.runtimeAVLASource;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.vlDefOpName = plan.vlDefOpName;
  description.vlScopeOpName = plan.vlScopeOpName;
  description.vlUses = plan.vlUses;
  description.emitCLoopKind = plan.emitCLoopKind;
  description.emitCLoopInductionName = plan.emitCLoopInductionName;
  description.emitCFullChunkVLName = plan.emitCFullChunkVLName;
  description.emitCLoopVLName = plan.emitCLoopVLName;
  description.remainingAVLMetadata = plan.remainingAVLMetadata;
  description.pointerAdvanceMetadata = plan.pointerAdvanceMetadata;
  description.boundedSlice = plan.boundedSlice;
  description.multiVL = plan.multiVL;
}

llvm::Expected<RVVSelectedBodyContractionRouteFamilyPlan>
deriveRVVSelectedBodyContractionRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyContractionRouteOperation(operation))
    return makeRVVEmitCRouteProviderError(
        "requested contraction route-family plan for non-contraction RVV "
        "operation");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          getRVVSelectedBodyContractionRuntimeABIOrder(operation),
          "contraction route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  const RVVSelectedBodyTypedConfigFacts &typedConfig =
      analysis.typedConfigFacts;
  const bool isProductReductionDequantClamp =
      operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isDeferredWideProductReductionDequantization =
      operation ==
      RVVSelectedBodyOperationKind::
          WideningProductDeferredAccumulateReduceDequantizeF32;
  const bool isProductReductionDequantization =
      operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      isProductReductionDequantClamp ||
      isDeferredWideProductReductionDequantization;
  const bool isProductReductionChain =
      operation == RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      isProductReductionDequantization;
  // The deferred-wide i16 dot-reduce terminal kind (2nd kernel family): a
  // dot-reduce route whose realized body decomposes the narrow fused
  // tcrv_rvv.widening_dot_reduce into widening_product -> deferred_accumulate ->
  // standalone_reduce. Its accumulator/result layouts come from the trailing
  // tcrv_rvv.standalone_reduce, its source operands from the widening_product
  // head. Its ROUTE IDENTITY mirrors the narrow dot-reduce (source i16mf2, result
  // i32m1); only the realized PRIMITIVE intrinsics are wide.
  const bool isDeferredWideDotReduce =
      operation ==
      RVVSelectedBodyOperationKind::
          WideningProductDeferredDotAccumulateReduceAdd;
  mlir::Value lhsSourceValue =
      (isProductReductionChain || isDeferredWideDotReduce)
          ? productSlotLhs(analysis.slice)
          : analysis.slice.arithmeticLhs;
  mlir::Value rhsSourceValue =
      (isProductReductionChain || isDeferredWideDotReduce)
          ? productSlotRhs(analysis.slice)
          : analysis.slice.arithmeticRhs;
  llvm::Expected<RVVContractionVectorFacts> lhsSourceFacts =
      deriveContractionVectorFacts(
          lhsSourceValue, "lhs", "contraction route-family plan");
  if (!lhsSourceFacts)
    return lhsSourceFacts.takeError();
  llvm::Expected<RVVContractionVectorFacts> rhsSourceFacts =
      deriveContractionVectorFacts(
          rhsSourceValue, "rhs", "contraction route-family plan");
  if (!rhsSourceFacts)
    return rhsSourceFacts.takeError();
  if (llvm::Error error = requireMatchingContractionSourceFacts(
          *lhsSourceFacts, *rhsSourceFacts, "contraction route-family plan"))
    return std::move(error);
  std::optional<RVVContractionVectorFacts> wideningProductResultFacts;
  if (operation == RVVSelectedBodyOperationKind::WideningProduct) {
    llvm::Expected<RVVContractionVectorFacts> derivedResultFacts =
        deriveContractionVectorFacts(analysis.slice.wideningProductOp.getResult(),
                                     "widening product result",
                                     "contraction route-family plan");
    if (!derivedResultFacts)
      return derivedResultFacts.takeError();
    wideningProductResultFacts = std::move(*derivedResultFacts);
  }
  std::optional<RVVContractionVectorFacts> productFacts;
  std::optional<RVVContractionVectorFacts> reductionResultFacts;
  if (isProductReductionChain) {
    // In the deferred-wide chain the trailing standalone_reduce consumes the
    // i32m8 accumulate result; reduceInputSlotResult returns that wide carrier
    // when present, else the narrow product result (I5).
    if (analysis.slice.standaloneReduceOp.getInput() !=
        reduceInputSlotResult(analysis.slice))
      return makeRVVEmitCRouteProviderError(
          "product-reduction contraction route-family plan requires "
          "tcrv_rvv.standalone_reduce input to consume the selected "
          "tcrv_rvv.widening_product result");
    if (isProductReductionDequantization &&
        !analysis.slice.dequantizeOp)
      return makeRVVEmitCRouteProviderError(
          "product-reduction dequantization contraction route-family plan "
          "requires tcrv_rvv.dequantize in the selected RVV body");
    // Two carriers of the i32 product-reduction result feed the dequant:
    //   - legacy two-scope body: the gearbox_cross_region_handoff output;
    //   - single-scope typed body (Stage 3 flip): the standalone_reduce result
    //     directly, with no handoff op present.
    if (isProductReductionDequantization &&
        analysis.slice.gearboxCrossRegionHandoffOp &&
        (analysis.slice.gearboxCrossRegionHandoffOp.getInput() !=
             analysis.slice.standaloneReduceOp.getResult() ||
         analysis.slice.gearboxCrossRegionHandoffOp.getOutput() !=
             analysis.slice.dequantizeOp.getSource() ||
         analysis.slice.gearboxCrossRegionHandoffOp.getVl() !=
             analysis.slice.withVL.getVl() ||
         analysis.slice.gearboxCrossRegionHandoffOp.getRuntimeAvl() !=
             analysis.slice.setvl.getAvl()))
      return makeRVVEmitCRouteProviderError(
          "product-reduction dequantization contraction route-family plan "
          "requires tcrv_rvv.gearbox_cross_region_handoff to structurally "
          "forward the selected standalone_reduce result, bind the selected "
          "with_vl token, and consume the selected runtime n/AVL SSA value "
          "before tcrv_rvv.dequantize");
    if (isProductReductionDequantization &&
        !analysis.slice.gearboxCrossRegionHandoffOp &&
        analysis.slice.dequantizeOp.getSource() !=
            analysis.slice.standaloneReduceOp.getResult())
      return makeRVVEmitCRouteProviderError(
          "product-reduction dequantization contraction route-family plan "
          "requires tcrv_rvv.dequantize to consume the selected "
          "tcrv_rvv.standalone_reduce i32 result in the single-scope typed "
          "body");
    if (isProductReductionDequantization &&
        analysis.slice.dequantScaleABI.role !=
            support::RuntimeABIParameterRole::DequantScaleValue)
      return makeRVVEmitCRouteProviderError(
          "product-reduction dequantization contraction route-family plan "
          "requires runtime scale ABI role dequant-scale-value");
    if (isProductReductionDequantClamp) {
      if (!analysis.slice.lowerBoundScalarSplat ||
          !analysis.slice.upperBoundScalarSplat ||
          !analysis.slice.compareOp || !analysis.slice.secondaryCompareOp ||
          !analysis.slice.selectOp || !analysis.slice.secondarySelectOp)
        return makeRVVEmitCRouteProviderError(
            "product-reduction dequant-clamp contraction route-family plan "
            "requires lower/upper bound splats and two compare/select stages "
            "in the selected RVV body");
      if (analysis.slice.lowerBoundABI.role !=
              support::RuntimeABIParameterRole::LowerBoundScalarValue ||
          analysis.slice.upperBoundABI.role !=
              support::RuntimeABIParameterRole::UpperBoundScalarValue)
        return makeRVVEmitCRouteProviderError(
            "product-reduction dequant-clamp contraction route-family plan "
            "requires lower_bound and upper_bound runtime ABI roles");
      if (analysis.slice.lowerBoundScalarSplat.getBroadcast() !=
              analysis.slice.lowerBoundValue ||
          analysis.slice.upperBoundScalarSplat.getBroadcast() !=
              analysis.slice.upperBoundValue ||
          analysis.slice.compareOp.getLhs() !=
              analysis.slice.dequantizeOp.getResult() ||
          analysis.slice.compareOp.getRhs() !=
              analysis.slice.lowerBoundScalarSplat.getBroadcast() ||
          analysis.slice.selectOp.getMask() !=
              analysis.slice.compareOp.getMask() ||
          analysis.slice.selectOp.getTrueValue() !=
              analysis.slice.lowerBoundScalarSplat.getBroadcast() ||
          analysis.slice.selectOp.getFalseValue() !=
              analysis.slice.dequantizeOp.getResult() ||
          analysis.slice.secondaryCompareOp.getLhs() !=
              analysis.slice.upperBoundScalarSplat.getBroadcast() ||
          analysis.slice.secondaryCompareOp.getRhs() !=
              analysis.slice.selectOp.getSelected() ||
          analysis.slice.secondarySelectOp.getMask() !=
              analysis.slice.secondaryCompareOp.getMask() ||
          analysis.slice.secondarySelectOp.getTrueValue() !=
              analysis.slice.upperBoundScalarSplat.getBroadcast() ||
          analysis.slice.secondarySelectOp.getFalseValue() !=
              analysis.slice.selectOp.getSelected() ||
          analysis.slice.genericStore.getValue() !=
              analysis.slice.secondarySelectOp.getSelected())
        return makeRVVEmitCRouteProviderError(
            "product-reduction dequant-clamp contraction route-family plan "
            "requires dequant result -> lower clamp -> upper clamp -> f32 "
            "store structural dataflow");
    }
    llvm::Expected<RVVContractionVectorFacts> derivedProductFacts =
        deriveContractionVectorFacts(productSlotResult(analysis.slice),
                                     "product intermediate",
                                     "contraction route-family plan");
    if (!derivedProductFacts)
      return derivedProductFacts.takeError();
    productFacts = std::move(*derivedProductFacts);
    llvm::Expected<RVVContractionVectorFacts> derivedReductionResultFacts =
        deriveContractionVectorFacts(analysis.slice.standaloneReduceOp.getResult(),
                                     "product-reduction result",
                                     "contraction route-family plan");
    if (!derivedReductionResultFacts)
      return derivedReductionResultFacts.takeError();
    reductionResultFacts = std::move(*derivedReductionResultFacts);
  }

  RVVSelectedBodyContractionRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesWideningMAcc =
      operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  plan.usesWideningProduct =
      operation == RVVSelectedBodyOperationKind::WideningProduct;
  plan.usesProductReductionChain = isProductReductionChain;
  plan.usesProductReductionDequantization = isProductReductionDequantization;
  plan.usesProductReductionDequantClamp = isProductReductionDequantClamp;
  plan.usesDotReduction =
      isRVVSelectedBodyContractionDotReduction(operation);
  plan.usesComputedMask =
      isRVVSelectedBodyContractionComputedMask(operation);
  plan.usesStridedInputs =
      isRVVSelectedBodyContractionStridedInputs(operation);
  const bool isStandaloneUnsignedWideningProduct =
      plan.usesWideningProduct && lhsSourceFacts->isUnsigned &&
      rhsSourceFacts->isUnsigned && wideningProductResultFacts &&
      wideningProductResultFacts->isUnsigned;
  const bool isSignedProductReductionChain =
      plan.usesProductReductionChain && productFacts && reductionResultFacts &&
      !lhsSourceFacts->isUnsigned && !rhsSourceFacts->isUnsigned &&
      !productFacts->isUnsigned && !reductionResultFacts->isUnsigned;
  const bool isUnsignedProductReductionChain =
      plan.usesProductReductionChain &&
      !plan.usesProductReductionDequantization && productFacts &&
      reductionResultFacts && lhsSourceFacts->isUnsigned &&
      rhsSourceFacts->isUnsigned && productFacts->isUnsigned &&
      reductionResultFacts->isUnsigned;
  if (plan.usesProductReductionChain && !isSignedProductReductionChain &&
      !isUnsignedProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        "product-reduction contraction route-family plan requires source, "
        "product, accumulator, and result facts to form either a signed "
        "i8/i16/i32 chain or a plain unsigned u8/u16/u32 chain; unsigned "
        "dequantization/clamp is not part of this bounded plain "
        "product-reduction support path");
  plan.usesScalarSeed = plan.usesDotReduction;
  plan.usesVectorAccumulator = plan.usesWideningMAcc;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.typedConfigFactsID = typedConfig.factsID;
  plan.elementTypeName =
      isProductReductionDequantization
          ? getContractionFloatElementTypeName(typedConfig.sew)
      : reductionResultFacts
          ? reductionResultFacts->elementTypeName
      : wideningProductResultFacts
          ? wideningProductResultFacts->elementTypeName
          : typedConfig.elementTypeName;
  plan.elementBitWidth = typedConfig.elementBitWidth;
  plan.sew = typedConfig.sew;
  plan.lmul = typedConfig.lmul;
  plan.tailPolicy = typedConfig.tailPolicy;
  plan.maskPolicy = typedConfig.maskPolicy;
  plan.configContractID = typedConfig.configContractID;
  plan.familyPlanID = kRVVContractionRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile =
      isProductReductionDequantClamp
          ? "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1"
      : isProductReductionDequantization
          ? "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1"
      : isProductReductionChain
          ? (isUnsignedProductReductionChain
                 ? "rvv-v1-u8mf4-u16mf2-u32m1-product-reduction-contraction-leaf-profile.v1"
                 : "rvv-v1-i8mf4-i16mf2-i32m1-product-reduction-contraction-leaf-profile.v1")
          : getContractionTargetLeafProfile(lhsSourceFacts->sew,
                                            lhsSourceFacts->lmul,
                                            typedConfig.sew,
                                            typedConfig.lmul,
                                            isStandaloneUnsignedWideningProduct);
  plan.providerSupportedMirror = kRVVContractionProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations = kRVVContractionRequiredHeaderDeclarations;
  plan.cTypeMappingSummary =
      isProductReductionDequantClamp
          ? "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat/clamped:float-e32m1,scale:float,lower:float,upper:float"
      : isProductReductionDequantization
          ? "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float"
      : isProductReductionChain && productFacts
          ? getContractionProductReductionChainCTypeMappingSummary(
                lhsSourceFacts->sew, lhsSourceFacts->lmul, productFacts->sew,
                productFacts->lmul, typedConfig.sew, typedConfig.lmul,
                isUnsignedProductReductionChain)
          : getContractionCTypeMappingSummary(lhsSourceFacts->sew,
                                              lhsSourceFacts->lmul,
                                              typedConfig.sew,
                                              typedConfig.lmul,
                                              isStandaloneUnsignedWideningProduct);
  plan.vlCType = typedConfig.vlCType;
  plan.resultVectorTypeName =
      isProductReductionDequantization
          ? getContractionFloatVectorTypeName(typedConfig.sew,
                                              typedConfig.lmul)
      : reductionResultFacts
          ? reductionResultFacts->vectorTypeName
      : wideningProductResultFacts
          ? wideningProductResultFacts->vectorTypeName
          : typedConfig.vectorTypeName;
  plan.resultVectorCType =
      isProductReductionDequantization
          ? getContractionFloatVectorCType(typedConfig.sew, typedConfig.lmul)
      : reductionResultFacts
          ? reductionResultFacts->vectorCType
      : wideningProductResultFacts
          ? wideningProductResultFacts->vectorCType
          : typedConfig.vectorCType;
  plan.maskTypeName = plan.usesComputedMask ? typedConfig.maskTypeName : "";
  plan.maskCType = plan.usesComputedMask ? typedConfig.maskCType : "";
  plan.setVLIntrinsic = typedConfig.setVLIntrinsic;
  plan.sourceElementTypeName = lhsSourceFacts->elementTypeName;
  plan.sourceElementBitWidth = lhsSourceFacts->elementBitWidth;
  plan.sourceSEW = lhsSourceFacts->sew;
  plan.sourceLMUL = lhsSourceFacts->lmul;
  plan.sourceVectorTypeName = lhsSourceFacts->vectorTypeName;
  plan.sourceVectorCType = lhsSourceFacts->vectorCType;
  plan.sourceVectorLoadIntrinsic = lhsSourceFacts->vectorLoadIntrinsic;
  if (productFacts) {
    plan.productElementTypeName = productFacts->elementTypeName;
    plan.productElementBitWidth = productFacts->elementBitWidth;
    plan.productSEW = productFacts->sew;
    plan.productLMUL = productFacts->lmul;
    plan.productVectorTypeName = productFacts->vectorTypeName;
    plan.productVectorCType = productFacts->vectorCType;
  }
  if (plan.usesStridedInputs)
    plan.stridedLoadIntrinsic = lhsSourceFacts->stridedLoadIntrinsic;
  plan.storeIntrinsic =
      isProductReductionDequantization
          ? getContractionFloatStoreIntrinsic(typedConfig.sew,
                                              typedConfig.lmul)
      : reductionResultFacts
          ? reductionResultFacts->vectorStoreIntrinsic
      : wideningProductResultFacts
          ? wideningProductResultFacts->vectorStoreIntrinsic
          : typedConfig.storeIntrinsic;

  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (plan.usesComputedMask) {
    plan.runtimeABIParameters.push_back(analysis.slice.dotLHSABI);
    plan.runtimeABIParameters.push_back(analysis.slice.dotRHSABI);
  }
  if (!plan.usesWideningProduct)
    plan.runtimeABIParameters.push_back(analysis.slice.accumulatorABI);
  if (plan.usesProductReductionDequantization)
    plan.runtimeABIParameters.push_back(analysis.slice.dequantScaleABI);
  if (plan.usesProductReductionDequantClamp) {
    plan.runtimeABIParameters.push_back(analysis.slice.lowerBoundABI);
    plan.runtimeABIParameters.push_back(analysis.slice.upperBoundABI);
  }
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(
      plan.runtimeControlPlan.runtimeAVLParameter);
  if (plan.usesStridedInputs) {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsStrideABI);
    plan.runtimeABIParameters.push_back(analysis.slice.rhsStrideABI);
  }

  if (plan.usesWideningMAcc) {
    plan.wideningMAccArithmeticKind =
        analysis.slice.wideningMAccOp.getKind();
    plan.accumulatorLayout =
        analysis.slice.wideningMAccOp.getAccumulatorLayout();
    plan.resultLayout = analysis.slice.wideningMAccOp.getResultLayout();
    plan.relation = analysis.slice.wideningMAccOp.getMaccRelation();
    plan.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    plan.contractionComputeIntrinsic =
        getContractionWideningMAccIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
            typedConfig.lmul, plan.relation);
  } else if (plan.usesWideningProduct) {
    plan.wideningProductRelation =
        analysis.slice.wideningProductOp.getProductRelation();
    plan.relation = plan.wideningProductRelation;
    plan.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    plan.wideningProductIntrinsic =
        getContractionWideningProductIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
            typedConfig.lmul, plan.relation);
  } else if (plan.usesProductReductionChain) {
    plan.accumulatorLayout =
        analysis.slice.standaloneReduceOp.getAccumulatorLayout();
    plan.resultLayout = analysis.slice.standaloneReduceOp.getResultLayout();
    plan.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    plan.wideningProductRelation = productSlotRelation(analysis.slice);
    plan.productReductionChainRelation =
        getContractionProductReductionChainRelation(
            plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
            plan.productLMUL, typedConfig.sew, typedConfig.lmul,
            isUnsignedProductReductionChain);
    plan.relation = plan.productReductionChainRelation;
    plan.wideningProductIntrinsic =
        getContractionWideningProductIntrinsic(
            plan.sourceSEW, plan.sourceLMUL, plan.productSEW,
            plan.productLMUL, plan.wideningProductRelation);
    // The deferred-wide chain reduces the i32m8 deferred accumulate with a PLAIN
    // same-width vredsum (not the narrow widening vwredsum of the i16 product);
    // its trailing reduce is __riscv_vredsum_vs_i32m8_i32m1 (I5, derived from the
    // realized i32m8 accumulate -> i32m1 reduce, mirroring the primitive fact).
    plan.contractionComputeIntrinsic =
        plan.operation ==
                RVVSelectedBodyOperationKind::
                    WideningProductDeferredAccumulateReduceDequantizeF32
            ? llvm::StringRef("__riscv_vredsum_vs_i32m8_i32m1")
            : getContractionWideningReductionIntrinsic(
                  plan.productSEW, plan.productLMUL, typedConfig.sew,
                  typedConfig.lmul, plan.productReductionChainRelation);
    plan.scalarSeedSplatIntrinsic =
        getContractionScalarSeedSplatIntrinsic(typedConfig.sew,
                                               typedConfig.lmul,
                                               isUnsignedProductReductionChain);
    plan.reductionStoreVL = kRVVWideningDotProductStoreVL;
    if (plan.usesProductReductionDequantization) {
      plan.dequantizationRelation =
          analysis.slice.dequantizeOp.getDequantRelation();
      plan.dequantScaleRole = kRVVContractionDequantScaleRole;
      plan.dequantScaleCType = kRVVContractionDequantScaleCType;
      plan.dequantScaleName = kRVVContractionDequantScaleName;
      plan.rhsBroadcastIntrinsic =
          getContractionFloatScalarSplatIntrinsic(typedConfig.sew,
                                                  typedConfig.lmul);
      if (plan.usesProductReductionDequantClamp) {
        plan.lowerBoundRole = kRVVContractionLowerBoundRole;
        plan.upperBoundRole = kRVVContractionUpperBoundRole;
        plan.lowerBoundCType = kRVVContractionF32ScalarCType;
        plan.upperBoundCType = kRVVContractionF32ScalarCType;
        plan.boundOrder = kRVVContractionClampBoundOrder;
        plan.clampRelation =
            kRVVContractionProductReductionDequantClampRelation;
        plan.selectLayout = kRVVContractionClampSelectLayout;
        plan.comparePredicateKind = analysis.slice.compareOp.getKind();
        plan.secondaryComparePredicateKind =
            analysis.slice.secondaryCompareOp.getKind();
        plan.compareIntrinsic =
            getContractionFloatLessThanCompareIntrinsic(typedConfig.sew,
                                                        typedConfig.lmul);
        plan.secondaryCompareIntrinsic =
            getContractionFloatLessThanCompareIntrinsic(typedConfig.sew,
                                                        typedConfig.lmul);
        plan.maskedMergeIntrinsic =
            getContractionFloatSelectIntrinsic(typedConfig.sew,
                                               typedConfig.lmul);
        plan.maskTypeName = internContractionDerivedText(
            (llvm::Twine("!tcrv_rvv.mask<") +
             getContractionFloatElementTypeName(typedConfig.sew) + ", \"" +
             typedConfig.lmul + "\">")
                .str());
        plan.maskCType =
            getContractionMaskCType(typedConfig.sew, typedConfig.lmul);
      }
    }
  } else {
    if (plan.usesComputedMask) {
      plan.accumulatorLayout =
          analysis.slice.maskedWideningDotReduceOp.getAccumulatorLayout();
      plan.resultLayout =
          analysis.slice.maskedWideningDotReduceOp.getResultLayout();
      plan.relation =
          analysis.slice.maskedWideningDotReduceOp.getDotProductRelation();
      plan.maskRole = analysis.slice.maskedWideningDotReduceOp.getMaskRole();
      plan.maskSource =
          analysis.slice.maskedWideningDotReduceOp.getMaskSource();
      plan.maskMemoryForm =
          analysis.slice.maskedWideningDotReduceOp.getMaskMemoryForm();
      plan.compareIntrinsic =
          getContractionSignedLessThanCompareIntrinsic(typedConfig.sew,
                                                       typedConfig.lmul);
      plan.maskedMergeIntrinsic =
          getContractionSelectIntrinsic(typedConfig.sew, typedConfig.lmul);
      plan.maskedWideningProductIntrinsic =
          getContractionMaskedWideningProductIntrinsic(
              plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
              typedConfig.lmul, plan.relation);
      plan.inactiveLaneZeroingRequirement =
          kRVVContractionMaskedInactiveLaneZeroingRequirement;
    } else if (isDeferredWideDotReduce) {
      // The deferred-wide i16 dot-reduce has NO fused tcrv_rvv.widening_dot_reduce
      // op; its accumulator seed/result are carried by the trailing
      // tcrv_rvv.standalone_reduce. The route IDENTITY layouts/relation are the
      // narrow dot-reduce's (the logical op is unchanged): scalar i32 seed,
      // dot-reduction store, narrow signed-i16mf2 dot relation. Read the seed
      // structurally from the standalone_reduce (I5), use the narrow identity
      // layout/relation constants for the route facts.
      plan.accumulatorLayout = kRVVWideningDotProductAccumulatorLayout;
      plan.resultLayout = kRVVWideningDotProductResultLayout;
      plan.relation = getContractionWideningDotProductRelation(
          lhsSourceFacts->sew, lhsSourceFacts->lmul, typedConfig.sew,
          typedConfig.lmul);
    } else {
      plan.accumulatorLayout =
          analysis.slice.wideningDotReduceOp.getAccumulatorLayout();
      plan.resultLayout = analysis.slice.wideningDotReduceOp.getResultLayout();
      plan.relation =
          analysis.slice.wideningDotReduceOp.getDotProductRelation();
    }

    // The deferred-wide i16 dot-reduce reduces the i32m8 deferred accumulate with
    // a PLAIN same-width vredsum to i32m1 (NOT the narrow per-iteration fused
    // dot-reduce's i32m1 vredsum); its trailing reduce is the wide
    // __riscv_vredsum_vs_i32m8_i32m1, and its widening product is the wide
    // __riscv_vwmul_vv_i32m8. Both mirror the realized ops (I5); the route relation
    // stays the narrow identity relation.
    plan.contractionComputeIntrinsic =
        isDeferredWideDotReduce
            ? llvm::StringRef("__riscv_vredsum_vs_i32m8_i32m1")
        : plan.relation == getContractionWideningDotProductRelation(
                               plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
                               typedConfig.lmul)
            ? getContractionReductionIntrinsic(typedConfig.sew,
                                               typedConfig.lmul)
            : llvm::StringRef();
    plan.wideningProductIntrinsic =
        isDeferredWideDotReduce
            ? llvm::StringRef("__riscv_vwmul_vv_i32m8")
            : getContractionWideningProductIntrinsic(
                  plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
                  typedConfig.lmul, plan.relation);
    plan.scalarSeedSplatIntrinsic =
        plan.relation == getContractionWideningDotProductRelation(
                             plan.sourceSEW, plan.sourceLMUL, typedConfig.sew,
                             typedConfig.lmul)
            ? getContractionScalarSeedSplatIntrinsic(typedConfig.sew,
                                                     typedConfig.lmul)
            : llvm::StringRef();
    plan.reductionStoreVL = kRVVWideningDotProductStoreVL;
    if (operation == RVVSelectedBodyOperationKind::
                         ComputedMaskStridedInputWideningDotReduceAdd)
      plan.sourceAccumulatorResultContract =
          kRVVComputedMaskStridedInputWideningDotSourceAccumulatorResultContract;
    if (plan.usesStridedInputs) {
      plan.stridedMemoryLayout =
          plan.usesComputedMask
              ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
              : kRVVStridedInputWideningDotMemoryLayout;
      plan.lhsStrideSource = kRVVLHSStrideSource;
      plan.rhsStrideSource = kRVVRHSStrideSource;
      plan.sourceMemoryForm = kRVVStridedInputDotSourceMemoryForm;
      plan.destinationMemoryForm = kRVVDestinationMemoryForm;
    }
  }

  populateRVVLowPrecisionPrimitiveFacts(plan);

  if (expectsRVVLowPrecisionContractionResourceSelection(plan)) {
    if (plan.usesProductReductionDequantization) {
      llvm::Expected<RVVLowPrecisionContractionResourceSelection> selection =
          deriveRVVLowPrecisionContractionResourceSelectionFromPassFacts(
              plan, analysis.slice, analysis.selectedTargetCapabilityFacts,
              analysis.slice.withVL.getOperation(),
              analysis.description.lowPrecisionSelectedDispatchPolicyBoundary,
              "contraction route-family plan derivation");
      if (!selection)
        return selection.takeError();
      plan.lowPrecisionResourceSelection = std::move(*selection);
    } else {
      plan.lowPrecisionResourceSelection =
          deriveRVVLowPrecisionContractionResourceSelection(
              plan, analysis.selectedTargetCapabilityFacts);
    }
    if (llvm::Error error =
            populateRVVLowPrecisionSelectedBodyRealizationAdmissionProof(
                plan.lowPrecisionResourceSelection,
                analysis.description.lowPrecisionSelectedDispatchPolicyBoundary,
                "contraction route-family plan realization admission proof"))
      return std::move(error);
  }

  if (llvm::Error error =
          validateRVVSelectedBodyContractionRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyContractionRouteFamilyPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyContractionRuntimeAVLVLControlPlanToDescription(
      plan.runtimeControlPlan, description);
  // For the deferred-wide (N3) route the runtime control plan runs at the i8m2
  // STRIP config, but the route description's LOGICAL config is the i32m1/f32m1
  // RESULT config (plan.sew/lmul). Restore the result config + its config/VL
  // contract IDs (which the control-plan application overwrote with the strip
  // config) so the route profile / c-type mapping / header mirror key off the
  // result config. The narrow route's control plan == result config, so this is a
  // no-op there. The runtime-AVL/loop fields (control plan id, setvl intrinsic)
  // stay strip-config (the loop genuinely runs at i8m2).
  // The deferred-wide i16 dot-reduce route (2nd kernel family) shares the same
  // dual-config structure: the control plan runs at the i16m4 STRIP config, but
  // the route description's LOGICAL config is the i32m1 RESULT config. Restore it
  // the same way the byte deferred-wide dequant route does.
  if (plan.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32 ||
      plan.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredDotAccumulateReduceAdd) {
    description.sew = plan.sew;
    description.lmul = plan.lmul;
    const tcrv::rvv::RVVSelectedBodyConfigVLContract &resultContract =
        tcrv::rvv::getRVVSelectedBodyConfigVLContract(
            plan.sew, plan.lmul, plan.runtimeControlPlan.policy);
    description.configContractID = resultContract.configContractID;
    description.runtimeVLContractID = resultContract.runtimeVLContractID;
    description.runtimeAVLASource = resultContract.runtimeAVLASource;
    description.boundedSlice = resultContract.boundedSlice;
    description.multiVL = resultContract.multiVL;
  }
  description.elementTypeName = plan.elementTypeName;
  description.resultElementTypeName = plan.elementTypeName;
  description.contractionRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.resultVectorTypeName;
  description.vectorCType = plan.resultVectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.sourceSEW = plan.sourceSEW;
  description.sourceLMUL = plan.sourceLMUL;
  description.sourceVectorTypeName = plan.sourceVectorTypeName;
  description.sourceVectorCType = plan.sourceVectorCType;
  description.sourceVectorLoadIntrinsic = plan.sourceVectorLoadIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  populateRVVLowPrecisionPrimitiveRoutePayload(
      description.lowPrecisionPrimitiveRoutePayload, plan);
  populateRVVLowPrecisionPrimitiveDescriptionMirrorsFromPayload(description);
  description.lowPrecisionResourceSelection =
      plan.lowPrecisionResourceSelection;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());

  if (plan.usesWideningMAcc) {
    description.maccArithmeticKind = plan.wideningMAccArithmeticKind;
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
    description.wideningMAccAccumulatorLayout = plan.accumulatorLayout;
    description.wideningMAccResultLayout = plan.resultLayout;
    description.wideningMAccRelation = plan.relation;
    description.intrinsic = plan.contractionComputeIntrinsic;
    return;
  }

  if (plan.usesWideningProduct) {
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
    description.wideningProductRelation = plan.wideningProductRelation;
    description.wideningProductMultiplicandRoleSummary =
        plan.wideningProductMultiplicandRoleSummary;
    description.wideningProductExtensionPolicy =
        plan.wideningProductExtensionPolicy;
    description.wideningProductIntrinsic = plan.wideningProductIntrinsic;
    description.intrinsic = plan.wideningProductIntrinsic;
    return;
  }

  if (plan.usesProductReductionChain) {
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
    description.productElementTypeName = plan.productElementTypeName;
    description.productSEW = plan.productSEW;
    description.productLMUL = plan.productLMUL;
    description.productVectorTypeName = plan.productVectorTypeName;
    description.productVectorCType = plan.productVectorCType;
    description.reductionAccumulatorLayout = plan.accumulatorLayout;
    description.reductionResultLayout = plan.resultLayout;
    description.wideningProductRelation = plan.wideningProductRelation;
    description.wideningProductMultiplicandRoleSummary =
        plan.wideningProductMultiplicandRoleSummary;
    description.wideningProductExtensionPolicy =
        plan.wideningProductExtensionPolicy;
    description.productReductionChainRelation =
        plan.productReductionChainRelation;
    description.intrinsic = plan.contractionComputeIntrinsic;
    description.wideningProductIntrinsic = plan.wideningProductIntrinsic;
    description.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
    description.reductionStoreVL = plan.reductionStoreVL;
    description.standaloneReductionScalarResultRuntimeBoundary =
        plan.usesProductReductionDequantization
            ? kRVVProductReductionDequantVectorCarryBoundary
            : kRVVProductReductionOutCarryBoundary;
    if (plan.usesProductReductionDequantization) {
      description.dequantizationRelation = plan.dequantizationRelation;
      description.dequantScaleRole = plan.dequantScaleRole;
      description.dequantScaleCType = plan.dequantScaleCType;
      description.dequantScaleName = plan.dequantScaleName;
      description.rhsBroadcastIntrinsic = plan.rhsBroadcastIntrinsic;
      if (plan.usesProductReductionDequantClamp) {
        description.lowerBoundRole = plan.lowerBoundRole;
        description.upperBoundRole = plan.upperBoundRole;
        description.lowerBoundCType = plan.lowerBoundCType;
        description.upperBoundCType = plan.upperBoundCType;
        description.boundOrder = plan.boundOrder;
        description.clampRelation = plan.clampRelation;
        description.selectLayout = plan.selectLayout;
        description.comparePredicateKind = plan.comparePredicateKind;
        description.secondaryComparePredicateKind =
            plan.secondaryComparePredicateKind;
        description.compareIntrinsic = plan.compareIntrinsic;
        description.secondaryCompareIntrinsic =
            plan.secondaryCompareIntrinsic;
        description.maskedMergeIntrinsic = plan.maskedMergeIntrinsic;
        description.rhsBroadcastIntrinsic = plan.rhsBroadcastIntrinsic;
      }
    }
    return;
  }

  description.wideningDotProductAccumulatorLayout = plan.accumulatorLayout;
  description.wideningDotProductResultLayout = plan.resultLayout;
  description.wideningDotProductRelation = plan.relation;
  description.wideningDotSourceAccumulatorResultContract =
      plan.sourceAccumulatorResultContract;
  description.intrinsic = plan.contractionComputeIntrinsic;
  description.compareIntrinsic = plan.compareIntrinsic;
  description.maskedMergeIntrinsic = plan.maskedMergeIntrinsic;
  description.wideningProductIntrinsic = plan.wideningProductIntrinsic;
  description.maskedWideningProductIntrinsic =
      plan.maskedWideningProductIntrinsic;
  description.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
  description.reductionStoreVL = plan.reductionStoreVL;
  description.inactiveLaneZeroingRequirement =
      plan.inactiveLaneZeroingRequirement;
  if (plan.usesComputedMask) {
    description.maskRole = plan.maskRole;
    description.maskSource = plan.maskSource;
    description.maskMemoryForm = plan.maskMemoryForm;
  }
  if (plan.usesStridedInputs) {
    description.stridedLoadIntrinsic = plan.stridedLoadIntrinsic;
    description.stridedMemoryLayout = plan.stridedMemoryLayout;
    description.lhsStrideSource = plan.lhsStrideSource;
    description.rhsStrideSource = plan.rhsStrideSource;
    description.sourceMemoryForm = plan.sourceMemoryForm;
    description.destinationMemoryForm = plan.destinationMemoryForm;
  }
}

llvm::Error requireRVVSelectedBodyContractionDescriptionField(
    llvm::StringRef context, llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " contraction route description requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireRVVSelectedBodyContractionDescriptionIntegerField(
    llvm::StringRef context, llvm::StringRef field, std::int64_t actual,
    std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " contraction route description requires " +
      field + " " + llvm::Twine(expected) + " but found " +
      llvm::Twine(actual));
}

llvm::Error verifyRVVSelectedBodyContractionRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!isRVVSelectedBodyContractionRouteOperation(description.operation))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requested contraction route description verification for "
        "non-contraction operation");

  const bool usesWideningMAcc =
      description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool usesWideningProduct =
      description.operation == RVVSelectedBodyOperationKind::WideningProduct;
  const bool usesProductReductionDequantClamp =
      description.operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool usesProductReductionDequantization =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      usesProductReductionDequantClamp ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32;
  const bool usesProductReductionChain =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
      usesProductReductionDequantization;
  const bool usesDotReduction =
      isRVVSelectedBodyContractionDotReduction(description.operation);
  const bool usesComputedMask =
      isRVVSelectedBodyContractionComputedMask(description.operation);
  const bool usesStridedInputs =
      isRVVSelectedBodyContractionStridedInputs(description.operation);
  const std::optional<RVVWideningProductRouteFacts> wideningProductFacts =
      usesWideningProduct ? getRVVWideningProductRouteFacts(description)
                           : std::nullopt;
  if (usesWideningProduct && !wideningProductFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " widening-product contraction route description requires "
        "provider-owned signed or unsigned canonical route facts");
  const std::optional<RVVWideningDotReduceRouteFacts> wideningDotFacts =
      usesProductReductionChain ? getRVVWideningDotReduceRouteFacts(description)
                                : std::nullopt;
  if (usesProductReductionChain && !wideningDotFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " product-reduction contraction route description requires "
        "provider-owned signed or unsigned canonical widening-reduction "
        "route facts");

  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "family plan", description.contractionRouteFamilyPlanID,
          kRVVContractionRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          getRVVSelectedBodyContractionRuntimeABIOrder(description.operation)))
    return error;
  const bool supportsNarrowProductReductionChain =
      description.sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULMF4() &&
      description.productSEW == tcrv::rvv::getRVVSEW16Bits() &&
      description.productLMUL == tcrv::rvv::getRVVLMULMF2() &&
      description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1();
  // The deferred-wide (N3) chain runs i8m2 -> i16m4 -> i32m1 (the i32m8 deferred
  // accumulate is the structural intermediate). Mirror the plan-level support.
  const bool supportsDeferredWideProductReductionChain =
      description.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32 &&
      description.sourceSEW == tcrv::rvv::getRVVSEW8Bits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULM2() &&
      description.productSEW == tcrv::rvv::getRVVSEW16Bits() &&
      description.productLMUL == tcrv::rvv::getRVVLMULM4() &&
      description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1();
  const bool supportsProductReductionChain =
      supportsNarrowProductReductionChain ||
      supportsDeferredWideProductReductionChain;
  if (usesProductReductionChain && !supportsProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " product-reduction contraction route description requires "
        "provider-derived 8-bit source -> 16-bit product -> 32-bit "
        "accumulator/result SEW/LMUL facts");
  // The deferred-wide i16 dot-reduce route description (2nd kernel family) carries
  // the wide realized source i16m4 -> result i32m1; admit it as a parallel config
  // (mirrors the plan-level supportsDeferredWideDotReduceConfig).
  const bool supportsDeferredWideDotReduceConfig =
      description.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredDotAccumulateReduceAdd &&
      description.sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
      description.sourceLMUL == tcrv::rvv::getRVVLMULM4() &&
      description.sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      description.lmul == tcrv::rvv::getRVVLMULM1();
  if (!usesProductReductionChain && !supportsDeferredWideDotReduceConfig &&
      !isSupportedContractionSourceResultConfig(description.sourceSEW,
                                                description.sourceLMUL,
                                                description.sew,
                                                description.lmul))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " contraction route description requires supported provider-derived "
        "source/result SEW/LMUL facts");
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "target leaf profile", description.targetLeafProfile,
          usesProductReductionDequantClamp
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-"
                    "dequant-clamp-leaf-profile.v1")
      : usesProductReductionDequantization
              ? llvm::StringRef(
                    "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-"
                    "dequantization-leaf-profile.v1")
          : usesProductReductionChain
              ? wideningDotFacts->targetLeafProfile
          : usesWideningProduct
              ? wideningProductFacts->targetLeafProfile
              : getContractionTargetLeafProfile(description.sourceSEW,
                                                description.sourceLMUL,
                                                description.sew,
                                                description.lmul)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "provider_supported_mirror",
          description.providerSupportedMirror,
          kRVVContractionProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "required header declarations",
          description.requiredHeaderDeclarations,
          kRVVContractionRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "C type mapping summary", description.cTypeMappingSummary,
          usesProductReductionDequantClamp
              ? llvm::StringRef(
                    "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,"
                    "seed:signed-i32,accumulator:signed-e32m1,"
                    "dequant-splat/clamped:float-e32m1,scale:float,"
                    "lower:float,upper:float")
      : usesProductReductionDequantization
              ? llvm::StringRef(
                    "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,"
                    "seed:signed-i32,accumulator:signed-e32m1,"
                    "dequant-splat:float-e32m1,scale:float")
          : usesProductReductionChain
              ? wideningDotFacts->cTypeMappingSummary
          : usesWideningProduct
              ? wideningProductFacts->cTypeMappingSummary
              : getContractionCTypeMappingSummary(description.sourceSEW,
                                                  description.sourceLMUL,
                                                  description.sew,
                                                  description.lmul)))
    return error;
  if (llvm::Error error =
          verifyRVVLowPrecisionContractionResourceDescriptionSelection(
              description, context))
    return error;
  const RVVLowPrecisionWideningReductionPrimitiveFacts *primitiveFacts =
      usesProductReductionChain
          ? &wideningDotFacts->lowPrecisionWideningReductionPrimitiveFacts
          : nullptr;
  if (usesWideningProduct || usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive contract",
            description.lowPrecisionPrimitiveContractID,
            kRVVLowPrecisionPrimitiveContractID))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive kind",
            description.lowPrecisionPrimitiveKind,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveKind
                : llvm::StringRef(
                      primitiveFacts->lowPrecisionPrimitiveKind)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source dtype",
            description.lowPrecisionPrimitiveSourceElementTypeName,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveSourceElementTypeName
                : llvm::StringRef(primitiveFacts->sourceElementTypeName)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source signedness",
            description.lowPrecisionPrimitiveSourceSignedness,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveSourceSignedness
                : llvm::StringRef(primitiveFacts->sourceSignedness)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source load",
            description.lowPrecisionPrimitiveSourceLoadKind,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveSourceLoadKind
                : llvm::StringRef(primitiveFacts->sourceLoadKind)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source extension",
            description.lowPrecisionPrimitiveSourceExtensionKind,
            usesWideningProduct
                ? wideningProductFacts
                      ->lowPrecisionPrimitiveSourceExtensionKind
                : llvm::StringRef(primitiveFacts->sourceExtensionKind)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive product dtype",
            description.lowPrecisionPrimitiveProductElementTypeName,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveProductElementTypeName
                : llvm::StringRef(primitiveFacts->productElementTypeName)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive accumulator dtype",
            description.lowPrecisionPrimitiveAccumulatorElementTypeName,
            usesWideningProduct
                ? llvm::StringRef()
                : llvm::StringRef(primitiveFacts->accumulatorElementTypeName)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive result dtype",
            description.lowPrecisionPrimitiveResultElementTypeName,
            usesWideningProduct
                ? wideningProductFacts->lowPrecisionPrimitiveResultElementTypeName
                : llvm::StringRef(primitiveFacts->finalResultElementTypeName)))
      return error;
  } else {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive contract",
            description.lowPrecisionPrimitiveContractID, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive kind",
            description.lowPrecisionPrimitiveKind, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source dtype",
            description.lowPrecisionPrimitiveSourceElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source signedness",
            description.lowPrecisionPrimitiveSourceSignedness, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source load",
            description.lowPrecisionPrimitiveSourceLoadKind, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive source extension",
            description.lowPrecisionPrimitiveSourceExtensionKind, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive product dtype",
            description.lowPrecisionPrimitiveProductElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive accumulator dtype",
            description.lowPrecisionPrimitiveAccumulatorElementTypeName, ""))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "low-precision primitive result dtype",
            description.lowPrecisionPrimitiveResultElementTypeName, ""))
      return error;
  }

  const RVVLowPrecisionPrimitiveRoutePayload &primitivePayload =
      description.lowPrecisionPrimitiveRoutePayload;
  if (usesWideningProduct || usesProductReductionChain) {
    if (!primitivePayload.hasPayload)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " contraction route description requires provider-owned "
          "low-precision primitive route payload before route construction");
    if (primitivePayload.isProductReductionChain != usesProductReductionChain)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " contraction route description low-precision primitive route "
          "payload must mirror the selected product-reduction boundary");
    if (usesProductReductionChain) {
      if (llvm::Error error =
              verifyRVVLowPrecisionPrimitiveRoutePayloadFromWideningReductionFacts(
                  primitivePayload, *primitiveFacts, description.tailPolicy,
                  description.maskPolicy, description.runtimeControlPlanID,
                  description.runtimeAVLASource, context))
        return error;
    }

    auto requirePayloadField =
        [&](llvm::StringRef field, llvm::StringRef actual,
            llvm::StringRef expected) -> llvm::Error {
      return requireRVVSelectedBodyContractionDescriptionField(
          context,
          (llvm::Twine("low-precision primitive route payload ") + field)
              .str(),
          actual, expected);
    };
    auto requirePayloadIntegerField =
        [&](llvm::StringRef field, std::int64_t actual,
            std::int64_t expected) -> llvm::Error {
      return requireRVVSelectedBodyContractionDescriptionIntegerField(
          context,
          (llvm::Twine("low-precision primitive route payload ") + field)
              .str(),
          actual, expected);
    };

#define TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(FIELD, ACTUAL, EXPECTED)    \
  if (llvm::Error error = requirePayloadField((FIELD), (ACTUAL), (EXPECTED)))  \
    return error
#define TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_INTEGER(FIELD, ACTUAL, EXPECTED)   \
  if (llvm::Error error =                                                       \
          requirePayloadIntegerField((FIELD), (ACTUAL), (EXPECTED)))           \
    return error

    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "contract", primitivePayload.contractID,
        description.lowPrecisionPrimitiveContractID);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "kind", primitivePayload.kind,
        description.lowPrecisionPrimitiveKind);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "source dtype", primitivePayload.sourceElementTypeName,
        description.lowPrecisionPrimitiveSourceElementTypeName);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "source signedness", primitivePayload.sourceSignedness,
        description.lowPrecisionPrimitiveSourceSignedness);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "source load", primitivePayload.sourceLoadKind,
        description.lowPrecisionPrimitiveSourceLoadKind);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "source extension", primitivePayload.sourceExtensionKind,
        description.lowPrecisionPrimitiveSourceExtensionKind);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "product dtype", primitivePayload.productElementTypeName,
        description.lowPrecisionPrimitiveProductElementTypeName);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "accumulator dtype", primitivePayload.accumulatorElementTypeName,
        description.lowPrecisionPrimitiveAccumulatorElementTypeName);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "result dtype", primitivePayload.resultElementTypeName,
        description.lowPrecisionPrimitiveResultElementTypeName);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_INTEGER(
        "source SEW", primitivePayload.sourceSEW, description.sourceSEW);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "source LMUL", primitivePayload.sourceLMUL, description.sourceLMUL);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_INTEGER(
        "product SEW", primitivePayload.productSEW,
        usesProductReductionChain ? description.productSEW : description.sew);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "product LMUL", primitivePayload.productLMUL,
        usesProductReductionChain ? description.productLMUL
                                  : description.lmul);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_INTEGER(
        "accumulator SEW", primitivePayload.accumulatorSEW,
        description.lowPrecisionPrimitiveAccumulatorElementTypeName.empty()
            ? 0
            : description.sew);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "accumulator LMUL", primitivePayload.accumulatorLMUL,
        description.lowPrecisionPrimitiveAccumulatorElementTypeName.empty()
            ? llvm::StringRef()
            : description.lmul);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_INTEGER(
        "result SEW", primitivePayload.resultSEW, description.sew);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "result LMUL", primitivePayload.resultLMUL, description.lmul);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "tail policy", primitivePayload.tailPolicy, description.tailPolicy);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "mask policy", primitivePayload.maskPolicy, description.maskPolicy);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "runtime control plan", primitivePayload.runtimeControlPlanID,
        description.runtimeControlPlanID);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "runtime AVL source", primitivePayload.runtimeAVLASource,
        description.runtimeAVLASource);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "widening product relation",
        primitivePayload.wideningProductRelation,
        description.wideningProductRelation);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "widening product intrinsic",
        primitivePayload.wideningProductIntrinsic,
        description.wideningProductIntrinsic);
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "product-reduction chain relation",
        primitivePayload.productReductionChainRelation,
        usesProductReductionChain ? description.productReductionChainRelation
                                  : llvm::StringRef());
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "reduction intrinsic", primitivePayload.reductionIntrinsic,
        usesProductReductionChain ? description.intrinsic
                                  : llvm::StringRef());
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "scalar seed splat intrinsic",
        primitivePayload.scalarSeedSplatIntrinsic,
        usesProductReductionChain ? description.scalarSeedSplatIntrinsic
                                  : llvm::StringRef());
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "accumulator layout", primitivePayload.accumulatorLayout,
        usesProductReductionChain ? description.reductionAccumulatorLayout
                                  : llvm::StringRef());
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "result layout", primitivePayload.resultLayout,
        usesProductReductionChain ? description.reductionResultLayout
                                  : llvm::StringRef());
    TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING(
        "reduction store VL", primitivePayload.reductionStoreVL,
        usesProductReductionChain ? description.reductionStoreVL
                                  : llvm::StringRef());

#undef TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_STRING
#undef TCRV_REQUIRE_DESC_PRIMITIVE_PAYLOAD_INTEGER
  } else if (primitivePayload.hasPayload) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " contraction route description rejects low-precision primitive "
        "route payload without a low-precision primitive route");
  }

  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector type", description.sourceVectorTypeName,
          usesWideningProduct
              ? wideningProductFacts->sourceVectorTypeName
          : usesProductReductionChain
              ? llvm::StringRef(primitiveFacts->sourceVectorTypeName)
              : getContractionVectorTypeName(description.sourceSEW,
                                             description.sourceLMUL)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector C type", description.sourceVectorCType,
          usesWideningProduct
              ? wideningProductFacts->sourceVectorCType
          : usesProductReductionChain
              ? llvm::StringRef(primitiveFacts->sourceVectorCType)
              : getContractionSignedVectorCType(description.sourceSEW,
                                                description.sourceLMUL)))
    return error;
  if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
          context, "source vector-load intrinsic",
          description.sourceVectorLoadIntrinsic,
          usesWideningProduct
              ? wideningProductFacts->sourceVectorLoadIntrinsic
          : usesProductReductionChain
              ? wideningDotFacts->sourceVectorLoadIntrinsic
              : getContractionVectorLoadIntrinsic(description.sourceSEW,
                                                  description.sourceLMUL)))
    return error;
  if (usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "product element type", description.productElementTypeName,
            primitiveFacts->productElementTypeName))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "product vector type", description.productVectorTypeName,
            primitiveFacts->productVectorTypeName))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "product vector C type", description.productVectorCType,
            primitiveFacts->productVectorCType))
      return error;
  }

  if (usesWideningMAcc) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate arithmetic kind",
            description.maccArithmeticKind, kRVVPreRealizedWideningMAccOpKind))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate accumulator layout",
            description.wideningMAccAccumulatorLayout,
            kRVVWideningMAccAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate result layout",
            description.wideningMAccResultLayout,
            kRVVWideningMAccResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate relation",
            description.wideningMAccRelation,
            getContractionWideningMAccRelation(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening multiply-accumulate intrinsic",
            description.intrinsic,
            getContractionWideningMAccIntrinsic(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul,
                description.wideningMAccRelation)))
      return error;
  }

  if (usesWideningProduct) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product relation",
            description.wideningProductRelation,
            wideningProductFacts->wideningProductRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product multiplicand roles",
            description.wideningProductMultiplicandRoleSummary,
            wideningProductFacts->wideningProductMultiplicandRoleSummary))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product extension policy",
            description.wideningProductExtensionPolicy,
            wideningProductFacts->wideningProductExtensionPolicy))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic,
            wideningProductFacts->wideningProductIntrinsic))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "compute intrinsic", description.intrinsic,
            description.wideningProductIntrinsic))
      return error;
  }

  if (usesProductReductionChain) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVUnitStrideSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout,
            kRVVProductReductionResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product relation",
            description.wideningProductRelation,
            primitiveFacts->wideningProductRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product multiplicand roles",
            description.wideningProductMultiplicandRoleSummary,
            primitiveFacts->sourceSignedness == "unsigned"
                ? llvm::StringRef(
                      kRVVLowPrecisionUnsignedWideningProductMultiplicandRoles)
                : llvm::StringRef(
                      kRVVLowPrecisionSignedWideningProductMultiplicandRoles)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product extension policy",
            description.wideningProductExtensionPolicy,
            primitiveFacts->sourceSignedness == "unsigned"
                ? llvm::StringRef(
                      kRVVLowPrecisionUnsignedWideningProductExtensionPolicy)
                : llvm::StringRef(
                      kRVVLowPrecisionSignedWideningProductExtensionPolicy)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "product-reduction relation",
            description.productReductionChainRelation,
            wideningDotFacts->productReductionChainRelation))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic,
            wideningDotFacts->wideningProductIntrinsic))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening reduction intrinsic", description.intrinsic,
            wideningDotFacts->reductionIntrinsic))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic,
            wideningDotFacts->scalarSeedSplatIntrinsic))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction store VL", description.reductionStoreVL,
            kRVVWideningDotProductStoreVL))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "scalar result runtime boundary",
            description.standaloneReductionScalarResultRuntimeBoundary,
            usesProductReductionDequantization
                ? kRVVProductReductionDequantVectorCarryBoundary
                : kRVVProductReductionOutCarryBoundary))
      return error;
    if (usesProductReductionDequantization) {
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "dequantization relation",
                  description.dequantizationRelation,
                  kRVVContractionDequantizationRelation))
        return error;
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "dequant scale role", description.dequantScaleRole,
                  kRVVContractionDequantScaleRole))
        return error;
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "dequant scale C type",
                  description.dequantScaleCType,
                  kRVVContractionDequantScaleCType))
        return error;
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "dequant scale name", description.dequantScaleName,
                  kRVVContractionDequantScaleName))
        return error;
      if (llvm::Error error =
              requireRVVSelectedBodyContractionDescriptionField(
                  context, "post-loop scalar dequant splat intrinsic",
                  description.rhsBroadcastIntrinsic,
                  getContractionFloatScalarSplatIntrinsic(description.sew,
                                                          description.lmul)))
        return error;
      if (usesProductReductionDequantClamp) {
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "lower bound role", description.lowerBoundRole,
                    kRVVContractionLowerBoundRole))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "upper bound role", description.upperBoundRole,
                    kRVVContractionUpperBoundRole))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "lower bound C type", description.lowerBoundCType,
                    kRVVContractionF32ScalarCType))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "upper bound C type", description.upperBoundCType,
                    kRVVContractionF32ScalarCType))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "bound order", description.boundOrder,
                    kRVVContractionClampBoundOrder))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "clamp relation", description.clampRelation,
                    kRVVContractionProductReductionDequantClampRelation))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "select layout", description.selectLayout,
                    kRVVContractionClampSelectLayout))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "compare predicate kind",
                    description.comparePredicateKind,
                    kRVVPreRealizedPredicateKind))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "secondary compare predicate kind",
                    description.secondaryComparePredicateKind,
                    kRVVPreRealizedPredicateKind))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "compare intrinsic", description.compareIntrinsic,
                    getContractionFloatLessThanCompareIntrinsic(
                        description.sew, description.lmul)))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "secondary compare intrinsic",
                    description.secondaryCompareIntrinsic,
                    getContractionFloatLessThanCompareIntrinsic(
                        description.sew, description.lmul)))
          return error;
        if (llvm::Error error =
                requireRVVSelectedBodyContractionDescriptionField(
                    context, "masked merge intrinsic",
                    description.maskedMergeIntrinsic,
                    getContractionFloatSelectIntrinsic(description.sew,
                                                       description.lmul)))
          return error;
      } else if (!description.lowerBoundRole.empty() ||
                 !description.upperBoundRole.empty() ||
                 !description.lowerBoundCType.empty() ||
                 !description.upperBoundCType.empty() ||
                 !description.boundOrder.empty() ||
                 !description.clampRelation.empty() ||
                 !description.selectLayout.empty() ||
                 !description.comparePredicateKind.empty() ||
                 !description.secondaryComparePredicateKind.empty() ||
                 !description.compareIntrinsic.empty() ||
                 !description.secondaryCompareIntrinsic.empty() ||
                 !description.maskedMergeIntrinsic.empty()) {
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " product-reduction dequantization route description must not "
            "carry clamp mirrors unless it is the dequant-clamp operation");
      }
    } else if (!description.dequantizationRelation.empty() ||
               !description.dequantizeConvertIntrinsic.empty() ||
               !description.dequantizeScaleIntrinsic.empty() ||
               !description.dequantScaleRole.empty() ||
               !description.dequantScaleCType.empty() ||
               !description.dequantScaleName.empty() ||
               !description.rhsBroadcastIntrinsic.empty()) {
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " plain product-reduction contraction route description must not "
          "carry dequantization mirrors");
    }
    return llvm::Error::success();
  }

  if (usesDotReduction) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product accumulator layout",
            description.wideningDotProductAccumulatorLayout,
            kRVVWideningDotProductAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product result layout",
            description.wideningDotProductResultLayout,
            kRVVWideningDotProductResultLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product relation",
            description.wideningDotProductRelation,
            getContractionWideningDotProductRelation(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot source/accumulator/result contract",
            description.wideningDotSourceAccumulatorResultContract,
            description.operation ==
                    RVVSelectedBodyOperationKind::
                        ComputedMaskStridedInputWideningDotReduceAdd
                ? llvm::StringRef(
                      kRVVComputedMaskStridedInputWideningDotSourceAccumulatorResultContract)
                : llvm::StringRef()))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening product intrinsic",
            description.wideningProductIntrinsic,
            getContractionWideningProductIntrinsic(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul,
                description.wideningDotProductRelation)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "reduction intrinsic", description.intrinsic,
            getContractionReductionIntrinsic(description.sew,
                                             description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "scalar seed splat intrinsic",
            description.scalarSeedSplatIntrinsic,
            getContractionScalarSeedSplatIntrinsic(description.sew,
                                                   description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "widening dot-product reduction store VL",
            description.reductionStoreVL, kRVVWideningDotProductStoreVL))
      return error;
  }

  if (usesComputedMask) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            getContractionSignedLessThanCompareIntrinsic(description.sew,
                                                         description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic,
            getContractionSelectIntrinsic(description.sew, description.lmul)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "masked widening product intrinsic",
            description.maskedWideningProductIntrinsic,
            getContractionMaskedWideningProductIntrinsic(
                description.sourceSEW, description.sourceLMUL,
                description.sew, description.lmul,
                description.wideningDotProductRelation)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "inactive-lane zeroing requirement",
            description.inactiveLaneZeroingRequirement,
            kRVVContractionMaskedInactiveLaneZeroingRequirement))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask role", description.maskRole,
            kRVVMaskedPredicateMaskRole))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask source", description.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "mask memory form", description.maskMemoryForm,
            kRVVComputedMaskMemoryMaskMemoryForm))
      return error;
  }

  if (usesStridedInputs) {
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "strided-load intrinsic",
            description.stridedLoadIntrinsic,
            getContractionStridedLoadIntrinsic(description.sourceSEW,
                                               description.sourceLMUL)))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "strided memory layout",
            description.stridedMemoryLayout,
            usesComputedMask ? kRVVComputedMaskStridedInputWideningDotMemoryLayout
                             : kRVVStridedInputWideningDotMemoryLayout))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "lhs stride source", description.lhsStrideSource,
            kRVVLHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "rhs stride source", description.rhsStrideSource,
            kRVVRHSStrideSource))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "source memory form", description.sourceMemoryForm,
            kRVVStridedInputDotSourceMemoryForm))
      return error;
    if (llvm::Error error = requireRVVSelectedBodyContractionDescriptionField(
            context, "destination memory form",
            description.destinationMemoryForm, kRVVDestinationMemoryForm))
      return error;
  }

  return llvm::Error::success();
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    return kRVVWideningMAccOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningProduct:
    return kRVVWideningProductOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    return kRVVWideningProductReductionChainOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredAccumulateReduceDequantizeF32:
    return kRVVWideningProductReductionDequantizeOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    return kRVVWideningProductReductionDequantClampF32OperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  // The deferred-wide i16 dot-reduce terminal kind (2nd kernel family) shares the
  // narrow dot-reduce route identity, so it binds operands via the SAME
  // lhs/rhs/acc/out/n dot-reduce operand-binding plan.
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredDotAccumulateReduceAdd:
    return kRVVWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return kRVVStridedInputWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return kRVVComputedMaskWideningDotReduceOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID;
  default:
    return std::nullopt;
  }
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyContractionRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (planID == kRVVWideningProductOperandBindingPlanID ||
      planID == kRVVUnsignedWideningProductOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVWideningMAccOperandBindingPlanID ||
      planID == kRVVWideningProductReductionChainOperandBindingPlanID ||
      planID == kRVVWideningProductReductionDequantizeOperandBindingPlanID ||
      planID ==
          kRVVWideningProductReductionDequantClampF32OperandBindingPlanID ||
      planID == kRVVWideningDotReduceOperandBindingPlanID ||
      planID == kRVVStridedInputWideningDotReduceOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if ((planID == kRVVWideningProductReductionDequantizeOperandBindingPlanID ||
         planID ==
             kRVVWideningProductReductionDequantClampF32OperandBindingPlanID) &&
        logicalOperand == "scale")
      return RuntimeABIParameterRole::DequantScaleValue;
    if (planID ==
        kRVVWideningProductReductionDequantClampF32OperandBindingPlanID) {
      if (logicalOperand == "lower_bound")
        return RuntimeABIParameterRole::LowerBoundScalarValue;
      if (logicalOperand == "upper_bound")
        return RuntimeABIParameterRole::UpperBoundScalarValue;
    }
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (planID == kRVVStridedInputWideningDotReduceOperandBindingPlanID) {
      if (logicalOperand == "lhs_stride")
        return RuntimeABIParameterRole::LHSInputStride;
      if (logicalOperand == "rhs_stride")
        return RuntimeABIParameterRole::RHSInputStride;
    }
  }
  if (planID == kRVVComputedMaskWideningDotReduceOperandBindingPlanID ||
      planID ==
          kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "dot_lhs")
      return RuntimeABIParameterRole::DotLHSInputBuffer;
    if (logicalOperand == "dot_rhs")
      return RuntimeABIParameterRole::DotRHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (planID ==
        kRVVComputedMaskStridedInputWideningDotReduceOperandBindingPlanID) {
      if (logicalOperand == "lhs_stride")
        return RuntimeABIParameterRole::LHSInputStride;
      if (logicalOperand == "rhs_stride")
        return RuntimeABIParameterRole::RHSInputStride;
    }
  }
  return std::nullopt;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyContractionRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyRouteSlice &slice = analysis.slice;
  RVVRouteOperandBindingPlan plan;
  llvm::StringRef expectedRuntimeABIOrder =
      getContractionRuntimeABIOrder(slice.arithmeticKind);
  std::optional<llvm::StringRef> planID =
      getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
          slice.arithmeticKind);
  if (!planID)
    return plan;

  if (!analysis.contractionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
        " route requires the contraction route-family plan before deriving "
        "widening-contraction operand bindings");
  if (analysis.contractionRouteFamilyPlan->operation != slice.arithmeticKind)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
        " route requires same-operation contraction route-family plan facts");

  const bool isUnsignedWideningProduct =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::WideningProduct &&
      analysis.description.wideningProductRelation ==
          getContractionWideningProductRelation(
              tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
              tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true);
  const bool isUnsignedProductReduction =
      slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::WideningProductReduceAdd &&
      analysis.description.wideningProductRelation ==
          getContractionWideningProductRelation(
              tcrv::rvv::getRVVSEW8Bits(), tcrv::rvv::getRVVLMULMF4(),
              tcrv::rvv::getRVVSEW16Bits(), tcrv::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true);
  plan.planID = (isUnsignedWideningProduct
                     ? llvm::StringRef(
                           kRVVUnsignedWideningProductOperandBindingPlanID)
                     : *planID)
                    .str();
  llvm::StringRef context;
  switch (slice.arithmeticKind) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    context = "widening_macc_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wmacc-lhs", "src-i16mf2", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wmacc-rhs", "src-i16mf2", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "acc-load", "wmacc-acc", "acc-i32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "res-store", "res-i32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::WideningProduct:
    context = "widening_product route";
    {
      const llvm::StringRef sourceWidthUse =
          isUnsignedWideningProduct ? "src-u8mf4" : "src-i8mf4";
      const llvm::StringRef resultWidthUse =
          isUnsignedWideningProduct ? "res-u16mf2" : "res-i16mf2";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", sourceWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", sourceWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "res-store", resultWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
    }
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd: {
    context = "widening_product_reduce_add route";
    const llvm::StringRef sourceWidthUse =
        isUnsignedProductReduction ? "src-u8mf4" : "src-i8mf4";
    const llvm::StringRef accumulatorWidthUse =
        isUnsignedProductReduction ? "u32" : "i32";
    const llvm::StringRef resultWidthUse =
        isUnsignedProductReduction ? "res-u32m1" : "res-i32m1";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", sourceWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", sourceWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI,
        {"abi", "seed", "wred", accumulatorWidthUse, "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "acc-state", "store",
                                       resultWidthUse, "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  }
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    context = "widening_product_reduce_dequantize_f32 route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "wred", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "scale", slice.dequantScaleABI,
        {"abi", "runtime-scale", "scale-f32", "dequant", "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "dequant-result", "store", "res-f32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredAccumulateReduceDequantizeF32:
    // Same logical dequant route; the deferred-wide realization widens the
    // source strip to i8m2 (vs narrow i8mf4) feeding the i32m8 deferred
    // accumulate. The src-i8m2 marker mirrors the realized wide strip (I5).
    context = "widening_product_reduce_dequantize_f32 route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", "src-i8m2", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", "src-i8m2", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "wred", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "scale", slice.dequantScaleABI,
        {"abi", "runtime-scale", "scale-f32", "dequant", "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "dequant-result", "store", "res-f32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    context = "widening_product_reduce_dequant_clamp_f32 route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "src-load", "wprod-lhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "src-load", "wprod-rhs", "src-i8mf4", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "wred", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "scale", slice.dequantScaleABI,
        {"abi", "runtime-scale", "scale-f32", "dequant", "hdr"});
    addContractionRouteOperandBinding(
        plan, "lower_bound", slice.lowerBoundABI,
        {"abi", "runtime-lower", "splat", "compare", "select", "hdr"});
    addContractionRouteOperandBinding(
        plan, "upper_bound", slice.upperBoundABI,
        {"abi", "runtime-upper", "splat", "compare", "select", "hdr"});
    addContractionRouteOperandBinding(
        plan, "out", slice.outABI,
        {"abi", "clamped-dequant-result", "store", "res-f32m1", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  // The deferred-wide i16 dot-reduce terminal kind binds the SAME lhs/rhs/acc/
  // out/n operands as the narrow dot-reduce (route identity preserved; the wide
  // realization is internal). Same i16 source / i32 seed+store markers.
  case RVVSelectedBodyOperationKind::
      WideningProductDeferredDotAccumulateReduceAdd:
    context = "widening_dot_reduce_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "ld", "dot-lhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI, {"abi", "ld", "dot-rhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    context = "strided_input_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "sld", "dot-lhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "rhs", slice.rhsABI, {"abi", "sld", "dot-rhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    addContractionRouteOperandBinding(plan, "lhs_stride", slice.lhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    addContractionRouteOperandBinding(plan, "rhs_stride", slice.rhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    context = "computed_masked_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(plan, "cmp_lhs", slice.lhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "cmp_rhs", slice.rhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_lhs", slice.dotLHSABI,
                                      {"abi", "ld", "mlhs", "i16", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_rhs", slice.dotRHSABI,
                                      {"abi", "ld", "mrhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    context = "computed_masked_strided_input_widening_dot_reduce_add route";
    addContractionRouteOperandBinding(plan, "cmp_lhs", slice.lhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "cmp_rhs", slice.rhsABI,
                                      {"abi", "cmp", "mask", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_lhs", slice.dotLHSABI,
                                      {"abi", "sld", "mlhs", "i16", "hdr"});
    addContractionRouteOperandBinding(plan, "dot_rhs", slice.dotRHSABI,
                                      {"abi", "sld", "mrhs", "i16", "hdr"});
    addContractionRouteOperandBinding(
        plan, "acc", slice.accumulatorABI, {"abi", "seed", "red", "i32", "hdr"});
    addContractionRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store", "i32", "hdr"});
    addContractionRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop", "hdr"});
    addContractionRouteOperandBinding(plan, "lhs_stride", slice.lhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    addContractionRouteOperandBinding(plan, "rhs_stride", slice.rhsStrideABI,
                                      {"abi", "str", "addr", "hdr"});
    break;
  default:
    return plan;
  }

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
        " requires runtime ABI parameter mirrors to match the binding plan");
  return plan;
}


} // namespace tianchenrv::plugin::rvv
