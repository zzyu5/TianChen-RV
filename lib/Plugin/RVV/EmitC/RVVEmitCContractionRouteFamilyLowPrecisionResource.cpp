//===- RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp ------------===//
//
// Mechanical transport for the formula-constructed low-precision primitive
// plan. Candidate construction and selection live before typed-body
// realization; this file neither enumerates candidates nor chooses a plan.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"

#include "RVVEmitCContractionRouteFamilyInternal.h"

#include "Weft/Plugin/RVV/RVVContractionRouteIdentity.h"

#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {

llvm::Error requireRVVSelectedBodyContractionPlanField(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("contraction route-family target-leaf/profile validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireRVVSelectedBodyContractionDerivedLeaf(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual,
    llvm::StringRef derivationInput) {
  if (!actual.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("contraction route-family target-leaf/profile validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) +
      "' requires provider-derived " + field +
      " from selected typed RVV body/config facts '" + derivationInput + "'");
}

llvm::StringRef getRVVLowPrecisionPrimitiveKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (plan.usesWideningProduct) {
    if (plan.wideningProductRelation ==
        getContractionWideningProductRelation(
            weft::rvv::getRVVSEW8Bits(), weft::rvv::getRVVLMULMF4(),
            weft::rvv::getRVVSEW16Bits(), weft::rvv::getRVVLMULMF2(),
            /*isUnsigned=*/true))
      return kRVVLowPrecisionPrimitiveUnsignedProductKind;
    return kRVVLowPrecisionPrimitiveSignedProductKind;
  }
  if (plan.usesProductReductionDequantClamp)
    return kRVVLowPrecisionPrimitiveSignedProductReductionDequantClampKind;
  if (plan.usesProductReductionDequantization)
    return kRVVLowPrecisionPrimitiveSignedProductReductionDequantKind;
  if (plan.usesProductReductionChain) {
    if (plan.wideningProductRelation ==
        getContractionWideningProductRelation(
            weft::rvv::getRVVSEW8Bits(), weft::rvv::getRVVLMULMF4(),
            weft::rvv::getRVVSEW16Bits(), weft::rvv::getRVVLMULMF2(),
            /*isUnsigned=*/true))
      return kRVVLowPrecisionPrimitiveUnsignedProductReductionKind;
    return kRVVLowPrecisionPrimitiveSignedProductReductionKind;
  }
  return {};
}

llvm::StringRef getRVVLowPrecisionPrimitiveSourceSignedness(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  // P1f C4 codebook: the u8 gather-index source is UNSIGNED even though the
  // product/result stay signed (asymmetric). Gated on the codebook head so every
  // other route keeps its relation-derived signedness verbatim.
  if (plan.usesCodebookProductReduction)
    return kRVVLowPrecisionSourceSignednessUnsigned;
  if ((plan.usesWideningProduct || plan.usesProductReductionChain) &&
      plan.wideningProductRelation ==
          getContractionWideningProductRelation(
              weft::rvv::getRVVSEW8Bits(), weft::rvv::getRVVLMULMF4(),
              weft::rvv::getRVVSEW16Bits(), weft::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true))
    return kRVVLowPrecisionSourceSignednessUnsigned;
  if (plan.usesWideningProduct || plan.usesProductReductionChain)
  return kRVVLowPrecisionSourceSignednessSigned;
  return {};
}

llvm::StringRef getRVVLowPrecisionPrimitiveSourceLoadKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (plan.usesWideningProduct || plan.usesProductReductionChain)
    return kRVVLowPrecisionPrimitiveSourceLoadKind;
  return {};
}

llvm::StringRef getRVVLowPrecisionPrimitiveSourceExtensionKind(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  // P1f C4 codebook: the u8 gather-index source is zero-extended (unsigned) even
  // though the signed widening product extends the gathered i8 values. Mirrors the
  // unsigned source signedness above; gated on the codebook head.
  if (plan.usesCodebookProductReduction)
    return kRVVLowPrecisionPrimitiveUnsignedSourceExtensionKind;
  if ((plan.usesWideningProduct || plan.usesProductReductionChain) &&
      plan.wideningProductRelation ==
          getContractionWideningProductRelation(
              weft::rvv::getRVVSEW8Bits(), weft::rvv::getRVVLMULMF4(),
              weft::rvv::getRVVSEW16Bits(), weft::rvv::getRVVLMULMF2(),
              /*isUnsigned=*/true))
    return kRVVLowPrecisionPrimitiveUnsignedSourceExtensionKind;
  if (plan.usesWideningProduct || plan.usesProductReductionChain)
    return kRVVLowPrecisionPrimitiveSignedSourceExtensionKind;
  return {};
}

static bool isRVVUnsignedLowPrecisionWideningProductPlan(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  return (plan.usesWideningProduct || plan.usesProductReductionChain) &&
         plan.wideningProductRelation ==
             getContractionWideningProductRelation(
                 weft::rvv::getRVVSEW8Bits(), weft::rvv::getRVVLMULMF4(),
                 weft::rvv::getRVVSEW16Bits(), weft::rvv::getRVVLMULMF2(),
                 /*isUnsigned=*/true);
}

llvm::StringRef getRVVWideningProductMultiplicandRoleSummary(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.usesWideningProduct && !plan.usesProductReductionChain)
    return {};
  // 1c: derive from the single ContractionRouteIdentity source (byte-identical
  // to the retired kRVVLowPrecision{Signed,Unsigned}WideningProductMultiplicand-
  // Roles). This is the producer feeding plan/description.widening-
  // ProductMultiplicandRoleSummary.
  return getContractionMultiplicandRoleSummary(
      "weft_rvv.widening_product",
      /*isSigned=*/!isRVVUnsignedLowPrecisionWideningProductPlan(plan));
}

llvm::StringRef getRVVWideningProductExtensionPolicy(
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.usesWideningProduct && !plan.usesProductReductionChain)
    return {};
  return isRVVUnsignedLowPrecisionWideningProductPlan(plan)
             ? llvm::StringRef(
                   kRVVLowPrecisionUnsignedWideningProductExtensionPolicy)
             : llvm::StringRef(
                   kRVVLowPrecisionSignedWideningProductExtensionPolicy);
}

void populateRVVLowPrecisionPrimitiveFacts(
    RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  if (!plan.usesWideningProduct && !plan.usesProductReductionChain)
    return;

  plan.lowPrecisionPrimitiveContractID = kRVVLowPrecisionPrimitiveContractID;
  plan.lowPrecisionPrimitiveKind = getRVVLowPrecisionPrimitiveKind(plan);
  plan.lowPrecisionPrimitiveSourceElementTypeName =
      plan.sourceElementTypeName;
  plan.lowPrecisionPrimitiveSourceSignedness =
      getRVVLowPrecisionPrimitiveSourceSignedness(plan);
  plan.lowPrecisionPrimitiveSourceLoadKind =
      getRVVLowPrecisionPrimitiveSourceLoadKind(plan);
  plan.lowPrecisionPrimitiveSourceExtensionKind =
      getRVVLowPrecisionPrimitiveSourceExtensionKind(plan);
  if (plan.usesWideningProduct) {
    plan.lowPrecisionPrimitiveProductElementTypeName = plan.elementTypeName;
    plan.lowPrecisionPrimitiveAccumulatorElementTypeName = "";
    plan.lowPrecisionPrimitiveResultElementTypeName = plan.elementTypeName;
    plan.wideningProductMultiplicandRoleSummary =
        getRVVWideningProductMultiplicandRoleSummary(plan);
    plan.wideningProductExtensionPolicy =
        getRVVWideningProductExtensionPolicy(plan);
    return;
  }

  plan.lowPrecisionPrimitiveProductElementTypeName =
      plan.productElementTypeName;
  plan.lowPrecisionPrimitiveAccumulatorElementTypeName =
      plan.usesProductReductionDequantization
          ? getContractionIntegerElementTypeName(plan.sew)
          : plan.elementTypeName;
  plan.lowPrecisionPrimitiveResultElementTypeName = plan.elementTypeName;
  plan.wideningProductMultiplicandRoleSummary =
      getRVVWideningProductMultiplicandRoleSummary(plan);
  plan.wideningProductExtensionPolicy =
      getRVVWideningProductExtensionPolicy(plan);
}

void populateRVVLowPrecisionPrimitiveRoutePayload(
    RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVSelectedBodyContractionRouteFamilyPlan &plan) {
  payload = RVVLowPrecisionPrimitiveRoutePayload();
  if (plan.lowPrecisionPrimitiveContractID.empty())
    return;

  payload.hasPayload = true;
  payload.isProductReductionChain = plan.usesProductReductionChain;
  payload.contractID = plan.lowPrecisionPrimitiveContractID;
  payload.kind = plan.lowPrecisionPrimitiveKind;
  payload.sourceElementTypeName =
      plan.lowPrecisionPrimitiveSourceElementTypeName;
  payload.sourceSignedness = plan.lowPrecisionPrimitiveSourceSignedness;
  payload.sourceLoadKind = plan.lowPrecisionPrimitiveSourceLoadKind;
  payload.sourceExtensionKind = plan.lowPrecisionPrimitiveSourceExtensionKind;
  payload.productElementTypeName =
      plan.lowPrecisionPrimitiveProductElementTypeName;
  payload.accumulatorElementTypeName =
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName;
  payload.resultElementTypeName =
      plan.lowPrecisionPrimitiveResultElementTypeName;

  payload.sourceSEW = plan.sourceSEW;
  payload.sourceLMUL = plan.sourceLMUL;
  payload.productSEW =
      plan.usesProductReductionChain ? plan.productSEW : plan.sew;
  payload.productLMUL =
      plan.usesProductReductionChain ? plan.productLMUL : plan.lmul;
  if (!plan.lowPrecisionPrimitiveAccumulatorElementTypeName.empty()) {
    payload.accumulatorSEW = plan.sew;
    payload.accumulatorLMUL = plan.lmul;
  }
  payload.resultSEW = plan.sew;
  payload.resultLMUL = plan.lmul;

  payload.tailPolicy = plan.tailPolicy;
  payload.maskPolicy = plan.maskPolicy;
  payload.runtimeControlPlanID = plan.runtimeControlPlan.controlPlanID;
  payload.runtimeAVLASource = plan.runtimeControlPlan.runtimeAVLASource;

  payload.wideningProductRelation = plan.wideningProductRelation;
  payload.wideningProductIntrinsic = plan.wideningProductIntrinsic;
  if (!plan.usesProductReductionChain)
    return;

  payload.productReductionChainRelation = plan.productReductionChainRelation;
  payload.reductionIntrinsic = plan.contractionComputeIntrinsic;
  payload.scalarSeedSplatIntrinsic = plan.scalarSeedSplatIntrinsic;
  payload.accumulatorLayout = plan.accumulatorLayout;
  payload.resultLayout = plan.resultLayout;
  payload.reductionStoreVL = plan.reductionStoreVL;
}

void populateRVVLowPrecisionPrimitiveDescriptionMirrorsFromPayload(
    RVVSelectedBodyEmitCRouteDescription &description) {
  const RVVLowPrecisionPrimitiveRoutePayload &payload =
      description.lowPrecisionPrimitiveRoutePayload;
  if (!payload.hasPayload) {
    description.lowPrecisionPrimitiveContractID = {};
    description.lowPrecisionPrimitiveKind = {};
    description.lowPrecisionPrimitiveSourceElementTypeName = {};
    description.lowPrecisionPrimitiveSourceSignedness = {};
    description.lowPrecisionPrimitiveSourceLoadKind = {};
    description.lowPrecisionPrimitiveSourceExtensionKind = {};
    description.lowPrecisionPrimitiveProductElementTypeName = {};
    description.lowPrecisionPrimitiveAccumulatorElementTypeName = {};
    description.lowPrecisionPrimitiveResultElementTypeName = {};
    return;
  }

  description.lowPrecisionPrimitiveContractID = payload.contractID;
  description.lowPrecisionPrimitiveKind = payload.kind;
  description.lowPrecisionPrimitiveSourceElementTypeName =
      payload.sourceElementTypeName;
  description.lowPrecisionPrimitiveSourceSignedness = payload.sourceSignedness;
  description.lowPrecisionPrimitiveSourceLoadKind = payload.sourceLoadKind;
  description.lowPrecisionPrimitiveSourceExtensionKind =
      payload.sourceExtensionKind;
  description.lowPrecisionPrimitiveProductElementTypeName =
      payload.productElementTypeName;
  description.lowPrecisionPrimitiveAccumulatorElementTypeName =
      payload.accumulatorElementTypeName;
  description.lowPrecisionPrimitiveResultElementTypeName =
      payload.resultElementTypeName;
}

llvm::Error verifyRVVLowPrecisionPrimitiveDescriptionMirrorsFromPayload(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const RVVLowPrecisionPrimitiveRoutePayload &payload =
      description.lowPrecisionPrimitiveRoutePayload;

  auto requireMirror =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route-description mirror '" + field +
        "' must mirror the provider-built primitive route payload; expected '" +
        expected + "' but saw '" + actual + "'");
  };

  if (!payload.hasPayload) {
    if (llvm::Error error =
            requireMirror("contract", description.lowPrecisionPrimitiveContractID,
                          ""))
      return error;
    if (llvm::Error error =
            requireMirror("kind", description.lowPrecisionPrimitiveKind, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "source dtype",
            description.lowPrecisionPrimitiveSourceElementTypeName, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "source signedness",
            description.lowPrecisionPrimitiveSourceSignedness, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "source load", description.lowPrecisionPrimitiveSourceLoadKind, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "source extension",
            description.lowPrecisionPrimitiveSourceExtensionKind, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "product dtype",
            description.lowPrecisionPrimitiveProductElementTypeName, ""))
      return error;
    if (llvm::Error error = requireMirror(
            "accumulator dtype",
            description.lowPrecisionPrimitiveAccumulatorElementTypeName, ""))
      return error;
    return requireMirror(
        "result dtype", description.lowPrecisionPrimitiveResultElementTypeName,
        "");
  }

  if (llvm::Error error = requireMirror(
          "contract", description.lowPrecisionPrimitiveContractID,
          payload.contractID))
    return error;
  if (llvm::Error error = requireMirror(
          "kind", description.lowPrecisionPrimitiveKind, payload.kind))
    return error;
  if (llvm::Error error = requireMirror(
          "source dtype",
          description.lowPrecisionPrimitiveSourceElementTypeName,
          payload.sourceElementTypeName))
    return error;
  if (llvm::Error error = requireMirror(
          "source signedness",
          description.lowPrecisionPrimitiveSourceSignedness,
          payload.sourceSignedness))
    return error;
  if (llvm::Error error = requireMirror(
          "source load", description.lowPrecisionPrimitiveSourceLoadKind,
          payload.sourceLoadKind))
    return error;
  if (llvm::Error error = requireMirror(
          "source extension",
          description.lowPrecisionPrimitiveSourceExtensionKind,
          payload.sourceExtensionKind))
    return error;
  if (llvm::Error error = requireMirror(
          "product dtype",
          description.lowPrecisionPrimitiveProductElementTypeName,
          payload.productElementTypeName))
    return error;
  if (llvm::Error error = requireMirror(
          "accumulator dtype",
          description.lowPrecisionPrimitiveAccumulatorElementTypeName,
          payload.accumulatorElementTypeName))
    return error;
  return requireMirror(
      "result dtype", description.lowPrecisionPrimitiveResultElementTypeName,
      payload.resultElementTypeName);
}

llvm::Error verifyRVVLowPrecisionPrimitiveRoutePayloadFromWideningReductionFacts(
    const RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVLowPrecisionWideningReductionPrimitiveFacts &primitiveFacts,
    llvm::StringRef tailPolicy, llvm::StringRef maskPolicy,
    llvm::StringRef runtimeControlPlanID,
    llvm::StringRef runtimeAVLASource, llvm::StringRef context) {
  if (!primitiveFacts.hasFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires provider-owned low-precision widening-reduction primitive "
        "facts before validating primitive route payload");
  if (!payload.hasPayload)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires provider-owned low-precision primitive route payload "
        "before validating widening-reduction primitive facts");
  if (!payload.isProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires product-reduction low-precision primitive route payload "
        "before validating widening-reduction primitive facts");

  auto requireString =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload field '" + field +
        "' must mirror provider-owned widening-reduction primitive facts; "
        "expected '" +
        expected + "' but saw '" + actual + "'");
  };
  auto requireInteger =
      [&](llvm::StringRef field, std::int64_t actual,
          std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload field '" + field +
        "' must mirror provider-owned widening-reduction primitive facts; "
        "expected " +
        llvm::Twine(expected) + " but saw " + llvm::Twine(actual));
  };

#define WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(FIELD, ACTUAL, EXPECTED)   \
  if (llvm::Error error = requireString((FIELD), (ACTUAL), (EXPECTED)))        \
    return error
#define WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(FIELD, ACTUAL, EXPECTED)  \
  if (llvm::Error error = requireInteger((FIELD), (ACTUAL), (EXPECTED)))       \
    return error

  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "contract", payload.contractID,
      primitiveFacts.lowPrecisionPrimitiveContractID);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "kind", payload.kind, primitiveFacts.lowPrecisionPrimitiveKind);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source dtype", payload.sourceElementTypeName,
      primitiveFacts.sourceElementTypeName);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source signedness", payload.sourceSignedness,
      primitiveFacts.sourceSignedness);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source load", payload.sourceLoadKind, primitiveFacts.sourceLoadKind);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source extension", payload.sourceExtensionKind,
      primitiveFacts.sourceExtensionKind);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "product dtype", payload.productElementTypeName,
      primitiveFacts.productElementTypeName);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "accumulator dtype", payload.accumulatorElementTypeName,
      primitiveFacts.accumulatorElementTypeName);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "result dtype", payload.resultElementTypeName,
      primitiveFacts.finalResultElementTypeName);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(
      "source SEW", payload.sourceSEW, primitiveFacts.sourceSEW);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "source LMUL", payload.sourceLMUL, primitiveFacts.sourceLMUL);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(
      "product SEW", payload.productSEW, primitiveFacts.productSEW);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "product LMUL", payload.productLMUL, primitiveFacts.productLMUL);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(
      "accumulator SEW", payload.accumulatorSEW,
      primitiveFacts.accumulatorSEW);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "accumulator LMUL", payload.accumulatorLMUL,
      primitiveFacts.accumulatorLMUL);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER(
      "result SEW", payload.resultSEW, primitiveFacts.reductionResultSEW);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "result LMUL", payload.resultLMUL, primitiveFacts.reductionResultLMUL);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "tail policy", payload.tailPolicy, tailPolicy);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "mask policy", payload.maskPolicy, maskPolicy);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "runtime control plan", payload.runtimeControlPlanID,
      runtimeControlPlanID);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "runtime AVL source", payload.runtimeAVLASource, runtimeAVLASource);
  // P1e C3 (offset-binary N=3): the provider-owned primitive facts derive the
  // config-keyed signed widening relation (signed-i8mf4xi8mf4-to-i16mf2), but
  // the offset-binary route's payload carries its OWN op-owned canonical product
  // relation (offset-binary-i4mf4-x-i8mf4x2-to-i16mf2, flowed from
  // plan.wideningProductRelation via payload assignment above). For that route
  // the mirror validates against the payload's own op-owned relation -- the same
  // gated recognition the plan-level relation mirror uses
  // (RVVEmitCContractionRouteFamilyValidation.cpp:704-715). Every existing
  // route's payload relation equals the config-derived facts relation, so the
  // gate never fires for them and this stays byte-exact.
  constexpr llvm::StringLiteral kRVVOffsetBinaryProductRelation(
      "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2");
  // The C4 codebook route likewise carries its OWN op-owned canonical product
  // relation (codebook-gather-i8-x-i8x2-to-i16); validate the payload against its
  // own relation for that route too. Byte-exact for every other route.
  constexpr llvm::StringLiteral kRVVCodebookProductRelation(
      "codebook-gather-i8-x-i8x2-to-i16");
  const llvm::StringRef expectedWideningProductRelation =
      (payload.wideningProductRelation == kRVVOffsetBinaryProductRelation ||
       payload.wideningProductRelation == kRVVCodebookProductRelation)
          ? llvm::StringRef(payload.wideningProductRelation)
          : llvm::StringRef(primitiveFacts.wideningProductRelation);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "widening product relation", payload.wideningProductRelation,
      expectedWideningProductRelation);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "product-reduction chain relation",
      payload.productReductionChainRelation,
      primitiveFacts.productReductionChainRelation);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "widening product intrinsic", payload.wideningProductIntrinsic,
      primitiveFacts.wideningProductIntrinsic);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "reduction intrinsic", payload.reductionIntrinsic,
      primitiveFacts.reductionIntrinsic);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "scalar seed splat intrinsic", payload.scalarSeedSplatIntrinsic,
      primitiveFacts.scalarSeedSplatIntrinsic);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "accumulator layout", payload.accumulatorLayout,
      primitiveFacts.accumulatorLayout);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "result layout", payload.resultLayout, primitiveFacts.resultLayout);
  WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING(
      "reduction store VL", payload.reductionStoreVL,
      primitiveFacts.reductionStoreVL);

#undef WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_STRING
#undef WEFT_REQUIRE_PRIMITIVE_FACT_PAYLOAD_INTEGER

  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionPrimitiveRoutePayloadFromPlan(
    const RVVLowPrecisionPrimitiveRoutePayload &payload,
    const RVVSelectedBodyContractionRouteFamilyPlan &plan,
    llvm::StringRef context) {
  const bool expectsPayload = !plan.lowPrecisionPrimitiveContractID.empty();
  if (!expectsPayload) {
    if (!payload.hasPayload)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " rejects low-precision primitive route payload on a contraction "
        "route without provider-owned low-precision primitive facts");
  }
  if (!payload.hasPayload)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires provider-owned low-precision primitive route payload "
        "before route construction");

  auto requireString =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload field '" + field +
        "' must mirror the validated provider route-family plan; expected '" +
        expected + "' but saw '" + actual + "'");
  };
  auto requireInteger =
      [&](llvm::StringRef field, std::int64_t actual,
          std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload field '" + field +
        "' must mirror the validated provider route-family plan; expected " +
        llvm::Twine(expected) + " but saw " + llvm::Twine(actual));
  };

  if (payload.isProductReductionChain != plan.usesProductReductionChain)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " low-precision primitive route payload product-reduction boundary "
        "must mirror the validated provider route-family plan");
  if (plan.usesProductReductionChain) {
    // P1f C4: the codebook route's asymmetric flag keeps the payload facts' source
    // u8 while the product/result stay signed (byte-identical for symmetric routes,
    // where the flag is false).
    std::optional<RVVLowPrecisionWideningReductionPrimitiveFacts>
        primitiveFacts = getRVVLowPrecisionWideningReductionPrimitiveFacts(
            plan.operation, isRVVUnsignedLowPrecisionWideningProductPlan(plan),
            plan.sourceLMUL, plan.productLMUL,
            plan.usesCodebookProductReduction);
    if (!primitiveFacts)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires provider-owned widening-reduction primitive facts before "
          "validating low-precision primitive route payload");
    if (llvm::Error error =
            verifyRVVLowPrecisionPrimitiveRoutePayloadFromWideningReductionFacts(
                payload, *primitiveFacts, plan.tailPolicy, plan.maskPolicy,
                plan.runtimeControlPlan.controlPlanID,
                plan.runtimeControlPlan.runtimeAVLASource, context))
      return error;
  }

#define WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(FIELD, ACTUAL, EXPECTED)         \
  if (llvm::Error error = requireString((FIELD), (ACTUAL), (EXPECTED)))        \
    return error
#define WEFT_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER(FIELD, ACTUAL, EXPECTED)        \
  if (llvm::Error error = requireInteger((FIELD), (ACTUAL), (EXPECTED)))       \
    return error

  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "contract", payload.contractID, plan.lowPrecisionPrimitiveContractID);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "kind", payload.kind, plan.lowPrecisionPrimitiveKind);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "source dtype", payload.sourceElementTypeName,
      plan.lowPrecisionPrimitiveSourceElementTypeName);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "source signedness", payload.sourceSignedness,
      plan.lowPrecisionPrimitiveSourceSignedness);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "source load", payload.sourceLoadKind,
      plan.lowPrecisionPrimitiveSourceLoadKind);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "source extension", payload.sourceExtensionKind,
      plan.lowPrecisionPrimitiveSourceExtensionKind);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "product dtype", payload.productElementTypeName,
      plan.lowPrecisionPrimitiveProductElementTypeName);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "accumulator dtype", payload.accumulatorElementTypeName,
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "result dtype", payload.resultElementTypeName,
      plan.lowPrecisionPrimitiveResultElementTypeName);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER("source SEW", payload.sourceSEW,
                                         plan.sourceSEW);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING("source LMUL", payload.sourceLMUL,
                                        plan.sourceLMUL);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER(
      "product SEW", payload.productSEW,
      plan.usesProductReductionChain ? plan.productSEW : plan.sew);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "product LMUL", payload.productLMUL,
      plan.usesProductReductionChain ? plan.productLMUL : plan.lmul);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER(
      "accumulator SEW", payload.accumulatorSEW,
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName.empty()
          ? 0
          : plan.sew);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "accumulator LMUL", payload.accumulatorLMUL,
      plan.lowPrecisionPrimitiveAccumulatorElementTypeName.empty()
          ? llvm::StringRef()
          : plan.lmul);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER("result SEW", payload.resultSEW,
                                         plan.sew);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING("result LMUL", payload.resultLMUL,
                                        plan.lmul);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING("tail policy", payload.tailPolicy,
                                        plan.tailPolicy);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING("mask policy", payload.maskPolicy,
                                        plan.maskPolicy);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "runtime control plan", payload.runtimeControlPlanID,
      plan.runtimeControlPlan.controlPlanID);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "runtime AVL source", payload.runtimeAVLASource,
      plan.runtimeControlPlan.runtimeAVLASource);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "widening product relation", payload.wideningProductRelation,
      plan.wideningProductRelation);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "widening product intrinsic", payload.wideningProductIntrinsic,
      plan.wideningProductIntrinsic);
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "product-reduction chain relation",
      payload.productReductionChainRelation,
      plan.usesProductReductionChain ? plan.productReductionChainRelation
                                     : llvm::StringRef());
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "reduction intrinsic", payload.reductionIntrinsic,
      plan.usesProductReductionChain ? plan.contractionComputeIntrinsic
                                     : llvm::StringRef());
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "scalar seed splat intrinsic", payload.scalarSeedSplatIntrinsic,
      plan.usesProductReductionChain ? plan.scalarSeedSplatIntrinsic
                                     : llvm::StringRef());
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "accumulator layout", payload.accumulatorLayout,
      plan.usesProductReductionChain ? plan.accumulatorLayout
                                     : llvm::StringRef());
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "result layout", payload.resultLayout,
      plan.usesProductReductionChain ? plan.resultLayout
                                     : llvm::StringRef());
  WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING(
      "reduction store VL", payload.reductionStoreVL,
      plan.usesProductReductionChain ? plan.reductionStoreVL
                                     : llvm::StringRef());

#undef WEFT_REQUIRE_PRIMITIVE_PAYLOAD_STRING
#undef WEFT_REQUIRE_PRIMITIVE_PAYLOAD_INTEGER

  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
