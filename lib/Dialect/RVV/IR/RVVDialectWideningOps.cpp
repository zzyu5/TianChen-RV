//===- RVVDialectWideningOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's widening / contraction ops: WideningMAcc, WideningDotReduce, WideningProduct.
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: widening / contraction ops: WideningMAcc, WideningDotReduce, WideningProduct,
// PackedI4NibbleUnpackProduct, MaskedWideningDotReduce, WideningConvert,
// Dequantize
//
//===----------------------------------------------------------------------===//

#include "RVVDialectInternal.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace tianchenrv::tcrv::rvv;

mlir::LogicalResult WideningMAccOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_macc keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningMAccAttr(attrName))
      return emitOpError()
             << "only accepts generic widening multiply-accumulate "
                "attributes 'kind', 'accumulator_layout', 'result_layout', "
                "and 'macc_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningMAccKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"signed_widening_macc_add\" for "
              "the bounded Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-widening-multiply-accumulate-result-to-output-buffer\" "
              "for the bounded Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccRelation(getMaccRelation()))
    return emitOpError()
           << "currently supports only macc_relation "
              "\"signed-i16mf2xi16mf2-plus-i32m1-to-i32m1\" for the bounded "
              "Stage 2 widening multiply-accumulate route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i16 generic RVV vector operands, one i32 "
              "accumulator vector operand, one !tcrv_rvv.vl operand, and one "
              "i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
              "widening multiply-accumulate route";
  if (!isGenericRVVVectorI32M1(getAccumulator().getType()) ||
      !isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires accumulator and result vectors to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed widening "
              "multiply-accumulate route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "accumulator/result SEW/LMUL metadata for widening macc";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl accumulator/result config "
              "to be SEW32 LMUL m1 for the bounded signed widening macc "
              "route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening macc";

  return mlir::success();
}

mlir::LogicalResult WideningDotReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_dot_reduce keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningDotReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic widening dot-product reduction "
                "attributes 'kind', 'accumulator_layout', 'result_layout', "
                "and 'dot_product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningDotReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_widening_dot_reduce_add\" for the bounded Stage 2 "
              "widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 widening dot-product reduction route";
  if (!isSupportedGenericWideningDotProductRelation(
          getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded Stage 2 widening dot-product reduction route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i16 generic RVV vector operands, one i32 "
              "accumulator seed runtime ABI operand, one !tcrv_rvv.vl "
              "operand, and one i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
              "widening dot-product reduction route";
  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
              "widening dot-product reduction route";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  if (!seedBinding || seedBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "for the bounded signed widening dot-product reduction route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "result SEW/LMUL metadata for widening dot-product reduction";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW32 LMUL m1 for the bounded signed widening dot-product "
              "reduction route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening dot-product reduction";

  return mlir::success();
}

mlir::LogicalResult WideningProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_product keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningProductKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"signed_widening_product\" or "
              "\"unsigned_widening_product\" for the bounded Stage 2 "
              "low-precision widening-product typed surface";
  if (!isSupportedGenericWideningProductRelation(getProductRelation()))
    return emitOpError()
           << "currently supports only product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\", "
              "\"unsigned-u8mf4xu8mf4-to-u16mf2\", or "
              "\"signed-i8m2xi8m2-to-i16m4\" (the deferred-wide max-legal-LMUL "
              "rung) for the bounded Stage 2 low-precision widening-product "
              "typed surface";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i8 generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one i16 generic RVV vector result";
  const bool isUnsignedProduct = getKind() == "unsigned_widening_product";
  // The deferred-wide max-legal-LMUL rung (N3 schedule, the measured ssh-rvv
  // winner): i8m2 x i8m2 -> i16m4 feeding tcrv_rvv.widening_accumulate. This is
  // a PARALLEL signed verifier branch -- the narrow i8mf4 branch is unchanged.
  const bool isWideDeferredProduct =
      getKind() == "signed_widening_product" &&
      isSupportedGenericWideningProductWideDeferredRelation(getProductRelation());
  if (isWideDeferredProduct) {
    if (!isGenericRVVVectorSignedI8M2(getLhs().getType()) ||
        !isGenericRVVVectorSignedI8M2(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<i8, \"m2\"> for the deferred-wide "
                "max-legal-LMUL widening-product rung";
    if (!isGenericRVVVectorSignedI16M4(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<i16, \"m4\"> for the deferred-wide "
                "max-legal-LMUL widening-product rung";
    if (!llvm::isa<VLType>(getVl().getType()))
      return emitOpError() << "requires runtime VL operand to have "
                              "!tcrv_rvv.vl type";
    auto withVL = verifyNestedDataflowOp(op);
    if (mlir::failed(withVL))
      return mlir::failure();
    if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
      return mlir::failure();
    auto expectedSEW =
        (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
    auto expectedLMUL =
        (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
    if (!expectedSEW || !expectedLMUL)
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit "
                "SEW/LMUL metadata for the deferred-wide widening product";
    if (expectedSEW.getInt() != getRVVSEW8Bits() ||
        expectedLMUL.getValue() != getRVVLMULM2())
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl config to be SEW8 LMUL m2 "
                "for the deferred-wide max-legal-LMUL widening-product rung";
    if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
                "metadata for widening product";
    return mlir::success();
  }
  // The 2nd-family (i16 dot-reduce) deferred-wide rung: i16m4 x i16m4 -> i32m8,
  // a SINGLE widening step where the widened product already equals the i32
  // accumulator width. Feeds a NON-widening tcrv_rvv.deferred_accumulate
  // (vadd.vv). PARALLEL signed branch -- the byte i8m2 and narrow i8mf4 branches
  // are unchanged.
  const bool isWideDotReduceProduct =
      getKind() == "signed_widening_product" &&
      isSupportedGenericWideningProductWideDotReduceRelation(
          getProductRelation());
  if (isWideDotReduceProduct) {
    if (!isGenericRVVVectorSignedI16M4(getLhs().getType()) ||
        !isGenericRVVVectorSignedI16M4(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<i16, \"m4\"> for the deferred-wide "
                "dot-reduce max-legal-LMUL widening-product rung";
    if (!isGenericRVVVectorI32M8(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<i32, \"m8\"> for the deferred-wide "
                "dot-reduce max-legal-LMUL widening-product rung";
    if (!llvm::isa<VLType>(getVl().getType()))
      return emitOpError() << "requires runtime VL operand to have "
                              "!tcrv_rvv.vl type";
    auto withVL = verifyNestedDataflowOp(op);
    if (mlir::failed(withVL))
      return mlir::failure();
    if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
      return mlir::failure();
    auto expectedSEW =
        (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
    auto expectedLMUL =
        (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
    if (!expectedSEW || !expectedLMUL)
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit "
                "SEW/LMUL metadata for the deferred-wide dot-reduce product";
    if (expectedSEW.getInt() != getRVVSEW16Bits() ||
        expectedLMUL.getValue() != getRVVLMULM4())
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl config to be SEW16 LMUL m4 "
                "for the deferred-wide dot-reduce max-legal-LMUL "
                "widening-product rung";
    if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
                "metadata for widening product";
    return mlir::success();
  }
  if (isUnsignedProduct) {
    if (getProductRelation() != "unsigned-u8mf4xu8mf4-to-u16mf2")
      return emitOpError()
             << "requires product_relation "
                "\"unsigned-u8mf4xu8mf4-to-u16mf2\" when kind is "
                "\"unsigned_widening_product\"";
    if (!isGenericRVVVectorUnsignedI8MF4(getLhs().getType()) ||
        !isGenericRVVVectorUnsignedI8MF4(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<ui8, \"mf4\"> for the bounded unsigned "
                "low-precision widening-product typed surface";
    if (!isGenericRVVVectorUnsignedI16MF2(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<ui16, \"mf2\"> for the bounded unsigned "
                "low-precision widening-product typed surface";
  } else {
    if (getProductRelation() != "signed-i8mf4xi8mf4-to-i16mf2")
      return emitOpError()
             << "requires product_relation "
                "\"signed-i8mf4xi8mf4-to-i16mf2\" when kind is "
                "\"signed_widening_product\"";
    if (!isGenericRVVVectorSignedI8MF4(getLhs().getType()) ||
        !isGenericRVVVectorSignedI8MF4(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<i8, \"mf4\"> for the bounded signed "
                "low-precision widening-product route";
    if (!isGenericRVVVectorSignedI16MF2(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
                "low-precision widening-product route";
  }
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit result "
              "SEW/LMUL metadata for widening product";
  const bool isStandaloneProductConfig =
      expectedSEW.getInt() == getRVVSEW16Bits() &&
      expectedLMUL.getValue() == getRVVLMULMF2();
  const bool isProductReductionChainConfig =
      isBoundedWideningProductReductionChainProduct(*this, *withVL);
  if (!isStandaloneProductConfig && !isProductReductionChainConfig)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW16 LMUL mf2 for the bounded signed low-precision "
              "widening-product route, or SEW32 LMUL m1 when the i16 product "
              "feeds the bounded i16-to-i32 standalone widening reduction "
              "chain";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening product";

  return mlir::success();
}

mlir::LogicalResult WideningAccumulateOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_accumulate keeps source/result SEW/LMUL/"
                "policy on typed vector values and setvl/with_vl, runtime "
                "n/AVL/VL in the surrounding control-plane IR, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedWideningAccumulateAttr(attrName))
      return emitOpError()
             << "only accepts generic deferred widening accumulate attributes "
                "'kind' and 'accumulate_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningAccumulateKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_widening_accumulate_add\" for the bounded N3 "
              "deferred-wide widening accumulate route";
  if (!isSupportedGenericWideningAccumulateRelation(getAccumulateRelation()))
    return emitOpError()
           << "currently supports only accumulate_relation "
              "\"signed-i16m4-into-i32m8-deferred-add\" for the bounded N3 "
              "deferred-wide widening accumulate route";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i16 LMUL m4 widening-product operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m8 vector result";
  if (!isGenericRVVVectorSignedI16M4(getProduct().getType()))
    return emitOpError()
           << "requires product operand to have type "
              "!tcrv_rvv.vector<i16, \"m4\"> for the deferred-wide widening "
              "accumulate route";
  if (!isGenericRVVVectorI32M8(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m8\"> for the deferred-wide widening "
              "accumulate route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  // The deferred-wide accumulate consumes a bounded i8m2 x i8m2 -> i16m4 signed
  // widening product (the structural marker that the body is the deferred-wide
  // algorithm). This keeps emission body-determined (I5): the conversion does
  // not infer the deferred mode from metadata, it follows op identity.
  auto product = getProduct().getDefiningOp<WideningProductOp>();
  if (!product)
    return emitOpError()
           << "requires product operand to be produced by a bounded "
              "tcrv_rvv.widening_product inside the selected RVV typed body";
  if (product.getKind() != "signed_widening_product" ||
      !isSupportedGenericWideningProductWideDeferredRelation(
          product.getProductRelation()))
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to use "
              "kind \"signed_widening_product\" and product_relation "
              "\"signed-i8m2xi8m2-to-i16m4\" for the deferred-wide widening "
              "accumulate route";
  if (product.getVl() != getVl())
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to consume "
              "the same !tcrv_rvv.vl token as tcrv_rvv.widening_accumulate";
  if (product->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to be in "
              "the same tcrv_rvv.with_vl body as tcrv_rvv.widening_accumulate";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW/LMUL "
              "metadata for the deferred-wide widening accumulate";
  if (expectedSEW.getInt() != getRVVSEW8Bits() ||
      expectedLMUL.getValue() != getRVVLMULM2())
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl config to be SEW8 LMUL m2 "
              "for the deferred-wide max-legal-LMUL widening accumulate route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the deferred-wide widening accumulate";

  return mlir::success();
}

mlir::LogicalResult DeferredAccumulateOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.deferred_accumulate keeps source/result SEW/LMUL/"
                "policy on typed vector values and setvl/with_vl, runtime "
                "n/AVL/VL in the surrounding control-plane IR, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedDeferredAccumulateAttr(attrName))
      return emitOpError()
             << "only accepts generic deferred accumulate attributes "
                "'kind' and 'accumulate_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericDeferredAccumulateKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_deferred_accumulate_add\" for the bounded N3 "
              "deferred-wide dot-reduce accumulate route";
  if (!isSupportedGenericDeferredAccumulateRelation(getAccumulateRelation()))
    return emitOpError()
           << "currently supports only accumulate_relation "
              "\"signed-i32m8-into-i32m8-deferred-add\" for the bounded N3 "
              "deferred-wide dot-reduce accumulate route";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i32 LMUL m8 widening-product operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m8 vector result";
  if (!isGenericRVVVectorI32M8(getProduct().getType()))
    return emitOpError()
           << "requires product operand to have type "
              "!tcrv_rvv.vector<i32, \"m8\"> for the deferred-wide dot-reduce "
              "accumulate route";
  if (!isGenericRVVVectorI32M8(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m8\"> for the deferred-wide dot-reduce "
              "accumulate route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  // The deferred-wide dot-reduce accumulate consumes a bounded i16m4 x i16m4 ->
  // i32m8 signed widening product (the structural marker that the body is the
  // deferred-wide dot-reduce algorithm). This keeps emission body-determined
  // (I5): the conversion follows op identity, not metadata.
  auto product = getProduct().getDefiningOp<WideningProductOp>();
  if (!product)
    return emitOpError()
           << "requires product operand to be produced by a bounded "
              "tcrv_rvv.widening_product inside the selected RVV typed body";
  if (product.getKind() != "signed_widening_product" ||
      !isSupportedGenericWideningProductWideDotReduceRelation(
          product.getProductRelation()))
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to use "
              "kind \"signed_widening_product\" and product_relation "
              "\"signed-i16m4xi16m4-to-i32m8\" for the deferred-wide dot-reduce "
              "accumulate route";
  if (product.getVl() != getVl())
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to consume "
              "the same !tcrv_rvv.vl token as tcrv_rvv.deferred_accumulate";
  if (product->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to be in "
              "the same tcrv_rvv.with_vl body as tcrv_rvv.deferred_accumulate";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW/LMUL "
              "metadata for the deferred-wide dot-reduce accumulate";
  if (expectedSEW.getInt() != getRVVSEW16Bits() ||
      expectedLMUL.getValue() != getRVVLMULM4())
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl config to be SEW16 LMUL m4 "
              "for the deferred-wide dot-reduce max-legal-LMUL accumulate route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the deferred-wide dot-reduce accumulate";

  return mlir::success();
}

mlir::LogicalResult PackedI4NibbleUnpackProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.packed_i4_nibble_unpack_product keeps source/"
                "result SEW/LMUL/policy on typed vector values and "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "signed_packed_i4_nibble_unpack_product")
    return emitOpError()
           << "currently supports only kind "
              "\"signed_packed_i4_nibble_unpack_product\" for the bounded "
              "Stage 3 packed-i4 nibble-unpack widening-product typed surface";
  if (getProductRelation() != "signed-i8mf4xi8mf4-to-i16mf2")
    return emitOpError()
           << "requires product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\" for the bounded packed-i4 "
              "nibble-unpack widening-product route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires two i8 LMUL mf4 packed source operands, one "
              "!tcrv_rvv.vl operand, and one i16 LMUL mf2 result";
  if (!isGenericRVVVectorSignedI8MF4(getLhs().getType()) ||
      !isGenericRVVVectorSignedI8MF4(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i8, \"mf4\"> for the bounded packed-i4 "
              "nibble-unpack widening-product route";
  if (!isGenericRVVVectorSignedI16MF2(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded packed-i4 "
              "nibble-unpack widening-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for packed-i4 nibble-unpack widening product";

  return mlir::success();
}

mlir::LogicalResult MaskedWideningDotReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_widening_dot_reduce keeps mask "
                "provenance, source/result SEW/LMUL/policy on typed values "
                "and setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedMaskedWideningDotReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic masked widening dot-product reduction "
                "attributes 'kind', 'mask_role', 'mask_source', "
                "'mask_memory_form', 'accumulator_layout', 'result_layout', "
                "and 'dot_product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedWideningDotReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_masked_widening_dot_reduce_add\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded Stage 2 masked "
              "widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 masked widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 masked widening dot-product reduction route";
  if (!isSupportedGenericWideningDotProductRelation(
          getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded Stage 2 masked widening dot-product reduction "
              "route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires compare-produced mask, lhs and rhs i16 generic RVV "
              "vector operands, one i32 accumulator seed runtime ABI operand, "
              "one !tcrv_rvv.vl operand, and one i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed masked "
              "widening dot-product reduction route";
  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed masked "
              "widening dot-product reduction route";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  if (!seedBinding || seedBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "for the bounded signed masked widening dot-product reduction "
              "route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getKind() != "slt")
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to use kind "
              "\"slt\" for the bounded computed-mask widening dot-product "
              "reduction route";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_widening_dot_reduce";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as "
              "tcrv_rvv.masked_widening_dot_reduce";
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "result SEW/LMUL metadata for masked widening dot-product "
              "reduction";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW32 LMUL m1 for the bounded signed masked widening "
              "dot-product reduction route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for masked widening dot-product reduction";

  return mlir::success();
}

mlir::LogicalResult WideningConvertOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_convert keeps source/destination "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningConvertAttr(attrName))
      return emitOpError()
             << "only accepts generic widening conversion attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningConvertKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"widen_i32_to_i64\" or "
              "\"sign_extend_widen_vf2\" for the bounded Stage 2 widening "
              "conversion routes";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one source generic RVV vector operand, one "
              "!tcrv_rvv.vl operand, and one destination generic RVV vector "
              "result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "destination SEW/LMUL metadata for widening conversion";

  if (getKind() == "widen_i32_to_i64") {
    if (!isGenericRVVVectorI32M1(getSource().getType()))
      return emitOpError()
             << "requires source vector type to be "
                "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
                "i32-to-i64 widening conversion route";
    if (!isGenericRVVVectorI64M2(getResult().getType()))
      return emitOpError()
             << "requires result vector type to be "
                "!tcrv_rvv.vector<i64, \"m2\"> for the bounded signed "
                "i32-to-i64 widening conversion route";
    if (!isRVVSelectedBodyI64M2Config(expectedSEW.getInt(),
                                      expectedLMUL.getValue()))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl destination config to "
                "be SEW64 LMUL m2 for the bounded signed i32-to-i64 "
                "widening conversion route";
  } else {
    if (!isGenericRVVVectorI16MF2(getSource().getType()))
      return emitOpError()
             << "requires source vector type to be "
                "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
                "i16-to-i32 widening conversion route";
    if (!isGenericRVVVectorI32M1(getResult().getType()))
      return emitOpError()
             << "requires result vector type to be "
                "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
                "i16-to-i32 widening conversion route";
    if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                   expectedLMUL.getValue()))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl destination config to "
                "be SEW32 LMUL m1 for the bounded signed i16-to-i32 "
                "widening conversion route";
  }
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening conversion";

  return mlir::success();
}

mlir::LogicalResult DequantizeOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.dequantize keeps source/result dtype, "
                "SEW/LMUL/policy, and runtime scale authority on typed vector "
                "values, runtime ABI SSA, and setvl/with_vl, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedDequantizeAttr(attrName))
      return emitOpError()
             << "only accepts generic dequantization attributes 'kind' and "
                "'dequant_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericDequantizeKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"i32_to_f32_scaled\" for the "
              "bounded Stage 2 i32-to-f32 dequantization route";
  if (!isSupportedGenericDequantizeRelation(getDequantRelation()))
    return emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded Stage 2 "
              "i32-to-f32 dequantization route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i32 source generic RVV vector operand, one "
              "runtime f32 scale ABI operand, one !tcrv_rvv.vl operand, and "
              "one f32 destination generic RVV vector result";
  if (!isGenericRVVVectorI32M1(getSource().getType()))
    return emitOpError()
           << "requires source vector type to be "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded i32-to-f32 "
              "dequantization route";
  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector type to be "
              "!tcrv_rvv.vector<f32, \"m1\"> for the bounded i32-to-f32 "
              "dequantization route";
  if (!llvm::isa<RuntimeABIValueType>(getScale().getType()))
    return emitOpError()
           << "requires scale operand to have !tcrv_rvv.runtime_abi_value "
              "type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::
               DequantScaleValue})))
    return mlir::failure();
  RuntimeABIValueOp scaleBinding = getScale().getDefiningOp<RuntimeABIValueOp>();
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return emitOpError()
           << "requires runtime scale operand C type 'float' for the bounded "
              "i32-to-f32 dequantization route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  // Deferred-wide dequant (N3 max-legal-LMUL schedule): the dequant sources the
  // trailing tcrv_rvv.standalone_reduce (i32m1) whose input is the i32m8
  // tcrv_rvv.widening_accumulate. The source/result dtype (i32m1 -> f32m1) is
  // the SAME as the narrow path; only the enclosing with_vl is the strip config
  // (SEW8 LMUL m2), so the SEW32-pinned result-vector check does not apply. This
  // is a PARALLEL branch keyed on the deferred-accumulate structural marker.
  if (auto deferredReduce =
          getSource().getDefiningOp<StandaloneReduceOp>()) {
    if (deferredReduce.getInput().getDefiningOp<WideningAccumulateOp>()) {
      if (deferredReduce.getVl() != getVl())
        return emitOpError()
               << "requires the deferred-wide trailing "
                  "tcrv_rvv.standalone_reduce to consume the same "
                  "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
      if (deferredReduce->getParentOp() != op->getParentOp())
        return emitOpError()
               << "requires the deferred-wide trailing "
                  "tcrv_rvv.standalone_reduce to be in the same "
                  "tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
      // Source i32m1 / result f32m1 already checked above; the deferred-wide
      // path keeps those, and skips the SEW32/m1 with_vl pin (strip is SEW8/m2).
      return mlir::success();
    }
  }

  auto sourceLoad = getSource().getDefiningOp<LoadOp>();
  auto sourceReduction = getSource().getDefiningOp<StandaloneReduceOp>();
  auto sourceHandoff = getSource().getDefiningOp<GearboxCrossRegionHandoffOp>();
  if (!sourceLoad && !sourceReduction && !sourceHandoff)
    return emitOpError()
           << "requires source vector to be produced by tcrv_rvv.load or by "
              "a bounded tcrv_rvv.widening_product -> "
              "tcrv_rvv.standalone_reduce chain, optionally through "
              "tcrv_rvv.gearbox_cross_region_handoff, inside the selected "
              "RVV typed body";
  if (sourceLoad) {
    if (sourceLoad.getVl() != getVl())
      return emitOpError()
             << "requires source-producing tcrv_rvv.load to consume the same "
                "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
    if (sourceLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires source-producing tcrv_rvv.load to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
  }
  if (sourceReduction) {
    // The reduce input is either a plain widening product or the signed packed-i4
    // nibble-unpack widening product (the Stage-3 typed packed-i4 surface). Both
    // are bounded i8mf4 -> i16mf2 signed product chains feeding the i32 reduce.
    mlir::Operation *productOp = sourceReduction.getInput().getDefiningOp();
    auto product = llvm::dyn_cast_or_null<WideningProductOp>(productOp);
    auto packed =
        llvm::dyn_cast_or_null<PackedI4NibbleUnpackProductOp>(productOp);
    if (!product && !packed)
      return emitOpError()
             << "requires source-producing tcrv_rvv.standalone_reduce to "
                "consume a bounded tcrv_rvv.widening_product or "
                "tcrv_rvv.packed_i4_nibble_unpack_product result for the "
                "low-precision product-reduction dequantization route";
    const bool productKindOK =
        product ? product.getKind() == "signed_widening_product"
                : packed.getKind() == "signed_packed_i4_nibble_unpack_product";
    if (sourceReduction.getKind() != "signed_widening_reduce_add" ||
        !productKindOK)
      return emitOpError()
             << "requires source-producing product-reduction chain to use "
                "signed_widening_product or "
                "signed_packed_i4_nibble_unpack_product followed by "
                "signed_widening_reduce_add";
    mlir::Value productVL = product ? product.getVl() : packed.getVl();
    if (sourceReduction.getVl() != getVl() || productVL != getVl())
      return emitOpError()
             << "requires source-producing product and "
                "tcrv_rvv.standalone_reduce to consume the same "
                "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
    if (sourceReduction->getParentOp() != op->getParentOp() ||
        productOp->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires source-producing product-reduction chain to be in "
                "the same tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
  }
  if (sourceHandoff) {
    auto reduction = sourceHandoff.getInput().getDefiningOp<StandaloneReduceOp>();
    if (!reduction)
      return emitOpError()
             << "requires source-producing Gearbox handoff to consume a "
                "tcrv_rvv.standalone_reduce result";
    // The reduce input is either a plain widening product or the signed
    // packed-i4 nibble-unpack widening product (the Stage-3 typed packed-i4
    // surface); both feed the i32 reduce -> handoff -> dequant chain.
    mlir::Operation *productOp = reduction.getInput().getDefiningOp();
    if (!llvm::isa_and_nonnull<WideningProductOp,
                               PackedI4NibbleUnpackProductOp>(productOp))
      return emitOpError()
             << "requires source-producing Gearbox handoff reduction to "
                "consume a bounded tcrv_rvv.widening_product or "
                "tcrv_rvv.packed_i4_nibble_unpack_product result";
    mlir::Value productVL =
        llvm::isa<WideningProductOp>(productOp)
            ? llvm::cast<WideningProductOp>(productOp).getVl()
            : llvm::cast<PackedI4NibbleUnpackProductOp>(productOp).getVl();
    if (sourceHandoff.getVl() != getVl() || reduction.getVl() != getVl() ||
        productVL != getVl())
      return emitOpError()
             << "requires source-producing Gearbox handoff, product, and "
                "standalone reduction to consume the same !tcrv_rvv.vl token "
                "as tcrv_rvv.dequantize";
    WithVLOp producerWithVL =
        llvm::dyn_cast_or_null<WithVLOp>(sourceHandoff->getParentOp());
    if (!producerWithVL ||
        reduction->getParentOp() != producerWithVL.getOperation() ||
        productOp->getParentOp() != producerWithVL.getOperation() ||
        (!isAncestorWithVL(producerWithVL, op) &&
         producerWithVL.getOperation() != op->getParentOp()))
      return emitOpError()
             << "requires source-producing Gearbox handoff chain to be in "
                "the same producer tcrv_rvv.with_vl body as the handoff, and "
                "that producer scope must enclose or match the dequantize "
                "consumer scope";
  }

  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getSource(),
                                                    "source")))
    return mlir::failure();
  return verifyDequantizeResultVectorForWithVL(op, getResult(), "result");
}
