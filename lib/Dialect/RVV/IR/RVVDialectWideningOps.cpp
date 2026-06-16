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

mlir::LogicalResult PackedI4OffsetBinaryXI8ProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.packed_i4_offset_binary_x_i8_product keeps source/"
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

  if (getKind() != "signed_packed_i4_offset_binary_x_i8_product")
    return emitOpError()
           << "currently supports only kind "
              "\"signed_packed_i4_offset_binary_x_i8_product\" for the bounded "
              "Stage 4 asymmetric offset-binary packed-i4 x plain-i8 "
              "widening-product typed surface";
  if (getProductRelation() != "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2")
    return emitOpError()
           << "requires product_relation "
              "\"offset-binary-i4mf4-x-i8mf4x2-to-i16mf2\" for the bounded "
              "asymmetric offset-binary packed-i4 x plain-i8 widening-product "
              "route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i8 LMUL mf4 packed-i4 weight operand, two i8 LMUL "
              "mf4 plain-int8 activation operands, one !tcrv_rvv.vl operand, "
              "and one i16 LMUL mf2 result";
  if (!isGenericRVVVectorSignedI8MF4(getWeight().getType()))
    return emitOpError()
           << "requires the packed-i4 weight source vector to have type "
              "!tcrv_rvv.vector<i8, \"mf4\"> for the asymmetric offset-binary "
              "packed-i4 x plain-i8 widening-product route";
  if (!isGenericRVVVectorSignedI8MF4(getActivationLow().getType()) ||
      !isGenericRVVVectorSignedI8MF4(getActivationHigh().getType()))
    return emitOpError()
           << "requires the low and high plain-int8 activation source vectors "
              "to have type !tcrv_rvv.vector<i8, \"mf4\"> for the asymmetric "
              "offset-binary packed-i4 x plain-i8 widening-product route";
  if (!isGenericRVVVectorSignedI16MF2(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the asymmetric offset-binary "
              "packed-i4 x plain-i8 widening-product route";
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
              "metadata for asymmetric offset-binary packed-i4 x plain-i8 "
              "widening product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ40Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, and the block-format structural facts. Anything else
  // -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected
  // name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    // The bounded block-format / shape-knob attributes, plus the N3
    // autotuner's resource-provenance audit trail. The schedule producer pass
    // (MaterializeRVVQ40Schedule) stamps the chosen shape knobs alongside a
    // "tcrv_rvv.q4_0_schedule.*" provenance namespace (the candidate count, the
    // selected cost, the Zvl128b capability fact, the vreg budget) so the choice
    // is a PROVABLE resource-aware selection, not a manual constant. The
    // provenance is mirror metadata (I4): it records the derivation, it does not
    // carry executable config. Accept any attr in that bounded namespace.
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" ||
           name.starts_with("tcrv_rvv.q4_0_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_0_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', and "
                "'activation_high_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q4_0_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_0_q8_0_block_dot\" for "
              "the bounded ggml Q4_0 x Q8_0 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q4_0 x Q8_0 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK8_0 == 32,
  // block_q4_0 stride 18, block_q8_0 stride 34, quants at byte offset +2, the
  // q8 high half at +16 within the block payload. Pin them so a malformed
  // typed body cannot lower under the block-dot emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q4_0 x "
                            "Q8_0 block dot-product route";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_q4_0) for the "
              "ggml Q4_0 x Q8_0 block dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q4_0 x Q8_0 block dot-product route";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (quants follow the inline fp16 "
              "scale) for the ggml Q4_0 x Q8_0 block dot-product route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q4_0 x Q8_0 block dot-product route";

  // The optional integer-core LMUL is a bounded resource/scheduling fact: the
  // per-block dot-product core anchors at "mf4" (the INC-2a default) or "m1"
  // (the ggml-matching one-vwredsum-per-block anchor). Both are byte-exact; any
  // other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf4" && *coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf4\" or \"m1\" (the bounded "
                "byte-exact resource anchors for the ggml Q4_0 x Q8_0 block "
                "dot-product integer core); got \""
             << *coreLmul << "\"";
  }

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // the outer block loop processes 1 (default), 2, or 4 blocks per iteration. It
  // is byte-exact (the per-block fp32 folds stay in strict ascending order; only
  // the independent integer cores overlap). Any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml Q4_0 x Q8_0 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob: the
  // inner half-block strip loop is kept ("robust", default -- correct at any
  // VLEN) or elided ("elided" -- a single vsetvl_e8m1(16) + one vwredsum per
  // half-block, correct ONLY at VLEN >= 128). Any other spelling is rejected
  // fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml Q4_0 x Q8_0 "
                "block dot-product); got \""
             << *stripElision << "\"";
    // The elided form drops the inner strip loop and emits a single
    // vsetvl_e8m1(16) per half-block; it is correct only when the integer core
    // anchors at m1 (mf4's vsetvl_e32m1 VLMAX is 4 at VLEN=128, which would
    // silently drop 12 of the 16 nibble bytes). Reject the silently-wrong
    // combination fail-closed (I7) so the autotuner cannot request it.
    if (*stripElision == "elided" &&
        getIntegerCoreLmul().value_or("mf4") != "m1")
      return emitOpError()
             << "strip_elision \"elided\" requires integer_core_lmul \"m1\" "
                "(the single-vsetvl_e8m1(16) half-block cover is correct only at "
                "the m1 anchor; the mf4 anchor's vsetvl_e32m1 VLMAX would drop "
                "12 of 16 nibble bytes)";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. Their C types pin the
  // ggml ABI byte layout the emission depends on.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_0 x Q8_0 block dot-product route";
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
              "metadata for the ggml Q4_0 x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlGemmTileQ40Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, the block-format structural facts, and the bounded
  // activation-column count M. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedGemmTileAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" || name == "activation_cols";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_0_q8_0_gemm_tile keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedGemmTileAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_0 x Q8_0 GEMM tile attributes "
                "'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', "
                "'activation_high_byte_offset', and 'activation_cols'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q4_0_q8_0_gemm_tile")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_0_q8_0_gemm_tile\" for "
              "the bounded ggml Q4_0 x Q8_0 GEMM tile typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q4_0 x Q8_0 GEMM tile route";
  // ggml's externally-defined block format (ggml-common.h), identical to the
  // per-row block dot: QK8_0 == 32, block_q4_0 stride 18, block_q8_0 stride 34,
  // quants at byte offset +2, the q8 high half at +16. Pin them so a malformed
  // typed body cannot lower under the GEMM tile emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q4_0 x "
                            "Q8_0 GEMM tile route";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_q4_0) for the "
              "ggml Q4_0 x Q8_0 GEMM tile route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q4_0 x Q8_0 GEMM tile route";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (quants follow the inline fp16 "
              "scale) for the ggml Q4_0 x Q8_0 GEMM tile route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q4_0 x Q8_0 GEMM tile route";

  // The bounded activation-column count M: G1 fixes a small tile so the inner
  // M-column loop and the M-wide fp32 accumulator array stay register-bounded.
  // Reject M outside the bounded [1, 16] band fail-closed (I7); the autotuner's
  // measurement-tuned M-block is G3.
  int64_t activationCols = getActivationCols();
  if (activationCols < 1 || activationCols > 16)
    return emitOpError()
           << "requires activation_cols in [1, 16] (the bounded G1 GEMM tile "
              "column count; the measurement-tuned M-block is G3); got "
           << activationCols;

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one activation column-stride, one output pointer, one runtime "
              "element-count runtime ABI operand, one !tcrv_rvv.vl operand, and "
              "one i32 LMUL m1 result";

  // The buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. The column stride is
  // the runtime byte distance between two adjacent activation columns.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 columns "
              "byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, M outputs)";
  if (!llvm::isa<mlir::IndexType>(getActivationColumnStride().getType()))
    return emitOpError()
           << "requires the activation column-stride operand to be a runtime "
              "index value (the per-column activation byte stride)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_0 x Q8_0 GEMM tile route";
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
              "metadata for the ggml Q4_0 x Q8_0 GEMM tile";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ80Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, and the block-format structural facts, plus the
  // bounded shape knobs and the N3 autotuner's resource-provenance audit trail.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" ||
           name.starts_with("tcrv_rvv.q8_0_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q8_0_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', and 'quant_byte_offset'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q8_0_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q8_0_q8_0_block_dot\" for "
              "the bounded ggml Q8_0 x Q8_0 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q8_0 x Q8_0 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK8_0 == 32, BOTH
  // operands are block_q8_0 (stride 34), quants at byte offset +2. There is NO
  // high-half split here (q8_0 is a single contiguous 32-element block), so the
  // op carries no activation_high_byte_offset.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q8_0 x "
                            "Q8_0 block dot-product route";
  if (getWeightBlockStride() != 34)
    return emitOpError()
           << "requires weight_block_stride == 34 (sizeof block_q8_0) for the "
              "ggml Q8_0 x Q8_0 block dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q8_0 x Q8_0 block dot-product route";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (quants follow the inline fp16 "
              "scale) for the ggml Q8_0 x Q8_0 block dot-product route";

  // The optional integer-core LMUL is a bounded resource/scheduling fact: the
  // per-block dot-product core anchors at "m2" (the ggml-matching
  // one-vwredsum-per-block anchor, the whole 32-element block in one strip),
  // "m1" (two 16-element strips), or "mf4" (eight 4-element strips). All three
  // are byte-exact; any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf4" && *coreLmul != "m1" && *coreLmul != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf4\", \"m1\", or \"m2\" (the "
                "bounded byte-exact resource anchors for the ggml Q8_0 x Q8_0 "
                "block dot-product integer core); got \""
             << *coreLmul << "\"";
  }

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // the outer block loop processes 1 (default), 2, or 4 blocks per iteration. It
  // is byte-exact (the per-block fp32 folds stay in strict ascending order; only
  // the independent integer cores overlap). Any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml Q8_0 x Q8_0 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob: the
  // inner strip loop is kept ("robust", default -- correct at any VLEN) or
  // elided ("elided" -- a single vsetvl_e8m2(32) + one vwredsum over the whole
  // block, correct ONLY at VLEN >= 128). Any other spelling is rejected
  // fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml Q8_0 x Q8_0 "
                "block dot-product); got \""
             << *stripElision << "\"";
    // The elided form drops the inner strip loop and emits a single
    // vsetvl_e8m2(32) per block; it is correct only when the integer core
    // anchors at m2 (m1's vsetvl_e8m1 VLMAX is 16 at VLEN=128, mf4's
    // vsetvl_e32m1 VLMAX is 4 -- either would silently drop block bytes). Reject
    // the silently-wrong combination fail-closed (I7) so the autotuner cannot
    // request it.
    if (*stripElision == "elided" &&
        getIntegerCoreLmul().value_or("mf4") != "m2")
      return emitOpError()
             << "strip_elision \"elided\" requires integer_core_lmul \"m2\" "
                "(the single-vsetvl_e8m2(32) whole-block cover is correct only "
                "at the m2 anchor; the m1/mf4 anchors' VLMAX would drop block "
                "bytes)";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one lhs base pointer, one rhs base pointer, one output "
              "pointer, one runtime element-count runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // lhs/rhs bases address the AoS block_q8_0 byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. Their C types pin the
  // ggml ABI byte layout the emission depends on.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the lhs base operand to bind a runtime ABI value of C "
              "type 'const uint8_t *' (the AoS block_q8_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the rhs base operand to bind a runtime ABI value of C "
              "type 'const uint8_t *' (the AoS block_q8_0 byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q8_0 x Q8_0 block dot-product route";
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
              "metadata for the ggml Q8_0 x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ41Q81Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16-plus-min scale model, the block-format structural facts (including
  // the SECOND-scale byte offsets), the bounded shape knobs, and the N3
  // autotuner's resource-provenance audit trail. Anything else -- a forbidden
  // local element_count/SEW/LMUL/policy attr, or an unexpected name -- is
  // rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" ||
           name == "weight_min_byte_offset" ||
           name == "activation_sum_byte_offset" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" ||
           name.starts_with("tcrv_rvv.q4_1_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_1_q8_1_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', "
                "'activation_high_byte_offset', 'weight_min_byte_offset', and "
                "'activation_sum_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q4_1_q8_1_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_1_q8_1_block_dot\" for "
              "the bounded ggml Q4_1 x Q8_1 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y-plus-min")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y-plus-min\" "
              "for the ggml Q4_1 x Q8_1 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK8_1 == 32,
  // block_q4_1 stride 20 ({fp16 d; fp16 m; uint8 qs[16]}), block_q8_1 stride 36
  // ({fp16 d; fp16 s; int8 qs[32]}), quants at byte offset +4 (after BOTH inline
  // fp16 scales), the q8 high half at +16, the second scale (m / s) at +2. Pin
  // them so a malformed typed body cannot lower under the block-dot emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_1) for the ggml Q4_1 x "
                            "Q8_1 block dot-product route";
  if (getWeightBlockStride() != 20)
    return emitOpError()
           << "requires weight_block_stride == 20 (sizeof block_q4_1) for the "
              "ggml Q4_1 x Q8_1 block dot-product route";
  if (getActivationBlockStride() != 36)
    return emitOpError()
           << "requires activation_block_stride == 36 (sizeof block_q8_1) for "
              "the ggml Q4_1 x Q8_1 block dot-product route";
  if (getQuantByteOffset() != 4)
    return emitOpError()
           << "requires quant_byte_offset == 4 (quants follow the two inline "
              "fp16 scales) for the ggml Q4_1 x Q8_1 block dot-product route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q4_1 x Q8_1 block dot-product route";
  if (getWeightMinByteOffset() != 2)
    return emitOpError()
           << "requires weight_min_byte_offset == 2 (the block_q4_1 min m "
              "follows the inline fp16 delta d) for the ggml Q4_1 x Q8_1 block "
              "dot-product route";
  if (getActivationSumByteOffset() != 2)
    return emitOpError()
           << "requires activation_sum_byte_offset == 2 (the block_q8_1 scaled "
              "sum s follows the inline fp16 delta d) for the ggml Q4_1 x Q8_1 "
              "block dot-product route";

  // The optional integer-core LMUL is a bounded resource/scheduling fact: the
  // per-block dot-product core anchors at "mf4" (the default) or "m1" (the
  // ggml-matching one-vwredsum-per-half-block anchor). q4_1's nibble half-block
  // matches q4_0's shape, so the anchor set is q4_0's {mf4, m1}, NOT q8_0's. Both
  // are byte-exact; any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf4" && *coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf4\" or \"m1\" (the bounded "
                "byte-exact resource anchors for the ggml Q4_1 x Q8_1 block "
                "dot-product integer core); got \""
             << *coreLmul << "\"";
  }

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // 1 (default), 2, or 4 blocks per outer iteration. Byte-exact (the per-block
  // fp32 folds stay in strict ascending order); any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml Q4_1 x Q8_1 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob:
  // "robust" (default, correct at any VLEN) or "elided" (a single
  // vsetvl_e8m1(16) + one vwredsum per half-block, correct ONLY at VLEN >= 128).
  // Any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml Q4_1 x Q8_1 "
                "block dot-product); got \""
             << *stripElision << "\"";
    // The elided form drops the inner strip loop and emits a single
    // vsetvl_e8m1(16) per half-block; it is correct only when the integer core
    // anchors at m1 (mf4's vsetvl_e32m1 VLMAX is 4 at VLEN=128, which would
    // silently drop 12 of 16 nibble bytes). Reject the silently-wrong
    // combination fail-closed (I7) so the autotuner cannot request it.
    if (*stripElision == "elided" &&
        getIntegerCoreLmul().value_or("mf4") != "m1")
      return emitOpError()
             << "strip_elision \"elided\" requires integer_core_lmul \"m1\" "
                "(the single-vsetvl_e8m1(16) half-block cover is correct only at "
                "the m1 anchor; the mf4 anchor's vsetvl_e32m1 VLMAX would drop "
                "12 of 16 nibble bytes)";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. Their C types pin the
  // ggml ABI byte layout the emission depends on.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_1 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_1 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_1 x Q8_1 block dot-product route";
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
              "metadata for the ggml Q4_1 x Q8_1 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ6KQ8KAux32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // per-sub-block int8 scale model, and the super-block-format structural facts.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q6_k_q8_k_aux32_partial keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_qh_byte_offset', 'weight_scales_byte_offset', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q6_k_q8_k_aux32_partial")
    return emitOpError()
           << "currently supports only kind \"ggml_q6_k_q8_k_aux32_partial\" "
              "for the bounded ggml Q6_K x Q8_K super-block integer partial "
              "typed surface";
  if (getScaleModel() != "per-sub-block-int8-scale-i32-domain")
    return emitOpError()
           << "requires scale_model \"per-sub-block-int8-scale-i32-domain\" for "
              "the ggml Q6_K x Q8_K super-block integer partial route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 16 sub-blocks of 16 elements, block_q6_K stride 210 (ql@0|qh@128|scales@192|
  // d@208), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so a malformed
  // typed body cannot lower under the super-block partial emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q6_K x "
                            "Q8_K super-block integer partial route";
  if (getSubBlock() != 16)
    return emitOpError()
           << "requires sub_block == 16 (16-element sub-block scale boundary) "
              "for the ggml Q6_K x Q8_K super-block integer partial route";
  if (getWeightBlockStride() != 210)
    return emitOpError()
           << "requires weight_block_stride == 210 (sizeof block_q6_K) for the "
              "ggml Q6_K x Q8_K super-block integer partial route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q6_K x Q8_K super-block integer partial route";
  if (getWeightQhByteOffset() != 128)
    return emitOpError()
           << "requires weight_qh_byte_offset == 128 (qh follows ql[128]) for "
              "the ggml Q6_K x Q8_K super-block integer partial route";
  if (getWeightScalesByteOffset() != 192)
    return emitOpError()
           << "requires weight_scales_byte_offset == 192 (scales follow "
              "ql[128]+qh[64]) for the ggml Q6_K x Q8_K super-block integer "
              "partial route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q6_K x Q8_K super-block integer partial route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one aux32 output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is an int32_t * (the 8-lane aux32 integer-state destination -- NOT
  // the fp32 *s of the K2 fold), and the element count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q6_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'int32_t *' (the per-super-block aux32[8] integer-state "
              "destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q6_K x Q8_K super-block integer partial "
              "route";
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
              "metadata for the ggml Q6_K x Q8_K super-block integer partial";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ6KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // deferred-fp32-fold scale model, and the super-block-format structural facts
  // (now including the fp16 weight scale d @208 and the fp32 activation scale
  // d @0 the K2 fold reads). Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_d_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q6_k_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_qh_byte_offset', 'weight_scales_byte_offset', "
                "'weight_d_byte_offset', 'activation_d_byte_offset', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q6_k_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q6_k_q8_k_block_dot\" for the "
              "bounded ggml Q6_K x Q8_K super-block full block dot-product typed "
              "surface";
  if (getScaleModel() !=
      "per-sub-block-int8-scale-i32-domain-deferred-fp32-fold")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-int8-scale-i32-domain-deferred-fp32-fold\" for the "
              "ggml Q6_K x Q8_K super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 16 sub-blocks of 16 elements, block_q6_K stride 210 (ql@0|qh@128|scales@192|
  // d@208), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so a malformed
  // typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q6_K x "
                            "Q8_K super-block full block dot-product route";
  if (getSubBlock() != 16)
    return emitOpError()
           << "requires sub_block == 16 (16-element sub-block scale boundary) "
              "for the ggml Q6_K x Q8_K super-block full block dot-product route";
  if (getWeightBlockStride() != 210)
    return emitOpError()
           << "requires weight_block_stride == 210 (sizeof block_q6_K) for the "
              "ggml Q6_K x Q8_K super-block full block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q6_K x Q8_K super-block full block dot-product route";
  if (getWeightQhByteOffset() != 128)
    return emitOpError()
           << "requires weight_qh_byte_offset == 128 (qh follows ql[128]) for "
              "the ggml Q6_K x Q8_K super-block full block dot-product route";
  if (getWeightScalesByteOffset() != 192)
    return emitOpError()
           << "requires weight_scales_byte_offset == 192 (scales follow "
              "ql[128]+qh[64]) for the ggml Q6_K x Q8_K super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 208)
    return emitOpError()
           << "requires weight_d_byte_offset == 208 (the fp16 super-block scale "
              "d follows ql[128]+qh[64]+scales[16]) for the ggml Q6_K x Q8_K "
              "super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml Q6_K x Q8_K super-block full block "
              "dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q6_K x Q8_K super-block full block dot-product "
              "route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination -- NOT K1's
  // int32_t * aux32 state), and the element count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q6_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q6_K x Q8_K super-block full block "
              "dot-product route";
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
              "metadata for the ggml Q6_K x Q8_K super-block full block "
              "dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlVecScaleF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind plus
  // the optional resource/scheduling strip-LMUL knob. Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name --
  // is rejected fail-closed (I7). The knob is named "strip_lmul" (not the
  // forbidden with_vl/setvl "lmul" spelling), exactly as the sibling block-dot
  // op uses "integer_core_lmul", so the I5 boundary check stays untouched.
  auto isAllowedScaleAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "strip_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_vec_scale_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedScaleAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 scale attributes 'kind' and "
                "'strip_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_vec_scale_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_vec_scale_f32\" for the "
              "bounded ggml f32 in-place elementwise scale typed surface";

  // The optional strip-LMUL is a bounded resource/scheduling fact: the f32 strip
  // loop anchors at m1/m2/m4/m8 (default m8, matching ggml's hand-written path).
  // All are byte-exact (every lane is multiplied by the same scalar v; the
  // runtime vsetvl_e32m<L>(n-i) re-strips correctly for any VLEN). Any other
  // spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> stripLmul = getStripLmul()) {
    if (*stripLmul != "m1" && *stripLmul != "m2" && *stripLmul != "m4" &&
        *stripLmul != "m8")
      return emitOpError()
             << "only accepts strip_lmul \"m1\", \"m2\", \"m4\", or \"m8\" (the "
                "bounded byte-exact f32-strip resource anchors for the ggml "
                "f32 elementwise scale); got \""
             << *stripLmul << "\"";
  }

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one in-place f32 buffer pointer, one runtime f32 scalar, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, and one f32 LMUL m1 result";

  // The in-place buffer is read AND written (y[i] *= v) -- it is the FIRST
  // forward-pass op whose single buffer is both input and output. It binds a
  // runtime ABI value of C type 'float *'; the scalar binds a runtime ABI value
  // of C type 'float' (the ggml `v` multiplier); the element count carries n.
  RuntimeABIValueOp bufferBinding =
      getBuffer().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scalarBinding =
      getScalar().getDefiningOp<RuntimeABIValueOp>();
  if (!bufferBinding || bufferBinding.getCType() != "float *")
    return emitOpError()
           << "requires the in-place buffer operand to bind a runtime ABI value "
              "of C type 'float *' (the ggml y[] buffer read and written in "
              "place)";
  if (!scalarBinding || scalarBinding.getCType() != "float")
    return emitOpError()
           << "requires the scalar operand to bind a runtime ABI value of C "
              "type 'float' (the ggml v multiplier)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 elementwise scale route";
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
              "metadata for the ggml f32 elementwise scale";

  return mlir::success();
}

mlir::LogicalResult GgmlRmsNormF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind plus
  // the optional resource/scheduling NORMALIZE strip-LMUL knob. Anything else --
  // a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name
  // -- is rejected fail-closed (I7). The knob is named "strip_lmul" (not the
  // forbidden with_vl/setvl "lmul" spelling), exactly as the sibling f32 scale
  // op, so the I5 boundary check stays untouched. The strip knob governs only
  // the vectorized normalize tail; the Sx^2 reduction is always scalar-double.
  auto isAllowedRmsNormAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "strip_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_rms_norm_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime ne00/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedRmsNormAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 rms_norm attributes 'kind' and "
                "'strip_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_rms_norm_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_rms_norm_f32\" for the "
              "bounded ggml f32 rms_norm typed surface";

  // The optional strip-LMUL is a bounded resource/scheduling fact: the NORMALIZE
  // strip loop (y[i] = x[i] * scale) anchors at m1/m2/m4/m8 (default m8). All are
  // byte-exact (every lane is multiplied by the same scalar scale; the runtime
  // vsetvl_e32m<L>(ne00-i) re-strips correctly for any VLEN). The reduction is
  // scalar-double regardless of this knob. Any other spelling is rejected (I7).
  if (std::optional<llvm::StringRef> stripLmul = getStripLmul()) {
    if (*stripLmul != "m1" && *stripLmul != "m2" && *stripLmul != "m4" &&
        *stripLmul != "m8")
      return emitOpError()
             << "only accepts strip_lmul \"m1\", \"m2\", \"m4\", or \"m8\" (the "
                "bounded byte-exact f32-strip resource anchors for the ggml "
                "f32 rms_norm normalize tail); got \""
             << *stripLmul << "\"";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one f32 output "
              "pointer, one runtime f32 eps, one runtime element-count runtime "
              "ABI operand, one !tcrv_rvv.vl operand, and one f32 LMUL m1 result";

  // The input is read-only (const float *), the output is written (float *), eps
  // binds a runtime f32. ggml's rms_norm reads x and writes y (the non-fused,
  // no-weight path); the byte-exactness depends on x[i]*x[i] being a true f32
  // product widened to double, so the input must be a real f32 buffer.
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp epsBinding = getEps().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] row read for the Sx^2 reduction "
              "and the normalize)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] normalized output buffer)";
  if (!epsBinding || epsBinding.getCType() != "float")
    return emitOpError()
           << "requires the eps operand to bind a runtime ABI value of C type "
              "'float' (the ggml runtime eps)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime ne00 index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 rms_norm route";
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
              "metadata for the ggml f32 rms_norm";

  return mlir::success();
}

mlir::LogicalResult GgmlVecSiluF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attr (I4): the operation kind. There
  // is no resource/scheduling knob this cut -- the strip loop and the exp
  // polynomial are fixed at m2 (matching ggml's vsetvl_e32m2 path and the
  // m2-tied vbool16_t/vuint32m2_t mask/reinterpret types). Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name --
  // is rejected fail-closed (I7).
  auto isAllowedSiluAttr = [](llvm::StringRef name) { return name == "kind"; };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_vec_silu_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedSiluAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 silu attribute 'kind'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_vec_silu_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_vec_silu_f32\" for the "
              "bounded ggml f32 silu (vectorized-transcendental) typed surface";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one f32 output "
              "pointer, one runtime element-count runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one f32 LMUL m1 result";

  // The input is read-only (const float *), the output is written (float *).
  // ggml's ggml_vec_silu_f32 reads x and writes y (vec.cpp:380); the
  // byte-exactness depends on the exp polynomial running on real f32 lanes, so
  // the input must be a real f32 buffer.
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] row read for the silu)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] silu output buffer)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 silu route";
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
              "metadata for the ggml f32 silu";

  return mlir::success();
}

mlir::LogicalResult GgmlVecSoftMaxF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attr (I4): the operation kind. There
  // is no resource/scheduling knob this cut -- the strip loop and the exp
  // polynomial are fixed at m2 (matching ggml's vsetvl_e32m2 path and the
  // m2-tied mask/reinterpret types), and the f64 widening-reduce accumulator is
  // f64m1 to match ggml's vfwredusum_vs_f32m2_f64m1 fold exactly. Anything else
  // -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected
  // name -- is rejected fail-closed (I7).
  auto isAllowedSoftMaxAttr = [](llvm::StringRef name) {
    return name == "kind";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_vec_soft_max_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedSoftMaxAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 soft_max attribute 'kind'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_vec_soft_max_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_vec_soft_max_f32\" for the "
              "bounded ggml f32 soft_max (reduction + vectorized-transcendental) "
              "typed surface";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one f32 output pointer, one read-only f32 input "
              "pointer, one runtime f32 max, one runtime element-count runtime "
              "ABI operand, one !tcrv_rvv.vl operand, and one f32 LMUL m1 result";

  // ggml's ggml_vec_soft_max_f32 writes y[i] = e^{x[i]-max} (out), reads x[]
  // (in), takes the scalar max (in), and returns the f64 sum. The output is
  // written (float *), the input is read-only (const float *), max binds a
  // runtime f32. The byte-exactness depends on the exp polynomial running on
  // real f32 lanes and the f64 widening reduce, so x must be a real f32 buffer.
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp maxBinding = getMax().getDefiningOp<RuntimeABIValueOp>();
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] = e^{x-max} output buffer)";
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] row read for the soft_max)";
  if (!maxBinding || maxBinding.getCType() != "float")
    return emitOpError()
           << "requires the max operand to bind a runtime ABI value of C type "
              "'float' (the ggml runtime row max subtracted before exp)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 soft_max route";
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
              "metadata for the ggml f32 soft_max";

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
